#!/usr/bin/env python3
"""Run the staged protocol from one settings file, and stop at a failed gate.

Every thalweg tool exits non-zero when its gate fails.  This runner chains
them: it runs the stages of a settings file in order, keeps a record of what
finished, and stops at the first step that fails, so that nothing is built on
a stage that did not pass.  Run it again and it carries on from that step.

The settings file is JSON:

  {
    "vars":     {"psf": "system/alad_wb.psf", "pool": "pool.json"},
    "launcher": {"mode": "local", "command": "namd3 +p4 {conf}", "parallel": 6,
                 "hang_minutes": 30},
    "stages": [
      {"name": "end states", "steps": [
          {"runlist": "ends/runs.list"},
          {"tool": "cv_select.py classifier --cv {pool} --psf {psf} ...", "gate": "basin classifier"}
      ]},
      ...
    ]
  }

A step is one of
  "tool"     a thalweg script and its arguments.  {name} is replaced from
             "vars"; shell wildcards are expanded.
  "runlist"  a file with one NAMD config per line.  Each is run through the
             launcher unless its log already ends cleanly; a config may start
             with "#PRE other.conf" lines, run first (see init_string.py).
  "shell"    any other command.
and may carry "gate": a label printed when the step fails.

Launchers for run lists
  "local"    runs the configs here, "parallel" at a time, each with
             "command" ({conf} is the config).  Use it inside one batch
             allocation, or on a workstation.  A run none of whose files has
             changed for "hang_minutes" is killed and counted as failed
             (NAMD can sit idle forever after a Tcl error).
  "sbatch"   submits the list as a job array and waits for it:
             "command": "sbatch --wait --account=NAME --array=1-{n}%20 {thalweg}/sbatch/namd_array.sbatch {list}".
             The account is yours to name: there is no default.  CFACV_BASEDIR
             must be exported; the array script has its own hang guard
             (HANG_MINUTES).

Options
  --dry-run        print the plan, with what is already done, and run nothing
  --from STAGE     forget the record from this stage on, and run from there
                   (a NAMD run is still skipped if its log is newer than its
                   config and ends cleanly; delete the logs to force those)
  --only STAGE     run this stage alone, whatever the record says, and leave
                   the record as it is (a rerun on the side)
  --status         print the record

One runner works in a directory at a time (a lock file says who); if it is
interrupted it stops the runs it started, and cancels the job array it
submitted.  Changing how many runs go at once does not redo anything.

The record is .thalweg/<settings name>-<tag>.state.json in the directory you run
in, and the logs go to .thalweg/<settings name>.logs/: the record belongs to
the run directory, so one settings file can serve several.  A step is redone
when its command changes, and so is every step after it.  A NAMD run is
redone when its config is newer than its log.  {thalweg} is the toolkit's
directory.  Braces that do not name a variable are left alone.

Example:
  thalweg_run.py examples/alad_cvsel.json --dry-run
  thalweg_run.py examples/alad_cvsel.json
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class Stop(Exception):
    pass


# Every child runs in its own process group (so a hung NAMD can be killed
# whole), which also takes it out of the terminal's group: Ctrl-C no longer
# reaches it.  So the runner keeps track of what it started and takes it down
# itself when it is interrupted, and never leaves a second NAMD writing the
# files the next invocation will open.
LIVE = set()            # Popen objects of running children
JOBS = set()            # SLURM job ids submitted with sbatch --wait
ABORT = threading.Event()


def kill_group(p, sig=signal.SIGKILL):
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def take_down(*_):
    ABORT.set()
    for p in list(LIVE):
        kill_group(p, signal.SIGTERM)
    time.sleep(0.5)
    for p in list(LIVE):
        kill_group(p)
    for j in list(JOBS):
        subprocess.run(["scancel", j], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def subst(text, v):
    """Replace {name} for the names under "vars" (and the ones the runner
    supplies).  Any other braces are left exactly as written, so awk, Tcl,
    JSON and ${HOME} pass through."""
    return re.sub(r"(?<!\$)\{(\w+)\}", lambda m: str(v[m.group(1)]) if m.group(1) in v else m.group(0), text)


def stem_of(conf):
    return conf[:-5] if conf.endswith(".conf") else conf


def log_complete(log, conf=None):
    """NAMD prints 'End of program' after a FATAL ERROR too.  A log older than
    its config belongs to an earlier version of that run and does not count."""
    if not os.path.exists(log) or os.path.getsize(log) == 0:
        return False
    if conf is not None:
        # equal times are not proof the log came after: run it again
        if not os.path.exists(conf) or os.path.getmtime(log) <= os.path.getmtime(conf):
            return False
    with open(log, errors="replace") as fh:
        text = fh.read()
    return "End of program" in text and "FATAL ERROR" not in text


def newest_change(stem):
    t = 0.0
    for f in glob.glob(glob.escape(stem) + ".*") + glob.glob(glob.escape(stem) + "_*"):
        try:
            t = max(t, os.path.getmtime(f))
        except OSError:
            pass
    return t


def run_namd(conf, command, hang_minutes):
    """One NAMD run with a guard against a hung process.  Returns '' or an error."""
    stem = stem_of(conf)
    log = stem + ".log"
    if log_complete(log, conf):
        return ""
    with open(log, "w") as out:
        # its own process group, so a kill reaches NAMD and not just the shell that started it
        p = subprocess.Popen(subst(command, {"conf": shlex.quote(conf)}), shell=True, stdout=out, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        LIVE.add(p)
        seen = time.time(); last = 0.0; poll = 0.5
        while p.poll() is None:
            time.sleep(poll); poll = min(poll * 2, 20.0)
            if hang_minutes <= 0:
                continue
            t = newest_change(stem)
            if t > last:
                last, seen = t, time.time()
            elif time.time() - seen > hang_minutes * 60:
                kill_group(p, signal.SIGTERM); time.sleep(0.3); kill_group(p)
                p.wait(); LIVE.discard(p)
                # one hang usually means all of them will (a Tcl error in a shared script)
                ABORT.set()
                return f"{conf}: no output for {hang_minutes} minutes; killed as hung"
    LIVE.discard(p)
    if ABORT.is_set() and p.returncode != 0:
        return f"{conf}: stopped"
    if p.returncode != 0:
        return f"{conf}: exit code {p.returncode} (see {log})"
    if not log_complete(log, conf):
        return f"{conf}: did not finish cleanly (see {log})"
    return ""


def run_with_pre(conf, command, hang_minutes):
    try:
        with open(conf) as fh:
            pre = [l.split()[1] for l in fh if l.startswith("#PRE ") and len(l.split()) > 1]
    except OSError as e:
        return f"{conf}: {e.strerror}"
    for c in pre + [conf]:
        if ABORT.is_set():
            return f"{conf}: not started (an earlier run hung, or the runner was interrupted)"
        err = run_namd(c, command, hang_minutes)
        if err:
            return err
    return ""


def do_runlist(path, launcher, logfh):
    try:
        with open(path) as fh:
            confs = [l.strip() for l in fh if l.strip() and not l.strip().startswith("#")]
    except OSError as e:
        raise Stop(f"run list {path}: {e.strerror}")
    if not confs:
        raise Stop(f"{path} lists no runs")
    mode = launcher.get("mode", "local")
    if mode == "sbatch":
        cmd = subst(launcher["command"], {"n": len(confs), "list": shlex.quote(path)})
        if "--account" not in cmd and " -A" not in cmd:
            raise Stop("launcher: the sbatch command names no --account; which account is charged is yours to decide")
        if "--wait" not in cmd and " -W" not in cmd:
            raise Stop("launcher: the sbatch command needs --wait, or the runner cannot tell when the array ends")
        print(f"      {cmd}", flush=True)
        if all(log_complete(stem_of(c) + ".log", c) for c in confs):
            return f"{len(confs)} runs (all already complete; nothing submitted)"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
        LIVE.add(p); job = None; lines = []
        for line in p.stdout:
            logfh.write(line); lines.append(line.rstrip())
            m = re.search(r"Submitted batch job (\d+)", line)
            if m:
                job = m.group(1); JOBS.add(job); print(f"      job {job}", flush=True)
        p.wait(); LIVE.discard(p)
        if job:
            JOBS.discard(job)
        if p.returncode and job is None:
            raise Stop(f"sbatch failed (exit {p.returncode}): {lines[-1] if lines else 'no message'}")
        bad = [c for c in confs if not log_complete(stem_of(c) + ".log", c)]
        if p.returncode or bad:
            raise Stop(f"{len(bad)} of {len(confs)} runs in {path} did not finish cleanly"
                       + (f", e.g. {bad[0]}" if bad else f" (sbatch exit {p.returncode})"))
        return f"{len(confs)} runs"
    if mode != "local":
        raise Stop(f"launcher: unknown mode '{mode}'")
    command = launcher.get("command", "namd3 +p4 {conf}")
    hang = float(launcher.get("hang_minutes", 30))
    skipped = sum(log_complete(stem_of(c) + ".log", c) for c in confs)
    errors = []; par = max(1, int(launcher.get("parallel", 1))); streak = [0]

    def one(c):
        err = run_with_pre(c, command, hang)
        # as many failures in a row as run at once: the fault is in what they share
        streak[0] = streak[0] + 1 if err else 0
        if streak[0] >= max(par, 3):
            ABORT.set()
        return err

    with cf.ThreadPoolExecutor(max_workers=par) as pool:
        for err in pool.map(one, confs):
            if err:
                errors.append(err); logfh.write(err + "\n")
    if errors:
        real = [e for e in errors if "not started" not in e]
        ABORT.clear()
        raise Stop(f"{len(real)} of {len(confs)} runs in {path} failed" + (f" and {len(errors) - len(real)} were not started"
                   if len(errors) > len(real) else "") + f"; first: {(real or errors)[0]}")
    return f"{len(confs)} runs" + (f" ({skipped} already complete)" if skipped else "")


def do_command(cmd, logfh):
    """Run, copy the output to the log, and echo the lines worth seeing."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                         start_new_session=True)
    LIVE.add(p)
    tail = []
    for line in p.stdout:
        logfh.write(line)
        if "Warning" in line or "warnings.warn" in line:
            continue
        tail.append(line.rstrip()); tail = tail[-6:]
        if "GATE" in line or line.startswith("WARNING"):
            print("      " + line.rstrip(), flush=True)
    p.wait(); LIVE.discard(p)
    return p.returncode, tail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--status", action="store_true")
    ap.add_argument("--from", dest="start"); ap.add_argument("--only")
    a = ap.parse_args(argv)

    try:
        with open(a.settings) as fh:
            s = json.load(fh)
        stages = s["stages"]
    except OSError as e:
        sys.exit(f"{a.settings}: {e.strerror}")
    except (ValueError, KeyError) as e:
        sys.exit(f"{a.settings}: not a settings file ({e})")
    v = dict(s.get("vars", {}))
    for name in ("conf", "n", "list", "thalweg"):
        if name in v:
            sys.exit(f"settings: \"{name}\" is a name the runner fills in itself; call the variable something else")
    v["thalweg"] = HERE
    launcher = {k: (subst(x, v) if isinstance(x, str) else x) for k, x in s.get("launcher", {}).items()}
    names = [st["name"] for st in stages]
    if len(set(names)) != len(names):
        sys.exit("settings: two stages have the same name")
    for want in (a.start, a.only):
        if want and want not in names:
            sys.exit(f"no stage named '{want}'; the stages are: {', '.join(names)}")
    # The record belongs to the directory the commands run in, not to the
    # settings file: one settings file can serve several run directories.
    # (the name carries a short hash of the settings file's path, so two files called the same do not share one)
    tag = hashlib.sha1(os.path.abspath(a.settings).encode()).hexdigest()[:6]
    base = os.path.join(".thalweg", os.path.splitext(os.path.basename(a.settings))[0] + "-" + tag)
    state_path = base + ".state.json"
    try:
        state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    except ValueError:
        sys.exit(f"{state_path} is damaged; delete it to start the record again")

    def save_state():
        try:
            os.makedirs(".thalweg", exist_ok=True)
            with open(state_path + ".tmp", "w") as fh:
                json.dump(state, fh, indent=1)
            os.replace(state_path + ".tmp", state_path)
        except OSError as e:
            raise Stop(f"cannot write the record in {os.getcwd()}/.thalweg: {e.strerror}")

    def on_signal(*_):
        # stop the children FIRST: the worker threads are waiting on them, and
        # the pool will not let the main thread go until they return
        take_down()
        raise KeyboardInterrupt()

    # one runner per run directory: a second one would start every unfinished run again, into the same files
    lock = base + ".lock"; have_lock = False
    if not (a.dry_run or a.status):
        try:
            os.makedirs(".thalweg", exist_ok=True)
            for attempt in (0, 1):
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, f"{socket.gethostname()} {os.getpid()}\n".encode()); os.close(fd); have_lock = True
                    break
                except FileExistsError:
                    host, pid = (open(lock).read().split() + ["?", "0"])[:2]
                    stale = host == socket.gethostname() and pid.isdigit() and not os.path.exists(f"/proc/{pid}")
                    if stale and attempt == 0:
                        os.remove(lock); continue
                    sys.exit(f"another thalweg_run.py is working in this directory (host {host}, process {pid}). "
                             f"Use --status to look. If it is gone, delete {lock}.")
        except OSError as e:
            sys.exit(f"cannot write in {os.getcwd()}/.thalweg: {e.strerror}")
        for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sg, on_signal)

    if a.start:
        for n in names[names.index(a.start):]:
            for k in [k for k in state if k.startswith(n + "#")]:
                del state[k]
        if not (a.dry_run or a.status):
            save_state()                                      # forgotten for good, even if the next step fails
    logdir = base + ".logs"
    if not (a.dry_run or a.status):
        os.makedirs(logdir, exist_ok=True)
    prev_sig = ""; redone = False; warned = set()

    failed = None
    try:
        for si, st in enumerate(stages):
            skip_stage = bool(a.only and st["name"] != a.only)
            if not skip_stage:
                print(f"[{si + 1}/{len(stages)}] {st['name']}", flush=True)
            for ti, step in enumerate(st["steps"]):
                kind = next((k for k in ("tool", "runlist", "shell") if k in step), None)
                if kind is None:
                    raise Stop(f"stage '{st['name']}', step {ti + 1}: needs one of tool, runlist, shell")
                what = subst(step[kind], v)
                for name in sorted(set(re.findall(r"(?<!\$)\{(\w+)\}", what))) if not skip_stage else []:
                    if name not in warned:
                        warned.add(name)
                        print(f"   note: {{{name}}} is not defined under \"vars\"; left as written")
                cmd = (f"{shlex.quote(sys.executable)} {shlex.quote(HERE)}/{what}" if kind == "tool" else what)
                key = f"{st['name']}#{ti}"
                # A step's signature includes every step before it, so a change
                # upstream undoes everything downstream; and once one step has
                # run in this invocation, nothing after it counts as done.
                # (of the launcher only the mode counts: running six at a time
                # instead of four is not a different calculation)
                sig = hashlib.sha1((prev_sig + kind + cmd + (launcher.get("mode", "local") if kind == "runlist" else "")).encode()).hexdigest()[:12]
                prev_sig = sig
                if skip_stage:
                    continue
                done = state.get(key, {}).get("sig") == sig and not redone and not a.only
                label = f"   {ti + 1}. {kind}: {what if len(what) < 110 else what[:107] + '...'}"
                if a.status or a.dry_run:
                    print(f"{label}   [{'done ' + state[key]['when'] if done else 'to do'}]")
                    redone = redone or not done
                    continue
                if done:
                    print(f"{label}   [done {state[key]['when']}]", flush=True)
                    continue
                print(label, flush=True)
                redone = True
                if key in state and not a.only:
                    del state[key]; save_state()
                t0 = time.time()
                logname = os.path.join(logdir, f"{si + 1:02d}_{ti + 1:02d}_{st['name'].replace(' ', '_').replace('/', '_')}.log")
                with open(logname, "w") as logfh:
                    if kind == "runlist":
                        note = do_runlist(what, launcher, logfh)
                    else:
                        rc, tail = do_command(cmd, logfh)
                        if rc:
                            for line in tail:
                                print("      | " + line)
                            raise Stop(f"stage '{st['name']}', step {ti + 1} failed (exit {rc})"
                                       + (f": GATE {step['gate']} did not pass" if step.get("gate") else "")
                                       + f". Log: {logname}")
                        note = tail[-1] if tail else ""
                if not a.only:                               # --only is a rerun on the side; it leaves the record alone
                    state[key] = {"sig": sig, "when": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0, 1)}
                    save_state()
                print(f"      ok in {time.time() - t0:.0f} s{': ' + note if note else ''}", flush=True)
    except Stop as e:
        failed = str(e)
    except KeyboardInterrupt:
        take_down()
        failed = "interrupted; the runs it had started were stopped" + (" and their SLURM jobs cancelled" if JOBS else "")
    finally:
        if have_lock:
            try:
                os.remove(lock)
            except OSError:
                pass
    if failed:
        print(f"STOPPED: {failed}")
        print("Nothing after this step was run. Fix it and run again WITHOUT --from; finished steps are kept.")
        return 1
    if not (a.dry_run or a.status):
        print("all stages finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
