"""The stage runner: order, the record, stopping at a failed step, resuming."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "thalweg_run.py")


def run(settings, *args):
    r = subprocess.run([sys.executable, RUN, str(settings), *args], capture_output=True, text=True, cwd=os.path.dirname(settings))
    return r.returncode, r.stdout + r.stderr


def write(tmp_path, gate_ok, command="sh {conf}"):
    # "NAMD" here is a shell script that writes what a finished run writes
    for i in range(3):
        (tmp_path / f"r{i}.conf").write_text(f"echo run {i} >> order.txt; echo 'End of program'\n")
    (tmp_path / "runs.list").write_text("r0.conf\nr1.conf\nr2.conf\n")
    s = {"vars": {"word": "hello"},
         "launcher": {"mode": "local", "command": command, "parallel": 2, "hang_minutes": 0.02},
         "stages": [
             {"name": "first", "steps": [{"shell": "echo {word} > a.txt"}, {"runlist": "runs.list"}]},
             {"name": "gate", "steps": [{"shell": f"echo 'GATE toy: {'PASS' if gate_ok else 'FAIL'}'; exit {0 if gate_ok else 1}", "gate": "toy"}]},
             {"name": "last", "steps": [{"shell": "echo done > z.txt"}]}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s)); return p


def test_stops_at_a_failed_gate_and_resumes(tmp_path):
    p = write(tmp_path, gate_ok=False)
    rc, out = run(p)
    assert rc == 1 and "STOPPED" in out and "GATE toy did not pass" in out
    assert (tmp_path / "a.txt").read_text().strip() == "hello"
    assert not (tmp_path / "z.txt").exists()                     # nothing after the gate ran
    assert len((tmp_path / "order.txt").read_text().split("\n")) == 4
    # fix the gate: the finished steps are not run again
    write(tmp_path, gate_ok=True)
    rc, out = run(p)
    assert rc == 0 and "all stages finished" in out and out.count("[done") == 2
    assert (tmp_path / "z.txt").exists()
    assert len((tmp_path / "order.txt").read_text().split("\n")) == 4      # the runs were not repeated
    rc, out = run(p, "--dry-run")
    assert rc == 0 and out.count("[done") == 4 and "to do" not in out
    rc, out = run(p, "--from", "gate", "--dry-run")
    assert out.count("to do") == 2


def test_failed_and_hung_runs_stop_the_runner(tmp_path):
    p = write(tmp_path, gate_ok=True)
    (tmp_path / "r1.conf").write_text("echo 'FATAL ERROR: no'; echo 'End of program'\n")
    rc, out = run(p)
    assert rc == 1 and "1 of 3 runs" in out and "r1.conf" in out
    (tmp_path / "r1.conf").write_text("sleep 30\n")              # says nothing: hung
    rc, out = run(p)
    assert rc == 1 and "killed as hung" in out


def test_undefined_variable_and_missing_account(tmp_path):
    p = write(tmp_path, gate_ok=True)
    s = json.loads(p.read_text()); s["stages"][0]["steps"][0]["shell"] = "echo {nope}"; p.write_text(json.dumps(s))
    rc, out = run(p, "--dry-run")
    assert "{nope} is not defined" in out                        # left as written, and said so
    s["stages"][0]["steps"][0]["shell"] = "true"
    s["launcher"] = {"mode": "sbatch", "command": "sbatch --wait --array=1-{n} x.sbatch {list}"}; p.write_text(json.dumps(s))
    rc, out = run(p)
    assert rc == 1 and "--account" in out


def chain(tmp_path, n):
    """setup writes n configs and the list; the runs; then an analysis that counts results."""
    s = {"launcher": {"mode": "local", "command": "sh {conf}", "parallel": 2, "hang_minutes": 0},
         "stages": [
             {"name": "setup", "steps": [{"shell": f"for i in $(seq 1 {n}); do echo \"echo $i > out$i.txt; echo 'End of program'\" > r$i.conf; done; ls r*.conf > runs.list"}]},
             {"name": "runs", "steps": [{"runlist": "runs.list"}]},
             {"name": "analyse", "steps": [{"shell": "ls out*.txt | wc -l > result.txt; awk '{print $1}' result.txt > /dev/null; echo ${HOME} > /dev/null"}]}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s)); return p


def test_a_changed_upstream_step_redoes_everything_after_it(tmp_path):
    p = chain(tmp_path, 2)
    rc, out = run(p)
    assert rc == 0 and (tmp_path / "result.txt").read_text().strip() == "2"
    assert list((tmp_path / ".thalweg").glob("s-*.state.json")) and not (tmp_path / "s.json.state.json").exists()
    assert not list((tmp_path / ".thalweg").glob("*.lock"))            # the lock is gone when the runner ends
    rc, out = run(p)
    assert rc == 0 and out.count("[done") == 3                    # nothing changed, nothing rerun
    import time; time.sleep(1.1)                                    # so rewritten configs are newer than their logs
    chain(tmp_path, 4)
    rc, out = run(p)
    assert rc == 0 and "[done" not in out
    assert (tmp_path / "result.txt").read_text().strip() == "4"   # not the stale 2


def test_from_survives_a_failure_and_the_record_is_per_directory(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
    s = {"stages": [{"name": "one", "steps": [{"shell": "test -f ok && echo x >> ran.txt"}]},
                    {"name": "two", "steps": [{"shell": "echo y >> ran.txt"}]}]}
    p = tmp_path / "shared.json"; p.write_text(json.dumps(s))
    def go(d, *args):
        r = subprocess.run([sys.executable, RUN, str(p), *args], capture_output=True, text=True, cwd=d)
        return r.returncode, r.stdout
    (a / "ok").write_text("")
    assert go(a)[0] == 0
    rc, out = go(b)                                                 # same settings file, another directory: nothing is done there
    assert rc == 1 and "[done" not in out and not (b / "ran.txt").exists()
    (a / "ok").unlink()
    assert go(a, "--from", "one")[0] == 1                          # forgotten, then fails
    rc, out = go(a)                                                 # a plain rerun must NOT call it done
    assert rc == 1 and "all stages finished" not in out


def test_hang_kill_reaches_the_real_process(tmp_path):
    (tmp_path / "r.conf").write_text("sleep 47\n")
    (tmp_path / "runs.list").write_text("r.conf\n")
    s = {"launcher": {"mode": "local", "command": "sh {conf} && true", "hang_minutes": 0.02},
         "stages": [{"name": "x", "steps": [{"runlist": "runs.list"}]}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s))
    rc, out = run(p)
    assert rc == 1 and "killed as hung" in out
    import time; time.sleep(0.5)
    left = subprocess.run(["pgrep", "-x", "-f", "sleep 47"], capture_output=True, text=True).stdout.split()
    assert left == []


def test_missing_config_is_a_clean_stop(tmp_path):
    (tmp_path / "runs.list").write_text("nope.conf\n")
    s = {"stages": [{"name": "x", "steps": [{"runlist": "runs.list"}]}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s))
    rc, out = run(p)
    assert rc == 1 and "STOPPED" in out and "Traceback" not in out


def test_only_reruns_and_leaves_the_record_and_launcher_size_redoes_nothing(tmp_path):
    p = chain(tmp_path, 2)
    assert run(p)[0] == 0
    rc, out = run(p, "--only", "analyse")
    assert rc == 0 and "[done" not in out and "ok in" in out       # it ran, whatever the record said
    rc, out = run(p, "--only", "analyse")
    assert "[done" not in out                                       # and again
    rc, out = run(p)
    assert rc == 0 and out.count("[done") == 3                      # the full run's record was not touched
    s = json.loads(p.read_text()); s["launcher"]["parallel"] = 5; p.write_text(json.dumps(s))
    rc, out = run(p)
    assert rc == 0 and out.count("[done") == 3                      # more at once is not a different calculation
    s["vars"] = {"conf": "r1.conf"}; p.write_text(json.dumps(s))
    rc, out = run(p)
    assert rc != 0 and "fills in itself" in out


def test_interrupt_takes_the_runs_down_and_a_second_runner_is_refused(tmp_path):
    import signal, time
    for i in range(2):
        (tmp_path / f"r{i}.conf").write_text("sleep 53\n")
    (tmp_path / "runs.list").write_text("r0.conf\nr1.conf\n")
    s = {"launcher": {"mode": "local", "command": "sh {conf} && true", "parallel": 2, "hang_minutes": 0},
         "stages": [{"name": "x", "steps": [{"runlist": "runs.list"}]}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s))
    proc = subprocess.Popen([sys.executable, RUN, str(p)], cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2.0)
    rc, out = run(p)                                                 # a second runner in the same directory
    assert rc != 0 and "another thalweg_run.py" in out
    proc.send_signal(signal.SIGTERM)
    out = proc.communicate(timeout=20)[0]
    assert proc.returncode == 1 and "interrupted" in out
    time.sleep(0.5)
    assert subprocess.run(["pgrep", "-x", "-f", "sleep 53"], capture_output=True, text=True).stdout.split() == []
    assert not list((tmp_path / ".thalweg").glob("*.lock"))
