#!/usr/bin/env python3
"""Stage 5: is the string converged, and have two strings merged?

A four-dimensional string cannot be judged by eye.  This reads the history
file the string driver writes on replica 0 and reports numbers.

  status   one string:
             * the current string, per CV against image index;
             * DRIFT: scaled distance, image by image, between the string
               averaged over the last window and over the window before it
               (a converged string does not drift);
             * SCATTER: scaled RMS fluctuation of the string about its mean
               within the last window (the thermal scatter of the estimate);
             * spacing of the images in scaled arc length.
           The gate passes when the largest drift is below --tol.  It writes
           the window-averaged string, which is the path the free-energy
           stage samples.  Two windows of a creeping string can agree to
           within --tol while it creeps, so for a history from the thalweg
           driver, which applies its own (stronger) stopping tests over half
           the run, the gate also needs the driver's verdict that the string
           stopped; a history that ends at the update cap, or before the
           driver's verdict, fails.  With fewer than two windows of frames
           there is nothing to judge, and the tool says so (the window is not
           shrunk to fit).
  compare  two strings: their mean scaled separation against their combined
           scatter.  Merged strings are one path; separated ones are two
           channels and both go to the free-energy stage.

Example:
  string_monitor.py status --history output/0/t4l_sm.job0.0.history \\
      --scales endpoints_scales.json --cv t4l_cvs.json --window 200 --out string0
  string_monitor.py compare --a string0_final.dat --b string1_final.dat \\
      --scales endpoints_scales.json --scatter-a 0.2 --scatter-b 0.25
"""
import argparse
import re
import sys

import numpy as np

from common import CVSet, load_json, path_length, save_json, wrap

_LINE = re.compile(r"^SM (FINAL)?iter\s*(\d+) replica (\d+) : reparam z (.*)$")


def read_history(paths):
    """Strings over time from one or more history files (job0, job1, ...).

    Returns (iters, z) with z of shape (frames, replicas, nCV).  A later file
    overrides an earlier one where their iterations overlap."""
    frames = {}; final = {}; thalweg = False; verdict = None
    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                if line.startswith("TW iter"):
                    thalweg = True
                    m = re.match(r"^TW iter\s*(\d+) : (finished, .*|converged.*)$", line.strip())
                    if m:
                        # the last word of the last file is the driver's verdict
                        if m.group(2).startswith("finished"):
                            verdict = (int(m.group(1)), m.group(2)[len("finished, "):])
                        elif verdict is None or verdict[0] < int(m.group(1)):
                            verdict = (int(m.group(1)), "converged")
                m = _LINE.match(line)
                if not m:
                    continue
                try:
                    vals = [float(v) for v in m.group(4).split()]
                except ValueError:
                    continue                                  # a line cut in the middle of a number
                # the thalweg driver's FINAL block is its own result, the string averaged over
                # its last window; it is not one more frame, and carries the last frame's number
                (final if m.group(1) else frames.setdefault(int(m.group(2)), {}))[int(m.group(3))] = vals
    read_history.thalweg = thalweg
    read_history.verdict = verdict
    if not frames:
        raise ValueError("no 'reparam z' lines found")
    nrep = max(len(f) for f in frames.values())
    ncv = max(len(v) for f in frames.values() for v in f.values())
    # drop a frame cut short, by lines or within a line
    iters = sorted(i for i, f in frames.items() if len(f) == nrep and all(len(v) == ncv for v in f.values()))
    if not iters:
        raise ValueError("no complete frame in the history")
    z = np.array([[frames[i][r] for r in range(nrep)] for i in iters])
    read_history.final = (np.array([final[r] for r in range(nrep)])
                          if len(final) == nrep and all(len(v) == ncv for v in final.values()) else None)
    return np.array(iters), z


def status(iters, z, scale, window, tol):
    """Convergence numbers for z of shape (frames, images, nCV)."""
    nf = len(z)
    if nf < 2 * window:
        raise ValueError(f"{nf} frames recorded; need at least {2 * window} for two windows of {window}")
    last, prev = z[-window:], z[-2 * window:-window]
    mean = last.mean(axis=0)
    drift = np.linalg.norm((mean - prev.mean(axis=0)) / scale, axis=1)
    scatter = np.sqrt(np.mean(np.sum(((last - mean) / scale) ** 2, axis=2), axis=0))
    s = path_length(mean, scale)
    seg = np.diff(s)
    # running distance from the final averaged string, to see the approach
    approach = np.sqrt(np.mean(np.sum(((z - mean) / scale) ** 2, axis=2), axis=1))
    # sharpest turn of the averaged string: a cosine below 0 is a hook, the
    # path doubling back on itself, usually at a free end
    d = np.diff(mean, axis=0) / scale
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
    turn = np.sum(d[:-1] * d[1:], axis=1)
    return {
        "turn_cos": turn, "sharpest_turn": float(turn.min()), "sharpest_turn_image": int(turn.argmin()) + 1,
        "iterations": [int(iters[0]), int(iters[-1])], "frames": nf, "window": window,
        "string": mean, "arc": s, "drift": drift, "scatter": scatter,
        "max_drift": float(drift.max()), "mean_scatter": float(scatter.mean()),
        "spacing_cv": float(seg.std() / seg.mean()), "length": float(s[-1]),
        "approach": approach, "tol": tol, "pass": bool(drift.max() < tol),
    }


def compare(a, b, scale, scatter_a, scatter_b, factor=2.0, periodic=None):
    """Are two strings (same image count) one path?

    `periodic` flags the CVs that wrap.  Strings are stored unwrapped, so two
    strings that are the same curve can differ by whole turns; the difference
    is wrapped before it is measured."""
    if a.shape != b.shape:
        raise ValueError(f"strings have different shapes {a.shape} and {b.shape}")
    diff_ab = a - b if periodic is None else wrap(a - b, periodic)
    d = np.linalg.norm(diff_ab / scale, axis=1)
    noise = float(np.hypot(scatter_a, scatter_b))
    return {"separation": d, "mean_separation": float(d.mean()), "max_separation": float(d.max()),
            "combined_scatter": noise, "merged": bool(d.max() < factor * noise), "factor": factor}


def plot(res, names, iters, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: no figure written")
        return
    z = res["string"]
    nd = z.shape[1]
    fig, ax = plt.subplots(2, nd, figsize=(3.2 * nd, 6))
    for j in range(nd):
        ax[0, j].plot(z[:, j], "o-")
        ax[0, j].set_xlabel("image"); ax[0, j].set_ylabel(names[j])
    ax[1, 0].plot(res["drift"], "o-", label="drift"); ax[1, 0].plot(res["scatter"], "s--", label="scatter")
    ax[1, 0].axhline(res["tol"], color="k", lw=0.8); ax[1, 0].set_xlabel("image"); ax[1, 0].legend()
    ax[1, 1].plot(iters, res["approach"]); ax[1, 1].set_xlabel("iteration")
    ax[1, 1].set_ylabel("distance from final string")
    for n, j in enumerate(range(2, nd)):                     # projections onto the last CV
        c = nd - 2 - n                                       # nearest CV first: (d6, chi1) for T4L
        ax[1, j].plot(z[:, c], z[:, nd - 1], "o-")
        ax[1, j].set_xlabel(names[c]); ax[1, j].set_ylabel(names[nd - 1])
    fig.tight_layout()
    fig.savefig(out + "_monitor.png", dpi=130)
    print(f"wrote {out}_monitor.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status")
    p.add_argument("--history", nargs="+", required=True, help="history file(s) of replica 0, in job order")
    p.add_argument("--scales", required=True)
    p.add_argument("--cv", help="CV definition JSON, for names")
    p.add_argument("--dual", action="store_true",
                   help="cfacv's own driver in dual mode: the first half of the replicas are the string. "
                        "NOT for the thalweg driver, which writes only the string's images")
    p.add_argument("--window", type=int, default=100,
                   help="frames per averaging window [100]; at least two windows of frames are needed")
    p.add_argument("--tol", type=float, default=0.25, help="gate: largest tolerated drift in scaled units [0.25]")
    p.add_argument("--out", default="string")
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--scales", required=True)
    p.add_argument("--scatter-a", type=float, required=True, help="mean scatter reported by 'status' for string a")
    p.add_argument("--scatter-b", type=float, required=True)
    p.add_argument("--factor", type=float, default=2.0, help="merged if the largest separation < factor x scatter")
    p.add_argument("--cv", help="CV definition JSON; needed when any CV is a torsion, so differences are wrapped")
    a = ap.parse_args(argv)
    try:
        scale = np.array(load_json(a.scales)["scale"], float)
    except OSError as e:
        sys.exit(f"{a.scales}: {e.strerror}")
    except (ValueError, KeyError) as e:
        sys.exit(f"{a.scales} is not a scales file ({e})")

    if a.cmd == "compare":
        per = None
        if a.cv:
            full = CVSet.load(a.cv); per = full.subset(full.string).periodic
        ss = []
        for f in (a.a, a.b):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z = np.atleast_2d(np.loadtxt(f))
            except (OSError, ValueError) as e:
                sys.exit(f"{f}: cannot read a string from it ({e})")
            if z.size == 0 or z.shape[0] < 2 or z.shape[1] != len(scale):
                sys.exit(f"{f}: not a string of {len(scale)} CVs (read {z.shape[0] if z.size else 0} rows)")
            ss.append(z)
        try:
            r = compare(ss[0], ss[1], scale, a.scatter_a, a.scatter_b, a.factor, per)
        except ValueError as e:
            sys.exit(str(e))
        print(f"separation: mean {r['mean_separation']:.2f}, largest {r['max_separation']:.2f} scaled units; "
              f"combined scatter {r['combined_scatter']:.2f}")
        print("MERGED: one path" if r["merged"] else "DISTINCT: two channels, both go to the free-energy stage")
        return 0

    try:
        iters, z = read_history(a.history)
    except OSError as e:
        sys.exit(f"{e.filename}: {e.strerror}")
    except ValueError as e:
        sys.exit(f"{', '.join(a.history)}: {e}")
    if a.dual:
        if read_history.thalweg:
            sys.exit("--dual is for cfacv's own driver. This history was written by the thalweg driver, which records "
                     "only the string's images: halving it would judge half a string.")
        z = z[:, : z.shape[1] // 2]
    if a.window < 5:
        sys.exit(f"--window {a.window}: a window of fewer than 5 frames averages nothing")
    if len(z) < 2 * a.window:
        # shrinking the window to fit (as this once did, down to 2 frames)
        # compares two noisy snapshots and calls their agreement convergence
        sys.exit(f"{len(z)} frames recorded: two windows of {a.window} need {2 * a.window}. Run the string longer, or "
                 f"choose a smaller --window deliberately (it must still span many times the string's relaxation)")
    if z.shape[2] != len(scale):
        sys.exit(f"history has {z.shape[2]} CVs, scales has {len(scale)}")
    names = CVSet.load(a.cv).string if a.cv else [f"cv{j}" for j in range(z.shape[2])]
    try:
        res = status(iters, z, scale, a.window, a.tol)
    except ValueError as e:
        sys.exit(f"cannot judge convergence yet: {e}")
    print(f"iterations {res['iterations'][0]}-{res['iterations'][1]}, {res['frames']} frames, "
          f"{z.shape[1]} images, window {a.window} frames")
    print(f"{'image':>5s} " + " ".join(f"{n:>9s}" for n in names) + f" {'drift':>7s} {'scatter':>8s}")
    for i, row in enumerate(res["string"]):
        print(f"{i:5d} " + " ".join(f"{v:9.4f}" for v in row) + f" {res['drift'][i]:7.3f} {res['scatter'][i]:8.3f}")
    print(f"length {res['length']:.2f} scaled units; spacing varies by {res['spacing_cv']:.1%}")
    if res["sharpest_turn"] < 0:
        print(f"WARNING: the string doubles back at image {res['sharpest_turn_image']} "
              f"(turn cosine {res['sharpest_turn']:.2f}); a profile along it would cross that region twice")
    two = res["pass"]
    if read_history.thalweg:
        v = read_history.verdict
        stopped = bool(v and v[1] in ("converged", "stopped by its plateau and force tests"))
        if v is None:
            print("the thalweg driver has not given its verdict yet (no 'finished' line): the run is not over")
        else:
            print(f"the thalweg driver's own verdict at update {v[0]}: {v[1]}")
        if not stopped:
            print("  the driver's stopping tests (drift over half the run, and the force across the string) did not pass; "
                  "two windows that agree do not make a string that has stopped creeping")
        res["driver_verdict"] = v[1] if v else None
        res["pass"] = bool(two and stopped)
    print(f"GATE path convergence: {'PASS' if res['pass'] else 'FAIL'} "
          f"(largest drift between the last two windows {res['max_drift']:.3f}, limit {a.tol}"
          + (f", {'within' if two else 'OVER'}" if read_history.thalweg else "")
          + f"; mean scatter {res['mean_scatter']:.3f}"
          + ("; and the driver's verdict" if read_history.thalweg else "") + ")")
    np.savetxt(a.out + "_final.dat", res["string"], fmt="%.6f",
               header=f"string averaged over the last {a.window} frames")
    if read_history.final is not None:
        np.savetxt(a.out + "_driver_final.dat", read_history.final, fmt="%.6f",
                   header="the thalweg driver's own result: the string averaged over its last window and respaced")
        d = np.sqrt(np.mean(np.sum(((read_history.final - res["string"]) / scale) ** 2, axis=1)))
        print(f"the driver's own final string is in {a.out}_driver_final.dat ({d:.3f} scaled units rms from the one above)")
    save_json(a.out + "_monitor.json", res)
    plot(res, names, iters, a.out)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
