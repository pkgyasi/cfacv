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
           stage samples.
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

from common import CVSet, load_json, path_length, save_json

_LINE = re.compile(r"^SM (?:FINAL)?iter\s*(\d+) replica (\d+) : reparam z (.*)$")


def read_history(paths):
    """Strings over time from one or more history files (job0, job1, ...).

    Returns (iters, z) with z of shape (frames, replicas, nCV).  A later file
    overrides an earlier one where their iterations overlap."""
    frames = {}
    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = _LINE.match(line)
                if m:
                    frames.setdefault(int(m.group(1)), {})[int(m.group(2))] = [float(v) for v in m.group(3).split()]
    if not frames:
        raise ValueError("no 'reparam z' lines found")
    nrep = max(len(f) for f in frames.values())
    iters = sorted(i for i, f in frames.items() if len(f) == nrep)   # drop a frame cut short
    z = np.array([[frames[i][r] for r in range(nrep)] for i in iters])
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
    return {
        "iterations": [int(iters[0]), int(iters[-1])], "frames": nf, "window": window,
        "string": mean, "arc": s, "drift": drift, "scatter": scatter,
        "max_drift": float(drift.max()), "mean_scatter": float(scatter.mean()),
        "spacing_cv": float(seg.std() / seg.mean()), "length": float(s[-1]),
        "approach": approach, "tol": tol, "pass": bool(drift.max() < tol),
    }


def compare(a, b, scale, scatter_a, scatter_b, factor=2.0):
    """Are two strings (same image count) one path?"""
    if a.shape != b.shape:
        raise ValueError(f"strings have different shapes {a.shape} and {b.shape}")
    d = np.linalg.norm((a - b) / scale, axis=1)
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
    p.add_argument("--dual", action="store_true", help="the first half of the replicas are the string")
    p.add_argument("--window", type=int, default=100, help="frames per averaging window")
    p.add_argument("--tol", type=float, default=0.25, help="gate: largest tolerated drift in scaled units [0.25]")
    p.add_argument("--out", default="string")
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--scales", required=True)
    p.add_argument("--scatter-a", type=float, required=True, help="mean scatter reported by 'status' for string a")
    p.add_argument("--scatter-b", type=float, required=True)
    p.add_argument("--factor", type=float, default=2.0, help="merged if the largest separation < factor x scatter")
    a = ap.parse_args(argv)
    scale = np.array(load_json(a.scales)["scale"])

    if a.cmd == "compare":
        r = compare(np.loadtxt(a.a), np.loadtxt(a.b), scale, a.scatter_a, a.scatter_b, a.factor)
        print(f"separation: mean {r['mean_separation']:.2f}, largest {r['max_separation']:.2f} scaled units; "
              f"combined scatter {r['combined_scatter']:.2f}")
        print("MERGED: one path" if r["merged"] else "DISTINCT: two channels, both go to the free-energy stage")
        return 0

    iters, z = read_history(a.history)
    if a.dual:
        z = z[:, : z.shape[1] // 2]
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
    print(f"GATE path convergence: {'PASS' if res['pass'] else 'FAIL'} "
          f"(largest drift {res['max_drift']:.3f}, limit {a.tol}; mean scatter {res['mean_scatter']:.3f})")
    np.savetxt(a.out + "_final.dat", res["string"], fmt="%.6f",
               header=f"string averaged over the last {a.window} frames")
    save_json(a.out + "_monitor.json", res)
    plot(res, names, iters, a.out)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
