#!/usr/bin/env python3
"""Stage 2: endpoint ensembles.

Measures every CV in the closed- and open-state trajectories and reports

  * per-CV mean and standard deviation in each state (circular for torsions),
  * per-CV separation: histogram overlap and a standardized effect size,
  * JOINT separation in the string CVs: the fraction of each state's frames
    that fall inside the other state's basin,
  * the minimum distance between the protein and its periodic images,

and writes the two files every later stage reads:

  <out>_scales.json  per-CV length scale (pooled within-basin standard deviation)
  <out>_basins.json  basin centre and width per string CV, per state

Example:
  endpoint_analysis.py --cv t4l_cvs.json --psf t4l.psf \\
      --closed c1.dcd c2.dcd c3.dcd --open o1.dcd o2.dcd o3.dcd \\
      --skip 1000 --cutoff 12 --out endpoints
"""
import argparse
import sys

import numpy as np

from common import CVCalculator, CVSet, circ_mean_sd, save_json, wrap


def min_image_distance(pos, box):
    """Shortest distance between any atom and any periodic image of any atom
    (images of the atom itself included).  Works for any cell."""
    from MDAnalysis.lib.mdamath import triclinic_vectors
    from scipy.spatial import cKDTree
    h = triclinic_vectors(box).astype(float)
    tree = cKDTree(pos)
    best = np.inf
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if (i, j, k) <= (0, 0, 0):
                    continue          # the other half follows by symmetry
                shift = i * h[0] + j * h[1] + k * h[2]
                d, _ = tree.query(pos + shift, k=1)
                best = min(best, d.min())
    return best


def measure(cvset, psf, dcds, skip, stride, image_sel, image_stride):
    """CV time series per trajectory and the minimum image distance seen."""
    import MDAnalysis as mda
    series, dmin = [], np.inf
    for dcd in dcds:
        u = mda.Universe(psf, dcd)
        calc = CVCalculator(cvset, u)
        prot = u.select_atoms(image_sel) if image_sel else None
        rows = []
        for n, ts in enumerate(u.trajectory[skip::stride]):
            rows.append(calc.frame())
            if prot is not None and n % image_stride == 0 and ts.dimensions is not None:
                dmin = min(dmin, min_image_distance(prot.positions.astype(float), ts.dimensions))
        if not rows:
            sys.exit(f"{dcd}: no frames left after skipping {skip}")
        series.append(np.array(rows))
    return series, dmin


def stats(x, periodic):
    """Mean and sd of each column; circular where periodic."""
    m, s = np.empty(x.shape[1]), np.empty(x.shape[1])
    for j in range(x.shape[1]):
        if periodic[j]:
            m[j], s[j] = circ_mean_sd(x[:, j])
        else:
            m[j], s[j] = x[:, j].mean(), x[:, j].std()
    return m, s


def overlap_coefficient(a, b, periodic, bins=60):
    """Shared area of two normalized histograms: 0 separate, 1 identical."""
    if periodic:
        lo, hi = -np.pi, np.pi
    else:
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        if hi <= lo:
            return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    return float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


def in_basin(x, centre, width, nsigma, periodic):
    """Frames with every CV within nsigma widths of the basin centre."""
    d = np.abs(wrap(x - centre, periodic)) / width
    return np.all(d <= nsigma, axis=1)


def analyse(closed, opened, cvset, nsigma):
    """closed/opened: lists of (frames, nCV) arrays, one per trajectory."""
    per = cvset.periodic
    C, O = np.vstack(closed), np.vstack(opened)
    mc, sc = stats(C, per)
    mo, so = stats(O, per)
    pooled = np.sqrt(0.5 * (sc ** 2 + so ** 2))
    sep = np.abs(wrap((mo - mc)[None, :], per)[0])
    rows = []
    for j, name in enumerate(cvset.names):
        # replica-to-replica spread of the mean: is one trajectory unlike the others?
        rep_c = [stats(t, per)[0][j] for t in closed]
        rep_o = [stats(t, per)[0][j] for t in opened]
        rows.append({
            "name": name, "type": cvset.types[j],
            "closed_mean": mc[j], "closed_sd": sc[j],
            "open_mean": mo[j], "open_sd": so[j],
            "effect_size": sep[j] / pooled[j] if pooled[j] > 0 else np.inf,
            "overlap": overlap_coefficient(C[:, j], O[:, j], per[j]),
            "closed_replica_means": rep_c, "open_replica_means": rep_o,
        })
    idx = [cvset.names.index(n) for n in cvset.string]
    sp = per[idx]
    joint = {
        "nsigma": nsigma,
        "closed_frames_in_open_basin": float(in_basin(C[:, idx], mo[idx], so[idx], nsigma, sp).mean()),
        "open_frames_in_closed_basin": float(in_basin(O[:, idx], mc[idx], sc[idx], nsigma, sp).mean()),
        "closed_frames_in_closed_basin": float(in_basin(C[:, idx], mc[idx], sc[idx], nsigma, sp).mean()),
        "open_frames_in_open_basin": float(in_basin(O[:, idx], mo[idx], so[idx], nsigma, sp).mean()),
        # distance between basin centres in units of the pooled width
        "scaled_separation": float(np.linalg.norm(sep[idx] / pooled[idx])),
    }
    scales = {"cvs": cvset.string, "scale": pooled[idx],
              "note": "pooled within-basin standard deviation; A for distances, rad for torsions"}
    basins = {"cvs": cvset.string, "types": [cvset.types[i] for i in idx], "nsigma": nsigma,
              "closed": {"centre": mc[idx], "width": sc[idx]},
              "open": {"centre": mo[idx], "width": so[idx]}}
    return rows, joint, scales, basins


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", required=True, help="CV definition JSON")
    ap.add_argument("--psf", required=True)
    ap.add_argument("--closed", nargs="+", required=True, help="closed-state trajectories")
    ap.add_argument("--open", nargs="+", required=True, dest="opened", help="open-state trajectories")
    ap.add_argument("--skip", type=int, default=0, help="frames to discard from the start of each trajectory")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--nsigma", type=float, default=2.0, help="basin half-width in standard deviations [2]")
    ap.add_argument("--image-sel", default="protein and not name H*",
                    help="atoms for the periodic-image check ('' disables it)")
    ap.add_argument("--image-stride", type=int, default=10, help="check images every this many analysed frames")
    ap.add_argument("--cutoff", type=float, default=12.0, help="nonbonded cutoff the image distance must exceed [12 A]")
    ap.add_argument("--max-joint-overlap", type=float, default=0.01,
                    help="gate: largest tolerated fraction of one state's frames inside the other's basin [0.01]")
    ap.add_argument("--out", default="endpoints", help="output prefix")
    a = ap.parse_args(argv)

    cvset = CVSet.load(a.cv)
    closed, dc = measure(cvset, a.psf, a.closed, a.skip, a.stride, a.image_sel, a.image_stride)
    opened, do = measure(cvset, a.psf, a.opened, a.skip, a.stride, a.image_sel, a.image_stride)
    rows, joint, scales, basins = analyse(closed, opened, cvset, a.nsigma)
    dmin = min(dc, do)

    print(f"{'CV':6s} {'type':5s} {'closed':>18s} {'open':>18s} {'effect':>7s} {'overlap':>8s}")
    for r in rows:
        f = np.degrees if r["type"] == "DIHED" else (lambda v: v)
        print(f"{r['name']:6s} {r['type']:5s} "
              f"{f(r['closed_mean']):9.3f} +/- {f(r['closed_sd']):5.3f} "
              f"{f(r['open_mean']):9.3f} +/- {f(r['open_sd']):5.3f} "
              f"{r['effect_size']:7.2f} {r['overlap']:8.3f}")
    print("(distances in A, torsions in degrees here; files are in A and rad)")
    print(f"\nJoint test in ({', '.join(cvset.string)}), basins of +/- {a.nsigma} sd:")
    print(f"  closed frames inside the OPEN basin : {joint['closed_frames_in_open_basin']:.4f}")
    print(f"  open frames inside the CLOSED basin : {joint['open_frames_in_closed_basin']:.4f}")
    print(f"  each state inside its own basin     : closed {joint['closed_frames_in_closed_basin']:.3f}, "
          f"open {joint['open_frames_in_open_basin']:.3f}")
    print(f"  separation of basin centres         : {joint['scaled_separation']:.2f} pooled sd")
    worst = max(joint["closed_frames_in_open_basin"], joint["open_frames_in_closed_basin"])
    ok_sep = worst <= a.max_joint_overlap
    print(f"GATE endpoint discrimination: {'PASS' if ok_sep else 'FAIL'} "
          f"(worst cross-basin fraction {worst:.4f}, limit {a.max_joint_overlap})")
    ok_img = True
    if a.image_sel:
        ok_img = dmin > a.cutoff
        print(f"GATE periodic images: {'PASS' if ok_img else 'FAIL'} "
              f"(closest approach {dmin:.2f} A, cutoff {a.cutoff} A)")

    save_json(a.out + "_cvstats.json", {"cvs": rows, "joint": joint, "min_image_distance": dmin})
    save_json(a.out + "_scales.json", scales)
    save_json(a.out + "_basins.json", basins)
    print(f"wrote {a.out}_cvstats.json, {a.out}_scales.json, {a.out}_basins.json")
    return 0 if (ok_sep and ok_img) else 1


if __name__ == "__main__":
    sys.exit(main())
