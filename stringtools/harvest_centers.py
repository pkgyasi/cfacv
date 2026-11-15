#!/usr/bin/env python3
"""Stage 3b: harvest centers from TAMD trajectories.

Walks the pooled trajectories and keeps a frame as a new center when it is
farther than --spacing from every center kept so far.  Distance is measured in
SCALED CV units (each CV divided by its length scale from endpoint analysis),
with torsions wrapped, so a center is as far from its neighbours in chi1 as in
d1.  cfacv's own harvester (tcl/cfacv_harvest_centers.tcl) takes one raw
distance for every CV, which is only right when all CVs have the same units.

For each center it writes the full atomic configuration, so the mean-force run
and, later, a string image can start from it:

  <outdir>/center00000.coor   NAMD binary coordinates
  <outdir>/center00000.xsc    the periodic cell of that frame
  <outdir>/mesh.dat           one line per center: the string-CV values
  <outdir>/centers.json       index, source trajectory and frame of each center

Example:
  harvest_centers.py --cv t4l_cvs.json --scales endpoints_scales.json \\
      --psf t4l.psf --dcd tamd_c*.dcd tamd_o*.dcd --spacing 1.0 --out centers
"""
import argparse
import os
import sys

import numpy as np

from common import CVCalculator, CVSet, load_json, save_json, sdist


def select_centers(points, scale, periodic, spacing, max_centers=None):
    """Greedy selection: indices of points pairwise farther apart than
    `spacing` in scaled units, in order of first appearance."""
    keep = []
    kept = np.empty((0, points.shape[1]))
    for i, p in enumerate(points):
        if len(keep) == 0 or sdist(kept, p, scale, periodic).min() > spacing:
            keep.append(i)
            kept = np.vstack([kept, p])
            if max_centers and len(keep) >= max_centers:
                break
    return keep


def write_xsc(path, dims):
    from MDAnalysis.lib.mdamath import triclinic_vectors
    h = triclinic_vectors(dims)
    with open(path, "w") as fh:
        fh.write("# NAMD extended system configuration output file\n")
        fh.write("#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z\n")
        fh.write("0 " + " ".join(f"{v:.6f}" for v in h.flatten()) + " 0 0 0\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", required=True)
    ap.add_argument("--scales", required=True, help="scales.json from endpoint_analysis.py")
    ap.add_argument("--psf", required=True)
    ap.add_argument("--dcd", nargs="+", required=True)
    ap.add_argument("--spacing", type=float, required=True, help="minimum scaled distance between centers")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-centers", type=int, default=None)
    ap.add_argument("--out", default="centers")
    a = ap.parse_args(argv)

    import MDAnalysis as mda
    full = CVSet.load(a.cv)
    cvset = full.subset(full.string)
    sc = load_json(a.scales)
    if sc["cvs"] != cvset.names:
        sys.exit(f"scales are for {sc['cvs']}, the string CVs are {cvset.names}")
    scale = np.array(sc["scale"])
    per = cvset.periodic

    # pass 1: CV values of every frame
    pts, src = [], []
    for dcd in a.dcd:
        u = mda.Universe(a.psf, dcd)
        z = CVCalculator(cvset, u).trajectory(step=a.stride)
        pts.append(z)
        src += [(dcd, i * a.stride) for i in range(len(z))]
    pts = np.vstack(pts)
    keep = select_centers(pts, scale, per, a.spacing, a.max_centers)
    print(f"{len(pts)} frames -> {len(keep)} centers at scaled spacing {a.spacing}")

    # pass 2: write the kept configurations
    os.makedirs(a.out, exist_ok=True)
    by_dcd = {}
    for n, i in enumerate(keep):
        by_dcd.setdefault(src[i][0], []).append((n, src[i][1]))
    for dcd, items in by_dcd.items():
        u = mda.Universe(a.psf, dcd)
        for n, frame in items:
            ts = u.trajectory[frame]
            u.atoms.write(os.path.join(a.out, f"center{n:05d}.coor"), file_format="NAMDBIN")
            if ts.dimensions is not None:
                write_xsc(os.path.join(a.out, f"center{n:05d}.xsc"), ts.dimensions)
    np.savetxt(os.path.join(a.out, "mesh.dat"), pts[keep], fmt="%.6f")
    save_json(os.path.join(a.out, "centers.json"),
              {"cvs": cvset.names, "types": cvset.types, "spacing": a.spacing, "scale": scale,
               "centers": [{"index": n, "dcd": src[i][0], "frame": src[i][1], "z": pts[i]}
                           for n, i in enumerate(keep)]})
    print(f"wrote {a.out}/mesh.dat, centers.json and {len(keep)} coordinate sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
