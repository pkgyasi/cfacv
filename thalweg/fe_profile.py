#!/usr/bin/env python3
"""Stage 6: free-energy profile along a converged string, with uncertainty.

Restrained MD is run at every image (set it up with meanforce.py, pointing
--points at the final string).  From each run this takes the mean force
f = k <theta - z> = -dF/dz and integrates it along the string:

    F_i = - sum_{j<i} 1/2 (f_j + f_j+1) . (z_j+1 - z_j)

in RAW CV units: a line integral does not depend on how the CVs are scaled.

It reports, for the Stage 6 gates:
  * the profile and its standard error, propagated from block averages of
    each image's mean force;
  * the barrier and the end-to-end free-energy difference, with errors;
  * OVERLAP of neighbouring restrained ensembles along the path;
  * STABILITY: the barrier from the first half of the data against all of it;
  * RESOLUTION: the barrier from every other image against all of them.

On overlap: with stiff restraints the ensembles of neighbouring images are
much narrower than their spacing and do not overlap at all.  Integrating a
mean force does not need them to; it needs the mean force to be resolved,
which is what the resolution check tests.  The overlap gate is kept because
the proposal names it; --min-overlap 0 turns it off.

`compare` checks that the closed-to-open free-energy difference agrees
between strings.  It must: that difference does not depend on the path.

Example:
  fe_profile.py compute --dir fe0 --scales endpoints_scales.json --out fe0_profile
  fe_profile.py compare --a fe0_profile.json --b fe1_profile.json
"""
import argparse
import os
import sys

import numpy as np

from common import block_mean_se, load_json, path_length, read_cvout, save_json, wrap


def integrate(z, f):
    """Profile from mean forces (f = -dF/dz) at the images z (continuous)."""
    dz = np.diff(z, axis=0)
    steps = -0.5 * np.sum((f[:-1] + f[1:]) * dz, axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def integration_weights(z):
    """W with F = W . f(flattened): F_i is linear in the mean forces, so the
    variance of F_i is sum_j W_ij^2 var(f_j) for independent images."""
    n, d = z.shape
    dz = np.diff(z, axis=0)
    W = np.zeros((n, n, d))
    for i in range(1, n):
        W[i] = W[i - 1]
        W[i, i - 1] += -0.5 * dz[i - 1]
        W[i, i] += -0.5 * dz[i - 1]
    return W


def profile_with_error(z, f, se):
    F = integrate(z, f)
    W = integration_weights(z)
    var = np.einsum("ijd,jd->i", W ** 2, se ** 2)
    return F, np.sqrt(var), W


def difference_error(W, se, i, j):
    """Standard error of F_i - F_j."""
    return float(np.sqrt(np.sum((W[i] - W[j]) ** 2 * se ** 2)))


def neighbour_overlap(theta, z, periodic, bins=40):
    """Overlap of neighbouring ensembles projected on the local tangent."""
    out = []
    for i in range(len(z) - 1):
        t = z[i + 1] - z[i]
        L2 = np.sum(t * t)
        a = wrap(theta[i] - z[i], periodic) @ t / L2          # 0 at image i, 1 at image i+1
        b = wrap(theta[i + 1] - z[i], periodic) @ t / L2
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
        hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
        out.append(float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum()))
    return np.array(out)


def compute(z, theta, k, periodic, scale, nblocks=5, min_overlap=0.05):
    """z: (n, D) images.  theta: list of (samples, D) CV series, one per image."""
    k = np.asarray(k, float)
    f, se = np.empty_like(z), np.empty_like(z)
    f_half = np.empty_like(z)
    for i, th in enumerate(theta):
        d = wrap(th - z[i], periodic) * k
        f[i], se[i], _ = block_mean_se(d, nblocks)
        f_half[i] = d[: len(d) // 2].mean(axis=0)
    F, Fse, W = profile_with_error(z, f, se)
    top = int(F.argmax())
    start = int(F[: top + 1].argmin())
    Fh = integrate(z, f_half)
    ov = neighbour_overlap(theta, z, periodic)
    res = {
        "z": z, "arc": path_length(z, scale), "F": F - F[start], "F_se": Fse, "mean_force": f, "mean_force_se": se,
        "top_image": top, "barrier": float(F[top] - F[start]), "barrier_se": difference_error(W, se, top, start),
        "delta_F": float(F[-1] - F[0]), "delta_F_se": difference_error(W, se, len(z) - 1, 0),
        "barrier_first_half": float(Fh[top] - Fh[start]),
        "overlap": ov, "min_overlap": float(ov.min()),
    }
    # The first half is part of the whole, so the two estimates differ by about
    # one standard error of the full estimate when nothing is wrong; three is
    # the alarm level.
    res["stable"] = bool(abs(res["barrier"] - res["barrier_first_half"]) <= 3 * res["barrier_se"])
    res["overlap_ok"] = bool(ov.min() >= min_overlap)
    # RESOLUTION: the same integral using every other image.  If the images
    # resolve the mean force, dropping half of them changes little.
    idx = np.unique(np.concatenate([np.arange(0, len(z), 2), [top, start, len(z) - 1]]))
    Fc = integrate(z[idx], f[idx])
    pos = {int(j): n for n, j in enumerate(idx)}
    res["barrier_half_images"] = float(Fc[pos[top]] - Fc[pos[start]])
    res["resolved"] = bool(abs(res["barrier"] - res["barrier_half_images"]) <= 2 * res["barrier_se"] + 0.5)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compute")
    p.add_argument("--dir", required=True, help="directory written by 'meanforce.py setup' and filled by the runs")
    p.add_argument("--scales", required=True)
    p.add_argument("--discard", type=float, default=0.2)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--min-overlap", type=float, default=0.05)
    p.add_argument("--out", default="profile")
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "compare":
        A, B = load_json(a.a), load_json(a.b)
        d = A["delta_F"] - B["delta_F"]
        se = float(np.hypot(A["delta_F_se"], B["delta_F_se"]))
        ok = abs(d) <= 2 * se
        print(f"closed-to-open free-energy difference: {A['delta_F']:.2f} +/- {A['delta_F_se']:.2f} "
              f"and {B['delta_F']:.2f} +/- {B['delta_F_se']:.2f} kcal/mol")
        print(f"GATE path independence: {'PASS' if ok else 'FAIL'} (difference {d:.2f}, combined error {se:.2f})")
        db = A["barrier"] - B["barrier"]
        seb = float(np.hypot(A["barrier_se"], B["barrier_se"]))
        verdict = "unresolved" if abs(db) <= 2 * seb else ("a is lower" if db < 0 else "b is lower")
        print(f"barriers: {A['barrier']:.2f} +/- {A['barrier_se']:.2f} and {B['barrier']:.2f} +/- "
              f"{B['barrier_se']:.2f}: {verdict}")
        return 0 if ok else 1

    s = load_json(os.path.join(a.dir, "setup.json"))
    z = np.array(s["points"])
    periodic = np.array([t == "DIHED" for t in s["types"]])
    scale = np.array(load_json(a.scales)["scale"])
    theta = []
    for i in range(len(z)):
        rep = read_cvout(os.path.join(a.dir, f"pt{i:05d}.cvout"))
        th = rep["Th"][1]
        theta.append(th[int(a.discard * len(th)):])
    res = compute(z, theta, s["k"], periodic, scale, a.blocks, a.min_overlap)
    print(f"{'image':>5s} {'arc':>7s} {'F':>8s} {'+/-':>6s} {'overlap->next':>13s}")
    for i in range(len(z)):
        ov = f"{res['overlap'][i]:13.3f}" if i < len(z) - 1 else ""
        print(f"{i:5d} {res['arc'][i]:7.2f} {res['F'][i]:8.2f} {res['F_se'][i]:6.2f} {ov}")
    print(f"barrier {res['barrier']:.2f} +/- {res['barrier_se']:.2f} kcal/mol at image {res['top_image']}; "
          f"end-to-end {res['delta_F']:.2f} +/- {res['delta_F_se']:.2f}")
    print(f"GATE overlap: {'PASS' if res['overlap_ok'] else 'FAIL'} (smallest {res['min_overlap']:.3f}, "
          f"limit {a.min_overlap})")
    print(f"GATE stability: {'PASS' if res['stable'] else 'FAIL'} (barrier from the first half of the data "
          f"{res['barrier_first_half']:.2f})")
    print(f"GATE resolution: {'PASS' if res['resolved'] else 'FAIL'} (barrier from every other image "
          f"{res['barrier_half_images']:.2f})")
    np.savetxt(a.out + ".dat", np.column_stack([np.arange(len(z)), res["arc"], res["F"], res["F_se"]]),
               fmt=["%d", "%.4f", "%.4f", "%.4f"], header="image  scaled_arc  F(kcal/mol)  standard_error")
    save_json(a.out + ".json", res)
    print(f"wrote {a.out}.dat and {a.out}.json")
    return 0 if (res["overlap_ok"] and res["stable"] and res["resolved"]) else 1


if __name__ == "__main__":
    sys.exit(main())
