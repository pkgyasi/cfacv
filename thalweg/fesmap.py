#!/usr/bin/env python3
"""Stage 3d: reconstruct the free-energy surface from mean forces
(single-sweep method, Maragliano and Vanden-Eijnden, J Chem Phys 128, 184110).

    F(z) = sum_k a_k exp(-|z - z_k|^2 / 2 sigma^2)

with one Gaussian on every center and |.| the SCALED, periodic-aware distance.
The coefficients minimize the squared difference between the gradient of F and
the measured gradients at the centers.  sigma is chosen by k-fold
cross-validation: the surface is fitted without a group of centers and asked
to predict the mean force at them.  That left-out error is the Stage 3 gate.

Input is the same centers-and-forces file that cfacv's Reconstruct reads
(force column = mean force = -dF/dz), so the two programs can be run on one
file and compared.  Reconstruct does not scale the CVs; this does.

Example:
  fesmap.py fit --forces forces.dat --scales endpoints_scales.json --out map.json
  fesmap.py eval --map map.json --points anchors.dat
"""
import argparse
import sys

import numpy as np

from common import PERIODIC_TYPES, diff, load_json, read_forces, save_json

KT_300 = 0.0019872041 * 300.0


class FESMap:
    """Gaussian radial-basis free-energy surface in scaled CV space."""

    def __init__(self, centers, coef, sigma, scale, periodic, types=None, offset=0.0):
        self.centers = np.asarray(centers, float)
        self.coef = np.asarray(coef, float)
        self.sigma = float(sigma)
        self.scale = np.asarray(scale, float)
        self.periodic = np.asarray(periodic, bool)
        self.types = types
        self.offset = float(offset)

    # -- evaluation ---------------------------------------------------------
    def _du(self, z):
        """Scaled displacements z - z_k: shape (P, K, D)."""
        z = np.atleast_2d(np.asarray(z, float))
        return diff(z[:, None, :], self.centers[None, :, :], self.periodic) / self.scale

    def value(self, z):
        du = self._du(z)
        phi = np.exp(-np.sum(du ** 2, axis=2) / (2 * self.sigma ** 2))
        return phi @ self.coef - self.offset

    def gradient(self, z):
        """dF/dz in raw CV units, shape (P, D)."""
        du = self._du(z)
        phi = np.exp(-np.sum(du ** 2, axis=2) / (2 * self.sigma ** 2))
        gu = -np.einsum("pk,pkd,k->pd", phi, du, self.coef) / self.sigma ** 2
        return gu / self.scale

    def nearest_center_distance(self, z):
        return np.sqrt(np.sum(self._du(z) ** 2, axis=2)).min(axis=1)

    # -- fitting ------------------------------------------------------------
    @staticmethod
    def _design(z_eval, centers, sigma, scale, periodic):
        """Matrix G with (G a) = scaled gradient of F at z_eval, flattened."""
        du = diff(z_eval[:, None, :], centers[None, :, :], periodic) / scale
        phi = np.exp(-np.sum(du ** 2, axis=2) / (2 * sigma ** 2))
        G = -(phi[:, :, None] * du) / sigma ** 2            # (P, K, D)
        return np.transpose(G, (0, 2, 1)).reshape(-1, centers.shape[0])

    @classmethod
    def fit(cls, z, f, sigma, scale, periodic, ridge=1e-8, types=None):
        z, f = np.asarray(z, float), np.asarray(f, float)
        scale = np.asarray(scale, float)
        G = cls._design(z, z, sigma, scale, periodic)
        target = (-f * scale).reshape(-1)                   # dF/du = -f * scale
        A = G.T @ G
        A[np.diag_indices_from(A)] += ridge * np.trace(A) / len(A)
        coef = np.linalg.solve(A, G.T @ target)
        m = cls(z, coef, sigma, scale, periodic, types)
        m.offset = float(m.value(z).min())                  # lowest center = 0
        return m

    def force_error(self, z, f):
        """Relative error of the predicted mean force at points z:
        |pred - f| / |f|, both in scaled units, over all points."""
        pred = -self.gradient(z) * self.scale
        ref = np.asarray(f, float) * self.scale
        return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

    # -- persistence --------------------------------------------------------
    def save(self, path, extra=None):
        d = {"centers": self.centers, "coef": self.coef, "sigma": self.sigma, "scale": self.scale,
             "periodic": self.periodic.tolist(), "types": self.types, "offset": self.offset}
        d.update(extra or {})
        save_json(path, d)

    @classmethod
    def load(cls, path):
        d = load_json(path)
        return cls(d["centers"], d["coef"], d["sigma"], d["scale"], d["periodic"], d.get("types"), d["offset"])


def cross_validate(z, f, sigma, scale, periodic, folds=5, seed=0, ridge=1e-8):
    """Left-out force error: fit without each fold, predict the fold."""
    n = len(z)
    order = np.random.default_rng(seed).permutation(n)
    num = den = 0.0
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        m = FESMap.fit(z[train], f[train], sigma, scale, periodic, ridge)
        pred = -m.gradient(z[test]) * scale
        ref = f[test] * scale
        num += np.sum((pred - ref) ** 2)
        den += np.sum(ref ** 2)
    return float(np.sqrt(num / den))


def fit_with_sigma_scan(z, f, scale, periodic, sigmas=None, folds=5, ridge=1e-8, types=None):
    """Scan sigma, keep the one with the smallest left-out force error."""
    if sigmas is None:
        # nearest-neighbour spacing sets the natural range
        du = diff(z[:, None, :], z[None, :, :], periodic) / scale
        d = np.sqrt(np.sum(du ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        nn = np.median(d.min(axis=1))
        sigmas = nn * np.array([0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    scan = [(float(s), cross_validate(z, f, s, scale, periodic, folds, ridge=ridge)) for s in sigmas]
    best = min(scan, key=lambda t: t[1])
    return FESMap.fit(z, f, best[0], scale, periodic, ridge, types), scan, best[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--forces", required=True, help="centers-and-forces file (Reconstruct format)")
    p.add_argument("--scales", required=True)
    p.add_argument("--sigmas", type=float, nargs="+", help="scaled widths to scan (default: from center spacing)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-error", type=float, default=0.5,
                   help="gate: largest tolerated left-out relative force error [0.5]")
    p.add_argument("--out", default="map.json")
    p = sub.add_parser("eval")
    p.add_argument("--map", required=True)
    p.add_argument("--points", required=True, help="text file, one CV-space point per line")
    a = ap.parse_args(argv)

    if a.cmd == "fit":
        z, f, types = read_forces(a.forces)
        scale = np.array(load_json(a.scales)["scale"])
        if len(scale) != z.shape[1]:
            sys.exit(f"{len(scale)} scales for a {z.shape[1]}-dimensional CV space")
        periodic = np.array([t in PERIODIC_TYPES for t in types])
        m, scan, err = fit_with_sigma_scan(z, f, scale, periodic,
                                           np.array(a.sigmas) if a.sigmas else None, a.folds, types=types)
        print("sigma (scaled)   left-out force error")
        for s, e in scan:
            print(f"  {s:10.4f}   {e:8.4f}{'   <- chosen' if s == m.sigma else ''}")
        F = m.value(z)
        print(f"{len(z)} centers, F at centers spans {F.min():.2f} to {F.max():.2f} kcal/mol")
        ok = err <= a.max_error
        print(f"GATE map quality: {'PASS' if ok else 'FAIL'} (left-out force error {err:.3f}, limit {a.max_error})")
        m.save(a.out, {"sigma_scan": scan, "leftout_force_error": err})
        print(f"wrote {a.out}")
        return 0 if ok else 1

    m = FESMap.load(a.map)
    pts = np.atleast_2d(np.loadtxt(a.points))
    for p, v, d in zip(pts, m.value(pts), m.nearest_center_distance(pts)):
        print(" ".join(f"{x:10.5f}" for x in p), f"{v:10.4f} {d:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
