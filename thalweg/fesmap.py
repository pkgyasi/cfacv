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

    def __init__(self, centers, coef, sigma, scale, periodic, types=None, offset=0.0, metric=None, M=None):
        self.centers = np.asarray(centers, float)
        self.coef = np.asarray(coef, float)
        self.sigma = float(sigma)
        # `scale` shapes the Gaussians.  `metric` is the per-CV length scale
        # used to measure distances along paths (the basin widths).  They are
        # different things: a smooth, stiff CV wants a wide Gaussian even
        # though its basin is narrow.
        self.scale = np.asarray(scale, float)
        self.metric = self.scale.copy() if metric is None else np.asarray(metric, float)
        self.periodic = np.asarray(periodic, bool)
        self.types = types
        self.offset = float(offset)
        # the metric tensor measured at each center (K, D, D), if it was; the
        # string moves along -M grad F
        self.M = None if M is None else np.asarray(M, float)
        # width, in metric units, of the average that `mobility` takes over the centers
        self.mobility_width = 0.5

    def set_metric_tensor(self, points, M):
        """Attach tensors measured at `points` to the centers they belong to."""
        points = np.asarray(points, float); M = np.asarray(M, float)
        d = np.abs(diff(self.centers[:, None, :], points[None, :, :], self.periodic)).max(axis=2)
        j = d.argmin(axis=1)
        if d[np.arange(len(j)), j].max() > 1e-4:
            raise ValueError("the metric tensors were not measured at this map's centers")
        self.M = M[j]

    def mobility(self, z):
        """Dimensionless tensor that turns the scaled gradient dF/du into a
        direction of motion in path units: S^-1 M S^-1 at the nearest center,
        divided by its typical size over all centers so that a step keeps its
        meaning.  The identity when no tensor was measured."""
        z = np.atleast_2d(np.asarray(z, float)); D = z.shape[1]
        if self.M is None:
            return np.broadcast_to(np.eye(D), (len(z), D, D))
        Mu = self.M / np.outer(self.metric, self.metric)
        Mu = Mu / np.median(np.trace(Mu, axis1=1, axis2=2) / D)
        dm = diff(z[:, None, :], self.centers[None, :, :], self.periodic) / self.metric
        d2 = np.sum(dm ** 2, axis=2)
        # a smooth average over the centers within about one metric unit, not
        # the nearest center's tensor alone: a field that jumps where the
        # nearest center changes gives a relaxation nothing to converge to
        w = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)) / self.mobility_width ** 2)
        w /= w.sum(axis=1, keepdims=True)
        return np.einsum("pk,kij->pij", w, Mu)

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
        """Distance to the nearest center, in metric (path) units."""
        z = np.atleast_2d(np.asarray(z, float))
        dm = diff(z[:, None, :], self.centers[None, :, :], self.periodic) / self.metric
        return np.sqrt(np.sum(dm ** 2, axis=2)).min(axis=1)

    # -- fitting ------------------------------------------------------------
    @staticmethod
    def _design(z_eval, centers, sigma, scale, periodic):
        """Matrix G with (G a) = scaled gradient of F at z_eval, flattened."""
        du = diff(z_eval[:, None, :], centers[None, :, :], periodic) / scale
        phi = np.exp(-np.sum(du ** 2, axis=2) / (2 * sigma ** 2))
        G = -(phi[:, :, None] * du) / sigma ** 2            # (P, K, D)
        return np.transpose(G, (0, 2, 1)).reshape(-1, centers.shape[0])

    @classmethod
    def fit(cls, z, f, sigma, scale, periodic, ridge=1e-8, types=None, metric=None):
        z, f = np.asarray(z, float), np.asarray(f, float)
        scale = np.asarray(scale, float)
        G = cls._design(z, z, sigma, scale, periodic)
        target = (-f * scale).reshape(-1)                   # dF/du = -f * scale
        A = G.T @ G
        A[np.diag_indices_from(A)] += ridge * np.trace(A) / len(A)
        coef = np.linalg.solve(A, G.T @ target)
        m = cls(z, coef, sigma, scale, periodic, types, metric=metric)
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
        d = {"centers": self.centers, "coef": self.coef, "sigma": self.sigma, "scale": self.scale, "metric": self.metric,
             "periodic": self.periodic.tolist(), "types": self.types, "offset": self.offset, "M": self.M}
        d.update(extra or {})
        save_json(path, d)

    @classmethod
    def load(cls, path):
        d = load_json(path)
        return cls(d["centers"], d["coef"], d["sigma"], d["scale"], d["periodic"], d.get("types"), d["offset"],
                   d.get("metric"), d.get("M"))


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


def fit_anisotropic(z, f, scale, periodic, factors=(0.5, 0.7, 1.0, 1.4, 2.0, 3.0), passes=2, folds=5,
                    ridge=1e-8, types=None, sigmas=None):
    """Choose a Gaussian width PER CV by cross-validation.

    The basin widths in `scale` are the right ruler for distances along a
    path, and the wrong shape for the Gaussians: the free energy is smooth
    along a stiff, nearly harmonic CV (wide Gaussian wanted, narrow basin) and
    can be sharp along a soft one.  On alanine dipeptide in four torsions,
    Gaussians shaped by the basin widths left 0.72 of the left-out force
    unexplained; shaping them per CV brought that to 0.2.

    Starts from the best single width, then tries each factor on each CV in
    turn, keeping a change when it lowers the left-out force error.
    Returns (map, record, left-out error); the map keeps `scale` as its
    distance metric."""
    scale = np.asarray(scale, float)
    m0, scan, err = fit_with_sigma_scan(z, f, scale, periodic, sigmas, folds, ridge, types)
    width = scale * m0.sigma
    record = [("isotropic", width.tolist(), err)]
    for _ in range(passes):
        changed = False
        for j in range(len(scale)):
            best = (1.0, err)
            for fac in factors:
                if fac == 1.0:
                    continue
                w = width.copy(); w[j] *= fac
                e = cross_validate(z, f, 1.0, w, periodic, folds, ridge=ridge)
                if e < best[1]:
                    best = (fac, e)
            if best[0] != 1.0:
                width[j] *= best[0]; err = best[1]; changed = True
                record.append((f"cv {j} x {best[0]}", width.tolist(), err))
        if not changed:
            break
    return FESMap.fit(z, f, 1.0, width, periodic, ridge, types, metric=scale), record, err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--forces", required=True, help="centers-and-forces file (Reconstruct format)")
    p.add_argument("--scales", required=True)
    p.add_argument("--sigmas", type=float, nargs="+", help="scaled widths to scan (default: from center spacing)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--isotropic", action="store_true",
                   help="one Gaussian width for every CV in units of its length scale (default: a width per CV)")
    p.add_argument("--max-error", type=float, default=0.5,
                   help="gate: largest tolerated left-out relative force error [0.5]")
    p.add_argument("--metric-tensor", help="metric.json from 'meanforce.py metric': stored with the map, and the "
                                           "path search then relaxes its strings along -M grad F")
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
        sig = np.array(a.sigmas) if a.sigmas else None
        if a.isotropic:
            m, scan, err = fit_with_sigma_scan(z, f, scale, periodic, sig, a.folds, types=types)
            m.metric = scale.copy()
            print("sigma (scaled)   left-out force error")
            for s, e in scan:
                print(f"  {s:10.4f}   {e:8.4f}{'   <- chosen' if s == m.sigma else ''}")
        else:
            m, scan, err = fit_anisotropic(z, f, scale, periodic, folds=a.folds, types=types, sigmas=sig)
            print("Gaussian width per CV (raw units)             left-out force error")
            for what, w, e in scan:
                print("  " + " ".join(f"{v:8.3f}" for v in w) + f"   {e:8.4f}   {what}")
        F = m.value(z)
        print(f"{len(z)} centers, F at centers spans {F.min():.2f} to {F.max():.2f} kcal/mol")
        ok = err <= a.max_error
        print(f"GATE map quality: {'PASS' if ok else 'FAIL'} (left-out force error {err:.3f}, limit {a.max_error})")
        if a.metric_tensor:
            mt = load_json(a.metric_tensor)
            try:
                m.set_metric_tensor(mt["points"], mt["M"])
            except ValueError as e:
                sys.exit(f"{a.metric_tensor}: {e}")
            print(f"metric tensor attached at {len(m.M)} centers")
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
