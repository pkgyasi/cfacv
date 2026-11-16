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


def check_inputs(z, f, scale, periodic, folds):
    """Refuse, with a message, the inputs a fit cannot mean anything on."""
    z, f = np.asarray(z, float), np.asarray(f, float)
    scale = np.asarray(scale, float)
    if not (np.all(np.isfinite(scale)) and np.all(scale > 0)):
        raise ValueError(f"every scale must be a finite positive length, got {scale.tolist()}")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(f))):
        bad = np.where(~(np.isfinite(z).all(axis=1) & np.isfinite(f).all(axis=1)))[0]
        raise ValueError(f"centers or forces are not finite on {len(bad)} rows (first: row {bad[0]})")
    if int(folds) < 2:
        raise ValueError(f"--folds {folds}: a left-out error needs at least 2 folds")
    if len(z) < 2 * int(folds):
        raise ValueError(f"{len(z)} distinct centers: too few for {folds}-fold cross-validation inside "
                         f"{folds}-fold cross-validation (at least {2 * int(folds)} wanted)")


def merge_duplicates(z, f, metric, periodic, tol=1e-6):
    """Centers closer than `tol` metric units are one center: their forces
    are averaged.  (A center listed twice, by a dual system or an appended
    rerun, makes the fit singular, sets the default width to zero, and in
    cross-validation lets a center predict its own copy.)
    Returns (z, f, number of rows merged away)."""
    z, f = np.asarray(z, float), np.asarray(f, float)
    lab = _groups(z, metric, periodic, tol)
    if lab.max() + 1 == len(z):
        return z, f, 0
    k = lab.max() + 1
    first = np.array([np.where(lab == g)[0][0] for g in range(k)])
    fz = np.array([f[lab == g].mean(axis=0) for g in range(k)])
    return z[first], fz, len(z) - k


def _groups(z, metric, periodic, tol):
    """Label centers joined by chains of distances below `tol` (metric units)."""
    du = diff(z[:, None, :], z[None, :, :], periodic) / metric
    near = np.sqrt(np.sum(du ** 2, axis=2)) < tol
    lab = -np.ones(len(z), int); k = 0
    for i in range(len(z)):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = k
        while stack:
            j = stack.pop()
            for q in np.where(near[j] & (lab < 0))[0]:
                lab[q] = k; stack.append(q)
        k += 1
    return lab


# centers this close (metric units) always fall in the same fold
TIE = 0.05


def fold_split(z, metric, periodic, folds, seed):
    """Fold of every center.  Centers within TIE metric units of each other go
    to the same fold, so that a near-copy of a center (a rerun a hair away)
    cannot predict it."""
    lab = _groups(np.asarray(z, float), np.asarray(metric, float), periodic, TIE)
    order = np.random.default_rng(seed).permutation(lab.max() + 1)
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    return rank[lab] % folds


def cross_validate(z, f, sigma, scale, periodic, folds=5, seed=0, ridge=1e-8, metric=None, residuals=False):
    """Left-out force error: fit without each fold, predict the fold.

    The Gaussians are shaped by `scale` (times sigma); the error is measured
    in `metric` units (default: `scale`), the units paths are measured in.
    The two must not be confused: measured in a candidate's own widths, the
    error of a CV is weighted by that CV's trial width, and narrowing the
    Gaussian of a badly predicted CV lowers the error without predicting
    anything better.  With residuals=True, also returns the left-out
    residual (predicted - measured) force at every center, in raw units."""
    z, f = np.asarray(z, float), np.asarray(f, float)
    scale = np.asarray(scale, float)
    metric = scale if metric is None else np.asarray(metric, float)
    fold = fold_split(z, metric, periodic, folds, seed)
    res = np.zeros_like(f)
    for k in range(folds):
        test = np.where(fold == k)[0]
        train = np.where(fold != k)[0]
        if len(test) == 0:
            continue
        if len(train) < 2:
            raise ValueError(f"{len(z)} centers: too few for {folds}-fold cross-validation")
        m = FESMap.fit(z[train], f[train], sigma, scale, periodic, ridge)
        res[test] = -m.gradient(z[test]) - f[test]
    den = np.sum((f * metric) ** 2)
    err = float(np.sqrt(np.sum((res * metric) ** 2) / den)) if den > 0 else np.inf
    if not np.isfinite(err):
        err = np.inf
    return (err, res) if residuals else err


def default_sigmas(z, scale, periodic):
    """Scaled widths to scan: multiples of the median nearest-neighbour spacing
    of the DISTINCT centers (a duplicated center is at distance 0 from its copy)."""
    du = diff(z[:, None, :], z[None, :, :], periodic) / scale
    d = np.sqrt(np.sum(du ** 2, axis=2))
    d[d < 1e-9] = np.inf
    nn = d.min(axis=1)
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        raise ValueError("all centers coincide")
    return float(np.median(nn)) * np.array([0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _select_isotropic(z, f, scale, periodic, sigmas, folds, ridge, metric, seed=0):
    if sigmas is None:
        sigmas = default_sigmas(z, scale, periodic)
    scan = [(float(s), cross_validate(z, f, s, scale, periodic, folds, seed, ridge, metric)) for s in sigmas]
    best = min(scan, key=lambda t: t[1])
    return best[0], scan, best[1]


def _select_anisotropic(z, f, scale, periodic, factors, passes, folds, ridge, metric, sigmas, seed=0):
    sigma, scan, err = _select_isotropic(z, f, scale, periodic, sigmas, folds, ridge, metric, seed)
    width = scale * sigma
    record = [("isotropic", width.tolist(), err)]
    for _ in range(passes):
        changed = False
        for j in range(len(scale)):
            best = (1.0, err)
            for fac in factors:
                if fac == 1.0:
                    continue
                w = width.copy(); w[j] *= fac
                e = cross_validate(z, f, 1.0, w, periodic, folds, seed, ridge, metric)
                if e < best[1]:
                    best = (fac, e)
            if best[0] != 1.0:
                width[j] *= best[0]; err = best[1]; changed = True
                record.append((f"cv {j} x {best[0]}", width.tolist(), err))
        if not changed:
            break
    return width, record, err


def nested_error(z, f, periodic, folds, metric, select, ridge=1e-8, seed=1):
    """Left-out force error of the whole procedure, width selection included
    (nested cross-validation): the widths are chosen again without each
    outer fold, by `select(z_train, f_train)` -> per-CV widths, and the
    resulting map is asked for the forces of that fold.  The best of many
    widths scored on one set of folds is an optimistic number for any of
    them; this one is not.  Measured in `metric` units."""
    fold = fold_split(z, metric, periodic, folds, seed)
    res = np.zeros_like(f)
    for k in range(folds):
        test = np.where(fold == k)[0]; train = np.where(fold != k)[0]
        if len(test) == 0:
            continue
        w = select(z[train], f[train])
        m = FESMap.fit(z[train], f[train], 1.0, w, periodic, ridge)
        res[test] = -m.gradient(z[test]) - f[test]
    err = float(np.sqrt(np.sum((res * metric) ** 2) / np.sum((f * metric) ** 2)))
    return err if np.isfinite(err) else np.inf


def _prepare(z, f, scale, periodic, folds):
    z, f = np.asarray(z, float), np.asarray(f, float)
    scale = np.asarray(scale, float)
    check_inputs(z, f, scale, periodic, folds)
    z, f, merged = merge_duplicates(z, f, scale, periodic)
    check_inputs(z, f, scale, periodic, folds)
    return z, f, scale, merged


def fit_with_sigma_scan(z, f, scale, periodic, sigmas=None, folds=5, ridge=1e-8, types=None):
    """Scan sigma, keep the one with the smallest left-out force error.
    Returns (map, scan, gate error); the gate error is the nested left-out
    error of the scan (see `nested_error`), not the scan's minimum.
    Duplicated centers are merged first (fit_with_sigma_scan.merged)."""
    z, f, scale, merged = _prepare(z, f, scale, periodic, folds)
    fit_with_sigma_scan.merged = merged
    sigma, scan, _ = _select_isotropic(z, f, scale, periodic, sigmas, folds, ridge, scale)
    err = nested_error(z, f, periodic, folds, scale,
                       lambda zt, ft: scale * _select_isotropic(zt, ft, scale, periodic, sigmas, folds, ridge, scale)[0],
                       ridge)
    return FESMap.fit(z, f, sigma, scale, periodic, ridge, types), scan, err


def fit_anisotropic(z, f, scale, periodic, factors=(0.5, 0.7, 1.0, 1.4, 2.0, 3.0), passes=2, folds=5,
                    ridge=1e-8, types=None, sigmas=None):
    """Choose a Gaussian width PER CV by cross-validation.

    The basin widths in `scale` are the right ruler for distances along a
    path, and the wrong shape for the Gaussians: the free energy is smooth
    along a stiff, nearly harmonic CV (wide Gaussian wanted, narrow basin) and
    can be sharp along a soft one.

    Starts from the best single width, then tries each factor on each CV in
    turn, keeping a change when it lowers the left-out force error.  That
    error is ALWAYS measured in `scale` units, whatever the trial widths.
    (Measured in the trial widths, as this did at first, narrowing a badly
    predicted CV's Gaussian down-weighted its error: on alanine dipeptide in
    four torsions it reported 0.23 for a map whose left-out error in path
    units is 0.36, and 0.21 for the mixed-CV map whose error is 0.50.)

    Returns (map, record, gate error); the map keeps `scale` as its distance
    metric.  The gate error is the nested left-out error of the whole
    selection (see `nested_error`): the last entry of `record` is the
    selection's own, optimistic, minimum.  Duplicated centers are merged
    first (fit_anisotropic.merged)."""
    z, f, scale, merged = _prepare(z, f, scale, periodic, folds)
    fit_anisotropic.merged = merged

    def select(zt, ft):
        return _select_anisotropic(zt, ft, scale, periodic, factors, passes, folds, ridge, scale, sigmas)

    width, record, _ = select(z, f)
    err = nested_error(z, f, periodic, folds, scale, lambda zt, ft: select(zt, ft)[0], ridge)
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
        fitter = fit_with_sigma_scan if a.isotropic else fit_anisotropic
        try:
            if a.isotropic:
                m, scan, err = fit_with_sigma_scan(z, f, scale, periodic, sig, a.folds, types=types)
            else:
                m, scan, err = fit_anisotropic(z, f, scale, periodic, folds=a.folds, types=types, sigmas=sig)
        except (ValueError, np.linalg.LinAlgError) as e:
            sys.exit(f"{a.forces}: {e}")
        if fitter.merged:
            print(f"{fitter.merged} centers duplicated others (within 1e-6 scaled units) and were merged into them, "
                  "forces averaged")
        z = m.centers
        if a.isotropic:
            m.metric = scale.copy()
            print("sigma (scaled)   left-out force error (selection folds)")
            for s, e in scan:
                print(f"  {s:10.4f}   {e:8.4f}{'   <- chosen' if s == m.sigma else ''}")
        else:
            print("Gaussian width per CV (raw units)             left-out force error (selection folds)")
            for what, w, e in scan:
                print("  " + " ".join(f"{v:8.3f}" for v in w) + f"   {e:8.4f}   {what}")
        print(f"left-out force error of the whole fit, width choice included (nested {a.folds}-fold), "
              f"in path units: {err:.3f}")
        if not np.isfinite(err):
            sys.exit(f"{a.forces}: the left-out force error is not finite: no map")
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
