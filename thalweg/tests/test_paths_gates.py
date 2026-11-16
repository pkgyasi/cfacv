"""Gates of the map fit and of the path search, on cases built to break them
(review 3: the map gate, duplicated centers, the support guard, the refit
statistics, channel splitting, disconnected basins, the tangent bias)."""
import json

import numpy as np
import pytest

import fesmap
import path_search as ps
from common import write_forces
from fesmap import FESMap, cross_validate, fit_anisotropic, fit_with_sigma_scan


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def metric_cv_error(z, f, width, metric, periodic, folds=5, seed=123):
    """Left-out force error of Gaussians of `width`, measured in `metric`
    units, on a fold split of its own: an independent ruler for the gate."""
    order = np.random.default_rng(seed).permutation(len(z))
    num = den = 0.0
    for k in range(folds):
        test = order[k::folds]; train = np.setdiff1d(order, test)
        m = FESMap.fit(z[train], f[train], 1.0, width, periodic)
        num += np.sum(((-m.gradient(z[test]) - f[test]) * metric) ** 2)
        den += np.sum((f[test] * metric) ** 2)
    return float(np.sqrt(num / den))


def bad_cv_data(seed=0, n=200):
    """CV 0 carries a smooth, well-measured force; CV 1 carries noise as
    large as the signal.  Both are equally important in path units."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-2, 2, (n, 2))
    f = np.column_stack([-2.0 * np.sin(z[:, 0]) * 1.5, rng.normal(0, 2.0, n)])
    return z, f


def ring_polar(n, seed, noise, h=4.0, half=None, gap=None):
    """The reviewer's ring (break/g2/mk.py): F = 20 (r-1)^2 + h sin^2 th,
    centred on (2, 2); minima at th = 0 and pi; two saddles at th = +-pi/2
    of the SAME height h."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.6, 1.4, n * 3); th = rng.uniform(-np.pi, np.pi, n * 3)
    if half == "upper":
        th = np.abs(th)
    if gap is not None:
        keep = np.abs(np.abs(th) - np.pi / 2) > gap
        r, th = r[keep], th[keep]
    r, th = r[:n], th[:n]
    x, y = r * np.cos(th) + 2, r * np.sin(th) + 2
    dFdr = 40 * (r - 1); dFdth = h * np.sin(2 * th)
    fx = -(dFdr * np.cos(th) - dFdth * np.sin(th) / r); fy = -(dFdr * np.sin(th) + dFdth * np.cos(th) / r)
    f = np.c_[fx, fy] + rng.normal(0, noise, (n, 2))
    return np.c_[x, y], f


def write_case(tmp_path, name, z, f, scale=(0.2, 0.2), types=("BOND", "BOND")):
    write_forces(str(tmp_path / f"{name}.dat"), z, f, list(types), "test")
    (tmp_path / "s.json").write_text(json.dumps({"scale": list(scale)}))
    (tmp_path / "b.json").write_text(json.dumps({"closed": {"centre": [1, 2]}, "open": {"centre": [3, 2]}}))
    return str(tmp_path / f"{name}.dat")


# --------------------------------------------------------------------------
# 1. the map gate is measured in path units, and is not the minimum of a scan
# --------------------------------------------------------------------------
def test_gate_is_measured_in_the_path_metric_not_the_trial_widths():
    z, f = bad_cv_data()
    metric = np.ones(2); per = np.zeros(2, bool)
    m, record, err = fit_anisotropic(z, f, metric, per)
    honest = metric_cv_error(z, f, m.scale, metric, per)
    # the reported gate value may not undercut an independent left-out error
    # of the very map it describes, measured in the units paths are measured in
    assert err > 0.9 * honest, (err, honest, record)


def test_cross_validate_ignores_the_width_when_measuring_the_error():
    """Same fit, error measured twice with different metrics: the width must
    not enter the ruler."""
    z, f = bad_cv_data(1)
    per = np.zeros(2, bool); w = np.array([0.8, 0.8])
    e1 = cross_validate(z, f, 1.0, w, per, metric=np.ones(2))
    w2 = np.array([0.8, 0.2])
    # narrowing the badly predicted CV's Gaussian must not make the ruler kinder
    e2 = cross_validate(z, f, 1.0, w2, per, metric=np.ones(2))
    assert e2 > 0.8 * e1, (e1, e2)


# --------------------------------------------------------------------------
# 2. duplicated centers, too few centers, bad folds, bad scales
# --------------------------------------------------------------------------
def smooth_data(n=120, seed=4):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-2, 2, (n, 2))
    f = np.column_stack([-2 * z[:, 0] + 0.5 * np.cos(z[:, 1]), -3 * z[:, 1]])
    return z, f


def test_duplicated_centers_give_a_finite_gate():
    z, f = smooth_data()
    zz, ff = np.vstack([z, z]), np.vstack([f, f])
    per = np.zeros(2, bool)
    for fit in (fit_with_sigma_scan, fit_anisotropic):
        m, _, err = fit(zz, ff, np.ones(2), per)
        assert np.isfinite(err) and err < 0.5, (fit.__name__, err)


def test_duplicated_noise_does_not_pass_the_gate():
    """A rerun appended to its own data (or noise duplicated a hair apart)
    must not let a center predict its own copy."""
    rng = np.random.default_rng(5)
    z = rng.uniform(-2, 2, (120, 2)); fn = rng.normal(0, 1, (120, 2))
    per = np.zeros(2, bool)
    for zz in (np.vstack([z, z]), np.vstack([z, z + 1e-3])):
        m, _, err = fit_anisotropic(zz, np.vstack([fn, fn]), np.ones(2), per)
        assert np.isfinite(err) and err > 0.5, err


@pytest.mark.parametrize("n,folds", [(1, 5), (2, 5), (5, 5), (60, 1)])
def test_fit_refuses_too_few_centers_or_folds_cleanly(tmp_path, n, folds):
    z, f = smooth_data(60)
    p = write_case(tmp_path, "few", z[:n], f[:n])
    with pytest.raises(SystemExit) as e:
        fesmap.main(["fit", "--forces", p, "--scales", str(tmp_path / "s.json"), "--folds", str(folds),
                     "--out", str(tmp_path / "m.json")])
    assert isinstance(e.value.code, str) and ("center" in e.value.code or "fold" in e.value.code), e.value.code


@pytest.mark.parametrize("scale", [(0.0, 0.2), (-0.2, 0.2), (float("nan"), 0.2)])
def test_fit_refuses_a_bad_scale(tmp_path, scale):
    z, f = smooth_data(60)
    p = write_case(tmp_path, "ok", z, f, scale=scale)
    with pytest.raises(SystemExit) as e:
        fesmap.main(["fit", "--forces", p, "--scales", str(tmp_path / "s.json"), "--out", str(tmp_path / "m.json")])
    assert isinstance(e.value.code, str) and "scale" in e.value.code, e.value.code


def test_fit_refuses_nan_forces(tmp_path):
    z, f = smooth_data(60); f[3, 0] = np.nan
    p = write_case(tmp_path, "nan", z, f)
    with pytest.raises(SystemExit) as e:
        fesmap.main(["fit", "--forces", p, "--scales", str(tmp_path / "s.json"), "--out", str(tmp_path / "m.json")])
    assert isinstance(e.value.code, str) and "finite" in e.value.code, e.value.code
