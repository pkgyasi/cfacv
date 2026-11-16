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


def test_fit_refuses_all_zero_forces(tmp_path):
    z, f = smooth_data(60)
    p = write_case(tmp_path, "zero", z, 0 * f)
    with pytest.raises(SystemExit) as e:
        fesmap.main(["fit", "--forces", p, "--scales", str(tmp_path / "s.json"), "--out", str(tmp_path / "m.json")])
    assert isinstance(e.value.code, str) and "force is zero" in e.value.code, e.value.code


# --------------------------------------------------------------------------
# analytic surfaces for the relaxation (the reviewer's ana.py / ana2.py)
# --------------------------------------------------------------------------
class Analytic(FESMap):
    """F(y) with y = M^-1/2 u, a constant tensor M; the minimum free-energy
    path is the image of the unit upper semicircle."""

    def __init__(self, Fy, gy, Mc, centers_y, useM=True):
        self.Fy, self.gy = Fy, gy
        self.metric = np.ones(2); self.scale = np.ones(2); self.periodic = np.zeros(2, bool)
        self.mobility_width = 0.5; self.offset = 0.0
        w, V = np.linalg.eigh(Mc)
        self.Mh = (V * np.sqrt(w)) @ V.T; self.Mih = (V / np.sqrt(w)) @ V.T
        self.centers = np.asarray(centers_y) @ self.Mh
        self.M = np.broadcast_to(Mc, (len(self.centers), 2, 2)).copy() if useM else None

    def value(self, z):
        return self.Fy(np.atleast_2d(z) @ self.Mih)

    def gradient(self, z):
        return self.gy(np.atleast_2d(z) @ self.Mih) @ self.Mih

    def truth(self):
        th = np.linspace(0, np.pi, 20001)
        return np.c_[np.cos(th), np.sin(th)] @ self.Mh


def ring_y(K=20.0, A=3.0, B=50.0):
    def Fy(y):
        r = np.linalg.norm(y, axis=1); w = np.clip(0.6 - r, 0, None)
        return K * (r - 1) ** 2 + A * (y[:, 1] / r) ** 2 + B * w ** 3

    def gy(y):
        r = np.linalg.norm(y, axis=1)[:, None]; rh = y / r; th = np.c_[-rh[:, 1], rh[:, 0]]
        s = rh[:, 1:2]; c = rh[:, 0:1]; w = np.clip(0.6 - r, 0, None)
        return (2 * K * (r - 1) - 3 * B * w ** 2) * rh + (2 * A * s * c / r) * th
    return Fy, gy


D_UP = 0.0203          # rms distance to the true path, 20 images, upwind only (2891a69)
ROT = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])


def full_centers():
    g = np.linspace(-1.5, 1.5, 13)
    return np.array([(a, b) for a in g for b in g if 0.5 < np.hypot(a, b) < 1.5 and b > -0.2])


# --------------------------------------------------------------------------
# 3. an image held back at the edge of the data is not a converged string
# --------------------------------------------------------------------------
def test_string_frozen_at_the_data_edge_is_not_converged():
    Fy, gy = ring_y()
    g = np.linspace(-1.5, 1.5, 61)
    inner = np.array([(a, b) for a in g for b in g if 0.3 < np.hypot(a, b) < 0.85 and b > -0.2])
    m = Analytic(Fy, gy, np.eye(2), np.vstack([inner, [[1, 0], [-1, 0], [0.95, 0.05], [-0.95, 0.05]]]), useM=False)
    th = np.linspace(0, np.pi, 7); start = np.c_[0.75 * np.cos(th), 0.75 * np.sin(th)]
    start[0] = [1, 0]; start[-1] = [-1, 0]
    p, it, conv = ps.relax_string(m, start, 20, support=0.1, cut=None)
    # the true path is the unit circle, out of the data's reach: this string is held, not relaxed
    assert ps._to_curve(p, m.truth(), np.ones(2)).max() > 0.1
    assert not conv
    assert ps.relax_string.info["frozen"]
    # with no support limit the same string converges onto the circle
    p, it, conv = ps.relax_string(m, start, 20, support=None, cut=None)
    assert conv and not ps.relax_string.info["frozen"]


# --------------------------------------------------------------------------
# 4/5. the refit statistics
# --------------------------------------------------------------------------
def ring_map_with_tensor(noise=0.3, seed=2):
    from test_map_and_paths import ring_centers
    P, V, g = ring_centers()
    f = -g + np.random.default_rng(seed).normal(0, noise, g.shape)
    m, _, _ = fit_with_sigma_scan(P, f, np.ones(2), np.zeros(2, bool))
    m.M = np.broadcast_to(np.diag([1.0, 3.0]), (len(m.centers), 2, 2)).copy()
    return m, P, f


def test_refits_relax_with_the_tensor_and_the_support_the_search_used(monkeypatch):
    m, P, f = ring_map_with_tensor()
    res = ps.search(m, np.array([-1.0, 0.0]), np.array([1.0, 0.0]), n_images=16, n_channels=2, n_random=0)
    seen = []
    orig = ps.relax_string

    def spy(mb, *a, **k):
        seen.append((mb.M is not None and len(mb.M) == len(mb.centers), k.get("support")))
        return orig(mb, *a, **k)
    monkeypatch.setattr(ps, "relax_string", spy)
    ps.barrier_uncertainty(m, P, f, res["channels"], n_boot=3, relax=True)
    assert seen and all(t for t, _ in seen), seen
    # the support the search uses by default (twice the median center spacing)
    assert all(s is not None and abs(s - res["support"]) < 1e-12 for _, s in seen), (seen, res["support"])


def test_a_refit_string_that_jumps_channel_is_counted_as_a_jump_not_a_hold():
    from test_map_and_paths import ring_centers
    P, V, g = ring_centers()
    f = -g + np.random.default_rng(2).normal(0, 0.3, g.shape)
    m, _, _ = fit_with_sigma_scan(P, f, np.ones(2), np.zeros(2, bool))
    res = ps.search(m, np.array([-1.0, 0.0]), np.array([1.0, 0.0]), n_images=16, n_channels=1, n_random=0)
    ch0 = res["channels"][0]
    # a "channel" that is really channel 0 placed off its route: every relaxation takes it back there
    fake = dict(ch0)
    fake["anchors"] = ch0["anchors"] + np.c_[np.zeros(16), 0.35 * np.sin(np.linspace(0, np.pi, 16))]
    fake["barrier"] = ch0["barrier"] + 2.0
    # channel identity by route, strictly: the fake is not channel 0's route
    out = ps.barrier_uncertainty(m, P, f, [ch0, fake], n_boot=6, relax=True,
                                 same=lambda a, b: ps.same_route(a, b, m.metric, 0.1))
    assert out[1][0] < 0.2, out[1]                  # a jump back onto channel 0 is not "channel 0 is lower"
    st = ps.barrier_uncertainty.stats
    assert st["jumped"][1] > 0.8, st


def test_refit_barrier_is_read_like_the_main_barrier():
    """The main barrier is the highest F read BETWEEN the images.  A coarse
    path whose top falls between its images (main barrier ~5) against a
    dense one (~4): the dense one is lower, and refits must say so instead
    of reading the coarse one at its images (~3)."""
    from test_map_and_paths import ring
    xs = np.arange(-1.5, 1.5 + 1e-9, 0.12)
    P = np.array([(x, y) for x in xs for y in xs])
    V, g = ring(P, hi_upper=2.0, hi_lower=1.0)
    keep = V < 9.0; P, g = P[keep], g[keep]
    m, _, _ = fit_with_sigma_scan(P, -g, np.ones(2), np.zeros(2, bool))
    coarse = np.array([[-1.0, 0.0], [-0.6, 0.8], [0.6, 0.8], [1.0, 0.0]])
    th = np.linspace(np.pi, 2 * np.pi, 41); dense = np.c_[np.cos(th), np.sin(th)]
    F0 = m.value(coarse[:1])[0]
    bc, bd = ps.fine_top(m, coarse) - F0, ps.fine_top(m, dense) - F0
    assert bd < bc - 0.5 and m.value(coarse).max() - F0 < bd - 0.5, (bc, bd)
    chans = [{"anchors": coarse, "barrier": bc}, {"anchors": dense, "barrier": bd}]
    sd, holds = ps.barrier_uncertainty(m, P, -g, chans, n_boot=10, relax=False)[:2]
    assert holds[0] < 0.2, holds


def _verdict(h, n):
    if hasattr(ps, "ranking_verdict"):
        return ps.ranking_verdict(h, n)
    be = np.sqrt(max(h * (1 - h), 1e-4) / n)               # the verdict main() printed at 2891a69
    return ("the order is firm" if h - 2 * be >= 0.9 else
            ("the order is likely" if h - 2 * be >= 0.7 else "the order is NOT established by this map"))


def test_equal_barriers_are_not_called_ordered():
    """The reviewer's g2 ring: two arcs with IDENTICAL saddles, forces with
    noise 3.  The ranking must come out 'NOT established' in most seeds."""
    verdicts = []
    for seed in range(4):
        z, f = ring_polar(300, seed, 3.0)
        m, _, _ = fit_with_sigma_scan(z, f, np.array([0.2, 0.2]), np.zeros(2, bool))
        res = ps.search(m, np.array([1.0, 2.0]), np.array([3.0, 2.0]), n_images=16, n_channels=2, n_random=0)
        if len(res["channels"]) < 2:
            continue
        sd, holds = ps.barrier_uncertainty(m, z, f, res["channels"], n_boot=40, relax=False, seed=seed)[:2]
        verdicts.append(_verdict(holds[0], 40))
    assert len(verdicts) >= 3
    assert sum(v.startswith("the order is NOT") for v in verdicts) >= 0.75 * len(verdicts), verdicts


# --------------------------------------------------------------------------
# 6. one arc is one channel
# --------------------------------------------------------------------------
def ring_search(z, f, n_random=12, seed=0):
    m, _, _ = fit_with_sigma_scan(z, f, np.array([0.2, 0.2]), np.zeros(2, bool))
    return ps.search(m, np.array([1.0, 2.0]), np.array([3.0, 2.0]), n_images=16, n_channels=3,
                     n_random=n_random, seed=seed)


def test_one_arc_is_one_channel_and_two_arcs_are_two():
    single = ring_search(*ring_polar(300, 0, 0.5, half="upper"))
    assert len(single["channels"]) == 1, [round(c["barrier"], 2) for c in single["channels"]]
    two = ring_search(*ring_polar(300, 0, 0.0))
    ys = [c["anchors"][:, 1].mean() - 2.0 for c in two["channels"]]
    assert len(two["channels"]) == 2 and min(ys) < -0.3 and max(ys) > 0.3, ys


# --------------------------------------------------------------------------
# 7. basins the centers do not connect
# --------------------------------------------------------------------------
def test_disconnected_basins_exit_with_a_message(tmp_path, capsys):
    z, f = ring_polar(300, 0, 0.5, gap=0.5)
    m, _, _ = fit_with_sigma_scan(z, f, np.array([0.2, 0.2]), np.zeros(2, bool))
    m.save(str(tmp_path / "m.json"))
    write_case(tmp_path, "gap", z, f)
    rc = ps.main(["--map", str(tmp_path / "m.json"), "--basins", str(tmp_path / "b.json"), "--random", "2",
                  "--out", str(tmp_path / "ch")])
    assert rc == 1
    assert "no route" in capsys.readouterr().out


# --------------------------------------------------------------------------
# 8. the upwind tangent's O(spacing) bias, and the central finishing pass
# --------------------------------------------------------------------------
def test_central_finish_removes_most_of_the_upwind_bias():
    Fy, gy = ring_y(K=20.0)
    m = Analytic(Fy, gy, ROT @ np.diag([1, 25.0]) @ ROT.T, full_centers())
    y0 = np.array([[1, 0], [0.9, 0.9], [-0.9, 0.9], [-1, 0.0]])
    p, _, c = ps.relax_string(m, y0 @ m.Mh, 20, cut=None, finish="central")
    assert c
    d = np.sqrt(np.mean(ps._to_curve(p, m.truth(), np.ones(2)) ** 2))
    # upwind tangents alone leave it D_UP from the true path (measured at 2891a69)
    assert d < 0.5 * D_UP, d
    assert ps.relax_string.info.get("finish") == "central"


# --------------------------------------------------------------------------
# 9. the smaller things
# --------------------------------------------------------------------------
def test_help_does_not_deny_the_tensor(capsys):
    with pytest.raises(SystemExit):
        ps.main(["-h"])
    out = capsys.readouterr().out
    assert "not the metric tensor" not in out


def test_across_metric_doc_does_not_claim_a_preconditioner():
    assert "as a preconditioner" not in ps.across_metric.__doc__


def test_a_random_string_matches_its_nearest_channel_not_the_first():
    a = np.c_[np.linspace(0, 1, 9), np.zeros(9)]
    cands = [{"anchors": a + [0, 0.08]}, {"anchors": a + [0, 0.01]}]
    assert ps.nearest_match(a, cands, np.ones(2)) is cands[1]


def test_spur_retry_also_starts_tensor_free(monkeypatch):
    Fy, gy = ring_y(K=20.0)
    m = Analytic(Fy, gy, ROT @ np.diag([1, 25.0]) @ ROT.T, full_centers())
    calls = []
    orig = ps._relax

    def spy(mm, *a, **k):
        calls.append(mm.M is None)
        return orig(mm, *a, **k)
    monkeypatch.setattr(ps, "_relax", spy)
    y0 = np.array([[1, 0], [0.9, 0.9], [-0.9, 0.9], [-1, 0.0]])
    ps.relax_string(m, y0 @ m.Mh, 12, cut=0.999)      # every turn counts as a spur
    assert ps.relax_string.info["cut"] or ps.relax_string.info["through_minimum"]
    # pre-stage without the tensor, then with it; and the retry the same way
    assert calls[:4] == [True, False, True, False], calls


def test_relaxation_reports_chatter_convergence():
    Fy, gy = ring_y(K=20.0)
    m = Analytic(Fy, gy, np.eye(2), full_centers())
    y0 = np.array([[1, 0], [0.9, 0.9], [-0.9, 0.9], [-1, 0.0]])
    ps.relax_string(m, y0 @ m.Mh, 12, cut=None)
    assert "chatter" in ps.relax_string.info


def test_random_strings_beyond_the_support_are_counted():
    res = ring_search(*ring_polar(300, 0, 0.5, half="upper"), n_random=12)
    assert "beyond_support_random" in res
    assert res["beyond_support_random"] + res["unconverged_random"] + \
        sum(c["random_hits"] for c in res["channels"]) + len(res["stray_barriers"]) == 12
