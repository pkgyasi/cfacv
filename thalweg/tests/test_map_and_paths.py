"""The map and the path search on surfaces whose answer is known."""
import numpy as np

from fesmap import FESMap, fit_with_sigma_scan
from path_search import search


def ring(p, hi_upper=2.0, hi_lower=5.0):
    """Two wells at (-1,0) and (1,0) joined by two arcs of a ring; the upper
    arc has the lower barrier.  Returns V and dV/dp (analytic)."""
    x, y = p[:, 0], p[:, 1]
    r = np.maximum(np.hypot(x, y), 1e-12)
    V = 20 * (r - 1) ** 2
    g = np.stack([40 * (r - 1) * x / r, 40 * (r - 1) * y / r], axis=1)
    for cx, cy, amp, w in ((-1, 0, -3, 0.15), (1, 0, -3, 0.15), (0, 1, hi_upper, 0.1), (0, -1, hi_lower, 0.1)):
        e = amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / w)
        V = V + e
        g[:, 0] += e * (-2 * (x - cx) / w)
        g[:, 1] += e * (-2 * (y - cy) / w)
    return V, g


def ring_centers(spacing=0.12, vmax=9.0):
    xs = np.arange(-1.5, 1.5 + 1e-9, spacing)
    P = np.array([(x, y) for x in xs for y in xs])
    V, g = ring(P)
    keep = V < vmax
    return P[keep], V[keep], g[keep]


def test_map_reproduces_surface_and_gate_error_is_small():
    P, V, g = ring_centers()
    rng = np.random.default_rng(1)
    f = -g + rng.normal(0, 0.3, g.shape)          # mean force with noise
    m, scan, err = fit_with_sigma_scan(P, f, np.ones(2), np.zeros(2, bool))
    assert err < 0.15, scan
    # differences of F, not F itself, are what the fit determines
    dv = (m.value(P) - m.value(P).min()) - (V - V.min())
    assert np.abs(dv).max() < 1.0
    assert np.sqrt(np.mean(dv ** 2)) < 0.3


def test_search_ranks_the_two_channels_and_finds_no_third():
    P, V, g = ring_centers()
    m, _, _ = fit_with_sigma_scan(P, -g, np.ones(2), np.zeros(2, bool))
    res = search(m, np.array([-1.0, 0.0]), np.array([1.0, 0.0]), n_images=24, n_channels=3, n_random=20)
    ch = res["channels"]
    # one route over the top; underneath, the route splits round its bump (inside and outside the ring),
    # which the search used to merge into one because it compared strings too coarsely
    assert 2 <= len(ch) <= 3, [c["barrier"] for c in ch]
    assert all(c["converged"] and not c["through_minimum"] for c in ch)
    # channel 0 goes over the top (y > 0), channel 1 underneath
    assert ch[0]["anchors"][:, 1].max() > 0.8 and ch[0]["anchors"][:, 1].min() > -0.3
    assert ch[1]["anchors"][:, 1].min() < -0.6 and ch[1]["anchors"][:, 1].max() < 0.3
    # the upper saddle is at +2 above wells at -3
    assert abs(ch[0]["barrier"] - 5.0) < 0.7
    # the lower route skirts its bump, so compare with the true surface
    # along the path the search returned instead of with a guessed saddle
    for c in ch:
        Vt, _ = ring(c["anchors"])
        assert abs(c["barrier"] - (Vt.max() - Vt[0])) < 0.7
    assert ch[0]["barrier"] < ch[1]["barrier"]
    assert res["unconverged_random"] == 0
    assert sum(c["random_hits"] for c in ch) == 20
    # every string starts at the closed well's minimum on the map, and barriers are measured from there
    assert all(np.allclose(c["anchors"][0], res["end_minima"][0]) and abs(c["F"][0]) < 1e-9 for c in ch)
    # images equally spaced, ends in the wells
    seg = np.linalg.norm(np.diff(ch[0]["anchors"], axis=0), axis=1)
    assert seg.std() / seg.mean() < 0.02
    assert np.linalg.norm(ch[0]["anchors"][0] - [-1, 0]) < 0.15
    assert np.linalg.norm(ch[0]["anchors"][-1] - [1, 0]) < 0.15


def test_four_dimensions_with_unequal_scales_and_a_torsion():
    """The ring lives in (CV0, CV3); CV3 is a torsion whose basin straddles
    +/-pi, CV0 is a distance thirty times larger in raw units.  CV1 and CV2
    are stiff bystanders."""
    P2, V, g2 = ring_centers(spacing=0.15)
    rng = np.random.default_rng(3)
    n = len(P2)
    scale = np.array([3.0, 1.0, 0.5, 0.1])
    z = np.empty((n, 4))
    z[:, 0] = 10 + scale[0] * P2[:, 0]
    z[:, 1] = rng.normal(0, 0.3, n)
    z[:, 2] = rng.normal(0, 0.15, n)
    raw3 = np.pi - 0.05 + scale[3] * P2[:, 1]      # ring centre next to the boundary
    z[:, 3] = (raw3 + np.pi) % (2 * np.pi) - np.pi
    dF = np.empty((n, 4))
    dF[:, 0] = g2[:, 0] / scale[0]
    dF[:, 1] = 8.0 * z[:, 1]
    dF[:, 2] = 30.0 * z[:, 2]
    dF[:, 3] = g2[:, 1] / scale[3]
    per = np.array([False, False, False, True])
    m, scan, err = fit_with_sigma_scan(z, -dF, scale, per)
    assert err < 0.3, scan
    closed = np.array([10 - 3.0, 0, 0, np.pi - 0.05])
    opened = np.array([10 + 3.0, 0, 0, np.pi - 0.05])
    res = search(m, closed, opened, n_images=20, n_channels=3, n_random=20)
    ch = res["channels"]
    assert len(ch) >= 2
    assert abs(ch[0]["barrier"] - 5.0) < 1.2 and ch[0]["barrier"] < ch[1]["barrier"]
    # the anchors are a CONTINUOUS path: no 2 pi jump between neighbours,
    # even though the torsion crosses +/-pi along the way
    for c in ch[:2]:
        assert np.abs(np.diff(c["anchors"][:, 3])).max() < 1.0
    a0 = ch[0]["anchors"]
    assert a0[:, 3].max() > np.pi                   # went over the boundary, unwrapped
    assert np.abs(a0[:, 1]).max() < 0.5 and np.abs(a0[:, 2]).max() < 0.3


def test_map_round_trips_through_json(tmp_path):
    P, V, g = ring_centers(spacing=0.25)
    m = FESMap.fit(P, -g, 0.3, np.ones(2), np.zeros(2, bool))
    m.save(tmp_path / "m.json")
    m2 = FESMap.load(tmp_path / "m.json")
    q = np.array([[0.2, 0.9], [-0.9, 0.1]])
    assert np.allclose(m.value(q), m2.value(q)) and np.allclose(m.gradient(q), m2.gradient(q))


def test_per_cv_widths_rescue_a_surface_the_basin_widths_misshape():
    """One CV is stiff and harmonic (narrow basin, smooth surface), the other
    soft with a sharp feature.  Gaussians shaped by the basin widths fit this
    badly; a width chosen per CV fits it well, and the path metric is kept."""
    from fesmap import fit_anisotropic
    rng = np.random.default_rng(11)
    n = 260
    z = np.column_stack([rng.uniform(-2.5, 2.5, n), rng.normal(0, 0.16, n)])
    # F = double well in x with a sharp barrier + stiff harmonic in y
    dF = np.column_stack([4 * (-2 * z[:, 0] / 0.09) * np.exp(-z[:, 0] ** 2 / 0.09) + 0.6 * z[:, 0], 23.0 * z[:, 1]])
    f = -dF + rng.normal(0, 0.2, dF.shape)
    basin = np.array([1.2, 0.16])                  # what endpoint analysis would report
    per = np.zeros(2, bool)
    iso, _, e_iso = fit_with_sigma_scan(z, f, basin, per)
    ani, record, e_ani = fit_anisotropic(z, f, basin, per)
    assert e_ani < 0.6 * e_iso, (e_iso, e_ani, record)
    assert np.allclose(ani.metric, basin)                       # distances still in basin units
    assert ani.scale[1] / ani.scale[0] > basin[1] / basin[0]    # relatively wider along the stiff CV


def test_ranking_uncertainty_and_unmatched_groups():
    from path_search import barrier_uncertainty, group_unassigned
    P, V, g = ring_centers()
    rng = np.random.default_rng(2)
    f = -g + rng.normal(0, 0.3, g.shape)
    m, _, _ = fit_with_sigma_scan(P, f, np.ones(2), np.zeros(2, bool))
    res = search(m, np.array([-1.0, 0.0]), np.array([1.0, 0.0]), n_images=20, n_channels=3, n_random=10)
    sd, holds = barrier_uncertainty(m, P, f, res["channels"], n_boot=15)
    assert len(sd) == len(res["channels"]) >= 2 and np.all(sd < 0.8) and np.all(sd > 0)
    # with the forces' own standard errors, the part of that error that is force noise comes back too
    sd3, holds3, noise = barrier_uncertainty(m, P, f, res["channels"], n_boot=15, f_se=np.full(f.shape, 0.3))
    assert len(noise) == len(sd3) and np.all(noise > 0)
    assert holds[0] >= 0.9                      # 5 against 8: the order is firm
    # two identical strays and one different one make two groups
    a = res["channels"][0]["anchors"]
    strays = [{"anchors": a + [0, 1.0], "barrier": 9.0}, {"anchors": a + [0, 1.01], "barrier": 9.1},
              {"anchors": a - [0, 2.0], "barrier": 12.0}]
    groups = group_unassigned(strays, np.ones(2), 0.3)
    assert [g["count"] for g in groups] == [2, 1] and groups[0]["barrier"] == 9.0


def test_close_barriers_are_reported_as_not_established():
    """Two channels with nearly the same barrier: the ranking must not be called firm."""
    from path_search import barrier_uncertainty
    xs = np.arange(-1.5, 1.5 + 1e-9, 0.15)
    P = np.array([(x, y) for x in xs for y in xs])
    V, g = ring(P, hi_upper=3.0, hi_lower=3.15)
    keep = V < 9.0
    P, g = P[keep], g[keep]
    rng = np.random.default_rng(7)
    f = -g + rng.normal(0, 1.0, g.shape)
    m, _, _ = fit_with_sigma_scan(P, f, np.ones(2), np.zeros(2, bool))
    res = search(m, np.array([-1.0, 0.0]), np.array([1.0, 0.0]), n_images=20, n_channels=3, n_random=6)
    assert len(res["channels"]) >= 2
    sd, holds = barrier_uncertainty(m, P, f, res["channels"], n_boot=20)
    assert holds[0] < 0.9


def test_spur_is_cut_and_curve_distance_ignores_a_shift_along_the_path():
    from path_search import cut_spurs, turn_cosines, path_distance
    from common import redistribute
    one = np.ones(2)
    # a straight path with a stretch that goes out sideways and comes back
    spur = np.array([[0, 0], [1, 0], [2, 0], [2, 0.9], [2, 0.05], [3, 0], [4, 0.0]], float)
    assert turn_cosines(spur, one).min() < -0.9
    cut = cut_spurs(spur, one)
    assert turn_cosines(cut, one).min() >= 0 and np.allclose(cut[0], spur[0]) and np.allclose(cut[-1], spur[-1])
    assert np.abs(cut[:, 1]).max() < 0.1                              # the excursion is gone
    gentle = np.array([[0, 0], [1, 0.2], [2, 0.5], [3, 1.2], [4, 2.0]], float)
    assert len(cut_spurs(gentle, one)) == 5
    # the same line, one string longer at one end: image-by-image they look apart, as curves they are not
    a = np.column_stack([np.linspace(0, 4, 16), np.zeros(16)])
    b = np.column_stack([np.linspace(0, 5, 16), np.zeros(16)])
    assert float(np.mean(np.linalg.norm(a - b, axis=1))) > 0.4
    assert path_distance(a, b[:13], one) < 0.05
    c = a + [0, 0.7]
    assert abs(path_distance(a, c, one) - 0.7) < 1e-9


def test_relaxation_keeps_its_ends_and_reports_convergence():
    P, V, g = ring_centers()
    m, _, _ = fit_with_sigma_scan(P, -g, np.ones(2), np.zeros(2, bool))
    from path_search import relax_string
    start = np.array([[-1.0, 0.0], [0.0, 1.3], [1.0, 0.0]])
    p, n, conv = relax_string(m, start, 15)
    assert conv and np.allclose(p[0], start[0]) and np.allclose(p[-1], start[-1])
    p2, n2, conv2 = relax_string(m, start, 15, max_iter=3)
    assert not conv2 and n2 == 3


def test_uncertainty_has_the_subsample_factor():
    """With half the centers left out each time the factor is 1; with a fifth left out it is 2."""
    from path_search import barrier_uncertainty
    P, V, g = ring_centers()
    rng = np.random.default_rng(2)
    f = -g + rng.normal(0, 0.3, g.shape)
    m, _, _ = fit_with_sigma_scan(P, f, np.ones(2), np.zeros(2, bool))
    res = search(m, np.array([-1.0, 0.0]), np.array([1.0, 0.0]), n_images=20, n_channels=2, n_random=0)
    assert len(res["channels"]) == 2                                  # a duplicate route no longer uses up the quota
    truth = []
    for s in range(25):                                               # the real spread: new noise each time
        fs = -g + np.random.default_rng(100 + s).normal(0, 0.3, g.shape)
        mb = type(m).fit(P, fs, m.sigma, m.scale, m.periodic, metric=m.metric)
        F = mb.value(np.asarray(res["channels"][0]["anchors"])); truth.append(F.max() - F[0])
    sd, _ = barrier_uncertainty(m, P, f, res["channels"], n_boot=40)
    assert 0.5 < sd[0] / np.std(truth, ddof=1) < 2.0                  # it was about 0.5 before the factor


def test_a_string_through_a_side_minimum_converges_and_is_flagged():
    """Three wells in a row with the middle one off to the side: A -> B -> C
    turns a sharp corner at B.  Cutting the corner out every iteration made a
    limit cycle (cut, fall back in, cut again) that never converged."""
    from path_search import relax_string, turn_cosines
    xs = np.arange(-2.0, 2.0 + 1e-9, 0.2); ys = np.arange(-0.6, 2.4 + 1e-9, 0.2)
    P = np.array([(x, y) for x in xs for y in ys])
    wells = [(-1.5, 0.0, 4.0), (0.0, 1.8, 5.0), (1.5, 0.0, 4.0)]
    def grad(p):
        g = np.zeros_like(p)
        for cx, cy, a in wells:
            d = p - [cx, cy]; e = a * np.exp(-np.sum(d * d, axis=1) / 0.5)
            g += (2 * d / 0.5) * e[:, None]
        return g
    m, _, _ = fit_with_sigma_scan(P, -grad(P), np.ones(2), np.zeros(2, bool))
    start = np.array([[-1.5, 0.0], [0.0, 0.3], [1.5, 0.0]])
    p, n, conv = relax_string(m, start, 14)
    assert conv and n < 4000
    if turn_cosines(p, np.ones(2)).min() < -0.5:                     # it went through B: say so
        assert relax_string.info["through_minimum"]
    assert np.allclose(p[0], start[0]) and np.allclose(p[-1], start[-1])


def test_descent_finds_the_basin_floor_and_distances_respect_periodicity():
    from path_search import descend, path_distance
    P, V, g = ring_centers()
    m, _, _ = fit_with_sigma_scan(P, -g, np.ones(2), np.zeros(2, bool))
    z = descend(m, np.array([-0.8, 0.25]))
    assert np.hypot(z[0] + 1.0, z[1]) < 0.08
    a = np.column_stack([np.linspace(0, 1, 9), np.linspace(2.5, 3.5, 9)])
    b = a - [0, 2 * np.pi]
    assert path_distance(a, b, np.ones(2)) > 5 and path_distance(a, b, np.ones(2), periodic=[False, True]) < 1e-9
