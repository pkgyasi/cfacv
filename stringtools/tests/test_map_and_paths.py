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
    assert len(ch) == 2, [c["barrier"] for c in ch]
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
    assert not res["unassigned"]
    assert sum(c["random_hits"] for c in ch) == 20
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
