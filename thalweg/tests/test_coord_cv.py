"""COORD, the analysis-only coordination number (2026-11-17)."""
import numpy as np
import pytest


def _universe(pos, names, box=None):
    import MDAnalysis as mda
    u = mda.Universe.empty(len(pos), n_residues=len(pos), atom_resindex=np.arange(len(pos)), trajectory=True)
    u.add_TopologyAttr("name", names); u.add_TopologyAttr("resid", np.arange(1, len(pos) + 1))
    u.add_TopologyAttr("masses", np.ones(len(pos)))
    u.atoms.positions = np.asarray(pos, float)
    if box is not None:
        u.dimensions = [box, box, box, 90, 90, 90]
    return u


def _s(r, r0=3.2, n=6, m=12):
    x = r / r0
    return n / m if abs(x - 1) < 1e-9 else (1 - x ** n) / (1 - x ** m)


def test_coord_counts_neighbours_with_the_switching_function():
    from common import CVSet, CVCalculator
    pos = [[0, 0, 0], [2.5, 0, 0], [0, 3.2, 0], [0, 0, 6.0], [10, 0, 0]]
    u = _universe(pos, ["SOD", "OH2", "OH2", "OH2", "OH2"])
    cv = CVSet([{"name": "n", "type": "COORD", "atoms": ["name SOD", "name OH2"]}])
    got = CVCalculator(cv, u).frame()[0]
    want = sum(_s(r) for r in (2.5, 3.2, 6.0, 10.0))
    assert got == pytest.approx(want, rel=1e-6)
    assert _s(3.2) == pytest.approx(0.5)            # the limit at r0 is n/m


def test_coord_uses_the_minimum_image():
    """A water 1 A outside the far face of the box is 2 A from an ion near the
    near face: it must count as a first-shell neighbour."""
    from common import CVSet, CVCalculator
    u = _universe([[1.0, 5, 5], [19.0, 5, 5]], ["SOD", "OH2"], box=20.0)
    cv = CVSet([{"name": "n", "type": "COORD", "atoms": ["name SOD", "name OH2"]}])
    assert CVCalculator(cv, u).frame()[0] == pytest.approx(_s(2.0), rel=1e-5)
    # without the box the same pair is 18 A apart and contributes almost nothing
    u2 = _universe([[1.0, 5, 5], [19.0, 5, 5]], ["SOD", "OH2"])
    assert CVCalculator(cv, u2).frame()[0] < 1e-3


def test_coord_options_are_checked_and_respected():
    from common import CVSet, CVCalculator
    u = _universe([[0, 0, 0], [3.0, 0, 0]], ["SOD", "OH2"])
    cv = CVSet([{"name": "n", "type": "COORD", "atoms": ["name SOD", "name OH2"], "r0": 2.0, "n": 8, "m": 16}])
    assert CVCalculator(cv, u).frame()[0] == pytest.approx(_s(3.0, 2.0, 8, 16), rel=1e-6)
    with pytest.raises(ValueError, match="m > n"):
        CVSet([{"name": "n", "type": "COORD", "atoms": ["a", "b"], "n": 12, "m": 6}])
    with pytest.raises(ValueError, match="positive"):
        CVSet([{"name": "n", "type": "COORD", "atoms": ["a", "b"], "r0": 0}])


def test_coord_cannot_span_a_string_or_reach_cfacv():
    from common import CVSet, CVCalculator
    import make_cfacv_inputs as mk
    cvs = [{"name": "r", "type": "BOND", "atoms": ["name SOD", "name CLA"]},
           {"name": "n", "type": "COORD", "atoms": ["name SOD", "name OH2"]}]
    assert CVSet(cvs).string == ["r"]                 # left out of the default string
    with pytest.raises(ValueError, match="analysis only"):
        CVSet(cvs, ["r", "n"])
    u = _universe([[0, 0, 0], [3, 0, 0], [0, 3, 0]], ["SOD", "CLA", "OH2"])
    with pytest.raises(ValueError, match="analysis only"):
        mk.build_groups(CVSet(cvs), u)
    with pytest.raises(ValueError, match="metric tensor"):
        CVCalculator(CVSet(cvs), u).metric_tensor()


def test_coord_is_a_plain_feature_for_selection():
    from common import CVSet
    from cv_select import featurize
    cv = CVSet([{"name": "r", "type": "BOND", "atoms": ["a", "b"]},
                {"name": "n", "type": "COORD", "atoms": ["a", "c"]},
                {"name": "t", "type": "DIHED", "atoms": ["a", "b", "c", "d"]}])
    F, owner = featurize(np.array([[3.0, 5.5, 0.1]]), cv.periodic)
    assert list(owner) == [0, 1, 2, 2] and F[0, 1] == 5.5
