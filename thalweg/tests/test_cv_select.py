"""CV selection on data whose answer is known (no MD)."""
import os
import pytest
import numpy as np

import cv_select as cs


def _shooting_data(n=600, seed=0, noise_only=False):
    """Six candidates.  The true committor depends on candidate 0 (a
    distance) and candidate 2 (a torsion, through its sine); the others are
    bystanders, one of them correlated with candidate 0."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(10, 2, n), rng.normal(0, 1, n), rng.uniform(-np.pi, np.pi, n),
                         rng.normal(5, 1, n), rng.uniform(-np.pi, np.pi, n), rng.normal(0, 1, n)])
    x[:, 3] += 0.4 * (x[:, 0] - 10)                      # a correlated bystander
    q = 0.9 * (x[:, 0] - 10) + 1.6 * np.sin(x[:, 2])
    p = 0.5 * (1 + np.tanh(q)) if not noise_only else np.full(n, 0.5)
    y = (rng.random(n) < p).astype(float)
    return x, y, np.array([False, False, True, False, True, False]), ["dA", "dB", "chiC", "dD", "chiE", "dF"]


def test_selection_recovers_the_two_real_coordinates():
    x, y, per, names = _shooting_data()
    res = cs.select_cvs(x, y, per, names, kmax=6)
    assert set(res["chosen"]) == {"dA", "chiC"}, res["steps"]
    assert res["pass"]
    assert res["cv_loglik"] > res["cv_loglik_best_single"] > res["cv_loglik_null"]
    # the step that was refused is recorded, with the reason visible in BIC
    assert res["steps"][-1]["kept"] is False


def test_selection_refuses_when_outcomes_are_coin_flips():
    x, y, per, names = _shooting_data(noise_only=True, seed=3)
    res = cs.select_cvs(x, y, per, names, kmax=6)
    assert not res["pass"]
    assert len(res["chosen"]) <= 1


def test_one_coordinate_is_enough_when_it_is():
    rng = np.random.default_rng(5)
    x = np.column_stack([rng.normal(0, 1, 500), rng.normal(0, 1, 500), rng.normal(0, 1, 500)])
    y = (rng.random(500) < 0.5 * (1 + np.tanh(2.0 * x[:, 1]))).astype(float)
    res = cs.select_cvs(x, y, np.zeros(3, bool), ["a", "b", "c"])
    assert res["chosen"] == ["b"] and res["pass"]


def test_classifier_separates_states_and_handles_a_torsion_at_the_boundary():
    rng = np.random.default_rng(1)
    def state(d, chi, n=2000):
        x = np.column_stack([rng.normal(d, 1.0, n), rng.normal(chi, 0.2, n), rng.normal(0, 1, n)])
        x[:, 1] = (x[:, 1] + np.pi) % (2 * np.pi) - np.pi
        return x
    per = np.array([False, True, False])
    C, O = state(9.0, -1.15), state(17.0, 3.07)            # open torsion straddles +/-pi
    clf = cs.BasinClassifier.train(C[:1000], O[:1000], per)
    assert np.mean(clf.basin(C[1000:]) == "open") == 0 and np.mean(clf.basin(O[1000:]) == "closed") == 0
    assert np.mean(clf.basin(C[1000:]) == "closed") > 0.9 and np.mean(clf.basin(O[1000:]) == "open") > 0.9
    mid = np.array([[13.0, 2.0, 0.0]])
    assert clf.basin(mid)[0] == ""                         # between the states: in neither basin
    # a trajectory that starts between and falls into the open basin
    traj = np.column_stack([np.linspace(13, 17, 30), np.linspace(2.0, 3.07, 30), np.zeros(30)])
    assert clf.first_basin(traj) == "open"
    assert clf.first_basin(np.repeat(mid, 10, axis=0)) == "U"


def test_classifier_round_trips(tmp_path):
    rng = np.random.default_rng(2)
    C, O = rng.normal(0, 1, (300, 2)), rng.normal(4, 1, (300, 2))
    clf = cs.BasinClassifier.train(C, O, np.zeros(2, bool))
    clf.save(tmp_path / "c.json")
    again = cs.BasinClassifier.load(tmp_path / "c.json")
    q = rng.normal(2, 2, (20, 2))
    assert list(clf.basin(q)) == list(again.basin(q))


def test_effect_size_and_spread_of_pairs():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, (4000, 3)); b = rng.normal([0, 3, 0.2], 1, (4000, 3))
    es = cs.effect_size(a, b)
    assert es[1] > 2.5 and es[0] < 0.2
    # circular: means either side of the boundary are close, not 2 pi apart
    ta = (rng.normal(3.1, 0.1, (3000, 1)) + np.pi) % (2 * np.pi) - np.pi
    tb = (rng.normal(-3.1, 0.1, (3000, 1)) + np.pi) % (2 * np.pi) - np.pi
    assert cs.effect_size(ta, tb, periodic=True)[0] < 1.5
    # near-duplicate residue pairs are skipped
    pairs = [(10, 100), (11, 101), (12, 99), (40, 80), (10, 60)]
    got = cs.pick_spread(np.array([9.0, 8.5, 8.0, 5.0, 4.0]), pairs, 3, gap=3)
    assert got == [(10, 100), (40, 80), (10, 60)]


def test_tcl_basin_test_agrees_with_python(tmp_path):
    """The basin test a shot runs inside NAMD (cvsel_shot.tcl) must give the
    same answer as BasinClassifier, on torsions and distances alike."""
    import shutil, subprocess
    tclsh = shutil.which("tclsh")
    if tclsh is None:
        pytest.skip("no tclsh")
    import cv_select as cs
    rng = np.random.default_rng(11)
    periodic = np.array([True, False, True])
    closed = np.column_stack([rng.normal(-2.9, 0.25, 400), rng.normal(5.0, 0.4, 400), rng.normal(1.0, 0.3, 400)])
    opened = np.column_stack([rng.normal(-1.2, 0.25, 400), rng.normal(8.0, 0.5, 400), rng.normal(1.1, 0.3, 400)])
    clf = cs.BasinClassifier.train(closed, opened, periodic)
    pts = np.vstack([closed[:40], opened[:40], 0.5 * (closed[:40] + opened[:40]),
                     np.column_stack([rng.uniform(-np.pi, np.pi, 60), rng.uniform(3, 10, 60), rng.uniform(-np.pi, np.pi, 60)])])
    want = [str(b) for b in clf.basin(pts)]
    assert {"closed", "open", ""} <= set(want)
    lst = lambda v: " ".join(f"{x:.8g}" for x in v)
    script = ["set shot_selftest 1", f"set cv_periodic {{{' '.join(str(int(p)) for p in periodic)}}}", "set basin_names {closed open}"]
    for k, s in clf.stats.items():
        script += [f"set basin_mean({k}) {{{lst(s['mean'])}}}", f"set basin_sd({k}) {{{lst(s['sd'])}}}", f"set basin_thr({k}) {s['thr']:.8g}",
                   f"set basin_zcap({k}) {s['zcap']:.8g}", f"set basin_cap({k}) {{{' '.join(str(int(v)) for v in s['cap'])}}}"]
    script.append(f"source {os.path.join(os.path.dirname(cs.__file__), 'cvsel_shot.tcl')}")
    script += [f'puts "=[shot_basin {{{lst(p)}}}]"' for p in pts]
    f = tmp_path / "t.tcl"; f.write_text("\n".join(script) + "\n")
    out = subprocess.run([tclsh, str(f)], capture_output=True, text=True, check=True).stdout.splitlines()
    assert [l[1:] for l in out] == want


def test_ambiguous_hex_serial_is_refused_for_a_cv_atom():
    import make_cfacv_inputs as mk
    assert mk.serial_field(99999, True) == "99999"
    assert mk.serial_field(148000, False) == "24220"          # an atom in no group: cfacv never reads it
    assert mk.serial_field(150000, True) == "249f0"           # has a letter: read as hexadecimal, correctly
    with pytest.raises(ValueError, match="24220"):
        mk.serial_field(148000, True)


def test_selection_ranks_by_bic_not_likelihood():
    """A torsion that just beats a distance on likelihood but costs two
    parameters must not end the search while the distance would pass."""
    import cv_select as cs
    hits = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = 400
        d = rng.normal(0, 1, n); t = rng.uniform(-np.pi, np.pi, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-2 * (0.9 * d + 0.25 * np.sin(t))))).astype(float)
        res = cs.select_cvs(np.column_stack([t, d]), y, np.array([True, False]), ["t", "d"])
        hits += "d" in res["chosen"]
    assert hits == 12


def test_one_coordinate_far_out_is_not_averaged_away():
    """Forty candidates; a point that sits in the closed state in thirty-nine
    of them and three spreads out in the one that tells the states apart is
    not in the closed basin.  The cap is fixed: it does not grow with the
    pool or with heavy tails."""
    import cv_select as cs
    rng = np.random.default_rng(21)
    n, D = 2000, 40
    for tails in ("gauss", "t3"):
        draw = (lambda s: rng.normal(0, 1, s)) if tails == "gauss" else (lambda s: rng.standard_t(3, s) / np.sqrt(3))
        closed = draw((n, D)); opened = draw((n, D)); opened[:, 0] += 8.0
        clf = cs.BasinClassifier.train(closed, opened, np.zeros(D, bool))
        assert clf.stats["closed"]["cap"].sum() == 1 and clf.stats["closed"]["cap"][0] == 1     # only the separating feature
        pts = draw((500, D)); pts[:, 0] = 3.0
        assert np.mean(clf.basin(pts) == "closed") == 0.0, tails
        assert np.mean(clf.basin(draw((500, D))) == "closed") > 0.85, tails                    # the state's own frames still are
    assert clf.stats["closed"]["separation"] > 6
    old = cs.BasinClassifier({k: {kk: vv for kk, vv in s.items() if kk in ("mean", "sd", "thr")} for k, s in clf.stats.items()}, clf.periodic)
    assert np.mean(old.basin(pts) == "closed") > 0.2             # a model saved without the cap behaves as before
    # entering a basin takes two consecutive tests
    traj = np.vstack([pts[:1] * 0 + 4.0, closed[:1], pts[:1] * 0 + 4.0, closed[1:3]])
    assert clf.first_basin(traj) == "closed" and clf.first_basin(traj[:3]) == "U"


def test_noise_candidates_do_not_pass_and_twins_are_one_sample():
    """Twenty candidates that mean nothing; each configuration's outcome is
    fixed by something outside the pool, and each is shot five times.  Counted
    as 200 independent points this passed every time."""
    import cv_select as cs
    passed = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, (40, 20)); out = rng.random(40) < 0.5
        X = np.repeat(x, 5, axis=0); y = np.repeat(out, 5).astype(float); g = np.repeat(np.arange(40), 5)
        res = cs.select_cvs(X, y, np.zeros(20, bool), [f"c{i}" for i in range(20)], groups=g, n_perm=20, seed=seed)
        passed += res["pass"]
    assert passed <= 1


def test_a_pair_that_only_helps_together_is_found_and_separable_outcomes_fail():
    import cv_select as cs
    rng = np.random.default_rng(0)
    n = 300
    a = rng.normal(0, 1, n); b = 0.97 * a + np.sqrt(1 - 0.97 ** 2) * rng.normal(0, 1, n)
    y = (rng.random(n) < 0.5 * (1 + np.tanh(4 * (a - b)))).astype(float)
    res = cs.select_cvs(np.column_stack([a, b, rng.normal(0, 1, n)]), y, np.zeros(3, bool), ["a", "b", "junk"], n_perm=20)
    assert set(res["chosen"]) >= {"a", "b"} and res["pass"]
    # outcomes fixed exactly by one CV, with no gap between the classes
    x = rng.normal(0, 1, (300, 1)); y = (x[:, 0] > 0).astype(float)
    res = cs.select_cvs(x, y, np.zeros(1, bool), ["x"], n_perm=10)
    assert res["chosen"] == ["x"] and res["separable"] and not res["pass"]
