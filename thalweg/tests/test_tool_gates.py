"""Gates and refusals found wanting by review (thalweg_review3).

Each test was written to FAIL on the code before its fix, for the reason the
review found; the commit that adds the fix says which."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.join(HERE, "..")


def fails_cleanly(fn, *args):
    """fn(*args) must stop with a message: SystemExit with a non-zero code or
    a non-zero return, never a traceback."""
    try:
        rc = fn(*args)
    except SystemExit as e:
        assert e.code not in (0, None), e.code
        return str(e.code)
    assert rc not in (0, None), rc
    return ""


# ===================================================================== 1, 2
def _true_plus_noise(seed, ncfg=60, shots=5, ncand=21, beta=1.0):
    rng = np.random.default_rng(seed)
    xc = rng.normal(0, 1, (ncfg, ncand))
    g = np.repeat(np.arange(ncfg), shots); x = xc[g]
    y = (rng.random(len(g)) < 0.5 * (1 + np.tanh(beta * x[:, 0]))).astype(float)
    return x, y, g


def test_noise_does_not_ride_along_after_a_real_cv():
    """One real coordinate and twenty noise candidates: before, every CV after
    the first was accepted on BIC alone, and noise sat in a PASSING selection
    in 12 of 20 trials."""
    import cv_select as cs
    bad = 0; found = 0
    for seed in range(10):
        x, y, g = _true_plus_noise(seed)
        r = cs.select_cvs(x, y, np.zeros(x.shape[1], bool), [f"c{i}" for i in range(x.shape[1])], groups=g, n_perm=20, seed=seed)
        found += r["pass"] and "c0" in r["chosen"]
        bad += r["pass"] and any(c != "c0" for c in r["chosen"])
        for s in r["steps"]:
            if s["kept"]:
                assert s["p_value"] <= 0.05          # every CV in the set passed its own test
    assert found >= 9 and bad <= 2, (found, bad)


def test_configurations_from_one_block_are_one_sample():
    """Pure-noise pool, each configuration duplicated four times under new
    configuration ids (frames of one trajectory, near-identical): with the
    duplicates permuted independently this passed 37 of 40 times."""
    import cv_select as cs
    passed = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        ncfg, m, ncand = 40, 4, 10
        xc = rng.normal(size=(ncfg, ncand)); yc = (rng.random((ncfg, m)) < 0.5).astype(float)
        X = np.repeat(xc, m, axis=0); Y = yc.ravel(); G = np.repeat(np.arange(ncfg), m)
        X = np.vstack([X] * 4); Y = np.concatenate([Y] * 4); B = np.concatenate([G] * 4)
        G = np.concatenate([G + k * ncfg for k in range(4)])
        r = cs.select_cvs(X, Y, np.zeros(ncand, bool), [f"n{i}" for i in range(ncand)], groups=G, blocks=B, n_perm=20, seed=seed)
        passed += r["pass"]
        assert r["n_blocks"] == ncfg
    assert passed <= 1, passed


def _outcomes_file(tmp_path, rows, ncv=3):
    pool = {"cvs": [{"name": f"c{i}", "type": "BOND", "atoms": ["index 0", "index 1"]} for i in range(ncv)]}
    (tmp_path / "pool.json").write_text(json.dumps(pool))
    (tmp_path / "out.json").write_text(json.dumps({"cvs": [f"c{i}" for i in range(ncv)], "rows": rows,
                                                   "counts": {"closed": 1, "open": 1, "U": 0, "missing": 0}}))
    return ["select", "--cv", str(tmp_path / "pool.json"), "--outcomes", str(tmp_path / "out.json"),
            "--out", str(tmp_path / "sel.json"), "--permutations", "19"]


def test_select_refuses_rows_without_a_configuration_and_warns_without_blocks(tmp_path, capsys):
    import cv_select as cs
    rng = np.random.default_rng(0)
    rows = [{"x": list(rng.normal(size=3)), "outcome": "open" if rng.random() < 0.5 else "closed", "start_basin": ""}
            for _ in range(80)]
    msg = fails_cleanly(cs.main, _outcomes_file(tmp_path, rows))
    assert "config" in msg
    for i, r in enumerate(rows):
        r["config"] = i // 2
    cs.main(_outcomes_file(tmp_path, rows))
    assert "block" in capsys.readouterr().out


def test_select_on_an_empty_outcomes_file_is_a_clean_stop(tmp_path):
    import cv_select as cs
    (tmp_path / "pool.json").write_text(json.dumps({"cvs": [{"name": "a", "type": "BOND", "atoms": ["index 0", "index 1"]}]}))
    (tmp_path / "e.json").write_text("{}")
    fails_cleanly(cs.main, ["select", "--cv", str(tmp_path / "pool.json"), "--outcomes", str(tmp_path / "e.json")])


# ------------------------------------------------- small systems on disk
def _universe(pos, names=None, resids=None, segids=None, box=None):
    mda = pytest.importorskip("MDAnalysis")
    n = len(pos)
    names = names or [f"A{i}" for i in range(n)]
    resids = resids or list(range(1, n + 1))
    segids = segids or ["A"] * n
    keys = list(dict.fromkeys(zip(segids, resids)))
    segs = list(dict.fromkeys(segids))
    u = mda.Universe.empty(n, n_residues=len(keys), n_segments=len(segs),
                           atom_resindex=[keys.index(k) for k in zip(segids, resids)],
                           residue_segindex=[segs.index(k[0]) for k in keys], trajectory=True)
    u.add_TopologyAttr("names", names); u.add_TopologyAttr("resids", [k[1] for k in keys])
    u.add_TopologyAttr("resnames", ["ALA"] * len(keys)); u.add_TopologyAttr("segids", segs)
    u.add_TopologyAttr("masses", np.full(n, 12.0)); u.add_TopologyAttr("types", ["C"] * n)
    u.atoms.positions = np.asarray(pos, np.float32)
    if box is not None:
        u.dimensions = box
    return u


def _write_coor(u, stem, box=None):
    u.atoms.write(stem + ".coor", file_format="NAMDBIN")
    if box is not None:
        with open(stem + ".xsc", "w") as fh:
            fh.write("# NAMD extended system configuration output file\n#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z\n")
            fh.write(f"0 {box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]} 0 0 0\n")


def test_shots_record_their_source_block_and_say_when_they_are_short(tmp_path, capsys, monkeypatch):
    mda = pytest.importorskip("MDAnalysis")
    import cv_select as cs, types
    monkeypatch.chdir(tmp_path)
    u = _universe(np.zeros((2, 3)), box=[30, 30, 30, 90, 90, 90])
    u.atoms.write("sys.pdb")
    for k in range(2):
        with mda.Writer(f"t{k}.dcd", 2) as w:
            for f in range(5):
                u.atoms.positions = np.array([[0, 0, 0], [1 + f + k, 0, 0]], np.float32); w.write(u.atoms)
    a = types.SimpleNamespace(psf="sys.pdb", dcd=["t0.dcd", "t1.dcd"], n=5, per_config=1, steps=10, dcdfreq=1, temperature=300.0,
                              seed=1, namd_base="base.namd", out="shots", classifier=None, label=None, cvinp=None, chunk=5,
                              between=None, cv=None, audit_fraction=0.0)
    cs.shots(a)
    s = json.load(open("shots/setup.json"))
    assert s["block"] == [0, 0, 1, 1]
    assert "fewer" in capsys.readouterr().out


# ======================================================================= 3
def test_outcomes_read_the_box_beside_a_configuration(tmp_path, monkeypatch):
    """A configuration written wrapped: two atoms one unit apart through the
    box edge.  Without the box the CV read 19."""
    pytest.importorskip("MDAnalysis")
    import cv_select as cs
    monkeypatch.chdir(tmp_path)
    box = [20.0, 20.0, 20.0, 90, 90, 90]
    u = _universe([[19.5, 5, 5], [0.5, 5, 5]])
    u.atoms.write("sys.pdb")
    os.makedirs("shots/configs")
    _write_coor(u, "shots/configs/config0000", box)
    (tmp_path / "pool.json").write_text(json.dumps({"cvs": [{"name": "d", "type": "BOND", "atoms": ["index 0", "index 1"]}]}))
    rng = np.random.default_rng(0)
    cs.BasinClassifier.train(rng.normal(0.5, 0.1, (200, 1)), rng.normal(9, 0.1, (200, 1)), np.zeros(1, bool)).save("clf.json")
    json.dump({"n_configs": 1, "per_config": 1, "steps": 10, "stop_on_basin": True, "audit": [], "block": [0]}, open("shots/setup.json", "w"))
    open("shots/c0000_s0.result", "w").write("open 10\n")
    cs.main(["outcomes", "--cv", "pool.json", "--classifier", "clf.json", "--psf", "sys.pdb", "--dir", "shots"])
    rows = json.load(open("shots/outcomes.json"))["rows"]
    assert abs(rows[0]["x"][0] - 1.0) < 1e-3 and rows[0]["block"] == 0


def test_triclinic_cell_is_read_not_dropped(tmp_path):
    import common
    p = tmp_path / "t.xsc"
    p.write_text("#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z\n0 30 0 0 15 25.98076 0 0 0 40 0 0 0\n")
    box = common.read_xsc(str(p))
    assert np.allclose(box[:3], [30, 30, 40], atol=1e-3) and abs(box[5] - 60.0) < 1e-3
    src = open(os.path.join(TW, "meanforce.py")).read()
    assert "float(xsc[1]), float(xsc[5]), float(xsc[9]), 90, 90, 90" not in src


# ======================================================================= 4
PSF = """PSF

       1 !NTITLE
 REMARKS test

       4 !NATOM
       1 A    1    ALA  CA   CT1    0.000000       12.0110           0
       2 A    1    ALA  CB   CT3    0.000000       12.0110           0
       3 A    1A   ALA  CA   CT1    0.000000       12.0110           0
       4 A    2    ALA  CA   CT1    0.000000       12.0110           0

       0 !NBOND: bonds

"""


def test_insertion_codes_in_the_psf_are_refused(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    import common
    p = tmp_path / "i.psf"; p.write_text(PSF)
    with pytest.raises(ValueError, match="1A"):
        common.check_psf(str(p))
    u = mda.Universe(str(p), np.zeros((1, 4, 3), np.float32))
    cv = common.CVSet([{"name": "d", "type": "BOND", "atoms": ["resid 1 and name CB", "resid 2 and name CA"]}])
    with pytest.raises(ValueError, match="1A"):
        common.CVCalculator(cv, u)


def test_a_single_atom_pick_that_matches_several_atoms_is_refused():
    pytest.importorskip("MDAnalysis")
    import common
    u = _universe(np.zeros((3, 3)), names=["CA", "CA", "CB"], resids=[5, 5, 6], segids=["A", "B", "A"])
    cv = common.CVSet([{"name": "d", "type": "BOND", "atoms": ["resid 5 and name CA", "resid 6 and name CB"]}])
    with pytest.raises(ValueError, match="2 atoms"):
        common.CVCalculator(cv, u)
    cv = common.CVSet([{"name": "d", "type": "BOND", "atoms": ["resid 5 and name CA CB", "resid 6 and name CB"]}])
    common.CVCalculator(cv, u)                                   # a group meant as a group is fine


# ======================================================================= 5
def test_a_group_wider_than_half_the_box_is_refused():
    pytest.importorskip("MDAnalysis")
    import common
    u = _universe([[0, 5, 5], [6, 5, 5], [12, 5, 5], [3, 5, 5]], box=[20.0, 20.0, 20.0, 90, 90, 90])
    cv = common.CVSet([{"name": "d", "type": "BOND", "atoms": ["index 0 1 2", "index 3"]}])
    with pytest.raises(ValueError, match="box"):
        common.CVCalculator(cv, u).frame()


# ======================================================================= 6
def _sim(p, ncfg=40, m=25, unc=0.0, T=200, seed=0, **kw):
    from committor import tally_counts
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(T):
        pc = p(rng, ncfg); c = []
        for q in pc:
            u = rng.binomial(m, unc); k = m - u; b = rng.binomial(k, q); c.append((k - b, b, u, 0))
        ok += tally_counts(c, **kw)["pass"]
    return ok / T


def test_tally_a_few_uncommitted_do_not_sink_a_true_transition_state():
    half = lambda r, n: np.full(n, 0.5)
    assert _sim(half) >= 0.9
    assert _sim(half, unc=0.05) >= 0.85                          # was 0.34
    assert _sim(half, unc=0.10) >= 0.8                           # was 0.00


def test_tally_is_an_equivalence_test_and_its_error_sees_the_configurations():
    from committor import tally_counts
    assert _sim(lambda r, n: np.full(n, 0.4), ncfg=10, m=20) <= 0.05
    assert _sim(lambda r, n: np.full(n, 0.4)) <= 0.05
    # every configuration at the window's edge, few of them: homogeneity has
    # little power here, so this is a rate, not a refusal (0.54 before)
    assert _sim(lambda r, n: np.repeat([0.4, 0.6], n // 2), ncfg=10, m=20) <= 0.3
    rng = np.random.default_rng(3)
    c = [(25 - b, b, 0, 0) for b in rng.binomial(25, np.repeat([0.3, 0.7], 20))]
    r = tally_counts(c)
    spread = np.std([b / 25 for _, b, _, _ in c], ddof=1) / np.sqrt(40)
    assert r["pooled_se"] >= 0.9 * spread and r["pooled_se"] > 1.5 * r["binomial_se"]


def test_tally_untestable_configurations_cannot_carry_the_pooled_value():
    from committor import tally_counts
    rng = np.random.default_rng(4)
    c = [(4, 0, 0, 0)] * 20 + [(0, 4, 0, 0)] * 20 + [(20 - b, b, 0, 0) for b in rng.binomial(20, 0.5, 5)]
    assert not tally_counts(c)["pass"]


def _shots(tmp_path, results, n_configs=1, shots=2):
    d = tmp_path / "shots"; d.mkdir()
    json.dump({"n_configs": n_configs, "shots": shots, "cap_steps": 10, "basins": ["closed", "open"]}, open(d / "setup.json", "w"))
    for (c, s), text in results.items():
        (d / f"c{c:02d}").mkdir(exist_ok=True)
        (d / f"c{c:02d}" / f"s{s:03d}.result").write_text(text)
    return str(d)


def test_tally_empty_or_strange_result_files_stop_cleanly(tmp_path):
    import committor
    d = _shots(tmp_path, {(0, 0): "", (0, 1): "open 10\n"})
    msg = fails_cleanly(committor.main, ["tally", "--dir", d])
    (tmp_path / "x").mkdir()
    d = _shots(tmp_path / "x", {(0, 0): "sideways 10\n", (0, 1): "open 10\n"})
    msg = fails_cleanly(committor.main, ["tally", "--dir", d])
    assert "sideways" in msg
    fails_cleanly(committor.main, ["tally", "--dir", str(tmp_path / "nowhere")])


# ======================================================================= 7
sys.path.insert(0, HERE)
from test_runner import run, chain  # noqa: E402


def test_a_gate_step_must_print_its_pass(tmp_path):
    s = {"stages": [{"name": "a", "steps": [{"shell": "echo 'GATE foo: FAIL'; exit 0", "gate": "foo"}]},
                    {"name": "b", "steps": [{"shell": "echo x > b.txt"}]}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s))
    rc, out = run(p)
    assert rc == 1 and not (tmp_path / "b.txt").exists()
    s["stages"][0]["steps"][0]["shell"] = "echo nothing; exit 0"
    p.write_text(json.dumps(s))
    rc, out = run(p)
    assert rc == 1 and "GATE" in out
    s["stages"][0]["steps"][0]["shell"] = "echo 'GATE foo: PASS'"
    p.write_text(json.dumps(s))
    assert run(p)[0] == 0


def test_a_step_is_redone_when_its_input_changes_or_its_output_is_gone(tmp_path):
    (tmp_path / "in.txt").write_text("1\n")
    s = {"stages": [{"name": "a", "steps": [{"shell": "cp in.txt out.txt; echo r >> ran.txt", "inputs": ["in.txt"], "outputs": ["out.txt"]}]},
                    {"name": "b", "steps": [{"shell": "echo b >> ranb.txt"}]}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s))
    assert run(p)[0] == 0
    rc, out = run(p)
    assert out.count("[done") == 2
    (tmp_path / "out.txt").unlink()
    assert run(p)[0] == 0 and (tmp_path / "out.txt").exists()
    assert len((tmp_path / "ranb.txt").read_text().split()) == 2        # downstream redone too
    time.sleep(1.1); (tmp_path / "in.txt").write_text("2\n")
    assert run(p)[0] == 0 and (tmp_path / "out.txt").read_text() == "2\n"


def test_only_invalidates_what_comes_after_and_does_not_claim_the_rest_finished(tmp_path):
    p = chain(tmp_path, 2)
    assert run(p)[0] == 0
    rc, out = run(p, "--only", "setup")
    assert rc == 0 and "all stages finished" not in out
    rc, out = run(p, "--dry-run")
    assert out.count("to do") == 2, out                              # runs and analyse are no longer done
    s = {"stages": [{"name": "a", "steps": [{"shell": "echo a"}]}, {"name": "b", "steps": [{"shell": "exit 1"}]}]}
    q = tmp_path / "t.json"; q.write_text(json.dumps(s))
    assert run(q)[0] == 1
    rc, out = run(q, "--only", "a")
    assert "all stages finished" not in out


def test_a_rerun_pre_stage_reruns_its_main_run(tmp_path):
    (tmp_path / "pre.conf").write_text("date +%s.%N > pre.out; echo 'End of program'\n")
    (tmp_path / "main.conf").write_text("#PRE pre.conf\ncp pre.out main.out; echo 'End of program'\n")
    (tmp_path / "runs.list").write_text("main.conf\n")
    s = {"launcher": {"mode": "local", "command": "sh {conf}", "hang_minutes": 0}, "stages": [{"name": "r", "steps": [{"runlist": "runs.list"}]}]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s))
    assert run(p)[0] == 0
    time.sleep(1.1); (tmp_path / "pre.conf").write_text("date +%s.%N > pre.out; echo again; echo 'End of program'\n")
    assert run(p, "--from", "r")[0] == 0
    assert (tmp_path / "main.out").read_text() == (tmp_path / "pre.out").read_text()


def test_status_piped_into_head_is_quiet(tmp_path):
    s = {"stages": [{"name": f"s{i}", "steps": [{"shell": f"echo {i} " + "x" * 100}]} for i in range(600)]}
    p = tmp_path / "s.json"; p.write_text(json.dumps(s))
    r = subprocess.run(f"{sys.executable} {os.path.join(TW, 'thalweg_run.py')} {p} --status | head -1", shell=True,
                       capture_output=True, text=True, cwd=tmp_path)
    assert "Traceback" not in r.stderr and "BrokenPipe" not in r.stderr, r.stderr


# ======================================================================= 8
def _string_dir(tmp_path, coords, nsamples=50, fit_names=None):
    """Three images; coords: list of (n, 3) positions, one per image."""
    pytest.importorskip("MDAnalysis")
    d = tmp_path / "s0"; d.mkdir()
    n = len(coords[0])
    u = _universe(coords[0], names=fit_names or ["X"] * n)
    u.atoms.write(str(tmp_path / "sys.pdb"))
    anchors = [[1.0 + i] for i in range(len(coords))]
    json.dump({"cvs": ["d"], "types": ["BOND"], "k": [10], "scale": [1.0], "anchors": anchors, "n_images": len(coords),
               "n_systems": len(coords), "root": "go"}, open(d / "setup.json", "w"))
    for i, c in enumerate(coords):
        u.atoms.positions = np.asarray(c, np.float32)
        u.atoms.write(str(d / f"go.job0.{i}.coor"), file_format="NAMDBIN")
        with open(d / f"warm_{i}.cvout", "w") as fh:
            for t in range(nsamples):
                fh.write(f"CFACV/C) Th {t * 100} {anchors[i][0] + 0.01 * np.sin(t):.5f}\n")
    return ["check", "--dir", str(d), "--psf", str(tmp_path / "sys.pdb"), "--fit", "all"]


def test_init_check_fails_a_collapsed_string(tmp_path):
    import init_string
    rng = np.random.default_rng(0)
    x = rng.normal(0, 3, (10, 3))
    assert fails_cleanly(init_string.main, _string_dir(tmp_path, [x, x, x])) == ""


def test_init_check_refuses_an_empty_fit_and_too_few_samples(tmp_path):
    import init_string
    rng = np.random.default_rng(1)
    xs = [rng.normal(0, 3, (10, 3)) + i for i in range(3)]
    args = _string_dir(tmp_path, xs, nsamples=50)
    assert init_string.main(args) == 0                       # a sound string passes
    msg = fails_cleanly(init_string.main, args[:-2])           # default fit 'protein and name CA' matches nothing here
    assert "match" in msg
    (tmp_path / "b").mkdir()
    assert fails_cleanly(init_string.main, _string_dir(tmp_path / "b", xs, nsamples=3)) is not None
    (tmp_path / "c").mkdir()
    args = _string_dir(tmp_path / "c", xs)
    open(os.path.join(args[2], "warm_1.cvout"), "w").close()
    fails_cleanly(init_string.main, args)
    fails_cleanly(init_string.main, ["check", "--dir", str(tmp_path / "nowhere"), "--psf", "x.psf"])


def test_coordinates_are_read_with_the_cell_beside_them(tmp_path):
    """init_string check, the tmd setup and outcomes read .coor files that
    NAMD may have written wrapped: the cell in the .xsc beside each must come
    with it."""
    pytest.importorskip("MDAnalysis")
    import common
    u = _universe([[19.5, 5, 5], [0.5, 5, 5]])
    u.atoms.write(str(tmp_path / "sys.pdb"))
    _write_coor(u, str(tmp_path / "c"), [20.0, 21.0, 22.0])
    v = common.load_namdbin(str(tmp_path / "sys.pdb"), str(tmp_path / "c.coor"))
    assert np.allclose(v.dimensions, [20, 21, 22, 90, 90, 90])
    for f in ("init_string.py", "cv_select.py"):
        src = open(os.path.join(TW, f)).read()
        assert not __import__("re").search(r"Universe\([^)]*\bformat=\"NAMDBIN\"", src), f


# ======================================================================= 9
def _tw_history(path, n, creep, verdict):
    rng = np.random.default_rng(2)
    base = np.column_stack([np.linspace(0, 1, 6), np.linspace(0, 1, 6)])
    with open(path, "w") as fh:
        for it in range(n):
            z = base + [0, creep * it] + rng.normal(0, 0.01, base.shape)
            for r, zr in enumerate(z):
                fh.write(f"SM iter {it * 100} replica {r} : reparam z {zr[0]} {zr[1]}\n")
            fh.write(f"TW iter {it * 100} : step 1e-5 across-string move 0.002 drift 0.1\n")
        if verdict:
            fh.write(f"TW iter {(n - 1) * 100} : finished, {verdict}\n")


def test_monitor_reports_the_driver_verdict_and_needs_enough_frames(tmp_path, capsys):
    import string_monitor as sm
    (tmp_path / "sc.json").write_text(json.dumps({"scale": [1.0, 1.0]}))
    h = tmp_path / "cap.history"; _tw_history(h, 400, 1e-5, "reached the update cap")
    rc = sm.main(["status", "--history", str(h), "--scales", str(tmp_path / "sc.json"), "--out", str(tmp_path / "a")])
    out = capsys.readouterr().out
    assert rc != 0 and "PASS" not in out and "update cap" in out
    h = tmp_path / "ok.history"; _tw_history(h, 400, 1e-5, "stopped by its plateau and force tests")
    assert sm.main(["status", "--history", str(h), "--scales", str(tmp_path / "sc.json"), "--out", str(tmp_path / "b")]) == 0
    h = tmp_path / "few.history"; _tw_history(h, 20, 0.0, "stopped by its plateau and force tests")
    fails_cleanly(sm.main, ["status", "--history", str(h), "--scales", str(tmp_path / "sc.json"), "--out", str(tmp_path / "c")])


def test_monitor_empty_inputs_stop_cleanly(tmp_path):
    import string_monitor as sm
    (tmp_path / "sc.json").write_text(json.dumps({"scale": [1.0, 1.0]}))
    (tmp_path / "e").write_text("")
    fails_cleanly(sm.main, ["status", "--history", str(tmp_path / "e"), "--scales", str(tmp_path / "sc.json")])
    fails_cleanly(sm.main, ["compare", "--a", str(tmp_path / "e"), "--b", str(tmp_path / "e"), "--scales", str(tmp_path / "sc.json"),
                            "--scatter-a", "0.1", "--scatter-b", "0.1"])


# ====================================================================== 10
def _mf_dir(tmp_path):
    (tmp_path / "cv.json").write_text(json.dumps({"cvs": [{"name": "a", "type": "BOND", "atoms": ["index 0", "index 1"]},
                                                          {"name": "b", "type": "BOND", "atoms": ["index 0", "index 2"]}]}))
    (tmp_path / "pts.dat").write_text("1 2\n3 4\n")
    os.makedirs(tmp_path / "centers")
    (tmp_path / "centers" / "mesh.dat").write_text("3 4\n1 2\n")
    for i in range(2):
        for ext in (".coor", ".xsc"):
            (tmp_path / "centers" / f"center{i:05d}{ext}").write_text("")
    return ["setup", "--cv", str(tmp_path / "cv.json"), "--points", str(tmp_path / "pts.dat"), "--centers", str(tmp_path / "centers"),
            "--k", "1", "1", "--steps", "10", "--namd-base", "b.namd", "--out", str(tmp_path / "mf")]


def test_meanforce_scales_must_match_and_metric_follows_the_starts(tmp_path):
    import meanforce as mf
    args = _mf_dir(tmp_path)
    (tmp_path / "sc.json").write_text(json.dumps({"cvs": ["b", "a"], "scale": [1, 1]}))
    msg = fails_cleanly(mf.main, args + ["--scales", str(tmp_path / "sc.json")])
    assert "scales" in msg
    (tmp_path / "sc.json").write_text(json.dumps({"cvs": ["a", "b"], "scale": [1, 1]}))
    mf.main(args + ["--scales", str(tmp_path / "sc.json")])
    s = json.load(open(tmp_path / "mf" / "setup.json"))
    assert [os.path.basename(x) for x in s["starts"]] == ["center00001", "center00000"]
    msg = fails_cleanly(mf.main, ["metric", "--dir", str(tmp_path / "mf"), "--start", str(tmp_path / "centers" / "center{i:05d}"),
                                  "--cv", str(tmp_path / "cv.json"), "--psf", "x.psf"])
    assert "nearest" in msg


def test_meanforce_collect_fails_on_unusable_points(tmp_path):
    import meanforce as mf
    d = tmp_path / "mf"; d.mkdir()
    json.dump({"cvs": ["a"], "types": ["BOND"], "k": [10.0], "points": [[1.0], [2.0]]}, open(d / "setup.json", "w"))
    with open(d / "pt00000.cvout", "w") as fh:
        for t in range(100):
            fh.write(f"CFACV/C) Z {t} 1.0\nCFACV/C) Th {t} {1.0 + 0.01 * np.sin(t):.4f}\n")
    args = ["collect", "--dir", str(d), "--out", str(tmp_path / "f.dat")]
    fails_cleanly(mf.main, args)
    assert mf.main(args + ["--allow-missing"]) == 0
    fails_cleanly(mf.main, ["collect", "--dir", str(tmp_path / "nowhere")])


# ====================================================================== 11
def _two_chain_system(tmp_path, nres=8):
    mda = pytest.importorskip("MDAnalysis")
    rng = np.random.default_rng(1)
    resn = ["ALA", "ASP", "LYS", "LEU", "GLU", "ARG", "VAL", "SER"][:nres]
    gam = {"ASP": "CG", "LYS": "CG", "LEU": "CG", "GLU": "CG", "ARG": "CG", "VAL": "CG1", "SER": "OG"}
    extra = {"ASP": ["CG"], "LYS": ["NZ"], "GLU": ["CD"], "ARG": ["CZ"]}
    names, resids, segs, rns = [], [], [], []
    for seg in ("A", "B"):
        for i, rn in enumerate(resn):
            for at in ["N", "CA", "CB"] + ([gam[rn]] if rn in gam else []) + [x for x in extra.get(rn, []) if x != gam.get(rn)]:
                names.append(at); resids.append(i + 1); segs.append(seg); rns.append(rn)
    u = _universe(np.zeros((len(names), 3)), names=names, resids=resids, segids=segs)
    keys = list(dict.fromkeys(zip(segs, resids)))
    u.residues.resnames = [rns[list(zip(segs, resids)).index(k)] for k in keys]
    n = len(names); base = rng.normal(0, 6, (n, 3))
    for tag, shift in (("closed", 0.0), ("open", 6.0)):
        pos = base.copy(); pos[np.array(segs) == "B", 0] += shift
        frames = pos[None] + rng.normal(0, 0.1, (30, n, 3))
        u.load_new(frames.astype(np.float32)); u.dimensions = [80, 80, 80, 90, 90, 90]
        u.atoms.write(str(tmp_path / "sys.pdb"))
        with mda.Writer(str(tmp_path / f"{tag}.dcd"), n) as w:
            for ts in u.trajectory:
                w.write(u.atoms)
    return u


def _cand_args(tmp_path, **kw):
    import types
    d = dict(psf=str(tmp_path / "sys.pdb"), closed=[str(tmp_path / "closed.dcd")], opened=[str(tmp_path / "open.dcd")],
             protein="all", skip=0, stride=1, min_sep=2, n_ca=6, gap=1, n_bridge=3, max_bridge=1e9, n_chi=0,
             include=None, out=str(tmp_path / "cand"))
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_candidates_bridges_chains_and_published_duplicates(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    import cv_select as cs
    _two_chain_system(tmp_path)
    # no acid-base pair is ever within 0.1 A: no salt-bridge candidate at all
    cs.candidates(_cand_args(tmp_path, max_bridge=0.1))
    pool = json.load(open(tmp_path / "cand.json"))["cvs"]
    assert not [c for c in pool if c["name"].startswith("sb_")]
    # --min-sep is a separation along ONE chain: pairs across chains always qualify
    cs.candidates(_cand_args(tmp_path, min_sep=20, n_bridge=0))
    pool = json.load(open(tmp_path / "cand.json"))["cvs"]
    ca = [c for c in pool if c["name"].startswith("ca_")]
    assert ca
    assert all(c["name"].split("_")[1][0] != c["name"].split("_")[2][0] for c in ca)
    # a published CV that names the same atoms in other words is not added twice
    uu = mda.Universe(str(tmp_path / "sys.pdb"))
    first = ca[0]
    idx = [uu.select_atoms(s)[0].index + 1 for s in first["atoms"]]
    (tmp_path / "pub.json").write_text(json.dumps({"cvs": [{"name": "pubd", "type": "BOND", "atoms": [f"bynum {i}" for i in idx]}]}))
    cs.candidates(_cand_args(tmp_path, min_sep=20, n_bridge=0, include=str(tmp_path / "pub.json")))
    names = [c["name"] for c in json.load(open(tmp_path / "cand.json"))["cvs"]]
    assert "pubd" not in names


def test_pick_spread_compares_residues_within_a_chain_only():
    import cv_select as cs
    pairs = [(7, 20), (8, 20)]                                  # residue 7 ends chain A, 8 starts chain B
    seg = np.array(["A"] * 8 + ["B"] * 20)
    got = cs.pick_spread(np.array([2.0, 1.0]), pairs, 2, gap=3, seg=seg)
    assert got == [(7, 20), (8, 20)]
