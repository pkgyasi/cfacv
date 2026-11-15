"""Unit tests of the pieces that need no MD: CV arithmetic, file formats,
endpoint statistics, center selection, mean force, initiation files,
convergence monitoring, the profile and the committor tally."""
import json
import os

import numpy as np
import pytest

import common
from common import CVSet


# ---------------------------------------------------------------- common
def test_dihedral_matches_mdanalysis_and_sign_convention():
    from MDAnalysis.lib.distances import calc_dihedrals
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 4, 3)).astype(np.float32)
    mine = common.cv_values_from_points("DIHED", [p[:, i].astype(float) for i in range(4)])
    ref = calc_dihedrals(p[:, 0], p[:, 1], p[:, 2], p[:, 3])
    assert np.allclose(mine, ref, atol=1e-5)


def test_wrap_unwrap_center_and_redistribute():
    per = np.array([False, True])
    assert np.allclose(common.diff([1.0, 3.1], [0.0, -3.1], per), [1.0, 3.1 + 3.1 - 2 * np.pi])
    path = np.array([[0, 2.9], [1, 3.1], [2, -3.0], [3, -2.8]])   # crosses +pi
    u = common.unwrap_path(path, per)
    assert np.all(np.diff(u[:, 1]) > 0) and np.abs(np.diff(u[:, 1])).max() < 0.5
    c = common.center_periodic(u + [0, 4 * np.pi], per)
    assert np.allclose(common.wrap(c - u, per), 0, atol=1e-9)        # same path, moved by whole turns
    assert np.abs(c[:, 1]).max() < np.pi + 0.5                       # and brought next to (-pi, pi]
    r = common.redistribute(u, 9, np.array([1.0, 0.1]))
    seg = np.diff(common.path_length(r, np.array([1.0, 0.1])))
    assert np.allclose(seg, seg[0], rtol=1e-2)
    assert np.allclose(r[0], u[0]) and np.allclose(r[-1], u[-1])


def test_restraint_file_keeps_string_continuous_and_refuses_out_of_range(tmp_path):
    f = tmp_path / "r.inp"
    common.write_restraints(f, [12.5, 3.4], [10.0, 100.0], ["BOND", "DIHED"])
    lines = f.read_text().splitlines()
    assert lines[0] == "1 0 {k 10.0} {rf HARMONIC} {b 12.500000}"
    assert lines[1] == "0 1 {k 100.0} {rf PERIODIC} {b 3.400000}"   # NOT wrapped to -2.88
    with pytest.raises(ValueError):
        common.write_restraints(f, [12.5, 6.3], [10.0, 100.0], ["BOND", "DIHED"])


def test_read_cvout_and_mean_force_with_wrapping(tmp_path):
    rng = np.random.default_rng(2)
    z = np.array([8.0, 3.10])                      # torsion target next to +pi
    k = np.array([10.0, 100.0])
    true_f = np.array([-3.0, 4.0])                 # f = k <theta - z>
    n = 4000
    th = z + true_f / k + rng.normal(0, [0.2, 0.08], (n, 2))
    th[:, 1] = (th[:, 1] + np.pi) % (2 * np.pi) - np.pi      # as cfacv reports it
    f = tmp_path / "x.cvout"
    with open(f, "w") as fh:
        fh.write("Info: unrelated NAMD line\n")
        for i in range(n):
            fh.write(f"CFACV/C) Z  {i * 100:10d} {z[0]:11.5f}{z[1]:11.5f}\n")
            fh.write(f"CFACV/C) Th {i * 100:10d} {th[i, 0]:11.5f}{th[i, 1]:11.5f}\n")
    rep = common.read_cvout(f)
    assert rep["Th"][1].shape == (n, 2)
    zz, ff, se, _ = common.mean_force(rep, k, np.array([False, True]), discard=0.1)
    assert np.allclose(zz, z)
    assert np.all(np.abs(ff - true_f) < 4 * se + 0.05), (ff, se)
    assert np.all(se > 0) and np.all(se < 0.5)


def test_forces_file_round_trip(tmp_path):
    z = np.array([[1.0, 2.0], [3.0, -1.0]]); f = np.array([[0.5, -0.5], [1.5, 2.5]])
    common.write_forces(tmp_path / "f.dat", z, f, ["BOND", "DIHED"], "test")
    z2, f2, t = common.read_forces(tmp_path / "f.dat")
    assert np.allclose(z, z2) and np.allclose(f, f2) and t == ["BOND", "DIHED"]


# ------------------------------------------------------- endpoint analysis
def _cvset():
    return CVSet([{"name": "d1", "type": "BOND", "atoms": ["a", "b"]},
                  {"name": "chi1", "type": "DIHED", "atoms": ["a", "b", "c", "d"]},
                  {"name": "dx", "type": "BOND", "atoms": ["a", "b"]}], ["d1", "chi1"])


def test_endpoint_analysis_separates_and_handles_the_boundary():
    from endpoint_analysis import analyse
    rng = np.random.default_rng(4)
    def state(d1, chi, n=3000):
        x = np.column_stack([rng.normal(d1, 1.0, n), rng.normal(chi, 0.15, n), rng.normal(5, 1, n)])
        x[:, 1] = (x[:, 1] + np.pi) % (2 * np.pi) - np.pi
        return x
    closed = [state(9.0, 1.16) for _ in range(3)]
    opened = [state(19.0, 3.07) for _ in range(3)]          # 176 degrees: straddles +/-pi
    rows, joint, scales, basins = analyse(closed, opened, _cvset(), 2.0)
    chi = rows[1]
    assert abs(chi["open_mean"] - 3.07) < 0.02 and abs(chi["open_sd"] - 0.15) < 0.02   # circular statistics
    assert rows[0]["overlap"] < 0.01 and rows[2]["overlap"] > 0.8                     # dx does not separate
    assert joint["closed_frames_in_open_basin"] == 0.0 and joint["open_frames_in_closed_basin"] == 0.0
    assert 0.85 < joint["open_frames_in_open_basin"] < 0.95                           # ~0.954^2
    assert scales["cvs"] == ["d1", "chi1"] and np.allclose(scales["scale"], [1.0, 0.15], atol=0.03)
    assert basins["types"] == ["BOND", "DIHED"]


def test_endpoint_analysis_reports_overlapping_states():
    from endpoint_analysis import analyse
    rng = np.random.default_rng(5)
    a = [np.column_stack([rng.normal(9, 1, 2000), rng.normal(1.0, 0.2, 2000), rng.normal(5, 1, 2000)])]
    b = [np.column_stack([rng.normal(9.5, 1, 2000), rng.normal(1.1, 0.2, 2000), rng.normal(5, 1, 2000)])]
    _, joint, _, _ = analyse(a, b, _cvset(), 2.0)
    assert joint["closed_frames_in_open_basin"] > 0.5


def test_min_image_distance():
    from endpoint_analysis import min_image_distance
    pos = np.array([[1.0, 5, 5], [9.0, 5, 5]])
    box = np.array([12.0, 30, 30, 90, 90, 90])
    assert abs(min_image_distance(pos, box) - 4.0) < 1e-6     # 9 -> 13 is 4 from... 1+12


# ----------------------------------------------------------------- harvest
def test_center_selection_uses_scaled_periodic_distance():
    from harvest_centers import select_centers
    scale, per = np.array([2.0, 0.1]), np.array([False, True])
    pts = np.array([[10.0, 3.10], [10.5, -3.12],     # 0.25 apart in d, 0.06 rad across the boundary
                    [10.0, 2.70],                    # 0.40 rad = 4 scaled units away
                    [13.0, 3.10]])                   # 1.5 scaled units away in d
    assert select_centers(pts, scale, per, 1.0) == [0, 2, 3]
    assert select_centers(pts, np.ones(2), np.zeros(2, bool), 1.0) == [0, 1, 3]   # what raw units would do


# ------------------------------------------------------------- initiation
def test_init_string_setup_writes_what_the_driver_reads(tmp_path, monkeypatch):
    import init_string
    monkeypatch.chdir(tmp_path)
    cv = {"cvs": [{"name": "d1", "type": "BOND", "atoms": ["a", "b"]},
                  {"name": "chi1", "type": "DIHED", "atoms": ["a", "b", "c", "d"]}]}
    json.dump(cv, open("cv.json", "w"))
    json.dump({"cvs": ["d1", "chi1"], "scale": [1.0, 0.2]}, open("scales.json", "w"))
    os.mkdir("centers")
    mesh = np.array([[9.0, 1.2], [12.0, 2.0], [15.0, 3.0], [19.0, -3.1]])
    np.savetxt("centers/mesh.dat", mesh)
    for i in range(4):
        open(f"centers/center{i:05d}.coor", "w").close(); open(f"centers/center{i:05d}.xsc", "w").close()
    # anchors given WRAPPED, as a careless caller might: the last one is past +pi
    np.savetxt("anchors.dat", np.array([[9.0, 1.2], [12.2, 2.1], [15.1, 2.9], [18.8, -3.0]]))
    open("base.namd", "w").close()
    init_string.main(["setup", "--cv", "cv.json", "--anchors", "anchors.dat", "--scales", "scales.json",
                      "--centers", "centers", "--k", "10", "100", "--namd-base", "base.namd", "--dual",
                      "--out", "s0"])
    s = json.load(open("s0/setup.json"))
    assert s["n_systems"] == 8 and s["center"] == [0, 1, 2, 3]
    a = np.loadtxt("s0/anchors.dat")
    assert np.all(np.diff(a[:, 1]) > 0) and abs(a[-1, 1] - (2 * np.pi - 3.0)) < 1e-6   # continuous
    # the driver reads the b values of restr.job0.N.inp as the string
    b = [float(l.split("{b ")[1].strip().rstrip("}")) for l in open("s0/restr.job0.3.inp")]
    assert np.allclose(b, a[3], atol=1e-6)
    assert open("s0/restr.job0.7.inp").read() == open("s0/restr.job0.3.inp").read()    # dual partner
    assert "PERIODIC" in open("s0/restr.job0.0.inp").read().splitlines()[1]
    listed = open("s0/warmup.list").read().split()
    assert len(listed) == 8
    last = open(listed[5]).read()
    assert last.startswith("#PRE s0/warm_5_s0.conf\n") and "\ncd ..\n" in last
    assert "bincoordinates  s0/warm_5_s0.coor" in last and "outputname      s0/go.job0.5" in last
    first = open("s0/warm_5_s0.conf").read()
    assert "bincoordinates  centers/center00001.coor" in first and "temperature" in first
    assert "{k 1.0}" in open("s0/warm_5_s0_restr.inp").read()       # 0.1 x 10
    assert "SMPARAMS(zscale) {1.00000 0.20000}" in open("s0/string_stage_snippet.namd").read()


# --------------------------------------------------------------- monitoring
def _history(path, strings, start=0):
    with open(path, "w") as fh:
        for it, s in enumerate(strings):
            for r, z in enumerate(s):
                fh.write(f"SM iter {start + it * 50} replica {r} : gradient dF/dz 0 0\n")
                fh.write(f"SM iter {start + it * 50} replica {r} : reparam z {z[0]} {z[1]}\n")


def test_monitor_tells_converged_from_drifting(tmp_path):
    import string_monitor as sm
    rng = np.random.default_rng(6)
    base = np.column_stack([np.linspace(9, 19, 12), np.linspace(1.2, 3.3, 12)])
    scale = np.array([1.0, 0.2])
    still = [base + rng.normal(0, [0.05, 0.01], base.shape) for _ in range(300)]
    drifting = [base + [0.0, 0.004 * i] + rng.normal(0, [0.05, 0.01], base.shape) for i in range(300)]
    _history(tmp_path / "a.history", still)
    _history(tmp_path / "b.history", drifting)
    it, z = sm.read_history([tmp_path / "a.history"])
    assert z.shape == (300, 12, 2) and it[1] == 50
    ra = sm.status(it, z, scale, 100, 0.25)
    assert ra["pass"] and ra["max_drift"] < 0.05 and 0.05 < ra["mean_scatter"] < 0.1
    it, z = sm.read_history([tmp_path / "b.history"])
    rb = sm.status(it, z, scale, 100, 0.25)
    assert not rb["pass"] and abs(rb["max_drift"] - 0.004 * 100 / 0.2) < 0.3
    # merged or distinct
    c = sm.compare(ra["string"], ra["string"] + [0.02, 0.0], scale, ra["mean_scatter"], ra["mean_scatter"])
    assert c["merged"]
    other = base + np.column_stack([np.zeros(12), 0.6 * np.sin(np.linspace(0, np.pi, 12))])
    assert not sm.compare(ra["string"], other, scale, ra["mean_scatter"], ra["mean_scatter"])["merged"]


def test_monitor_joins_restart_files_and_drops_a_cut_frame(tmp_path):
    import string_monitor as sm
    base = np.column_stack([np.linspace(0, 1, 4), np.linspace(0, 1, 4)])
    _history(tmp_path / "j0", [base] * 3)
    _history(tmp_path / "j1", [base + 1] * 3, start=100)        # overlaps iteration 100
    with open(tmp_path / "j1", "a") as fh:
        fh.write("SM iter 250 replica 0 : reparam z 9 9\n")       # job killed mid-frame
    it, z = sm.read_history([tmp_path / "j0", tmp_path / "j1"])
    assert list(it) == [0, 50, 100, 150, 200] and z[2, 0, 0] == 1.0


# ------------------------------------------------------------------ profile
def test_profile_recovers_a_known_barrier_and_its_error_is_honest():
    import fe_profile as fp
    rng = np.random.default_rng(7)
    n, k = 21, np.array([10.0, 100.0])
    s = np.linspace(0, 1, n)
    z = np.column_stack([9 + 10 * s, 3.0 + 0.8 * s])            # torsion runs past +pi
    Ftrue = 6 * np.sin(np.pi * s) ** 2 + 2 * s                   # barrier 7.0 at s~0.53, end +2
    dFds = 6 * np.pi * np.sin(2 * np.pi * s) + 2
    t = np.array([10.0, 0.8])
    grad = np.outer(dFds, t / np.sum(t * t))                     # dF/dz along the path
    barriers = []
    for rep in range(6):
        theta = []
        for i in range(n):
            th = z[i] - grad[i] / k + rng.normal(0, np.sqrt(0.596 / k), (3000, 2))
            th[:, 1] = (th[:, 1] + np.pi) % (2 * np.pi) - np.pi
            theta.append(th)
        r = fp.compute(z, theta, k, np.array([False, True]), np.array([1.0, 0.2]))
        barriers.append(r["barrier"])
    assert abs(r["delta_F"] - 2.0) < 4 * r["delta_F_se"] + 0.1
    assert abs(np.mean(barriers) - Ftrue.max()) < 0.25
    # the quoted error should be of the size of the actual replicate spread
    assert 0.4 < r["barrier_se"] / np.std(barriers, ddof=1) < 2.5
    assert r["overlap_ok"], r["min_overlap"]
    assert r["stable"], (r["barrier"], r["barrier_first_half"], r["barrier_se"])
    assert abs(r["top_image"] - 11) <= 1
    assert r["resolved"], (r["barrier"], r["barrier_half_images"])


def test_profile_flags_missing_overlap():
    import fe_profile as fp
    rng = np.random.default_rng(8)
    z = np.column_stack([np.linspace(0, 10, 5), np.zeros(5)])   # images 2.5 apart, ensembles 0.05 wide
    theta = [zi + rng.normal(0, 0.05, (500, 2)) for zi in z]
    r = fp.compute(z, theta, [200.0, 200.0], np.zeros(2, bool), np.ones(2))
    assert not r["overlap_ok"] and r["min_overlap"] == 0.0


def test_profile_flags_a_barrier_the_images_do_not_resolve():
    import fe_profile as fp
    rng = np.random.default_rng(10)
    n, k = 9, np.array([50.0])
    s = np.linspace(0, 1, n)
    z = (10 * s)[:, None]
    dFdz = np.zeros(n); dFdz[4] = 30.0                         # a spike seen by ONE image
    theta = [z[i] - dFdz[i] / k + rng.normal(0, 0.1, (2000, 1)) for i in range(n)]
    r = fp.compute(z, theta, k, np.zeros(1, bool), np.ones(1))
    assert not r["resolved"]


# ---------------------------------------------------------------- committor
def test_committor_tally_accepts_a_transition_state_only():
    from committor import tally_counts
    rng = np.random.default_rng(9)
    good = [(20 - b, b, 0, 0) for b in rng.binomial(20, 0.5, 10)]
    r = tally_counts(good)
    assert r["pass"] and abs(r["pooled_p_open"] - 0.5) < 0.1 and abs(r["pooled_se"] - 0.035) < 0.003
    # pooled 0.5, but every configuration commits one way: NOT a transition state
    split = [(20, 0, 0, 0)] * 5 + [(0, 20, 0, 0)] * 5
    r = tally_counts(split)
    assert r["centred"] and not r["homogeneous"] and not r["pass"]
    # on the closed side of the barrier
    r = tally_counts([(17, 3, 0, 0)] * 10)
    assert not r["centred"] and not r["pass"]
    # too many trajectories never committed
    r = tally_counts([(6, 6, 8, 0)] * 10)
    assert r["centred"] and not r["few_uncommitted"] and not r["pass"]
    # unfinished runs can never pass
    assert not tally_counts([(10, 10, 0, 0)] * 9 + [(5, 5, 0, 10)])["pass"]
