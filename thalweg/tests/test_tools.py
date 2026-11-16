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
    # the same curve stored one whole turn away in the torsion is still the same curve
    turned = ra["string"] + [0.0, 2 * np.pi]
    assert not sm.compare(ra["string"], turned, scale, ra["mean_scatter"], ra["mean_scatter"])["merged"]
    assert sm.compare(ra["string"], turned, scale, ra["mean_scatter"], ra["mean_scatter"],
                      periodic=np.array([False, True]))["merged"]
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
    assert not r["few_uncommitted"] and not r["pass"]
    # ... and where they would have gone brackets p: 8/8/4 reads 0.50, but is 0.40 to 0.60 or worse
    r = tally_counts([(8, 8, 4, 0)] * 10)
    assert abs(r["pooled_p_open"] - 0.5) < 1e-9 and r["p_bounds"][0] < r["window"][0] and not r["centred"]
    # one configuration with all the shots and nine with one each: nothing to test sameness with
    r = tally_counts([(45, 46, 0, 0)] + [(1, 0, 0, 0)] * 9)
    assert not r["testable"] and not r["pass"]
    # unfinished runs can never pass
    assert not tally_counts([(10, 10, 0, 0)] * 9 + [(5, 5, 0, 10)])["pass"]


def test_refine_puts_images_where_the_force_changes():
    import fe_profile as fp
    z = np.column_stack([np.linspace(0, 8, 9), np.zeros(9)])
    f = np.zeros((9, 2)); f[4, 0] = 30.0                     # the force spikes at image 4
    new, split = fp.refine_images(z, f, np.ones(2), 4)
    assert len(new) == 13 and set(split) <= {2, 3, 4, 5}     # only the segments around the spike
    # a steep but LINEAR ramp needs no images (the trapezoid rule is exact there); the kink does
    f2 = np.zeros((9, 2)); f2[:5, 0] = 20.0 * np.arange(5); f2[5:, 0] = 80.0
    new2, split2 = fp.refine_images(z, f2, np.ones(2), 3)
    assert set(split2) <= {3, 4}
    assert np.all(np.diff(new[:, 0]) > 0)                    # still an ordered path
    assert np.allclose(new[0], z[0]) and np.allclose(new[-1], z[-1])


def test_profile_from_driver_forces(tmp_path):
    """A string that moves, then stops: only the frames after it stopped are
    used, and the profile of F = 3 sin^2(pi x / 2) along x in [0, 1] comes back."""
    import fe_profile as fp
    rng = np.random.default_rng(4)
    n, nfr = 11, 60
    zfin = np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
    lines = ["# update image n_updates z dF/dz"]
    for fr in range(nfr):
        shift = 3.0 * np.exp(-fr / 4.0)                      # early frames are far from the final string
        for i in range(n):
            z = zfin[i] + [0, shift]
            g = np.array([3 * np.pi / 2 * np.sin(np.pi * z[0]), 0.0]) + rng.normal(0, 0.3, 2)
            lines.append(f"{(fr + 1) * 100} {i} 100 {z[0]} {z[1]} {g[0]} {g[1]}")
    f1, f2 = tmp_path / "a.forces", tmp_path / "b.forces"       # two legs
    cut = 1 + 20 * n
    f1.write_text("\n".join(lines[:cut]) + "\n"); f2.write_text(lines[0] + "\n" + "\n".join(lines[cut:]) + "\n")
    res = fp.from_driver([str(f1), str(f2)], np.ones(2))
    assert 30 < res["frames_used"] < nfr                     # the moving frames were dropped
    assert abs(res["barrier"] - 3.0) < 0.15 and res["top_image"] == 10
    assert res["stable"] and res["resolved"]
    with pytest.raises(ValueError):
        fp.from_driver([str(f1)], np.ones(2), max_drift=1e-6)


def test_metric_tensor_against_closed_forms():
    mda = pytest.importorskip("MDAnalysis")
    rng = np.random.default_rng(3)
    u = mda.Universe.empty(6, trajectory=True)
    masses = np.array([12.0, 14.0, 1.0, 16.0, 12.0, 32.0])
    u.add_TopologyAttr("masses", masses)
    u.atoms.positions = rng.normal(0, 2.0, (6, 3))
    cvs = CVSet([{"name": "d", "type": "BOND", "atoms": ["index 0", "index 1"]},
                 {"name": "dg", "type": "BOND", "atoms": ["index 0 2", "index 3"]},
                 {"name": "far", "type": "BOND", "atoms": ["index 4", "index 5"]},
                 {"name": "t", "type": "DIHED", "atoms": ["index 0", "index 1", "index 3", "index 4"]}])
    M = common.CVCalculator(cvs, u).metric_tensor()
    assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
    assert abs(M[0, 0] - (1 / 12 + 1 / 14)) < 1e-6           # a distance: the sum of the inverse masses
    assert abs(M[1, 1] - (1 / 13 + 1 / 16)) < 1e-6           # a group counts as one body of its total mass
    assert abs(M[0, 2]) < 1e-12 and abs(M[1, 2]) < 1e-12     # no shared atoms, no coupling
    assert abs(M[0, 1]) > 1e-4                               # atom 0 is in both
    # the dihedral, by brute force on the atoms themselves
    def tors(x):
        return common._dihedral(x[0], x[1], x[3], x[4])
    x0 = u.atoms.positions.astype(float); g = np.zeros((6, 3))
    for a in range(6):
        for k in range(3):
            hi, lo = x0.copy(), x0.copy(); hi[a, k] += 1e-5; lo[a, k] -= 1e-5
            g[a, k] = (tors(hi) - tors(lo)) / 2e-5
    assert abs(M[3, 3] - np.sum(g ** 2 / masses[:, None])) < 1e-5


def test_mobility_bends_the_relaxed_string():
    """On a tilted quadratic valley a path relaxed with an anisotropic tensor
    ends somewhere else than one relaxed without it; and the tensor survives
    a save and a load."""
    from fesmap import FESMap, fit_with_sigma_scan
    from path_search import relax_string
    xs = np.arange(-1.5, 1.5 + 1e-9, 0.25)
    P = np.array([(x, y) for x in xs for y in xs])
    f = -np.column_stack([0.4 * P[:, 0], 4.0 * (P[:, 1] - 0.5 * np.sin(2 * P[:, 0]))])   # minus the gradient
    m, _, _ = fit_with_sigma_scan(P, f, np.ones(2), np.zeros(2, bool))
    start = np.column_stack([np.linspace(-1.2, 1.2, 12), np.full(12, 0.6)])
    plain, _, _ = relax_string(m, start, 12, max_iter=300)
    m.set_metric_tensor(P, np.tile(np.array([[1.0, 0.9], [0.9, 1.0]]), (len(P), 1, 1)))
    bent, _, _ = relax_string(m, start, 12, max_iter=300)
    assert np.abs(bent - plain).max() > 0.02
    with pytest.raises(ValueError):
        m.set_metric_tensor(P + 0.1, np.tile(np.eye(2), (len(P), 1, 1)))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m.save(os.path.join(d, "m.json")); m2 = FESMap.load(os.path.join(d, "m.json"))
    assert np.allclose(m2.M, m.M) and np.allclose(m2.mobility(P[:3]), m.mobility(P[:3]))


def test_metric_relaxation_reaches_the_path_M_grad_F_defines():
    """Two wells joined by a curved valley, and a tensor whose eigenvalues
    differ 200-fold with the slow direction nearly across the valley.  The
    relaxation must converge, must end on the path the textbook flow along
    -M grad F reaches when run until it truly stops, and that path must
    differ from the one found without the tensor."""
    from fesmap import fit_with_sigma_scan
    import path_search as ps
    P = np.array([(x, y) for x in np.arange(-1.6, 1.6 + 1e-9, 0.2) for y in np.arange(-1.2, 1.2 + 1e-9, 0.2)])
    w = P[:, 1] - 0.5 * np.sin(1.5 * P[:, 0])
    f = -np.column_stack([4 * P[:, 0] * (P[:, 0] ** 2 - 1) - 6 * w * 0.75 * np.cos(1.5 * P[:, 0]), 6 * w])
    m, _, _ = fit_with_sigma_scan(P, f, np.ones(2), np.zeros(2, bool))
    e0, e1 = ps.descend(m, [-1, -0.5]), ps.descend(m, [1, 0.5])
    start = np.linspace(e0, e1, 16); start[1:-1, 1] += 0.4
    plain, _, conv0 = ps.relax_string(m, start, 16)
    c, s_ = np.cos(1.3), np.sin(1.3); R = np.array([[c, -s_], [s_, c]])
    m.set_metric_tensor(P, np.tile(R @ np.diag([1.0, 200.0]) @ R.T, (len(P), 1, 1)))
    new, n_it, conv = ps.relax_string(m, start, 16)
    assert conv0 and conv and n_it < 2000
    path = ps.redistribute(start.copy(), 16, m.metric)        # the textbook flow: images along -M grad F, ends held, respaced
    gm = np.einsum("pij,pj->pi", m.mobility(m.centers), m.gradient(m.centers))
    dt = 0.05 / np.percentile(np.linalg.norm(gm, axis=1), 90)
    for _ in range(20000):
        q = path - dt * np.einsum("pij,pj->pi", m.mobility(path), m.gradient(path)); q[0], q[-1] = path[0], path[-1]
        path = ps.redistribute(q, 16, m.metric)
    assert ps.path_distance(new, path, m.metric) < 0.02       # the same path as the textbook flow
    assert ps.path_distance(plain, new, m.metric) > 0.1       # and not the one the tensor-free relaxation finds
    assert np.abs(ps.across_metric(m, new, m.gradient(new))[1:-1]).max() < 0.1


def test_upwind_tangents():
    import path_search as ps
    u = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    t = ps.upwind_tangents(u, np.array([0.0, 1.0, 2.0, 3.0]))             # rising: look ahead
    assert np.allclose(t[1], [0, 1]) and np.allclose(t[2], [1, 0])
    t = ps.upwind_tangents(u, np.array([3.0, 2.0, 1.0, 0.0]))             # falling: look back
    assert np.allclose(t[1], [1, 0]) and np.allclose(t[2], [0, 1])
    t = ps.upwind_tangents(u, np.array([0.0, 1.0, 1.0, 0.0]))             # a flat top: a mix, still unit length
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0)


def test_metric_from_cfacv_log(tmp_path):
    import meanforce as mf
    rng = np.random.default_rng(5)
    true = np.array([[0.3, -0.1], [-0.1, 0.8]])
    lines = ["Info: something", "CFACV) Average gradF 1.0 2.0"]
    for i in range(50):
        M = true + (5.0 if i < 10 else 0.0) + rng.normal(0, 0.01, (2, 2))       # the first fifth is off
        lines.append("CFACV) Average MT " + " ".join(str(v) for v in M.ravel()))
    p = tmp_path / "pt.log"; p.write_text("\n".join(lines) + "\n")
    M = mf.metric_from_log(str(p), 2, discard=0.2)
    assert np.allclose(M, true, atol=0.01) and np.allclose(M, M.T)
    with pytest.raises(ValueError):
        mf.metric_from_log(str(p), 3)


def _force_frames(zs, rng, n=11):
    lines = ["# update image n_updates z dF/dz"]
    for fr, shift in enumerate(zs):
        for i in range(n):
            x = i / (n - 1.0)
            g = np.array([3 * np.pi / 2 * np.sin(np.pi * x), 0.0]) + rng.normal(0, 0.3, 2)
            lines.append(f"{(fr + 1) * 100} {i} 100 {x} {shift if i == 5 else 0.0} {g[0]} {g[1]}")
    return lines


def test_driver_profile_refuses_a_drifting_string_and_one_moving_image(tmp_path):
    import fe_profile as fp
    rng = np.random.default_rng(8)
    # one image out of eleven drifts steadily the whole time: an average over images would hide it
    f = tmp_path / "drift.forces"; f.write_text("\n".join(_force_frames(0.01 * np.arange(200), rng)) + "\n")
    with pytest.raises(ValueError, match="not at rest"):
        fp.from_driver([str(f)], np.ones(2))
    # the same string, stopped for its last 120 frames: the last half of the run is at rest, so a profile comes
    # (stopped for only its last 80, the last half is not: refused since 2026-11-15)
    f.write_text("\n".join(_force_frames(np.concatenate([0.01 * np.arange(80), np.full(120, 0.8)]), rng)) + "\n")
    res = fp.from_driver([str(f)], np.ones(2))
    # the last half of the run at least; the tail of the creep that fits within the limit comes with it
    # (151 frames here; 175 before the window's first half had to pass as well)
    assert 100 <= res["frames_used"] <= 155 and res["systematic_in_window"] <= 0.2 and res["stationary"]


def test_plateau_tells_a_resting_string_from_a_creeping_one(tmp_path):
    """Both strings wander alike between neighbouring stretches; only one goes anywhere."""
    import fe_profile as fp
    rng = np.random.default_rng(21)
    n, nfr = 12, 800
    base = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    w = np.zeros((nfr, n)); 
    for f in range(1, nfr):
        w[f] = 0.8 * w[f - 1] + rng.normal(0, 0.04, n)        # slow wander across the string
    rest = np.repeat(base[None], nfr, axis=0); rest[:, :, 1] += w
    creep = rest.copy(); creep[:, 6:, 1] += 0.001 * np.arange(nfr)[:, None]   # the far half creeps 0.8 over the run
    s_rest, f_rest = fp.plateau_motion(rest[nfr // 2:], np.ones(2))
    s_creep, f_creep = fp.plateau_motion(creep[nfr // 2:], np.ones(2))
    assert s_rest < 0.1 and 0.22 < s_creep < 0.36             # 0.1 per block: sqrt(8) x 0.1 = 0.28
    assert f_rest < 0.1 and f_creep < 0.1                     # a limit of 0.1 on neighbouring stretches passes both
    assert fp.stationary(rest[nfr // 2:], np.ones(2), 0.2)[0] and not fp.stationary(creep[nfr // 2:], np.ones(2), 0.2)[0]
    # sliding along the path is not movement: the same string with its images respaced
    slid = base.copy(); slid[1:-1, 0] += 0.3
    Z = np.concatenate([np.repeat(base[None], 40, axis=0), np.repeat(slid[None], 40, axis=0)])
    assert fp.plateau_motion(Z, np.ones(2))[0] < 1e-6


def test_driver_force_legs_overlap_and_cut_frames(tmp_path):
    import fe_profile as fp
    rng = np.random.default_rng(9)
    lines = _force_frames(np.zeros(40), rng)
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_text("\n".join(lines[:1 + 25 * 11 + 4]) + "\n")              # killed mid-frame after 25 frames
    b.write_text(lines[0] + "\n")                                         # a leg that wrote nothing
    c.write_text(lines[0] + "\n" + "\n".join(lines[1 + 20 * 11:]) + "\n")  # resumed from frame 20: five frames repeated
    upd, Z, G, nup = fp.read_driver_forces([str(a), str(b), str(c)])
    assert len(upd) == 40 and list(upd) == sorted(set(upd))


def _toy_profile(rng, n=13, m=4000, corr=0.0, drift_image=None, drift=0.0, relax=0.0):
    """Restrained-run samples for a string along (1, 1) in two CVs, F = 3 sin^2(pi s) + s."""
    import fe_profile as fp
    s = np.linspace(0, 1, n); z = np.column_stack([s, s]); k = np.array([200.0, 200.0])
    dFds = 3 * np.pi * np.sin(2 * np.pi * s) + 1.0
    grad = np.outer(dFds, [0.5, 0.5])
    theta = []
    for i in range(n):
        a, b = rng.normal(0, 1, (2, m))
        e = np.column_stack([a, corr * a + np.sqrt(1 - corr ** 2) * b]) * 0.055
        th = z[i] - grad[i] / k + e
        if drift_image == i:
            th[:, 0] -= drift / k[0] * np.linspace(-0.5, 0.5, m)       # the force at this image drifts by `drift`
        if relax:
            th[:, 0] -= relax / k[0] * np.exp(-np.arange(m) / (0.05 * m))  # every run starts off its mean
        theta.append(th)
    return z, theta, k


def test_error_keeps_the_correlation_between_components():
    """Force noise correlated between the two CVs: along (1, 1) it adds up
    when the correlation is positive and cancels when negative.  A separate
    error bar per component gets both cases wrong."""
    import fe_profile as fp
    for corr, lo, hi in ((0.95, 0.6, 1.7), (-0.95, 0.6, 1.7)):
        vals, ses = [], []
        for rep in range(40):
            z, theta, k = _toy_profile(np.random.default_rng(300 + rep), corr=corr, m=1500)
            r = fp.compute(z, theta, k, np.zeros(2, bool), np.ones(2), discard=0.0)
            vals.append(r["delta_F"]); ses.append(r["delta_F_se"])
        ratio = np.mean(ses) / np.std(vals, ddof=1)
        assert lo < ratio < hi, (corr, ratio)


def test_stability_gate_can_fail_and_looks_at_the_end_to_end_difference():
    import fe_profile as fp
    z, theta, k = _toy_profile(np.random.default_rng(1))
    r = fp.compute(z, theta, k, np.zeros(2, bool), np.ones(2), discard=0.0)
    assert r["stable"] and r["stable_barrier"] and r["stable_delta_F"] and r["resolved"]
    assert abs(r["barrier"] - 3.53) < 0.3 and abs(r["delta_F"] - 1.0) < 4 * r["delta_F_se"] + 0.05
    # one image's force drifts through its run: with the old test (first half
    # against the whole, error inflated by the drift itself) this could not fail
    z, theta, k = _toy_profile(np.random.default_rng(1), drift_image=3, drift=40.0)
    r = fp.compute(z, theta, k, np.zeros(2, bool), np.ones(2), discard=0.0)
    assert not r["stable"]
    # a drift AFTER the top leaves the barrier alone and moves only the end-to-end difference
    z, theta, k = _toy_profile(np.random.default_rng(1), drift_image=10, drift=40.0)
    r = fp.compute(z, theta, k, np.zeros(2, bool), np.ones(2), discard=0.0)
    assert r["stable_barrier"] and not r["stable_delta_F"] and not r["stable"]


def test_resolution_gate_looks_at_the_end_to_end_difference():
    """A narrow drop after the top, seen by one image: the barrier is fine,
    the end-to-end difference is not."""
    import fe_profile as fp
    rng = np.random.default_rng(12)
    n, k = 21, np.array([50.0])
    z = np.linspace(0, 10, n)[:, None]
    dFdz = 0.8 * np.sin(np.pi * z[:, 0] / 5.0); dFdz[15] = 60.0
    theta = [z[i] - dFdz[i] / k + rng.normal(0, 0.1, (2000, 1)) for i in range(n)]
    r = fp.compute(z, theta, k, np.zeros(1, bool), np.ones(1), discard=0.0)
    assert not r["resolved_delta_F"] and not r["resolved"]


def test_equilibration_is_detected_not_assumed():
    import fe_profile as fp
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 4000); x[:600] += 6 * np.exp(-np.arange(600) / 150.0)
    t0, g, neff = fp.detect_equilibration(x)
    assert 200 <= t0 <= 1300
    assert fp.detect_equilibration(rng.normal(0, 1, 4000))[0] <= 250          # nothing to drop from a stationary series
    assert abs(fp.statistical_inefficiency(rng.normal(0, 1, 5000)) - 1) < 0.3
    ar = np.zeros(20000); e = rng.normal(0, 1, 20000)
    for i in range(1, 20000):
        ar[i] = 0.9 * ar[i - 1] + e[i]
    assert 12 < fp.statistical_inefficiency(ar) < 28                           # (1 + 0.9) / (1 - 0.9) = 19
    # runs that start off their mean: auto drops it, a profile with no discard is biased
    z, theta, k = _toy_profile(np.random.default_rng(2), relax=60.0)
    auto = fp.compute(z, [t.copy() for t in theta], k, np.zeros(2, bool), np.ones(2))
    none = fp.compute(z, [t.copy() for t in theta], k, np.zeros(2, bool), np.ones(2), discard=0.0)
    assert np.median(auto["discarded_fraction"]) > 0.05
    assert abs(auto["delta_F"] - 1.0) < abs(none["delta_F"] - 1.0)


def test_group_centre_is_whole_across_the_box_edge():
    """A two-atom group cut by the box edge: its centre is between the atoms
    on one side of the edge, not in the middle of the box."""
    mda = pytest.importorskip("MDAnalysis")
    from common import CVSet, CVCalculator
    u = mda.Universe.empty(4, trajectory=True)
    u.add_TopologyAttr("masses", np.ones(4)); u.add_TopologyAttr("names", ["A", "B", "C", "D"])
    u.dimensions = [20.0, 20.0, 20.0, 90, 90, 90]
    # group 1: atoms at x = 19.5 and 0.5 (one unit apart through the edge); group 2: atoms at x = 4.5 and 5.5
    u.atoms.positions = np.array([[19.5, 5, 5], [0.5, 5, 5], [4.5, 5, 5], [5.5, 5, 5]], dtype=np.float32)
    cv = CVSet([{"name": "d", "type": "BOND", "atoms": ["index 0 1", "index 2 3"]}])
    d = CVCalculator(cv, u).frame()[0]
    assert abs(d - 5.0) < 1e-4            # centres at x = 20 (= 0) and 5; the naive centre of mass would sit at x = 10


def test_candidates_name_the_chain_when_there_is_more_than_one(tmp_path):
    """Two chains with the same residue numbers: every candidate must select
    one atom per group, and its name must say which chain."""
    mda = pytest.importorskip("MDAnalysis")
    import cv_select as cs, json, types
    rng = np.random.default_rng(1)
    resn = ["ALA", "ASP", "LYS", "LEU", "GLU", "ARG", "VAL", "SER"]
    names, resids, segs, resnames = [], [], [], []
    gam = {"ASP": "CG", "LYS": "CG", "LEU": "CG", "GLU": "CG", "ARG": "CG", "VAL": "CG1", "SER": "OG"}
    extra = {"ASP": ["CG"], "LYS": ["NZ"], "GLU": ["CD"], "ARG": ["CZ"]}
    for seg in ("A", "B"):
        for i, rn in enumerate(resn):
            ats = ["N", "CA", "CB"] + ([gam[rn]] if rn in gam else []) + [x for x in extra.get(rn, []) if x != gam.get(rn)]
            for at in ats: names.append(at); resids.append(i + 1); segs.append(seg); resnames.append(rn)
    n = len(names)
    keys = list(dict.fromkeys(zip(segs, resids)))
    resindex = np.array([keys.index(k) for k in zip(segs, resids)])
    u = mda.Universe.empty(n, n_residues=len(keys), n_segments=2, atom_resindex=resindex,
                           residue_segindex=[0 if k[0] == "A" else 1 for k in keys], trajectory=True)
    u.add_TopologyAttr("names", names); u.add_TopologyAttr("resids", [k[1] for k in keys])
    u.add_TopologyAttr("resnames", [resnames[list(resindex).index(r)] for r in range(len(keys))])
    u.add_TopologyAttr("segids", ["A", "B"]); u.add_TopologyAttr("masses", np.full(n, 12.0))
    base = rng.normal(0, 6, (n, 3))
    def write(tag, shift):
        pos = base.copy(); pos[np.array(segs) == "B"] += shift          # chain B moves between the states
        frames = pos[None] + rng.normal(0, 0.1, (30, n, 3))
        u.load_new(frames.astype(np.float32)); u.dimensions = [80, 80, 80, 90, 90, 90]
        u.atoms.write(str(tmp_path / "sys.pdb"))
        with mda.Writer(str(tmp_path / f"{tag}.dcd"), n) as w:
            for ts in u.trajectory: w.write(u.atoms)
    write("closed", np.zeros(3)); write("open", np.array([6.0, 0, 0]))
    a = types.SimpleNamespace(psf=str(tmp_path / "sys.pdb"), closed=[str(tmp_path / "closed.dcd")], opened=[str(tmp_path / "open.dcd")],
                              protein="all", skip=0, stride=1, min_sep=2, n_ca=6, gap=1, n_bridge=3, max_bridge=1e9, n_chi=3,
                              include=None, out=str(tmp_path / "cand"))
    cs.candidates(a)
    pool = json.load(open(str(tmp_path / "cand") + ".json")) if os.path.exists(str(tmp_path / "cand") + ".json") else json.load(open(a.out))
    uu = mda.Universe(str(tmp_path / "sys.pdb"))
    assert pool["cvs"]
    for c in pool["cvs"]:
        assert any(ch in c["name"] for ch in ("A", "B")), c["name"]
        for sel in c["atoms"]:
            assert "segid" in sel and len(uu.select_atoms(sel)) == 1, sel
    assert any(c["name"].startswith("ca_A") and "_B" in c["name"] for c in pool["cvs"])      # an inter-chain distance, which is what moved


def test_spline_integration_beats_the_trapezoid_rule_on_a_curved_path():
    """F = 3 sin^2(pi x / 2) + 2 y^2 along a quarter-circle-like path sampled at few images."""
    import fe_profile as fp
    F = lambda p: 3 * np.sin(np.pi * p[:, 0] / 2) ** 2 + 2 * p[:, 1] ** 2
    f = lambda p: -np.column_stack([3 * np.pi / 2 * np.sin(np.pi * p[:, 0]), 4 * p[:, 1]])     # minus the gradient
    t = np.linspace(0, 1, 9)
    z = np.column_stack([t, 0.5 * np.sin(np.pi * t)])                                           # a bent path from (0,0) to (1,0)
    exact = F(z) - F(z[:1])
    err_spline = np.abs(fp.integrate(z, f(z)) - exact).max()
    err_trap = np.abs(fp.integrate_trapezoid(z, f(z)) - exact).max()
    assert err_spline < 0.03 and err_trap > 4 * err_spline
    # linear in the forces, so block profiles add up
    a, b = f(z), f(z)[::-1].copy()
    assert np.allclose(fp.integrate(z, a + 2 * b), fp.integrate(z, a) + 2 * fp.integrate(z, b))
    # three images: falls back to the trapezoid rule
    assert np.allclose(fp.integrate(z[:3], f(z[:3])), fp.integrate_trapezoid(z[:3], f(z[:3])))


def test_reference_from_unbiased_frames_is_the_smoothed_free_energy():
    """Frames drawn from a known 2-D density: the reference at a few points
    must equal the free energy of that density smoothed by the restraint's
    Gaussian (worked out on a grid), and must differ from the unsmoothed one
    by the amount the smoothing predicts."""
    import fe_profile as fp
    rng = np.random.default_rng(12); kT = 0.596; k = 40.0
    Fx = lambda x: 3.0 * (x * x - 1) ** 2                      # double well in x, barrier 3 at x = 0
    x = rng.uniform(-2, 2, 4_000_000); x = x[rng.random(len(x)) < np.exp(-Fx(x) / kT)]
    y = rng.normal(0, np.sqrt(kT / 5.0), len(x))               # harmonic in y, k_y = 5
    runs = np.array_split(np.column_stack([x, y]), 20)
    z = np.array([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    R = fp.reference_smoothed(z, runs, k, np.zeros(2, bool), kT=kT, n_boot=300)
    g = np.linspace(-2.5, 2.5, 20001); rho = np.exp(-Fx(g) / kT)
    smooth = np.array([-kT * np.log(np.trapezoid(rho * np.exp(-0.5 * k * (g - zi) ** 2 / kT), g)) for zi in z[:, 0]])
    smooth -= smooth[0]
    assert np.abs(R["F"] - smooth).max() < 4 * R["F_se"].max() + 0.03
    assert abs(R["barrier"] - smooth.max()) < 0.1
    assert 3.0 - smooth.max() > 0.15                           # the restraint flattens the barrier, and the reference follows it
    assert R["effective_frames"].min() > 50
    with pytest.raises(ValueError):
        fp.reference_smoothed(np.array([[-1.0, 0.0], [9.0, 0.0]]), runs, k, np.zeros(2, bool), kT=kT, n_boot=10)
