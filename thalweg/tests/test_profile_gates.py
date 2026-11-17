"""Defects in fe_profile.py found by review (thalweg_review3), one test each.

Each test failed on 2891a69 for the reason the review gave, except where
its docstring says otherwise."""
import io
import contextlib
import json
import os

import numpy as np
import pytest


# ------------------------------------------------------------------ integration
def _surface(w):
    """The reviewer's 2-D surface: a Gaussian barrier of width w on a tilted, curved valley."""
    def F(p):
        x, y = p[..., 0], p[..., 1]
        return 3 * np.exp(-x ** 2 / (2 * w * w)) * (1 + 0.3 * y) + 0.5 * y ** 2 + 0.3 * x

    def g(p):
        x, y = p[..., 0], p[..., 1]; e = 3 * np.exp(-x ** 2 / (2 * w * w))
        return np.stack([-x / (w * w) * e * (1 + 0.3 * y) + 0.3, 0.3 * e + y], -1)
    return F, g


def _path(n, uneven=False):
    u = np.linspace(-2, 2, n)
    if uneven:                    # the spacing a refinement leaves: 2:1 and 4:1
        u = np.unique(np.concatenate([np.linspace(-2, 2, n // 2), np.linspace(-0.5, 0.5, n - n // 2)]))
    return np.stack([u, np.sin(1.5 * u)], -1)


def test_a_near_constant_cv_does_not_set_the_path_parameter():
    """Finding 1. A third CV that is constant up to 1e-9 and carries no force
    must not change the profile.  With each CV's step divided by its own span
    along the string, that CV's noise set the spline parameter."""
    import fe_profile as fp
    rng = np.random.default_rng(0)
    F, g = _surface(0.3)
    z = _path(16); f = -g(z); T = F(z) - F(z[0])
    zz = np.column_stack([z, 1e-9 * rng.standard_normal(len(z))])
    ff = np.column_stack([f, np.zeros(len(z))])
    smp = [np.tile(v, (20, 1)) + 1e-7 * rng.standard_normal((20, 3)) for v in ff]
    r = fp.from_force_samples(zz, smp, np.ones(3), 10)
    smp2 = [s[:, :2] for s in smp]
    r2 = fp.from_force_samples(z, smp2, np.ones(2), 10)
    assert abs(r["barrier"] - r2["barrier"]) < 1e-3, (r["barrier"], r2["barrier"])
    assert abs(r["delta_F"] - T[-1]) < 0.05


def _weights(z, fun):
    n, d = z.shape; M = np.zeros((n, n * d))
    for j in range(n * d):
        e = np.zeros(n * d); e[j] = 1.0; M[:, j] = fun(z, e.reshape(n, d))
    return M


def test_integration_does_not_amplify_force_noise():
    """Finding 2. A global cubic spline through the force spreads one image's
    noise over its neighbours with alternating sign: on refined (uneven)
    spacing the noise in F was 2.2 times the trapezoid rule's."""
    import fe_profile as fp
    for z in (_path(16), _path(16, True), _path(32, True)):
        Wn, Wt = _weights(z, fp.integrate), _weights(z, fp.integrate_trapezoid)
        ratio = (np.linalg.norm(Wn[1:], axis=1) / np.linalg.norm(Wt[1:], axis=1)).max()
        assert ratio < 1.3, ratio


def test_integration_is_accurate_on_the_reviewers_surfaces():
    """Finding 2. Over sharp and smooth barriers, even and refined spacing,
    the median of the largest error must beat the trapezoid rule clearly,
    and the rule must not overshoot where the force changes sign."""
    import fe_profile as fp
    errs, errs_t = [], []
    for w in (0.6, 0.3, 0.15, 0.08):
        F, g = _surface(w)
        for n, uneven in ((8, False), (16, False), (16, True), (32, False), (32, True)):
            z = _path(n, uneven); f = -g(z); T = F(z) - F(z[0])
            errs.append(np.abs(fp.integrate(z, f) - T).max())
            errs_t.append(np.abs(fp.integrate_trapezoid(z, f) - T).max())
    assert np.median(errs) < 0.6 * np.median(errs_t), (np.median(errs), np.median(errs_t))
    assert np.median(errs) < 0.09, np.median(errs)
    # a path that is not smooth (images off a smooth curve, exact forces there): the
    # rule follows the images, it does not assume a tangent
    rng = np.random.default_rng(3)
    F, g = _surface(0.3); z0 = _path(32); e = []
    for _ in range(20):
        z = z0.copy(); z[1:-1, 1] += 0.05 * rng.standard_normal(30)
        e.append(np.abs(fp.integrate(z, -g(z)) - (F(z) - F(z[0]))).max())
    assert np.mean(e) < 0.03, np.mean(e)


# ------------------------------------------------------------------ resolution
def test_resolution_tests_the_difference_against_its_own_error():
    """Finding 10. A slow fluctuation shared by every image makes the barrier
    uncertain (se 0.6 to 1.7) but moves the all-image and half-image profiles
    alike: their difference, 0.8 kcal/mol from a feature only the dropped
    images see, is determined to 0.03.  The limit used to be t95 x the
    barrier's error + 0.5, which let it through every time."""
    import fe_profile as fp
    for seed in range(4):
        rng = np.random.default_rng(seed); n = 21; x = np.linspace(0, 10, n); z = x[:, None]
        dF = 3 * np.pi / 10 * np.sin(2 * np.pi * x / 10); dF[7] += 1.6
        common = np.repeat(rng.normal(0, 0.6, 10), 40)[:, None]
        smp = [np.full((400, 1), -dF[i]) + common + rng.normal(0, 0.3, (400, 1)) for i in range(n)]
        r = fp.from_force_samples(z, smp, np.ones(1), 10)
        assert abs(r["barrier"] - r["barrier_half_images"]) > 0.7
        assert not r["resolved_barrier"], (r["barrier"], r["barrier_half_images"], r["barrier_se"])


def test_resolution_drops_every_other_image_even_at_the_top():
    """Finding 10. Keeping the top and start images in the half set made the
    check blind to anything next to them: with a local integration rule, a
    spike in the force at the image between them changed nothing.  (This
    passes on 2891a69 only because its global spline carried the spike to
    images far away; with the new rule and the old choice of images it fails.)"""
    import fe_profile as fp
    rng = np.random.default_rng(10)
    n = 9; z = np.linspace(0, 10, n)[:, None]
    dFdz = np.zeros(n); dFdz[4] = 30.0
    smp = [np.full((200, 1), -dFdz[i]) + rng.normal(0, 0.05, (200, 1)) for i in range(n)]
    r = fp.from_force_samples(z, smp, np.ones(1), 10)
    assert not r["resolved"]


# ------------------------------------------------------------------ helpers for the command line
def _write_compute_dir(d, z, theta, k, types):
    """A directory as 'meanforce.py setup' leaves it, filled with runs."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "setup.json"), "w") as fh:
        json.dump({"cvs": [f"c{j}" for j in range(z.shape[1])], "types": types, "k": list(map(float, k)),
                   "points": z.tolist()}, fh)
    for i, th in enumerate(theta):
        with open(os.path.join(d, f"pt{i:05d}.cvout"), "w") as fh:
            for st, v in enumerate(th):
                fh.write(f"CFACV/C) Th {st * 10} " + " ".join(f"{x:.6f}" for x in v) + "\n")
    sc = os.path.join(d, "scales.json")
    with open(sc, "w") as fh:
        json.dump({"scale": [1.0] * z.shape[1]}, fh)
    return sc


def _main(args):
    import fe_profile as fp
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = fp.main(args)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
            print(e.code)
    return rc, buf.getvalue()


def _ramp_runs(rng, n=9, m=4000, k=100.0, noise=None):
    """Restrained runs along one distance CV, F = 2 sin^2(pi s): overlapping ensembles."""
    z = np.linspace(0.0, 1.0, n)[:, None]
    dFdz = 2 * np.pi * np.sin(2 * np.pi * z[:, 0])
    sd = np.sqrt(0.596 / k)
    theta = []
    for i in range(n):
        e = rng.normal(0, sd, (m, 1)) if noise is None else noise(i, m) * sd
        theta.append(z[i] - dFdz[i] / k + e)
    return z, theta, k


# ------------------------------------------------------------------ finding 3: warnings that must be gates
def test_compute_fails_when_blocks_are_not_longer_than_the_correlation(tmp_path):
    """Finding 3. With the force correlated over a whole block, every error is
    too small; that was a WARNING and the exit code ignored it."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(4)

    def ar(i, m):
        a = 0.998
        return lfilter([1], [1, -a], rng.normal(0, np.sqrt(1 - a * a), (m, 1)), axis=0)
    z, theta, k = _ramp_runs(rng, noise=ar)
    sc = _write_compute_dir(str(tmp_path / "r"), z, theta, [k], ["BOND"])
    rc, out = _main(["compute", "--dir", str(tmp_path / "r"), "--scales", sc, "--discard", "0", "--out", str(tmp_path / "p")])
    assert "GATE overlap: PASS" in out and "GATE stability: PASS" in out and "GATE resolution: PASS" in out, out
    assert rc == 1, out


def test_compute_fails_when_one_run_was_cut_short(tmp_path):
    """Finding 3. One image with 2% of the others' samples: a WARNING only."""
    rng = np.random.default_rng(5)
    z, theta, k = _ramp_runs(rng)
    theta[4] = theta[4][:80]
    sc = _write_compute_dir(str(tmp_path / "r"), z, theta, [k], ["BOND"])
    rc, out = _main(["compute", "--dir", str(tmp_path / "r"), "--scales", sc, "--discard", "0", "--out", str(tmp_path / "p")])
    assert "GATE overlap: PASS" in out and "GATE resolution: PASS" in out, out
    assert rc == 1, out


def _driver_lines(rng, nfr, nup, n=11, sd=0.3):
    lines = ["# update image n_updates z dF/dz"]
    for fr in range(nfr):
        for i in range(n):
            x = i / (n - 1.0)
            g = np.array([3 * np.pi / 2 * np.sin(np.pi * x), 0.0]) + rng.normal(0, sd, 2)
            lines.append(f"{(fr + 1) * nup} {i} {nup} {x} 0.0 {g[0]} {g[1]}")
    return lines


def test_driver_fails_on_too_few_updates(tmp_path):
    """Finding 3 and 12. Under 20,000 updates per image was a NOTE (with stale
    numbers in it); the exit code ignored it."""
    rng = np.random.default_rng(6)
    f = tmp_path / "few.forces"; f.write_text("\n".join(_driver_lines(rng, 60, 100)) + "\n")
    sc = tmp_path / "s.json"; sc.write_text(json.dumps({"scale": [1.0, 1.0]}))
    rc, out = _main(["driver", "--forces", str(f), "--scales", str(sc), "--out", str(tmp_path / "o"), "--step", "1e-4"])
    assert "GATE string at rest: PASS" in out and "GATE stability: PASS" in out and "GATE resolution: PASS" in out, out
    assert "0.3 kcal/mol low" not in out
    assert rc == 1, out
    # the same run with 1000 updates a frame is long enough
    f.write_text("\n".join(_driver_lines(np.random.default_rng(6), 60, 1000)) + "\n")
    rc, out = _main(["driver", "--forces", str(f), "--scales", str(sc), "--out", str(tmp_path / "o"), "--step", "1e-4"])
    assert rc == 0, out


# ------------------------------------------------------------------ finding 11
def test_restraint_width_uses_the_temperature(tmp_path):
    """Finding 11. kT was 0.596 whatever the temperature."""
    rng = np.random.default_rng(7)
    z, theta, k = _ramp_runs(rng)
    sc = _write_compute_dir(str(tmp_path / "r"), z, theta, [k], ["BOND"])
    rc, out = _main(["compute", "--dir", str(tmp_path / "r"), "--scales", sc, "--discard", "0", "--temperature", "600",
                     "--out", str(tmp_path / "p")])
    P = json.load(open(str(tmp_path / "p") + ".json"))
    assert abs(P["restraint_width"][0] - np.sqrt(0.0019872041 * 600 / k)) < 1e-6, P.get("restraint_width")


# ------------------------------------------------------------------ finding 4: one image drifting
def _drift_runs(seed, D, n=31, m=4000, at=12):
    rng = np.random.default_rng(seed)
    z, theta, k = _ramp_runs(rng, n=n, m=m)
    theta[at][:, 0] -= D / k * np.linspace(-0.5, 0.5, m)       # this image's force drifts by D over its run
    return z, theta, k


def test_stability_sees_one_image_drifting():
    """Finding 4. One image's force drifting by 3 kcal/mol/unit over its run
    (31 images) passed both aggregate stability tests; the image's own trend
    is 7 to 10 standard errors."""
    import fe_profile as fp
    for seed in range(4):
        z, theta, k = _drift_runs(seed, 3.0)
        r = fp.compute(z, theta, [k], np.zeros(1, bool), np.ones(1), discard=0.0)
        assert not r["stable"], seed
        assert r["drifting_images"] == [12], r["drifting_images"]


def test_per_image_drift_test_is_quiet_on_stationary_runs():
    import fe_profile as fp
    from scipy.signal import lfilter
    fails = 0
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)

        def ar(i, m):
            a = 0.9
            return lfilter([1], [1, -a], rng.normal(0, np.sqrt(1 - a * a), (m, 1)), axis=0)
        z, theta, k = _ramp_runs(rng, n=31, noise=ar)
        r = fp.compute(z, theta, [k], np.zeros(1, bool), np.ones(1), discard=0.0)
        fails += bool(r["drifting_images"])
    assert fails <= 1, fails


G5 = "/home/DREXEL/pg565/fep/fusion_peptide/6cm3_9d9d0_equilibration_results/t4l_paper/thalweg_review3/break/g5"


@pytest.mark.skipif(not os.path.isdir(G5 + "/drift16_0.05"), reason="the reviewer's mutated alanine runs are not here")
def test_stability_sees_the_reviewers_psi_drift(tmp_path):
    """Finding 4, on the reviewer's copy of feR0r with psi at image 16 drifted
    by 0.05 rad over its run: the barrier moved 3.35 -> 3.76 and every gate passed."""
    sc = G5 + "/../../../validation_alad4d/scales.json"
    rc, out = _main(["compute", "--dir", G5 + "/drift16_0.05", "--scales", sc, "--out", str(tmp_path / "p")])
    assert "GATE stability: FAIL" in out, out
    assert rc == 1


# ------------------------------------------------------------------ findings 5, 6, 7: the reference gate
KT = 0.0019872041 * 300.0


def _cvs(tmp_path, d):
    """A CV definition file with d distances."""
    p = tmp_path / "cvs.json"
    p.write_text(json.dumps({"cvs": [{"name": f"d{j}", "type": "BOND", "atoms": ["bynum 1", "bynum 2"]} for j in range(d)]}))
    return str(p)


def _double_well_runs(tmp_path, seed=12, nruns=20, ntot=2_000_000, Fbar=3.0):
    """iid frames from F(x) = Fbar (x^2 - 1)^2 + harmonic y, written as cfacv reports, one per run."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, ntot); x = x[rng.random(len(x)) < np.exp(-Fbar * (x * x - 1) ** 2 / KT)]
    y = rng.normal(0, np.sqrt(KT / 5.0), len(x))
    files = []
    for a, r in enumerate(np.array_split(np.column_stack([x, y]), nruns)):
        p = tmp_path / f"u{a}.cvout"; files.append(str(p))
        p.write_text("".join(f"CFACV/C) Th {s} {v[0]:.6f} {v[1]:.6f}\n" for s, v in enumerate(r)))
    return files


def _smoothed(zx, k, Fbar=3.0):
    g = np.linspace(-2.5, 2.5, 20001); rho = np.exp(-Fbar * (g * g - 1) ** 2 / KT)
    return np.array([-KT * np.log(np.trapezoid(rho * np.exp(-0.5 * k * (g - a) ** 2 / KT), g)) for a in zx])


def _profile_json(path, z, F, se, types=("BOND", "BOND")):
    top = int(F.argmax()); start = int(F[: top + 1].argmin()); F = F - F[start]
    P = {"z": z.tolist(), "F": F.tolist(), "F_se": [se] * len(z), "barrier": float(F[top] - F[start]), "barrier_se": se,
         "delta_F": float(F[-1] - F[0]), "delta_F_se": se, "top_image": top, "start_image": start}
    if types is not None:
        P["types"] = list(types)
    path.write_text(json.dumps(P))
    return P


def test_reference_zeroes_profile_and_reference_at_the_same_image(tmp_path):
    """Finding 5. A profile that starts on a wall is zeroed at its minimum
    (start_image 2 here) and the reference at image 0, and the barriers were
    F[top]-F[start] against F.max()-F[0]: an exact profile failed."""
    files = _double_well_runs(tmp_path)
    k = 40.0
    z = np.column_stack([np.linspace(-1.35, 1.0, 17), np.zeros(17)])
    F = _smoothed(z[:, 0], k)
    P = _profile_json(tmp_path / "p.json", z, F, 0.05)
    assert P["start_image"] > 0 and F[0] - F[P["start_image"]] > 0.4
    rc, out = _main(["reference", "--profile", str(tmp_path / "p.json"), "--cvout", *files, "--k", str(k), "--cv", _cvs(tmp_path, 2)])
    assert rc == 0, out


G6 = "/home/DREXEL/pg565/fep/fusion_peptide/6cm3_9d9d0_equilibration_results/t4l_paper/thalweg_review3/break/g6"
ALAD = "/home/DREXEL/pg565/fep/fusion_peptide/6cm3_9d9d0_equilibration_results/t4l_paper/validation_alad4d"


@pytest.mark.skipif(not os.path.isfile(G6 + "/perfect_start1.json"), reason="the reviewer's files are not here")
def test_reference_passes_the_reviewers_perfect_profile_with_start_image_1():
    """Finding 5 on the reviewer's case: the alanine reference itself, zeroed
    at image 1, failed by 0.63 kcal/mol at every image."""
    refs = [f"{ALAD}/ref/ref{i}.cvout" for i in range(48)]
    rc, out = _main(["reference", "--profile", G6 + "/perfect_start1.json", "--cvout", *refs, "--k", "100",
                     "--cv", ALAD + "/cvs.json"])
    assert rc == 0, out


def test_reference_says_what_it_cannot_see(tmp_path):
    """Finding 6. A profile 20% low with an honest error of 0.15 is 2.3
    combined errors from the reference: the 3-error gate cannot catch it.
    It must say so, by printing the smallest difference it would catch;
    40% low it must fail."""
    files = _double_well_runs(tmp_path)
    k = 40.0
    z = np.column_stack([np.linspace(-1.0, 1.0, 17), np.zeros(17)])
    F = _smoothed(z[:, 0], k); F -= F[0]
    _profile_json(tmp_path / "p8.json", z, 0.8 * F, 0.2)
    rc, out = _main(["reference", "--profile", str(tmp_path / "p8.json"), "--cvout", *files, "--k", str(k), "--cv", _cvs(tmp_path, 2)])
    assert "would not show" in out, out
    line = [l for l in out.splitlines() if "would not show" in l][0]
    smallest = float(line.split("under")[1].split()[0])
    assert smallest > 0.2 * F.max()                     # it says, in numbers, that it could not have seen this one
    _profile_json(tmp_path / "p6.json", z, 0.6 * F, 0.2)
    rc, out = _main(["reference", "--profile", str(tmp_path / "p6.json"), "--cvout", *files, "--k", str(k), "--cv", _cvs(tmp_path, 2)])
    assert rc == 1, out


def test_reference_reports_each_difference_with_its_own_error(tmp_path):
    """Third review, merge check. The largest difference (large error there)
    was printed next to the largest z-score, which belonged to another image:
    alanine feR1 read '0.40 kcal/mol (image 16); in combined standard errors
    10.6' and passed, because image 16 was 1.3 errors off and the 10.6 was
    image 1's, off by 0.22."""
    files = _double_well_runs(tmp_path)
    k = 40.0
    z = np.column_stack([np.linspace(-1.0, 1.0, 17), np.zeros(17)])
    F = _smoothed(z[:, 0], k); F -= F[0]
    P = _profile_json(tmp_path / "p.json", z, F, 0.05)
    F2 = np.array(P["F"]); se = np.full(len(z), 0.05)
    F2[3] += 0.25; se[3] = 0.02          # small difference, many errors
    F2[12] += 0.45; se[12] = 0.6         # large difference, few errors
    P["F"] = F2.tolist(); P["F_se"] = se.tolist()
    (tmp_path / "p.json").write_text(json.dumps(P))
    rc, out = _main(["reference", "--profile", str(tmp_path / "p.json"), "--cvout", *files, "--k", str(k), "--cv", _cvs(tmp_path, 2)])
    line = [l for l in out.splitlines() if l.startswith("largest difference at an image")][0]
    assert "(image 12," in line, line
    z_at_largest = float(line.split("(image 12,")[1].split("combined")[0])
    assert z_at_largest < 4.0, line      # image 12's own error, not image 3's


def test_reference_fails_when_it_is_too_thin_at_the_barrier(tmp_path):
    """Finding 6. Images with under --min-frames effective unbiased frames were
    dropped from the per-image test without a word in the verdict; they are the
    barrier images.  With the top thin the reference cannot test the barrier."""
    files = _double_well_runs(tmp_path, ntot=2_500_000, Fbar=6.0, nruns=20)
    k = 40.0
    z = np.column_stack([np.linspace(-1.0, 1.0, 17), np.zeros(17)])
    F = _smoothed(z[:, 0], k, Fbar=6.0); F -= F[0]
    _profile_json(tmp_path / "p.json", z, F, 0.1)               # right, but nothing can show it is
    rc, out = _main(["reference", "--profile", str(tmp_path / "p.json"), "--cvout", *files, "--k", str(k), "--cv", _cvs(tmp_path, 2)])
    assert "fewer than 20" in out
    assert rc == 1, out


def test_reference_counts_the_bootstrap_resamples_it_could_not_use(tmp_path):
    """Finding 6. A resample with no frame near some image was dropped
    silently, which leaves out exactly the resamples that would widen the
    error.  Two runs of twenty reach the far image: one resample in eight
    has none of them."""
    import fe_profile as fp
    rng = np.random.default_rng(3)
    runs = [rng.normal(-1.2, 0.1, (3000, 1)) for _ in range(18)] + [rng.normal(1.2, 0.1, (3000, 1)) for _ in range(2)]
    files = []
    for a, r in enumerate(runs):
        p = tmp_path / f"u{a}.cvout"; files.append(str(p))
        p.write_text("".join(f"CFACV/C) Th {s} {v[0]:.6f}\n" for s, v in enumerate(r)))
    z = np.linspace(-1.2, 1.2, 9)[:, None]
    R = fp.reference_smoothed(z, runs, 400.0, [False], kT=KT, n_boot=400)
    F = np.asarray(R["F"]); top = int(F.argmax())
    P = {"z": z.tolist(), "F": F.tolist(), "F_se": [0.1] * 9, "barrier": float(F[top] - F[: top + 1].min()), "barrier_se": 0.1,
         "delta_F": float(F[-1]), "delta_F_se": 0.1, "top_image": top, "start_image": int(F[: top + 1].argmin()), "types": ["BOND"]}
    (tmp_path / "p.json").write_text(json.dumps(P))
    rc, out = _main(["reference", "--profile", str(tmp_path / "p.json"), "--cvout", *files, "--k", "400", "--cv", _cvs(tmp_path, 1)])
    assert "resamples" in out and rc == 1, out


def test_reference_needs_the_cv_types(tmp_path):
    """Finding 7. Without --cv every CV was wrapped as a torsion, distances included."""
    files = _double_well_runs(tmp_path, ntot=400_000)
    z = np.column_stack([np.linspace(-1.0, 1.0, 9), np.zeros(9)])
    _profile_json(tmp_path / "p.json", z, _smoothed(z[:, 0], 40.0), 0.05, types=None)
    rc, out = _main(["reference", "--profile", str(tmp_path / "p.json"), "--cvout", *files, "--k", "40"])
    assert rc not in (0, None) and "--cv" in out, out


def test_compute_stores_the_cv_types(tmp_path):
    """Finding 7: compute writes the CV types into the profile."""
    rng = np.random.default_rng(5)
    z, theta, k = _ramp_runs(rng)
    sc = _write_compute_dir(str(tmp_path / "r"), z, theta, [k], ["BOND"])
    _main(["compute", "--dir", str(tmp_path / "r"), "--scales", sc, "--discard", "0", "--out", str(tmp_path / "p")])
    assert json.load(open(str(tmp_path / "p") + ".json")).get("types") == ["BOND"]


# ------------------------------------------------------------------ finding 8: is the driver's string at rest?
def _frames(shift_fn, rng, L=400, nimg=12, noise=0.03):
    base = np.column_stack([np.linspace(0, 3, nimg), np.sin(np.linspace(0, 3, nimg))])
    return np.array([base + shift_fn(t) + noise * rng.standard_normal(base.shape) for t in np.linspace(0, 1, L)])


def test_an_excursion_out_and_back_is_not_rest():
    """Finding 8. Every image goes 0.6 out and comes back: the first and last
    quarters agree, so the old plateau statistic was 0 and it passed.  The
    rest test reads the scatter of the block means about the line: refused.
    (400 frames of 100 updates at h = 1e-4.)"""
    import fe_profile as fp
    rng = np.random.default_rng(7)
    Z = _frames(lambda t: 0.6 * np.sin(np.pi * t) * np.array([0.0, 1.0]), rng)
    assert not fp.at_rest(Z, np.ones(2), 1e-4, 40000)[0]
    assert fp.at_rest(_frames(lambda t: 0.0 * t, rng), np.ones(2), 1e-4, 40000)[0]


def test_a_creep_is_judged_by_what_it_is_worth():
    """Finding 8, restated on 2026-11-16.  The old test held a creep to 0.2
    scaled units across the window whatever drove it.  The rest test holds it
    to what the force driving it dissipates: a creep M over W updates at step
    h costs M^2 / (h W).  At h W = 4, 0.25 across the window (0.016 kcal/mol)
    is rest and 0.8 (0.16) is not; at h W = 0.4 even 0.25 (0.16) is not."""
    import fe_profile as fp
    rng = np.random.default_rng(7)
    Z = _frames(lambda t: 0.25 * t * np.array([0.0, 1.0]), rng)
    assert fp.at_rest(Z, np.ones(2), 1e-4, 40000)[0]
    assert not fp.at_rest(Z, np.ones(2), 1e-5, 40000)[0]
    Z = _frames(lambda t: 0.8 * t * np.array([0.0, 1.0]), rng)
    assert not fp.at_rest(Z, np.ones(2), 1e-4, 40000)[0]


def _creep_file(path, nfr, rate, rng, nup=100, n=11):
    lines = ["# update image n_updates z dF/dz"]
    for fr in range(nfr):
        for i in range(n):
            x = i / (n - 1.0)
            g = np.array([3 * np.pi / 2 * np.sin(np.pi * x), 0.0]) + rng.normal(0, 0.3, 2)
            y = rate * fr if i >= 5 else 0.0                     # half the string creeps across it all the time
            lines.append(f"{(fr + 1) * nup} {i} {nup} {x} {y + rng.normal(0, 0.01)} {g[0]} {g[1]}")
    path.write_text("\n".join(lines) + "\n")


def test_last_does_not_skip_the_half_run_test(tmp_path):
    """Finding 8. --last N tested only the last N frames: a string creeping the
    whole run passed with a short enough N."""
    import fe_profile as fp
    f = tmp_path / "creep.forces"; _creep_file(f, 400, 0.002, np.random.default_rng(1))
    with pytest.raises(ValueError):
        fp.from_driver([str(f)], np.ones(2), step=1e-4)
    for last in (40, 100):
        with pytest.raises(ValueError):
            fp.from_driver([str(f)], np.ones(2), last=last, step=1e-4)


def test_a_short_run_is_refused(tmp_path):
    """Finding 8. 16 frames passed (the least was two per block)."""
    import fe_profile as fp
    rng = np.random.default_rng(2)
    for nfr in (16, 30):
        f = tmp_path / f"s{nfr}.forces"; f.write_text("\n".join(_driver_lines(rng, nfr, 100)) + "\n")
        with pytest.raises(ValueError):
            fp.from_driver([str(f)], np.ones(2), step=1e-4)


def test_legs_in_the_wrong_order_are_refused(tmp_path):
    """Finding 8. A later leg given first was silently thrown away by the earlier
    one (its frames are all 'after the restart point')."""
    import fe_profile as fp
    lines = _driver_lines(np.random.default_rng(3), 200, 100)
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("\n".join(lines[:1 + 100 * 11]) + "\n"); b.write_text(lines[0] + "\n" + "\n".join(lines[1 + 100 * 11:]) + "\n")
    assert len(fp.read_driver_forces([str(a), str(b)])[0]) == 200
    with pytest.raises(ValueError, match="order"):
        fp.read_driver_forces([str(b), str(a)])


FINAL = ALAD + "/chk/final"


@pytest.mark.skipif(not os.path.isfile(FINAL + "/pr1b.forces"), reason="the alanine driver runs are not here")
def test_the_known_creeping_run_is_refused_in_every_mode_and_the_resting_ones_are_not():
    """Finding 8 on real runs: pr1b creeps (refused by default, but passed with
    --last 40 or 100); pr0D2, pr0D3, pr1D1 and pr0R must still give a profile."""
    import fe_profile as fp
    scale = np.array(json.load(open(ALAD + "/scales.json"))["scale"])
    for last in (None, 40, 100, 200):
        with pytest.raises(ValueError):
            fp.from_driver([FINAL + "/pr1b.forces"], scale, last=last)
    for name in ("pr0D2", "pr1D1", "pr0R"):
        r = fp.from_driver([f"{FINAL}/{name}.forces"], scale)
        assert r["stationary"], name
    # pr0D3 is the shortest (a 19,200-update half): since 2026-11-16 its movement bound is worth 0.055 kcal/mol,
    # just over rest_tol, and it is refused as not yet SHOWN at rest (its barrier against unbiased dynamics was
    # flat, 3.56-3.57: the safe way to be wrong).  It must be refused for that reason and no other.
    with pytest.raises(ValueError, match="worth up to 0.05"):
        fp.from_driver([f"{FINAL}/pr0D3.forces"], scale)


# ------------------------------------------------------------------ finding 9: overlap
def test_overlap_is_not_fooled_by_one_outlying_frame():
    """Finding 9. Bins spanned min to max: one frame far out put both disjoint
    ensembles in one or two bins and the overlap read 0.5 to 1."""
    import fe_profile as fp
    rng = np.random.default_rng(7)
    z = np.array([[0.0, 0.0], [1.0, 0.0]]); per = [False, False]
    a = z[0] + 0.077 * rng.standard_normal((4000, 2)); b = z[1] + 0.077 * rng.standard_normal((4000, 2))
    assert fp.neighbour_overlap([a, b], z, per)[0] < 0.01
    a[5] = [-300.0, 0.0]
    assert fp.neighbour_overlap([a, b], z, per)[0] < 0.01
    # and it still measures overlap: two Gaussians 2 sd apart overlap by 2 Phi(-1) = 0.317
    zz = np.array([[0.0, 0.0], [0.154, 0.0]])
    a = zz[0] + 0.077 * rng.standard_normal((20000, 2)); b = zz[1] + 0.077 * rng.standard_normal((20000, 2))
    b[:3] = [[40.0, 0.0]] * 3
    assert abs(fp.neighbour_overlap([a, b], zz, per)[0] - 0.317) < 0.03


# ------------------------------------------------------------------ follow-up: an honest correlation time
def _ar1(rng, m, a):
    from scipy.signal import lfilter
    return lfilter([1], [1, -a], rng.normal(0, np.sqrt(1 - a * a), m))


def test_block_inefficiency_sees_a_slow_weak_tail():
    """The first-zero-crossing sum stops where noise first takes the
    autocorrelation below zero: a weak slow component under a strong fast
    one (as in the alanine runs' forces) is cut off and g reads about 3x low.
    Fast AR(1) a=0.5 (g 3, variance 1) plus slow a=0.999 (g 1999, variance
    0.01): g = (3 + 0.01 x 1999) / 1.01 = 22.8."""
    import fe_profile as fp
    true = (3.0 + 0.01 * 1999.0) / 1.01
    old, new = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        x = _ar1(rng, 150000, 0.5) + np.sqrt(0.01) * _ar1(rng, 150000, 0.999)
        old.append(fp.statistical_inefficiency(x)); new.append(fp.block_inefficiency(x))
    assert np.median(old) < true / 2                       # the old estimate, for the record
    assert abs(np.median(new) / true - 1) < 0.3, (np.median(new), true)
    # a plain AR(1) with g = 19: still within 30%
    r = [fp.block_inefficiency(_ar1(np.random.default_rng(10 + s), 40000, 0.9)) for s in range(8)]
    assert abs(np.median(r) / 19.0 - 1) < 0.3, r


def test_block_length_gate_uses_the_block_estimate(monkeypatch):
    """The block-length gate reads g from block_inefficiency.  (Where that
    gate bites, a correlation time near a hundredth of the run, the two
    estimates agree; they part on long runs with a weak slow tail.)"""
    import fe_profile as fp
    rng = np.random.default_rng(3)
    z = np.linspace(0, 1, 7)[:, None]
    smp = [rng.normal(0, 1, (4000, 1)) for _ in range(7)]
    assert fp.from_force_samples(z, smp, np.ones(1), 10)["blocks_ok"]
    monkeypatch.setattr(fp, "block_inefficiency", lambda x: 200.0)
    assert not fp.from_force_samples(z, smp, np.ones(1), 10)["blocks_ok"]
