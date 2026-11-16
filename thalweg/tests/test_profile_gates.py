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
    rc, out = _main(["driver", "--forces", str(f), "--scales", str(sc), "--out", str(tmp_path / "o")])
    assert "GATE stationary string: PASS" in out and "GATE stability: PASS" in out and "GATE resolution: PASS" in out, out
    assert "0.3 kcal/mol low" not in out
    assert rc == 1, out
    # the same run with 1000 updates a frame is long enough
    f.write_text("\n".join(_driver_lines(np.random.default_rng(6), 60, 1000)) + "\n")
    rc, out = _main(["driver", "--forces", str(f), "--scales", str(sc), "--out", str(tmp_path / "o")])
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
