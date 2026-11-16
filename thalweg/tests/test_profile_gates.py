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
