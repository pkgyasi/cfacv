#!/usr/bin/env python3
"""Stage 6: free-energy profile along a converged string, with uncertainty.

Restrained MD is run at every image (set it up with meanforce.py, pointing
--points at the final string).  From each run this takes the mean force
f = k <theta - z> = -dF/dz and integrates it along the string,

    F(s) = - int_0^s f(s') . z'(s') ds',

with z(s) a cubic spline through the images and f(s) a local cubic Hermite
interpolant of the forces, s the arc length in scaled CV units (see
`integrate` for why this rule and not the trapezoid rule or a spline through
the forces).  The line integral itself does not depend on how the CVs are
scaled; only the curve drawn between the images does.

It reports, for the Stage 6 gates:
  * the profile and its standard error: each run is cut into blocks, every
    block is integrated into a whole profile, and the error is the spread
    over those profiles (so correlation between the CV components of an
    image's force is kept);
  * the barrier and the end-to-end free-energy difference, with errors;
  * OVERLAP of neighbouring restrained ensembles along the path;
  * STABILITY: the first half of the data against the second, each with its
    own error, for the barrier AND the end-to-end difference;
  * RESOLUTION: every other image against all of them, for both numbers.
The start of each run is dropped by detecting its equilibration (the start
that leaves the most independent samples), unless --discard gives a fraction.

On overlap: with stiff restraints the ensembles of neighbouring images are
much narrower than their spacing and do not overlap at all.  Integrating a
mean force does not need them to; it needs the mean force to be resolved,
which is what the resolution check tests.  The overlap gate is kept because
the proposal names it; --min-overlap 0 turns it off.

`compare` checks that the closed-to-open free-energy difference agrees
between strings.  It must: that difference does not depend on the path.

Example:
  fe_profile.py compute --dir fe0 --scales endpoints_scales.json --out fe0_profile
  fe_profile.py compare --a fe0_profile.json --b fe1_profile.json
"""
import argparse
import os
import sys

import numpy as np

from common import block_mean_se, load_json, path_length, read_cvout, save_json, wrap


def integrate_trapezoid(z, f):
    """Profile from mean forces (f = -dF/dz) at the images z (continuous), by the trapezoid rule."""
    dz = np.diff(z, axis=0)
    steps = -0.5 * np.sum((f[:-1] + f[1:]) * dz, axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def path_parameter(z, scale=None):
    """Arc length along the images in the CVs' scaled units (the units the rest
    of thalweg measures distance in).  Without scales, in the CVs' own units.

    (Until 2026-11-15 each CV's steps were divided by the range that CV
    covers along the string.  A CV that hardly moves then sets the parameter
    from its noise: on feR0r zeta, spanning 0.044 rad, carried 54% of it.)"""
    z = np.asarray(z, float)
    sc = np.ones(z.shape[1]) if scale is None else np.asarray(scale, float) * np.ones(z.shape[1])
    return np.concatenate([[0.0], np.cumsum(np.sqrt(((np.diff(z, axis=0) / sc) ** 2).sum(axis=1)) + 1e-12)])


def _node_slopes(s, f):
    """dF/ds at the nodes from the three-point (non-uniform) difference, one-sided
    at the ends.  Linear in f and local: one image's noise reaches two segments."""
    h = np.diff(s); d = np.empty_like(f)
    h0, h1 = h[:-1, None], h[1:, None]
    d[1:-1] = (-h1 / (h0 * (h0 + h1)) * f[:-2] + (h1 - h0) / (h0 * h1) * f[1:-1] + h0 / (h1 * (h0 + h1)) * f[2:])
    d[0] = (f[1] - f[0]) / h[0]; d[-1] = (f[-1] - f[-2]) / h[-1]
    return d


def integrate(z, f, scale=None, s=None, at=None):
    """Profile from mean forces (f = -dF/dz) at the images z (continuous).

    F = -int f . dz along a curve through the images.  The path is a cubic
    spline z(s) through the images, s the scaled arc length (path_parameter).
    The force is a LOCAL cubic Hermite interpolant of each component, with
    the slope at an image from it and its two neighbours, so the integrand
    f(s) . z'(s) is a polynomial of degree five on each segment and is
    integrated exactly (Gauss-Legendre, 3 points).  The result is linear in
    the forces, so block profiles average to the profile of the mean force.

    Why not the alternatives (numbers from the review cases, 2026-11-15):
      * trapezoid rule (path straight, force linear between images): on
        16-image alanine strings 0.2 kcal/mol low at the barrier; over eight
        real strings the rms difference from the unbiased reference is 0.25
        kcal/mol against 0.20 here;
      * a global cubic spline through the force (the rule until 2026-11-15):
        rms 0.22, but it overshoots where the force changes sign (feR1f image
        20: 4.82, here 4.75, trapezoid 4.69, reference 4.56) and spreads one
        image's noise over its neighbours: 1.5 to 2.2 times the trapezoid's
        noise on refined spacing, against at most 1.15 here;
      * the force projected on the spline's tangent at each image, then
        integrated: fails on a path whose images are not on a smooth curve
        (0.4 kcal/mol off at 0.05 of jitter, here 0.02);
      * PCHIP or Akima interpolation of the force: as accurate as this, but
        not linear in the forces.
    With fewer than four images it falls back to the trapezoid rule.

    s: the images' parameter values, if not path_parameter(z, scale) (the
    resolution check passes the full string's values for a subset of its
    images).  at: parameter values to return F at, instead of the images'
    (F is then read off the same curve between images)."""
    z = np.asarray(z, float); f = np.asarray(f, float)
    if len(z) < 4 and at is None:
        return integrate_trapezoid(z, f)
    s = path_parameter(z, scale) if s is None else np.asarray(s, float)
    n = len(s); h = np.diff(s)
    if n < 4:                                                  # straight segments, force linear along them
        def dz(x):
            k = np.clip(np.searchsorted(s, x, "right") - 1, 0, n - 2)
            return (z[k + 1] - z[k]) / h[k, None]
        df = None
    else:
        from scipy.interpolate import CubicSpline
        dz = CubicSpline(s, z).derivative(); df = _node_slopes(s, f)
    x, w = np.polynomial.legendre.leggauss(3)

    def piece(k, t):
        """-int f . dz from s[k] to s[k] + t h[k], for arrays k and t in [0, 1]."""
        out = np.zeros(len(k)); hk = h[k]
        for xi, wi in zip(x, w):
            u = 0.5 * t * (xi + 1.0)
            if df is None:
                fu = (1 - u)[:, None] * f[k] + u[:, None] * f[k + 1]
            else:
                h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
                fu = (h00[:, None] * f[k] + (h10 * hk)[:, None] * df[k] + h01[:, None] * f[k + 1]
                      + (h11 * hk)[:, None] * df[k + 1])
            out -= 0.5 * wi * t * hk * (fu * dz(s[k] + u * hk)).sum(axis=1)
        return out
    seg = np.arange(n - 1)
    nodes = np.concatenate([[0.0], np.cumsum(piece(seg, np.ones(n - 1)))])
    if at is None:
        return nodes
    at = np.atleast_1d(np.asarray(at, float))
    k = np.clip(np.searchsorted(s, at, "right") - 1, 0, n - 2)
    return nodes[k] + piece(k, (at - s[k]) / h[k])


def neighbour_overlap(theta, z, periodic, bins=40):
    """Overlap of neighbouring ensembles projected on the local tangent."""
    out = []
    for i in range(len(z) - 1):
        t = z[i + 1] - z[i]
        L2 = np.sum(t * t)
        a = wrap(theta[i] - z[i], periodic) @ t / L2          # 0 at image i, 1 at image i+1
        b = wrap(theta[i + 1] - z[i], periodic) @ t / L2
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
        hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
        out.append(float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum()))
    return np.array(out)


def refine_images(z, f, scale, n_extra, must_split=()):
    """Anchors with n_extra images added where the mean force curves most,
    after one image in the middle of every segment listed in must_split
    (segments whose two ensembles do not overlap: curvature does not see
    those, and on a 47-image alanine string two such segments were all that
    kept the overlap gate from passing).

    The segments next to the largest second difference of the scaled mean
    force are split at their midpoints, largest first; a segment already
    split is scored on its halves.  Returns (new anchors, indices of
    the original segments that were split)."""
    z = [np.asarray(p, float) for p in z]; f = [np.asarray(v, float) for v in f]
    seg = list(range(len(z) - 1)); split = []
    for s0 in sorted(set(int(v) for v in must_split), reverse=True):      # from the far end, so indices stay valid
        z.insert(s0 + 1, 0.5 * (z[s0] + z[s0 + 1])); f.insert(s0 + 1, 0.5 * (f[s0] + f[s0 + 1]))
        split.append(seg[s0]); seg.insert(s0 + 1, seg[s0])
    for _ in range(n_extra):
        # the trapezoid rule is exact for a force that changes linearly; its
        # error follows the CURVATURE of the force, so a segment is scored by
        # the second difference at its two ends
        n = len(z)
        # (as a change of slope, so unequal spacing after a split does not
        # look like curvature)
        h = [max(np.linalg.norm((z[i + 1] - z[i]) / scale), 1e-12) for i in range(n - 1)]
        slope = [(f[i + 1] - f[i]) * scale / h[i] for i in range(n - 1)]
        curv = [np.linalg.norm(slope[i] - slope[i - 1]) if 0 < i < n - 1 else 0.0 for i in range(n)]
        score = [max(curv[i], curv[i + 1]) * h[i] for i in range(n - 1)]
        i = int(np.argmax(score))
        z.insert(i + 1, 0.5 * (z[i] + z[i + 1])); f.insert(i + 1, 0.5 * (f[i] + f[i + 1]))
        split.append(seg[i]); seg.insert(i + 1, seg[i])
        # the two halves each carry half the change, so they drop down the ranking
    return np.array(z), split


def statistical_inefficiency(x):
    """g = 1 + 2 sum of the normalised autocorrelation, summed until it first
    falls to zero.  N / g is the number of independent samples in x."""
    x = np.asarray(x, float); n = len(x)
    d = x - x.mean(); v = float(np.dot(d, d)) / n
    if n < 4 or v <= 0:
        return 1.0
    g = 1.0
    for t in range(1, n - 1):
        c = float(np.dot(d[:-t], d[t:])) / ((n - t) * v)
        if c <= 0:
            break
        g += 2.0 * c * (1.0 - t / n)
    return max(g, 1.0)


def detect_equilibration(x, grid=20, max_fraction=0.6, floor=0.0, gain=1.5):
    """How much of the start of a series to drop (after Chodera, JCTC 2016):
    the start that leaves the most independent samples, (N - t0) / g(t0).  A
    series still relaxing has a long correlation time until the relaxation is
    cut off.

    Used bare, that rule is noisy: on stationary correlated data it cuts 5 to
    20% for nothing and makes the error look smaller than it is, and it does
    not see a weak slow relaxation under the noise.  So `floor` is always
    dropped, and more is dropped only when that raises the number of
    independent samples by the factor `gain`.  Returns (t0, g, N_eff)."""
    x = np.asarray(x, float); n = len(x)
    t_floor = int(floor * n)
    if n - t_floor < 8:
        return (t_floor, 1.0, float(max(n - t_floor, 0)))
    g0 = statistical_inefficiency(x[t_floor:]); base = (t_floor, g0, (n - t_floor) / g0)
    best = base
    for t0 in np.unique(np.linspace(t_floor, max_fraction * n, grid).astype(int)):
        if n - t0 < 8:
            break
        g = statistical_inefficiency(x[t0:])
        neff = (n - t0) / g
        if neff > best[2]:
            best = (int(t0), g, neff)
    return best if best[2] >= gain * base[2] else base


def from_force_samples(z, samples, scale, nblocks=10, resolution_tol=0.5, drift_blocks=20, drift_alpha=0.01):
    """Profile and its checks from per-image samples of the mean force.

    z: (n, D) images.  samples: list of (m_i, D) arrays, each row one sample
    of f = -dF/dz at that image, in time order.

    ERRORS.  Each image's samples are cut into nblocks consecutive blocks;
    block b of every image is integrated into a whole profile; the error of
    anything read off the profile is the spread of that thing over the block
    profiles.  Propagating through the integral this way, instead of from a
    separate error bar on each force component, keeps the correlation between
    the components of one image's force.  With few blocks the error itself is
    uncertain, so limits use Student's t (t95), not 2.

    STABILITY.  The first and second half of every run give two independent
    profiles.  The barrier and the end-to-end difference must agree between
    them within their combined error, each from the blocks of its own half:
    a drift cannot hide in an error bar it inflated itself.

    RESOLUTION.  The same two numbers from every other image."""
    from scipy.stats import t as student
    z = np.asarray(z, float); n = len(z)
    nb = int(nblocks)
    if nb < 6 or nb % 2:
        raise ValueError(f"{nblocks} blocks: need an even number, at least 6 (three against three is the least a "
                         "comparison of halves can use)")
    short = min(len(s) for s in samples)
    if short < 2 * nb:
        raise ValueError(f"an image has only {short} force samples; {nb} blocks need at least {2 * nb}")
    fb = np.empty((nb, n, z.shape[1]))
    for i, d in enumerate(samples):
        for b, chunk in enumerate(np.array_split(np.asarray(d, float), nb)):
            fb[b, i] = chunk.mean(axis=0)
    f = np.array([np.asarray(d, float).mean(axis=0) for d in samples])
    Fb = np.array([integrate(z, fb[b], scale) for b in range(nb)])           # one profile per block
    F = integrate(z, f, scale)
    top = int(F.argmax()); start = int(F[: top + 1].argmin())

    def err(q, rows=slice(None)):                                      # standard error of a quantity over the blocks
        v = np.asarray(q)[rows]
        return float(v.std(ddof=1) / np.sqrt(len(v)))

    h = nb // 2; first, second = slice(0, h), slice(h, nb)
    q_bar, q_dF = Fb[:, top] - Fb[:, start], Fb[:, -1] - Fb[:, 0]
    t95 = float(student.ppf(0.975, nb - 1)); t99h = float(student.ppf(0.995, nb - 2))
    res = {
        "z": z, "arc": path_length(z, scale), "F": F - F[start],
        "F_se": np.array([err(Fb[:, i] - Fb[:, start]) for i in range(n)]),
        "mean_force": f, "mean_force_se": fb.std(axis=0, ddof=1) / np.sqrt(nb),
        "top_image": top, "start_image": start, "blocks": nb, "t95": t95,
        "barrier": float(F[top] - F[start]), "barrier_se": err(q_bar),
        "delta_F": float(F[-1] - F[0]), "delta_F_se": err(q_dF),
    }
    ok = True
    for name, q in (("barrier", q_bar), ("delta_F", q_dF)):
        a, b = float(q[first].mean()), float(q[second].mean())
        se = float(np.hypot(err(q, first), err(q, second)))
        res[name + "_first_half"], res[name + "_second_half"], res[name + "_halves_se"] = a, b, se
        res["stable_" + name] = bool(abs(a - b) <= t99h * se)
        ok = ok and res["stable_" + name]
    # ... and a trend through all the blocks: a steady drift splits evenly
    # between the halves' own spreads and is seen much sooner as a slope
    t99s = float(student.ppf(0.995, nb - 2)); xb = np.arange(nb) - (nb - 1) / 2.0
    for name, q in (("barrier", q_bar), ("delta_F", q_dF)):
        slope = float(np.dot(xb, q) / np.dot(xb, xb)); r = q - q.mean() - slope * xb
        se_s = float(np.sqrt(np.dot(r, r) / (nb - 2) / np.dot(xb, xb)))
        res[name + "_trend"] = slope * (nb - 1); res[name + "_trend_se"] = se_s * (nb - 1)   # change from first block to last
        res["trend_ok_" + name] = bool(abs(slope) <= t99s * se_s)
        res["stable_" + name] = bool(res["stable_" + name] and res["trend_ok_" + name])
        ok = ok and res["trend_ok_" + name]
        # the smallest steady drift over the run this gate would have caught half the time
        res[name + "_detectable"] = t99s * se_s * (nb - 1)
    # ... and every image on its own.  One image's force drifting moves the
    # profile from that image on, but in the two aggregate numbers it is
    # diluted by every other image's noise: on feR0r a 0.05 rad drift of psi
    # at one image moved the barrier from 3.35 to 3.76 and passed both.  So
    # each image's force along the string (times the length of string it
    # stands for: a contribution to F) gets a trend test through drift_blocks
    # blocks of its run, at a family-wise level drift_alpha over the images
    # (Bonferroni).  The blocks' own scatter is the error, so correlation
    # shorter than a block does not matter; the first-zero-crossing
    # inefficiency underestimates slow tails 2-3x on the alanine runs, which
    # made a sample-level test fail every real string.
    nbt = int(max(6, min(drift_blocks, short // 2)))
    xt = np.arange(nbt) - (nbt - 1) / 2.0; sxx = float(np.dot(xt, xt))
    tv, chg = np.zeros(n), np.zeros(n)
    for i, d in enumerate(samples):
        tng = 0.5 * (z[min(i + 1, n - 1)] - z[max(i - 1, 0)])
        bm = np.array([c.mean() for c in np.array_split(np.asarray(d, float) @ tng, nbt)])
        sl = float(np.dot(xt, bm - bm.mean()) / sxx); r = bm - bm.mean() - sl * xt
        se_sl = float(np.sqrt(max(np.dot(r, r), 1e-300) / (nbt - 2) / sxx))
        tv[i], chg[i] = sl / se_sl, sl * (nbt - 1)
    crit = float(student.isf(drift_alpha / 2.0 / n, nbt - 2))
    res["image_trend_t"] = tv; res["image_trend_change"] = chg; res["image_trend_limit"] = crit
    res["image_trend_blocks"] = nbt; res["drift_alpha"] = float(drift_alpha)
    res["drifting_images"] = [int(i) for i in np.flatnonzero(np.abs(tv) > crit)]
    res["images_stationary"] = not res["drifting_images"]
    ok = ok and res["images_stationary"]
    res["stable"] = bool(ok); res["stability_limit_t"] = t99h
    # BLOCKS MUST BE LONGER THAN THE CORRELATION TIME, or every error above is
    # too small (with a correlation time of a fifth of the run the quoted
    # error is less than half the true one)
    worst = 0.0
    for i, d in enumerate(samples):
        d = np.asarray(d, float)
        tng = z[min(i + 1, n - 1)] - z[max(i - 1, 0)]
        worst = max(worst, statistical_inefficiency(d @ tng) * nb / len(d))
    res["correlation_to_block_ratio"] = float(worst); res["blocks_ok"] = bool(worst <= 0.2)
    # unequal runs: an image with a fraction of the others' samples sets the error and nobody notices
    lens = np.array([len(s) for s in samples]); res["samples"] = lens
    res["short_image"] = int(lens.argmin()) if lens.min() < 0.5 * np.median(lens) else None
    res["samples_ok"] = res["short_image"] is None
    # is the top where it seems?  with another image within the error of the
    # top the barrier's position is not determined, and picking the highest
    # of several noisy values biases the barrier upward
    Ftop = F - F[top]
    near = [int(i) for i in range(n) if i != top and abs(i - top) > 1
            and abs(Ftop[i]) <= t95 * err(Fb[:, i] - Fb[:, top])]
    res["top_ambiguous_with"] = near
    # every other image (and the last), the profile read off THEIR curve at
    # every image: what the coarser string predicts where the images were
    # dropped.  (Until 2026-11-15 the top and start images were kept, so a
    # feature between them could not show, and the limit used the barrier's
    # own error, which is not the error of the difference of two profiles
    # from the same data; that is now taken block by block.)
    sp = path_parameter(z, scale)
    idx = np.arange(0, n, 2)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    Fc = integrate(z[idx], f[idx], s=sp[idx], at=sp)
    Fcb = np.array([integrate(z[idx], fb[b][idx], s=sp[idx], at=sp) for b in range(nb)])
    res["barrier_half_images"] = float(Fc[top] - Fc[start])
    res["delta_F_half_images"] = float(Fc[-1] - Fc[0])
    for name, q, qc in (("barrier", q_bar, Fcb[:, top] - Fcb[:, start]), ("delta_F", q_dF, Fcb[:, -1] - Fcb[:, 0])):
        d = res[name] - res[name + "_half_images"]; se_d = err(q - qc)
        res[name + "_resolution_se"] = se_d
        # beyond the noise, 0.5 kcal/mol: with an error that falls at least
        # four-fold when the spacing halves, the full string's own spacing
        # error is under a third of that, 0.17 kcal/mol (0.3 kT)
        res["resolved_" + name] = bool(abs(d) <= t95 * se_d + resolution_tol)
    res["resolution_tol"] = float(resolution_tol)
    res["resolved"] = bool(res["resolved_barrier"] and res["resolved_delta_F"])
    # a rule whose error falls four-fold when the spacing halves leaves a
    # third of (all - half) with all the images.  The Hermite rule does better
    # than that on a smooth profile, so this is on the safe side.
    res["barrier_discretisation"] = abs(res["barrier"] - res["barrier_half_images"]) / 3.0
    res["delta_F_discretisation"] = abs(res["delta_F"] - res["delta_F_half_images"]) / 3.0
    Ft = integrate_trapezoid(z, f)
    res["barrier_trapezoid"] = float(Ft[top] - Ft[start]); res["delta_F_trapezoid"] = float(Ft[-1] - Ft[0])
    return res


def print_gates(res):
    w = res.get("restraint_width")
    if w is not None and np.max(w) > 0.3:
        print("NOTE: the restraint lets a CV move by " + ", ".join(f"{v:.2f}" for v in np.asarray(w))
              + " of its length scale (sqrt(kT/k) / scale, per CV). The profile is of the free energy smoothed over "
              "that width, exactly; features narrower than it are flattened. A stiffer restraint narrows it; "
              "'fe_profile.py reference' compares like with like whatever the stiffness.")
    t = res["stability_limit_t"]
    print(f"GATE stability: {'PASS' if res['stable'] else 'FAIL'} (first against second half of the data, each with its own error, "
          f"and the trend through all {res['blocks']} blocks; limit {t:.1f} standard errors)")
    for name, label in (("barrier", "barrier"), ("delta_F", "end-to-end")):
        d = res[name + "_first_half"] - res[name + "_second_half"]
        print(f"     {label:10s} halves {res[name + '_first_half']:6.2f} and {res[name + '_second_half']:6.2f} (differ by {d:+.2f} +/- "
              f"{res[name + '_halves_se']:.2f}); change over the run {res[name + '_trend']:+.2f} +/- {res[name + '_trend_se']:.2f}  "
              f"{'ok' if res['stable_' + name] else 'NOT STABLE'}; a steady drift under {res[name + '_detectable']:.2f} would not show")
    tv = np.abs(res["image_trend_t"]); w = int(tv.argmax())
    print(f"     per image  trend of each image's force through {res['image_trend_blocks']} blocks: largest {tv[w]:.1f} standard "
          f"errors (image {w}, its contribution to F changed by {res['image_trend_change'][w]:+.2f}); limit {res['image_trend_limit']:.1f} "
          f"(family-wise {res['drift_alpha']:g} over {len(tv)} images)  "
          + ("ok" if res["images_stationary"] else f"DRIFTING at images {res['drifting_images']}"))
    print(f"GATE block length: {'PASS' if res['blocks_ok'] else 'FAIL'} (the force stays correlated for "
          f"{res['correlation_to_block_ratio']:.0%} of a block at worst; limit 20%)")
    if not res["blocks_ok"]:
        print("     blocks have to be several correlation times long; as it is, every error here is too small. Run longer.")
    if res["short_image"] is not None:
        print(f"GATE equal runs: FAIL (image {res['short_image']} has {int(res['samples'].min())} samples against a median of "
              f"{int(np.median(res['samples']))}; its error dominates and its run may not have relaxed. Finish that run.)")
    else:
        print("GATE equal runs: PASS (no image has under half the median number of samples)")
    if res["top_ambiguous_with"]:
        print(f"note: images {res['top_ambiguous_with']} are as high as the top (image {res['top_image']}) within the error: where "
              "the barrier is, is not settled, and the highest of several noisy values overstates it")
    print(f"GATE resolution: {'PASS' if res['resolved'] else 'FAIL'} (every other image against all of them, the "
          "difference against its own error)")
    for name, label in (("barrier", "barrier"), ("delta_F", "end-to-end")):
        print(f"     {label:10s} {res[name]:7.2f} with all images, {res[name + '_half_images']:7.2f} with half (difference "
              f"{res[name] - res[name + '_half_images']:+.2f} +/- {res[name + '_resolution_se']:.2f}, limit {res['t95']:.1f} errors + "
              f"{res['resolution_tol']:g})  {'ok' if res['resolved_' + name] else 'NOT RESOLVED'}; estimated error from the spacing "
              f"{res[name + '_discretisation']:.2f}")


def read_driver_forces(paths):
    """The .forces files of a thalweg driver run (one per leg, in order).
    Returns (updates, z[frame, image, D], g[frame, image, D], n_updates per frame).

    A leg that was killed leaves frames past the restart point the next leg
    resumed from; when a leg starts, the earlier legs' frames at or after its
    first update are dropped.  A line or a frame cut short is dropped."""
    frames = {}; width = None; n = None
    for p in paths:
        rows = []
        with open(p, errors="replace") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                try:
                    v = [float(t) for t in line.split()]
                except ValueError:
                    continue                                  # cut in the middle of a number
                if width is None:
                    width = len(v)
                if len(v) == width:
                    rows.append(v)
        if not rows:
            continue
        a = np.array(rows)
        n = int(a[:, 1].max()) + 1 if n is None else n
        first = int(a[0, 0])
        for k in [k for k in frames if k >= first]:
            del frames[k]
        for u in np.unique(a[:, 0]):
            fr = a[a[:, 0] == u]
            if len(fr) == n and np.all(fr[:, 1] == np.arange(n)):
                frames[int(u)] = fr                           # anything else is a frame cut short
    if not frames:
        raise ValueError("the force files hold no whole frames")
    r = np.array([frames[k] for k in sorted(frames)])
    D = (r.shape[2] - 3) // 2
    return r[:, 0, 0].astype(int), r[:, :, 3:3 + D], r[:, :, 3 + D:], r[:, 0, 2].astype(int)


def window_motion(Z, scale):
    """How a string moved across a window of frames.  Per image: a straight
    line is fitted to each CV against the frame number.  Returns
      net     the fitted change from first frame to last, as a scaled distance, largest over images;
      t       the largest slope / its standard error over images and CVs (is the drift real?);
      wander  the largest scaled distance of any quarter of the window from the window's mean
              (an excursion out and back has no net drift at all)."""
    Z = np.asarray(Z, float) / scale; L = len(Z)
    xb = np.arange(L) - (L - 1) / 2.0; sxx = float(np.dot(xb, xb))
    slope = np.einsum("f,fid->id", xb, Z) / sxx
    resid = Z - Z.mean(axis=0) - xb[:, None, None] * slope
    se = np.sqrt(np.maximum((resid ** 2).sum(axis=0) / max(L - 2, 1) / sxx, 1e-300))
    net = float(np.sqrt(((slope * (L - 1)) ** 2).sum(axis=1)).max())
    quarters = np.array([c.mean(axis=0) for c in np.array_split(Z, 4)]) - Z.mean(axis=0)
    wander = float(np.sqrt((quarters ** 2).sum(axis=2)).max())
    return net, float(np.abs(slope / se).max()), wander


def window_drift(Z, scale):   # kept for callers; from_driver uses window_motion
    """How far the string moved across a window of frames: the scaled distance
    between each image's mean position in the first and in the second half,
    largest over the images."""
    h = len(Z) // 2
    d = (Z[:h].mean(axis=0) - Z[len(Z) - h:].mean(axis=0)) / scale
    return float(np.sqrt((d ** 2).sum(axis=1)).max())


def _to_curve(pts, curve):
    """Distance from each point to the polyline `curve` (both already scaled)."""
    a = curve[:-1]; ab = curve[1:] - a; l2 = np.maximum((ab * ab).sum(axis=1), 1e-300)
    out = np.empty(len(pts))
    for i, p in enumerate(pts):
        t = np.clip(((p - a) * ab).sum(axis=1) / l2, 0.0, 1.0)
        out[i] = np.sqrt((((a + t[:, None] * ab) - p) ** 2).sum(axis=1).min())
    return out


def _fine(x, per_segment=12):
    """A fine polyline along the natural cubic spline through the points x."""
    from scipy.interpolate import CubicSpline
    s = np.concatenate([[0.0], np.cumsum(np.sqrt((np.diff(x, axis=0) ** 2).sum(axis=1)) + 1e-12)])
    return CubicSpline(s, x, bc_type="natural")(np.linspace(0.0, s[-1], (len(x) - 1) * per_segment + 1))


def plateau_motion(Z, scale):
    """The driver's plateau test (tw::plateau), on these frames.

    The frames are cut into four equal blocks.  For every image, g2 is its
    squared distance, as a curve, between the first block's mean string and
    the last's, and f2 the same between neighbouring blocks (mean of the
    three pairs).  Returns
      systematic  the largest sqrt(g2 - f2) over the images: the movement
                  across these frames that the string's own wander between
                  neighbouring blocks does not account for;
      wander      the rms of sqrt(f2) over the images."""
    Z = np.asarray(Z, float) / scale
    L = len(Z) // 4
    if L < 1:
        raise ValueError("need at least 4 frames")
    Q = [Z[len(Z) - (4 - j) * L: len(Z) - (3 - j) * L].mean(axis=0) for j in range(4)]

    def sq(A, B):
        return (_to_curve(B, _fine(A)) ** 2 + _to_curve(A, _fine(B)) ** 2) / 2.0
    g2 = sq(Q[0], Q[3]); f2 = (sq(Q[0], Q[1]) + sq(Q[1], Q[2]) + sq(Q[2], Q[3])) / 3.0
    return float(np.sqrt(np.clip(g2 - f2, 0.0, None)).max()), float(np.sqrt(f2.mean()))


def stationary(Z, scale, max_drift):
    """Has the string come to rest over these frames?  The same question the
    driver asks, answered the same way (plateau_motion): no image has moved
    across them by more than max_drift beyond the string's own wander.

    (Until 2026-11-15 this compared a fitted net motion and the largest
    excursion with a limit of 0.03.  A string that moves on a noisy force
    with a constant step wanders by 0.06 to 0.07 between any two stretches
    of its run, at rest or not, so that test failed every real run.)
    Returns (ok, net, t, wander, systematic); net and t are the straight-line
    fit of window_motion, kept for the report."""
    net, t, _ = window_motion(Z, scale)
    systematic, wander = plateau_motion(Z, scale)
    return bool(systematic <= max_drift), net, t, wander, systematic


def from_driver(paths, scale, last=None, nblocks=6, max_drift=0.2):
    """Free-energy profile from the forces a thalweg driver run recorded.

    Each frame holds every image's mean gradient over that frame's updates
    and the mean position they were measured at.  Only frames from after the
    string came to rest are used: the longest run of final frames that
    `stationary` accepts and that covers at least the last half of the run.
    (A creeping string always has SOME short final stretch over which it
    moved less than any limit; half the run is what the driver tests too.)
    `last` uses that many final frames instead, but they are tested the same
    way and the result says so if they fail."""
    upd, Z, G, nup = read_driver_forces(paths)
    need = 2 * nblocks
    if len(Z) < need:
        raise ValueError(f"{len(Z)} frames recorded; need at least {need}")
    if last is None:
        first = None
        for f in range(0, max(1, min(len(Z) // 2, len(Z) - need) + 1)):
            if stationary(Z[f:], scale, max_drift)[0]:
                first = f; break
        if first is None:
            half = Z[min(len(Z) // 2, len(Z) - need):]
            _, net, t, wander, systematic = stationary(half, scale, max_drift)
            raise ValueError(f"the string is still moving: over the last {len(half)} frames one image moved "
                             f"{systematic:.3f} scaled units beyond the string's own wander of {wander:.3f} "
                             f"(limit {max_drift}). Run it longer, or pin its ends if one is creeping.")
    else:
        first = max(0, len(Z) - last)
    use = slice(first, len(Z))
    nf = len(Z) - first
    if nf < need:
        raise ValueError(f"{nf} frames given; need at least {need}")
    ok, net, t, wander, systematic = stationary(Z[use], scale, max_drift)
    zbar = Z[use].mean(axis=0)
    res = from_force_samples(zbar, [-G[use, i, :] for i in range(Z.shape[1])], scale, nblocks)
    res.update({"frames_used": nf, "frames_total": len(Z), "first_update": int(upd[first]), "last_update": int(upd[-1]),
                "updates_per_frame": int(nup[-1]), "updates_used": int(nup[use].sum()), "drift_in_window": net, "drift_t": t, "wander_in_window": wander,
                "systematic_in_window": systematic, "stationary": ok})
    return res


def compute(z, theta, k, periodic, scale, nblocks=10, min_overlap=0.05, discard="auto", resolution_tol=0.5, floor=0.1,
            kT=0.0019872041 * 300.0):
    """z: (n, D) images.  theta: list of (samples, D) CV series, one per image,
    whole (nothing discarded yet).  discard: a fraction, or "auto" to find
    each image's own equilibration time from its force along the string."""
    k = np.asarray(k, float); z = np.asarray(z, float); n = len(z)
    forces = [wrap(np.asarray(th, float) - z[i], periodic) * k for i, th in enumerate(theta)]
    dropped = []
    for i in range(n):
        m = len(forces[i])
        if discard == "auto":
            # the part of the force that enters the integral: along the local direction of the string
            t = z[min(i + 1, n - 1)] - z[max(i - 1, 0)]
            t0 = detect_equilibration(forces[i] @ t, floor=floor)[0]
        else:
            t0 = int(float(discard) * m)
        dropped.append(t0 / m)
        forces[i] = forces[i][t0:]
    theta = [np.asarray(th)[len(th) - len(f):] for th, f in zip(theta, forces)]     # the caller's list is left alone
    res = from_force_samples(z, forces, scale, nblocks, resolution_tol)
    res["discarded_fraction"] = np.array(dropped)
    ov = neighbour_overlap(theta, z, periodic)
    res["overlap"] = ov; res["min_overlap"] = float(ov.min())
    res["overlap_ok"] = bool(ov.min() >= min_overlap)
    # the restraint's own width against each CV's length scale: what the runs
    # measure is the free energy smoothed over that width (see reference_smoothed)
    res["restraint_width"] = np.sqrt(kT / np.maximum(k, 1e-12)) / np.asarray(scale, float)
    res["kT"] = float(kT)
    return res


def reference_smoothed(z, runs, k, periodic, kT=0.596, n_boot=3000, seed=0, zero=0):
    """The profile restrained runs SHOULD give, from unbiased trajectories.

    With a harmonic restraint of force constant k in each CV the mean force
    k<z - theta> is the gradient of F_k(z) = -kT ln [rho * G](z), the free
    energy of the unbiased density rho smoothed by a Gaussian of variance
    kT/k per CV.  That is exact for any k, not a stiff-restraint limit.  So
    with unbiased frames theta_t,

        F_k(z_i) - F_k(z_0) = -kT ln [ sum_t w_i(t) / sum_t w_0(t) ],
        w_i(t) = exp(-sum_j k_j (z_ij - theta_tj)^2 / 2kT),

    estimates the very quantity mean-force integration along the images
    gives, with nothing to choose.  Counting frames within a radius of each
    image does not: it averages the density over the ball, and on a 4-D test
    density with a barrier of 3.30 it returned 2.80 at a radius of 0.6 and
    1.90 at 1.0.

    runs: list of (frames, D) arrays, one per independent trajectory.
    Errors come from resampling whole trajectories.  F is zeroed at image
    `zero` (a profile's start image, so the two are compared like with like),
    and the barrier is defined as a profile's is: F[top] - F[start], top the
    highest image and start the lowest before it.  A resample with no frame
    near some image has no F there and is dropped; how many were is returned
    (dropping them leaves out the resamples that would widen the error).
    Returns a dict."""
    z = np.asarray(z, float); k = np.asarray(k, float) * np.ones(z.shape[1]); periodic = np.asarray(periodic, bool)
    S = np.zeros((len(runs), len(z))); S2 = np.zeros((len(runs), len(z)))
    for a, x in enumerate(runs):
        x = np.asarray(x, float)
        for i, zi in enumerate(z):
            d = wrap(x - zi, periodic)
            w = np.exp(-0.5 * (d * d * k).sum(axis=1) / kT); S[a, i] = w.sum(); S2[a, i] = (w * w).sum()
    tot = S.sum(axis=0)
    if tot.min() <= 0:
        raise ValueError("no unbiased frame near some image: the trajectories never visit that part of the path")
    F = -kT * np.log(tot / tot[zero]); rng = np.random.default_rng(seed); Fb = []
    for _ in range(n_boot):
        c = S[rng.integers(0, len(runs), len(runs))].sum(axis=0)
        if c.min() > 0:
            Fb.append(-kT * np.log(c / c[zero]))
    if not Fb:
        raise ValueError("every bootstrap resample left some image with no unbiased frame near it")
    Fb = np.array(Fb); top = int(F.argmax()); start = int(F[: top + 1].argmin())
    return {"z": z, "F": F, "F_se": Fb.std(axis=0), "zero_image": int(zero),
            "barrier": float(F[top] - F[start]), "barrier_se": float((Fb[:, top] - Fb[:, start]).std()),
            "top_image": top, "start_image": start, "delta_F": float(F[-1] - F[0]), "delta_F_se": float((Fb[:, -1] - Fb[:, 0]).std()),
            "effective_frames": tot ** 2 / S2.sum(axis=0), "k": k, "runs": len(runs), "resamples_used": len(Fb),
            "resamples_dropped": int(n_boot - len(Fb))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compute")
    p.add_argument("--dir", required=True, help="directory written by 'meanforce.py setup' and filled by the runs")
    p.add_argument("--scales", required=True)
    p.add_argument("--discard", default="auto",
                   help="leading fraction of each run to drop, or 'auto': drop --floor, and more at an image only where "
                        "that clearly raises its number of independent samples [auto]")
    p.add_argument("--floor", type=float, default=0.1, help="fraction always dropped with --discard auto [0.1]")
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--resolution-tol", type=float, default=0.5,
                   help="allowed change, beyond the error, when half the images are dropped (kcal/mol) [0.5]")
    p.add_argument("--min-overlap", type=float, default=0.05)
    p.add_argument("--refine", type=int, default=0,
                   help="write anchors with this many images added where the mean force changes fastest "
                        "(default: as many as the string has segments, when the resolution check fails)")
    p.add_argument("--temperature", type=float, default=300.0, help="of the restrained runs, for the restraint's width [300]")
    p.add_argument("--out", default="profile")
    p = sub.add_parser("driver", help="profile from the forces a thalweg driver run recorded: no extra MD")
    p.add_argument("--forces", nargs="+", required=True, help="the run's .forces files, one per leg, in order")
    p.add_argument("--scales", required=True)
    p.add_argument("--last", type=int, help="use this many final frames (default: the longest final run over which the "
                                           "string did not move); they are tested the same way")
    p.add_argument("--max-drift", type=float, default=0.2,
                   help="largest scaled distance any image may have moved across the frames used, beyond the "
                        "string's own wander between neighbouring quarters of them (the driver's plateau test) [0.2]")
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--min-updates", type=int, default=20000, help="updates per image the profile must rest on [20000]")
    p.add_argument("--out", default="profile_driver")
    p = sub.add_parser("reference", help="what a profile should be, from unbiased trajectories (see reference_smoothed)")
    p.add_argument("--profile", required=True, help="a profile JSON from 'compute' or 'driver' (its images are used)")
    p.add_argument("--cvout", nargs="+", required=True, help="cfacv CV reports of unbiased runs, one per trajectory, in the profile's CVs")
    p.add_argument("--k", type=float, nargs="+", required=True, help="the restraint force constant the profile's runs used, one per CV or one for all")
    p.add_argument("--cv", help="CV definition JSON, to know which CVs are periodic (required when the profile does not "
                   "record its CV types)")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--tolerance", type=float, default=0.3, help="a difference at an image counts against the gate only above this "
                   "many kcal/mol as well as above 4 standard errors [0.3, half of kT at 300 K]: near a basin both errors "
                   "are a few thousandths and any difference is many of them")
    p.add_argument("--min-frames", type=float, default=20.0, help="effective unbiased frames below which the reference is thin "
                   "at an image; thin at the top or next to it fails the gate [20]")
    p.add_argument("--max-dropped", type=float, default=0.05, help="largest fraction of bootstrap resamples that may be "
                   "dropped for leaving an image empty [0.05]")
    p.add_argument("--out")
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "compare":
        A, B = load_json(a.a), load_json(a.b)
        d = A["delta_F"] - B["delta_F"]
        se = float(np.hypot(A["delta_F_se"], B["delta_F_se"]))
        t = max(A.get("t95", 2.0), B.get("t95", 2.0))                 # few blocks: Student's t, not 2
        ok = abs(d) <= t * se
        print(f"closed-to-open free-energy difference: {A['delta_F']:.2f} +/- {A['delta_F_se']:.2f} "
              f"and {B['delta_F']:.2f} +/- {B['delta_F_se']:.2f} kcal/mol")
        print(f"GATE path independence: {'PASS' if ok else 'FAIL'} (difference {d:.2f}, combined error {se:.2f}, limit {t:.1f} errors)")
        for nm, P in (("a", A), ("b", B)):
            if not (P.get("stable", True) and P.get("resolved", True)):
                print(f"WARNING: profile {nm} did not pass its own stability and resolution gates; this comparison rests on it")
        db = A["barrier"] - B["barrier"]
        seb = float(np.hypot(A["barrier_se"], B["barrier_se"]))
        verdict = "unresolved" if abs(db) <= t * seb else ("a is lower" if db < 0 else "b is lower")
        print(f"barriers: {A['barrier']:.2f} +/- {A['barrier_se']:.2f} and {B['barrier']:.2f} +/- "
              f"{B['barrier_se']:.2f}: {verdict}")
        return 0 if ok else 1

    if a.cmd == "reference":
        P = load_json(a.profile); z = np.array(P["z"], float)
        if a.cv:
            from common import CVSet
            full = CVSet.load(a.cv); sub_ = full.subset(full.string); periodic = sub_.periodic
            if len(periodic) != z.shape[1]:
                sys.exit(f"{a.cv} defines {len(periodic)} string CVs; the profile has {z.shape[1]}")
            if "types" in P and [t == "DIHED" for t in P["types"]] != list(periodic):
                sys.exit(f"{a.cv} gives CV types {sub_.types}; the profile was computed with {P['types']}")
        elif "types" in P:
            periodic = np.array([t == "DIHED" for t in P["types"]])
        else:
            sys.exit("the profile does not say which of its CVs are periodic (a 'driver' profile, or one from before "
                     "2026-11-15): give --cv")
        runs = [read_cvout(f)["Th"][1] for f in a.cvout]
        start = int(P.get("start_image", 0))
        try:
            R = reference_smoothed(z, runs, a.k, periodic, kT=0.0019872041 * a.temperature, zero=start)
        except ValueError as e:
            sys.exit(str(e))
        F = np.array(P["F"], float); F = F - F[start]
        se = np.sqrt(np.array(P.get("F_se", np.zeros(len(z))), float) ** 2 + R["F_se"] ** 2)
        print(f"{R['runs']} unbiased trajectories; restraint width sqrt(kT/k) = " + ", ".join(f"{v:.3f}" for v in np.sqrt(0.0019872041 * a.temperature / R["k"])))
        print(f"both zeroed at image {start}, the profile's start")
        print(f"{'image':>5s} {'profile':>8s} {'reference':>9s} {'+/-':>6s} {'frames':>7s}")
        for i in range(len(z)):
            print(f"{i:5d} {F[i]:8.2f} {R['F'][i]:9.2f} {R['F_se'][i]:6.2f} {R['effective_frames'][i]:7.0f}")
        print(f"barrier      profile {P['barrier']:.2f} +/- {P['barrier_se']:.2f} (image {P.get('top_image', int(F.argmax()))} over "
              f"{start})   reference {R['barrier']:.2f} +/- {R['barrier_se']:.2f} (image {R['top_image']} over {R['start_image']})")
        print(f"end-to-end   profile {P['delta_F']:.2f} +/- {P['delta_F_se']:.2f}   reference {R['delta_F']:.2f} +/- {R['delta_F_se']:.2f}")
        ok = True
        # THIN IMAGES are not dropped: they are compared with their (larger)
        # error, and where they are the barrier's own the reference cannot
        # test the barrier at all
        thin = R["effective_frames"] < a.min_frames
        if thin.any():
            print(f"images {np.flatnonzero(thin).tolist()} have fewer than {a.min_frames:g} effective unbiased frames: the reference "
                  "says little there")
        tops = {int(P.get("top_image", int(F.argmax()))), int(R["top_image"])}
        near_top = sorted(j for t in tops for j in (t - 1, t, t + 1) if 0 <= j < len(z) and thin[j])
        if near_top:
            ok = False
            print(f"FAIL: images {sorted(set(near_top))} at the top have fewer than {a.min_frames:g} effective unbiased frames: "
                  "the unbiased runs do not sample the barrier well enough to test it")
        # BOOTSTRAP: resamples with an empty image were dropped
        frac = R["resamples_dropped"] / float(R["resamples_dropped"] + R["resamples_used"])
        if R["resamples_dropped"]:
            print(f"{R['resamples_dropped']} of {R['resamples_dropped'] + R['resamples_used']} bootstrap resamples ({frac:.1%}) left "
                  "some image with no unbiased frame and were dropped; the reference's errors are too small by an unknown amount")
        if frac > a.max_dropped:
            ok = False
            print(f"FAIL: more than {a.max_dropped:.0%} of the bootstrap resamples were dropped: too few runs reach some image")
        d = np.abs(F - R["F"]); use = np.ones(len(z), bool); use[start] = False
        zs = d / np.maximum(se, 1e-12)
        zmax = float(zs[use].max()) if use.any() else float("nan")
        bad = use & (zs > 4.0) & (d > a.tolerance)
        comb = float(np.hypot(P["barrier_se"], R["barrier_se"]))
        db = abs(P["barrier"] - R["barrier"]) / comb
        ok = bool(ok and db <= 3.0 and not bad.any())
        print(f"largest difference at an image {d[use].max():.2f} kcal/mol (image {int(np.flatnonzero(use)[d[use].argmax()])}); in "
              f"combined standard errors {zmax:.1f}; barriers {db:.1f} combined standard errors apart")
        t0 = int(R["top_image"])
        print(f"     a barrier difference under {3.0 * comb:.2f} kcal/mol would not show; at the top image, a difference under "
              f"{max(4.0 * se[t0], a.tolerance):.2f}")
        if bad.any():
            print(f"images {np.flatnonzero(bad).tolist()} differ by more than 4 standard errors and more than {a.tolerance} kcal/mol")
        print(f"GATE profile against unbiased dynamics: {'PASS' if ok else 'FAIL'} (barrier within 3 combined standard errors; "
              f"no image off by both 4 standard errors and {a.tolerance} kcal/mol; the top sampled; under "
              f"{a.max_dropped:.0%} of resamples dropped)")
        if a.out:
            save_json(a.out, R)
        return 0 if ok else 1
    if a.cmd == "driver":
        scale = np.array(load_json(a.scales)["scale"])
        try:
            res = from_driver(a.forces, scale, a.last, a.blocks, a.max_drift)
        except ValueError as e:
            sys.exit(str(e))
        z = res["z"]
        print(f"{res['frames_used']} of {res['frames_total']} frames used (updates {res['first_update']} to {res['last_update']}, "
              f"{res['updates_per_frame']} updates each); across them one image moved {res['systematic_in_window']:.3f} "
              f"scaled units beyond the string's own wander of {res['wander_in_window']:.3f}")
        print(f"GATE stationary string: {'PASS' if res['stationary'] else 'FAIL'} (limit {a.max_drift} scaled units)")
        res["updates_ok"] = bool(res["updates_used"] >= a.min_updates)
        print(f"GATE updates: {'PASS' if res['updates_ok'] else 'FAIL'} ({res['updates_used']} updates per image went into this "
              f"profile; limit {a.min_updates})")
        if not res["updates_ok"]:
            print("     on alanine dipeptide a driver profile from 5,500 updates passed every other gate and was well off the "
                  "unbiased reference at the barrier and end to end: with so few, blocks are shorter than the force stays "
                  "correlated and the errors are too small.")
        print(f"{'image':>5s} {'arc':>7s} {'F':>8s} {'+/-':>6s}")
        for i in range(len(z)):
            print(f"{i:5d} {res['arc'][i]:7.2f} {res['F'][i]:8.2f} {res['F_se'][i]:6.2f}")
        print(f"barrier {res['barrier']:.2f} +/- {res['barrier_se']:.2f} kcal/mol at image {res['top_image']}; "
              f"end-to-end {res['delta_F']:.2f} +/- {res['delta_F_se']:.2f}")
        print_gates(res)
        print("This profile shares its sampling with the string's own refinement. It tells you whether the dedicated "
              "restrained runs are worth starting and what to expect from them; it does not replace them.")
        np.savetxt(a.out + ".dat", np.column_stack([np.arange(len(z)), res["arc"], res["F"], res["F_se"]]),
                   fmt=["%d", "%.4f", "%.4f", "%.4f"], header="image  scaled_arc  F(kcal/mol)  standard_error")
        save_json(a.out + ".json", res)
        print(f"wrote {a.out}.dat and {a.out}.json")
        return 0 if (res["stationary"] and res["stable"] and res["resolved"] and res["blocks_ok"] and res["samples_ok"]
                      and res["updates_ok"]) else 1

    s = load_json(os.path.join(a.dir, "setup.json"))
    z = np.array(s["points"])
    periodic = np.array([t == "DIHED" for t in s["types"]])
    scale = np.array(load_json(a.scales)["scale"])
    theta = []
    for i in range(len(z)):
        try:
            th = read_cvout(os.path.join(a.dir, f"pt{i:05d}.cvout"))["Th"][1]
        except (OSError, KeyError) as e:
            sys.exit(f"image {i}: no usable restraint report ({e}). A profile needs every image.")
        if th.shape[1] != z.shape[1]:
            sys.exit(f"pt{i:05d}.cvout has {th.shape[1]} CVs, the string has {z.shape[1]}")
        theta.append(th)
    sc = load_json(a.scales)
    if "cvs" in sc and "cvs" in s and list(sc["cvs"]) != list(s["cvs"]):
        sys.exit(f"scales are for {sc['cvs']}, the runs are for {s['cvs']}")
    discard = a.discard if a.discard == "auto" else float(a.discard)
    try:
        res = compute(z, theta, s["k"], periodic, scale, a.blocks, a.min_overlap, discard, a.resolution_tol, a.floor,
                      kT=0.0019872041 * a.temperature)
    except ValueError as e:
        sys.exit(str(e))
    res["types"] = list(s["types"]); res["cvs"] = list(s.get("cvs", []))      # 'reference' needs to know which CVs are periodic
    dr = res["discarded_fraction"]
    print(f"discarded as equilibration: {'a floor plus what is detected per image' if discard == 'auto' else 'fixed'}; "
          f"median {np.median(dr):.0%}, most {dr.max():.0%} (image {int(dr.argmax())}); {res['blocks']} blocks; "
          f"95% limits are {res['t95']:.2f} standard errors")
    if discard == "auto" and dr.max() > a.floor + 0.02:
        print("     more than the floor at images: " + ", ".join(f"{i} ({v:.0%})" for i, v in enumerate(dr) if v > a.floor + 0.02))
    print(f"{'image':>5s} {'arc':>7s} {'F':>8s} {'+/-':>6s} {'overlap->next':>13s}")
    for i in range(len(z)):
        ov = f"{res['overlap'][i]:13.3f}" if i < len(z) - 1 else ""
        print(f"{i:5d} {res['arc'][i]:7.2f} {res['F'][i]:8.2f} {res['F_se'][i]:6.2f} {ov}")
    print(f"barrier {res['barrier']:.2f} +/- {res['barrier_se']:.2f} kcal/mol at image {res['top_image']}; "
          f"end-to-end {res['delta_F']:.2f} +/- {res['delta_F_se']:.2f}")
    print(f"GATE overlap: {'PASS' if res['overlap_ok'] else 'FAIL'} (smallest {res['min_overlap']:.3f}, "
          f"limit {a.min_overlap})")
    print_gates(res)
    np.savetxt(a.out + ".dat", np.column_stack([np.arange(len(z)), res["arc"], res["F"], res["F_se"]]),
               fmt=["%d", "%.4f", "%.4f", "%.4f"], header="image  scaled_arc  F(kcal/mol)  standard_error")
    n_extra = a.refine if a.refine else (0 if res["resolved"] else len(z) - 1)
    low = [int(i) for i in np.flatnonzero(np.asarray(res["overlap"]) < a.min_overlap)]
    if low and not n_extra:
        n_extra = -1                                         # only the segments that do not overlap
    if n_extra:
        new, split = refine_images(z, res["mean_force"], scale, max(n_extra, 0), must_split=low)
        np.savetxt(a.out + "_refined.dat", new, fmt="%.6f",
                   header=f"{len(z)} images plus {len(low)} in segments whose ensembles do not overlap and {max(n_extra, 0)} "
                          "where the mean force changes fastest")
        res["refined_images"] = len(new); res["split_segments"] = sorted(set(split))
        print(f"wrote {a.out}_refined.dat: {len(new)} images; extra images went into segments {sorted(set(split))}. "
              "Prepare them with init_string.py or meanforce.py and recompute.")
    save_json(a.out + ".json", res)
    print(f"wrote {a.out}.dat and {a.out}.json")
    return 0 if (res["overlap_ok"] and res["stable"] and res["resolved"] and res["blocks_ok"] and res["samples_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
