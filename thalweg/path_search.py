#!/usr/bin/env python3
"""Stage 4: global search of the free-energy map for the lowest-barrier
routes between the two basins, and the image anchors of the initial strings.

The string method relaxes a path downhill; it cannot leave the channel it
starts in.  This step looks at the whole mapped region first:

  1. GRAPH.  Every center is a node, joined to its nearest neighbours.  An
     edge's height is the highest free energy along it.  The best route is
     the one whose highest point is lowest (a minimax path), and among routes
     with that same barrier, the one that minimizes the integral of
     exp(F/kT) along the path, which keeps it in the valley floor.
  2. ALTERNATIVES.  The neighbourhood of the route's bottleneck is removed
     and the search repeated, giving distinct channels in order of barrier.
  3. RELAXATION.  Each route is relaxed on the fitted surface as a
     zero-temperature string (steepest descent plus redistribution).
  4. CROSS-CHECK.  Strings started from many random paths are relaxed the
     same way and assigned to the channels found; any that fits none is
     reported as a channel the graph search missed.

Everything here costs seconds: no MD is run.  The result is only as wide as
the region the centers cover, and the relaxation uses the scaled Euclidean
metric, not the metric tensor; the MD string refinement of Stage 5 corrects
both the path and its barrier.

Example:
  path_search.py --map map.json --basins endpoints_basins.json \\
      --images 20 --channels 3 --out channels
"""
import argparse
import heapq
import os
import sys

import numpy as np

from common import diff, load_json, path_length, redistribute, save_json, unwrap_path
from fesmap import KT_300, FESMap


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------
def build_graph(m, k=8, samples=7):
    """k-nearest-neighbour graph over the centers.

    Returns {(i, j): (height, length, argmax point)} for i < j, where height
    is the largest F at `samples` points along the straight segment."""
    z = m.centers
    n = len(z)
    du = diff(z[:, None, :], z[None, :, :], m.periodic) / m.metric
    d = np.sqrt(np.sum(du ** 2, axis=2))
    edges = {}
    t = np.linspace(0.0, 1.0, samples)
    for i in range(n):
        for j in np.argsort(d[i])[1:k + 1]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if (a, b) in edges:
                continue
            step = diff(z[b], z[a], m.periodic)
            pts = z[a] + t[:, None] * step
            F = m.value(pts)
            edges[(a, b)] = (float(F.max()), float(d[a, b]), pts[int(F.argmax())])
    return edges


def adjacency(edges, n, removed=()):
    removed = set(removed)
    adj = [[] for _ in range(n)]
    for (a, b), (h, length, _) in edges.items():
        if a in removed or b in removed:
            continue
        adj[a].append((b, h, length))
        adj[b].append((a, h, length))
    return adj


def minimax_barrier(adj, src, dst):
    """Smallest possible value of the highest edge on any src-dst path."""
    best = [np.inf] * len(adj)
    best[src] = -np.inf
    heap = [(-np.inf, src)]
    while heap:
        b, i = heapq.heappop(heap)
        if b > best[i]:
            continue
        if i == dst:
            return b
        for j, h, _ in adj[i]:
            nb = max(b, h)
            if nb < best[j]:
                best[j] = nb
                heapq.heappush(heap, (nb, j))
    return np.inf


def valley_path(adj, src, dst, cap, kT):
    """Among paths whose edges are all <= cap, minimize sum len * exp(h/kT)."""
    dist = [np.inf] * len(adj)
    prev = [-1] * len(adj)
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        c, i = heapq.heappop(heap)
        if c > dist[i]:
            continue
        if i == dst:
            break
        for j, h, length in adj[i]:
            if h > cap:
                continue
            nc = c + length * np.exp(min(h / kT, 700.0))
            if nc < dist[j]:
                dist[j], prev[j] = nc, i
                heapq.heappush(heap, (nc, j))
    if not np.isfinite(dist[dst]):
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def find_channels(m, src, dst, n_channels=3, k=8, exclude_radius=None, tol=0.25, kT=KT_300):
    """Distinct routes in order of barrier.  After each, the centers within
    `exclude_radius` (scaled) of its bottleneck are removed."""
    edges = build_graph(m, k)
    n = len(m.centers)
    if exclude_radius is None:
        # Two typical edges.  This is a compromise: on a sparse 4-D map it
        # removes a sixth of the centers around each bottleneck, hiding any
        # alternative saddle a few units away (the random-start strings are
        # what catches those); one edge is too little on a dense map, where
        # the "alternative" just steps around the hole in the same channel.
        exclude_radius = 2.0 * np.median([e[1] for e in edges.values()])
    removed, out = set(), []
    for _ in range(n_channels):
        adj = adjacency(edges, n, removed)
        cap = minimax_barrier(adj, src, dst)
        if not np.isfinite(cap):
            break
        nodes = valley_path(adj, src, dst, cap + tol, kT)
        # the bottleneck: highest edge on the route
        hs = [edges[(min(a, b), max(a, b))] for a, b in zip(nodes[:-1], nodes[1:])]
        top = max(hs, key=lambda e: e[0])
        out.append({"nodes": nodes, "barrier_point": top[2], "top": top[0]})
        near = np.sqrt(np.sum((diff(m.centers, top[2], m.periodic) / m.metric) ** 2, axis=1)) < exclude_radius
        new = {int(i) for i in np.where(near)[0]} - {src, dst}
        if not new:
            break
        removed |= new
    return out, exclude_radius


# --------------------------------------------------------------------------
# zero-temperature string on the fitted surface
# --------------------------------------------------------------------------
def turn_cosines(path, scale):
    """Cosine of the turn at each interior image (1 = straight on, -1 = straight back)."""
    d = np.diff(np.asarray(path, float) / scale, axis=0)
    n = np.maximum(np.linalg.norm(d, axis=1), 1e-300)
    return np.sum(d[:-1] * d[1:], axis=1) / (n[:-1] * n[1:])


def cut_spurs(path, scale, cos_min=0.0):
    """Drop interior images where the path turns back on itself.

    A stretch of string that falls into a side minimum stays there as a spur:
    out and back along the same line.  Equal-arc redistribution keeps it, the
    spur uses up images and shifts every other image along the path.  The
    sharpest such turn is removed, and again, until none is sharper than
    cos_min; the caller redistributes afterwards."""
    path = np.asarray(path, float)
    while len(path) > 3:
        c = turn_cosines(path, scale)
        i = int(np.argmin(c))
        if c[i] >= cos_min:
            break
        path = np.delete(path, i + 1, axis=0)
    return path


def upwind_tangents(u, F):
    """Unit tangents of a string from the neighbour on the UPHILL side of
    each image (both neighbours, weighted, at a maximum or minimum of F along
    the string): the tangent of Henkelman and Jonsson, J Chem Phys 113, 9978
    (2000).  A flow that pushes images along the string as well as across it
    carries a perturbation along the string, and central differences then
    amplify it: with them two of three alanine dipeptide routes cycled for
    ever at 6e-3 scaled units per step; with these all three converge."""
    u = np.asarray(u, float); F = np.asarray(F, float)
    t = np.gradient(u, axis=0)
    if len(u) > 2:
        a = u[2:] - u[1:-1]; b = u[1:-1] - u[:-2]
        up = F[2:] - F[1:-1]; dn = F[:-2] - F[1:-1]
        hi = np.maximum(np.abs(up), np.abs(dn))[:, None]; lo = np.minimum(np.abs(up), np.abs(dn))[:, None]
        rising = (F[2:] > F[1:-1]) & (F[1:-1] > F[:-2]); falling = (F[2:] < F[1:-1]) & (F[1:-1] < F[:-2])
        mixed = np.where((F[2:] > F[:-2])[:, None], a * hi + b * lo, a * lo + b * hi)
        t[1:-1] = np.where(rising[:, None], a, np.where(falling[:, None], b, mixed))
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)


def across_metric(m, path, g_u):
    """The part of the scaled gradient that a minimum free-energy path does
    not balance: g minus its component along n = M^-1 t / |M^-1 t|, with t the
    path's tangent and M the (scaled) metric tensor there.  It vanishes where
    M grad F is parallel to the tangent, which is what defines the path.

    The string is moved along MINUS THIS, not along -M grad F.  Both flows
    stop at the same paths.  Moving along -M grad F is the textbook form, but
    here M's eigenvalues differ 250-fold (torsions that barely move the atoms
    against ones that swing a whole group), so a step small enough for the
    fast directions leaves the slow ones, the ones the path runs along,
    almost still.  On the alanine dipeptide map such relaxations met their
    stopping test 0.03 to 0.09 scaled units short of where they ended after
    200,000 iterations (six minutes a string), with 0.04 to 0.31 of this
    residual left; this flow reaches the same paths, to 0.01, in under 2,000
    iterations.  It is the same descent with
    M^-1 as a preconditioner (it lowers F at a rate >= 0 by Cauchy-Schwarz in
    the M^-1 inner product), has the conditioning of plain descent, and needs
    no special step or iteration cap.  It does need upwind tangents (see
    `upwind_tangents`), because unlike -M grad F it uses the tangent."""
    t = upwind_tangents(path / m.metric, m.value(path))
    n = np.linalg.solve(m.mobility(path), t[:, :, None])[:, :, 0]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    return g_u - np.sum(g_u * n, axis=1, keepdims=True) * n


def _relax(m, path, n_images, dt, stop, max_iter, support, fix_ends, window=100):
    """Returns (path, iterations, converged).  Converged means either that no
    image moved more than `stop` in an iteration, or that the string has
    settled into a small chatter: its mean over one window of iterations
    lies within 5 x stop of its mean over the window before, while single
    iterations still move it by up to 50 x stop.  In the second case the
    window mean is returned.  (Upwind tangents switch sides where the profile
    is flat, and a string there can jitter by a few 1e-4 scaled units for
    ever without going anywhere; a string that is still travelling moves its
    window mean by a hundred times its per-iteration move and is not let
    through.)  _relax.chatter holds the per-iteration move in that case."""
    converged = False
    _relax.chatter = 0.0
    acc = np.zeros_like(path); nacc = 0; prev_mean = None; big = 0.0
    for it in range(max_iter):
        g_u = m.gradient(path) * m.metric                    # dF/du
        if m.M is not None:
            g_u = across_metric(m, path, g_u)
        new = path - dt * g_u * m.metric
        if fix_ends:
            new[0], new[-1] = path[0], path[-1]
        if support is not None:
            d_new = m.nearest_center_distance(new)
            far = (d_new > support) & (d_new >= m.nearest_center_distance(path))
            new[far] = path[far]
        new = redistribute(new, n_images, m.metric)
        move = np.max(np.linalg.norm((new - path) / m.metric, axis=1))
        path = new
        if move < stop:
            converged = True
            break
        acc += path; nacc += 1; big = max(big, move)
        if nacc == window:
            mean = acc / nacc
            if prev_mean is not None and big < 50.0 * stop and \
                    np.max(np.linalg.norm((mean - prev_mean) / m.metric, axis=1)) < 5.0 * stop:
                path = redistribute(mean, n_images, m.metric)
                if fix_ends:
                    path[0], path[-1] = new[0], new[-1]
                converged = True; _relax.chatter = float(big)
                break
            prev_mean = mean; acc = np.zeros_like(path); nacc = 0; big = 0.0
    return path, it + 1, converged


def relax_string(m, path, n_images, dt=None, max_iter=None, tol=1e-4, support=None, fix_ends=True, cut=-0.5):
    """Descent in scaled coordinates plus equal-arc redistribution.  With a
    metric tensor on the map the images move against the part of grad F that
    a minimum free-energy path does not balance (see `across_metric`), which
    stops where M grad F is parallel to the path; without one, along -grad F.

    The two end images stay where they start (fix_ends): free ends slide to
    whichever sub-minimum of the basin is nearest, so two relaxations of the
    same channel can end in different places and differ in length.  Start
    them at the basin's minimum on the map (see `descend`), not at a center
    up its wall, or every string begins with a stub down to the floor.

    SPURS.  A string that turns back on itself (turn cosine below `cut`) is
    dealt with AFTER it has converged, once: the doubled-back images are
    removed and the string relaxed again.  If it converges without the spur,
    that is the result.  If the spur comes back, the string really does pass
    through a side minimum, and the converged string WITH it is returned and
    flagged (`through_minimum`): it is two legs, not one.  Cutting inside the
    loop, as this did at first, makes a limit cycle of exactly that case:
    cut, fall back in, cut again, never converging.

    `support`: an image may not move farther from the data than this scaled
    distance unless the move brings it closer, because the fitted surface
    means nothing out there (a sum of Gaussians goes flat away from its data).

    Returns (path, iterations, converged); relax_string.info holds
    {"through_minimum": bool, "cut": bool} for the last call."""
    path = redistribute(unwrap_path(path, m.periodic), n_images, m.metric)
    # a step that moves the steepest center about 0.05 metric units
    gc = m.gradient(m.centers) * m.metric
    dt_plain = 0.05 / max(np.percentile(np.linalg.norm(gc, axis=1), 90), 1e-12)
    if dt is None:
        dt = dt_plain
    if max_iter is None:
        max_iter = 4000
    # the stopping test is on move / dt, a residual force, so it means the
    # same thing whatever the step
    stop = tol * dt / dt_plain
    n_pre = 0
    if m.M is not None:
        # first to the tensor-free path, then on from there with the tensor: a
        # string still far from any path (a graph route, a random start) can
        # double back on itself under the tensor flow and never settle, and
        # started from the tensor-free path it does not
        M, m.M = m.M, None
        try:
            path, n_pre, _ = _relax(m, path, n_images, dt, stop, max_iter, support, fix_ends)
        finally:
            m.M = M
    path, n_it, conv = _relax(m, path, n_images, dt, stop, max_iter, support, fix_ends)
    n_it += n_pre
    info = {"through_minimum": False, "cut": False}
    if cut is not None and conv and turn_cosines(path, m.metric).min() < cut:
        trial = redistribute(cut_spurs(path, m.metric, cut), n_images, m.metric)
        trial, n2, conv2 = _relax(m, trial, n_images, dt, stop, max_iter, support, fix_ends)
        n_it += n2
        if conv2 and turn_cosines(trial, m.metric).min() >= cut:
            path = trial; info["cut"] = True
        else:
            info["through_minimum"] = True
    relax_string.info = info
    return path, n_it, conv


def descend(m, z0, support=None, max_iter=5000, tol=1e-5):
    """The minimum of the map reached by steepest descent from z0 (scaled
    coordinates).  The center nearest a basin's centre can sit a kcal/mol up
    the basin's wall; this is where a string should end."""
    z = np.array(z0, float)[None, :]
    gc = m.gradient(m.centers) * m.metric
    dt = 0.05 / max(np.percentile(np.linalg.norm(gc, axis=1), 90), 1e-12)
    for _ in range(max_iter):
        step = dt * (m.gradient(z) * m.metric) * m.metric
        new = z - step
        if support is not None and m.nearest_center_distance(new)[0] > support:
            break
        z = new
        if np.linalg.norm(step / m.metric) < tol:
            break
    return z[0]


def profile_on_map(m, path):
    s = path_length(path, m.metric)
    F = m.value(path)
    return s, F


def _to_curve(a, b, scale):
    """Scaled distance from each point of a to the polyline through b."""
    a = np.asarray(a, float) / scale; b = np.asarray(b, float) / scale
    p0, d = b[:-1], np.diff(b, axis=0)
    L2 = np.maximum(np.sum(d * d, axis=1), 1e-300)
    t = np.clip(np.einsum("isk,sk->is", a[:, None, :] - p0[None, :, :], d) / L2, 0.0, 1.0)
    near = p0[None, :, :] + t[:, :, None] * d[None, :, :]
    return np.sqrt(np.sum((a[:, None, :] - near) ** 2, axis=2)).min(axis=1)


def align_periodic(a, b, periodic):
    """b shifted by whole turns in its periodic CVs so that it starts on the
    same 2 pi image as a."""
    a = np.asarray(a, float); b = np.asarray(b, float).copy()
    if periodic is not None:
        per = np.asarray(periodic, bool)
        b[:, per] += 2 * np.pi * np.round((a[0, per] - b[0, per]) / (2 * np.pi))
    return b


def path_distance(a, b, scale, periodic=None, largest=False):
    """How far apart two strings lie as CURVES, in scaled units: the rms
    distance from the images of each to the line through the other, the
    larger of the two directions.  Comparing image i with image i instead
    counts a shift along the path (one string a little longer at one end) as
    if the strings were apart.  With `periodic`, b is first moved onto a's
    2 pi image.  With `largest`, returns (rms, largest single distance): two
    strings that part company over a short stretch have a small rms."""
    b = align_periodic(a, b, periodic)
    d1, d2 = _to_curve(a, b, scale), _to_curve(b, a, scale)
    rms = float(max(np.sqrt(np.mean(d1 ** 2)), np.sqrt(np.mean(d2 ** 2))))
    return (rms, float(max(d1.max(), d2.max()))) if largest else rms


def same_route(a, b, scale, tol):
    """One route: close as curves on average, and nowhere farther apart than twice that."""
    rms, big = path_distance(a, b, scale, largest=True)
    return rms < tol and big < 2.0 * tol


def fine_top(m, path, n=10):
    """Highest F along the string, read between the images as well as at them."""
    fine = redistribute(np.asarray(path, float), (len(path) - 1) * n + 1, m.metric)
    return float(m.value(fine).max())


def climb_to_top(F):
    """From the lowest point before the top, to the top.  Used on its own
    this measures every channel from a different reference; the channels are
    ranked by the top above the COMMON start, the closed basin's minimum on
    the map, where every string now begins."""
    F = np.asarray(F, float); top = int(F.argmax())
    return float(F[top] - F[: top + 1].min())


def largest_climb(F):
    """The largest rise in F going forward along the string: max over i < j of
    F[j] - F[i].  It exceeds the barrier when there is a second climb, after
    the top, out of a minimum lower than the one the string started from."""
    F = np.asarray(F, float)
    return float(np.max(F - np.minimum.accumulate(F)))


def random_start(m, src, dst, rng, n_way=2):
    """A random initial path: src -> random centers -> dst."""
    way = rng.choice(len(m.centers), size=rng.integers(1, n_way + 1), replace=False)
    return m.centers[[src] + [int(w) for w in way] + [dst]]


# --------------------------------------------------------------------------
def nearest_center(m, point):
    return int(np.argmin(np.sqrt(np.sum((diff(m.centers, point, m.periodic) / m.metric) ** 2, axis=1))))


def _channel(m, relaxed, n_it, conv, info, F_ref, found_by, extra=None):
    s, F = profile_on_map(m, relaxed)
    top = fine_top(m, relaxed)
    c = {"anchors": relaxed, "arc": s, "F": F - F_ref, "barrier": float(top - F_ref), "top_image": int(F.argmax()),
         "climb_to_top": climb_to_top(F), "largest_climb": largest_climb(F),
         "iterations": n_it, "converged": bool(conv), "through_minimum": bool(info["through_minimum"]),
         "sharpest_turn": float(turn_cosines(relaxed, m.metric).min()),
         "max_distance_to_center": float(m.nearest_center_distance(relaxed).max()),
         "random_hits": 0, "hit_barriers": [], "found_by": found_by}
    c.update(extra or {})
    return c


def search(m, z_closed, z_open, n_images=20, n_channels=3, k=8, n_random=50, seed=0,
           exclude_radius=None, same_channel=None, support=None):
    src, dst = nearest_center(m, z_closed), nearest_center(m, z_open)
    if src == dst:
        raise ValueError("both basins map onto the same center: the map does not separate them")
    if support is None:
        # twice the typical spacing of the centers, in metric units
        dm = diff(m.centers[:, None, :], m.centers[None, :, :], m.periodic) / m.metric
        dd = np.sqrt(np.sum(dm ** 2, axis=2)); np.fill_diagonal(dd, np.inf)
        support = 2.0 * float(np.median(dd.min(axis=1)))
    # every string runs between the two basins' minima ON THE MAP, reached by
    # descent from the centers nearest the basin centres; barriers are
    # measured from the closed one
    z_src, z_dst = descend(m, m.centers[src], support), descend(m, m.centers[dst], support)
    F_ref = float(m.value(z_src[None, :])[0])
    # graph routes: more than asked for, since some relax into a channel already found
    routes, excl = find_channels(m, src, dst, 3 * n_channels, k, exclude_radius)
    if not routes:
        raise ValueError("no route between the basins: the centers do not connect them")
    if same_channel is None:
        # two strings are the same route when they lie within half a center
        # spacing of each other as curves.  (The exclusion radius, used
        # before, is several units: lower variants of a channel were counted
        # as that channel and never reported.)
        same_channel = 0.25 * support

    def ends_on(nodes_path, target):
        # the route as unwrapped points, with the basin minima on its ends, on the route's own 2 pi image
        p = unwrap_path(nodes_path, m.periodic)
        a = p[0] + diff(z_src, p[0], m.periodic); b = p[-1] + diff(target, p[-1], m.periodic)
        return np.vstack([a, p, b])

    channels = []
    for r in routes:
        if len(channels) >= n_channels:
            break
        relaxed, n_it, conv = relax_string(m, ends_on(m.centers[r["nodes"]], z_dst), n_images, support=support)
        # a relaxed route may fall into a channel already found
        if any(same_route(relaxed, c["anchors"], m.metric, same_channel) for c in channels):
            continue
        channels.append(_channel(m, relaxed, n_it, conv, relax_string.info, F_ref, "graph search",
                                 {"nodes": r["nodes"], "graph_barrier": r["top"] - F_ref}))
    rng = np.random.default_rng(seed)
    loose = []; n_unconverged = 0
    for _ in range(n_random):
        relaxed, n_it, conv = relax_string(m, ends_on(random_start(m, src, dst, rng), z_dst), n_images, support=support)
        info = dict(relax_string.info)
        if not conv:
            n_unconverged += 1                               # not a relaxed string: it joins nothing and proves nothing
            continue
        hit = [c for c in channels if same_route(relaxed, c["anchors"], m.metric, same_channel)]
        if hit:
            c = min(hit, key=lambda c: path_distance(relaxed, c["anchors"], m.metric))
            c["random_hits"] += 1; c["hit_barriers"].append(float(fine_top(m, relaxed) - F_ref))
        else:
            loose.append((relaxed, n_it, info))
    # Random starts that agree with each other but with no graph route ARE a
    # route: the graph search removes a ball round each bottleneck and cannot
    # return a second saddle inside it.  They join the table like any other.
    # ... when at least two of them agree.  One string on its own is a stray:
    # it is counted and its barrier listed, and that is all.
    cands = []
    for relaxed, n_it, info in sorted(loose, key=lambda t: fine_top(m, t[0])):
        hit = [c for c in channels + cands if same_route(relaxed, c["anchors"], m.metric, same_channel)]
        if hit:
            hit[0]["random_hits"] += 1; hit[0]["hit_barriers"].append(float(fine_top(m, relaxed) - F_ref))
        elif m.nearest_center_distance(relaxed).max() <= support:
            c = _channel(m, relaxed, n_it, True, info, F_ref, "random starts only"); c["random_hits"] = 1
            c["hit_barriers"].append(c["barrier"]); cands.append(c)
    strays = sorted(c["barrier"] for c in cands if c["random_hits"] < 2)
    channels += [c for c in cands if c["random_hits"] >= 2]
    channels.sort(key=lambda c: c["barrier"])
    return {"src": src, "dst": dst, "channels": channels, "unconverged_random": n_unconverged, "stray_barriers": strays,
            "end_minima": [z_src, z_dst], "end_centers_F": [float(m.value(m.centers[[src]])[0]) - F_ref,
                                                            float(m.value(m.centers[[dst]])[0]) - F_ref],
            "exclude_radius": excl, "same_channel": same_channel, "support": support}


def barrier_uncertainty(m, z, f, channels, n_boot=100, frac=0.8, seed=0, f_se=None, relax=False):
    """How much of each channel's barrier is the map and how much the centers.

    The map is refitted n_boot times on random subsets of the centers, with
    the Gaussian widths held fixed, and each channel's barrier is read again
    along its (fixed) path.  Returns per-channel standard errors and, for
    each channel after the first, the share of refits in which the first
    channel is still the lower one.

    A subset of a fraction `frac` drawn without replacement varies LESS than
    the full sample's own uncertainty: the spread has to be multiplied by
    sqrt(frac / (1 - frac)), which is 2 for the default 0.8 (the delete-d
    jackknife).  Both numbers returned include that factor.

    With relax=False the paths are held fixed and each barrier is read along
    the same images on every refit.  With relax=True each channel's string is
    relaxed again on every refit (ends taken down to that map's minima), so
    the error includes how far the route itself moves when the map changes;
    barrier_uncertainty.moved then holds, per channel, the rms distance the
    relaxed strings lie from the original and the share of refits on which
    the string did not converge (those refits keep the fixed path).  Either
    way the routes are not SEARCHED for again, so a refit that would prefer
    a route not in the list does not show.

    What this error is: mostly how much the barrier depends on WHICH centers
    the map has, not the noise in their forces.  With f_se (the standard
    errors of the forces) a third return value gives the part that is force
    noise alone, from refits with the forces perturbed by their errors."""
    rng = np.random.default_rng(seed)
    B = np.empty((n_boot, len(channels)))
    moved = np.zeros((n_boot, len(channels))); failed = np.zeros(len(channels))
    for b in range(n_boot):
        idx = rng.choice(len(z), size=int(frac * len(z)), replace=False)
        mb = FESMap.fit(z[idx], f[idx], m.sigma, m.scale, m.periodic, metric=m.metric)
        for c, ch in enumerate(channels):
            path = np.asarray(ch["anchors"], float)
            if relax:
                start = path.copy(); start[0], start[-1] = descend(mb, path[0]), descend(mb, path[-1])
                new, _, conv = relax_string(mb, start, len(path))
                if conv:
                    moved[b, c] = path_distance(new, path, m.metric); path = new
                else:
                    failed[c] += 1
            F = mb.value(path)
            B[b, c] = F.max() - F[0]                          # every string starts at the same point
    barrier_uncertainty.moved = (np.sqrt((moved ** 2).mean(axis=0)), failed / n_boot) if relax else None
    k = np.sqrt(frac / (1.0 - frac))
    sd = k * B.std(axis=0, ddof=1)
    holds = []
    for c in range(1, len(channels)):
        d = B[:, c] - B[:, 0]
        holds.append(float(np.mean(d.mean() + k * (d - d.mean()) > 0)))
    if f_se is None:
        return sd, holds
    N = np.empty((max(20, n_boot // 3), len(channels)))
    for b in range(len(N)):
        mb = FESMap.fit(z, f + rng.normal(0, 1, f.shape) * f_se, m.sigma, m.scale, m.periodic, metric=m.metric)
        for c, ch in enumerate(channels):
            F = mb.value(np.asarray(ch["anchors"])); N[b, c] = F.max() - F[0]
    return sd, holds, N.std(axis=0, ddof=1)


def group_unassigned(unassigned, scale, same_channel):     # kept for callers; search now promotes these to channels
    """Group the random-start strings that matched no channel.  Each group is
    a route the graph search did not return: usually a higher one, sometimes
    a string caught on the edge of the mapped region."""
    groups = []
    for u in sorted(unassigned, key=lambda u: u["barrier"]):
        for g in groups:
            if path_distance(np.asarray(u["anchors"]), np.asarray(g["anchors"]), scale) < same_channel:
                g["count"] += 1
                break
        else:
            groups.append({"anchors": u["anchors"], "barrier": u["barrier"], "count": 1})
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--forces", help="the centers-and-forces file the map was fitted to; enables the "
                    "uncertainty of the barriers and of the channel ranking")
    ap.add_argument("--bootstrap", type=int, default=100, help="map refits for that uncertainty [100]")
    ap.add_argument("--route-points", help="also write the images of every route that lie farther than --route-gap from "
                    "any center, one point per line: places to measure the mean force next, so that the map "
                    "is refitted with data on the routes themselves ('meanforce.py setup --points ... --centers ...')")
    ap.add_argument("--route-gap", type=float, default=0.3, help="scaled distance from the nearest center beyond which a "
                    "route image is worth a new center [0.3]")
    ap.add_argument("--fixed-paths", action="store_true",
                    help="read each refit's barriers along the fixed strings instead of relaxing every string again "
                         "on every refit (faster; the error then leaves out how far a route moves with the map)")
    ap.add_argument("--no-metric-tensor", action="store_true",
                    help="relax along -grad F even if the map carries a measured metric tensor")
    ap.add_argument("--basins", required=True, help="basins.json from endpoint_analysis.py")
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--neighbours", type=int, default=8)
    ap.add_argument("--random", type=int, default=50, help="random-start strings for the cross-check")
    ap.add_argument("--exclude-radius", type=float, help="scaled radius removed around each bottleneck")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="channels")
    a = ap.parse_args(argv)

    m = FESMap.load(a.map)
    if a.no_metric_tensor:
        m.M = None
    print("strings relaxed to where M grad F lies along them, with the metric tensor measured at the centers" if m.M is not None else
          "strings relaxed along -grad F (no metric tensor on the map: see 'meanforce.py metric')")
    b = load_json(a.basins)
    res = search(m, np.array(b["closed"]["centre"]), np.array(b["open"]["centre"]), a.images, a.channels,
                 a.neighbours, a.random, a.seed, a.exclude_radius)
    e = res["end_centers_F"]
    print(f"closed basin -> center {res['src']}, open basin -> center {res['dst']}; the strings run between the map's minima "
          f"below them ({e[0]:.2f} and {e[1]:.2f} kcal/mol lower than those centers); barriers are from the closed minimum")
    sd = holds = noise = None
    if a.forces:
        from common import read_forces
        zf, ff, _ = read_forces(a.forces)
        fse = np.loadtxt(a.forces + ".se") if os.path.exists(a.forces + ".se") else None
        if fse is not None and fse.shape != ff.shape:
            fse = None
        mplain = FESMap.load(a.map); mplain.M = None     # refits carry no tensor: relax them without one
        out = barrier_uncertainty(mplain, zf, ff, res["channels"], a.bootstrap, seed=a.seed, f_se=fse, relax=not a.fixed_paths)
        sd, holds = out[0], out[1]; noise = out[2] if len(out) > 2 else None
        for i, c in enumerate(res["channels"]):
            c["barrier_sd"] = float(sd[i])
            if noise is not None:
                c["barrier_sd_force_noise"] = float(noise[i])
        res["ranking_holds"] = holds
    print(f"{'channel':>7s} {'barrier':>9s} {'+/-':>6s} {'top image':>9s} {'random hits':>11s} {'farthest from data':>18s}  found by")
    for i, c in enumerate(res["channels"]):
        print(f"{i:7d} {c['barrier']:9.2f} {(f'{sd[i]:6.2f}' if sd is not None else '     -')} {c['top_image']:9d} "
              f"{c['random_hits']:11d} {c['max_distance_to_center']:18.2f}  {c['found_by']}")
        np.savetxt(f"{a.out}_anchors_{i}.dat", c["anchors"], fmt="%.6f",
                   header=f"channel {i} ({c['found_by']}): {a.images} images, map barrier {c['barrier']:.3f} kcal/mol")
    print("(barriers in kcal/mol from the coarse map: for ranking only)")
    if a.route_points:
        pts = []
        for c in res["channels"]:
            anc = np.asarray(c["anchors"], float)
            far = m.nearest_center_distance(anc) > a.route_gap
            for p in anc[far]:
                p = np.array([((v + np.pi) % (2 * np.pi)) - np.pi if per else v for v, per in zip(p, m.periodic)])
                if not pts or min(np.linalg.norm(diff(np.array(pts), p, m.periodic) / m.metric, axis=1)) > a.route_gap / 2:
                    pts.append(p)
        np.savetxt(a.route_points, np.array(pts).reshape(-1, len(m.metric)), fmt="%.6f",
                   header=f"route images farther than {a.route_gap} scaled units from every center")
        print(f"wrote {len(pts)} route images with no center within {a.route_gap} scaled units to {a.route_points}")
    if sd is not None:
        mv = getattr(barrier_uncertainty, "moved", None)
        if mv is not None:
            print("on each refit every string was relaxed again: they moved (rms, scaled units) "
                  + ", ".join(f"{v:.2f}" for v in mv[0])
                  + ("; did not converge on " + ", ".join(f"{100 * v:.0f}%" for v in mv[1]) + " of refits" if mv[1].max() > 0 else ""))
            res["refit_moved"] = [float(v) for v in mv[0]]; res["refit_unconverged"] = [float(v) for v in mv[1]]
        print("the +/- is how much a barrier depends on WHICH centers the map has (refits on subsets, "
              + ("strings relaxed again on each)" if mv is not None else "paths fixed)")
              + (f"; the part that is noise in the forces is {', '.join(f'{v:.2f}' for v in noise)}" if noise is not None else ""))
    for i, c in enumerate(res["channels"]):
        if not c["converged"]:
            print(f"WARNING: channel {i} did not converge in {c['iterations']} iterations: its string is not a relaxed one")
        if c["through_minimum"]:
            print(f"note: channel {i} passes through a side minimum (it turns back on itself, cosine {c['sharpest_turn']:.2f}, and "
                  "does so again when the turn is cut out): it is two legs, and wants an image count that resolves the corner")
        elif c["sharpest_turn"] < 0.3:
            print(f"WARNING: channel {i} turns sharply (turn cosine {c['sharpest_turn']:.2f}): look at it before starting MD from it")
        if c["largest_climb"] > c["climb_to_top"] + 0.25:
            print(f"note: channel {i} has a second climb after its top; the largest climb along it is {c['largest_climb']:.2f} kcal/mol")
        hb = c["hit_barriers"]
        if len(hb) > 1:
            tol = max(2 * c.get("barrier_sd", 0.0), 0.3)
            low = [b for b in hb if b < c["barrier"] - tol]
            print(f"   channel {i}: the {len(hb)} random-start strings on it have barriers {min(hb):.2f} to {max(hb):.2f}"
                  + (f"; {len(low)} are LOWER than the channel by more than {tol:.2f}: a lower variant of this route exists" if low else ""))
    if res["unconverged_random"]:
        print(f"WARNING: {res['unconverged_random']} of {a.random} random-start strings did not converge and were left out")
    if holds:
        for i, h in enumerate(holds, start=1):
            be = np.sqrt(max(h * (1 - h), 1e-4) / a.bootstrap)
            verdict = ("the order is firm" if h - 2 * be >= 0.9 else
                       ("the order is likely" if h - 2 * be >= 0.7 else "the order is NOT established by this map"))
            print(f"channel 0 below channel {i} in {h:.0%} (+/- {be:.0%}) of {a.bootstrap} map refits: {verdict}")
    if res["stray_barriers"]:
        sb = res["stray_barriers"]; best = res["channels"][0]["barrier"]
        print(f"{len(sb)} random-start strings matched no channel and no other string (barriers {sb[0]:.2f} to {sb[-1]:.2f}): strays, not listed"
              + (f". {sum(1 for b in sb if b < best)} of them are LOWER than channel 0: raise --random and look again" if sb[0] < best else ""))
    nr = sum(1 for c in res["channels"] if c["found_by"] != "graph search")
    if nr:
        print(f"{nr} of these channels were found only by random-start strings: routes the graph search could not return")
    save_json(a.out + ".json", res)
    print(f"wrote {a.out}.json and {len(res['channels'])} anchor table(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
