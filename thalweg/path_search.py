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
def relax_string(m, path, n_images, dt=None, max_iter=4000, tol=1e-4, support=None):
    """Steepest descent in scaled coordinates plus equal-arc redistribution.

    `support`: images are not allowed farther than this scaled distance from
    the nearest center, because the fitted surface means nothing out there
    (a sum of Gaussians goes flat away from its data)."""
    path = redistribute(unwrap_path(path, m.periodic), n_images, m.metric)
    if dt is None:
        # a step that moves the steepest center about 0.05 metric units
        g0 = np.linalg.norm(m.gradient(m.centers) * m.metric, axis=1)
        dt = 0.05 / max(np.percentile(g0, 90), 1e-12)
    for it in range(max_iter):
        g_u = m.gradient(path) * m.metric                    # dF/du
        new = path - dt * g_u * m.metric                     # du = -dt dF/du
        if support is not None:
            far = m.nearest_center_distance(new) > support
            new[far] = path[far]
        new = redistribute(new, n_images, m.metric)
        move = np.max(np.linalg.norm((new - path) / m.metric, axis=1))
        path = new
        if move < tol:
            break
    return path, it + 1


def profile_on_map(m, path):
    s = path_length(path, m.metric)
    F = m.value(path)
    return s, F


def path_distance(a, b, scale):
    """Mean scaled distance between two strings with the same image count."""
    return float(np.mean(np.linalg.norm((a - b) / scale, axis=1)))


def random_start(m, src, dst, rng, n_way=2):
    """A random initial path: src -> random centers -> dst."""
    way = rng.choice(len(m.centers), size=rng.integers(1, n_way + 1), replace=False)
    return m.centers[[src] + [int(w) for w in way] + [dst]]


# --------------------------------------------------------------------------
def nearest_center(m, point):
    return int(np.argmin(np.sqrt(np.sum((diff(m.centers, point, m.periodic) / m.metric) ** 2, axis=1))))


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
    routes, excl = find_channels(m, src, dst, n_channels, k, exclude_radius)
    if not routes:
        raise ValueError("no route between the basins: the centers do not connect them")
    if same_channel is None:
        same_channel = excl
    channels = []
    for r in routes:
        relaxed, n_it = relax_string(m, m.centers[r["nodes"]], n_images, support=support)
        # a relaxed route may fall into a channel already found
        if any(path_distance(relaxed, c["anchors"], m.metric) < same_channel for c in channels):
            continue
        s, F = profile_on_map(m, relaxed)
        channels.append({
            "nodes": r["nodes"], "anchors": relaxed, "arc": s, "F": F,
            "graph_barrier": r["top"] - float(m.value(m.centers[[src]])[0]),
            "barrier": float(F.max() - F[0]), "top_image": int(F.argmax()),
            "iterations": n_it,
            "max_distance_to_center": float(m.nearest_center_distance(relaxed).max()),
            "random_hits": 0,
        })
    rng = np.random.default_rng(seed)
    unassigned = []
    for _ in range(n_random):
        relaxed, _ = relax_string(m, random_start(m, src, dst, rng), n_images, support=support)
        d = [path_distance(relaxed, c["anchors"], m.metric) for c in channels]
        j = int(np.argmin(d))
        if d[j] < same_channel:
            channels[j]["random_hits"] += 1
        else:
            s, F = profile_on_map(m, relaxed)
            unassigned.append({"anchors": relaxed, "barrier": float(F.max() - F[0]), "distance": d[j]})
    channels.sort(key=lambda c: c["barrier"])
    return {"src": src, "dst": dst, "channels": channels, "unassigned": unassigned,
            "exclude_radius": excl, "same_channel": same_channel, "support": support}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
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
    b = load_json(a.basins)
    res = search(m, np.array(b["closed"]["centre"]), np.array(b["open"]["centre"]), a.images, a.channels,
                 a.neighbours, a.random, a.seed, a.exclude_radius)
    print(f"closed basin -> center {res['src']}, open basin -> center {res['dst']}")
    print(f"{'channel':>7s} {'barrier':>9s} {'top image':>9s} {'random hits':>11s} {'farthest from data':>18s}")
    for i, c in enumerate(res["channels"]):
        print(f"{i:7d} {c['barrier']:9.2f} {c['top_image']:9d} {c['random_hits']:11d} "
              f"{c['max_distance_to_center']:18.2f}")
        np.savetxt(f"{a.out}_anchors_{i}.dat", c["anchors"], fmt="%.6f",
                   header=f"channel {i}: {a.images} images, map barrier {c['barrier']:.3f} kcal/mol")
    print("(barriers in kcal/mol from the coarse map: for ranking only)")
    if res["unassigned"]:
        lo = min(u["barrier"] for u in res["unassigned"])
        print(f"WARNING: {len(res['unassigned'])} of {a.random} random-start strings match no channel "
              f"(lowest barrier among them {lo:.2f} kcal/mol): inspect {a.out}.json")
    else:
        print(f"all {a.random} random-start strings relaxed into a channel found by the graph search")
    save_json(a.out + ".json", res)
    print(f"wrote {a.out}.json and {len(res['channels'])} anchor table(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
