"""Shared pieces for the string-method tools: CV definitions, periodic
arithmetic in CV space, and readers for the files cfacv writes.

Units follow cfacv: distances in angstroms, angles and torsions in radians.

A CV definition file is JSON:

    {"cvs": [
        {"name": "d1",   "type": "BOND",  "atoms": ["resid 22 and name CA",
                                                     "resid 141 and name CA"]},
        {"name": "chi1", "type": "DIHED", "atoms": ["resid 4 and name N", ...]}
     ],
     "string": ["d1", "d2", "d6", "chi1"]}

Each entry of "atoms" is an MDAnalysis selection; a selection of several atoms
means their centre of mass, which is what a cfacv "center" is.  "string" names
the CVs that span the string, in cv.inp order; the others are measured only.
"""
import json
import re

import numpy as np

TWO_PI = 2.0 * np.pi
PERIODIC_TYPES = {"DIHED"}


# --------------------------------------------------------------------------
# CV definitions
# --------------------------------------------------------------------------
class CVSet:
    def __init__(self, cvs, string=None):
        self.cvs = cvs
        self.names = [c["name"] for c in cvs]
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate CV names")
        for c in cvs:
            n = {"BOND": 2, "ANGLE": 3, "DIHED": 4}.get(c["type"])
            if n is None:
                raise ValueError(f"CV {c['name']}: unsupported type {c['type']}")
            if len(c["atoms"]) != n:
                raise ValueError(f"CV {c['name']}: {c['type']} needs {n} atom selections")
        self.string = list(string) if string else list(self.names)
        for s in self.string:
            if s not in self.names:
                raise ValueError(f"string CV {s} is not defined")

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["cvs"], d.get("string"))

    def subset(self, names):
        return CVSet([c for n in names for c in self.cvs if c["name"] == n], names)

    @property
    def periodic(self):
        """Boolean array: is each CV periodic (domain 2 pi)?"""
        return np.array([c["type"] in PERIODIC_TYPES for c in self.cvs])

    @property
    def types(self):
        return [c["type"] for c in self.cvs]


def _dihedral(p0, p1, p2, p3):
    """IUPAC dihedral in radians, (-pi, pi].  Arrays of shape (..., 3)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.arctan2(y, x)


def cv_values_from_points(cvtype, pts):
    """pts: list of (..., 3) arrays, one per atom group of the CV."""
    if cvtype == "BOND":
        return np.linalg.norm(pts[1] - pts[0], axis=-1)
    if cvtype == "ANGLE":
        a = pts[0] - pts[1]
        b = pts[2] - pts[1]
        c = np.sum(a * b, axis=-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
        return np.arccos(np.clip(c, -1.0, 1.0))
    if cvtype == "DIHED":
        return _dihedral(*pts)
    raise ValueError(cvtype)


class CVCalculator:
    """Evaluates a CVSet on an MDAnalysis Universe.

    Each group is made whole (its atoms brought to the image nearest its
    first atom) before its centre of mass is taken, and the groups of one CV
    are then brought to the image nearest the first group, so a CV is right
    even when the trajectory was written with wrapping on and a group
    straddles the box edge.
    """

    def __init__(self, cvset, universe):
        self.cvset = cvset
        self.u = universe
        self.groups = []
        for c in cvset.cvs:
            gs = []
            for sel in c["atoms"]:
                g = universe.select_atoms(sel)
                if g.n_atoms == 0:
                    raise ValueError(f"CV {c['name']}: selection '{sel}' matches no atoms")
                gs.append(g)
            self.groups.append(gs)

    def _centers(self, gs):
        from MDAnalysis.lib.distances import minimize_vectors
        box = self.u.dimensions
        have_box = box is not None and np.all(box[:3] > 0)

        def centre(g):
            if g.n_atoms == 1:
                return g.positions[0].astype(float)
            if not have_box:
                return g.center_of_mass()
            # a group cut in two by the box edge (a wrapped trajectory) has its
            # centre of mass in the middle of the box: bring every atom to the
            # image nearest the group's first atom before averaging.  Right for
            # any group smaller than half the box.
            p0 = g.positions[0].astype(float)
            d = minimize_vectors((g.positions - g.positions[0]).astype(np.float32), box).astype(float)
            m = g.masses
            return p0 + (d * m[:, None]).sum(axis=0) / m.sum() if m.sum() > 0 else p0 + d.mean(axis=0)
        pts = [centre(g) for g in gs]
        if have_box:
            ref = pts[0]
            pts = [ref] + [ref + minimize_vectors((p - ref)[None, :].astype(np.float32), box)[0]
                           for p in pts[1:]]
        return [np.asarray(p, dtype=float) for p in pts]

    def frame(self):
        """CV values for the current frame, in CVSet order."""
        return np.array([cv_values_from_points(c["type"], self._centers(gs))
                         for c, gs in zip(self.cvset.cvs, self.groups)])

    def trajectory(self, start=None, stop=None, step=None):
        out = [self.frame() for _ in self.u.trajectory[start:stop:step]]
        return np.array(out)

    def metric_tensor(self, h=1e-4):
        """M_ij = sum over atoms of (1/m) grad(theta_i) . grad(theta_j) for the
        current frame, in (CV unit)^2 per amu.

        This is the tensor the string method in collective variables moves
        with: an image goes along -M grad F, not along -grad F.  A CV here is
        a function of the centers of mass of its atom groups, so its gradient
        on atom a of group g is (m_a / M_g) dtheta/dR_g; dtheta/dR_g is taken
        by central differences."""
        grads = []                                            # per CV: {atom index: 3-vector}
        for c, gs in zip(self.cvset.cvs, self.groups):
            pts = self._centers(gs); per = c["type"] in PERIODIC_TYPES
            ga = {}
            for k, g in enumerate(gs):
                d = np.zeros(3)
                for x in range(3):
                    hi = [p.copy() for p in pts]; lo = [p.copy() for p in pts]
                    hi[k][x] += h; lo[k][x] -= h
                    dv = cv_values_from_points(c["type"], hi) - cv_values_from_points(c["type"], lo)
                    if per:
                        dv = (dv + np.pi) % (2 * np.pi) - np.pi
                    d[x] = dv / (2 * h)
                w = g.masses / g.masses.sum()
                for idx, wa in zip(g.indices, w):
                    ga[idx] = ga.get(idx, 0.0) + wa * d
            grads.append(ga)
        mass = self.u.atoms.masses
        n = len(grads); M = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                common = grads[i].keys() & grads[j].keys()
                M[i, j] = M[j, i] = sum(float(grads[i][a] @ grads[j][a]) / mass[a] for a in common)
        return M


# --------------------------------------------------------------------------
# periodic arithmetic in CV space
# --------------------------------------------------------------------------
def wrap(d, periodic):
    """Wrap differences d (..., D) of periodic components into (-pi, pi]."""
    d = np.array(d, dtype=float, copy=True)
    p = np.asarray(periodic, dtype=bool)
    if p.any():
        d[..., p] = (d[..., p] + np.pi) % TWO_PI - np.pi
    return d


def diff(a, b, periodic):
    """a - b with periodic components wrapped."""
    return wrap(np.asarray(a, float) - np.asarray(b, float), periodic)


def sdist(a, b, scale, periodic):
    """Scaled distance between points (or arrays of points) of CV space."""
    return np.linalg.norm(diff(a, b, periodic) / np.asarray(scale, float), axis=-1)


def unwrap_path(path, periodic):
    """Make a path continuous: each periodic component of each point is moved
    by multiples of 2 pi to be nearest the previous point.  A string must be
    stored this way, because cfacv never wraps string coordinates."""
    path = np.array(path, dtype=float, copy=True)
    for i in range(1, len(path)):
        path[i] = path[i - 1] + diff(path[i], path[i - 1], periodic)
    return path


def circ_mean_sd(x):
    """Circular mean and standard deviation (radians) of angles x."""
    s, c = np.sin(x).mean(), np.cos(x).mean()
    m = np.arctan2(s, c)
    return m, np.std(wrap((x - m)[:, None], [True])[:, 0])


def path_length(path, scale):
    """Cumulative scaled arc length of an (already continuous) path."""
    seg = np.linalg.norm(np.diff(path, axis=0) / np.asarray(scale, float), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def redistribute(path, n, scale):
    """n points at equal scaled arc length along a continuous polyline."""
    path = np.asarray(path, float)
    L = path_length(path, scale)
    if L[-1] <= 0:
        raise ValueError("path has zero length")
    s = np.linspace(0.0, L[-1], n)
    out = np.empty((n, path.shape[1]))
    for d in range(path.shape[1]):
        out[:, d] = np.interp(s, L, path[:, d])
    return out


# --------------------------------------------------------------------------
# files cfacv reads and writes
# --------------------------------------------------------------------------
_CVOUT = re.compile(r"^CFACV/C\)\s+(Z|Th|FD)\s+(-?\d+)\s+(.*)$")


def read_cvout(path):
    """Read a cfacv restraint report (TAMDoutputFile, or a NAMD log).

    Returns {"Z": (steps, values), "Th": (steps, values)} for the kinds
    present: Z is the restraint target, Th the instantaneous CV value.
    """
    acc = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _CVOUT.match(line)
            if not m:
                continue
            try:
                vals = [float(v) for v in m.group(3).split()]
            except ValueError:
                continue
            acc.setdefault(m.group(1), []).append((int(m.group(2)), vals))
    out = {}
    for k, rows in acc.items():
        # a line cut short (a walltime kill in mid-write) is dropped; cutting
        # every row to its width instead would silently lose CVs
        widths = [len(r[1]) for r in rows]
        n = max(set(widths), key=widths.count)
        rows = [r for r in rows if len(r[1]) == n]
        out[k] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    return out


def read_forces(path):
    """Read a single-sweep centers-and-forces file (the format Reconstruct
    reads): comments, 'NCV NDATA', a line of CV type keywords, a blank line,
    then NDATA rows of NCV positions followed by NCV mean forces.

    The force column is the mean force f = k <theta - z> = -dF/dz.
    """
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    ncv, ndata = (int(v) for v in lines[0].split()[:2])
    types = lines[1].split()
    rows = [l.split() for l in lines[2:] if l.strip()]
    a = np.array(rows[:ndata], dtype=float)
    if a.shape != (ndata, 2 * ncv):
        raise ValueError(f"{path}: expected {ndata} rows of {2 * ncv} numbers, got {a.shape}")
    return a[:, :ncv], a[:, ncv:], types


def write_forces(path, z, f, types, comment=""):
    with open(path, "w") as fh:
        fh.write(f"# {comment}\n{z.shape[1]} {z.shape[0]}\n{' '.join(types)}\n\n")
        for zi, fi in zip(z, f):
            fh.write(" ".join(f"{v:.8f}" for v in list(zi) + list(fi)) + "\n")


def restraint_line(i, ncv, k, rf, b=None, extra=""):
    """One line of a cfacv restraint file: a restraint on CV i alone."""
    coef = " ".join("1" if j == i else "0" for j in range(ncv))
    s = f"{coef} {{k {k}}} {{rf {rf}}}"
    if b is not None:
        s += f" {{b {b:.6f}}}"
    return s + (" " + extra if extra else "")


def write_restraints(path, target, ks, types):
    """Restraint file holding every string CV at `target` (one line per CV).
    Torsions get a PERIODIC restraint, everything else HARMONIC.

    A torsion target is written as given, NOT wrapped into (-pi, pi]: the
    string driver reads these targets back as the string itself, and a string
    must be continuous.  cfacv's periodic restraint wraps (value - target)
    once.  The value is in (-pi, pi], so that is correct exactly when the
    target is strictly inside (-2 pi, 2 pi); a target outside that range is
    refused here.  The same limit holds while the string evolves: a torsion
    coordinate of a string must not drift past +/-2 pi."""
    n = len(target)
    with open(path, "w") as fh:
        for i in range(n):
            rf = "PERIODIC" if types[i] in PERIODIC_TYPES else "HARMONIC"
            if rf == "PERIODIC" and not abs(target[i]) < TWO_PI - 0.05:
                raise ValueError(f"torsion target {target[i]:.3f} rad is outside (-2 pi, 2 pi), where "
                                 "cfacv's periodic restraint is wrong; shift the path by a multiple of "
                                 "2 pi (center_periodic), or the path winds too far in this torsion")
            fh.write(restraint_line(i, n, ks[i], rf, target[i]) + "\n")


def center_periodic(path, periodic):
    """Shift each periodic column of a continuous path by a multiple of 2 pi
    so that its midrange is as close to zero as possible."""
    path = np.array(path, dtype=float, copy=True)
    for j in np.where(np.asarray(periodic, bool))[0]:
        mid = 0.5 * (path[:, j].min() + path[:, j].max())
        path[:, j] -= TWO_PI * np.round(mid / TWO_PI)
    return path


def block_mean_se(x, nblocks=5):
    """Mean of each column of x and its standard error from block averages.
    Returns (mean, se, block_means)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 4:
        raise ValueError(f"{len(x)} samples: too few for an error estimate")
    nb = max(2, min(nblocks, len(x) // 2))
    blocks = np.array([b.mean(axis=0) for b in np.array_split(x, nb)])
    return x.mean(axis=0), blocks.std(axis=0, ddof=1) / np.sqrt(nb), blocks


def mean_force(cvout, k, periodic, discard=0.2, nblocks=5):
    """Mean force f = k <theta - z> = -dF/dz from a cfacv restraint report.

    cvout: dict from read_cvout with "Z" (targets) and "Th" (CV values).
    Returns (z, f, se, theta) with theta the retained CV samples."""
    if "Z" not in cvout or "Th" not in cvout:
        raise ValueError("restraint report needs both Z and Th lines (TAMDoutputlevel 3)")
    Z, Th = cvout["Z"][1], cvout["Th"][1]
    n = min(len(Z), len(Th))
    i0 = int(discard * n)
    if n - i0 < 4:
        raise ValueError(f"only {n - i0} samples left after discarding {discard:.0%}")
    Z, Th = Z[i0:n], Th[i0:n]
    d = wrap(Th - Z, periodic) * np.asarray(k, float)
    f, se, _ = block_mean_se(d, nblocks)
    return Z[-1], f, se, Th


def back_to_root(conf_path):
    """Relative path from a config file's directory back to the current
    directory, for the `cd` line that generated NAMD configs start with."""
    import os
    return os.path.relpath(os.getcwd(), os.path.dirname(os.path.abspath(conf_path)))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def save_json(path, obj):
    def enc(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        raise TypeError(type(o))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=enc)
        fh.write("\n")
