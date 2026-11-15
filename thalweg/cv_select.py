#!/usr/bin/env python3
"""Stage 3: choose the collective variables from the system's own data.

The string method needs CVs before it can start, and a published set is a
hypothesis about this system, not a measurement of it.  This stage picks the
CVs from the end-state ensembles and from where trajectories started between
the states end up, and treats any published set as one more candidate.

  candidates  enumerate coordinates that differ between the two end states
              (C-alpha distances, salt-bridge distances, side-chain chi1
              torsions), rank each kind by effect size, and write them with
              any published CVs as one candidate pool.
  tmd         write targeted-MD runs that drive each end state toward the
              other on RMSD.  No CVs are assumed, which is the point: these
              only supply configurations between the basins.
  shots       take configurations along those runs and write unbiased
              trajectories of fixed length from each, with fresh velocities.
  classifier  describe each end state over the WHOLE pool, so a configuration
              can be placed in one basin, the other, or neither.
  outcomes    read the shots back and record the basin each reached first.
  select      likelihood maximization (Peters and Trout, J Chem Phys 125,
              054108): fit p(open) = 1/2 (1 + tanh q), q linear in a few
              candidates, to the outcomes; grow the set one candidate at a
              time while the Bayesian information criterion improves.

Targeted MD is itself biased, so its configurations sample the transition
region imperfectly.  The committor test at the end of the protocol is the
check on what this stage selects.

Example:
  cv_select.py candidates --psf sys.psf --closed c*.dcd --open o*.dcd \\
      --include t4l_cvs.json --out cand
  cv_select.py tmd --psf closed.psf --coor closed.coor --xsc closed.xsc \\
      --target-psf open.psf --target-coor open.coor --namd-base base.namd \\
      --runs 5 --steps 10000000 --out tmd_c2o
  cv_select.py shots --psf closed.psf --dcd tmd_c2o/*.dcd --n 100 \\
      --steps 1000000 --namd-base base.namd --out shots
  cv_select.py classifier --cv cand.json --psf sys.psf --closed c*.dcd --open o*.dcd --out clf.json
  cv_select.py outcomes --cv cand.json --classifier clf.json --psf closed.psf --dir shots
  cv_select.py select --cv cand.json --outcomes shots/outcomes.json --published t4l_cvs.json
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np

from common import CVCalculator, CVSet, back_to_root, load_json, save_json, wrap

ACID = {"ASP": "CG", "GLU": "CD"}
BASE = {"ARG": "CZ", "LYS": "NZ"}
GAMMA = ("CG", "CG1", "OG", "OG1", "SG")


# --------------------------------------------------------------------------
# features: a CV value becomes one number, or (sin, cos) for a torsion
# --------------------------------------------------------------------------
def featurize(x, periodic):
    """x: (n, nCV).  Returns (features, owner) where owner[j] is the CV that
    feature j belongs to.  A torsion contributes its sine and cosine."""
    cols, owner = [], []
    for j in range(x.shape[1]):
        if periodic[j]:
            cols += [np.sin(x[:, j]), np.cos(x[:, j])]; owner += [j, j]
        else:
            cols.append(x[:, j]); owner.append(j)
    return np.column_stack(cols), np.array(owner)


def effect_size(a, b, periodic=False):
    """|difference of means| / pooled sd; circular for torsions."""
    if periodic:
        ma, mb = np.arctan2(np.sin(a).mean(0), np.cos(a).mean(0)), np.arctan2(np.sin(b).mean(0), np.cos(b).mean(0))
        sa = np.sqrt(np.mean(wrap(a - ma, np.ones(a.shape[1], bool)) ** 2, axis=0))
        sb = np.sqrt(np.mean(wrap(b - mb, np.ones(b.shape[1], bool)) ** 2, axis=0))
        d = np.abs(wrap((mb - ma)[None, :], np.ones(len(ma), bool))[0])
    else:
        d, sa, sb = np.abs(b.mean(0) - a.mean(0)), a.std(0), b.std(0)
    return d / np.maximum(np.sqrt(0.5 * (sa ** 2 + sb ** 2)), 1e-9)


def pick_spread(score, pairs, n, gap):
    """Top-n pairs by score, skipping a pair whose two residues are both
    within `gap` of the residues of a pair already taken (near-duplicates)."""
    out = []
    for k in np.argsort(-score):
        i, j = pairs[k]
        if all(not ((abs(i - a) <= gap and abs(j - b) <= gap) or (abs(i - b) <= gap and abs(j - a) <= gap)) for a, b in out):
            out.append((int(i), int(j)))
            if len(out) == n:
                break
    return out


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------
def _frames(psf, dcds, sel_groups, skip, stride):
    """Positions of several single-atom-per-residue groups over trajectories."""
    import MDAnalysis as mda
    out = [[] for _ in sel_groups]
    for dcd in dcds:
        u = mda.Universe(psf, dcd)
        groups = [u.atoms[idx] for idx in sel_groups]
        for _ in u.trajectory[skip::stride]:
            for o, g in zip(out, groups):
                o.append(g.positions.astype(float))
    return [np.array(o) for o in out]


def _dihedrals(p):
    b0, b1, b2 = p[:, :, 0] - p[:, :, 1], p[:, :, 2] - p[:, :, 1], p[:, :, 3] - p[:, :, 2]
    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, -1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, -1, keepdims=True) * b1
    return np.arctan2(np.sum(np.cross(b1, v) * w, -1), np.sum(v * w, -1))


def candidates(a):
    import MDAnalysis as mda
    u = mda.Universe(a.psf, a.closed[0])
    prot = u.select_atoms(a.protein)
    res = prot.residues
    seg = f"({a.protein}) and "
    ca = [r.atoms.select_atoms("name CA") for r in res]
    keep = [k for k, g in enumerate(ca) if len(g) == 1]
    ca_idx = np.array([ca[k][0].index for k in keep]); ca_res = np.array([res[k].resid for k in keep])
    # salt-bridge ends
    acid = [(r.resid, r.atoms.select_atoms(f"name {ACID[r.resname]}")[0].index) for r in res if r.resname in ACID and len(r.atoms.select_atoms(f"name {ACID[r.resname]}"))]
    base = [(r.resid, r.atoms.select_atoms(f"name {BASE[r.resname]}")[0].index) for r in res if r.resname in BASE and len(r.atoms.select_atoms(f"name {BASE[r.resname]}"))]
    # chi1 = N-CA-CB-XG
    chi = []
    for r in res:
        n, c, b = (r.atoms.select_atoms(f"name {x}") for x in ("N", "CA", "CB"))
        g = [x for nm in GAMMA for x in r.atoms.select_atoms(f"name {nm}")][:1]
        if len(n) == len(c) == len(b) == 1 and g:
            chi.append((r.resid, [n[0].index, c[0].index, b[0].index, g[0].index], g[0].name))
    groups = [ca_idx, np.array([i for _, i in acid], int), np.array([i for _, i in base], int),
              np.array([q for _, quad, _ in chi for q in quad], int)]
    C = _frames(a.psf, a.closed, groups, a.skip, a.stride)
    O = _frames(a.psf, a.opened, groups, a.skip, a.stride)
    cvs, table = [], []

    # C-alpha pairs at least --min-sep apart in sequence
    ii, jj = np.triu_indices(len(ca_idx), k=a.min_sep)
    d = lambda P: np.linalg.norm(P[:, ii] - P[:, jj], axis=-1)
    es = effect_size(d(C[0]), d(O[0]))
    for i, j in pick_spread(es, list(zip(ca_res[ii], ca_res[jj])), a.n_ca, a.gap):
        k = np.where((ca_res[ii] == i) & (ca_res[jj] == j))[0][0]
        cvs.append({"name": f"ca_{i}_{j}", "type": "BOND", "atoms": [f"{seg}resid {i} and name CA", f"{seg}resid {j} and name CA"]})
        table.append((cvs[-1]["name"], float(es[k])))
    # salt bridges
    if len(acid) and len(base):
        pa, pb = np.meshgrid(np.arange(len(acid)), np.arange(len(base)), indexing="ij")
        pa, pb = pa.ravel(), pb.ravel()
        d = lambda A, B: np.linalg.norm(A[:, pa] - B[:, pb], axis=-1)
        dc, do = d(C[1], C[2]), d(O[1], O[2])
        es = effect_size(dc, do)
        es[np.minimum(dc.mean(0), do.mean(0)) > a.max_bridge] = 0      # never in contact in either state
        for k in np.argsort(-es)[:a.n_bridge]:
            (ra, _), (rb, _) = acid[pa[k]], base[pb[k]]
            rn_a, rn_b = res[[r.resid for r in res].index(ra)].resname, res[[r.resid for r in res].index(rb)].resname
            cvs.append({"name": f"sb_{rb}_{ra}", "type": "BOND",
                        "atoms": [f"{seg}resid {rb} and name {BASE[rn_b]}", f"{seg}resid {ra} and name {ACID[rn_a]}"]})
            table.append((cvs[-1]["name"], float(es[k])))
    # chi1
    if chi:
        t = lambda P: _dihedrals(P.reshape(len(P), len(chi), 4, 3))
        es = effect_size(t(C[3]), t(O[3]), periodic=True)
        for k in np.argsort(-es)[:a.n_chi]:
            rid, _, gname = chi[k]
            cvs.append({"name": f"chi1_{rid}", "type": "DIHED",
                        "atoms": [f"{seg}resid {rid} and name {x}" for x in ("N", "CA", "CB", gname)]})
            table.append((cvs[-1]["name"], float(es[k])))
    # published CVs join the pool under their own names
    if a.include:
        pub = load_json(a.include)["cvs"]
        have = {tuple(c["atoms"]) for c in cvs}
        for c in pub:
            if tuple(c["atoms"]) not in have and c["name"] not in {x["name"] for x in cvs}:
                cvs.append(c); table.append((c["name"], float("nan")))
    save_json(a.out + ".json", {"cvs": cvs, "_note": "candidate pool from cv_select.py candidates; effect sizes in " + a.out + "_ranking.json"})
    save_json(a.out + "_ranking.json", {"effect_size": dict(table)})
    print(f"{len(cvs)} candidates ({a.n_ca} C-alpha distances, up to {a.n_bridge} salt bridges, {a.n_chi} chi1 torsions"
          + (f", plus published CVs from {a.include}" if a.include else "") + ")")
    for n, e in sorted(table, key=lambda t: -(t[1] if t[1] == t[1] else -1))[:12]:
        print(f"  {n:16s} effect size {e:6.2f}" if e == e else f"  {n:16s} published")
    print(f"wrote {a.out}.json")


# --------------------------------------------------------------------------
# targeted MD and shots
# --------------------------------------------------------------------------
TMD_CONF = """# generated by cv_select.py: targeted MD toward the other end state, run {i}
cd {root}
bincoordinates  {start}.coor
extendedSystem  {start}.xsc
temperature     {temperature}
seed            {seed}
outputname      {out}/tmd{i:02d}
dcdfreq         {dcdfreq}
source          {base}
TMD             on
TMDk            {k}
TMDOutputFreq   {dcdfreq}
TMDFile         {out}/target.pdb
TMDFirstStep    0
TMDLastStep     {steps}
TMDFinalRMSD    {final}
run {steps}
"""

SHOT_CONF = """# generated by cv_select.py: unbiased trajectory {s} from configuration {c}
cd {root}
bincoordinates  {start}.coor
extendedSystem  {start}.xsc
temperature     {temperature}   ; # fresh velocities from this shot's seed
seed            {seed}
outputname      {stem}
dcdfreq         {dcdfreq}
source          {base}
run {steps}
"""


# With a basin model in hand before the shots are run, each one can stop the
# moment it commits instead of running its full length.
SHOT_STOP = """if {{![info exists env(CFACV_BASEDIR)]}} {{ puts "set CFACV_BASEDIR"; exit 1 }}
set CFACV_BASEDIR $env(CFACV_BASEDIR)
tclforces       on
tclforcesscript $CFACV_BASEDIR/tcl/cfacv_tclforces.tcl
set labelPDB        {label}
set cvINP           {cvinp}
set restrPARAMS     {{{{k 0.0}}}}  ; # measure the CVs, apply no bias
set TAMDof          {chunk}
set TAMDoutputlevel 2
set TAMDoutputFile  {stem}.cvout
set shot_chunk   {chunk}
set shot_cap     {steps}
set shot_result  {stem}.result
set basin_names  {{{names}}}
{basins}
set cv_periodic  {{{periodic}}}
source {tcl}
"""


def tmd(a):
    import MDAnalysis as mda
    u = mda.Universe(a.psf, a.coor, format="NAMDBIN")
    t = mda.Universe(a.target_psf, a.target_coor, format="NAMDBIN")
    su, st = u.select_atoms(a.sel), t.select_atoms(a.sel)
    if su.n_atoms != st.n_atoms or list(su.names) != list(st.names) or list(su.resids) != list(st.resids):
        sys.exit(f"selection '{a.sel}' does not match between the two systems ({su.n_atoms} and {st.n_atoms} atoms); "
                 "the end states must have the same sequence")
    os.makedirs(a.out, exist_ok=True)
    # the target file is the RUNNING system's atom list, with the other
    # state's coordinates on the selected atoms and occupancy 1 on them
    # NAMD does not move the target onto the running structure, so do it
    # here: best-fit the other state's selected atoms onto the current ones.
    from MDAnalysis.analysis.align import rotation_matrix
    cur, tgt = su.positions.astype(float), st.positions.astype(float)
    c0, t0 = cur.mean(0), tgt.mean(0)
    R, rmsd0 = rotation_matrix(tgt - t0, cur - c0)
    fitted = (tgt - t0) @ R.T + c0
    pos = u.atoms.positions.copy()          # .positions returns a copy: assign the whole array back
    pos[su.indices] = fitted
    u.atoms.positions = pos
    occ = np.zeros(u.atoms.n_atoms); occ[su.indices] = 1.0
    u.add_TopologyAttr("occupancies", occ)
    u.add_TopologyAttr("tempfactors", np.zeros(u.atoms.n_atoms))
    u.atoms.write(os.path.join(a.out, "target.pdb"))
    stem = a.coor[:-5]
    confs = []
    for i in range(a.runs):
        conf = os.path.join(a.out, f"tmd{i:02d}.conf")
        with open(conf, "w") as fh:
            fh.write(TMD_CONF.format(i=i, root=back_to_root(conf), start=stem, temperature=a.temperature, seed=a.seed + i,
                                     out=a.out, dcdfreq=a.dcdfreq, base=a.namd_base, k=a.k, steps=a.steps, final=a.final_rmsd))
        confs.append(conf)
    with open(os.path.join(a.out, "runs.list"), "w") as fh:
        fh.write("\n".join(confs) + "\n")
    print(f"{a.runs} targeted-MD runs set up in {a.out}/ on {su.n_atoms} atoms ('{a.sel}'); "
          f"RMSD to the target after fitting {rmsd0:.2f} A, to be driven to {a.final_rmsd} A")
    if rmsd0 < a.final_rmsd + 0.2:
        print("WARNING: the start is already about as close to the target as the run is asked to bring it")


def shots(a):
    import MDAnalysis as mda
    from harvest_centers import write_xsc
    os.makedirs(os.path.join(a.out, "configs"), exist_ok=True)
    stop = None
    if a.classifier:
        if not (a.label and a.cvinp):
            sys.exit("--classifier needs --label and --cvinp: the cfacv inputs of the SAME pool, in the same order "
                     "(make_cfacv_inputs.py writes them from the pool's JSON)")
        import shutil
        clf = BasinClassifier.load(a.classifier)
        ncv = sum(1 for l in open(a.cvinp) if l.strip() and not l.lstrip().startswith("#"))
        if ncv != len(clf.periodic):
            sys.exit(f"{a.cvinp} defines {ncv} CVs, the basin model was trained on {len(clf.periodic)}")
        tcl = os.path.join(a.out, "cvsel_shot.tcl")
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvsel_shot.tcl"), tcl)
        lst = lambda v: " ".join(f"{x:.8g}" for x in v)
        basins = "\n".join(f"set basin_mean({k}) {{{lst(s['mean'])}}}\nset basin_sd({k}) {{{lst(s['sd'])}}}\nset basin_thr({k}) {s['thr']:.8g}"
                           + (f"\nset basin_zmax({k}) {s['zmax']:.8g}" if np.isfinite(s["zmax"]) else "")
                           + (f"\nset basin_zcap({k}) {s['zcap']:.8g}\nset basin_cap({k}) {{{' '.join(str(int(v)) for v in s['cap'])}}}"
                              if np.isfinite(s["zcap"]) else "")
                           for k, s in clf.stats.items())
        stop = dict(label=a.label, cvinp=a.cvinp, chunk=a.chunk, names=" ".join(clf.stats), basins=basins,
                    periodic=" ".join(str(int(p)) for p in clf.periodic), tcl=tcl)
    # equal numbers from each trajectory, evenly spaced along it
    per = max(1, a.n // len(a.dcd)); c = 0; confs = []
    # AUDIT: some configurations run their full length without stopping, so
    # that 'outcomes' can check the one assumption stopping rests on: that a
    # trajectory which enters a basin stays there
    every = int(round(1 / a.audit_fraction)) if (stop and a.audit_fraction > 0) else 0
    audit = set(range(every // 2, per * len(a.dcd), every)) if every else set()
    for dcd in a.dcd:
        u = mda.Universe(a.psf, dcd)
        nf = len(u.trajectory)
        for fr in np.unique(np.linspace(0, nf - 1, per).astype(int)):
            ts = u.trajectory[fr]
            stem = os.path.join(a.out, "configs", f"config{c:04d}")
            u.atoms.write(stem + ".coor", file_format="NAMDBIN"); write_xsc(stem + ".xsc", ts.dimensions)
            for s in range(a.per_config):
                st = os.path.join(a.out, f"c{c:04d}_s{s}")
                text = SHOT_CONF.format(s=s, c=c, root=back_to_root(st + ".conf"), start=stem, temperature=a.temperature,
                                        seed=a.seed + 10 * c + s, stem=st, dcdfreq=a.dcdfreq, base=a.namd_base, steps=a.steps)
                if stop and c not in audit:
                    text = text.replace(f"run {a.steps}\n", SHOT_STOP.format(stem=st, steps=a.steps, **stop))
                elif stop:
                    text = text.replace(f"dcdfreq         {a.dcdfreq}", f"dcdfreq         {a.chunk}")   # tested at the same cadence
                with open(st + ".conf", "w") as fh:
                    fh.write(text)
                confs.append(st + ".conf")
            c += 1
    with open(os.path.join(a.out, "runs.list"), "w") as fh:
        fh.write("\n".join(confs) + "\n")
    save_json(os.path.join(a.out, "setup.json"), {"n_configs": c, "per_config": a.per_config, "steps": a.steps,
                                                  "stop_on_basin": bool(stop), "audit": sorted(k for k in audit if k < c)})
    print(f"{len(confs)} unbiased trajectories from {c} configurations set up in {a.out}/"
          + (f"; each stops when it enters a basin (tested every {a.chunk} steps, cap {a.steps})" if stop else ""))


# --------------------------------------------------------------------------
# basin classifier over the whole pool
# --------------------------------------------------------------------------
def fit_logistic(F, y, l2=1e-2, iters=400):
    """Logistic regression by Newton steps.  F standardized, y in {0, 1}."""
    X = np.column_stack([np.ones(len(F)), F]); w = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / len(y) + R @ w
        H = (X * (p * (1 - p))[:, None]).T @ X / len(y) + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


class BasinClassifier:
    """Which basin, if either, a configuration is in.

    Each end state is described by the mean and spread of every feature in
    its own ensemble.  A configuration's distance from a state is the mean
    squared z-score over the features; it is IN that state's basin when that
    distance is within the range the state's own frames show (their
    `quantile`) and smaller than its distance from the other state.

    One coordinate far out must not be averaged away by many that agree: with
    forty candidates, a configuration four spreads out in a single coordinate
    still has a small MEAN squared z-score, and would be called in the basin
    before that coordinate (possibly the slow one the selection is looking
    for) has moved.  So every feature that tells the two states apart (their
    means differ by more than `sep_min` pooled spreads) must also be within
    `zcap` spreads of this state's mean.  The cap is a fixed number, 2.5, not
    a quantile of the data: a quantile of the largest |z| grows with the size
    of the pool and with the weight of the tails (3.3 to 16 spreads in tests),
    and by then the basin's edge is at or past the barrier.  A basin has to
    be a CORE of its state for "the first basin entered" to mean anything.
    `separation` records how many of its own spreads apart the states are in
    their best feature; below about 6 no such core exists.

    A dividing surface would not do.  A logistic classifier trained on two
    well-separated states is certain about everything, including a point
    half way between them, so a trajectory would count as committed the
    moment it crossed the surface.  A basin has to be a region the state
    actually occupies, with room between the two for "neither"."""

    def __init__(self, stats, periodic):
        self.stats = {k: {kk: np.asarray(vv, float) if kk not in ("thr", "zmax") else float(vv) for kk, vv in v.items()} for k, v in stats.items()}
        for s in self.stats.values():
            s.setdefault("zmax", float("inf"))                # models saved before the caps existed
            s.setdefault("zcap", float("inf"))
            if "cap" not in s:
                s["cap"] = np.zeros(len(s["mean"]))
            s["zcap"] = float(s["zcap"])
        self.periodic = np.asarray(periodic, bool)

    @classmethod
    def train(cls, closed, opened, periodic, quantile=0.95, zcap=2.5, sep_min=4.0):
        stats = {}
        Fs = {name: featurize(x, periodic)[0] for name, x in (("closed", closed), ("open", opened))}
        mc, mo = Fs["closed"].mean(0), Fs["open"].mean(0)
        sc, so = np.maximum(Fs["closed"].std(0), 1e-6), np.maximum(Fs["open"].std(0), 1e-6)
        sep = np.abs(mc - mo) / np.sqrt(0.5 * (sc ** 2 + so ** 2))       # in pooled spreads
        cap = (sep > sep_min).astype(float)
        for name, F, mean, sd in (("closed", Fs["closed"], mc, sc), ("open", Fs["open"], mo, so)):
            d = np.mean(((F - mean) / sd) ** 2, axis=1)
            stats[name] = {"mean": mean, "sd": sd, "thr": float(np.quantile(d, quantile)), "cap": cap, "zcap": float(zcap),
                           "separation": float(np.max(np.abs(mc - mo) / sd))}
        return cls(stats, periodic)

    def distances(self, x):
        F, _ = featurize(np.atleast_2d(x), self.periodic)
        return {k: np.mean(((F - s["mean"]) / s["sd"]) ** 2, axis=1) for k, s in self.stats.items()}

    def within_caps(self, x):
        """Per state: is every capped feature within zcap spreads (and, for a
        model saved with the earlier rule, every feature within zmax)?"""
        F, _ = featurize(np.atleast_2d(x), self.periodic)
        out = {}
        for k, s in self.stats.items():
            zs = np.abs((F - s["mean"]) / s["sd"])
            capped = np.where(s["cap"] > 0, zs, 0.0).max(axis=1) if zs.shape[1] else np.zeros(len(F))
            out[k] = (capped <= s["zcap"]) & (zs.max(axis=1) <= s["zmax"])
        return out

    def basin(self, x):
        """'closed', 'open' or '' for each row."""
        d = self.distances(x); w = self.within_caps(x)
        c = (d["closed"] <= self.stats["closed"]["thr"]) & (d["closed"] < d["open"]) & w["closed"]
        o = (d["open"] <= self.stats["open"]["thr"]) & (d["open"] < d["closed"]) & w["open"]
        return np.where(o, "open", np.where(c, "closed", ""))

    def first_basin(self, traj, confirm=2):
        """The basin a trajectory enters first, or 'U' if it enters neither.
        Entered means found there on `confirm` consecutive tests: one test is
        a fluctuation."""
        run, last = 0, ""
        for b in self.basin(traj):
            b = str(b)
            run = run + 1 if (b and b == last) else (1 if b else 0)
            last = b
            if b and run >= confirm:
                return b
        return "U"

    def save(self, path):
        save_json(path, {"stats": self.stats, "periodic": self.periodic.tolist()})

    @classmethod
    def load(cls, path):
        d = load_json(path)
        return cls(d["stats"], d["periodic"])


def _measure(cvset, psf, dcds, skip=0, stride=1):
    import MDAnalysis as mda
    out = []
    for dcd in dcds:
        u = mda.Universe(psf, dcd)
        out.append(CVCalculator(cvset, u).trajectory(start=skip, step=stride))
    return np.vstack(out)


def classifier(a):
    cvset = CVSet.load(a.cv)
    C, O = _measure(cvset, a.psf, a.closed, a.skip, a.stride), _measure(cvset, a.psf, a.opened, a.skip, a.stride)
    # held-out check: train on the first half of each state, test on the second
    hc, ho = len(C) // 2, len(O) // 2
    half = BasinClassifier.train(C[:hc], O[:ho], cvset.periodic, a.quantile)
    bc, bo = half.basin(C[hc:]), half.basin(O[ho:])
    wrong = float(np.mean(bc == "open") + np.mean(bo == "closed")) / 2
    unassigned = float(np.mean(bc == "") + np.mean(bo == "")) / 2
    clf = BasinClassifier.train(C, O, cvset.periodic, a.quantile)
    clf.save(a.out)
    ok = wrong <= a.max_wrong
    print(f"{len(cvset.names)} candidates, {len(C)} closed and {len(O)} open frames")
    print(f"held-out frames put in the WRONG basin: {wrong:.4f}; in neither: {unassigned:.3f}")
    F0, owner = featurize(C[:1], cvset.periodic)
    ncap = int(clf.stats["closed"]["cap"].sum())
    sepc, sepo = clf.stats["closed"]["separation"], clf.stats["open"]["separation"]
    capped = sorted({cvset.names[owner[j]] for j in np.where(clf.stats["closed"]["cap"] > 0)[0]})
    print(f"the states are {min(sepc, sepo):.1f} of their own spreads apart in their best feature; "
          f"{ncap} feature(s) tell them apart and are capped at {clf.stats['closed']['zcap']} spreads: {', '.join(capped) or 'none'}")
    if min(sepc, sepo) < 6.0:
        print("WARNING: under about 6 spreads apart no core of either state is clear of the barrier, and stopping a trajectory "
              "the first time it is found in a basin will mislabel some outcomes. Check the audit that 'outcomes' reports.")
    print(f"GATE basin classifier: {'PASS' if ok else 'FAIL'} (limit on wrong-basin fraction {a.max_wrong})")
    print(f"wrote {a.out}")
    return 0 if ok else 1


def outcomes(a):
    import MDAnalysis as mda
    cvset = CVSet.load(a.cv); clf = BasinClassifier.load(a.classifier)
    s = load_json(os.path.join(a.dir, "setup.json"))
    rows = []; count = {"closed": 0, "open": 0, "U": 0, "missing": 0}; used = 0; stopped = 0
    audit_set = set(s.get("audit", [])); audit_n = audit_bad = 0
    for c in range(s["n_configs"]):
        stem = os.path.join(a.dir, "configs", f"config{c:04d}")
        x0 = CVCalculator(cvset, mda.Universe(a.psf, stem + ".coor", format="NAMDBIN")).frame()
        for sh in range(s["per_config"]):
            dcd = os.path.join(a.dir, f"c{c:04d}_s{sh}.dcd")
            result = os.path.join(a.dir, f"c{c:04d}_s{sh}.result")
            if os.path.exists(result):
                # the shot tested itself as it ran and stopped on entering a basin
                tok = open(result).read().split()
                if len(tok) < 2:
                    count["missing"] += 1; continue      # a result file cut short
                res = tok[0]; used += int(tok[1]); stopped += 1
                if res not in count:
                    sys.exit(f"{result}: unknown outcome '{res}'")
                count[res] += 1
            elif s.get("stop_on_basin") and c not in audit_set:
                count["missing"] += 1; continue          # a self-stopping shot that did not finish
            elif not os.path.exists(dcd):
                count["missing"] += 1; continue
            else:
                traj = CVCalculator(cvset, mda.Universe(a.psf, dcd)).trajectory()
                res = clf.first_basin(traj); count[res] += 1
                if c in audit_set and res != "U":
                    # where it is at the end of its full length (most common basin over the last tenth)
                    tail = [str(b) for b in clf.basin(traj[-max(2, len(traj) // 10):]) if b]
                    if tail:
                        end = max(set(tail), key=tail.count)
                        audit_n += 1; audit_bad += int(end != res)
            rows.append({"config": c, "shot": sh, "x": x0, "outcome": res, "start_basin": str(clf.basin(x0)[0])})
    audit = {"n": audit_n, "mismatch": audit_bad, "mismatch_fraction": (audit_bad / audit_n if audit_n else 0.0)}
    save_json(os.path.join(a.dir, "outcomes.json"), {"cvs": cvset.names, "rows": rows, "counts": count, "audit": audit})
    n = count["closed"] + count["open"]
    print(f"{n} committed trajectories ({count['closed']} closed, {count['open']} open), {count['U']} uncommitted, {count['missing']} missing")
    if stopped:
        full = stopped * s["steps"]
        print(f"the shots stopped on entering a basin: {used} MD steps used of the {full} a fixed length would have cost ({used / full:.0%})")
    if audit_set:
        print(f"audit: {audit_n} trajectories ran their full length; {audit_bad} ended in a different basin from the one they "
              f"entered first ({audit['mismatch_fraction']:.0%})" + ("" if audit['mismatch_fraction'] <= 0.05 else
              ": MORE THAN 5%. The basins are not cores of the states; stopping at first entry mislabels outcomes."))
    if n and min(count["closed"], count["open"]) / n < 0.1:
        print("WARNING: fewer than 10% of outcomes are in one basin; the shooting points sit on one side of the barrier")
    print(f"wrote {a.dir}/outcomes.json")


# --------------------------------------------------------------------------
# likelihood maximization
# --------------------------------------------------------------------------
def fit_committor(F, y, l2=1e-3):
    """Maximize sum ln p with p(open) = 1/2 (1 + tanh q), q = w0 + F.w.
    That is logistic regression in 2q.  Returns (weights for q, lnL)."""
    w = fit_logistic(F, y, l2) / 2.0
    return w, loglik(F, y, w)


def loglik(F, y, w):
    q = w[0] + F @ w[1:] if F.shape[1] else np.full(len(y), w[0])
    p = np.clip(0.5 * (1 + np.tanh(q)), 1e-12, 1 - 1e-12)
    return float(np.sum(np.where(y == 1, np.log(p), np.log(1 - p))))


def cv_loglik(F, y, folds=5, seed=0, groups=None):
    """Mean left-out log likelihood per point.  Rows that share a group (two
    shots from one configuration) are always left out together: they have the
    same CV values, so one in the training set gives the other away."""
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    ug = np.unique(groups); order = np.random.default_rng(seed).permutation(len(ug)); tot = 0.0
    folds = max(2, min(folds, len(ug)))
    for k in range(folds):
        te = np.isin(groups, ug[order[k::folds]])
        w, _ = fit_committor(F[~te], y[~te])
        tot += loglik(F[te], y[te], w)
    return tot / len(y)


def _greedy(F, owner, y, n_eff, kmax, ncv):
    """Forward selection by BIC.  Returns (chosen CV indices, steps, lnL of the null, single-CV lnL)."""
    cols = lambda cvs: np.concatenate([np.where(owner == c)[0] for c in cvs]) if cvs else np.array([], int)
    bic = lambda ll, k: -2 * ll + (k + 1) * np.log(n_eff)
    _, ll0 = fit_committor(F[:, :0], y); best_bic = bic(ll0, 0)
    chosen, steps, single = [], [], {}
    while len(chosen) < kmax:
        trials = []
        for c in range(ncv):
            if c in chosen:
                continue
            idx = cols(chosen + [c]); _, ll = fit_committor(F[:, idx], y)
            trials.append((ll, (c,), len(idx)))
            if not chosen:
                single[c] = ll
        if not trials:
            break
        # rank by BIC, not likelihood: a torsion costs two parameters and a
        # distance one, so the best likelihood can fail the BIC test while a
        # cheaper candidate would have passed
        ll, add, k = min(trials, key=lambda t: bic(t[0], t[2]))
        if bic(ll, k) >= best_bic and len(chosen) + 2 <= kmax:
            # Two CVs can matter only together (two distances across one
            # hinge whose DIFFERENCE is the coordinate): neither passes alone.
            # Before stopping, try every pair.
            rest = [c for c in range(ncv) if c not in chosen]
            pairs = []
            for i, c1 in enumerate(rest):
                for c2 in rest[i + 1:]:
                    idx = cols(chosen + [c1, c2]); _, l2 = fit_committor(F[:, idx], y)
                    pairs.append((l2, (c1, c2), len(idx)))
            if pairs:
                l2, add2, k2 = min(pairs, key=lambda t: bic(t[0], t[2]))
                if bic(l2, k2) < best_bic:
                    ll, add, k = l2, add2, k2
        b = bic(ll, k)
        steps.append({"added": list(add), "lnL": ll, "BIC": float(b), "kept": bool(b < best_bic)})
        if b >= best_bic:
            break
        best_bic = b; chosen += list(add)
    return chosen, steps, ll0, single


def select_cvs(x, y, periodic, names, kmax=6, groups=None, n_perm=40, seed=0):
    """Greedy forward selection of CVs by likelihood, stopped by BIC.

    x: (n, nCV) CV values at the shooting points; y: 1 open, 0 closed;
    groups: the configuration each row was shot from.  Shots from one
    configuration are not independent evidence about the CVs (they have the
    same CV values), so the sample size in BIC is the number of
    configurations, cross-validation leaves whole configurations out, and the
    gate is a permutation test: the outcomes are reassigned to other
    configurations n_perm times, the whole selection is rerun each time, and
    the real gain in log likelihood has to beat 95% of those.  (With twin
    shots counted as independent, a pool of pure-noise candidates passed the
    earlier gate three times in four.)

    Returns a dict with the chosen CVs and the table of steps."""
    F, owner = featurize(x, periodic)
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-9)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    ug, ginv = np.unique(groups, return_inverse=True)
    n_eff = len(ug)
    cols = lambda cvs: np.concatenate([np.where(owner == c)[0] for c in cvs]) if cvs else np.array([], int)
    chosen, steps, ll0, single = _greedy(F, owner, y, n_eff, kmax, len(names))
    for s in steps:
        s["added"] = "+".join(names[c] for c in s["added"])
    idx = cols(chosen)
    out = {"chosen": [names[c] for c in chosen], "steps": steps, "lnL_null": ll0, "n_configurations": int(n_eff),
           "single": {names[c]: v for c, v in single.items()}}
    if not chosen:
        out["pass"] = False
        return out
    w, ll = fit_committor(F[:, idx], y)
    gain = ll - ll0
    out["cv_loglik"] = cv_loglik(F[:, idx], y, groups=groups)
    best1 = max(single, key=single.get)
    out["best_single"] = names[best1]
    out["cv_loglik_best_single"] = cv_loglik(F[:, cols([best1])], y, groups=groups)
    out["cv_loglik_null"] = cv_loglik(F[:, :0], y, groups=groups)
    # separable outcomes: almost no point has a fitted probability away from 0 or 1,
    # and the likelihoods are then set by the fit's penalty, not the data
    q = w[0] + F[:, idx] @ w[1:]; pfit = 0.5 * (1 + np.tanh(q))
    out["undecided_fraction"] = float(np.mean((pfit > 0.1) & (pfit < 0.9)))
    out["training_errors"] = int(np.sum((pfit > 0.5) != (y == 1)))
    # (almost) no point on the wrong side of the fit: a committor that is 0 or
    # 1 everywhere, which real shooting outcomes near a barrier never give
    out["separable"] = bool(out["undecided_fraction"] < 0.1 or out["training_errors"] < 0.02 * len(y))
    # permutation null: move each configuration's outcomes to another configuration
    rng = np.random.default_rng(seed)
    first = np.array([np.where(ginv == g)[0][0] for g in range(n_eff)])
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(n_eff)
        Fp = F[first[perm]][ginv]                              # row r now carries the CVs of another configuration
        ch, _, l0, _ = _greedy(Fp, owner, y, n_eff, kmax, len(names))
        null.append(fit_committor(Fp[:, cols(ch)], y)[1] - l0 if ch else 0.0)
    null = np.array(null)
    out["gain"] = float(gain); out["null_gain_95"] = float(np.quantile(null, 0.95)) if n_perm else 0.0
    out["p_value"] = float((1 + np.sum(null >= gain)) / (1 + n_perm)) if n_perm else float("nan")
    out["pass"] = bool((n_perm == 0 or out["p_value"] <= 0.05) and not out["separable"]
                       and out["cv_loglik"] > out["cv_loglik_null"])
    return out


def select(a):
    d = load_json(a.outcomes); names = d["cvs"]
    cvset = CVSet.load(a.cv)
    if cvset.names != names:
        sys.exit("the outcomes were recorded for a different candidate pool")
    rows = [r for r in d["rows"] if r["outcome"] in ("closed", "open")]
    # A shot that starts inside a basin says nothing about the barrier: any
    # coordinate that tells the states apart explains it.  Enough of them make
    # the outcomes separable, and the ranking is then set by the fit's penalty.
    inside = [r for r in rows if r.get("start_basin")]
    if inside and not a.keep_basin_starts:
        rows = [r for r in rows if not r.get("start_basin")]
        print(f"{len(inside)} trajectories started inside a basin and are left out (--keep-basin-starts keeps them)")
    cnt = d.get("counts", {})
    lost = cnt.get("U", 0) + cnt.get("missing", 0)
    total = lost + cnt.get("closed", 0) + cnt.get("open", 0)
    if total and lost / total > 0.2:
        print(f"WARNING: {lost} of {total} trajectories are uncommitted or missing; slow committers near the barrier "
              "are the ones lost, which biases the selection. Lengthen the shots.")
    if len(rows) < 30:
        sys.exit(f"only {len(rows)} committed trajectories; too few to select from")
    x = np.array([r["x"] for r in rows]); y = np.array([r["outcome"] == "open" for r in rows], float)
    groups = np.array([r.get("config", i) for i, r in enumerate(rows)])
    ncfg = len(np.unique(groups)); minority = min(y.mean(), 1 - y.mean())
    print(f"{len(rows)} committed trajectories from {ncfg} configurations, {int(y.sum())} open and {int(len(y) - y.sum())} closed; "
          f"{len(names)} candidates")
    problems = []
    if minority < 0.1:
        problems.append(f"only {minority:.0%} of the outcomes are in one basin: the shooting points sit on one side of the barrier")
    if ncfg < 30:
        problems.append(f"only {ncfg} configurations: shots from one configuration share their CV values, so this is the sample size")
    if total and lost / total > 0.2:
        problems.append(f"{lost} of {total} trajectories are uncommitted or missing")
    audit = d.get("audit")
    if audit and audit["n"] and audit["mismatch_fraction"] > 0.05:
        problems.append(f"{audit['mismatch_fraction']:.0%} of the audited trajectories ended in a different basin from the one they "
                        "entered first: the basins are not stable-state cores, so stopping at first entry mislabels outcomes")
    res = select_cvs(x, y, cvset.periodic, names, a.kmax, groups=groups, n_perm=a.permutations)
    print("step  candidate added        lnL       BIC   kept")
    for i, s in enumerate(res["steps"]):
        print(f"{i + 1:4d}  {s['added']:18s} {s['lnL']:9.2f} {s['BIC']:9.2f}   {'yes' if s['kept'] else 'no (BIC did not improve)'}")
    print("selected:", ", ".join(res["chosen"]) or "(none)")
    if res.get("separable"):
        problems.append(f"the outcomes are separable ({res['training_errors']} points on the wrong side of the fit, "
                        f"{res['undecided_fraction']:.0%} with a fitted probability between 0.1 and 0.9): the likelihoods are "
                        "set by the fit's penalty, not the data, and the shooting points do not sample the barrier")
    if res["chosen"]:
        print(f"gain in log likelihood {res['gain']:.1f}; with the outcomes moved to other configurations the selection gains "
              f"{res['null_gain_95']:.1f} or less 95% of the time (p = {res['p_value']:.3f})")
    if res["chosen"]:
        print(f"left-out log likelihood per point: selected {res['cv_loglik']:.3f}, best single ({res['best_single']}) "
              f"{res['cv_loglik_best_single']:.3f}, no CVs {res['cv_loglik_null']:.3f}")
    if a.published:
        pub = load_json(a.published); pset = pub.get("string") or [c["name"] for c in pub["cvs"]]
        got = [p for p in pset if p in res["chosen"]]
        print(f"published set {pset}: {len(got)} of {len(pset)} recovered ({', '.join(got) or 'none'}); "
              f"selected but not published: {', '.join(c for c in res['chosen'] if c not in pset) or 'none'}")
        res["published"] = pset; res["recovered"] = got
    for pr in problems:
        print("PROBLEM: " + pr)
    res["problems"] = problems
    res["pass"] = bool(res["pass"] and not problems)
    print(f"GATE CV selection: {'PASS' if res['pass'] else 'FAIL'}"
          + ("" if res["pass"] or not res["chosen"] else " (a selection was made, but see above)"))
    save_json(a.out, res)
    full = {"cvs": cvset.cvs, "string": res["chosen"]}
    save_json(a.out.replace(".json", "") + "_cvs.json", full)
    print(f"wrote {a.out} and {a.out.replace('.json', '')}_cvs.json (the pool with the selected set as its string CVs)")
    return 0 if res["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def ends(p):
        p.add_argument("--psf", required=True)
        p.add_argument("--closed", nargs="+", required=True); p.add_argument("--open", nargs="+", required=True, dest="opened")
        p.add_argument("--skip", type=int, default=0); p.add_argument("--stride", type=int, default=1)
    p = sub.add_parser("candidates"); ends(p)
    p.add_argument("--protein", default="protein"); p.add_argument("--include", help="published CV JSON to add to the pool")
    p.add_argument("--n-ca", type=int, default=20); p.add_argument("--n-bridge", type=int, default=8); p.add_argument("--n-chi", type=int, default=10)
    p.add_argument("--min-sep", type=int, default=4, help="least sequence separation of a C-alpha pair")
    p.add_argument("--gap", type=int, default=3, help="residue window that makes two C-alpha pairs near-duplicates")
    p.add_argument("--max-bridge", type=float, default=8.0, help="ignore acid-base pairs farther than this in both states (A)")
    p.add_argument("--out", default="candidates")
    p = sub.add_parser("tmd")
    p.add_argument("--psf", required=True); p.add_argument("--coor", required=True); p.add_argument("--xsc")
    p.add_argument("--target-psf", required=True); p.add_argument("--target-coor", required=True)
    p.add_argument("--sel", default="protein and not name H*"); p.add_argument("--runs", type=int, default=5)
    p.add_argument("--steps", type=int, required=True); p.add_argument("--k", type=float, default=200.0)
    p.add_argument("--final-rmsd", type=float, default=0.5); p.add_argument("--dcdfreq", type=int, default=5000)
    p.add_argument("--temperature", type=float, default=300.0); p.add_argument("--seed", type=int, default=31000)
    p.add_argument("--namd-base", required=True); p.add_argument("--out", default="tmd")
    p = sub.add_parser("shots")
    p.add_argument("--psf", required=True); p.add_argument("--dcd", nargs="+", required=True)
    p.add_argument("--n", type=int, default=200, help="configurations in all"); p.add_argument("--per-config", type=int, default=2)
    p.add_argument("--steps", type=int, required=True); p.add_argument("--dcdfreq", type=int, default=2500)
    p.add_argument("--temperature", type=float, default=300.0); p.add_argument("--seed", type=int, default=52000)
    p.add_argument("--namd-base", required=True); p.add_argument("--out", default="shots")
    p.add_argument("--classifier", help="basin model from 'classifier': each shot then stops when it enters a basin "
                                        "(--steps becomes the cap)")
    p.add_argument("--label", help="cfacv label PDB of the candidate pool (with --classifier)")
    p.add_argument("--cvinp", help="cfacv cv.inp of the candidate pool, same CV order as the pool JSON (with --classifier)")
    p.add_argument("--chunk", type=int, default=500, help="MD steps between basin tests [500]")
    p.add_argument("--audit-fraction", type=float, default=0.1,
                   help="share of the configurations whose shots run their full length without stopping, to check "
                        "that a trajectory which enters a basin stays there [0.1]; 0 for none")
    p = sub.add_parser("classifier"); ends(p)
    p.add_argument("--cv", required=True)
    p.add_argument("--quantile", type=float, default=0.95, help="share of a state's own frames its basin must contain [0.95]")
    p.add_argument("--max-wrong", type=float, default=0.01); p.add_argument("--out", default="classifier.json")
    p = sub.add_parser("outcomes")
    p.add_argument("--cv", required=True); p.add_argument("--classifier", required=True); p.add_argument("--psf", required=True); p.add_argument("--dir", default="shots")
    p = sub.add_parser("select")
    p.add_argument("--cv", required=True); p.add_argument("--outcomes", required=True); p.add_argument("--published")
    p.add_argument("--kmax", type=int, default=6); p.add_argument("--out", default="selection.json")
    p.add_argument("--permutations", type=int, default=40, help="reruns of the selection on reassigned outcomes, for the gate [40]")
    p.add_argument("--keep-basin-starts", action="store_true",
                   help="keep trajectories whose starting configuration was already inside a basin")
    a = ap.parse_args(argv)
    r = {"candidates": candidates, "tmd": tmd, "shots": shots, "classifier": classifier, "outcomes": outcomes, "select": select}[a.cmd](a)
    return r or 0


if __name__ == "__main__":
    sys.exit(main())
