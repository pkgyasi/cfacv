# thalweg

A thalweg is the line along the bottom of a valley, the course water follows
downhill. This toolkit finds the equivalent in a free-energy landscape: the
lowest route between two states.

Scripts around cfacv's string method for a CV space of any dimension, with
CVs of mixed units (distances and torsions). They implement the staged
protocol of the T4 lysozyme proposal: explore, map, search the map for the
lowest-barrier routes, start strings from them, refine, profile, and test the
transition state.

Written 2026-11-14 as `stringtools`; renamed thalweg the same day. Fork-local: not part of upstream cfacv.

## What is here

| stage | file | what it does |
|---|---|---|
| 2 | `endpoint_analysis.py` | CV statistics of the two end states, the joint separation test, the periodic-image check; writes `scales.json` and `basins.json` |
| 3 | `harvest_centers.py` | centers from TAMD trajectories at a minimum SCALED spacing; keeps each center's configuration |
| 3, 6 | `meanforce.py` | sets up restrained MD at a list of points; collects block-averaged mean forces |
| 3 | `fesmap.py` | single-sweep free-energy map in scaled, periodic-aware CV space; sigma by cross-validation |
| 4 | `path_search.py` | lowest-barrier routes on the map (graph search, alternatives, zero-temperature strings, random-start cross-check) |
| 4 | `init_string.py` | image anchors -> the files the cfacv string driver starts from; warm-ups; the initiation gate |
| 5 | `string_monitor.py` | convergence of a string from its history file; merged-or-distinct for two strings |
| 6 | `fe_profile.py` | profile along a string with propagated error, overlap and stability checks; path-independence check |
| 7 | `committor.py`, `committor_shot.tcl` | unbiased shots that stop on basin entry; the transition-state gate |
| 5 | `driver/` | a string driver of its own: wrapped torsions, averaged forces with a measured step and a stopping test, spline respacing. See `driver/README.md`. cfacv's driver is unchanged and still usable |
| 1 | `make_cfacv_inputs.py` | writes cfacv's label PDB and CV file from the CV definition JSON |
| all | `sbatch/namd_array.sbatch` | one NAMD run per array task from a list file |
| | `common.py` | CV definitions, periodic arithmetic, readers for cfacv's files |
| | `t4l_cvs.json` | the eleven T4 lysozyme CVs of Abou-Hatab and Abrams 2025 |

Every tool prints a `GATE ...: PASS/FAIL` line where the protocol has a gate,
and exits non-zero on FAIL. Run any of them with `-h` for its options and an
example.

## The change inside cfacv that these rely on

`SMPARAMS(zscale)`, a list of one length scale per CV, in the string-method
`.conf`. The reparameterization then measures arc length with each CV divided
by its scale. Without it a 10 A change in a distance takes every image and a
2 rad torsion flip gets one or two. Unset, the behavior is the old one. The
string update and the mean force do not use the scale.

It is documented in `docs/source/string_method.md`, including the caution
that a lopsided scale with too few images changes the path found.

## Conventions

- **Units** are cfacv's: angstroms and radians.
- **Mean force** is `f = k <theta - z> = -dF/dz`, the column `ForcesFromLog`
  and `Reconstruct` use.
- **Scaled distance**: every distance in CV space (center spacing, arc
  length, "near", "merged") divides each CV by its entry in `scales.json`,
  the pooled within-basin standard deviation from the endpoint runs.
- **Torsions on a string are continuous, not wrapped.** cfacv never wraps
  string coordinates, so an anchor table may hold a torsion beyond pi.
  `init_string.py` unwraps the table and shifts it by whole turns to sit next
  to (-pi, pi]; cfacv's `PERIODIC` restraint wraps `value - target` once,
  which is right for any target inside (-2 pi, 2 pi).
- **Paths in generated NAMD configs are relative to the directory where the
  setup command was run.** NAMD changes into a config's own directory before
  reading it; each generated config starts with a `cd` back. Run setup in the
  directory you will submit from, and keep that directory on persistent
  scratch.
- **The NAMD base file** (`--namd-base`) holds the system and the dynamics
  and nothing else: structure, parameters, cutoffs, PME, thermostat,
  barostat, timestep. No inputs, outputs, tclforces or `run`.

## The string stage needs two lines changed from the example

The alanine dipeptide example gives every CV one harmonic force constant
(`set restrPARAMS {{k 100}}`). That is wrong for CVs of different units and
for a torsion near +/-180 degrees. `init_string.py setup` writes
`string_stage_snippet.namd` with the lines to use instead: each replica reads
its own `restr.job0.N.inp`, which carries a force constant per CV and a
`PERIODIC` restraint on each torsion.

## A restraint that is not a string coordinate

In string mode every cfacv restraint is a coordinate of the string. A helper
restraint (the alpha3 helix length in T4 lysozyme) therefore cannot be a
cfacv restraint; put it in the NAMD base file by other means (colvars,
`extraBonds`). That combination has not been tested here.

## Order of use

```
endpoint_analysis.py                      -> endpoints_scales.json, endpoints_basins.json
(run TAMD with cfacv)
harvest_centers.py                        -> centers/
meanforce.py setup ; array ; collect      -> forces.dat
fesmap.py fit                             -> map.json
path_search.py                            -> channels_anchors_N.dat
init_string.py setup ; array ; check      -> stringN/  (+ string_stage_snippet.namd)
(run the string with cfacv)
string_monitor.py status / compare        -> stringN_final.dat
meanforce.py setup --points stringN_final.dat ; array
fe_profile.py compute / compare           -> profile
committor.py pick ; setup ; array ; tally
```

## Tests

```
python3 -m pytest thalweg/tests       # no MD needed, about a minute
tclsh tests/reparam_zscale.tcl .          # the zscale change, through the Tcl interface
```

`tests/run_list_local.sh <list>` runs a list of configs on a workstation with
the same rules as the SLURM wrapper.

## What has and has not been run

Run for real, on alanine dipeptide with NAMD 3 on a workstation: CV values
from these scripts against cfacv's own, center harvesting, the mean-force
runs, the map, the path search, the string warm-ups and the committor shots.

Run for real on picotte: the string driver with `SMPARAMS(zscale)` and with
per-replica restraint files, on alanine dipeptide (jobs 26768091-26768093).

NOT shown: the free-energy map did not pass its own gate on short alanine
dipeptide data (left-out force error 0.65 with 118 centers of 8 ps; limit
0.5). NOT run: anything on T4 lysozyme, and the SLURM wrapper on picotte.

## Limits worth remembering

- The map is only as wide as the region the centers cover. `path_search.py`
  keeps strings within `2 sigma` of the nearest center and reports how far
  each path strays from the data.
- Barriers read from the map rank channels. The reported barrier comes from
  the refined string and `fe_profile.py`.
- The zero-temperature strings on the map use the scaled Euclidean metric,
  not the metric tensor. The MD string of Stage 5 uses the metric tensor.
- `t4l_cvs.json` uses the paper's residue numbers and takes chi1 as
  N-CA-CB-CG. Check both against the built PSF.
