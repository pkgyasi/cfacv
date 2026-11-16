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
| 2 | `endpoint_analysis.py` | CV statistics of the two end states, the joint separation test, the periodic-image check; writes `<out>_scales.json`, `<out>_basins.json` and `<out>_cvstats.json` (default prefix `endpoints`) |
| all | `thalweg_run.py` | runs the stages of one settings file in order, keeps a record in `.thalweg/` of the run directory, and stops at the first failed gate; run it again and it carries on. A gate step counts as passed only when it exits 0 AND prints a `GATE ...: PASS` line and no FAIL. A step is redone when its command, a declared input (size and time) or its run list changes, or when a declared output (or a tool's `--out`) is missing; a changed step redoes everything after it; a NAMD run is redone when its config, or a `#PRE` stage before it, is newer than its log; `--only` reruns one stage and clears every later one; one runner per directory; an interrupt stops what it started. `examples/alad_cvsel.json` is a complete settings file |
| 3 | `cv_select.py`, `cvsel_shot.tcl` | chooses the CVs from the system's own data: candidate enumeration, targeted-MD and shot setup, a basin model over the whole candidate pool, and likelihood maximization with a BIC stop. Each CV after the first must also pass its own conditional permutation test (the chosen CVs fixed, the others permuted between blocks), so noise does not ride along behind a real CV. The block is the source trajectory of a configuration (recorded by `shots`, carried by `outcomes`): it is the sample size, the cross-validation fold and the permutation unit; without block ids clustered configurations look independent, and `select` warns. A basin is a core of its state (the features that tell the states apart are capped at 2.5 spreads, and entering takes two consecutive tests); with the basin model built first each shot stops when it enters one, and a tenth of them run on as an audit of that |
| 3 | `harvest_centers.py` | centers from TAMD trajectories at a minimum SCALED spacing; keeps each center's configuration |
| 3, 6 | `meanforce.py` | sets up restrained MD at a list of points; collects block-averaged mean forces; `metric` measures the metric tensor at each point from its configuration (no MD) |
| 3 | `fesmap.py` | single-sweep free-energy map in scaled, periodic-aware CV space; sigma by cross-validation |
| 4 | `path_search.py` | lowest-barrier routes on the map: graph search, alternatives, zero-temperature strings between the two basins' minima on the map, and random-start strings, whose routes join the table when the graph search could not return them. With `--forces` it refits the map on subsets of the centers, resampling the force noise (from `forces.dat.se`, or else the map's left-out residual), and says how firm the ranking is; a refit whose string jumps channel or does not converge counts against the ranking, not for it. Two routes are one channel when every blend of them stays on the data and within 0.25 kcal/mol of both tops. A string with an image held back by the support guard is reported as not converged. On the alanine map: 7 channels, the lowest (2.71 kcal/mol) firmly below three of them and only likely below the other three (81 to 94% of refits). `--central-finish` adds a central-tangent finishing pass (opt-in: it converged on 5 of 8 channels and brought none closer to MD). A string through a side minimum is reported as such. When the map carries a metric tensor (`fesmap.py fit --metric-tensor`, opt-in) strings relax to where M grad F lies along them, the condition that defines a minimum free-energy path. They are moved against the part of grad F the path does not balance, with M^-1 as preconditioner and upwind tangents, after a first relaxation without the tensor; the tensor is a smooth average over nearby centers. On alanine dipeptide these strings lie 0.17 to 0.28 scaled units from the MD strings of three routes, the tensor-free ones 0.37 to 0.63; barriers and their order are the same, since both kinds of path cross the same saddles |
| 4 | `init_string.py` | image anchors -> the files the cfacv string driver starts from; warm-ups; the initiation gate |
| 5 | `string_monitor.py` | convergence of a string from its history file (for a thalweg-driver history the gate needs the driver's own stop; a run that reached its cap fails; fewer than two windows of frames is refused); merged-or-distinct for two strings; warns when the string doubles back on itself |
| 6 | `fe_profile.py` | profile along a string. Errors from block profiles (each block of every run integrated into a whole profile), with Student-t limits; gates on overlap (fixed bins in robust-spread units), stability (halves and trend for the barrier and the end-to-end difference, and a trend test on every image's force), resolution (every other image, the difference against its own error), block length (correlation from block averaging) and equal runs. `--refine` adds images where the force curves. `driver` gives a first profile from the forces the thalweg driver recorded, only when the last half of the run is at rest (creep and wander gates, in every mode including `--last`) and at least 20,000 updates per image went in; legs must be given in order. `reference` gates a profile against unbiased trajectories. `compare` is the path-independence check |
| 7 | `committor.py`, `committor_shot.tcl` | unbiased shots that stop on basin entry; the transition-state gate: at least 100 committed shots, at least 5 configurations of 10 holding at least 75% of them, configurations alike (chi-square), and an equivalence test: the 95% interval of the pooled committor, with an error that includes the spread between configurations, inside 0.5 +/- `--equiv` (0.1), with the uncommitted shots bracketed both ways inside the same window. With 200 shots a true 0.5 passes only about 57% of the time; plan on 1,000 |
| 5 | `driver/` | a string driver of its own: wrapped torsions, a measured step, images that move on each update's force with the AVERAGED string as the result, free ends that slide to their basins or pinned ends, a stopping test (movement over the last half beyond the noise, wander, force left across the string as rms and per image, profile change), spline respacing, restart. See `driver/README.md`. cfacv's driver is unchanged and still usable |
| 1 | `make_cfacv_inputs.py` | writes cfacv's label PDB and CV file from the CV definition JSON |
| all | `sbatch/namd_array.sbatch` | one NAMD run per array task from a list file; kills a run that has written nothing for `HANG_MINUTES` |
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
make_cfacv_inputs.py                      -> label.pdb, cv.inp           (from the CV JSON)
cv_select.py candidates ; classifier ; tmd ; shots ; outcomes ; select   (if the CVs are to be chosen)
(run TAMD with cfacv)
harvest_centers.py                        -> centers/
meanforce.py setup ; array ; collect      -> forces.dat   (; metric -> metric.json, optional)
fesmap.py fit                             -> map.json
path_search.py --forces forces.dat        -> channels_anchors_N.dat
init_string.py setup ; array ; check      -> stringN/  (+ string_stage_snippet.namd)
(run the string: thalweg/driver, or cfacv's own driver)
string_monitor.py status / compare        -> stringN_final.dat  (+ stringN_driver_final.dat)
fe_profile.py driver                      -> a first profile, free
meanforce.py setup --points stringN_final.dat ; array
fe_profile.py compute / compare           -> profile
fe_profile.py reference                   -> the profile against unbiased trajectories, where there are any
committor.py pick ; setup ; array ; tally
```

`thalweg_run.py settings.json` runs any stretch of this from one settings
file and stops at the first gate that fails.

## Python on picotte

The system Python on picotte has numpy and scipy but not MDAnalysis, which
the scripts that read structures and trajectories need. An environment with
all of it lives on group storage:

    source /ifs/groups/abramsGrp/pg565/envs/thalweg_env.sh

It loads the Python 3.11 module, activates the environment, and sets
`THALWEG` and `CFACV_BASEDIR`. In a batch script that runs `module purge`,
source it after the purge. A purge also unloads the SLURM module, so do not
call `sbatch` from a shell where you have purged.

## Tests

From the repository root:

```
python3 -m pytest thalweg/tests                     # no MD needed, under a minute
tclsh thalweg/tests/string_math_test.tcl            # the driver's arithmetic, under plain tclsh
namd3 thalweg/tests/string_math_in_namd.namd        # the same arithmetic inside NAMD's Tcl
tclsh tests/reparam_zscale.tcl .                    # the zscale change, through cfacv's Tcl interface
```

`thalweg/tests/run_list_local.sh <list>` runs a list of configs on a
workstation with the rules of the SLURM wrapper (skip a run whose log ends
cleanly and is newer than its config; `#PRE` chains; blank and `#` lines
ignored), without its hang guard.

## What has and has not been run

Validated end to end on alanine dipeptide in four torsions (CHARMM22, TIP3P,
NAMD 3), on a workstation and on picotte: end-state analysis, center
harvesting, mean forces at 483 centers, the map (nested left-out force error 0.35 in path units,
with a Gaussian width per CV; the 0.23 stored before the third review was in the wrong units), the path search, string initiation, string
refinement with cfacv's driver and with the thalweg driver, profiles from
dedicated restrained runs, the committor shots, CV selection through
`thalweg_run.py`, and the profile from the driver's own forces.
Under the stopping rule of 2026-11-15 (before the third review) the driver
stopped five pinned-end alanine runs by its own tests. That rule was fooled
by out-and-back motion and by slow creep, and was replaced. Under the
present rule no alanine run has stopped itself: three 120,000-update runs
(lower channel with free and with pinned ends, second channel with free
ends) all reached their cap, the closest at 0.202 against 0.2. The whole workflow has also been run once on
the same molecule in mixed units (two torsions in radians and a distance
between two-atom groups in angstrom): see the paper's validation section.

T4 lysozyme: a smoke test only. A closed system (2LZM) and an open one (150L
chain D) were built; a 12-image string between the closed basin and a point
part way to open was initiated and run with the thalweg driver for 21,000
updates without reaching its stopping test. Nothing on T4 lysozyme is a
result.

Not run: the SLURM array wrapper as a job array on picotte (the runner's
sbatch mode); pinned
ends on a full closed-to-open string; a helper restraint that is not a
string coordinate.

Three rounds of independent code review (2026-11-15) found and removed
defects that had let gates pass for the wrong reason; the tests now include
a case for each, written to fail on the code before the fix. A gate that has never been seen to fail on bad data has not been
tested: see `tests/` for which have.

## Limits worth remembering

- The map is only as wide as the region the centers cover. `path_search.py`
  keeps strings from moving farther than two center spacings from the data
  and reports how far each path strays. On sparse random centers that guard
  marks most random-start strings not converged; `--support` widens it.
- Refitting a map with the corrected width selection can choose widths that
  pass the gate but give worse paths: on the alanine map the refit's strings
  lie 0.7 to 1.8 scaled units from the MD strings, against 0.17 to 0.28 for
  the stored widths. Compare a refitted map's strings against MD before
  adopting it.
- Barriers read from the map rank channels; their error bar is mostly how
  much the barrier depends on which centers the map has. The reported
  barrier comes from the refined string and `fe_profile.py`.
- The profile is the free energy of the CVs integrated along the path. It
  is not a free energy that includes the width of the transition tube.
- What restrained runs measure is, exactly, the gradient of the free energy
  smoothed by a Gaussian of variance kT/k in each CV (Maragliano, Roux and
  Vanden-Eijnden 2014, eq. 25). `fe_profile.py` prints that width against
  each CV's length scale. `fe_profile.py reference` computes the same
  smoothed free energy at a profile's images from unbiased trajectories
  and gates the profile against it; it has no radius or bin to choose.
  Counting frames within a radius of each image, which an earlier
  validation here used, is biased low: on a 4-D test density with a
  barrier of 3.30 it returned 2.80 at a radius of 0.6 scaled units.
- The mean force is integrated along a cubic-spline path parametrised by
  arc length in the scaled units, with the force interpolated by local cubic
  Hermite pieces and integrated exactly by Gauss-Legendre; it is linear in
  the forces. It amplifies force noise little (1.03 to 1.19 times the
  trapezoid rule's) and does not overshoot where the force changes sign, as
  a global spline did. On the driver's own 16-image profiles a global
  spline was still closer to the reference (rms 0.15 against 0.20).
- A first-look profile from the driver's own forces is not a result. On
  16-image alanine strings it came out 0.3 to 0.5 kcal/mol below the
  unbiased reference, at single images up to 0.6 (1.4 in one run), with
  error bars of 0.02 to 0.1 that do not cover that; the reference gate now
  fails them. The profile of record is the dedicated restrained runs.
- The path search's relaxation counts a string as converged when it moves
  less than its tolerance per iteration, or when it has settled into a
  small chatter whose average no longer moves; it then returns the
  average.
- The driver's step is constant, so the string itself does not converge:
  it fluctuates about the path, and the AVERAGED string is the result, with
  a bias of the order of the step.
- The driver's stopping tests read the last half of the run (at least
  `TW(plateau_span)` updates): no movement between any two of eight blocks
  beyond the noise, small wander, and no force left across the string, rms
  or at any one image. With the residual flow the force left on alanine
  strings is 0.05 to 0.25 kcal/mol per scaled unit. The images next to
  pinned ends swing 0.2 to 0.45 scaled units over a few thousand updates,
  and the test reads that as movement, so in practice the runs reach their
  cap. The profile tool's own stationarity check on the same runs finds a
  creep of about 0.12; the two disagree, which is open. See
  `driver/README.md`.
- Shots for CV selection need configurations BETWEEN the basins. Evenly
  spaced targeted-MD frames mostly start inside one (alanine dipeptide: 38
  usable configurations of 180, 6 with shots going both ways). Frames from
  the images around the barrier tops of MD strings do far better (`shots
  --dcd` takes any trajectories: 183 usable of 210, 82 going both ways), but
  they come from runs restrained in the string's CVs, so they vary little in
  those CVs across the path and cannot show that one of them matters.
  Exploration frames that the basin model puts in neither basin (`shots
  --between`) are unrestrained and need no string, but on alanine dipeptide
  44% of their shots never reached either basin in 50 ps (they sit in other
  basins) and 12% of the audited ones left the basin they first entered, so
  that selection fails its gate. Either way the committor test is what
  checks the outcome.
- The map's barriers are too low where it has no centers on the route:
  2.71 and 3.92 kcal/mol for two alanine routes against 3.3 and 4.4 to 4.6
  from MD. `path_search.py --route-points` writes the route images that
  have no center nearby, `meanforce.py setup --centers` starts runs at them
  from the nearest stored configuration, and a refit with those 131 points
  gave 3.26 and 4.47. The strings' positions did not move closer to the MD
  strings.
- Candidate CVs name the segment as well as the residue number when the
  selection holds more than one segment, and a group's centre of mass is
  taken after its atoms are brought to one periodic image. Neither has been
  run on a real multi-chain system.
- `t4l_cvs.json` uses the paper's residue numbers and takes chi1 as
  N-CA-CB-CG. The CVs come from a steered-MD and TAMD study in another force
  field and engine; that they are adequate string CVs here is a hypothesis.
