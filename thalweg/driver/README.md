# thalweg string driver

A string-method driver for NAMD replicas that sits beside cfacv's own
`tcl/stringmethod.tcl` and does not change it. Each replica still runs cfacv's
`cfacv_tclforces_stringmethod.tcl`, with the same label PDB, CV file and
restraint files. What differs is what the master does with the mean forces.

Written 2026-11-14.

| | cfacv `stringmethod.tcl` | this driver |
|---|---|---|
| torsions | never wrapped; a string must not cross +/-pi | string kept unwrapped, replicas restrained to the wrapped point |
| how an image moves | along -M g, g from the last `steps_per_run` MD steps | by default against the part of g that a minimum free-energy path does not balance (`TW(flow) residual`), g from the same few MD steps and the metric tensor averaged over `TW(memory)` updates; `TW(flow) metric` moves along -M g |
| the result | the string at the last update | the AVERAGE of the string over the last window of frames (iterate averaging): the momentary string fluctuates about the path and never settles, because the step is constant |
| step size | `SMPARAMS(stepsize)`, chosen by hand | measured: the string stands still for `TW(calibrate)` updates, then the step is the smaller of two limits: the fastest image moves `TW(max_move)` scaled units per update, or the noise alone moves an image less than half the drift tolerance over one test window |
| when it stops | after `num_runs` updates | when, over the last half of the run, the averaged string has not moved beyond its own wander and no force is left across it, and the profile along it is not demonstrably changing; or at `num_runs` |
| end images | move with the force | move with the force, or are pinned (`TW(pin_first)`, `TW(pin_last)`); an image next to an end that doubles back is dropped |
| respacing | straight segments between images | a cubic spline through the images |
| two systems per image | needed for independent force and metric estimates | optional (`TW(dual)`); the running averages make it unnecessary |

## Files

- `string_math.tcl` the arithmetic, pure Tcl; tested by `tests/string_math_test.tcl`
  under plain tclsh with a mock replica.
- `string_driver.tcl` the NAMD side: replicas, restarts, the history file.
- `job0.conf`, `job1.conf`, `string.conf` templates for a first leg, a
  continuation and the settings.

## Using it

Prepare the string with `init_string.py` as before. In the string settings
file use the variables listed at the top of `string_driver.tcl`; in the NAMD
base file keep

    tclforcesscript  $CFACV_BASEDIR/tcl/cfacv_tclforces_stringmethod.tcl
    set restrINP     [format "<dir>/restr.job0.%s.inp" [myReplica]]

Do not set `langevinTemp`, `outputEnergies`, `dcdFreq`, input or output names
in the base file: the driver sets them, and NAMD refuses a parameter given
twice.

The history file is `<output_root>.job<N>.0.history` on replica 0, with
`SM iter ... reparam z ...` lines in cfacv's format, so `string_monitor.py
status` reads it (do not pass `--dual`; a run that stopped itself has few
frames, and the monitor then shortens its window). Each frame's string is
the MEAN over that frame's updates, unwrapped (a torsion may lie beyond
pi). The `SM FINALiter` block at the end is the result: the string averaged
over the last window and respaced. Lines starting `TW iter` carry the step,
the largest across-string move, the short-window drift (as a curve, and per image), the plateau statistic and the wander, the residual force, its noise,
the residual with the noise taken out, and the profile change.

Next to it, `<output_root>.job<N>.0.forces` holds one line per image per
recorded frame: the update, the image, the number of updates averaged, the
image's position and the mean gradient dF/dz there over that frame (a plain
average, not the weighted one the string moves with). `fe_profile.py driver`
integrates the frames from after the string stopped into a free-energy
profile with error bars, at no extra MD. It shares its sampling with the
refinement, so it is a first look, not the profile of record.

## Choosing the settings

- `TW(max_move)`: how fast the string moves right after calibration. 0.002
  scaled units per update means a string one scaled unit from its path needs
  on the order of a thousand updates.
- `TW(memory)`: longer is smoother and slower to respond. The force used is
  an average over `memory x steps_per_run` MD steps.
- **When the run ends.** Three tests, all passing on `TW(conv_count)`
  evaluations in a row and holding (below), and only once the last half of
  the history since calibration covers at least `TW(plateau_span)` updates
  (10,000) and 8 x `TW(conv_window)` frames. A stop happens only on a frame
  on which the tests were evaluated (every `TW(plateau_every)` frames).
  1. *Plateau.* The last half of the frames since calibration is cut into
     eight blocks. For each image, the largest distance (as a curve) between
     the mean strings of ANY two blocks is taken, less the distance the noise
     alone puts between two block means, which is measured inside the blocks
     (from the two halves of each). The largest of these over the images
     must be below `TW(plateau_tol)` (0.2 scaled units), and so must the
     noise term itself (reported as `wander`, worst image).
  2. *Force.* Over the same span, the force left across the string, rms over
     the images that follow the force and with its noise (from four blocks)
     taken out, is below `TW(force_tol)` (0.3 kcal/mol per scaled unit, half
     of kT at 300 K); no single image's is above `TW(force_image_tol)`
     (default -1 = three times `force_tol`, 0.9; 0 turns the per-image cap
     off); and the force must have been measured (a run that recorded no
     forces does not pass, which it used to). At a free end the whole
     gradient counts, not only its part across the string.
  3. *Profile.* The free-energy profile from one window's forces and from
     the next's is not demonstrably changing by more than `TW(profile_tol)`
     (0.5 kcal/mol).

  **What the plateau test can and cannot see.** Creep slower than about
  `plateau_tol / plateau_span` per update (2e-5 scaled units with the
  defaults) passes by construction; that is the choice `plateau_span`
  makes. With a residual-flow step h a force F drives the string at h F per
  update, so on alanine dipeptide (h about 3e-4) the default lets through
  only what a force of about 0.07 kcal/mol per scaled unit would cause, a
  quarter of `force_tol`. Faster creep, out-and-back excursions, slow
  oscillations, random walks and an end creeping across or along the path
  all fail it. A string whose block means scatter by about `plateau_tol` or
  more cannot be told from one that moves that much and does not stop until
  its blocks are long enough to average the scatter down. Measured on a
  12-image test string fed frames directly (`frame_updates 20`, so the
  default span is 500 frames; AR(1) wander of 0.067 rms per image; 10 seeds;
  1500 frames, 2500 for the resting rows):

  | frames fed (per frame) | before: stopped | after: stopped |
  |---|---|---|
  | resting, wander 0.067 rms | 10/10 at 199 | 10/10 at 1255 |
  | resting, wander 0.17 rms | 10/10 at 218-549 | 4/10 by 2500 (2023-2423) |
  | resting, wander 0.67 rms | 0/10 | 0/10 |
  | all creeping 0.0001 / 0.0002 | 10/10 at 199 | 10/10 at 1255-1375 (accepted: under 2e-5 per update) |
  | all creeping 0.0003 / 0.0004 | 10/10 at 199 | 0/10 |
  | all creeping 0.0005 / 0.001 / 0.0015 | 10/10 at 199-281 | 0/10 |
  | all creeping 0.002 / 0.003 | 1/10, 0/10 | 0/10 |
  | all oscillating, amplitude 0.5, period 107 / 400 | 10/10 / 10/10 | 0/10 / 0/10 |
  | all oscillating, amplitude 1.0, period 53 | 10/10 | 0/10 |
  | far half oscillating, amplitude 0.3, period 120 / 300 | 10/10 / 10/10 | 0/10 / 0/10 |
  | far half random walk, 0.012 a frame | 10/10 | 0/10 |
  | end image creeping across, 0.002 | 10/10 | 0/10 |
  | end image creeping along, 0.003 | 10/10 | 0/10 |

  The previous test (first block against last, less neighbouring blocks,
  from 8 x `conv_window` frames on) is the "before" column. It stopped
  nearly everything at the earliest frame it could. The plateau test is
  costlier (36 curve comparisons per evaluation); it runs every
  `plateau_every` frames.

  **What this means for the alanine runs.** Replayed through the new test,
  the five alanine strings that stopped themselves (pr0D, pr0D2, pr0D3,
  pr1D1, pr1D2) would not have stopped by the plateau test within their
  lengths: over the last half of each, the images next to the pinned ends
  (and one or two more) made out-and-back excursions of 0.2 to 0.45 scaled
  units between blocks of 2,300 to 3,400 updates, while the noise inside a
  block was about 0.1. The old statistic cancelled these against the
  neighbouring-block term. Whether those excursions matter for the profile
  is a separate question; the stopping rule no longer hides them.

  **Which flow.** With `TW(flow) metric` each image moves along -M g, the
  update as published. The step then has to suit the directions in which M
  is largest, and on alanine dipeptide M's eigenvalues differ about
  250-fold, so the others lag: strings came to rest by the plateau test
  with 0.3 to 0.5 kcal/mol per scaled unit still across them, the same in
  both halves of an 80,000-update run, and the dedicated 500 ps runs at an
  older string's images showed the same (0.53). It is not the restraint
  (doubling its force constant left 0.49) and not the image count (16, 24
  and 36 images left 0.53, 0.41 and 0.39). With `TW(flow) residual`, the
  default, each image moves against g minus its component along
  n = M^-1 t / |M^-1 t|: the same strings are fixed points, and every
  direction moves on the same footing. The step came out ten times larger
  and the force left fell to 0.13 on the lower channel and 0.2 to 0.3 on
  the second. A lower-channel string started from the path search reached
  its resting place in about 8,000 updates and stopped itself at 10,200;
  with the metric flow the same string took 34,200.

  **Holding.** The tests must keep passing while the history grows by
  `TW(plateau_hold)` (a quarter). Evaluations a few frames apart read
  nearly the same half of the run; a string that was still creeping passed
  three of them in a row at its first evaluations and failed a few thousand
  updates later.

  **A shrinking step.** `TW(step_decay) a` (0 = off) makes the step fall as
  (update at rest / update)^a once the plateau and force tests have passed
  AND held, never before: a string with a vanishing step looks at rest
  wherever it is. If the plateau test fails later, the rest point is cleared
  and the step goes back to its calibrated value. Since a run normally stops
  as soon as the tests hold, the decay only acts when something keeps the
  run going (`TW(min_runs)`, the profile test, a large `TW(conv_count)`).
  (The alanine figures once quoted here were from the earlier rule, which
  started the decay on a single plateau pass.)

  **Calibration.** The force noise is measured from blocks of the
  calibration period and again from blocks twice as long; the larger is
  used and the ratio is written to the history. It is the noise of the
  WORST image (it was the rms over the images, so one noisy image among
  many barely counted), and at a free end both the across and along
  directions count. Above about 1.3 the blocks
  are shorter than the force stays correlated: lengthen `TW(calibrate)`.

  **History.** The plateau and force tests read frames since calibration,
  kept as means over chunks so that there are never more than
  `TW(hist_max)` of them (at least 64); restart files stay small however
  long the run. Merging keeps the newest frames.
  Each history line carries both sets of numbers: `drift`, `image_drift`,
  `residual_force`, `force_noise`, `residual_less_noise` over the short
  window, and `systematic_drift`, `wander`, `long_residual`, `long_noise`,
  `long_residual_less_noise`, `long_residual_worst_image` over the long
  span. The `converged` line carries the long-span numbers the stop used.
  `string_monitor.py` applies a two-window drift test with its own defaults
  (window 100, tolerance 0.25). An end that is pinned or held is left out of
  the force test; in the drift tests it counts only while it is still
  walking to its pin. `TW(conv_tol)` still sets the step (see above).
- `TW(polyak)`: 1 by default. Each update the images move on that update's
  own force with the measured small step (the on-the-fly string update), and
  the string that is recorded, tested and written at the end is the average
  over a frame or a window (iterate averaging, after Polyak and Juditsky).
  The step is constant, so this does not meet the Robbins-Monro conditions:
  the momentary string does not converge, the averaged one does, up to a
  bias of the order of the step. Stepping on an exponentially averaged force
  instead, `TW(polyak) 0` and what this driver did at first, is exactly
  Polyak's heavy-ball iteration with momentum 1 - 1/memory: on a straight
  valley the string coasted past the floor by as much as it started above
  it. It is kept for comparison and should not be used.

- `TW(pin_first)`, `TW(pin_last)`: a point in CV space for an end image
  (the basin centre from `endpoint_analysis.py`, which is the free-energy
  minimum only as far as the basin is harmonic). That end ignores the force,
  walks to the pin at `TW(pin_speed)` and stays; it is left out of the step
  and of the stopping tests. Both free and fixed ends are in published use.
- **Free ends** (no pin, `TW(evolve_ends) 1`). Under `TW(flow) metric` an
  end moves along -M g like every image. Under `TW(flow) residual` an end
  moves against its whole gradient in scaled units (steepest descent, -g s^2
  in raw units): the residual vanishes on any minimum free-energy path, so
  ends moved by it never slid into their basins (ends started half way up
  the walls of a double well stayed there). Scaled steepest descent has the
  same units as the residual move, so one measured step suits both; -M g
  would not. A free end's whole move, along the path as well as across it,
  counts in the step size and the noise, and its whole gradient in the force
  test.
- **Restarts.** The state file records the parameters the step was set
  under (`flow`, `calibrate`, `nblocks`, `frame_updates`, `max_move`,
  `noise_frac`, `conv_tol`, `conv_window`, `polyak`, `stepsize`). A restart
  that asks for a different `flow`, `calibrate` or `frame_updates` is
  refused (a step calibrated under one flow moved strings up to 250 times
  too far under the other); one that changes the others is reported in the
  log and history, and the step stays as calibrated. A config that names no
  flow continues with the state's. A state file from before `TW(flow)`
  existed ran the metric flow and is treated so. `TW(frame_updates)`, if
  set, must equal `runs_per_frame`. Whole-number settings (`calibrate`,
  `conv_window`, `curve_reach`, ...) are checked when set. The restart files
  a leg started from are deleted once it has written its own.
- `TW(hook_cos)`: the image NEXT TO an end that makes the string double back
  (cosine of the turn below this value, default 0) is dropped before the
  images are respaced, so an end that has slid down into its basin does not
  leave a hook behind. Set it to -2 to switch this off (-1 is not enough: an exact reversal can round to just below -1).

## Limits

- The step is set once and then constant (unless `TW(step_decay)`). If the calibration period is too
  short the measured forces are noisy and the step can be too large.
- A spline through noisy images can wiggle; `TW(spline) 0` falls back to
  straight segments.
- An error on the master ends that replica with `exit 1`; the others end when
  MPI notices. NAMD does not exit by itself after a Tcl error in the cfacv
  script.
