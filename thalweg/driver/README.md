# thalweg string driver

A string-method driver for NAMD replicas that sits beside cfacv's own
`tcl/stringmethod.tcl` and does not change it. Each replica still runs cfacv's
`cfacv_tclforces_stringmethod.tcl`, with the same label PDB, CV file and
restraint files. What differs is what the master does with the mean forces.

Written 2026-11-14.

| | cfacv `stringmethod.tcl` | this driver |
|---|---|---|
| torsions | never wrapped; a string must not cross +/-pi | string kept unwrapped, replicas restrained to the wrapped point |
| mean force used for an update | the last `steps_per_run` MD steps | the same (this is the on-the-fly string update), with a metric tensor averaged over `TW(memory)` updates |
| the result | the string at the last update | the AVERAGE of the string over the last window of frames (iterate averaging): the momentary string fluctuates about the path and never settles, because the step is constant |
| step size | `SMPARAMS(stepsize)`, chosen by hand | measured: the string stands still for `TW(calibrate)` updates, then the step is the smaller of two limits: the fastest image moves `TW(max_move)` scaled units per update, or the noise alone moves an image less than half the drift tolerance over one test window |
| when it stops | after `num_runs` updates | when the averaged string has stopped drifting, no force across it is demonstrably left, and the profile along it is not demonstrably changing; or at `num_runs` |
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
the largest across-string move, the drift, the residual force, its noise,
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
- `TW(conv_window)`, `TW(conv_tol)`, `TW(conv_count)`: the run ends when the
  window-averaged string has moved less than `conv_tol` scaled units between
  consecutive windows, `conv_count` frames in a row, AND the force left
  across the string over the last window, with the noise in that number
  taken out, is below `TW(force_tol)` (0.3 kcal/mol per scaled unit, half of
  kT at 300 K), AND the free-energy profile computed from one window's
  forces and from the next's is not demonstrably changing by more than
  `TW(profile_tol)` (0.5 kcal/mol), AND both windows lie after the
  calibration period.
  The drift test is the one the string-method literature uses (a plateau in
  the string's displacement); it is given a number here. The force and
  profile tests are additions. The force test does NOT wait until the force
  is resolved below its tolerance: for a solvated protein that needs
  nanoseconds per image, as much as a free-energy profile costs. It holds
  the run while a force is demonstrably there. Each history line gives the
  force, its noise and the force with the noise taken out, so how well it
  was resolved is on record. A longer `TW(conv_window)` resolves it better.
  `string_monitor.py` applies only the drift test, with its own defaults
  (window 100, tolerance 0.25), so the two can disagree: the driver's is the
  stricter. An end that is pinned or held is left out of all of them.
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
- `TW(hook_cos)`: the image NEXT TO an end that makes the string double back
  (cosine of the turn below this value, default 0) is dropped before the
  images are respaced, so an end that has slid down into its basin does not
  leave a hook behind. Set it to -2 to switch this off (-1 is not enough: an exact reversal can round to just below -1).

## Limits

- The step is set once and then constant. If the calibration period is too
  short the measured forces are noisy and the step can be too large.
- A spline through noisy images can wiggle; `TW(spline) 0` falls back to
  straight segments.
- An error on the master ends that replica with `exit 1`; the others end when
  MPI notices. NAMD does not exit by itself after a Tcl error in the cfacv
  script.
