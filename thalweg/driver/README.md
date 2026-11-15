# thalweg string driver

A string-method driver for NAMD replicas that sits beside cfacv's own
`tcl/stringmethod.tcl` and does not change it. Each replica still runs cfacv's
`cfacv_tclforces_stringmethod.tcl`, with the same label PDB, CV file and
restraint files. What differs is what the master does with the mean forces.

Written 2026-11-14.

| | cfacv `stringmethod.tcl` | this driver |
|---|---|---|
| torsions | never wrapped; a string must not cross +/-pi | string kept unwrapped, replicas restrained to the wrapped point |
| mean force used for an update | the last `steps_per_run` MD steps | a running average over `TW(memory)` updates |
| step size | `SMPARAMS(stepsize)`, chosen by hand | measured: the string stands still for `TW(calibrate)` updates, then the step is set so the fastest image moves `TW(max_move)` scaled units per update |
| when it stops | after `num_runs` updates | when the string has stopped drifting, or at `num_runs` |
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

The history file is `<output_root>.job<N>.0.history` on replica 0, with the
same `SM iter ... reparam z ...` lines as cfacv's driver, so
`string_monitor.py status` reads it. Lines starting `TW iter` carry the step,
the largest across-string move and the drift.

## Choosing the settings

- `TW(max_move)`: how fast the string moves right after calibration. 0.002
  scaled units per update means a string one scaled unit from its path needs
  on the order of a thousand updates.
- `TW(memory)`: longer is smoother and slower to respond. The force used is
  an average over `memory x steps_per_run` MD steps.
- `TW(conv_window)`, `TW(conv_tol)`, `TW(conv_count)`: the run ends when the
  window-averaged string has moved less than `conv_tol` scaled units between
  consecutive windows, `conv_count` frames in a row. This is the same test
  `string_monitor.py` applies afterwards.

## Limits

- The step is set once and then constant. If the calibration period is too
  short the measured forces are noisy and the step can be too large.
- A spline through noisy images can wiggle; `TW(spline) 0` falls back to
  straight segments.
- An error on the master ends that replica with `exit 1`; the others end when
  MPI notices. NAMD does not exit by itself after a Tcl error in the cfacv
  script.
