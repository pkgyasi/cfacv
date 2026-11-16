# thalweg string-method driver for NAMD replicas.
#
# A separate driver, not a change to cfacv's own tcl/stringmethod.tcl.  It
# uses cfacv exactly as that driver does on each replica (the same tclforces
# script, the same restraint files, the same accumulated mean force and metric
# tensor) and replaces what the master does with the result:
#
#   - torsions may run past +/-pi: the string is kept unwrapped and each
#     replica is restrained to the wrapped point;
#   - the mean force is a running average, the step size is measured in a
#     calibration period instead of guessed, and the run stops by itself when
#     the string has come to rest (no movement over the last half of the run
#     beyond its own wander) with no force left across it;
#   - images are respaced along a spline through them.
#
# The arithmetic is in string_math.tcl, which is tested without NAMD.
#
# A config file sets, before sourcing this file, the same variables cfacv's
# driver uses:
#     TEMP steps_per_run num_runs runs_per_frame frames_per_restart
#     outputEnergiesEvery namd_config_file init_restr_file initial_root
#     output_root
# and, in the array TW, any of (defaults in brackets):
#     TW(zscale)       one length scale per CV [1 for each, with a warning]
#     TW(periodic)     0/1 per CV [1 for each DIHED in cvINP]
#     TW(dual)         1 = two systems per image, metric tensor from the partner [0]
#     TW(polyak)       1 = the images move on each update's own force and the
#                      AVERAGED string is the result (Polyak-Ruppert); 0 = they
#                      move on the averaged force, which overshoots [1]
#     TW(memory)       updates the running averages of the metric tensor and of
#                      the force remember [200].  With polyak on the force
#                      average only sets the step at calibration
#     TW(calibrate)    updates with the string held still, to set the step [200]
#     TW(max_move)     signal limit on the step: largest across-string move
#                      per update right after calibration, scaled units [0.002]
#     TW(noise_frac)   noise limit on the step: the force noise alone may move
#                      an image this fraction of conv_tol in one window of the
#                      drift test [0.5].  The step is the smaller of the two.
#     TW(stepsize)     set the step by hand instead (0 = measure it) [0]
#     TW(spline)       1 = spline respacing, 0 = straight segments [1]
#     TW(evolve_ends)  1 = end images move too [1]
#     TW(pin_first), TW(pin_last)   a point in CV space for the first / last
#                      image, usually the basin centre from endpoint_analysis.py.
#                      That end then ignores the force: it walks to the pin at
#                      TW(pin_speed) scaled units per update [0.002] and stays.
#                      Without this an end image wanders inside its basin and
#                      leaves a bend in the string.  [not set]
#     TW(hook_cos)     an image next to an end is dropped, and the string
#                      respaced, when the turn there has a cosine below this
#                      [0 = sharper than 90 degrees]; -2 turns it off
#     TW(conv_window)  frames per averaging window of the drift test [20]
#     TW(conv_tol)     drift, in scaled units, below which a frame counts as still [0.1]
#     TW(flow)         residual = images move against the part of grad F that
#                      a minimum free-energy path does not balance [residual];
#                      metric = images move along -M grad F, the string method
#                      as published.  Both stop at the same strings.  With
#                      metric the step has to suit the directions in which M
#                      is largest, and the others lag: on alanine dipeptide
#                      such runs came to rest with 0.3 to 0.5 kcal/mol per
#                      scaled unit left across the string, residual runs with
#                      0.13 to 0.3.
#     TW(plateau_hold) the stopping tests must keep passing while the history
#                      grows by this fraction [0.25]
#     TW(step_decay)   0 = constant step [0]; a > 0 = once the string has
#                      passed the plateau test the step shrinks as
#                      (update at rest / update)^a.  Use 0.5 < a <= 1.
#     TW(hist_max)     most entries kept of the history the plateau and force
#                      tests read [512]; beyond it neighbours are merged
#     TW(curve_reach)  images either side within which a curve distance looks
#                      for the nearest point [4]
#     TW(plateau)      1 = the drift test is the PLATEAU test: over the last
#                      half of the run (four blocks) no image has moved by
#                      TW(plateau_tol) scaled units [0.2] or more beyond the
#                      string's own wander between neighbouring blocks.  It
#                      needs 8 x conv_window frames after calibration.  0 =
#                      the older test on two neighbouring windows against
#                      conv_tol, which cannot tell a creeping string from a
#                      resting one [1].  conv_tol still sets the step.  The
#                      statistic is recomputed every TW(plateau_every) frames [5].
#     TW(drift_curve)  1 = the neighbouring-window drift (reported, and the
#                      test when plateau is 0) is the distance between the two windows'
#                      strings as CURVES, so images that only slide along the
#                      path (they all do when an end moves) are not counted;
#                      0 = the largest displacement of any one image [1].
#                      Both are written to the history (drift, image_drift).
#     TW(force_tol)    and the force left across the string over the last
#                      window of frames, rms over the images and with its own
#                      noise taken out, must be below this, in kcal/mol per
#                      scaled unit [0.3, half of kT at 300 K]; 0 turns it off
#     TW(profile_tol)  and the free-energy profile from one window's forces and
#                      the next's must not differ by more than this (kcal/mol)
#                      at any image, beyond its own noise [0.5]; 0 turns it off
#     TW(conv_count)   consecutive frames passing all the tests that end the run [3]
#     (TW(conv_force), TW(force_res), TW(memory_max) and TW(max_step_factor),
#      from earlier force tests, are accepted and ignored.)
#     TW(fine) [12] points per segment of the spline; TW(nblocks) [10] blocks for
#     the noise at calibration; TW(frame_updates) updates per recorded frame
#     [runs_per_frame]; TW(debug) 1 traces the first updates on stderr.
# Optional: restart_file (a state file from an earlier leg), first_timestep.
#     TW(min_runs)     never stop before this many updates [0]
# num_runs is the cap: the run ends there if the drift test has not ended it.
#
# The history file has the same "SM iter N replica i : reparam z ..." lines
# as cfacv's driver, so string_monitor.py reads either.

foreach v {TEMP steps_per_run num_runs runs_per_frame frames_per_restart outputEnergiesEvery
           namd_config_file init_restr_file initial_root output_root} {
    if {![info exists $v]} { puts stderr "TW) ERROR: the config must set $v"; exit 1 }
}
if {![info exists CFACV_BASEDIR]} {
    if {[info exists env(CFACV_BASEDIR)]} { set CFACV_BASEDIR $env(CFACV_BASEDIR) } else { puts stderr "TW) ERROR: set CFACV_BASEDIR"; exit 1 }
}
source ${CFACV_BASEDIR}/tcl/cfacv.tcl
source [file join [file dirname [info script]] string_math.tcl]

replicaBarrier
set nr [numReplicas]
# NOT a short name: cfacv's tclforces script is read into this same
# interpreter at the first `run` and uses r, i, key, nR, ... as globals.  A
# replica index kept in $r is silently replaced by a restraint specification
# there, after which no replica believes it is the master and all of them
# wait for each other.
set tw_me [myReplica]
set replica_id $tw_me

array set TWDEF {dual 0 min_runs 0 debug 0}
foreach k [array names TWDEF] { if {![info exists TW($k)]} { set TW($k) $TWDEF($k) } }
set nimg [expr {$TW(dual) ? $nr / 2 : $nr}]
if {$TW(dual) && $nr % 2} { puts stderr "TW) ERROR: dual mode needs an even number of replicas, got $nr"; exit 1 }
puts "TW) replica $tw_me of $nr; $nimg images[expr {$TW(dual) ? {, dual mode} : {}}]"

# restart: the job config sets restart_file, written by an earlier leg
if {[info exists restart_file]} {
    source $restart_file
    puts "TW) resuming from $restart_file at update $i_run"
} else {
    set i_job 0; set i_run 0; set i_step 0
    if {[info exists first_timestep]} { set i_step $first_timestep }
}
set job_output_root "$output_root.job$i_job"
set job_initial_root "$initial_root.job$i_job"

if {$i_run} {
    bincoordinates [format "$restart_root.$replica_id.coor" $replica_id]
    binvelocities  [format "$restart_root.$replica_id.vel" $replica_id]
    extendedSystem [format "$restart_root.$replica_id.xsc" $replica_id]
} else {
    bincoordinates $job_initial_root.$replica_id.coor
    if {[file exists $job_initial_root.$replica_id.vel]} { binvelocities $job_initial_root.$replica_id.vel } else { temperature $TEMP }
    extendedSystem $job_initial_root.$replica_id.xsc
}
firsttimestep $i_step
langevinTemp $TEMP
outputname [format $job_output_root.$replica_id $replica_id]
outputEnergies $outputEnergiesEvery
dcdFreq [expr {$steps_per_run * $runs_per_frame}]

source $namd_config_file

proc tw_fail {msg} { puts stderr "TW) ERROR: $msg"; flush stderr; exit 1 }
# TW(debug) 1 traces the first few updates on stderr, which is not buffered,
# so a stall can be located (NAMD's own log is, and loses its last lines).
proc tw_dbg {msg} { global TW tw_me i_run; if {$TW(debug) && $i_run < 3} { puts stderr "TW-DBG replica $tw_me update $i_run: $msg"; flush stderr } }

# ---- the master builds the string -------------------------------------------
set stop 0
if {$tw_me == 0} {
    set history_file [open [format "$job_output_root.$replica_id.history" $replica_id] "w"]
    fconfigure $history_file -buffering line
    # one line per image per recorded frame: update, image, updates averaged,
    # the image's position, and the mean gradient dF/dz there over the frame.
    # fe_profile.py driver turns the frames after the string stopped into a
    # free-energy profile, at no extra MD.
    set force_file [open [format "$job_output_root.$replica_id.forces" $replica_id] "w"]
    fconfigure $force_file -buffering line
    puts $force_file "# update image n_updates z(1..D) dF/dz(1..D)"
    if {[info exists tw_state]} {
        eval $tw_state
        if {$tw::N != $nimg} { tw_fail "the restart holds $tw::N images, this run has $nimg" }
        # the step was set in an earlier leg; do not announce a calibration again
        if {$tw::h > 0} { set step_reported 1 }
    } else {
        set z0 {}
        for {set i 0} {$i < $nimg} {incr i} {
            set fn [format $init_restr_file $i]
            if {[catch {open $fn r} fh]} { tw_fail "cannot read $fn" }
            set p {}
            foreach line [split [read -nonewline $fh] \n] {
                foreach elem $line {
                    if {[llength $elem] > 1 && ([lindex $elem 0] eq "b" || [lindex $elem 0] eq "z")} { lappend p [lindex $elem 1] }
                }
            }
            close $fh
            lappend z0 $p
        }
        set nz [llength [lindex $z0 0]]
        if {![info exists TW(periodic)]} {
            if {![info exists cvINP]} { tw_fail "set TW(periodic), or set cvINP in $namd_config_file so it can be read from there" }
            set TW(periodic) {}
            set fh [open $cvINP r]
            foreach line [split [read $fh] \n] {
                set line [string trim $line]
                if {$line eq "" || [string index $line 0] eq "#"} continue
                lappend TW(periodic) [expr {[lindex $line 0] eq "DIHED"}]
            }
            close $fh
        }
        if {![info exists TW(zscale)]} {
            set TW(zscale) [lrepeat $nz 1.0]
            puts "TW) WARNING: TW(zscale) not set; every CV gets length scale 1, which is wrong for CVs of different units"
        }
        if {[catch {tw::init $z0 $TW(zscale) $TW(periodic)} err]} { tw_fail $err }
    }
    # one recorded frame is runs_per_frame updates; the noise limit on the step needs to know
    if {![info exists TW(frame_updates)]} { set TW(frame_updates) $runs_per_frame }
    set kept_memory $tw::P(memory)
    foreach k [array names TW] {
        if {$k in {zscale periodic dual min_runs debug}} continue
        if {[catch {tw::set_param $k $TW($k)} err]} { tw_fail $err }
    }
    # on a restart, a memory lengthened at calibration outranks the config's
    if {[info exists tw_state] && $kept_memory > $tw::P(memory)} { tw::set_param memory $kept_memory }
    # a pin of the wrong length is refused now, not after the whole calibration
    foreach k {pin_first pin_last} {
        set n [llength $tw::P($k)]
        if {$n && $n != $tw::D} { tw_fail "TW($k) has $n values, the string has $tw::D CVs" }
    }
    puts "TW) $tw::N images in $tw::D CVs; length scales $tw::scale; periodic $tw::periodic"
    puts "TW) memory $tw::P(memory), calibrate $tw::P(calibrate), max_move $tw::P(max_move), spline $tw::P(spline), stopping tests: [expr {$tw::P(plateau) ? "plateau < $tw::P(plateau_tol) and force < $tw::P(force_tol) over the last half of the run" : "drift over $tw::P(conv_window) frames < $tw::P(conv_tol)"}], x $tw::P(conv_count)"
    set tg [tw::targets]
    for {set i 1} {$i < $nr} {incr i} { replicaSend [list 0 [lindex $tg [expr {$i % $nimg}]]] $i }
    set z [lindex $tg 0]
} else {
    set msg [replicaRecv 0]
    set z [lindex $msg 1]
}
set new_z 1
puts "TW) replica $tw_me starts at z = $z"

# ---- main loop ----------------------------------------------------------------
while {$i_run < $num_runs && !$stop} {
    tw_dbg "run"
    run $steps_per_run
    tw_dbg "run returned"

    # this replica's mean force dF/dz and metric tensor since the last update
    DataSpace_Tally $ds
    set nzr [llength $z]
    set g [ArrayToList [DataSpace_g $ds] $nzr]
    set M [ArrayToList [DataSpace_MM $ds] [expr {$nzr * $nzr}]]
    if {$tw_me == 0} {
        set rep_g(0) $g; set rep_M(0) $M
        for {set i 1} {$i < $nr} {incr i} { lassign [replicaRecv $i] rep_g($i) rep_M($i) }
    } else {
        replicaSend [list $g $M] 0
    }
    tw_dbg "forces exchanged"
    replicaBarrier
    tw_dbg "barrier 1 passed"

    set next_run [expr {$i_run + 1}]
    if {$tw_me == 0} {
        set gl {}; set Ml {}
        for {set i 0} {$i < $nimg} {incr i} {
            lappend gl $rep_g($i)
            lappend Ml $rep_M([expr {$TW(dual) ? $i + $nimg : $i}])
        }
        if {[catch { tw::accumulate $gl $Ml; set info [tw::advance] } err]} { tw_fail $err }
        if {[dict get $info moved] && ![info exists step_reported]} {
            set step_reported 1
            set which [expr {$tw::P(stepsize) > 0 ? "set by hand" : ($tw::h_noise < $tw::h_signal ? "noise limit" : "signal limit")}]
            set fac [dict get $info factor]
            if {$fac > 1} {
                puts "TW) the force is too noisy for the longest memory: MD steps per update $steps_per_run -> [expr {$steps_per_run * $fac}]"
                puts $history_file "TW iter $next_run : steps per update $steps_per_run -> [expr {$steps_per_run * $fac}]"
            }
            puts "TW) calibration over at update $next_run: step $tw::h ($which; signal limit $tw::h_signal, noise limit $tw::h_noise, noise per update $tw::noise1); averaging memory $tw::P(memory) updates"
            puts $history_file "TW iter $next_run : calibrated step $tw::h ($which) signal_limit $tw::h_signal noise_limit $tw::h_noise noise_per_update $tw::noise1 memory $tw::P(memory) flow $tw::P(flow) noise_ratio_longer_blocks $tw::noise_ratio"
            if {$tw::noise_ratio > 1.3} {
                puts "TW) WARNING: blocks twice as long give [format %.2f $tw::noise_ratio] times the force noise: the force stays correlated across a calibration block, the noise is underestimated and the step may be too large. Lengthen TW(calibrate)."
                puts $history_file "TW iter $next_run : WARNING calibration blocks shorter than the force's correlation time (ratio $tw::noise_ratio)"
            }
        }
        if {$next_run % $runs_per_frame == 0} {
            # this frame's mean gradients, and the mean positions they were measured at
            lassign [tw::frame_force] fn fg
            set fz [tw::frame_z]
            if {$fn > 0} {
                for {set i 0} {$i < $nimg} {incr i} { puts $force_file "$next_run $i $fn [lindex $fz $i] [lindex $fg $i]" }
            }
            set drift [tw::record_frame $fz $fg]
            # the string that is reported is the frame's average, not the momentary one
            for {set i 0} {$i < $nimg} {incr i} { puts $history_file "SM iter $next_run replica $i : reparam z [lindex $fz $i]" }
            puts $history_file "TW iter $next_run : step [dict get $info h] across-string move [dict get $info perp] force/noise $tw::force_ratio drift $drift image_drift $tw::d_image systematic_drift $tw::d_sys wander $tw::d_floor residual_force $tw::f_res force_noise $tw::f_noise residual_less_noise $tw::f_clean profile_change $tw::p_change long_residual $tw::fl_res long_noise $tw::fl_noise long_residual_less_noise $tw::fl_clean"
            if {[tw::converged] && $next_run >= $TW(min_runs)} {
                set stop 1
                puts "TW) the string has stopped: movement over the last half of the run beyond its own wander $tw::d_sys (wander $tw::d_floor), force across it over that span $tw::fl_res kcal/mol per scaled unit (noise $tw::fl_noise; $tw::fl_clean with the noise taken out), profile change $tw::p_change kcal/mol; ending at update $next_run"
                puts $history_file "TW iter $next_run : converged, systematic_drift $tw::d_sys residual_force $tw::f_res profile_change $tw::p_change"
            }
        }
        set tg [tw::targets]
        set fac [dict get $info factor]
        for {set i 1} {$i < $nr} {incr i} { replicaSend [list $stop [lindex $tg [expr {$i % $nimg}]] $fac] $i }
        set z [lindex $tg 0]
    } else {
        lassign [replicaRecv 0] stop z fac
    }
    # count this update's MD steps before the number per update changes
    incr i_step $steps_per_run
    if {$fac > 1} { set steps_per_run [expr {$steps_per_run * $fac}] }
    tw_dbg "new target $z"
    set new_z 1
    Tcl_DataSpace_resetAccumulators $ds
    replicaBarrier
    tw_dbg "barrier 2 passed"

    set i_run $next_run

    # restart files: every few frames, at the cap, and when the drift test ends the run
    if {$i_run % ($runs_per_frame * $frames_per_restart) == 0 || $i_run == $num_runs || $stop} {
        set restart_root "$job_output_root.restart$i_run"
        output [format $restart_root.$replica_id $replica_id]
        replicaBarrier
        if {$tw_me == 0} {
            set fh [open [format "$restart_root.state.tcl" 0] w]
            puts $fh [list set i_job [expr {$i_job + 1}]]
            puts $fh [list set i_run $i_run]
            puts $fh [list set i_step $i_step]
            puts $fh [list set steps_per_run $steps_per_run]
            puts $fh [list set restart_root $restart_root]
            puts $fh [list set tw_state [tw::dump_state]]
            close $fh
        }
        replicaBarrier
        if {[info exists old_restart_root]} {
            set old [format $old_restart_root $replica_id]
            foreach e {coor vel xsc} { file delete $old.$replica_id.$e }
            if {$tw_me == 0} { file delete [format "$old_restart_root.state.tcl" 0] }
        }
        set old_restart_root $restart_root
    }
}

replicaBarrier
if {$tw_me == 0} {
    # the result is the string averaged over the last window of frames
    set zfin [tw::final_string]
    for {set i 0} {$i < $nimg} {incr i} { puts $history_file "SM FINALiter $i_run replica $i : reparam z [lindex $zfin $i]" }
    # where the force left across the string sits: one number per image
    set fi {}; set over {}
    foreach e $tw::f_image { lassign $e img v; lappend fi "$img:[format %.2f $v]"; if {$tw::P(force_tol) > 0 && $v > $tw::P(force_tol)} { lappend over $img } }
    puts $history_file "TW iter $i_run : residual_by_image [join $fi { }]"
    puts $history_file "TW iter $i_run : at the end, systematic_drift $tw::d_sys (tolerance $tw::P(plateau_tol)) wander $tw::d_floor long_residual_less_noise $tw::fl_clean (tolerance $tw::P(force_tol))"
    if {!$stop && $tw::P(plateau) && $tw::d_sys >= 0 && $tw::d_sys < $tw::P(plateau_tol) && $tw::fl_clean > $tw::P(force_tol)} {
        puts "TW) the string has come to rest (it moved $tw::d_sys over the last half of the run beyond its wander of $tw::d_floor) but a force of $tw::fl_clean kcal/mol per scaled unit is left across it.  Running longer will not remove it."
        puts $history_file "TW iter $i_run : at rest with a force left across the string"
    }
    if {[llength $over]} {
        puts "TW) images with a force across the string above $tw::P(force_tol) at the end: [join $over {, }].  If the string is at rest, running longer will not remove it; on alanine dipeptide such places were bends of the path and more images removed only part of it. Check the profile's gates before deciding whether it matters."
        puts $history_file "TW iter $i_run : above_force_tol [join $over { }]"
    }
    puts $history_file "TW iter $i_run : finished, [expr {$stop ? {stopped by its plateau and force tests} : {reached the update cap}}]"
    close $history_file
    close $force_file
    puts "TW) finished at update $i_run: [expr {$stop ? {stopped by its plateau and force tests} : {reached the update cap without passing its stopping tests}}]"
}
replicaBarrier
