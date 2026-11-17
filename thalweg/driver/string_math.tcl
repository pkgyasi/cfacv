# String arithmetic for the thalweg driver.  Pure Tcl: no NAMD and no
# cfacv commands, so it runs under plain tclsh and can be tested there.
#
# It keeps the string on the master and does three things that cfacv's own
# stringmethod.tcl does not:
#
#   1. TORSIONS THAT WRAP.  The string is held unwrapped, as a continuous
#      curve that may run past +/-pi.  Arc length and the update use it as it
#      is.  What a replica is restrained to is the same point wrapped into
#      (-pi, pi], which cfacv's PERIODIC restraint handles for any value.
#
#   2. AVERAGED POSITIONS AND A MEASURED STEP.  Each update the images move
#      on the force of that update alone, with a small step; what is
#      reported, tested for convergence and written at the end is the
#      AVERAGE of the string over many updates (Polyak-Ruppert averaging,
#      the standard remedy in stochastic approximation).  Averaging the
#      FORCE and stepping on that average, which this driver did first, is
#      the heavy-ball method: the average carries inertia, the string coasts
#      on a force it has already left behind, overshoots and rings.  The
#      running average of the force is still kept, but only to measure how
#      much force is left across the string.  TW(polyak) 0 restores the old
#      behaviour.  The step size is not guessed: the
#      string is held still for a calibration period, in which both the SIZE
#      of the force across the string and its NOISE are measured.  The step
#      is the smaller of two limits:
#        signal  the fastest image moves max_move per update;
#        noise   the noise alone, acting as a random walk for one window of
#                the drift test, moves an image less than noise_frac x
#                conv_tol.
#      The first set of runs used the signal limit only.  It chose a step
#      three times too large for the noise, the string jittered, and the
#      drift test could never pass.  After calibration the step is constant,
#      so the string slows by itself as the forces across it vanish, and the
#      drift test says when it has stopped.
#
#   3. SPLINE RESPACING.  Images are respaced along a cubic spline through
#      them, not along straight segments, which cuts corners when a bend has
#      few images.
#
# Units: every distance is measured with each CV divided by its length scale.

namespace eval tw {
    variable P
    array set P {
        memory      200
        calibrate   200
        max_move    0.002
        stepsize    0
        spline      1
        fine        12
        evolve_ends 1
        conv_window 20
        conv_tol    0.1
        conv_count  3
        conv_force  1.5
        force_res   0.15
        polyak      1
        force_tol   0.3
        profile_tol 0.5
        memory_max  10000
        max_step_factor 10
        hook_cos    0.0
        pin_first   {}
        pin_last    {}
        pin_speed   0.002
        noise_frac  0.5
        drift_curve 1
        plateau     1
        plateau_tol 0.2
        plateau_every 5
        hist_max    512
        curve_reach 4
        flow        residual
        plateau_hold 0.25
        step_decay  0.0
        frame_updates 100
        nblocks     10
        plateau_span 10000
        force_image_tol -1
        rest_tol    0.05
        rest_move_max 1.0
    }
    # what the restart file says the step was calibrated under (tw::dump_state,
    # tw::restart_check); empty in a state file older than this record
    variable saved_P {}
    variable blk {}
    variable h_signal 0.0
    variable h_noise 0.0
    variable noise1 0.0
    variable force_ratio -1.0
    # this update's gradients (the string moves on these when polyak is on)
    variable G_now {}
    # sum of the string's positions over the updates since the last frame
    variable Z_sum {}
    variable Z_n 0
    # one free-energy profile per recorded frame, from that frame's own forces
    variable prof {}
    # the frame-mean gradients of the recorded frames (parallel to frames)
    variable gframes {}
    # residual force across the string (rms over the images that follow the
    # force) and the noise left in it, both in kcal/mol per scaled unit
    variable f_res -1.0
    variable f_noise -1.0
    variable f_clean -1.0
    variable p_change -1.0
    # the two measures of drift between the last two windows: each image's
    # own displacement, and the distance between the two strings as curves
    variable d_image -1.0
    variable d_curve -1.0
    # the plateau test: every frame-mean string recorded after calibration,
    # the largest movement of any image over the last half of them that the
    # string's own wander does not account for, and that wander (rms over
    # the images, between neighbouring blocks)
    variable zhist {}
    variable ghist {}
    # the history is kept in chunks of hist_chunk frames (their means), so
    # that it stays below hist_max entries however long the run; a chunk in
    # the making is in hz_sum / hg_sum (sums over hz_n frames)
    variable hist_chunk 1
    variable hz_sum {}
    variable hg_sum {}
    variable hz_n 0
    variable h_since 0
    # the step right after calibration and the update at which the string
    # first passed the plateau test (for step_decay); -1 = not yet
    variable h0 0.0
    variable n_rest -1
    # calibration: the noise from blocks twice as long, over the noise from
    # the blocks as they are (above about 1.3 the blocks are too short)
    variable noise_ratio -1.0
    # frames in the history when the present run of passing evaluations began
    variable k_pass -1
    # consecutive evaluations of the long-span tests that passed
    variable nplat 0
    # the force across the string over the last half of the run: rms over
    # the images, its noise (from the four blocks), and with the noise out
    variable fl_res -1.0
    # the largest per-image long-span force (tw::long_force)
    variable fl_image -1.0
    variable fl_noise -1.0
    variable fl_clean -1.0
    # the rest test (tw::rest): cost of the string's movement and of its
    # position's uncertainty (kcal/mol, worst image), the movement bound
    # (scaled units) and the worst image
    variable d_sys -1.0
    variable d_floor -1.0
    variable d_move -1.0
    variable d_worst -1
    # the long-span profile test (tw::long_profile): lower bound on the change, kcal/mol
    variable p_long -1.0
    # residual force at each image that follows the force, over the last
    # window and with its noise taken out: a list of {image value}
    variable f_image {}
    # plain (not exponentially weighted) sum of the gradients since the last
    # recorded frame: independent samples of dF/dz for the free-energy profile
    variable F_sum {}
    variable F_n 0
    variable step_factor 1
    variable N 0
    variable D 0
    variable scale {}
    variable periodic {}
    variable z {}
    variable S_g {}
    variable S_M {}
    variable W 0.0
    variable h 0.0
    variable nacc 0
    variable frames {}
    variable nconv 0
    variable PI 3.14159265358979323846
}

# Parameters that index lists or count frames must be whole numbers: a value
# like 200.5 used to pass here and crash lindex hundreds of updates later.
proc tw::set_param {name value} {
    variable P
    if {![info exists P($name)]} { error "thalweg: unknown parameter '$name' (known: [lsort [array names P]])" }
    if {$name in {calibrate nblocks conv_window conv_count frame_updates curve_reach fine hist_max plateau_every
                  evolve_ends spline polyak plateau drift_curve}} {
        if {![string is integer -strict $value] || $value < 0} { error "thalweg: TW($name) must be a whole number >= 0, got '$value'" }
    } elseif {$name eq "flow"} {
        if {$value ni {metric residual}} { error "thalweg: TW(flow) must be metric or residual, got '$value'" }
    } elseif {$name ni {pin_first pin_last}} {
        if {![string is double -strict $value]} { error "thalweg: TW($name) must be a number, got '$value'" }
    }
    if {$name eq "plateau_tol"} {
        puts "thalweg: TW(plateau_tol) is no longer used (the plateau test was replaced on 2026-11-16 by the rest test, TW(rest_tol) in kcal/mol); ignored"
    }
    set P($name) $value
}

# A recorded frame is runs_per_frame updates; the noise limit on the step and
# the plateau test's span read TW(frame_updates).  A different value made both
# silently wrong, so it is refused.
proc tw::check_frame_updates {runs_per_frame} {
    variable P
    if {$P(frame_updates) != $runs_per_frame} {
        error "thalweg: TW(frame_updates) is $P(frame_updates) but runs_per_frame is $runs_per_frame; a recorded frame is runs_per_frame updates, so leave TW(frame_updates) unset"
    }
}

proc tw::wrap {x} {
    variable PI
    set y [expr {fmod($x + $PI, 2.0 * $PI)}]
    if {$y <= 0.0} { set y [expr {$y + 2.0 * $PI}] }
    return [expr {$y - $PI}]
}

# z0: list of N points, each a list of D values.  Torsion values may arrive
# wrapped; the string is made continuous here, each image nearest the last.
proc tw::init {z0 scale periodic} {
    variable N; variable D; variable z; variable S_g; variable S_M; variable W; variable h
    variable nacc; variable frames; variable nconv; variable PI
    variable F_sum {}; variable F_n 0
    variable G_now {}; variable Z_sum {}; variable Z_n 0; variable prof {}
    variable f_res -1.0; variable f_noise -1.0; variable p_change -1.0; variable force_ratio -1.0
    variable gframes {}; variable f_clean -1.0; variable saved_P {}; variable fl_image -1.0
    variable zhist {}; variable d_sys -1.0; variable d_floor -1.0; variable d_move -1.0; variable d_worst -1; variable p_long -1.0; variable f_image {}
    variable ghist {}; variable nplat 0; variable fl_res -1.0; variable fl_noise -1.0; variable fl_clean -1.0
    variable hist_chunk 1; variable hz_sum {}; variable hg_sum {}; variable hz_n 0; variable h_since 0
    variable h0 0.0; variable n_rest -1; variable noise_ratio -1.0; variable k_pass -1
    variable d_image -1.0; variable d_curve -1.0
    set N [llength $z0]
    set D [llength [lindex $z0 0]]
    if {$N < 3} { error "thalweg: a string needs at least 3 images, got $N" }
    if {[llength $scale] != $D || [llength $periodic] != $D} {
        error "thalweg: need $D length scales and $D periodic flags, got [llength $scale] and [llength $periodic]"
    }
    foreach s $scale { if {![string is double -strict $s] || $s <= 0} { error "thalweg: length scales must be positive, got '$s'" } }
    foreach p $z0 { if {[llength $p] != $D} { error "thalweg: images have different numbers of CVs" } }
    set tw::scale $scale
    set tw::periodic $periodic
    set z [list [lindex $z0 0]]
    for {set i 1} {$i < $N} {incr i} {
        set prev [lindex $z end]; set p {}
        for {set d 0} {$d < $D} {incr d} {
            set v [lindex $z0 $i $d]
            if {[lindex $periodic $d]} {
                set pv [lindex $prev $d]
                set v [expr {$pv + [tw::wrap [expr {$v - $pv}]]}]
            }
            lappend p $v
        }
        lappend z $p
    }
    set S_g {}; set S_M {}
    for {set i 0} {$i < $N} {incr i} {
        lappend S_g [lrepeat $D 0.0]
        lappend S_M [lrepeat [expr {$D * $D}] 0.0]
    }
    set W 0.0; set h 0.0; set nacc 0; set frames {}; set nconv 0
    set tw::blk {}; set tw::h_signal 0.0; set tw::h_noise 0.0; set tw::noise1 0.0
}

# What each replica is restrained to: the image, with torsions wrapped.
proc tw::targets {} {
    variable z; variable periodic
    set out {}
    foreach p $z {
        set q {}
        foreach v $p per $periodic { lappend q [expr {$per ? [tw::wrap $v] : $v}] }
        lappend out $q
    }
    return $out
}

# Fold one update's measurements into the running averages.
# glist: N gradients dF/dz.  Mlist: N metric tensors, D*D values each.
# The average is S/W with S <- (1-a) S + a x and W <- (1-a) W + a, which is
# an ordinary mean while few samples exist and forgets with memory 1/a later.
proc tw::accumulate {glist Mlist} {
    variable N; variable D; variable S_g; variable S_M; variable W; variable P; variable nacc
    if {[llength $glist] != $N || [llength $Mlist] != $N} { error "thalweg: expected $N gradients and metric tensors" }
    set a [expr {1.0 / $P(memory)}]
    # during calibration every sample counts equally, and the gradients are
    # also summed in blocks, so their noise can be measured afterwards
    if {$nacc < $P(calibrate)} {
        set a [expr {1.0 / ($nacc + 1)}]
        variable blk
        # NOT min() or max(): inside NAMD's Tcl those are not expr functions.
        # NAMD's handler for unknown commands takes them for config
        # parameters and they return an empty string or an error.
        set B [expr {$P(calibrate) / $P(nblocks)}]
        if {$B < 1} { set B 1 }
        set b [expr {$nacc / $B}]
        if {$b > $P(nblocks) - 1} { set b [expr {$P(nblocks) - 1}] }
        while {[llength $blk] <= $b} { lappend blk [list 0 {}] }
        lassign [lindex $blk $b] cnt sums
        if {$cnt == 0} {
            set sums $glist
        } else {
            set ns {}
            foreach s $sums g $glist { set t {}; foreach u $s v $g { lappend t [expr {$u + $v}] }; lappend ns $t }
            set sums $ns
        }
        lset blk $b [list [expr {$cnt + 1}] $sums]
    }
    set ng {}; set nM {}
    foreach sg $S_g g $glist sM $S_M M $Mlist {
        set t {}; foreach s $sg x $g { lappend t [expr {(1.0 - $a) * $s + $a * $x}] }
        lappend ng $t
        set t {}; foreach s $sM x $M { lappend t [expr {(1.0 - $a) * $s + $a * $x}] }
        lappend nM $t
    }
    set S_g $ng; set S_M $nM
    variable G_now; variable Z_sum; variable Z_n; variable z
    set G_now $glist
    # the positions these forces were measured at: the string before this update's move
    if {$Z_n == 0} {
        set Z_sum $z
    } else {
        set ns {}
        foreach s $Z_sum p $z { set t {}; foreach u $s v $p { lappend t [expr {$u + $v}] }; lappend ns $t }
        set Z_sum $ns
    }
    incr Z_n
    variable F_sum; variable F_n
    if {$F_n == 0} {
        set F_sum $glist
    } else {
        set ns {}
        foreach s $F_sum g $glist { set t {}; foreach u $s v $g { lappend t [expr {$u + $v}] }; lappend ns $t }
        set F_sum $ns
    }
    incr F_n
    set W [expr {$nacc < $P(calibrate) ? 1.0 : (1.0 - $a) * $W + $a}]
    incr nacc
}

# The move each image would make with step 1: -M.g, from the averages.
# The move of every image, per unit step and in raw CV units, for a list of
# gradients (one per image).
#   flow metric    -M g, the string method as published.
#   flow residual  minus the part of g that a minimum free-energy path does
#                  not balance: g less its component along n = M^-1 t / |M^-1 t|
#                  (tw::residual_vec).  The two stop at the same strings.  With
#                  -M g the step has to be small enough for the directions in
#                  which M is large, and the directions in which it is small,
#                  often the ones the path runs along, then barely move: on
#                  alanine dipeptide M's eigenvalues differ about 250-fold.
#                  This form moves every direction on the same footing.
#                  A FREE END image (not pinned, not held) moves against its
#                  whole gradient in scaled units, -g (s^2 in raw units): the
#                  residual is zero on any MFEP, so an end moved by it never
#                  slid down into its basin (ends started half way up the walls
#                  of a double well stayed there).  Steepest descent in the
#                  scaled coordinates has the same units as the residual move,
#                  so the measured step suits both; -M g would not (M's
#                  eigenvalues span 250-fold on alanine).  An end's minimum is
#                  a minimum whatever the metric, so it stops in the same place.
# gscale divides the gradients first (the running sum S_g wants 1/W).
proc tw::flow_moves {glist {gscale 1.0}} {
    variable D; variable S_M; variable W; variable P; variable z; variable scale; variable N
    set out {}
    if {$P(flow) eq "residual"} {
        array set nv {}
        foreach e [tw::normals $z] { set nv([lindex $e 0]) [lindex $e 1] }
        for {set i 0} {$i < $N} {incr i} {
            set g {}; foreach v [lindex $glist $i] { lappend g [expr {$v / $gscale}] }
            if {[tw::held $i] || ![info exists nv($i)]} { lappend out [lrepeat $D 0.0]; continue }
            if {[tw::free_end $i]} {
                set m {}; foreach v $g sc $scale { lappend m [expr {-$v * $sc * $sc}] }
            } else {
                set m {}; foreach r [tw::residual_vec $g $nv($i)] sc $scale { lappend m [expr {-$r * $sc}] }
            }
            lappend out $m
        }
        return $out
    }
    foreach sg $glist sM $S_M {
        set m {}
        for {set i 0} {$i < $D} {incr i} {
            set t 0.0
            for {set j 0} {$j < $D} {incr j} { set t [expr {$t + [lindex $sM [expr {$i * $D + $j}]] * [lindex $sg $j]}] }
            lappend m [expr {-$t / ($W * $gscale)}]
        }
        lappend out $m
    }
    return $out
}
proc tw::raw_move {} {
    variable S_g; variable W
    return [tw::flow_moves $S_g $W]
}

# Unit tangents of the string in scaled coordinates.
proc tw::tangents {zz} {
    variable scale
    set n [llength $zz]; set out {}
    for {set i 0} {$i < $n} {incr i} {
        set a [lindex $zz [expr {$i > 0 ? $i - 1 : 0}]]
        set b [lindex $zz [expr {$i < $n - 1 ? $i + 1 : $n - 1}]]
        set t {}; set nn 0.0
        foreach x $a y $b s $scale { set v [expr {($y - $x) / $s}]; lappend t $v; set nn [expr {$nn + $v * $v}] }
        set nn [expr {sqrt($nn)}]
        if {$nn < 1e-300} { set nn 1.0 }
        set u {}; foreach v $t { lappend u [expr {$v / $nn}] }
        lappend out $u
    }
    return $out
}

# Is image i an end that does not follow the force (pinned, or held by
# evolve_ends 0)?  Such an end keeps whatever force its position has: that
# force can never relax, so it must not enter the step, the memory, the noise
# or the stopping test, or a pin a little off the valley floor keeps the run
# going to its cap.
proc tw::held {i} {
    variable P; variable N
    if {$i == 0} { return [expr {[llength $P(pin_first)] > 0 || !$P(evolve_ends)}] }
    if {$i == $N - 1} { return [expr {[llength $P(pin_last)] > 0 || !$P(evolve_ends)}] }
    return 0
}

proc tw::free_end {i} {
    variable N
    return [expr {($i == 0 || $i == $N - 1) && ![tw::held $i]}]
}

# The part of image img's move m (raw units, per unit step) that is progress,
# in scaled units: for an interior image the part ACROSS the string (movement
# along it is undone by respacing); for a free end all of it, because an end
# that moves along the path lengthens or shortens the string, and respacing
# keeps the ends where they are.
proc tw::counted_move {img m t} {
    variable scale
    set ms {}; set dot 0.0
    foreach v $m s $scale tt $t { set x [expr {$v / $s}]; lappend ms $x; set dot [expr {$dot + $x * $tt}] }
    if {[tw::free_end $img]} { return $ms }
    set q {}; foreach x $ms tt $t { lappend q [expr {$x - $dot * $tt}] }
    return $q
}

# Size of each image's counted move per unit step (tw::counted_move), for the
# images that follow the force (held ends are left out).
proc tw::perp_sizes {moves} {
    variable z
    set out {}; set img -1
    foreach m $moves t [tw::tangents $z] {
        incr img
        if {[tw::held $img]} continue
        set nn 0.0; foreach y [tw::counted_move $img $m $t] { set nn [expr {$nn + $y * $y}] }
        lappend out [expr {sqrt($nn)}]
    }
    return $out
}

proc tw::calibrated {} { variable nacc; variable P; return [expr {$nacc >= $P(calibrate)}] }

# Noise of the across-string move, per update and for step 1, in scaled
# units: the largest over the images.  Measured from the spread of block
# means of the gradient during calibration, so correlation between
# successive updates is included.  Returns 0 when it cannot be measured.
proc tw::noise_per_update {} {
    variable blk; variable z; variable scale; variable D; variable N; variable P
    variable noise_ratio
    set nb 0; foreach b $blk { if {[lindex $b 0] > 0} { incr nb } }
    if {$nb < 3} { return 0.0 }
    # every block's mean gradient -> its move at unit step -> scaled -> across the string
    set tans [tw::tangents $z]
    set pvs {}; set fvs {}; set lens {}
    foreach b $blk {
        lassign $b cnt sums
        if {$cnt == 0} continue
        set moves [tw::flow_moves $sums $cnt]
        set pv {}
        for {set img 0} {$img < $N} {incr img} {
            if {[tw::held $img]} { lappend pv {}; continue }
            lappend pv [tw::counted_move $img [lindex $moves $img] [lindex $tans $img]]
        }
        lappend pvs $pv; lappend lens $cnt
    }
    lassign [tw::block_noise $pvs $lens] n1 n2
    # Blocks shorter than the time over which the force stays correlated make
    # the noise look smaller than it is, and the step then comes out too
    # large.  Blocks twice as long (neighbours merged) give a second
    # estimate; the larger of the two is used, and their ratio is kept.
    set noise_ratio [expr {$n1 > 0 && $n2 > 0 ? $n2 / $n1 : -1.0}]
    return [expr {$n2 > $n1 ? $n2 : $n1}]
}

# Noise per update of a vector quantity given as block means (vecs: per
# block, per image, a vector or {} for a held image; lens: updates per
# block): the LARGEST over the images of sqrt(variance of a block mean x
# block length).  (It was the rms over the images, which let one noisy image
# among 24 count for a fifth of its noise; the step's promise is per image.)
# Returns that, and the same from blocks twice as long (0 when fewer than
# three such blocks exist).
proc tw::block_noise {vecs lens} {
    set out {}
    foreach level {1 2} {
        set V $vecs; set Ln $lens
        if {$level == 2} {
            set V {}; set Ln {}
            for {set b 0} {$b + 1 < [llength $vecs]} {incr b 2} {
                set va [lindex $vecs $b]; set vb [lindex $vecs [expr {$b + 1}]]
                set la [lindex $lens $b]; set lb [lindex $lens [expr {$b + 1}]]
                set merged {}
                foreach x $va y $vb {
                    if {[llength $x] == 0} { lappend merged {}; continue }
                    set q {}; foreach u $x v $y { lappend q [expr {($u * $la + $v * $lb) / double($la + $lb)}] }
                    lappend merged $q
                }
                lappend V $merged; lappend Ln [expr {$la + $lb}]
            }
        }
        set n [llength $V]
        if {$n < 3} { lappend out 0.0; continue }
        set len 0.0; foreach l $Ln { set len [expr {$len + $l}] }
        set worst 0.0
        set nI [llength [lindex $V 0]]
        for {set img 0} {$img < $nI} {incr img} {
            if {[llength [lindex $V 0 $img]] == 0} continue
            set D [llength [lindex $V 0 $img]]; set sq 0.0
            for {set d 0} {$d < $D} {incr d} {
                set m 0.0; foreach v $V { set m [expr {$m + [lindex $v $img $d]}] }
                set m [expr {$m / $n}]
                foreach v $V { set sq [expr {$sq + pow([lindex $v $img $d] - $m, 2)}] }
            }
            set v [expr {$sq / ($n - 1) * ($len / $n)}]
            if {$v > $worst} { set worst $v }
        }
        lappend out [expr {sqrt($worst)}]
    }
    return $out
}

# Natural cubic spline: second derivatives at the knots.
proc tw::spline_m {x y} {
    set n [llength $x]
    if {$n < 3} { return [lrepeat $n 0.0] }
    set u [list 0.0]; set m [list 0.0]
    for {set i 1} {$i < $n - 1} {incr i} {
        set h0 [expr {[lindex $x $i] - [lindex $x [expr {$i - 1}]]}]
        set h1 [expr {[lindex $x [expr {$i + 1}]] - [lindex $x $i]}]
        set sig [expr {$h0 / ($h0 + $h1)}]
        set p [expr {$sig * [lindex $m end] + 2.0}]
        lappend m [expr {($sig - 1.0) / $p}]
        set dd [expr {([lindex $y [expr {$i + 1}]] - [lindex $y $i]) / $h1 - ([lindex $y $i] - [lindex $y [expr {$i - 1}]]) / $h0}]
        lappend u [expr {(6.0 * $dd / ($h0 + $h1) - $sig * [lindex $u end]) / $p}]
    }
    set out [lrepeat $n 0.0]
    for {set k [expr {$n - 2}]} {$k >= 1} {incr k -1} {
        lset out $k [expr {[lindex $m $k] * [lindex $out [expr {$k + 1}]] + [lindex $u $k]}]
    }
    return $out
}

# Cosine of the turn the string makes at image i (between the segment into
# it and the segment out of it), in scaled coordinates.
proc tw::turn_cos {zz i} {
    variable scale
    set a [lindex $zz [expr {$i - 1}]]; set b [lindex $zz $i]; set c [lindex $zz [expr {$i + 1}]]
    set dot 0.0; set n1 0.0; set n2 0.0
    foreach x $a y $b w $c s $scale {
        set u [expr {($y - $x) / $s}]; set v [expr {($w - $y) / $s}]
        set dot [expr {$dot + $u * $v}]; set n1 [expr {$n1 + $u * $u}]; set n2 [expr {$n2 + $v * $v}]
    }
    if {$n1 <= 0 || $n2 <= 0} { return 1.0 }
    return [expr {$dot / sqrt($n1 * $n2)}]
}

# Remove hooks at the ends.  A free end image slides to the bottom of its
# basin, and when that lies behind its neighbour the string doubles back on
# itself there (seen on T4 lysozyme: turns sharper than 90 degrees at both
# ends).  The path then enters the basin, leaves and comes back, which no
# profile should integrate over.  While the turn at the image next to an end
# is sharper than hook_cos, that image is dropped; respacing afterwards
# restores the image count along the shortened path.
proc tw::unhook {zz} {
    variable P
    while {[llength $zz] > 3 && [tw::turn_cos $zz 1] < $P(hook_cos)} { set zz [lreplace $zz 1 1] }
    while {[llength $zz] > 3 && [tw::turn_cos $zz [expr {[llength $zz] - 2}]] < $P(hook_cos)} { set zz [lreplace $zz end-1 end-1] }
    return $zz
}

# Respace at equal scaled arc length into nout images (default: as many as
# given).  The end images do not move.
proc tw::reparam {zz {nout 0}} {
    variable scale; variable P
    set n [llength $zz]; set D [llength [lindex $zz 0]]
    if {$nout <= 0} { set nout $n }
    # chord-length parameter of the images
    set s [list 0.0]
    for {set i 1} {$i < $n} {incr i} {
        set d 0.0
        foreach a [lindex $zz [expr {$i - 1}]] b [lindex $zz $i] sc $scale { set d [expr {$d + pow(($b - $a) / $sc, 2)}] }
        lappend s [expr {[lindex $s end] + sqrt($d) + 1e-12}]
    }
    # a fine polyline along the curve through the images
    set cols {}; set ms {}
    for {set d 0} {$d < $D} {incr d} {
        set y {}; foreach p $zz { lappend y [lindex $p $d] }
        lappend cols $y
        lappend ms [expr {$P(spline) ? [tw::spline_m $s $y] : [lrepeat $n 0.0]}]
    }
    set nf [expr {$P(spline) ? $P(fine) : 1}]
    set fine {}
    for {set k 0} {$k < $n - 1} {incr k} {
        set x0 [lindex $s $k]; set x1 [lindex $s [expr {$k + 1}]]; set hh [expr {$x1 - $x0}]
        for {set j 0} {$j < $nf} {incr j} {
            set B [expr {double($j) / $nf}]; set A [expr {1.0 - $B}]
            set p {}
            for {set d 0} {$d < $D} {incr d} {
                set y [lindex $cols $d]; set m [lindex $ms $d]
                lappend p [expr {$A * [lindex $y $k] + $B * [lindex $y [expr {$k + 1}]]
                    + (($A * $A * $A - $A) * [lindex $m $k] + ($B * $B * $B - $B) * [lindex $m [expr {$k + 1}]]) * $hh * $hh / 6.0}]
            }
            lappend fine $p
        }
    }
    lappend fine [lindex $zz end]
    # arc length along the fine polyline, then equal steps along it
    set L [list 0.0]
    for {set i 1} {$i < [llength $fine]} {incr i} {
        set d 0.0
        foreach a [lindex $fine [expr {$i - 1}]] b [lindex $fine $i] sc $scale { set d [expr {$d + pow(($b - $a) / $sc, 2)}] }
        lappend L [expr {[lindex $L end] + sqrt($d)}]
    }
    set total [lindex $L end]
    set out [list [lindex $zz 0]]; set k 1; set nL [llength $L]
    for {set i 1} {$i < $nout - 1} {incr i} {
        set t [expr {$total * $i / ($nout - 1.0)}]
        while {$k < $nL - 1 && [lindex $L $k] < $t} { incr k }
        set l0 [lindex $L [expr {$k - 1}]]; set l1 [lindex $L $k]
        set f [expr {$l1 > $l0 ? ($t - $l0) / ($l1 - $l0) : 0.0}]
        set p {}
        foreach a [lindex $fine [expr {$k - 1}]] b [lindex $fine $k] { lappend p [expr {$a + $f * ($b - $a)}] }
        lappend out $p
    }
    lappend out [lindex $zz end]
    return $out
}

# One update of the string.  Call after accumulate.  Returns a dict:
#   moved 0|1, h, perp (largest across-string move this update, scaled).
# An end image that is pinned does not follow the force.  It walks to its
# pin in a straight line, at most pin_speed scaled units per update (so the
# replica restrained to it is never jerked), and then stays there.  The pin
# is taken at the periodic image nearest the end, so the string stays
# continuous.  Returns the new position and the scaled distance still to go.
proc tw::toward_pin {p pin} {
    variable scale; variable periodic; variable P
    if {[llength $pin] != [llength $p]} { error "thalweg: a pin has [llength $pin] values, the string has [llength $p] CVs" }
    set d {}; set len 0.0
    foreach v $p t $pin s $scale per $periodic {
        set dv [expr {$t - $v}]
        if {$per} { set dv [tw::wrap $dv] }
        lappend d $dv; set len [expr {$len + ($dv / $s) * ($dv / $s)}]
    }
    set len [expr {sqrt($len)}]
    set f [expr {$len > $P(pin_speed) ? $P(pin_speed) / $len : 1.0}]
    set q {}; foreach v $p dv $d { lappend q [expr {$v + $f * $dv}] }
    return [list $q [expr {$len * (1.0 - $f)}]]
}

proc tw::advance {} {
    variable z; variable h; variable P; variable N
    if {![tw::calibrated]} { return [dict create moved 0 h 0.0 perp 0.0 factor 1] }
    set factor 1
    # the move from the AVERAGED force: needed to set the step, and to move
    # on when polyak is off; not worked out on the updates that do not use it
    set moves {}
    if {$h == 0.0 || !$P(polyak)} { set moves [tw::raw_move] }
    if {$h == 0.0} {
        # set the step once, from forces averaged while the string stood still
        variable h_signal; variable h_noise; variable noise1
        set big 0.0
        foreach v [tw::perp_sizes $moves] { if {$v > $big} { set big $v } }
        if {$big <= 0.0 && $P(stepsize) <= 0} { error "thalweg: the averaged force across the string is exactly zero; cannot set a step (set TW(stepsize) to give one by hand)" }
        set h_signal [expr {$big > 0 ? $P(max_move) / $big : 0.0}]
        set noise1 [tw::noise_per_update]
        # a random walk of the noise alone over one drift-test window
        set h_noise [expr {$noise1 > 0 ? $P(noise_frac) * $P(conv_tol) / ($noise1 * sqrt($P(conv_window) * $P(frame_updates))) : $h_signal}]
        if {$P(stepsize) > 0} { set h $P(stepsize) } else { set h [expr {$h_noise < $h_signal ? $h_noise : $h_signal}] }
        variable h0; set h0 $h
        # what the step was set under, for a restart to check (tw::restart_check)
        variable saved_P; set saved_P [tw::step_params]
        set tw::step_factor 1
        # (An earlier version lengthened the force-averaging memory here, up
        # to 10,000 updates, and multiplied the MD steps per update, so that
        # an average of the force could resolve a fixed tolerance.  That cost
        # nanoseconds per image before a string was allowed to stop, and the
        # long average remembered the force of the starting string.  The
        # stopping test now measures the force over the last window of
        # frames and allows for its own noise; see tw::record_frame.)
    }
    # Once the string is at rest (the rest and force tests have passed and
    # held, tw::record_frame sets n_rest), the step may shrink:
    # h = h0 (n_rest / n)^step_decay.  A constant step leaves the string
    # wandering by an amount set by the step, and its average biased by as
    # much; a step that shrinks as a power between 1/2 and 1 of the update
    # count removes both (Robbins and Monro).  It is NOT shrunk before the
    # string is at rest: a string with a vanishing step looks at rest
    # wherever it is.  If the plateau test later fails, n_rest is cleared and
    # the step goes back to h0.
    variable h0; variable n_rest; variable nacc
    if {$P(step_decay) > 0 && $h0 > 0} {
        if {$n_rest > 0} { set h [expr {$h0 * pow(double($n_rest) / $nacc, $P(step_decay))}] } else { set h $h0 }
    }
    # the images move on this update's own force; the averaged one is kept
    # for the stopping test only
    if {$P(polyak)} { set moves [tw::inst_move] }
    set perp 0.0
    foreach v [tw::perp_sizes $moves] { if {$v * $h > $perp} { set perp [expr {$v * $h}] } }
    set nz {}; set i 0
    foreach p $z m $moves {
        if {$i == 0 && [llength $P(pin_first)]} {
            lappend nz [lindex [tw::toward_pin $p $P(pin_first)] 0]
        } elseif {$i == $N - 1 && [llength $P(pin_last)]} {
            lappend nz [lindex [tw::toward_pin $p $P(pin_last)] 0]
        } elseif {!$P(evolve_ends) && ($i == 0 || $i == $N - 1)} {
            lappend nz $p
        } else {
            set q {}; foreach v $p dv $m { lappend q [expr {$v + $h * $dv}] }
            lappend nz $q
        }
        incr i
    }
    set z [tw::reparam [tw::unhook $nz] [llength $nz]]
    # the step factor is reported once, on the update that ends calibration
    variable step_factor
    set factor $step_factor; set step_factor 1
    return [dict create moved 1 h $h perp $perp factor $factor]
}

# Scaled distance from point p to the polyline through the points of c.
proc tw::point_to_curve {p c {first 0} {last -1}} {
    variable scale
    set best -1.0
    if {$last < 0 || $last > [llength $c] - 1} { set last [expr {[llength $c] - 1}] }
    if {$first < 0} { set first 0 }
    if {$first > $last - 1} { set first [expr {$last - 1}] }
    set a [lindex $c $first]
    foreach b [lrange $c [expr {$first + 1}] $last] {
        set ab2 0.0; set apab 0.0
        foreach u $a v $b x $p sc $scale {
            set e [expr {($v - $u) / $sc}]; set q [expr {($x - $u) / $sc}]
            set ab2 [expr {$ab2 + $e * $e}]; set apab [expr {$apab + $q * $e}]
        }
        set t [expr {$ab2 > 0 ? $apab / $ab2 : 0.0}]
        if {$t < 0.0} { set t 0.0 } elseif {$t > 1.0} { set t 1.0 }
        set d2 0.0
        foreach u $a v $b x $p sc $scale { set d2 [expr {$d2 + pow(($x - $u - $t * ($v - $u)) / $sc, 2)}] }
        if {$best < 0 || $d2 < $best} { set best $d2 }
        set a $b
    }
    return [expr {sqrt($best)}]
}

# How far apart two strings are as curves: the largest scaled distance from
# an image of one to the spline through the other, taken both ways.  Two
# strings that trace the same path with their images in different places are
# at distance zero; one that extends past the other's end is at the distance
# it extends.
proc tw::curve_distance {A B} {
    variable P
    set nf [expr {([llength $A] - 1) * ($P(spline) ? $P(fine) : 1) + 1}]
    set fa [tw::reparam $A $nf]; set fb [tw::reparam $B $nf]
    # an image is looked for within curve_reach images of its own place on
    # the other string, which keeps the cost proportional to the number of
    # images; one that has slid farther than that reads as far away
    set per [expr {($nf - 1) / ([llength $A] - 1)}]; set w [expr {$P(curve_reach) * $per}]
    set worst 0.0; set i 0
    foreach pa $A pb $B {
        set c [expr {$i * $per}]
        set v [tw::point_to_curve $pb $fa [expr {$c - $w}] [expr {$c + $w}]]; if {$v > $worst} { set worst $v }
        set v [tw::point_to_curve $pa $fb [expr {$c - $w}] [expr {$c + $w}]]; if {$v > $worst} { set worst $v }
        incr i
    }
    return $worst
}

# For each image, the mean of two squared scaled distances: from that image
# of B to the curve through A, and from that image of A to the curve through B.
proc tw::image_curve_sq {A B} {
    variable P
    set nf [expr {([llength $A] - 1) * ($P(spline) ? $P(fine) : 1) + 1}]
    set fa [tw::reparam $A $nf]; set fb [tw::reparam $B $nf]
    set per [expr {($nf - 1) / ([llength $A] - 1)}]; set w [expr {$P(curve_reach) * $per}]
    set out {}; set i 0
    foreach pa $A pb $B {
        set c [expr {$i * $per}]
        set u [tw::point_to_curve $pb $fa [expr {$c - $w}] [expr {$c + $w}]]
        set v [tw::point_to_curve $pa $fb [expr {$c - $w}] [expr {$c + $w}]]
        lappend out [expr {($u * $u + $v * $v) / 2.0}]
        incr i
    }
    return $out
}

# Mean string over frames first..last (inclusive) of a list of strings.
proc tw::mean_string {fr first last} {
    set n [llength [lindex $fr $first]]; set D [llength [lindex $fr $first 0]]
    set cnt [expr {$last - $first + 1}]; set out {}
    for {set i 0} {$i < $n} {incr i} {
        set p {}
        for {set d 0} {$d < $D} {incr d} {
            set a 0.0
            for {set f $first} {$f <= $last} {incr f} { set a [expr {$a + [lindex $fr $f $i $d]}] }
            lappend p [expr {$a / $cnt}]
        }
        lappend out $p
    }
    return $out
}

# Add one frame to the history the rest and force tests read.  Frames are
# kept as means over chunks of hist_chunk frames; when there are hist_max
# chunks (at least 64, so that the plateau test's sixteen half-blocks have
# two entries each), neighbours are merged and the chunk length doubles, so
# the history and the restart file that carries it stay bounded however long
# the run.  With an odd number of entries the newest has no partner: it goes
# back into the chunk in the making, as the frames it stands for (it used to
# be dropped).  Merging repeats until the history is under the cap (a
# restart with a smaller hist_max needs more than one pass).
# Returns 1 when a chunk was completed.
proc tw::history_add {zf gf} {
    variable zhist; variable ghist; variable hist_chunk; variable hz_sum; variable hg_sum; variable hz_n; variable h_since
    variable P
    incr h_since
    if {$hz_n == 0} {
        set hz_sum $zf; set hg_sum $gf
    } else {
        set hz_sum [tw::add_strings $hz_sum $zf]
        # a chunk has forces only if every frame in it had them
        set hg_sum [expr {[llength $hg_sum] && [llength $gf] ? [tw::add_strings $hg_sum $gf] : {}}]
    }
    incr hz_n
    if {$hz_n < $hist_chunk} { return 0 }
    lappend zhist [tw::scale_string $hz_sum [expr {1.0 / $hz_n}]]
    lappend ghist [expr {[llength $hg_sum] ? [tw::scale_string $hg_sum [expr {1.0 / $hz_n}]] : {}}]
    set hz_n 0; set hz_sum {}; set hg_sum {}
    set cap [expr {$P(hist_max) < 64 ? 64 : $P(hist_max) - $P(hist_max) % 2}]
    while {[llength $zhist] >= $cap} {
        if {[llength $zhist] % 2} {
            # the newest entry, hist_chunk frames, goes back to the chunk in the making
            set zl [lindex $zhist end]; set gl [lindex $ghist end]
            set zhist [lrange $zhist 0 end-1]; set ghist [lrange $ghist 0 end-1]
            set zs [tw::scale_string $zl $hist_chunk]
            set gs [expr {[llength $gl] ? [tw::scale_string $gl $hist_chunk] : {}}]
            if {$hz_n == 0} {
                set hz_sum $zs; set hg_sum $gs
            } else {
                set hz_sum [tw::add_strings $zs $hz_sum]
                set hg_sum [expr {[llength $gs] && [llength $hg_sum] ? [tw::add_strings $gs $hg_sum] : {}}]
            }
            incr hz_n $hist_chunk
        }
        set nz {}; set ng {}
        for {set i 0} {$i + 1 < [llength $zhist]} {incr i 2} {
            lappend nz [tw::scale_string [tw::add_strings [lindex $zhist $i] [lindex $zhist [expr {$i + 1}]]] 0.5]
            set a [lindex $ghist $i]; set b [lindex $ghist [expr {$i + 1}]]
            lappend ng [expr {[llength $a] && [llength $b] ? [tw::scale_string [tw::add_strings $a $b] 0.5] : {}}]
        }
        set zhist $nz; set ghist $ng; set hist_chunk [expr {$hist_chunk * 2}]
    }
    return 1
}
proc tw::add_strings {a b} {
    set out {}
    foreach p $a q $b { set t {}; foreach u $p v $q { lappend t [expr {$u + $v}] }; lappend out $t }
    return $out
}
proc tw::scale_string {a f} {
    set out {}
    foreach p $a { set t {}; foreach u $p { lappend t [expr {$u * $f}] }; lappend out $t }
    return $out
}

# THE REST TEST (replaced the plateau test on 2026-11-16).  Is the string at
# rest, judged by what its motion is worth in free energy?
#
# Over the last half of the history since calibration, cut into B = 8 equal
# blocks: for every image that moves (not held), d is its displacement from
# the half's mean string (scaled units), across that string (the part along
# its tangent, central differences, removed), or all of it at a free end.  A
# straight line through the eight block means of d gives a slope b per block
# and, from the scatter of the block means about the line, its standard
# error s per CV (batch means: correlation shorter than a block cannot make
# noise look like movement, and a slower one widens the error instead).
#   U    = (|b| + t sqrt(sum s^2)) x 8      upper (95%, t = 1.943, 6 dof)
#          bound on how far the image moved across the half, scaled units;
#   cost = U^2 / (h W)   W the half's updates: with the residual flow a
#          movement M over W updates is driven by a mean force M / (h W),
#          which does M^2 / (h W) of work on the way, kcal/mol;
#   prec = t^2 sigma^2 / (h W)   sigma^2 the image's mean squared d: what
#          not knowing where the averaged string sits costs, 1/2 kappa MCSE^2
#          with kappa ~ 2 / (h tau) for a constant step in a well of curvature
#          kappa, so the correlation time tau cancels and need not be known.
# The string is at rest when cost and prec are at most rest_tol at every
# image (0.05 kcal/mol, about kT/12) and no image's U exceeds rest_move_max
# (1 scaled unit, one basin width).  h is the step in force (the smaller of
# the current step and the calibrated one, so a decaying step only makes the
# test stricter).
#
# Why not the plateau test (largest distance between any two of the eight
# block means, less the noise inside the blocks, against 0.2 scaled units):
#   - on the alanine strings the images stay correlated over 10,000 to
#     20,000 updates (tau ~ 2 / (h kappa)), longer than half a block, so the
#     in-block noise was too small and a string AT REST failed: a synthetic
#     resting string with each image's measured wander and correlation passed
#     0.2 in 16% (pr0P) and 26% (pr1F) of trials, and no 120,000-update run
#     ever stopped;
#   - 0.2 over 60,000 updates is the movement a force of 0.015 kcal/mol per
#     scaled unit causes, twenty times below force_tol: the two tests did not
#     agree on what at rest means;
#   - judged against unbiased dynamics, the barrier along those strings did
#     not change after 10,000 updates (3.59-3.60, 3.56-3.57, 4.87-4.91
#     kcal/mol, errors 0.15-0.2): positions in soft directions wandered
#     where the free energy does not care.
# What this test lets through by construction: a creep driven by a force
# below sqrt(rest_tol / (h W)) (0.06-0.08 kcal/mol per scaled unit on the
# alanine runs), and a mode whose correlation time exceeds the half cannot be
# shown at rest (the run goes on, which is the safe way to be wrong).
# Assumes the residual flow; with the metric flow the alanine mobilities are
# all >= 1 in scaled units, so the cost is overestimated.
# Returns {cost prec move worst}, or {-1 -1 -1 -1} while the history is too
# short: fewer than 8 x conv_window frames, a half shorter than plateau_span
# updates, or fewer than two entries per block.
proc tw::rest {} {
    variable zhist; variable P; variable hist_chunk; variable scale; variable N; variable D
    variable h; variable h0
    set k [llength $zhist]; set B 8; set T 1.9431802803927816
    set L [expr {$k / (2 * $B)}]
    set kf [expr {$k * $hist_chunk}]
    if {$L < 2 || $kf < 8 * $P(conv_window) || 0.5 * $kf * $P(frame_updates) < $P(plateau_span)} { return {-1.0 -1.0 -1.0 -1} }
    set hh [expr {$h > 0 && $h < $h0 ? $h : $h0}]
    if {$hh <= 0} { return {-1.0 -1.0 -1.0 -1} }
    set k0 [expr {$k - $B * $L}]
    set W [expr {double($B * $L * $hist_chunk * $P(frame_updates))}]
    # mean string of the half and of each block, scaled
    set zbar [tw::mean_string $zhist $k0 [expr {$k - 1}]]
    set Q {}
    for {set j 0} {$j < $B} {incr j} { lappend Q [tw::mean_string $zhist [expr {$k0 + $j * $L}] [expr {$k0 + ($j + 1) * $L - 1}]] }
    set zs {}; foreach p $zbar { set q {}; foreach v $p s $scale { lappend q [expr {$v / $s}] }; lappend zs $q }
    set cost_w -1.0; set prec_w -1.0; set move_w 0.0; set worst -1; set bad -1.0
    for {set i 0} {$i < $N} {incr i} {
        if {[tw::held $i]} continue
        set ip [expr {$i > 0 ? $i - 1 : 0}]; set in [expr {$i < $N - 1 ? $i + 1 : $N - 1}]
        set t {}; set tt 0.0
        foreach a [lindex $zs $ip] b [lindex $zs $in] { set c [expr {$b - $a}]; lappend t $c; set tt [expr {$tt + $c * $c}] }
        set tt [expr {$tt > 0 ? sqrt($tt) : 1.0}]
        set tu {}; foreach c $t { lappend tu [expr {$c / $tt}] }
        set free [tw::free_end $i]
        set zi [lindex $zs $i]
        # block means of the displacement, projected
        set bm {}
        foreach q $Q {
            set x {}; foreach v [lindex $q $i] s $scale m $zi { lappend x [expr {$v / $s - $m}] }
            lappend bm [tw::rest_project $x $tu $free]
        }
        # straight line through the eight block means, per CV
        set b2 0.0; set s2 0.0
        for {set c 0} {$c < $D} {incr c} {
            set sl 0.0; set mean 0.0
            for {set j 0} {$j < $B} {incr j} {
                set y [lindex $bm $j $c]
                set sl [expr {$sl + ($j - 3.5) * $y}]; set mean [expr {$mean + $y}]
            }
            set sl [expr {$sl / 42.0}]; set mean [expr {$mean / $B}]
            set rr 0.0
            for {set j 0} {$j < $B} {incr j} { set r [expr {[lindex $bm $j $c] - $mean - ($j - 3.5) * $sl}]; set rr [expr {$rr + $r * $r}] }
            set b2 [expr {$b2 + $sl * $sl}]; set s2 [expr {$s2 + $rr / 6.0 / 42.0}]
        }
        set U [expr {(sqrt($b2) + $T * sqrt($s2)) * $B}]
        # mean squared displacement over the half's entries
        set sq 0.0
        for {set f $k0} {$f < $k} {incr f} {
            set x {}; foreach v [lindex $zhist $f $i] s $scale m $zi { lappend x [expr {$v / $s - $m}] }
            foreach c [tw::rest_project $x $tu $free] { set sq [expr {$sq + $c * $c}] }
        }
        set sq [expr {$sq / ($B * $L)}]
        set cost [expr {$U * $U / ($hh * $W)}]; set prec [expr {$T * $T * $sq / ($hh * $W)}]
        if {$cost > $cost_w} { set cost_w $cost }
        if {$prec > $prec_w} { set prec_w $prec }
        if {$U > $move_w} { set move_w $U }
        set m [expr {$cost > $prec ? $cost : $prec}]
        if {$m > $bad} { set bad $m; set worst $i }
    }
    return [list $cost_w $prec_w $move_w $worst]
}

# Displacement x (scaled) across the unit tangent tu, or all of it at a free end.
proc tw::rest_project {x tu free} {
    if {$free} { return $x }
    set dot 0.0; foreach a $x b $tu { set dot [expr {$dot + $a * $b}] }
    set out {}; foreach a $x b $tu { lappend out [expr {$a - $dot * $b}] }
    return $out
}

# THE PROFILE TEST over the same half (replaced the two-window test on
# 2026-11-16).  Each of the eight blocks' mean gradients is integrated along
# that block's mean string (tw::profile): its barrier (highest image less
# the lowest before it) and end-to-end difference.  A straight line through
# the eight values of each; the stop is blocked only when the profile is
# DEMONSTRABLY changing: the LOWER (95%) bound of the change across the half,
# (|slope| - t se) x 8, above profile_tol.  The driver's profiles are noisy
# first looks (the dedicated runs settle the profile); this guards against
# stopping while one is clearly still moving.  The two-window form it
# replaces took twenty frames on each side and treated them as independent,
# so its error was five to ten times too small and it blocked almost every
# stop on noise.  Returns the larger lower bound (kcal/mol), or -1 without
# the data.
proc tw::long_profile {} {
    variable zhist; variable ghist; variable P
    set k [llength $zhist]; set B 8; set T 1.9431802803927816
    set L [expr {$k / (2 * $B)}]
    if {$L < 2 || [llength $ghist] != $k} { return -1.0 }
    set k0 [expr {$k - $B * $L}]
    set bar {}; set dF {}
    for {set j 0} {$j < $B} {incr j} {
        set a [expr {$k0 + $j * $L}]; set b [expr {$a + $L - 1}]
        for {set f $a} {$f <= $b} {incr f} { if {[llength [lindex $ghist $f]] != [llength [lindex $zhist $f]]} { return -1.0 } }
        set F [tw::profile [tw::mean_string $zhist $a $b] [tw::mean_string $ghist $a $b]]
        set top 0; set fmax [lindex $F 0]
        for {set i 1} {$i < [llength $F]} {incr i} { if {[lindex $F $i] > $fmax} { set fmax [lindex $F $i]; set top $i } }
        set fmin [lindex $F 0]; for {set i 1} {$i <= $top} {incr i} { if {[lindex $F $i] < $fmin} { set fmin [lindex $F $i] } }
        lappend bar [expr {$fmax - $fmin}]; lappend dF [lindex $F end]
    }
    set worst 0.0
    foreach v [list $bar $dF] {
        set sl 0.0; set mean 0.0
        for {set j 0} {$j < $B} {incr j} { set sl [expr {$sl + ($j - 3.5) * [lindex $v $j]}]; set mean [expr {$mean + [lindex $v $j]}] }
        set sl [expr {$sl / 42.0}]; set mean [expr {$mean / $B}]
        set rr 0.0
        for {set j 0} {$j < $B} {incr j} { set r [expr {[lindex $v $j] - $mean - ($j - 3.5) * $sl}]; set rr [expr {$rr + $r * $r}] }
        set lo [expr {(abs($sl) - $T * sqrt($rr / 6.0 / 42.0)) * $B}]
        if {$lo > $worst} { set worst $lo }
    }
    return $worst
}

# A string's fine curve (as tw::image_curve_sq builds it), and the per-image
# squared curve distances from two strings and their fine curves, so that a
# block's curve is built once and not once per pair.
proc tw::fine_curve {A} {
    variable P
    return [tw::reparam $A [expr {([llength $A] - 1) * ($P(spline) ? $P(fine) : 1) + 1}]]
}
proc tw::image_curve_sq_f {A fa B fb} {
    variable P
    set per [expr {([llength $fa] - 1) / ([llength $A] - 1)}]; set w [expr {$P(curve_reach) * $per}]
    set out {}; set i 0
    foreach pa $A pb $B {
        set c [expr {$i * $per}]
        set u [tw::point_to_curve $pb $fa [expr {$c - $w}] [expr {$c + $w}]]
        set v [tw::point_to_curve $pa $fb [expr {$c - $w}] [expr {$c + $w}]]
        lappend out [expr {($u * $u + $v * $v) / 2.0}]
        incr i
    }
    return $out
}

# THE FORCE LEFT ACROSS THE STRING, over the same span as the plateau test:
# the last half of the frames since calibration, in four blocks.  At each
# image that follows the force, the part of the mean gradient that the
# string does not balance (tw::residual_vec, with the normal taken on the
# span's mean string) is averaged over each block; the mean of the four is
# the residual and their scatter its noise.  Returns {rms noise clean} over
# the images and sets tw::f_image to {image clean} pairs (tw::fl_image is
# the largest of them); -1s when there is too little history or no forces
# were recorded.  At a free end the whole gradient counts (tw::force_vec).
#
# A window of ten or twenty frames is too short for this: on the alanine
# dipeptide strings the noise-corrected number then swings between 0.2 and
# 0.8 from one window to the next, and a run could stop on a low swing.
# Over the long span it is steady, and on those strings it is NOT zero: it
# sits at a few images, at the path's sharp bends, at 0.5 to 1.4 kcal/mol
# per scaled unit, the same in both halves of a run and nearly the same
# with 24 and with 36 images.  That is a string that has come to rest with
# a force left, not one that is still relaxing; why it is left is not
# established.
proc tw::long_force {} {
    variable zhist; variable ghist; variable P; variable D; variable f_image; variable hist_chunk; variable fl_image
    set k [llength $zhist]; set L [expr {$k / 8}]; set fl_image -1.0
    if {$L * $hist_chunk < $P(conv_window) || $L < 1 || [llength $ghist] != $k} { return {-1.0 -1.0 -1.0} }
    set k0 [expr {$k - 4 * $L}]
    for {set f $k0} {$f < $k} {incr f} { if {[llength [lindex $ghist $f]] != [llength [lindex $zhist $f]]} { return {-1.0 -1.0 -1.0} } }
    set zbar [tw::mean_string $zhist $k0 [expr {$k - 1}]]
    set gb {}
    for {set j 0} {$j < 4} {incr j} { lappend gb [tw::mean_string $ghist [expr {$k0 + $j * $L}] [expr {$k0 + ($j + 1) * $L - 1}]] }
    set tot 0.0; set vtot 0.0; set m 0; set f_image {}
    foreach e [tw::normals $zbar] {
        lassign $e img n
        set sum [lrepeat $D 0.0]; set sq [lrepeat $D 0.0]
        foreach g $gb {
            set r [tw::force_vec $img [lindex $g $img] $n]
            set ns {}; set nq {}
            foreach a $sum b $sq c $r { lappend ns [expr {$a + $c}]; lappend nq [expr {$b + $c * $c}] }
            set sum $ns; set sq $nq
        }
        set ti 0.0; set vi 0.0
        foreach a $sum b $sq {
            set mean [expr {$a / 4.0}]
            set ti [expr {$ti + $mean * $mean}]
            set vi [expr {$vi + ($b / 4.0 - $mean * $mean) / 3.0}]
        }
        set tot [expr {$tot + $ti}]; set vtot [expr {$vtot + $vi}]
        set fi [expr {$ti > $vi ? sqrt($ti - $vi) : 0.0}]
        lappend f_image [list $img $fi]
        if {$fi > $fl_image} { set fl_image $fi }
        incr m
    }
    if {$m == 0} { set fl_image 0.0; return {0.0 0.0 0.0} }
    set clean [expr {($tot - $vtot) / $m}]
    return [list [expr {sqrt($tot / $m)}] [expr {$vtot > 0 ? sqrt($vtot / $m) : 0.0}] [expr {$clean > 0 ? sqrt($clean) : 0.0}]]
}

# Convergence needs FOUR things, all for conv_count frames in a row.
#
#   drift    over the last half of the run no image's block means differ,
#            beyond their own noise, by plateau_tol or more, and that noise
#            is below plateau_tol too (tw::plateau; with
#            plateau 0, the older test: the window-averaged string has moved
#            less than conv_tol between consecutive windows, which on real
#            runs sits at the wander and passes or fails by chance);
#   force    the force left across the string over the last half of the run
#            (tw::long_force; with plateau 0, over the last window of
#            frames), rms over the images that follow the force and with
#            the noise in that number taken out, is below force_tol
#            (kcal/mol per scaled unit; 0.3 is half of kT at 300 K), no
#            image's is above force_image_tol, and it was measured at all;
#   profile  the free-energy profile from the forces of one window and of
#            the next agree to within profile_tol (kcal/mol) at every image;
#   age      both windows lie after the calibration period.
#
# Drift alone is not enough: a small step makes any string drift slowly, and
# slow looks like still.  The force test asks what the string method is
# about, in absolute units, not relative to where the run started.  The
# profile test asks whether the answer the string is for has stopped
# changing.
#
# zf, gf: this frame's mean positions and mean gradients (tw::frame_z,
# tw::frame_force).  Returns the drift the test used (tw::d_curve, or
# tw::d_image when drift_curve is 0; both are kept), or -1
# while there is too little history.  tw::f_res, tw::f_noise and
# tw::p_change hold the other tests' values.
proc tw::record_frame {{zf {}} {gf {}}} {
    variable z; variable frames; variable P; variable scale; variable nconv; variable prof
    if {[llength $zf] == 0} { set zf $z }
    variable gframes
    if {[llength $prof] != [llength $frames] || [llength $gframes] != [llength $frames]} { set prof {}; set frames {}; set gframes {} }
    lappend frames $zf
    lappend gframes $gf
    # the plateau test's history: frames wholly after calibration
    variable zhist; variable nacc
    variable ghist
    set chunk_done 0
    if {$nacc >= $P(calibrate) + $P(frame_updates)} { set chunk_done [tw::history_add $zf $gf] }
    lappend prof [expr {[llength $gf] ? [tw::profile $zf $gf] : {}}]
    set w $P(conv_window)
    if {[llength $frames] > 2 * $w} {
        set frames [lrange $frames end-[expr {2 * $w - 1}] end]
        set prof [lrange $prof end-[expr {2 * $w - 1}] end]
        set gframes [lrange $gframes end-[expr {2 * $w - 1}] end]
    }
    if {[llength $frames] < 2 * $w} { set tw::d_sys -1.0; set tw::d_floor -1.0; return -1.0 }
    # The two windows' mean strings, and how far apart they are.
    #   d_image  the largest displacement of any one image.  It counts an
    #            image that has only slid ALONG the path, which happens to
    #            every image whenever an end moves and the string is respaced.
    #   d_curve  the largest distance from an image of either window to the
    #            other window's string taken as a curve (the spline through
    #            its images).  Sliding along the path costs nothing; a string
    #            that has grown or shrunk at an end is still counted, by the
    #            length gained or lost.
    # The stopping test uses d_curve unless TW(drift_curve) is 0.
    variable d_image; variable d_curve
    set n [llength $z]; set D [llength [lindex $z 0]]
    set A {}; set B {}
    for {set i 0} {$i < $n} {incr i} {
        set pa {}; set pb {}
        for {set d 0} {$d < $D} {incr d} {
            set a 0.0; set b 0.0
            for {set f 0} {$f < $w} {incr f} {
                set a [expr {$a + [lindex $frames $f $i $d]}]
                set b [expr {$b + [lindex $frames [expr {$f + $w}] $i $d]}]
            }
            lappend pa [expr {$a / $w}]; lappend pb [expr {$b / $w}]
        }
        lappend A $pa; lappend B $pb
    }
    set d_image 0.0
    foreach pa $A pb $B {
        set d2 0.0
        foreach a $pa b $pb sc $scale { set d2 [expr {$d2 + pow(($b - $a) / $sc, 2)}] }
        if {sqrt($d2) > $d_image} { set d_image [expr {sqrt($d2)}] }
    }
    set d_curve [tw::curve_distance $A $B]
    set worst [expr {$P(drift_curve) ? $d_curve : $d_image}]
    variable d_sys; variable d_floor; variable d_move; variable d_worst; variable p_long
    # (worked out every plateau_every frames: it changes slowly and is the
    # costliest thing here)
    variable fl_res; variable fl_noise; variable fl_clean; variable nplat
    set evaluated 0
    variable h_since
    if {$chunk_done && ($d_sys < 0 || $h_since >= $P(plateau_every))} {
        set h_since 0
        lassign [tw::rest] d_sys d_floor d_move d_worst
        if {$P(plateau)} { lassign [tw::long_force] fl_res fl_noise fl_clean; set p_long [tw::long_profile] }
        set evaluated [expr {$d_sys >= 0}]
    }
    set drift_ok [expr {$P(plateau) ? ($d_sys >= 0 && $d_sys <= $P(rest_tol) && $d_floor <= $P(rest_tol) && $d_move <= $P(rest_move_max)) : ($worst < $P(conv_tol))}]
    variable p_change
    if {$P(plateau)} {
        # the long-span profile test (tw::long_profile), worked out with the
        # rest test: blocks the stop only on a change demonstrated beyond
        # profile_tol
        set p_change $p_long
        set prof_ok [expr {$P(profile_tol) <= 0 || $p_change < 0 || $p_change <= $P(profile_tol)}]
    } else {
        # the profile of one window against the next (TW(plateau) 0 only)
        set p_change -1.0; set have 1; set p_sig 0
        foreach pf $prof { if {[llength $pf] != $n} { set have 0 } }
        if {$have} {
            # The stop is blocked when the profile is DEMONSTRABLY still changing:
            # by more than profile_tol and by more than three standard errors of
            # the difference (from the frame-to-frame spread in each window).  A
            # difference inside its own noise is not evidence of change; how well
            # the profile is resolved is for the dedicated runs to settle.
            set p_change 0.0; set p_sig 0.0
            for {set i 0} {$i < $n} {incr i} {
                set a 0.0; set b 0.0; set a2 0.0; set b2 0.0
                for {set f 0} {$f < $w} {incr f} {
                    set u [lindex $prof $f $i]; set v [lindex $prof [expr {$f + $w}] $i]
                    set a [expr {$a + $u}]; set b [expr {$b + $v}]
                    set a2 [expr {$a2 + $u * $u}]; set b2 [expr {$b2 + $v * $v}]
                }
                set a [expr {$a / $w}]; set b [expr {$b / $w}]
                set c [expr {abs($b - $a)}]
                set var [expr {$w > 1 ? (($a2 / $w - $a * $a) + ($b2 / $w - $b * $b)) / ($w - 1) : 0.0}]
                set se [expr {$var > 0 ? sqrt($var) : 0.0}]
                if {$c > $p_change} { set p_change $c }
                if {$c > $P(profile_tol) && $c > 3.0 * $se} { set p_sig 1 }
            }
        }
        set prof_ok [expr {$P(profile_tol) <= 0 || $p_change < 0 || !$p_sig}]
    }
    # The force left across the string over the last window, with the noise
    # in that number measured from the window itself and taken out.  The run
    # is not held until the force is RESOLVED below the tolerance: for a
    # protein in water that takes nanoseconds per image, the price of a
    # free-energy profile, and no published string calculation stops on an
    # absolute force.  It is held while the force is demonstrably there.
    variable noise1; variable force_ratio; variable f_res; variable f_noise; variable f_clean
    if {$P(plateau)} {
        # short window: reported only (window_force would overwrite the
        # per-image list of the long span, so keep that)
        variable f_image; set keep $f_image
        lassign [tw::window_force $w] f_res f_noise f_clean
        set f_image $keep
        # No measurement is not a pass (it was).  And the rms over the images
        # let one image carry force_tol x sqrt(images): one image at 1.0 among
        # twelve stopped.  So no image may exceed force_image_tol either
        # (default three times force_tol, 0.9: replaying the five converged
        # alanine runs' recorded forces, with the metric tensor of an earlier
        # run on the same channel, the largest per-image force over their
        # last halves was 0.67, and 0.64 at a stop; twice force_tol would have
        # refused two of the five).
        variable fl_image
        set itol [expr {$P(force_image_tol) < 0 ? 3.0 * $P(force_tol) : $P(force_image_tol)}]
        set force_ok [expr {$P(force_tol) <= 0 || ($fl_res >= 0 && $fl_clean <= $P(force_tol) && ($itol <= 0 || $fl_image <= $itol))}]
    } else {
        lassign [tw::window_force $w] f_res f_noise f_clean
        set force_ok [expr {$P(force_tol) <= 0 || ($f_res >= 0 && $f_clean <= $P(force_tol))}]
    }
    set force_ratio [expr {$f_noise > 0 ? $f_res / $f_noise : -1.0}]
    # both windows must lie after the calibration period, when the string stood still
    variable nacc
    set aged [expr {$nacc >= $P(calibrate) + 2 * $w * $P(frame_updates)}]
    if {$P(plateau)} {
        # the long-span tests change slowly and are worked out every
        # plateau_every frames, so conv_count counts EVALUATIONS that pass in
        # a row, not frames; the profile test still has to pass on the frame
        # ... and the passes must HOLD while the history grows by the fraction
        # plateau_hold.  Evaluations a few frames apart read almost the same
        # half of the run, so three of them in a row is one verdict, not
        # three: a string still creeping passed that way at its very first
        # evaluations and failed a few thousand updates later.
        # A stop, and the start of step_decay, happen only on a frame on which
        # the tests were evaluated (the hold used to complete on a frame up to
        # plateau_every - 1 frames after the evaluation it rested on): nconv
        # rises only there, and in between keeps its value unless the
        # profile test fails.  The
        # step decays once plateau, force and hold have all passed (not on a
        # single plateau pass), and goes back to h0 when the plateau fails.
        variable k_pass; variable hist_chunk; variable n_rest
        set kf [expr {[llength $zhist] * $hist_chunk}]
        if {!$prof_ok} { set nconv 0 }
        if {$evaluated} {
            if {$drift_ok && $force_ok && $aged} { incr nplat; if {$k_pass < 0} { set k_pass $kf } } else { set nplat 0; set k_pass -1 }
            set held [expr {$k_pass >= 0 && $kf >= (1.0 + $P(plateau_hold)) * $k_pass}]
            if {$held && $n_rest < 0} { set n_rest $nacc }
            if {!$drift_ok && $n_rest >= 0} { set n_rest -1 }
            set nconv [expr {$prof_ok && $held ? $nplat : 0}]
        }
    } elseif {$drift_ok && $force_ok && $prof_ok && $aged} { incr nconv } else { set nconv 0 }
    return $worst
}

proc tw::converged {} { variable nconv; variable P; return [expr {$nconv >= $P(conv_count)}] }

# Everything needed to continue after a restart, as a Tcl script.
# Mean gradient dF/dz at each image over the updates since the last call,
# and the number of updates it covers; starts a new frame.
# Mean position of each image over the updates since the last call: where
# the frame's forces were measured, and the string that is reported.
proc tw::frame_z {} {
    variable Z_sum; variable Z_n; variable z
    if {$Z_n == 0} { return $z }
    set out {}
    foreach s $Z_sum { set t {}; foreach u $s { lappend t [expr {$u / $Z_n}] }; lappend out $t }
    set Z_n 0; set Z_sum {}
    return $out
}

# The string to report: the average of the last window of recorded frames,
# respaced.  With polyak on, this, not the momentary tw::z, is the result.
proc tw::final_string {} {
    variable frames; variable P; variable z; variable N; variable D
    set w $P(conv_window)
    if {[llength $frames] == 0} { return $z }
    set use [lrange $frames end-[expr {$w - 1}] end]
    set out {}
    for {set i 0} {$i < $N} {incr i} {
        set p {}
        for {set d 0} {$d < $D} {incr d} {
            set s 0.0; foreach f $use { set s [expr {$s + [lindex $f $i $d]}] }
            lappend p [expr {$s / [llength $use]}]
        }
        lappend out $p
    }
    return [tw::reparam $out $N]
}

# Free-energy profile along a string from gradients dF/dz at its images
# (trapezoid rule, raw units); starts at 0 at the first image.
proc tw::profile {zz gg} {
    set F {0.0}; set acc 0.0
    for {set i 1} {$i < [llength $zz]} {incr i} {
        foreach a [lindex $zz [expr {$i - 1}]] b [lindex $zz $i] ga [lindex $gg [expr {$i - 1}]] gb [lindex $gg $i] {
            set acc [expr {$acc + 0.5 * ($ga + $gb) * ($b - $a)}]
        }
        lappend F $acc
    }
    return $F
}

# Solve A x = b for a small dense matrix (row-major list, n x n) by Gaussian
# elimination with partial pivoting.
proc tw::solve {A b} {
    set n [llength $b]
    set M {}
    for {set i 0} {$i < $n} {incr i} { lappend M [concat [lrange $A [expr {$i * $n}] [expr {$i * $n + $n - 1}]] [lindex $b $i]] }
    for {set c 0} {$c < $n} {incr c} {
        set piv $c; set big [expr {abs([lindex $M $c $c])}]
        for {set r [expr {$c + 1}]} {$r < $n} {incr r} { if {abs([lindex $M $r $c]) > $big} { set big [expr {abs([lindex $M $r $c])}]; set piv $r } }
        if {$big < 1e-300} { error "thalweg: singular metric tensor" }
        if {$piv != $c} { set tmp [lindex $M $c]; lset M $c [lindex $M $piv]; lset M $piv $tmp }
        for {set r 0} {$r < $n} {incr r} {
            if {$r == $c} continue
            set f [expr {[lindex $M $r $c] / [lindex $M $c $c]}]
            if {$f == 0.0} continue
            set row {}; foreach u [lindex $M $r] v [lindex $M $c] { lappend row [expr {$u - $f * $v}] }
            lset M $r $row
        }
    }
    set out {}
    for {set i 0} {$i < $n} {incr i} { lappend out [expr {[lindex $M $i $n] / [lindex $M $i $i]}] }
    return $out
}

# The averaged metric tensor of image i in scaled units, M_ab / (s_a s_b).
proc tw::mobility_matrix {i} {
    variable S_M; variable W; variable scale; variable D
    set sM [lindex $S_M $i]; set out {}
    for {set a 0} {$a < $D} {incr a} {
        for {set b 0} {$b < $D} {incr b} {
            lappend out [expr {[lindex $sM [expr {$a * $D + $b}]] / $W / ([lindex $scale $a] * [lindex $scale $b])}]
        }
    }
    return $out
}

# A move per unit step (scaled units), turned back into the force that
# causes it: M^-1 applied to it, in kcal/mol per scaled unit.  Dividing by one
# typical mobility instead is wrong when the tensor is uneven: for the four
# alanine torsions the scaled mobilities run from 1 to 70, so a force along a
# slow CV was understated twenty-fold and a string stopped at once.
proc tw::to_force {i mv} {
    set f [tw::solve [tw::mobility_matrix $i] $mv]
    set s 0.0; foreach v $f { set s [expr {$s + $v * $v}] }
    return [expr {sqrt($s)}]
}

# Counted part (tw::counted_move) of each free image's scaled move, as a vector.
proc tw::perp_vectors {moves} {
    variable z
    set out {}; set img -1
    foreach m $moves t [tw::tangents $z] {
        incr img
        if {[tw::held $img]} continue
        lappend out [list $img [tw::counted_move $img $m $t]]
    }
    return $out
}

# Unit tangents of the string in scaled units, from the derivative of the
# SAME cubic spline the respacing uses.  A centred-difference chord is off by
# 0.1 to 0.2 rad where the path bends, and that error times the force along
# the path reads as a force across it: on an exact MFEP with exact forces the
# chord tangent left a "residual" of 0.4 to 0.8 at 12 to 24 images.
proc tw::spline_tangents {zz} {
    variable scale; variable P
    set n [llength $zz]; set D [llength [lindex $zz 0]]
    if {!$P(spline) || $n < 3} { return [tw::tangents $zz] }
    set s [list 0.0]
    for {set i 1} {$i < $n} {incr i} {
        set d 0.0
        foreach a [lindex $zz [expr {$i - 1}]] b [lindex $zz $i] sc $scale { set d [expr {$d + pow(($b - $a) / $sc, 2)}] }
        lappend s [expr {[lindex $s end] + sqrt($d) + 1e-12}]
    }
    set der {}
    for {set d 0} {$d < $D} {incr d} {
        set y {}; foreach p $zz { lappend y [expr {[lindex $p $d] / [lindex $scale $d]}] }
        set m [tw::spline_m $s $y]; set col {}
        for {set i 0} {$i < $n} {incr i} {
            if {$i < $n - 1} {
                set hh [expr {[lindex $s [expr {$i + 1}]] - [lindex $s $i]}]
                lappend col [expr {([lindex $y [expr {$i + 1}]] - [lindex $y $i]) / $hh - $hh * (2.0 * [lindex $m $i] + [lindex $m [expr {$i + 1}]]) / 6.0}]
            } else {
                set hh [expr {[lindex $s $i] - [lindex $s [expr {$i - 1}]]}]
                lappend col [expr {([lindex $y $i] - [lindex $y [expr {$i - 1}]]) / $hh + $hh * (2.0 * [lindex $m $i] + [lindex $m [expr {$i - 1}]]) / 6.0}]
            }
        }
        lappend der $col
    }
    set out {}
    for {set i 0} {$i < $n} {incr i} {
        set t {}; set nn 0.0
        for {set d 0} {$d < $D} {incr d} { set v [lindex $der $d $i]; lappend t $v; set nn [expr {$nn + $v * $v}] }
        set nn [expr {sqrt($nn)}]; if {$nn < 1e-300} { set nn 1.0 }
        set u {}; foreach v $t { lappend u [expr {$v / $nn}] }
        lappend out $u
    }
    return $out
}

# The direction the gradient must have on a minimum free-energy path, at each
# image of zz that follows the force: n = M^-1 t / |M^-1 t| (scaled units).
# On an MFEP M grad F is parallel to the tangent t, so grad F is parallel to
# n, and what is left of the gradient after its part along n is taken out,
#     r = g - (g . n) n,
# is the force that should not be there.  It is exactly zero on an MFEP and
# can never exceed |g|.  (Projecting M g across t and then applying M^-1,
# which is also zero on an MFEP, is an oblique projection: with two coupled
# CVs it returned 7 times, and at a coupling of 0.99 seventy times, the force
# put in.)  Returns a list of {image n}.
proc tw::normals {zz} {
    variable N
    set out {}; set tans [tw::spline_tangents $zz]
    for {set i 0} {$i < $N} {incr i} {
        if {[tw::held $i]} continue
        set v [tw::solve [tw::mobility_matrix $i] [lindex $tans $i]]
        set nn 0.0; foreach c $v { set nn [expr {$nn + $c * $c}] }
        set nn [expr {sqrt($nn)}]; if {$nn < 1e-300} { set nn 1.0 }
        set u {}; foreach c $v { lappend u [expr {$c / $nn}] }
        lappend out [list $i $u]
    }
    return $out
}

# The residual vector at one image: gradient g (raw units) against normal n.
proc tw::residual_vec {g n} {
    variable scale
    set gu {}; set dot 0.0
    foreach v $g s $scale c $n { set w [expr {$v * $s}]; lappend gu $w; set dot [expr {$dot + $w * $c}] }
    set r {}; foreach w $gu c $n { lappend r [expr {$w - $dot * $c}] }
    return $r
}

# The force that should not be there at image img, scaled units: the residual
# (tw::residual_vec) inside the string; at a free end the whole gradient,
# because a free end comes to rest at a minimum, where the gradient vanishes
# (tw::flow_moves).
proc tw::force_vec {img g n} {
    variable scale
    if {[tw::free_end $img]} { set out {}; foreach v $g s $scale { lappend out [expr {$v * $s}] }; return $out }
    return [tw::residual_vec $g $n]
}

# Residual force across the string at each image that follows the force,
# from the running average of the gradient (kcal/mol per scaled unit).
proc tw::residual_forces {} {
    variable z; variable S_g; variable W
    set out {}
    foreach e [tw::normals $z] {
        lassign $e img n
        set g {}; foreach v [lindex $S_g $img] { lappend g [expr {$v / $W}] }
        set s 0.0; foreach c [tw::force_vec $img $g $n] { set s [expr {$s + $c * $c}] }
        lappend out [expr {sqrt($s)}]
    }
    return $out
}

# The same from the last `w` recorded frames: the mean gradients of those
# frames at the mean string of those frames.  Returns {rms residual, the
# noise in it, the residual with that noise taken out}.  The noise is from
# the frame-to-frame spread, so it is measured on the run itself, at the
# string as it now is, and forgets everything older than the window.
proc tw::window_force {w} {
    variable frames; variable gframes; variable N; variable D
    set zs [lrange $frames end-[expr {$w - 1}] end]; set gs [lrange $gframes end-[expr {$w - 1}] end]
    foreach g $gs { if {[llength $g] != $N} { return {-1.0 -1.0 -1.0} } }
    set zbar {}
    for {set i 0} {$i < $N} {incr i} {
        set p {}
        for {set d 0} {$d < $D} {incr d} { set s 0.0; foreach f $zs { set s [expr {$s + [lindex $f $i $d]}] }; lappend p [expr {$s / $w}] }
        lappend zbar $p
    }
    set tot 0.0; set vtot 0.0; set m 0
    variable f_image; set f_image {}
    foreach e [tw::normals $zbar] {
        set tot_i 0.0; set v_i 0.0
        lassign $e img n
        set sum [lrepeat $D 0.0]; set sq [lrepeat $D 0.0]
        foreach g $gs {
            set r [tw::force_vec $img [lindex $g $img] $n]
            set ns {}; set nq {}
            foreach a $sum b $sq c $r { lappend ns [expr {$a + $c}]; lappend nq [expr {$b + $c * $c}] }
            set sum $ns; set sq $nq
        }
        foreach a $sum b $sq {
            set mean [expr {$a / $w}]
            set tot [expr {$tot + $mean * $mean}]; set tot_i [expr {$tot_i + $mean * $mean}]
            if {$w > 1} {
                set vtot [expr {$vtot + ($b / $w - $mean * $mean) / ($w - 1)}]
                set v_i [expr {$v_i + ($b / $w - $mean * $mean) / ($w - 1)}]
            }
        }
        lappend f_image [list $img [expr {$tot_i > $v_i ? sqrt($tot_i - $v_i) : 0.0}]]
        incr m
    }
    if {$m == 0} { return {0.0 0.0 0.0} }
    set fres [expr {sqrt($tot / $m)}]; set fnoise [expr {$vtot > 0 ? sqrt($vtot / $m) : 0.0}]
    set clean [expr {$tot / $m - $vtot / $m}]
    return [list $fres $fnoise [expr {$clean > 0 ? sqrt($clean) : 0.0}]]
}

# The move per unit step from THIS update's gradients alone.
proc tw::inst_move {} {
    variable G_now
    return [tw::flow_moves $G_now]
}

proc tw::frame_force {} {
    variable F_sum; variable F_n
    if {$F_n == 0} { return [list 0 {}] }
    set out {}
    foreach s $F_sum { set t {}; foreach u $s { lappend t [expr {$u / $F_n}] }; lappend out $t }
    set n $F_n; set F_n 0; set F_sum {}
    return [list $n $out]
}

# The parameters the step is set under.  flow and calibrate change what the
# step means or when it is set; the others only set its size.
proc tw::step_params {} {
    variable P
    set out {}
    foreach k {flow calibrate nblocks frame_updates max_move noise_frac conv_tol conv_window polyak stepsize} { lappend out $k $P($k) }
    return $out
}

proc tw::dump_state {} {
    variable h
    set out ""
    foreach v {N D scale periodic z S_g S_M W h nacc frames nconv blk h_signal h_noise noise1 force_ratio F_sum F_n
               G_now Z_sum Z_n prof f_res f_noise p_change gframes f_clean d_image d_curve
               zhist d_sys d_floor d_move d_worst p_long f_image ghist nplat fl_res fl_noise fl_clean fl_image
               hist_chunk hz_sum hg_sum hz_n h_since h0 n_rest noise_ratio k_pass} {
        append out [list set tw::$v [set tw::$v]] "\n"
    }
    # the memory may have been lengthened at calibration; keep that choice
    append out [list set tw::P(memory) $tw::P(memory)] "\n"
    # what the step was set under (before calibration ends: what it will be set under)
    append out [list set tw::saved_P [expr {$h > 0 && [llength $tw::saved_P] ? $tw::saved_P : [tw::step_params]}]] "\n"
    return $out
}

# Load a state written by tw::dump_state, by this version or an older one.
# What an older file does not carry is set first: no step record (it ran the
# metric flow, the only one there was), no rest point, and h0 taken from h
# (before step_decay existed the step never changed after calibration;
# without this, step_decay was silently off on such a restart).
proc tw::load_state {script} {
    foreach {v x} {saved_P {} h0 -1.0 n_rest -1 k_pass -1 nplat 0 fl_image -1.0 hist_chunk 1 hz_sum {} hg_sum {} hz_n 0 h_since 0
                   zhist {} ghist {} noise_ratio -1.0 d_sys -1.0 d_floor -1.0 fl_res -1.0 fl_noise -1.0 fl_clean -1.0} {
        set tw::$v $x
    }
    uplevel #0 $script
    if {$tw::h0 < 0} { set tw::h0 $tw::h }
    if {[llength $tw::saved_P] == 0 && $tw::h > 0} { set tw::saved_P {flow metric} }
}

# The flow the loaded state ran (an older state: metric); {} before calibration.
proc tw::state_flow {} {
    variable saved_P; variable h
    if {$h <= 0} { return {} }
    if {[dict exists $saved_P flow]} { return [dict get $saved_P flow] }
    return metric
}

# After a restart, with the config's parameters applied: refuse what would
# make the calibrated step wrong, and report what no longer matches it.
#   flow      the two flows' steps differ by up to the spread of M's
#             eigenvalues (250-fold on alanine); a step calibrated under one
#             moved strings up to 250 times too far under the other.
#   calibrate when the step is set and how the noise blocks are cut.
#   frame_updates  the history's frames and the plateau span are counted in it.
#   nblocks   (only while still calibrating) how the noise blocks are cut.
# max_move, noise_frac, conv_tol, conv_window, polyak and stepsize only set
# the step's size, which is already set: a change has no effect and is
# reported.  Returns the report ({} when all match).
proc tw::restart_check {} {
    variable saved_P; variable P; variable h
    set warn {}
    set sf [tw::state_flow]
    if {$sf ne {} && $sf ne $P(flow)} {
        error "thalweg: the restart state's step was calibrated under TW(flow) $sf[expr {[dict exists $saved_P flow] ? {} : { (a state file from before TW(flow) existed)}}]; this run asks for TW(flow) $P(flow).  Continue it with TW(flow) $sf, or start a new run."
    }
    foreach k {calibrate frame_updates nblocks} {
        if {$k eq "nblocks" && $h > 0} continue
        if {[dict exists $saved_P $k] && [dict get $saved_P $k] != $P($k)} {
            error "thalweg: the restart state was run with TW($k) [dict get $saved_P $k]; this run asks for $P($k).  Continue it with [dict get $saved_P $k], or start a new run."
        }
    }
    if {$h > 0} {
        foreach k {max_move noise_frac conv_tol conv_window polyak stepsize} {
            if {[dict exists $saved_P $k] && [dict get $saved_P $k] != $P($k)} {
                lappend warn "TW($k) is $P($k) but the step was calibrated with [dict get $saved_P $k]; the step stays $h"
            }
        }
    }
    return [join $warn "; "]
}
