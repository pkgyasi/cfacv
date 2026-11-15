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
        frame_updates 100
        nblocks     10
    }
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
    # noise per update of the across-string force, kcal/mol per scaled unit (worst image)
    variable noise_f1 0.0
    variable p_change -1.0
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

proc tw::set_param {name value} {
    variable P
    if {![info exists P($name)]} { error "thalweg: unknown parameter '$name' (known: [lsort [array names P]])" }
    set P($name) $value
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
    variable noise_f1 0.0; variable gframes {}; variable f_clean -1.0
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
proc tw::raw_move {} {
    variable N; variable D; variable S_g; variable S_M; variable W
    set out {}
    foreach sg $S_g sM $S_M {
        set m {}
        for {set i 0} {$i < $D} {incr i} {
            set t 0.0
            for {set j 0} {$j < $D} {incr j} { set t [expr {$t + [lindex $sM [expr {$i * $D + $j}]] * [lindex $sg $j]}] }
            lappend m [expr {-$t / ($W * $W)}]
        }
        lappend out $m
    }
    return $out
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

# Size, in scaled units, of the part of each move that is ACROSS the string.
# Movement along the string is undone by respacing, so it is not progress.
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

# Across-string size of each image's move per unit step, for the images that
# follow the force (held ends are left out).
proc tw::perp_sizes {moves} {
    variable z; variable scale
    set out {}; set img -1
    foreach m $moves t [tw::tangents $z] {
        incr img
        if {[tw::held $img]} continue
        set ms {}; set dot 0.0
        foreach v $m s $scale tt $t { set x [expr {$v / $s}]; lappend ms $x; set dot [expr {$dot + $x * $tt}] }
        set nn 0.0
        foreach x $ms tt $t { set y [expr {$x - $dot * $tt}]; set nn [expr {$nn + $y * $y}] }
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
    variable blk; variable z; variable scale; variable S_M; variable W; variable D
    set nb 0; foreach b $blk { if {[lindex $b 0] > 0} { incr nb } }
    if {$nb < 3} { return 0.0 }
    set tans [tw::tangents $z]; set worst 0.0; set worstf 0.0; set img 0; set nimg 0
    variable noise_f1
    foreach t $tans sM $S_M {
        if {[tw::held $img]} { incr img; continue }
        set mean [lrepeat $D 0.0]; set sq 0.0; set sqf 0.0; set len 0.0; set vecs {}; set fvecs {}
        set Mu [tw::mobility_matrix $img]
        foreach b $blk {
            lassign $b cnt sums
            if {$cnt == 0} continue
            set len [expr {$len + $cnt}]
            # block-mean gradient -> move with step 1 -> scaled -> across the string
            set g {}; foreach v [lindex $sums $img] { lappend g [expr {$v / $cnt}] }
            set ms {}; set dot 0.0
            for {set i 0} {$i < $D} {incr i} {
                set u 0.0
                for {set j 0} {$j < $D} {incr j} { set u [expr {$u + [lindex $sM [expr {$i * $D + $j}]] * [lindex $g $j]}] }
                set u [expr {-$u / $W / [lindex $scale $i]}]
                lappend ms $u; set dot [expr {$dot + $u * [lindex $t $i]}]
            }
            set pv {}; foreach u $ms tt $t { lappend pv [expr {$u - $dot * $tt}] }
            lappend vecs $pv
            lappend fvecs [tw::solve $Mu $pv]                 ;# the same, as a force
        }
        set n [llength $vecs]
        for {set i 0} {$i < $D} {incr i} {
            set m 0.0; foreach v $vecs { set m [expr {$m + [lindex $v $i]}] }
            set m [expr {$m / $n}]
            foreach v $vecs { set sq [expr {$sq + pow([lindex $v $i] - $m, 2)}] }
            set m 0.0; foreach v $fvecs { set m [expr {$m + [lindex $v $i]}] }
            set m [expr {$m / $n}]
            foreach v $fvecs { set sqf [expr {$sqf + pow([lindex $v $i] - $m, 2)}] }
        }
        # variance of a block mean, times the block length = variance per update
        set s1 [expr {sqrt($sq / ($n - 1) * ($len / $n))}]
        # pooled over the images (root mean square): the largest of N noisy
        # estimates overstates the noise by half
        set worst [expr {$worst + $s1 * $s1}]; incr nimg
        set s1 [expr {sqrt($sqf / ($n - 1) * ($len / $n))}]
        set worstf [expr {$worstf + $s1 * $s1}]
        incr img
    }
    if {$nimg == 0} { return 0.0 }
    set noise_f1 [expr {sqrt($worstf / $nimg)}]
    return [expr {sqrt($worst / $nimg)}]
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
    set moves [tw::raw_move]
    if {$h == 0.0} {
        # set the step once, from forces averaged while the string stood still
        variable h_signal; variable h_noise; variable noise1
        set big 0.0
        foreach v [tw::perp_sizes $moves] { if {$v > $big} { set big $v } }
        if {$big <= 0.0} { error "thalweg: the averaged force across the string is exactly zero; cannot set a step" }
        set h_signal [expr {$P(max_move) / $big}]
        set noise1 [tw::noise_per_update]
        # a random walk of the noise alone over one drift-test window
        set h_noise [expr {$noise1 > 0 ? $P(noise_frac) * $P(conv_tol) / ($noise1 * sqrt($P(conv_window) * $P(frame_updates))) : $h_signal}]
        if {$P(stepsize) > 0} { set h $P(stepsize) } else { set h [expr {$h_noise < $h_signal ? $h_noise : $h_signal}] }
        set tw::step_factor 1
        # (An earlier version lengthened the force-averaging memory here, up
        # to 10,000 updates, and multiplied the MD steps per update, so that
        # an average of the force could resolve a fixed tolerance.  That cost
        # nanoseconds per image before a string was allowed to stop, and the
        # long average remembered the force of the starting string.  The
        # stopping test now measures the force over the last window of
        # frames and allows for its own noise; see tw::record_frame.)
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

# Convergence needs FOUR things, all for conv_count frames in a row.
#
#   drift    the window-averaged string has moved less than conv_tol between
#            consecutive windows;
#   force    the force left across the string over the last window of
#            frames, rms over the images that follow the force and with the
#            noise in that number taken out, is below force_tol (kcal/mol
#            per scaled unit; 0.3 is half of kT at 300 K);
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
# tw::frame_force).  Returns the largest scaled drift of any image, or -1
# while there is too little history.  tw::f_res, tw::f_noise and
# tw::p_change hold the other tests' values.
proc tw::record_frame {{zf {}} {gf {}}} {
    variable z; variable frames; variable P; variable scale; variable nconv; variable prof
    if {[llength $zf] == 0} { set zf $z }
    variable gframes
    if {[llength $prof] != [llength $frames] || [llength $gframes] != [llength $frames]} { set prof {}; set frames {}; set gframes {} }
    lappend frames $zf
    lappend gframes $gf
    lappend prof [expr {[llength $gf] ? [tw::profile $zf $gf] : {}}]
    set w $P(conv_window)
    if {[llength $frames] > 2 * $w} {
        set frames [lrange $frames end-[expr {2 * $w - 1}] end]
        set prof [lrange $prof end-[expr {2 * $w - 1}] end]
        set gframes [lrange $gframes end-[expr {2 * $w - 1}] end]
    }
    if {[llength $frames] < 2 * $w} { return -1.0 }
    set n [llength $z]; set D [llength [lindex $z 0]]; set worst 0.0
    for {set i 0} {$i < $n} {incr i} {
        set d2 0.0
        for {set d 0} {$d < $D} {incr d} {
            set a 0.0; set b 0.0
            for {set f 0} {$f < $w} {incr f} {
                set a [expr {$a + [lindex $frames $f $i $d]}]
                set b [expr {$b + [lindex $frames [expr {$f + $w}] $i $d]}]
            }
            set d2 [expr {$d2 + pow(($b - $a) / $w / [lindex $scale $d], 2)}]
        }
        if {sqrt($d2) > $worst} { set worst [expr {sqrt($d2)}] }
    }
    # the profile of one window against the next
    variable p_change
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
    # The force left across the string over the last window, with the noise
    # in that number measured from the window itself and taken out.  The run
    # is not held until the force is RESOLVED below the tolerance: for a
    # protein in water that takes nanoseconds per image, the price of a
    # free-energy profile, and no published string calculation stops on an
    # absolute force.  It is held while the force is demonstrably there.
    variable noise1; variable force_ratio; variable f_res; variable f_noise; variable f_clean
    lassign [tw::window_force $w] f_res f_noise f_clean
    set force_ratio [expr {$f_noise > 0 ? $f_res / $f_noise : -1.0}]
    set force_ok [expr {$P(force_tol) <= 0 || $f_res < 0 || $f_clean <= $P(force_tol)}]
    # both windows must lie after the calibration period, when the string stood still
    variable nacc
    set aged [expr {$nacc >= $P(calibrate) + 2 * $w * $P(frame_updates)}]
    if {$worst < $P(conv_tol) && $force_ok && $prof_ok && $aged} { incr nconv } else { set nconv 0 }
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

# Across-string part of each free image's scaled move, as a vector.
proc tw::perp_vectors {moves} {
    variable z; variable scale
    set out {}; set img -1
    foreach m $moves t [tw::tangents $z] {
        incr img
        if {[tw::held $img]} continue
        set ms {}; set dot 0.0
        foreach v $m s $scale tt $t { set u [expr {$v / $s}]; lappend ms $u; set dot [expr {$dot + $u * $tt}] }
        set pv {}; foreach u $ms tt $t { lappend pv [expr {$u - $dot * $tt}] }
        lappend out [list $img $pv]
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

# Residual force across the string at each image that follows the force,
# from the running average of the gradient (kcal/mol per scaled unit).
proc tw::residual_forces {} {
    variable z; variable S_g; variable W
    set out {}
    foreach e [tw::normals $z] {
        lassign $e img n
        set g {}; foreach v [lindex $S_g $img] { lappend g [expr {$v / $W}] }
        set s 0.0; foreach c [tw::residual_vec $g $n] { set s [expr {$s + $c * $c}] }
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
    foreach e [tw::normals $zbar] {
        lassign $e img n
        set sum [lrepeat $D 0.0]; set sq [lrepeat $D 0.0]
        foreach g $gs {
            set r [tw::residual_vec [lindex $g $img] $n]
            set ns {}; set nq {}
            foreach a $sum b $sq c $r { lappend ns [expr {$a + $c}]; lappend nq [expr {$b + $c * $c}] }
            set sum $ns; set sq $nq
        }
        foreach a $sum b $sq {
            set mean [expr {$a / $w}]
            set tot [expr {$tot + $mean * $mean}]
            if {$w > 1} { set vtot [expr {$vtot + ($b / $w - $mean * $mean) / ($w - 1)}] }
        }
        incr m
    }
    if {$m == 0} { return {0.0 0.0 0.0} }
    set fres [expr {sqrt($tot / $m)}]; set fnoise [expr {$vtot > 0 ? sqrt($vtot / $m) : 0.0}]
    set clean [expr {$tot / $m - $vtot / $m}]
    return [list $fres $fnoise [expr {$clean > 0 ? sqrt($clean) : 0.0}]]
}

# The move per unit step from THIS update's gradients alone.
proc tw::inst_move {} {
    variable D; variable G_now; variable S_M; variable W
    set out {}
    foreach g $G_now sM $S_M {
        set m {}
        for {set i 0} {$i < $D} {incr i} {
            set t 0.0
            for {set j 0} {$j < $D} {incr j} { set t [expr {$t + [lindex $sM [expr {$i * $D + $j}]] * [lindex $g $j]}] }
            lappend m [expr {-$t / $W}]
        }
        lappend out $m
    }
    return $out
}

proc tw::frame_force {} {
    variable F_sum; variable F_n
    if {$F_n == 0} { return [list 0 {}] }
    set out {}
    foreach s $F_sum { set t {}; foreach u $s { lappend t [expr {$u / $F_n}] }; lappend out $t }
    set n $F_n; set F_n 0; set F_sum {}
    return [list $n $out]
}

proc tw::dump_state {} {
    set out ""
    foreach v {N D scale periodic z S_g S_M W h nacc frames nconv blk h_signal h_noise noise1 force_ratio F_sum F_n
               G_now Z_sum Z_n prof f_res f_noise p_change noise_f1 gframes f_clean} {
        append out [list set tw::$v [set tw::$v]] "\n"
    }
    # the memory may have been lengthened at calibration; keep that choice
    append out [list set tw::P(memory) $tw::P(memory)] "\n"
    return $out
}
