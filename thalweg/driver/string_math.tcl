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
#   2. AN AVERAGED FORCE AND A MEASURED STEP.  The mean force and metric
#      tensor of each image are running averages over many updates, not the
#      few MD steps of the last update.  The step size is not guessed: the
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
        memory_max  10000
        noise_frac  0.5
        frame_updates 100
        nblocks     10
    }
    variable blk {}
    variable h_signal 0.0
    variable h_noise 0.0
    variable noise1 0.0
    variable force_ratio -1.0
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
proc tw::perp_sizes {moves} {
    variable z; variable scale
    set out {}
    foreach m $moves t [tw::tangents $z] {
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
    set tans [tw::tangents $z]; set worst 0.0; set img 0
    foreach t $tans sM $S_M {
        set mean [lrepeat $D 0.0]; set sq 0.0; set len 0.0; set vecs {}
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
        }
        set n [llength $vecs]
        for {set i 0} {$i < $D} {incr i} {
            set m 0.0; foreach v $vecs { set m [expr {$m + [lindex $v $i]}] }
            set m [expr {$m / $n}]
            foreach v $vecs { set sq [expr {$sq + pow([lindex $v $i] - $m, 2)}] }
        }
        # variance of a block mean, times the block length = variance per update
        set s1 [expr {sqrt($sq / ($n - 1) * ($len / $n))}]
        if {$s1 > $worst} { set worst $s1 }
        incr img
    }
    return $worst
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

# Respace the images at equal scaled arc length.  The end images do not move.
proc tw::reparam {zz} {
    variable scale; variable P
    set n [llength $zz]; set D [llength [lindex $zz 0]]
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
    for {set i 1} {$i < $n - 1} {incr i} {
        set t [expr {$total * $i / ($n - 1.0)}]
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
proc tw::advance {} {
    variable z; variable h; variable P; variable N
    if {![tw::calibrated]} { return [dict create moved 0 h 0.0 perp 0.0] }
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
        # Lengthen the averaging memory until the noise left in the average is
        # force_res of the force the string starts with.  Otherwise "the force
        # is down to its noise" can be true from the first frame: with a
        # memory of 200 updates the alanine strings passed that test at once,
        # a quarter of the way to their path.
        if {$noise1 > 0} {
            set s 0.0; set m 0
            foreach v [tw::perp_sizes $moves] { set s [expr {$s + $v * $v}]; incr m }
            set sig [expr {sqrt($s / $m)}]
            if {$sig > 0} {
                set need [expr {int(pow($noise1 / ($P(force_res) * $sig), 2) / 2.0) + 1}]
                if {$need > $P(memory_max)} { set need $P(memory_max) }
                if {$need > $P(memory)} { set P(memory) $need }
            }
        }
    }
    set perp 0.0
    foreach v [tw::perp_sizes $moves] { if {$v * $h > $perp} { set perp [expr {$v * $h}] } }
    set nz {}; set i 0
    foreach p $z m $moves {
        if {!$P(evolve_ends) && ($i == 0 || $i == $N - 1)} {
            lappend nz $p
        } else {
            set q {}; foreach v $p dv $m { lappend q [expr {$v + $h * $dv}] }
            lappend nz $q
        }
        incr i
    }
    set z [tw::reparam $nz]
    return [dict create moved 1 h $h perp $perp]
}

# Convergence needs TWO things, both for conv_count frames in a row.
#
#   drift  the window-averaged string has moved less than conv_tol between
#          consecutive windows;
#   force  the averaged force across the string is no larger than conv_force
#          times what noise alone would leave in the average.
#
# Drift alone is not enough.  A small step makes any string drift slowly, and
# the first noise-limited runs stopped after 3,000 updates a quarter of the
# way to their path, because slow looked like still.  The force test asks the
# question the string method is about: is there still a force across the
# path?  It can only be answered down to the noise in the running average,
# which is noise_per_update / sqrt(2 x memory); a longer memory resolves a
# smaller force.
#
# Returns the largest scaled drift of any image, or -1 while there is too
# little history.  tw::force_ratio holds the force test's value.
proc tw::record_frame {} {
    variable z; variable frames; variable P; variable scale; variable nconv
    lappend frames $z
    set w $P(conv_window)
    if {[llength $frames] > 2 * $w} { set frames [lrange $frames end-[expr {2 * $w - 1}] end] }
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
    # root-mean-square, over the images, of the averaged across-string force,
    # in units of the noise left in that average
    variable noise1; variable force_ratio
    set force_ok 1
    if {$noise1 > 0} {
        set s 0.0; set m 0
        foreach v [tw::perp_sizes [tw::raw_move]] { set s [expr {$s + $v * $v}]; incr m }
        set force_ratio [expr {sqrt($s / $m) / ($noise1 / sqrt(2.0 * $P(memory)))}]
        set force_ok [expr {$force_ratio <= $P(conv_force)}]
    }
    # and the running average must have turned over a few times since the
    # string started to move, or it still remembers the calibration period
    variable nacc
    set aged [expr {$nacc >= $P(calibrate) + 4 * $P(memory)}]
    if {$worst < $P(conv_tol) && $force_ok && $aged} { incr nconv } else { set nconv 0 }
    return $worst
}

proc tw::converged {} { variable nconv; variable P; return [expr {$nconv >= $P(conv_count)}] }

# Everything needed to continue after a restart, as a Tcl script.
proc tw::dump_state {} {
    set out ""
    foreach v {N D scale periodic z S_g S_M W h nacc frames nconv blk h_signal h_noise noise1 force_ratio} {
        append out [list set tw::$v [set tw::$v]] "\n"
    }
    # the memory may have been lengthened at calibration; keep that choice
    append out [list set tw::P(memory) $tw::P(memory)] "\n"
    return $out
}
