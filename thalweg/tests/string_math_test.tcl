# Tests of driver/string_math.tcl under plain tclsh: no NAMD, no cluster.
#   tclsh tests/string_math_test.tcl
# A mock "replica" returns the gradient of a known surface at the point it is
# restrained to, plus noise, so the whole master loop can be run here.
source [file join [file dirname [info script]] .. driver string_math.tcl]
set npass 0; set nfail 0
proc check {name ok} { global npass nfail; if {$ok} { incr npass } else { incr nfail; puts "FAIL $name" } }
set PI 3.14159265358979323846
expr {srand(20261114)}
proc gauss {} { return [expr {sqrt(-2.0 * log(1.0 - rand())) * cos(6.283185307179586 * rand())}] }

# two wells at (-1,0) and (1,0) on a ring; the upper arc has the lower barrier
proc gradV {x y} {
    set r [expr {max(hypot($x, $y), 1e-9)}]
    set gx [expr {40.0 * ($r - 1) * $x / $r}]; set gy [expr {40.0 * ($r - 1) * $y / $r}]
    foreach b {{-1 0 -3 0.15} {1 0 -3 0.15} {0 1 2 0.1} {0 -1 5 0.1}} {
        lassign $b cx cy a w
        set e [expr {$a * exp(-(($x - $cx) ** 2 + ($y - $cy) ** 2) / $w)}]
        set gx [expr {$gx + $e * (-2 * ($x - $cx) / $w)}]; set gy [expr {$gy + $e * (-2 * ($y - $cy) / $w)}]
    }
    return [list $gx $gy]
}
proc V {x y} {
    set r [expr {max(hypot($x, $y), 1e-9)}]; set v [expr {20.0 * ($r - 1) ** 2}]
    foreach b {{-1 0 -3 0.15} {1 0 -3 0.15} {0 1 2 0.1} {0 -1 5 0.1}} { lassign $b cx cy a w; set v [expr {$v + $a * exp(-(($x - $cx) ** 2 + ($y - $cy) ** 2) / $w)}] }
    return $v
}

# ---- 1. wrapping and continuity
check "wrap 3.5 -> -2.78" [expr {abs([tw::wrap 3.5] - (3.5 - 2 * $PI)) < 1e-12}]
check "wrap -pi stays in (-pi, pi]" [expr {abs([tw::wrap [expr {-$PI}]] - $PI) < 1e-9}]
tw::init {{8.0 2.9} {9.0 3.1} {10.0 -3.0} {11.0 -2.8}} {1.0 0.2} {0 1}
set z $tw::z
check "string made continuous across +pi" [expr {abs([lindex $z 2 1] - (2 * $PI - 3.0)) < 1e-9 && [lindex $z 3 1] > [lindex $z 2 1]}]
set t [tw::targets]
check "targets wrapped back into (-pi, pi]" [expr {abs([lindex $t 2 1] + 3.0) < 1e-9 && abs([lindex $t 3 1] + 2.8) < 1e-9 && abs([lindex $t 1 1] - 3.1) < 1e-9}]
check "non-periodic CV untouched" [expr {[lindex $t 3 0] == 11.0}]

# ---- 2. respacing: equal arc, ends fixed, and a spline cuts corners less
proc on_arc {angles} { set o {}; foreach a $angles { lappend o [list [expr {-cos($a)}] [expr {sin($a)}]] }; return $o }
set uneven [on_arc {0.0 0.2 0.5 1.9 3.14159265358979}]
foreach mode {1 0} name {spline linear} {
    tw::init $uneven {1.0 1.0} {0 0}; tw::set_param spline $mode
    set r [tw::reparam $tw::z]
    set worst 0.0; foreach p $r { set d [expr {abs(hypot([lindex $p 0], [lindex $p 1]) - 1.0)}]; if {$d > $worst} { set worst $d } }
    set dev($name) $worst
    set seg {}; for {set i 1} {$i < 5} {incr i} { lappend seg [expr {hypot([lindex $r $i 0] - [lindex $r [expr {$i-1}] 0], [lindex $r $i 1] - [lindex $r [expr {$i-1}] 1])}] }
    set spread($name) [expr {([tcl::mathfunc::max {*}$seg] - [tcl::mathfunc::min {*}$seg]) / [tcl::mathfunc::max {*}$seg]}]
    check "$name: ends unmoved" [expr {[lindex $r 0] eq [lindex $uneven 0] && [lindex $r end] eq [lindex $uneven end]}]
}
check "spline stays nearer the true arc than straight segments ($dev(spline) vs $dev(linear))" [expr {$dev(spline) < 0.5 * $dev(linear)}]
check "spline spacing nearly equal ($spread(spline))" [expr {$spread(spline) < 0.2}]
tw::set_param spline 1

# ---- 2b. a hook at an end is removed and the image count kept
tw::init {{0.0 0.0} {-0.3 0.05} {0.5 0.0} {1.0 0.0} {1.5 0.0} {2.0 0.0} {2.3 0.1} {2.1 0.4}} {1.0 1.0} {0 0}
check "hook detected at the start (cos [tw::turn_cos $tw::z 1])" [expr {[tw::turn_cos $tw::z 1] < 0}]
set u [tw::unhook $tw::z]
check "hooked images dropped at both ends ([llength $u] left of 8)" [expr {[llength $u] == 6 && [lindex $u 0] eq {0.0 0.0} && [lindex $u end] eq {2.1 0.4}}]
set r [tw::reparam $u 8]
check "respacing restores 8 images" [expr {[llength $r] == 8}]
set worst 1.0; for {set i 1} {$i < 7} {incr i} { set c [tw::turn_cos $r $i]; if {$c < $worst} { set worst $c } }
check "no turn sharper than 90 degrees afterwards (sharpest cos $worst)" [expr {$worst > 0}]
check "a gentle bend is left alone" [expr {[llength [tw::unhook {{0 0} {1 0.2} {2 0.5} {3 1.2} {4 2.0}}]] == 5}]

# ---- 2b. pinned ends: walk to the pin at pin_speed, then stay; periodic pins by the short way
tw::init {{0.0 3.0} {1.0 3.0} {2.0 3.0} {3.0 3.0}} {1.0 0.5} {0 1}
tw::set_param pin_speed 0.01
lassign [tw::toward_pin {0.0 3.0} {0.03 3.0}] q left
check "pin: one step of pin_speed toward it" [expr {abs([lindex $q 0] - 0.01) < 1e-12 && abs($left - 0.02) < 1e-12}]
lassign [tw::toward_pin {0.0 3.0} {0.004 3.0}] q left
check "pin: lands exactly when closer than one step" [expr {abs([lindex $q 0] - 0.004) < 1e-12 && $left == 0.0}]
lassign [tw::toward_pin {0.0 3.0} [list 0.0 -3.1]] q left
check "pin: a torsion pin is approached the short way across pi" [expr {[lindex $q 1] > 3.0 && [lindex $q 1] < 3.01}]
check "pin: wrong length refused" [catch {tw::toward_pin {0.0 3.0} {1.0}}]
# in the loop: a force that would push the ends the wrong way; the pins win
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
tw::set_param calibrate 2; tw::set_param memory 2; tw::set_param stepsize 1e-3; tw::set_param pin_speed 0.05
tw::set_param pin_first {0.0 0.4}; tw::set_param pin_last {3.0 -0.3}
set I {1 0 0 1}
for {set it 0} {$it < 30} {incr it} {
    tw::accumulate {{0 1e-3} {0 1e-3} {0 1e-3} {0 1e-3}} [list $I $I $I $I]
    tw::advance
}
check "pinned ends arrive and stay ([lindex $tw::z 0] / [lindex $tw::z end])" [expr {abs([lindex $tw::z 0 1] - 0.4) < 1e-9 && abs([lindex $tw::z end 1] + 0.3) < 1e-9 && abs([lindex $tw::z 0 0]) < 1e-9}]
# a held end's force is left out of the step and the stopping test
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
set big {{0 50.0} {0 1.0} {0 2.0} {0 70.0}}
tw::set_param pin_first {}; tw::set_param pin_last {}
check "free ends: all four images counted" [expr {[llength [tw::perp_sizes $big]] == 4}]
tw::set_param pin_first {0.0 0.0}
check "pinned first end left out" [expr {[llength [tw::perp_sizes $big]] == 3 && abs([lindex [tw::perp_sizes $big] 0] - 1.0) < 1e-12}]
tw::set_param pin_first {}; tw::set_param evolve_ends 0
check "evolve_ends 0: both ends left out" [expr {[llength [tw::perp_sizes $big]] == 2 && abs([lindex [tw::perp_sizes $big] 1] - 2.0) < 1e-12}]
tw::set_param evolve_ends 1
tw::set_param pin_first {}; tw::set_param pin_last {}; tw::set_param pin_speed 0.002
tw::set_param calibrate 200; tw::set_param memory 200; tw::set_param stepsize 0

# ---- 3. the whole loop on a known surface, with noisy forces
set NOISE 4.0
proc run_string {z0 scale periodic shift maxit} {
    global NOISE
    # shift: added to CV 1 of the surface, to move the problem across +/-pi
    tw::init $z0 $scale $periodic
    set it 0; set first_perp -1; set stop 0
    while {$it < $maxit && !$stop} {
        set gl {}; set Ml {}
        foreach p [tw::targets] {
            lassign [gradV [lindex $p 0] [tw::wrap [expr {[lindex $p 1] - $shift}]]] gx gy
            lappend gl [list [expr {$gx + $NOISE * [gauss]}] [expr {$gy + $NOISE * [gauss]}]]
            lappend Ml {1.0 0.0 0.0 1.0}
        }
        tw::accumulate $gl $Ml
        set info [tw::advance]
        # a real replica would now run `factor` times as many MD steps per update
        if {[dict get $info factor] > 1} { set ::LAST_FACTOR [dict get $info factor]; set NOISE [expr {$NOISE / sqrt([dict get $info factor])}] }
        if {[dict get $info moved] && $first_perp < 0} { set first_perp [dict get $info perp] }
        incr it
        if {$it % 25 == 0} { lassign [tw::frame_force] fn fg; tw::record_frame [tw::frame_z] $fg; set stop [tw::converged] }
    }
    return [list $it $first_perp $stop]
}
# (plateau_span: this surface relaxes in a few thousand updates; the default
#  is for proteins and would hold these runs to the cap)
foreach {k v} {memory 100 calibrate 200 max_move 0.004 conv_window 8 conv_tol 0.06 conv_count 3 plateau_span 2000} { tw::set_param $k $v }
set z0 {}; for {set i 0} {$i < 14} {incr i} { set s [expr {$i / 13.0}]; lappend z0 [list [expr {-1 + 2 * $s}] [expr {0.9 * sin($PI * $s) * (1 + 0.3 * sin(9 * $s))}]] }
lassign [run_string $z0 {1.0 1.0} {0 0} 0.0 20000] it first_perp stop
check "stopped by its own drift test, not by the iteration cap (after $it updates)" [expr {$stop && $it < 20000}]
check "step is the smaller of the signal and noise limits" [expr {abs($tw::h - min($tw::h_signal, $tw::h_noise)) < 1e-15 && $tw::h_noise > 0 && $tw::h_signal > 0}]
check "first move no larger than requested ($first_perp)" [expr {$first_perp <= 0.004 + 1e-12}]
# the worst image's, and at a free end both directions count: 4 sqrt(2) = 5.7
check "measured force noise per update is about the 4.0 sqrt(2) put in at a free end ($tw::noise1)" [expr {$tw::noise1 > 4.5 && $tw::noise1 < 8.5}]
set worst 0.0; set top -1e9
foreach p [lrange [tw::final_string] 1 end-1] { set d [expr {abs(hypot([lindex $p 0], [lindex $p 1]) - 1.0)}]; if {$d > $worst} { set worst $d }; set v [V {*}$p]; if {$v > $top} { set top $v } }
check "interior images on the ring to within 0.12 (worst $worst)" [expr {$worst < 0.12}]
check "went over the top: barrier near +2 (highest image $top)" [expr {abs($top - 2.0) < 0.5}]
check "ends in the wells" [expr {hypot([lindex [tw::final_string] 0 0] + 1, [lindex [tw::final_string] 0 1]) < 0.1 && hypot([lindex [tw::final_string] end 0] - 1, [lindex [tw::final_string] end 1]) < 0.1}]
set ref [tw::final_string]

# ---- 4. the same problem moved so the path crosses +/-pi in CV 1
set shift [expr {$PI - 0.3}]
set z0s {}; foreach p $z0 { lappend z0s [list [lindex $p 0] [tw::wrap [expr {[lindex $p 1] + $shift}]]] }
lassign [run_string $z0s {1.0 1.0} {0 1} $shift 20000] it2 fp2 stop2
check "periodic: stopped by its drift test (after $it2 updates)" [expr {$stop2 && $it2 < 20000}]
set jump 0.0; for {set i 1} {$i < 14} {incr i} { set d [expr {abs([lindex [tw::final_string] $i 1] - [lindex [tw::final_string] [expr {$i-1}] 1])}]; if {$d > $jump} { set jump $d } }
check "periodic: string is continuous, no 2 pi jump (largest step $jump)" [expr {$jump < 1.0}]
set past 0; foreach p [tw::final_string] { if {[lindex $p 1] > $PI} { set past 1 } }
check "periodic: string runs past +pi unwrapped" $past
set worst 0.0
foreach p [tw::final_string] q $ref { set d [expr {hypot([lindex $p 0] - [lindex $q 0], [lindex $p 1] - $shift - [lindex $q 1])}]; if {$d > $worst} { set worst $d } }
check "periodic: same path as the unshifted problem to within 0.12 (worst $worst)" [expr {$worst < 0.12}]

# ---- 4b. heavy noise: the noise limit must bind, and the string must still settle
set NOISE 25.0
# this surface is stiff (forces of order 40 a unit off the ring): a residual
# of 2 is to it what the default 0.3 is to a protein's free-energy surface
foreach {k v} {frame_updates 25 force_tol 2.0} { tw::set_param $k $v }
lassign [run_string $z0 {1.0 1.0} {0 0} 0.0 60000] it3 fp3 stop3
check "heavy noise: the noise limit set the step ($tw::h_noise < $tw::h_signal)" [expr {$tw::h_noise < $tw::h_signal && abs($tw::h - $tw::h_noise) < 1e-15}]
check "heavy noise: measured noise is about the 25 sqrt(2) put in at a free end ($tw::noise1)" [expr {$tw::noise1 > 28 && $tw::noise1 < 53}]
check "heavy noise: still stops by its drift test (after $it3 updates)" [expr {$stop3 && $it3 < 60000}]
set worst 0.0
foreach p [lrange [tw::final_string] 1 end-1] { set d [expr {abs(hypot([lindex $p 0], [lindex $p 1]) - 1.0)}]; if {$d > $worst} { set worst $d } }
check "heavy noise: still on the ring to within 0.15 (worst $worst)" [expr {$worst < 0.15}]
# the signal-only rule on the same problem, for contrast: the step it would have used
check "heavy noise: the old signal-only rule would have stepped [format %.1f [expr {$tw::h_signal / $tw::h}]] times larger" [expr {$tw::h_signal / $tw::h > 2.0}]
set NOISE 4.0

# ---- 4c. slow is not still: a tiny hand-set step far from the path must NOT stop.
# (The drift test alone passed here on the cluster, a quarter of the way in.)
tw::set_param stepsize 0.00002; tw::set_param force_tol 0.3
lassign [run_string $z0 {1.0 1.0} {0 0} 0.0 3000] it4 fp4 stop4
set far 0.0; foreach p [lrange $tw::z 1 end-1] { set d [expr {abs(hypot([lindex $p 0], [lindex $p 1]) - 1.0)}]; if {$d > $far} { set far $d } }
check "slow string far from its path (off by $far) is not declared converged" [expr {!$stop4 && $far > 0.05}]
check "the force test is what refused it (residual force $tw::f_res against a tolerance of 0.3)" [expr {$tw::f_res > 0.3}]
tw::set_param stepsize 0; tw::set_param force_tol 2.0

# ---- 4d. noise too high for the longest memory: ask for more MD steps per update
set NOISE 60.0; set LAST_FACTOR 1
foreach {k v} {memory 100 memory_max 400 max_step_factor 10} { tw::set_param $k $v }
lassign [run_string $z0 {1.0 1.0} {0 0} 0.0 60000] it5 fp5 stop5
# (the driver no longer lengthens its memory or multiplies the MD steps per update:
#  the force is judged over the last window of frames, with its noise taken out)
check "very heavy noise: memory and steps per update left as set ($tw::P(memory), factor $LAST_FACTOR)" [expr {$tw::P(memory) == 100 && $LAST_FACTOR == 1}]
check "very heavy noise: still stops by its own tests (after $it5 updates)" [expr {$stop5 && $it5 < 60000}]
set worst 0.0
foreach p [lrange [tw::final_string] 1 end-1] { set d [expr {abs(hypot([lindex $p 0], [lindex $p 1]) - 1.0)}]; if {$d > $worst} { set worst $d } }
# (noise per update one and a half times the surface's own force scale, and no extra MD asked for any more)
check "very heavy noise: still on the ring to within 0.25 (worst $worst)" [expr {$worst < 0.25}]
set NOISE 4.0
foreach {k v} {memory_max 10000} { tw::set_param $k $v }

# ---- 5. state survives a restart
set saved [tw::dump_state]; set zb $tw::z; set hb $tw::h; set nb $tw::nacc
tw::init {{0 0} {1 1} {2 2}} {1 1} {0 0}
eval $saved
check "state restored" [expr {$tw::z eq $zb && $tw::h == $hb && $tw::nacc == $nb && $tw::N == 14}]
check "state restored: the frames, their gradients and profiles came back too" [expr {[llength $tw::frames] > 0 && [llength $tw::gframes] == [llength $tw::frames] && [llength $tw::prof] == [llength $tw::frames]}]

# ---- 5b. nothing in the driver may use expr's min() or max(): NAMD's Tcl
# does not have them (its unknown-command handler swallows the call), which
# is invisible under tclsh and stalled two cluster jobs.
set bad 0
foreach f {string_math.tcl string_driver.tcl} {
    set fh [open [file join [file dirname [info script]] .. driver $f]]
    foreach line [split [read $fh] \n] {
        set code [regsub {#.*$} $line ""]
        if {[regexp {\m(min|max)\(} $code]} { incr bad; puts "  $f: $line" }
    }
    close $fh
}
check "no min() or max() in driver code" [expr {$bad == 0}]

# ---- 5b. the string moves on the force of the moment; its AVERAGE is the result
# A straight valley V = y^2/2 along x, the string 0.1 above its floor, little
# noise.  Stepping on the averaged force (the old way) is the heavy-ball
# method: the string coasts past the floor to about -0.1 before turning.
proc valley {polyak} {
    foreach {k v} {memory 200 calibrate 200 max_move 0.002 conv_window 20 conv_tol 0.1 conv_count 3 frame_updates 100 stepsize 0 nblocks 10 force_tol 0.3 profile_tol 0.5} { tw::set_param $k $v }
    tw::set_param polyak $polyak
    expr {srand(3)}
    set z0 {}; for {set i 0} {$i < 8} {incr i} { lappend z0 [list $i 0.1] }
    tw::init $z0 {1 1} {0 0}
    set ymin 1e9; set stop 0
    for {set it 0} {$it < 30000 && !$stop} {incr it} {
        set gl {}; set Ml {}
        foreach p $tw::z { lappend gl [list [expr {0.02 * [gauss]}] [expr {[lindex $p 1] + 0.02 * [gauss]}]]; lappend Ml {1 0 0 1} }
        tw::accumulate $gl $Ml; tw::advance
        if {($it + 1) % 100 == 0} {
            lassign [tw::frame_force] fn fg; set fz [tw::frame_z]
            set y 0.0; foreach p $fz { set y [expr {$y + [lindex $p 1] / 8.0}] }
            if {$y < $ymin} { set ymin $y }
            tw::record_frame $fz $fg; set stop [tw::converged]
        }
    }
    set y 0.0; foreach p [tw::final_string] { set y [expr {$y + [lindex $p 1] / 8.0}] }
    return [list $ymin $stop $y]
}
lassign [valley 0] ymin_old stop_old yfin_old
lassign [valley 1] ymin_new stop_new yfin_new
check "force averaging overshoots the valley floor (lowest $ymin_old)" [expr {$ymin_old < -0.05}]
check "position averaging does not (lowest $ymin_new)" [expr {$ymin_new > -0.01}]
check "it stops by its own tests, on the floor (final string at $yfin_new)" [expr {$stop_new && abs($yfin_new) < 0.005}]
check "the residual force is in force units and small ($tw::f_res)" [expr {$tw::f_res >= 0 && $tw::f_res < 0.05}]
check "the profile test saw data (change $tw::p_change)" [expr {$tw::p_change >= 0 && $tw::p_change < 0.5}]
# the profile along a string from its gradients: F = y^2/2 along a line in y
set zz {}; set gg {}
for {set i 0} {$i <= 10} {incr i} { set y [expr {0.2 * $i}]; lappend zz [list 0.0 $y]; lappend gg [list 0.0 $y] }
check "trapezoid profile of a linear force is exact" [expr {abs([lindex [tw::profile $zz $gg] end] - 2.0) < 1e-12}]
# a profile still changing between windows blocks the stop, whatever the drift and the force say
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
foreach {k v} {calibrate 2 memory 2 stepsize 1e-9 conv_window 2 conv_count 1 profile_tol 0.5 force_tol 0.3 polyak 1 frame_updates 5 plateau 0} { tw::set_param $k $v }
set I {1 0 0 1}
for {set it 0} {$it < 40} {incr it} {
    set gx [expr {$it < 20 ? 1.0 : 0.0}]                    ;# the force along the string changes half way: F(end) 3 -> 0
    tw::accumulate [list [list $gx 1e-6] [list $gx 1e-6] [list $gx 1e-6] [list $gx 1e-6]] [list $I $I $I $I]; tw::advance
    if {($it + 1) % 5 == 0} { lassign [tw::frame_force] fn fg; tw::record_frame [tw::frame_z] $fg
        if {$it == 29} { set mid_ok [tw::converged]; set mid_change $tw::p_change } }
}
check "a changing profile blocks the stop (change $mid_change)" [expr {!$mid_ok && $mid_change > 0.5}]
check "and a settled one allows it (change $tw::p_change)" [expr {[tw::converged] && $tw::p_change < 0.5}]
foreach {k v} {calibrate 200 memory 200 stepsize 0 conv_window 20 conv_count 3} { tw::set_param $k $v }
tw::set_param force_tol 0.3; tw::set_param frame_updates 100; tw::set_param plateau 1

# ---- 5c. the residual force is a FORCE, whatever the metric tensor looks like
check "solve: 3x3 system" [expr {abs([lindex [tw::solve {2 1 0  1 3 1  0 1 4} {3 5 5}] 0] - 1.0) < 1e-12 && abs([lindex [tw::solve {2 1 0  1 3 1  0 1 4} {3 5 5}] 2] - 1.0) < 1e-12}]
# a string along x; the same gradient of 2 across it at every image; the
# mobility across the string is 1 in one run and 50 in the other
foreach mob {1.0 50.0} {
    tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
    foreach {k v} {calibrate 2 memory 2 stepsize 1e-12 conv_window 2 polyak 1} { tw::set_param $k $v }
    set Mm [list 1.0 0.0 0.0 $mob]
    for {set it 0} {$it < 6} {incr it} { tw::accumulate {{0 2.0} {0 2.0} {0 2.0} {0 2.0}} [list $Mm $Mm $Mm $Mm]; tw::advance }
    set r [tw::residual_forces]
    check "residual force is 2 with an across-string mobility of $mob ([lindex $r 0])" [expr {abs([lindex $r 0] - 2.0) < 1e-6}]
}
foreach {k v} {calibrate 200 memory 200 stepsize 0 conv_window 20} { tw::set_param $k $v }

# ---- 5d. on an exact MFEP with exact forces the residual is (nearly) nothing, bends and coupled CVs included
# F = 3 (x^2 - 1)^2 + 10 (y - 0.6 (1 - x^2))^2; with M = identity the MFEP is the valley floor y = 0.6 (1 - x^2)
# only approximately, so the true path is found by relaxing a fine noise-free string first.
proc gF {x y} { set c [expr {$y - 0.6 * (1 - $x * $x)}]; return [list [expr {12.0 * $x * ($x * $x - 1) + 20.0 * $c * 1.2 * $x}] [expr {20.0 * $c}]] }
proc Mat {x} { set ph [expr {0.5 + 0.8 * $x}]; set c [expr {cos($ph)}]; set s [expr {sin($ph)}]; set A 8.0
    return [list [expr {$c * $c + $A * $s * $s}] [expr {(1 - $A) * $c * $s}] [expr {(1 - $A) * $c * $s}] [expr {$s * $s + $A * $c * $c}]] }
set z0 {}; for {set i 0} {$i < 16} {incr i} { set xx [expr {-1.0 + 2.0 * $i / 15.0}]; lappend z0 [list $xx [expr {0.6 * (1 - $xx * $xx) + 0.3}]] }
tw::init $z0 {1.0 1.0} {0 0}
foreach {k v} {calibrate 20 memory 20 stepsize 0.002 conv_window 4 frame_updates 50 polyak 1 force_tol 0.3 profile_tol 0 conv_count 3 plateau_span 1000} { tw::set_param $k $v }
for {set it 0} {$it < 6000} {incr it} {
    set gl {}; set Ml {}
    foreach p $tw::z { lappend gl [gF {*}$p]; lappend Ml [Mat [lindex $p 0]] }
    tw::accumulate $gl $Ml; tw::advance
    if {($it + 1) % 50 == 0} { lassign [tw::frame_force] fn fg; tw::record_frame [tw::frame_z] $fg }
}
check "exact forces, curved path, coupled tensor: the string has stopped (drift test and force test both pass)" [tw::converged]
check "the residual on the converged string is small ($tw::f_res; the chord-tangent version left 0.4 here)" [expr {$tw::f_res < 0.15}]
# a force of 1 put straight across a straight string reads as 1 however strongly the two CVs are coupled
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
foreach {k v} {calibrate 2 memory 2 stepsize 1e-12} { tw::set_param $k $v }
foreach off {0.0 0.9 0.99} {
    set Mm [list 1.0 $off $off 1.0]
    tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
    for {set it 0} {$it < 6} {incr it} { tw::accumulate {{0 1.0} {0 1.0} {0 1.0} {0 1.0}} [list $Mm $Mm $Mm $Mm]; tw::advance }
    set r [lindex [tw::residual_forces] 0]
    check "a unit force never reads as more than 1 (coupling $off: $r)" [expr {$r <= 1.0 + 1e-9 && $r > 0.1}]
}
foreach {k v} {calibrate 200 memory 200 stepsize 0 conv_window 20 frame_updates 100 profile_tol 0.5} { tw::set_param $k $v }

# ---- 5e. drift as a curve: sliding along the path is not movement, moving across it is
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
# a quarter circle, and the same quarter circle with its inner images slid along it
proc arcpts {angles {dr 0.0}} { set o {}; foreach a $angles { lappend o [list [expr {(2.0 + $dr) * cos($a)}] [expr {(2.0 + $dr) * sin($a)}]] }; return $o }
set A [arcpts {0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4}]
set Bslide [arcpts {0.0 0.3 0.5 0.7 0.9 1.1 1.3 1.4}]
set dslide [tw::curve_distance $A $Bslide]
check "images slid 0.2 along the path: the curves agree to the spline's own error ($dslide)" [expr {$dslide < 0.01}]
set Bout [arcpts {0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4} 0.15]
set dout [tw::curve_distance $A $Bout]
check "a string moved 0.15 across its path is 0.15 away ($dout)" [expr {abs($dout - 0.15) < 0.005}]
set Bboth [arcpts {0.0 0.3 0.5 0.7 0.9 1.1 1.3 1.4} 0.15]
set dboth [tw::curve_distance $A $Bboth]
check "slid and moved across: only the move across counts ($dboth)" [expr {abs($dboth - 0.15) < 0.005}]
set Bgrow [arcpts {0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.5}]
set dgrow [tw::curve_distance $A $Bgrow]
check "an end that has grown 0.2 along the path is counted ($dgrow)" [expr {abs($dgrow - 0.2) < 0.01}]
check "one bulged image is seen ([tw::curve_distance $A [lreplace $A 3 3 [list [expr {2.3 * cos(0.6)}] [expr {2.3 * sin(0.6)}]]]])" \
    [expr {[tw::curve_distance $A [lreplace $A 3 3 [list [expr {2.3 * cos(0.6)}] [expr {2.3 * sin(0.6)}]]]] > 0.25}]
# the stopping test itself: frames that slide pass on the curve and fail per image; frames that move across fail both
proc feed {first second} {
    tw::init $first {1.0 1.0} {0 0}
    foreach {k v} {calibrate 0 conv_window 2 conv_count 1 force_tol 0 profile_tol 0 frame_updates 1 plateau 0} { tw::set_param $k $v }
    set tw::nacc 100
    foreach f [list $first $first $second $second] { set d [tw::record_frame $f {}] }
    return $d
}
feed $A $Bslide
check "sliding: curve drift passes ($tw::d_curve), image drift would not ($tw::d_image)" [expr {[tw::converged] && $tw::d_curve < 0.01 && $tw::d_image > 0.15}]
tw::set_param drift_curve 0; feed $A $Bslide; set old_blocks [expr {![tw::converged]}]; tw::set_param drift_curve 1
check "drift_curve 0 restores the per-image test, which blocks on sliding" $old_blocks
feed $A $Bout
check "moving across: the curve drift blocks the stop ($tw::d_curve)" [expr {![tw::converged] && $tw::d_curve > 0.1}]
feed $A $Bgrow
check "a growing end blocks the stop ($tw::d_curve)" [expr {![tw::converged]}]
foreach {k v} {calibrate 200 memory 200 stepsize 0 conv_window 20 conv_count 3 frame_updates 100 profile_tol 0.5 force_tol 0.3 plateau 1} { tw::set_param $k $v }

# ---- 5f. the plateau test: a resting string that wanders passes; one that creeps under the same wander does not
# 12 images on a line in x; every frame each image is displaced across the line by slow noise
# (an AR(1) wander) and, in the second case, also by a steady creep of the far half of the string
proc wander_run {creep nfr} {
    tw::init {{0 0} {1 0} {2 0} {3 0} {4 0} {5 0} {6 0} {7 0} {8 0} {9 0} {10 0} {11 0}} {1.0 1.0} {0 0}
    foreach {k v} {calibrate 0 conv_window 20 conv_count 1 force_tol 0 profile_tol 0 frame_updates 1 plateau 1 plateau_tol 0.2 plateau_span 200} { tw::set_param $k $v }
    set tw::nacc 1000
    set w [lrepeat 12 0.0]; set short {}; set sys {}; set stopped 0
    for {set f 0} {$f < $nfr} {incr f} {
        set nw {}; set fr {}; set i 0
        foreach y $w {
            set y [expr {0.8 * $y + 0.04 * [gauss]}]; lappend nw $y
            lappend fr [list $i.0 [expr {$y + ($i >= 6 ? $creep * $f : 0.0)}]]; incr i
        }
        set w $nw
        set d [tw::record_frame $fr {}]
        # (counts evaluations whose plateau statistic and wander both pass; the stop also needs the hold)
        if {$tw::d_sys >= 0} { lappend short $d; lappend sys $tw::d_sys; if {$tw::d_sys < 0.2 && $tw::d_floor < 0.2} { incr stopped } }
    }
    set ms 0.0; foreach v $short { set ms [expr {$ms + $v}] }
    set tail [lrange $sys end-49 end]; set my 0.0; foreach v $tail { set my [expr {$my + $v}] }
    return [list [expr {$ms / [llength $short]}] [expr {$my / [llength $tail]}] [expr {double($stopped) / [llength $sys]}]]
}
expr {srand(11)}
lassign [wander_run 0.0 600] rest_short rest_sys rest_frac
lassign [wander_run 0.004 600] creep_short creep_sys creep_frac
check "resting but wandering: the plateau test passes most of the time (systematic $rest_sys, passes [format %.2f $rest_frac])" [expr {$rest_sys < 0.2 && $rest_frac > 0.8}]
check "creeping 0.4 per 100 frames: the plateau test never passes (systematic $creep_sys, passes $creep_frac)" [expr {$creep_sys > 0.2 && $creep_frac == 0.0}]
check "the neighbouring-window drift cannot tell them apart (resting $rest_short, creeping $creep_short)" [expr {abs($creep_short - $rest_short) < 0.5 * $rest_short}]
tw::init {{0 0} {1 0} {2 0} {3 0}} {1.0 1.0} {0 0}
foreach {k v} {conv_window 5 frame_updates 1 calibrate 0} { tw::set_param $k $v }
set tw::nacc 1000
for {set f 0} {$f < 39} {incr f} { tw::record_frame {{0 0} {1 0} {2 0} {3 0}} {} }
check "too little history: no verdict and no stop (39 of 40 frames)" [expr {$tw::d_sys < 0 && ![tw::converged]}]
foreach {k v} {calibrate 200 memory 200 stepsize 0 conv_window 20 conv_count 3 frame_updates 100 profile_tol 0.5 force_tol 0.3 plateau 1 plateau_tol 0.2} { tw::set_param $k $v }

# ---- 5g. the two flows stop at the same string when the metric tensor is far from the identity
proc run_flow {flow maxit} {
    global z0 NOISE
    tw::init $z0 {1.0 1.0} {0 0}
    foreach {k v} [list flow $flow memory 100 calibrate 200 max_move 0.004 conv_window 8 conv_count 3 force_tol 0 profile_tol 0 frame_updates 25 pin_first {-1.0 0.0} pin_last {1.0 0.0}] { tw::set_param $k $v }
    set it 0; set stop 0
    while {$it < $maxit && !$stop} {
        set gl {}; set Ml {}
        foreach p [tw::targets] {
            lassign [gradV {*}$p] gx gy
            lappend gl [list [expr {$gx + $NOISE * [gauss]}] [expr {$gy + $NOISE * [gauss]}]]
            lappend Ml {1.0 0.6 0.6 30.0}                       ;# slow along x, fast along y, coupled
        }
        tw::accumulate $gl $Ml; tw::advance; incr it
        if {$it % 25 == 0} { lassign [tw::frame_force] fn fg; tw::record_frame [tw::frame_z] $fg; set stop [tw::converged] }
    }
    return [list [tw::final_string] $it $stop $tw::h]
}
expr {srand(5)}
lassign [run_flow metric 30000] s_metric it_m stop_m h_m
lassign [run_flow residual 30000] s_resid it_r stop_r h_r
set dflow [tw::curve_distance $s_metric $s_resid]
check "metric flow and residual flow end on the same string (apart by $dflow; $it_m and $it_r updates)" [expr {$dflow < 0.12}]
check "both came to rest by the plateau test" [expr {$stop_m && $stop_r}]
check "the residual flow's step is not held down by the fast direction (step $h_r against $h_m)" [expr {$h_r > 3.0 * $h_m}]
foreach {k v} {flow metric pin_first {} pin_last {} force_tol 0.3 profile_tol 0.5 frame_updates 100 conv_window 20 max_move 0.002 memory 200} { tw::set_param $k $v }

# ---- 5h. the history stays bounded, and the plateau test reads the same from it
proc wander_bounded {creep nfr cap} {
    tw::init {{0 0} {1 0} {2 0} {3 0} {4 0} {5 0} {6 0} {7 0} {8 0} {9 0} {10 0} {11 0}} {1.0 1.0} {0 0}
    foreach {k v} [list calibrate 0 conv_window 20 conv_count 1 force_tol 0 profile_tol 0 frame_updates 1 plateau 1 plateau_tol 0.2 hist_max $cap] { tw::set_param $k $v }
    set tw::nacc 1000
    set w [lrepeat 12 0.0]; set big 0
    for {set f 0} {$f < $nfr} {incr f} {
        set nw {}; set fr {}; set i 0
        foreach y $w { set y [expr {0.8 * $y + 0.04 * [gauss]}]; lappend nw $y; lappend fr [list $i.0 [expr {$y + ($i >= 6 ? $creep * $f : 0.0)}]]; incr i }
        set w $nw; tw::record_frame $fr {}
        if {[llength $tw::zhist] > $big} { set big [llength $tw::zhist] }
    }
    return [list $tw::d_sys $big $tw::hist_chunk]
}
expr {srand(11)}
# (the cap is at least 64 entries now: the plateau test's sixteen half-blocks need two each)
lassign [wander_bounded 0.0 1500 64] rest_b big_b chunk_b
lassign [wander_bounded 0.002 1500 64] creep_b big_c chunk_c
check "1500 frames kept in at most 64 entries (most $big_b, $chunk_b frames each)" [expr {$big_b <= 64 && $chunk_b >= 32}]
check "bounded history: at rest reads $rest_b, creeping reads $creep_b" [expr {$rest_b < 0.2 && $creep_b > 0.2}]
foreach {k v} {calibrate 200 conv_window 20 conv_count 3 frame_updates 100 profile_tol 0.5 force_tol 0.3 hist_max 512} { tw::set_param $k $v }

# ---- 5i. the step shrinks only after the string is at rest
foreach {k v} {memory 100 calibrate 200 max_move 0.004 conv_window 8 conv_tol 0.06 conv_count 1000000 step_decay 0.6 force_tol 0 profile_tol 0 frame_updates 25} { tw::set_param $k $v }
expr {srand(3)}
tw::init $z0 {1.0 1.0} {0 0}
set h_before -1; set h_first -1
for {set it 1} {$it <= 12000} {incr it} {
    set gl {}; set Ml {}
    foreach p [tw::targets] { lassign [gradV {*}$p] gx gy; lappend gl [list [expr {$gx + 4.0 * [gauss]}] [expr {$gy + 4.0 * [gauss]}]]; lappend Ml {1.0 0.0 0.0 1.0} }
    tw::accumulate $gl $Ml; tw::advance
    if {$it == 300} { set h_first $tw::h }
    if {$tw::n_rest < 0} { set h_before $tw::h }
    if {$it % 25 == 0} { lassign [tw::frame_force] fn fg; tw::record_frame [tw::frame_z] $fg }
}
check "step constant until the string is at rest (rest found at update $tw::n_rest)" [expr {$tw::n_rest > 300 && abs($h_before - $h_first) < 1e-15}]
check "and smaller afterwards ($tw::h against $tw::h0)" [expr {$tw::h < 0.9 * $tw::h0 && abs($tw::h - $tw::h0 * pow(double($tw::n_rest) / $tw::nacc, 0.6)) < 1e-12}]
foreach {k v} {memory 200 calibrate 200 max_move 0.002 conv_window 20 conv_tol 0.1 conv_count 3 step_decay 0.0 force_tol 0.3 profile_tol 0.5 frame_updates 100} { tw::set_param $k $v }

# ---- 5j. calibration noise from blocks that are too short is seen to be too small
expr {srand(9)}
proc blocks_of {phi nblk blen} {
    # a series with correlation phi between neighbours, variance 1, cut into block means
    set x 0.0; set out {}
    for {set b 0} {$b < $nblk} {incr b} {
        set sum 0.0
        for {set i 0} {$i < $blen} {incr i} { set x [expr {$phi * $x + sqrt(1 - $phi * $phi) * [gauss]}]; set sum [expr {$sum + $x}] }
        lappend out [list [list [expr {$sum / $blen}]]]
    }
    return $out
}
set r_white 0.0; set r_corr 0.0
for {set rep 0} {$rep < 200} {incr rep} {
    lassign [tw::block_noise [blocks_of 0.0 10 20] [lrepeat 10 20]] a b; set r_white [expr {$r_white + $b / $a / 200.0}]
    lassign [tw::block_noise [blocks_of 0.97 10 20] [lrepeat 10 20]] a b; set r_corr [expr {$r_corr + $b / $a / 200.0}]
}
check "independent samples: longer blocks give the same noise (ratio $r_white)" [expr {abs($r_white - 1.0) < 0.12}]
check "samples correlated over 30 updates, blocks of 20: longer blocks give more (ratio $r_corr)" [expr {$r_corr > 1.2}]

# ==== 7. review round 3: each test below failed on 2891a69 for the reason named
set TDIR [file join [file dirname [info script]] .. driver]
proc reset_defaults {} {
    foreach {k v} {memory 200 calibrate 200 max_move 0.002 stepsize 0 spline 1 evolve_ends 1 conv_window 20 conv_tol 0.1
                   conv_count 3 polyak 1 force_tol 0.3 profile_tol 0.5 pin_first {} pin_last {} noise_frac 0.5 drift_curve 1
                   plateau 1 plateau_tol 0.2 plateau_every 5 hist_max 512 curve_reach 4 flow residual plateau_hold 0.25
                   step_decay 0.0 frame_updates 100 nblocks 10 plateau_span 10000 force_image_tol -1} { tw::set_param $k $v }
}
# a string of 12 images on the x axis, fed frames directly (no forces unless given);
# mode: all = every image creeps across; end = the last image creeps across;
# endalong = the last image creeps along; osc = all oscillate; hosc = the far half
# oscillates; rw = the far half random-walks; noise is AR(1) wander on every image.
proc hrun {mode rate noise nfr {extra {}} {T 100}} {
    set z0 {}; for {set i 0} {$i < 12} {incr i} { lappend z0 [list $i.0 0.0] }
    tw::init $z0 {1.0 1.0} {0 0}
    reset_defaults
    foreach {k v} [concat {calibrate 0 conv_window 20 conv_count 3 force_tol 0 profile_tol 0 frame_updates 20} $extra] { tw::set_param $k $v }
    set tw::nacc 1000
    set w [lrepeat 12 0.0]; set rw 0.0
    for {set f 0} {$f < $nfr} {incr f} {
        if {$mode eq "rw"} { set rw [expr {$rw + $rate * [gauss]}] }
        set nw {}; set fr {}; set i 0
        foreach y $w {
            set y [expr {0.8 * $y + $noise * [gauss]}]; lappend nw $y
            set x $i.0; set yy $y
            switch $mode {
                all { set yy [expr {$y + $rate * $f}] }
                end { if {$i == 11} { set yy [expr {$y + $rate * $f}] } }
                endalong { if {$i == 11} { set x [expr {$x + $rate * $f}] } }
                osc { set yy [expr {$y + $rate * sin(6.283185307179586 * $f / $T)}] }
                hosc { if {$i >= 6} { set yy [expr {$y + $rate * sin(6.283185307179586 * $f / $T)}] } }
                rw { if {$i >= 6} { set yy [expr {$y + $rw}] } }
            }
            lappend fr [list $x $yy]; incr i
        }
        set w $nw
        tw::record_frame $fr {}
        if {[tw::converged]} { return $f }
    }
    return -1
}

# ---- 7.1 free ends under the residual flow slide into their basins (finding 1)
# F = 2 (x^2 - 1)^2 + 5 y^2: minima at x = +-1; the ends start half way up the walls
proc gEnds {x y} { return [list [expr {8.0 * $x * ($x * $x - 1)}] [expr {10.0 * $y}]] }
foreach nz {0.0 3.0} {
    reset_defaults; expr {srand(1)}
    set z0 {}; for {set i 0} {$i < 8} {incr i} { lappend z0 [list [expr {-0.5 + $i / 7.0}] 0.05] }
    tw::init $z0 {1 1} {0 0}
    foreach {k v} {calibrate 2 memory 2 stepsize 0.01} { tw::set_param $k $v }
    set xa 0.0; set xb 0.0; set nav 0
    for {set it 0} {$it < 4000} {incr it} {
        set gl {}; set Ml {}
        foreach p $tw::z { lassign [gEnds {*}$p] gx gy; lappend gl [list [expr {$gx + $nz * [gauss]}] [expr {$gy + $nz * [gauss]}]]; lappend Ml {1 0 0 1} }
        tw::accumulate $gl $Ml; tw::advance
        if {$it >= 2000} { set xa [expr {$xa + [lindex $tw::z 0 0]}]; set xb [expr {$xb + [lindex $tw::z end 0]}]; incr nav }
    }
    set xa [expr {$xa / $nav}]; set xb [expr {$xb / $nav}]
    check "7.1 residual flow, noise $nz: free ends reach the minima at -1 and 1 (mean $xa, $xb)" [expr {abs($xa + 1) < 0.08 && abs($xb - 1) < 0.08}]
}

# ---- 7.2 a hand-set step needs no measurable force; a free end's move along the path counts (finding 7)
reset_defaults
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
foreach {k v} {calibrate 2 memory 2 stepsize 0.001} { tw::set_param $k $v }
set err [catch { for {set it 0} {$it < 4} {incr it} { tw::accumulate [lrepeat 4 {0.0 0.0}] [lrepeat 4 {1 0 0 1}]; tw::advance } } msg]
check "7.2 stepsize set by hand and zero force: no error ($msg)" [expr {!$err && $tw::h == 0.001}]
set ps [tw::perp_sizes {{1.0 0.0} {1.0 0.0} {1.0 0.0} {1.0 0.0}}]
check "7.2 a free end's move along the path counts in the step size ($ps)" [expr {abs([lindex $ps 0] - 1.0) < 1e-12 && abs([lindex $ps end] - 1.0) < 1e-12 && [lindex $ps 1] < 1e-12}]

# ---- 7.3 the restart state carries the flow and the step's parameters; another flow is refused (finding 2)
reset_defaults
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0} {4.0 0.0}} {1.0 1.0} {0 0}
foreach {k v} {calibrate 2 memory 2 flow residual} { tw::set_param $k $v }
for {set it 0} {$it < 3} {incr it} { tw::accumulate [lrepeat 5 {0.0 1.0}] [lrepeat 5 {1 0 0 250}]; tw::advance }
set sres [tw::dump_state]
check "7.3 the state records the flow it was calibrated under" [regexp {flow residual} $sres]
catch {tw::load_state $sres}; tw::set_param flow metric
set err [catch {tw::restart_check} msg]
check "7.3 a residual-calibrated state restarted under flow metric is refused ($msg)" [expr {$err && [string match "*flow*residual*" $msg]}]
tw::set_param flow residual
set err [catch {tw::restart_check} msg]
check "7.3 the same flow is accepted ($msg)" [expr {!$err}]
# a state from before TW(flow) existed ran the metric flow
set sold {}; foreach line [split $sres \n] { if {![string match "*tw::saved_P*" $line]} { append sold $line \n } }
catch {tw::load_state $sold}; tw::set_param flow residual
set err [catch {tw::restart_check} msg]
check "7.3 an old state (no flow recorded) is taken as metric: residual refused ($msg)" [expr {$err && [string match "*metric*" $msg]}]
tw::set_param flow metric
set err [catch {tw::restart_check} msg]
check "7.3 an old state under flow metric is accepted ($msg)" [expr {!$err}]
catch {tw::load_state $sres}; tw::set_param flow residual; tw::set_param calibrate 50
set err [catch {tw::restart_check} msg]
check "7.3 a different calibrate is refused ($msg)" [expr {$err && [string match "*calibrate*" $msg]}]
tw::set_param calibrate 2; tw::set_param max_move 0.004
set err [catch {tw::restart_check} msg]
check "7.3 a different max_move is reported, not refused ($msg)" [expr {!$err && [string match "*max_move*" $msg]}]

# ---- 7.4 the plateau test (finding 3).  Frames of 20 updates and the default
# plateau_span, so the test reads at least 10,000 updates (500 frames).
set f3 {}
foreach {lab mode rate noise T} {
    "every image creeping 0.001 per frame"            all 0.001 0.04 100
    "far half oscillating, amplitude 0.3, period 120" hosc 0.3 0.04 120
    "all oscillating, amplitude 0.5, period 107"      osc 0.5 0.04 107
    "far half random-walking, 0.012 per frame"        rw 0.012 0.04 100
    "end image creeping across, 0.002 per frame"      end 0.002 0.04 100
    "end image creeping along, 0.003 per frame"       endalong 0.003 0.04 100
} {
    set stops {}
    foreach seed {1 2} { expr {srand($seed)}; lappend stops [hrun $mode $rate $noise 1500 {} $T] }
    check "7.4 $lab: never stops in 1500 frames (stop frames $stops)" [expr {[lindex $stops 0] < 0 && [lindex $stops 1] < 0}]
}
set stops {}
foreach seed {1 2} { expr {srand($seed)}; lappend stops [hrun all 0.0 0.04 2000] }
check "7.4 a resting string with wander 0.067 rms still stops (stop frames $stops)" [expr {[lindex $stops 0] > 0 && [lindex $stops 1] > 0}]

# ---- 7.5 the step decays only after the full stop criteria held, and is restored if the string moves on (finding 4)
proc strline {y} { set fr {}; for {set i 0} {$i < 12} {incr i} { lappend fr [list $i.0 [expr {$i >= 6 ? $y : 0.0}]] }; return $fr }
reset_defaults
tw::init [strline 0] {1.0 1.0} {0 0}
foreach {k v} {calibrate 0 conv_window 20 conv_count 1000 force_tol 0 profile_tol 0 frame_updates 20 step_decay 0.6} { tw::set_param $k $v }
set tw::nacc 1000; set first_pass -1; set rest_at_first -2
for {set f 0} {$f < 1600} {incr f} {
    incr tw::nacc 20; tw::record_frame [strline 0.0] {}
    if {$tw::nplat == 1 && $first_pass < 0} { set first_pass $f; set rest_at_first $tw::n_rest }
}
check "7.5 a single plateau pass does not start the decay (n_rest $rest_at_first at the first pass, frame $first_pass)" [expr {$first_pass > 0 && $rest_at_first < 0}]
check "7.5 the decay starts once the pass has held (n_rest $tw::n_rest)" [expr {$tw::n_rest > 0}]
for {set f 0} {$f < 800} {incr f} { incr tw::nacc 20; tw::record_frame [strline [expr {0.01 * $f}]] {} }
check "7.5 a string that moves again resets it (systematic $tw::d_sys, n_rest $tw::n_rest)" [expr {$tw::d_sys > 0.2 && $tw::n_rest < 0}]
reset_defaults
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
foreach {k v} {calibrate 2 memory 2 stepsize 0.001 step_decay 0.6} { tw::set_param $k $v }
for {set it 0} {$it < 3} {incr it} { tw::accumulate [lrepeat 4 {0.0 1.0}] [lrepeat 4 {1 0 0 1}]; tw::advance }
set tw::h 0.0001; set tw::n_rest -1
tw::accumulate [lrepeat 4 {0.0 1.0}] [lrepeat 4 {1 0 0 1}]; tw::advance
check "7.5 with the rest point reset the step is back to h0 ($tw::h, h0 $tw::h0)" [expr {abs($tw::h - $tw::h0) < 1e-15}]

# ---- 7.6 calibration noise is the worst image's, as promised (finding 5)
expr {srand(4)}; set vecs {}
for {set b 0} {$b < 10} {incr b} { set v {}; for {set i 0} {$i < 24} {incr i} { set s [expr {$i == 5 ? 10.0 : 1.0}]; lappend v [list [expr {$s * [gauss] / sqrt(20)}]] }; lappend vecs $v }
set bn [lindex [tw::block_noise $vecs [lrepeat 10 20]] 0]
check "7.6 one image with ten times the noise sets the noise ($bn; that image's own is about 10)" [expr {$bn > 6.0}]
check "7.6 the unused noise_f1 is gone" [expr {![info exists tw::noise_f1]}]

# ---- 7.7 the force test: one image may not carry the whole tolerance; no forces is no pass (finding 6)
proc frun {G nfr} {
    reset_defaults
    set z0 {}; for {set i 0} {$i < 12} {incr i} { lappend z0 [list $i.0 0.0] }
    tw::init $z0 {1.0 1.0} {0 0}
    foreach {k v} {calibrate 0 conv_window 20 force_tol 0.3 profile_tol 0 frame_updates 20} { tw::set_param $k $v }
    tw::accumulate [lrepeat 12 {0.0 0.0}] [lrepeat 12 {1.0 0.0 0.0 1.0}]
    set tw::nacc 1000
    for {set f 0} {$f < $nfr} {incr f} {
        set fr {}; set gf {}
        for {set i 0} {$i < 12} {incr i} { lappend fr [list $i.0 [expr {0.01 * [gauss]}]]; lappend gf [list 0.0 [expr {($i == 5 ? $G : 0.0) + 0.05 * [gauss]}]] }
        if {$G < 0} { set gf {} }
        tw::record_frame $fr $gf
        if {[tw::converged]} { return $f }
    }
    return -1
}
expr {srand(1)}
set s1 [frun 1.0 1600]
check "7.7 one image of 12 with a force of 1.0 across the string does not stop (stop $s1; rms over images $tw::fl_clean)" [expr {$s1 < 0}]
set s0 [frun 0.5 1600]
check "7.7 one image with 0.5 still stops (stop $s0)" [expr {$s0 > 0}]
set sn [frun -1 1600]
check "7.7 no forces recorded: the force test cannot pass (stop $sn, long residual $tw::fl_res)" [expr {$sn < 0}]

# ---- 7.8 a run stops only on a frame on which the tests were evaluated (finding 8)
reset_defaults
tw::init [strline 0] {1.0 1.0} {0 0}
foreach {k v} {calibrate 0 conv_window 21 conv_count 1 force_tol 0 profile_tol 0 frame_updates 20} { tw::set_param $k $v }
set tw::nacc 1000; set ok8 -1
for {set f 0} {$f < 2000} {incr f} {
    incr tw::nacc 20; tw::record_frame [strline 0.0] {}
    if {[tw::converged]} { set ok8 [expr {$tw::h_since == 0}]; break }
}
check "7.8 the stopping frame was an evaluated one (stopped at $f)" [expr {$ok8 == 1}]

# ---- 7.9 merging the history keeps the newest frames and gets under the cap (finding 9)
reset_defaults
tw::init [strline 0] {1.0 1.0} {0 0}
set tw::zhist {}; set tw::ghist {}
for {set j 0} {$j < 70} {incr j} { lappend tw::zhist [strline $j]; lappend tw::ghist {} }
tw::set_param hist_max 64
tw::history_add [strline 70] {}
tw::history_add [strline 71] {}
check "7.9 an odd-length merge keeps the newest frame (last entry [lindex $tw::zhist end end 1], expected 70.5)" [expr {abs([lindex $tw::zhist end end 1] - 70.5) < 1e-9}]
set tw::zhist {}; set tw::ghist {}; set tw::hist_chunk 1; set tw::hz_n 0
for {set j 0} {$j < 200} {incr j} { lappend tw::zhist [strline $j]; lappend tw::ghist {} }
tw::history_add [strline 200] {}
check "7.9 a history far over the cap is merged until under it ([llength $tw::zhist] entries, cap 64)" [expr {[llength $tw::zhist] < 64}]

# ---- 7.10 the driver's messages (finding 10)
set fh [open [file join $TDIR string_driver.tcl]]; set drv [read $fh]; close $fh
check "7.10 the 'converged' history line reports the long-span force, not the short-window one" [regexp {converged,[^\n]*fl_clean} $drv]
check "7.10 no claim that running longer will not remove the force" [expr {![string match "*will not remove it*" $drv]}]

# ---- 7.11 restarts from older state files and bad settings (finding 11)
reset_defaults
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
foreach {k v} {calibrate 2 memory 2 stepsize 0.001} { tw::set_param $k $v }
for {set it 0} {$it < 3} {incr it} { tw::accumulate [lrepeat 4 {0.0 1.0}] [lrepeat 4 {1 0 0 1}]; tw::advance }
set sold {}; foreach line [split [tw::dump_state] \n] { if {![string match "set tw::h0 *" $line]} { append sold $line \n } }
set tw::h0 0.0
catch {tw::load_state $sold}
check "7.11 a state without h0 takes it from h, so step_decay still works (h0 $tw::h0, h $tw::h)" [expr {$tw::h0 == $tw::h && $tw::h > 0}]
check "7.11 a non-integer calibrate is refused" [catch {tw::set_param calibrate 200.5}]
check "7.11 a non-integer curve_reach is refused" [catch {tw::set_param curve_reach 2.5}]
check "7.11 an unknown flow is refused" [catch {tw::set_param flow metrc}]
reset_defaults
tw::set_param frame_updates 50
set err [catch {tw::check_frame_updates 100} msg]
check "7.11 TW(frame_updates) different from runs_per_frame is refused ($msg)" [expr {$err && [string match "*runs_per_frame*" $msg]}]
reset_defaults

# ---- 7.12 the averaged move is not worked out on updates that do not use it (finding 12)
reset_defaults
tw::init {{0.0 0.0} {1.0 0.0} {2.0 0.0} {3.0 0.0}} {1.0 1.0} {0 0}
foreach {k v} {calibrate 2 memory 2} { tw::set_param $k $v }
rename tw::raw_move tw::raw_move_orig
set nraw 0
proc tw::raw_move {} { incr ::nraw; return [tw::raw_move_orig] }
for {set it 0} {$it < 3} {incr it} { tw::accumulate [lrepeat 4 {0.0 1.0}] [lrepeat 4 {1 0 0 1}]; tw::advance }
set nraw 0
for {set it 0} {$it < 10} {incr it} { tw::accumulate [lrepeat 4 {0.0 1.0}] [lrepeat 4 {1 0 0 1}]; tw::advance }
rename tw::raw_move {}; rename tw::raw_move_orig tw::raw_move
check "7.12 with polyak on, the averaged move is not computed after calibration ($nraw calls in 10 updates)" [expr {$nraw == 0}]
reset_defaults

# ---- 6. bad input is refused
check "too few images refused" [catch {tw::init {{0 0} {1 1}} {1 1} {0 0}}]
check "non-positive scale refused" [catch {tw::init {{0 0} {1 1} {2 2}} {1 0} {0 0}}]
check "unknown parameter refused" [catch {tw::set_param nonsense 1}]

puts "string_math: $npass passed, $nfail failed"
exit [expr {$nfail > 0}]
