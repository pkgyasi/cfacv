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
foreach {k v} {memory 100 calibrate 200 max_move 0.004 conv_window 8 conv_tol 0.06 conv_count 3} { tw::set_param $k $v }
set z0 {}; for {set i 0} {$i < 14} {incr i} { set s [expr {$i / 13.0}]; lappend z0 [list [expr {-1 + 2 * $s}] [expr {0.9 * sin($PI * $s) * (1 + 0.3 * sin(9 * $s))}]] }
lassign [run_string $z0 {1.0 1.0} {0 0} 0.0 20000] it first_perp stop
check "stopped by its own drift test, not by the iteration cap (after $it updates)" [expr {$stop && $it < 20000}]
check "step is the smaller of the signal and noise limits" [expr {abs($tw::h - min($tw::h_signal, $tw::h_noise)) < 1e-15 && $tw::h_noise > 0 && $tw::h_signal > 0}]
check "first move no larger than requested ($first_perp)" [expr {$first_perp <= 0.004 + 1e-12}]
check "measured force noise per update is about the 4.0 put in ($tw::noise1)" [expr {$tw::noise1 > 2.5 && $tw::noise1 < 6.0}]
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
check "heavy noise: measured noise is about the 25 put in ($tw::noise1)" [expr {$tw::noise1 > 15 && $tw::noise1 < 38}]
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
foreach {k v} {calibrate 20 memory 20 stepsize 0.002 conv_window 4 frame_updates 50 polyak 1 force_tol 0.3 profile_tol 0 conv_count 3} { tw::set_param $k $v }
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
    foreach {k v} {calibrate 0 conv_window 20 conv_count 1 force_tol 0 profile_tol 0 frame_updates 1 plateau 1 plateau_tol 0.2} { tw::set_param $k $v }
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
        if {$tw::d_sys >= 0} { lappend short $d; lappend sys $tw::d_sys; if {[tw::converged]} { incr stopped } }
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
lassign [wander_bounded 0.0 1500 32] rest_b big_b chunk_b
lassign [wander_bounded 0.002 1500 32] creep_b big_c chunk_c
check "1500 frames kept in at most 32 entries (most $big_b, $chunk_b frames each)" [expr {$big_b <= 32 && $chunk_b >= 32}]
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

# ---- 6. bad input is refused
check "too few images refused" [catch {tw::init {{0 0} {1 1}} {1 1} {0 0}}]
check "non-positive scale refused" [catch {tw::init {{0 0} {1 1} {2 2}} {1 0} {0 0}}]
check "unknown parameter refused" [catch {tw::set_param nonsense 1}]

puts "string_math: $npass passed, $nfail failed"
exit [expr {$nfail > 0}]
