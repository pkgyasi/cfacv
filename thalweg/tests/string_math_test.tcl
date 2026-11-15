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
        if {[dict get $info moved] && $first_perp < 0} { set first_perp [dict get $info perp] }
        incr it
        if {$it % 25 == 0} { tw::record_frame; set stop [tw::converged] }
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
foreach p [lrange $tw::z 1 end-1] { set d [expr {abs(hypot([lindex $p 0], [lindex $p 1]) - 1.0)}]; if {$d > $worst} { set worst $d }; set v [V {*}$p]; if {$v > $top} { set top $v } }
check "interior images on the ring to within 0.12 (worst $worst)" [expr {$worst < 0.12}]
check "went over the top: barrier near +2 (highest image $top)" [expr {abs($top - 2.0) < 0.5}]
check "ends in the wells" [expr {hypot([lindex $tw::z 0 0] + 1, [lindex $tw::z 0 1]) < 0.1 && hypot([lindex $tw::z end 0] - 1, [lindex $tw::z end 1]) < 0.1}]
set ref $tw::z

# ---- 4. the same problem moved so the path crosses +/-pi in CV 1
set shift [expr {$PI - 0.3}]
set z0s {}; foreach p $z0 { lappend z0s [list [lindex $p 0] [tw::wrap [expr {[lindex $p 1] + $shift}]]] }
lassign [run_string $z0s {1.0 1.0} {0 1} $shift 20000] it2 fp2 stop2
check "periodic: stopped by its drift test (after $it2 updates)" [expr {$stop2 && $it2 < 20000}]
set jump 0.0; for {set i 1} {$i < 14} {incr i} { set d [expr {abs([lindex $tw::z $i 1] - [lindex $tw::z [expr {$i-1}] 1])}]; if {$d > $jump} { set jump $d } }
check "periodic: string is continuous, no 2 pi jump (largest step $jump)" [expr {$jump < 1.0}]
set past 0; foreach p $tw::z { if {[lindex $p 1] > $PI} { set past 1 } }
check "periodic: string runs past +pi unwrapped" $past
set worst 0.0
foreach p $tw::z q $ref { set d [expr {hypot([lindex $p 0] - [lindex $q 0], [lindex $p 1] - $shift - [lindex $q 1])}]; if {$d > $worst} { set worst $d } }
check "periodic: same path as the unshifted problem to within 0.12 (worst $worst)" [expr {$worst < 0.12}]

# ---- 4b. heavy noise: the noise limit must bind, and the string must still settle
set NOISE 25.0
foreach {k v} {frame_updates 25} { tw::set_param $k $v }
lassign [run_string $z0 {1.0 1.0} {0 0} 0.0 60000] it3 fp3 stop3
check "heavy noise: the noise limit set the step ($tw::h_noise < $tw::h_signal)" [expr {$tw::h_noise < $tw::h_signal && abs($tw::h - $tw::h_noise) < 1e-15}]
check "heavy noise: measured noise is about the 25 put in ($tw::noise1)" [expr {$tw::noise1 > 15 && $tw::noise1 < 38}]
check "heavy noise: still stops by its drift test (after $it3 updates)" [expr {$stop3 && $it3 < 60000}]
set worst 0.0
foreach p [lrange $tw::z 1 end-1] { set d [expr {abs(hypot([lindex $p 0], [lindex $p 1]) - 1.0)}]; if {$d > $worst} { set worst $d } }
check "heavy noise: still on the ring to within 0.15 (worst $worst)" [expr {$worst < 0.15}]
# the signal-only rule on the same problem, for contrast: the step it would have used
check "heavy noise: the old signal-only rule would have stepped [format %.1f [expr {$tw::h_signal / $tw::h}]] times larger" [expr {$tw::h_signal / $tw::h > 2.0}]
set NOISE 4.0

# ---- 4c. slow is not still: a tiny hand-set step far from the path must NOT stop.
# (The drift test alone passed here on the cluster, a quarter of the way in.)
tw::set_param stepsize 0.00002
lassign [run_string $z0 {1.0 1.0} {0 0} 0.0 3000] it4 fp4 stop4
set far 0.0; foreach p [lrange $tw::z 1 end-1] { set d [expr {abs(hypot([lindex $p 0], [lindex $p 1]) - 1.0)}]; if {$d > $far} { set far $d } }
check "slow string far from its path (off by $far) is not declared converged" [expr {!$stop4 && $far > 0.05}]
check "the force test is what refused it (force/noise $tw::force_ratio)" [expr {$tw::force_ratio > 1.5}]
tw::set_param stepsize 0

# ---- 5. state survives a restart
set saved [tw::dump_state]; set zb $tw::z; set hb $tw::h; set nb $tw::nacc
tw::init {{0 0} {1 1} {2 2}} {1 1} {0 0}
eval $saved
check "state restored" [expr {$tw::z eq $zb && $tw::h == $hb && $tw::nacc == $nb && $tw::N == 14}]

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

# ---- 6. bad input is refused
check "too few images refused" [catch {tw::init {{0 0} {1 1}} {1 1} {0 0}}]
check "non-positive scale refused" [catch {tw::init {{0 0} {1 1} {2 2}} {1 0} {0 0}}]
check "unknown parameter refused" [catch {tw::set_param nonsense 1}]

puts "string_math: $npass passed, $nfail failed"
exit [expr {$nfail > 0}]
