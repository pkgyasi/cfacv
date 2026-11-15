# One CV-selection shot: unbiased MD from a configuration on a targeted-MD
# path, stopped as soon as it enters a basin of the candidate pool.  Sourced
# by the configs that `cv_select.py shots --classifier ...` writes, AFTER the
# system, the dynamics and tclforces are set.
#
# The basin test is the one in cv_select.py (BasinClassifier): a state is
# described by the mean and spread of every feature in its own ensemble; the
# distance from a state is the mean squared z-score over the features; a
# configuration is IN a basin when that distance is within the state's own
# threshold and smaller than its distance from every other state.  A torsion
# contributes its sine and cosine as two features.
#
# Expects these Tcl variables:
#   shot_chunk    MD steps between basin tests
#   shot_cap      give up after this many steps (outcome U, "uncommitted")
#   shot_result   file to write the outcome to
#   basin_names   e.g. {closed open}
#   basin_mean(name), basin_sd(name)   lists, one value per FEATURE
#   basin_thr(name)                    threshold on the distance
#   basin_zcap(name), basin_cap(name)  cap, in spreads, on each feature flagged 1
#                                      in basin_cap: the features that tell the
#                                      states apart (optional)
#   basin_zmax(name)                   cap on the largest single |z| (older models)
#   shot_confirm                       consecutive tests in a basin that count as
#                                      entering it [2]
#   cv_periodic   list of 0/1, one per CV, in the order of cv.inp
#
# cfacv runs with zero force constants here: it measures the CVs and applies
# no bias.

proc shot_cvs {} {
    global ds cv_periodic
    set n [llength $cv_periodic]
    set ones {}
    set zeros {}
    for {set i 0} {$i < $n} {incr i} { lappend ones 1; lappend zeros 0.0 }
    set act [intListToArray $ones]
    set res [ListToArray $zeros]
    DataSpace_WriteCVtoArray $ds $act $res
    return [ArrayToList $res $n]
}

proc shot_features {cv} {
    global cv_periodic
    set f {}
    foreach v $cv p $cv_periodic {
        if {$p} { lappend f [expr {sin($v)}] [expr {cos($v)}] } else { lappend f $v }
    }
    return $f
}

# returns {mean squared z-score, largest |z|, largest |z| among the capped features}
proc shot_distance {f name} {
    global basin_mean basin_sd basin_cap
    set s 0.0; set big 0.0; set bigc 0.0
    set cap [expr {[info exists basin_cap($name)] ? $basin_cap($name) : {}}]
    set j 0
    foreach x $f m $basin_mean($name) w $basin_sd($name) {
        set zz [expr {abs(($x - $m) / $w)}]
        set s [expr {$s + $zz * $zz}]
        if {$zz > $big} { set big $zz }
        if {[llength $cap] && [lindex $cap $j] && $zz > $bigc} { set bigc $zz }
        incr j
    }
    return [list [expr {$s / [llength $f]}] $big $bigc]
}

proc shot_basin {cv} {
    global basin_names basin_thr basin_zmax basin_zcap
    set f [shot_features $cv]
    set best ""; set dbest 0.0; set zbest 0.0; set cbest 0.0; set tie 0
    foreach name $basin_names {
        lassign [shot_distance $f $name] d big bigc
        if {$best eq "" || $d < $dbest} { set best $name; set dbest $d; set zbest $big; set cbest $bigc; set tie 0 } elseif {$d == $dbest} { set tie 1 }
    }
    # one coordinate far out is not averaged away by the rest: see BasinClassifier
    if {!$tie && $dbest <= $basin_thr($best)
        && (![info exists basin_zmax($best)] || $zbest <= $basin_zmax($best))
        && (![info exists basin_zcap($best)] || $cbest <= $basin_zcap($best))} { return $best }
    return ""
}

if {![info exists shot_selftest]} {
    if {![info exists shot_confirm]} { set shot_confirm 2 }
    set shot_last ""; set shot_run 0
    set shot_steps 0
    set shot_outcome U
    set shot_cv {}
    while {$shot_steps < $shot_cap} {
        run $shot_chunk
        incr shot_steps $shot_chunk
        set shot_cv [shot_cvs]
        set b [shot_basin $shot_cv]
        # entered = found there on shot_confirm consecutive tests; one is a fluctuation
        if {$b ne "" && $b eq $shot_last} { incr shot_run } elseif {$b ne ""} { set shot_run 1 } else { set shot_run 0 }
        set shot_last $b
        if {$b ne "" && $shot_run >= $shot_confirm} { set shot_outcome $b; break }
    }
    set fp [open $shot_result w]
    puts $fp "$shot_outcome $shot_steps $shot_cv"
    close $fp
    puts "CVSEL) outcome $shot_outcome after $shot_steps steps"
}
