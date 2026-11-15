# One committor trajectory: unbiased MD from a transition-state candidate,
# stopped as soon as it enters a basin.  Sourced by the configs that
# committor.py writes, AFTER the system, the dynamics and tclforces are set.
#
# Expects these Tcl variables:
#   shot_chunk    MD steps between basin tests
#   shot_cap      give up after this many steps (outcome U, "uncommitted")
#   shot_result   file to write the outcome to
#   basin_names   e.g. {closed open}
#   basin_centre(name), basin_width(name)   lists, one value per CV
#   basin_nsigma  half-width of a basin in units of basin_width
#   cv_periodic   list of 0/1, one per CV
#
# cfacv runs with zero force constants here: it measures the CVs and applies
# no bias.  The CV values are read back from the cfacv dataspace after each
# chunk; $ds exists once the first `run` has loaded the tclforces script.

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

proc shot_basin {cv} {
    global basin_names basin_centre basin_width basin_nsigma cv_periodic
    set pi 3.14159265358979323846
    foreach name $basin_names {
        set inside 1
        foreach v $cv c $basin_centre($name) w $basin_width($name) p $cv_periodic {
            set d [expr {$v - $c}]
            if {$p} {
                while {$d >  $pi} { set d [expr {$d - 2*$pi}] }
                while {$d < -$pi} { set d [expr {$d + 2*$pi}] }
            }
            if {abs($d) > $basin_nsigma * $w} { set inside 0; break }
        }
        if {$inside} { return $name }
    }
    return ""
}

set shot_steps 0
set shot_outcome U
set shot_cv {}
while {$shot_steps < $shot_cap} {
    run $shot_chunk
    incr shot_steps $shot_chunk
    set shot_cv [shot_cvs]
    set b [shot_basin $shot_cv]
    if {$b ne ""} { set shot_outcome $b; break }
}
set fp [open $shot_result w]
puts $fp "$shot_outcome $shot_steps $shot_cv"
close $fp
puts "COMMITTOR) outcome $shot_outcome after $shot_steps steps; CVs $shot_cv"
