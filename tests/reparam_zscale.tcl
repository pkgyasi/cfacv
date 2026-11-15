# Reparameterization with per-CV length scales, SMPARAMS(zscale).
#   tclsh tests/reparam_zscale.tcl <cfacv base dir>
# Prints a FAIL line for each failed check and a summary line; exit status 1
# on any failure.
set base [lindex $argv 0]
load $base/lib/cfacv.so cfa_cvlibc
proc setz {sm i l} { set a [SMDataSpace_image_z $sm $i]; set k 0; foreach v $l { array_setitem $a $k $v; incr k } }
proc getz {sm i n} { set a [SMDataSpace_image_z $sm $i]; set l {}; for {set k 0} {$k<$n} {incr k} { lappend l [array_getitem $a $k] }; return $l }
proc sdist {a b w} { set d 0.0; foreach x $a y $b s $w { set d [expr {$d + pow(($y-$x)/$s,2)}] }; return [expr {sqrt($d)}] }
set fail 0
set npass 0
set nfail 0
proc check {name ok} { global fail npass nfail; if {$ok} { incr npass } else { puts "FAIL $name"; incr nfail; set fail 1 } }

# an unevenly spaced bent path in (distance [A], torsion [rad])
set pts {{4.0 1.0} {4.5 1.02} {9.0 1.05} {12.0 1.10} {12.5 2.2} {12.8 2.9} {13.0 3.0} {14.0 3.05}}
set ni [llength $pts]
foreach {label w} {unscaled {1.0 1.0} scaled {3.0 0.15}} {
  set sm [New_stringMethod_Dataspace $ni 2 0 0.0 1 0]
  SMDataSpace_set_reparam_tol $sm 1.e-9 500
  if {$label eq "scaled"} { set zs [SMDataSpace_zscale $sm]; array_setitem $zs 0 3.0; array_setitem $zs 1 0.15 }
  set i 0; foreach p $pts { setz $sm $i $p; incr i }
  SMDataSpace_reparameterize $sm
  set d {}
  for {set i 1} {$i<$ni} {incr i} { lappend d [sdist [getz $sm [expr {$i-1}] 2] [getz $sm $i 2] $w] }
  set mn [tcl::mathfunc::min {*}$d]; set mx [tcl::mathfunc::max {*}$d]
  check "$label: equal spacing in the chosen metric (min $mn max $mx)" [expr {($mx-$mn) < 1.e-6}]
  check "$label: ends unmoved" [expr {[getz $sm 0 2] eq {4.0 1.0} && [getz $sm [expr {$ni-1}] 2] eq {14.0 3.05}}]
  # how many images resolve the torsion change (z1 between 1.15 and 2.85)?
  set nt 0; for {set i 0} {$i<$ni} {incr i} { set t [lindex [getz $sm $i 2] 1]; if {$t>1.15 && $t<2.85} {incr nt} }
  set n_$label $nt
  # idempotence: a second call on an already equal-spaced string must not move it
  set before {}; for {set i 0} {$i<$ni} {incr i} { lappend before [getz $sm $i 2] }
  SMDataSpace_reparameterize $sm
  set worst 0.0
  for {set i 0} {$i<$ni} {incr i} { set dd [sdist [lindex $before $i] [getz $sm $i 2] {1.0 1.0}]; if {$dd>$worst} {set worst $dd} }
  check "$label: second call leaves the string in place (max move $worst)" [expr {$worst < 1.e-6}]
}
check "scaling puts more images on the torsion step ($n_scaled vs $n_unscaled)" [expr {$n_scaled > $n_unscaled}]
puts "reparam_zscale: $npass passed, $nfail failed"
exit $fail
