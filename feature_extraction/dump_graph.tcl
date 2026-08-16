# Reads an OpenROAD .odb checkpoint and dumps the netlist as a graph:
# instances (cell nodes, with geometry) and nets (hub nodes, with the list
# of instance indices they connect) -- for GNN-based hotspot prediction as
# an alternative to the raster/CNN pipeline (feature_extraction/dump_layout.tcl).
#
# Star-expansion representation (cell nodes <-> net hub nodes, bipartite)
# instead of clique expansion (all-pairs edges per net): a single net with
# 50 pins would add ~1225 edges under clique expansion but only 50 under
# star expansion, and star expansion is the standard choice in netlist-GNN
# literature for exactly this reason.
#
# Env vars (set by caller):
#   ODB_IN   - path to input .odb file (post-global-placement, e.g. 3_3_place_gp.odb)
#   JSON_OUT - path to output .json file

set odb_in $::env(ODB_IN)
set json_out $::env(JSON_OUT)

read_db $odb_in

set block [ord::get_db_block]
set dbu [$block getDbUnitsPerMicron]
set die [$block getDieArea]

set fh [open $json_out w]
puts $fh "\{"
puts $fh "\"dbu_per_micron\": $dbu,"
puts $fh [format "\"die\": {\"llx\":%d,\"lly\":%d,\"urx\":%d,\"ury\":%d}," \
  [$die xMin] [$die yMin] [$die xMax] [$die yMax]]

# --- Instances: assign each a 0-based index, used by nets below to refer back ---
puts -nonewline $fh "\"instances\": \["
set insts [$block getInsts]
set n [llength $insts]
set i 0
set inst_index [dict create]
foreach inst $insts {
  dict set inst_index [$inst getName] $i
  set bbox [$inst getBBox]
  set master [$inst getMaster]
  set is_macro [expr {[$master getType] eq "BLOCK" ? "true" : "false"}]
  puts -nonewline $fh [format "{\"llx\":%d,\"lly\":%d,\"urx\":%d,\"ury\":%d,\"is_macro\":%s}" \
     [$bbox xMin] [$bbox yMin] [$bbox xMax] [$bbox yMax] $is_macro]
  incr i
  if {$i < $n} { puts -nonewline $fh "," }
}
puts $fh "\],"

# --- Nets: list of instance indices connected (dedup), plus any IO pin bbox centers ---
puts -nonewline $fh "\"nets\": \["
set nets [$block getNets]
set n [llength $nets]
set i 0
foreach net $nets {
  set seen [dict create]
  set idxs {}
  foreach iterm [$net getITerms] {
    set inst [$iterm getInst]
    set name [$inst getName]
    if {![dict exists $seen $name]} {
      dict set seen $name 1
      lappend idxs [dict get $inst_index $name]
    }
  }
  set io_pts {}
  foreach bterm [$net getBTerms] {
    foreach bpin [$bterm getBPins] {
      set bbox [$bpin getBBox]
      set cx [expr {([$bbox xMin] + [$bbox xMax]) / 2}]
      set cy [expr {([$bbox yMin] + [$bbox yMax]) / 2}]
      lappend io_pts [list $cx $cy]
    }
  }
  set idxstrs {}
  foreach x $idxs { lappend idxstrs $x }
  set iostrs {}
  foreach p $io_pts { lappend iostrs [format "\[%d,%d\]" [lindex $p 0] [lindex $p 1]] }
  puts -nonewline $fh [format "{\"insts\":\[%s\],\"io_pins\":\[%s\]}" [join $idxstrs ","] [join $iostrs ","]]
  incr i
  if {$i < $n} { puts -nonewline $fh "," }
}
puts $fh "\]"
puts $fh "\}"
close $fh

puts "Wrote $json_out"
exit
