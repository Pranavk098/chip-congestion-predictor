# Reads an OpenROAD .odb checkpoint and dumps instance/pin/net geometry as JSON
# for downstream Python rasterization (cell density, pin density, RUDY estimate).
# All coordinates are in DBU (database units); dbu_per_micron lets Python convert.
#
# Env vars (set by caller):
#   ODB_IN   - path to input .odb file (e.g. 3_place.odb, post-placement)
#   JSON_OUT - path to output .json file

set odb_in $::env(ODB_IN)
set json_out $::env(JSON_OUT)

read_db $odb_in

set block [ord::get_db_block]
set dbu [$block getDbUnitsPerMicron]

set die [$block getDieArea]
set core [$block getCoreArea]

proc rect_json {r} {
  return [format "{\"llx\":%d,\"lly\":%d,\"urx\":%d,\"ury\":%d}" \
    [$r xMin] [$r yMin] [$r xMax] [$r yMax]]
}

set fh [open $json_out w]
puts $fh "\{"
puts $fh "\"dbu_per_micron\": $dbu,"
puts $fh "\"die\": [rect_json $die],"
puts $fh "\"core\": [rect_json $core],"

# --- Instances (for cell density) ---
puts -nonewline $fh "\"instances\": \["
set insts [$block getInsts]
set n [llength $insts]
set i 0
foreach inst $insts {
  set bbox [$inst getBBox]
  set placed [$inst isPlaced]
  set master [$inst getMaster]
  set is_macro [expr {[$master getType] eq "BLOCK" ? "true" : "false"}]
  puts -nonewline $fh [format "{\"llx\":%d,\"lly\":%d,\"urx\":%d,\"ury\":%d,\"placed\":%s,\"is_macro\":%s}" \
     [$bbox xMin] [$bbox yMin] [$bbox xMax] [$bbox yMax] [expr {$placed ? "true" : "false"}] $is_macro]
  incr i
  if {$i < $n} { puts -nonewline $fh "," }
}
puts $fh "\],"

# --- Nets -> pin center coordinates (for pin density + RUDY) ---
# Pin location approximated as owning-instance bbox center for ITerms (standard
# simplification for global congestion estimation) and BPin bbox center for IO.
puts -nonewline $fh "\"nets\": \["
set nets [$block getNets]
set n [llength $nets]
set i 0
foreach net $nets {
  set pts {}
  foreach iterm [$net getITerms] {
    set inst [$iterm getInst]
    set bbox [$inst getBBox]
    set cx [expr {([$bbox xMin] + [$bbox xMax]) / 2}]
    set cy [expr {([$bbox yMin] + [$bbox yMax]) / 2}]
    lappend pts [list $cx $cy]
  }
  foreach bterm [$net getBTerms] {
    foreach bpin [$bterm getBPins] {
      set bbox [$bpin getBBox]
      set cx [expr {([$bbox xMin] + [$bbox xMax]) / 2}]
      set cy [expr {([$bbox yMin] + [$bbox yMax]) / 2}]
      lappend pts [list $cx $cy]
    }
  }
  set ptstrs {}
  foreach p $pts { lappend ptstrs [format "\[%d,%d\]" [lindex $p 0] [lindex $p 1]] }
  puts -nonewline $fh [format "\[%s\]" [join $ptstrs ","]]
  incr i
  if {$i < $n} { puts -nonewline $fh "," }
}
puts $fh "\]"
puts $fh "\}"
close $fh

puts "Wrote $json_out"
exit
