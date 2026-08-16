# Standalone fast/coarse global-route pass, run separately from the real
# flow's `route` stage. Loads the post-CTS checkpoint (same input the real
# global_route.tcl uses) and runs global_route with a tight iteration cap,
# so its congestion report is a fast, coarse ESTIMATE distinct from the
# fully-converged congestion report the real flow stage produces (which
# runs up to 30 congestion-driven iterations). This gives a genuine
# "early GR" vs "final GR" pair, matching CircuitNet's
# congestion_early_global_routing vs congestion_global_routing distinction,
# rather than reusing one pass for both.
#
# Env vars:
#   CTS_ODB_IN     - path to the post-CTS .odb checkpoint (4_cts.odb)
#   EGR_REPORT_OUT - path to write the fast-pass congestion report

set cts_odb $::env(CTS_ODB_IN)
set report_out $::env(EGR_REPORT_OUT)

read_db $cts_odb

# Same pin_access step the real flow runs before global routing, so the
# routing resource estimate is comparable.
if { [catch { pin_access } errMsg] } {
  puts "pin_access warning (continuing): $errMsg"
}

if { [catch {
  global_route -congestion_iterations 1 -congestion_report_file $report_out
} errMsg] } {
  puts "early global_route warning (continuing): $errMsg"
}

puts "Wrote $report_out"
exit
