# Congestion and placement diagnostics for a routed (or placed) OOC checkpoint.
# Usage: vivado -mode batch -nojournal -log congestion.log -source fpga/ooc/ooc_congestion.tcl \
#          -tclargs <checkpoint.dcp> <result_dir>
set dcp    [lindex $argv 0]
set outdir [lindex $argv 1]
file mkdir $outdir
open_checkpoint $dcp
report_design_analysis -congestion -file [file join $outdir congestion.rpt]
report_design_analysis -complexity -file [file join $outdir complexity.rpt]
report_high_fanout_nets -timing -load_types -max_nets 30 -file [file join $outdir high_fanout.rpt]
report_utilization -slr -file [file join $outdir util_slr.rpt]
report_utilization -hierarchical -hierarchical_depth 4 -file [file join $outdir util_hier.rpt]
