# Place and route the out-of-context checkpoint written by ooc_synth.tcl.
# Normally run through run_ooc.sh; see README.md in this directory.
# Usage:
#   vivado -mode batch -nojournal -log impl.log -source fpga/ooc/ooc_impl.tcl \
#     -tclargs <dir with synth.dcp> [clock_ns] [result_dir] [place_directive] [route_directive] [insert_bufg 0|1]
# clock_ns replaces the synthesis clock constraint (empty: keep the checkpoint's).
# Reports and checkpoints go to result_dir (default: the synth dir).
# The default directives are the ones that route the full subsystem legally: with
# Default/Default the router leaves 10k-100k nets unrouted in score/lsu.
set synthdir [lindex $argv 0]
set period   [expr {[llength $argv] > 1 ? [lindex $argv 1] : {}}]
set outdir   [expr {[llength $argv] > 2 ? [lindex $argv 2] : $synthdir}]
set pdir     [expr {[llength $argv] > 3 ? [lindex $argv 3] : "AltSpreadLogic_high"}]
set rdir     [expr {[llength $argv] > 4 ? [lindex $argv 4] : "AlternateCLBRouting"}]
set ibufg    [expr {[llength $argv] > 5 ? [lindex $argv 5] : 1}]
file mkdir $outdir

open_checkpoint [file join $synthdir synth.dcp]
if {$period ne {}} {
  # Replaces the synthesis-time clk (Vivado warns that it overrides the old one).
  create_clock -name clk -period $period [get_ports io_clk_i]
}

# Out of context the clock port has no buffer, so the clock is routed on general
# interconnect with several ns of skew (setting HD.CLK_SRC did not fix this). Insert a
# BUFG between io_clk_i and its loads so the clock uses the global clock network, as it
# would inside a real top level.
if {$ibufg} {
  set pnet  [get_nets -of_objects [get_ports io_clk_i]]
  # Top-level sinks only (hierarchical module clock pins and top-level cells); the
  # nets below each module pin follow automatically.
  set loads [get_pins -of_objects $pnet -filter {DIRECTION == IN}]
  set nleaf [llength [get_pins -leaf -of_objects $pnet -filter {DIRECTION == IN}]]
  puts "insert_bufg: moving [llength $loads] top-level sinks ($nleaf leaf loads) of $pnet onto a BUFG"
  create_cell -reference BUFG ooc_clk_bufg
  create_net ooc_clk_bufg_o
  disconnect_net -net $pnet -objects $loads
  connect_net -net ooc_clk_bufg_o -objects [concat [get_pins ooc_clk_bufg/O] $loads]
  connect_net -net $pnet -objects [get_pins ooc_clk_bufg/I]
  puts "insert_bufg: port net loads now [llength [get_pins -leaf -of_objects $pnet -filter {DIRECTION == IN}]], bufg loads [llength [get_pins -leaf -of_objects [get_nets ooc_clk_bufg_o] -filter {DIRECTION == IN}]]"
}

opt_design
place_design -directive $pdir
phys_opt_design
write_checkpoint -force [file join $outdir placed.dcp]
route_design -directive $rdir
# Save the routed result before post-route phys_opt: phys_opt_design crashed (SIGSEGV in
# librdi_route) on two runs after a legal route, losing the unsaved checkpoint.
write_checkpoint -force [file join $outdir routed.dcp]
report_route_status      -file [file join $outdir impl_route_status.rpt]
report_utilization       -file [file join $outdir impl_utilization.rpt]
report_timing_summary    -file [file join $outdir impl_timing.rpt] -max_paths 10
report_clock_utilization -file [file join $outdir impl_clock_util.rpt]
report_power             -file [file join $outdir impl_power.rpt]

phys_opt_design
write_checkpoint -force [file join $outdir routed_physopt.dcp]
report_timing_summary    -file [file join $outdir impl_timing_physopt.rpt] -max_paths 10
report_power             -file [file join $outdir impl_power_physopt.rpt]
