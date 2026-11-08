# Out-of-context Vivado synthesis of the CoralNPU subsystem for the F2 part (VU47P).
# Normally run through run_ooc.sh; see README.md in this directory.
# Usage:
#   vivado -mode batch -nojournal -log synth.log -source fpga/ooc/ooc_synth.tcl \
#     -tclargs <CoralNPUChiselSubsystem.sv> <out_dir> [clock_ns] [extra define list] [synth_directive]
# clock_ns defaults to 20 (50 MHz, the project target clock).
set sv      [lindex $argv 0]
set outdir  [lindex $argv 1]
set period  [expr {[llength $argv] > 2 ? [lindex $argv 2] : 20.000}]
set extra   [expr {[llength $argv] > 3 ? [lindex $argv 3] : {}}]
set sdir    [expr {[llength $argv] > 4 ? [lindex $argv 4] : {}}]
set part    xcvu47p-fsvh2892-2-e
set top     CoralNPUChiselSubsystem
set root    [file normalize [file join [file dirname [info script]] .. ..]]

file mkdir $outdir
# RVV defines from RVV_CORE_MINI_AXI_VOPTS (hdl/chisel/src/coralnpu/flags.bzl), plus
# FPGA_XILINX as in upstream's FPGA synth target (fpga/coralnpu_soc.core). FPGA_XILINX
# turns ClockGate into a BUFGCE and the multi_fifo storage into LUTRAM; without it the
# clock gate is a latch + LUT driving ~67k registers on fabric and hold fails.
set defs [concat {USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON SYNTHESIS FPGA_XILINX} $extra]
set incs [list \
  [file join $root hdl verilog rvv inc] \
  [file join $root hdl verilog rvv design] \
  [file join $root hdl verilog rvv common]]

read_verilog -sv $sv
set sargs [list -top $top -part $part -mode out_of_context \
  -verilog_define $defs -include_dirs $incs -flatten_hierarchy rebuilt]
if {$sdir ne {}} { lappend sargs -directive $sdir }
synth_design {*}$sargs

# Clock: io_clk_i is the core clock. Other ports are left unconstrained (OOC).
create_clock -name clk -period $period [get_ports io_clk_i]

report_utilization -file [file join $outdir utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $outdir utilization_hier.rpt]
report_timing_summary -file [file join $outdir timing_synth.rpt]
write_checkpoint -force [file join $outdir synth.dcp]
