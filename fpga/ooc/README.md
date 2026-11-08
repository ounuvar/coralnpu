# Out-of-context synthesis on the VU47P

These scripts measure the hardware cost of a configuration: LUT, FF, DSP, BRAM and URAM use, WNS and WHS, and power for the full `CoralNPUChiselSubsystem` on the AWS F2 part (`xcvu47p-fsvh2892-2-e`). Run them the same way for the baseline and every idea, so the numbers compare. The baseline results are on issue #6.

The target clock is **50 MHz (20 ns)**. Upstream builds its vector bitstreams at 50 MHz too (`fpga/BUILD`). A configuration that does not meet setup and hold at 20 ns with zero routing errors does not count as a result.

## Requirements

- Vivado 2023.2. On the UT ECE LRC servers `run_ooc.sh` sets up Vivado 2023.2.2 and the license server. The license there stops at version 2023.10, so newer Vivado will not run. Elsewhere, set `VIVADO_BIN` and `XILINXD_LICENSE_FILE`.
- About 13 GB of memory and about 2 hours: about 30 minutes of synthesis, then about 1.5 hours of place and route.
- One Synthesis and one Implementation license.

## Running

1. Emit the Verilog for the configuration from the commit you are measuring:

   ```bash
   bazel build //hdl/chisel/src/soc:coralnpu_chisel_subsystem_cc_library_emit_verilog
   ```

   The netlist is `bazel-bin/hdl/chisel/src/soc/CoralNPUChiselSubsystem.sv`. Copy it into the run directory so the run keeps its own input.

2. Run synthesis and implementation in the background, outside the source tree:

   ```bash
   nohup fpga/ooc/run_ooc.sh <run_dir>/CoralNPUChiselSubsystem.sv <run_dir> \
     > <run_dir>/run.log 2>&1 &
   tail -f <run_dir>/synth.log    # then <run_dir>/impl.log
   ```

   Optional third and fourth arguments set the clock period in ns (default 20) and extra SystemVerilog defines for an idea's switch, for example `"IDEA1_ON"`.

3. Check the result in `<run_dir>/out/`. In `impl_route_status.rpt`, `# of nets with routing errors` must be 0. In `impl_timing.rpt`, WNS and WHS must both be at least 0, and the report must say `All user specified timing constraints are met`. `impl_utilization.rpt` and `impl_power.rpt` hold the area and power to report.

4. Post the configuration, commit hash, Vivado version and these numbers on the idea's issue, next to the baseline from issue #6.

## The F2 top level (`chip_f2`)

`chip_f2` (`fpga/rtl/chip_f2.sv`) is the full SoC that goes into the F2 shell: the subsystem, peripherals, debug module, and an MMCM that derives the core clock from the shell's 250 MHz `clk_main_a0`. It has no pins of its own. The shell drives it, so it is synthesized out of context too, but through the upstream FuseSoC flow rather than `ooc_synth.tcl`. Use it to check that a configuration still closes timing with everything around the subsystem. The per-idea numbers still come from the subsystem flow above.

1. Synthesize one configuration (`vector` or `matrix`, both with 1 MB ITCM and DTCM and a 50 MHz core clock). Vivado must be on the `--repo_env=PATH` in `.bazelrc.user`, and the `nonhermetic` repository must be refetched after changing it (`bazel fetch --force --repo=@@+coralnpu_deps_ext+nonhermetic`):

   ```bash
   bazel build //fpga:build_chip_f2_synth_only_vector
   ```

   This takes about 45 minutes. The checkpoint is `synth-vivado/com.google.coralnpu_fpga_chip_f2_0.1.runs/synth_1/chip_f2.dcp` under `$(bazel info bazel-bin)/fpga/build.build_chip_f2_synth_only_vector/com.google.coralnpu_fpga_chip_f2_0.1/`. Copy it into a run directory.

2. Place and route it with `ooc_impl.tcl`. Pass the checkpoint, the 4 ns period of `clk_main_a0`, the result directory, the default directives, and the clock port:

   ```bash
   vivado -mode batch -nojournal -log <run_dir>/impl.log -source fpga/ooc/ooc_impl.tcl \
     -tclargs <run_dir>/synth.dcp 4.000 <run_dir>/out AltSpreadLogic_high AlternateCLBRouting 1 clk_main_a0
   ```

   Vivado derives the core, SPI master and ISP clocks from the MMCM, so `impl_timing.rpt` lists them under `clk_main_a0`. The core clock is `clk_10_unbuf` at 20 ns. Check it the same way as step 3 above. The `//fpga:build_chip_f2_elab_only_*` targets only elaborate, which takes a few minutes and catches connection errors.

## What the flow does, and why

- `ooc_synth.tcl` synthesizes in out-of-context mode with the RVV defines from `RVV_CORE_MINI_AXI_VOPTS` (`hdl/chisel/src/coralnpu/flags.bzl`) plus `FPGA_XILINX`, as in upstream's FPGA synth target (`fpga/coralnpu_soc.core`). Leave `FPGA_XILINX` on. It makes `ClockGate` a `BUFGCE` and puts the `multi_fifo` storage in LUTRAM. Without it the clock gate is a latch and a LUT that drives 67k registers over general routing, and hold fails by about 4.7 ns.
- `ooc_impl.tcl` inserts a `BUFG` on `io_clk_i`, so the clock uses the global clock network as it would in a real top level. Out of context the port has no buffer otherwise.
- Placement uses `AltSpreadLogic_high` and routing uses `AlternateCLBRouting`. With the default directives the router left 13k to 100k nets unrouted in the congested `score/lsu` logic.
- The routed checkpoint and its reports are written before the last `phys_opt_design`, because Vivado 2023.2 has crashed in that step (SIGSEGV in `librdi_route.so`). If a run crashes there, the results are already saved.
- `ooc_congestion.tcl <checkpoint.dcp> <result_dir>` writes congestion, high-fanout and per-SLR reports for a placed or routed checkpoint, for investigating a run that does not route.

Ports other than the clock stay unconstrained out of context, so the I/O timing is not measured. The `Route 35-198 ... HD.PARTPIN_LOCS` warnings about this are expected.
