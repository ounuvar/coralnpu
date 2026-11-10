// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// AWS HDK test: host registers of the CoralNPU CL. Checks the ID, waits for DDR calibration,
// and reads the first ITCM line back over the SPI loader (the SoC answers even before a
// program is loaded). Run: make test_coralnpu_regs (see fpga/f2/README.md).
module test_coralnpu_regs ();
  import tb_type_defines_pkg::*;
  `include "common_base_test.svh"

  // Macro arguments are substituted everywhere in the body, port names included, so they must
  // not be called addr or data. F2_ERROR is a task call, so `if (...) `F2_ERROR(...); else` works.
  task automatic f2_error(input string msg);
    $error("[%t] %s", $realtime, msg);
    error_count++;
  endtask
  `define F2_OCL_WRITE(a, d) tb.poke_ocl(.addr(a), .data(d))
  `define F2_OCL_READ(a, d) tb.peek_ocl(.addr(a), .data(d))
  `define F2_ERROR(msg) f2_error(msg)
  `include "f2_host_tasks.svh"

  initial begin
    logic [ 31:0] v;
    logic [127:0] line;

    tb.power_up(.clk_recipe_a(ClockRecipe::A0));

    // Take sh_ddr out of reset through its stats interface, as initialize_ddr() in the HDK's
    // cl_dram_hbm_dma example does. Without it DDR never calibrates in simulation.
    tb.poke_stat(.addr(8'h0c), .intf("ddr"), .data(32'h0000_0000));
    tb.poke_stat(.addr(8'h0c), .intf("ddr"), .data(32'h0000_0001));
    tb.poke_stat(.addr(8'h0c), .intf("ddr"), .data(32'h0000_0000));

    `F2_OCL_READ(F2RegId, v);
    if (v != F2IdValue) `F2_ERROR($sformatf("ID 0x%08h", v));
    else $display("[%t] CL ID ok", $realtime);

    // DDR calibration (STATUS[2]); the micron model takes tens of microseconds.
    for (int i = 0; i < 1000; i++) begin
      `F2_OCL_READ(F2RegStatus, v);
      if (v[2]) break;
      tb.nsec_delay(1000);
    end
    if (!v[2]) `F2_ERROR("DDR not ready");
    else $display("[%t] DDR ready", $realtime);

    `F2_OCL_WRITE(F2RegSpiCsb, 1);
    `F2_OCL_WRITE(F2RegSpiIdle, 4);
    f2_spi_read_line(32'h0, line);
    $display("[%t] ITCM line 0 = 0x%032h", $realtime, line);

    tb.power_down();
    report_pass_fail_status();
    $finish;
  end
endmodule
