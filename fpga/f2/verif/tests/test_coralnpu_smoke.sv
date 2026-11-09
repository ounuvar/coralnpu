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

// AWS HDK test: loads a program over the CL's SPI loader, runs it and checks the result, the
// console and DDR through the shell's sh_ddr. Pass the image built by
// //fpga:f2_smoke_test_image:
//   make test_coralnpu_smoke PLUSARGS=+image=/path/to/f2_smoke_test.f2img
module test_coralnpu_smoke ();
  import tb_type_defines_pkg::*;
  `include "common_base_test.svh"

  `define F2_OCL_WRITE(addr, data) tb.poke_ocl(.addr(addr), .data(data))
  `define F2_OCL_READ(addr, data) tb.peek_ocl(.addr(addr), .data(data))
  `define F2_ERROR(msg) \
    begin \
      $error("[%t] %s", $realtime, msg); \
      error_count++; \
    end
  `include "f2_host_tasks.svh"

  initial begin
    string image, console;
    int timeout_us;
    logic [31:0] v, entry, result_addr, result;
    logic [127:0] line;

    if (!$value$plusargs("image=%s", image)) image = "f2_smoke_test.f2img";
    if (!$value$plusargs("timeout_us=%d", timeout_us)) timeout_us = 20000;

    tb.power_up(.clk_recipe_a(ClockRecipe::A0));

    `F2_OCL_READ(F2RegId, v);
    if (v != F2IdValue) `F2_ERROR($sformatf("ID 0x%08h", v));
    for (int i = 0; i < 1000; i++) begin
      `F2_OCL_READ(F2RegStatus, v);
      if (v[2]) break;
      tb.nsec_delay(1000);
    end
    if (!v[2]) `F2_ERROR("DDR not ready");

    `F2_OCL_WRITE(F2RegSpiCsb, 1);
    `F2_OCL_WRITE(F2RegSpiIdle, 4);
    f2_load_image(image, entry, result_addr);
    f2_start_core(entry);

    for (int t = 0; t < timeout_us; t += 10) begin
      `F2_OCL_READ(F2RegStatus, v);
      if (v[0]) break;
      tb.nsec_delay(10000);
    end
    if (!v[0]) `F2_ERROR($sformatf("core did not halt within %0d us", timeout_us));
    if (v[1]) `F2_ERROR("core halted with fault");

    // The UART keeps sending after the core halts: wait until the console FIFO stops growing.
    begin
      logic [31:0] count, last;
      last = '1;
      forever begin
        tb.nsec_delay(200000);
        `F2_OCL_READ(F2RegUartCount, count);
        if (count == last) break;
        last = count;
      end
    end
    console = "";
    f2_read_console(console);
    $display("[%t] console: \"%s\"", $realtime, console);
    if (!f2_contains(console, "PASS")) `F2_ERROR("console does not contain PASS");

    if (result_addr != 0) begin
      f2_spi_read_word(result_addr, result);
      if (result != F2Pass) `F2_ERROR($sformatf("f2_result 0x%08h", result));
    end

    // DDR through the SoC and sh_ddr: words 0 and 1 of the first beat (fpga/sw/f2_smoke_test.c).
    f2_spi_read_line(32'h8000_0000, line);
    if (line[31:0] != 32'hA500_0000 || line[63:32] != 32'hA401_0101) begin
      `F2_ERROR($sformatf("DDR beat 0: 0x%032h", line));
    end

    tb.power_down();
    report_pass_fail_status();
    $finish;
  end
endmodule
