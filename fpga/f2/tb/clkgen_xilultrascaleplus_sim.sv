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

// Behavioral stand-in for clkgen_xilultrascaleplus (fpga/rtl), for the local Verilator
// testbench only: Verilator cannot simulate the MMCM. Same module name and ports; it makes the
// core, ISP and SPI master clocks from delays (needs --timing) and "locks" 64 input cycles after
// rst_ni rises. The output clocks are not phase-related to clk_i, which the SoC does not need:
// every crossing between clk_main_a0 and the core clock is asynchronous.
module clkgen_xilultrascaleplus #(
    parameter int ClockFrequencyMhz = 50,
    parameter bit AddClkBuf = 1,
    parameter bit DiffClkIn = 1,
    parameter real ClkInPeriodNs = 10.0,
    parameter int ClkInDivide = 1,
    parameter real ClkInMult = 12.0
) (
    input  clk_i,
    input  clk_n_i,
    input  rst_ni,
    input  srst_ni,
    output clk_main_o,
    output clk_isp_o,
    output clk_spim_o,
    output rst_no,
    output locked_o
);

  localparam real MainHalfNs = 500.0 / ClockFrequencyMhz;

  logic clk_main = 1'b0;
  logic clk_isp = 1'b0;
  logic clk_spim = 1'b0;

  always #(MainHalfNs * 1ns) clk_main = !clk_main;
  always #(50ns) clk_isp = !clk_isp;  // 10 MHz
  always #(5ns) clk_spim = !clk_spim;  // 100 MHz

  logic [6:0] lock_count;
  logic       locked;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      lock_count <= '0;
      locked     <= 1'b0;
    end else if (!locked) begin
      lock_count <= lock_count + 7'd1;
      locked     <= (lock_count == 7'd63);
    end
  end

  assign clk_main_o = clk_main;
  assign clk_isp_o  = clk_isp;
  assign clk_spim_o = clk_spim;
  assign rst_no     = locked & rst_ni & srst_ni;
  assign locked_o   = locked;

  logic unused;
  assign unused = ^{clk_n_i, AddClkBuf, DiffClkIn};

endmodule
