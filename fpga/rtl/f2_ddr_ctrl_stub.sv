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

// Responder for the SoC's DDR controller register window (0x7000_0000, 4 KB).
//
// On Nexus this window is the MIG's AXI-Lite control port. The AWS shell's sh_ddr has no such
// port, so on F2 every write is accepted and dropped, and reads return 0 except offset 0x0,
// which returns {31'b0, ddr_ready} so software can poll for DDR calibration. Every response is
// OKAY, so a stray access cannot hang the SoC's crossbar.
module f2_ddr_ctrl_stub #(
    parameter int unsigned AddrWidth = 32
) (
    input clk_i,
    input rst_ni,

    input ddr_ready_i,

    input                        aw_valid,
    output logic                 aw_ready,
    input        [AddrWidth-1:0] aw_addr,
    input                        w_valid,
    output logic                 w_ready,
    input        [         31:0] w_data,
    output logic                 b_valid,
    input                        b_ready,
    output logic [          1:0] b_resp,
    input                        ar_valid,
    output logic                 ar_ready,
    input        [AddrWidth-1:0] ar_addr,
    output logic                 r_valid,
    input                        r_ready,
    output logic [         31:0] r_data,
    output logic [          1:0] r_resp
);

  // One write and one read in flight at a time.
  wire wr_take = aw_valid && w_valid && (!b_valid || b_ready);
  wire rd_take = ar_valid && (!r_valid || r_ready);

  assign aw_ready = wr_take;
  assign w_ready  = wr_take;
  assign ar_ready = rd_take;
  assign b_resp   = 2'b00;
  assign r_resp   = 2'b00;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      b_valid <= 1'b0;
      r_valid <= 1'b0;
      r_data  <= '0;
    end else begin
      if (wr_take) b_valid <= 1'b1;
      else if (b_ready) b_valid <= 1'b0;
      if (rd_take) begin
        r_valid <= 1'b1;
        r_data  <= (ar_addr[11:2] == '0) ? {31'b0, ddr_ready_i} : 32'b0;
      end else if (r_ready) begin
        r_valid <= 1'b0;
      end
    end
  end

  logic unused;
  assign unused = ^{aw_addr, w_data, ar_addr[AddrWidth-1:12], ar_addr[1:0]};

endmodule
