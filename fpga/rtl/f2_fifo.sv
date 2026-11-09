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

// First-word-fall-through FIFO with its storage in distributed RAM (LUTRAM). Same handshake as
// prim_fifo_sync with Pass = 0, which keeps its storage in flip-flops: the 1024 x 8 console FIFO
// took 3,439 LUTs and 8,295 FFs that way, and takes 262 LUTs (160 of them LUTRAM) and 31 FFs here
// (VU47P, out of context, #14). Depth must be a power of 2.
module f2_fifo #(
    parameter  int unsigned Width  = 8,
    parameter  int unsigned Depth  = 16,
    localparam int unsigned AddrW  = $clog2(Depth),
    localparam int unsigned DepthW = $clog2(Depth + 1)
) (
    input                     clk_i,
    input                     rst_ni,
    input                     wvalid_i,
    output logic              wready_o,
    input        [ Width-1:0] wdata_i,
    output logic              rvalid_o,
    input                     rready_i,
    output logic [ Width-1:0] rdata_o,
    output logic [DepthW-1:0] depth_o
);
  (* ram_style = "distributed" *) logic [Width-1:0] mem[Depth];

  logic [AddrW-1:0] wptr, rptr;
  logic [DepthW-1:0] count;

  wire push = wvalid_i && wready_o;
  wire pop = rvalid_o && rready_i;

  assign wready_o = (count != DepthW'(Depth));
  assign rvalid_o = (count != '0);
  assign rdata_o  = mem[rptr];
  assign depth_o  = count;

  always_ff @(posedge clk_i) begin
    if (push) mem[wptr] <= wdata_i;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wptr  <= '0;
      rptr  <= '0;
      count <= '0;
    end else begin
      if (push) wptr <= wptr + 1'b1;
      if (pop) rptr <= rptr + 1'b1;
      count <= count + DepthW'(push) - DepthW'(pop);
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (Depth >= 2 && (Depth & (Depth - 1)) == 0)
    else $fatal(1, "f2_fifo: Depth (%0d) must be a power of 2", Depth);
  end
`endif
endmodule
