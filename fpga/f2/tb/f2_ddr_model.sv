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

// Behavioral model of the AWS shell's DDR (sh_ddr) AXI4 subordinate for the local testbench:
// 512-bit data, 16-bit ID, single-beat transfers only (all the CL issues). Memory is sparse
// (64-byte lines, unwritten bytes read as 0). Reads and writes each go through a fixed-latency
// pipeline, and ready is withheld every few cycles to exercise back-pressure.
module f2_ddr_model #(
    parameter int unsigned Latency = 12
) (
    input clk_i,
    input rst_ni,

    input                awvalid,
    output logic         awready,
    input        [ 63:0] awaddr,
    input        [ 15:0] awid,
    input        [  7:0] awlen,
    input                wvalid,
    output logic         wready,
    input        [511:0] wdata,
    input        [ 63:0] wstrb,
    input                wlast,
    output logic         bvalid,
    input                bready,
    output logic [ 15:0] bid,
    output logic [  1:0] bresp,
    input                arvalid,
    output logic         arready,
    input        [ 63:0] araddr,
    input        [ 15:0] arid,
    input        [  7:0] arlen,
    output logic         rvalid,
    input                rready,
    output logic [ 15:0] rid,
    output logic [511:0] rdata,
    output logic [  1:0] rresp,
    output logic         rlast
);

  logic [511:0] mem[logic [57:0]];

  function automatic logic [511:0] read_line(logic [57:0] line);
    return mem.exists(line) ? mem[line] : '0;
  endfunction

  // Backdoor read of a 32-bit word at a DDR byte offset, for the testbench.
  function automatic logic [31:0] peek32(logic [63:0] offset);
    logic [511:0] l = read_line(offset[63:6]);
    return l[offset[5:2]*32+:32];
  endfunction

  logic [7:0] cycle;
  always_ff @(posedge clk_i) cycle <= cycle + 8'd1;
  wire stall = (cycle[2:0] == 3'd5);

  // Writes: accept AW and W together, apply, respond after Latency cycles.
  int unsigned b_due[$];
  logic [15:0] b_ids[$];
  assign awready = awvalid && wvalid && !stall && (b_ids.size() < 8);
  assign wready  = awready;

  // Reads: respond after Latency cycles, in order.
  int unsigned r_due[$];
  logic [15:0] r_ids[$];
  logic [511:0] r_data[$];
  assign arready = arvalid && !stall && (r_ids.size() < 8);

  int unsigned now;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      now <= 0;
      bvalid <= 1'b0;
      rvalid <= 1'b0;
      bid <= '0;
      rid <= '0;
      rdata <= '0;
      b_due.delete();
      b_ids.delete();
      r_due.delete();
      r_ids.delete();
      r_data.delete();
    end else begin
      now <= now + 1;
      if (awvalid && awready) begin
        automatic logic [511:0] l = read_line(awaddr[63:6]);
        for (int i = 0; i < 64; i++) if (wstrb[i]) l[i*8+:8] = wdata[i*8+:8];
        mem[awaddr[63:6]] = l;
        b_due.push_back(now + Latency);
        b_ids.push_back(awid);
        if (awlen != 0 || !wlast) $error("f2_ddr_model: burst write not supported");
      end
      if (arvalid && arready) begin
        r_due.push_back(now + Latency);
        r_ids.push_back(arid);
        r_data.push_back(read_line(araddr[63:6]));
        if (arlen != 0) $error("f2_ddr_model: burst read not supported");
      end
      if (bvalid && bready) begin
        bvalid <= 1'b0;
        void'(b_ids.pop_front());
        void'(b_due.pop_front());
      end else if (!bvalid && b_ids.size() > 0 && b_due[0] <= now) begin
        bvalid <= 1'b1;
        bid    <= b_ids[0];
      end
      if (rvalid && rready) begin
        rvalid <= 1'b0;
        void'(r_ids.pop_front());
        void'(r_due.pop_front());
        void'(r_data.pop_front());
      end else if (!rvalid && r_ids.size() > 0 && r_due[0] <= now) begin
        rvalid <= 1'b1;
        rid    <= r_ids[0];
        rdata  <= r_data[0];
      end
    end
  end

  assign bresp = 2'b00;
  assign rresp = 2'b00;
  assign rlast = 1'b1;

endmodule
