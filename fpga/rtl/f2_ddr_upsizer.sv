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

// Connects the SoC's 256-bit DDR AXI manager (chip_f2 ddr_mem_axi_*) to the AWS F2 shell's
// 512-bit DDR AXI subordinate (sh_ddr cl_sh_ddr_axi_*). Both sides run on clk_main_a0.
//
// The SoC reaches DDR through TLUL2Axi, which only issues single-beat transfers (len = 0) of at
// most 32 bytes, so no burst splitting or packing is needed:
//   - Address: the SoC maps DDR at 0x8000_0000 (2 GB), so the DDR offset is addr[30:0].
//   - Writes: the 256-bit data is copied to both halves and the strobes are placed in the half
//     selected by addr[5].
//   - Reads: addr[5] travels in bit 1 of the AXI ID (the shell's ID is 16 bits; the SoC uses 1),
//     so the returning RID selects the half with no per-transaction state.
// AW and W are accepted together into one holding register, which a subordinate may do (it may
// wait for both valids before raising either ready).
module f2_ddr_upsizer #(
    parameter int unsigned InAddrWidth  = 32,
    parameter int unsigned OutAddrWidth = 64,
    parameter int unsigned OutIdWidth   = 16
) (
    input clk_i,
    input rst_ni,

    // 256-bit AXI4 subordinate port, from the SoC.
    input                          s_aw_valid,
    output logic                   s_aw_ready,
    input        [InAddrWidth-1:0] s_aw_addr,
    input        [            0:0] s_aw_id,
    input        [            7:0] s_aw_len,
    input        [            2:0] s_aw_size,
    input                          s_w_valid,
    output logic                   s_w_ready,
    input        [          255:0] s_w_data,
    input        [           31:0] s_w_strb,
    input                          s_w_last,
    output logic                   s_b_valid,
    input                          s_b_ready,
    output logic [            0:0] s_b_id,
    output logic [            1:0] s_b_resp,
    input                          s_ar_valid,
    output logic                   s_ar_ready,
    input        [InAddrWidth-1:0] s_ar_addr,
    input        [            0:0] s_ar_id,
    input        [            7:0] s_ar_len,
    input        [            2:0] s_ar_size,
    output logic                   s_r_valid,
    input                          s_r_ready,
    output logic [          255:0] s_r_data,
    output logic [            0:0] s_r_id,
    output logic [            1:0] s_r_resp,
    output logic                   s_r_last,

    // 512-bit AXI4 manager port, to the shell's DDR.
    output logic                    m_awvalid,
    input                           m_awready,
    output logic [OutAddrWidth-1:0] m_awaddr,
    output logic [  OutIdWidth-1:0] m_awid,
    output logic [             7:0] m_awlen,
    output logic [             2:0] m_awsize,
    output logic [             1:0] m_awburst,
    output logic                    m_wvalid,
    input                           m_wready,
    output logic [           511:0] m_wdata,
    output logic [            63:0] m_wstrb,
    output logic                    m_wlast,
    input                           m_bvalid,
    output logic                    m_bready,
    input        [  OutIdWidth-1:0] m_bid,
    input        [             1:0] m_bresp,
    output logic                    m_arvalid,
    input                           m_arready,
    output logic [OutAddrWidth-1:0] m_araddr,
    output logic [  OutIdWidth-1:0] m_arid,
    output logic [             7:0] m_arlen,
    output logic [             2:0] m_arsize,
    output logic [             1:0] m_arburst,
    input                           m_rvalid,
    output logic                    m_rready,
    input        [  OutIdWidth-1:0] m_rid,
    input        [           511:0] m_rdata,
    input        [             1:0] m_rresp,
    input                           m_rlast
);

  // DDR offset of an SoC address in the 0x8000_0000 window.
  function automatic logic [OutAddrWidth-1:0] ddr_offset(logic [InAddrWidth-1:0] addr);
    return OutAddrWidth'(addr[30:0]);
  endfunction

  // ---------------------------------------------------------------------------------------------
  // Write: AW and W are taken together into one holding register.
  // ---------------------------------------------------------------------------------------------
  logic                    wr_full;  // holding register occupied
  logic                    aw_pend;  // AW not yet accepted downstream
  logic                    w_pend;  // W not yet accepted downstream
  logic [OutAddrWidth-1:0] wr_addr;
  logic [             0:0] wr_id;
  logic [             2:0] wr_size;
  logic [           255:0] wr_data;
  logic [            31:0] wr_strb;
  logic                    wr_hi;

  wire                     wr_take = s_aw_valid && s_w_valid && !wr_full;
  wire                     aw_done = m_awvalid && m_awready;
  wire                     w_done = m_wvalid && m_wready;

  assign s_aw_ready = wr_take;
  assign s_w_ready  = wr_take;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_full <= 1'b0;
      aw_pend <= 1'b0;
      w_pend  <= 1'b0;
      wr_addr <= '0;
      wr_id   <= '0;
      wr_size <= '0;
      wr_data <= '0;
      wr_strb <= '0;
      wr_hi   <= 1'b0;
    end else if (wr_take) begin
      wr_full <= 1'b1;
      aw_pend <= 1'b1;
      w_pend  <= 1'b1;
      wr_addr <= ddr_offset(s_aw_addr);
      wr_id   <= s_aw_id;
      wr_size <= s_aw_size;
      wr_data <= s_w_data;
      wr_strb <= s_w_strb;
      wr_hi   <= s_aw_addr[5];
    end else begin
      if (aw_done) aw_pend <= 1'b0;
      if (w_done) w_pend <= 1'b0;
      if ((!aw_pend || aw_done) && (!w_pend || w_done)) wr_full <= 1'b0;
    end
  end

  assign m_awvalid = aw_pend;
  assign m_awaddr  = wr_addr;
  assign m_awid    = OutIdWidth'(wr_id);
  assign m_awlen   = 8'd0;
  assign m_awsize  = wr_size;
  assign m_awburst = 2'b01;  // INCR
  assign m_wvalid  = w_pend;
  assign m_wdata   = {wr_data, wr_data};
  assign m_wstrb   = wr_hi ? {wr_strb, 32'b0} : {32'b0, wr_strb};
  assign m_wlast   = 1'b1;

  assign s_b_valid = m_bvalid;
  assign m_bready  = s_b_ready;
  assign s_b_id    = m_bid[0];
  assign s_b_resp  = m_bresp;

  // ---------------------------------------------------------------------------------------------
  // Read: pass-through, with the 256-bit half carried in ID bit 1.
  // ---------------------------------------------------------------------------------------------
  assign m_arvalid = s_ar_valid;
  assign s_ar_ready = m_arready;
  assign m_araddr = ddr_offset(s_ar_addr);
  assign m_arid = OutIdWidth'({s_ar_addr[5], s_ar_id});
  assign m_arlen = 8'd0;
  assign m_arsize = s_ar_size;
  assign m_arburst = 2'b01;

  assign s_r_valid = m_rvalid;
  assign m_rready = s_r_ready;
  assign s_r_data = m_rid[1] ? m_rdata[511:256] : m_rdata[255:0];
  assign s_r_id = m_rid[0];
  assign s_r_resp = m_rresp;
  assign s_r_last = m_rlast;

  // The SoC never bursts and stays inside a 256-bit beat; anything else would be mistranslated.
`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (rst_ni && s_aw_valid && s_aw_ready) begin
      assert (s_aw_len == 0 && s_aw_size <= 3'd5)
      else $error("f2_ddr_upsizer: unsupported write len=%0d size=%0d", s_aw_len, s_aw_size);
    end
    if (rst_ni && s_w_valid && s_w_ready) begin
      assert (s_w_last)
      else $error("f2_ddr_upsizer: write without wlast");
    end
    if (rst_ni && s_ar_valid && s_ar_ready) begin
      assert (s_ar_len == 0 && s_ar_size <= 3'd5)
      else $error("f2_ddr_upsizer: unsupported read len=%0d size=%0d", s_ar_len, s_ar_size);
    end
  end
`endif

  logic unused;
  assign unused = ^{m_rid[OutIdWidth-1:2], m_bid[OutIdWidth-1:1], s_aw_len, s_ar_len, s_w_last};

endmodule
