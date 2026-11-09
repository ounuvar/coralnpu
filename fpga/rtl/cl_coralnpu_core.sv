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

// Everything in the CoralNPU AWS F2 custom logic (CL) except the shell interface itself:
// chip_f2 (the SoC), the host control registers on OCL, the DDR width converter and the DDR
// control-window stub. The HDK top (fpga/f2/design/cl_coralnpu.sv) wraps this module with the
// shell ports, sh_ddr and the tie-offs; this module has no HDK dependencies, so it also runs in
// the local Verilator testbench (fpga/f2/tb).
//
// Host path: OCL register writes drive an SPI controller into the SoC's SPI device port, which
// speaks the upstream SPI load protocol (Spi2TLULV2). The SoC's UART1 (console) is captured into
// a FIFO the host reads over OCL.
module cl_coralnpu_core #(
    parameter MemInitFile = "",
    parameter int ClockFrequencyMhz = 50,
    parameter int ItcmSizeKBytes = 1024,
    parameter int DtcmSizeKBytes = 1024
) (
    input clk_main_a0,
    input rst_main_n,

    // OCL, AXI4-Lite subordinate.
    input        [31:0] ocl_awaddr,
    input               ocl_awvalid,
    output logic        ocl_awready,
    input        [31:0] ocl_wdata,
    input        [ 3:0] ocl_wstrb,
    input               ocl_wvalid,
    output logic        ocl_wready,
    output logic [ 1:0] ocl_bresp,
    output logic        ocl_bvalid,
    input               ocl_bready,
    input        [31:0] ocl_araddr,
    input               ocl_arvalid,
    output logic        ocl_arready,
    output logic [31:0] ocl_rdata,
    output logic [ 1:0] ocl_rresp,
    output logic        ocl_rvalid,
    input               ocl_rready,

    // DDR, 512-bit AXI4 manager to sh_ddr.
    input                ddr_ready_i,
    output logic         ddr_awvalid,
    input                ddr_awready,
    output logic [ 63:0] ddr_awaddr,
    output logic [ 15:0] ddr_awid,
    output logic [  7:0] ddr_awlen,
    output logic [  2:0] ddr_awsize,
    output logic [  1:0] ddr_awburst,
    output logic         ddr_wvalid,
    input                ddr_wready,
    output logic [511:0] ddr_wdata,
    output logic [ 63:0] ddr_wstrb,
    output logic         ddr_wlast,
    input                ddr_bvalid,
    output logic         ddr_bready,
    input        [ 15:0] ddr_bid,
    input        [  1:0] ddr_bresp,
    output logic         ddr_arvalid,
    input                ddr_arready,
    output logic [ 63:0] ddr_araddr,
    output logic [ 15:0] ddr_arid,
    output logic [  7:0] ddr_arlen,
    output logic [  2:0] ddr_arsize,
    output logic [  1:0] ddr_arburst,
    input                ddr_rvalid,
    output logic         ddr_rready,
    input        [ 15:0] ddr_rid,
    input        [511:0] ddr_rdata,
    input        [  1:0] ddr_rresp,
    input                ddr_rlast,

    // Status for the shell's virtual LEDs.
    output logic halted_o,
    output logic fault_o
);

  // ---------------------------------------------------------------------------------------------
  // Host registers.
  // ---------------------------------------------------------------------------------------------
  logic soc_reset;
  logic spi_sclk, spi_csb, spi_mosi, spi_miso;
  logic [1:0] uart_tx;
  logic halted, fault;

  f2_ocl_regs u_regs (
      .clk_i      (clk_main_a0),
      .rst_ni     (rst_main_n),
      .awaddr     (ocl_awaddr),
      .awvalid    (ocl_awvalid),
      .awready    (ocl_awready),
      .wdata      (ocl_wdata),
      .wstrb      (ocl_wstrb),
      .wvalid     (ocl_wvalid),
      .wready     (ocl_wready),
      .bresp      (ocl_bresp),
      .bvalid     (ocl_bvalid),
      .bready     (ocl_bready),
      .araddr     (ocl_araddr),
      .arvalid    (ocl_arvalid),
      .arready    (ocl_arready),
      .rdata      (ocl_rdata),
      .rresp      (ocl_rresp),
      .rvalid     (ocl_rvalid),
      .rready     (ocl_rready),
      .soc_reset_o(soc_reset),
      .halted_i   (halted),
      .fault_i    (fault),
      .ddr_ready_i(ddr_ready_i),
      .sclk_o     (spi_sclk),
      .csb_o      (spi_csb),
      .mosi_o     (spi_mosi),
      .miso_i     (spi_miso),
      .uart_tx_i  (uart_tx[1])
  );

  assign halted_o = halted;
  assign fault_o  = fault;

  // SoC reset: the shell reset or CTRL[0]. Registered so the SoC sees a clean, synchronous edge.
  // The SoC's DDR-side logic (and the converter below) share it, so reset the SoC only while it
  // has no DDR access in flight.
  logic soc_rst_n;
  always_ff @(posedge clk_main_a0 or negedge rst_main_n) begin
    if (!rst_main_n) soc_rst_n <= 1'b0;
    else soc_rst_n <= !soc_reset;
  end

  // The SPI clock is a register output; put it on a global buffer on the FPGA.
  logic spi_sclk_buf;
`ifdef FPGA_XILINX
  BUFG u_spi_sclk_bufg (
      .I(spi_sclk),
      .O(spi_sclk_buf)
  );
`else
  assign spi_sclk_buf = spi_sclk;
`endif

  // ---------------------------------------------------------------------------------------------
  // SoC.
  // ---------------------------------------------------------------------------------------------
  logic mem_aw_valid, mem_aw_ready;
  logic [31:0] mem_aw_addr;
  logic [ 0:0] mem_aw_id;
  logic [ 7:0] mem_aw_len;
  logic [ 2:0] mem_aw_size;
  logic mem_w_valid, mem_w_ready;
  logic [255:0] mem_w_data;
  logic [ 31:0] mem_w_strb;
  logic         mem_w_last;
  logic mem_b_valid, mem_b_ready;
  logic [0:0] mem_b_id;
  logic [1:0] mem_b_resp;
  logic mem_ar_valid, mem_ar_ready;
  logic [31:0] mem_ar_addr;
  logic [ 0:0] mem_ar_id;
  logic [ 7:0] mem_ar_len;
  logic [ 2:0] mem_ar_size;
  logic mem_r_valid, mem_r_ready;
  logic [255:0] mem_r_data;
  logic [  0:0] mem_r_id;
  logic [  1:0] mem_r_resp;
  logic         mem_r_last;

  logic ctl_aw_valid, ctl_aw_ready, ctl_w_valid, ctl_w_ready, ctl_b_valid, ctl_b_ready;
  logic ctl_ar_valid, ctl_ar_ready, ctl_r_valid, ctl_r_ready;
  logic [31:0] ctl_aw_addr, ctl_w_data, ctl_ar_addr, ctl_r_data;
  logic [1:0] ctl_b_resp, ctl_r_resp;

  chip_f2 #(
      .MemInitFile      (MemInitFile),
      .ClockFrequencyMhz(ClockFrequencyMhz),
      .ItcmSizeKBytes   (ItcmSizeKBytes),
      .DtcmSizeKBytes   (DtcmSizeKBytes)
  ) u_chip (
      .clk_main_a0              (clk_main_a0),
      .rst_main_n               (soc_rst_n),
      .spi_clk_i                (spi_sclk_buf),
      .spi_csb_i                (spi_csb),
      .spi_mosi_i               (spi_mosi),
      .spi_miso_o               (spi_miso),
      // Debug module JTAG unused on F2 for now: hold the TAP in reset.
      .tck_i                    (1'b0),
      .tms_i                    (1'b1),
      .trst_ni                  (1'b0),
      .td_i                     (1'b0),
      .td_o                     (),
      .uart_tx_o                (uart_tx),
      .uart_rx_i                (2'b11),
      .io_halted                (halted),
      .io_fault                 (fault),
      .ddr_mem_axi_aw_valid     (mem_aw_valid),
      .ddr_mem_axi_aw_ready     (mem_aw_ready),
      .ddr_mem_axi_aw_bits_addr (mem_aw_addr),
      .ddr_mem_axi_aw_bits_prot (),
      .ddr_mem_axi_aw_bits_id   (mem_aw_id),
      .ddr_mem_axi_aw_bits_len  (mem_aw_len),
      .ddr_mem_axi_aw_bits_size (mem_aw_size),
      .ddr_mem_axi_aw_bits_burst(),
      .ddr_mem_axi_aw_bits_lock (),
      .ddr_mem_axi_aw_bits_cache(),
      .ddr_mem_axi_aw_bits_qos  (),
      .ddr_mem_axi_w_valid      (mem_w_valid),
      .ddr_mem_axi_w_ready      (mem_w_ready),
      .ddr_mem_axi_w_bits_data  (mem_w_data),
      .ddr_mem_axi_w_bits_last  (mem_w_last),
      .ddr_mem_axi_w_bits_strb  (mem_w_strb),
      .ddr_mem_axi_b_valid      (mem_b_valid),
      .ddr_mem_axi_b_ready      (mem_b_ready),
      .ddr_mem_axi_b_bits_id    (mem_b_id),
      .ddr_mem_axi_b_bits_resp  (mem_b_resp),
      .ddr_mem_axi_ar_valid     (mem_ar_valid),
      .ddr_mem_axi_ar_ready     (mem_ar_ready),
      .ddr_mem_axi_ar_bits_addr (mem_ar_addr),
      .ddr_mem_axi_ar_bits_prot (),
      .ddr_mem_axi_ar_bits_id   (mem_ar_id),
      .ddr_mem_axi_ar_bits_len  (mem_ar_len),
      .ddr_mem_axi_ar_bits_size (mem_ar_size),
      .ddr_mem_axi_ar_bits_burst(),
      .ddr_mem_axi_ar_bits_lock (),
      .ddr_mem_axi_ar_bits_cache(),
      .ddr_mem_axi_ar_bits_qos  (),
      .ddr_mem_axi_r_valid      (mem_r_valid),
      .ddr_mem_axi_r_ready      (mem_r_ready),
      .ddr_mem_axi_r_bits_data  (mem_r_data),
      .ddr_mem_axi_r_bits_id    (mem_r_id),
      .ddr_mem_axi_r_bits_resp  (mem_r_resp),
      .ddr_mem_axi_r_bits_last  (mem_r_last),
      .ddr_ctrl_axi_aw_valid    (ctl_aw_valid),
      .ddr_ctrl_axi_aw_ready    (ctl_aw_ready),
      .ddr_ctrl_axi_aw_bits_addr(ctl_aw_addr),
      .ddr_ctrl_axi_w_valid     (ctl_w_valid),
      .ddr_ctrl_axi_w_ready     (ctl_w_ready),
      .ddr_ctrl_axi_w_bits_data (ctl_w_data),
      .ddr_ctrl_axi_b_valid     (ctl_b_valid),
      .ddr_ctrl_axi_b_ready     (ctl_b_ready),
      .ddr_ctrl_axi_b_bits_resp (ctl_b_resp),
      .ddr_ctrl_axi_ar_valid    (ctl_ar_valid),
      .ddr_ctrl_axi_ar_ready    (ctl_ar_ready),
      .ddr_ctrl_axi_ar_bits_addr(ctl_ar_addr),
      .ddr_ctrl_axi_r_valid     (ctl_r_valid),
      .ddr_ctrl_axi_r_ready     (ctl_r_ready),
      .ddr_ctrl_axi_r_bits_data (ctl_r_data),
      .ddr_ctrl_axi_r_bits_resp (ctl_r_resp)
  );

  // ---------------------------------------------------------------------------------------------
  // DDR.
  // ---------------------------------------------------------------------------------------------
  f2_ddr_upsizer u_ddr_upsizer (
      .clk_i     (clk_main_a0),
      .rst_ni    (soc_rst_n),
      .s_aw_valid(mem_aw_valid),
      .s_aw_ready(mem_aw_ready),
      .s_aw_addr (mem_aw_addr),
      .s_aw_id   (mem_aw_id),
      .s_aw_len  (mem_aw_len),
      .s_aw_size (mem_aw_size),
      .s_w_valid (mem_w_valid),
      .s_w_ready (mem_w_ready),
      .s_w_data  (mem_w_data),
      .s_w_strb  (mem_w_strb),
      .s_w_last  (mem_w_last),
      .s_b_valid (mem_b_valid),
      .s_b_ready (mem_b_ready),
      .s_b_id    (mem_b_id),
      .s_b_resp  (mem_b_resp),
      .s_ar_valid(mem_ar_valid),
      .s_ar_ready(mem_ar_ready),
      .s_ar_addr (mem_ar_addr),
      .s_ar_id   (mem_ar_id),
      .s_ar_len  (mem_ar_len),
      .s_ar_size (mem_ar_size),
      .s_r_valid (mem_r_valid),
      .s_r_ready (mem_r_ready),
      .s_r_data  (mem_r_data),
      .s_r_id    (mem_r_id),
      .s_r_resp  (mem_r_resp),
      .s_r_last  (mem_r_last),
      .m_awvalid (ddr_awvalid),
      .m_awready (ddr_awready),
      .m_awaddr  (ddr_awaddr),
      .m_awid    (ddr_awid),
      .m_awlen   (ddr_awlen),
      .m_awsize  (ddr_awsize),
      .m_awburst (ddr_awburst),
      .m_wvalid  (ddr_wvalid),
      .m_wready  (ddr_wready),
      .m_wdata   (ddr_wdata),
      .m_wstrb   (ddr_wstrb),
      .m_wlast   (ddr_wlast),
      .m_bvalid  (ddr_bvalid),
      .m_bready  (ddr_bready),
      .m_bid     (ddr_bid),
      .m_bresp   (ddr_bresp),
      .m_arvalid (ddr_arvalid),
      .m_arready (ddr_arready),
      .m_araddr  (ddr_araddr),
      .m_arid    (ddr_arid),
      .m_arlen   (ddr_arlen),
      .m_arsize  (ddr_arsize),
      .m_arburst (ddr_arburst),
      .m_rvalid  (ddr_rvalid),
      .m_rready  (ddr_rready),
      .m_rid     (ddr_rid),
      .m_rdata   (ddr_rdata),
      .m_rresp   (ddr_rresp),
      .m_rlast   (ddr_rlast)
  );

  f2_ddr_ctrl_stub u_ddr_ctrl_stub (
      .clk_i      (clk_main_a0),
      .rst_ni     (soc_rst_n),
      .ddr_ready_i(ddr_ready_i),
      .aw_valid   (ctl_aw_valid),
      .aw_ready   (ctl_aw_ready),
      .aw_addr    (ctl_aw_addr),
      .w_valid    (ctl_w_valid),
      .w_ready    (ctl_w_ready),
      .w_data     (ctl_w_data),
      .b_valid    (ctl_b_valid),
      .b_ready    (ctl_b_ready),
      .b_resp     (ctl_b_resp),
      .ar_valid   (ctl_ar_valid),
      .ar_ready   (ctl_ar_ready),
      .ar_addr    (ctl_ar_addr),
      .r_valid    (ctl_r_valid),
      .r_ready    (ctl_r_ready),
      .r_data     (ctl_r_data),
      .r_resp     (ctl_r_resp)
  );

endmodule
