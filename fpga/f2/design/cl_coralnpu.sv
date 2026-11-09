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

// CoralNPU custom logic (CL) for the AWS F2 shell. Built with the AWS HDK (fpga/f2/README.md):
// the port list comes from the HDK's cl_ports.vh, and sh_ddr, lib_pipe and xpm_cdc_async_rst
// come from the HDK and Vivado.
//
//   OCL (BAR0)  -> cl_coralnpu_core host registers (SPI loader, UART console, reset, status)
//   sh_ddr      <- cl_coralnpu_core DDR port (SoC 0x8000_0000 window -> DDR offset 0)
//   vLED[2:0]   =  {DDR ready, fault, halted}
//   PCIS, PCIM, SDA, interrupts, HBM monitor, debug bridge: unused and tied off.
module cl_coralnpu #(
    parameter EN_DDR = 1,
    parameter EN_HBM = 0,
    // SoC configuration; synth_cl_coralnpu.tcl sets these from the exported FuseSoC project.
    parameter MemInitFile = "",
    parameter int ClockFrequencyMhz = 50,
    parameter int ItcmSizeKBytes = 1024,
    parameter int DtcmSizeKBytes = 1024
) (
    `include "cl_ports.vh"
);

  `include "cl_id_defines.vh"

  // -------------------------------------------------------------------------------------------
  // Resets.
  // -------------------------------------------------------------------------------------------
  logic rst_main_n_sync;

  xpm_cdc_async_rst u_rst_sync (
      .src_arst (rst_main_n),
      .dest_clk (clk_main_a0),
      .dest_arst(rst_main_n_sync)
  );

  // Function-level reset: acknowledge it one cycle later; the CL state is all reset by
  // rst_main_n, which the shell asserts around an FLR.
  always_ff @(posedge clk_main_a0) cl_sh_flr_done <= sh_cl_flr_assert;

  // -------------------------------------------------------------------------------------------
  // CoralNPU.
  // -------------------------------------------------------------------------------------------
  logic         ddr_ready;
  logic         halted;
  logic         fault;

  logic         ddr_awvalid;
  logic         ddr_awready;
  logic [ 63:0] ddr_awaddr;
  logic [ 15:0] ddr_awid;
  logic [  7:0] ddr_awlen;
  logic [  2:0] ddr_awsize;
  logic [  1:0] ddr_awburst;
  logic         ddr_wvalid;
  logic         ddr_wready;
  logic [511:0] ddr_wdata;
  logic [ 63:0] ddr_wstrb;
  logic         ddr_wlast;
  logic         ddr_bvalid;
  logic         ddr_bready;
  logic [ 15:0] ddr_bid;
  logic [  1:0] ddr_bresp;
  logic         ddr_arvalid;
  logic         ddr_arready;
  logic [ 63:0] ddr_araddr;
  logic [ 15:0] ddr_arid;
  logic [  7:0] ddr_arlen;
  logic [  2:0] ddr_arsize;
  logic [  1:0] ddr_arburst;
  logic         ddr_rvalid;
  logic         ddr_rready;
  logic [ 15:0] ddr_rid;
  logic [511:0] ddr_rdata;
  logic [  1:0] ddr_rresp;
  logic         ddr_rlast;

  cl_coralnpu_core #(
      .MemInitFile      (MemInitFile),
      .ClockFrequencyMhz(ClockFrequencyMhz),
      .ItcmSizeKBytes   (ItcmSizeKBytes),
      .DtcmSizeKBytes   (DtcmSizeKBytes)
  ) u_core (
      .clk_main_a0(clk_main_a0),
      .rst_main_n (rst_main_n_sync),
      .ocl_awaddr (ocl_cl_awaddr),
      .ocl_awvalid(ocl_cl_awvalid),
      .ocl_awready(cl_ocl_awready),
      .ocl_wdata  (ocl_cl_wdata),
      .ocl_wstrb  (ocl_cl_wstrb),
      .ocl_wvalid (ocl_cl_wvalid),
      .ocl_wready (cl_ocl_wready),
      .ocl_bresp  (cl_ocl_bresp),
      .ocl_bvalid (cl_ocl_bvalid),
      .ocl_bready (ocl_cl_bready),
      .ocl_araddr (ocl_cl_araddr),
      .ocl_arvalid(ocl_cl_arvalid),
      .ocl_arready(cl_ocl_arready),
      .ocl_rdata  (cl_ocl_rdata),
      .ocl_rresp  (cl_ocl_rresp),
      .ocl_rvalid (cl_ocl_rvalid),
      .ocl_rready (ocl_cl_rready),
      .ddr_ready_i(ddr_ready),
      .ddr_awvalid,
      .ddr_awready,
      .ddr_awaddr,
      .ddr_awid,
      .ddr_awlen,
      .ddr_awsize,
      .ddr_awburst,
      .ddr_wvalid,
      .ddr_wready,
      .ddr_wdata,
      .ddr_wstrb,
      .ddr_wlast,
      .ddr_bvalid,
      .ddr_bready,
      .ddr_bid,
      .ddr_bresp,
      .ddr_arvalid,
      .ddr_arready,
      .ddr_araddr,
      .ddr_arid,
      .ddr_arlen,
      .ddr_arsize,
      .ddr_arburst,
      .ddr_rvalid,
      .ddr_rready,
      .ddr_rid,
      .ddr_rdata,
      .ddr_rresp,
      .ddr_rlast,
      .halted_o   (halted),
      .fault_o    (fault)
  );

  // -------------------------------------------------------------------------------------------
  // DDR: sh_ddr, with its stats bus piped to the shell as in the HDK's cl_dram_hbm_dma example.
  // -------------------------------------------------------------------------------------------
  localparam int NumStatStages = 8;

  logic [ 7:0] ddr_stat_addr_q;
  logic        ddr_stat_wr_q;
  logic        ddr_stat_rd_q;
  logic [31:0] ddr_stat_wdata_q;
  logic        ddr_stat_ack_q;
  logic [31:0] ddr_stat_rdata_q;
  logic [ 7:0] ddr_stat_int_q;

  lib_pipe #(
      .WIDTH (1 + 1 + 8 + 32),
      .STAGES(NumStatStages)
  ) u_pipe_ddr_stat (
      .clk    (clk_main_a0),
      .rst_n  (rst_main_n_sync),
      .in_bus ({sh_cl_ddr_stat_wr, sh_cl_ddr_stat_rd, sh_cl_ddr_stat_addr, sh_cl_ddr_stat_wdata}),
      .out_bus({ddr_stat_wr_q, ddr_stat_rd_q, ddr_stat_addr_q, ddr_stat_wdata_q})
  );

  lib_pipe #(
      .WIDTH (1 + 8 + 32),
      .STAGES(NumStatStages)
  ) u_pipe_ddr_stat_ack (
      .clk    (clk_main_a0),
      .rst_n  (rst_main_n_sync),
      .in_bus ({ddr_stat_ack_q, ddr_stat_int_q, ddr_stat_rdata_q}),
      .out_bus({cl_sh_ddr_stat_ack, cl_sh_ddr_stat_int, cl_sh_ddr_stat_rdata})
  );

  sh_ddr #(
      .DDR_PRESENT(EN_DDR)
  ) SH_DDR (
      .clk                  (clk_main_a0),
      .rst_n                (rst_main_n_sync),
      .stat_clk             (clk_main_a0),
      .stat_rst_n           (rst_main_n_sync),
      .CLK_DIMM_DP          (CLK_DIMM_DP),
      .CLK_DIMM_DN          (CLK_DIMM_DN),
      .M_ACT_N              (M_ACT_N),
      .M_MA                 (M_MA),
      .M_BA                 (M_BA),
      .M_BG                 (M_BG),
      .M_CKE                (M_CKE),
      .M_ODT                (M_ODT),
      .M_CS_N               (M_CS_N),
      .M_CLK_DN             (M_CLK_DN),
      .M_CLK_DP             (M_CLK_DP),
      .M_PAR                (M_PAR),
      .M_DQ                 (M_DQ),
      .M_ECC                (M_ECC),
      .M_DQS_DP             (M_DQS_DP),
      .M_DQS_DN             (M_DQS_DN),
      .cl_RST_DIMM_N        (RST_DIMM_N),
      .cl_sh_ddr_axi_awid   (ddr_awid),
      .cl_sh_ddr_axi_awaddr (ddr_awaddr),
      .cl_sh_ddr_axi_awlen  (ddr_awlen),
      .cl_sh_ddr_axi_awsize (ddr_awsize),
      .cl_sh_ddr_axi_awvalid(ddr_awvalid),
      .cl_sh_ddr_axi_awburst(ddr_awburst),
      .cl_sh_ddr_axi_awuser (1'b0),
      .cl_sh_ddr_axi_awready(ddr_awready),
      .cl_sh_ddr_axi_wdata  (ddr_wdata),
      .cl_sh_ddr_axi_wstrb  (ddr_wstrb),
      .cl_sh_ddr_axi_wlast  (ddr_wlast),
      .cl_sh_ddr_axi_wvalid (ddr_wvalid),
      .cl_sh_ddr_axi_wready (ddr_wready),
      .cl_sh_ddr_axi_bid    (ddr_bid),
      .cl_sh_ddr_axi_bresp  (ddr_bresp),
      .cl_sh_ddr_axi_bvalid (ddr_bvalid),
      .cl_sh_ddr_axi_bready (ddr_bready),
      .cl_sh_ddr_axi_arid   (ddr_arid),
      .cl_sh_ddr_axi_araddr (ddr_araddr),
      .cl_sh_ddr_axi_arlen  (ddr_arlen),
      .cl_sh_ddr_axi_arsize (ddr_arsize),
      .cl_sh_ddr_axi_arvalid(ddr_arvalid),
      .cl_sh_ddr_axi_arburst(ddr_arburst),
      .cl_sh_ddr_axi_aruser (1'b0),
      .cl_sh_ddr_axi_arready(ddr_arready),
      .cl_sh_ddr_axi_rid    (ddr_rid),
      .cl_sh_ddr_axi_rdata  (ddr_rdata),
      .cl_sh_ddr_axi_rresp  (ddr_rresp),
      .cl_sh_ddr_axi_rlast  (ddr_rlast),
      .cl_sh_ddr_axi_rvalid (ddr_rvalid),
      .cl_sh_ddr_axi_rready (ddr_rready),
      .sh_ddr_stat_bus_addr (ddr_stat_addr_q),
      .sh_ddr_stat_bus_wdata(ddr_stat_wdata_q),
      .sh_ddr_stat_bus_wr   (ddr_stat_wr_q),
      .sh_ddr_stat_bus_rd   (ddr_stat_rd_q),
      .sh_ddr_stat_bus_ack  (ddr_stat_ack_q),
      .sh_ddr_stat_bus_rdata(ddr_stat_rdata_q),
      .ddr_sh_stat_int      (ddr_stat_int_q),
      .sh_cl_ddr_is_ready   (ddr_ready)
  );

  // -------------------------------------------------------------------------------------------
  // Shell status and IDs.
  // -------------------------------------------------------------------------------------------
  always_comb begin
    cl_sh_id0         = `CL_SH_ID0;
    cl_sh_id1         = `CL_SH_ID1;
    cl_sh_status0     = 32'h434E_5055;  // "CNPU"
    cl_sh_status1     = {29'd0, ddr_ready, fault, halted};
    cl_sh_status2     = 32'd0;
    cl_sh_status_vled = {13'd0, ddr_ready, fault, halted};
  end

  // -------------------------------------------------------------------------------------------
  // Unused interfaces.
  // -------------------------------------------------------------------------------------------
  // Written out rather than with the HDK's unused_*_template.inc files, which drive only some
  // bits of a few of these ports.
  always_comb begin
    cl_sh_pcim_awid    = '0;
    cl_sh_pcim_awaddr  = '0;
    cl_sh_pcim_awlen   = '0;
    cl_sh_pcim_awsize  = '0;
    cl_sh_pcim_awburst = 2'b01;
    cl_sh_pcim_awcache = '0;
    cl_sh_pcim_awlock  = '0;
    cl_sh_pcim_awprot  = '0;
    cl_sh_pcim_awqos   = '0;
    cl_sh_pcim_awuser  = '0;
    cl_sh_pcim_awvalid = 1'b0;
    cl_sh_pcim_wid     = '0;
    cl_sh_pcim_wdata   = '0;
    cl_sh_pcim_wstrb   = '0;
    cl_sh_pcim_wlast   = 1'b0;
    cl_sh_pcim_wuser   = '0;
    cl_sh_pcim_wvalid  = 1'b0;
    cl_sh_pcim_bready  = 1'b0;
    cl_sh_pcim_arid    = '0;
    cl_sh_pcim_araddr  = '0;
    cl_sh_pcim_arlen   = '0;
    cl_sh_pcim_arsize  = '0;
    cl_sh_pcim_arburst = 2'b01;
    cl_sh_pcim_arcache = '0;
    cl_sh_pcim_arlock  = '0;
    cl_sh_pcim_arprot  = '0;
    cl_sh_pcim_arqos   = '0;
    cl_sh_pcim_aruser  = '0;
    cl_sh_pcim_arvalid = 1'b0;
    cl_sh_pcim_rready  = 1'b0;
  end

  // PCIS (host DMA into the CL) is not used: the host loads programs through OCL and SPI.
  always_comb begin
    cl_sh_dma_pcis_awready = 1'b0;
    cl_sh_dma_pcis_wready  = 1'b0;
    cl_sh_dma_pcis_bid     = '0;
    cl_sh_dma_pcis_bresp   = '0;
    cl_sh_dma_pcis_bvalid  = 1'b0;
    cl_sh_dma_pcis_arready = 1'b0;
    cl_sh_dma_pcis_rid     = '0;
    cl_sh_dma_pcis_rdata   = '0;
    cl_sh_dma_pcis_rresp   = '0;
    cl_sh_dma_pcis_rlast   = 1'b0;
    cl_sh_dma_pcis_ruser   = '0;
    cl_sh_dma_pcis_rvalid  = 1'b0;
    cl_sh_dma_wr_full      = 1'b0;
    cl_sh_dma_rd_full      = 1'b0;
  end

  always_comb begin
    cl_sda_awready      = 1'b0;
    cl_sda_wready       = 1'b0;
    cl_sda_bvalid       = 1'b0;
    cl_sda_bresp        = '0;
    cl_sda_arready      = 1'b0;
    cl_sda_rvalid       = 1'b0;
    cl_sda_rdata        = '0;
    cl_sda_rresp        = '0;
    cl_sh_apppf_irq_req = '0;
  end

  always_comb begin
    tdo               = 1'b0;
    hbm_apb_paddr_1   = '0;
    hbm_apb_pprot_1   = '0;
    hbm_apb_psel_1    = '0;
    hbm_apb_penable_1 = '0;
    hbm_apb_pwrite_1  = '0;
    hbm_apb_pwdata_1  = '0;
    hbm_apb_pstrb_1   = '0;
    hbm_apb_pready_1  = '0;
    hbm_apb_prdata_1  = '0;
    hbm_apb_pslverr_1 = '0;
    hbm_apb_paddr_0   = '0;
    hbm_apb_pprot_0   = '0;
    hbm_apb_psel_0    = '0;
    hbm_apb_penable_0 = '0;
    hbm_apb_pwrite_0  = '0;
    hbm_apb_pwdata_0  = '0;
    hbm_apb_pstrb_0   = '0;
    hbm_apb_pready_0  = '0;
    hbm_apb_prdata_0  = '0;
    hbm_apb_pslverr_0 = '0;
    PCIE_EP_TXP       = '0;
    PCIE_EP_TXN       = '0;
    PCIE_RP_PERSTN    = '0;
    PCIE_RP_TXP       = '0;
    PCIE_RP_TXN       = '0;
  end

endmodule
