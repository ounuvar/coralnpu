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

// CoralNPU SoC top for the AWS F2 card (VU47P, xcvu47p-fsvh2892-2-e).
//
// Unlike chip_nexus, this top has no board pins: on F2 the custom logic (CL) sits inside the AWS
// shell, which owns every pin and gives the CL a 250 MHz clock (clk_main_a0), a reset and AXI
// ports. The F2 CL wrapper (issue #14) instantiates this module and connects:
//   - ddr_mem_axi_* to the shell's DDR4 controller (sh_ddr, 512-bit AXI on clk_main_a0), through
//     a 256-to-512-bit width converter;
//   - ddr_ctrl_axi_* to a small register responder (the SoC's DDR controller register window;
//     sh_ddr has no such port);
//   - the SPI device port, JTAG and UART to whatever host access path #14 and #15 choose.
// Both AXI ports run on clk_main_a0; the SoC crosses them into the core clock internally.
//
// Peripherals with no counterpart on F2 (camera, SPI flash, SPI master, GPIO, I2C) are tied off.
module chip_f2 #(
    parameter MemInitFile = "",
    parameter int ClockFrequencyMhz = 50,
    parameter int IspClockFrequencyMhz = 10,
    parameter int SpimClockFrequencyMhz = 100,
    parameter int BootAddr = 0,
    parameter int EnableAutoboot = 0,
    parameter int ItcmSizeKBytes = 1024,
    parameter int DtcmSizeKBytes = 1024,
    parameter int AddrWidth = 32
) (
    // From the shell: 250 MHz main clock and its active-low synchronous reset.
    input clk_main_a0,
    input rst_main_n,

    // SPI device port (the upstream host loading path).
    input        spi_clk_i,
    input        spi_csb_i,
    input        spi_mosi_i,
    output logic spi_miso_o,

    // Debug module JTAG.
    input  tck_i,
    input  tms_i,
    input  trst_ni,
    input  td_i,
    output td_o,

    output [1:0] uart_tx_o,
    input  [1:0] uart_rx_i,

    output logic io_halted,
    output logic io_fault,

    // DDR memory, AXI4 manager, on clk_main_a0.
    output                 ddr_mem_axi_aw_valid,
    input                  ddr_mem_axi_aw_ready,
    output [AddrWidth-1:0] ddr_mem_axi_aw_bits_addr,
    output [          2:0] ddr_mem_axi_aw_bits_prot,
    output [          0:0] ddr_mem_axi_aw_bits_id,
    output [          7:0] ddr_mem_axi_aw_bits_len,
    output [          2:0] ddr_mem_axi_aw_bits_size,
    output [          1:0] ddr_mem_axi_aw_bits_burst,
    output                 ddr_mem_axi_aw_bits_lock,
    output [          3:0] ddr_mem_axi_aw_bits_cache,
    output [          3:0] ddr_mem_axi_aw_bits_qos,
    output                 ddr_mem_axi_w_valid,
    input                  ddr_mem_axi_w_ready,
    output [        255:0] ddr_mem_axi_w_bits_data,
    output                 ddr_mem_axi_w_bits_last,
    output [         31:0] ddr_mem_axi_w_bits_strb,
    input                  ddr_mem_axi_b_valid,
    output                 ddr_mem_axi_b_ready,
    input  [          0:0] ddr_mem_axi_b_bits_id,
    input  [          1:0] ddr_mem_axi_b_bits_resp,
    output                 ddr_mem_axi_ar_valid,
    input                  ddr_mem_axi_ar_ready,
    output [AddrWidth-1:0] ddr_mem_axi_ar_bits_addr,
    output [          2:0] ddr_mem_axi_ar_bits_prot,
    output [          0:0] ddr_mem_axi_ar_bits_id,
    output [          7:0] ddr_mem_axi_ar_bits_len,
    output [          2:0] ddr_mem_axi_ar_bits_size,
    output [          1:0] ddr_mem_axi_ar_bits_burst,
    output                 ddr_mem_axi_ar_bits_lock,
    output [          3:0] ddr_mem_axi_ar_bits_cache,
    output [          3:0] ddr_mem_axi_ar_bits_qos,
    input                  ddr_mem_axi_r_valid,
    output                 ddr_mem_axi_r_ready,
    input  [        255:0] ddr_mem_axi_r_bits_data,
    input  [          0:0] ddr_mem_axi_r_bits_id,
    input  [          1:0] ddr_mem_axi_r_bits_resp,
    input                  ddr_mem_axi_r_bits_last,

    // DDR controller register window, AXI4-Lite style manager, on clk_main_a0.
    output                 ddr_ctrl_axi_aw_valid,
    input                  ddr_ctrl_axi_aw_ready,
    output [AddrWidth-1:0] ddr_ctrl_axi_aw_bits_addr,
    output                 ddr_ctrl_axi_w_valid,
    input                  ddr_ctrl_axi_w_ready,
    output [         31:0] ddr_ctrl_axi_w_bits_data,
    input                  ddr_ctrl_axi_b_valid,
    output                 ddr_ctrl_axi_b_ready,
    input  [          1:0] ddr_ctrl_axi_b_bits_resp,
    output                 ddr_ctrl_axi_ar_valid,
    input                  ddr_ctrl_axi_ar_ready,
    output [AddrWidth-1:0] ddr_ctrl_axi_ar_bits_addr,
    input                  ddr_ctrl_axi_r_valid,
    output                 ddr_ctrl_axi_r_ready,
    input  [         31:0] ddr_ctrl_axi_r_bits_data,
    input  [          1:0] ddr_ctrl_axi_r_bits_resp
);

  logic clk;
  logic rst_n;
  logic clk_isp;
  logic clk_spim;
  logic locked;

  // clk_main_a0 is already on a global buffer, so the MMCM takes it single-ended.
  // 250 MHz / 5 * 24 = 1200 MHz VCO, the same VCO as the Nexus 100 MHz input, so the output
  // dividers (core clock, 100 MHz SPI master, 10 MHz ISP) are unchanged.
  clkgen_xilultrascaleplus #(
      .ClockFrequencyMhz(ClockFrequencyMhz),
      .DiffClkIn(1'b0),
      .ClkInPeriodNs(4.0),
      .ClkInDivide(5),
      .ClkInMult(24.0)
  ) i_clkgen (
      .clk_i(clk_main_a0),
      .clk_n_i(1'b0),
      .rst_ni(rst_main_n),
      .srst_ni(1'b1),
      .clk_main_o(clk),
      .clk_isp_o(clk_isp),
      .clk_spim_o(clk_spim),
      .rst_no(rst_n),
      .locked_o(locked)
  );

  top_pkg::uart_sideband_i_t [1:0] uart_sideband_i;
  top_pkg::uart_sideband_o_t [1:0] uart_sideband_o;

  assign uart_sideband_i[0].cio_rx = uart_rx_i[0];
  assign uart_sideband_i[1].cio_rx = uart_rx_i[1];
  assign uart_tx_o[0] = uart_sideband_o[0].cio_tx;
  assign uart_tx_o[1] = uart_sideband_o[1].cio_tx;

  logic dm_req_valid, dm_req_ready;
  dm::dmi_req_t dm_req;
  logic dm_rsp_valid, dm_rsp_ready;
  dm::dmi_resp_t dm_rsp;
  logic dmi_rst_n;

  dmi_jtag #(
      .IdcodeValue(32'h04f5484d)
  ) i_jtag (
      .clk_i(clk),
      .rst_ni(rst_n),
      .testmode_i(1'b0),
      .test_rst_ni(1'b1),
      .dmi_rst_no(dmi_rst_n),
      .dmi_req_o(dm_req),
      .dmi_req_valid_o(dm_req_valid),
      .dmi_req_ready_i(dm_req_ready),
      .dmi_resp_i(dm_rsp),
      .dmi_resp_ready_o(dm_rsp_ready),
      .dmi_resp_valid_i(dm_rsp_valid),
      .tck_i(tck_i),
      .tms_i(tms_i),
      .trst_ni(trst_ni),
      .td_i(td_i),
      .td_o(td_o),
      .tdo_oe_o(  /*tdo_oe_o*/)
  );

  // Outputs of tied-off peripherals, left unconnected.
  logic spim_sclk, spim_csb, spim_mosi;
  logic spim_flash_sclk, spim_flash_csb, spim_flash_mosi, spim_flash_rst_n;
  logic [7:0] gpio_out, gpio_en;
  logic scl_out, scl_en, sda_out, sda_en;
  logic cam_trig;

  coralnpu_soc #(
      .MemInitFile(MemInitFile),
      .ClockFrequencyMhz(ClockFrequencyMhz),
      .IspClockFrequencyMhz(IspClockFrequencyMhz),
      .SpimClockFrequencyMhz(SpimClockFrequencyMhz),
      .EnableAutoboot(EnableAutoboot),
      .ItcmSizeKBytes(ItcmSizeKBytes),
      .DtcmSizeKBytes(DtcmSizeKBytes),
      .AddrWidth(AddrWidth)
  ) i_coralnpu_soc (
      .clk_i(clk),
      .clk_isp_i(clk_isp),
      .rst_ni(rst_n),
      .spi_clk_i(spi_clk_i),
      .spi_csb_i(spi_csb_i),
      .spi_mosi_i(spi_mosi_i),
      .spi_miso_o(spi_miso_o),
      .spim_sclk_o(spim_sclk),
      .spim_csb_o(spim_csb),
      .spim_mosi_o(spim_mosi),
      .spim_miso_i(1'b0),
      .spim_clk_i(clk_spim),
      .boot_addr_i(AddrWidth'(BootAddr)),
      .spim_flash_sclk_o(spim_flash_sclk),
      .spim_flash_csb_o(spim_flash_csb),
      .spim_flash_mosi_o(spim_flash_mosi),
      .spim_flash_miso_i(1'b0),
      .spim_flash_clk_i(clk_spim),
      .spim_flash_rst_no(spim_flash_rst_n),
      .gpio_o(gpio_out),
      .gpio_en_o(gpio_en),
      .gpio_i('0),
      .ISP_DVP_D0(1'b0),
      .ISP_DVP_D1(1'b0),
      .ISP_DVP_D2(1'b0),
      .ISP_DVP_D3(1'b0),
      .ISP_DVP_D4(1'b0),
      .ISP_DVP_D5(1'b0),
      .ISP_DVP_D6(1'b0),
      .ISP_DVP_D7(1'b0),
      .ISP_DVP_PCLK(1'b0),
      .ISP_DVP_HSYNC(1'b0),
      .ISP_DVP_VSYNC(1'b0),
      .CAM_INT(1'b0),
      .CAM_TRIG(cam_trig),
      .scanmode_i('0),
      .uart_sideband_i(uart_sideband_i),
      .uart_sideband_o(uart_sideband_o),
      // I2C lines idle high, as with the pull-ups on a real bus.
      .scl_i(1'b1),
      .scl_o(scl_out),
      .scl_en_o(scl_en),
      .sda_i(1'b1),
      .sda_o(sda_out),
      .sda_en_o(sda_en),
      .io_halted(io_halted),
      .io_fault(io_fault),
      .ddr_clk_i(clk_main_a0),
      .ddr_rst(~rst_main_n),
      .io_ddr_ctrl_axi_aw_valid(ddr_ctrl_axi_aw_valid),
      .io_ddr_ctrl_axi_aw_ready(ddr_ctrl_axi_aw_ready),
      .io_ddr_ctrl_axi_aw_bits_addr(ddr_ctrl_axi_aw_bits_addr),
      .io_ddr_ctrl_axi_w_valid(ddr_ctrl_axi_w_valid),
      .io_ddr_ctrl_axi_w_ready(ddr_ctrl_axi_w_ready),
      .io_ddr_ctrl_axi_w_bits_data(ddr_ctrl_axi_w_bits_data),
      .io_ddr_ctrl_axi_b_valid(ddr_ctrl_axi_b_valid),
      .io_ddr_ctrl_axi_b_ready(ddr_ctrl_axi_b_ready),
      .io_ddr_ctrl_axi_b_bits_resp(ddr_ctrl_axi_b_bits_resp),
      .io_ddr_ctrl_axi_ar_valid(ddr_ctrl_axi_ar_valid),
      .io_ddr_ctrl_axi_ar_ready(ddr_ctrl_axi_ar_ready),
      .io_ddr_ctrl_axi_ar_bits_addr(ddr_ctrl_axi_ar_bits_addr),
      .io_ddr_ctrl_axi_r_valid(ddr_ctrl_axi_r_valid),
      .io_ddr_ctrl_axi_r_ready(ddr_ctrl_axi_r_ready),
      .io_ddr_ctrl_axi_r_bits_data(ddr_ctrl_axi_r_bits_data),
      .io_ddr_ctrl_axi_r_bits_resp(ddr_ctrl_axi_r_bits_resp),
      .io_ddr_mem_axi_aw_valid(ddr_mem_axi_aw_valid),
      .io_ddr_mem_axi_aw_ready(ddr_mem_axi_aw_ready),
      .io_ddr_mem_axi_aw_bits_addr(ddr_mem_axi_aw_bits_addr),
      .io_ddr_mem_axi_aw_bits_prot(ddr_mem_axi_aw_bits_prot),
      .io_ddr_mem_axi_aw_bits_id(ddr_mem_axi_aw_bits_id),
      .io_ddr_mem_axi_aw_bits_len(ddr_mem_axi_aw_bits_len),
      .io_ddr_mem_axi_aw_bits_size(ddr_mem_axi_aw_bits_size),
      .io_ddr_mem_axi_aw_bits_burst(ddr_mem_axi_aw_bits_burst),
      .io_ddr_mem_axi_aw_bits_lock(ddr_mem_axi_aw_bits_lock),
      .io_ddr_mem_axi_aw_bits_cache(ddr_mem_axi_aw_bits_cache),
      .io_ddr_mem_axi_aw_bits_qos(ddr_mem_axi_aw_bits_qos),
      .io_ddr_mem_axi_w_valid(ddr_mem_axi_w_valid),
      .io_ddr_mem_axi_w_ready(ddr_mem_axi_w_ready),
      .io_ddr_mem_axi_w_bits_data(ddr_mem_axi_w_bits_data),
      .io_ddr_mem_axi_w_bits_last(ddr_mem_axi_w_bits_last),
      .io_ddr_mem_axi_w_bits_strb(ddr_mem_axi_w_bits_strb),
      .io_ddr_mem_axi_b_valid(ddr_mem_axi_b_valid),
      .io_ddr_mem_axi_b_ready(ddr_mem_axi_b_ready),
      .io_ddr_mem_axi_b_bits_id(ddr_mem_axi_b_bits_id),
      .io_ddr_mem_axi_b_bits_resp(ddr_mem_axi_b_bits_resp),
      .io_ddr_mem_axi_ar_valid(ddr_mem_axi_ar_valid),
      .io_ddr_mem_axi_ar_ready(ddr_mem_axi_ar_ready),
      .io_ddr_mem_axi_ar_bits_addr(ddr_mem_axi_ar_bits_addr),
      .io_ddr_mem_axi_ar_bits_prot(ddr_mem_axi_ar_bits_prot),
      .io_ddr_mem_axi_ar_bits_id(ddr_mem_axi_ar_bits_id),
      .io_ddr_mem_axi_ar_bits_len(ddr_mem_axi_ar_bits_len),
      .io_ddr_mem_axi_ar_bits_size(ddr_mem_axi_ar_bits_size),
      .io_ddr_mem_axi_ar_bits_burst(ddr_mem_axi_ar_bits_burst),
      .io_ddr_mem_axi_ar_bits_lock(ddr_mem_axi_ar_bits_lock),
      .io_ddr_mem_axi_ar_bits_cache(ddr_mem_axi_ar_bits_cache),
      .io_ddr_mem_axi_ar_bits_qos(ddr_mem_axi_ar_bits_qos),
      .io_ddr_mem_axi_r_valid(ddr_mem_axi_r_valid),
      .io_ddr_mem_axi_r_ready(ddr_mem_axi_r_ready),
      .io_ddr_mem_axi_r_bits_data(ddr_mem_axi_r_bits_data),
      .io_ddr_mem_axi_r_bits_id(ddr_mem_axi_r_bits_id),
      .io_ddr_mem_axi_r_bits_resp(ddr_mem_axi_r_bits_resp),
      .io_ddr_mem_axi_r_bits_last(ddr_mem_axi_r_bits_last),
      .io_dm_req_valid(dm_req_valid),
      .io_dm_req_ready(dm_req_ready),
      .io_dm_req_bits_address(dm_req.addr),
      .io_dm_req_bits_data(dm_req.data),
      .io_dm_req_bits_op(dm_req.op),
      .io_dm_rsp_ready(dm_rsp_ready),
      .io_dm_rsp_valid(dm_rsp_valid),
      .io_dm_rsp_bits_data(dm_rsp.data),
      .io_dm_rsp_bits_op(dm_rsp.resp)
  );

endmodule
