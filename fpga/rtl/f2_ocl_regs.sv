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

// Host control registers on the AWS F2 OCL port (AXI4-Lite, BAR0 of the application PF), on
// clk_main_a0. The register map is documented in fpga/f2/README.md; keep the two in step.
//
//   0x00 ID           RO  0x434E5055 ("CNPU")
//   0x04 VERSION      RO  0x00010000
//   0x08 CTRL         RW  [0] hold the SoC in reset (default 0: SoC runs out of reset)
//   0x0C STATUS       RO  [0] halted [1] fault [2] DDR ready [3] SPI busy
//                     W1C [4] SPI RX overflow [5] UART RX overflow
//   0x10 SPI_HALF     RW  [7:0] SCLK half period in clk_main_a0 cycles (min 3, default 5)
//   0x14 UART_BIT     RW  [15:0] UART bit time in clk_main_a0 cycles (default 2170, 115200 baud)
//   0x20 SPI_CSB      WO  queue: CSB = data[0]
//   0x24 SPI_TX1      WO  queue: send byte data[7:0]
//   0x28 SPI_TX4      WO  queue: send bytes data[7:0], [15:8], [23:16], [31:24]
//   0x2C SPI_RX       WO  queue: send data[15:0] zero bytes, capturing MISO into the RX FIFO
//   0x30 SPI_IDLE     WO  queue: send data[15:0] zero bytes, discarding MISO
//   0x34 SPI_RXDATA   RO  pop: [8] valid [7:0] byte
//   0x38 SPI_RXCOUNT  RO  bytes in the RX FIFO
//   0x3C SPI_CMDFREE  RO  free command-queue entries
//   0x40 UART_RXDATA  RO  pop: [8] valid [7:0] byte (SoC UART1, the console)
//   0x44 UART_RXCOUNT RO  bytes in the UART FIFO
//   0x48 CYCLES_LO    RO  free-running clk_main_a0 counter, low word (latches CYCLES_HI)
//   0x4C CYCLES_HI    RO  high word latched by the last CYCLES_LO read
//
// A write to a queue register when the queue is full is held (no BVALID) until an entry frees
// up, so the host should check SPI_CMDFREE before long bursts. Unmapped offsets read 0 and
// ignore writes; every response is OKAY.
module f2_ocl_regs #(
    parameter int unsigned SpiCmdDepth = 32,
    parameter int unsigned SpiRxDepth  = 256,
    parameter int unsigned UartDepth   = 1024
) (
    input clk_i,
    input rst_ni,

    // AXI4-Lite subordinate (OCL).
    input        [31:0] awaddr,
    input               awvalid,
    output logic        awready,
    input        [31:0] wdata,
    input        [ 3:0] wstrb,
    input               wvalid,
    output logic        wready,
    output logic [ 1:0] bresp,
    output logic        bvalid,
    input               bready,
    input        [31:0] araddr,
    input               arvalid,
    output logic        arready,
    output logic [31:0] rdata,
    output logic [ 1:0] rresp,
    output logic        rvalid,
    input               rready,

    // SoC control and status.
    output logic soc_reset_o,
    input        halted_i,
    input        fault_i,
    input        ddr_ready_i,

    // SPI pins (to the SoC's SPI device) and UART console input.
    output logic sclk_o,
    output logic csb_o,
    output logic mosi_o,
    input        miso_i,
    input        uart_tx_i
);

  localparam logic [31:0] IdValue = 32'h434E_5055;
  localparam logic [31:0] VersionValue = 32'h0001_0000;

  typedef enum logic [5:0] {
    RegId         = 6'h00,
    RegVersion    = 6'h01,
    RegCtrl       = 6'h02,
    RegStatus     = 6'h03,
    RegSpiHalf    = 6'h04,
    RegUartBit    = 6'h05,
    RegSpiCsb     = 6'h08,
    RegSpiTx1     = 6'h09,
    RegSpiTx4     = 6'h0A,
    RegSpiRx      = 6'h0B,
    RegSpiIdle    = 6'h0C,
    RegSpiRxData  = 6'h0D,
    RegSpiRxCount = 6'h0E,
    RegSpiCmdFree = 6'h0F,
    RegUartData   = 6'h10,
    RegUartCount  = 6'h11,
    RegCyclesLo   = 6'h12,
    RegCyclesHi   = 6'h13
  } reg_e;

  logic        ctrl_reset;
  logic [ 7:0] spi_half;
  logic [15:0] uart_bit;
  logic [63:0] cycles;
  logic [31:0] cycles_hi_q;

  // SPI controller.
  logic spi_cmd_valid, spi_cmd_ready;
  logic [1:0] spi_cmd_op, spi_cmd_cnt;
  logic [31:0] spi_cmd_data;
  logic [$clog2(SpiCmdDepth+1)-1:0] spi_cmd_depth;
  logic spi_rx_valid, spi_rx_pop;
  logic [7:0] spi_rx_data;
  logic [$clog2(SpiRxDepth+1)-1:0] spi_rx_depth;
  logic spi_rx_ovf, spi_rx_ovf_clr, spi_busy;

  f2_spi_host #(
      .CmdDepth(SpiCmdDepth),
      .RxDepth (SpiRxDepth)
  ) u_spi_host (
      .clk_i,
      .rst_ni,
      .half_period_i    (spi_half),
      .cmd_valid_i      (spi_cmd_valid),
      .cmd_ready_o      (spi_cmd_ready),
      .cmd_op_i         (spi_cmd_op),
      .cmd_cnt_i        (spi_cmd_cnt),
      .cmd_data_i       (spi_cmd_data),
      .cmd_depth_o      (spi_cmd_depth),
      .rx_valid_o       (spi_rx_valid),
      .rx_ready_i       (spi_rx_pop),
      .rx_data_o        (spi_rx_data),
      .rx_depth_o       (spi_rx_depth),
      .rx_overflow_o    (spi_rx_ovf),
      .rx_overflow_clr_i(spi_rx_ovf_clr),
      .busy_o           (spi_busy),
      .sclk_o,
      .csb_o,
      .mosi_o,
      .miso_i
  );

  // UART console capture.
  logic uart_valid, uart_pop;
  logic [7:0] uart_data;
  logic [$clog2(UartDepth+1)-1:0] uart_depth;
  logic uart_ovf, uart_ovf_clr;

  f2_uart_rx #(
      .Depth(UartDepth)
  ) u_uart_rx (
      .clk_i,
      .rst_ni,
      .bit_cycles_i  (uart_bit),
      .rx_i          (uart_tx_i),
      .valid_o       (uart_valid),
      .ready_i       (uart_pop),
      .data_o        (uart_data),
      .depth_o       (uart_depth),
      .overflow_o    (uart_ovf),
      .overflow_clr_i(uart_ovf_clr)
  );

  // ---------------------------------------------------------------------------------------------
  // Writes: take AW and W together; hold queue writes while the SPI queue is full.
  // ---------------------------------------------------------------------------------------------
  wire [5:0] wr_reg = awaddr[7:2];
  wire       wr_is_queue = (wr_reg inside {RegSpiCsb, RegSpiTx1, RegSpiTx4, RegSpiRx, RegSpiIdle});
  wire       wr_count_zero = (wdata[15:0] == 16'd0);
  wire       wr_ok = !wr_is_queue || spi_cmd_ready;
  wire       wr_take = awvalid && wvalid && !bvalid && wr_ok;

  assign awready = wr_take;
  assign wready  = wr_take;
  assign bresp   = 2'b00;

  // Queue command for this write (SPI_RX / SPI_IDLE with a zero count queue nothing).
  always_comb begin
    spi_cmd_valid = 1'b0;
    spi_cmd_op    = 2'd0;
    spi_cmd_cnt   = 2'd0;
    spi_cmd_data  = wdata;
    if (wr_take) begin
      unique case (wr_reg)
        RegSpiCsb: begin
          spi_cmd_valid = 1'b1;
          spi_cmd_op    = 2'd0;
        end
        RegSpiTx1: begin
          spi_cmd_valid = 1'b1;
          spi_cmd_op    = 2'd1;
          spi_cmd_cnt   = 2'd0;
        end
        RegSpiTx4: begin
          spi_cmd_valid = 1'b1;
          spi_cmd_op    = 2'd1;
          spi_cmd_cnt   = 2'd3;
        end
        RegSpiRx: begin
          spi_cmd_valid = !wr_count_zero;
          spi_cmd_op    = 2'd2;
          spi_cmd_data  = {16'd0, wdata[15:0] - 16'd1};
        end
        RegSpiIdle: begin
          spi_cmd_valid = !wr_count_zero;
          spi_cmd_op    = 2'd3;
          spi_cmd_data  = {16'd0, wdata[15:0] - 16'd1};
        end
        default: ;
      endcase
    end
  end

  assign spi_rx_ovf_clr = wr_take && (wr_reg == RegStatus) && wdata[4];
  assign uart_ovf_clr   = wr_take && (wr_reg == RegStatus) && wdata[5];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      bvalid     <= 1'b0;
      ctrl_reset <= 1'b0;
      spi_half   <= 8'd5;
      uart_bit   <= 16'd2170;
    end else begin
      if (wr_take) bvalid <= 1'b1;
      else if (bready) bvalid <= 1'b0;
      if (wr_take) begin
        unique case (wr_reg)
          RegCtrl:    if (wstrb[0]) ctrl_reset <= wdata[0];
          RegSpiHalf: if (wstrb[0]) spi_half <= wdata[7:0];
          RegUartBit: begin
            if (wstrb[0]) uart_bit[7:0] <= wdata[7:0];
            if (wstrb[1]) uart_bit[15:8] <= wdata[15:8];
          end
          default: ;
        endcase
      end
    end
  end

  assign soc_reset_o = ctrl_reset;

  // ---------------------------------------------------------------------------------------------
  // Reads: one at a time; FIFO data registers pop when the read is accepted.
  // ---------------------------------------------------------------------------------------------
  wire [5:0] rd_reg = araddr[7:2];
  wire       rd_take = arvalid && !rvalid;

  assign arready    = rd_take;
  assign rresp      = 2'b00;
  assign spi_rx_pop = rd_take && (rd_reg == RegSpiRxData) && spi_rx_valid;
  assign uart_pop   = rd_take && (rd_reg == RegUartData) && uart_valid;

  logic [31:0] rd_value;
  always_comb begin
    rd_value = 32'd0;
    unique case (rd_reg)
      RegId: rd_value = IdValue;
      RegVersion: rd_value = VersionValue;
      RegCtrl: rd_value = {31'd0, ctrl_reset};
      RegStatus: rd_value = {26'd0, uart_ovf, spi_rx_ovf, spi_busy, ddr_ready_i, fault_i, halted_i};
      RegSpiHalf: rd_value = {24'd0, spi_half};
      RegUartBit: rd_value = {16'd0, uart_bit};
      RegSpiRxData: rd_value = {23'd0, spi_rx_valid, spi_rx_data};
      RegSpiRxCount: rd_value = 32'(spi_rx_depth);
      RegSpiCmdFree: rd_value = 32'(SpiCmdDepth) - 32'(spi_cmd_depth);
      RegUartData: rd_value = {23'd0, uart_valid, uart_data};
      RegUartCount: rd_value = 32'(uart_depth);
      RegCyclesLo: rd_value = cycles[31:0];
      RegCyclesHi: rd_value = cycles_hi_q;
      default: rd_value = 32'd0;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rvalid      <= 1'b0;
      rdata       <= '0;
      cycles      <= '0;
      cycles_hi_q <= '0;
    end else begin
      cycles <= cycles + 64'd1;
      if (rd_take) begin
        rvalid <= 1'b1;
        rdata  <= rd_value;
        if (rd_reg == RegCyclesLo) cycles_hi_q <= cycles[63:32];
      end else if (rready) begin
        rvalid <= 1'b0;
      end
    end
  end

  logic unused;
  assign unused = ^{awaddr[31:8], awaddr[1:0], araddr[31:8], araddr[1:0], wstrb[3:2]};

endmodule
