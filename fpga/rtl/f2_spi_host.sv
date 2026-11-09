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

// SPI controller (mode 0, MSB first) that drives the SoC's SPI device port (Spi2TLULV2), so the
// F2 host can use the upstream SPI load protocol through OCL register writes.
//
// The host queues commands; the engine runs them in order:
//   CmdCsb   : set CSB to data[0] (0 = selected).
//   CmdTx    : shift out bytes data[7:0], data[15:8], ... (count = cnt + 1, 1 to 4), discarding
//              MISO.
//   CmdRx    : shift out data[15:0] + 1 zero bytes, pushing each MISO byte into the RX FIFO.
//   CmdIdle  : shift out data[15:0] + 1 zero bytes, discarding MISO. The SoC's SPI clock domain
//              only advances while SCLK toggles, so the host ends each frame with idle bytes to
//              drain the bridge's clock-crossing queues (the cocotb SPIMaster does the same).
// SCLK has a half period of half_period_i clk_i cycles (at least 3: MISO is sampled through a
// 2-flop synchronizer at the end of the high phase, after the device updated it on the rising
// edge).
module f2_spi_host #(
    parameter int unsigned CmdDepth = 32,
    parameter int unsigned RxDepth  = 256
) (
    input clk_i,
    input rst_ni,

    input [7:0] half_period_i,

    // Command queue.
    input                                 cmd_valid_i,
    output logic                          cmd_ready_o,
    input        [                   1:0] cmd_op_i,
    input        [                   1:0] cmd_cnt_i,
    input        [                  31:0] cmd_data_i,
    output logic [$clog2(CmdDepth+1)-1:0] cmd_depth_o,

    // Received bytes.
    output logic                         rx_valid_o,
    input                                rx_ready_i,
    output logic [                  7:0] rx_data_o,
    output logic [$clog2(RxDepth+1)-1:0] rx_depth_o,
    output logic                         rx_overflow_o,     // a byte was dropped (sticky)
    input                                rx_overflow_clr_i,

    output logic busy_o,  // a command is queued or running

    // SPI pins (to the SoC's SPI device).
    output logic sclk_o,
    output logic csb_o,
    output logic mosi_o,
    input        miso_i
);

  localparam logic [1:0] CmdCsb = 2'd0;
  localparam logic [1:0] CmdTx = 2'd1;
  localparam logic [1:0] CmdRx = 2'd2;
  localparam logic [1:0] CmdIdle = 2'd3;

  localparam int unsigned CmdW = 2 + 2 + 32;

  // Command FIFO.
  logic            q_valid;
  logic            q_ready;
  logic [CmdW-1:0] q_data;

  f2_fifo #(
      .Width(CmdW),
      .Depth(CmdDepth)
  ) u_cmd_fifo (
      .clk_i,
      .rst_ni,
      .wvalid_i(cmd_valid_i),
      .wready_o(cmd_ready_o),
      .wdata_i ({cmd_op_i, cmd_cnt_i, cmd_data_i}),
      .rvalid_o(q_valid),
      .rready_i(q_ready),
      .rdata_o (q_data),
      .depth_o (cmd_depth_o)
  );

  // RX FIFO.
  logic       rx_push;
  logic [7:0] rx_byte;
  logic       rx_wready;

  f2_fifo #(
      .Width(8),
      .Depth(RxDepth)
  ) u_rx_fifo (
      .clk_i,
      .rst_ni,
      .wvalid_i(rx_push),
      .wready_o(rx_wready),
      .wdata_i (rx_byte),
      .rvalid_o(rx_valid_o),
      .rready_i(rx_ready_i),
      .rdata_o (rx_data_o),
      .depth_o (rx_depth_o)
  );

  // MISO synchronizer.
  logic miso_q1, miso_q2;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      miso_q1 <= 1'b0;
      miso_q2 <= 1'b0;
    end else begin
      miso_q1 <= miso_i;
      miso_q2 <= miso_q1;
    end
  end

  typedef enum logic [1:0] {
    StIdle,
    StLow,   // SCLK low, MOSI set up
    StHigh,  // SCLK high
    StGap    // settle after a CSB change
  } state_e;

  state_e        state;
  logic   [ 7:0] timer;
  logic   [ 2:0] bit_idx;  // 7 down to 0
  logic   [ 7:0] tx_sr;
  logic   [ 7:0] rx_sr;
  logic   [31:0] tx_word;  // bytes still to send for CmdTx, LSB first
  logic   [15:0] bytes_left;  // bytes after the current one
  logic          capture;
  logic          is_tx;

  wire    [ 1:0] q_op = q_data[35:34];
  wire    [ 1:0] q_cnt = q_data[33:32];
  wire    [31:0] q_arg = q_data[31:0];
  wire    [ 7:0] half = (half_period_i < 8'd3) ? 8'd3 : half_period_i;
  wire           timer_done = (timer == 8'd1);

  assign q_ready = (state == StIdle);
  assign busy_o  = q_valid || (state != StIdle);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state         <= StIdle;
      timer         <= '0;
      bit_idx       <= '0;
      tx_sr         <= '0;
      rx_sr         <= '0;
      tx_word       <= '0;
      bytes_left    <= '0;
      capture       <= 1'b0;
      is_tx         <= 1'b0;
      sclk_o        <= 1'b0;
      csb_o         <= 1'b1;
      mosi_o        <= 1'b0;
      rx_push       <= 1'b0;
      rx_byte       <= '0;
      rx_overflow_o <= 1'b0;
    end else begin
      rx_push <= 1'b0;
      if (rx_overflow_clr_i) rx_overflow_o <= 1'b0;
      if (rx_push && !rx_wready) rx_overflow_o <= 1'b1;

      unique case (state)
        StIdle: begin
          if (q_valid) begin
            if (q_op == CmdCsb) begin
              csb_o <= q_arg[0];
              timer <= half;
              state <= StGap;
            end else begin
              is_tx      <= (q_op == CmdTx);
              capture    <= (q_op == CmdRx);
              tx_sr      <= (q_op == CmdTx) ? q_arg[7:0] : 8'h00;
              tx_word    <= {8'h00, q_arg[31:8]};
              bytes_left <= (q_op == CmdTx) ? 16'(q_cnt) : q_arg[15:0];
              mosi_o     <= (q_op == CmdTx) ? q_arg[7] : 1'b0;
              bit_idx    <= 3'd7;
              timer      <= half;
              state      <= StLow;
            end
          end
        end

        StGap: begin
          if (timer_done) state <= StIdle;
          else timer <= timer - 8'd1;
        end

        StLow: begin
          if (timer_done) begin
            sclk_o <= 1'b1;
            timer  <= half;
            state  <= StHigh;
          end else begin
            timer <= timer - 8'd1;
          end
        end

        StHigh: begin
          if (timer_done) begin
            // Sample MISO before the falling edge.
            sclk_o <= 1'b0;
            rx_sr  <= {rx_sr[6:0], miso_q2};
            timer  <= half;
            if (bit_idx != 3'd0) begin
              bit_idx <= bit_idx - 3'd1;
              mosi_o  <= tx_sr[bit_idx-3'd1];
              state   <= StLow;
            end else begin
              if (capture) begin
                rx_push <= 1'b1;
                rx_byte <= {rx_sr[6:0], miso_q2};
              end
              if (bytes_left != 16'd0) begin
                bytes_left <= bytes_left - 16'd1;
                bit_idx    <= 3'd7;
                tx_sr      <= is_tx ? tx_word[7:0] : 8'h00;
                mosi_o     <= is_tx ? tx_word[7] : 1'b0;
                tx_word    <= {8'h00, tx_word[31:8]};
                state      <= StLow;
              end else begin
                mosi_o <= 1'b0;
                state  <= StIdle;
              end
            end
          end else begin
            timer <= timer - 8'd1;
          end
        end

        default: state <= StIdle;
      endcase
    end
  end

endmodule
