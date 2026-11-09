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

// UART receiver (8N1) that captures the SoC's console output into a FIFO for the F2 host.
// bit_cycles_i is the bit time in clk_i cycles (for example 250 MHz / 115200 = 2170).
module f2_uart_rx #(
    parameter int unsigned Depth = 1024
) (
    input clk_i,
    input rst_ni,

    input [15:0] bit_cycles_i,
    input        rx_i,

    output logic                       valid_o,
    input                              ready_i,
    output logic [                7:0] data_o,
    output logic [$clog2(Depth+1)-1:0] depth_o,
    output logic                       overflow_o,     // sticky
    input                              overflow_clr_i
);

  logic rx_q1, rx_q2;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rx_q1 <= 1'b1;
      rx_q2 <= 1'b1;
    end else begin
      rx_q1 <= rx_i;
      rx_q2 <= rx_q1;
    end
  end

  typedef enum logic [1:0] {
    StIdle,
    StStart,
    StData,
    StStop
  } state_e;

  state_e        state;
  logic   [15:0] timer;
  logic   [ 2:0] bit_idx;
  logic   [ 7:0] sr;
  logic          push;
  logic          wready;

  f2_fifo #(
      .Width(8),
      .Depth(Depth)
  ) u_fifo (
      .clk_i,
      .rst_ni,
      .wvalid_i(push),
      .wready_o(wready),
      .wdata_i (sr),
      .rvalid_o(valid_o),
      .rready_i(ready_i),
      .rdata_o (data_o),
      .depth_o (depth_o)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state      <= StIdle;
      timer      <= '0;
      bit_idx    <= '0;
      sr         <= '0;
      push       <= 1'b0;
      overflow_o <= 1'b0;
    end else begin
      push <= 1'b0;
      if (overflow_clr_i) overflow_o <= 1'b0;
      if (push && !wready) overflow_o <= 1'b1;
      unique case (state)
        StIdle:
        if (!rx_q2) begin
          // Middle of the start bit.
          timer <= {1'b0, bit_cycles_i[15:1]};
          state <= StStart;
        end
        StStart:
        if (timer == 16'd0) begin
          if (!rx_q2) begin
            timer   <= bit_cycles_i;
            bit_idx <= 3'd0;
            state   <= StData;
          end else begin
            state <= StIdle;  // glitch
          end
        end else begin
          timer <= timer - 16'd1;
        end
        StData:
        if (timer == 16'd0) begin
          sr    <= {rx_q2, sr[7:1]};  // LSB first
          timer <= bit_cycles_i;
          if (bit_idx == 3'd7) state <= StStop;
          bit_idx <= bit_idx + 3'd1;
        end else begin
          timer <= timer - 16'd1;
        end
        StStop:
        if (timer == 16'd0) begin
          push  <= rx_q2;  // drop framing errors
          state <= StIdle;
        end else begin
          timer <= timer - 16'd1;
        end
        default: state <= StIdle;
      endcase
    end
  end

endmodule
