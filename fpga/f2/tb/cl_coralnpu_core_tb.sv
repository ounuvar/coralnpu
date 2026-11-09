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

`timescale 1ns / 1ps

// Local Verilator testbench for cl_coralnpu_core, the AWS F2 CL without the shell. It plays the
// host the way the F2 runtime will: OCL register accesses only. It loads a program over the SPI
// controller, starts the core, waits for it to halt, then checks the UART console, a result
// word read back over SPI, and DDR (through the SoC over SPI, and directly in the DDR model).
//
// Plusargs:
//   +image=<file>       program image from elf_to_f2_image.py (needs symbol f2_result)
//   +timeout_us=<n>     simulated time limit for the program to halt (default 20000)
//   +expect=<string>    text the console must contain (default PASS)
//   +ddr_check=1        also check the DDR pattern written by fpga/sw/f2_smoke_test.c
module cl_coralnpu_core_tb #(
    parameter int ClockFrequencyMhz = 50,
    parameter int ItcmSizeKBytes = 1024,
    parameter int DtcmSizeKBytes = 1024
);

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  always #2ns clk = !clk;  // clk_main_a0, 250 MHz

  // OCL.
  logic [31:0] awaddr = '0, wdata = '0, araddr = '0;
  logic [3:0] wstrb = '0;
  logic awvalid = 1'b0, wvalid = 1'b0, arvalid = 1'b0, bready = 1'b1, rready = 1'b1;
  logic awready, wready, bvalid, arready, rvalid;
  logic [1:0] bresp, rresp;
  logic [31:0] rdata;

  // DDR.
  logic ddr_awvalid, ddr_awready, ddr_wvalid, ddr_wready, ddr_bvalid, ddr_bready;
  logic ddr_arvalid, ddr_arready, ddr_rvalid, ddr_rready, ddr_wlast, ddr_rlast;
  logic [63:0] ddr_awaddr, ddr_araddr, ddr_wstrb;
  logic [15:0] ddr_awid, ddr_arid, ddr_bid, ddr_rid;
  logic [7:0] ddr_awlen, ddr_arlen;
  logic [2:0] ddr_awsize, ddr_arsize;
  logic [1:0] ddr_awburst, ddr_arburst, ddr_bresp, ddr_rresp;
  logic [511:0] ddr_wdata, ddr_rdata;
  logic halted, fault;

  cl_coralnpu_core #(
      .ClockFrequencyMhz(ClockFrequencyMhz),
      .ItcmSizeKBytes   (ItcmSizeKBytes),
      .DtcmSizeKBytes   (DtcmSizeKBytes)
  ) dut (
      .clk_main_a0(clk),
      .rst_main_n (rst_n),
      .ocl_awaddr (awaddr),
      .ocl_awvalid(awvalid),
      .ocl_awready(awready),
      .ocl_wdata  (wdata),
      .ocl_wstrb  (wstrb),
      .ocl_wvalid (wvalid),
      .ocl_wready (wready),
      .ocl_bresp  (bresp),
      .ocl_bvalid (bvalid),
      .ocl_bready (bready),
      .ocl_araddr (araddr),
      .ocl_arvalid(arvalid),
      .ocl_arready(arready),
      .ocl_rdata  (rdata),
      .ocl_rresp  (rresp),
      .ocl_rvalid (rvalid),
      .ocl_rready (rready),
      .ddr_ready_i(1'b1),
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

  f2_ddr_model u_ddr (
      .clk_i  (clk),
      .rst_ni (rst_n),
      .awvalid(ddr_awvalid),
      .awready(ddr_awready),
      .awaddr (ddr_awaddr),
      .awid   (ddr_awid),
      .awlen  (ddr_awlen),
      .wvalid (ddr_wvalid),
      .wready (ddr_wready),
      .wdata  (ddr_wdata),
      .wstrb  (ddr_wstrb),
      .wlast  (ddr_wlast),
      .bvalid (ddr_bvalid),
      .bready (ddr_bready),
      .bid    (ddr_bid),
      .bresp  (ddr_bresp),
      .arvalid(ddr_arvalid),
      .arready(ddr_arready),
      .araddr (ddr_araddr),
      .arid   (ddr_arid),
      .arlen  (ddr_arlen),
      .rvalid (ddr_rvalid),
      .rready (ddr_rready),
      .rid    (ddr_rid),
      .rdata  (ddr_rdata),
      .rresp  (ddr_rresp),
      .rlast  (ddr_rlast)
  );

  int errors = 0;

  task automatic fail(string msg);
    $display("[tb] ERROR: %s", msg);
    errors++;
  endtask

  // -------------------------------------------------------------------------------------------
  // OCL accesses. Inputs change on the falling edge; handshakes complete on the rising edge.
  // -------------------------------------------------------------------------------------------
  task automatic ocl_write(input logic [31:0] addr, input logic [31:0] data);
    @(negedge clk);
    awaddr  = addr;
    wdata   = data;
    wstrb   = 4'hF;
    awvalid = 1'b1;
    wvalid  = 1'b1;
    #1;
    while (!awready) begin
      @(negedge clk);
      #1;
    end
    @(negedge clk);
    awvalid = 1'b0;
    wvalid  = 1'b0;
    #1;
    while (!bvalid) begin
      @(negedge clk);
      #1;
    end
    if (bresp != 2'b00) fail($sformatf("OCL write 0x%0h: bresp %0d", addr, bresp));
  endtask

  task automatic ocl_read(input logic [31:0] addr, output logic [31:0] data);
    @(negedge clk);
    araddr  = addr;
    arvalid = 1'b1;
    #1;
    while (!arready) begin
      @(negedge clk);
      #1;
    end
    @(negedge clk);
    arvalid = 1'b0;
    #1;
    while (!rvalid) begin
      @(negedge clk);
      #1;
    end
    data = rdata;
    if (rresp != 2'b00) fail($sformatf("OCL read 0x%0h: rresp %0d", addr, rresp));
  endtask

  // Host protocol shared with the AWS HDK tests.
  `define F2_OCL_WRITE(addr, data) ocl_write(addr, data)
  `define F2_OCL_READ(addr, data) ocl_read(addr, data)
  `define F2_ERROR(msg) fail(msg)
  `include "f2_host_tasks.svh"

  // Expected DDR contents written by fpga/sw/f2_smoke_test.c.
  function automatic logic [31:0] pattern(int i);
    return 32'hA500_0000 ^ (32'(i) * 32'h0101_0101);
  endfunction

  // -------------------------------------------------------------------------------------------
  // Test.
  // -------------------------------------------------------------------------------------------
  initial begin
    string image, expect_text, console;
    int timeout_us, ddr_check;
    logic [31:0] v, status, result, entry, result_addr;
    logic [127:0] line;
    time t_start;
    automatic int offsets[9] = '{0, 1, 7, 8, 9, 15, 16, 31, 1024};

    if (!$value$plusargs("image=%s", image)) image = "";
    if (!$value$plusargs("timeout_us=%d", timeout_us)) timeout_us = 20000;
    if (!$value$plusargs("expect=%s", expect_text)) expect_text = "PASS";
    if (!$value$plusargs("ddr_check=%d", ddr_check)) ddr_check = 0;

    repeat (20) @(posedge clk);
    rst_n = 1'b1;
    repeat (200) @(posedge clk);

    ocl_read(F2RegId, v);
    if (v != F2IdValue) fail($sformatf("ID 0x%08h", v));
    ocl_read(F2RegSpiCmdFree, v);
    $display("[tb] OCL up, ID ok, %0d free SPI command entries", v);

    ocl_write(F2RegSpiHalf, 3);  // 41.7 MHz SCLK keeps the simulation short
    ocl_write(F2RegSpiCsb, 1);
    ocl_write(F2RegSpiIdle, 4);

    if (image == "") begin
      fail("no +image");
    end else begin
      f2_load_image(image, entry, result_addr);

      // Read back the first word of the image (ITCM) to check the SPI read path.
      f2_spi_read_word(entry, v);
      $display("[tb] word at entry 0x%08h = 0x%08h", entry, v);

      f2_start_core(entry);
      t_start = $time;
      $display("[tb] core started at %0t", $time);

      // Poll every 2000 cycles of clk_main_a0 (8 us).
      for (int polls = 0; polls * 8 < timeout_us; polls++) begin
        repeat (2000) @(posedge clk);
        ocl_read(F2RegStatus, status);
        if (status[0]) break;
      end
      if (!status[0]) fail($sformatf("core did not halt within %0d us", timeout_us));
      else $display("[tb] core halted at %0t (%0t after start)", $time, $time - t_start);
      if (status[1]) fail("core halted with fault");

      // The SoC's UART keeps sending after the core halts (115200 baud: ~87 us per character),
      // so wait until the console FIFO stops growing for 200 us.
      begin
        logic [31:0] count, last;
        last = '1;
        forever begin
          repeat (50000) @(posedge clk);
          ocl_read(F2RegUartCount, count);
          if (count == last) break;
          last = count;
        end
      end
      console = "";
      f2_read_console(console);
      $display("[tb] console: \"%s\"", console);
      if (!f2_contains(console, expect_text)) begin
        fail($sformatf("console does not contain \"%s\"", expect_text));
      end

      if (result_addr != 0) begin
        f2_spi_read_word(result_addr, result);
        $display("[tb] f2_result @0x%08h = 0x%08h", result_addr, result);
        if (result != F2Pass) fail("f2_result is not PASS");
      end

      if (ddr_check != 0) begin
        // Through the SoC: SPI read of the first DDR beat (words 0, 1 written; 2, 3 not).
        f2_spi_read_line(32'h8000_0000, line);
        if (line[31:0] != pattern(0) || line[63:32] != pattern(1) || line[127:64] != '0) begin
          fail($sformatf("SPI read of DDR beat 0: 0x%032h", line));
        end
        // Directly in the DDR model: every word, in both 256-bit halves of a 512-bit line.
        for (int i = 0; i < 9; i++) begin
          v = u_ddr.peek32(64'(offsets[i] * 4));
          if (v != pattern(i)) begin
            fail($sformatf("DDR word %0d: 0x%08h, expected 0x%08h", offsets[i], v, pattern(i)));
          end
        end
        v = u_ddr.peek32(64'h30);
        if (v != 32'h1122_3344) fail($sformatf("DDR 0x30: 0x%08h, expected 0x11223344", v));
        $display("[tb] DDR contents checked");
      end
    end

    // Soft reset through CTRL: the SoC goes back to reset, the host registers keep working.
    ocl_write(F2RegCtrl, 1);
    repeat (50) @(posedge clk);
    ocl_read(F2RegStatus, status);
    if (status[0]) fail("halted still set while the SoC is held in reset");
    ocl_write(F2RegCtrl, 0);
    ocl_read(F2RegId, v);
    if (v != F2IdValue) fail("ID after soft reset");

    if (errors == 0) $display("[tb] TEST PASSED");
    else $display("[tb] TEST FAILED: %0d error(s)", errors);
    $finish;
  end

  // Hard limit, in case the test itself hangs.
  initial begin
    #(200ms);
    $display("[tb] TEST FAILED: simulation time limit");
    $finish;
  end

endmodule
