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

// Host-side protocol for the CoralNPU F2 CL, shared by the local Verilator testbench
// (fpga/f2/tb) and the AWS HDK tests (fpga/f2/verif/tests). The F2 host software (#15)
// implements the same steps over the OCL BAR.
//
// Before including, define:
//   `F2_OCL_WRITE(addr, data)   32-bit OCL write (blocking until the write response)
//   `F2_OCL_READ(addr, data)    32-bit OCL read into `data`
//   `F2_ERROR(msg)              report an error and count it

`ifndef F2_HOST_TASKS_SVH
`define F2_HOST_TASKS_SVH

// OCL register offsets (fpga/rtl/f2_ocl_regs.sv).
localparam logic [31:0] F2RegId = 32'h00;
localparam logic [31:0] F2RegCtrl = 32'h08;
localparam logic [31:0] F2RegStatus = 32'h0C;
localparam logic [31:0] F2RegSpiHalf = 32'h10;
localparam logic [31:0] F2RegUartBit = 32'h14;
localparam logic [31:0] F2RegSpiCsb = 32'h20;
localparam logic [31:0] F2RegSpiTx1 = 32'h24;
localparam logic [31:0] F2RegSpiTx4 = 32'h28;
localparam logic [31:0] F2RegSpiRx = 32'h2C;
localparam logic [31:0] F2RegSpiIdle = 32'h30;
localparam logic [31:0] F2RegSpiRxData = 32'h34;
localparam logic [31:0] F2RegSpiRxCount = 32'h38;
localparam logic [31:0] F2RegSpiCmdFree = 32'h3C;
localparam logic [31:0] F2RegUartData = 32'h40;
localparam logic [31:0] F2RegUartCount = 32'h44;
localparam logic [31:0] F2IdValue = 32'h434E_5055;

localparam logic [31:0] F2CsrBase = 32'h0020_0000;  // highmem coralnpu_device CSRs
localparam logic [31:0] F2Pass = 32'h600D_F00D;  // fpga/sw/f2_smoke_test.c
// Bytes clocked in per SPI read; the response token arrives after a variable delay (longer for
// DDR, which crosses into the DDR clock domain). The RX FIFO holds 256.
localparam int F2SpiReadWindow = 200;

task automatic f2_spi_wait_idle();
  logic [31:0] status;
  do `F2_OCL_READ(F2RegStatus, status); while (status[3]);
endtask

// Upstream SPI frame (Spi2TLULV2): [op, addr (4 B, big endian), len (2 B, beats - 1)].
task automatic f2_spi_header(input logic [7:0] op, input logic [31:0] addr, input logic [15:0] len);
  `F2_OCL_WRITE(F2RegSpiTx4, {addr[15:8], addr[23:16], addr[31:24], op});
  `F2_OCL_WRITE(F2RegSpiTx1, {24'd0, addr[7:0]});
  `F2_OCL_WRITE(F2RegSpiTx1, {24'd0, len[15:8]});
  `F2_OCL_WRITE(F2RegSpiTx1, {24'd0, len[7:0]});
endtask

// Write `beats` 16-byte beats, words[first*4 ...], to addr (16-byte aligned).
task automatic f2_spi_write(input logic [31:0] addr, ref logic [31:0] words[$], input int first,
                            input int beats);
  `F2_OCL_WRITE(F2RegSpiCsb, 0);
  f2_spi_header(8'h02, addr, 16'(beats - 1));
  for (int i = first * 4; i < (first + beats) * 4; i++) `F2_OCL_WRITE(F2RegSpiTx4, words[i]);
  `F2_OCL_WRITE(F2RegSpiIdle, 16);  // drain the bridge's clock crossing
  `F2_OCL_WRITE(F2RegSpiCsb, 1);
endtask

// `width` bits (at most 32) of a byte stream starting at bit `bit_off`, MSB of byte 0 first.
function automatic logic [31:0] f2_bits(ref byte unsigned bytes[$], input int bit_off,
                                        input int width);
  logic [31:0] v = '0;
  for (int i = 0; i < width; i++) begin
    int b = bit_off + i;
    v = {v[30:0], bytes[b/8][7-(b%8)]};
  end
  return v;
endfunction

// Read one 16-byte line. The response starts with a 0xFE token after a variable delay.
task automatic f2_spi_read_line(input logic [31:0] addr, output logic [127:0] line);
  logic [31:0] count, v;
  byte unsigned rx[$];
  int sync;
  `F2_OCL_WRITE(F2RegSpiCsb, 0);
  f2_spi_header(8'h01, addr, 16'd0);
  `F2_OCL_WRITE(F2RegSpiRx, F2SpiReadWindow);
  `F2_OCL_WRITE(F2RegSpiIdle, 4);
  `F2_OCL_WRITE(F2RegSpiCsb, 1);
  f2_spi_wait_idle();
  `F2_OCL_READ(F2RegSpiRxCount, count);
  for (int i = 0; i < count; i++) begin
    `F2_OCL_READ(F2RegSpiRxData, v);
    rx.push_back(v[7:0]);
  end
  // The token is usually byte-aligned, but not always (the response can start at any SPI clock
  // edge), so search at every bit offset, MSB first, as the upstream cocotb SPIMaster does.
  sync = -1;
  for (int b = 0; b + 8 + 128 <= rx.size() * 8; b++) begin
    if (f2_bits(rx, b, 8) == 8'hFE) begin
      sync = b + 8;
      break;
    end
  end
  line = '0;
  if (sync < 0) begin
    string bytes = "";
    foreach (rx[i]) bytes = {bytes, $sformatf(" %02h", rx[i])};
    `F2_ERROR($sformatf("SPI read 0x%08h: no response token in %0d bytes:%s", addr, rx.size(), bytes
              ));
  end else begin
    for (int i = 0; i < 16; i++) line[i*8+:8] = 8'(f2_bits(rx, sync + i * 8, 8));
  end
endtask

task automatic f2_spi_read_word(input logic [31:0] addr, output logic [31:0] value);
  logic [127:0] line;
  f2_spi_read_line({addr[31:4], 4'h0}, line);
  value = line[addr[3:2]*32+:32];
endtask

// Read-modify-write of the 16-byte line holding addr (SPI writes are whole lines).
task automatic f2_spi_write_word(input logic [31:0] addr, input logic [31:0] value);
  logic [127:0] line;
  logic [ 31:0] words[$];
  f2_spi_read_line({addr[31:4], 4'h0}, line);
  line[addr[3:2]*32+:32] = value;
  for (int i = 0; i < 4; i++) words.push_back(line[i*32+:32]);
  f2_spi_write({addr[31:4], 4'h0}, words, 0, 1);
endtask

// Load an image written by fpga/f2/tb/elf_to_f2_image.py. Returns the entry point and the
// address of symbol f2_result (0 if absent).
task automatic f2_load_image(input string path, output logic [31:0] entry,
                             output logic [31:0] result_addr);
  int fd, n, beats, total;
  string tag, name;
  logic [31:0] addr, w0, w1, w2, w3;
  logic [31:0] words[$];
  entry = '0;
  result_addr = '0;
  total = 0;
  fd = $fopen(path, "r");
  if (fd == 0) begin
    `F2_ERROR($sformatf("cannot open image %s", path));
    return;
  end
  while (!$feof(
      fd
  )) begin
    n = $fscanf(fd, "%s", tag);
    if (n != 1) break;
    if (tag == "S") begin
      n = $fscanf(fd, "%h %h", addr, beats);
      words.delete();
      for (int i = 0; i < beats; i++) begin
        n = $fscanf(fd, "%h %h %h %h", w0, w1, w2, w3);
        words.push_back(w0);
        words.push_back(w1);
        words.push_back(w2);
        words.push_back(w3);
      end
      for (int b = 0; b < beats; b += 64) begin  // frames of up to 1 KB
        f2_spi_write(addr + 32'(b * 16), words, b, (beats - b > 64) ? 64 : beats - b);
      end
      total += beats * 16;
    end else if (tag == "Y") begin
      n = $fscanf(fd, "%s %h", name, addr);
      if (name == "f2_result") result_addr = addr;
    end else if (tag == "E") begin
      n = $fscanf(fd, "%h", entry);
    end
  end
  $fclose(fd);
  f2_spi_wait_idle();
  $display("[f2] loaded %s: %0d bytes, entry 0x%08h", path, total, entry);
endtask

// Set the start PC, release the clock gate, release reset (as the upstream SPI test does).
task automatic f2_start_core(input logic [31:0] entry);
  f2_spi_write_word(F2CsrBase + 32'h4, entry);
  f2_spi_write_word(F2CsrBase, 32'h1);
  f2_spi_write_word(F2CsrBase, 32'h0);
endtask

// Drain the UART console FIFO.
task automatic f2_read_console(inout string console);
  logic [31:0] v;
  forever begin
    `F2_OCL_READ(F2RegUartData, v);
    if (!v[8]) break;
    console = {console, string'(v[7:0])};
  end
endtask

function automatic bit f2_contains(string text, string pattern);
  for (int i = 0; i + pattern.len() <= text.len(); i++) begin
    if (text.substr(i, i + pattern.len() - 1) == pattern) return 1;
  end
  return 0;
endfunction

`endif  // F2_HOST_TASKS_SVH
