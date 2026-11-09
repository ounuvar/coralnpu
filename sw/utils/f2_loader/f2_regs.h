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

// Host registers of the CoralNPU F2 custom logic on OCL (BAR0), as defined by
// fpga/rtl/f2_ocl_regs.sv and documented in fpga/f2/README.md. The same offsets
// are in fpga/f2/verif/tests/f2_host_tasks.svh.

#ifndef SW_UTILS_F2_LOADER_F2_REGS_H_
#define SW_UTILS_F2_LOADER_F2_REGS_H_

#include <cstdint>

namespace f2 {

constexpr uint32_t kRegId         = 0x00;
constexpr uint32_t kRegVersion    = 0x04;
constexpr uint32_t kRegCtrl       = 0x08;
constexpr uint32_t kRegStatus     = 0x0C;
constexpr uint32_t kRegSpiHalf    = 0x10;
constexpr uint32_t kRegUartBit    = 0x14;
constexpr uint32_t kRegSpiCsb     = 0x20;  // queued
constexpr uint32_t kRegSpiTx1     = 0x24;  // queued
constexpr uint32_t kRegSpiTx4     = 0x28;  // queued
constexpr uint32_t kRegSpiRx      = 0x2C;  // queued
constexpr uint32_t kRegSpiIdle    = 0x30;  // queued
constexpr uint32_t kRegSpiRxData  = 0x34;  // pops
constexpr uint32_t kRegSpiRxCount = 0x38;
constexpr uint32_t kRegSpiCmdFree = 0x3C;
constexpr uint32_t kRegUartData   = 0x40;  // pops
constexpr uint32_t kRegUartCount  = 0x44;
constexpr uint32_t kRegCyclesLo   = 0x48;
constexpr uint32_t kRegCyclesHi   = 0x4C;

constexpr uint32_t kIdValue = 0x434E5055;  // "CNPU"

constexpr uint32_t kCtrlHoldReset = 1u << 0;

constexpr uint32_t kStatusHalted        = 1u << 0;
constexpr uint32_t kStatusFault         = 1u << 1;
constexpr uint32_t kStatusDdrReady      = 1u << 2;
constexpr uint32_t kStatusSpiBusy       = 1u << 3;
constexpr uint32_t kStatusSpiRxOverflow = 1u << 4;  // W1C
constexpr uint32_t kStatusUartOverflow  = 1u << 5;  // W1C

constexpr uint32_t kRxValid = 1u << 8;  // SPI_RXDATA and UART_RXDATA

constexpr uint32_t kSpiCmdQueueDepth = 32;
constexpr uint32_t kSpiRxFifoDepth   = 256;

// Spi2TLULV2 frame opcodes.
constexpr uint8_t kSpiOpRead  = 0x01;
constexpr uint8_t kSpiOpWrite = 0x02;
constexpr uint8_t kSpiToken   = 0xFE;

constexpr uint32_t kLineBytes = 16;
// Bytes clocked in per one-line read. The response token arrives after a
// variable delay (longer for DDR). Must stay below kSpiRxFifoDepth.
constexpr uint32_t kSpiReadWindow = 200;
// Beats per write frame (1 KB), as in the testbench.
constexpr uint32_t kMaxWriteBeats = 64;

// coralnpu_device CSRs in the highmem configuration (the one F2 builds).
constexpr uint32_t kCsrBaseHighmem = 0x00200000;

}  // namespace f2

#endif  // SW_UTILS_F2_LOADER_F2_REGS_H_
