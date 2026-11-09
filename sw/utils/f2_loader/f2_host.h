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

// Host side of the CoralNPU F2 CL: SoC memory access over the CL's SPI
// controller, core control and the UART console, all through OCL registers.
// The steps follow fpga/f2/README.md ("Loading and running a program") and
// fpga/f2/verif/tests/f2_host_tasks.svh.

#ifndef SW_UTILS_F2_LOADER_F2_HOST_H_
#define SW_UTILS_F2_LOADER_F2_HOST_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sw/utils/f2_loader/ocl.h"

namespace f2 {

class F2Host {
 public:
  explicit F2Host(Ocl *ocl) : ocl_(ocl) {}

  // Checks the ID register and puts the SPI link in its idle state. Call once
  // before anything else. Fails if no CL (or a different one) answers.
  bool Init();

  // SPI clock half period in clk_main_a0 cycles (3 or more).
  void SetSpiHalf(uint32_t cycles);

  // SoC memory. Addresses are SoC addresses (ITCM, DTCM, CSRs, DDR). Writes
  // that do not cover whole 16-byte lines read the line first.
  bool WriteData(uint32_t addr, const uint8_t *data, size_t len);
  bool ReadData(uint32_t addr, size_t len, uint8_t *out);
  bool WriteWord(uint32_t addr, uint32_t value);
  bool ReadWord(uint32_t addr, uint32_t *value);
  // Whole 16-byte lines; addr must be 16-byte aligned.
  bool WriteLines(uint32_t addr, const uint8_t *data, size_t beats);
  bool ReadLine(uint32_t addr, uint8_t out[16]);

  // Core control through the coralnpu_device CSRs at csr_base.
  bool SetEntryPoint(uint32_t csr_base, uint32_t entry);
  // Releases the clock gate and reset, as the upstream loader does.
  bool StartCore(uint32_t csr_base);

  // Holds the whole SoC in reset (CTRL[0]) and releases it.
  void SoftReset();

  uint32_t Status();
  // Pops the console FIFO until it is empty.
  std::string ReadConsole();

  const std::string &error() const { return error_; }

  // Finds the 0xFE response token at any bit offset (MSB of each byte first)
  // and extracts the 16 bytes after it. Returns the bit offset of the first
  // data bit, or -1 if there is no token followed by 128 bits.
  static int FindLine(const std::vector<uint8_t> &rx, uint8_t out[16]);

 private:
  // Queued registers: never write one while the command queue is full. A full
  // queue stalls the OCL write in the CL, and the F2 shell times out OCL
  // accesses the CL does not answer.
  void Queue(uint32_t reg, uint32_t value);
  void Header(uint8_t op, uint32_t addr, uint16_t len);
  bool WaitSpiIdle();
  bool Fail(const std::string &msg);

  Ocl *ocl_;
  uint32_t free_    = 0;  // command queue entries known to be free
  bool queue_stuck_ = false;
  std::string error_;
};

}  // namespace f2

#endif  // SW_UTILS_F2_LOADER_F2_HOST_H_
