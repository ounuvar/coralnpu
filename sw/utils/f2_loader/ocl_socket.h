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

// OCL backend for the local Verilator testbench: requests over a Unix socket to
// cl_coralnpu_core_tb running with +host_socket=<path> (fpga/f2/tb).
//
// Protocol: each request is three little-endian 32-bit words (op, offset,
// data), answered by one 32-bit word (the read value, or 0).

#ifndef SW_UTILS_F2_LOADER_OCL_SOCKET_H_
#define SW_UTILS_F2_LOADER_OCL_SOCKET_H_

#include <cstdint>
#include <memory>
#include <string>

#include "sw/utils/f2_loader/ocl.h"

namespace f2 {

// Must match fpga/f2/tb/cl_coralnpu_core_tb.sv.
constexpr uint32_t kSocketOpRead  = 1;
constexpr uint32_t kSocketOpWrite = 2;
constexpr uint32_t kSocketOpWait  = 3;  // data = clk_main_a0 cycles
constexpr uint32_t kSocketOpDone  = 4;

constexpr uint32_t kClkMainMhz = 250;  // clk_main_a0, clock recipe A0

class SocketOcl : public Ocl {
 public:
  // Connects to the testbench, retrying for up to timeout_s while it starts.
  static std::unique_ptr<SocketOcl> Connect(const std::string &path, double timeout_s,
                                            std::string *error);
  ~SocketOcl() override;  // ends the session

  uint32_t Read32(uint32_t offset) override;
  void Write32(uint32_t offset, uint32_t value) override;
  void SleepUs(uint32_t us) override;

 private:
  explicit SocketOcl(int fd) : fd_(fd) {}
  uint32_t Request(uint32_t op, uint32_t offset, uint32_t data);

  int fd_;
};

}  // namespace f2

#endif  // SW_UTILS_F2_LOADER_OCL_SOCKET_H_
