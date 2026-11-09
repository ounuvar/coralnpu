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

// Access to the CL's OCL registers. Backends: the F2 BAR (ocl_pci.h), a socket
// to the local Verilator testbench (ocl_socket.h), and a model in the tests.

#ifndef SW_UTILS_F2_LOADER_OCL_H_
#define SW_UTILS_F2_LOADER_OCL_H_

#include <cstdint>

namespace f2 {

class Ocl {
 public:
  virtual ~Ocl() = default;
  // 32-bit register access at a byte offset (4-byte aligned) in BAR0.
  virtual uint32_t Read32(uint32_t offset)              = 0;
  virtual void Write32(uint32_t offset, uint32_t value) = 0;
  // Lets time pass: a real sleep on F2, simulated clk_main_a0 cycles in the
  // testbench.
  virtual void SleepUs(uint32_t us) = 0;
};

}  // namespace f2

#endif  // SW_UTILS_F2_LOADER_OCL_H_
