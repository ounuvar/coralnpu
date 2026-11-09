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

// OCL backend for an F2 instance: BAR0 of the FPGA's application PF, mapped
// through sysfs. This is the mapping the aws-fpga SDK's fpga_pci_attach()
// makes (open resource0 with O_SYNC, mmap), without the SDK as a dependency.
// Needs root, and an AFI loaded with fpga-load-local-image.

#ifndef SW_UTILS_F2_LOADER_OCL_PCI_H_
#define SW_UTILS_F2_LOADER_OCL_PCI_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "sw/utils/f2_loader/ocl.h"

namespace f2 {

class PciOcl : public Ocl {
 public:
  // FPGA slot N: the Nth management PF (vendor 0x1d0f, device 0x9248) in
  // /sys/bus/pci/devices order; the application PF is function 0 of the same
  // device, as in the SDK's fpga_pci_sysfs.c.
  static std::string SlotToBdf(int slot, std::string *error,
                               const std::string &sysfs = "/sys/bus/pci/devices");
  // bdf: "0000:34:00.0".
  static std::unique_ptr<PciOcl> Open(const std::string &bdf, std::string *error);
  ~PciOcl() override;

  uint32_t Read32(uint32_t offset) override;
  void Write32(uint32_t offset, uint32_t value) override;
  void SleepUs(uint32_t us) override;

 private:
  PciOcl(volatile uint8_t *base, size_t size) : base_(base), size_(size) {}

  volatile uint8_t *base_;
  size_t size_;
};

}  // namespace f2

#endif  // SW_UTILS_F2_LOADER_OCL_PCI_H_
