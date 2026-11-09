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

// The loadable segments of a 32-bit little-endian RISC-V ELF, read without
// libelf so the loader builds anywhere.

#ifndef SW_UTILS_F2_LOADER_ELF_IMAGE_H_
#define SW_UTILS_F2_LOADER_ELF_IMAGE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace f2 {

struct ElfSegment {
  uint32_t addr;              // p_paddr, as fpga/f2/tb/elf_to_f2_image.py uses
  std::vector<uint8_t> data;  // the p_filesz bytes; .bss is cleared by the program
};

struct ElfImage {
  uint32_t entry = 0;
  std::vector<ElfSegment> segments;
};

// Reads the PT_LOAD segments with file data. On failure returns false and sets
// *error.
bool ReadElfImage(const std::string &path, ElfImage *image, std::string *error);
bool ParseElfImage(const std::vector<uint8_t> &file, ElfImage *image, std::string *error);

}  // namespace f2

#endif  // SW_UTILS_F2_LOADER_ELF_IMAGE_H_
