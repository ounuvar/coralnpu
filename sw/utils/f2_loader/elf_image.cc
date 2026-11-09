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

#include "sw/utils/f2_loader/elf_image.h"

#include <fstream>
#include <iterator>

namespace f2 {
namespace {

// ELF32 field offsets (System V gABI).
constexpr size_t kEhdrSize       = 52;
constexpr size_t kEntry          = 24;
constexpr size_t kPhoff          = 28;
constexpr size_t kPhentsize      = 42;
constexpr size_t kPhnum          = 44;
constexpr size_t kPhdrSize       = 32;
constexpr uint32_t kPtLoad       = 1;
constexpr uint16_t kMachineRiscv = 243;

uint32_t U32(const std::vector<uint8_t> &f, size_t o) {
  return static_cast<uint32_t>(f[o]) | static_cast<uint32_t>(f[o + 1]) << 8 |
         static_cast<uint32_t>(f[o + 2]) << 16 | static_cast<uint32_t>(f[o + 3]) << 24;
}

uint16_t U16(const std::vector<uint8_t> &f, size_t o) {
  return static_cast<uint16_t>(f[o] | f[o + 1] << 8);
}

}  // namespace

bool ParseElfImage(const std::vector<uint8_t> &f, ElfImage *image, std::string *error) {
  if (f.size() < kEhdrSize || f[0] != 0x7F || f[1] != 'E' || f[2] != 'L' || f[3] != 'F') {
    *error = "not an ELF file";
    return false;
  }
  if (f[4] != 1 || f[5] != 1) {
    *error = "not a 32-bit little-endian ELF";
    return false;
  }
  if (U16(f, 18) != kMachineRiscv) {
    *error = "not a RISC-V ELF";
    return false;
  }
  const uint32_t phoff     = U32(f, kPhoff);
  const uint16_t phentsize = U16(f, kPhentsize);
  const uint16_t phnum     = U16(f, kPhnum);
  if (phentsize < kPhdrSize || phoff + static_cast<uint64_t>(phnum) * phentsize > f.size()) {
    *error = "bad program header table";
    return false;
  }
  image->entry = U32(f, kEntry);
  image->segments.clear();
  for (uint16_t i = 0; i < phnum; ++i) {
    const size_t ph = phoff + static_cast<size_t>(i) * phentsize;
    if (U32(f, ph) != kPtLoad)
      continue;
    const uint32_t offset = U32(f, ph + 4);
    const uint32_t paddr  = U32(f, ph + 12);
    const uint32_t filesz = U32(f, ph + 16);
    if (filesz == 0)
      continue;
    if (static_cast<uint64_t>(offset) + filesz > f.size()) {
      *error = "segment " + std::to_string(i) + " runs past the end of the file";
      return false;
    }
    image->segments.push_back(
        {paddr, std::vector<uint8_t>(f.begin() + offset, f.begin() + offset + filesz)});
  }
  return true;
}

bool ReadElfImage(const std::string &path, ElfImage *image, std::string *error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  const std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
  if (!ParseElfImage(file, image, error)) {
    *error = path + ": " + *error;
    return false;
  }
  return true;
}

}  // namespace f2
