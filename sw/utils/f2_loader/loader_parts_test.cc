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

// The ELF reader and the F2 slot lookup.

#include <sys/stat.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sw/utils/f2_loader/elf_image.h"
#include "sw/utils/f2_loader/ocl_pci.h"

namespace f2 {
namespace {

void Put32(std::vector<uint8_t> &f, size_t o, uint32_t v) {
  for (int i = 0; i < 4; ++i)
    f[o + i] = static_cast<uint8_t>(v >> (8 * i));
}
void Put16(std::vector<uint8_t> &f, size_t o, uint16_t v) {
  f[o]     = static_cast<uint8_t>(v);
  f[o + 1] = static_cast<uint8_t>(v >> 8);
}

// A RISC-V ELF32 with three program headers: a PT_LOAD with data, a PT_LOAD
// with no file data (.bss), and a non-load header.
std::vector<uint8_t> TinyElf() {
  std::vector<uint8_t> f(52 + 3 * 32 + 8, 0);
  f[0] = 0x7F;
  f[1] = 'E';
  f[2] = 'L';
  f[3] = 'F';
  f[4] = 1;            // ELFCLASS32
  f[5] = 1;            // little endian
  Put16(f, 18, 243);   // EM_RISCV
  Put32(f, 24, 0x80);  // entry
  Put32(f, 28, 52);    // phoff
  Put16(f, 42, 32);    // phentsize
  Put16(f, 44, 3);     // phnum
  const size_t data = 52 + 3 * 32;
  // PT_LOAD: 8 bytes at paddr 0x10000 (vaddr differs on purpose).
  Put32(f, 52 + 0, 1);
  Put32(f, 52 + 4, data);
  Put32(f, 52 + 8, 0x90000);
  Put32(f, 52 + 12, 0x10000);
  Put32(f, 52 + 16, 8);
  // PT_LOAD with filesz 0.
  Put32(f, 84 + 0, 1);
  Put32(f, 84 + 12, 0x20000);
  // PT_NOTE.
  Put32(f, 116 + 0, 4);
  for (int i = 0; i < 8; ++i)
    f[data + i] = static_cast<uint8_t>(0xA0 + i);
  return f;
}

TEST(ElfImage, LoadSegmentsAtPaddr) {
  ElfImage image;
  std::string error;
  ASSERT_TRUE(ParseElfImage(TinyElf(), &image, &error)) << error;
  EXPECT_EQ(image.entry, 0x80u);
  ASSERT_EQ(image.segments.size(), 1u);
  EXPECT_EQ(image.segments[0].addr, 0x10000u);
  EXPECT_EQ(image.segments[0].data,
            (std::vector<uint8_t>{0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7}));
}

TEST(ElfImage, Rejects) {
  ElfImage image;
  std::string error;
  std::vector<uint8_t> f = TinyElf();
  f[4]                   = 2;  // ELFCLASS64
  EXPECT_FALSE(ParseElfImage(f, &image, &error));
  f = TinyElf();
  Put16(f, 18, 62);  // x86-64
  EXPECT_FALSE(ParseElfImage(f, &image, &error));
  f = TinyElf();
  Put32(f, 52 + 16, 4096);  // runs past the end
  EXPECT_FALSE(ParseElfImage(f, &image, &error));
  EXPECT_FALSE(ParseElfImage(std::vector<uint8_t>(10, 0), &image, &error));
}

void Device(const std::string &root, const std::string &bdf, const char *vendor,
            const char *device) {
  const std::string dir = root + "/" + bdf;
  mkdir(dir.c_str(), 0755);
  std::ofstream(dir + "/vendor") << vendor << "\n";
  std::ofstream(dir + "/device") << device << "\n";
}

TEST(PciOcl, SlotToBdf) {
  const std::string root = std::string(getenv("TEST_TMPDIR")) + "/sysfs";
  mkdir(root.c_str(), 0755);
  // Two F2 FPGAs (management PF .1, application PF .0) and an unrelated device.
  Device(root, "0000:36:00.1", "0x1d0f", "0x9248");
  Device(root, "0000:36:00.0", "0x1d0f", "0x9048");
  Device(root, "0000:34:00.1", "0x1d0f", "0x9248");
  Device(root, "0000:34:00.0", "0x1d0f", "0x9048");
  Device(root, "0000:00:1f.0", "0x8086", "0x1234");
  std::string error;
  EXPECT_EQ(PciOcl::SlotToBdf(0, &error, root), "0000:34:00.0") << error;
  EXPECT_EQ(PciOcl::SlotToBdf(1, &error, root), "0000:36:00.0") << error;
  EXPECT_EQ(PciOcl::SlotToBdf(2, &error, root), "");
  EXPECT_NE(error.find("not found"), std::string::npos) << error;
}

}  // namespace
}  // namespace f2
