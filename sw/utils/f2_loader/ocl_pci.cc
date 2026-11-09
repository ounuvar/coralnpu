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

#include "sw/utils/f2_loader/ocl_pci.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace f2 {
namespace {

constexpr uint32_t kAmazonVendor = 0x1d0f;
constexpr uint32_t kF2MgmtDevice = 0x9248;

bool ReadHex(const std::string &path, uint32_t *value) {
  std::ifstream in(path);
  std::string s;
  if (!(in >> s))
    return false;
  *value = static_cast<uint32_t>(strtoul(s.c_str(), nullptr, 16));
  return true;
}

}  // namespace

std::string PciOcl::SlotToBdf(int slot, std::string *error, const std::string &sysfs) {
  DIR *dir = opendir(sysfs.c_str());
  if (!dir) {
    *error = "cannot list " + sysfs + ": " + strerror(errno);
    return "";
  }
  std::vector<std::string> mgmt;
  while (dirent *e = readdir(dir)) {
    const std::string name = e->d_name;
    if (name[0] == '.')
      continue;
    uint32_t vendor, device;
    if (ReadHex(sysfs + "/" + name + "/vendor", &vendor) &&
        ReadHex(sysfs + "/" + name + "/device", &device) && vendor == kAmazonVendor &&
        device == kF2MgmtDevice) {
      mgmt.push_back(name);
    }
  }
  closedir(dir);
  std::sort(mgmt.begin(), mgmt.end());
  if (slot < 0 || static_cast<size_t>(slot) >= mgmt.size()) {
    *error = "FPGA slot " + std::to_string(slot) + " not found (" + std::to_string(mgmt.size()) +
             " F2 FPGA(s) in " + sysfs + ")";
    return "";
  }
  std::string bdf  = mgmt[slot];
  const size_t dot = bdf.rfind('.');
  if (dot == std::string::npos) {
    *error = "unexpected PCI address " + bdf;
    return "";
  }
  return bdf.substr(0, dot + 1) + "0";  // the application PF
}

std::unique_ptr<PciOcl> PciOcl::Open(const std::string &bdf, std::string *error) {
  const std::string path = "/sys/bus/pci/devices/" + bdf + "/resource0";
  const int fd           = open(path.c_str(), O_RDWR | O_SYNC);
  if (fd < 0) {
    *error = "cannot open " + path + ": " + strerror(errno) +
             (errno == EACCES   ? " (run as root)"
              : errno == ENOENT ? " (is an AFI loaded?)"
                                : "");
    return nullptr;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    *error = "cannot size " + path;
    close(fd);
    return nullptr;
  }
  const size_t size = static_cast<size_t>(st.st_size);
  void *p           = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    *error = "cannot map " + path + ": " + strerror(errno);
    return nullptr;
  }
  return std::unique_ptr<PciOcl>(new PciOcl(static_cast<volatile uint8_t *>(p), size));
}

PciOcl::~PciOcl() { munmap(const_cast<uint8_t *>(base_), size_); }

uint32_t PciOcl::Read32(uint32_t offset) {
  if (offset % 4 || offset + 4 > size_)
    return 0xFFFFFFFF;
  return *reinterpret_cast<volatile uint32_t *>(base_ + offset);
}

void PciOcl::Write32(uint32_t offset, uint32_t value) {
  if (offset % 4 || offset + 4 > size_)
    return;
  *reinterpret_cast<volatile uint32_t *>(base_ + offset) = value;
}

void PciOcl::SleepUs(uint32_t us) { usleep(us); }

}  // namespace f2
