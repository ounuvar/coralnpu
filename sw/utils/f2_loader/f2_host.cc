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

#include "sw/utils/f2_loader/f2_host.h"

#include <cstdio>
#include <cstring>

#include "sw/utils/f2_loader/f2_regs.h"

namespace f2 {
namespace {

// Register reads before giving up on the SPI controller (busy, queue full).
// Each poll is an OCL read, so this is well over a second on F2.
constexpr int kMaxPolls = 1000000;

std::string Hex(uint32_t v) {
  char buf[16];
  snprintf(buf, sizeof(buf), "0x%08x", v);
  return buf;
}

uint32_t LoadLe32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
         static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
}

}  // namespace

bool F2Host::Fail(const std::string &msg) {
  error_ = msg;
  return false;
}

bool F2Host::Init() {
  const uint32_t id = ocl_->Read32(kRegId);
  if (id != kIdValue) {
    return Fail("OCL ID register reads " + Hex(id) + ", expected " + Hex(kIdValue) +
                " (is the CoralNPU AFI loaded?)");
  }
  free_ = 0;
  Queue(kRegSpiCsb, 1);
  Queue(kRegSpiIdle, 4);
  return WaitSpiIdle();
}

void F2Host::SetSpiHalf(uint32_t cycles) { ocl_->Write32(kRegSpiHalf, cycles); }

void F2Host::Queue(uint32_t reg, uint32_t value) {
  if (queue_stuck_)
    return;
  for (int polls = 0; free_ == 0; ++polls) {
    if (polls == kMaxPolls) {
      queue_stuck_ = true;  // reported by the next WaitSpiIdle()
      return;
    }
    free_ = ocl_->Read32(kRegSpiCmdFree);
    if (free_ > kSpiCmdQueueDepth)
      free_ = 0;  // not a CL value; keep polling
  }
  ocl_->Write32(reg, value);
  --free_;
}

bool F2Host::WaitSpiIdle() {
  if (queue_stuck_)
    return Fail("SPI command queue stays full");
  for (int polls = 0; polls < kMaxPolls; ++polls) {
    if (!(ocl_->Read32(kRegStatus) & kStatusSpiBusy))
      return true;
  }
  return Fail("SPI controller stays busy");
}

// Spi2TLULV2 frame header: [op, addr (4 bytes, big endian), len (2 bytes, beats - 1)].
void F2Host::Header(uint8_t op, uint32_t addr, uint16_t len) {
  Queue(kRegSpiTx4,
        op | ((addr >> 24) & 0xFF) << 8 | ((addr >> 16) & 0xFF) << 16 | ((addr >> 8) & 0xFF) << 24);
  Queue(kRegSpiTx1, addr & 0xFF);
  Queue(kRegSpiTx1, len >> 8);
  Queue(kRegSpiTx1, len & 0xFF);
}

bool F2Host::WriteLines(uint32_t addr, const uint8_t *data, size_t beats) {
  if (addr % kLineBytes)
    return Fail("WriteLines: address " + Hex(addr) + " not 16-byte aligned");
  for (size_t done = 0; done < beats;) {
    const size_t n = beats - done < kMaxWriteBeats ? beats - done : kMaxWriteBeats;
    Queue(kRegSpiCsb, 0);
    Header(kSpiOpWrite, addr + static_cast<uint32_t>(done * kLineBytes),
           static_cast<uint16_t>(n - 1));
    const uint8_t *p = data + done * kLineBytes;
    for (size_t i = 0; i < n * kLineBytes; i += 4)
      Queue(kRegSpiTx4, LoadLe32(p + i));
    Queue(kRegSpiIdle, 16);  // the bridge only moves data while the SPI clock runs
    Queue(kRegSpiCsb, 1);
    done += n;
  }
  return WaitSpiIdle();
}

int F2Host::FindLine(const std::vector<uint8_t> &rx, uint8_t out[16]) {
  auto bit           = [&rx](size_t b) { return (rx[b / 8] >> (7 - b % 8)) & 1; };
  const size_t total = rx.size() * 8;
  uint32_t window    = 0;
  for (size_t b = 0; b + 128 < total; ++b) {
    window = ((window << 1) | bit(b)) & 0xFF;
    if (b < 7 || window != kSpiToken)
      continue;
    const size_t start = b + 1;
    if (start + 128 > total)
      return -1;
    for (int i = 0; i < 16; ++i) {
      uint8_t v = 0;
      for (int k = 0; k < 8; ++k)
        v = static_cast<uint8_t>(v << 1 | bit(start + i * 8 + k));
      out[i] = v;
    }
    return static_cast<int>(start);
  }
  return -1;
}

bool F2Host::ReadLine(uint32_t addr, uint8_t out[16]) {
  if (addr % kLineBytes)
    return Fail("ReadLine: address " + Hex(addr) + " not 16-byte aligned");
  // Bytes left from an earlier, failed read would shift this one.
  if (!WaitSpiIdle())
    return false;
  for (uint32_t n = ocl_->Read32(kRegSpiRxCount); n > 0; --n)
    ocl_->Read32(kRegSpiRxData);
  Queue(kRegSpiCsb, 0);
  Header(kSpiOpRead, addr, 0);
  Queue(kRegSpiRx, kSpiReadWindow);
  Queue(kRegSpiIdle, 4);
  Queue(kRegSpiCsb, 1);
  if (!WaitSpiIdle())
    return false;
  std::vector<uint8_t> rx;
  for (uint32_t n = ocl_->Read32(kRegSpiRxCount); n > 0; --n) {
    const uint32_t v = ocl_->Read32(kRegSpiRxData);
    if (!(v & kRxValid))
      break;
    rx.push_back(static_cast<uint8_t>(v));
  }
  if (FindLine(rx, out) < 0) {
    std::string bytes;
    for (size_t i = 0; i < rx.size() && i < 32; ++i) {
      char b[4];
      snprintf(b, sizeof(b), " %02x", rx[i]);
      bytes += b;
    }
    return Fail("SPI read " + Hex(addr) + ": no response token in " + std::to_string(rx.size()) +
                " bytes:" + bytes + (rx.size() > 32 ? " ..." : ""));
  }
  return true;
}

bool F2Host::ReadData(uint32_t addr, size_t len, uint8_t *out) {
  size_t done = 0;
  while (done < len) {
    const uint32_t cur  = addr + static_cast<uint32_t>(done);
    const uint32_t line = cur & ~(kLineBytes - 1);
    const size_t off    = cur - line;
    uint8_t buf[16];
    if (!ReadLine(line, buf))
      return false;
    const size_t n = len - done < kLineBytes - off ? len - done : kLineBytes - off;
    memcpy(out + done, buf + off, n);
    done += n;
  }
  return true;
}

bool F2Host::WriteData(uint32_t addr, const uint8_t *data, size_t len) {
  if (len == 0)
    return true;
  const uint32_t first = addr & ~(kLineBytes - 1);
  const uint32_t end   = addr + static_cast<uint32_t>(len);
  const uint32_t last  = (end + kLineBytes - 1) & ~(kLineBytes - 1);
  std::vector<uint8_t> buf(last - first);
  // A partial first or last line keeps the bytes around the data. When both
  // ends fall in one line, it is read once.
  const uint32_t last_line = last - kLineBytes;
  if (addr != first && !ReadLine(first, buf.data()))
    return false;
  if (end != last && !(last_line == first && addr != first) &&
      !ReadLine(last_line, buf.data() + (last_line - first))) {
    return false;
  }
  memcpy(buf.data() + (addr - first), data, len);
  return WriteLines(first, buf.data(), buf.size() / kLineBytes);
}

bool F2Host::WriteWord(uint32_t addr, uint32_t value) {
  uint8_t b[4] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
                  static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)};
  return WriteData(addr, b, 4);
}

bool F2Host::ReadWord(uint32_t addr, uint32_t *value) {
  uint8_t b[4];
  if (!ReadData(addr, 4, b))
    return false;
  *value = LoadLe32(b);
  return true;
}

bool F2Host::SetEntryPoint(uint32_t csr_base, uint32_t entry) {
  return WriteWord(csr_base + 4, entry);
}

bool F2Host::StartCore(uint32_t csr_base) {
  if (!WriteWord(csr_base, 1))
    return false;
  ocl_->SleepUs(1);
  return WriteWord(csr_base, 0);
}

void F2Host::SoftReset() {
  ocl_->Write32(kRegCtrl, kCtrlHoldReset);
  ocl_->SleepUs(10);
  ocl_->Write32(kRegCtrl, 0);
  ocl_->SleepUs(10);
}

uint32_t F2Host::Status() { return ocl_->Read32(kRegStatus); }

std::string F2Host::ReadConsole() {
  std::string text;
  for (;;) {
    const uint32_t v = ocl_->Read32(kRegUartData);
    if (!(v & kRxValid))
      break;
    text.push_back(static_cast<char>(v & 0xFF));
  }
  return text;
}

}  // namespace f2
