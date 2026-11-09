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

// F2Host against a model of the CL's OCL registers and the SoC's SPI bridge.
// The RTL itself is checked end to end by the testbench in fpga/f2/tb.

#include "sw/utils/f2_loader/f2_host.h"

#include <cstring>
#include <deque>
#include <map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sw/utils/f2_loader/f2_regs.h"

namespace f2 {
namespace {

// The CL's registers and SPI controller, and the SoC's Spi2TLULV2 bridge, at
// the level of bytes. The SPI controller runs one queued command every
// `accesses_per_command` OCL accesses (or several per access when negative),
// so a host that ignores SPI_CMDFREE overflows the queue.
class FakeCl : public Ocl {
 public:
  uint32_t id              = kIdValue;
  int accesses_per_command = -3;  // 3 commands per access
  int latency_bits         = 13;  // response token start, in bits after the header
  bool idle_miso_one       = false;
  bool respond             = true;  // false: the bridge never answers a read
  bool halted              = false;
  std::map<uint32_t, uint8_t> mem;
  std::deque<uint8_t> console;
  std::vector<uint32_t> ctrl_writes;
  std::vector<std::pair<uint32_t, uint32_t>> word_writes;  // (line addr, word 0) per write frame
  int overflow_writes = 0;                                 // queued writes while the queue was full
  int frames          = 0;

  uint32_t Read32(uint32_t offset) override {
    Tick();
    switch (offset) {
      case kRegId:
        return id;
      case kRegStatus:
        return (halted ? kStatusHalted : 0) | (queue_.empty() ? 0 : kStatusSpiBusy);
      case kRegSpiCmdFree:
        return kSpiCmdQueueDepth - static_cast<uint32_t>(queue_.size());
      case kRegSpiRxCount:
        return static_cast<uint32_t>(rx_.size());
      case kRegSpiRxData: {
        if (rx_.empty())
          return 0;
        const uint8_t b = rx_.front();
        rx_.pop_front();
        return kRxValid | b;
      }
      case kRegUartData: {
        if (console.empty())
          return 0;
        const uint8_t b = console.front();
        console.pop_front();
        return kRxValid | b;
      }
      default:
        return 0;
    }
  }

  void Write32(uint32_t offset, uint32_t value) override {
    Tick();
    if (offset >= kRegSpiCsb && offset <= kRegSpiIdle) {
      if (queue_.size() >= kSpiCmdQueueDepth) {
        ++overflow_writes;  // the real CL would stall this OCL write
        return;
      }
      queue_.emplace_back(offset, value);
    } else if (offset == kRegCtrl) {
      ctrl_writes.push_back(value);
    }
  }

  void SleepUs(uint32_t) override { Drain(1 << 30); }

  uint8_t Mem(uint32_t a) const {
    auto it = mem.find(a);
    return it == mem.end() ? 0 : it->second;
  }

 private:
  void Tick() {
    if (accesses_per_command < 0) {
      Drain(-accesses_per_command);
    } else if (++accesses_ % accesses_per_command == 0) {
      Drain(1);
    }
  }

  void Drain(int n) {
    while (n-- > 0 && !queue_.empty()) {
      auto [reg, value] = queue_.front();
      queue_.pop_front();
      Execute(reg, value);
    }
  }

  void Execute(uint32_t reg, uint32_t value) {
    switch (reg) {
      case kRegSpiCsb:
        if (!csb_ && (value & 1))
          EndFrame();
        if (csb_ && !(value & 1))
          BeginFrame();
        csb_ = value & 1;
        break;
      case kRegSpiTx1:
        Clock(static_cast<uint8_t>(value), false);
        break;
      case kRegSpiTx4:
        for (int i = 0; i < 4; ++i)
          Clock(static_cast<uint8_t>(value >> (8 * i)), false);
        break;
      case kRegSpiRx:
        for (uint32_t i = 0; i < (value & 0xFFFF); ++i)
          Clock(0, true);
        break;
      case kRegSpiIdle:
        for (uint32_t i = 0; i < (value & 0xFFFF); ++i)
          Clock(0, false);
        break;
    }
  }

  void BeginFrame() {
    mosi_.clear();
    resp_bits_.clear();
  }

  void EndFrame() {
    ++frames;
    if (mosi_.size() < 7 || mosi_[0] != kSpiOpWrite)
      return;
    const uint32_t addr = Addr();
    const size_t beats  = (static_cast<size_t>(mosi_[5]) << 8 | mosi_[6]) + 1;
    if (mosi_.size() < 7 + beats * 16)
      return;  // short frame: the bridge writes nothing
    for (size_t i = 0; i < beats * 16; ++i)
      mem[addr + static_cast<uint32_t>(i)] = mosi_[7 + i];
    word_writes.emplace_back(addr, static_cast<uint32_t>(mosi_[7]) | mosi_[8] << 8 |
                                       mosi_[9] << 16 | static_cast<uint32_t>(mosi_[10]) << 24);
  }

  uint32_t Addr() const {
    return static_cast<uint32_t>(mosi_[1]) << 24 | mosi_[2] << 16 | mosi_[3] << 8 | mosi_[4];
  }

  // One SPI byte: MOSI in, MISO out (MSB first).
  void Clock(uint8_t mosi, bool capture) {
    if (csb_)
      return;
    const size_t bit0 = mosi_.size() * 8;
    mosi_.push_back(mosi);
    if (mosi_.size() == 7 && mosi_[0] == kSpiOpRead && respond) {
      // Token, then the 16 bytes of the line, lowest address first.
      const uint32_t addr        = Addr();
      std::vector<uint8_t> bytes = {kSpiToken};
      for (uint32_t i = 0; i < 16; ++i)
        bytes.push_back(Mem(addr + i));
      resp_start_ = 7 * 8 + latency_bits;
      for (uint8_t b : bytes)
        for (int k = 7; k >= 0; --k)
          resp_bits_.push_back((b >> k) & 1);
    }
    uint8_t miso = 0;
    for (int k = 0; k < 8; ++k) {
      const size_t bit = bit0 + k;
      int v            = idle_miso_one ? 1 : 0;
      if (!resp_bits_.empty() && bit >= resp_start_ && bit < resp_start_ + resp_bits_.size())
        v = resp_bits_[bit - resp_start_];
      miso = static_cast<uint8_t>(miso << 1 | v);
    }
    if (capture && rx_.size() < kSpiRxFifoDepth)
      rx_.push_back(miso);
  }

  int accesses_ = 0;
  std::deque<std::pair<uint32_t, uint32_t>> queue_;
  std::deque<uint8_t> rx_;
  bool csb_ = true;
  std::vector<uint8_t> mosi_;
  std::vector<int> resp_bits_;
  size_t resp_start_ = 0;
};

std::vector<uint8_t> Pattern(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = static_cast<uint8_t>(seed + i * 7 + (i >> 5));
  return v;
}

TEST(FindLine, TokenAtEveryBitOffset) {
  const std::vector<uint8_t> line = Pattern(16, 0x31);
  for (int idle = 0; idle <= 1; ++idle) {
    for (int off = 0; off < 24; ++off) {
      std::vector<int> bits(off, idle);
      for (int k = 7; k >= 0; --k)
        bits.push_back((kSpiToken >> k) & 1);
      for (uint8_t b : line)
        for (int k = 7; k >= 0; --k)
          bits.push_back((b >> k) & 1);
      while (bits.size() % 8)
        bits.push_back(idle);
      std::vector<uint8_t> rx(bits.size() / 8);
      for (size_t i = 0; i < bits.size(); ++i)
        rx[i / 8] |= bits[i] << (7 - i % 8);
      uint8_t out[16] = {};
      ASSERT_EQ(F2Host::FindLine(rx, out), off + 8) << "idle " << idle << " offset " << off;
      EXPECT_EQ(memcmp(out, line.data(), 16), 0) << "idle " << idle << " offset " << off;
    }
  }
}

TEST(FindLine, NoTokenOrTruncated) {
  uint8_t out[16];
  EXPECT_EQ(F2Host::FindLine(std::vector<uint8_t>(200, 0), out), -1);
  EXPECT_EQ(F2Host::FindLine(std::vector<uint8_t>(200, 0xFF), out), -1);
  std::vector<uint8_t> truncated(10, 0);
  truncated[0] = kSpiToken;  // only 72 bits follow
  EXPECT_EQ(F2Host::FindLine(truncated, out), -1);
}

TEST(F2Host, InitChecksTheId) {
  FakeCl cl;
  cl.id = 0xFFFFFFFF;  // what a read returns with no AFI
  F2Host host(&cl);
  EXPECT_FALSE(host.Init());
  EXPECT_NE(host.error().find("0xffffffff"), std::string::npos) << host.error();
}

// Unaligned writes keep the bytes around them, and reads return what was
// written, for several response delays and both MISO idle levels.
TEST(F2Host, WriteThenReadBack) {
  for (int latency : {0, 1, 7, 8, 13, 40}) {
    for (bool idle_one : {false, true}) {
      FakeCl cl;
      cl.latency_bits  = latency;
      cl.idle_miso_one = idle_one;
      F2Host host(&cl);
      ASSERT_TRUE(host.Init()) << host.error();
      std::vector<uint8_t> background = Pattern(256, 0xA0);
      ASSERT_TRUE(host.WriteData(0x1000, background.data(), background.size())) << host.error();
      for (auto [addr, len] : {std::pair<uint32_t, size_t>{0x1000, 16},
                               {0x1003, 5},
                               {0x1013, 30},
                               {0x1040, 64},
                               {0x1071, 1}}) {
        const std::vector<uint8_t> data = Pattern(len, static_cast<uint8_t>(addr));
        ASSERT_TRUE(host.WriteData(addr, data.data(), len)) << host.error();
        std::vector<uint8_t> want = background;
        for (size_t i = 0; i < len; ++i)
          want[addr - 0x1000 + i] = data[i];
        std::vector<uint8_t> got(256);
        ASSERT_TRUE(host.ReadData(0x1000, got.size(), got.data())) << host.error();
        EXPECT_EQ(got, want) << "latency " << latency << " idle " << idle_one << " write 0x"
                             << std::hex << addr << "+" << std::dec << len;
        for (size_t i = 0; i < 256; ++i)
          ASSERT_EQ(cl.Mem(0x1000 + i), want[i]);
        background = want;  // the next case writes on top of this one
      }
      EXPECT_EQ(cl.overflow_writes, 0);
    }
  }
}

TEST(F2Host, NeverWritesAFullCommandQueue) {
  FakeCl cl;
  cl.accesses_per_command = 4;  // SPI controller slower than the host
  F2Host host(&cl);
  ASSERT_TRUE(host.Init()) << host.error();
  const std::vector<uint8_t> image = Pattern(259 * 16, 0x11);  // 259 lines: 1 KB frames
  ASSERT_TRUE(host.WriteData(0x2000, image.data(), image.size())) << host.error();
  EXPECT_EQ(cl.overflow_writes, 0);
  EXPECT_EQ(cl.frames, 5);  // 4 frames of 64 beats and one of 3; whole lines, so no reads
  for (size_t i = 0; i < image.size(); ++i)
    ASSERT_EQ(cl.Mem(0x2000 + i), image[i]) << i;
}

TEST(F2Host, ReadWithoutResponseFails) {
  FakeCl cl;
  cl.respond = false;
  F2Host host(&cl);
  ASSERT_TRUE(host.Init()) << host.error();
  uint32_t v;
  EXPECT_FALSE(host.ReadWord(0x100, &v));
  EXPECT_NE(host.error().find("no response token"), std::string::npos) << host.error();
}

TEST(F2Host, StartCoreWritesEntryThenPulsesTheGate) {
  FakeCl cl;
  F2Host host(&cl);
  ASSERT_TRUE(host.Init()) << host.error();
  ASSERT_TRUE(host.SetEntryPoint(kCsrBaseHighmem, 0x80)) << host.error();
  ASSERT_TRUE(host.StartCore(kCsrBaseHighmem)) << host.error();
  uint32_t entry;
  ASSERT_TRUE(host.ReadWord(kCsrBaseHighmem + 4, &entry));
  EXPECT_EQ(entry, 0x80u);
  // Word 0 of the CSR line, frame by frame: 0 (entry write), 1, then 0.
  std::vector<uint32_t> word0;
  for (auto [addr, w] : cl.word_writes)
    if (addr == kCsrBaseHighmem)
      word0.push_back(w);
  EXPECT_EQ(word0, (std::vector<uint32_t>{0, 1, 0}));
}

TEST(F2Host, ConsoleAndSoftReset) {
  FakeCl cl;
  for (char c : std::string("F2\nPASS\n"))
    cl.console.push_back(static_cast<uint8_t>(c));
  F2Host host(&cl);
  ASSERT_TRUE(host.Init()) << host.error();
  EXPECT_EQ(host.ReadConsole(), "F2\nPASS\n");
  EXPECT_EQ(host.ReadConsole(), "");
  host.SoftReset();
  EXPECT_EQ(cl.ctrl_writes, (std::vector<uint32_t>{kCtrlHoldReset, 0}));
}

}  // namespace
}  // namespace f2
