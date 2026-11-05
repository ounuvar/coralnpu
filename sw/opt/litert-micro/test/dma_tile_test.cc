// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Host tests for the copy interface (#11), memcpy backend: the beat width and descriptor flag
// helpers, and the chain's bookkeeping.

#include "sw/opt/litert-micro/dma_tile.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace coralnpu_v2::opt::litert_micro {
namespace {

// The loop the trailing-zero count replaced.
uint32_t BeatLog2Reference(uintptr_t src, uintptr_t dst, uint32_t bytes, uint32_t max_log2) {
  uint32_t log2 = max_log2;
  while (log2 > 0) {
    const uintptr_t mask = (static_cast<uintptr_t>(1) << log2) - 1;
    if (((src | dst | bytes) & mask) == 0)
      break;
    --log2;
  }
  return log2;
}

TEST(BeatLog2, MatchesTheReferenceLoopOnRandomAndEdgeValues) {
  std::mt19937 rng(7);
  for (uint32_t max_log2 = 0; max_log2 <= 4; ++max_log2) {
    for (int i = 0; i < 20000; ++i) {
      // Mix of fully random values and values with chosen low zero bits.
      const uint32_t shift_a = rng() % 6, shift_b = rng() % 6, shift_c = rng() % 6;
      const uintptr_t src  = static_cast<uintptr_t>(rng()) << shift_a;
      const uintptr_t dst  = static_cast<uintptr_t>(rng()) << shift_b;
      const uint32_t bytes = (rng() % 0xFFFFFF + 1) << shift_c;
      ASSERT_EQ(BeatLog2(src, dst, bytes, max_log2), BeatLog2Reference(src, dst, bytes, max_log2))
          << src << " " << dst << " " << bytes << " " << max_log2;
    }
  }
  // Edges: all-zero low bits, a single misaligned operand, and alignment above the maximum.
  EXPECT_EQ(BeatLog2(0, 0, 0, 4), 4u);
  EXPECT_EQ(BeatLog2(64, 128, 256, 4), 4u);
  EXPECT_EQ(BeatLog2(64, 128, 256, 2), 2u);
  EXPECT_EQ(BeatLog2(65, 128, 256, 4), 0u);
  EXPECT_EQ(BeatLog2(64, 130, 256, 4), 1u);
  EXPECT_EQ(BeatLog2(64, 128, 260, 4), 2u);
  EXPECT_EQ(BeatLog2(16, 32, 48, 4), 4u);
}

TEST(CopyLenFlags, PutsTheLengthInTheLowBitsAndTheBeatSizeAboveIt) {
  EXPECT_EQ(CopyLenFlags(0, 0), 0u);
  EXPECT_EQ(CopyLenFlags(256, 4), 256u | (4u << 24));
  EXPECT_EQ(CopyLenFlags(0xFFFFFF, 4), 0xFFFFFFu | (4u << 24));
  // The same encoding as dma_make_len_flags(len, width, 0, 0, 0) in fpga/sw/dma.c.
  for (uint32_t bytes : {1u, 16u, 4096u, 65536u, 0xFFFFFFu}) {
    for (uint32_t log2 = 0; log2 <= 4; ++log2) {
      EXPECT_EQ(CopyLenFlags(bytes, log2), (bytes & 0xFFFFFF) | ((log2 & 0x7) << 24));
    }
  }
}

// Chain storage the way the wrapper provides it: 32-byte aligned, StorageBytes(capacity) bytes.
struct Storage {
  explicit Storage(int capacity) : bytes(CopyChain::StorageBytes(capacity) + 32) {}
  void *get() {
    return reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(bytes.data()) + 31) &
                                    ~uintptr_t{31});
  }
  std::vector<uint8_t> bytes;
};

TEST(CopyChain, CopiesInOrderAndRejectsEmptyOrOverfullChains) {
  std::vector<uint8_t> src(256), dst(256, 0);
  for (int i = 0; i < 256; ++i)
    src[i] = static_cast<uint8_t>(i);
  Storage storage(3);
  CopyChain chain;
  chain.Init(storage.get(), 3);
  EXPECT_EQ(chain.size(), 0);
  EXPECT_EQ(chain.capacity(), 3);
  EXPECT_FALSE(chain.Add(src.data(), dst.data(), 0));  // empty copy
  EXPECT_TRUE(chain.Add(src.data(), dst.data(), 64));
  EXPECT_TRUE(chain.Add(src.data() + 64, dst.data() + 128, 64));
  EXPECT_TRUE(chain.Add(src.data() + 128, dst.data() + 64, 64));
  EXPECT_FALSE(chain.Add(src.data(), dst.data(), 16));  // full
  EXPECT_EQ(chain.size(), 3);
  EXPECT_TRUE(chain.Run());
  EXPECT_EQ(std::memcmp(dst.data(), src.data(), 64), 0);
  EXPECT_EQ(std::memcmp(dst.data() + 128, src.data() + 64, 64), 0);
  EXPECT_EQ(std::memcmp(dst.data() + 64, src.data() + 128, 64), 0);
  EXPECT_EQ(dst[192], 0);  // untouched
  chain.Clear();
  EXPECT_EQ(chain.size(), 0);
  EXPECT_TRUE(chain.Run());  // an empty chain is a no-op
}

TEST(CopyChain, StorageIsSizedPerChainAndTwoChainsDoNotOverlap) {
  EXPECT_EQ(CopyChain::StorageBytes(1), 32u);
  EXPECT_EQ(CopyChain::StorageBytes(37), 37u * 32);
  // Two chains side by side in one arena, as the layout places them.
  Storage arena(5 + 4);
  uint8_t *base = static_cast<uint8_t *>(arena.get());
  CopyChain a, b;
  a.Init(base, 5);
  b.Init(base + CopyChain::StorageBytes(5), 4);
  std::vector<uint8_t> src(64, 0xAB), dst_a(64, 0), dst_b(64, 0);
  for (int i = 0; i < 5; ++i)
    EXPECT_TRUE(a.Add(src.data() + i, dst_a.data() + i, 8));
  for (int i = 0; i < 4; ++i)
    EXPECT_TRUE(b.Add(src.data() + i, dst_b.data() + i, 8));
  EXPECT_FALSE(a.Add(src.data(), dst_a.data(), 8));
  EXPECT_FALSE(b.Add(src.data(), dst_b.data(), 8));
  EXPECT_TRUE(a.Run());
  EXPECT_TRUE(b.Run());
  EXPECT_EQ(dst_a[11], 0xAB);
  EXPECT_EQ(dst_b[10], 0xAB);
}

#ifdef CORALNPU_COPY_AT_WAIT
// The other extreme of when a DMA moves data: nothing has moved after Start, everything has after
// Wait. Code that is correct for both orders is correct on the DMA.
TEST(CopyChain, CopyAtWaitMovesDataOnlyAtWait) {
  std::vector<uint8_t> src(64, 0x5C), dst(64, 0);
  Storage storage(2);
  CopyChain chain;
  chain.Init(storage.get(), 2);
  EXPECT_TRUE(chain.Add(src.data(), dst.data(), 32));
  EXPECT_TRUE(chain.Add(src.data() + 32, dst.data() + 32, 32));
  chain.Start();
  EXPECT_EQ(dst[0], 0);  // nothing has moved yet
  EXPECT_EQ(dst[63], 0);
  src[5] = 0x11;  // a source changed between Start and Wait is what the copy sees
  EXPECT_TRUE(chain.Wait());
  EXPECT_EQ(dst[0], 0x5C);
  EXPECT_EQ(dst[5], 0x11);
  EXPECT_EQ(dst[63], 0x5C);
}
#else
TEST(CopyChain, CopyAtStartMovesDataAtStart) {
  std::vector<uint8_t> src(64, 0x5C), dst(64, 0);
  Storage storage(2);
  CopyChain chain;
  chain.Init(storage.get(), 2);
  EXPECT_TRUE(chain.Add(src.data(), dst.data(), 64));
  chain.Start();
  EXPECT_EQ(dst[0], 0x5C);  // already moved
  src[5] = 0x11;            // too late to be seen
  EXPECT_TRUE(chain.Wait());
  EXPECT_EQ(dst[5], 0x5C);
}
#endif

}  // namespace
}  // namespace coralnpu_v2::opt::litert_micro
