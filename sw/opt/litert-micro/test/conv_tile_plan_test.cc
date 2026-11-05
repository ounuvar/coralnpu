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

// Host tests for the pointwise tile planner (#11).

#include "sw/opt/litert-micro/conv_tile_plan.h"

#include <cstdint>
#include <cstdio>
#include <vector>

#include "gtest/gtest.h"

namespace coralnpu_v2::opt::litert_micro {
namespace {

struct Layer {
  int index, pixels, in_d, out_d;
};

// MobileNet v1 0.25 at 224x224: the 1x1 conv layers (pixels = output height x width).
const std::vector<Layer> kLayers = {
    {2, 112 * 112, 8, 16}, {5, 56 * 56, 16, 32},   {7, 56 * 56, 32, 32},    {10, 28 * 28, 32, 64},
    {12, 28 * 28, 64, 64}, {15, 14 * 14, 64, 128}, {17, 14 * 14, 128, 128}, {28, 7 * 7, 128, 256},
    {30, 7 * 7, 256, 256}, {32, 1 * 1, 256, 5},
};

void ExpectConsistent(const Layer &l, bool int16_weights, uint32_t budget) {
  const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, int16_weights, budget);
  ASSERT_TRUE(p.tiled) << "layer " << l.index;
  EXPECT_LE(p.footprint, budget) << "layer " << l.index;
  EXPECT_EQ(p.footprint, PointwiseLayoutFor(l.in_d, p.channel_chunk, p.pixels_per_tile,
                                            p.double_buffer ? 2 : 1, int16_weights)
                                 .total +
                             kDescriptorBytes);
  // The chunks cover every output channel, and there are no empty chunks.
  EXPECT_GE(p.num_chunks * p.channel_chunk, l.out_d);
  EXPECT_LT((p.num_chunks - 1) * p.channel_chunk, l.out_d);
  EXPECT_LE(p.channel_chunk, l.out_d);
  // A tile is at least kMinTilePixels pixels unless the whole layer is smaller.
  EXPECT_GE(p.pixels_per_tile, l.pixels < kMinTilePixels ? l.pixels : kMinTilePixels);
  EXPECT_LE(p.pixels_per_tile, l.pixels);
  EXPECT_LE(p.pixels_per_tile, kMaxTilePixels);
}

TEST(ConvTilePlan, MobileNetLayersFitTheDefaultBudget) {
  for (const Layer &l : kLayers) {
    for (bool int16_weights : {false, true}) {
      ExpectConsistent(l, int16_weights, CORALNPU_TILE_BUDGET_BYTES);
    }
  }
}

TEST(ConvTilePlan, PrintPlans) {
  // One DMA beat (16 bytes) takes about 82 cycles with the HBM latency model (measured in #11).
  std::printf(
      "layer pixels in->out | rvv: chunk x n, tile px, buffers, bytes, DMA bytes, est. DMA kcycles "
      "| scalar: chunk x n, tile px\n");
  for (const Layer &l : kLayers) {
    const ConvTilePlan r = PlanPointwise(l.pixels, l.in_d, l.out_d, true);
    const ConvTilePlan s = PlanPointwise(l.pixels, l.in_d, l.out_d, false);
    std::printf(
        "L%-2d %5d %3d->%3d | %3d x%2d, %3d px, %d buf, %5u B, %7llu B, %5.0f k | %3d x%2d, %3d "
        "px\n",
        l.index, l.pixels, l.in_d, l.out_d, r.channel_chunk, r.num_chunks, r.pixels_per_tile,
        r.double_buffer ? 2 : 1, r.footprint, static_cast<unsigned long long>(r.dma_bytes),
        r.dma_bytes / 16.0 * 82 / 1000, s.channel_chunk, s.num_chunks, s.pixels_per_tile);
  }
}

TEST(ConvTilePlan, DmaBytesCountFilterOncePerLayerAndInputOncePerChunk) {
  EXPECT_EQ(PointwiseDmaBytes(100, 64, 128, 1), 128u * 64 + 12 * 128 + 100u * 64 + 100u * 128);
  EXPECT_EQ(PointwiseDmaBytes(100, 64, 128, 4), 128u * 64 + 12 * 128 + 4 * 100u * 64 + 100u * 128);
  for (const Layer &l : kLayers) {
    const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, true);
    EXPECT_EQ(p.dma_bytes, PointwiseDmaBytes(l.pixels, l.in_d, l.out_d, p.num_chunks));
  }
}

TEST(ConvTilePlan, SmallLayersAreNotChunkedAndLargeOnesAre) {
  for (const Layer &l : kLayers) {
    const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, true);
    if (l.index <= 12 || l.index == 32) {
      EXPECT_EQ(p.num_chunks, 1) << "layer " << l.index;
    } else {
      // Layers 15, 17, 28 and 30 do not fit unchunked with the kernel's int16 weights.
      EXPECT_GT(p.num_chunks, 1) << "layer " << l.index;
    }
  }
}

TEST(ConvTilePlan, ScalarKernelNeedsNoMoreChunksThanRvv) {
  for (const Layer &l : kLayers) {
    const ConvTilePlan r = PlanPointwise(l.pixels, l.in_d, l.out_d, true);
    const ConvTilePlan s = PlanPointwise(l.pixels, l.in_d, l.out_d, false);
    EXPECT_LE(s.num_chunks, r.num_chunks) << "layer " << l.index;
  }
}

TEST(ConvTilePlan, LargeBudgetKeepsEveryLayerWholeAndDoubleBuffered) {
  for (const Layer &l : kLayers) {
    const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, true, 1024 * 1024);
    EXPECT_EQ(p.num_chunks, 1) << "layer " << l.index;
    EXPECT_TRUE(p.double_buffer) << "layer " << l.index;
  }
}

TEST(ConvTilePlan, BiggerBudgetNeverMakesChunksSmaller) {
  for (const Layer &l : kLayers) {
    int previous_chunk = 0;
    for (uint32_t budget = 4 * 1024; budget <= 256 * 1024; budget += 2 * 1024) {
      const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, true, budget);
      if (!p.tiled) {
        EXPECT_EQ(previous_chunk, 0) << "layer " << l.index << " budget " << budget;
        continue;
      }
      EXPECT_LE(p.footprint, budget);
      EXPECT_GE(p.channel_chunk, previous_chunk) << "layer " << l.index << " budget " << budget;
      previous_chunk = p.channel_chunk;
    }
  }
}

TEST(ConvTilePlan, DescriptorsAreInsideTheBudget) {
  EXPECT_EQ(kDescriptorBytes, 32u * (5 + 2 * 34 + 3));
  EXPECT_EQ(ArenaBudget(CORALNPU_TILE_BUDGET_BYTES), CORALNPU_TILE_BUDGET_BYTES - kDescriptorBytes);
  EXPECT_EQ(ArenaBudget(100), 0u);
}

TEST(ConvTilePlan, ImpossibleBudgetIsNotTiled) {
  for (const Layer &l : kLayers) {
    EXPECT_FALSE(PlanPointwise(l.pixels, l.in_d, l.out_d, true, 256).tiled) << l.index;
  }
  EXPECT_FALSE(PlanPointwise(0, 8, 8, true).tiled);
  EXPECT_FALSE(PlanPointwise(8, 0, 8, true).tiled);
  EXPECT_FALSE(PlanPointwise(8, 8, 0, true).tiled);
}

TEST(ConvTilePlan, LayerSmallerThanAMinimumTileIsOneTile) {
  const ConvTilePlan p = PlanPointwise(3, 16, 16, true);
  ASSERT_TRUE(p.tiled);
  EXPECT_EQ(p.pixels_per_tile, 3);
}

TEST(ConvTilePlan, LayoutIsAlignedOrderedAndSizedByTheBuffersInUse) {
  for (int buffers : {1, 2}) {
    for (bool int16_weights : {false, true}) {
      const PointwiseLayout l  = PointwiseLayoutFor(128, 48, 20, buffers, int16_weights);
      const uint32_t offsets[] = {
          l.filter,     l.weights16,   l.bias_raw, l.bias_folded, l.multiplier, l.shift_raw,
          l.shift_left, l.shift_right, l.accs,     l.filter2,     l.bias_raw2,  l.multiplier2,
          l.shift_raw2, l.in[0],       l.out[0],   l.in[1],       l.out[1]};
      uint32_t previous = 0;
      for (uint32_t o : offsets) {
        EXPECT_EQ(o % 16, 0u);
        EXPECT_GE(o, previous);
        previous = o;
      }
      EXPECT_LE(previous, l.total);
      EXPECT_EQ(l.total % 16, 0u);
      // The second input and output tiles take no space for single buffering.
      if (buffers == 1)
        EXPECT_EQ(l.out[1], l.total);
      // The int16 weights and accumulators exist only for the RVV kernel.
      EXPECT_EQ(l.bias_raw - l.weights16, int16_weights ? 2u * 128 * 48 : 0u);
    }
  }
}

TEST(ConvTilePlan, PrefetchAddsASecondSetOfRawBuffers) {
  const PointwiseLayout off = PointwiseLayoutFor(128, 48, 20, 2, true, false);
  const PointwiseLayout on  = PointwiseLayoutFor(128, 48, 20, 2, true, true);
  EXPECT_EQ(off.filter2, off.bias_raw2);  // nothing reserved without prefetch
  // The second set mirrors the first: filter rows, bias, multipliers and shifts of one chunk.
  EXPECT_EQ(on.bias_raw2 - on.filter2, RoundUp16(48u * 128));
  EXPECT_EQ(on.multiplier2 - on.bias_raw2, RoundUp16(4u * 48));
  EXPECT_EQ(on.shift_raw2 - on.multiplier2, RoundUp16(4u * 48));
  EXPECT_EQ(on.total - off.total, RoundUp16(48u * 128) + 3 * RoundUp16(4u * 48));
}

TEST(ConvTilePlan, MobileNetLayersFitWithPrefetchAndNeverUseMoreThanTheBudget) {
  for (const Layer &l : kLayers) {
    const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, true,
                                         CORALNPU_TILE_BUDGET_BYTES, /*prefetch=*/true);
    ASSERT_TRUE(p.tiled) << "layer " << l.index;
    EXPECT_LE(p.footprint, CORALNPU_TILE_BUDGET_BYTES) << "layer " << l.index;
    EXPECT_EQ(p.footprint, PointwiseLayoutFor(l.in_d, p.channel_chunk, p.pixels_per_tile,
                                              p.double_buffer ? 2 : 1, true, true)
                                   .total +
                               kDescriptorBytes);
    // Prefetch costs memory, so it never allows a bigger chunk than without it.
    const ConvTilePlan q = PlanPointwise(l.pixels, l.in_d, l.out_d, true,
                                         CORALNPU_TILE_BUDGET_BYTES, /*prefetch=*/false);
    EXPECT_LE(p.channel_chunk, q.channel_chunk) << "layer " << l.index;
  }
}

TEST(ConvTilePlan, Round4MakesTilesAMultipleOfTheKernelBlockWithoutChangingTheChunking) {
  for (const Layer &l : kLayers) {
    for (bool prefetch : {false, true}) {
      const ConvTilePlan a = PlanPointwise(l.pixels, l.in_d, l.out_d, true,
                                           CORALNPU_TILE_BUDGET_BYTES, prefetch, false);
      const ConvTilePlan b = PlanPointwise(l.pixels, l.in_d, l.out_d, true,
                                           CORALNPU_TILE_BUDGET_BYTES, prefetch, true);
      ASSERT_TRUE(a.tiled && b.tiled) << "layer " << l.index;
      EXPECT_EQ(a.channel_chunk, b.channel_chunk) << "layer " << l.index;
      EXPECT_EQ(a.double_buffer, b.double_buffer) << "layer " << l.index;
      EXPECT_LE(b.pixels_per_tile, a.pixels_per_tile);
      EXPECT_LT(a.pixels_per_tile - b.pixels_per_tile, 4) << "layer " << l.index;
      if (a.pixels_per_tile >= 8) {
        EXPECT_EQ(b.pixels_per_tile % 4, 0) << "layer " << l.index;
      } else {
        EXPECT_EQ(a.pixels_per_tile, b.pixels_per_tile) << "layer " << l.index;
      }
      EXPECT_LE(b.footprint, a.footprint);
    }
  }
}

TEST(ConvTilePlan, ChainCapacityCoversTheLongestChain) {
  // Outputs (one per pixel of a full tile) + next input + the prefetch schedule's extra copies.
  EXPECT_GE(kMaxChainCopies, static_cast<uint32_t>(kMaxTilePixels) + 2 +
                                 (kPrefetchDefault ? kPrefetchExtraCopies : 0));
  // The last tile's chain without the spread: outputs, the next chunk's filter, bias, multiplier
  // and shift copies, and its first input tile.
  if (kPrefetchDefault) {
    EXPECT_GE(kMaxChainCopies, static_cast<uint32_t>(kMaxTilePixels) + 4 + 1);
  }
  // Every plan the planner returns keeps its tile within the chain capacity.
  for (const Layer &l : kLayers) {
    const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, true);
    ASSERT_TRUE(p.tiled);
    EXPECT_LE(static_cast<uint32_t>(p.pixels_per_tile) + 5, kMaxChainCopies + 3);
  }
}

}  // namespace
}  // namespace coralnpu_v2::opt::litert_micro
