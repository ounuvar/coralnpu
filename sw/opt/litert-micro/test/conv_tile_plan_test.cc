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

#include <algorithm>
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

constexpr uint32_t kBudget = CORALNPU_TILE_BUDGET_BYTES;

// Everything a plan promises, for one layer: it fits, its chunks cover every channel, and its tiles
// cover every pixel exactly.
void ExpectConsistent(const Layer &l, bool int16_weights, uint32_t budget, bool prefetch,
                      bool merge) {
  const ConvTilePlan p =
      PlanPointwise(l.pixels, l.in_d, l.out_d, int16_weights, budget, prefetch, merge);
  ASSERT_TRUE(p.tiled) << "layer " << l.index;
  const bool chunked = p.channel_chunk < l.out_d;
  const uint32_t expected =
      PointwiseLayoutFor(l.in_d, p.channel_chunk, p.buffer_pixels, p.double_buffer ? 2 : 1,
                         int16_weights, prefetch, chunked)
          .total;
  EXPECT_EQ(p.footprint, expected) << "layer " << l.index;
  EXPECT_LE(p.footprint, budget) << "layer " << l.index;
  // Channel chunks cover every output channel, with no empty chunk.
  EXPECT_GE(p.num_chunks * p.channel_chunk, l.out_d);
  EXPECT_LT((p.num_chunks - 1) * p.channel_chunk, l.out_d);
  EXPECT_LE(p.channel_chunk, l.out_d);
  // Tiles cover every pixel exactly: all but the last are pixels_per_tile.
  ASSERT_GE(p.tiles, 1);
  EXPECT_EQ((p.tiles - 1) * p.pixels_per_tile + p.last_tile_pixels, l.pixels)
      << "layer " << l.index;
  EXPECT_GE(p.last_tile_pixels, 1);
  EXPECT_GE(p.pixels_per_tile, 1);
  // The buffers hold the largest tile and never more than the planner may use.
  EXPECT_EQ(p.buffer_pixels, std::max(p.pixels_per_tile, p.last_tile_pixels));
  EXPECT_LE(p.buffer_pixels, kMaxTilePixels);
  // Without merging the last tile is never longer than the others.
  if (!merge)
    EXPECT_LE(p.last_tile_pixels, p.pixels_per_tile) << "layer " << l.index;
}

TEST(ConvTilePlan, EveryPlanIsConsistentForEveryLayerKernelAndSwitch) {
  for (const Layer &l : kLayers)
    for (bool int16_weights : {false, true})
      for (bool prefetch : {false, true})
        for (bool merge : {false, true})
          ExpectConsistent(l, int16_weights, kBudget, prefetch, merge);
}

TEST(ConvTilePlan, PrintPlans) {
  std::printf(
      "layer pixels in->out | rvv, prefetch, merge: chunk x n, tile x tiles + last, buffers, "
      "bytes | no merge\n");
  for (const Layer &l : kLayers) {
    const ConvTilePlan m = PlanPointwise(l.pixels, l.in_d, l.out_d, true, kBudget, true, true);
    const ConvTilePlan n = PlanPointwise(l.pixels, l.in_d, l.out_d, true, kBudget, true, false);
    std::printf("L%-2d %5d %3d->%3d | %3d x%2d, %2d x %3d + %2d, %d buf, %5u B | %2d x %3d + %2d\n",
                l.index, l.pixels, l.in_d, l.out_d, m.channel_chunk, m.num_chunks,
                m.pixels_per_tile, m.tiles - 1, m.last_tile_pixels, m.double_buffer ? 2 : 1,
                m.footprint, n.pixels_per_tile, n.tiles - 1, n.last_tile_pixels);
  }
}

TEST(ConvTilePlan, SmallLayersAreNotChunkedAndLargeOnesAre) {
  for (const Layer &l : kLayers) {
    const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, true, kBudget, false, false);
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
    const ConvTilePlan r = PlanPointwise(l.pixels, l.in_d, l.out_d, true, kBudget, true, false);
    const ConvTilePlan s = PlanPointwise(l.pixels, l.in_d, l.out_d, false, kBudget, true, false);
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
  EXPECT_EQ(p.tiles, 1);
  EXPECT_EQ(p.pixels_per_tile, 3);
  EXPECT_EQ(p.last_tile_pixels, 3);
}

TEST(ConvTilePlan, LayoutPutsTheChainsFirstAndIsAlignedOrderedAndSizedByTheBuffersInUse) {
  for (int buffers : {1, 2}) {
    for (bool int16_weights : {false, true}) {
      for (bool prefetch : {false, true}) {
        for (bool chunked : {false, true}) {
          const PointwiseLayout l =
              PointwiseLayoutFor(128, 48, 20, buffers, int16_weights, prefetch, chunked);
          // The chains come first and are 32-byte aligned (they hold DMA descriptors).
          EXPECT_EQ(l.prologue_chain, 0u);
          EXPECT_EQ(l.chain[0], kDescriptorBytes * kPrologueCopies);
          EXPECT_EQ(l.chain[1], l.chain[0] + kDescriptorBytes * l.chain_capacity);
          EXPECT_EQ(l.chain[0] % 32, 0u);
          EXPECT_EQ(l.chain[1] % 32, 0u);
          EXPECT_EQ(l.chain_capacity, ChainCopies(20, chunked, prefetch));
          const uint32_t offsets[] = {
              l.filter,     l.weights16,   l.bias_raw, l.bias_folded, l.multiplier, l.shift_raw,
              l.shift_left, l.shift_right, l.accs,     l.filter2,     l.bias_raw2,  l.multiplier2,
              l.shift_raw2, l.in[0],       l.out[0],   l.in[1],       l.out[1]};
          uint32_t previous = l.chain[1] + kDescriptorBytes * l.chain_capacity;
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
          // The int16 weights exist only for the RVV kernel.
          EXPECT_EQ(l.bias_raw - l.weights16, int16_weights ? 2u * 128 * 48 : 0u);
        }
      }
    }
  }
}

TEST(ConvTilePlan, ChainCapacityCoversTheLongestChain) {
  // Chunked: one output copy per pixel of the largest tile, plus the next input, plus with prefetch
  // the bias, multiplier and shift of the next chunk, a piece of its filter and its first input.
  EXPECT_EQ(ChainCopies(32, true, false), 33);
  EXPECT_EQ(ChainCopies(32, true, true), 37);
  // Not chunked: the whole tile's output is one copy.
  EXPECT_EQ(ChainCopies(32, false, false), 2);
  EXPECT_EQ(ChainCopies(32, false, true), 6);
}

TEST(ConvTilePlan, PrefetchAddsASecondSetOfRawBuffers) {
  const PointwiseLayout off = PointwiseLayoutFor(128, 48, 20, 2, true, false, true);
  const PointwiseLayout on  = PointwiseLayoutFor(128, 48, 20, 2, true, true, true);
  EXPECT_EQ(off.filter2, off.bias_raw2);  // nothing reserved without prefetch
  EXPECT_EQ(on.bias_raw2 - on.filter2, RoundUp16(48u * 128));
  EXPECT_EQ(on.multiplier2 - on.bias_raw2, RoundUp16(4u * 48));
  EXPECT_EQ(on.shift_raw2 - on.multiplier2, RoundUp16(4u * 48));
  // The raw buffers and the longer chains it needs.
  const uint32_t chains =
      kDescriptorBytes * 2 * (ChainCopies(20, true, true) - ChainCopies(20, true, false));
  EXPECT_EQ(on.total - off.total, RoundUp16(48u * 128) + 3 * RoundUp16(4u * 48) + chains);
}

TEST(ConvTilePlan, DmaBytesCountFilterOncePerLayerAndInputOncePerChunk) {
  EXPECT_EQ(PointwiseDmaBytes(100, 64, 128, 1), 128u * 64 + 12 * 128 + 100u * 64 + 100u * 128);
  EXPECT_EQ(PointwiseDmaBytes(100, 64, 128, 4), 128u * 64 + 12 * 128 + 4 * 100u * 64 + 100u * 128);
  for (const Layer &l : kLayers) {
    const ConvTilePlan p = PlanPointwise(l.pixels, l.in_d, l.out_d, true);
    EXPECT_EQ(p.dma_bytes, PointwiseDmaBytes(l.pixels, l.in_d, l.out_d, p.num_chunks));
  }
}

// ---- tile splitting --------------------------------------------------------------------------

TEST(SplitPixels, WithoutMergeTilesAreAsLargeAsTheBuffersAllow) {
  const TileSplit s = SplitPixels(49, 12, 32, 128, false);
  EXPECT_EQ(s.tile, 12);
  EXPECT_EQ(s.tiles, 5);
  EXPECT_EQ(s.last, 1);
}

TEST(SplitPixels, ALayerThatFitsIsOneTile) {
  for (bool merge : {false, true}) {
    const TileSplit s = SplitPixels(10, 12, 32, 128, merge);
    EXPECT_EQ(s.tiles, 1);
    EXPECT_EQ(s.tile, 10);
    EXPECT_EQ(s.last, 10);
  }
}

TEST(SplitPixels, MergeFoldsALeftoverPixelIntoTheLastTile) {
  // 49 pixels, buffers for 16: 12, 12, 12, 13 beats 16, 16, 16, 1 (a one-pixel tile cannot hide the
  // next tile's copy) and 8 x 6 + 1.
  const TileSplit s = SplitPixels(49, 16, 32, 128, true);
  EXPECT_EQ(s.tile, 12);
  EXPECT_EQ(s.tiles, 4);
  EXPECT_EQ(s.last, 13);
  EXPECT_LE(s.last, 16);
}

TEST(SplitPixels, MergeKeepsTilesAtMultiplesOfTheKernelBlock) {
  for (int pixels : {33, 49, 99, 196, 784, 3136}) {
    for (int max_tile : {8, 11, 14, 16, 30, 32}) {
      const TileSplit s = SplitPixels(pixels, max_tile, 32, 128, true);
      EXPECT_EQ((s.tiles - 1) * s.tile + s.last, pixels) << pixels << " " << max_tile;
      EXPECT_LE(std::max(s.tile, s.last), max_tile) << pixels << " " << max_tile;
      if (s.tile >= kMinTilePixels && pixels > max_tile)
        EXPECT_EQ(s.tile % 4, 0) << pixels << " " << max_tile;
    }
  }
}

TEST(SplitPixels, MergeIsNeverCostlierThanTheDefaultByTheModel) {
  for (int pixels : {33, 49, 99, 196, 784}) {
    for (int max_tile : {8, 11, 16, 32}) {
      const TileSplit a = SplitPixels(pixels, max_tile, 32, 128, false);
      const TileSplit b = SplitPixels(pixels, max_tile, 32, 128, true);
      EXPECT_LE(SplitCost(b, 32, 128), SplitCost(a, 32, 128)) << pixels << " " << max_tile;
    }
  }
}

TEST(SplitPixels, ATinyLastTileCostsMoreThanAMergedOne) {
  // 49 pixels as 12 x 4 + 1 against 12 x 3 + 13 (chunk 32, in_d 128).
  EXPECT_GT(SplitCost({12, 5, 1}, 32, 128), SplitCost({12, 4, 13}, 32, 128));
}

TEST(SplitPixels, TheCostModelPricesBlocksAndLeftoverPixels) {
  // Four pixels are one block; five are a block and a pixel on the slower path.
  EXPECT_EQ(TileComputeCycles(4, 32, 128), kBlockCycles);
  EXPECT_EQ(TileComputeCycles(5, 32, 128), kBlockCycles + kSlowCycles);
  EXPECT_EQ(TileComputeCycles(3, 32, 128), 3 * kSlowCycles);
  // A leftover pixel costs more than a pixel in a block.
  EXPECT_GT(kSlowCycles * 4, kBlockCycles);
  // The model scales with chunk x in_d.
  EXPECT_EQ(TileComputeCycles(4, 64, 128), 2 * kBlockCycles);
  EXPECT_EQ(TileInputCopyCycles(8, 256), 8 * 256 / 16 * kBeatCycles);
}

TEST(ConvTilePlan, MergePlansForTheMeasuredLayersAreTheIntendedOnes) {
  // Layers 28 and 30 and the two layer-17 runs of #11, RVV kernel, prefetch, default budget.
  struct Expect {
    int pixels, in_d, out_d, tile, tiles, last;
  };
  const Expect expected[] = {
      {49, 128, 256, 12, 4, 13},  // layer 28: 12, 12, 12, 13
      {49, 256, 256, 8, 6, 9},    // layer 30: 8 x 5 + 9
      {99, 128, 128, 12, 8, 15},  // layer 17, 99 pixels: 12 x 7 + 15
      {33, 128, 128, 12, 3, 9},   // layer 17, 33 pixels: 12, 12, 9 (no tiny tile)
  };
  for (const Expect &e : expected) {
    const ConvTilePlan p = PlanPointwise(e.pixels, e.in_d, e.out_d, true, kBudget, true, true);
    ASSERT_TRUE(p.tiled);
    EXPECT_EQ(p.pixels_per_tile, e.tile) << e.pixels << " " << e.in_d << " " << e.out_d;
    EXPECT_EQ(p.tiles, e.tiles) << e.pixels << " " << e.in_d << " " << e.out_d;
    EXPECT_EQ(p.last_tile_pixels, e.last) << e.pixels << " " << e.in_d << " " << e.out_d;
  }
}

}  // namespace
}  // namespace coralnpu_v2::opt::litert_micro
