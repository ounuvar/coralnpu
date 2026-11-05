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

// Host tests for the row planner of DMA-tiled spatial convolution (#11).

#include "sw/opt/litert-micro/conv_row_plan.h"

#include "gtest/gtest.h"

namespace coralnpu_v2::opt::litert_micro {
namespace {

// MobileNet v1 layer 0: 224x224x3 to 112x112x8, 3x3 stride 2, SAME (no padding above).
constexpr RowGeom kLayer0 = {224, 224, 3, 112, 112, 8, 3, 3, 2, 0, 216, 0};

TEST(RowTileFor, FirstTileOfLayer0HasNoPaddingAbove) {
  const RowTile t = RowTileFor(kLayer0, 0, 4);
  EXPECT_EQ(t.in_start, 0);
  EXPECT_EQ(t.in_rows, 9);  // (4 - 1) * 2 + 3
  EXPECT_EQ(t.pad_top, 0);
}

TEST(RowTileFor, LastTileIsClippedAtTheBottom) {
  const RowTile t = RowTileFor(kLayer0, 108, 4);  // rows 216 to 224 (224 is below the image)
  EXPECT_EQ(t.in_start, 216);
  EXPECT_EQ(t.in_rows, 8);
  EXPECT_EQ(t.pad_top, 0);
}

TEST(RowTileFor, PaddingAboveIsTheKernelsPadForTheFirstTileOnly) {
  RowGeom g           = kLayer0;
  g.pad_h             = 1;
  const RowTile first = RowTileFor(g, 0, 4);
  EXPECT_EQ(first.in_start, 0);
  EXPECT_EQ(first.in_rows, 8);  // rows -1 to 7, clipped to 0 to 7
  EXPECT_EQ(first.pad_top, 1);
  const RowTile second = RowTileFor(g, 4, 4);
  EXPECT_EQ(second.in_start, 7);
  EXPECT_EQ(second.pad_top, 0);
}

TEST(RowTileFor, EveryOutputRowSeesTheSameInputRowsAsUntiled) {
  // For each output row, the input row of tap ky is in_start + (o - o0) * stride - pad_top + ky in
  // the tile and o * stride - pad_h + ky in the whole image.
  for (int pad : {0, 1, 2}) {
    for (int stride : {1, 2, 3}) {
      RowGeom g = {30, 8, 3, 0, 8, 8, 3, 3, stride, pad, 216, 0};
      g.out_h   = (g.in_h + 2 * pad - g.fh) / stride + 1;
      for (int rows : {1, 2, 5}) {
        for (int o0 = 0; o0 < g.out_h; o0 += rows) {
          const int nr    = std::min(rows, g.out_h - o0);
          const RowTile t = RowTileFor(g, o0, nr);
          for (int o = o0; o < o0 + nr; ++o) {
            for (int ky = 0; ky < g.fh; ++ky) {
              const int whole = o * stride - pad + ky;
              const int tile  = t.in_start + (o - o0) * stride - t.pad_top + ky;
              // In the tile, taps outside [0, in_rows) are padding or beyond the image.
              const bool whole_in = whole >= 0 && whole < g.in_h;
              const bool tile_in  = tile >= t.in_start && tile < t.in_start + t.in_rows;
              EXPECT_EQ(whole_in, tile_in);
              if (whole_in)
                EXPECT_EQ(whole, tile);
            }
          }
        }
      }
    }
  }
}

TEST(PlanRows, Layer0FitsDoubleBufferedAtDefaultBudget) {
  const RowPlan p = PlanRows(kLayer0, CORALNPU_TILE_BUDGET_BYTES);
  ASSERT_TRUE(p.tiled);
  EXPECT_TRUE(p.double_buffer);
  EXPECT_GE(p.rows_per_tile, 2);
  EXPECT_LE(p.footprint, CORALNPU_TILE_BUDGET_BYTES);
  EXPECT_EQ(p.tiles, (112 + p.rows_per_tile - 1) / p.rows_per_tile);
  EXPECT_LE((p.tiles - 1) * p.rows_per_tile, 112);
}

TEST(PlanRows, FootprintMatchesTheLayoutAndNeverExceedsTheBudget) {
  for (uint32_t budget : {1024u, 2048u, 4096u, 8192u, 24576u}) {
    const RowPlan p = PlanRows(kLayer0, budget);
    if (p.tiled) {
      EXPECT_LE(p.footprint, budget);
      EXPECT_EQ(p.footprint, RowLayoutFor(kLayer0, p.rows_per_tile, p.double_buffer ? 2 : 1).total);
    }
  }
}

TEST(PlanRows, TooSmallABudgetIsNotTiled) { EXPECT_FALSE(PlanRows(kLayer0, 600).tiled); }

TEST(PlanRows, TilesCoverEveryOutputRowAndAreEvened) {
  const RowPlan p = PlanRows(kLayer0, CORALNPU_TILE_BUDGET_BYTES);
  ASSERT_TRUE(p.tiled);
  const int last = 112 - (p.tiles - 1) * p.rows_per_tile;
  EXPECT_GE(last, 1);
  EXPECT_LE(last, p.rows_per_tile);
  // Evened: no tile is more than one row smaller than the nominal size.
  EXPECT_GE(last, p.rows_per_tile - p.tiles);
}

TEST(RowLayout, BuffersAreDisjointAndAligned) {
  const RowLayout l = RowLayoutFor(kLayer0, 4, 2);
  EXPECT_EQ(l.chain % 32, 0u);
  for (uint32_t off : {l.filter, l.bias, l.multiplier, l.shift_raw, l.shift_left, l.shift_right,
                       l.in[0], l.in[1], l.out[0], l.out[1]}) {
    EXPECT_EQ(off % 16, 0u);
    EXPECT_LT(off, l.total + 1);
  }
  EXPECT_GE(l.in[1], l.in[0] + 9u * 224 * 3);
  EXPECT_GE(l.out[1], l.out[0] + 4u * 112 * 8);
}

TEST(RowLayout, ScratchGrowsWithRowsAndFitsInTheBudget) {
  // Depthwise layer 1: 112x112x8, 3x3, stride 1, SAME; accumulators are 4 bytes per output value.
  const RowGeom g   = {112, 112, 8, 112, 112, 8, 3, 3, 1, 1, 72, 112 * 8 * 4};
  const RowLayout l = RowLayoutFor(g, 2, 2);
  EXPECT_GE(l.total - l.scratch, 2u * 112 * 8 * 4);
  const RowPlan p = PlanRows(g, CORALNPU_TILE_BUDGET_BYTES);
  ASSERT_TRUE(p.tiled);
  EXPECT_LE(p.footprint, CORALNPU_TILE_BUDGET_BYTES);
  EXPECT_GE(p.rows_per_tile, 1);
  // The accumulators for the tile are part of the footprint.
  EXPECT_GE(p.footprint, static_cast<uint32_t>(p.rows_per_tile) * 112 * 8 * 4);
}

}  // namespace
}  // namespace coralnpu_v2::opt::litert_micro
