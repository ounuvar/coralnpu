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

// Tile planner for DMA-tiled spatial convolution (#11): filters larger than 1x1, such as the 3x3
// stride-2 first layer of MobileNet.
//
// A tile is a band of output rows for all output channels. Its input is the band of input rows
// the filter reaches, clipped to the image: neighbouring tiles overlap by (filter height - stride)
// rows, which are fetched again rather than kept. The whole filter must fit next to the buffers;
// there is no output-channel chunking here (layer 0 has 216 bytes of filter).
// Depthwise convs (one filter plane per channel) use the same planner; their kernel also needs
// per-row accumulator scratch (RowGeom::scratch_row_bytes). A kernel can also have fixed scratch
// and a prepared copy of the filter (RowGeom::scratch_fixed_bytes, prepared_bytes).
//
// RowLayout is the one description of the DTCM buffers: the planner sizes a plan with it and the
// wrapper carves its arena with the same offsets.
//
// Portable: no CoralNPU headers, so it builds for the host (unit tests) and for CoralNPU.

#ifndef SW_OPT_LITERT_MICRO_CONV_ROW_PLAN_H_
#define SW_OPT_LITERT_MICRO_CONV_ROW_PLAN_H_

#include <algorithm>
#include <cstdint>

#include "sw/opt/litert-micro/conv_tile_plan.h"

namespace coralnpu_v2::opt::litert_micro {

// The shape of a spatial conv, one group, dilation 1.
struct RowGeom {
  int in_h, in_w, in_d;
  int out_h, out_w, out_d;
  int fh, fw;
  int stride_h;
  int pad_h;  // rows of padding above the image
  // Bytes of filter (conv: out_d * fh * fw * in_d; depthwise: fh * fw * out_d).
  int filter_bytes;
  // Scratch the kernel needs per output row of a tile, in bytes (depthwise: the int32 accumulators,
  // out_w * out_d * 4); 0 if none.
  int scratch_row_bytes;
  // Scratch the kernel needs per tile whatever its size (the RVV 3x3 conv: accumulators for a few
  // pixels), in bytes; 0 if none.
  int scratch_fixed_bytes;
  // Bytes of filter data prepared once per layer from the local filter (the RVV 3x3 conv: its int16
  // weight layout); 0 if none.
  int prepared_bytes;
};

// Copies in one chain: filter, bias, multipliers, shifts, and the first input tile.
inline constexpr int kRowChainCopies = 5;

// Input rows a tile of `rows` output rows reaches, before clipping to the image.
constexpr int RowInputRows(const RowGeom &g, int rows) { return (rows - 1) * g.stride_h + g.fh; }

// Where a tile of output rows [o0, o0 + rows) gets its input, and the padding the kernel is told.
struct RowTile {
  int in_start;  // first input row, clipped to the image
  int in_rows;   // number of input rows, clipped to the image; 0 if the tile reaches none
  int pad_top;   // padding rows above in_start, as the kernel sees them
};
constexpr RowTile RowTileFor(const RowGeom &g, int o0, int rows) {
  const int first = o0 * g.stride_h - g.pad_h;
  const int last  = first + RowInputRows(g, rows);  // one past the last row reached
  const int s     = first < 0 ? 0 : first;
  const int e     = last > g.in_h ? g.in_h : last;
  return RowTile{s, e > s ? e - s : 0, s - o0 * g.stride_h + g.pad_h};
}

struct RowLayout {
  uint32_t chain, filter, prepared, bias, multiplier, shift_raw, shift_left, shift_right, in[2],
      out[2], scratch, total;
};

constexpr RowLayout RowLayoutFor(const RowGeom &g, int rows, int buffers) {
  RowLayout l{};
  uint32_t at = 0;
  l.chain     = at;
  at += RoundUp16(kDescriptorBytes * kRowChainCopies);
  l.filter = at;
  at += RoundUp16(static_cast<uint32_t>(g.filter_bytes));
  l.prepared = at;
  at += RoundUp16(static_cast<uint32_t>(g.prepared_bytes));
  l.bias = at;
  at += RoundUp16(static_cast<uint32_t>(g.out_d) * 4);
  l.multiplier = at;
  at += RoundUp16(static_cast<uint32_t>(g.out_d) * 4);
  l.shift_raw = at;
  at += RoundUp16(static_cast<uint32_t>(g.out_d) * 4);
  l.shift_left = at;
  at += RoundUp16(static_cast<uint32_t>(g.out_d));
  l.shift_right = at;
  at += RoundUp16(static_cast<uint32_t>(g.out_d));
  const uint32_t in_bytes =
      RoundUp16(static_cast<uint32_t>(RowInputRows(g, rows)) * g.in_w * g.in_d);
  const uint32_t out_bytes = RoundUp16(static_cast<uint32_t>(rows) * g.out_w * g.out_d);
  for (int b = 0; b < 2; ++b) {
    const bool used = b < buffers;
    l.in[b]         = at;
    at += used ? in_bytes : 0;
    l.out[b] = at;
    at += used ? out_bytes : 0;
  }
  l.scratch = at;
  at += RoundUp16(static_cast<uint32_t>(rows) * static_cast<uint32_t>(g.scratch_row_bytes) +
                  static_cast<uint32_t>(g.scratch_fixed_bytes));
  l.total = at;
  return l;
}

struct RowPlan {
  bool tiled;
  int rows_per_tile;  // output rows per tile (the last tile may have fewer)
  int tiles;          // tiles per image
  bool double_buffer;
  uint32_t footprint;  // bytes of arena used
};

// Largest `rows` such that the layout fits `budget`, or 0.
constexpr int LargestRows(const RowGeom &g, int buffers, uint32_t budget) {
  int best = 0;
  for (int r = 1; r <= g.out_h; ++r) {
    if (RowLayoutFor(g, r, buffers).total > budget)
      break;
    best = r;
  }
  return best;
}

// Picks rows per tile for the budget: double buffered if at least two rows per tile fit that way
// (or the whole image is one tile), otherwise single buffered. Tiles are then evened out, so that
// the last one is not much smaller than the others.
constexpr RowPlan PlanRows(const RowGeom &g, uint32_t budget) {
  RowPlan p{};
  if (g.out_h <= 0 || g.out_w <= 0 || g.stride_h <= 0 || g.fh <= 0)
    return p;
  int rows     = LargestRows(g, 2, budget);
  int buffers  = 2;
  const int lo = g.out_h < 2 ? g.out_h : 2;
  if (rows < lo) {
    rows    = LargestRows(g, 1, budget);
    buffers = 1;
  }
  if (rows < 1)
    return p;
  const int tiles = (g.out_h + rows - 1) / rows;
  rows            = (g.out_h + tiles - 1) / tiles;
  p.tiled         = true;
  p.rows_per_tile = rows;
  p.tiles         = tiles;
  p.double_buffer = buffers == 2;
  p.footprint     = RowLayoutFor(g, rows, buffers).total;
  return p;
}

// PlanRows, or no plan if `budget` is more than the arena holds (`arena_bytes`).
constexpr RowPlan PlanRowsInArena(const RowGeom &g, uint32_t budget, uint32_t arena_bytes) {
  return budget > arena_bytes ? RowPlan{} : PlanRows(g, budget);
}

// A layer's tile plan: which tiled path runs it, if any, and that path's plan. It depends only on
// the layer's shapes and parameters and on build-time settings, so ConvPrepare and
// DepthwiseConvPrepare compute it once and every inference reuses it (#11).
struct LayerTilePlan {
  enum Kind : uint8_t { kNone = 0, kPointwise, kRow };
  Kind kind;
  ConvTilePlan pointwise;  // kind == kPointwise
  RowPlan row;             // kind == kRow
};

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_CONV_ROW_PLAN_H_
