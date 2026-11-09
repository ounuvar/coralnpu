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

// The DTCM arena of the DMA-tiled paths and the row-tiled loop (#11), shared by the conv and
// depthwise kernels. One layer runs at a time, so there is one arena for all of them: the pointwise
// tiler (conv_tiled.cc) and the row tiler below carve it by their own layouts.
//
// Header only, so that conv and depthwise libraries can both use it; the arena and the counters are
// function-local statics of inline functions, which exist once per program.

#ifndef SW_OPT_LITERT_MICRO_ROW_TILED_CORE_H_
#define SW_OPT_LITERT_MICRO_ROW_TILED_CORE_H_

#include <algorithm>
#include <cstdint>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/litert-micro/conv_row_plan.h"
#include "sw/opt/litert-micro/conv_tile_plan.h"
#include "sw/opt/litert-micro/dma_tile.h"

namespace coralnpu_v2::opt::litert_micro {

// Cycle counters for the tiled paths, collected only when built with CORALNPU_TILING_STATS (all
// zero otherwise). They add up over every call since the last reset.
struct ConvTiledStats {
  uint32_t chunks;         // output-channel chunks run (row-tiled: layers run)
  uint32_t tiles;          // tiles run
  uint32_t prologue;       // blocking copies at the start of a chunk
  uint32_t prepare;        // shift and weight preparation
  uint32_t compute;        // the kernel
  uint32_t wait;           // waiting for a chain after the kernel finished (overlapped copies)
  uint32_t blocking_copy;  // blocking copies: single-buffer input and all output write-backs
  uint32_t build;          // building copy chains (descriptors) for the tiles, not waiting
};

inline constexpr uint32_t kTileArenaBytes = CORALNPU_TILE_BUDGET_BYTES;

// The arena: buffers and copy-chain descriptors of the layer being run (32-byte aligned for the
// descriptors).
inline uint8_t *TileArena() {
  alignas(32) static uint8_t arena[kTileArenaBytes];
  return arena;
}

inline ConvTiledStats &TileStats() {
  static ConvTiledStats stats;
  return stats;
}

#ifdef CORALNPU_TILING_STATS
inline uint32_t TileCycles() {
  uint32_t v;
  asm volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}
#define RT_STAT_START(t)      const uint32_t t = TileCycles()
#define RT_STAT_ADD(field, t) TileStats().field += TileCycles() - (t)
#define RT_STAT_COUNT(field)  ++TileStats().field
#else
#define RT_STAT_START(t)
#define RT_STAT_ADD(field, t)
#define RT_STAT_COUNT(field)
#endif

// What a tile kernel gets: the tile's rows, local buffers, and the layer's parameters in local
// copies. `pad_top` is the padding above the first input row, as the kernel should see it.
struct RowTileArgs {
  int pad_top, in_rows, out_rows;
  const int8_t *in;
  int8_t *out;
  const int8_t *filter;
  const int32_t *bias;  // null if the layer has no bias
  const int32_t *multiplier;
  const int32_t *shift_raw;
  const uint8_t *shift_left;
  const uint8_t *shift_right;
  // RowGeom::scratch_row_bytes per output row plus scratch_fixed_bytes, 16-byte aligned; null if
  // none.
  int32_t *scratch;
  const void *prepared;  // what the prepare step wrote (RowGeom::prepared_bytes); null if none
};

// What the prepare step gets: the layer's filter in DTCM and where to write the prepared copy.
struct RowPrepareArgs {
  const int8_t *filter;
  void *prepared;  // RowGeom::prepared_bytes, 16-byte aligned
};

// The layer's data in external memory (or wherever the caller has it).
struct RowData {
  const int8_t *input;
  int8_t *output;
  int batches;
  const int8_t *filter;
  const int32_t *bias;  // may be null
  const int32_t *multiplier;
  const int32_t *shift;
};

// Plans the layer for `budget` and runs it in tiles of output rows; `kernel(const RowTileArgs&)`
// computes one tile on the local buffers and returns false on failure. Returns false without
// writing the output if no plan fits or a tile would have no input rows; returns false after the
// output may have been partly written if the kernel or a copy fails. Either way no copy is still
// running when it returns, so the caller can recompute the layer untiled. `plan_out`, if not
// null, receives the plan.
//
// `prepare(const RowPrepareArgs&)` runs once per layer, after the filter is in DTCM and before the
// first tile, to fill RowGeom::prepared_bytes of prepared filter data; the overload without it is
// for kernels that need none.
template <typename Prepare, typename Kernel>
bool RunRowTiled(const RowGeom &g, const RowData &d, uint32_t budget, RowPlan *plan_out,
                 Prepare &&prepare, Kernel &&kernel) {
  if (budget > kTileArenaBytes)
    return false;
  const RowPlan plan = PlanRows(g, budget);
  if (plan_out)
    *plan_out = plan;
  if (!plan.tiled)
    return false;
  const int rows = plan.rows_per_tile;
  // A tile whose rows all fall in the padding has no input to copy: leave such shapes untiled.
  for (int t = 0; t < plan.tiles; ++t) {
    const int o0 = t * rows;
    if (RowTileFor(g, o0, std::min(rows, g.out_h - o0)).in_rows <= 0)
      return false;
  }

  const int buffers   = plan.double_buffer ? 2 : 1;
  const RowLayout l   = RowLayoutFor(g, rows, buffers);
  uint8_t *const base = TileArena();
  CopyChain c;
  c.Init(base + l.chain, kRowChainCopies);
  // Every return below, including the failures, waits for a started chain first.
  DrainOnExit drain(c);
  int8_t *const filter       = reinterpret_cast<int8_t *>(base + l.filter);
  int32_t *const bias        = reinterpret_cast<int32_t *>(base + l.bias);
  int32_t *const multiplier  = reinterpret_cast<int32_t *>(base + l.multiplier);
  int32_t *const shift_raw   = reinterpret_cast<int32_t *>(base + l.shift_raw);
  uint8_t *const shift_left  = base + l.shift_left;
  uint8_t *const shift_right = base + l.shift_right;
  int32_t *const scratch     = g.scratch_row_bytes > 0 || g.scratch_fixed_bytes > 0
                                   ? reinterpret_cast<int32_t *>(base + l.scratch)
                                   : nullptr;
  void *const prepared       = g.prepared_bytes > 0 ? base + l.prepared : nullptr;
  int8_t *const in_buf[2]  = {reinterpret_cast<int8_t *>(base + l.in[0]),
                              reinterpret_cast<int8_t *>(base + l.in[1])};
  int8_t *const out_buf[2] = {reinterpret_cast<int8_t *>(base + l.out[0]),
                              reinterpret_cast<int8_t *>(base + l.out[1])};

  const int64_t in_row_bytes  = static_cast<int64_t>(g.in_w) * g.in_d;
  const int64_t out_row_bytes = static_cast<int64_t>(g.out_w) * g.out_d;
  const int total             = d.batches * plan.tiles;
  // Tile i is tile (i % tiles) of image (i / tiles).
  struct Tile {
    const int8_t *src;
    int8_t *dst;
    uint32_t in_bytes, out_bytes;
    int out_rows, in_rows, pad_top;
  };
  const auto tile_at = [&](int i) {
    const int b      = i / plan.tiles;
    const int o0     = (i % plan.tiles) * rows;
    const int nr     = std::min(rows, g.out_h - o0);
    const RowTile rt = RowTileFor(g, o0, nr);
    Tile t;
    t.src       = d.input + (static_cast<int64_t>(b) * g.in_h + rt.in_start) * in_row_bytes;
    t.dst       = d.output + (static_cast<int64_t>(b) * g.out_h + o0) * out_row_bytes;
    t.in_bytes  = static_cast<uint32_t>(rt.in_rows * in_row_bytes);
    t.out_bytes = static_cast<uint32_t>(nr * out_row_bytes);
    t.out_rows  = nr;
    t.in_rows   = rt.in_rows;
    t.pad_top   = rt.pad_top;
    return t;
  };

  // Filter, bias, multipliers, shifts and the first input tile.
  {
    RT_STAT_COUNT(chunks);
    RT_STAT_START(t0);
    const Tile first = tile_at(0);
    c.Clear();
    bool ok = c.Add(d.filter, filter, static_cast<uint32_t>(g.filter_bytes));
    if (d.bias)
      ok &= c.Add(d.bias, bias, 4u * g.out_d);
    ok &= c.Add(d.multiplier, multiplier, 4u * g.out_d);
    ok &= c.Add(d.shift, shift_raw, 4u * g.out_d);
    ok &= c.Add(first.src, in_buf[0], first.in_bytes);
    if (!ok || !c.Run())
      return false;
    RT_STAT_ADD(prologue, t0);
  }
  {
    RT_STAT_START(tp);
    PrepareShiftParams(shift_left, shift_right, shift_raw, g.out_d);
    if (prepared)
      prepare(RowPrepareArgs{filter, prepared});
    RT_STAT_ADD(prepare, tp);
  }

  const auto compute = [&](const Tile &t, int8_t *in, int8_t *out) {
    RowTileArgs a;
    a.pad_top     = t.pad_top;
    a.in_rows     = t.in_rows;
    a.out_rows    = t.out_rows;
    a.in          = in;
    a.out         = out;
    a.filter      = filter;
    a.bias        = d.bias ? bias : nullptr;
    a.multiplier  = multiplier;
    a.shift_raw   = shift_raw;
    a.shift_left  = shift_left;
    a.shift_right = shift_right;
    a.scratch     = scratch;
    a.prepared    = prepared;
    RT_STAT_COUNT(tiles);
    RT_STAT_START(tc);
    const bool ok = kernel(a);
    RT_STAT_ADD(compute, tc);
    return ok;
  };

  if (buffers == 2) {
    // Each tile's chain moves the previous tile's output out and the next tile's input in while
    // the kernel computes this tile.
    for (int i = 0; i < total; ++i) {
      const int cur = i & 1;
      const Tile t  = tile_at(i);
      c.Clear();
      bool ok = true;
      {
        RT_STAT_START(tb);
        if (i > 0) {
          const Tile p = tile_at(i - 1);
          ok &= c.Add(out_buf[cur ^ 1], p.dst, p.out_bytes);
        }
        if (i + 1 < total) {
          const Tile n = tile_at(i + 1);
          ok &= c.Add(n.src, in_buf[cur ^ 1], n.in_bytes);
        }
        RT_STAT_ADD(build, tb);
      }
      if (!ok)
        return false;
      c.Start();
      if (!compute(t, in_buf[cur], out_buf[cur]))
        return false;
      RT_STAT_START(tw);
      if (!c.Wait())
        return false;
      RT_STAT_ADD(wait, tw);
    }
    const Tile last = tile_at(total - 1);
    c.Clear();
    RT_STAT_START(tf);
    if (!c.Add(out_buf[(total - 1) & 1], last.dst, last.out_bytes) || !c.Run())
      return false;
    RT_STAT_ADD(blocking_copy, tf);
  } else {
    for (int i = 0; i < total; ++i) {
      const Tile t = tile_at(i);
      c.Clear();
      RT_STAT_START(ti);
      if (i > 0 && (!c.Add(t.src, in_buf[0], t.in_bytes) || !c.Run()))
        return false;
      RT_STAT_ADD(blocking_copy, ti);
      if (!compute(t, in_buf[0], out_buf[0]))
        return false;
      c.Clear();
      RT_STAT_START(to);
      if (!c.Add(out_buf[0], t.dst, t.out_bytes) || !c.Run())
        return false;
      RT_STAT_ADD(blocking_copy, to);
    }
  }
  return true;
}

template <typename Kernel>
bool RunRowTiled(const RowGeom &g, const RowData &d, uint32_t budget, RowPlan *plan_out,
                 Kernel &&kernel) {
  return RunRowTiled(g, d, budget, plan_out, [](const RowPrepareArgs &) {}, kernel);
}

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_ROW_TILED_CORE_H_
