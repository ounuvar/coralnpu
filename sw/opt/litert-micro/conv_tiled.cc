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

#include "sw/opt/litert-micro/conv_tiled.h"

#include <algorithm>
#include <cstdint>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/litert-micro/conv_pointwise.h"
#include "sw/opt/litert-micro/dma_tile.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

namespace coralnpu_v2::opt::litert_micro {
namespace {

#if CORALNPU_RVV_CONV
constexpr bool kRvv = true;
#else
constexpr bool kRvv = false;
#endif

constexpr uint32_t kArenaBytes = kTileArenaBytes;

// Everything the tiled path keeps in DTCM: one arena that the buffers and the copy chains' DMA
// descriptors are carved from by the plan's layout. One layer at a time, so one set. The copy
// chains' storage lives in the arena (32-byte aligned for the descriptors).
CopyChain g_prologue;
CopyChain g_chains[2];

#ifdef CORALNPU_TILING_STATS
inline uint32_t Cycles() {
  uint32_t v;
  asm volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}
#define STAT_START(t)      const uint32_t t = Cycles()
#define STAT_ADD(field, t) TileStats().field += Cycles() - (t)
#define STAT_COUNT(field)  ++TileStats().field
#else
#define STAT_START(t)
#define STAT_ADD(field, t)
#define STAT_COUNT(field)
#endif

// The per-chunk raw buffers: filter rows, bias, multipliers and shifts of one chunk.
struct Raw {
  int8_t *filter;
  int32_t *bias;
  int32_t *mult;
  int32_t *shift;
};

}  // namespace

void ConvTiledResetStats() { TileStats() = ConvTiledStats{}; }
ConvTiledStats ConvTiledGetStats() { return TileStats(); }

bool ConvTiledUsesRvvKernel() { return kRvv; }

namespace {

// The plan for a pointwise conv (1x1 filter, stride 1, no padding, one group), or no plan if the
// layer is not one or nothing fits `budget`.
ConvTilePlan PlanPointwiseLayer(const tflite::ConvParams &params,
                                const tflite::RuntimeShape &input_shape,
                                const tflite::RuntimeShape &filter_shape,
                                const tflite::RuntimeShape &output_shape, uint32_t budget) {
  const ConvTilePlan none{};
  if (params.stride_width != 1 || params.stride_height != 1 || params.padding_values.width != 0 ||
      params.padding_values.height != 0) {
    return none;
  }
  if (input_shape.DimensionsCount() != 4 || filter_shape.DimensionsCount() != 4 ||
      output_shape.DimensionsCount() != 4) {
    return none;
  }
  const int in_d  = input_shape.Dims(3);
  const int out_d = output_shape.Dims(3);
  if (filter_shape.Dims(1) != 1 || filter_shape.Dims(2) != 1 || filter_shape.Dims(0) != out_d ||
      filter_shape.Dims(3) != in_d || input_shape.Dims(0) != output_shape.Dims(0) ||
      input_shape.Dims(1) != output_shape.Dims(1) || input_shape.Dims(2) != output_shape.Dims(2)) {
    return none;
  }
  if (budget > kArenaBytes)
    return none;
  const int pixels = input_shape.Dims(0) * input_shape.Dims(1) * input_shape.Dims(2);
  return PlanPointwise(pixels, in_d, out_d, kRvv, budget, kPrefetchDefault, kMergeDefault);
}

// Runs a pointwise conv with `plan` (from PlanPointwiseLayer for the same shapes).
bool RunPointwise(const ConvTilePlan &plan, const tflite::ConvParams &params,
                  const int32_t *output_multiplier, const int32_t *output_shift,
                  const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                  const int8_t *filter_data, const int32_t *bias_data,
                  const tflite::RuntimeShape &output_shape, int8_t *output_data) {
  if (!plan.tiled)
    return false;
  const int in_d  = input_shape.Dims(3);
  const int out_d = output_shape.Dims(3);

  const int tile            = plan.pixels_per_tile;
  const int tiles           = plan.tiles;
  const int chunk           = plan.channel_chunk;
  const int buffers         = plan.double_buffer ? 2 : 1;
  const PointwiseLayout lay = PointwiseLayoutFor(in_d, chunk, plan.buffer_pixels, buffers, kRvv,
                                                 kPrefetchDefault, chunk < out_d);
  g_prologue.Init(TileArena() + lay.prologue_chain, kPrologueCopies);
  for (int b = 0; b < 2; ++b)
    g_chains[b].Init(TileArena() + lay.chain[b], lay.chain_capacity);
  // Every return below, including the failures, waits for any started chain first (the
  // overlapped flush or prefetch may still be running when a later step fails).
  DrainOnExit drain(g_prologue, g_chains[0], g_chains[1]);

  uint8_t *const base                         = TileArena();
  const Raw raws[2]                           = {{reinterpret_cast<int8_t *>(base + lay.filter),
                                                  reinterpret_cast<int32_t *>(base + lay.bias_raw),
                                                  reinterpret_cast<int32_t *>(base + lay.multiplier),
                                                  reinterpret_cast<int32_t *>(base + lay.shift_raw)},
                                                 {reinterpret_cast<int8_t *>(base + lay.filter2),
                                                  reinterpret_cast<int32_t *>(base + lay.bias_raw2),
                                                  reinterpret_cast<int32_t *>(base + lay.multiplier2),
                                                  reinterpret_cast<int32_t *>(base + lay.shift_raw2)}};
  [[maybe_unused]] int16_t *const weights16   = reinterpret_cast<int16_t *>(base + lay.weights16);
  [[maybe_unused]] int32_t *const bias_folded = reinterpret_cast<int32_t *>(base + lay.bias_folded);
  uint8_t *const shift_left                   = base + lay.shift_left;
  uint8_t *const shift_right                  = base + lay.shift_right;
  [[maybe_unused]] int32_t *const accs        = reinterpret_cast<int32_t *>(base + lay.accs);
  int8_t *const in_tile[2]                    = {reinterpret_cast<int8_t *>(base + lay.in[0]),
                                                 reinterpret_cast<int8_t *>(base + lay.in[1])};
  int8_t *const out_tile[2]                   = {reinterpret_cast<int8_t *>(base + lay.out[0]),
                                                 reinterpret_cast<int8_t *>(base + lay.out[1])};

  // Tiles are `tile` pixels, except the last, which holds what is left (merged: more than `tile`).
  const auto pixels_in_tile = [&](int t) { return t == tiles - 1 ? plan.last_tile_pixels : tile; };
  const auto tile_start     = [&](int t) { return t * tile; };

  // Adds part `part` of `parts` of the copies that bring chunk [n0, n0 + chunk) of the filter,
  // bias, multipliers and shifts into the raw buffers `r`. The filter rows are contiguous in the
  // OHWI filter, so they split into pieces; part 0 also carries the small bias, multiplier and
  // shift copies. Empty pieces add nothing.
  const auto add_raw_part = [&](auto &c, int n0, const Raw &r, int part, int parts) {
    const int cn         = std::min(chunk, out_d - n0);
    const uint32_t total = static_cast<uint32_t>(cn) * in_d;
    const uint32_t per   = RoundUp16((total + parts - 1) / parts);
    const uint32_t off   = std::min(per * static_cast<uint32_t>(part), total);
    const uint32_t len   = std::min(per, total - off);
    bool added           = true;
    if (len > 0) {
      added &= c.Add(filter_data + static_cast<size_t>(n0) * in_d + off, r.filter + off, len);
    }
    if (part == 0) {
      if (bias_data)
        added &= c.Add(bias_data + n0, r.bias, 4u * cn);
      added &= c.Add(output_multiplier + n0, r.mult, 4u * cn);
      added &= c.Add(output_shift + n0, r.shift, 4u * cn);
    }
    return added;
  };
  const auto add_raw = [&](auto &c, int n0, const Raw &r) { return add_raw_part(c, n0, r, 0, 1); };

  bool ok         = true;
  int rs          = 0;      // which raw buffer set holds the current chunk
  int buf0        = 0;      // input/output buffer of the current chunk's first tile
  bool prefetched = false;  // raw set rs (and, double buffered, the first input tile) are in place
  bool flush_pending = false;  // the last tile's output copy of the previous chunk is still running
  for (int n0 = 0; n0 < out_d; n0 += chunk) {
    STAT_COUNT(chunks);
    const int cn             = std::min(chunk, out_d - n0);
    const Raw &r             = raws[rs];
    const bool more_chunks   = n0 + chunk < out_d;
    const bool prefetch_next = kPrefetchDefault && more_chunks;
    const int np0            = pixels_in_tile(0);

    // The chunk's filter rows are contiguous in the OHWI filter, so one copy brings them in. The
    // first input tile comes with them, unless the previous chunk already prefetched it.
    if (!prefetched) {
      STAT_START(t0);
      g_prologue.Clear();
      ok &= add_raw(g_prologue, n0, r);
      ok &= g_prologue.Add(input_data, in_tile[buf0], static_cast<uint32_t>(np0) * in_d);
      if (!ok || !g_prologue.Run())
        return false;
      STAT_ADD(prologue, t0);
    } else if (buffers == 1) {
      // The raw buffers were prefetched; the first input tile was not (no spare buffer).
      STAT_START(t0);
      g_prologue.Clear();
      ok &= g_prologue.Add(input_data, in_tile[buf0], static_cast<uint32_t>(np0) * in_d);
      if (!ok || !g_prologue.Run())
        return false;
      STAT_ADD(prologue, t0);
    }
    prefetched = false;

    {
      STAT_START(t0);
      PrepareShiftParams(shift_left, shift_right, r.shift, cn);
#if CORALNPU_RVV_CONV
      Conv1x1Prepare(params, in_d, cn, r.filter, bias_data ? r.bias : nullptr, weights16,
                     bias_folded);
#endif
      STAT_ADD(prepare, t0);
    }

    // The previous chunk's output flush, started without waiting, must be done before this chunk
    // builds or starts any copy chain (chain 0 is shared, and the DMA runs one transfer at a time).
    if (flush_pending) {
      STAT_START(tw);
      if (!g_chains[0].Wait())
        return false;
      STAT_ADD(blocking_copy, tw);
      flush_pending = false;
    }

    const auto compute = [&](int buf, int np) {
      STAT_START(t0);
#if CORALNPU_RVV_CONV
      Conv1x1Run(params, r.mult, shift_left, shift_right, in_d, cn, np, in_tile[buf], weights16,
                 bias_folded, accs, out_tile[buf]);
#else
      const int32_t in_dims[4]   = {1, 1, np, in_d};
      const int32_t flt_dims[4]  = {cn, 1, 1, in_d};
      const int32_t out_dims[4]  = {1, 1, np, cn};
      const int32_t bias_dims[1] = {cn};
      tflite::reference_integer_ops::ConvPerChannel(
          params, r.mult, r.shift, tflite::RuntimeShape(4, in_dims), in_tile[buf],
          tflite::RuntimeShape(4, flt_dims), r.filter, tflite::RuntimeShape(1, bias_dims),
          bias_data ? r.bias : nullptr, tflite::RuntimeShape(4, out_dims), out_tile[buf]);
#endif
      STAT_ADD(compute, t0);
      STAT_COUNT(tiles);
    };

    // Writes tile t's output (in out_tile[buf]) back: one copy if the tile holds every channel,
    // otherwise one copy per pixel, since the DMA has no strided mode.
    const auto add_output = [&](auto &c, int t, int buf) {
      const int np = pixels_in_tile(t);
      int8_t *dst  = output_data + static_cast<size_t>(tile_start(t)) * out_d;
      if (cn == out_d) {
        return c.Add(out_tile[buf], dst, static_cast<uint32_t>(np) * out_d);
      }
      // One copy per pixel, with the pixel's slice of the channels: equal copies, strided.
      return c.AddStrided(out_tile[buf], static_cast<uint32_t>(cn), dst + n0,
                          static_cast<uint32_t>(out_d), static_cast<uint32_t>(cn), np);
    };

    if (buffers == 2) {
      // While tile t computes, the DMA writes tile t-1's output back and brings tile t+1 in. During
      // the last tile it also prefetches the next chunk's raw buffers and first input tile.
      int last_buf = buf0;
      for (int t = 0; t < tiles; ++t) {
        const int buf = buf0 ^ (t & 1);
        last_buf      = buf;
        CopyChain &c  = g_chains[t & 1];
        STAT_START(tb);
        c.Clear();
        if (t > 0)
          ok &= add_output(c, t - 1, buf ^ 1);
        if (t + 1 < tiles) {
          ok &= c.Add(input_data + static_cast<size_t>(tile_start(t + 1)) * in_d, in_tile[buf ^ 1],
                      static_cast<uint32_t>(pixels_in_tile(t + 1)) * in_d);
        }
        if (prefetch_next) {
          if (kSpreadDefault)
            ok &= add_raw_part(c, n0 + chunk, raws[rs ^ 1], t, tiles);
          else if (t + 1 == tiles)
            ok &= add_raw(c, n0 + chunk, raws[rs ^ 1]);
          if (t + 1 == tiles)
            ok &= c.Add(input_data, in_tile[buf ^ 1], static_cast<uint32_t>(np0) * in_d);
        }
        STAT_ADD(build, tb);
        if (!ok)
          return false;
        c.Start();
        compute(buf, pixels_in_tile(t));
        STAT_START(t1);
        if (!c.Wait())
          return false;
        STAT_ADD(wait, t1);
      }
      STAT_START(t2);
      CopyChain &last = g_chains[0];
      {
        STAT_START(tb);
        last.Clear();
        ok &= add_output(last, tiles - 1, last_buf);
        STAT_ADD(build, tb);
      }
      if (!ok)
        return false;
      if (kOverlapFlushDefault && prefetch_next) {
        // The next chunk starts with weight preparation (no DMA), which runs while this copies.
        last.Start();
        flush_pending = true;
      } else if (!last.Run()) {
        return false;
      }
      STAT_ADD(blocking_copy, t2);
      buf0 = last_buf ^ 1;  // the next chunk's first tile uses the spare buffer
    } else {
      CopyChain &c = g_chains[0];
      for (int t = 0; t < tiles; ++t) {
        c.Clear();
        if (t > 0) {
          STAT_START(t1);
          ok &= c.Add(input_data + static_cast<size_t>(tile_start(t)) * in_d, in_tile[0],
                      static_cast<uint32_t>(pixels_in_tile(t)) * in_d);
          if (!ok || !c.Run())
            return false;
          STAT_ADD(blocking_copy, t1);
          c.Clear();
        }
        // The DMA prefetches the next chunk's raw buffers: during the last tile, or spread over
        // all.
        const bool prefetch_now = prefetch_next && (kSpreadDefault || t == tiles - 1);
        CopyChain &pf           = g_chains[1];
        if (prefetch_now) {
          STAT_START(tb);
          pf.Clear();
          if (kSpreadDefault)
            ok &= add_raw_part(pf, n0 + chunk, raws[rs ^ 1], t, tiles);
          else
            ok &= add_raw(pf, n0 + chunk, raws[rs ^ 1]);
          STAT_ADD(build, tb);
          if (!ok)
            return false;
          pf.Start();
        }
        compute(0, pixels_in_tile(t));
        if (prefetch_now) {
          STAT_START(t3);
          if (!pf.Wait())
            return false;
          STAT_ADD(wait, t3);
        }
        {
          STAT_START(tb);
          ok &= add_output(c, t, 0);
          STAT_ADD(build, tb);
        }
        STAT_START(t2);
        if (!ok || !c.Run())
          return false;
        STAT_ADD(blocking_copy, t2);
      }
    }
    if (prefetch_next) {
      prefetched = true;
      rs ^= 1;
    }
  }
  return true;
}

}  // namespace

bool ConvPointwiseTiledWithBudget(const tflite::ConvParams &params,
                                  const int32_t *output_multiplier, const int32_t *output_shift,
                                  const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                                  const tflite::RuntimeShape &filter_shape,
                                  const int8_t *filter_data, const int32_t *bias_data,
                                  const tflite::RuntimeShape &output_shape, int8_t *output_data,
                                  uint32_t budget, ConvTilePlan *plan_out) {
  const ConvTilePlan plan =
      PlanPointwiseLayer(params, input_shape, filter_shape, output_shape, budget);
  if (plan_out)
    *plan_out = plan;
  return RunPointwise(plan, params, output_multiplier, output_shift, input_shape, input_data,
                      filter_data, bias_data, output_shape, output_data);
}

namespace {

// The RVV 3x3 kernel requantizes every kTileAccPixels pixels, so its accumulators take
// kTileAccPixels * out_d * 4 bytes of the arena (2 KB for MobileNet's layer 0) instead of a whole
// output row (14 KB).
constexpr int kTileAccPixels = 16;

// Whether tiles run the RVV 3x3 kernel (on weights prepared once per layer), or the TFLite
// reference kernel.
bool RowUsesConv3x3([[maybe_unused]] const tflite::RuntimeShape &filter_shape) {
#if CORALNPU_RVV_CONV
  return filter_shape.Dims(1) == 3 && filter_shape.Dims(2) == 3;
#else
  return false;
#endif
}

// The row geometry of a spatial conv (one group, dilation 1), with the RVV 3x3 kernel's prepared
// weights and accumulators when it runs the tiles. False if the layer is not one the row tiler
// handles.
bool ConvRowGeom(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                 const tflite::RuntimeShape &filter_shape, const tflite::RuntimeShape &output_shape,
                 RowGeom *geom) {
  if (input_shape.DimensionsCount() != 4 || filter_shape.DimensionsCount() != 4 ||
      output_shape.DimensionsCount() != 4) {
    return false;
  }
  if (params.dilation_width_factor != 1 || params.dilation_height_factor != 1)
    return false;
  RowGeom g{};
  g.in_h         = input_shape.Dims(1);
  g.in_w         = input_shape.Dims(2);
  g.in_d         = input_shape.Dims(3);
  g.out_h        = output_shape.Dims(1);
  g.out_w        = output_shape.Dims(2);
  g.out_d        = output_shape.Dims(3);
  g.fh           = filter_shape.Dims(1);
  g.fw           = filter_shape.Dims(2);
  g.stride_h     = params.stride_height;
  g.pad_h        = params.padding_values.height;
  g.filter_bytes = g.out_d * g.fh * g.fw * g.in_d;
  // One group only: the filter's depth is the input depth.
  if (input_shape.Dims(0) != output_shape.Dims(0) || filter_shape.Dims(0) != g.out_d ||
      filter_shape.Dims(3) != g.in_d || g.stride_h < 1 || g.pad_h < 0 || g.in_h < 1) {
    return false;
  }
#if CORALNPU_RVV_CONV
  if (RowUsesConv3x3(filter_shape)) {
    g.prepared_bytes      = Conv3x3WeightCount(g.out_d, g.fh * g.fw * g.in_d) * 2;
    g.scratch_fixed_bytes = kTileAccPixels * g.out_d * 4;
  }
#endif
  *geom = g;
  return true;
}

// Runs a spatial conv with `plan` (from PlanRowsInArena for the same RowGeom).
bool RunConvRow(const RowGeom &g, const RowPlan &plan, const tflite::ConvParams &params,
                const int32_t *output_multiplier, const int32_t *output_shift, int batches,
                const int8_t *input_data, const tflite::RuntimeShape &filter_shape,
                const int8_t *filter_data, const int32_t *bias_data, int8_t *output_data) {
  [[maybe_unused]] const bool conv3x3 = RowUsesConv3x3(filter_shape);
  [[maybe_unused]] const int k_size   = g.fh * g.fw * g.in_d;
  RowData d{};
  d.input            = input_data;
  d.output           = output_data;
  d.batches          = batches;
  d.filter           = filter_data;
  d.bias             = bias_data;
  d.multiplier       = output_multiplier;
  d.shift            = output_shift;
  const auto prepare = [&]([[maybe_unused]] const RowPrepareArgs &p) {
#if CORALNPU_RVV_CONV
    Conv3x3Prepare(g.out_d, k_size, p.filter, static_cast<int16_t *>(p.prepared));
#endif
  };
  return RunRowTiled(g, d, plan, prepare, [&](const RowTileArgs &a) {
    tflite::ConvParams sub    = params;
    sub.padding_values.height = a.pad_top;
    const int32_t in_dims[4]  = {1, a.in_rows, g.in_w, g.in_d};
    const int32_t out_dims[4] = {1, a.out_rows, g.out_w, g.out_d};
    const tflite::RuntimeShape in_shape(4, in_dims), out_shape(4, out_dims);
#if CORALNPU_RVV_CONV
    if (conv3x3) {
      Conv3x3Run(sub, a.multiplier, a.shift_left, a.shift_right, in_shape, a.in, filter_shape,
                 a.filter, static_cast<const int16_t *>(a.prepared), a.bias, out_shape, a.out,
                 a.scratch, kTileAccPixels);
      return true;
    }
#endif
    const int32_t bias_dims[1] = {g.out_d};
    tflite::reference_integer_ops::ConvPerChannel(
        sub, a.multiplier, a.shift_raw, in_shape, a.in, filter_shape, a.filter,
        tflite::RuntimeShape(1, bias_dims), a.bias, out_shape, a.out);
    return true;
  });
}

}  // namespace

bool ConvRowTiledWithBudget(const tflite::ConvParams &params, const int32_t *output_multiplier,
                            const int32_t *output_shift, const tflite::RuntimeShape &input_shape,
                            const int8_t *input_data, const tflite::RuntimeShape &filter_shape,
                            const int8_t *filter_data, const int32_t *bias_data,
                            const tflite::RuntimeShape &output_shape, int8_t *output_data,
                            uint32_t budget, RowPlan *plan_out) {
  RowGeom g{};
  const bool ok      = ConvRowGeom(params, input_shape, filter_shape, output_shape, &g);
  const RowPlan plan = ok ? PlanRowsInArena(g, budget, kTileArenaBytes) : RowPlan{};
  if (plan_out)
    *plan_out = plan;
  return ok && RunConvRow(g, plan, params, output_multiplier, output_shift, input_shape.Dims(0),
                          input_data, filter_shape, filter_data, bias_data, output_data);
}

LayerTilePlan PlanConvTiled(const tflite::ConvParams &params,
                            const tflite::RuntimeShape &input_shape,
                            const tflite::RuntimeShape &filter_shape,
                            const tflite::RuntimeShape &output_shape) {
  LayerTilePlan p{};
  p.pointwise = PlanPointwiseLayer(params, input_shape, filter_shape, output_shape,
                                   CORALNPU_TILE_BUDGET_BYTES);
  if (p.pointwise.tiled) {
    p.kind = LayerTilePlan::kPointwise;
    return p;
  }
  // Row tiling only for 3x3 filters: the 4x4 layers keep their optimized untiled kernels (#11
  // decision), and nothing else in MobileNet needs it.
  RowGeom g{};
  if (filter_shape.DimensionsCount() == 4 && filter_shape.Dims(1) == 3 &&
      filter_shape.Dims(2) == 3 &&
      ConvRowGeom(params, input_shape, filter_shape, output_shape, &g)) {
    p.row = PlanRowsInArena(g, CORALNPU_TILE_BUDGET_BYTES, kTileArenaBytes);
    if (p.row.tiled)
      p.kind = LayerTilePlan::kRow;
  }
  return p;
}

bool ConvTiled(const LayerTilePlan &plan, const tflite::ConvParams &params,
               const int32_t *output_multiplier, const int32_t *output_shift,
               const tflite::RuntimeShape &input_shape, const int8_t *input_data,
               const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
               const int32_t *bias_data, const tflite::RuntimeShape &output_shape,
               int8_t *output_data) {
  switch (plan.kind) {
    case LayerTilePlan::kPointwise:
      return RunPointwise(plan.pointwise, params, output_multiplier, output_shift, input_shape,
                          input_data, filter_data, bias_data, output_shape, output_data);
    case LayerTilePlan::kRow: {
      RowGeom g{};
      return ConvRowGeom(params, input_shape, filter_shape, output_shape, &g) &&
             RunConvRow(g, plan.row, params, output_multiplier, output_shift, input_shape.Dims(0),
                        input_data, filter_shape, filter_data, bias_data, output_data);
    }
    default:
      return false;
  }
}

}  // namespace coralnpu_v2::opt::litert_micro
