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

constexpr uint32_t kArenaBytes = ArenaBudget(CORALNPU_TILE_BUDGET_BYTES);

// Everything the tiled path keeps in DTCM: the arena the buffers are carved from, and the copy
// chains with their DMA descriptors. One layer at a time, so one set.
alignas(16) uint8_t g_arena[kArenaBytes];
CopyChain<kPrologueCopies> g_prologue;
CopyChain<kMaxChainCopies> g_chains[2];

}  // namespace

bool ConvTiledUsesRvvKernel() { return kRvv; }

bool ConvPointwiseTiledWithBudget(const tflite::ConvParams &params,
                                  const int32_t *output_multiplier, const int32_t *output_shift,
                                  const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                                  const tflite::RuntimeShape &filter_shape,
                                  const int8_t *filter_data, const int32_t *bias_data,
                                  const tflite::RuntimeShape &output_shape, int8_t *output_data,
                                  uint32_t budget, ConvTilePlan *plan_out) {
  // Only pointwise convs: 1x1 filter, stride 1, no padding, one group.
  if (params.stride_width != 1 || params.stride_height != 1 || params.padding_values.width != 0 ||
      params.padding_values.height != 0) {
    return false;
  }
  if (input_shape.DimensionsCount() != 4 || filter_shape.DimensionsCount() != 4 ||
      output_shape.DimensionsCount() != 4) {
    return false;
  }
  const int in_d  = input_shape.Dims(3);
  const int out_d = output_shape.Dims(3);
  if (filter_shape.Dims(1) != 1 || filter_shape.Dims(2) != 1 || filter_shape.Dims(0) != out_d ||
      filter_shape.Dims(3) != in_d || input_shape.Dims(0) != output_shape.Dims(0) ||
      input_shape.Dims(1) != output_shape.Dims(1) || input_shape.Dims(2) != output_shape.Dims(2)) {
    return false;
  }
  if (ArenaBudget(budget) > kArenaBytes)
    return false;
  const int pixels = input_shape.Dims(0) * input_shape.Dims(1) * input_shape.Dims(2);

  const ConvTilePlan plan = PlanPointwise(pixels, in_d, out_d, kRvv, budget);
  if (plan_out)
    *plan_out = plan;
  if (!plan.tiled)
    return false;

  const int tile            = plan.pixels_per_tile;
  const int chunk           = plan.channel_chunk;
  const int buffers         = plan.double_buffer ? 2 : 1;
  const int tiles           = (pixels + tile - 1) / tile;
  const PointwiseLayout lay = PointwiseLayoutFor(in_d, chunk, tile, buffers, kRvv);

  uint8_t *const base                         = g_arena;
  int8_t *const filter                        = reinterpret_cast<int8_t *>(base + lay.filter);
  [[maybe_unused]] int16_t *const weights16   = reinterpret_cast<int16_t *>(base + lay.weights16);
  int32_t *const bias_raw                     = reinterpret_cast<int32_t *>(base + lay.bias_raw);
  [[maybe_unused]] int32_t *const bias_folded = reinterpret_cast<int32_t *>(base + lay.bias_folded);
  int32_t *const mult                         = reinterpret_cast<int32_t *>(base + lay.multiplier);
  int32_t *const shift_raw                    = reinterpret_cast<int32_t *>(base + lay.shift_raw);
  uint8_t *const shift_left                   = base + lay.shift_left;
  uint8_t *const shift_right                  = base + lay.shift_right;
  [[maybe_unused]] int32_t *const accs        = reinterpret_cast<int32_t *>(base + lay.accs);
  int8_t *const in_tile[2]                    = {reinterpret_cast<int8_t *>(base + lay.in[0]),
                                                 reinterpret_cast<int8_t *>(base + lay.in[1])};
  int8_t *const out_tile[2]                   = {reinterpret_cast<int8_t *>(base + lay.out[0]),
                                                 reinterpret_cast<int8_t *>(base + lay.out[1])};

  const auto pixels_in_tile = [&](int t) { return std::min(tile, pixels - t * tile); };

  bool ok = true;
  for (int n0 = 0; n0 < out_d; n0 += chunk) {
    const int cn = std::min(chunk, out_d - n0);

    // The chunk's filter rows are contiguous in the OHWI filter, so one copy brings them in. The
    // first input tile comes with them.
    g_prologue.Clear();
    ok &= g_prologue.Add(filter_data + static_cast<size_t>(n0) * in_d, filter,
                         static_cast<uint32_t>(cn) * in_d);
    if (bias_data)
      ok &= g_prologue.Add(bias_data + n0, bias_raw, 4u * cn);
    ok &= g_prologue.Add(output_multiplier + n0, mult, 4u * cn);
    ok &= g_prologue.Add(output_shift + n0, shift_raw, 4u * cn);
    ok &= g_prologue.Add(input_data, in_tile[0], static_cast<uint32_t>(pixels_in_tile(0)) * in_d);
    if (!ok || !g_prologue.Run())
      return false;

    PrepareShiftParams(shift_left, shift_right, shift_raw, cn);
#if CORALNPU_RVV_CONV
    Conv1x1Prepare(params, in_d, cn, filter, bias_data ? bias_raw : nullptr, weights16,
                   bias_folded);
#endif

    const auto compute = [&](int buf, int np) {
#if CORALNPU_RVV_CONV
      Conv1x1Run(params, mult, shift_left, shift_right, in_d, cn, np, in_tile[buf], weights16,
                 bias_folded, accs, out_tile[buf]);
#else
      const int32_t in_dims[4]   = {1, 1, np, in_d};
      const int32_t flt_dims[4]  = {cn, 1, 1, in_d};
      const int32_t out_dims[4]  = {1, 1, np, cn};
      const int32_t bias_dims[1] = {cn};
      tflite::reference_integer_ops::ConvPerChannel(
          params, mult, shift_raw, tflite::RuntimeShape(4, in_dims), in_tile[buf],
          tflite::RuntimeShape(4, flt_dims), filter, tflite::RuntimeShape(1, bias_dims),
          bias_data ? bias_raw : nullptr, tflite::RuntimeShape(4, out_dims), out_tile[buf]);
#endif
    };

    // Writes tile t's output (in out_tile[buf]) back: one copy if the tile holds every channel,
    // otherwise one copy per pixel, since the DMA has no strided mode.
    const auto add_output = [&](CopyChain<kMaxChainCopies> &c, int t, int buf) {
      const int np = pixels_in_tile(t);
      int8_t *dst  = output_data + static_cast<size_t>(t) * tile * out_d;
      if (cn == out_d) {
        return c.Add(out_tile[buf], dst, static_cast<uint32_t>(np) * out_d);
      }
      bool added = true;
      for (int p = 0; p < np; ++p) {
        added &= c.Add(out_tile[buf] + static_cast<size_t>(p) * cn,
                       dst + static_cast<size_t>(p) * out_d + n0, static_cast<uint32_t>(cn));
      }
      return added;
    };

    if (buffers == 2) {
      // While tile t computes, the DMA writes tile t-1's output back and brings tile t+1 in.
      for (int t = 0; t < tiles; ++t) {
        const int buf                 = t & 1;
        CopyChain<kMaxChainCopies> &c = g_chains[buf];
        c.Clear();
        if (t > 0)
          ok &= add_output(c, t - 1, buf ^ 1);
        if (t + 1 < tiles) {
          ok &= c.Add(input_data + static_cast<size_t>(t + 1) * tile * in_d, in_tile[buf ^ 1],
                      static_cast<uint32_t>(pixels_in_tile(t + 1)) * in_d);
        }
        if (!ok)
          return false;
        c.Start();
        compute(buf, pixels_in_tile(t));
        if (!c.Wait())
          return false;
      }
      CopyChain<kMaxChainCopies> &last = g_chains[0];
      last.Clear();
      ok &= add_output(last, tiles - 1, (tiles - 1) & 1);
      if (!ok || !last.Run())
        return false;
    } else {
      CopyChain<kMaxChainCopies> &c = g_chains[0];
      for (int t = 0; t < tiles; ++t) {
        c.Clear();
        if (t > 0) {
          ok &= c.Add(input_data + static_cast<size_t>(t) * tile * in_d, in_tile[0],
                      static_cast<uint32_t>(pixels_in_tile(t)) * in_d);
          if (!ok || !c.Run())
            return false;
          c.Clear();
        }
        compute(0, pixels_in_tile(t));
        ok &= add_output(c, t, 0);
        if (!ok || !c.Run())
          return false;
      }
    }
  }
  return true;
}

bool ConvPointwiseTiled(const tflite::ConvParams &params, const int32_t *output_multiplier,
                        const int32_t *output_shift, const tflite::RuntimeShape &input_shape,
                        const int8_t *input_data, const tflite::RuntimeShape &filter_shape,
                        const int8_t *filter_data, const int32_t *bias_data,
                        const tflite::RuntimeShape &output_shape, int8_t *output_data) {
  return ConvPointwiseTiledWithBudget(
      params, output_multiplier, output_shift, input_shape, input_data, filter_shape, filter_data,
      bias_data, output_shape, output_data, CORALNPU_TILE_BUDGET_BYTES, nullptr);
}

}  // namespace coralnpu_v2::opt::litert_micro
