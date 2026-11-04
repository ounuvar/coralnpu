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

// Tile planner for DMA-tiled pointwise (1x1, stride 1, no padding) convolution (#11).
//
// A tile is a group of output pixels (consecutive in memory, since a 1x1 conv has no halo) for a
// chunk of output channels. The planner picks the pixels per tile, the output-channel chunk and
// whether to double buffer, so that every buffer the tiled path needs fits in a DTCM budget.
//
// PointwiseLayout is the single description of those buffers: the planner sizes a plan with it
// and the tiled wrapper carves its DTCM arena with the same offsets, so the two cannot disagree.
//
// Portable: no CoralNPU headers, so it builds for the host (unit tests) and for CoralNPU.

#ifndef SW_OPT_LITERT_MICRO_CONV_TILE_PLAN_H_
#define SW_OPT_LITERT_MICRO_CONV_TILE_PLAN_H_

#include <cstdint>

// DTCM bytes the tiler may use. The default design point is a 32 KB DTCM with 8 KB left for the
// stack, .data and heap. Override with -DCORALNPU_TILE_BUDGET_BYTES=<n> (for example for the
// 1 MB highmem builds).
#ifndef CORALNPU_TILE_BUDGET_BYTES
#define CORALNPU_TILE_BUDGET_BYTES (24 * 1024)
#endif

namespace coralnpu_v2::opt::litert_micro {

// Pixels the RVV pointwise kernel works on at once (kPixels in conv.cc): sizes its accumulators.
inline constexpr uint32_t kPointwiseKernelPixels = 4;
// Output channels come in chunks that are multiples of this (one 16-byte DMA beat of int8).
inline constexpr int kChannelGranule = 16;
// Smallest tile worth the per-tile overhead (descriptor setup, one DMA wait), unless the whole
// layer is smaller.
inline constexpr int kMinTilePixels = 8;
// Larger tiles do not amortize the per-tile overhead any further and only cost DTCM.
inline constexpr int kMaxTilePixels = 64;

constexpr uint32_t RoundUp16(uint32_t v) { return (v + 15u) & ~15u; }

// What the tiled path keeps in DTCM, in bytes, laid out one after the other, 16-byte aligned.
//   filter       raw int8 filter rows of the chunk, [chunk][in_d]
//   weights16    transposed int16 weights, [in_d][chunk] (RVV kernel only)
//   bias_raw     int32 bias of the chunk
//   bias_folded  int32 bias with the input offset folded in (RVV kernel only)
//   multiplier   int32 per-channel output multiplier of the chunk
//   shift_raw    int32 per-channel output shift of the chunk
//   shift_left / shift_right  the uint8 forms the kernel reads
//   accs         int32 accumulators, kPointwiseKernelPixels x chunk (RVV kernel only)
//   in[b]        input tile b: pixels x in_d int8
//   out[b]       output tile b: pixels x chunk int8
struct PointwiseLayout {
  uint32_t filter, weights16, bias_raw, bias_folded, multiplier, shift_raw, shift_left, shift_right,
      accs;
  uint32_t in[2], out[2];
  uint32_t total;
};

constexpr PointwiseLayout PointwiseLayoutFor(int in_d, int chunk, int pixels, int buffers,
                                             bool int16_weights) {
  PointwiseLayout l{};
  uint32_t at = 0;
  auto take   = [&at](uint32_t bytes) {
    const uint32_t offset = at;
    at += RoundUp16(bytes);
    return offset;
  };
  const uint32_t c = static_cast<uint32_t>(chunk);
  const uint32_t d = static_cast<uint32_t>(in_d);
  const uint32_t p = static_cast<uint32_t>(pixels);
  l.filter         = take(c * d);
  l.weights16      = take(int16_weights ? 2 * c * d : 0);
  l.bias_raw       = take(4 * c);
  l.bias_folded    = take(int16_weights ? 4 * c : 0);
  l.multiplier     = take(4 * c);
  l.shift_raw      = take(4 * c);
  l.shift_left     = take(c);
  l.shift_right    = take(c);
  l.accs           = take(int16_weights ? 4 * kPointwiseKernelPixels * c : 0);
  for (int b = 0; b < 2; ++b) {
    const bool used = b < buffers;
    l.in[b]         = take(used ? p * d : 0);
    l.out[b]        = take(used ? p * c : 0);
  }
  l.total = at;
  return l;
}

struct ConvTilePlan {
  bool tiled          = false;  // false: use the untiled path
  int pixels_per_tile = 0;      // output pixels per tile (the last tile may be smaller)
  int channel_chunk   = 0;      // output channels per chunk (the last chunk may be smaller)
  int num_chunks      = 0;
  bool double_buffer  = false;
  uint32_t footprint  = 0;  // DTCM bytes the plan needs
  uint64_t dma_bytes  = 0;  // bytes the DMA moves for the whole layer
};

// Plans a pointwise conv with `pixels` output pixels (batch x height x width), `in_d` input and
// `out_d` output channels. `int16_weights` selects the RVV kernel's buffers (the transposed int16
// weights are twice the filter's size); false is for the scalar reference kernel.
//
// Prefers the largest channel chunk (fewer passes over the input), starting with all channels and
// then splitting into 2, 3, ... balanced chunks. For a chunk it prefers double buffering, and falls
// back to single buffering when double buffering would leave fewer than kMinTilePixels pixels per
// tile. If no chunk of at least kChannelGranule channels fits, the plan is not tiled. Bytes the DMA
// moves for the whole layer: the filter, bias and per-channel parameters once, the input once per
// channel chunk, the output once.
constexpr uint64_t PointwiseDmaBytes(int pixels, int in_d, int out_d, int num_chunks) {
  return static_cast<uint64_t>(out_d) * in_d + 12ull * out_d +
         static_cast<uint64_t>(num_chunks) * pixels * in_d + static_cast<uint64_t>(pixels) * out_d;
}

constexpr ConvTilePlan PlanPointwise(int pixels, int in_d, int out_d, bool int16_weights,
                                     uint32_t budget = CORALNPU_TILE_BUDGET_BYTES) {
  ConvTilePlan none{};
  if (pixels <= 0 || in_d <= 0 || out_d <= 0)
    return none;
  const int min_tile = pixels < kMinTilePixels ? pixels : kMinTilePixels;
  const int max_tile = pixels < kMaxTilePixels ? pixels : kMaxTilePixels;
  // Try one chunk (all channels), then 2, 3, ... balanced chunks, each rounded up to a granule.
  int previous = 0;
  for (int parts = 1;; ++parts) {
    int chunk = out_d;
    if (parts > 1) {
      const int even = (out_d + parts - 1) / parts;
      chunk          = ((even + kChannelGranule - 1) / kChannelGranule) * kChannelGranule;
    }
    if (chunk >= out_d && parts > 1)
      return none;  // rounding left nothing smaller to try
    if (chunk == previous)
      continue;
    previous = chunk;
    for (int buffers = 2; buffers >= 1; --buffers) {
      // The footprint grows with the tile, so the largest tile that fits is found by search.
      int best = 0;
      for (int t = max_tile; t >= min_tile; --t) {
        if (PointwiseLayoutFor(in_d, chunk, t, buffers, int16_weights).total <= budget) {
          best = t;
          break;
        }
      }
      if (best > 0) {
        ConvTilePlan plan;
        plan.tiled           = true;
        plan.pixels_per_tile = best;
        plan.channel_chunk   = chunk;
        plan.num_chunks      = (out_d + chunk - 1) / chunk;
        plan.double_buffer   = buffers == 2;
        plan.footprint       = PointwiseLayoutFor(in_d, chunk, best, buffers, int16_weights).total;
        plan.dma_bytes       = PointwiseDmaBytes(pixels, in_d, out_d, plan.num_chunks);
        return plan;
      }
    }
    if (chunk <= kChannelGranule)
      return none;  // cannot go below one granule
  }
}

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_CONV_TILE_PLAN_H_
