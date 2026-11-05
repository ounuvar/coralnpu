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
// whether to double buffer, so that every buffer the tiled path needs, including the DMA copy
// chains, fits in a DTCM budget.
//
// PointwiseLayout is the single description of those buffers: the planner sizes a plan with it
// and the tiled wrapper carves its DTCM arena with the same offsets, so the two cannot disagree.
//
// Portable: no CoralNPU headers, so it builds for the host (unit tests) and for CoralNPU.

#ifndef SW_OPT_LITERT_MICRO_CONV_TILE_PLAN_H_
#define SW_OPT_LITERT_MICRO_CONV_TILE_PLAN_H_

#include <cstdint>

// DTCM bytes the tiler may use, all of it: buffers and DMA descriptors. The default design point is
// a 32 KB DTCM with 8 KB left for the stack, .data and heap. Override with
// -DCORALNPU_TILE_BUDGET_BYTES=<n> (for example for the 1 MB highmem builds).
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
inline constexpr int kMaxTilePixels = 32;

// Prefetch (CORALNPU_TILING_PREFETCH): while a chunk's last tile computes, the DMA brings in the
// next chunk's filter rows, bias, multipliers and shifts (a second set of raw buffers) and, when
// double buffering, its first input tile (into the spare input buffer), so the next chunk does not
// start with a blocking copy. The copies join the last tile's chain.
#ifdef CORALNPU_TILING_PREFETCH
inline constexpr bool kPrefetchDefault = true;
#else
inline constexpr bool kPrefetchDefault = false;
#endif

// With prefetch, CORALNPU_TILING_SPREAD spreads the next chunk's filter rows over all the tiles of
// the current chunk (one piece per tile) instead of copying them during the last tile only, so no
// single tile's chain is much longer than its compute.
#ifdef CORALNPU_TILING_SPREAD
inline constexpr bool kSpreadDefault = true;
#else
inline constexpr bool kSpreadDefault = false;
#endif

// CORALNPU_TILING_MERGE chooses the tile size by a cost model (see TileSplit) instead of filling
// the buffers: tiles that are multiples of the kernel's 4-pixel block, and leftover pixels merged
// into the last tile so that no tile is tiny. Without it, tiles are as large as the buffers allow
// and the last tile takes what is left, which can be a pixel or two.
#ifdef CORALNPU_TILING_MERGE
inline constexpr bool kMergeDefault = true;
#else
inline constexpr bool kMergeDefault = false;
#endif

constexpr uint32_t RoundUp16(uint32_t v) { return (v + 15u) & ~15u; }

// DMA descriptors are 32 bytes. The prologue chain holds the filter, bias, multiplier and shift of
// a chunk and its first input tile.
inline constexpr uint32_t kDescriptorBytes = 32;
inline constexpr int kPrologueCopies       = 5;

// Copies in the longest tile chain: one output copy per pixel when the output channels are chunked
// (the DMA has no strided mode, so each pixel's slice is its own copy), otherwise one; the next
// tile's input; and with prefetch up to four more (in the last tile: the bias, multiplier and shift
// of the next chunk, a piece of its filter and its first input tile).
constexpr int ChainCopies(int pixels, bool chunked, bool prefetch) {
  return (chunked ? pixels : 1) + (prefetch ? 5 : 1);
}

// What the tiled path keeps in DTCM, in bytes, laid out one after the other. The copy chains come
// first so that their descriptors are 32-byte aligned; everything else is 16-byte aligned.
//   prologue_chain, chain[2]  DMA descriptor storage of the three copy chains
//   filter       raw int8 filter rows of the chunk, [chunk][in_d]
//   weights16    transposed int16 weights, [in_d][chunk] (RVV kernel only)
//   bias_raw     int32 bias of the chunk
//   bias_folded  int32 bias with the input offset folded in (RVV kernel only)
//   multiplier   int32 per-channel output multiplier of the chunk
//   shift_raw    int32 per-channel output shift of the chunk
//   shift_left / shift_right  the uint8 forms the kernel reads
//   accs         int32 accumulators, kPointwiseKernelPixels x chunk (RVV kernel only)
//   filter2, bias_raw2, multiplier2, shift_raw2  a second set of the raw buffers for the next
//                chunk (prefetch only)
//   in[b]        input tile b: pixels x in_d int8
//   out[b]       output tile b: pixels x chunk int8
struct PointwiseLayout {
  uint32_t prologue_chain, chain[2];
  int chain_capacity;  // copies each tile chain can hold
  uint32_t filter, weights16, bias_raw, bias_folded, multiplier, shift_raw, shift_left, shift_right,
      accs, filter2, bias_raw2, multiplier2, shift_raw2;
  uint32_t in[2], out[2];
  uint32_t total;
};

// `pixels` is the largest tile (the buffers hold that many); `chunked` is whether the output
// channels are split into several chunks.
constexpr PointwiseLayout PointwiseLayoutFor(int in_d, int chunk, int pixels, int buffers,
                                             bool int16_weights, bool prefetch, bool chunked) {
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
  l.chain_capacity = ChainCopies(pixels, chunked, prefetch);
  l.prologue_chain = take(kDescriptorBytes * kPrologueCopies);
  for (int b = 0; b < 2; ++b)
    l.chain[b] = take(kDescriptorBytes * static_cast<uint32_t>(l.chain_capacity));
  l.filter      = take(c * d);
  l.weights16   = take(int16_weights ? 2 * c * d : 0);
  l.bias_raw    = take(4 * c);
  l.bias_folded = take(int16_weights ? 4 * c : 0);
  l.multiplier  = take(4 * c);
  l.shift_raw   = take(4 * c);
  l.shift_left  = take(c);
  l.shift_right = take(c);
  l.accs        = take(int16_weights ? 4 * kPointwiseKernelPixels * c : 0);
  l.filter2     = take(prefetch ? c * d : 0);
  l.bias_raw2   = take(prefetch ? 4 * c : 0);
  l.multiplier2 = take(prefetch ? 4 * c : 0);
  l.shift_raw2  = take(prefetch ? 4 * c : 0);
  for (int b = 0; b < 2; ++b) {
    const bool used = b < buffers;
    l.in[b]         = take(used ? p * d : 0);
    l.out[b]        = take(used ? p * c : 0);
  }
  l.total = at;
  return l;
}

struct ConvTilePlan {
  bool tiled           = false;  // false: use the untiled path
  int pixels_per_tile  = 0;      // pixels in every tile but the last
  int tiles            = 0;
  int last_tile_pixels = 0;  // pixels in the last tile (the others have pixels_per_tile)
  int buffer_pixels    = 0;  // the buffers hold this many pixels: the largest tile
  int channel_chunk    = 0;  // output channels per chunk (the last chunk may be smaller)
  int num_chunks       = 0;
  bool double_buffer   = false;
  uint32_t footprint   = 0;  // DTCM bytes the plan needs
  uint64_t dma_bytes   = 0;  // bytes the DMA moves for the whole layer
};

// Bytes the DMA moves for the whole layer: the filter, bias and per-channel parameters once, the
// input once per channel chunk, the output once.
constexpr uint64_t PointwiseDmaBytes(int pixels, int in_d, int out_d, int num_chunks) {
  return static_cast<uint64_t>(out_d) * in_d + 12ull * out_d +
         static_cast<uint64_t>(num_chunks) * pixels * in_d + static_cast<uint64_t>(pixels) * out_d;
}

// How a layer's pixels are split into tiles: `tiles` tiles of `tile` pixels except the last, which
// has `last` pixels (merged: more than `tile`).
struct TileSplit {
  int tile, tiles, last;
};

// Cost model for the tile choice, in cycles, from the measurements on #11 (RVV kernel, HBM latency
// model; scaled by chunk x in_d, which the measured layers kept at 4096): a block of 4 pixels
// costs about 7265 cycles and each leftover pixel on the slower path about 2720; each tile costs
// about 1000 cycles of overhead (building chains, starting and waiting); and while a tile computes,
// the DMA is copying the next tile's input at one 16-byte beat per 82 cycles, so a tile that
// computes for less than that copy makes the core wait for the difference.
inline constexpr uint64_t kBlockCycles  = 7265;
inline constexpr uint64_t kSlowCycles   = 2720;
inline constexpr uint64_t kTileOverhead = 1000;
inline constexpr uint64_t kBeatCycles   = 82;

constexpr uint64_t TileComputeCycles(int pixels, int chunk, int in_d) {
  const uint64_t blocks = static_cast<uint64_t>(pixels) / kPointwiseKernelPixels;
  const uint64_t slow   = static_cast<uint64_t>(pixels) % kPointwiseKernelPixels;
  return (blocks * kBlockCycles + slow * kSlowCycles) * static_cast<uint64_t>(chunk) *
         static_cast<uint64_t>(in_d) / 4096;
}

constexpr uint64_t TileInputCopyCycles(int pixels, int in_d) {
  return (static_cast<uint64_t>(pixels) * static_cast<uint64_t>(in_d) + 15) / 16 * kBeatCycles;
}

// Estimated cycles for one chunk with the given split.
constexpr uint64_t SplitCost(TileSplit s, int chunk, int in_d) {
  uint64_t cost = 0;
  for (int t = 0; t < s.tiles; ++t) {
    const int np           = t == s.tiles - 1 ? s.last : s.tile;
    const int next         = t + 1 < s.tiles ? (t + 1 == s.tiles - 1 ? s.last : s.tile) : s.tile;
    const uint64_t compute = TileComputeCycles(np, chunk, in_d);
    const uint64_t copy    = TileInputCopyCycles(next, in_d);
    cost += compute + (copy > compute ? copy - compute : 0) + kTileOverhead;
  }
  return cost;
}

// The split into tiles whose buffers hold at most `max_tile` pixels. Without `merge`: tiles of
// max_tile pixels and a last tile with the rest. With it: the candidate with the lowest cost, over
// every tile size from max_tile down to the minimum, each as leftover pixels in a tile of their own
// (when they fit no other way) or merged into the last tile (when that tile still fits the
// buffers). Sizes above the minimum must be multiples of the kernel's 4-pixel block.
constexpr TileSplit SplitPixels(int pixels, int max_tile, int chunk, int in_d, bool merge) {
  if (pixels <= max_tile)
    return {pixels, 1, pixels};
  TileSplit best{max_tile, (pixels + max_tile - 1) / max_tile,
                 pixels - ((pixels + max_tile - 1) / max_tile - 1) * max_tile};
  if (!merge)
    return best;
  uint64_t best_cost = SplitCost(best, chunk, in_d);
  const int min_tile = pixels < kMinTilePixels ? pixels : kMinTilePixels;
  for (int s = max_tile; s >= min_tile; --s) {
    if (s >= kMinTilePixels && s % static_cast<int>(kPointwiseKernelPixels) != 0)
      continue;
    const int n = pixels / s;
    const int r = pixels % s;
    if (n == 0)
      continue;
    TileSplit candidates[2] = {};
    int count               = 0;
    if (r == 0)
      candidates[count++] = {s, n, s};
    else {
      candidates[count++] = {s, n + 1, r};
      if (s + r <= max_tile)
        candidates[count++] = {s, n, s + r};
    }
    for (int i = 0; i < count; ++i) {
      const uint64_t cost = SplitCost(candidates[i], chunk, in_d);
      if (cost < best_cost) {
        best_cost = cost;
        best      = candidates[i];
      }
    }
  }
  return best;
}

// Plans a pointwise conv with `pixels` output pixels (batch x height x width), `in_d` input and
// `out_d` output channels. `budget` is the DTCM for buffers and descriptors. `int16_weights`
// selects the RVV kernel's buffers (the transposed int16 weights are twice the filter's size);
// false is for the scalar reference kernel.
//
// Prefers the largest channel chunk (fewer passes over the input), starting with all channels and
// then splitting into 2, 3, ... balanced chunks. For a chunk it prefers double buffering, and falls
// back to single buffering when double buffering would leave fewer than kMinTilePixels pixels per
// tile. If no chunk of at least kChannelGranule channels fits, the plan is not tiled.
constexpr ConvTilePlan PlanPointwise(int pixels, int in_d, int out_d, bool int16_weights,
                                     uint32_t budget = CORALNPU_TILE_BUDGET_BYTES,
                                     bool prefetch = kPrefetchDefault, bool merge = kMergeDefault) {
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
    previous           = chunk;
    const bool chunked = chunk < out_d;
    for (int buffers = 2; buffers >= 1; --buffers) {
      // The footprint grows with the tile, so the largest tile that fits is found by search.
      int best = 0;
      for (int t = max_tile; t >= min_tile; --t) {
        if (PointwiseLayoutFor(in_d, chunk, t, buffers, int16_weights, prefetch, chunked).total <=
            budget) {
          best = t;
          break;
        }
      }
      if (best > 0) {
        const TileSplit split = SplitPixels(pixels, best, chunk, in_d, merge);
        // The buffers need to hold only the largest tile of the split.
        const int largest = split.last > split.tile ? split.last : split.tile;
        ConvTilePlan plan;
        plan.tiled            = true;
        plan.pixels_per_tile  = split.tile;
        plan.tiles            = split.tiles;
        plan.last_tile_pixels = split.last;
        plan.buffer_pixels    = largest;
        plan.channel_chunk    = chunk;
        plan.num_chunks       = (out_d + chunk - 1) / chunk;
        plan.double_buffer    = buffers == 2;
        plan.footprint =
            PointwiseLayoutFor(in_d, chunk, largest, buffers, int16_weights, prefetch, chunked)
                .total;
        plan.dma_bytes = PointwiseDmaBytes(pixels, in_d, out_d, plan.num_chunks);
        return plan;
      }
    }
    if (chunk <= kChannelGranule)
      return none;  // cannot go below one granule
  }
}

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_CONV_TILE_PLAN_H_
