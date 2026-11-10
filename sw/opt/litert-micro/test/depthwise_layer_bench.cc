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

// A depthwise conv layer through the real kernel entry (DepthwiseConvPerChannel) on the SoC
// testharness (#11), timed with mcycle/minstret. Built three ways:
//   local: everything in DTCM (the ceiling)
//   ext:   input, filter, bias, parameters, output and the accumulator scratch in external memory
//          (0x20000000), untiled
//   tiled: the same data in external memory, with the DMA row-tiled path linked in
// Shape: DW_H x DW_W input pixels, DW_C channels (depth multiplier 1), 3x3, DW_STRIDE, SAME. The
// output offset is 5 rather than -128 (#23). Result variables use the same names as
// ext_mem_tile_bench.cc so that test_ext_mem_tile.py reads them.

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/litert-micro/row_tiled_core.h"
#include "sw/utils/utils.h"

using namespace coralnpu_v2::opt::litert_micro;

#ifndef DW_H
#error "define DW_H, DW_W, DW_C, DW_STRIDE"
#endif

#ifdef PLACE_EXT
#define DATA_SECTION ".extdata"
#else
#define DATA_SECTION ".data"
#endif

namespace {

constexpr int kInH  = DW_H;
constexpr int kInW  = DW_W;
constexpr int kC    = DW_C;
constexpr int kS    = DW_STRIDE;
constexpr int kF    = 3;
constexpr int kOutH = (kInH + kS - 1) / kS;
constexpr int kOutW = (kInW + kS - 1) / kS;
constexpr int kPadH = ((kOutH - 1) * kS + kF - kInH > 0) ? ((kOutH - 1) * kS + kF - kInH) / 2 : 0;
constexpr int kPadW = ((kOutW - 1) * kS + kF - kInW > 0) ? ((kOutW - 1) * kS + kF - kInW) / 2 : 0;

int8_t input_data[kInH * kInW * kC] __attribute__((section(DATA_SECTION), aligned(16)));
int8_t filter_data[kF * kF * kC] __attribute__((section(DATA_SECTION), aligned(16)));
int32_t bias_data[kC] __attribute__((section(DATA_SECTION), aligned(16)));
int32_t multiplier[kC] __attribute__((section(DATA_SECTION), aligned(16)));
int32_t shift[kC] __attribute__((section(DATA_SECTION), aligned(16)));
int8_t output_data[kOutH * kOutW * kC] __attribute__((section(DATA_SECTION), aligned(16)));
// The untiled kernel's accumulators: the whole output as int32, as MobileNet's scratch buffer.
int32_t accs[kOutH * kOutW * kC] __attribute__((section(DATA_SECTION), aligned(16)));

// One result per 16-byte beat: the cocotb test host is 32 bits wide and reads only the first word
// of each 128-bit DTCM beat.
#define RESULT_VAR(type, name) type name __attribute__((section(".data"), used, aligned(16))) = 0

uint32_t lcg_state = 12345;
int8_t NextByte() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<int8_t>(lcg_state >> 24);
}

}  // namespace

extern "C" {
RESULT_VAR(uint32_t, tile_cycles_lo);
RESULT_VAR(uint32_t, tile_cycles_hi);
RESULT_VAR(uint32_t, tile_instret_lo);
RESULT_VAR(uint32_t, tile_instret_hi);
RESULT_VAR(int32_t, tile_checksum);
RESULT_VAR(int32_t, tile_done);
#ifdef BENCH_STATS
RESULT_VAR(uint32_t, stat_chunks);
RESULT_VAR(uint32_t, stat_tiles);
RESULT_VAR(uint32_t, stat_prologue);
RESULT_VAR(uint32_t, stat_prepare);
RESULT_VAR(uint32_t, stat_compute);
RESULT_VAR(uint32_t, stat_wait);
RESULT_VAR(uint32_t, stat_blocking_copy);
RESULT_VAR(uint32_t, stat_build);
#endif
}

int main(void) {
  for (int i = 0; i < kInH * kInW * kC; ++i)
    input_data[i] = NextByte();
  for (int i = 0; i < kF * kF * kC; ++i)
    filter_data[i] = NextByte();
  for (int i = 0; i < kC; ++i) {
    bias_data[i]  = (i * 37) - 2000;
    multiplier[i] = 0x40000000 + i * 4096;
    shift[i]      = -8;
  }

  const int32_t input_dims[4]  = {1, kInH, kInW, kC};
  const int32_t filter_dims[4] = {1, kF, kF, kC};
  const int32_t output_dims[4] = {1, kOutH, kOutW, kC};
  const int32_t bias_dims[1]   = {kC};
  const tflite::RuntimeShape input_shape(4, input_dims), filter_shape(4, filter_dims),
      output_shape(4, output_dims), bias_shape(1, bias_dims);

  tflite::DepthwiseParams params;
  params.stride_width             = kS;
  params.stride_height            = kS;
  params.dilation_width_factor    = 1;
  params.dilation_height_factor   = 1;
  params.padding_values.width     = kPadW;
  params.padding_values.height    = kPadH;
  params.depth_multiplier         = 1;
  params.input_offset             = 128;
  params.weights_offset           = 0;
  params.output_offset            = 5;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

#ifdef BENCH_STATS
  TileStats() = ConvTiledStats{};
#endif
  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
  DepthwiseConvPerChannel(params, multiplier, shift, input_shape, input_data, filter_shape,
                          filter_data, bias_shape, bias_data, output_shape, output_data, accs);
  const uint64_t instret = minstret_read() - i0;
  const uint64_t cycles  = mcycle_read() - c0;

#ifdef BENCH_STATS
  const ConvTiledStats stats = TileStats();
  stat_chunks                = stats.chunks;
  stat_tiles                 = stats.tiles;
  stat_prologue              = stats.prologue;
  stat_prepare               = stats.prepare;
  stat_compute               = stats.compute;
  stat_wait                  = stats.wait;
  stat_blocking_copy         = stats.blocking_copy;
  stat_build                 = stats.build;
#endif
  tile_instret_lo = static_cast<uint32_t>(instret);
  tile_instret_hi = static_cast<uint32_t>(instret >> 32);
  tile_cycles_lo  = static_cast<uint32_t>(cycles);
  tile_cycles_hi  = static_cast<uint32_t>(cycles >> 32);

  int32_t sum = 0;
  for (int i = 0; i < kOutH * kOutW * kC; ++i)
    sum = sum * 31 + output_data[i];
  tile_checksum = sum;
  tile_done     = 0x600D;
  return 0;
}
