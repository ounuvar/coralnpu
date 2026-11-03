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

// Tile-sized conv benchmark for the DMA tiling study (#11).
//
// Runs one tile of MobileNet layer 17 (a 1x1 conv, 128 -> 128 channels, one output row of 14
// pixels, 229,376 MACs) through the same ConvPerChannel dispatcher the model uses, timed with
// mcycle/minstret. Where the tile's data lives is chosen at build time:
//   default             input, filter, bias and output all in DTCM (no external memory traffic)
//   -DTILE_IN_EXTMEM    all four in external memory (what an untiled kernel sees)
// Comparing the two on a simulation with external-memory latency shows what the external
// accesses cost the kernel, which is the most DMA tiling can recover.
//
// Data is generated here (fixed LCG) so no test-side plumbing is needed; the output checksum lets
// a run be compared against another placement.

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/micro/kernels/conv.h"

using namespace coralnpu_v2::opt::litert_micro;

#ifdef TILE_IN_EXTMEM
#define TILE_SECTION ".extdata"
#else
#define TILE_SECTION ".data"
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 14
#endif
constexpr int kWidth    = TILE_WIDTH;
constexpr int kInDepth  = 128;
constexpr int kOutDepth = 128;

extern "C" {
int8_t tile_input[kWidth * kInDepth] __attribute__((section(TILE_SECTION), aligned(16)));
int8_t tile_filter[kOutDepth * kInDepth] __attribute__((section(TILE_SECTION), aligned(16)));
int32_t tile_bias[kOutDepth] __attribute__((section(TILE_SECTION), aligned(16)));
int8_t tile_output[kWidth * kOutDepth] __attribute__((section(TILE_SECTION), aligned(16)));

// Always in DTCM: scratch and results the host reads back.
int32_t per_channel_multiplier[kOutDepth] __attribute__((section(".data")));
int32_t per_channel_shift[kOutDepth] __attribute__((section(".data")));
int32_t accs_buf[64] __attribute__((section(".data"), aligned(16)));

// One result per 16-byte beat: the cocotb test host is 32 bits wide and reads only the first word
// of each 128-bit DTCM beat, so every result sits at the start of its own beat.
#define RESULT_VAR(type, name) type name __attribute__((section(".data"), used, aligned(16))) = 0
RESULT_VAR(uint32_t, tile_cycles_lo);
RESULT_VAR(uint32_t, tile_cycles_hi);
RESULT_VAR(uint32_t, tile_instret_lo);
RESULT_VAR(uint32_t, tile_instret_hi);
RESULT_VAR(int32_t, tile_checksum);
RESULT_VAR(int32_t, tile_done);
}

static TfLiteStatus MockRequestScratchBuffer(TfLiteContext *, size_t, int *buffer_idx) {
  *buffer_idx = 0;
  return kTfLiteOk;
}
static void *MockGetScratchBuffer(TfLiteContext *, int) { return accs_buf; }

static uint32_t lcg_state = 12345;
static int8_t NextByte() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<int8_t>(lcg_state >> 24);
}

int main(void) {
  for (int i = 0; i < kWidth * kInDepth; ++i)
    tile_input[i] = NextByte();
  for (int i = 0; i < kOutDepth * kInDepth; ++i)
    tile_filter[i] = NextByte();
  for (int i = 0; i < kOutDepth; ++i) {
    tile_bias[i]              = (i * 37) - 2000;
    per_channel_multiplier[i] = 0x40000000 + i * 4096;
    per_channel_shift[i]      = -8;
  }

  const int32_t input_dims[4]  = {1, 1, kWidth, kInDepth};
  const int32_t filter_dims[4] = {kOutDepth, 1, 1, kInDepth};
  const int32_t output_dims[4] = {1, 1, kWidth, kOutDepth};
  const int32_t bias_dims[1]   = {kOutDepth};
  const tflite::RuntimeShape input_shape(4, input_dims);
  const tflite::RuntimeShape filter_shape(4, filter_dims);
  const tflite::RuntimeShape output_shape(4, output_dims);
  const tflite::RuntimeShape bias_shape(1, bias_dims);

  tflite::ConvParams params;
  params.stride_width             = 1;
  params.stride_height            = 1;
  params.dilation_width_factor    = 1;
  params.dilation_height_factor   = 1;
  params.padding_values.width     = 0;
  params.padding_values.height    = 0;
  params.input_offset             = 128;
  params.weights_offset           = 0;
  params.output_offset            = -128;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  TfLiteContext context;
  context.GetScratchBuffer            = MockGetScratchBuffer;
  context.RequestScratchBufferInArena = MockRequestScratchBuffer;
  OpDataConvCustom data;
  memset(&data, 0, sizeof(data));
  data.per_channel_output_multiplier = per_channel_multiplier;
  data.per_channel_output_shift      = per_channel_shift;
  data.output_activation_min         = -128;
  data.output_activation_max         = 127;

  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
  ConvPerChannel(params, data, per_channel_multiplier, per_channel_shift, &context, input_shape,
                 tile_input, filter_shape, tile_filter, bias_shape, tile_bias, output_shape,
                 tile_output);
  const uint64_t instret = minstret_read() - i0;
  const uint64_t cycles  = mcycle_read() - c0;
  tile_instret_lo        = static_cast<uint32_t>(instret);
  tile_instret_hi        = static_cast<uint32_t>(instret >> 32);
  tile_cycles_lo         = static_cast<uint32_t>(cycles);
  tile_cycles_hi         = static_cast<uint32_t>(cycles >> 32);

  int32_t sum = 0;
  for (int i = 0; i < kWidth * kOutDepth; ++i)
    sum = sum * 31 + tile_output[i];
  tile_checksum = sum;
  tile_done     = 0x600D;
  return 0;
}
