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

// A pointwise conv layer through the real dispatcher (ConvPerChannel) on the SoC testharness (#11),
// timed with mcycle/minstret. The same source is built three ways:
//   local: the RVV kernel, everything in DTCM (the ceiling)
//   ext:   the RVV kernel, input, filter, bias, parameters and output in external memory
//   (0x20000000) tiled: the same data in external memory, with the DMA-tiled path linked in
// Shape: LAYER_PIXELS output pixels, IN_D input and OUT_D output channels. The output offset is 5
// rather than -128 (#23). Result variables use the same names as ext_mem_tile_bench.cc so that
// test_ext_mem_tile.py reads them.

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/micro/kernels/conv.h"

using namespace coralnpu_v2::opt::litert_micro;

#ifndef LAYER_PIXELS
#error "define LAYER_PIXELS"
#endif
#ifndef IN_D
#error "define IN_D"
#endif
#ifndef OUT_D
#error "define OUT_D"
#endif

#ifdef PLACE_EXT
#define DATA_SECTION ".extdata"
#else
#define DATA_SECTION ".data"
#endif

namespace {

constexpr int kPixels = LAYER_PIXELS;
constexpr int kInD    = IN_D;
constexpr int kOutD   = OUT_D;

int8_t input_data[kPixels * kInD] __attribute__((section(DATA_SECTION), aligned(16)));
int8_t filter_data[kOutD * kInD] __attribute__((section(DATA_SECTION), aligned(16)));
int32_t bias_data[kOutD] __attribute__((section(DATA_SECTION), aligned(16)));
int32_t multiplier[kOutD] __attribute__((section(DATA_SECTION), aligned(16)));
int32_t shift[kOutD] __attribute__((section(DATA_SECTION), aligned(16)));
int8_t output_data[kPixels * kOutD] __attribute__((section(DATA_SECTION), aligned(16)));

// One result per 16-byte beat: the cocotb test host is 32 bits wide and reads only the first word
// of each 128-bit DTCM beat.
#define RESULT_VAR(type, name) type name __attribute__((section(".data"), used, aligned(16))) = 0

uint32_t lcg_state = 12345;
int8_t NextByte() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<int8_t>(lcg_state >> 24);
}

TfLiteStatus MockRequestScratchBuffer(TfLiteContext *, size_t, int *idx) {
  *idx = 0;
  return kTfLiteOk;
}
void *MockGetScratchBuffer(TfLiteContext *, int) { return nullptr; }

}  // namespace

extern "C" {
RESULT_VAR(uint32_t, tile_cycles_lo);
RESULT_VAR(uint32_t, tile_cycles_hi);
RESULT_VAR(uint32_t, tile_instret_lo);
RESULT_VAR(uint32_t, tile_instret_hi);
RESULT_VAR(int32_t, tile_checksum);
RESULT_VAR(int32_t, tile_done);
}

int main(void) {
  for (int i = 0; i < kPixels * kInD; ++i)
    input_data[i] = NextByte();
  for (int i = 0; i < kOutD * kInD; ++i)
    filter_data[i] = NextByte();
  for (int i = 0; i < kOutD; ++i) {
    bias_data[i]  = (i * 37) - 2000;
    multiplier[i] = 0x40000000 + i * 4096;
    shift[i]      = -8;
  }

  const int32_t input_dims[4]  = {1, 1, kPixels, kInD};
  const int32_t filter_dims[4] = {kOutD, 1, 1, kInD};
  const int32_t output_dims[4] = {1, 1, kPixels, kOutD};
  const int32_t bias_dims[1]   = {kOutD};
  const tflite::RuntimeShape input_shape(4, input_dims), filter_shape(4, filter_dims),
      output_shape(4, output_dims), bias_shape(1, bias_dims);

  tflite::ConvParams params;
  params.stride_width             = 1;
  params.stride_height            = 1;
  params.dilation_width_factor    = 1;
  params.dilation_height_factor   = 1;
  params.padding_values.width     = 0;
  params.padding_values.height    = 0;
  params.input_offset             = 128;
  params.weights_offset           = 0;
  params.output_offset            = 5;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  TfLiteContext context;
  context.GetScratchBuffer            = MockGetScratchBuffer;
  context.RequestScratchBufferInArena = MockRequestScratchBuffer;
  OpDataConvCustom data;
  memset(&data, 0, sizeof(data));
  data.per_channel_output_multiplier = multiplier;
  data.per_channel_output_shift      = shift;
  data.output_activation_min         = -128;
  data.output_activation_max         = 127;

  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
  ConvPerChannel(params, data, multiplier, shift, &context, input_shape, input_data, filter_shape,
                 filter_data, bias_shape, bias_data, output_shape, output_data);
  const uint64_t instret = minstret_read() - i0;
  const uint64_t cycles  = mcycle_read() - c0;

  tile_instret_lo = static_cast<uint32_t>(instret);
  tile_instret_hi = static_cast<uint32_t>(instret >> 32);
  tile_cycles_lo  = static_cast<uint32_t>(cycles);
  tile_cycles_hi  = static_cast<uint32_t>(cycles >> 32);

  int32_t sum = 0;
  for (int i = 0; i < kPixels * kOutD; ++i)
    sum = sum * 31 + output_data[i];
  tile_checksum = sum;
  tile_done     = 0x600D;
  return 0;
}
