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

// Tiled versus untiled pointwise conv (#11). For each case the layer runs once through the
// dispatcher (ConvPerChannel, untiled) and then through ConvPointwiseTiledWithBudget at several
// DTCM budgets, which gives different tile sizes, channel chunks and buffering. The tiled output
// must equal the untiled output of the same kernel, byte for byte. That isolates tiling bugs
// (chunk boundaries, tile edges, buffer swaps). It does not check the kernel against TFLite: see
// #23. Runs on npusim with the memcpy copy path. The test binary does not enable conv_tiling, so
// ConvPerChannel is the untiled path.
//
// Output offsets other than -128 and varied activation ranges are used on purpose (#23).

#include "sw/opt/litert-micro/conv_tiled.h"

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "tensorflow/lite/micro/kernels/conv.h"

using namespace coralnpu_v2::opt::litert_micro;

namespace {

constexpr int kMaxPixels = 128;
constexpr int kMaxIn     = 64;
constexpr int kMaxOut    = 96;

uint32_t lcg_state = 4242;
uint32_t Rand() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return lcg_state;
}
int8_t RandByte() { return static_cast<int8_t>(Rand() >> 24); }

struct Quant {
  int32_t input_offset, output_offset, act_min, act_max;
};
constexpr Quant kQuants[] = {
    {128, -128, -128, 127}, {128, -3, -128, 127},  {0, 7, -100, 50},
    {-7, 100, -128, 127},   {21, -128, -128, 127}, {128, 0, 0, 127},
};

struct Case {
  int n, h, w, in_d, out_d;
  bool bias;
};
constexpr Case kCases[] = {
    {1, 1, 1, 16, 16, true},  {1, 1, 3, 8, 24, true},    {1, 3, 3, 13, 21, true},
    {1, 3, 11, 16, 32, true}, {1, 8, 8, 24, 40, false},  {1, 10, 10, 32, 48, true},
    {1, 5, 5, 64, 96, true},  {2, 4, 4, 16, 64, true},   {1, 7, 7, 40, 5, true},
    {1, 6, 7, 64, 80, false}, {1, 12, 10, 16, 16, true},
};
constexpr uint32_t kBudgets[] = {3072, 4096, 6144, 8192, 12288, 24576};

int8_t input_data[kMaxPixels * kMaxIn] __attribute__((aligned(16)));
int8_t filter_data[kMaxOut * kMaxIn] __attribute__((aligned(16)));
int32_t bias_data[kMaxOut] __attribute__((aligned(16)));
int32_t multiplier[kMaxOut] __attribute__((aligned(16)));
int32_t shift[kMaxOut] __attribute__((aligned(16)));
int8_t out_ref[kMaxPixels * kMaxOut] __attribute__((aligned(16)));
int8_t out_tiled[kMaxPixels * kMaxOut] __attribute__((aligned(16)));
int8_t scratch[1 << 16] __attribute__((aligned(16)));

TfLiteStatus MockRequestScratchBuffer(TfLiteContext *, size_t, int *idx) {
  *idx = 0;
  return kTfLiteOk;
}
void *MockGetScratchBuffer(TfLiteContext *, int) { return scratch; }

}  // namespace

extern "C" {
#define RESULT(name) volatile int32_t name __attribute__((used, retain, aligned(16))) = 0
RESULT(done);
RESULT(cases_run);
RESULT(cases_tiled);
RESULT(cases_not_tiled);
RESULT(cases_chunked);
RESULT(cases_multi_tile);
RESULT(cases_double_buffer);
RESULT(cases_single_buffer);
RESULT(cases_chunked_double);
RESULT(cases_chunked_single);
RESULT(mismatches);
RESULT(selftest_detected);
RESULT(rvv_kernel);
}

int main() {
  rvv_kernel     = ConvTiledUsesRvvKernel() ? 1 : 0;
  int case_index = 0;
  for (const Case &c : kCases) {
    const Quant &q = kQuants[case_index % (sizeof(kQuants) / sizeof(kQuants[0]))];
    ++case_index;
    const int pixels = c.n * c.h * c.w;

    for (int i = 0; i < pixels * c.in_d; ++i)
      input_data[i] = RandByte();
    for (int i = 0; i < c.out_d * c.in_d; ++i)
      filter_data[i] = RandByte();
    for (int i = 0; i < c.out_d; ++i) {
      bias_data[i]  = static_cast<int32_t>(Rand() % 8001) - 4000;
      multiplier[i] = 0x40000000 + static_cast<int32_t>(Rand() % 0x3fffffff);
      shift[i]      = -static_cast<int32_t>(Rand() % 11) + 2;  // -8 to 2: mostly right shifts
    }

    tflite::ConvParams params;
    params.stride_width             = 1;
    params.stride_height            = 1;
    params.dilation_width_factor    = 1;
    params.dilation_height_factor   = 1;
    params.padding_values.width     = 0;
    params.padding_values.height    = 0;
    params.input_offset             = q.input_offset;
    params.weights_offset           = 0;
    params.output_offset            = q.output_offset;
    params.quantized_activation_min = q.act_min;
    params.quantized_activation_max = q.act_max;

    const int32_t in_dims[4]   = {c.n, c.h, c.w, c.in_d};
    const int32_t flt_dims[4]  = {c.out_d, 1, 1, c.in_d};
    const int32_t out_dims[4]  = {c.n, c.h, c.w, c.out_d};
    const int32_t bias_dims[1] = {c.out_d};
    const tflite::RuntimeShape in_shape(4, in_dims), flt_shape(4, flt_dims), out_shape(4, out_dims),
        bias_shape(1, bias_dims);
    const int32_t *bias = c.bias ? bias_data : nullptr;

    TfLiteContext context;
    context.GetScratchBuffer            = MockGetScratchBuffer;
    context.RequestScratchBufferInArena = MockRequestScratchBuffer;
    OpDataConvCustom data;
    memset(&data, 0, sizeof(data));
    data.per_channel_output_multiplier = multiplier;
    data.per_channel_output_shift      = shift;
    data.output_activation_min         = q.act_min;
    data.output_activation_max         = q.act_max;

    memset(out_ref, 0x5A, sizeof(out_ref));
    ConvPerChannel(params, data, multiplier, shift, &context, in_shape, input_data, flt_shape,
                   filter_data, bias_shape, bias, out_shape, out_ref);

    for (uint32_t budget : kBudgets) {
      ++cases_run;
      // Bytes the tiled path does not write stay 0xA5, so a missed tile shows up as a mismatch.
      memset(out_tiled, 0xA5, sizeof(out_tiled));
      for (int i = 0; i < pixels * c.out_d; ++i)
        out_tiled[i] = 0xA5;
      ConvTilePlan plan;
      const bool tiled =
          ConvPointwiseTiledWithBudget(params, multiplier, shift, in_shape, input_data, flt_shape,
                                       filter_data, bias, out_shape, out_tiled, budget, &plan);
      if (!tiled) {
        ++cases_not_tiled;
        continue;
      }
      ++cases_tiled;
      if (plan.num_chunks > 1)
        ++cases_chunked;
      if (plan.pixels_per_tile < pixels)
        ++cases_multi_tile;
      if (plan.double_buffer) {
        ++cases_double_buffer;
        if (plan.num_chunks > 1)
          ++cases_chunked_double;
      } else {
        ++cases_single_buffer;
        if (plan.num_chunks > 1)
          ++cases_chunked_single;
      }

      if (memcmp(out_ref, out_tiled, static_cast<size_t>(pixels) * c.out_d) != 0)
        ++mismatches;

      // Self-test of the comparison: one flipped byte must be detected.
      if (!selftest_detected) {
        out_tiled[pixels * c.out_d - 1] = static_cast<int8_t>(out_tiled[pixels * c.out_d - 1] ^ 1);
        if (memcmp(out_ref, out_tiled, static_cast<size_t>(pixels) * c.out_d) != 0) {
          selftest_detected = 1;
        }
      }
    }
  }
  done = 1;
  return 0;
}
