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

// Row-tiled versus untiled spatial conv (#11): 3x3 stride 2 like MobileNet's first layer, other
// strides and filter sizes, SAME and VALID padding, odd heights, batches. Each case runs once
// through the dispatcher (ConvPerChannel, untiled) and then through ConvRowTiledWithBudget at
// several DTCM budgets (different rows per tile, single or double buffering). The tiled output must
// equal the untiled output of the same kernel, byte for byte. Runs on npusim with the memcpy copy
// path. Output offsets other than -128 are used on purpose (#23).

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/conv_tiled.h"
#include "tensorflow/lite/micro/kernels/conv.h"

using namespace coralnpu_v2::opt::litert_micro;

namespace {

constexpr int kMaxIn  = 1 << 14;
constexpr int kMaxOut = 1 << 14;
constexpr int kMaxCh  = 32;
constexpr int kMaxFlt = 32 * 5 * 5 * 8;

uint32_t lcg_state = 777;
uint32_t Rand() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return lcg_state;
}
int8_t RandByte() { return static_cast<int8_t>(Rand() >> 24); }

struct Quant {
  int32_t input_offset, output_offset, act_min, act_max;
};
constexpr Quant kQuants[] = {
    {128, -128, -128, 127},
    {128, -3, -128, 127},
    {0, 7, -100, 50},
    {-7, 100, -128, 127},
};

struct Case {
  int n, h, w, in_d, out_d, f, stride;
  bool same;
  bool bias;
};
constexpr Case kCases[] = {
    {1, 16, 16, 3, 8, 3, 2, true, true},    // layer 0 in miniature
    {1, 17, 13, 3, 8, 3, 2, true, true},    // odd height and width
    {1, 15, 9, 4, 16, 3, 1, true, true},    // stride 1, SAME
    {1, 14, 10, 8, 8, 3, 1, false, false},  // VALID
    {1, 21, 11, 3, 8, 3, 2, false, true},   // stride 2, VALID
    {2, 12, 12, 3, 8, 3, 2, true, true},    // batch of two
    {1, 20, 8, 2, 4, 5, 2, true, true},     // 5x5
    {1, 9, 9, 3, 8, 3, 3, true, true},      // stride 3
    {1, 6, 30, 3, 8, 3, 2, true, false},    // wide
    {1, 3, 7, 3, 8, 3, 2, true, true},      // fewer rows than a tile
    {1, 1, 8, 3, 8, 3, 1, true, true},      // one row
};
constexpr uint32_t kBudgets[] = {1024, 1536, 2048, 3072, 4096, 6144, 12288, 24576};

int8_t input_data[kMaxIn] __attribute__((aligned(16)));
int8_t filter_data[kMaxFlt] __attribute__((aligned(16)));
int32_t bias_data[kMaxCh] __attribute__((aligned(16)));
int32_t multiplier[kMaxCh] __attribute__((aligned(16)));
int32_t shift[kMaxCh] __attribute__((aligned(16)));
int8_t out_ref[kMaxOut] __attribute__((aligned(16)));
int8_t out_tiled[kMaxOut + 256] __attribute__((aligned(16)));
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
RESULT(cases_multi_tile);
RESULT(cases_double_buffer);
RESULT(cases_single_buffer);
RESULT(cases_padded_multi_tile);  // a multi-tile case whose first tile has padding rows above
RESULT(cases_stride2_multi_tile);
RESULT(guard_violations);  // bytes written past the end of the output
RESULT(plan_failures);     // the plan said tiled but the wrapper returned false
RESULT(first_failure);     // case index * 16 + budget index of the first such failure, plus 1
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
    const int out_h = c.same ? (c.h + c.stride - 1) / c.stride : (c.h - c.f) / c.stride + 1;
    const int out_w = c.same ? (c.w + c.stride - 1) / c.stride : (c.w - c.f) / c.stride + 1;
    const int pad_h =
        c.same ? ((out_h - 1) * c.stride + c.f - c.h > 0 ? ((out_h - 1) * c.stride + c.f - c.h) / 2
                                                         : 0)
               : 0;
    const int pad_w =
        c.same ? ((out_w - 1) * c.stride + c.f - c.w > 0 ? ((out_w - 1) * c.stride + c.f - c.w) / 2
                                                         : 0)
               : 0;
    const int in_size  = c.n * c.h * c.w * c.in_d;
    const int out_size = c.n * out_h * out_w * c.out_d;
    const int flt_size = c.out_d * c.f * c.f * c.in_d;

    for (int i = 0; i < in_size; ++i)
      input_data[i] = RandByte();
    for (int i = 0; i < flt_size; ++i)
      filter_data[i] = RandByte();
    for (int i = 0; i < c.out_d; ++i) {
      bias_data[i]  = static_cast<int32_t>(Rand() % 8001) - 4000;
      multiplier[i] = 0x40000000 + static_cast<int32_t>(Rand() % 0x3fffffff);
      shift[i]      = -static_cast<int32_t>(Rand() % 8) - 4;  // right shifts: spread-out outputs
    }

    tflite::ConvParams params;
    params.stride_width             = c.stride;
    params.stride_height            = c.stride;
    params.dilation_width_factor    = 1;
    params.dilation_height_factor   = 1;
    params.padding_values.width     = pad_w;
    params.padding_values.height    = pad_h;
    params.input_offset             = q.input_offset;
    params.weights_offset           = 0;
    params.output_offset            = q.output_offset;
    params.quantized_activation_min = q.act_min;
    params.quantized_activation_max = q.act_max;

    const int32_t in_dims[4]   = {c.n, c.h, c.w, c.in_d};
    const int32_t flt_dims[4]  = {c.out_d, c.f, c.f, c.in_d};
    const int32_t out_dims[4]  = {c.n, out_h, out_w, c.out_d};
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

    int budget_index = -1;
    for (uint32_t budget : kBudgets) {
      ++budget_index;
      ++cases_run;
      // Bytes the tiled path does not write stay 0xA5, so a missed tile shows up as a mismatch.
      memset(out_tiled, 0xA5, sizeof(out_tiled));
      RowPlan plan;
      const bool tiled =
          ConvRowTiledWithBudget(params, multiplier, shift, in_shape, input_data, flt_shape,
                                 filter_data, bias, out_shape, out_tiled, budget, &plan);
      if (!tiled) {
        ++cases_not_tiled;
        if (plan.tiled) {
          if (!plan_failures)
            first_failure = (case_index - 1) * 16 + budget_index + 1;
          ++plan_failures;
        }
        continue;
      }
      ++cases_tiled;
      // Nothing may be written after the last output byte: the area behind it still holds 0xA5.
      for (int i = out_size; i < out_size + 256; ++i) {
        if (out_tiled[i] != static_cast<int8_t>(0xA5)) {
          ++guard_violations;
          break;
        }
      }
      if (plan.tiles > 1) {
        ++cases_multi_tile;
        if (pad_h > 0)
          ++cases_padded_multi_tile;
        if (c.stride == 2)
          ++cases_stride2_multi_tile;
      }
      if (plan.double_buffer)
        ++cases_double_buffer;
      else
        ++cases_single_buffer;

      if (memcmp(out_ref, out_tiled, static_cast<size_t>(out_size)) != 0)
        ++mismatches;

      // Self-test of the comparison: one flipped byte must be detected.
      if (!selftest_detected) {
        out_tiled[out_size - 1] = static_cast<int8_t>(out_tiled[out_size - 1] ^ 1);
        if (memcmp(out_ref, out_tiled, static_cast<size_t>(out_size)) != 0)
          selftest_detected = 1;
      }
    }
  }
  done = 1;
  return 0;
}
