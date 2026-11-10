// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The DMA-tiled paths (#11) against TFLite, byte for byte: pointwise conv, the row-tiled 3x3
// conv and row-tiled depthwise conv, each at several DTCM budgets.
//
// For every output byte the test computes the int32 accumulator itself and requantizes it two
// ways: as TFLite does (requant_ref.h, double rounding, ties away from zero) and as the RVV
// kernels do (ties toward +infinity, issue #23). A tiled byte must equal TFLite's value, or else
// equal the kernels' value where the two differ (a #23 tie). Anything else is a tiling bug.
//
// With the reference kernel inside the conv tiles (built without rvv_conv), the conv paths must
// match TFLite exactly. The depthwise tiles always run the RVV depthwise kernel, so they may show
// #23 ties in either build. The model itself is checked against reference_integer_ops on every
// case. Output offsets other than -128 and small right shifts are used so that ties occur.

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/conv_tiled.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/litert-micro/requant_ref.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

using namespace coralnpu_v2::opt::litert_micro;
namespace rq = coralnpu_v2::opt::litert_micro;

namespace {

uint32_t lcg_state = 20261108;
uint32_t Rand() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return lcg_state;
}
int8_t RandByte() { return static_cast<int8_t>(Rand() >> 24); }

struct Quant {
  int32_t input_offset, output_offset, act_min, act_max;
};
constexpr Quant kQuants[] = {
    {128, -3, -128, 127}, {0, 7, -100, 50},      {-7, 100, -128, 127},
    {128, 0, -128, 127},  {21, -128, -128, 127},
};

enum Kind { kPointwise = 0, kRow3x3 = 1, kDepthwise = 2, kKinds = 3 };

struct Case {
  Kind kind;
  int h, w, in_d, out_d, f, stride;
  bool same;
};
constexpr Case kCases[] = {
    {kPointwise, 8, 8, 16, 24, 1, 1, false},   {kPointwise, 6, 7, 32, 40, 1, 1, false},
    {kPointwise, 10, 10, 64, 96, 1, 1, false}, {kPointwise, 3, 11, 13, 21, 1, 1, false},
    {kRow3x3, 32, 32, 3, 8, 3, 2, true},       {kRow3x3, 21, 17, 3, 13, 3, 2, true},
    {kRow3x3, 16, 16, 8, 16, 3, 1, true},      {kRow3x3, 15, 9, 3, 8, 3, 1, false},
    {kDepthwise, 16, 16, 16, 16, 3, 1, true},  {kDepthwise, 16, 16, 32, 32, 3, 2, true},
    {kDepthwise, 13, 11, 8, 8, 3, 1, true},    {kDepthwise, 14, 14, 24, 24, 3, 2, false},
};
constexpr uint32_t kBudgets[] = {2048, 4096, 8192, 24576};

constexpr int kMaxIn  = 32 * 32 * 64;
constexpr int kMaxOut = 32 * 32 * 96;
constexpr int kMaxFlt = 96 * 3 * 3 * 64;
constexpr int kMaxCh  = 96;

int8_t input_data[kMaxIn] __attribute__((aligned(16)));
int8_t filter_data[kMaxFlt] __attribute__((aligned(16)));
int32_t bias_data[kMaxCh] __attribute__((aligned(16)));
int32_t multiplier[kMaxCh] __attribute__((aligned(16)));
int32_t shift[kMaxCh] __attribute__((aligned(16)));
int32_t accs[kMaxOut] __attribute__((aligned(16)));
int8_t want_tflite[kMaxOut] __attribute__((aligned(16)));
int8_t want_rnu[kMaxOut] __attribute__((aligned(16)));
int8_t out_ref[kMaxOut] __attribute__((aligned(16)));
int8_t out_tiled[kMaxOut] __attribute__((aligned(16)));

// The RVV kernels' requantization: the same multiply, then a right shift that rounds ties toward
// +infinity (vssra with vxrm = rnu, or vsra after adding 1 << (e - 1)).
int32_t RequantRnu(int32_t acc, int32_t mult, int32_t sh, const Quant &q) {
  const int32_t s = sh < rq::kRequantMinShift
                        ? rq::kRequantMinShift
                        : (sh > rq::kRequantMaxShift ? rq::kRequantMaxShift : sh);
  const int left  = s > 0 ? s : 0;
  const int right = s > 0 ? 0 : -s;
  int32_t y       = rq::RequantDoublingHighMul(rq::RequantLeftShift(acc, left), mult);
  if (right > 0)
    y = static_cast<int32_t>((static_cast<int64_t>(y) + (int64_t{1} << (right - 1))) >> right);
  y = static_cast<int32_t>(static_cast<uint32_t>(y) + static_cast<uint32_t>(q.output_offset));
  return y < q.act_min ? q.act_min : (y > q.act_max ? q.act_max : y);
}

}  // namespace

extern "C" {
#define RESULT(name) volatile int32_t name __attribute__((used, retain, aligned(16))) = 0
RESULT(done);
RESULT(rvv_kernel);
RESULT(model_mismatches);  // the requant model against reference_integer_ops
RESULT(possible_ties);     // output bytes where TFLite and the kernels' rounding differ
// Per kind (pointwise, 3x3 row, depthwise): tiled runs, bytes checked, bytes equal to TFLite,
// bytes that are #23 ties, and bytes that are neither.
int32_t tiled_runs[kKinds] __attribute__((used, retain, aligned(16)));
int32_t bytes_checked[kKinds] __attribute__((used, retain, aligned(16)));
int32_t bytes_tie[kKinds] __attribute__((used, retain, aligned(16)));
int32_t bytes_other[kKinds] __attribute__((used, retain, aligned(16)));
RESULT(first_other);  // case index * 16 + budget index of the first bad byte, plus 1
}

int main() {
  rvv_kernel     = ConvTiledUsesRvvKernel() ? 1 : 0;
  int case_index = 0;
  for (const Case &c : kCases) {
    const Quant &q     = kQuants[case_index % (sizeof(kQuants) / sizeof(kQuants[0]))];
    const bool dw      = c.kind == kDepthwise;
    const int out_d    = c.out_d;
    const int out_h    = c.same ? (c.h + c.stride - 1) / c.stride : (c.h - c.f) / c.stride + 1;
    const int out_w    = c.same ? (c.w + c.stride - 1) / c.stride : (c.w - c.f) / c.stride + 1;
    const int pad_h    = c.same && (out_h - 1) * c.stride + c.f - c.h > 0
                             ? ((out_h - 1) * c.stride + c.f - c.h) / 2
                             : 0;
    const int pad_w    = c.same && (out_w - 1) * c.stride + c.f - c.w > 0
                             ? ((out_w - 1) * c.stride + c.f - c.w) / 2
                             : 0;
    const int in_size  = c.h * c.w * c.in_d;
    const int out_size = out_h * out_w * out_d;
    const int flt_size = dw ? c.f * c.f * out_d : out_d * c.f * c.f * c.in_d;

    for (int i = 0; i < in_size; ++i)
      input_data[i] = RandByte();
    for (int i = 0; i < flt_size; ++i)
      filter_data[i] = RandByte();
    for (int i = 0; i < out_d; ++i) {
      bias_data[i] = static_cast<int32_t>(Rand() % 8001) - 4000;
      // Every fourth channel halves exactly, which makes rounding ties common.
      multiplier[i] =
          i % 4 == 0 ? 0x40000000 : 0x40000000 + static_cast<int32_t>(Rand() % 0x3fffffff);
      shift[i] = -static_cast<int32_t>(Rand() % 5) - 1;  // right shifts of 1 to 5
    }

    // The accumulators, then the expected bytes both ways.
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        for (int oc = 0; oc < out_d; ++oc) {
          int32_t acc = bias_data[oc];
          for (int ky = 0; ky < c.f; ++ky) {
            const int iy = oy * c.stride - pad_h + ky;
            for (int kx = 0; kx < c.f; ++kx) {
              const int ix = ox * c.stride - pad_w + kx;
              if (iy < 0 || iy >= c.h || ix < 0 || ix >= c.w)
                continue;
              if (dw) {
                const int32_t x = input_data[(iy * c.w + ix) * c.in_d + oc] + q.input_offset;
                acc += x * filter_data[(ky * c.f + kx) * out_d + oc];
              } else {
                for (int ic = 0; ic < c.in_d; ++ic) {
                  const int32_t x = input_data[(iy * c.w + ix) * c.in_d + ic] + q.input_offset;
                  acc += x * filter_data[((oc * c.f + ky) * c.f + kx) * c.in_d + ic];
                }
              }
            }
          }
          const int o    = (oy * out_w + ox) * out_d + oc;
          accs[o]        = acc;
          want_tflite[o] = static_cast<int8_t>(rq::RequantRef(
              acc, multiplier[oc], shift[oc], q.output_offset, q.act_min, q.act_max));
          want_rnu[o]    = static_cast<int8_t>(RequantRnu(acc, multiplier[oc], shift[oc], q));
          possible_ties += want_tflite[o] != want_rnu[o];
        }
      }
    }

    const int32_t in_dims[4]   = {1, c.h, c.w, c.in_d};
    const int32_t out_dims[4]  = {1, out_h, out_w, out_d};
    const int32_t bias_dims[1] = {out_d};
    const int32_t cflt_dims[4] = {out_d, c.f, c.f, c.in_d};
    const int32_t dflt_dims[4] = {1, c.f, c.f, out_d};
    const tflite::RuntimeShape in_shape(4, in_dims), out_shape(4, out_dims),
        bias_shape(1, bias_dims), flt_shape(4, dw ? dflt_dims : cflt_dims);

    tflite::ConvParams cp{};
    tflite::DepthwiseParams dp{};
    cp.stride_width = cp.stride_height = dp.stride_width = dp.stride_height = c.stride;
    cp.dilation_width_factor = cp.dilation_height_factor = 1;
    dp.dilation_width_factor = dp.dilation_height_factor = 1;
    cp.padding_values.width = dp.padding_values.width = pad_w;
    cp.padding_values.height = dp.padding_values.height = pad_h;
    cp.input_offset = dp.input_offset = q.input_offset;
    cp.output_offset = dp.output_offset = q.output_offset;
    cp.quantized_activation_min = dp.quantized_activation_min = q.act_min;
    cp.quantized_activation_max = dp.quantized_activation_max = q.act_max;
    dp.depth_multiplier                                       = 1;

    // The model against the TFLite reference kernel.
    if (dw) {
      tflite::reference_integer_ops::DepthwiseConvPerChannel(
          dp, multiplier, shift, in_shape, input_data, flt_shape, filter_data, bias_shape,
          bias_data, out_shape, out_ref);
    } else {
      tflite::reference_integer_ops::ConvPerChannel(cp, multiplier, shift, in_shape, input_data,
                                                    flt_shape, filter_data, bias_shape, bias_data,
                                                    out_shape, out_ref);
    }
    for (int o = 0; o < out_size; ++o)
      model_mismatches += out_ref[o] != want_tflite[o];

    int budget_index = -1;
    for (uint32_t budget : kBudgets) {
      ++budget_index;
      memset(out_tiled, 0xA5, sizeof(out_tiled));
      bool tiled = false;
      if (c.kind == kPointwise) {
        tiled = ConvPointwiseTiledWithBudget(cp, multiplier, shift, in_shape, input_data, flt_shape,
                                             filter_data, bias_data, out_shape, out_tiled, budget,
                                             nullptr);
      } else if (c.kind == kRow3x3) {
        tiled =
            ConvRowTiledWithBudget(cp, multiplier, shift, in_shape, input_data, flt_shape,
                                   filter_data, bias_data, out_shape, out_tiled, budget, nullptr);
      } else {
        tiled = DepthwiseRowTiledWithBudget(dp, multiplier, shift, in_shape, input_data, flt_shape,
                                            filter_data, bias_data, out_shape, out_tiled, budget,
                                            nullptr);
      }
      if (!tiled)
        continue;
      ++tiled_runs[c.kind];
      for (int o = 0; o < out_size; ++o) {
        ++bytes_checked[c.kind];
        if (out_tiled[o] == want_tflite[o])
          continue;
        if (out_tiled[o] == want_rnu[o]) {
          ++bytes_tie[c.kind];
          continue;
        }
        ++bytes_other[c.kind];
        if (first_other == 0)
          first_other = case_index * 16 + budget_index + 1;
      }
    }
    ++case_index;
  }
  done = 1;
  return 0;
}
