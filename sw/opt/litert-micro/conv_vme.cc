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

#include "sw/opt/litert-micro/conv_vme.h"

#include <riscv_vector.h>

#include <cstdint>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/litert-micro/vme_util.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace coralnpu_v2::opt::litert_micro::vme {
namespace {

constexpr size_t Align16(size_t bytes) { return (bytes + 15) & ~size_t{15}; }
constexpr int RoundUp(int x, int m) { return (x + m - 1) / m * m; }

int FilterK(const tflite::RuntimeShape &filter_shape) {
  return filter_shape.Dims(1) * filter_shape.Dims(2) * filter_shape.Dims(3);
}

// Writes `count` bytes from src into column `m` of the K-major A block, one byte
// per 16-byte row, starting at row k.
inline void ScatterColumn(int8_t *a_block, int k, int m, const int8_t *src, int count) {
  int8_t *dst = a_block + k * kTileDim + m;
  while (count > 0) {
    const size_t vl = __riscv_vsetvl_e8m8(count);
    __riscv_vsse8_v_i8m8(dst, kTileDim, __riscv_vle8_v_i8m8(src, vl), vl);
    src += vl;
    dst += vl * kTileDim;
    count -= vl;
  }
}

inline void FillColumn(int8_t *a_block, int k, int m, int8_t value, int count) {
  int8_t *dst = a_block + k * kTileDim + m;
  while (count > 0) {
    const size_t vl = __riscv_vsetvl_e8m8(count);
    __riscv_vsse8_v_i8m8(dst, kTileDim, __riscv_vmv_v_x_i8m8(value, vl), vl);
    dst += vl * kTileDim;
    count -= vl;
  }
}

// Accumulates one group of NT output-channel blocks for the 16 pixels in
// a_block and stores the first `rows` tile rows into accs (row stride
// accs_stride int32s).
template <int NT>
void GemmGroup(const int8_t *a_block, const int8_t *b_group, int k_padded, int rows, int32_t *accs,
               int accs_stride) {
  ConfigureTileMoves();
  ZeroAccTiles<NT>();
  ConfigureInt8Matmul();
  const int b_tile_stride = k_padded * kTileDim;
  for (int k = 0; k < k_padded; k += kTileK) {
    MatmulStep<NT>(a_block + k * kTileDim, b_group + k * kTileDim, b_tile_stride);
  }
  ConfigureTileMoves();
  for (int m = 0; m < rows; ++m) {
    int32_t *row = accs + m * accs_stride;
    StoreAccRow<0>(m, row);
    if constexpr (NT > 1)
      StoreAccRow<1>(m, row + kTileDim);
    if constexpr (NT > 2)
      StoreAccRow<2>(m, row + 2 * kTileDim);
    if constexpr (NT > 3)
      StoreAccRow<3>(m, row + 3 * kTileDim);
  }
}

}  // namespace

size_t ConvVmePersistentBytes(const tflite::RuntimeShape &filter_shape) {
  const int n = filter_shape.Dims(0);
  return Align16(RoundUp(n, kTileDim) * RoundUp(FilterK(filter_shape), kTileK)) +
         Align16(n * sizeof(int32_t)) + 2 * Align16(n);
}

size_t ConvVmeScratchBytes(const tflite::RuntimeShape &filter_shape) {
  const int n_padded = RoundUp(filter_shape.Dims(0), kTileDim);
  return Align16(RoundUp(FilterK(filter_shape), kTileK) * kTileDim) +
         kTileDim * n_padded * sizeof(int32_t);
}

ConvVmeWeights ConvVmePrepare(const tflite::ConvParams &params, const int32_t *output_shift,
                              const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
                              const int32_t *bias_data, void *persistent) {
  ConvVmeWeights w;
  w.filter_height  = filter_shape.Dims(1);
  w.filter_width   = filter_shape.Dims(2);
  w.n              = filter_shape.Dims(0);
  w.k              = FilterK(filter_shape);
  w.k_padded       = RoundUp(w.k, kTileK);
  const int blocks = RoundUp(w.n, kTileDim) / kTileDim;

  uint8_t *p      = static_cast<uint8_t *>(persistent);
  w.packed_filter = reinterpret_cast<int8_t *>(p);
  p += Align16(blocks * kTileDim * w.k_padded);
  w.bias = reinterpret_cast<int32_t *>(p);
  p += Align16(w.n * sizeof(int32_t));
  w.shift_left = p;
  p += Align16(w.n);
  w.shift_right = p;

  // B operand: per 16-channel block, K-major rows of 16 channels.
  for (int blk = 0; blk < blocks; ++blk) {
    int8_t *dst = w.packed_filter + blk * w.k_padded * kTileDim;
    for (int k = 0; k < w.k_padded; ++k) {
      for (int j = 0; j < kTileDim; ++j) {
        const int n           = blk * kTileDim + j;
        dst[k * kTileDim + j] = (n < w.n && k < w.k) ? filter_data[n * w.k + k] : 0;
      }
    }
  }

  // The engine multiplies raw int8 inputs, so fold sum_k((x + input_offset) * w)
  // = sum_k(x * w) + input_offset * sum_k(w) into the bias.
  for (int n = 0; n < w.n; ++n) {
    int32_t sum = 0;
    for (int k = 0; k < w.k; ++k)
      sum += filter_data[n * w.k + k];
    w.bias[n] = (bias_data ? bias_data[n] : 0) + params.input_offset * sum;
  }

  PrepareShiftParams(w.shift_left, w.shift_right, output_shift, w.n);
  return w;
}

void ConvPerChannelVme(const tflite::ConvParams &params, const ConvVmeWeights &weights,
                       const int32_t *output_multiplier, const tflite::RuntimeShape &input_shape,
                       const int8_t *input_data, const tflite::RuntimeShape &output_shape,
                       int8_t *output_data, void *scratch) {
  const int batches       = input_shape.Dims(0);
  const int input_height  = input_shape.Dims(1);
  const int input_width   = input_shape.Dims(2);
  const int input_depth   = input_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width  = output_shape.Dims(2);
  const int output_depth  = weights.n;
  const int n_padded      = RoundUp(output_depth, kTileDim);
  const int blocks        = n_padded / kTileDim;
  const int pixels        = output_height * output_width;

  int8_t *a_block = static_cast<int8_t *>(scratch);
  int32_t *accs   = reinterpret_cast<int32_t *>(static_cast<uint8_t *>(scratch) +
                                                Align16(weights.k_padded * kTileDim));
  // Padded taps must contribute zero: (pad + input_offset) * w == 0.
  const int8_t pad_value = static_cast<int8_t>(-params.input_offset);

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t *input = input_data + batch * input_height * input_width * input_depth;
    int8_t *output      = output_data + batch * pixels * output_depth;
    for (int pix0 = 0; pix0 < pixels; pix0 += kTileDim) {
      const int rows = pixels - pix0 < kTileDim ? pixels - pix0 : kTileDim;

      // im2col: column m of a_block holds the receptive field of pixel pix0+m,
      // in (ky, kx, ic) order to match the OHWI filter. Columns >= rows and
      // rows >= k are never read back (B is zero there).
      for (int m = 0; m < rows; ++m) {
        const int out_y = (pix0 + m) / output_width;
        const int out_x = (pix0 + m) % output_width;
        const int in_y0 = out_y * params.stride_height - params.padding_values.height;
        const int in_x0 = out_x * params.stride_width - params.padding_values.width;
        int k           = 0;
        for (int ky = 0; ky < weights.filter_height; ++ky) {
          const int in_y = in_y0 + params.dilation_height_factor * ky;
          for (int kx = 0; kx < weights.filter_width; ++kx) {
            const int in_x = in_x0 + params.dilation_width_factor * kx;
            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
              ScatterColumn(a_block, k, m, input + (in_y * input_width + in_x) * input_depth,
                            input_depth);
            } else {
              FillColumn(a_block, k, m, pad_value, input_depth);
            }
            k += input_depth;
          }
        }
      }

      for (int blk = 0; blk < blocks; blk += kMaxAccTiles) {
        const int8_t *b_group = weights.packed_filter + blk * weights.k_padded * kTileDim;
        int32_t *accs_group   = accs + blk * kTileDim;
        switch (blocks - blk < kMaxAccTiles ? blocks - blk : kMaxAccTiles) {
          case 1:
            GemmGroup<1>(a_block, b_group, weights.k_padded, rows, accs_group, n_padded);
            break;
          case 2:
            GemmGroup<2>(a_block, b_group, weights.k_padded, rows, accs_group, n_padded);
            break;
          case 3:
            GemmGroup<3>(a_block, b_group, weights.k_padded, rows, accs_group, n_padded);
            break;
          default:
            GemmGroup<4>(a_block, b_group, weights.k_padded, rows, accs_group, n_padded);
            break;
        }
      }

      // accs rows are n_padded wide; PostprocessAcc wants output_depth-wide
      // rows. Compact them in place when the widths differ (dest never passes
      // source, so a forward copy is safe), then requantize all rows at once:
      // one call per tile instead of one per pixel.
      if (n_padded != output_depth) {
        for (int m = 1; m < rows; ++m) {
          const int32_t *src = accs + m * n_padded;
          int32_t *dst       = accs + m * output_depth;
          for (int c = 0; c < output_depth;) {
            const size_t vl = __riscv_vsetvl_e32m8(output_depth - c);
            __riscv_vse32_v_i32m8(dst + c, __riscv_vle32_v_i32m8(src + c, vl), vl);
            c += vl;
          }
        }
      }
      PostprocessAcc(accs, weights.bias, weights.shift_left, output_multiplier, weights.shift_right,
                     params.output_offset, params.quantized_activation_min,
                     params.quantized_activation_max, output + pix0 * output_depth, rows,
                     output_depth);
    }
  }
}

}  // namespace coralnpu_v2::opt::litert_micro::vme
