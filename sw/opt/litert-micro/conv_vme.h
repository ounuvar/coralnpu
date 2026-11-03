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

// INT8 per-channel convolution on the matrix engine (idea 5, #12): im2col into
// a K-major A operand, GEMM on 16x16 int32 tiles, then the same requantization
// as the RVV kernels (PostprocessAcc). Bit-exact with
// tflite::reference_integer_ops::ConvPerChannel wherever the RVV kernels are.

#ifndef SW_OPT_LITERT_MICRO_CONV_VME_H_
#define SW_OPT_LITERT_MICRO_CONV_VME_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro::vme {

// Per-layer data computed once (TFLite Prepare). Not timed in benchmarks, like
// the weight repacking of the RVV kernels.
struct ConvVmeWeights {
  int8_t *packed_filter;  // [N/16 blocks][Kp][16], zero padded
  int32_t *bias;          // bias + input_offset * sum_k(filter), per channel
  uint8_t *shift_left;    // PrepareShiftParams output, per channel
  uint8_t *shift_right;
  int filter_height;
  int filter_width;
  int k;         // filter_height * filter_width * input_depth
  int k_padded;  // k rounded up to a multiple of 4
  int n;         // output channels
};

// Bytes of persistent memory ConvVmePrepare needs.
size_t ConvVmePersistentBytes(const tflite::RuntimeShape &filter_shape);

// Bytes of scratch memory ConvPerChannelVme needs.
size_t ConvVmeScratchBytes(const tflite::RuntimeShape &filter_shape);

// Packs the filter and per-channel parameters into `persistent` (16-byte
// aligned, ConvVmePersistentBytes long).
ConvVmeWeights ConvVmePrepare(const tflite::ConvParams &params, const int32_t *output_shift,
                              const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
                              const int32_t *bias_data, void *persistent);

// Same contract as reference_integer_ops::ConvPerChannel. Requires int8 with
// zero filter offset and input_offset in [-127, 128]. `scratch` is 16-byte
// aligned and ConvVmeScratchBytes long.
void ConvPerChannelVme(const tflite::ConvParams &params, const ConvVmeWeights &weights,
                       const int32_t *output_multiplier, const tflite::RuntimeShape &input_shape,
                       const int8_t *input_data, const tflite::RuntimeShape &output_shape,
                       int8_t *output_data, void *scratch);

}  // namespace coralnpu_v2::opt::litert_micro::vme

#endif  // SW_OPT_LITERT_MICRO_CONV_VME_H_
