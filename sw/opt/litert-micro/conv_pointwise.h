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

// The two steps of the RVV pointwise (1x1) conv kernel and the 3x3 kernel (configuration S, #26),
// exposed for the DMA-tiled paths (#11). Only declared with CORALNPU_RVV_CONV, like the kernel
// itself.

#ifndef SW_OPT_LITERT_MICRO_CONV_POINTWISE_H_
#define SW_OPT_LITERT_MICRO_CONV_POINTWISE_H_

#include <cstdint>

#include "tensorflow/lite/kernels/internal/types.h"

#if CORALNPU_RVV_CONV

namespace coralnpu_v2::opt::litert_micro {

// Transposes `out_d` rows of an OHWI filter ([out_d][in_d] int8) to [in_d][out_d] int16 in
// `weights` and writes the bias with the input offset folded in to `bias` (out_d values).
// `bias_data` may be null (zero bias).
void Conv1x1Prepare(const tflite::ConvParams &params, int in_d, int out_d,
                    const int8_t *filter_data, const int32_t *bias_data, int16_t *weights,
                    int32_t *bias);

// Runs the kernel for `pixels` pixels on prepared weights and bias. `accs` holds 4 * out_d int32
// values. `output_multiplier`, `shift_left` and `shift_right` are indexed 0 to out_d - 1.
void Conv1x1Run(const tflite::ConvParams &params, const int32_t *output_multiplier,
                const uint8_t *shift_left, const uint8_t *shift_right, int in_d, int out_d,
                int pixels, const int8_t *input_data, const int16_t *weights, const int32_t *bias,
                int32_t *accs, int8_t *output_data);

// The 3x3 conv kernel (MobileNet's first layer) in two steps, for the row-tiled path (#11), which
// prepares once per layer and runs once per tile with its own buffers.
//
// Number of int16 values Conv3x3Prepare writes for `out_d` output channels and k_size =
// fh * fw * in_d filter values per channel.
int Conv3x3WeightCount(int out_d, int k_size);

// Lays out an OHWI filter as the int16 weights Conv3x3Run reads (Conv3x3WeightCount values).
void Conv3x3Prepare(int out_d, int k_size, const int8_t *filter_data, int16_t *weights);

// Runs the kernel on prepared weights. `filter_data` is still read for the pixels whose taps reach
// into the padding. The output is requantized every `acc_pixels` pixels of a row (at most out_w),
// so `accs` holds acc_pixels * out_d int32 values. `shift_left` and `shift_right` are indexed by
// output channel.
void Conv3x3Run(const tflite::ConvParams &params, const int32_t *output_multiplier,
                const uint8_t *shift_left, const uint8_t *shift_right,
                const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
                const int16_t *weights, const int32_t *bias_data,
                const tflite::RuntimeShape &output_shape, int8_t *output_data, int32_t *accs,
                int acc_pixels);

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // CORALNPU_RVV_CONV

#endif  // SW_OPT_LITERT_MICRO_CONV_POINTWISE_H_
