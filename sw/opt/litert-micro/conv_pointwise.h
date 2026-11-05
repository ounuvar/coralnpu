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

// The 3x3 conv kernel (MobileNet's first layer), exposed for the row-tiled path (#11). It
// allocates its own scratch and returns false if that fails. `shift_left` and `shift_right` are
// indexed by output channel.
bool Conv3x3PerChannel(const tflite::ConvParams &params, const int32_t *output_multiplier,
                       const uint8_t *shift_left, const uint8_t *shift_right,
                       const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                       const tflite::RuntimeShape &filter_shape, const int8_t *filter_data,
                       const int32_t *bias_data, const tflite::RuntimeShape &output_shape,
                       int8_t *output_data);

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // CORALNPU_RVV_CONV

#endif  // SW_OPT_LITERT_MICRO_CONV_POINTWISE_H_
