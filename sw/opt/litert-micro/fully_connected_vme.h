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

// INT8 fully connected layer on the matrix engine (idea 5, #12), run as a 1x1
// convolution over a 1 x batches image. Batches map to the tile's M dimension,
// so batch 1 uses one of its 16 rows.

#ifndef SW_OPT_LITERT_MICRO_FULLY_CONNECTED_VME_H_
#define SW_OPT_LITERT_MICRO_FULLY_CONNECTED_VME_H_

#include <cstddef>
#include <cstdint>

#include "sw/opt/litert-micro/conv_vme.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro::vme {

struct FullyConnectedVmeWeights {
  ConvVmeWeights conv;
  int32_t *output_multiplier;  // per-tensor multiplier expanded per channel
};

size_t FullyConnectedVmePersistentBytes(const tflite::RuntimeShape &filter_shape);
size_t FullyConnectedVmeScratchBytes(const tflite::RuntimeShape &filter_shape);

// Requires params.weights_offset == 0 and input_offset in [-127, 128].
FullyConnectedVmeWeights FullyConnectedVmePrepare(const tflite::FullyConnectedParams &params,
                                                  const tflite::RuntimeShape &filter_shape,
                                                  const int8_t *filter_data,
                                                  const int32_t *bias_data, void *persistent);

// Same contract as reference_integer_ops::FullyConnected (per-tensor).
void FullyConnectedVme(const tflite::FullyConnectedParams &params,
                       const FullyConnectedVmeWeights &weights,
                       const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                       const tflite::RuntimeShape &output_shape, int8_t *output_data,
                       void *scratch);

}  // namespace coralnpu_v2::opt::litert_micro::vme

#endif  // SW_OPT_LITERT_MICRO_FULLY_CONNECTED_VME_H_
