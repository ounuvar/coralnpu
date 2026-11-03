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

#include "sw/opt/litert-micro/fully_connected_vme.h"

#include <cstdint>

#include "tensorflow/lite/kernels/internal/common.h"

namespace coralnpu_v2::opt::litert_micro::vme {
namespace {

constexpr size_t Align16(size_t bytes) { return (bytes + 15) & ~size_t{15}; }

int OutputDepth(const tflite::RuntimeShape &filter_shape) {
  return filter_shape.Dims(filter_shape.DimensionsCount() - 2);
}

int AccumDepth(const tflite::RuntimeShape &filter_shape) {
  return filter_shape.Dims(filter_shape.DimensionsCount() - 1);
}

// The filter as a [N, 1, 1, K] OHWI conv filter.
tflite::RuntimeShape AsConvFilter(const tflite::RuntimeShape &filter_shape) {
  const int32_t dims[4] = {OutputDepth(filter_shape), 1, 1, AccumDepth(filter_shape)};
  return tflite::RuntimeShape(4, dims);
}

tflite::ConvParams AsConvParams(const tflite::FullyConnectedParams &params) {
  tflite::ConvParams conv;
  conv.padding_values.width     = 0;
  conv.padding_values.height    = 0;
  conv.stride_width             = 1;
  conv.stride_height            = 1;
  conv.dilation_width_factor    = 1;
  conv.dilation_height_factor   = 1;
  conv.input_offset             = params.input_offset;
  conv.weights_offset           = 0;
  conv.output_offset            = params.output_offset;
  conv.quantized_activation_min = params.quantized_activation_min;
  conv.quantized_activation_max = params.quantized_activation_max;
  return conv;
}

}  // namespace

size_t FullyConnectedVmePersistentBytes(const tflite::RuntimeShape &filter_shape) {
  const int n = OutputDepth(filter_shape);
  // Conv weights, the expanded multiplier, and the expanded shift (only needed
  // while preparing, but kept to avoid a separate scratch request).
  return ConvVmePersistentBytes(AsConvFilter(filter_shape)) + 2 * Align16(n * sizeof(int32_t));
}

size_t FullyConnectedVmeScratchBytes(const tflite::RuntimeShape &filter_shape) {
  return ConvVmeScratchBytes(AsConvFilter(filter_shape));
}

FullyConnectedVmeWeights FullyConnectedVmePrepare(const tflite::FullyConnectedParams &params,
                                                  const tflite::RuntimeShape &filter_shape,
                                                  const int8_t *filter_data,
                                                  const int32_t *bias_data, void *persistent) {
  const tflite::RuntimeShape conv_filter = AsConvFilter(filter_shape);
  const int n                            = OutputDepth(filter_shape);
  uint8_t *p = static_cast<uint8_t *>(persistent) + ConvVmePersistentBytes(conv_filter);

  FullyConnectedVmeWeights w;
  w.output_multiplier = reinterpret_cast<int32_t *>(p);
  int32_t *shift      = reinterpret_cast<int32_t *>(p + Align16(n * sizeof(int32_t)));
  for (int c = 0; c < n; ++c) {
    w.output_multiplier[c] = params.output_multiplier;
    shift[c]               = params.output_shift;
  }
  w.conv =
      ConvVmePrepare(AsConvParams(params), shift, conv_filter, filter_data, bias_data, persistent);
  return w;
}

void FullyConnectedVme(const tflite::FullyConnectedParams &params,
                       const FullyConnectedVmeWeights &weights,
                       const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                       const tflite::RuntimeShape &output_shape, int8_t *output_data,
                       void *scratch) {
  const int output_dims = output_shape.DimensionsCount();
  const int batches     = tflite::FlatSizeSkipDim(output_shape, output_dims - 1);
  const int k           = weights.conv.k;
  (void)input_shape;
  const int32_t image_dims[4]     = {1, 1, batches, k};
  const int32_t out_image_dims[4] = {1, 1, batches, weights.conv.n};
  ConvPerChannelVme(AsConvParams(params), weights.conv, weights.output_multiplier,
                    tflite::RuntimeShape(4, image_dims), input_data,
                    tflite::RuntimeShape(4, out_image_dims), output_data, scratch);
}

}  // namespace coralnpu_v2::opt::litert_micro::vme
