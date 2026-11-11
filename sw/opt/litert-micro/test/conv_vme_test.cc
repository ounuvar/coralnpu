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

// Runs reference_integer_ops::ConvPerChannel and the matrix-engine kernel on
// the same inputs; conv_vme_test.py compares the outputs.

#include "sw/opt/litert-micro/conv_vme.h"

#include <cstdint>

#include "sw/utils/utils.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

extern "C" {
int8_t input_data[65536] __attribute__((section(".data"), aligned(16)))      = {0};
int8_t filter_data[65536] __attribute__((section(".data"), aligned(16)))     = {0};
int32_t bias_data[256] __attribute__((section(".data"), aligned(16)))        = {0};
int8_t output_data[65536] __attribute__((section(".data"), aligned(16)))     = {0};
int8_t output_data_ref[65536] __attribute__((section(".data"), aligned(16))) = {0};

// Whether this ELF was built with --//sw/opt/litert-micro:tflite_ties (#23).
int32_t tflite_ties_enabled __attribute__((section(".data"), used)) = CORALNPU_TFLITE_TIES;

int32_t input_dims[4] __attribute__((section(".data")))  = {0};
int32_t filter_dims[4] __attribute__((section(".data"))) = {0};
int32_t bias_dims[4] __attribute__((section(".data")))   = {0};
int32_t output_dims[4] __attribute__((section(".data"))) = {0};

int32_t params_stride_width __attribute__((section(".data")))    = 0;
int32_t params_stride_height __attribute__((section(".data")))   = 0;
int32_t params_dilation_width __attribute__((section(".data")))  = 1;
int32_t params_dilation_height __attribute__((section(".data"))) = 1;
int32_t params_padding_width __attribute__((section(".data")))   = 0;
int32_t params_padding_height __attribute__((section(".data")))  = 0;
int32_t params_input_offset __attribute__((section(".data")))    = 0;
int32_t params_output_offset __attribute__((section(".data")))   = 0;
int32_t params_activation_min __attribute__((section(".data")))  = 0;
int32_t params_activation_max __attribute__((section(".data")))  = 0;

int32_t per_channel_multiplier[256] __attribute__((section(".data"))) = {0};
int32_t per_channel_shift[256] __attribute__((section(".data")))      = {0};

uint64_t ref_cycles __attribute__((section(".data"), used)) = 0;
uint64_t opt_cycles __attribute__((section(".data"), used)) = 0;
int32_t heartbeat __attribute__((section(".data"), used))   = 0xAA;
// The RTL targets clear this and check against a host copy of the reference
// (tflite_int8_ref.py); the scalar reference is ~98% of their cycles (#63).
int32_t run_reference __attribute__((section(".data"), used)) = 1;

int8_t persistent_buffer[131072] __attribute__((section(".data"), aligned(16))) = {0};
int8_t scratch_buffer[65536] __attribute__((section(".data"), aligned(16)))     = {0};

static tflite::ConvParams MakeParams() {
  tflite::ConvParams params;
  params.stride_width             = params_stride_width;
  params.stride_height            = params_stride_height;
  params.dilation_width_factor    = params_dilation_width;
  params.dilation_height_factor   = params_dilation_height;
  params.padding_values.width     = params_padding_width;
  params.padding_values.height    = params_padding_height;
  params.input_offset             = params_input_offset;
  params.output_offset            = params_output_offset;
  params.quantized_activation_min = params_activation_min;
  params.quantized_activation_max = params_activation_max;
  return params;
}

__attribute__((used, retain)) void run_ref() {
  heartbeat                       = 0x11;
  const tflite::ConvParams params = MakeParams();
  uint64_t start                  = mcycle_read();
  tflite::reference_integer_ops::ConvPerChannel(
      params, per_channel_multiplier, per_channel_shift, tflite::RuntimeShape(4, input_dims),
      input_data, tflite::RuntimeShape(4, filter_dims), filter_data,
      tflite::RuntimeShape(4, bias_dims), bias_data, tflite::RuntimeShape(4, output_dims),
      output_data_ref);
  ref_cycles = mcycle_read() - start;
  heartbeat  = 0x12;
}

__attribute__((used, retain)) void run_vme() {
  heartbeat                       = 0x21;
  const tflite::ConvParams params = MakeParams();
  namespace vme                   = coralnpu_v2::opt::litert_micro::vme;
  const tflite::RuntimeShape filter_shape(4, filter_dims);
  if (vme::ConvVmePersistentBytes(filter_shape) > sizeof(persistent_buffer) ||
      vme::ConvVmeScratchBytes(filter_shape) > sizeof(scratch_buffer)) {
    heartbeat = 0xEE;
    return;
  }
  // Prepare (weight packing) is per-layer setup and is not timed.
  const vme::ConvVmeWeights weights = vme::ConvVmePrepare(
      params, per_channel_shift, filter_shape, filter_data, bias_data, persistent_buffer);
  uint64_t start = mcycle_read();
  vme::ConvPerChannelVme(params, weights, per_channel_multiplier,
                         tflite::RuntimeShape(4, input_dims), input_data,
                         tflite::RuntimeShape(4, output_dims), output_data, scratch_buffer);
  opt_cycles = mcycle_read() - start;
  heartbeat  = 0x22;
}

int main(void) {
  if (run_reference) {
    run_ref();
  }
  run_vme();
  return 0;
}
}
