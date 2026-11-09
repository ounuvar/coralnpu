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

// Per-layer benchmark for idea 5 (#12): runs one MobileNet conv layer through
// the baseline dispatcher (litert_micro::ConvPerChannel, what ConvEval runs)
// and through the matrix-engine kernel, timing both with mcycle/minstret.
// mobilenet_conv_bench.py drives it layer by layer and compares outputs.

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/conv_vme.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/micro/kernels/conv.h"

using namespace coralnpu_v2::opt::litert_micro;

extern "C" {
// Sized for MobileNet v1 0.25 at 224x224: largest input 224x224x3, largest
// output 112x112x16, largest filter 256x1x1x256.
int8_t input_data[150528] __attribute__((section(".data"), aligned(16)))      = {0};
int8_t filter_data[65536] __attribute__((section(".data"), aligned(16)))      = {0};
int32_t bias_data[256] __attribute__((section(".data"), aligned(16)))         = {0};
int8_t output_baseline[200704] __attribute__((section(".data"), aligned(16))) = {0};
int8_t output_vme[200704] __attribute__((section(".data"), aligned(16)))      = {0};

// Whether this ELF was built with --//sw/opt/litert-micro:tflite_ties (#23).
int32_t tflite_ties_enabled __attribute__((section(".data"), used)) = CORALNPU_TFLITE_TIES;

int32_t input_dims[4] __attribute__((section(".data")))  = {0};
int32_t filter_dims[4] __attribute__((section(".data"))) = {0};
int32_t output_dims[4] __attribute__((section(".data"))) = {0};

int32_t params_stride __attribute__((section(".data")))         = 1;
int32_t params_padding_height __attribute__((section(".data"))) = 0;
int32_t params_padding_width __attribute__((section(".data")))  = 0;
int32_t params_input_offset __attribute__((section(".data")))   = 0;
int32_t params_output_offset __attribute__((section(".data")))  = 0;
int32_t params_activation_min __attribute__((section(".data"))) = 0;
int32_t params_activation_max __attribute__((section(".data"))) = 0;

int32_t per_channel_multiplier[256] __attribute__((section(".data"))) = {0};
int32_t per_channel_shift[256] __attribute__((section(".data")))      = {0};

uint64_t baseline_cycles __attribute__((section(".data"), used))  = 0;
uint64_t baseline_instret __attribute__((section(".data"), used)) = 0;
uint64_t vme_cycles __attribute__((section(".data"), used))       = 0;
uint64_t vme_instret __attribute__((section(".data"), used))      = 0;
int32_t heartbeat __attribute__((section(".data"), used))         = 0xAA;

int8_t persistent_buffer[81920] __attribute__((section(".data"), aligned(16))) = {0};
int8_t scratch_buffer[32768] __attribute__((section(".data"), aligned(16)))    = {0};

static TfLiteStatus MockRequestScratchBuffer(TfLiteContext *, size_t, int *buffer_idx) {
  *buffer_idx = 0;
  return kTfLiteOk;
}
static void *MockGetScratchBuffer(TfLiteContext *, int) { return scratch_buffer; }
static void *MockAllocatePersistentBuffer(TfLiteContext *, size_t) { return persistent_buffer; }

static tflite::ConvParams MakeParams() {
  tflite::ConvParams params;
  params.stride_width             = params_stride;
  params.stride_height            = params_stride;
  params.dilation_width_factor    = 1;
  params.dilation_height_factor   = 1;
  params.padding_values.width     = params_padding_width;
  params.padding_values.height    = params_padding_height;
  params.input_offset             = params_input_offset;
  params.output_offset            = params_output_offset;
  params.quantized_activation_min = params_activation_min;
  params.quantized_activation_max = params_activation_max;
  return params;
}

__attribute__((used, retain)) void run_baseline() {
  heartbeat                       = 0x11;
  const tflite::ConvParams params = MakeParams();
  TfLiteContext context;
  context.GetScratchBuffer            = MockGetScratchBuffer;
  context.RequestScratchBufferInArena = MockRequestScratchBuffer;
  context.AllocatePersistentBuffer    = MockAllocatePersistentBuffer;
  OpDataConvCustom data;
  memset(&data, 0, sizeof(data));
  data.per_channel_output_multiplier = per_channel_multiplier;
  data.per_channel_output_shift      = per_channel_shift;
  data.output_activation_min         = params_activation_min;
  data.output_activation_max         = params_activation_max;
  const int32_t bias_dims[1]         = {filter_dims[0]};

  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
  ConvPerChannel(params, data, per_channel_multiplier, per_channel_shift, &context,
                 tflite::RuntimeShape(4, input_dims), input_data,
                 tflite::RuntimeShape(4, filter_dims), filter_data,
                 tflite::RuntimeShape(1, bias_dims), bias_data,
                 tflite::RuntimeShape(4, output_dims), output_baseline);
  baseline_instret = minstret_read() - i0;
  baseline_cycles  = mcycle_read() - c0;
  heartbeat        = 0x12;
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
  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
  vme::ConvPerChannelVme(params, weights, per_channel_multiplier,
                         tflite::RuntimeShape(4, input_dims), input_data,
                         tflite::RuntimeShape(4, output_dims), output_vme, scratch_buffer);
  vme_instret = minstret_read() - i0;
  vme_cycles  = mcycle_read() - c0;
  heartbeat   = 0x22;
}

int main(void) {
  run_baseline();
  run_vme();
  return 0;
}
}
