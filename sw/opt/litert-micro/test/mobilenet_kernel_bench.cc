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

// Baseline per-layer kernel benchmark (#3): runs one MobileNet conv or
// depthwise conv layer through the optimized dispatcher (what ConvEval and
// DepthwiseEval run) and, optionally, through the TFLite reference kernel,
// timing both with mcycle/minstret. mobilenet_kernel_bench.py drives it layer
// by layer and compares the two outputs.
//
// Placement follows tests/cocotb/tutorial/tfmicro/depthwise_conv_test.cc:
// activations and the accumulator scratch in DTCM (the tensor arena), weights
// and bias in external memory (the model flatbuffer).

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/utils/perf_counters.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/conv.h"

using namespace coralnpu_v2::opt::litert_micro;

enum Kernel : int32_t {
  KERNEL_CONV      = 0,
  KERNEL_DEPTHWISE = 1,
};

extern "C" {
// Sized for MobileNet v1 0.25 at 224x224: largest input 113x113x16 (padded
// depthwise), largest output 112x112x16, largest filter 256x1x1x256, largest
// depthwise accumulator 112x112x8 int32.
int8_t input_data[204304] __attribute__((section(".data"), aligned(16))) = {0};
int8_t output_opt[200704] __attribute__((section(".data"), aligned(16))) = {0};
int32_t accs_buf[100352] __attribute__((section(".data"), aligned(16)))  = {0};
int8_t output_ref[200704] __attribute__((section(".extdata"), aligned(16)));
int8_t filter_data[65536] __attribute__((section(".extdata"), aligned(16)));
int32_t bias_data[256] __attribute__((section(".extdata"), aligned(16)));

int32_t kernel __attribute__((section(".data")))        = KERNEL_CONV;
int32_t run_reference __attribute__((section(".data"))) = 1;

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

uint64_t opt_cycles __attribute__((section(".data"), used))  = 0;
uint64_t opt_instret __attribute__((section(".data"), used)) = 0;
uint64_t ref_cycles __attribute__((section(".data"), used))  = 0;
uint64_t ref_instret __attribute__((section(".data"), used)) = 0;
#if CORALNPU_PERF_COUNTERS
// #59: stall counter deltas over the timed kernel call.
int32_t perf_counters_enabled __attribute__((section(".data"), used))                 = 1;
uint64_t opt_perf[coralnpu_v2::kPerfCounters] __attribute__((section(".data"), used)) = {};
#endif
int32_t heartbeat __attribute__((section(".data"), used))    = 0xAA;

static TfLiteStatus MockRequestScratchBuffer(TfLiteContext *, size_t, int *buffer_idx) {
  *buffer_idx = 0;
  return kTfLiteOk;
}
static void *MockGetScratchBuffer(TfLiteContext *, int) { return accs_buf; }

static tflite::ConvParams MakeConvParams() {
  tflite::ConvParams params;
  params.stride_width             = params_stride;
  params.stride_height            = params_stride;
  params.dilation_width_factor    = 1;
  params.dilation_height_factor   = 1;
  params.padding_values.width     = params_padding_width;
  params.padding_values.height    = params_padding_height;
  params.input_offset             = params_input_offset;
  params.weights_offset           = 0;
  params.output_offset            = params_output_offset;
  params.quantized_activation_min = params_activation_min;
  params.quantized_activation_max = params_activation_max;
  return params;
}

static tflite::DepthwiseParams MakeDepthwiseParams() {
  tflite::DepthwiseParams params;
  params.stride_width             = params_stride;
  params.stride_height            = params_stride;
  params.dilation_width_factor    = 1;
  params.dilation_height_factor   = 1;
  params.depth_multiplier         = 1;
  params.padding_values.width     = params_padding_width;
  params.padding_values.height    = params_padding_height;
  params.input_offset             = params_input_offset;
  params.weights_offset           = 0;
  params.output_offset            = params_output_offset;
  params.quantized_activation_min = params_activation_min;
  params.quantized_activation_max = params_activation_max;
  return params;
}

__attribute__((used, retain)) void run_opt() {
  heartbeat = 0x11;
  const tflite::RuntimeShape input_shape(4, input_dims);
  const tflite::RuntimeShape filter_shape(4, filter_dims);
  const tflite::RuntimeShape output_shape(4, output_dims);
  const int32_t bias_dims[1] = {output_dims[3]};
  const tflite::RuntimeShape bias_shape(1, bias_dims);

  if (kernel == KERNEL_CONV) {
    const tflite::ConvParams params = MakeConvParams();
    TfLiteContext context;
    context.GetScratchBuffer            = MockGetScratchBuffer;
    context.RequestScratchBufferInArena = MockRequestScratchBuffer;
    OpDataConvCustom data;
    memset(&data, 0, sizeof(data));
    data.per_channel_output_multiplier = per_channel_multiplier;
    data.per_channel_output_shift      = per_channel_shift;
    data.output_activation_min         = params_activation_min;
    data.output_activation_max         = params_activation_max;

#if CORALNPU_PERF_COUNTERS
    uint64_t perf0[coralnpu_v2::kPerfCounters];
    coralnpu_v2::PerfBegin(perf0);
#endif
    const uint64_t c0 = mcycle_read();
    const uint64_t i0 = minstret_read();
    ConvPerChannel(params, data, per_channel_multiplier, per_channel_shift, &context, input_shape,
                   input_data, filter_shape, filter_data, bias_shape, bias_data, output_shape,
                   output_opt);
    opt_instret = minstret_read() - i0;
    opt_cycles  = mcycle_read() - c0;
#if CORALNPU_PERF_COUNTERS
    coralnpu_v2::PerfEnd(perf0, opt_perf);
#endif
  } else {
    const tflite::DepthwiseParams params = MakeDepthwiseParams();
#if CORALNPU_PERF_COUNTERS
    uint64_t perf0[coralnpu_v2::kPerfCounters];
    coralnpu_v2::PerfBegin(perf0);
#endif
    const uint64_t c0                    = mcycle_read();
    const uint64_t i0                    = minstret_read();
    DepthwiseConvPerChannel(params, per_channel_multiplier, per_channel_shift, input_shape,
                            input_data, filter_shape, filter_data, bias_shape, bias_data,
                            output_shape, output_opt, accs_buf);
    opt_instret = minstret_read() - i0;
    opt_cycles  = mcycle_read() - c0;
#if CORALNPU_PERF_COUNTERS
    coralnpu_v2::PerfEnd(perf0, opt_perf);
#endif
  }
  heartbeat = 0x12;
}

__attribute__((used, retain)) void run_ref() {
  heartbeat = 0x21;
  const tflite::RuntimeShape input_shape(4, input_dims);
  const tflite::RuntimeShape filter_shape(4, filter_dims);
  const tflite::RuntimeShape output_shape(4, output_dims);
  const int32_t bias_dims[1] = {output_dims[3]};
  const tflite::RuntimeShape bias_shape(1, bias_dims);

  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
  if (kernel == KERNEL_CONV) {
    tflite::reference_integer_ops::ConvPerChannel(
        MakeConvParams(), per_channel_multiplier, per_channel_shift, input_shape, input_data,
        filter_shape, filter_data, bias_shape, bias_data, output_shape, output_ref);
  } else {
    tflite::reference_integer_ops::DepthwiseConvPerChannel(
        MakeDepthwiseParams(), per_channel_multiplier, per_channel_shift, input_shape, input_data,
        filter_shape, filter_data, bias_shape, bias_data, output_shape, output_ref);
  }
  ref_instret = minstret_read() - i0;
  ref_cycles  = mcycle_read() - c0;
  heartbeat   = 0x22;
}

int main(void) {
  run_opt();
  if (run_reference) {
    run_ref();
  }
  return 0;
}
}
