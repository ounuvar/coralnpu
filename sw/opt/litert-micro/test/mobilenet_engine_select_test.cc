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

// Full MobileNet through the CONV_2D op with the per-layer engine choice (idea
// 5, #12). Runs the model three times on the same input: every conv on the
// vector unit, the per-layer choice, and every conv on the matrix engine. The
// CONV_2D invoke is wrapped to checksum each conv layer's output as it is
// produced and to record which engine ran it; mobilenet_engine_select_test.py
// requires the checksums of all three runs to match and reports the
// instructions/cycles of each Invoke.

#include <climits>
#include <cstdint>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_224_int8_dummy.h"

namespace {

constexpr int kRuns          = 3;
constexpr int kMaxConvLayers = 32;
// Matrix-engine input-depth thresholds: all vector, per-layer, all matrix.
constexpr int kThresholds[kRuns] = {INT_MAX, 16, 0};

using OpResolver = tflite::MicroMutableOpResolver<10>;

TfLiteStatus (*conv_invoke)(TfLiteContext *, TfLiteNode *) = nullptr;
int current_run                                            = 0;

TfLiteStatus CheckedConvInvoke(TfLiteContext *context, TfLiteNode *node);

TfLiteStatus RegisterOps(OpResolver &op_resolver) {
  TFLMRegistration conv = coralnpu_v2::opt::litert_micro::Register_CONV_2D();
  conv_invoke           = conv.invoke;
  conv.invoke           = CheckedConvInvoke;
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(conv));
  TF_LITE_ENSURE_STATUS(
      op_resolver.AddDepthwiseConv2D(coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddShape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
  return kTfLiteOk;
}

// FNV-1a.
uint32_t Checksum(const uint8_t *data, size_t bytes) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < bytes; ++i) {
    hash = (hash ^ data[i]) * 16777619u;
  }
  return hash;
}

}  // namespace

extern "C" {
constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".extdata"), aligned(16)));

// Per run: conv layers run, how many of them on the matrix engine, and the
// checksum of each conv layer's output.
int32_t conv_layers[kRuns] __attribute__((section(".data"), used))                = {};
int32_t matrix_layers[kRuns] __attribute__((section(".data"), used))              = {};
uint32_t checksums[kRuns][kMaxConvLayers] __attribute__((section(".data"), used)) = {};
uint64_t invoke_cycles[kRuns] __attribute__((section(".data"), used))             = {};
uint64_t invoke_instret[kRuns] __attribute__((section(".data"), used))            = {};
int8_t model_output[kRuns][5] __attribute__((section(".data"), used))             = {};
uint32_t arena_used[kRuns] __attribute__((section(".data"), used))                = {};
// Bit i set: do run i. The test may clear bits to save RTL time.
int32_t run_mask __attribute__((section(".data"), used)) = (1 << kRuns) - 1;
// 0x100 + run index while a run is in progress, 0x22 when all runs finished,
// 0xA0 / 0xB0 + run index if AllocateTensors / Invoke failed.
int32_t heartbeat __attribute__((section(".data"), used)) = 0xAA;
}

namespace {

TfLiteStatus CheckedConvInvoke(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_STATUS(conv_invoke(context, node));
  const int layer = conv_layers[current_run]++;
  TF_LITE_ENSURE(context, layer < kMaxConvLayers);
  const auto *data =
      static_cast<const coralnpu_v2::opt::litert_micro::OpDataConvCustom *>(node->user_data);
  if (data->vme_weights != nullptr) {
    ++matrix_layers[current_run];
  }
  const TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  size_t bytes                   = 0;
  TF_LITE_ENSURE_STATUS(tflite::TfLiteEvalTensorByteLength(output, &bytes));
  checksums[current_run][layer] = Checksum(static_cast<const uint8_t *>(output->data.data), bytes);
  return kTfLiteOk;
}

}  // namespace

static int RunModel(int run, const tflite::Model *model, OpResolver &op_resolver) {
  heartbeat   = 0x100 + run;
  current_run = run;
  coralnpu_v2::opt::litert_micro::SetMatrixEngineMinInputDepth(kThresholds[run]);
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 0xA0 + run;
  }
  // Deterministic input, the same for every run.
  TfLiteTensor *input = interpreter.input(0);
  uint32_t state      = 12345;
  for (size_t i = 0; i < input->bytes; ++i) {
    state               = state * 1664525u + 1013904223u;
    input->data.int8[i] = static_cast<int8_t>(state >> 24);
  }

  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
  if (interpreter.Invoke() != kTfLiteOk) {
    return 0xB0 + run;
  }
  invoke_instret[run] = minstret_read() - i0;
  invoke_cycles[run]  = mcycle_read() - c0;
  arena_used[run]     = interpreter.arena_used_bytes();

  const TfLiteTensor *output = interpreter.output(0);
  for (int i = 0; i < 5; ++i) {
    model_output[run][i] = output->data.int8[i];
  }
  return 0;
}

int main(void) {
  const tflite::Model *model = tflite::GetModel(g_25_224_int8_dummy_model_data);
  OpResolver op_resolver;
  if (RegisterOps(op_resolver) != kTfLiteOk) {
    heartbeat = 0xEE;
    return 1;
  }
  for (int run = 0; run < kRuns; ++run) {
    if (!(run_mask & (1 << run))) {
      continue;
    }
    if (const int status = RunModel(run, model, op_resolver)) {
      heartbeat = status;
      return 1;
    }
  }
  heartbeat = 0x22;
  return 0;
}
