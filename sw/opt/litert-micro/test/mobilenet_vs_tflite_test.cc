// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Full MobileNet through the interpreter with the optimized CONV_2D and DEPTHWISE_CONV_2D, each
// layer checked against TFLite's reference kernel on the same input (#11).
//
// Every conv and depthwise invoke runs the optimized kernel (DMA-tiled with
// --//sw/opt/litert-micro:conv_tiling), keeps its output, runs TFLite's reference Eval on the
// same node (OpDataConvCustom extends tflite::OpDataConv, and the optimized Prepare runs TFLite's
// Prepare first), compares the two byte by byte, and puts the optimized output back so the model
// continues on the optimized path. The optimized kernels round negative ties toward +infinity
// where TFLite rounds them away from zero (#23), so a byte may be TFLite's value plus one; any
// other difference is a bug. The final output's checksum lets a conv_tiling build be compared
// with a build without it.

#include <cstdint>
#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_224_int8_dummy.h"

namespace {

constexpr int kMaxLayers   = 32;
constexpr size_t kMaxBytes = 512 * 1024;  // the largest layer output of MobileNet v1 0.25 at 224

using OpResolver = tflite::MicroMutableOpResolver<10>;

TfLiteStatus (*opt_conv)(TfLiteContext *, TfLiteNode *) = nullptr;
TfLiteStatus (*opt_dw)(TfLiteContext *, TfLiteNode *)   = nullptr;
TfLiteStatus (*ref_conv)(TfLiteContext *, TfLiteNode *) = nullptr;
TfLiteStatus (*ref_dw)(TfLiteContext *, TfLiteNode *)   = nullptr;

uint32_t Fnv(const uint8_t *data, size_t bytes) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < bytes; ++i)
    hash = (hash ^ data[i]) * 16777619u;
  return hash;
}

}  // namespace

extern "C" {
constexpr size_t kTensorArenaSize = 2 * 1024 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".extdata"), aligned(16)));
int8_t saved[kMaxBytes] __attribute__((section(".extdata"), aligned(16)));

// Per layer, in model order: 1 for depthwise, 0 for conv; bytes; bytes equal to TFLite; bytes
// that are TFLite's value plus one (#23 ties); bytes that differ otherwise.
int32_t layers __attribute__((section(".data"), used))                     = 0;
int32_t layer_is_dw[kMaxLayers] __attribute__((section(".data"), used))    = {};
int32_t layer_bytes[kMaxLayers] __attribute__((section(".data"), used))    = {};
int32_t layer_equal[kMaxLayers] __attribute__((section(".data"), used))    = {};
int32_t layer_plus_one[kMaxLayers] __attribute__((section(".data"), used)) = {};
int32_t layer_other[kMaxLayers] __attribute__((section(".data"), used))    = {};
uint32_t output_checksum __attribute__((section(".data"), used))           = 0;
int8_t model_output[8] __attribute__((section(".data"), used))             = {};
// 0x22 when finished; 0xA0 / 0xB0 if AllocateTensors / Invoke failed; 0xC0 if a layer output is
// larger than the save buffer or there are too many layers.
int32_t heartbeat __attribute__((section(".data"), used)) = 0xAA;
}

namespace {

TfLiteStatus Checked(TfLiteContext *context, TfLiteNode *node, bool dw) {
  TF_LITE_ENSURE_STATUS((dw ? opt_dw : opt_conv)(context, node));
  const TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  size_t bytes                   = 0;
  TF_LITE_ENSURE_STATUS(tflite::TfLiteEvalTensorByteLength(output, &bytes));
  const int layer = layers;
  if (bytes > kMaxBytes || layer >= kMaxLayers) {
    heartbeat = 0xC0;
    return kTfLiteError;
  }
  int8_t *out = static_cast<int8_t *>(output->data.data);
  memcpy(saved, out, bytes);
  TF_LITE_ENSURE_STATUS((dw ? ref_dw : ref_conv)(context, node));  // TFLite's result in `out`
  int32_t equal = 0, plus_one = 0, other = 0;
  for (size_t i = 0; i < bytes; ++i) {
    const int d = static_cast<int>(saved[i]) - static_cast<int>(out[i]);
    equal += d == 0;
    plus_one += d == 1;
    other += d != 0 && d != 1;
  }
  memcpy(out, saved, bytes);  // continue on the optimized path
  layer_is_dw[layer]    = dw ? 1 : 0;
  layer_bytes[layer]    = static_cast<int32_t>(bytes);
  layer_equal[layer]    = equal;
  layer_plus_one[layer] = plus_one;
  layer_other[layer]    = other;
  layers                = layer + 1;
  return kTfLiteOk;
}

TfLiteStatus CheckedConv(TfLiteContext *context, TfLiteNode *node) {
  return Checked(context, node, false);
}
TfLiteStatus CheckedDw(TfLiteContext *context, TfLiteNode *node) {
  return Checked(context, node, true);
}

TfLiteStatus RegisterOps(OpResolver &op_resolver) {
  TFLMRegistration conv = coralnpu_v2::opt::litert_micro::Register_CONV_2D();
  TFLMRegistration dw   = coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D();
  opt_conv              = conv.invoke;
  opt_dw                = dw.invoke;
  ref_conv              = tflite::Register_CONV_2D().invoke;
  ref_dw                = tflite::Register_DEPTHWISE_CONV_2D().invoke;
  conv.invoke           = CheckedConv;
  dw.invoke             = CheckedDw;
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(conv));
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(dw));
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

}  // namespace

int main(void) {
  const tflite::Model *model = tflite::GetModel(g_25_224_int8_dummy_model_data);
  OpResolver op_resolver;
  if (RegisterOps(op_resolver) != kTfLiteOk) {
    heartbeat = 0xEE;
    return 1;
  }
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    heartbeat = 0xA0;
    return 1;
  }
  // The same deterministic input as mobilenet_engine_select_test.
  TfLiteTensor *input = interpreter.input(0);
  uint32_t state      = 12345;
  for (size_t i = 0; i < input->bytes; ++i) {
    state               = state * 1664525u + 1013904223u;
    input->data.int8[i] = static_cast<int8_t>(state >> 24);
  }
  if (interpreter.Invoke() != kTfLiteOk) {
    if (heartbeat == 0xAA)
      heartbeat = 0xB0;
    return 1;
  }
  const TfLiteTensor *output = interpreter.output(0);
  output_checksum = Fnv(reinterpret_cast<const uint8_t *>(output->data.int8), output->bytes);
  for (int i = 0; i < 8; ++i)
    model_output[i] = output->data.int8[i];
  heartbeat = 0x22;
  return 0;
}
