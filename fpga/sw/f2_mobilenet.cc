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

// Full MobileNet v1 0.25 (224, INT8) on the F2 custom logic (#16), timed and checked in one run.
//
// Pass 1 (timed): the interpreter with the optimized CONV_2D and DEPTHWISE_CONV_2D. mcycle and
// minstret around Invoke give the full-model numbers; each conv and depthwise layer is timed too.
// Pass 2 (checked): the same input again; every conv and depthwise layer runs the optimized kernel,
// then TFLite's reference kernel on the same node, and the two outputs are compared byte by byte.
// The optimized output is put back, so the model continues on the optimized path. The optimized
// kernels round negative ties toward +infinity where TFLite rounds away from zero (#23), so a byte
// may be TFLite's value plus one; any other difference is a bug. Both passes must give the same
// output checksum.
//
// Results: the f2_mnet struct (read by the host over SPI) and a short summary on UART1. f2_result
// is F2_PASS when both passes finished, no layer differs other than by a #23 tie, and the
// checksums match. The tensor arena is in the 4 MB SRAM at 0x20000000 (.extbss, not loaded).

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fpga/sw/uart.h"
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

constexpr uint32_t kPass       = 0x600DF00Du;
constexpr uint32_t kMagic      = 0x4D4E4554u;  // "MNET"
constexpr int kMaxLayers       = 32;
constexpr size_t kArenaSize    = 2 * 1024 * 1024;
constexpr size_t kMaxLayerSize = 512 * 1024;  // the largest conv/dw output at 224

}  // namespace

extern "C" {

// Host-visible results, one block. Keep the layout in step with fpga/f2/tb/f2_mobilenet.py.
struct F2Mnet {
  uint32_t magic;      // kMagic once the program started
  uint32_t stage;      // progress: 1 setup, 2 timed pass, 3 checked pass, 4 done; 0xE0+ errors
  uint32_t layers;     // conv and depthwise layers seen in the checked pass
  uint32_t cycles_lo;  // timed pass: mcycle and minstret around Invoke
  uint32_t cycles_hi;
  uint32_t instret_lo;
  uint32_t instret_hi;
  uint32_t checksum_timed;    // FNV-1a of the output tensor, pass 1
  uint32_t checksum_checked;  // and pass 2
  uint32_t total_plus_one;    // over all layers
  uint32_t total_other;
  int8_t output[8];  // first bytes of the output tensor (pass 1)
  int32_t is_dw[kMaxLayers];
  int32_t bytes[kMaxLayers];
  uint32_t layer_cycles[kMaxLayers];  // pass 1, optimized kernel only
  int32_t equal[kMaxLayers];
  int32_t plus_one[kMaxLayers];
  int32_t other[kMaxLayers];
};

volatile uint32_t f2_result = 0;
// The host (or the npusim test, which has no UART) can clear this before starting the core.
volatile uint32_t f2_uart_enable                                    = 1;
F2Mnet f2_mnet __attribute__((section(".data"), aligned(16), used)) = {};

uint8_t tensor_arena[kArenaSize] __attribute__((section(".extbss"), aligned(16)));
int8_t saved_output[kMaxLayerSize] __attribute__((section(".extbss"), aligned(16)));
}

namespace {

using OpResolver = tflite::MicroMutableOpResolver<10>;
using Invoke     = TfLiteStatus (*)(TfLiteContext *, TfLiteNode *);

Invoke opt_conv = nullptr;
Invoke opt_dw   = nullptr;
Invoke ref_conv = nullptr;
Invoke ref_dw   = nullptr;
bool checking   = false;  // pass 2
int layer       = 0;

void Puts(const char *s) {
  if (f2_uart_enable)
    uart_puts(s);
}

void PutHex(uint32_t v) {
  if (f2_uart_enable)
    uart_puthex32(v);
}

uint32_t Fnv(const int8_t *data, size_t bytes) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < bytes; ++i)
    hash = (hash ^ static_cast<uint8_t>(data[i])) * 16777619u;
  return hash;
}

TfLiteStatus Layer(TfLiteContext *context, TfLiteNode *node, bool dw) {
  const int k = layer;
  if (k >= kMaxLayers) {
    f2_mnet.stage = 0xE3;
    return kTfLiteError;
  }
  layer = k + 1;
  if (!checking) {
    const uint64_t start    = mcycle_read();
    const TfLiteStatus s    = (dw ? opt_dw : opt_conv)(context, node);
    f2_mnet.layer_cycles[k] = static_cast<uint32_t>(mcycle_read() - start);
    return s;
  }
  TF_LITE_ENSURE_STATUS((dw ? opt_dw : opt_conv)(context, node));
  const TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  size_t bytes                   = 0;
  TF_LITE_ENSURE_STATUS(tflite::TfLiteEvalTensorByteLength(output, &bytes));
  if (bytes > kMaxLayerSize) {
    f2_mnet.stage = 0xE4;
    return kTfLiteError;
  }
  int8_t *out = static_cast<int8_t *>(output->data.data);
  memcpy(saved_output, out, bytes);
  TF_LITE_ENSURE_STATUS((dw ? ref_dw : ref_conv)(context, node));  // TFLite's result in `out`
  int32_t equal = 0, plus_one = 0, other = 0;
  for (size_t i = 0; i < bytes; ++i) {
    const int d = static_cast<int>(saved_output[i]) - static_cast<int>(out[i]);
    equal += d == 0;
    plus_one += d == 1;
    other += d != 0 && d != 1;
  }
  memcpy(out, saved_output, bytes);  // continue on the optimized path
  f2_mnet.is_dw[k]    = dw ? 1 : 0;
  f2_mnet.bytes[k]    = static_cast<int32_t>(bytes);
  f2_mnet.equal[k]    = equal;
  f2_mnet.plus_one[k] = plus_one;
  f2_mnet.other[k]    = other;
  f2_mnet.layers      = static_cast<uint32_t>(k + 1);
  f2_mnet.total_plus_one += static_cast<uint32_t>(plus_one);
  f2_mnet.total_other += static_cast<uint32_t>(other);
  return kTfLiteOk;
}

TfLiteStatus Conv(TfLiteContext *context, TfLiteNode *node) { return Layer(context, node, false); }
TfLiteStatus Dw(TfLiteContext *context, TfLiteNode *node) { return Layer(context, node, true); }

TfLiteStatus RegisterOps(OpResolver &op_resolver) {
  TFLMRegistration conv = coralnpu_v2::opt::litert_micro::Register_CONV_2D();
  TFLMRegistration dw   = coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D();
  opt_conv              = conv.invoke;
  opt_dw                = dw.invoke;
  ref_conv              = tflite::Register_CONV_2D().invoke;
  ref_dw                = tflite::Register_DEPTHWISE_CONV_2D().invoke;
  conv.invoke           = Conv;
  dw.invoke             = Dw;
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

// The same deterministic input as mobilenet_vs_tflite_test (#11).
void FillInput(TfLiteTensor *input) {
  uint32_t state = 12345;
  for (size_t i = 0; i < input->bytes; ++i) {
    state               = state * 1664525u + 1013904223u;
    input->data.int8[i] = static_cast<int8_t>(state >> 24);
  }
}

uint32_t Fail(uint32_t stage) {
  f2_mnet.stage = stage;
  f2_result     = 0xBAD00000u | stage;
  Puts("FAIL ");
  PutHex(stage);
  Puts("\r\n");
  return 1;
}

}  // namespace

int main(void) {
  f2_mnet.magic = kMagic;
  f2_mnet.stage = 1;
  if (f2_uart_enable)
    uart_init();
  Puts("MNET\r\n");

  OpResolver op_resolver;
  if (RegisterOps(op_resolver) != kTfLiteOk)
    return Fail(0xE0);
  const tflite::Model *model = tflite::GetModel(g_25_224_int8_dummy_model_data);
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk)
    return Fail(0xE1);
  TfLiteTensor *input        = interpreter.input(0);
  const TfLiteTensor *output = interpreter.output(0);

  // Pass 1: timed.
  f2_mnet.stage = 2;
  FillInput(input);
  const uint64_t cycles0  = mcycle_read();
  const uint64_t instret0 = minstret_read();
  const TfLiteStatus s1   = interpreter.Invoke();
  const uint64_t cycles   = mcycle_read() - cycles0;
  const uint64_t instret  = minstret_read() - instret0;
  if (s1 != kTfLiteOk)
    return Fail(f2_mnet.stage == 2 ? 0xE2 : f2_mnet.stage);
  f2_mnet.cycles_lo      = static_cast<uint32_t>(cycles);
  f2_mnet.cycles_hi      = static_cast<uint32_t>(cycles >> 32);
  f2_mnet.instret_lo     = static_cast<uint32_t>(instret);
  f2_mnet.instret_hi     = static_cast<uint32_t>(instret >> 32);
  f2_mnet.checksum_timed = Fnv(output->data.int8, output->bytes);
  for (int i = 0; i < 8; ++i)
    f2_mnet.output[i] = output->data.int8[i];
  Puts("cycles ");
  PutHex(f2_mnet.cycles_hi);
  PutHex(f2_mnet.cycles_lo);
  Puts(" instret ");
  PutHex(f2_mnet.instret_hi);
  PutHex(f2_mnet.instret_lo);
  Puts("\r\nchecksum ");
  PutHex(f2_mnet.checksum_timed);
  Puts("\r\n");

  // Pass 2: checked against TFLite. Invoke reuses the arena, so the input is filled again.
  f2_mnet.stage = 3;
  checking      = true;
  layer         = 0;
  FillInput(input);
  if (interpreter.Invoke() != kTfLiteOk)
    return Fail(f2_mnet.stage == 3 ? 0xE5 : f2_mnet.stage);
  f2_mnet.checksum_checked = Fnv(output->data.int8, output->bytes);
  Puts("layers ");
  PutHex(f2_mnet.layers);
  Puts(" plus_one ");
  PutHex(f2_mnet.total_plus_one);
  Puts(" other ");
  PutHex(f2_mnet.total_other);
  Puts("\r\n");

  f2_mnet.stage = 4;
  if (f2_mnet.layers == 0 || f2_mnet.total_other != 0 ||
      f2_mnet.checksum_checked != f2_mnet.checksum_timed)
    return Fail(0xE6);
  f2_result = kPass;
  Puts("PASS\r\n");
  return 0;
}
