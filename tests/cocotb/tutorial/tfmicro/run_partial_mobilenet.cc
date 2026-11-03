// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "sw/opt/litert-micro/depthwise_conv.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_partial_layers.h"

namespace {
using MobilenetOpResolver = tflite::MicroMutableOpResolver<2>;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;
TfLiteStatus RegisterOps(MobilenetOpResolver &op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  return kTfLiteOk;
}
}  // namespace

// Per-op profiling. Cycles and retired instructions come from the core's own
// mcycle/minstret counters (not from npusim, which is a functional ISS).
// Results are left in global arrays that the cocotb test reads after halt.
// Each op costs a few CSR reads, so the overhead is tens of cycles per op,
// negligible against the layers measured.
extern "C" {
constexpr uint32_t kMaxProfileEvents = 128;
uint32_t profile_num_events;                   // events seen (may exceed the max)
uint32_t profile_tag_addr[kMaxProfileEvents];  // address of the op-name string
uint32_t profile_cycles[kMaxProfileEvents];    // mcycle delta of the op
uint32_t profile_instret[kMaxProfileEvents];   // minstret delta of the op
}

namespace {
inline uint32_t ReadMcycle() {
  uint32_t v;
  asm volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}
inline uint32_t ReadMinstret() {
  uint32_t v;
  asm volatile("csrr %0, minstret" : "=r"(v));
  return v;
}

class CycleProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char *tag) override {
    const uint32_t idx = profile_num_events++;
    if (idx < kMaxProfileEvents) {
      profile_tag_addr[idx] = reinterpret_cast<uint32_t>(tag);
      start_cycles_[idx]    = ReadMcycle();
      start_instret_[idx]   = ReadMinstret();
    }
    return idx;
  }
  void EndEvent(uint32_t idx) override {
    if (idx < kMaxProfileEvents) {
      profile_cycles[idx]  = ReadMcycle() - start_cycles_[idx];
      profile_instret[idx] = ReadMinstret() - start_instret_[idx];
    }
  }

 private:
  uint32_t start_cycles_[kMaxProfileEvents];
  uint32_t start_instret_[kMaxProfileEvents];
};
}  // namespace

extern "C" {
// aligned(16)
constexpr size_t kTensorArenaSize = 2048 * 1024;
int8_t inference_status           = -1;
char inference_status_message[31] __attribute__((section(".data"), aligned(16)));
uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".extdata"), aligned(16)));
}

int main(int argc, char **argv) {
  const tflite::Model *model = tflite::GetModel(g_mobilenet_v1_025_partial_layers_model_data);
  MobilenetOpResolver op_resolver;
  RegisterOps(op_resolver);
  std::strncpy(inference_status_message, "Halted after op resolver", 31);
  CycleProfiler profiler;
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize, nullptr,
                                       &profiler);
  std::strncpy(inference_status_message, "Halted after Interpreter setup", 31);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::strncpy(inference_status_message, "Error during AllocateTensors", 31);
    return -1;
  }
  if (interpreter.Invoke() != kTfLiteOk) {
    std::strncpy(inference_status_message, "Error during Invoke", 31);
    return -1;
  }
  std::strncpy(inference_status_message, "Invoke successful", 31);
  inference_status = 0;
  return 0;
}
