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

// Tile-sized conv benchmark for the DMA tiling study (#11).
//
// Runs one tile of MobileNet layer 17 (a 1x1 conv, 128 -> 128 channels, one output row of 14
// pixels, 229,376 MACs) through the same ConvPerChannel dispatcher the model uses, timed with
// mcycle/minstret. Where the tile's data lives is chosen at build time:
//   default             input, filter, bias and output all in DTCM (no external memory traffic)
//   -DTILE_IN_EXTMEM    all four in external memory (what an untiled kernel sees)
//   -DTILE_DMA          the tile lives in external memory, the DMA copies filter, bias and input
//                       into DTCM, the kernel runs there, and the DMA copies the output back
//                       (-DDMA_BEAT_LOG2=2 for 4-byte beats, 4 for 16-byte beats)
// Comparing the two on a simulation with external-memory latency shows what the external
// accesses cost the kernel, which is the most DMA tiling can recover.
//
// Data is generated here (fixed LCG) so no test-side plumbing is needed; the output checksum lets
// a run be compared against another placement.

#include <cstdint>
#include <cstring>

#ifdef TILE_DMA
#include "fpga/sw/dma.h"
#endif
#include "sw/opt/litert-micro/conv.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/micro/kernels/conv.h"

using namespace coralnpu_v2::opt::litert_micro;

#if defined(TILE_IN_EXTMEM)
#define TILE_SECTION ".extdata"
#else
#define TILE_SECTION ".data"
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 14
#endif
constexpr int kWidth    = TILE_WIDTH;
constexpr int kInDepth  = 128;
constexpr int kOutDepth = 128;

#ifndef DMA_BEAT_LOG2
#define DMA_BEAT_LOG2 2
#endif

extern "C" {
#ifdef TILE_DMA
// The tile's home is external memory; the kernel works on copies in DTCM.
int8_t ext_input[kWidth * kInDepth] __attribute__((section(".extdata"), aligned(16)));
int8_t ext_filter[kOutDepth * kInDepth] __attribute__((section(".extdata"), aligned(16)));
int32_t ext_bias[kOutDepth] __attribute__((section(".extdata"), aligned(16)));
int8_t ext_output[kWidth * kOutDepth] __attribute__((section(".extdata"), aligned(16)));
#endif
int8_t tile_input[kWidth * kInDepth] __attribute__((section(TILE_SECTION), aligned(16)));
int8_t tile_filter[kOutDepth * kInDepth] __attribute__((section(TILE_SECTION), aligned(16)));
int32_t tile_bias[kOutDepth] __attribute__((section(TILE_SECTION), aligned(16)));
int8_t tile_output[kWidth * kOutDepth] __attribute__((section(TILE_SECTION), aligned(16)));

// Always in DTCM: scratch and results the host reads back.
int32_t per_channel_multiplier[kOutDepth] __attribute__((section(".data")));
int32_t per_channel_shift[kOutDepth] __attribute__((section(".data")));
int32_t accs_buf[64] __attribute__((section(".data"), aligned(16)));

// One result per 16-byte beat: the cocotb test host is 32 bits wide and reads only the first word
// of each 128-bit DTCM beat, so every result sits at the start of its own beat.
#define RESULT_VAR(type, name) type name __attribute__((section(".data"), used, aligned(16))) = 0
RESULT_VAR(uint32_t, tile_cycles_lo);
RESULT_VAR(uint32_t, tile_cycles_hi);
RESULT_VAR(uint32_t, tile_instret_lo);
RESULT_VAR(uint32_t, tile_instret_hi);
RESULT_VAR(uint32_t, dma_in_cycles);
RESULT_VAR(uint32_t, compute_cycles);
RESULT_VAR(uint32_t, dma_out_cycles);
RESULT_VAR(int32_t, tile_checksum);
RESULT_VAR(int32_t, tile_done);
}

static TfLiteStatus MockRequestScratchBuffer(TfLiteContext *, size_t, int *buffer_idx) {
  *buffer_idx = 0;
  return kTfLiteOk;
}
static void *MockGetScratchBuffer(TfLiteContext *, int) { return accs_buf; }

static uint32_t lcg_state = 12345;
static int8_t NextByte() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<int8_t>(lcg_state >> 24);
}

#ifdef TILE_DMA
// Where the tile data is created and where the final output is read from.
#define SRC_INPUT  ext_input
#define SRC_FILTER ext_filter
#define SRC_BIAS   ext_bias
#define SRC_OUTPUT ext_output

static struct dma_descriptor dma_desc[4] __attribute__((section(".data"), aligned(32)));

static void SetDescriptor(struct dma_descriptor *d, const void *src, void *dst, uint32_t bytes,
                          struct dma_descriptor *next) {
  d->src_addr  = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src));
  d->dst_addr  = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst));
  d->len_flags = dma_make_len_flags(bytes, DMA_BEAT_LOG2, 0, 0, 0);
  d->next_desc = next ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(next)) : 0;
  d->poll_addr = d->poll_mask = d->poll_value = d->reserved = 0;
}

// Returns false if the DMA reported an error or never finished.
static bool RunDma(struct dma_descriptor *first) {
  dma_start(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(first)));
  return dma_wait_done() == 0;
}
#else
#define SRC_INPUT  tile_input
#define SRC_FILTER tile_filter
#define SRC_BIAS   tile_bias
#define SRC_OUTPUT tile_output
#endif

int main(void) {
  for (int i = 0; i < kWidth * kInDepth; ++i)
    SRC_INPUT[i] = NextByte();
  for (int i = 0; i < kOutDepth * kInDepth; ++i)
    SRC_FILTER[i] = NextByte();
  for (int i = 0; i < kOutDepth; ++i) {
    SRC_BIAS[i]               = (i * 37) - 2000;
    per_channel_multiplier[i] = 0x40000000 + i * 4096;
    per_channel_shift[i]      = -8;
  }

  const int32_t input_dims[4]  = {1, 1, kWidth, kInDepth};
  const int32_t filter_dims[4] = {kOutDepth, 1, 1, kInDepth};
  const int32_t output_dims[4] = {1, 1, kWidth, kOutDepth};
  const int32_t bias_dims[1]   = {kOutDepth};
  const tflite::RuntimeShape input_shape(4, input_dims);
  const tflite::RuntimeShape filter_shape(4, filter_dims);
  const tflite::RuntimeShape output_shape(4, output_dims);
  const tflite::RuntimeShape bias_shape(1, bias_dims);

  tflite::ConvParams params;
  params.stride_width             = 1;
  params.stride_height            = 1;
  params.dilation_width_factor    = 1;
  params.dilation_height_factor   = 1;
  params.padding_values.width     = 0;
  params.padding_values.height    = 0;
  params.input_offset             = 128;
  params.weights_offset           = 0;
  params.output_offset            = -128;
  params.quantized_activation_min = -128;
  params.quantized_activation_max = 127;

  TfLiteContext context;
  context.GetScratchBuffer            = MockGetScratchBuffer;
  context.RequestScratchBufferInArena = MockRequestScratchBuffer;
  OpDataConvCustom data;
  memset(&data, 0, sizeof(data));
  data.per_channel_output_multiplier = per_channel_multiplier;
  data.per_channel_output_shift      = per_channel_shift;
  data.output_activation_min         = -128;
  data.output_activation_max         = 127;

  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
#ifdef TILE_DMA
  // One descriptor chain brings in the input row, the filter and the bias.
  SetDescriptor(&dma_desc[0], ext_input, tile_input, kWidth * kInDepth, &dma_desc[1]);
  SetDescriptor(&dma_desc[1], ext_filter, tile_filter, kOutDepth * kInDepth, &dma_desc[2]);
  SetDescriptor(&dma_desc[2], ext_bias, tile_bias, kOutDepth * sizeof(int32_t), nullptr);
  const bool in_ok = RunDma(&dma_desc[0]);
#endif
  const uint64_t t1 = mcycle_read();
  ConvPerChannel(params, data, per_channel_multiplier, per_channel_shift, &context, input_shape,
                 tile_input, filter_shape, tile_filter, bias_shape, tile_bias, output_shape,
                 tile_output);
  const uint64_t t2 = mcycle_read();
#ifdef TILE_DMA
  SetDescriptor(&dma_desc[3], tile_output, ext_output, kWidth * kOutDepth, nullptr);
  const bool out_ok = RunDma(&dma_desc[3]);
#endif
  const uint64_t t3      = mcycle_read();
  const uint64_t instret = minstret_read() - i0;
  const uint64_t cycles  = t3 - c0;
  tile_instret_lo        = static_cast<uint32_t>(instret);
  tile_instret_hi        = static_cast<uint32_t>(instret >> 32);
  tile_cycles_lo         = static_cast<uint32_t>(cycles);
  tile_cycles_hi         = static_cast<uint32_t>(cycles >> 32);
  dma_in_cycles          = static_cast<uint32_t>(t1 - c0);
  compute_cycles         = static_cast<uint32_t>(t2 - t1);
  dma_out_cycles         = static_cast<uint32_t>(t3 - t2);

  int32_t sum = 0;
  for (int i = 0; i < kWidth * kOutDepth; ++i)
    sum = sum * 31 + SRC_OUTPUT[i];
  tile_checksum = sum;
#ifdef TILE_DMA
  tile_done = (in_ok && out_ok) ? 0x600D : 0xBAD;
#else
  tile_done = 0x600D;
#endif
  return 0;
}
