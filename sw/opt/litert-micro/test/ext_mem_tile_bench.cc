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
// Runs tiles of MobileNet layer 17 (a 1x1 conv, 128 -> 128 channels; one tile is one output row of
// 14 pixels, 229,376 MACs) through the same ConvPerChannel dispatcher the model uses, timed with
// mcycle/minstret. Where the data lives is chosen at build time:
//   default             input, filter, bias and output all in DTCM (no external memory traffic)
//   -DTILE_IN_EXTMEM    all four in external memory (what an untiled kernel sees)
//   -DTILE_DMA          the data lives in external memory; the DMA copies the filter and bias in
//                       once and each input row into DTCM, the kernel runs there, and the DMA
//                       copies each output row back. The core waits for every copy.
//                       (-DDMA_BEAT_LOG2=2 for 4-byte beats, 4 for 16-byte beats)
//   -DTILE_DMA -DTILE_DOUBLE_BUFFER
//                       same copies, but overlapped with compute: while the kernel works on row r
//                       from one buffer, one DMA chain writes out row r-1 and reads in row r+1
//                       using the other buffers (ping-pong).
// -DTILE_ROWS=N runs N rows (default 1). Compare against the other placements on a simulation with
// external-memory latency to see what the external accesses cost and what tiling recovers.
//
// Data is generated here (fixed LCG) so no test-side plumbing is needed; the output checksum covers
// every row, so a run can be compared against another placement.

#include <cstdint>
#include <cstring>

#ifdef TILE_DMA
#include "fpga/sw/dma.h"
#endif
#include "sw/opt/litert-micro/conv.h"
#include "sw/utils/utils.h"
#ifdef TILE_CHECK_REFERENCE
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#endif
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
#ifndef TILE_ROWS
#define TILE_ROWS 1
#endif
constexpr int kWidth    = TILE_WIDTH;
constexpr int kRows     = TILE_ROWS;
constexpr int kInDepth  = 128;
constexpr int kOutDepth = 128;
constexpr int kRowIn    = kWidth * kInDepth;
constexpr int kRowOut   = kWidth * kOutDepth;

#ifndef DMA_BEAT_LOG2
#define DMA_BEAT_LOG2 2
#endif

extern "C" {
#ifdef TILE_DMA
// The data's home is external memory; the kernel works on copies in DTCM (two buffers for input
// and for output, so the DMA and the kernel never touch the same one when double buffering).
int8_t ext_input[kRows * kRowIn] __attribute__((section(".extdata"), aligned(16)));
int8_t ext_filter[kOutDepth * kInDepth] __attribute__((section(".extdata"), aligned(16)));
int32_t ext_bias[kOutDepth] __attribute__((section(".extdata"), aligned(16)));
int8_t ext_output[kRows * kRowOut] __attribute__((section(".extdata"), aligned(16)));
int8_t tile_input[2][kRowIn] __attribute__((section(".data"), aligned(16)));
int8_t tile_output[2][kRowOut] __attribute__((section(".data"), aligned(16)));
#else
int8_t tile_input[kRows * kRowIn] __attribute__((section(TILE_SECTION), aligned(16)));
int8_t tile_output[kRows * kRowOut] __attribute__((section(TILE_SECTION), aligned(16)));
#endif
int8_t tile_filter[kOutDepth * kInDepth] __attribute__((section(TILE_SECTION), aligned(16)));
int32_t tile_bias[kOutDepth] __attribute__((section(TILE_SECTION), aligned(16)));

// Always in DTCM: scratch and results the host reads back.
int32_t per_channel_multiplier[kOutDepth] __attribute__((section(".data")));
int32_t per_channel_shift[kOutDepth] __attribute__((section(".data")));
int32_t accs_buf[64] __attribute__((section(".data"), aligned(16)));
#ifdef TILE_CHECK_REFERENCE
// Output of the TFLite Micro reference kernel, compared byte for byte with the optimized kernel's.
int8_t ref_output[kRows * kRowOut] __attribute__((section(".data"), aligned(16)));
#endif

// One result per 16-byte beat: the cocotb test host is 32 bits wide and reads only the first word
// of each 128-bit DTCM beat, so every result sits at the start of its own beat.
#define RESULT_VAR(type, name) type name __attribute__((section(".data"), used, aligned(16))) = 0
RESULT_VAR(uint32_t, tile_cycles_lo);
RESULT_VAR(uint32_t, tile_cycles_hi);
RESULT_VAR(uint32_t, tile_instret_lo);
RESULT_VAR(uint32_t, tile_instret_hi);
RESULT_VAR(uint32_t, dma_in_cycles);     // core blocked on input copies
RESULT_VAR(uint32_t, compute_cycles);    // time in the kernel
RESULT_VAR(uint32_t, dma_out_cycles);    // core blocked on output copies
RESULT_VAR(uint32_t, dma_stall_cycles);  // double buffering: core waiting after compute finished
RESULT_VAR(int32_t, tile_checksum);
RESULT_VAR(int32_t, tile_done);
RESULT_VAR(uint32_t, tile_ref_mismatch);  // bytes where optimized and reference outputs differ
RESULT_VAR(int32_t, tile_ref_done);       // 0x600D once the reference comparison has run
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
// Where the data is created and where the final output is read from.
#define SRC_INPUT  ext_input
#define SRC_FILTER ext_filter
#define SRC_BIAS   ext_bias
#define SRC_OUTPUT ext_output

#ifdef TILE_DOUBLE_BUFFER
constexpr bool kDoubleBuffer = true;
#else
constexpr bool kDoubleBuffer = false;
#endif
// Which DTCM buffer holds row r.
static inline int BufferOf(int r) { return kDoubleBuffer ? (r & 1) : 0; }

// Descriptors live in DTCM: the DMA reads them from there. The prologue chain, one two-descriptor
// chain per buffer parity (the DMA may still be reading one while the next is being built), and a
// single descriptor for the blocking copies of the serial schedule.
static struct dma_descriptor dma_prologue[3] __attribute__((section(".data"), aligned(32)));
#ifdef TILE_DOUBLE_BUFFER
static struct dma_descriptor dma_chain[2][2] __attribute__((section(".data"), aligned(32)));
#endif
static struct dma_descriptor dma_single __attribute__((section(".data"), aligned(32)));

static uint32_t Addr(const void *p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

static void SetDescriptor(struct dma_descriptor *d, const void *src, void *dst, uint32_t bytes,
                          struct dma_descriptor *next) {
  d->src_addr  = Addr(src);
  d->dst_addr  = Addr(dst);
  d->len_flags = dma_make_len_flags(bytes, DMA_BEAT_LOG2, 0, 0, 0);
  d->next_desc = next ? Addr(next) : 0;
  d->poll_addr = d->poll_mask = d->poll_value = d->reserved = 0;
}

#ifdef TILE_DOUBLE_BUFFER
// Adds a copy to the end of a chain (first/last start out null).
static void Append(struct dma_descriptor **first, struct dma_descriptor **last,
                   struct dma_descriptor *d, const void *src, void *dst, uint32_t bytes) {
  SetDescriptor(d, src, dst, bytes, nullptr);
  if (*last)
    (*last)->next_desc = Addr(d);
  else
    *first = d;
  *last = d;
}
#endif

// Returns false if the DMA reported an error or never finished.
static bool RunDma(struct dma_descriptor *first) {
  dma_start(Addr(first));
  return dma_wait_done() == 0;
}
#else
#define SRC_INPUT  tile_input
#define SRC_FILTER tile_filter
#define SRC_BIAS   tile_bias
#define SRC_OUTPUT tile_output
#endif

int main(void) {
  for (int i = 0; i < kRows * kRowIn; ++i)
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

  auto compute = [&](const int8_t *in, int8_t *out) {
    ConvPerChannel(params, data, per_channel_multiplier, per_channel_shift, &context, input_shape,
                   in, filter_shape, tile_filter, bias_shape, tile_bias, output_shape, out);
  };

  uint64_t t_in = 0, t_compute = 0, t_out = 0, t_stall = 0;
  bool ok = true;

  const uint64_t c0 = mcycle_read();
  const uint64_t i0 = minstret_read();
#ifdef TILE_DMA
  // Prologue: one chain brings in the filter, the bias and input row 0.
  {
    SetDescriptor(&dma_prologue[0], ext_filter, tile_filter, kOutDepth * kInDepth,
                  &dma_prologue[1]);
    SetDescriptor(&dma_prologue[1], ext_bias, tile_bias, kOutDepth * sizeof(int32_t),
                  &dma_prologue[2]);
    SetDescriptor(&dma_prologue[2], ext_input, tile_input[BufferOf(0)], kRowIn, nullptr);
    const uint64_t a = mcycle_read();
    ok &= RunDma(&dma_prologue[0]);
    t_in += mcycle_read() - a;
  }
  for (int r = 0; r < kRows; ++r) {
#ifdef TILE_DOUBLE_BUFFER
    // One chain per row, moving data for the neighbouring rows while this row computes:
    // row r-1's output out to external memory, row r+1's input in.
    struct dma_descriptor *first = nullptr, *last = nullptr;
    struct dma_descriptor *d = dma_chain[r & 1];
    int n                    = 0;
    if (r > 0) {
      Append(&first, &last, &d[n++], tile_output[BufferOf(r - 1)], ext_output + (r - 1) * kRowOut,
             kRowOut);
    }
    if (r + 1 < kRows) {
      Append(&first, &last, &d[n++], ext_input + (r + 1) * kRowIn, tile_input[BufferOf(r + 1)],
             kRowIn);
    }
    if (first)
      dma_start(Addr(first));
    const uint64_t b = mcycle_read();
    compute(tile_input[BufferOf(r)], tile_output[BufferOf(r)]);
    const uint64_t c = mcycle_read();
    t_compute += c - b;
    if (first) {
      ok &= dma_wait_done() == 0;
      t_stall += mcycle_read() - c;
    }
#else
    if (r > 0) {
      SetDescriptor(&dma_single, ext_input + r * kRowIn, tile_input[BufferOf(r)], kRowIn, nullptr);
      const uint64_t a = mcycle_read();
      ok &= RunDma(&dma_single);
      t_in += mcycle_read() - a;
    }
    const uint64_t b = mcycle_read();
    compute(tile_input[BufferOf(r)], tile_output[BufferOf(r)]);
    const uint64_t c = mcycle_read();
    t_compute += c - b;
    SetDescriptor(&dma_single, tile_output[BufferOf(r)], ext_output + r * kRowOut, kRowOut,
                  nullptr);
    ok &= RunDma(&dma_single);
    t_out += mcycle_read() - c;
#endif
  }
#ifdef TILE_DOUBLE_BUFFER
  // Epilogue: the last row's output has not been copied out yet.
  {
    SetDescriptor(&dma_single, tile_output[BufferOf(kRows - 1)], ext_output + (kRows - 1) * kRowOut,
                  kRowOut, nullptr);
    const uint64_t a = mcycle_read();
    ok &= RunDma(&dma_single);
    t_out += mcycle_read() - a;
  }
#endif
#else
  for (int r = 0; r < kRows; ++r) {
    const uint64_t b = mcycle_read();
    compute(tile_input + r * kRowIn, tile_output + r * kRowOut);
    t_compute += mcycle_read() - b;
  }
#endif
  const uint64_t t3      = mcycle_read();
  const uint64_t instret = minstret_read() - i0;
  const uint64_t cycles  = t3 - c0;
  tile_instret_lo        = static_cast<uint32_t>(instret);
  tile_instret_hi        = static_cast<uint32_t>(instret >> 32);
  tile_cycles_lo         = static_cast<uint32_t>(cycles);
  tile_cycles_hi         = static_cast<uint32_t>(cycles >> 32);
  dma_in_cycles          = static_cast<uint32_t>(t_in);
  compute_cycles         = static_cast<uint32_t>(t_compute);
  dma_out_cycles         = static_cast<uint32_t>(t_out);
  dma_stall_cycles       = static_cast<uint32_t>(t_stall);

  int32_t sum = 0;
  for (int i = 0; i < kRows * kRowOut; ++i)
    sum = sum * 31 + SRC_OUTPUT[i];
  tile_checksum = sum;
  tile_done     = ok ? 0x600D : 0xBAD;

#ifdef TILE_CHECK_REFERENCE
  // Outside the timed region: the reference kernel on the same data, compared byte for byte.
  static_assert(kRows == 1, "the reference check covers one row");
  tflite::reference_integer_ops::ConvPerChannel(params, per_channel_multiplier, per_channel_shift,
                                                input_shape, tile_input, filter_shape, tile_filter,
                                                bias_shape, tile_bias, output_shape, ref_output);
  uint32_t mismatches = 0;
  for (int i = 0; i < kRowOut; ++i)
    mismatches += (ref_output[i] != tile_output[i]);
  tile_ref_mismatch = mismatches;
  tile_ref_done     = 0x600D;
#endif
  return 0;
}
