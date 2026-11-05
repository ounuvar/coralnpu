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

// DMA-tiled pointwise (1x1, stride 1, no padding) convolution (#11).
//
// Runs the layer in tiles of output pixels and chunks of output channels, copying each tile into a
// DTCM arena with the copy chains of dma_tile.h and writing the result back. The copies run on the
// DMA engine (CORALNPU_DMA_TILING) or as memcpy. The kernel on each tile is the RVV pointwise
// kernel with CORALNPU_RVV_CONV, and the TFLite reference kernel otherwise.

#ifndef SW_OPT_LITERT_MICRO_CONV_TILED_H_
#define SW_OPT_LITERT_MICRO_CONV_TILED_H_

#include <cstdint>

#include "sw/opt/litert-micro/conv_tile_plan.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace coralnpu_v2::opt::litert_micro {

// Cycle counters for the tiled path, collected only when built with CORALNPU_TILING_STATS (all
// zero otherwise). They add up over every call since the last reset.
struct ConvTiledStats {
  uint32_t chunks;         // output-channel chunks run
  uint32_t tiles;          // tiles run
  uint32_t prologue;       // blocking copies at the start of a chunk
  uint32_t prepare;        // shift and weight preparation
  uint32_t compute;        // the kernel
  uint32_t wait;           // waiting for a chain after the kernel finished (overlapped copies)
  uint32_t blocking_copy;  // blocking copies: single-buffer input and all output write-backs
  uint32_t build;          // building copy chains (descriptors) for the tiles, not waiting
};
void ConvTiledResetStats();
ConvTiledStats ConvTiledGetStats();

// True if the tiles are computed with the RVV kernel, false for the reference kernel.
bool ConvTiledUsesRvvKernel();

// Runs the layer tiled and returns true, or returns false without writing the output if it is not
// a pointwise conv this path handles or no tile plan fits `budget` (DTCM bytes for the arena and
// descriptors; at most CORALNPU_TILE_BUDGET_BYTES). `bias_data` may be null. `plan_out`, if not
// null, receives the plan that was used. On a DMA error it returns false after the output may have
// been partly written; the caller then recomputes the whole layer untiled.
bool ConvPointwiseTiledWithBudget(const tflite::ConvParams &params,
                                  const int32_t *output_multiplier, const int32_t *output_shift,
                                  const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                                  const tflite::RuntimeShape &filter_shape,
                                  const int8_t *filter_data, const int32_t *bias_data,
                                  const tflite::RuntimeShape &output_shape, int8_t *output_data,
                                  uint32_t budget, ConvTilePlan *plan_out);

// Same with the default budget.
bool ConvPointwiseTiled(const tflite::ConvParams &params, const int32_t *output_multiplier,
                        const int32_t *output_shift, const tflite::RuntimeShape &input_shape,
                        const int8_t *input_data, const tflite::RuntimeShape &filter_shape,
                        const int8_t *filter_data, const int32_t *bias_data,
                        const tflite::RuntimeShape &output_shape, int8_t *output_data);

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_CONV_TILED_H_
