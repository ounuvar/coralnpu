// Copyright 2025 Google LLC
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

#ifndef SW_OPT_LITERT_MICRO_ACCUMULATOR_UTIL_H_
#define SW_OPT_LITERT_MICRO_ACCUMULATOR_UTIL_H_

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

#if TFLITE_SINGLE_ROUNDING
#error "TFLITE_SINGLE_ROUNDING is not supported"
#endif  // TFLITE_SINGLE_ROUNDING

namespace coralnpu_v2::opt::litert_micro {
inline void PrepareShiftParams(uint8_t *left, uint8_t *right, const int32_t *shift_in, int out_d) {
  int out_ch        = 0;
  size_t out_ch_rem = out_d;
  while (out_ch_rem > 0) {
    const size_t vl          = __riscv_vsetvl_e32m8(out_ch_rem);
    const vint32m8_t shift32 = __riscv_vle32_v_i32m8(&shift_in[out_ch], vl);
    const vint16m4_t shift16 = __riscv_vncvt_x_x_w_i16m4(shift32, vl);
    const vint8m2_t shift8   = __riscv_vncvt_x_x_w_i8m2(shift16, vl);
    const vint8m2_t neg      = __riscv_vneg_v_i8m2(shift8, vl);
    // Positive values shift to LEFT, negative shift to right.
    const vuint8m2_t shl = __riscv_vreinterpret_v_i8m2_u8m2(__riscv_vmax_vx_i8m2(shift8, 0, vl));
    const vuint8m2_t shr = __riscv_vreinterpret_v_i8m2_u8m2(__riscv_vmax_vx_i8m2(neg, 0, vl));
    __riscv_vse8_v_u8m2(&left[out_ch], shl, vl);
    __riscv_vse8_v_u8m2(&right[out_ch], shr, vl);
    out_ch += vl;
    out_ch_rem -= vl;
  }
}

#if CORALNPU_FUSED_REQUANT
// Idea 2 (#9): PostprocessAcc with the fused requantization instructions.
// vrequant.vv does TFLite's MultiplyByQuantizedMultiplier (both shifts,
// ties away from zero) and vrqclamp.vx adds the output offset and clamps, so
// the result is already int8 and the two narrowings are exact. The
// instructions have no toolchain mnemonics, so the loop body is .insn with
// fixed register groups: v0 bias, v8 accumulators, v16 multipliers, v24
// shifts (e32, LMUL=8).
inline void PostprocessAccFused(const int32_t *accs, const int32_t *bias_data,
                                const uint8_t *lshift, const int32_t *multiplier,
                                const uint8_t *rshift, int32_t out_offset, int8_t out_min,
                                int8_t out_max, int8_t *out_data, int out_w, int out_d) {
  // vrqclamp.vx operand: {hi[31:24], lo[23:16], offset[15:0]}.
  const uint32_t clamp = (static_cast<uint32_t>(static_cast<uint8_t>(out_max)) << 24) |
                         (static_cast<uint32_t>(static_cast<uint8_t>(out_min)) << 16) |
                         (static_cast<uint32_t>(out_offset) & 0xffff);
  int32_t shift[32] __attribute__((aligned(16)));  // VLMAX at e32/m8, VLEN=128
  int out_ch = 0;
  while (out_ch < out_d) {
    const size_t vl = __riscv_vsetvl_e32m8(out_d - out_ch);
    // TFLite convention: positive shifts left, negative shifts right.
    const vint32m8_t lsh = __riscv_vreinterpret_v_u32m8_i32m8(
        __riscv_vzext_vf4_u32m8(__riscv_vle8_v_u8m2(&lshift[out_ch], vl), vl));
    const vint32m8_t rsh = __riscv_vreinterpret_v_u32m8_i32m8(
        __riscv_vzext_vf4_u32m8(__riscv_vle8_v_u8m2(&rshift[out_ch], vl), vl));
    __riscv_vse32_v_i32m8(shift, __riscv_vsub_vv_i32m8(lsh, rsh, vl), vl);
    if (bias_data) {
      asm volatile(
          "vsetvli zero, %0, e32, m8, ta, ma\n"
          "vle32.v v0, (%1)\n"
          :
          : "r"(vl), "r"(&bias_data[out_ch])
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
    } else {
      asm volatile(
          "vsetvli zero, %0, e32, m8, ta, ma\n"
          "vmv.v.i v0, 0\n"
          :
          : "r"(vl)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }
    asm volatile(
        "vle32.v v16, (%0)\n"
        "vle32.v v24, (%1)\n"
        :
        : "r"(&multiplier[out_ch]), "r"(shift)
        : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "v31", "memory");
    for (int out_x = 0; out_x < out_w; ++out_x) {
      asm volatile(
          "vsetvli zero, %[vl], e32, m8, ta, ma\n"
          "vle32.v v8, (%[in])\n"
          "vadd.vv v8, v8, v0\n"
          ".insn r 0x57, 0x2, 0x27, x8, x24, x16\n"      // vrequant.vv v8, v16, v24
          ".insn r 0x57, 0x6, 0x23, x8, %[clamp], x8\n"  // vrqclamp.vx v8, v8, clamp
          "vsetvli zero, %[vl], e16, m4, ta, ma\n"
          "vnsrl.wi v8, v8, 0\n"
          "vsetvli zero, %[vl], e8, m2, ta, ma\n"
          "vnsrl.wi v8, v8, 0\n"
          "vse8.v v8, (%[out])\n"
          :
          : [vl] "r"(vl), [in] "r"(&accs[out_x * out_d + out_ch]), [clamp] "r"(clamp),
            [out] "r"(&out_data[out_x * out_d + out_ch])
          : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory");
    }
    out_ch += vl;
  }
}
#endif  // CORALNPU_FUSED_REQUANT

// TODO(davidgao): use a param structure for reuse?
inline void PostprocessAcc(const int32_t *accs, const int32_t *bias_data, const uint8_t *lshift,
                           const int32_t *multiplier, const uint8_t *rshift, int32_t out_offset,
                           int8_t out_min, int8_t out_max, int8_t *out_data, int out_w, int out_d) {
#if CORALNPU_FUSED_REQUANT
  return PostprocessAccFused(accs, bias_data, lshift, multiplier, rshift, out_offset, out_min,
                             out_max, out_data, out_w, out_d);
#endif                          // CORALNPU_FUSED_REQUANT
  constexpr uint32_t vxrm = 0;  // round-to-nearest-up

  int out_ch        = 0;
  size_t out_ch_rem = out_d;
  while (out_ch_rem > 0) {
    const size_t vl = __riscv_vsetvl_e32m8(out_ch_rem);
    const vint32m8_t bias_val =
        bias_data ? __riscv_vle32_v_i32m8(&bias_data[out_ch], vl) : __riscv_vmv_v_x_i32m8(0, vl);
    const vint32m8_t mul_val = __riscv_vle32_v_i32m8(&multiplier[out_ch], vl);
    const vuint8m2_t lsh8    = __riscv_vle8_v_u8m2(&lshift[out_ch], vl);
    const vuint8m2_t rsh8    = __riscv_vle8_v_u8m2(&rshift[out_ch], vl);
    const vuint32m8_t lsh32  = __riscv_vzext_vf4_u32m8(lsh8, vl);
    const vuint32m8_t rsh32  = __riscv_vzext_vf4_u32m8(rsh8, vl);
    for (int out_x = 0; out_x < out_w; ++out_x) {
      vint32m8_t acc = __riscv_vle32_v_i32m8(&accs[out_x * out_d + out_ch], vl);
      // Apply bias
      acc = __riscv_vadd_vv_i32m8(acc, bias_val, vl);
      // int8 uses left shift BEFORE multiplier
      acc = __riscv_vsll_vv_i32m8(acc, lsh32, vl);
      acc = __riscv_vsmul_vv_i32m8(acc, mul_val, vxrm, vl);
      // int8 uses rounding right shift
      acc = __riscv_vssra_vv_i32m8(acc, rsh32, vxrm, vl);
      // Apply offset
      acc = __riscv_vadd_vx_i32m8(acc, out_offset, vl);
      // Narrow down, saturating
      const vint16m4_t out16 = __riscv_vnclip_wx_i16m4(acc, 0, vxrm, vl);
      vint8m2_t out8         = __riscv_vnclip_wx_i8m2(out16, 0, vxrm, vl);
      // Apply clamping
      out8 = __riscv_vmax_vx_i8m2(out8, out_min, vl);
      out8 = __riscv_vmin_vx_i8m2(out8, out_max, vl);
      // Write out
      __riscv_vse8_v_i8m2(&out_data[out_x * out_d + out_ch], out8, vl);
    }
    out_ch += vl;
    out_ch_rem -= vl;
  }
}

inline void PostprocessAcc16(const int32_t *accs, const int32_t *bias_data, const uint8_t *lshift,
                             const int32_t *multiplier, const uint8_t *rshift, int32_t out_offset,
                             int16_t out_min, int16_t out_max, int16_t *out_data, int out_w,
                             int out_d) {
  // Scalar post-processing to ensure absolute bit-exactness with TFLM.
  // The user requested correctness over performance for this stage.
  for (int out_x = 0; out_x < out_w; ++out_x) {
    for (int c = 0; c < out_d; ++c) {
      int64_t acc = (int64_t)accs[out_x * out_d + c];
      if (bias_data) {
        acc += (int64_t)bias_data[c];
      }

      int shift      = (int8_t)lshift[c] - (int8_t)rshift[c];
      int32_t result = tflite::MultiplyByQuantizedMultiplier(acc, multiplier[c], shift);

      result                      = std::max<int32_t>(result, out_min);
      result                      = std::min<int32_t>(result, out_max);
      out_data[out_x * out_d + c] = (int16_t)result;
    }
  }
}
}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_ACCUMULATOR_UTIL_H_
