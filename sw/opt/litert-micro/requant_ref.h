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

// Scalar reference model for the fused requantization instruction (idea 2,
// issue #9). It is the golden model for the ISS patches and the RTL, so it is
// written from bit-level definitions instead of calling gemmlowp, and builds
// for both the host and CoralNPU.
//
// One element of the instruction computes, for SEW=32:
//
//   y = MultiplyByQuantizedMultiplier(acc, multiplier, shift)  // TFLite,
//                                                              // double rounding
//   y = clamp(y + output_offset, act_min, act_max)
//
// which is what reference_integer_ops::ConvPerChannel and friends do per
// output element after the bias is added.

#ifndef SW_OPT_LITERT_MICRO_REQUANT_REF_H_
#define SW_OPT_LITERT_MICRO_REQUANT_REF_H_

#include <cstdint>

namespace coralnpu_v2::opt::litert_micro {

// The shift operand is a signed int32 in TFLite's convention (positive shifts
// left before the multiply, negative shifts right after it). The instruction
// saturates it to this range, which covers every shift TFLite produces.
inline constexpr int32_t kRequantMinShift = -31;
inline constexpr int32_t kRequantMaxShift = 31;

// acc << left_shift, wrapping modulo 2^32. TFLite computes x * (1 << shift) in
// int32; this is what that compiles to.
inline int32_t RequantLeftShift(int32_t acc, int left_shift) {
  return static_cast<int32_t>(static_cast<uint32_t>(acc) << left_shift);
}

// gemmlowp SaturatingRoundingDoublingHighMul: round(a * b / 2^31) with ties
// toward +infinity, saturating the single overflow case a == b == INT32_MIN.
// Same result as RVV vsmul with vxrm = rnu.
inline int32_t RequantDoublingHighMul(int32_t a, int32_t b) {
  if (a == INT32_MIN && b == INT32_MIN)
    return INT32_MAX;
  const int64_t product = static_cast<int64_t>(a) * b;
  return static_cast<int32_t>((product + (int64_t{1} << 30)) >> 31);
}

// gemmlowp RoundingDivideByPOT: x / 2^exponent with ties away from zero. This
// differs from RVV vssra (vxrm = rnu), which rounds negative ties up.
inline int32_t RequantRoundingRightShift(int32_t x, int exponent) {
  if (exponent == 0)
    return x;
  const int32_t mask      = static_cast<int32_t>((int64_t{1} << exponent) - 1);
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

// One element of the fused requantization instruction.
inline int32_t RequantRef(int32_t acc, int32_t multiplier, int32_t shift, int32_t output_offset,
                          int32_t act_min, int32_t act_max) {
  const int32_t sat_shift = shift < kRequantMinShift
                                ? kRequantMinShift
                                : (shift > kRequantMaxShift ? kRequantMaxShift : shift);
  const int left_shift    = sat_shift > 0 ? sat_shift : 0;
  const int right_shift   = sat_shift > 0 ? 0 : -sat_shift;

  int32_t y = RequantDoublingHighMul(RequantLeftShift(acc, left_shift), multiplier);
  y         = RequantRoundingRightShift(y, right_shift);
  // The offset add wraps like the int32 addition in the TFLite kernels.
  y = static_cast<int32_t>(static_cast<uint32_t>(y) + static_cast<uint32_t>(output_offset));
  y = y < act_min ? act_min : y;
  y = y > act_max ? act_max : y;
  return y;
}

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_REQUANT_REF_H_
