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

// Golden tests for the requantization reference model (issue #9, step 2):
// bit-exact against TFLite's double-rounding MultiplyByQuantizedMultiplier.

#include "sw/opt/litert-micro/requant_ref.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "fixedpoint/fixedpoint.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/kernels/internal/common.h"

#if TFLITE_SINGLE_ROUNDING
#error "The reference model targets double rounding (TFLITE_SINGLE_ROUNDING=0)"
#endif  // TFLITE_SINGLE_ROUNDING

namespace coralnpu_v2::opt::litert_micro {
namespace {

// What the TFLite reference kernels compute per output element.
int32_t TfliteRequant(int32_t acc, int32_t multiplier, int32_t shift, int32_t output_offset,
                      int32_t act_min, int32_t act_max) {
  int32_t y = tflite::MultiplyByQuantizedMultiplier(acc, multiplier, shift);
  y += output_offset;
  return std::min(std::max(y, act_min), act_max);
}

// True when TFLite's int32 arithmetic for these inputs is free of signed
// overflow (undefined behavior), so its result is a valid golden value.
bool TfliteIsDefined(int32_t acc, int32_t multiplier, int32_t shift, int32_t output_offset) {
  if (shift > 0) {
    const int64_t shifted = static_cast<int64_t>(acc) * (int64_t{1} << shift);
    if (shifted != static_cast<int32_t>(shifted))
      return false;
  }
  const int64_t y   = tflite::MultiplyByQuantizedMultiplier(acc, multiplier, shift);
  const int64_t sum = y + output_offset;
  return sum == static_cast<int32_t>(sum);
}

constexpr int32_t kInt8Min = std::numeric_limits<int8_t>::min();
constexpr int32_t kInt8Max = std::numeric_limits<int8_t>::max();

TEST(RequantRefTest, DoublingHighMulMatchesGemmlowp) {
  std::mt19937 rng(1);
  std::uniform_int_distribution<int32_t> any;
  const int32_t edges[] = {0,       1,          -1,        2,         -2,
                           1 << 30, -(1 << 30), INT32_MAX, INT32_MIN, INT32_MIN + 1};
  for (int32_t a : edges) {
    for (int32_t b : edges) {
      EXPECT_EQ(RequantDoublingHighMul(a, b), gemmlowp::SaturatingRoundingDoublingHighMul(a, b))
          << a << " * " << b;
    }
  }
  for (int i = 0; i < 1000000; ++i) {
    const int32_t a = any(rng), b = any(rng);
    ASSERT_EQ(RequantDoublingHighMul(a, b), gemmlowp::SaturatingRoundingDoublingHighMul(a, b))
        << a << " * " << b;
  }
}

TEST(RequantRefTest, RoundingRightShiftMatchesGemmlowp) {
  std::mt19937 rng(2);
  std::uniform_int_distribution<int32_t> any;
  for (int e = 0; e <= 31; ++e) {
    // Every remainder pattern around zero, including exact ties of both signs.
    for (int32_t x = -4096; x <= 4096; ++x) {
      ASSERT_EQ(RequantRoundingRightShift(x, e), gemmlowp::RoundingDivideByPOT(x, e))
          << x << " >> " << e;
    }
    for (int32_t x : {INT32_MIN, INT32_MIN + 1, INT32_MAX, INT32_MAX - 1}) {
      ASSERT_EQ(RequantRoundingRightShift(x, e), gemmlowp::RoundingDivideByPOT(x, e))
          << x << " >> " << e;
    }
    for (int i = 0; i < 20000; ++i) {
      const int32_t x = any(rng);
      ASSERT_EQ(RequantRoundingRightShift(x, e), gemmlowp::RoundingDivideByPOT(x, e))
          << x << " >> " << e;
    }
  }
}

// Documents why the fused instruction cannot reuse vssra's rounding: RVV
// round-to-nearest-up sends negative ties toward +infinity, gemmlowp away from
// zero. The existing kernels (vssra, or vsra plus a 1 << (e-1) nudge) round
// like VssraRnu.
TEST(RequantRefTest, NegativeTiesDifferFromVssraRnu) {
  auto vssra_rnu = [](int32_t x, int e) {
    return e == 0 ? x : static_cast<int32_t>((x >> e) + ((x >> (e - 1)) & 1));
  };
  EXPECT_EQ(RequantRoundingRightShift(-3, 1), -2);              // -1.5 -> -2
  EXPECT_EQ(vssra_rnu(-3, 1), -1);                              // -1.5 -> -1
  EXPECT_EQ(RequantRoundingRightShift(3, 1), vssra_rnu(3, 1));  // positive ties agree
  for (int e = 1; e <= 10; ++e) {
    for (int32_t x = -2048; x <= 2048; ++x) {
      const bool negative_tie = x < 0 && (x & ((1 << e) - 1)) == (1 << (e - 1));
      EXPECT_EQ(RequantRoundingRightShift(x, e) != vssra_rnu(x, e), negative_tie)
          << x << " >> " << e;
    }
  }
}

TEST(RequantRefTest, MatchesTfliteRandomPerChannel) {
  std::mt19937 rng(3);
  // Ranges the TFLite converter produces: multipliers in [2^30, 2^31), shifts
  // mostly negative. Accumulators span the full int32 range.
  std::uniform_int_distribution<int32_t> acc_dist;
  std::uniform_int_distribution<int32_t> small_acc(-300000, 300000);
  std::uniform_int_distribution<int32_t> mult_dist(1 << 30, INT32_MAX);
  std::uniform_int_distribution<int32_t> shift_dist(-31, 7);
  std::uniform_int_distribution<int32_t> offset_dist(kInt8Min, kInt8Max);
  int checked = 0;
  for (int i = 0; i < 2000000; ++i) {
    const int32_t acc    = (i & 1) ? acc_dist(rng) : small_acc(rng);
    const int32_t mult   = mult_dist(rng);
    const int32_t shift  = shift_dist(rng);
    const int32_t offset = offset_dist(rng);
    if (!TfliteIsDefined(acc, mult, shift, offset))
      continue;
    ASSERT_EQ(RequantRef(acc, mult, shift, offset, kInt8Min, kInt8Max),
              TfliteRequant(acc, mult, shift, offset, kInt8Min, kInt8Max))
        << "acc=" << acc << " mult=" << mult << " shift=" << shift << " offset=" << offset;
    ++checked;
  }
  EXPECT_GT(checked, 1000000);
}

// Inputs whose result lands exactly on a rounding tie of the final right
// shift, for both signs. The existing conv test uses output_offset = -128,
// which clamps every negative tie away; these use offsets that keep them.
TEST(RequantRefTest, MatchesTfliteOnExactTies) {
  const int32_t mult = 1 << 30;  // SRDHM(x, 2^30) == x / 2 exactly for even x
  for (int e = 1; e <= 12; ++e) {
    for (int32_t q = -300; q <= 300; ++q) {
      // SRDHM(acc, 2^30) = acc / 2, so acc = 2 * (q * 2^e + 2^(e-1)) is a tie.
      const int32_t acc = 2 * (q * (1 << e) + (1 << (e - 1)));
      for (int32_t offset : {0, 5, 100, -50}) {
        ASSERT_EQ(RequantRef(acc, mult, -e, offset, kInt8Min, kInt8Max),
                  TfliteRequant(acc, mult, -e, offset, kInt8Min, kInt8Max))
            << "acc=" << acc << " shift=" << -e << " offset=" << offset;
      }
    }
  }
}

TEST(RequantRefTest, MatchesTfliteOnSaturationAndClamp) {
  const int32_t mult = 1518500250;  // ~0.707 in Q31
  // Accumulators far outside the int8 range clamp at both ends.
  for (int32_t acc : {INT32_MIN, INT32_MIN + 1, -100000000, 100000000, INT32_MAX - 1, INT32_MAX}) {
    for (int32_t shift : {-31, -20, -8, -1, 0}) {
      ASSERT_EQ(RequantRef(acc, mult, shift, 0, kInt8Min, kInt8Max),
                TfliteRequant(acc, mult, shift, 0, kInt8Min, kInt8Max))
          << "acc=" << acc << " shift=" << shift;
    }
  }
  // Fused activations narrow the clamp range (ReLU, ReLU6 at zero point -128).
  for (int32_t acc = -5000; acc <= 5000; acc += 7) {
    for (auto [lo, hi] :
         {std::pair{-128, 127}, std::pair{-128, -128 + 6 * 16}, std::pair{0, 127}}) {
      ASSERT_EQ(RequantRef(acc, mult, -4, -128, lo, hi), TfliteRequant(acc, mult, -4, -128, lo, hi))
          << "acc=" << acc << " lo=" << lo << " hi=" << hi;
    }
  }
}

TEST(RequantRefTest, MatchesTfliteOnShiftRange) {
  std::mt19937 rng(4);
  std::uniform_int_distribution<int32_t> acc_dist(-(1 << 20), 1 << 20);
  std::uniform_int_distribution<int32_t> mult_dist(1 << 30, INT32_MAX);
  for (int32_t shift = -31; shift <= 10; ++shift) {
    for (int i = 0; i < 20000; ++i) {
      const int32_t acc = acc_dist(rng), mult = mult_dist(rng);
      if (!TfliteIsDefined(acc, mult, shift, 0))
        continue;
      ASSERT_EQ(RequantRef(acc, mult, shift, 0, INT32_MIN, INT32_MAX),
                TfliteRequant(acc, mult, shift, 0, INT32_MIN, INT32_MAX))
          << "acc=" << acc << " mult=" << mult << " shift=" << shift;
    }
  }
  // Zero multiplier (scale 0) gives the offset.
  EXPECT_EQ(RequantRef(12345, 0, -5, 7, kInt8Min, kInt8Max), 7);
}

TEST(RequantRefTest, DefinedOutsideTfliteRange) {
  // Left shifts that overflow int32 wrap, as TFLite's x * (1 << shift) does
  // when compiled; TFLite itself leaves this undefined.
  EXPECT_EQ(RequantLeftShift(0x40000000, 1), INT32_MIN);
  EXPECT_EQ(RequantLeftShift(-1, 31), INT32_MIN);
  // Shifts beyond +-31 saturate.
  EXPECT_EQ(RequantRef(1000, 1 << 30, -100, 0, INT32_MIN, INT32_MAX),
            RequantRef(1000, 1 << 30, -31, 0, INT32_MIN, INT32_MAX));
  EXPECT_EQ(RequantRef(1, 1 << 30, 100, 0, INT32_MIN, INT32_MAX),
            RequantRef(1, 1 << 30, 31, 0, INT32_MIN, INT32_MAX));
}

// The per-channel loop as reference_integer_ops::ConvPerChannel runs it.
TEST(RequantRefTest, MatchesTflitePerChannelArrays) {
  std::mt19937 rng(5);
  constexpr int kChannels = 64, kPixels = 200;
  std::uniform_int_distribution<int32_t> mult_dist(1 << 30, INT32_MAX);
  std::uniform_int_distribution<int32_t> shift_dist(-12, 2);
  std::uniform_int_distribution<int32_t> acc_dist(-200000, 200000);
  std::vector<int32_t> mult(kChannels), shift(kChannels);
  for (int c = 0; c < kChannels; ++c) {
    mult[c]  = mult_dist(rng);
    shift[c] = shift_dist(rng);
  }
  for (int32_t offset : {-128, -3, 0, 17}) {
    for (int p = 0; p < kPixels; ++p) {
      for (int c = 0; c < kChannels; ++c) {
        const int32_t acc = acc_dist(rng);
        ASSERT_EQ(RequantRef(acc, mult[c], shift[c], offset, kInt8Min, kInt8Max),
                  TfliteRequant(acc, mult[c], shift[c], offset, kInt8Min, kInt8Max))
            << "c=" << c << " acc=" << acc << " offset=" << offset;
      }
    }
  }
}

}  // namespace
}  // namespace coralnpu_v2::opt::litert_micro
