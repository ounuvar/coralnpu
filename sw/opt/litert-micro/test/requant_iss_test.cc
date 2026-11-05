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

// ISS check for the fused requantization instructions (idea 2, #9, step 3).
// Runs unchanged on npusim (MPACT) and Spike: generates its own inputs,
// checks vrequant.vv and vrqclamp.vx against the reference model, and counts
// instructions for PostprocessAcc versus a fused postprocess loop.

#include <riscv_vector.h>

#include <cstdint>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/litert-micro/requant_ref.h"
#include "sw/utils/utils.h"

namespace rq = coralnpu_v2::opt::litert_micro;

namespace {

constexpr int kElems      = 256;  // instruction check
constexpr int kOutW       = 16;   // postprocess block: kOutW pixels x kOutD channels
constexpr int kOutD       = 64;
constexpr int32_t kActMin = -128, kActMax = 127;

uint32_t lcg_state = 12345;
uint32_t Rand() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return lcg_state;
}

// Packs the vrqclamp.vx scalar operand: {hi[31:24], lo[23:16], offset[15:0]}.
uint32_t PackClamp(int32_t offset, int32_t lo, int32_t hi) {
  return (static_cast<uint32_t>(hi & 0xff) << 24) | (static_cast<uint32_t>(lo & 0xff) << 16) |
         (static_cast<uint32_t>(offset) & 0xffff);
}

}  // namespace

extern "C" {
int32_t acc[kElems] __attribute__((aligned(16)));
int32_t mult[kElems] __attribute__((aligned(16)));
int32_t shift[kElems] __attribute__((aligned(16)));
int32_t got[kElems] __attribute__((aligned(16)));

int32_t accs[kOutW * kOutD] __attribute__((aligned(16)));
int32_t bias[kOutD] __attribute__((aligned(16)));
int32_t ch_mult[kOutD] __attribute__((aligned(16)));
int32_t ch_shift[kOutD] __attribute__((aligned(16)));
uint8_t shl[kOutD] __attribute__((aligned(16)));
uint8_t shr[kOutD] __attribute__((aligned(16)));
int8_t out_baseline[kOutW * kOutD] __attribute__((aligned(16)));
int8_t out_fused[kOutW * kOutD] __attribute__((aligned(16)));

// Results read by the test drivers.
volatile int32_t insn_mismatches __attribute__((used))   = -1;
volatile int32_t fused_mismatches __attribute__((used))  = -1;
volatile int32_t ties_checked __attribute__((used))      = 0;
// vrequant.vv alone (no clamp), and vrqclamp.vx alone, against the model.
volatile int32_t requant_mismatches __attribute__((used)) = -1;
volatile int32_t rqclamp_mismatches __attribute__((used)) = -1;
// First few instruction-check mismatches: {i, acc, mult, shift, offset, got, want}.
constexpr int kMaxBad = 8;
int32_t first_bad[kMaxBad][7] __attribute__((used));
volatile int32_t first_bad_count __attribute__((used)) = 0;
volatile uint32_t baseline_instret __attribute__((used)) = 0;
volatile uint32_t fused_instret __attribute__((used))    = 0;
volatile int32_t done __attribute__((used))              = 0;
}

// vd = vrequant.vv(vd = v8, vs2 = v16 multipliers, vs1 = v24 shifts), then
// vrqclamp.vx(v8, rs1 = packed). LMUL=8 so v8/v16/v24 are whole groups.
static void RequantClamp(const int32_t *in, const int32_t *m, const int32_t *s, uint32_t packed,
                         int32_t *out, int n) {
  while (n > 0) {
    size_t vl;
    asm volatile(
        "vsetvli %[vl], %[n], e32, m8, ta, ma\n"
        "vle32.v v8, (%[in])\n"
        "vle32.v v16, (%[m])\n"
        "vle32.v v24, (%[s])\n"
        ".insn r 0x57, 0x2, 0x27, x8, x24, x16\n"       // vrequant.vv v8, v16, v24
        ".insn r 0x57, 0x6, 0x23, x8, %[packed], x8\n"  // vrqclamp.vx v8, v8, packed
        "vse32.v v8, (%[out])\n"
        : [vl] "=&r"(vl)
        : [n] "r"(n), [in] "r"(in), [m] "r"(m), [s] "r"(s), [packed] "r"(packed), [out] "r"(out)
        : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
    in += vl;
    m += vl;
    s += vl;
    out += vl;
    n -= vl;
  }
}

// PostprocessAcc with the fused instructions: bias add, vrequant, vrqclamp,
// then two exact narrowings (the result is already in int8 range).
static void FusedPostprocess(const int32_t *in, const int32_t *bias_data, const int32_t *m,
                             const int32_t *s, uint32_t packed, int8_t *out, int out_w, int out_d) {
  for (int ch = 0; ch < out_d;) {
    size_t vl;
    asm volatile(
        "vsetvli %0, %1, e32, m8, ta, ma\n"
        "vle32.v v0, (%2)\n"
        "vle32.v v16, (%3)\n"
        "vle32.v v24, (%4)\n"
        : "=&r"(vl)
        : "r"(out_d - ch), "r"(bias_data + ch), "r"(m + ch), "r"(s + ch)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
    for (int x = 0; x < out_w; ++x) {
      asm volatile(
          "vsetvli zero, %[vl], e32, m8, ta, ma\n"
          "vle32.v v8, (%[in])\n"
          "vadd.vv v8, v8, v0\n"
          ".insn r 0x57, 0x2, 0x27, x8, x24, x16\n"       // vrequant.vv v8, v16, v24
          ".insn r 0x57, 0x6, 0x23, x8, %[packed], x8\n"  // vrqclamp.vx v8, v8, packed
          "vsetvli zero, %[vl], e16, m4, ta, ma\n"
          "vnsrl.wi v8, v8, 0\n"
          "vsetvli zero, %[vl], e8, m2, ta, ma\n"
          "vnsrl.wi v8, v8, 0\n"
          "vse8.v v8, (%[out])\n"
          :
          : [vl] "r"(vl), [in] "r"(in + x * out_d + ch), [packed] "r"(packed),
            [out] "r"(out + x * out_d + ch)
          : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory");
    }
    ch += vl;
  }
}

// One instruction over n elements at e32, m8: op 0 = vrequant.vv v8, v16, v24
// (in, m, s); op 1 = vrqclamp.vx v8, v8, packed (in only).
static void OneInsn(int op, const int32_t *in, const int32_t *m, const int32_t *s, uint32_t packed,
                    int32_t *out, int n) {
  while (n > 0) {
    size_t vl;
    asm volatile("vsetvli %[vl], %[n], e32, m8, ta, ma\n"
                 "vle32.v v8, (%[in])\n"
                 : [vl] "=&r"(vl)
                 : [n] "r"(n), [in] "r"(in)
                 : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory");
    if (op == 0) {
      asm volatile(
          "vle32.v v16, (%[m])\n"
          "vle32.v v24, (%[s])\n"
          ".insn r 0x57, 0x2, 0x27, x8, x24, x16\n"  // vrequant.vv v8, v16, v24
          :
          : [m] "r"(m), [s] "r"(s)
          : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
            "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    } else {
      asm volatile(".insn r 0x57, 0x6, 0x23, x8, %[packed], x8\n"  // vrqclamp.vx v8, v8, packed
                   :
                   : [packed] "r"(packed)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
    }
    asm volatile("vse32.v v8, (%[out])\n" : : [out] "r"(out) : "memory");
    in += vl;
    m += vl;
    s += vl;
    out += vl;
    n -= vl;
  }
}

int main(void) {
  // 1. Instruction check: random values, plus exact rounding ties of both
  // signs (multiplier 2^30 halves acc exactly), saturation and extreme shifts.
  for (int i = 0; i < kElems; ++i) {
    const uint32_t r = Rand();
    switch (i % 4) {
      case 0:  // random, TFLite-like ranges
        acc[i]   = static_cast<int32_t>(Rand()) >> (r % 12);
        mult[i]  = static_cast<int32_t>((1u << 30) | (Rand() >> 2));
        shift[i] = static_cast<int32_t>(r % 39) - 31;  // -31..7
        break;
      case 1: {  // exact ties: acc / 2 = q * 2^e + 2^(e-1)
        const int e     = 1 + static_cast<int>(r % 12);
        const int32_t q = static_cast<int32_t>(Rand() % 601) - 300;
        acc[i]          = 2 * (q * (1 << e) + (1 << (e - 1)));
        mult[i]         = 1 << 30;
        shift[i]        = -e;
        break;
      }
      case 2:  // saturation and wrap: extreme accumulators, out-of-range shifts
        acc[i]   = (r & 1) ? INT32_MIN : INT32_MAX - static_cast<int32_t>(r % 3);
        mult[i]  = (r & 2) ? INT32_MIN : INT32_MAX;
        shift[i] = static_cast<int32_t>(r % 80) - 40;
        break;
      default:  // full random
        acc[i]   = static_cast<int32_t>(Rand());
        mult[i]  = static_cast<int32_t>(Rand());
        shift[i] = static_cast<int32_t>(r % 63) - 31;
        break;
    }
  }
  const int32_t offsets[] = {-128, 0, 37};
  int32_t bad = 0, ties = 0;
  for (int32_t offset : offsets) {
    RequantClamp(acc, mult, shift, PackClamp(offset, kActMin, kActMax), got, kElems);
    for (int i = 0; i < kElems; ++i) {
      const int32_t want = rq::RequantRef(acc[i], mult[i], shift[i], offset, kActMin, kActMax);
      if (got[i] != want && first_bad_count < kMaxBad) {
        int32_t *rec = first_bad[first_bad_count];
        rec[0]       = i;
        rec[1]       = acc[i];
        rec[2]       = mult[i];
        rec[3]       = shift[i];
        rec[4]       = offset;
        rec[5]       = got[i];
        rec[6]       = want;
        first_bad_count = first_bad_count + 1;
      }
      bad += got[i] != want;
      ties += (i % 4 == 1) && want > kActMin && want < kActMax;
    }
  }
  insn_mismatches = bad;
  ties_checked    = ties;

  // Each instruction on its own. No clamp: offset 0 and the full int32 range.
  OneInsn(0, acc, mult, shift, 0, got, kElems);
  bad = 0;
  for (int i = 0; i < kElems; ++i) {
    bad += got[i] != rq::RequantRef(acc[i], mult[i], shift[i], 0, INT32_MIN, INT32_MAX);
  }
  requant_mismatches = bad;
  bad = 0;
  for (int32_t offset : offsets) {
    OneInsn(1, acc, nullptr, nullptr, PackClamp(offset, kActMin, kActMax), got, kElems);
    for (int i = 0; i < kElems; ++i) {
      int32_t want = static_cast<int32_t>(static_cast<uint32_t>(acc[i]) + static_cast<uint32_t>(offset));
      want         = want < kActMin ? kActMin : (want > kActMax ? kActMax : want);
      bad += got[i] != want;
    }
  }
  rqclamp_mismatches = bad;

  // 2. Postprocess block: existing PostprocessAcc vs the fused loop.
  for (int c = 0; c < kOutD; ++c) {
    bias[c]     = static_cast<int32_t>(Rand() % 2001) - 1000;
    ch_mult[c]  = static_cast<int32_t>((1u << 30) | (Rand() >> 2));
    ch_shift[c] = -static_cast<int32_t>(2 + Rand() % 9);
  }
  for (int i = 0; i < kOutW * kOutD; ++i) {
    accs[i] = static_cast<int32_t>(Rand() % 200001) - 100000;
  }
  rq::PrepareShiftParams(shl, shr, ch_shift, kOutD);
  const int32_t out_offset = 5;

  uint32_t i0 = static_cast<uint32_t>(minstret_read());
  rq::PostprocessAcc(accs, bias, shl, ch_mult, shr, out_offset, kActMin, kActMax, out_baseline,
                     kOutW, kOutD);
  baseline_instret = static_cast<uint32_t>(minstret_read()) - i0;

  i0 = static_cast<uint32_t>(minstret_read());
  FusedPostprocess(accs, bias, ch_mult, ch_shift, PackClamp(out_offset, kActMin, kActMax),
                   out_fused, kOutW, kOutD);
  fused_instret = static_cast<uint32_t>(minstret_read()) - i0;

  bad = 0;
  for (int x = 0; x < kOutW; ++x) {
    for (int c = 0; c < kOutD; ++c) {
      const int32_t want = rq::RequantRef(accs[x * kOutD + c] + bias[c], ch_mult[c], ch_shift[c],
                                          out_offset, kActMin, kActMax);
      bad += out_fused[x * kOutD + c] != want;
    }
  }
#if CORALNPU_FUSED_REQUANT
  // With --//sw/opt/litert-micro:fused_requant, PostprocessAcc itself takes
  // the fused path, so it must match the reference too (offset 5 keeps
  // negative rounding ties in range).
  for (int x = 0; x < kOutW; ++x) {
    for (int c = 0; c < kOutD; ++c) {
      const int32_t want = rq::RequantRef(accs[x * kOutD + c] + bias[c], ch_mult[c], ch_shift[c],
                                          out_offset, kActMin, kActMax);
      bad += out_baseline[x * kOutD + c] != want;
    }
  }
#endif  // CORALNPU_FUSED_REQUANT
  fused_mismatches = bad;
  done             = 1;
  return 0;
}
