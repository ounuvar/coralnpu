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

// #59: memory stall counters (mhpmcounter3 + i, see PerfCounters in
// hdl/chisel/src/coralnpu/scalar/Csr.scala). Runs five short loops and
// records, for each, the mcycle delta and the delta of every counter:
//   0 ALU only, 1 dependent loads from DTCM, 2 dependent loads from external
//   memory, 3 vector loads from external memory, 4 vector stores to external
//   memory.
// perf_counters_test.py checks the deltas. Needs a core built with
// enablePerfCounters (RvvCoreMiniPerfAxi).

#include <riscv_vector.h>
#include <stdint.h>

constexpr int kCounters = 8;
constexpr int kPhases   = 5;
constexpr int kWords    = 64;
constexpr int kChase    = 64;

// Column 0 is mcycle, columns 1..8 are the counters.
uint32_t results[kPhases][kCounters + 1] __attribute__((section(".data"), used));
volatile uint32_t sink __attribute__((section(".data"), used));

uint32_t dtcm_buf[kWords] __attribute__((section(".data"), aligned(16)));
uint32_t ext_buf[kWords] __attribute__((section(".extdata"), aligned(16)));

#define CSRR(csr)                                        \
  ({                                                     \
    uint32_t v_;                                         \
    asm volatile("csrr %0, " #csr : "=r"(v_)::"memory"); \
    v_;                                                  \
  })

static inline void Snapshot(uint32_t *s) {
  s[0] = CSRR(mcycle);
  s[1] = CSRR(0xb03);
  s[2] = CSRR(0xb04);
  s[3] = CSRR(0xb05);
  s[4] = CSRR(0xb06);
  s[5] = CSRR(0xb07);
  s[6] = CSRR(0xb08);
  s[7] = CSRR(0xb09);
  s[8] = CSRR(0xb0a);
}

static void Record(int phase, const uint32_t *before) {
  uint32_t after[kCounters + 1];
  Snapshot(after);
  for (int i = 0; i <= kCounters; ++i) {
    results[phase][i] = after[i] - before[i];
  }
}

// Pointer chase: each load's address depends on the previous load.
static uint32_t Chase(const uint32_t *buf) {
  uint32_t idx = 0;
  for (int i = 0; i < kChase; ++i) {
    idx = buf[idx];
  }
  return idx;
}

int main() {
  for (int i = 0; i < kWords; ++i) {
    const uint32_t next = (i * 17 + 5) % kWords;
    dtcm_buf[i]         = next;
    ext_buf[i]          = next;
  }
  uint32_t before[kCounters + 1];

  // 0: ALU only.
  Snapshot(before);
  uint32_t acc = 1;
  for (int i = 0; i < kChase; ++i) {
    asm volatile("addi %0, %0, 3\n xor %0, %0, %1" : "+r"(acc) : "r"(i));
  }
  Record(0, before);
  sink = acc;

  // 1: dependent loads from DTCM.
  Snapshot(before);
  sink = Chase(dtcm_buf);
  Record(1, before);

  // 2: dependent loads from external memory.
  Snapshot(before);
  sink = Chase(ext_buf);
  Record(2, before);

  // 3: vector loads from external memory.
  Snapshot(before);
  vuint32m1_t vacc = __riscv_vmv_v_x_u32m1(0, 4);
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < kWords; i += 4) {
      vacc = __riscv_vadd_vv_u32m1(vacc, __riscv_vle32_v_u32m1(&ext_buf[i], 4), 4);
    }
  }
  Record(3, before);
  sink = __riscv_vmv_x_s_u32m1_u32(vacc);

  // 4: vector stores to external memory.
  Snapshot(before);
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < kWords; i += 4) {
      __riscv_vse32_v_u32m1(&ext_buf[i], vacc, 4);
    }
  }
  Record(4, before);

  return 0;
}
