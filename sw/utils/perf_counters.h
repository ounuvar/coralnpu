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

#ifndef SW_UTILS_PERF_COUNTERS_H_
#define SW_UTILS_PERF_COUNTERS_H_

// #59: memory stall counters at mhpmcounter3 + i (PerfCounters in
// hdl/chisel/src/coralnpu/scalar/Csr.scala). Build with
// --//sw/utils:perf_counters to read them; only cores built with
// enablePerfCounters have them (they trap elsewhere, including npusim).
// Without the flag every function here compiles to nothing, so binaries are
// unchanged.

#include <cstdint>

#ifndef CORALNPU_PERF_COUNTERS
#define CORALNPU_PERF_COUNTERS 0
#endif

namespace coralnpu_v2 {

// memStall, lsuFull, tcmWait, extWait, fetchWait, vecDataWait, noDispatch,
// rvvFull.
constexpr int kPerfCounters = 8;

#if CORALNPU_PERF_COUNTERS
#define CORALNPU_PERF_READ64(lo, hi, out)                                                     \
  do {                                                                                        \
    uint32_t h_, l_, h2_;                                                                     \
    asm volatile("1: csrr %0, " #hi "\n csrr %1, " #lo "\n csrr %2, " #hi "\n bne %0, %2, 1b" \
                 : "=r"(h_), "=r"(l_), "=r"(h2_));                                            \
    (out) = (static_cast<uint64_t>(h_) << 32) | l_;                                           \
  } while (0)

inline void PerfRead(uint64_t *v) {
  CORALNPU_PERF_READ64(0xb03, 0xb83, v[0]);
  CORALNPU_PERF_READ64(0xb04, 0xb84, v[1]);
  CORALNPU_PERF_READ64(0xb05, 0xb85, v[2]);
  CORALNPU_PERF_READ64(0xb06, 0xb86, v[3]);
  CORALNPU_PERF_READ64(0xb07, 0xb87, v[4]);
  CORALNPU_PERF_READ64(0xb08, 0xb88, v[5]);
  CORALNPU_PERF_READ64(0xb09, 0xb89, v[6]);
  CORALNPU_PERF_READ64(0xb0a, 0xb8a, v[7]);
}
#undef CORALNPU_PERF_READ64
#else
inline void PerfRead(uint64_t *) {}
#endif  // CORALNPU_PERF_COUNTERS

// Snapshot before a region; PerfEnd stores the per-counter deltas.
inline void PerfBegin(uint64_t *start) { PerfRead(start); }

inline void PerfEnd(const uint64_t *start, uint64_t *delta) {
#if CORALNPU_PERF_COUNTERS
  uint64_t now[kPerfCounters];
  PerfRead(now);
  for (int i = 0; i < kPerfCounters; ++i) {
    delta[i] = now[i] - start[i];
  }
#else
  (void)start;
  (void)delta;
#endif
}

}  // namespace coralnpu_v2

#endif  // SW_UTILS_PERF_COUNTERS_H_
