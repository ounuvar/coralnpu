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

// Copy interface for DMA tiling (#11): a chain of copies that start together and finish together.
//
// With CORALNPU_DMA_TILING defined the chain runs on the DMA engine (SoC builds). Without it the
// same interface runs as plain memcpy, so functional tests on npusim, which has no DMA engine,
// exercise the same tiling code. The default build does not include this header's DMA code.
//
// The DMA engine has no 2D or strided mode, so every contiguous run is its own copy.

#ifndef SW_OPT_LITERT_MICRO_DMA_TILE_H_
#define SW_OPT_LITERT_MICRO_DMA_TILE_H_

#include <cstdint>
#include <cstring>

#ifdef CORALNPU_DMA_TILING
#include "fpga/sw/dma.h"
#endif

// Widest DMA beat the chain may use, log2 of bytes: 4 means 16-byte beats. The engine accepts
// 0 (1 B) to 4 (16 B); each copy uses the widest beat that source, destination and length allow.
#ifndef CORALNPU_DMA_BEAT_LOG2
#define CORALNPU_DMA_BEAT_LOG2 4
#endif

namespace coralnpu_v2::opt::litert_micro {

// Largest beat size, as log2 of bytes, such that source, destination and length are all multiples
// of it and it does not exceed `max_log2`.
constexpr uint32_t BeatLog2(uintptr_t src, uintptr_t dst, uint32_t bytes,
                            uint32_t max_log2 = CORALNPU_DMA_BEAT_LOG2) {
  uint32_t log2 = max_log2;
  while (log2 > 0) {
    const uintptr_t mask = (static_cast<uintptr_t>(1) << log2) - 1;
    if (((src | dst | bytes) & mask) == 0)
      break;
    --log2;
  }
  return log2;
}

// Up to kMaxCopies copies, run back to back. Descriptors (DMA) live in this object, so it must sit
// in DTCM (a static or a local) and must not be modified or destroyed while a transfer is running.
// A chain being built is separate from a chain being run, so double buffering uses two chains.
template <int kMaxCopies>
class alignas(32) CopyChain {
 public:
  void Clear() { count_ = 0; }
  int size() const { return count_; }

  // Appends one copy. Returns false (and adds nothing) if the chain is full or the copy is empty
  // or longer than the engine's 24-bit length field.
  bool Add(const void *src, void *dst, uint32_t bytes) {
    if (count_ >= kMaxCopies || bytes == 0 || bytes > kMaxBytes)
      return false;
#ifdef CORALNPU_DMA_TILING
    dma_descriptor &d = desc_[count_];
    d.src_addr        = Addr(src);
    d.dst_addr        = Addr(dst);
    d.len_flags       = dma_make_len_flags(bytes, BeatLog2(Addr(src), Addr(dst), bytes), 0, 0, 0);
    d.next_desc       = 0;
    d.poll_addr = d.poll_mask = d.poll_value = d.reserved = 0;
    if (count_ > 0)
      desc_[count_ - 1].next_desc = Addr(&d);
#else
    entries_[count_] = {src, dst, bytes};
#endif
    ++count_;
    return true;
  }

  // Starts the chain. DMA: returns immediately and the copies proceed in the background; call
  // Wait() before using the destinations or reusing the sources. memcpy: copies now.
  void Start() {
    if (count_ == 0)
      return;
#ifdef CORALNPU_DMA_TILING
    dma_start(Addr(&desc_[0]));
#else
    for (int i = 0; i < count_; ++i)
      std::memcpy(entries_[i].dst, entries_[i].src, entries_[i].n);
#endif
  }

  // Waits for the started chain. Returns false if the DMA reported an error or never finished.
  bool Wait() {
    if (count_ == 0)
      return true;
#ifdef CORALNPU_DMA_TILING
    return dma_wait_done() == 0;
#else
    return true;
#endif
  }

  // Start and wait.
  bool Run() {
    Start();
    return Wait();
  }

 private:
  static constexpr uint32_t kMaxBytes = 0xFFFFFF;
  int count_                          = 0;
#ifdef CORALNPU_DMA_TILING
  static uint32_t Addr(const void *p) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
  }
  dma_descriptor desc_[kMaxCopies];
#else
  struct Entry {
    const void *src;
    void *dst;
    uint32_t n;
  };
  Entry entries_[kMaxCopies];
#endif
};

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_DMA_TILE_H_
