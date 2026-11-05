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
//
// A DMA moves data at some unknown time between Start() and Wait(), so code that is correct on the
// DMA is correct whether the copies happen at Start() or at Wait(). The memcpy backend copies at
// Start(); with CORALNPU_COPY_AT_WAIT it copies at Wait(), the other extreme. Tests run both.

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
// of it and it does not exceed `max_log2`: the number of trailing zero bits of the three together.
constexpr uint32_t BeatLog2(uintptr_t src, uintptr_t dst, uint32_t bytes,
                            uint32_t max_log2 = CORALNPU_DMA_BEAT_LOG2) {
  const uint32_t all = static_cast<uint32_t>(src | dst | bytes);
  if (all == 0)
    return max_log2;
  const uint32_t low = static_cast<uint32_t>(__builtin_ctz(all));
  return low < max_log2 ? low : max_log2;
}

// The DMA descriptor's length and flags word for a plain copy (no fixed addresses, no polling), the
// same value as dma_make_len_flags(bytes, beat_log2, 0, 0, 0): length in bits 0 to 23, beat size
// in bits 24 to 26. Computed inline because that function is a call in another translation unit.
constexpr uint32_t CopyLenFlags(uint32_t bytes, uint32_t beat_log2) {
  return (bytes & 0xFFFFFFu) | ((beat_log2 & 0x7u) << 24);
}

// Up to `capacity` copies, run back to back. The storage for the copies (DMA descriptors) is
// provided by the caller, so it can sit in a DTCM arena sized for the plan: StorageBytes(capacity)
// bytes, aligned to 32, not modified or freed while a transfer is running. A chain being built is
// separate from a chain being run, so double buffering uses two chains.
class CopyChain {
 public:
  // The descriptors are 32 bytes, and so is the memcpy backend's entry at most.
  static constexpr uint32_t StorageBytes(int capacity) {
    return 32u * static_cast<uint32_t>(capacity);
  }

  // Uses `storage` for up to `capacity` copies and empties the chain. DMA: zeroes the storage, so
  // that the descriptor fields no copy sets (polling, reserved) are zero.
  void Init(void *storage, int capacity) {
    capacity_ = capacity;
    count_    = 0;
#ifdef CORALNPU_DMA_TILING
    desc_ = static_cast<dma_descriptor *>(storage);
    std::memset(storage, 0, StorageBytes(capacity));
#else
    entries_ = static_cast<Entry *>(storage);
    pending_ = false;
#endif
  }

  void Clear() { count_ = 0; }
  int size() const { return count_; }
  int capacity() const { return capacity_; }

  // Appends one copy. Returns false (and adds nothing) if the chain is full or the copy is empty
  // or longer than the engine's 24-bit length field.
  bool Add(const void *src, void *dst, uint32_t bytes) {
    if (count_ >= capacity_ || bytes == 0 || bytes > kMaxBytes)
      return false;
#ifdef CORALNPU_DMA_TILING
    dma_descriptor &d = desc_[count_];
    d.src_addr        = Addr(src);
    d.dst_addr        = Addr(dst);
    d.len_flags       = CopyLenFlags(bytes, BeatLog2(Addr(src), Addr(dst), bytes));
    d.next_desc       = 0;
    // poll_addr, poll_mask, poll_value and reserved stay zero: Init zeroed them and no copy sets
    // them.
    if (count_ > 0)
      desc_[count_ - 1].next_desc = Addr(&d);
#else
    entries_[count_] = {src, dst, bytes};
#endif
    ++count_;
    return true;
  }

  // Starts the chain. DMA: returns immediately and the copies proceed in the background; call
  // Wait() before using the destinations or reusing the sources. memcpy: copies now, or at Wait()
  // with CORALNPU_COPY_AT_WAIT.
  void Start() {
    if (count_ == 0)
      return;
#ifdef CORALNPU_DMA_TILING
    dma_start(Addr(&desc_[0]));
#elif defined(CORALNPU_COPY_AT_WAIT)
    pending_ = true;
#else
    CopyNow();
#endif
  }

  // Waits for the started chain. Returns false if the DMA reported an error or never finished.
  bool Wait() {
    if (count_ == 0)
      return true;
#ifdef CORALNPU_DMA_TILING
    return dma_wait_done() == 0;
#else
    if (pending_)
      CopyNow();
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
  int capacity_                       = 0;
  int count_                          = 0;
#ifdef CORALNPU_DMA_TILING
  static uint32_t Addr(const void *p) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
  }
  dma_descriptor *desc_ = nullptr;
#else
  struct Entry {
    const void *src;
    void *dst;
    uint32_t n;
  };
  static_assert(sizeof(Entry) <= 32, "an entry must fit in the 32 bytes StorageBytes gives it");
  void CopyNow() {
    for (int i = 0; i < count_; ++i)
      std::memcpy(entries_[i].dst, entries_[i].src, entries_[i].n);
    pending_ = false;
  }
  Entry *entries_ = nullptr;
  bool pending_   = false;
#endif
};

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_DMA_TILE_H_
