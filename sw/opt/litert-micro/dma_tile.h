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
constexpr uint32_t BeatLog2(uintptr_t src, uintptr_t dst, uint32_t bytes, uint32_t max_log2) {
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

// A chain of up to `capacity` copies, run back to back. The storage for the copies (DMA
// descriptors) is provided by the caller, so it can sit in a DTCM arena sized for the plan:
// StorageBytes(capacity) bytes, aligned to 32, not modified or freed while a transfer is running.
// A chain being built is separate from a chain being run, so double buffering uses two chains.
//
// Each backend is its own class, and CopyChain (below) names the one this build uses. One class
// whose members changed with the build flags would break the one-definition rule when two
// libraries built with different flags are linked together (#45 review).
//
// The interface every backend has:
//   Init(storage, capacity)   uses `storage` for up to `capacity` copies and empties the chain
//   Clear(), size(), capacity()
//   Add(src, dst, bytes)      appends one copy; false (adds nothing) if the chain is full or the
//                             copy is empty or longer than the engine's 24-bit length field
//   AddStrided(src, src_stride, dst, dst_stride, bytes, count)
//                             appends `count` equal copies: copy i goes from src + i * src_stride
//                             to dst + i * dst_stride. The DMA has no strided mode, so a strided
//                             transfer is a chain of equal copies; same result as `count` Adds
//   Start()                   starts the chain; the copies run in the background until Wait()
//   Wait()                    waits for the started chain; false if a copy failed. It never
//                             returns while the engine is still running.
//   Drain()                   Wait() if a chain was started and not yet waited for, else nothing.
//                             For early returns, so that no copy outlives the code that owns its
//                             buffers (see DrainOnExit).
//   Run()                     Start() then Wait()
class CopyChainBase {
 public:
  // The descriptors are 32 bytes, and so is the memcpy backend's entry at most.
  static constexpr uint32_t StorageBytes(int capacity) {
    return 32u * static_cast<uint32_t>(capacity);
  }
  void Clear() { count_ = 0; }
  int size() const { return count_; }
  int capacity() const { return capacity_; }

 protected:
  static constexpr uint32_t kMaxBytes = 0xFFFFFF;
  bool Fits(uint32_t bytes, int count) const {
    return count > 0 && bytes > 0 && bytes <= kMaxBytes && count <= capacity_ - count_;
  }
  int capacity_ = 0;
  int count_    = 0;
};

// memcpy backend, for builds without a DMA engine (npusim). It copies at Start(), or at Wait()
// with kCopyAtWait.
template <bool kCopyAtWait>
class MemcpyCopyChain : public CopyChainBase {
 public:
  void Init(void *storage, int capacity) {
    capacity_ = capacity;
    count_    = 0;
    entries_  = static_cast<Entry *>(storage);
    pending_  = false;
  }

  bool Add(const void *src, void *dst, uint32_t bytes) {
    if (!Fits(bytes, 1))
      return false;
    entries_[count_++] = {src, dst, bytes};
    return true;
  }

  bool AddStrided(const void *src, uint32_t src_stride, void *dst, uint32_t dst_stride,
                  uint32_t bytes, int count) {
    if (!Fits(bytes, count))
      return false;
    const uint8_t *cs = static_cast<const uint8_t *>(src);
    uint8_t *cd       = static_cast<uint8_t *>(dst);
    for (int i = 0; i < count; ++i) {
      entries_[count_ + i] = {cs, cd, bytes};
      cs += src_stride;
      cd += dst_stride;
    }
    count_ += count;
    return true;
  }

  void Start() {
    if (count_ == 0)
      return;
    if (kCopyAtWait)
      pending_ = true;
    else
      CopyNow();
  }

  bool Wait() {
    if (pending_)
      CopyNow();
    return true;
  }

  void Drain() { Wait(); }

  bool Run() {
    Start();
    return Wait();
  }

 private:
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
};

#ifdef CORALNPU_DMA_TILING
// DMA engine backend (SoC and F2 builds). Each copy uses the widest beat, up to 2^kMaxBeatLog2
// bytes, that its source, destination and length allow.
template <uint32_t kMaxBeatLog2>
class DmaCopyChain : public CopyChainBase {
 public:
  // Zeroes the storage, so that the descriptor fields no copy sets (polling, reserved) are zero.
  void Init(void *storage, int capacity) {
    capacity_ = capacity;
    count_    = 0;
    desc_     = static_cast<dma_descriptor *>(storage);
    running_  = false;
    std::memset(storage, 0, StorageBytes(capacity));
  }

  bool Add(const void *src, void *dst, uint32_t bytes) {
    if (!Fits(bytes, 1))
      return false;
    dma_descriptor &d = desc_[count_];
    d.src_addr        = Addr(src);
    d.dst_addr        = Addr(dst);
    d.len_flags       = CopyLenFlags(bytes, BeatLog2(Addr(src), Addr(dst), bytes, kMaxBeatLog2));
    d.next_desc       = 0;
    // poll_addr, poll_mask, poll_value and reserved stay zero: Init zeroed them and no copy sets
    // them.
    if (count_ > 0)
      desc_[count_ - 1].next_desc = Addr(&d);
    ++count_;
    return true;
  }

  bool AddStrided(const void *src, uint32_t src_stride, void *dst, uint32_t dst_stride,
                  uint32_t bytes, int count) {
    if (!Fits(bytes, count))
      return false;
    const int first = count_;
    // Every address is base + i * stride, so the alignment of all of them is the lowest set bit
    // of the bases, the strides and the length together.
    const uint32_t s = Addr(src), d = Addr(dst);
    uint32_t bits = s | d | bytes;
    if (count > 1)
      bits |= src_stride | dst_stride;
    const uint32_t flags       = CopyLenFlags(bytes, BeatLog2(0, 0, bits, kMaxBeatLog2));
    dma_descriptor *const base = desc_ + first;
    if (first > 0)
      desc_[first - 1].next_desc = Addr(base);
    uint32_t next = Addr(base) + 32;
    uint32_t cs = s, cd = d;
    dma_descriptor *p = base;
    for (int i = 0; i < count; ++i, ++p) {
      p->src_addr  = cs;
      p->dst_addr  = cd;
      p->len_flags = flags;
      p->next_desc = next;
      cs += src_stride;
      cd += dst_stride;
      next += 32;
    }
    (p - 1)->next_desc = 0;
    count_             = first + count;
    return true;
  }

  // Returns immediately; the copies proceed in the background. Call Wait() before using the
  // destinations or reusing the sources.
  void Start() {
    if (count_ == 0)
      return;
    dma_start(Addr(&desc_[0]));
    running_ = true;
  }

  // dma_wait_done() gives up after a bounded poll and returns -2 with the engine possibly still
  // running, and the engine has no working abort (DmaCsr: CTRL[2] is reserved). Returning then
  // would let the caller reuse buffers the DMA still writes, so on a timeout this keeps polling
  // STATUS.busy until the engine stops, and reports the failure after that.
  bool Wait() {
    if (!running_)
      return true;
    const int r = dma_wait_done();
    if (r == -2) {
      while (dma_get_status() & kStatusBusy) {
      }
    }
    running_ = false;
    return r == 0;
  }

  void Drain() { Wait(); }

  bool Run() {
    Start();
    return Wait();
  }

 private:
  static constexpr uint32_t kStatusBusy = 1u;  // STATUS[0], doc/peripherals/dma.md
  static uint32_t Addr(const void *p) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
  }
  dma_descriptor *desc_ = nullptr;
  bool running_         = false;
};
#endif  // CORALNPU_DMA_TILING

#if defined(CORALNPU_DMA_TILING)
using CopyChain = DmaCopyChain<CORALNPU_DMA_BEAT_LOG2>;
#elif defined(CORALNPU_COPY_AT_WAIT)
using CopyChain = MemcpyCopyChain<true>;
#else
using CopyChain = MemcpyCopyChain<false>;
#endif

// Drains the given chains when it goes out of scope, so that every return from a tiled kernel,
// including the early ones on failure, leaves no copy running. The caller may then reuse the
// buffers or recompute the layer untiled. A template on the chain type, like the chains, so that
// it is a different class in builds with different backends.
//   DrainOnExit drain(chain_a, chain_b);
template <typename Chain, int N>
class DrainOnExit {
 public:
  template <typename... Chains>
  explicit DrainOnExit(Chain &first, Chains &...rest) : chains_{&first, &rest...} {}
  ~DrainOnExit() {
    for (Chain *c : chains_)
      c->Drain();
  }
  DrainOnExit(const DrainOnExit &)            = delete;
  DrainOnExit &operator=(const DrainOnExit &) = delete;

 private:
  Chain *chains_[N];
};
template <typename Chain, typename... Chains>
DrainOnExit(Chain &, Chains &...) -> DrainOnExit<Chain, 1 + sizeof...(Chains)>;

}  // namespace coralnpu_v2::opt::litert_micro

#endif  // SW_OPT_LITERT_MICRO_DMA_TILE_H_
