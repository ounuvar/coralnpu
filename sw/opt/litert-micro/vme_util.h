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

// Matrix engine (VME, Zvt) helpers for INT8 GEMM kernels (idea 5, #12).
//
// The toolchain has no Zvt mnemonics, so every instruction is emitted as a raw
// encoding (see tests/cocotb/vme_test/vme_test_utils.h for the full set).
// Geometry at VLEN=128: 16x16 int32 accumulator tiles, and in INT8 mode
// (SEW8, mtwiden=3) each matmul consumes up to 4 K-rows:
//
//   C[m][n] += sum_{k<4} A[k][m] * B[k][n]
//
// A and B come from vector registers in K-major order: A rows in v8, v10, v12,
// v14 and B rows in v16, v18, v20, v22, 16 bytes each. int32 accumulators must
// use tiles 0, 4, 8 or 12. Any vset* clears mtype, so reconfigure after RVV
// code.

#ifndef SW_OPT_LITERT_MICRO_VME_UTIL_H_
#define SW_OPT_LITERT_MICRO_VME_UTIL_H_

#include <cstdint>

namespace coralnpu_v2::opt::litert_micro::vme {

inline constexpr int kTileDim     = 16;  // M and N of one accumulator tile
inline constexpr int kTileK       = 4;   // K rows per INT8 matmul
inline constexpr int kMaxAccTiles = 4;   // int32 tiles 0, 4, 8, 12

// mtype: tm[23:10] | tk[7:5] | mtwiden[1:0].
constexpr uint32_t MtypeValue(uint32_t tm, uint32_t tk, uint32_t mtwiden) {
  return ((tm & 0x3FFF) << 10) | ((tk & 0x7) << 5) | (mtwiden & 0x3);
}

// vtype: altfmt[8] | vma[7] | vta[6] | vsew[5:3] | vlmul[2:0].
inline constexpr uint32_t kVtypeSew8Lmul1Altfmt = 0x1C0;  // altfmt makes B (vs1) signed
inline constexpr uint32_t kVtypeSew32Lmul4      = 0xD2;

// [31:26 funct6] [25 vm=1] [24:20 vs2] [19:15 vs1/rs1] [14:12 funct3] [11:7 rd] [6:0 opcode]
constexpr uint32_t ZvtWord(uint32_t funct6, uint32_t vs2, uint32_t rs1, uint32_t funct3,
                           uint32_t rd, uint32_t opcode) {
  return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) |
         (opcode & 0x7Fu);
}

// vtmms (signed A): OP-VE, funct6=111100, funct3=000, A=v8, B=v16, rd=(tile<<1)|1.
constexpr uint32_t MatmulSignedWord(uint32_t tile) {
  return ZvtWord(0x3C, 8, 16, 0, (tile << 1) | 1, 0x77);
}
// vtzero: OP-V, funct6=010000, vs2=11110, funct3=110, rd=tile<<1.
constexpr uint32_t VtzeroWord(uint32_t tile) { return ZvtWord(0x10, 30, 0, 6, tile << 1, 0x57); }
// vtmv.v.t v4, a0: tile row selected by a0 into v4..v7.
inline constexpr uint32_t kVtmvVTWord = ZvtWord(0x10, 31, 10, 6, 4, 0x57);

// msetmtype rs1, rs2: mtype <- rs1, vtype <- rs2, vl <- 0.
inline __attribute__((always_inline)) void SetMtype(uint32_t mtype, uint32_t vtype) {
  register uint32_t a0_arg asm("a0") = mtype;
  register uint32_t a1_arg asm("a1") = vtype;
  asm volatile(".insn r 0b1010111, 0b111, 0b1000010, x0, a0, a1" : : "r"(a0_arg), "r"(a1_arg));
}

// msettn rd, rs1: vl (= tn) <- min(rs1, limits).
inline __attribute__((always_inline)) void SetTn(uint32_t tn) {
  register uint32_t a0_arg asm("a0") = tn;
  register uint32_t t0_out asm("t0");
  asm volatile(".insn r 0b1010111, 0b111, 0b1000100, t0, a0, x0" : "=r"(t0_out) : "r"(a0_arg));
}

// Signed int8 x signed int8 into int32 tiles, full 16x16x4 shape.
inline __attribute__((always_inline)) void ConfigureInt8Matmul() {
  SetMtype(MtypeValue(kTileDim, kTileK, /*mtwiden=*/3), kVtypeSew8Lmul1Altfmt);
  SetTn(kTileDim);
}

// SEW32/LMUL4: one 16 x int32 tile row per vtmv, full-tile vtzero.
inline __attribute__((always_inline)) void ConfigureTileMoves() {
  SetMtype(MtypeValue(kTileDim, 1, /*mtwiden=*/1), kVtypeSew32Lmul4);
  SetTn(kTileDim);
}

// Tile index for accumulator slot t (0..3).
constexpr uint32_t AccTile(int t) { return static_cast<uint32_t>(4 * t); }

// Zeroes accumulator slots 0..NT-1. Needs ConfigureTileMoves().
template <int NT>
inline __attribute__((always_inline)) void ZeroAccTiles() {
  asm volatile(".word %0" : : "i"(VtzeroWord(AccTile(0))) : "memory");
  if constexpr (NT > 1)
    asm volatile(".word %0" : : "i"(VtzeroWord(AccTile(1))) : "memory");
  if constexpr (NT > 2)
    asm volatile(".word %0" : : "i"(VtzeroWord(AccTile(2))) : "memory");
  if constexpr (NT > 3)
    asm volatile(".word %0" : : "i"(VtzeroWord(AccTile(3))) : "memory");
}

// One K-step for NT accumulator slots: loads 4 A rows (16 bytes each, K-major,
// at a) once, then for each slot t loads its 4 B rows from b + t * b_tile_stride
// and accumulates. Loads and matmuls share one asm block so nothing can use
// v8..v23 in between. Needs ConfigureInt8Matmul().
#define VME_LOAD_ROWS(VA, VB, VC, VD, PTR) \
  "vle8.v " VA ", (" PTR                   \
  ")\n"                                    \
  "addi   t1, " PTR                        \
  ", 16\n"                                 \
  "vle8.v " VB                             \
  ", (t1)\n"                               \
  "addi   t1, " PTR                        \
  ", 32\n"                                 \
  "vle8.v " VC                             \
  ", (t1)\n"                               \
  "addi   t1, " PTR                        \
  ", 48\n"                                 \
  "vle8.v " VD ", (t1)\n"
#define VME_A_ROWS(PTR)         VME_LOAD_ROWS("v8", "v10", "v12", "v14", PTR)
#define VME_B_MATMUL(PTR, WORD) VME_LOAD_ROWS("v16", "v18", "v20", "v22", PTR) ".word " WORD "\n"
#define VME_MATMUL_OPERANDS                                               \
  : "r"(a), "r"(b), "r"(b + b_tile_stride), "r"(b + 2 * b_tile_stride),                        \
    "r"(b + 3 * b_tile_stride), "i"(MatmulSignedWord(AccTile(0))),                             \
    "i"(MatmulSignedWord(AccTile(1))), "i"(MatmulSignedWord(AccTile(2))),                      \
    "i"(MatmulSignedWord(AccTile(3)))                                                          \
  : "t1", "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22", "memory"

template <int NT>
inline __attribute__((always_inline)) void MatmulStep(const int8_t *a, const int8_t *b,
                                                      int b_tile_stride) {
  static_assert(NT >= 1 && NT <= kMaxAccTiles);
  if constexpr (NT == 1) {
    asm volatile(VME_A_ROWS("%0") VME_B_MATMUL("%1", "%5") : VME_MATMUL_OPERANDS);
  } else if constexpr (NT == 2) {
    asm volatile(VME_A_ROWS("%0") VME_B_MATMUL("%1", "%5") VME_B_MATMUL("%2", "%6")
                 : VME_MATMUL_OPERANDS);
  } else if constexpr (NT == 3) {
    asm volatile(VME_A_ROWS("%0") VME_B_MATMUL("%1", "%5") VME_B_MATMUL("%2", "%6")
                     VME_B_MATMUL("%3", "%7")
                 : VME_MATMUL_OPERANDS);
  } else {
    asm volatile(VME_A_ROWS("%0") VME_B_MATMUL("%1", "%5") VME_B_MATMUL("%2", "%6")
                     VME_B_MATMUL("%3", "%7") VME_B_MATMUL("%4", "%8")
                 : VME_MATMUL_OPERANDS);
  }
}

#undef VME_MATMUL_OPERANDS
#undef VME_B_MATMUL
#undef VME_A_ROWS
#undef VME_LOAD_ROWS

// Stores row `row` (16 x int32) of accumulator slot T to dst. Needs
// ConfigureTileMoves().
template <int T>
inline __attribute__((always_inline)) void StoreAccRow(uint32_t row, int32_t *dst) {
  register uint32_t a0_arg asm("a0") = (AccTile(T) << 27) | row;  // TSS: tile | row pattern | index
  asm volatile(
      ".word   %0\n"
      "vse32.v v4, (%1)\n"
      :
      : "i"(kVtmvVTWord), "r"(dst), "r"(a0_arg)
      : "v4", "v5", "v6", "v7", "memory");
}

}  // namespace coralnpu_v2::opt::litert_micro::vme

#endif  // SW_OPT_LITERT_MICRO_VME_UTIL_H_
