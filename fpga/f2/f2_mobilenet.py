# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reads the results of fpga/sw/f2_mobilenet.cc (#16).

The layout of struct F2Mnet is kept here. `parse` takes the raw bytes of f2_mnet (from the npusim
test, or from `f2_loader --read_data_addr=<f2_mnet> --read_data_size=SIZE` on F2), `check`
lists what is wrong with them, and `report` formats them.

    python3 f2_mobilenet.py results.bin
"""

import struct
import sys

MAGIC = 0x4D4E4554
MAX_LAYERS = 32
_HEADER = struct.Struct("<11I8b")
_ARRAYS = [("is_dw", "i"), ("bytes", "i"), ("layer_cycles", "I"),
           ("equal", "i"), ("plus_one", "i"), ("other", "i")]
SIZE = _HEADER.size + len(_ARRAYS) * 4 * MAX_LAYERS


def parse(blob):
    if len(blob) < SIZE:
        raise ValueError(f"f2_mnet: {len(blob)} bytes, expected {SIZE}")
    h = _HEADER.unpack_from(blob, 0)
    r = {
        "magic": h[0],
        "stage": h[1],
        "layers": h[2],
        "cycles": h[3] | h[4] << 32,
        "instret": h[5] | h[6] << 32,
        "checksum_timed": h[7],
        "checksum_checked": h[8],
        "total_plus_one": h[9],
        "total_other": h[10],
        "output": list(h[11:19]),
    }
    off = _HEADER.size
    for name, fmt in _ARRAYS:
        r[name] = list(struct.unpack_from(f"<{MAX_LAYERS}{fmt}", blob, off))
        off += 4 * MAX_LAYERS
    return r


def check(r):
    errors = []
    if r["magic"] != MAGIC:
        errors.append(f"magic {r['magic']:#x}: the program did not start")
    if r["stage"] != 4:
        errors.append(f"stage {r['stage']:#x}: the program did not finish")
    if r["layers"] == 0:
        errors.append("no conv or depthwise layer ran")
    bad = [k for k in range(r["layers"]) if r["other"][k] != 0]
    if bad:
        errors.append(
            f"layers differing from TFLite other than by #23 ties: {bad}"
        )
    if r["checksum_timed"] != r["checksum_checked"]:
        errors.append(
            f"checksums differ: timed {r['checksum_timed']:#010x}, "
            f"checked {r['checksum_checked']:#010x}"
        )
    return errors


def report(r):
    n = r["layers"]
    lines = [
        f"f2_mobilenet: {r['cycles']} cycles, {r['instret']} instructions "
        f"(Invoke, optimized kernels)",
        f"  output checksum {r['checksum_timed']:#010x} (checked pass "
        f"{r['checksum_checked']:#010x}), first bytes {r['output']}",
        f"  {n} conv/dw layers, {sum(r['bytes'][:n])} bytes checked against "
        f"TFLite: +1 (#23 ties) {r['total_plus_one']}, other {r['total_other']}",
    ]
    conv_dw = sum(r["layer_cycles"][:n])
    lines.append(
        f"  conv/dw cycles {conv_dw} ({100.0 * conv_dw / max(r['cycles'], 1):.1f}% "
        f"of the model)"
    )
    for k in range(n):
        lines.append(
            f"  layer {k:2d} {'dw  ' if r['is_dw'][k] else 'conv'} "
            f"cycles {r['layer_cycles'][k]:10d} bytes {r['bytes'][k]:7d} "
            f"equal {r['equal'][k]:7d} +1 {r['plus_one'][k]:5d} "
            f"other {r['other'][k]}"
        )
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: f2_mobilenet.py <f2_mnet bytes>")
    with open(sys.argv[1], "rb") as f:
        r = parse(f.read())
    print(report(r))
    errors = check(r)
    if errors:
        raise SystemExit("FAIL:\n  " + "\n  ".join(errors))
    print("PASS")


if __name__ == "__main__":
    main()
