# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs mobilenet_vs_tflite_test.elf on npusim (#11).

Full MobileNet through the interpreter; every conv and depthwise layer is compared with TFLite's
reference kernel on the same input. A byte may be TFLite's value plus one (#23 tie); nothing else.
"""

import os

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_ARRAYS = [
    "layer_is_dw", "layer_bytes", "layer_equal", "layer_plus_one",
    "layer_other"
]
_SYMBOLS = ["heartbeat", "layers", "output_checksum", "model_output"] + _ARRAYS


@coralflow_test()
async def run_mobilenet_vs_tflite_test(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/mobilenet_vs_tflite_test.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(elf_file, _SYMBOLS)
    await fixture.run_to_halt(timeout_cycles=20_000_000_000)

    heartbeat = await fixture.read_word("heartbeat")
    assert heartbeat == 0x22, f"model did not finish: heartbeat {heartbeat:#x}"
    n = await fixture.read_word("layers")
    rows = []
    for k in range(n):
        rows.append({
            s: await fixture.read_word(s, offset=4 * k)
            for s in _ARRAYS
        })
    total = {s: sum(row[s] for row in rows) for s in _ARRAYS[1:]}
    checksum = await fixture.read_word("output_checksum")
    print(
        f"mobilenet_vs_tflite_test: {n} layers, {total['layer_bytes']} bytes: "
        f"equal {total['layer_equal']}, TFLite+1 (#23) {total['layer_plus_one']}, "
        f"other {total['layer_other']}; output checksum {checksum:#010x}",
        flush=True,
    )
    for k, row in enumerate(rows):
        print(
            f"  layer {k:2d} {'dw  ' if row['layer_is_dw'] else 'conv'} "
            f"bytes {row['layer_bytes']:7d} equal {row['layer_equal']:7d} "
            f"+1 {row['layer_plus_one']:5d} other {row['layer_other']}",
            flush=True,
        )
    assert n > 0, "no conv or depthwise layer ran"
    bad = [k for k, row in enumerate(rows) if row["layer_other"] != 0]
    assert not bad, f"layers with differences other than #23 ties: {bad}"


if __name__ == "__main__":
    coralflow_main()
