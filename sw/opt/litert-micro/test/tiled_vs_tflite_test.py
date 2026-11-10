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
"""Runs tiled_vs_tflite_test.elf on npusim (#11): the tiled paths against TFLite, byte for byte.

A byte may differ from TFLite only where the RVV kernels' rounding differs from TFLite's (#23).
"""

import os

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_SCALARS = [
    "done",
    "rvv_kernel",
    "model_mismatches",
    "possible_ties",
    "first_other",
]
_ARRAYS = ["tiled_runs", "bytes_checked", "bytes_tie", "bytes_other"]
_KINDS = ["pointwise", "row_3x3", "depthwise"]


@coralflow_test()
async def run_tiled_vs_tflite_test(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/tiled_vs_tflite_test.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(elf_file, _SCALARS + _ARRAYS)
    await fixture.run_to_halt(timeout_cycles=4_000_000_000)

    v = {s: await fixture.read_word(s) for s in _SCALARS}
    a = {}
    for s in _ARRAYS:
        a[s] = [
            await fixture.read_word(s, offset=4 * k)
            for k in range(len(_KINDS))
        ]
    rvv = v["rvv_kernel"] == 1
    print(
        f"tiled_vs_tflite_test: conv kernel={'rvv' if rvv else 'reference'} "
        f"model_mismatches={v['model_mismatches']} possible_ties={v['possible_ties']}",
        flush=True,
    )
    for k, kind in enumerate(_KINDS):
        print(
            f"  {kind}: tiled runs {a['tiled_runs'][k]}, bytes {a['bytes_checked'][k]}, "
            f"equal to TFLite {a['bytes_checked'][k] - a['bytes_tie'][k] - a['bytes_other'][k]}, "
            f"#23 ties {a['bytes_tie'][k]}, other {a['bytes_other'][k]}",
            flush=True,
        )
    assert v["done"] == 1, "program did not finish"
    assert v["model_mismatches"] == 0, (
        "the test's requant model disagrees with reference_integer_ops: " +
        str(v)
    )
    assert v["possible_ties"
             ] > 0, "no rounding ties occurred: the test cannot see #23"
    for k, kind in enumerate(_KINDS):
        assert a["tiled_runs"][k] > 0, f"{kind}: no case was tiled"
        assert a["bytes_other"][k] == 0, (
            f"{kind}: bytes that are neither TFLite's value nor a #23 tie "
            f"(first: case {(v['first_other'] - 1) // 16}, budget "
            f"{(v['first_other'] - 1) % 16})"
        )
    if not rvv:
        # The conv tiles run the TFLite reference kernel: exact.
        assert a["bytes_tie"][0] == 0 and a["bytes_tie"][1] == 0, a


if __name__ == "__main__":
    coralflow_main()
