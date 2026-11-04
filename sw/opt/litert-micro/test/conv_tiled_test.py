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
"""Runs conv_tiled_test.elf on npusim (#11): tiled equals untiled, byte for byte."""

import os

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_SYMBOLS = [
    "done",
    "cases_run",
    "cases_tiled",
    "cases_not_tiled",
    "cases_chunked",
    "cases_multi_tile",
    "cases_double_buffer",
    "cases_single_buffer",
    "mismatches",
    "selftest_detected",
    "rvv_kernel",
]


@coralflow_test()
async def run_conv_tiled_test(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/conv_tiled_test.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(elf_file, _SYMBOLS)
    await fixture.run_to_halt(timeout_cycles=2_000_000_000)

    v = {s: await fixture.read_word(s) for s in _SYMBOLS}
    print(
        f"conv_tiled_test: kernel={'rvv' if v['rvv_kernel'] else 'reference'} "
        f"runs={v['cases_run']} tiled={v['cases_tiled']} "
        f"not_tiled={v['cases_not_tiled']} chunked={v['cases_chunked']} "
        f"multi_tile={v['cases_multi_tile']} double={v['cases_double_buffer']} "
        f"single={v['cases_single_buffer']} mismatches={v['mismatches']}",
        flush=True,
    )
    assert v["done"] == 1, "program did not finish"
    assert v["mismatches"] == 0, v
    assert v["selftest_detected"] == 1, "the comparison missed a flipped byte"
    # The test must actually exercise the interesting plans.
    assert v["cases_tiled"] > 0, v
    assert v["cases_chunked"] > 0, "no case used output-channel chunks"
    assert v["cases_multi_tile"] > 0, "no case used more than one tile"
    assert v["cases_double_buffer"] > 0, "no case used double buffering"
    assert v["cases_single_buffer"] > 0, "no case used single buffering"
    assert v["cases_tiled"] + v["cases_not_tiled"] == v["cases_run"], v


if __name__ == "__main__":
    coralflow_main()
