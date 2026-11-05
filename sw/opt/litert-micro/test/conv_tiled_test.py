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
    "cases_chunked_double",
    "guard_violations",
    "plan_failures",
    "first_failure",
    "cases_chunked_single",
    "mismatches",
    "selftest_detected",
    "rvv_kernel",
]


@coralflow_test()
async def run_conv_tiled_test(fixture):
    r = runfiles.Create()
    # The test target conv_tiled_sim_test<variant>_npusim runs conv_tiled_test<variant>.elf.
    target = os.environ.get("TEST_TARGET", "")
    variant = target.split("conv_tiled_sim_test")[-1].rsplit("_npusim", 1)[0]
    elf_file = r.Rlocation(
        f"coralnpu_hw/sw/opt/litert-micro/test/conv_tiled_test{variant}.elf"
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
        f"single={v['cases_single_buffer']} "
        f"chunked_double={v['cases_chunked_double']} "
        f"chunked_single={v['cases_chunked_single']} "
        f"mismatches={v['mismatches']} guard={v['guard_violations']} "
        f"plan_failures={v['plan_failures']} "
        f"first_failure={v['first_failure']} variant='{variant}'",
        flush=True,
    )
    assert v["done"] == 1, "program did not finish"
    assert v["mismatches"] == 0, v
    assert v[
        "guard_violations"
    ] == 0, "the tiled path wrote past the end of the output: " + str(v)
    assert v["plan_failures"] == 0, (
        "the planner accepted a layer that the wrapper then failed to run "
        f"(first: case {(v['first_failure'] - 1) // 16}, budget "
        f"{(v['first_failure'] - 1) % 16}): {v}"
    )
    assert v["selftest_detected"] == 1, "the comparison missed a flipped byte"
    # The test must actually exercise the interesting plans.
    assert v["cases_tiled"] > 0, v
    assert v["cases_chunked"] > 0, "no case used output-channel chunks"
    assert v["cases_multi_tile"] > 0, "no case used more than one tile"
    assert v["cases_double_buffer"] > 0, "no case used double buffering"
    assert v["cases_single_buffer"] > 0, "no case used single buffering"
    assert v["cases_chunked_double"] > 0, "no chunked case was double buffered"
    assert v["cases_chunked_single"] > 0, "no chunked case was single buffered"
    assert v["cases_tiled"] + v["cases_not_tiled"] == v["cases_run"], v


if __name__ == "__main__":
    coralflow_main()
