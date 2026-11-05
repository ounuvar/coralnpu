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
"""Runs requant_iss_test.elf on npusim and on the REQUANT_ON RTL (idea 2, #9)."""

import os

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_SYMBOLS = [
    "done",
    "insn_mismatches",
    "fused_mismatches",
    "ties_checked",
    "baseline_instret",
    "fused_instret",
    "requant_mismatches",
    "rqclamp_mismatches",
    "first_bad_count",
    "first_bad",
]


def _signed(value):
    return value - (1 << 32) if value >= 1 << 31 else value


@coralflow_test()
async def run_requant_iss_test(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/requant_iss_test.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(elf_file, _SYMBOLS)
    await fixture.run_to_halt(timeout_cycles=50_000_000)

    values = {
        s: await fixture.read_word(s) for s in _SYMBOLS if s != "first_bad"
    }
    for k in range(min(_signed(values["first_bad_count"]), 8)):
        rec = [
            _signed(await fixture.read_word("first_bad", offset=4 * (7 * k + j)))
            for j in range(7)
        ]
        print("mismatch i={} acc={} mult={} shift={} offset={} got={} want={}"
              .format(*rec), flush=True)
    baseline = values["baseline_instret"]
    fused = values["fused_instret"]
    print(
        f"instret: PostprocessAcc {baseline}, fused {fused} "
        f"({baseline / max(fused, 1):.2f}x fewer)",
        flush=True
    )
    assert values["done"] == 1, "program did not finish"
    assert _signed(values["requant_mismatches"]) == 0, values
    assert _signed(values["rqclamp_mismatches"]) == 0, values
    assert _signed(values["insn_mismatches"]) == 0, values
    assert _signed(values["fused_mismatches"]) == 0, values
    assert _signed(values["ties_checked"]) > 0, values
    assert fused < baseline, values


if __name__ == "__main__":
    coralflow_main()
