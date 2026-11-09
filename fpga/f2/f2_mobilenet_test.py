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
"""Runs the F2 MobileNet program (fpga/sw/f2_mobilenet.cc) on a simulator (#16).

The same ELF runs on F2. The simulators have no UART1, so the console is turned off first.
"""

import os

import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

import f2_mobilenet

F2_PASS = 0x600DF00D


@coralflow_test()
async def run_f2_mobilenet(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation("coralnpu_hw/fpga/f2_mobilenet.elf")
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(
        elf_file, ["f2_result", "f2_uart_enable", "f2_mnet"]
    )
    await fixture.write_word("f2_uart_enable", 0)
    await fixture.run_to_halt(timeout_cycles=40_000_000_000)
    blob = await fixture.read(
        "f2_mnet", size=f2_mobilenet.SIZE, dtype=np.uint8
    )
    results = f2_mobilenet.parse(bytes(blob))
    print(f2_mobilenet.report(results), flush=True)
    errors = f2_mobilenet.check(results)
    assert not errors, "\n".join(errors)
    result = await fixture.read_word("f2_result")
    assert result == F2_PASS, f"f2_result {result:#010x}"


if __name__ == "__main__":
    coralflow_main()
