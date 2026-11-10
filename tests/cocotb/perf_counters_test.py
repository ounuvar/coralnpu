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
"""#59: checks the memory stall counters on a core with enablePerfCounters."""

from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test
from bazel_tools.tools.python.runfiles import runfiles
import logging
import numpy as np

# Same order as PerfCounters.names in hdl/chisel/src/coralnpu/scalar/Csr.scala.
NAMES = [
    "memStall", "lsuFull", "tcmWait", "extWait", "fetchWait", "vecDataWait",
    "noDispatch", "rvvFull"
]
PHASES = ["alu", "dtcm_chase", "ext_chase", "ext_vector", "ext_vstore"]


@coralflow_test
async def perf_counters_test(fixture):
    r = runfiles.Create()
    await fixture.load_elf_and_lookup_symbols(
        r.Rlocation('coralnpu_hw/tests/cocotb/perf_counters_test.elf'),
        ['results']
    )
    await fixture.run_to_halt(timeout_cycles=2000000)
    raw = await fixture.read(
        'results', dtype=np.uint32, shape=(len(PHASES), len(NAMES) + 1)
    )
    res = {}
    for p, phase in enumerate(PHASES):
        row = dict(zip(["cycles"] + NAMES, (int(v) for v in raw[p])))
        res[phase] = row
        logging.info(
            "%-11s %s", phase, " ".join(f"{k}={v}" for k, v in row.items())
        )
        print("CSV," + phase + "," + ",".join(str(v) for v in row.values()))

    for phase, row in res.items():
        for name in NAMES:
            assert row[name] <= row["cycles"], f"{phase}: {name} > cycles"
        assert row["memStall"] <= row["noDispatch"], \
            f"{phase}: memStall must be a subset of noDispatch"

    alu, dtcm, ext, vec, vst = (res[p] for p in PHASES)
    assert alu["extWait"] == 0 and dtcm["extWait"] == 0, \
        "no external accesses in the ALU and DTCM phases"
    assert alu["lsuFull"] == 0, "ALU phase has no loads"
    assert dtcm["memStall"] > alu["memStall"], "DTCM loads must stall"
    assert ext["extWait"] > 0, "external loads must wait on the bus"
    assert ext["memStall"] > dtcm["memStall"], \
        "external loads must stall longer than DTCM loads"
    assert ext["cycles"] > dtcm["cycles"]
    assert vec["extWait"] > 0, "vector external loads use the external bus"
    assert vst["extWait"] > 0, "vector external stores use the external bus"


if __name__ == "__main__":
    coralflow_main()
