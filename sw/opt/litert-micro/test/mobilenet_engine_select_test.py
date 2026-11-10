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
"""Full MobileNet with the per-layer vector/matrix choice (idea 5, #12).

The ELF runs the model three times through the CONV_2D op: all conv layers on
the vector unit, the per-layer choice, and all on the matrix engine. Every conv
layer's output must be identical across the three runs, and each run must use
the matrix engine on the expected layers. Prints one CSV row per run;
cycles are meaningful only on RTL (npusim counts instructions).

Build with both flags, or every run takes the vector path:
  bazel test --//sw/opt/litert-micro:rvv_conv \\
    --//sw/opt/litert-micro:conv_engine_select ...

The arena (about 1.3 MB) is in EXTMEM. On RTL each run is long; pick runs with
--test_env=MOBILENET_ENGINE_RUNS=all_vector,per_layer.
"""

import os

import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_RUNS = ["all_vector", "per_layer", "all_matrix"]
_MAX_CONV_LAYERS = 32
# MobileNet v1 0.25 has 15 conv layers; the per-layer choice keeps the two with
# fewer than 16 input channels on the vector unit.
_CONV_LAYERS = 15
_MATRIX_LAYERS = [0, 13, 15]

_SYMBOLS = [
    "conv_layers",
    "matrix_layers",
    "checksums",
    "invoke_cycles",
    "invoke_instret",
    "model_output",
    "heartbeat",
    "arena_used",
    "run_mask",
]

# #59: stall counters, present when built with --//sw/utils:read_perf_counters
# (order as PerfCounters.names in hdl/chisel/src/coralnpu/scalar/Csr.scala).
_PERF_NAMES = (
    "memStall", "lsuFull", "tcmWait", "extWait", "fetchWait", "vecDataWait",
    "noDispatch", "rvvFull"
)


def _has_perf(symbols):
    return bool(symbols) and symbols.get("perf_counters_enabled") is not None


@coralflow_test(highmem=True)
async def run_mobilenet_engine_select(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/mobilenet_engine_select_test.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    symbols = await fixture.load_elf_and_lookup_symbols(
        elf_file,
        _SYMBOLS,
        optional_symbols=["perf_counters_enabled", "invoke_perf"]
    )
    perf = _has_perf(symbols)

    selected = os.environ.get("MOBILENET_ENGINE_RUNS", "")
    runs = [
        i for i, name in enumerate(_RUNS)
        if not selected or name in selected.split(",")
    ]
    assert runs, f"no runs selected by MOBILENET_ENGINE_RUNS={selected}"
    await fixture.write_word("run_mask", sum(1 << i for i in runs))

    await fixture.run_to_halt(
        timeout_sec=24 * 3600, timeout_cycles=20_000_000_000
    )

    heartbeat = int(
        (await fixture.read("heartbeat", dtype=np.int32, shape=(1, )))[0]
    )
    assert heartbeat == 0x22, f"heartbeat {heartbeat:#x}"
    conv_layers = await fixture.read(
        "conv_layers", dtype=np.int32, shape=(len(_RUNS), )
    )
    matrix_layers = await fixture.read(
        "matrix_layers", dtype=np.int32, shape=(len(_RUNS), )
    )
    checksums = await fixture.read(
        "checksums", dtype=np.uint32, shape=(len(_RUNS), _MAX_CONV_LAYERS)
    )
    cycles = await fixture.read(
        "invoke_cycles", dtype=np.uint64, shape=(len(_RUNS), )
    )
    instret = await fixture.read(
        "invoke_instret", dtype=np.uint64, shape=(len(_RUNS), )
    )
    outputs = await fixture.read(
        "model_output", dtype=np.int8, shape=(len(_RUNS), 5)
    )
    arena = await fixture.read(
        "arena_used", dtype=np.uint32, shape=(len(_RUNS), )
    )

    print(
        "CSV,run,conv_layers,matrix_layers,invoke_cycles,invoke_instret,"
        "arena_bytes,output",
        flush=True
    )
    for i in runs:
        print(
            f"CSV,{_RUNS[i]},{conv_layers[i]},{matrix_layers[i]},{cycles[i]},"
            f"{instret[i]},{arena[i]},{' '.join(str(v) for v in outputs[i])}",
            flush=True
        )

    if perf:
        invoke_perf = await fixture.read(
            "invoke_perf",
            dtype=np.uint64,
            shape=(len(_RUNS), len(_PERF_NAMES))
        )
        print("PERF,run," + ",".join(_PERF_NAMES), flush=True)
        for i in runs:
            print(
                f"PERF,{_RUNS[i]}," +
                ",".join(str(int(v)) for v in invoke_perf[i]),
                flush=True
            )

    for i in runs:
        assert conv_layers[i] == _CONV_LAYERS, (_RUNS[i], conv_layers[i])
        assert matrix_layers[i] == _MATRIX_LAYERS[i], (
            f"{_RUNS[i]}: {matrix_layers[i]} matrix layers, expected "
            f"{_MATRIX_LAYERS[i]}; build with "
            "--//sw/opt/litert-micro:conv_engine_select"
        )
    first = runs[0]
    for i in runs[1:]:
        diff = np.nonzero(
            checksums[i, :_CONV_LAYERS] != checksums[first, :_CONV_LAYERS]
        )[0]
        assert diff.size == 0, (
            f"{_RUNS[i]} differs from {_RUNS[first]} in conv layers "
            f"{diff.tolist()}"
        )
    print(
        f"All {_CONV_LAYERS} conv outputs match across "
        f"{', '.join(_RUNS[i] for i in runs)}",
        flush=True
    )


if __name__ == "__main__":
    coralflow_main()
