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
]


@coralflow_test(highmem=True)
async def run_mobilenet_engine_select(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/mobilenet_engine_select_test.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(elf_file, _SYMBOLS)

    await fixture.run_to_halt(
        timeout_sec=6 * 3600, timeout_cycles=20_000_000_000
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

    print(
        "CSV,run,conv_layers,matrix_layers,invoke_cycles,invoke_instret,output",
        flush=True
    )
    for i, name in enumerate(_RUNS):
        print(
            f"CSV,{name},{conv_layers[i]},{matrix_layers[i]},{cycles[i]},"
            f"{instret[i]},{' '.join(str(v) for v in outputs[i])}",
            flush=True
        )

    assert list(conv_layers) == [_CONV_LAYERS] * len(_RUNS), conv_layers
    assert list(matrix_layers) == _MATRIX_LAYERS, (
        f"matrix layers per run {list(matrix_layers)}, expected "
        f"{_MATRIX_LAYERS}; build with --//sw/opt/litert-micro:conv_engine_select"
    )
    for i in range(1, len(_RUNS)):
        diff = np.nonzero(
            checksums[i, :_CONV_LAYERS] != checksums[0, :_CONV_LAYERS]
        )[0]
        assert diff.size == 0, (
            f"{_RUNS[i]} differs from {_RUNS[0]} in conv layers {diff.tolist()}"
        )
    print(
        f"All {_CONV_LAYERS} conv outputs match across the three runs",
        flush=True
    )


if __name__ == "__main__":
    coralflow_main()
