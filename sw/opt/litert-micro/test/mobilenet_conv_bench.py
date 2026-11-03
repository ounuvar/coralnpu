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
"""Per-layer baseline vs matrix-engine conv benchmark for MobileNet (#12).

Runs every CONV_2D layer of MobileNet v1 0.25 at 224x224 (the model in
tests/cocotb/tutorial/tfmicro/models) with random data, through the baseline
dispatcher and the matrix-engine kernel, and requires bit-identical outputs.
On these shapes the baseline dispatcher falls back to the TFLite reference
kernel, so this is also a bit-exactness check against the reference.

Prints one CSV row per layer. Cycles are meaningful only on RTL (the
verilator target); npusim counts are instruction-level.

`bazel test --test_env=MOBILENET_CONV_LAYERS=0,2,5` (op indices below) restricts
the run, since the full network is long on RTL.
"""

import os
import sys

import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_SYMBOLS = [
    "input_data",
    "filter_data",
    "bias_data",
    "output_baseline",
    "output_vme",
    "input_dims",
    "filter_dims",
    "output_dims",
    "params_stride",
    "params_padding_height",
    "params_padding_width",
    "params_input_offset",
    "params_output_offset",
    "params_activation_min",
    "params_activation_max",
    "per_channel_multiplier",
    "per_channel_shift",
    "baseline_cycles",
    "baseline_instret",
    "vme_cycles",
    "vme_instret",
    "heartbeat",
]

# (model op index, input HWC, filter OHWI, stride), all SAME padding. Read from
# mobilenet_v1_0.25_224_int8_dummy.tflite.
_LAYERS = [
    (0, (224, 224, 3), (8, 3, 3, 3), 2),
    (2, (112, 112, 8), (16, 1, 1, 8), 1),
    (5, (56, 56, 16), (32, 1, 1, 16), 1),
    (7, (56, 56, 32), (32, 1, 1, 32), 1),
    (10, (28, 28, 32), (64, 1, 1, 32), 1),
    (12, (28, 28, 64), (64, 1, 1, 64), 1),
    (15, (14, 14, 64), (128, 1, 1, 64), 1),
    (17, (14, 14, 128), (128, 1, 1, 128), 1),
    (19, (14, 14, 128), (128, 1, 1, 128), 1),
    (21, (14, 14, 128), (128, 1, 1, 128), 1),
    (23, (14, 14, 128), (128, 1, 1, 128), 1),
    (25, (14, 14, 128), (128, 1, 1, 128), 1),
    (28, (7, 7, 128), (256, 1, 1, 128), 1),
    (30, (7, 7, 256), (256, 1, 1, 256), 1),
    (32, (1, 1, 256), (5, 1, 1, 256), 1),
]


def _same_padding(in_size, filter_size, stride):
    out_size = (in_size + stride - 1) // stride
    total = max(0, (out_size - 1) * stride + filter_size - in_size)
    return out_size, total // 2  # TFLite puts the extra row/column last


async def _read_u64(fixture, name):
    return int.from_bytes((await fixture.read(name, 8)).tobytes(), "little")


async def _run_layer(fixture, op_index, in_hwc, filt, stride):
    # Seeded per layer, so a layer's data does not depend on which others run.
    rng = np.random.default_rng(56 + op_index)
    h, w, c = in_hwc
    n, fh, fw, _ = filt
    out_h, pad_h = _same_padding(h, fh, stride)
    out_w, pad_w = _same_padding(w, fw, stride)
    k = fh * fw * c
    # Random int8 data, as in conv_test.py. The shift scales with K so outputs
    # don't all saturate; output_offset stays -128 for the reason in
    # conv_vme_test.py.
    shift = -int(np.ceil(np.log2(k))) - 6
    await fixture.write("input_dims", np.array([1, h, w, c], dtype=np.int32))
    await fixture.write("filter_dims", np.array(filt, dtype=np.int32))
    await fixture.write(
        "output_dims", np.array([1, out_h, out_w, n], dtype=np.int32)
    )
    await fixture.write(
        "input_data", rng.integers(-128, 128, size=h * w * c, dtype=np.int8)
    )
    await fixture.write(
        "filter_data",
        rng.integers(-128, 128, size=int(np.prod(filt)), dtype=np.int8)
    )
    await fixture.write(
        "bias_data", rng.integers(-1000, 1000, size=n, dtype=np.int32)
    )
    await fixture.write(
        "per_channel_multiplier",
        rng.integers(1 << 30, (1 << 31) - 1, size=n, dtype=np.int32)
    )
    await fixture.write(
        "per_channel_shift",
        (shift + rng.integers(-1, 2, size=n)).astype(np.int32)
    )
    for name, value in [
        ("params_stride", stride),
        ("params_padding_height", pad_h),
        ("params_padding_width", pad_w),
        ("params_input_offset", int(rng.integers(-127, 129))),
        ("params_output_offset", -128),
        ("params_activation_min", -128),
        ("params_activation_max", 127),
    ]:
        await fixture.write_word(name, value)

    await fixture.run_to_halt(timeout_cycles=2_000_000_000)

    heartbeat = int.from_bytes((await fixture.read("heartbeat", 4)).tobytes(),
                               "little")
    if heartbeat != 0x22:
        print(
            f"layer {op_index}: FAILED, heartbeat {heartbeat:#x}", flush=True
        )
        sys.exit(1)
    size = out_h * out_w * n
    base = await fixture.read("output_baseline", dtype=np.int8, shape=(size, ))
    vme = await fixture.read("output_vme", dtype=np.int8, shape=(size, ))
    mismatches = int(np.sum(base != vme))
    unsaturated = int(np.sum((base > -128) & (base < 127)))
    counters = [
        await _read_u64(fixture, name) for name in
        ("baseline_cycles", "vme_cycles", "baseline_instret", "vme_instret")
    ]
    row = [
        op_index, f"{h}x{w}x{c}", f"{fh}x{fw}x{n}/s{stride}",
        out_h * out_w * n * k
    ] + counters + [mismatches, unsaturated, size]
    print("CSV," + ",".join(str(v) for v in row), flush=True)
    if mismatches:
        print(f"layer {op_index}: FAILED, {mismatches} mismatches", flush=True)
        sys.exit(1)


@coralflow_test(highmem=True)
async def run_mobilenet_conv_bench(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/mobilenet_conv_bench.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(elf_file, _SYMBOLS)

    selected = os.environ.get("MOBILENET_CONV_LAYERS", "")
    wanted = {int(x) for x in selected.split(",") if x} if selected else None
    print(
        "CSV,layer,input,filter,macs,baseline_cycles,vme_cycles,"
        "baseline_instret,vme_instret,mismatches,unsaturated,outputs",
        flush=True
    )
    for layer in _LAYERS:
        if wanted is None or layer[0] in wanted:
            await _run_layer(fixture, *layer)


if __name__ == "__main__":
    coralflow_main()
