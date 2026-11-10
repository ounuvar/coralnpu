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
"""Baseline per-layer kernel benchmark for MobileNet (#3).

Runs every CONV_2D and DEPTHWISE_CONV_2D layer of MobileNet v1 0.25 at
224x224 (the model in tests/cocotb/tutorial/tfmicro/models) with random data
through the optimized dispatcher, and by default also through the TFLite
reference kernel, requiring bit-identical outputs. On the RVV core the conv
dispatcher falls back to the reference kernel for every MobileNet shape (it
only vectorizes 4x4 filters); depthwise has a real RVV path.

Prints one CSV row per layer. Cycles are meaningful only on RTL (the
verilator target); npusim counts are instruction-level.

Environment:
  MOBILENET_KERNEL_LAYERS=0,1,4   run only these model op indices.
  MOBILENET_KERNEL_SKIP_REF=1     skip the reference run (and the check), for
                                  long RTL timing runs of layers already
                                  checked on npusim.
"""

import os
import sys

import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_SYMBOLS = [
    "input_data",
    "output_opt",
    "output_ref",
    "filter_data",
    "bias_data",
    "kernel",
    "run_reference",
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
    "opt_cycles",
    "opt_instret",
    "ref_cycles",
    "ref_instret",
    "heartbeat",
]

_CONV = 0
_DEPTHWISE = 1

# (model op index, kernel, input HWC, filter HW, output channels, stride,
# padding). Read from mobilenet_v1_0.25_224_int8_dummy.tflite. Stride-2
# depthwise layers follow a PAD op and use VALID padding on the padded input.
_LAYERS = [
    (0, _CONV, (224, 224, 3), (3, 3), 8, 2, "SAME"),
    (1, _DEPTHWISE, (112, 112, 8), (3, 3), 8, 1, "SAME"),
    (2, _CONV, (112, 112, 8), (1, 1), 16, 1, "SAME"),
    (4, _DEPTHWISE, (113, 113, 16), (3, 3), 16, 2, "VALID"),
    (5, _CONV, (56, 56, 16), (1, 1), 32, 1, "SAME"),
    (6, _DEPTHWISE, (56, 56, 32), (3, 3), 32, 1, "SAME"),
    (7, _CONV, (56, 56, 32), (1, 1), 32, 1, "SAME"),
    (9, _DEPTHWISE, (57, 57, 32), (3, 3), 32, 2, "VALID"),
    (10, _CONV, (28, 28, 32), (1, 1), 64, 1, "SAME"),
    (11, _DEPTHWISE, (28, 28, 64), (3, 3), 64, 1, "SAME"),
    (12, _CONV, (28, 28, 64), (1, 1), 64, 1, "SAME"),
    (14, _DEPTHWISE, (29, 29, 64), (3, 3), 64, 2, "VALID"),
    (15, _CONV, (14, 14, 64), (1, 1), 128, 1, "SAME"),
    (16, _DEPTHWISE, (14, 14, 128), (3, 3), 128, 1, "SAME"),
    (17, _CONV, (14, 14, 128), (1, 1), 128, 1, "SAME"),
    (18, _DEPTHWISE, (14, 14, 128), (3, 3), 128, 1, "SAME"),
    (19, _CONV, (14, 14, 128), (1, 1), 128, 1, "SAME"),
    (20, _DEPTHWISE, (14, 14, 128), (3, 3), 128, 1, "SAME"),
    (21, _CONV, (14, 14, 128), (1, 1), 128, 1, "SAME"),
    (22, _DEPTHWISE, (14, 14, 128), (3, 3), 128, 1, "SAME"),
    (23, _CONV, (14, 14, 128), (1, 1), 128, 1, "SAME"),
    (24, _DEPTHWISE, (14, 14, 128), (3, 3), 128, 1, "SAME"),
    (25, _CONV, (14, 14, 128), (1, 1), 128, 1, "SAME"),
    (27, _DEPTHWISE, (15, 15, 128), (3, 3), 128, 2, "VALID"),
    (28, _CONV, (7, 7, 128), (1, 1), 256, 1, "SAME"),
    (29, _DEPTHWISE, (7, 7, 256), (3, 3), 256, 1, "SAME"),
    (30, _CONV, (7, 7, 256), (1, 1), 256, 1, "SAME"),
    (32, _CONV, (1, 1, 256), (1, 1), 5, 1, "SAME"),
]

# #59: stall counters, present when built with --//sw/utils:perf_counters
# (order as PerfCounters.names in hdl/chisel/src/coralnpu/scalar/Csr.scala).
_PERF_NAMES = (
    "memStall", "lsuFull", "tcmWait", "extWait", "fetchWait", "vecDataWait",
    "noDispatch", "rvvFull"
)
_perf = False


def _has_perf(symbols):
    return bool(symbols) and symbols.get("perf_counters_enabled") is not None


def _out_and_pad(in_size, filter_size, stride, padding):
    if padding == "VALID":
        return (in_size - filter_size) // stride + 1, 0
    out_size = (in_size + stride - 1) // stride
    total = max(0, (out_size - 1) * stride + filter_size - in_size)
    return out_size, total // 2  # TFLite puts the extra row/column last


async def _read_u64(fixture, name):
    return int.from_bytes((await fixture.read(name, 8)).tobytes(), "little")


async def _run_layer(
    fixture, check, op_index, kernel, in_hwc, filt_hw, n, stride, padding
):
    # Seeded per layer, so a layer's data does not depend on which others run.
    rng = np.random.default_rng(300 + op_index)
    h, w, c = in_hwc
    fh, fw = filt_hw
    out_h, pad_h = _out_and_pad(h, fh, stride, padding)
    out_w, pad_w = _out_and_pad(w, fw, stride, padding)
    if kernel == _CONV:
        filt = (n, fh, fw, c)
        k = fh * fw * c
    else:
        filt = (1, fh, fw, n)
        k = fh * fw
    # Random int8 data, as in mobilenet_conv_bench.py: the shift scales with K
    # so outputs don't all saturate.
    shift = -int(np.ceil(np.log2(k))) - 6
    await fixture.write_word("kernel", kernel)
    await fixture.write_word("run_reference", 1 if check else 0)
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
    expected = 0x22 if check else 0x12
    if heartbeat != expected:
        print(
            f"layer {op_index}: FAILED, heartbeat {heartbeat:#x}", flush=True
        )
        sys.exit(1)
    size = out_h * out_w * n
    opt = await fixture.read("output_opt", dtype=np.int8, shape=(size, ))
    mismatches = ""
    if check:
        ref = await fixture.read("output_ref", dtype=np.int8, shape=(size, ))
        mismatches = int(np.sum(opt != ref))
    unsaturated = int(np.sum((opt > -128) & (opt < 127)))
    counters = [
        await _read_u64(fixture, "opt_cycles"), await
        _read_u64(fixture, "opt_instret")
    ]
    if check:
        counters += [
            await _read_u64(fixture, "ref_cycles"), await
            _read_u64(fixture, "ref_instret")
        ]
    else:
        counters += ["", ""]
    row = [
        op_index, "conv" if kernel == _CONV else "depthwise", f"{h}x{w}x{c}",
        f"{fh}x{fw}x{n}/s{stride}/{padding}", size * k
    ] + counters + [mismatches, unsaturated, size]
    print("CSV," + ",".join(str(v) for v in row), flush=True)
    if _perf:
        vals = await fixture.read(
            "opt_perf", dtype=np.uint64, shape=(len(_PERF_NAMES), )
        )
        print(
            f"PERF,{op_index},{'conv' if kernel == _CONV else 'depthwise'}," +
            ",".join(str(int(v)) for v in vals),
            flush=True
        )
    if mismatches:
        print(f"layer {op_index}: FAILED, {mismatches} mismatches", flush=True)
        sys.exit(1)


@coralflow_test(highmem=True)
async def run_mobilenet_kernel_bench(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/mobilenet_kernel_bench.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    global _perf
    symbols = await fixture.load_elf_and_lookup_symbols(
        elf_file,
        _SYMBOLS,
        optional_symbols=["perf_counters_enabled", "opt_perf"]
    )
    _perf = _has_perf(symbols)

    selected = os.environ.get("MOBILENET_KERNEL_LAYERS", "")
    wanted = {int(x) for x in selected.split(",") if x} if selected else None
    check = os.environ.get("MOBILENET_KERNEL_SKIP_REF", "") != "1"
    print(
        "CSV,layer,kernel,input,filter,macs,opt_cycles,opt_instret,"
        "ref_cycles,ref_instret,mismatches,unsaturated,outputs",
        flush=True
    )
    if _perf:
        print("PERF,layer,kernel," + ",".join(_PERF_NAMES), flush=True)
    for layer in _LAYERS:
        if wanted is None or layer[0] in wanted:
            await _run_layer(fixture, check, *layer)


if __name__ == "__main__":
    coralflow_main()
