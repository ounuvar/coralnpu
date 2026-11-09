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
"""Matrix-engine conv kernel vs reference_integer_ops::ConvPerChannel (#12)."""

import os
import sys

import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_SYMBOLS = [
    "tflite_ties_enabled",
    "input_dims",
    "input_data",
    "filter_dims",
    "filter_data",
    "bias_dims",
    "bias_data",
    "output_dims",
    "output_data",
    "output_data_ref",
    "params_stride_width",
    "params_stride_height",
    "params_dilation_width",
    "params_dilation_height",
    "params_padding_width",
    "params_padding_height",
    "params_input_offset",
    "params_output_offset",
    "params_activation_min",
    "params_activation_max",
    "per_channel_multiplier",
    "per_channel_shift",
    "ref_cycles",
    "opt_cycles",
    "heartbeat",
]

# Shapes cover the kernel's edge cases: K not a multiple of 4, channel counts
# that are not multiples of 16 or need more than one group of 4 tiles, SAME
# padding, strides, dilation, batches, and pixel counts that are not multiples
# of 16. The 1x1 cases are MobileNet v1 pointwise shapes.
_TEST_CASES = [
    dict(input=[1, 8, 8, 16], filter=[16, 1, 1, 16]),
    dict(input=[1, 7, 7, 64], filter=[80, 1, 1, 64]),
    dict(input=[1, 14, 14, 32], filter=[64, 1, 1, 32]),
    dict(
        input=[1, 9, 9, 3],
        filter=[32, 3, 3, 3],
        stride=(2, 2),
        padding="SAME"
    ),
    dict(input=[1, 10, 10, 16], filter=[20, 3, 3, 16], padding="SAME"),
    dict(input=[2, 6, 5, 8], filter=[72, 3, 3, 8]),
    dict(
        input=[1, 12, 12, 8],
        filter=[16, 3, 3, 8],
        dilation=(2, 2),
        padding="SAME"
    ),
    dict(input=[1, 10, 10, 16], filter=[48, 4, 4, 16]),
    dict(input=[1, 5, 5, 1], filter=[1, 1, 1, 1]),
]


def _geometry(input_shape, filter_shape, stride, dilation, padding):
    stride_h, stride_w = stride
    dil_h, dil_w = dilation
    eff_h = (filter_shape[1] - 1) * dil_h + 1
    eff_w = (filter_shape[2] - 1) * dil_w + 1
    if padding == "SAME":
        out_h = (input_shape[1] + stride_h - 1) // stride_h
        out_w = (input_shape[2] + stride_w - 1) // stride_w
        pad_h = max(0, (out_h - 1) * stride_h + eff_h - input_shape[1]) // 2
        pad_w = max(0, (out_w - 1) * stride_w + eff_w - input_shape[2]) // 2
    else:
        out_h = (input_shape[1] - eff_h) // stride_h + 1
        out_w = (input_shape[2] - eff_w) // stride_w + 1
        pad_h = pad_w = 0
    return [input_shape[0], out_h, out_w, filter_shape[0]], pad_h, pad_w


async def _run_case(
    fixture,
    rng,
    input,
    filter,
    stride=(1, 1),
    dilation=(1, 1),
    padding="VALID"
):
    output_shape, pad_h, pad_w = _geometry(
        input, filter, stride, dilation, padding
    )
    output_depth = filter[0]
    # input_offset = -zero_point; the kernel folds it into the bias, so
    # randomize it. With tflite_ties output_offset is random too, so negative
    # rounding ties reach the output; without it the kernels differ from TFLite
    # on those ties (#23) and -128 clamps them away.
    input_offset = int(rng.integers(-127, 129))
    ties = await fixture.read_word("tflite_ties_enabled")
    output_offset = int(rng.integers(-128, 128)) if ties else -128
    print(
        f"Testing: input {input} filter {filter} stride {stride} "
        f"dilation {dilation} {padding} input_offset {input_offset} "
        f"-> output {output_shape}",
        flush=True
    )

    await fixture.write("input_dims", np.array(input, dtype=np.int32))
    await fixture.write(
        "input_data", rng.integers(-128, 128, size=input, dtype=np.int8)
    )
    await fixture.write("filter_dims", np.array(filter, dtype=np.int32))
    await fixture.write(
        "filter_data", rng.integers(-128, 128, size=filter, dtype=np.int8)
    )
    await fixture.write("bias_dims", np.array([output_depth], dtype=np.int32))
    await fixture.write(
        "bias_data",
        rng.integers(-1000, 1000, size=(output_depth, ), dtype=np.int32)
    )
    await fixture.write("output_dims", np.array(output_shape, dtype=np.int32))
    for name, value in [
        ("params_stride_height", stride[0]),
        ("params_stride_width", stride[1]),
        ("params_dilation_height", dilation[0]),
        ("params_dilation_width", dilation[1]),
        ("params_padding_height", pad_h),
        ("params_padding_width", pad_w),
        ("params_input_offset", input_offset),
        ("params_output_offset", output_offset),
        ("params_activation_min", -128),
        ("params_activation_max", 127),
    ]:
        await fixture.write_word(name, value)
    await fixture.write(
        "per_channel_multiplier",
        rng.integers(
            1 << 30, (1 << 31) - 1, size=(output_depth, ), dtype=np.int32
        )
    )
    await fixture.write(
        "per_channel_shift",
        rng.integers(-10, -1, size=(output_depth, ), dtype=np.int32)
    )

    await fixture.run_to_halt(timeout_cycles=500_000_000)

    heartbeat = int.from_bytes((await fixture.read("heartbeat", 4)).tobytes(),
                               "little")
    if heartbeat != 0x22:
        print(f"  FAILED: heartbeat {heartbeat:#x}", flush=True)
        sys.exit(1)
    ref_cycles = int.from_bytes((await fixture.read("ref_cycles",
                                                    8)).tobytes(), "little")
    opt_cycles = int.from_bytes((await fixture.read("opt_cycles",
                                                    8)).tobytes(), "little")
    out_size = int(np.prod(output_shape))
    ref_out = await fixture.read(
        "output_data_ref", dtype=np.int8, shape=(out_size, )
    )
    vme_out = await fixture.read(
        "output_data", dtype=np.int8, shape=(out_size, )
    )
    print(f"  Ref cycles: {ref_cycles}  VME cycles: {opt_cycles}", flush=True)

    mismatches = int(np.sum(vme_out != ref_out))
    if mismatches:
        idx = int(np.flatnonzero(vme_out != ref_out)[0])
        print(
            f"  FAILED: {mismatches} mismatches; first at {idx}: "
            f"VME {vme_out[idx]}, ref {ref_out[idx]}",
            flush=True
        )
        sys.exit(1)
    # Guard against a vacuous pass where everything saturates.
    unsaturated = int(np.sum((ref_out > -128) & (ref_out < 127)))
    print(
        f"  SUCCESS: outputs match ({unsaturated}/{out_size} unsaturated).",
        flush=True
    )


@coralflow_test(highmem=True)
async def run_conv_vme_test(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/conv_vme_test.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(elf_file, _SYMBOLS)

    rng = np.random.default_rng(12)
    for case in _TEST_CASES:
        await _run_case(fixture, rng, **case)


if __name__ == "__main__":
    coralflow_main()
