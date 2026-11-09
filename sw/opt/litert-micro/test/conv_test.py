# Copyright 2026 Google LLC
import os
import sys
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

_CONV_SYMBOLS = [
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
    "tflite_ties_enabled",
]


async def run_test_case(
    fixture,
    input_shape,
    filter_shape,
    stride,
    padding_type="VALID",
    output_offset=-128,
):
    # Parameters
    filter_height = filter_shape[1]
    filter_width = filter_shape[2]
    output_depth = filter_shape[0]
    stride_height, stride_width = stride

    input_offset = 128
    activation_min = -128
    activation_max = 127

    if padding_type == "SAME":
        out_h = (input_shape[1] + stride_height - 1) // stride_height
        out_w = (input_shape[2] + stride_width - 1) // stride_width
        pad_h = (
            max(
                0, (out_h - 1) * stride_height + filter_height - input_shape[1]
            ) // 2
        )
        pad_w = max(
            0, (out_w - 1) * stride_width + filter_width - input_shape[2]
        ) // 2
    else:  # VALID
        out_h = (input_shape[1] - filter_height) // stride_height + 1
        out_w = (input_shape[2] - filter_width) // stride_width + 1
        pad_h = 0
        pad_w = 0

    output_shape = [input_shape[0], out_h, out_w, output_depth]

    print(
        f"Testing: Input {input_shape}, Filter {filter_shape}, Stride {stride}, Padding {padding_type}, Output offset {output_offset} -> Output {output_shape}",
        flush=True,
    )

    # Generate random data
    input_data = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
    filter_data = np.random.randint(
        -128, 127, size=filter_shape, dtype=np.int8
    )
    bias_data = np.random.randint(
        -1000, 1000, size=(output_depth, ), dtype=np.int32
    )
    per_channel_multiplier = np.random.randint(
        1073741824, 2147483647, size=(output_depth, ), dtype=np.int32
    )
    per_channel_shift = np.random.randint(
        -10, -1, size=(output_depth, ), dtype=np.int32
    )

    await fixture.write("input_dims", np.array(input_shape, dtype=np.int32))
    await fixture.write("input_data", input_data)
    await fixture.write("filter_dims", np.array(filter_shape, dtype=np.int32))
    await fixture.write("filter_data", filter_data)
    await fixture.write("bias_dims", np.array([output_depth], dtype=np.int32))
    await fixture.write("bias_data", bias_data)
    await fixture.write("output_dims", np.array(output_shape, dtype=np.int32))

    await fixture.write_word("params_stride_width", stride_width)
    await fixture.write_word("params_stride_height", stride_height)
    await fixture.write_word("params_padding_width", pad_w)
    await fixture.write_word("params_padding_height", pad_h)
    await fixture.write_word("params_input_offset", input_offset)
    await fixture.write_word("params_output_offset", output_offset)
    await fixture.write_word("params_activation_min", activation_min)
    await fixture.write_word("params_activation_max", activation_max)
    await fixture.write("per_channel_multiplier", per_channel_multiplier)
    await fixture.write("per_channel_shift", per_channel_shift)

    # Run Simulation
    await fixture.run_to_halt(timeout_cycles=500_000_000)

    # Read Results
    ref_cycles = int.from_bytes((await fixture.read("ref_cycles",
                                                    8)).tobytes(), "little")
    print(f"  Ref Cycles: {ref_cycles}")

    out_size = int(np.prod(output_shape))
    ref_out = await fixture.read(
        "output_data_ref", dtype=np.int8, shape=(out_size, )
    )

    opt_cycles = int.from_bytes((await fixture.read("opt_cycles",
                                                    8)).tobytes(), "little")
    print(f"  Opt Cycles: {opt_cycles}")

    if opt_cycles > 0:
        print(f"  Speedup: {ref_cycles / opt_cycles:.2f}x")

    opt_out = await fixture.read(
        "output_data", dtype=np.int8, shape=(out_size, )
    )

    # Verify
    mismatches = np.sum(opt_out != ref_out)
    if mismatches > 0:
        print(f"  FAILED: {mismatches} mismatches found!", flush=True)
        idx = np.where(opt_out != ref_out)[0][0]
        print(
            f"  First mismatch at index {idx}: Opt {opt_out[idx]}, Ref {ref_out[idx]}"
        )
        sys.exit(1)
    else:
        print("  SUCCESS: Outputs match.", flush=True)


@coralflow_test(highmem=True)
async def run_conv_sim_test(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/conv_test.elf"
    )

    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")

    await fixture.load_elf_and_lookup_symbols(elf_file, _CONV_SYMBOLS)

    test_cases = [
        # 1. Conv_4_4_16_StrideN (ic <= 16)
        {
            "input": [1, 10, 10, 16],
            "filter": [16, 4, 4, 16],
            "stride": (1, 1)
        },
        {
            "input": [1, 10, 10, 16],
            "filter": [16, 4, 4, 16],
            "stride": (2, 2)
        },
        {
            "input": [1, 10, 10, 16],
            "filter": [48, 4, 4, 16],
            "stride": (1, 1)
        },
        # 2. Conv_4_4_48_Stride1 (ic <= 48, stride 1)
        {
            "input": [1, 10, 10, 48],
            "filter": [16, 4, 4, 48],
            "stride": (1, 1)
        },
        # 3. Conv_48_4_4_48 (ic=48, oc=48)
        {
            "input": [1, 10, 10, 48],
            "filter": [48, 4, 4, 48],
            "stride": (1, 1)
        },
        {
            "input": [1, 10, 10, 48],
            "filter": [48, 4, 4, 48],
            "stride": (2, 2)
        },
        # 4. Conv2D_4x4 (Generic 4x4)
        {
            "input": [1, 8, 8, 32],
            "filter": [32, 4, 4, 32],
            "stride": (1, 1)
        },
        {
            "input": [1, 8, 8, 16],
            "filter": [48, 4, 4, 16],
            "stride": (1, 1)
        },
        # 5. fh=3, fw=3: Conv3x3PerChannel with --//sw/opt/litert-micro:rvv_conv,
        # the reference fallback without it.
        {
            "input": [1, 8, 8, 16],
            "filter": [16, 3, 3, 16],
            "stride": (1, 1)
        },
        # Input smaller than the filter: 2x2 input, 3x3 at stride 2, SAME
        # padding only at the bottom and right. No pixel is interior.
        {
            "input": [1, 2, 2, 3],
            "filter": [8, 3, 3, 3],
            "stride": (2, 2),
            "padding": "SAME"
        },
        # Padding on all sides, out_d below one 8-channel block.
        {
            "input": [1, 9, 11, 3],
            "filter": [5, 3, 3, 3],
            "stride": (1, 1),
            "padding": "SAME"
        },
        # Stride 2, odd input: padding on all sides.
        {
            "input": [1, 15, 15, 3],
            "filter": [8, 3, 3, 3],
            "stride": (2, 2),
            "padding": "SAME"
        },
        # Stride 2, even input: padding only at the bottom and right, as in
        # MobileNet's first layer.
        {
            "input": [1, 16, 18, 4],
            "filter": [12, 3, 3, 4],
            "stride": (2, 2),
            "padding": "SAME"
        },
        # Several 8-channel blocks, the last one partial.
        {
            "input": [1, 10, 13, 7],
            "filter": [20, 3, 3, 7],
            "stride": (1, 1),
            "padding": "SAME"
        },
        # 6. fh=1, fw=1, stride 1: Conv1x1PerChannel with rvv_conv. Pixel
        # count not a multiple of 4, channel counts not a multiple of 16.
        {
            "input": [1, 6, 7, 13],
            "filter": [21, 1, 1, 13],
            "stride": (1, 1)
        },
        {
            "input": [1, 5, 5, 40],
            "filter": [5, 1, 1, 40],
            "stride": (1, 1)
        },
        # 1x1 with stride 2 still takes the reference fallback.
        {
            "input": [1, 9, 9, 16],
            "filter": [24, 1, 1, 16],
            "stride": (2, 2)
        },
    ]

    # With output_offset = -128 every negative rounding tie clamps to -128,
    # which hides the rounding difference in #23. The kernels only match TFLite
    # on those ties with --//sw/opt/litert-micro:tflite_ties, so the other
    # offsets run in that build.
    ties = await fixture.read_word("tflite_ties_enabled")
    offsets = (-128, 0, 37) if ties else (-128, )
    for tc in test_cases:
        for output_offset in offsets:
            await run_test_case(
                fixture,
                tc["input"],
                tc["filter"],
                tc["stride"],
                tc.get("padding", "VALID"),
                output_offset=output_offset
            )


if __name__ == "__main__":
    coralflow_main()
