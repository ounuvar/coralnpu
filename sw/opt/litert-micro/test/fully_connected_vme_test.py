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
"""Matrix-engine FC kernel vs reference_integer_ops::FullyConnected (#12)."""

import os
import sys

import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import (
    coralflow_main, coralflow_test, get_target_environment
)
import tflite_int8_ref

_SYMBOLS = [
    "tflite_ties_enabled",
    "input_data",
    "filter_data",
    "bias_data",
    "output_data",
    "output_data_ref",
    "batches",
    "accum_depth",
    "output_depth",
    "params_input_offset",
    "params_output_offset",
    "params_output_multiplier",
    "params_output_shift",
    "params_activation_min",
    "params_activation_max",
    "ref_cycles",
    "opt_cycles",
    "heartbeat",
    "run_reference",
]

# (batches, accum_depth, output_depth). Covers batch 1 (one useful tile row),
# batches above 16, depths that are not multiples of 4 or 16, and an output
# depth past one group of 4 tiles. The last is MobileNet v1's FC with the
# class count cut to fit the test buffers.
_TEST_CASES = [
    (1, 64, 16),
    (1, 250, 37),
    (3, 128, 100),
    (17, 33, 70),
    (1, 1024, 200),
]


async def _run_case(fixture, rng, batches, accum_depth, output_depth):
    input_offset = int(rng.integers(-127, 129))
    print(
        f"Testing: batches {batches} accum_depth {accum_depth} "
        f"output_depth {output_depth} input_offset {input_offset}",
        flush=True
    )
    input_data = rng.integers(
        -128, 128, size=(batches * accum_depth, ), dtype=np.int8
    )
    await fixture.write("input_data", input_data)
    filter_data = rng.integers(
        -128, 128, size=(output_depth * accum_depth, ), dtype=np.int8
    )
    await fixture.write("filter_data", filter_data)
    bias_data = rng.integers(
        -1000, 1000, size=(output_depth, ), dtype=np.int32
    )
    await fixture.write("bias_data", bias_data)
    # Scale the right shift with the accumulator size so outputs do not all
    # saturate. output_offset is random with tflite_ties, so negative rounding
    # ties show, and -128 without it (#23).
    ties = await fixture.read_word("tflite_ties_enabled")
    output_offset = int(rng.integers(-128, 128)) if ties else -128
    shift = -int(np.ceil(np.log2(accum_depth))) - 7
    multiplier = int(rng.integers(1 << 30, (1 << 31) - 1))
    for name, value in [
        ("batches", batches),
        ("accum_depth", accum_depth),
        ("output_depth", output_depth),
        ("params_input_offset", input_offset),
        ("params_output_offset", output_offset),
        ("params_output_multiplier", multiplier),
        ("params_output_shift", shift),
        ("params_activation_min", -128),
        ("params_activation_max", 127),
    ]:
        await fixture.write_word(name, value)
    host_ref = tflite_int8_ref.fully_connected(
        input_data.reshape(batches, accum_depth),
        filter_data.reshape(output_depth, accum_depth), bias_data, multiplier,
        shift, input_offset, output_offset, -128, 127
    ).reshape(-1)
    on_core_ref = get_target_environment() == "npusim"
    await fixture.write_word("run_reference", int(on_core_ref))

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
    out_size = batches * output_depth
    if on_core_ref:
        # The C reference runs only here; the host copy must match it exactly.
        ref_out = await fixture.read(
            "output_data_ref", dtype=np.int8, shape=(out_size, )
        )
        host_diff = int(np.sum(host_ref != ref_out))
        if host_diff:
            print(
                f"  FAILED: host reference differs from TFLite's in "
                f"{host_diff} outputs",
                flush=True
            )
            sys.exit(1)
    else:
        ref_out = host_ref
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
    unsaturated = int(np.sum((ref_out > -128) & (ref_out < 127)))
    print(
        f"  SUCCESS: outputs match ({unsaturated}/{out_size} unsaturated).",
        flush=True
    )


@coralflow_test(highmem=True)
async def run_fully_connected_vme_test(fixture):
    r = runfiles.Create()
    elf_file = r.Rlocation(
        "coralnpu_hw/sw/opt/litert-micro/test/fully_connected_vme_test.elf"
    )
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found: {elf_file}")
    await fixture.load_elf_and_lookup_symbols(elf_file, _SYMBOLS)

    rng = np.random.default_rng(34)
    for case in _TEST_CASES:
        await _run_case(fixture, rng, *case)


if __name__ == "__main__":
    coralflow_main()
