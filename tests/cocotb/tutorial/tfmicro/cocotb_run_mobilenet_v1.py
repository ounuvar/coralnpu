# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test
import numpy as np

MAX_PROFILE_EVENTS = 128


async def print_layer_profile(fixture, total_cycles):
    """Prints per-op mcycle/minstret recorded by the firmware's CycleProfiler."""
    n_seen = int((await fixture.read_word('profile_num_events')))
    n = min(n_seen, MAX_PROFILE_EVENTS)
    tag_addr = np.asarray(
        await fixture.read('profile_tag_addr', n * 4, dtype=np.uint32)
    )
    cycles = np.asarray(
        await fixture.read('profile_cycles', n * 4, dtype=np.uint32)
    )
    instret = np.asarray(
        await fixture.read('profile_instret', n * 4, dtype=np.uint32)
    )
    # Raw values first, so the data survives even if the formatting below fails.
    print(
        f'RAW_PROFILE n_seen={n_seen} tag_addr={tag_addr.tolist()} '
        f'cycles={cycles.tolist()} instret={instret.tolist()}',
        flush=True,
    )
    rows = []
    for i in range(n):
        raw = bytes(await fixture.read(int(tag_addr[i]), 32))
        tag = raw.split(b'\0')[0].decode(errors='replace')
        rows.append((i, tag, int(cycles[i]), int(instret[i])))
    layer_total = sum(r[2] for r in rows)
    print(
        f'\nPer-layer profile ({n_seen} events recorded, {n} read)',
        flush=True
    )
    print('layer,op,cycles,instructions,ipc,pct_of_layer_cycles', flush=True)
    for i, tag, c, ins in rows:
        ipc = ins / c if c else 0.0
        pct = 100.0 * c / layer_total if layer_total else 0.0
        print(f'{i},{tag},{c},{ins},{ipc:.3f},{pct:.2f}', flush=True)
    print(
        f'TOTAL_LAYER_CYCLES={layer_total} '
        f'TOTAL_LAYER_INSTRUCTIONS={sum(r[3] for r in rows)} '
        f'TOTAL_PROGRAM_CYCLES={total_cycles}',
        flush=True,
    )
    assert n_seen <= MAX_PROFILE_EVENTS, 'Increase kMaxProfileEvents in firmware'


@coralflow_test(highmem=True)
async def core_mini_rvv_mobilenet_v1(fixture):
    r = runfiles.Create()
    await fixture.load_elf_and_lookup_symbols(
        r.Rlocation(
            'coralnpu_hw/tests/cocotb/tutorial/tfmicro/run_mobilenet_v1_025_partial_binary.elf'
        ),
        [
            'inference_status',
            'inference_status_message',
            'profile_num_events',
            'profile_tag_addr',
            'profile_cycles',
            'profile_instret',
        ],
    )
    # NOTE: Running the example in DEBUG mode is too slow could take more than 500Million cycles
    await fixture.run_to_halt(timeout_cycles=130_000_000)
    cycle_count = fixture.get_cycle_count()
    print(f'Total number of execution cycles: {cycle_count} \n', flush=True)
    # Print the profile first so a failing status check cannot hide the measurement.
    profile_error = None
    try:
        await print_layer_profile(fixture, cycle_count)
    except Exception as e:  # pylint: disable=broad-except
        profile_error = e
        print(f'PROFILE_ERROR: {type(e).__name__}: {e}', flush=True)
    # inference_status is a single int8_t in the firmware (read_word returns an unsigned int).
    tflite_inference_status = int(
        np.asarray(await fixture.read('inference_status', 1, dtype=np.int8))[0]
    )
    tflite_inference_message = bytes(
        (await fixture.read('inference_status_message', 31))
    ).split(b'\0')[0].decode(errors='replace')
    assert tflite_inference_status == 0, tflite_inference_message
    if profile_error is not None:
        raise profile_error
    print(' \n Partial mobilenet Invoke() successful \n', flush=True)


if __name__ == '__main__':
    coralflow_main()
