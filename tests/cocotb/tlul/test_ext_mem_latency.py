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
"""Calibration of the external SRAM (0x20000000) latency model, and DMA copy time.

Run against the default testharness (zero extra latency) and against the HBM-latency testharness
(#11). The difference between the two runs is the latency the model adds. Results are printed as
`RESULT ...` lines.

Times are measured on the host side in ns and converted at the core clock period (10 ns). The host
runs on an asynchronous clock, so single accesses carry a few cycles of clock-crossing jitter; the
averages and the difference between runs are what to read.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from cocotb.utils import get_sim_time
from coralnpu_test_utils.TileLinkULInterface import (
    TileLinkULInterface,
    create_a_channel_req,
)

CORE_PERIOD_NS = 10

DMA_BASE = 0x40050000
DMA_CTRL = DMA_BASE + 0x00
DMA_STATUS = DMA_BASE + 0x04
DMA_DESC_ADDR = DMA_BASE + 0x08

SRAM_BASE = 0x20000000


async def setup_dut(dut, host_if):
    cocotb.start_soon(Clock(dut.io_clk_i, CORE_PERIOD_NS, "ns").start())
    cocotb.start_soon(
        Clock(dut.io_async_ports_hosts_test_clock, 2 * CORE_PERIOD_NS,
              "ns").start()
    )
    dut.io_rst_ni.value = 0
    dut.io_async_ports_hosts_test_reset.value = 1
    dut.io_external_ports_gpio_i.value = 0
    await ClockCycles(dut.io_clk_i, 5)
    dut.io_rst_ni.value = 1
    dut.io_async_ports_hosts_test_reset.value = 0
    await ClockCycles(dut.io_clk_i, 100)


def new_host(dut):
    return TileLinkULInterface(
        dut,
        host_if_name="io_external_hosts_test_host_32",
        clock_name="io_async_ports_hosts_test_clock",
        reset_name="io_async_ports_hosts_test_reset",
        width=32,
    )


async def tl_write(host_if, addr, data, source=0x10):
    txn = create_a_channel_req(
        address=addr, data=data, mask=0xF, width=host_if.width, source=source
    )
    await host_if.host_put(txn)
    resp = await host_if.host_get_response()
    assert resp["error"] == 0, f"TL write error at 0x{addr:08X}"


async def tl_read(host_if, addr, source=0x10):
    txn = create_a_channel_req(
        address=addr, width=host_if.width, is_read=True, size=2, source=source
    )
    await host_if.host_put(txn)
    resp = await host_if.host_get_response()
    assert resp["error"] == 0, f"TL read error at 0x{addr:08X}"
    return int(resp["data"])


def cycles_since(t0_ns):
    return (get_sim_time("ns") - t0_ns) / CORE_PERIOD_NS


async def average(n, op):
    total = 0.0
    for i in range(n):
        t0 = get_sim_time("ns")
        await op(i)
        total += cycles_since(t0)
    return total / n


@cocotb.test()
async def test_sram_latency(dut):
    """Average cycles for one read and one write to SRAM, and to a CSR for reference."""
    host_if = new_host(dut)
    await host_if.init()
    await setup_dut(dut, host_if)

    n = 16
    for i in range(n):  # initialise so reads return defined data
        await tl_write(host_if, SRAM_BASE + 4 * i, i)
    sram_read = await average(n, lambda i: tl_read(host_if, SRAM_BASE + 4 * i))
    sram_write = await average(
        n, lambda i: tl_write(host_if, SRAM_BASE + 0x100 + 4 * i, i)
    )
    csr_read = await average(n, lambda i: tl_read(host_if, DMA_STATUS))
    for i in range(n):
        assert await tl_read(host_if, SRAM_BASE + 4 * i) == i
    print(
        f"RESULT sram_read_cycles={sram_read:.1f} sram_write_cycles={sram_write:.1f} "
        f"csr_read_cycles={csr_read:.1f}",
        flush=True,
    )


@cocotb.test()
async def test_dma_copy_time(dut):
    """Cycles for the DMA to copy N bytes SRAM to SRAM, timed from start until STATUS.done."""
    host_if = new_host(dut)
    await host_if.init()
    await setup_dut(dut, host_if)

    src = SRAM_BASE + 0x0000
    dst = SRAM_BASE + 0x8000
    desc = SRAM_BASE + 0x10000
    for size in (64, 512, 4096):
        pattern = [0xA5000000 + size + i for i in range(4)]
        for i, word in enumerate(pattern):
            await tl_write(host_if, src + 4 * i, word)
        words = [
            src,
            dst,
            (size & 0xFFFFFF) | (2 << 24),  # 4-byte beats, no fixed addresses
            0,
            0,
            0,
            0,
            0,
        ]
        for i, word in enumerate(words):
            await tl_write(host_if, desc + 4 * i, word)
        await tl_write(host_if, DMA_DESC_ADDR, desc)
        t0 = get_sim_time("ns")
        await tl_write(host_if, DMA_CTRL, 0x3)  # enable + start
        while True:
            status = await tl_read(host_if, DMA_STATUS)
            if status & 0x2:
                break
            assert cycles_since(t0) < 2_000_000, "DMA did not finish"
        cycles = cycles_since(t0)
        assert not (status & 0x4), f"DMA error 0x{status:08X}"
        for i, word in enumerate(pattern):
            assert await tl_read(host_if, dst + 4 * i) == word
        print(
            f"RESULT dma_bytes={size} cycles={cycles:.0f} "
            f"cycles_per_byte={cycles / size:.2f}",
            flush=True,
        )
        # Clear the done state for the next transfer.
        await tl_write(host_if, DMA_CTRL, 0x0)
