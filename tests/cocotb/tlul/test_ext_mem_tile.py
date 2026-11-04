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
"""Tile-sized conv benchmark on the SoC testharness (#11).

Loads sw/opt/litert-micro/test/ext_mem_tile_bench_{local,ext}.elf into the highmem subsystem, runs
it to halt, and reads back the cycle counts the firmware recorded with mcycle/minstret. Run it
against the highmem testharness without and with external-memory latency; the `local` placement
keeps all data in DTCM, the `ext` placement keeps it in external memory (0x20000000).

Prints a `RESULT placement=... cycles=... instret=... checksum=...` line.
"""

import cocotb
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from coralnpu_test_utils.backdoor import backdoor_load
from coralnpu_test_utils.TileLinkULInterface import (
    TileLinkULInterface,
    create_a_channel_req,
)
from elftools.elf.elffile import ELFFile

# Highmem peripheral space (CSR block of the core).
CORALNPU_RESET_CSR = 0x200000
CORALNPU_PC_CSR = 0x200004

ELF_DIR = "coralnpu_hw/sw/opt/litert-micro/test/"
TIMEOUT_CORE_CYCLES = 400_000_000


async def setup_dut(dut):
    for signal in dir(dut):
        if signal.startswith("io_external_devices_"):
            if "_d_valid" in signal or "_d_bits_" in signal or "_a_ready" in signal:
                try:
                    getattr(dut, signal).value = 0
                except Exception:
                    pass
        elif signal.startswith("io_external_hosts_"):
            if "_a_valid" in signal or "_a_bits_" in signal or "_d_ready" in signal:
                try:
                    getattr(dut, signal).value = 0
                except Exception:
                    pass
    for signal in dir(dut):
        if signal.startswith("io_async_ports_") and "hosts_test" not in signal:
            try:
                if "clock" in signal or "clk" in signal:
                    getattr(dut, signal).value = 0
                elif "reset" in signal or "rst" in signal:
                    getattr(dut, signal).value = 1
            except Exception:
                pass
    for signal in (
            "io_external_ports_te",
            "io_external_ports_dm_req_valid",
            "io_external_ports_dm_req_bits_address",
            "io_external_ports_dm_req_bits_data",
            "io_external_ports_dm_req_bits_op",
            "io_external_ports_dm_rsp_ready",
            "io_external_ports_spi_clk",
            "io_external_ports_spi_mosi",
            "io_external_ports_spim_miso",
            "io_external_ports_spim_clk_i",
            "io_external_ports_gpio_i",
            "io_external_ports_spim_flash_miso",
            "io_external_ports_spim_flash_clk_i",
            "io_external_ports_ext_intrs",
    ):
        try:
            getattr(dut, signal).value = 0
        except Exception:
            pass
    if hasattr(dut, "io_external_ports_spi_csb"):
        dut.io_external_ports_spi_csb.value = 1
    dut.io_external_ports_boot_addr.value = 0

    cocotb.start_soon(Clock(dut.io_clk_i, 10, "ns").start())
    cocotb.start_soon(
        Clock(dut.io_async_ports_hosts_test_clock, 20, "ns").start()
    )
    dut.io_rst_ni.value = 0
    dut.io_async_ports_hosts_test_reset.value = 1
    await ClockCycles(dut.io_clk_i, 5)
    dut.io_rst_ni.value = 1
    dut.io_async_ports_hosts_test_reset.value = 0
    await ClockCycles(dut.io_clk_i, 15)


async def tl_write(host_if, addr, data):
    txn = create_a_channel_req(
        address=addr, data=data, mask=0xF, width=host_if.width, source=0x10
    )
    await host_if.host_put(txn)
    resp = await host_if.host_get_response()
    assert resp["error"] == 0, f"write error at 0x{addr:08x}"


async def tl_read(host_if, addr):
    txn = create_a_channel_req(
        address=addr, width=host_if.width, is_read=True, size=2, source=0x10
    )
    await host_if.host_put(txn)
    resp = await host_if.host_get_response()
    assert resp["error"] == 0, f"read error at 0x{addr:08x}"
    return int(resp["data"])


def load_elf(dut, path):
    """Backdoor-loads the ELF and returns (entry point, symbol table)."""
    with open(path, "rb") as f:
        elf = ELFFile(f)
        for segment in elf.iter_segments(type="PT_LOAD"):
            data = np.frombuffer(segment.data(), dtype=np.uint8)
            if len(data):
                backdoor_load(segment.header.p_paddr, data)
        symbols = {
            s.name: s.entry.st_value
            for s in elf.get_section_by_name(".symtab").iter_symbols()
        }
        return elf.header.e_entry, symbols


async def run_tile(dut, elf_name):
    host_if = TileLinkULInterface(
        dut,
        host_if_name="io_external_hosts_test_host_32",
        clock_name="io_async_ports_hosts_test_clock",
        reset_name="io_async_ports_hosts_test_reset",
        width=32,
    )
    await host_if.init()
    await setup_dut(dut)

    elf_path = runfiles.Create().Rlocation(ELF_DIR + elf_name)
    assert elf_path, f"could not find {elf_name}"
    entry, sym = load_elf(dut, elf_path)

    # Sanity check before the core runs: the ELF is loaded and results start at zero.
    assert await tl_read(
        host_if, sym["tile_done"]
    ) == 0, "results not zero before run"

    await tl_write(host_if, CORALNPU_PC_CSR, entry)
    await tl_write(host_if, CORALNPU_RESET_CSR, 1)  # release clock gate
    await ClockCycles(dut.io_clk_i, 1)
    await tl_write(host_if, CORALNPU_RESET_CSR, 0)  # release reset

    waited = 0
    while dut.io_external_ports_halted.value != 1:
        await ClockCycles(dut.io_clk_i, 10000)
        waited += 10000
        assert waited < TIMEOUT_CORE_CYCLES, "program did not halt"
    assert dut.io_external_ports_fault.value == 0, "program halted with a fault"

    # The 32-bit host reads only the first word of each 128-bit DTCM beat, and the firmware puts
    # every result at the start of its own beat.
    async def result(name):
        return await tl_read(host_if, sym[name])

    done = await result("tile_done")
    assert done == 0x600D, f"benchmark did not finish: tile_done=0x{done:x}"
    cycles = (await result("tile_cycles_lo")
              ) | ((await result("tile_cycles_hi")) << 32)
    instret = (await result("tile_instret_lo")
               ) | ((await result("tile_instret_hi")) << 32)
    checksum = await result("tile_checksum")
    phases = ""
    if "dma" in elf_name or "_db" in elf_name:
        phases = (
            f" dma_in={await result('dma_in_cycles')}"
            f" compute={await result('compute_cycles')}"
            f" dma_out={await result('dma_out_cycles')}"
            f" dma_stall={await result('dma_stall_cycles')}"
        )
    elif "_r4" in elf_name:
        phases = f" compute={await result('compute_cycles')}"
    if "_ref" in elf_name:
        ref_done = await result("tile_ref_done")
        assert ref_done == 0x600D, f"reference comparison did not run: 0x{ref_done:x}"
        mismatches = await result("tile_ref_mismatch")
        phases += f" ref_mismatches={mismatches}"
        assert mismatches == 0, (
            f"optimized kernel differs from the reference kernel in {mismatches} bytes"
        )
    print(
        f"RESULT placement={elf_name} cycles={cycles} instret={instret}{phases} "
        f"checksum=0x{checksum:08x} halted_after_core_cycles~{waited}",
        flush=True,
    )


@cocotb.test()
async def test_tile_local(dut):
    """Tile with input, filter, bias and output in DTCM."""
    await run_tile(dut, "ext_mem_tile_bench_local.elf")


@cocotb.test()
async def test_tile_ext(dut):
    """Tile with input, filter, bias and output in external memory."""
    await run_tile(dut, "ext_mem_tile_bench_ext.elf")


@cocotb.test()
async def test_tile_dma4(dut):
    """Tile copied into DTCM by the DMA (4-byte beats), computed there, output copied back."""
    await run_tile(dut, "ext_mem_tile_bench_dma4.elf")


@cocotb.test()
async def test_tile_dma16(dut):
    """Same as test_tile_dma4 with 16-byte DMA beats."""
    await run_tile(dut, "ext_mem_tile_bench_dma16.elf")


@cocotb.test()
async def test_tile_ref(dut):
    """The all-DTCM tile also run through the TFLite Micro reference kernel; outputs must match."""
    await run_tile(dut, "ext_mem_tile_bench_ref.elf")


# Four rows (double buffering study): all in DTCM as the reference, one DMA transfer at a time,
# and DMA overlapped with compute (4 and 16 are the beat width in bytes).
@cocotb.test()
async def test_tile_local_r4(dut):
    """Four rows with everything in DTCM (reference output)."""
    await run_tile(dut, "ext_mem_tile_bench_local_r4.elf")


@cocotb.test()
async def test_tile_dma4_r4(dut):
    """Four rows, DMA with 4-byte beats, one transfer at a time."""
    await run_tile(dut, "ext_mem_tile_bench_dma4_r4.elf")


@cocotb.test()
async def test_tile_dma16_r4(dut):
    """Four rows, DMA with 16-byte beats, one transfer at a time."""
    await run_tile(dut, "ext_mem_tile_bench_dma16_r4.elf")


@cocotb.test()
async def test_tile_db4_r4(dut):
    """Four rows, double buffered DMA with 4-byte beats."""
    await run_tile(dut, "ext_mem_tile_bench_db4_r4.elf")


@cocotb.test()
async def test_tile_db16_r4(dut):
    """Four rows, double buffered DMA with 16-byte beats."""
    await run_tile(dut, "ext_mem_tile_bench_db16_r4.elf")
