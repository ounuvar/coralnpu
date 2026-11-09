# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs f2_loader (#15) against the cl_coralnpu_core Verilator testbench.

The testbench runs with +host_socket and serves the loader's OCL accesses, so
this is the host software that will run on F2, on the CL's RTL. It loads
fpga/sw/f2_smoke_test.c (with read-back), runs it, and checks the same things
as the testbench's own test: the console, f2_result, and the first DDR line.
"""

import os
import re
import subprocess
import sys
import tempfile

from elftools.elf.elffile import ELFFile

F2_PASS = 0x600DF00D


def ddr_pattern(i):
    return 0xA5000000 ^ ((i * 0x01010101) & 0xFFFFFFFF)


def symbol(elf_path, name):
    with open(elf_path, "rb") as f:
        syms = ELFFile(f).get_section_by_name(".symtab"
                                              ).get_symbol_by_name(name)
    if not syms:
        raise SystemExit(f"symbol {name} not found in {elf_path}")
    return syms[0]["st_value"]


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: run_f2_loader_tb.py <Vcl_coralnpu_core_tb> <f2_loader> <elf>"
        )
    tb, loader, elf = sys.argv[1:]
    result_addr = symbol(elf, "f2_result")
    # Unix socket paths are limited to 108 bytes, so keep it short.
    sock_dir = tempfile.mkdtemp(
        prefix="f2sock", dir="/tmp" if os.path.isdir("/tmp") else None
    )
    sock = os.path.join(sock_dir, "ocl")

    tb_log = os.path.join(
        os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", sock_dir), "tb.log"
    )
    with open(tb_log, "w") as tb_out:
        tb_proc = subprocess.Popen([tb, f"+host_socket={sock}"],
                                   stdout=tb_out,
                                   stderr=subprocess.STDOUT)
        cmd = [
            loader,
            f"--sim_socket={sock}",
            "--spi_half=3",  # 41.7 MHz SCLK keeps the simulation short
            f"--load_elf={elf}",
            "--verify",
            "--start_core",
            "--poll_halt=0.02",  # 20 ms of simulated time
            "--poll_interval_us=8",
            f"--read_word_addr=0x{result_addr:x}",
            "--read_line=0x80000000",
            "--console",
            "--console_idle_us=200",
        ]
        print(" ".join(cmd), flush=True)
        try:
            loader_proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3000,
                check=False
            )
        finally:
            try:
                tb_proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                tb_proc.kill()
    with open(tb_log) as f:
        tb_text = f.read()
    print("---- f2_loader ----")
    print(loader_proc.stdout)
    print("---- testbench ----")
    print(tb_text)

    out = loader_proc.stdout
    errors = []
    if loader_proc.returncode != 0:
        errors.append(f"f2_loader exit code {loader_proc.returncode}")
    if "[tb] HOST SESSION PASSED" not in tb_text:
        errors.append("testbench did not report HOST SESSION PASSED")
    if "Verify OK" not in out:
        errors.append("no read-back verification")
    m = re.search(r"DATA_WORD: 0x([0-9a-f]{8})", out)
    if not m or int(m.group(1), 16) != F2_PASS:
        errors.append(f"f2_result: {m.group(0) if m else 'not read'}")
    want_line = (
        f"0x80000000: 0x{0:08x}{0:08x}"
        f"{ddr_pattern(1):08x}{ddr_pattern(0):08x}"
    )
    if want_line not in out:
        errors.append(f"DDR line 0: expected '{want_line}'")
    console = "\n".join(
        l[len("CONSOLE: "):]
        for l in out.splitlines()
        if l.startswith("CONSOLE: ")
    )
    if "F2" not in console or "PASS" not in console:
        errors.append(f"console: {console!r}")
    if errors:
        raise SystemExit("f2_loader test failed:\n  " + "\n  ".join(errors))
    print("f2_loader test passed")


if __name__ == "__main__":
    main()
