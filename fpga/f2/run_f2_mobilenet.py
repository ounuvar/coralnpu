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
"""Runs fpga/sw/f2_mobilenet.cc on F2 with f2_loader and reports the results (#16).

    bazel build //fpga:run_f2_mobilenet
    sudo bazel-bin/fpga/run_f2_mobilenet --slot=0 --out_dir=results/<commit>-vector

Two f2_loader sessions: the first loads the ELF with read-back, runs it and prints the console and
f2_result; the second reads the f2_mnet results block (raw bytes, so it gets a session of its own).
Everything is saved to --out_dir: the loader output, f2_mnet.bin, and report.txt.
"""

import argparse
import os
import re
import subprocess
import sys

from bazel_tools.tools.python.runfiles import runfiles
from elftools.elf.elffile import ELFFile

import f2_mobilenet

F2_PASS = 0x600DF00D


def symbol(elf_path, name):
    with open(elf_path, "rb") as f:
        syms = ELFFile(f).get_section_by_name(".symtab"
                                              ).get_symbol_by_name(name)
    if not syms:
        raise SystemExit(f"symbol {name} not found in {elf_path}")
    return syms[0]["st_value"]


def main():
    r = runfiles.Create()
    parser = argparse.ArgumentParser(description=__doc__)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--slot", type=int, help="F2 FPGA slot")
    where.add_argument("--pci_bdf", help="application PF, e.g. 0000:34:00.0")
    where.add_argument("--sim_socket", help="local testbench (+host_socket)")
    parser.add_argument(
        "--loader",
        default=r.Rlocation("coralnpu_hw/sw/utils/f2_loader/f2_loader")
    )
    parser.add_argument(
        "--elf", default=r.Rlocation("coralnpu_hw/fpga/f2_mobilenet.elf")
    )
    parser.add_argument(
        "--poll_halt",
        type=float,
        default=600.0,
        help="seconds to wait for the core to halt"
    )
    parser.add_argument("--out_dir", default="f2_mobilenet_results")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="more f2_loader flags, for example --extra=--spi_half=3"
    )
    args = parser.parse_args()

    if args.slot is not None:
        target = [f"--slot={args.slot}"]
    elif args.pci_bdf:
        target = [f"--pci_bdf={args.pci_bdf}"]
    else:
        target = [f"--sim_socket={args.sim_socket}"]
    result_addr = symbol(args.elf, "f2_result")
    mnet_addr = symbol(args.elf, "f2_mnet")
    os.makedirs(args.out_dir, exist_ok=True)

    run = [
        args.loader, *target, f"--load_elf={args.elf}", "--verify",
        "--start_core", f"--poll_halt={args.poll_halt}",
        f"--read_word_addr=0x{result_addr:x}", "--console", *args.extra
    ]
    print(" ".join(run), flush=True)
    p1 = subprocess.run(
        run,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
    )
    with open(os.path.join(args.out_dir, "run.log"), "w") as f:
        f.write(p1.stdout)
    print(p1.stdout, flush=True)
    if p1.returncode != 0:
        raise SystemExit(f"f2_loader run: exit code {p1.returncode}")

    read = [
        args.loader, *target, f"--read_data_addr=0x{mnet_addr:x}",
        f"--read_data_size={f2_mobilenet.SIZE}", *args.extra
    ]
    print(" ".join(read), flush=True)
    p2 = subprocess.run(
        read, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    with open(os.path.join(args.out_dir, "read.log"), "wb") as f:
        f.write(p2.stderr)
    if p2.returncode != 0:
        sys.stderr.write(p2.stderr.decode(errors="replace"))
        raise SystemExit(f"f2_loader read: exit code {p2.returncode}")
    with open(os.path.join(args.out_dir, "f2_mnet.bin"), "wb") as f:
        f.write(p2.stdout)

    results = f2_mobilenet.parse(p2.stdout)
    errors = f2_mobilenet.check(results)
    m = re.search(r"DATA_WORD: 0x([0-9a-f]{8})", p1.stdout)
    if not m or int(m.group(1), 16) != F2_PASS:
        errors.append(f"f2_result: {m.group(0) if m else 'not read'}")
    text = f2_mobilenet.report(results)
    text += "\n" + ("FAIL:\n  " + "\n  ".join(errors) if errors else "PASS")
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
