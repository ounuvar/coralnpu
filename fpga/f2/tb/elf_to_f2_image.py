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
"""Converts an ELF into the text image the F2 testbench loads over SPI.

Only the loadable bytes of each PT_LOAD segment are emitted (not the gaps between ITCM and
DTCM), padded to 16-byte SPI beats. Format, one record per line, hex without 0x:

  S <addr> <beats>     segment start (16-byte aligned) and length in 16-byte beats
  <w0> <w1> <w2> <w3>  one line per beat, 32-bit little-endian words in address order
  Y <name> <addr>      address of a symbol listed with --symbol
  E <entry>            entry point, last line
"""

import argparse

from elftools.elf.elffile import ELFFile

BEAT = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf")
    parser.add_argument("out")
    parser.add_argument("--symbol", action="append", default=[])
    args = parser.parse_args()

    with open(args.elf, "rb") as f:
        elf = ELFFile(f)
        lines = []
        for seg in elf.iter_segments():
            if seg["p_type"] != "PT_LOAD" or seg["p_filesz"] == 0:
                continue
            addr = seg["p_paddr"]
            data = seg.data()
            start = addr - (addr % BEAT)
            data = bytes(addr - start) + data
            data += bytes((-len(data)) % BEAT)
            lines.append(f"S {start:08x} {len(data) // BEAT:x}")
            for off in range(0, len(data), BEAT):
                words = [
                    int.from_bytes(data[off + i:off + i + 4], "little")
                    for i in range(0, BEAT, 4)
                ]
                lines.append(" ".join(f"{w:08x}" for w in words))
        symtab = elf.get_section_by_name(".symtab")
        for name in args.symbol:
            syms = symtab.get_symbol_by_name(name) if symtab else None
            if not syms:
                raise SystemExit(f"symbol {name} not found in {args.elf}")
            lines.append(f"Y {name} {syms[0]['st_value']:08x}")
        lines.append(f"E {elf.header['e_entry']:08x}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
