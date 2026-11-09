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
"""Runs the cl_coralnpu_core Verilator testbench on a program image and checks the verdict."""

import subprocess
import sys


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: run_cl_coralnpu_tb.py <Vcl_coralnpu_core_tb> <image> [plusargs]"
        )
    cmd = [sys.argv[1], f"+image={sys.argv[2]}", "+ddr_check=1"] + sys.argv[3:]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
    )
    print(proc.stdout)
    if proc.returncode != 0 or "[tb] TEST PASSED" not in proc.stdout:
        raise SystemExit(f"testbench failed (exit code {proc.returncode})")


if __name__ == "__main__":
    main()
