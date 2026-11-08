#!/bin/bash
# Out-of-context synthesis, place and route of CoralNPUChiselSubsystem on the VU47P.
# See README.md in this directory.
#
# Usage: fpga/ooc/run_ooc.sh <CoralNPUChiselSubsystem.sv> <run_dir> [clock_ns] [extra defines]
#   clock_ns       default 20 (50 MHz, the project target clock)
#   extra defines  space-separated list added to the baseline defines, e.g. "IDEA1_ON"
#
# Writes synth.log and impl.log to <run_dir>, and checkpoints and reports to
# <run_dir>/synth and <run_dir>/out. Takes about 2 hours in total.
#
# VIVADO_BIN and XILINXD_LICENSE_FILE default to the UT ECE LRC servers.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '5,7p' "$0" >&2
  exit 2
fi

sv=$(realpath "$1")
run_dir=$2
clock_ns=${3:-20.000}
extra=${4:-}
script=$(realpath "$0")
ooc_dir=$(dirname "${script}")

export XILINXD_LICENSE_FILE=${XILINXD_LICENSE_FILE:-2100@harrison.ece.utexas.edu}
VIVADO_BIN=${VIVADO_BIN:-/usr/local/packages/xilinx_2023.2/Vivado/2023.2/bin}
export PATH=${VIVADO_BIN}:${PATH}

mkdir -p "${run_dir}"
run_dir=$(realpath "${run_dir}")
# Vivado writes .Xil/ and its journal into the working directory.
cd "${run_dir}"

vivado -mode batch -nojournal -log synth.log \
  -source "${ooc_dir}/ooc_synth.tcl" \
  -tclargs "${sv}" "${run_dir}/synth" "${clock_ns}" "${extra}"

vivado -mode batch -nojournal -log impl.log \
  -source "${ooc_dir}/ooc_impl.tcl" \
  -tclargs "${run_dir}/synth" "${clock_ns}" "${run_dir}/out"

grep -E "nets with routing errors" "${run_dir}/out/impl_route_status.rpt"
grep -E -A3 "WNS\(ns\)" "${run_dir}/out/impl_timing.rpt" | sed -n 3p
