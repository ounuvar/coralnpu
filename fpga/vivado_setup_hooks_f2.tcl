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

# Project setup for chip_f2 (AWS F2, VU47P). Sourced by FuseSoC's Vivado backend after the
# project is created.
#
# chip_f2 has no pins: on F2 it sits inside the AWS custom-logic wrapper, and the AWS build flow
# also synthesizes the CL out of context. So synthesis runs out of context here, and place and
# route runs on the resulting checkpoint with fpga/ooc/ooc_impl.tcl (see fpga/ooc/README.md).

# Same as upstream: an undriven-net warning becomes an error.
set_msg_config -id {[Synth 8-4445]} -new_severity ERROR

set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} \
    -objects [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
