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

# Timing for the CL logic around chip_f2 (cl_coralnpu_core). clk_main_a0 itself is created by
# the shell (or by clocks_f2.xdc out of context).

# SPI clock into the SoC's SPI device: a clk_main_a0 register toggled every SPI_HALF cycles
# (default 5, so clk_main_a0 / 10 = 25 MHz). Constrain it at the fastest setting the hardware
# allows (SPI_HALF = 3, clk_main_a0 / 6): the edges always fall on clk_main_a0 edges, so this
# also covers MOSI and CSB, which change on clk_main_a0 edges away from the SCLK rising edge.
create_generated_clock -name spi_sclk -divide_by 6 \
    -source [get_pins -hierarchical -filter {NAME =~ *u_spi_host/sclk_o_reg/C}] \
    [get_pins -hierarchical -filter {NAME =~ *u_spi_host/sclk_o_reg/Q}]

# MOSI, CSB and the SoC reset are launched from clk_main_a0 and captured on the SCLK rising
# edge. f2_spi_host changes MOSI and CSB only on the SCLK falling edge or while SCLK is idle,
# so at least SPI_HALF (3 or more) clk_main_a0 cycles (12 ns) away from each rising edge, and
# the reset is released long before the host starts SCLK. Setup: launch 3 cycles before the
# capture edge (a 12 ns requirement). Hold: launch 2 cycles after it (-8 ns, one cycle short of
# the guaranteed 3). The default checks the same clk_main_a0 edge that raises SCLK, which the
# BUFG insertion delay on SCLK cannot meet.
set_multicycle_path -setup -start 3 -from [get_clocks clk_main_a0] -to [get_clocks spi_sclk]
set_multicycle_path -hold -start 4 -from [get_clocks clk_main_a0] -to [get_clocks spi_sclk]

# The SoC's SPI device crosses into the core clock through its own asynchronous queues
# (Spi2TLULV2). Upstream (pins_nexus.xdc) puts the SPI clock in its own asynchronous group; do
# the same here, against every clock of the chip_f2 MMCM. Only the derivation from clk_main_a0
# makes Vivado treat them as related.
set_clock_groups -asynchronous -group [get_clocks spi_sclk] \
    -group [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ *i_clkgen/pll/CLKOUT*}]]

# MISO comes back from the SPI clock domain through a 2-flop synchronizer.
set_false_path -to [get_pins -hierarchical -filter {NAME =~ *u_spi_host/miso_q1_reg/D}]

# UART1 TX from the core clock domain into the CL's receiver, through a 2-flop synchronizer.
set_false_path -to [get_pins -hierarchical -filter {NAME =~ *u_uart_rx/rx_q1_reg/D}]
