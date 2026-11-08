# The AWS F2 shell's main clock into the CL: 250 MHz. The MMCM in chip_f2 derives the core
# (ClockFrequencyMhz), SPI master and ISP clocks from it, so Vivado generates those constraints.
create_clock -name clk_main_a0 -period 4.000 [get_ports clk_main_a0]
