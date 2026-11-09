# CoralNPU on AWS F2

This directory holds the custom logic (CL) that puts the CoralNPU SoC on an AWS F2 card (VU47P). It wraps `chip_f2` (`fpga/rtl/chip_f2.sv`, #13) with what the AWS shell needs: the host interface, external memory, clocks and reset (#14). The host software that loads programs and reads results is `f2_loader` (`sw/utils/f2_loader`, #15).

## Structure

| Block | File | Role |
| ----- | ---- | ---- |
| `cl_coralnpu` | `design/cl_coralnpu.sv` | HDK top: the shell ports (`cl_ports.vh`), `sh_ddr`, tie-offs for unused shell interfaces |
| `cl_coralnpu_core` | `../rtl/cl_coralnpu_core.sv` | Everything else, with no HDK dependency, so it also runs in the local testbench |
| `f2_ocl_regs` | `../rtl/f2_ocl_regs.sv` | Host registers on OCL (BAR0): reset, status, SPI loader, UART console |
| `f2_spi_host` | `../rtl/f2_spi_host.sv` | SPI controller that drives the SoC's SPI device port with the upstream load protocol |
| `f2_uart_rx` | `../rtl/f2_uart_rx.sv` | Captures UART1 (the console) into a FIFO the host reads |
| `f2_fifo` | `../rtl/f2_fifo.sv` | FIFO in distributed RAM for the SPI command, SPI RX and console buffers (`prim_fifo_sync` keeps its storage in flip-flops) |
| `f2_ddr_upsizer` | `../rtl/f2_ddr_upsizer.sv` | SoC 256-bit DDR AXI to the shell's 512-bit `sh_ddr` AXI |
| `f2_ddr_ctrl_stub` | `../rtl/f2_ddr_ctrl_stub.sv` | Answers the SoC's DDR controller register window, which `sh_ddr` does not have |

Clocks and reset: everything runs from the shell's `clk_main_a0` (clock recipe A0, 250 MHz) and `rst_main_n`. The MMCM in `chip_f2` makes the 50 MHz core clock. The host can hold the SoC in reset through `CTRL[0]`.

Host path: the host writes OCL registers, and `f2_spi_host` turns them into SPI frames for the SoC's `Spi2TLULV2` bridge, which reads and writes any SoC address (ITCM, DTCM, CSRs, DDR). This reuses the upstream loader protocol and needs no change to the SoC. PCIS (host DMA) is not used. If loading over OCL turns out too slow for large images, a PCIS path into the SoC is the next step.

DDR: the SoC's window at `0x8000_0000` (2 GB) maps to DDR offset 0. Every SoC DDR access is a single beat of at most 32 bytes (`TLUL2Axi` never bursts), so `f2_ddr_upsizer` places it in the right half of a 512-bit beat. It carries address bit 5 in AXI ID bit 1 so that read data comes back from the right half. Software should wait for DDR calibration (`STATUS[2]`, also readable by the SoC at `0x7000_0000`) before touching DDR.

## Host registers (OCL, BAR0)

All registers are 32 bits. Every access gets an OKAY response, and unmapped offsets read 0.

| Offset | Name | Access | Description |
| ------ | ---- | ------ | ----------- |
| `0x00` | `ID` | RO | `0x434E5055` ("CNPU") |
| `0x04` | `VERSION` | RO | `0x00010000` |
| `0x08` | `CTRL` | RW | `[0]` hold the SoC in reset (default 0, the SoC runs out of reset) |
| `0x0C` | `STATUS` | RO, W1C | `[0]` halted, `[1]` fault, `[2]` DDR ready, `[3]` SPI busy, `[4]` SPI RX overflow (W1C), `[5]` UART overflow (W1C) |
| `0x10` | `SPI_HALF` | RW | SPI clock half period in `clk_main_a0` cycles, 3 or more (default 5: 25 MHz) |
| `0x14` | `UART_BIT` | RW | UART bit time in `clk_main_a0` cycles (default 2170: 115200 baud) |
| `0x20` | `SPI_CSB` | WO, queued | Set chip select to `data[0]` (0 selects) |
| `0x24` | `SPI_TX1` | WO, queued | Send byte `data[7:0]` |
| `0x28` | `SPI_TX4` | WO, queued | Send bytes `data[7:0]`, `[15:8]`, `[23:16]`, `[31:24]` (a little-endian word) |
| `0x2C` | `SPI_RX` | WO, queued | Send `data[15:0]` zero bytes and capture what comes back into the RX FIFO |
| `0x30` | `SPI_IDLE` | WO, queued | Send `data[15:0]` zero bytes and discard what comes back |
| `0x34` | `SPI_RXDATA` | RO, pops | `[8]` valid, `[7:0]` received byte |
| `0x38` | `SPI_RXCOUNT` | RO | Bytes in the RX FIFO (256 deep) |
| `0x3C` | `SPI_CMDFREE` | RO | Free entries in the SPI command queue (32 deep) |
| `0x40` | `UART_RXDATA` | RO, pops | `[8]` valid, `[7:0]` console byte |
| `0x44` | `UART_RXCOUNT` | RO | Bytes in the console FIFO (1024 deep) |
| `0x48` | `CYCLES_LO` | RO | Free-running `clk_main_a0` counter, low word (latches `CYCLES_HI`) |
| `0x4C` | `CYCLES_HI` | RO | High word, latched by the last `CYCLES_LO` read |

The SPI command queue runs in order. A write to a queued register when the queue is full is held until an entry frees up, so check `SPI_CMDFREE` before long bursts. The shell times out OCL accesses that stall too long.

## Loading and running a program

The steps match the upstream cocotb SPI test (`tests/cocotb/tlul/test_subsystem.py`). `verif/tests/f2_host_tasks.svh` implements them, and both testbenches use it.

1. Write `SPI_CSB` = 1 and `SPI_IDLE` = 4 once after reset.
2. Write each 16-byte beat of the program. A frame is: `SPI_CSB` = 0, then the header `[0x02, address (4 bytes, big endian), beats - 1 (2 bytes, big endian)]`, then the data as `SPI_TX4` words in address order, then `SPI_IDLE` = 16 (the bridge only moves data while the SPI clock runs), then `SPI_CSB` = 1. Use frames of up to 64 beats.
3. To read a 16-byte line, send the header with op `0x01` and length 0, then `SPI_RX` = 200, `SPI_IDLE` = 4, `SPI_CSB` = 1. Wait for `STATUS[3]` to clear and pop `SPI_RXCOUNT` bytes. Treat them as one bit stream, MSB of each byte first, find the first `0xFE` at any bit offset (DDR reads in particular come back shifted by a few bits), and take the next 128 bits as 16 bytes, lowest address first.
4. Start the core through its CSRs, which are at `0x0020_0000` in the highmem configuration: write the entry point to `0x0020_0004`, then 1 and then 0 to `0x0020_0000`. SPI writes are whole 16-byte lines, so read, modify and write the line.
5. Poll `STATUS[0]` (halted) and `STATUS[1]` (fault). Read the console from `UART_RXDATA`, and read results from memory over SPI.

## Host software: `f2_loader`

`//sw/utils/f2_loader` implements the steps above. It has the same flags and order of operations as the upstream `//sw/utils/nexus_loader`, so scripts written for one can drive the other:

```bash
sudo f2_loader --slot=0 --load_elf=prog.elf --verify --start_core --poll_halt=10 --console
sudo f2_loader --slot=0 --read_data_addr=0x80000000 --read_data_size=4096 > out.bin
```

- **Where the CL is:** `--slot=N` (the Nth F2 FPGA), `--pci_bdf=0000:34:00.0`, or `--sim_socket=<path>` for the local testbench. On F2 it maps BAR0 of the application PF through sysfs (`/sys/bus/pci/devices/<BDF>/resource0`), the mapping the aws-fpga SDK's `fpga_pci_attach` makes, so it needs no SDK. It needs root and a loaded AFI (`fpga-load-local-image`).
- **Differences from `nexus_loader`:**
  - The CSR base defaults to highmem (`0x200000`), which every F2 build is.
  - `--start_core` without `--set_entry_point` uses the ELF's entry point.
  - Halt and fault come from `STATUS` rather than over SPI, and a fault ends `--poll_halt` with an error.
  - `--soft_reset` holds the SoC in reset through `CTRL[0]`.
  - There is no `--reset`: reloading the AFI is `fpga-load-local-image`.
  - `--console` prints the UART console as `CONSOLE:` lines once it stops growing.
  - `--spi_half` sets the SPI clock.
- **No stalled OCL access:** the loader reads `SPI_CMDFREE` before writing queued registers, so it never writes into a full command queue. A full queue would stall the OCL write until the shell times it out.

## Local testbench

`tb/cl_coralnpu_core_tb.sv` runs `cl_coralnpu_core` in Verilator with a behavioral clock generator (Verilator cannot simulate the MMCM) and a 512-bit DDR model. It plays the host over OCL only. It loads `fpga/sw/f2_smoke_test.c` over SPI, runs it, and checks the console, the result word, and DDR in both 256-bit halves of a beat, through the SoC and directly in the model. Then it checks the soft reset.

With `+host_socket=<path>`, the testbench instead serves OCL accesses from `f2_loader --sim_socket=<path>` over a Unix socket (`tb/f2_host_socket.cc`). Simulated time stops between requests, and the loader's sleeps become waits of simulated `clk_main_a0` cycles. `cl_coralnpu_tb_{vector,matrix}_loader_test` uses this to run the loader on the RTL: it loads the smoke test with read-back, runs it, and checks the console, the result word and the first DDR line.

```bash
bazel test //fpga:cl_coralnpu_tb_vector_test          # or _matrix; builds the simulator first
bazel test //fpga:cl_coralnpu_tb_vector_loader_test   # f2_loader against the testbench
bazel test //sw/utils/f2_loader:all                   # f2_loader unit tests
```

The test is tagged manual. Building the simulator takes several minutes.

## Connecting to the AWS instance

The F2 instance runs on EC2 and is reached over SSH with the team's key pair (`mlxlo-team-5.pem`). Each person keeps the key in their own location. Never commit it.

1. Restrict the key's permissions once, or SSH refuses to use it:

   ```bash
   chmod 400 /path/to/mlxlo-team-5.pem
   ```

2. Look up the instance's current public DNS name before every session. It changes each time the instance is stopped and started, so an old name will not work. In the EC2 console, open Instances, select the instance and copy "Public IPv4 DNS", for example `ec2-34-201-23-195.compute-1.amazonaws.com`. Use the host name only, without `https://`.

3. Connect. The login user is `mlxlo-team-5` for everyone:

   ```bash
   ssh -i /path/to/mlxlo-team-5.pem mlxlo-team-5@<public-dns>
   ```

## Building and simulating on AWS

The HDK needs Vivado 2024.1 or newer, so this part runs on the AWS instance, not on the LRC servers.

1. Package the CL on any machine with the repo (it runs FuseSoC and a Vivado elaboration):

   ```bash
   bazel build //fpga:cl_coralnpu_hdk_vector   # or _matrix
   ```

   This writes `cl_coralnpu_vector.tar.gz`: the CL in the HDK layout, with the SoC sources and a generated, ordered source list (`build/scripts/cl_coralnpu_sources.tcl`, `verif/scripts/top.*.f`).

2. On the AWS instance, set up the HDK from [aws-fpga](https://github.com/aws/aws-fpga) (branch `f2`) and unpack the CL:

   ```bash
   source hdk_setup.sh
   tar -xzf cl_coralnpu_vector.tar.gz -C $HDK_DIR/cl/developer_designs/
   export CL_DIR=$HDK_DIR/cl/developer_designs/cl_coralnpu
   ```

3. Simulate with the HDK testbench (shell model, DDR model):

   ```bash
   cd $CL_DIR/verif/scripts
   make test_coralnpu_regs
   make test_coralnpu_smoke PLUSARGS=+image=/path/to/f2_smoke_test.f2img
   ```

   Build the image with `bazel build //fpga:f2_smoke_test_image`.

4. Build the design checkpoint for an AFI:

   ```bash
   cd $CL_DIR/build/scripts
   ./aws_build_dcp_from_cl.py -c cl_coralnpu
   ```

   The timing constraints for the CL logic are in `build/constraints/cl_timing_user.xdc`, copied from `build/constraints/cl_coralnpu_timing.xdc`: the generated SPI clock, a 3-cycle multicycle path from `clk_main_a0` into it (MOSI and CSB change only on the SCLK falling edge), an asynchronous clock group between it and the `chip_f2` MMCM clocks (as upstream does for the SPI device clock), and false paths into the MISO and UART synchronizers. If the paths between the SoC and `sh_ddr` miss timing at 250 MHz, add the HDK's `axi_register_slice` between `cl_coralnpu_core` and `sh_ddr`.

## Not done yet

- These HDK steps have not been run: they need the AWS instance (#16, #17).
- JTAG debug is held in reset. The F2 shell's virtual JTAG could drive it later.
- `f2_loader` has run only against the local testbench; its F2 backend runs on the AWS instance (#16, #17).
