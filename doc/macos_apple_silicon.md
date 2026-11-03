# Building on Apple Silicon Macs

CoralNPU's official environment is x86_64 Linux (see
[coralnpu.dockerfile](../utils/coralnpu.dockerfile)). This branch also builds
and simulates natively on Apple Silicon (M1 or newer) Macs, without Docker.
Intel Macs are not supported.

## Setup

1. **Apple tools**

   ```sh
   xcode-select --install                              # clang, macOS SDK, make
   softwareupdate --install-rosetta --agree-to-license # for firtool (x86_64 only)
   ```

2. **Homebrew packages** (from [brew.sh](https://brew.sh))

   ```sh
   brew install bazelisk dtc
   brew tap riscv-software-src/riscv
   brew trust --formula riscv-software-src/riscv/riscv-gnu-toolchain
   brew install riscv-software-src/riscv/riscv-gnu-toolchain
   ```

   `riscv-gnu-toolchain` is the official RISC-V GNU toolchain (GCC, newlib,
   libstdc++). Homebrew's prebuilt bottle needs macOS 15 or newer; on older
   versions it builds from source, which can take an hour or more. Homebrew
   core's `riscv64-elf-gcc` will not work: it ships no C library.

3. **Check the setup**

   ```sh
   utils/check_macos_setup.sh           # a few seconds
   utils/check_macos_setup.sh --build   # also builds a RISC-V binary with Bazel
   ```

   Each failure prints the command that fixes it.

Use `bazelisk` wherever other docs say `bazel`; it runs the version pinned in
`.bazelversion`. Python, Java, Verilator and all other dependencies are
downloaded by Bazel.

## What works

These were run on an M5 Pro with macOS 26:

| Command | Result |
|---|---|
| `bazelisk build //examples:coralnpu_v2_hello_world_add_floats` | Builds the RISC-V ELF |
| `bazelisk build //tests/verilator_sim:core_mini_axi_sim`, then run it with `--binary <elf>` | Runs hello world to completion |
| `bazelisk test //tests/cocotb:core_mini_axi_sim_cocotb_verilator` | 27 pass, 3 skip (RVV-only tests on the scalar core) |
| `bazelisk run //tests/npusim_examples:npusim_run_mobilenet` | MobileNet runs on MPACT, `inference_status 0` |
| `bazelisk test //tests/cocotb:nop_stress_test_uvm` | Passes (RTL vs MPACT vs Spike co-sim) |
| `bazelisk test //tests/cocotb:vme_rvv_load_store_test_uvm` | Passes |

Not tested on macOS yet: the vector-core cocotb suite
(`rvv_core_mini_axi_sim_cocotb_verilator`), the larger UVM suites, FPGA
targets and synthesis. VCS targets need Synopsys tools and do not run on a
Mac.

## Differences from the Linux toolchain

- **Library code is a little slower.** Homebrew's libraries have no build
  for CoralNPU's exact architecture, so GCC links the `rv32im/ilp32`
  versions of newlib, libm and libstdc++. Your own code still uses the full
  `-march` (vector, float, bf16). Library calls such as `memcpy` and libm
  float functions run without hardware float or Zbb instructions. Results are
  the same, but cycle counts for library-heavy code can be higher than on
  Linux. Keep this in mind when benchmarking.
- **GCC must be version 16.1.0.** `toolchain/cc_toolchain_config.bzl` uses
  that version in its include paths. If Homebrew moves to a newer GCC, update
  `gcc_version` there. `check_macos_setup.sh` reports a mismatch.
- **firtool runs under Rosetta.** CIRCT publishes no arm64 macOS build. It is
  used only while generating Verilog from Chisel.

## How it works

On an arm64 Mac, `rules/coralnpu_toolchain.bzl` links Bazel's
`@toolchain_coralnpu_v2` repository to the Homebrew install, instead of
downloading Google's x86_64 Linux toolchain. Set
`CORALNPU_RISCV_TOOLCHAIN=<prefix>` to use a different install. The other
changes are limited to macOS or have no effect on Linux. Each one is
explained in the commit messages on the `apple-silicon` branch and in
comments next to the change.

## Troubleshooting

- **`cannot read spec file 'htif_nano.specs'`, or errors after reinstalling
  the toolchain**: run `bazelisk clean --expunge` so Bazel sets up the
  toolchain repository again.
- **Errors mentioning `x86_64` or `linux` toolchains**: check that your
  terminal isn't running under Rosetta (`uname -m` should print `arm64`).
- **`MODULE.bazel.lock` changes after a build**: Bazel keeps a separate
  lockfile entry per platform. Commit a new entry only for a platform you
  meant to add.
