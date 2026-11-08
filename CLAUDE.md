# CLAUDE.md

@AGENTS.md

This file holds the project's rules and method, for people and agents alike. Everything else (tasks, owners, status, results, decisions) lives in the GitHub issues on [`ounuvar/coralnpu`](https://github.com/ounuvar/coralnpu/issues), viewed on the project board. Do not add other planning files.

## Project context

This checkout is the base for a UT Austin HW/SW co-design course project, "HW/SW Optimization and FPGA Prototyping of Google Coral NPU" (Aayush Merchant, Orhan Unuvar, Christopher Rodolph). The goal is to speed up quantized (INT8) CNN inference (MobileNet v1) on Coral NPU, measured against the unmodified upstream design, and to prototype the result on an AWS F2 FPGA (AMD Virtex UltraScale+ HBM VU47P).

A result counts only if it is reported on all three axes:

- **Software:** cycles per kernel and per MobileNet layer, full-model cycles, instruction count, memory stall cycles.
- **Hardware:** added LUTs, FFs, DSPs, BRAM/URAM, power, and WNS at the target clock on the VU47P. The target clock is 50 MHz (decided on #6), measured with the out-of-context flow in `fpga/ooc/README.md`. A speedup that breaks timing closure is not a result.
- **Deployment:** the configuration builds, meets timing, and runs MobileNet correctly on F2.

## Phases and the five ideas

| Phase | What it produces | Issues |
| ----- | ---------------- | ------ |
| 0. Environment and baseline | Working builds; frozen `baseline-v0`; kernel, MobileNet and hardware baselines; profile | #1 to #7 |
| 1. The five ideas | Each idea behind a switch, tested and measured | #8 to #12 |
| 2. FPGA on AWS F2 | Baseline and every configuration running on F2 | #13 to #17 |

Phase 0 blocks phase 1. Phase 2 is the highest-risk track and runs in parallel from the start.

| # | Idea | HW | SW | Where to start |
| - | ---- | -- | -- | -------------- |
| 1 | INT8 dot-product instruction (4 × INT8 MACs into an INT32 accumulator) | yes | yes | `hdl/verilog/rvv/design/rvv_backend_mulmac.sv`, `rvv_backend_mac_unit.sv`, decode in `rvv_backend_decode_unit_ari.sv` and `hdl/chisel/src/coralnpu/rvv/RvvDecode.scala` |
| 2 | Fused requantization instruction (INT32 to INT8: scale, round, add offset, clamp) | yes | yes | Same RVV backend. The current multi-instruction path is in `sw/opt/litert-micro/accumulator_util.h` and `sw/opt/rvv_opt.h` |
| 3 | Depthwise convolution unit | yes | yes | Baseline kernel in `sw/opt/litert-micro/depthwise_conv.cc` (RVV intrinsics) |
| 4 | DMA tiling and double buffering into the 32 KB DTCM | no | yes | `hdl/chisel/src/dma/`, `hdl/chisel/src/bus/DmaEngine.scala`, driver `fpga/sw/dma.c`, docs `doc/sw/dma.md` and `doc/peripherals/dma.md` |
| 5 | Per-layer choice between the vector unit and the matrix engine | yes | yes | Matrix engine (VME/Zvt) in `hdl/verilog/rvv/design/Zvt/`, enabled by `enableVme` in `hdl/chisel/src/coralnpu/Parameters.scala`, tests in `tests/cocotb/vme_test/` |

Upstream has no dedicated dot-product or requantization instruction. Confirm this again before designing encodings (`git grep -i dot`, `git grep -i requant`).

## Ground rules

- **Baseline is frozen.** The unmodified commit is tagged `baseline-v0`. Every number is reported relative to it, with the commit hash it was measured on.
- **Every idea is a switch.** Hardware behind a `Parameters.scala` flag (default off, like `enableVme`) plus a `-D` define for the SystemVerilog side; kernel paths behind a compile-time flag. A build with all switches off must behave exactly like the baseline.
- **Correctness before speed.** Every kernel is checked bit-exact against the TFLite Micro reference kernels (`@tflite_micro//tensorflow/lite/kernels/internal:reference_base`) before it is timed.
- **Requantization must match TFLite rounding exactly** (`MultiplyByQuantizedMultiplier`: rounding-doubling high multiply plus rounding right shift). Test saturation, negative shifts, and per-channel multipliers.
- **Cycles come from RTL or FPGA, not from npusim.** `npusim` is the MPACT functional ISS: instruction counts and correctness only. Use `mcycle`/`minstret` (see `tests/cocotb/isa_cycle_bench.cc`, `sw/utils/metrics.py`).
- **Reported numbers come from the pinned Linux toolchain** (x86 or CI). Macs use a Homebrew RISC-V GCC that may compile kernels differently; Mac numbers are for debugging only.
- **Results and decisions go on the issue.** Post each measurement as a comment on its issue: configuration, commit, and the numbers. Record decisions (encodings, target clock, whether to build a depthwise unit) the same way.
- **One branch per issue, merged by pull request** into `main` with `Closes #N`, after CI passes. Keep pull requests small and rebase on `main` often.
- **GitHub Actions runs only the pull-request checks.** Never start the full `bazel test //...` (5 to 6 hours) on GitHub, by `gh workflow run` or the Run workflow button: run it by hand on a team Linux machine (see Machines). No self-hosted runners, since the repo is public. Verify only the targets you changed and batch fixes into fewer pushes. Keep the `push` trigger on `main` and add no `paths-ignore` for Markdown or `doc/`: with `strict: false` branch protection the post-merge run is the only check of what landed, and a path-skipped required check (`lint`) blocks docs-only pull requests. If GitHub disables workflows on the fork, the repo owner re-enables them on the Actions tab; then close and reopen any pull request that has no runs.

## Working on an issue

1. **Pick it up.** Assign yourself and move the card to In progress on the board.
2. **Branch from fresh `main`:** `git switch main && git pull && git switch -c <issue>-<short-name>` (for example `8-dot-product`).
3. **Work and test locally.** Run only the tests your change touches (see Common commands); the full suite is not part of the loop.
4. **Lint before pushing, where you can:** `BASE_BRANCH=origin/main ./utils/run_linters.sh --fix`. The script stops at the first missing linter. On a Mac without them, `pip install "clang-format==19.*" yapf` in a virtual environment covers C++ (`git-clang-format`) and Python (`yapf`); CI runs the rest.
5. **Open a pull request** with `Closes #<issue>` in the description and move the card to In review. CI takes a few minutes: lint, macro signatures, and the RTL Verilog build.
6. **Merge it yourself once CI passes.** Ask a teammate to review first only if you touched shared ground: the RVV decode or multiply files, `MODULE.bazel*`, `.bazelrc`, or `toolchain/`. Delete the branch after merging.
7. **Report on the issue.** Post measurements (configuration, commit, numbers) and decisions as comments on the issue before closing it.

Never push directly to `main`. The full `bazel test //...` (5 to 6 hours) runs only by hand on a team Linux machine, never on GitHub (see Ground rules), for example before measuring a configuration for the report.

## Recipe for a new vector instruction (ideas 1, 2, and 3 if it becomes an instruction)

Do the steps in order; each check must pass before the next step. Steps 1 to 4 need no RTL, so the instruction-count benefit is known before anyone touches hardware.

| Step | Work | Check |
| ---- | ---- | ----- |
| 1. Spec | Encoding, operand types, rounding and saturation rules, scalar C reference model | Team review |
| 2. Golden tests | C model vs TFLite Micro reference, including edge cases | Bit-exact |
| 3. ISS | New patch files for Spike (`third_party/spike/`) and MPACT (`third_party/coralnpu_mpact/`); co-sim fails without both | Runs on npusim; instruction-count gain visible |
| 4. Kernel | Kernel path using `.insn` inline assembly, behind a compile flag (pattern in `tests/cocotb/vme_test/*.cc`) | Kernel tests bit-exact on npusim |
| 5. RTL | Chisel decode (`RvvDecode.scala`), SV decode and datapath, behind the new flag | Builds flag off and on, with and without `ZVT_ON` |
| 6. RTL tests | cocotb test under `tests/cocotb/rvv/`, existing RVV regressions, UVM co-sim against both ISS models | All pass; flag-off build unchanged |
| 7. Measure | Kernel and MobileNet cycles on Verilator; out-of-context synthesis on VU47P at 50 MHz (`fpga/ooc/run_ooc.sh`) | Numbers posted on the issue |

Working in parallel: ideas 1 and 2 touch the same files, so both encodings are settled in #7 first. Each idea keeps its ISS changes in its own patch file named after the idea (for example `0100-idea1-dot-product.patch`), not the next free number. All new `Parameters.scala` flags go in together in one early pull request.

## Evaluation

Configurations, each measured from a recorded commit with the same model, input and clock: **B** (baseline), **1**, **2**, **1+2**, **3**, **4**, **5**, and **All** (every idea that helped). For each: kernel cycles per layer type, MobileNet cycles, instructions, stalls, LUT/FF/DSP/BRAM/URAM, WNS, power, and whether it ran on F2.

## Build targets

| Purpose | Target |
| ------- | ------ |
| Main development (ideas 1 to 3) | `//hdl/chisel/src/coralnpu:rvv_core_mini_axi_cc_library`, `//tests/verilator_sim:rvv_core_mini_axi_sim` |
| Larger programs (MobileNet) | `rvv_core_mini_highmem_axi_cc_library`, `rvv_core_mini_highmem_axi_sim` |
| Co-simulation of new instructions | `rvv_core_mini_verification_axi_cc_library` |
| Idea 5 and combined builds | `vme_core_mini_axi_cc_library` (same RVV backend with `ZVT_ON`) |
| FPGA | `//hdl/chisel/src/soc:coralnpu_chisel_subsystem_cc_library` (and `_highmem`, `_matrix`), wrapped by `fpga/ip/coralnpu_chisel_subsystem_*` |

Changes to the RVV backend must build and pass tests both with and without `ZVT_ON`, because the matrix configuration widens shared structs.

## Machines

| Machine | Who | Use |
| ------- | --- | --- |
| Apple Silicon | `ounuvar`, `AayushMerchant` | Day-to-day: Chisel, Verilator, cocotb, npusim. Setup: `doc/macos_apple_silicon.md` and `utils/check_macos_setup.sh`. No Vivado, VCS or AWS HDK; linters run in CI if not installed. |
| x86 Linux | `Capheuss` | Upstream Linux setup as-is (prebuilt toolchain, or `utils/coralnpu.dockerfile`). Numbers for the report; long RTL runs. Under WSL2: clone in `~/` not `/mnt/c`, 16 GB+ memory in `.wslconfig`, `git config core.autocrlf false`. |
| UT ECE LRC servers | Everyone | Vivado 2023.2.2 for out-of-context VU47P area and timing (`fpga/ooc/README.md`). Not new enough for the F2 HDK. |
| AWS instance | Everyone | F2 builds (the HDK needs Vivado 2024.1 or newer) and F2 runs. |
| GitHub Actions | Everyone | On every pull request and push to `main`: lint and macro signatures (`ci.yml`) and the RTL Verilog build (`main.yml`) on Ubuntu x86, a few minutes. The full `bazel test //...` never runs here; it runs by hand on a team Linux machine. |

## Repo map (project-relevant parts)

- `hdl/chisel/src/coralnpu/`: the core in Chisel/Scala. `scalar/` holds the RV32IMF front end, `rvv/` holds the Chisel RVV wrapper and decode, and `Parameters.scala` holds the feature flags (`enableRvv`, `enableVme`, `rvvVlen = 128`).
- `hdl/verilog/rvv/design/`: the SystemVerilog RVV backend (decode, dispatch, ALU, MUL/MAC, permutation/reduction, LSU remap, ROB). `Zvt/` is the matrix engine. `hdl/verilog/rvv/` is excluded from Verible formatting.
- `hdl/chisel/src/soc/`: SoC subsystem configs (default, highmem, matrix, matrix_highmem) consumed by `fpga/ip/coralnpu_chisel_subsystem_*`.
- `sw/opt/litert-micro/`: hand-optimized TFLite Micro kernels (conv, depthwise_conv, fully_connected, pooling, logistic), the software baseline to beat. Tests in `sw/opt/litert-micro/test/`.
- `tests/cocotb/`: RTL simulation tests. `rvv/arithmetics/` holds per-instruction templates; `tutorial/tfmicro/` runs MobileNet layers on RTL.
- `tests/uvm/`: co-simulation against the MPACT and Spike ISS models. Both are required in presubmit.
- `third_party/spike/` and `third_party/coralnpu_mpact/`: ISS models patched for Coral.
- `fpga/ooc/`: our out-of-context synthesis and place-and-route flow for VU47P area, timing and power.
- `fpga/`: the upstream FPGA flow for Google's internal "Nexus" board (`xcvu13p`, FuseSoC + Vivado). **AWS F2 is not supported upstream**; porting is project work. Lab-only details in `fpga/README.md` (zturn, `nexus_loader`, `*.corp.google.com`) do not apply to us.
- `doc/`: architecture docs. `doc/tutorials/npusim_mobilenet_tutorial.md` covers the end-to-end MobileNet flow.

## Common commands

```bash
# Core smoke test (RTL sim)
bazel run //tests/cocotb:core_mini_axi_sim_cocotb

# Build the Verilator simulator for the RVV core
bazel build //tests/verilator_sim:rvv_core_mini_axi_sim

# Emit Verilog for the RVV core (what CI builds)
bazel build //hdl/chisel/src/coralnpu:rvv_core_mini_axi_cc_library_emit_verilog

# Optimized kernel tests on npusim (functional; instruction counts, not cycles)
bazel test //sw/opt/litert-micro/test:conv_sim_test_npusim

# MobileNet layers on RTL (cycles; long run, tagged manual)
bazel test //tests/cocotb/tutorial/tfmicro:cocotb_run_mobilenet_v1_verilator

# End-to-end MobileNet on npusim (instruction counts)
bazel run //tests/npusim_examples:npusim_run_mobilenet

# Find targets
bazel query 'kind(".*test", //tests/cocotb/...)'
```

Target binaries are built with the `coralnpu_v2_binary` rule (`rules/coralnpu_v2.bzl`). Use `dtcm_size_kbytes` / `itcm_size_kbytes` for highmem builds. The toolchain targets `rv32imf_zve32x_zicsr_zifencei_zbb`.

## Conventions

- **Debug waveforms with `fst_query`**, never by converting to VCD (see AGENTS.md).
- **Run `./utils/run_linters.sh`** before committing (or rely on CI). Use `BASE_BRANCH=HEAD~N` for a commit series.
