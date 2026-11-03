# Building on a RHEL 8 host without root

CoralNPU's official Linux environment is the Ubuntu image in
[coralnpu.dockerfile](../utils/coralnpu.dockerfile). This page covers a shared
RHEL 8-family x86_64 machine (for example a university server) where you have no
root, no Docker, and the system toolchain is too old: no clang, no `lz4-devel`,
and a gcc 8 `libstdc++`. Everything is installed under one private directory.

## Setup

1. **Prerequisites.** `gcc-toolset-14` (or another recent toolset, set
   `CORALNPU_GCC_TOOLSET`) must be installed under `/opt/rh`. You also need
   `curl`, `make` and `python3`.

2. **Install the toolchain.**

   ```sh
   utils/setup_rh8_userspace.sh
   ```

   This installs, under `~/.coralnpu-env` (override with `CORALNPU_ENV_ROOT`):

   - bazelisk, linked as `~/.local/bin/bazel`, so Bazel follows `.bazelversion`
   - clang 19 from conda-forge (via micromamba), wrapped so it targets the system
     headers and `gcc-toolset`'s libstdc++ and links the C++ runtime statically
   - lz4 1.9.4, built from source (Verilator needs `lz4.h`)

   Downloads are pinned and checksum-verified. The lz4 digest is trust-on-first-use
   because lz4 publishes none.

3. **Put it on your shell.** Add `source ~/.coralnpu-env/env.sh` to `~/.bashrc`.
   It sets `BAZELISK_HOME` (keeps Bazel downloads out of your home directory) and
   puts `~/.local/bin` first on `PATH`.

4. **Write the Bazel config.**

   ```sh
   utils/setup_rh8_userspace.sh --write-bazelrc
   ```

   This writes `.bazelrc.user` (gitignored, imported by `.bazelrc`) with a private
   cache directory, the clang `PATH`, and the lz4 paths. It will not overwrite an
   existing file; it writes `.bazelrc.user.generated` instead.

5. **Verify.**

   ```sh
   utils/setup_rh8_userspace.sh --check
   bazel build //tests/verilator_sim:rvv_core_mini_highmem_axi_sim
   bazel test //sw/opt/litert-micro/test:conv_sim_test_npusim
   bazel test //tests/cocotb:core_mini_axi_sim_cocotb_verilator
   ```

## Things to know

- **Keep the cache on a filesystem you trust.** Bazel warns that an NFS output
  base "may lead to surprising failures". It worked in our runs.
- **Bazel 9 needs the repository contents cache outside the repo.** The generated
  config handles this.
- **If you change the clang wrapper**, bump `CLANG_WRAPPER_REV` in
  `.bazelrc.user` and run
  `bazel fetch --force --repo=@@rules_cc++cc_configure_extension+local_config_cc`.
  Bazel does not hash the wrapper, and caches the toolchain's list of built-in
  include directories.
- **Do not update clang in the micromamba environment.** The wrapper replaces
  the `clang-19` binary in place and an update would overwrite it. Re-run the
  script to repair it.
- **Deleting a Bazel cache needs write permission first.** Bazel makes its
  outputs read-only, so `rm -rf` fails partway. Run `bazel clean --expunge` from
  the clone, or `chmod -R u+w <cache dir>` before deleting.
- **VS Code may show thousands of "changes".** Each Bazel output base contains
  `execroot/_main`, a symlink copy of the workspace with a `.git` link, and VS
  Code's git extension opens it as a repository the moment a build creates it.
  The generated config moves the `bazel-*` convenience symlinks out of the
  workspace (`--symlink_prefix`). Also add your cache directory (default
  `~/.coralnpu-env/cache`) to `files.watcherExclude` in your VS Code settings,
  and list any `_main` that is already open in `git.ignoredRepositories`. The
  cache is not tracked by git.
- **Why the wrapper replaces the binary.** Bazel's auto-detected toolchain
  resolves the compiler to its real path, so a wrapper on `PATH` can be bypassed.
- **Why `host_clang` trusts `/`.** See the comment in `toolchain/host_clang/BUILD`.
  Without it Bazel rejects a compiler installed outside `/usr` and `/opt`.
