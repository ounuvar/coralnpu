#!/usr/bin/env bash
# shellcheck disable=SC2311,SC2312
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Sets up a no-root build environment for CoralNPU on a RHEL 8-family x86_64 host that has no
# system clang, no lz4-devel and an old libstdc++ (for example a shared university machine).
# Everything is installed under one private directory; nothing outside it is modified except one
# symlink in BIN_DIR (bazel -> bazelisk). See doc/linux_rh8_userspace.md.
#
# Usage:
#   utils/setup_rh8_userspace.sh                  # install everything (idempotent)
#   utils/setup_rh8_userspace.sh --check          # verify an existing install
#   utils/setup_rh8_userspace.sh --write-bazelrc  # install, then write <repo>/.bazelrc.user
#
# Environment overrides:
#   CORALNPU_ENV_ROOT    install root, mode 700 (default: ~/.coralnpu-env)
#   CORALNPU_BIN_DIR     where the bazel symlink goes (default: ~/.local/bin)
#   CORALNPU_GCC_TOOLSET gcc-toolset to use for C++ headers (default: /opt/rh/gcc-toolset-14)

set -euo pipefail

ROOT="${CORALNPU_ENV_ROOT:-${HOME}/.coralnpu-env}"
BIN_DIR="${CORALNPU_BIN_DIR:-${HOME}/.local/bin}"
GCC_TOOLSET="${CORALNPU_GCC_TOOLSET:-/opt/rh/gcc-toolset-14}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pinned downloads. The bazelisk and micromamba digests are the ones the projects publish.
# The lz4 digest was computed on first download (lz4 publishes none): trust-on-first-use.
BAZELISK_VERSION="v1.30.0"
BAZELISK_SHA256="a248ae2b2b62b5baebb748f6fba7203bff1a27e5d01b55399a8406744cb539e8"
MAMBA_VERSION="2.10.0-0"
MAMBA_SHA256="701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd"
LZ4_VERSION="1.9.4"
LZ4_SHA256="0b0e3aa07c8c063ddf40b082bdf7e37a1562bda40a0ff5272957f3e987e0e54b"
CLANG_VERSION="19.1.7"

MODE="install"
WRITE_BAZELRC=0
for arg in "$@"; do
  case "${arg}" in
    --check) MODE="check" ;;
    --write-bazelrc) WRITE_BAZELRC=1 ;;
    -h | --help)
      sed -n '16,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg} (try --help)" >&2
      exit 2
      ;;
  esac
done

log() { echo "==> $*"; }
die() {
  echo "ERROR: $*" >&2
  exit 1
}

# fetch URL DEST SHA256: download once, verify, and keep.
fetch() {
  local url="$1" dest="$2" sha="$3"
  if [[ -f "${dest}" ]] && [[ "$(sha256sum "${dest}" | cut -d' ' -f1)" == "${sha}" ]]; then
    return 0
  fi
  curl -fsSL -m 600 -o "${dest}.part" "${url}"
  if [[ "$(sha256sum "${dest}.part" | cut -d' ' -f1)" != "${sha}" ]]; then
    die "checksum mismatch for ${url}"
  fi
  mv -f "${dest}.part" "${dest}"
}

check_prereqs() {
  [[ "$(uname -m)" == "x86_64" ]] || die "x86_64 Linux only"
  for t in curl sha256sum tar make gcc python3; do
    command -v "${t}" > /dev/null || die "missing required tool: ${t}"
  done
  local gcc_dir
  gcc_dir="$(gcc_install_dir)"
  [[ -d "${gcc_dir}" ]] || die "gcc-toolset not found at ${GCC_TOOLSET} (set CORALNPU_GCC_TOOLSET)"
}

# gcc's install dir inside the toolset, e.g. .../lib/gcc/x86_64-redhat-linux/14
gcc_install_dir() {
  local d
  for d in "${GCC_TOOLSET}"/root/usr/lib/gcc/x86_64-redhat-linux/*; do
    if [[ -d "${d}" ]]; then
      echo "${d}"
      return 0
    fi
  done
  echo "${GCC_TOOLSET}/root/usr/lib/gcc/x86_64-redhat-linux/NOT_FOUND"
}

install_bazelisk() {
  log "bazelisk ${BAZELISK_VERSION}"
  mkdir -p "${ROOT}/dl" "${ROOT}/bazelisk" "${BIN_DIR}"
  fetch "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64" \
    "${ROOT}/dl/bazelisk" "${BAZELISK_SHA256}"
  install -m 755 "${ROOT}/dl/bazelisk" "${ROOT}/bin/bazelisk"
  # Keep any pre-existing plain bazel binary under another name; never delete it.
  if [[ -e "${BIN_DIR}/bazel" && ! -L "${BIN_DIR}/bazel" ]]; then
    mv "${BIN_DIR}/bazel" "${BIN_DIR}/bazel.pre-bazelisk"
  fi
  ln -sfn "${ROOT}/bin/bazelisk" "${BIN_DIR}/bazel"
}

install_micromamba() {
  log "micromamba ${MAMBA_VERSION}"
  fetch "https://github.com/mamba-org/micromamba-releases/releases/download/${MAMBA_VERSION}/micromamba-linux-64" \
    "${ROOT}/dl/micromamba" "${MAMBA_SHA256}"
  install -m 700 "${ROOT}/dl/micromamba" "${ROOT}/bin/micromamba"
}

install_clang() {
  if [[ -x "${ROOT}/clang19/bin/clang-19.real" ]]; then
    log "clang ${CLANG_VERSION} already installed"
    return 0
  fi
  log "clang ${CLANG_VERSION} (conda-forge)"
  MAMBA_ROOT_PREFIX="${ROOT}/mamba-root" "${ROOT}/bin/micromamba" create -y -p "${ROOT}/clang19" \
    --override-channels -c conda-forge "clang=${CLANG_VERSION}" "clangxx=${CLANG_VERSION}"
}

# The wrapper replaces the clang-19 binary in place (the real one becomes clang-19.real), so the
# flags apply however a build system resolves the compiler (clang, clang++, clang-19, a realpath).
install_wrapper() {
  local bin="${ROOT}/clang19/bin" gcc_dir
  gcc_dir="$(gcc_install_dir)"
  if [[ ! -e "${bin}/clang-19.real" ]]; then
    mv "${bin}/clang-19" "${bin}/clang-19.real"
  fi
  log "clang wrapper (system target, ${gcc_dir##*/lib/gcc/}, static libstdc++)"
  cat > "${bin}/clang-19.tmp" << 'EOF'
#!/bin/bash
# Wrapper installed IN PLACE of the conda-forge clang binary (the real one is clang-19.real).
# Targets this host's system headers + gcc-toolset's libstdc++, linked statically (the host
# libstdc++.so.6 is too old for the symbols the toolset's headers need). Also rewrites the
# toolchain's explicit -lstdc++/-latomic, including inside @params files (how Bazel passes link
# args). Rewritten params files go to a content-addressed cache in /tmp (never deleted here).
# Also puts the private lz4 (installed below) on the include and link paths. The FuseSoC Verilator
# build (//fpga:build_chip_verilator*) runs with a clean environment, so the C_INCLUDE_PATH and
# LIBRARY_PATH in .bazelrc.user never reach it; the compiler it uses is this wrapper.
real="$(dirname "$(readlink -f "$0")")/clang-19.real"
# Clang re-execs its own "executable" (this script) with -cc1 first for subprocess frontend
# jobs; pass those straight to the real binary, untouched.
case "${1:-}" in -cc1 | -cc1as | -cc1gen-reproducer) exec -a "$0" "${real}" "$@" ;; esac
extra=(--target=x86_64-redhat-linux-gnu --gcc-install-dir=@GCC_DIR@ -isystem @LZ4_DIR@/include)
args=()
link=1
cache="/tmp/clangwrap-$(id -un)"
mkdir -p "${cache}"
for a in "$@"; do
  case "${a}" in
    -c | -E | -S | -M | -MM | -fsyntax-only) link=0 ;;
    -lstdc++) a=-l:libstdc++.a ;;
    -latomic) a=-l:libatomic.a ;;
    @*)
      f="${a#@}"
      if [ -f "${f}" ]; then
        h="$(md5sum < "${f}" | cut -d' ' -f1)"
        t="${cache}/${h}"
        if [ ! -f "${t}" ]; then
          sed -e 's#^-lstdc++$#-l:libstdc++.a#' -e 's#^-latomic$#-l:libatomic.a#' < "${f}" > "${t}.$$" \
            && mv -f "${t}.$$" "${t}"
        fi
        a="@${t}"
      fi
      ;;
  esac
  args+=("${a}")
done
if [ "${link}" = 1 ]; then
  extra+=(-static-libstdc++ -static-libgcc -L@GCC_DIR@ -L@LZ4_DIR@/lib -Wl,-rpath,@LZ4_DIR@/lib)
fi
exec -a "$0" "${real}" "${extra[@]}" "${args[@]}"
EOF
  sed -i -e "s#@GCC_DIR@#${gcc_dir}#g" -e "s#@LZ4_DIR@#${ROOT}/local#g" "${bin}/clang-19.tmp"
  chmod 755 "${bin}/clang-19.tmp"
  mv -f "${bin}/clang-19.tmp" "${bin}/clang-19"
}

install_lz4() {
  if [[ -f "${ROOT}/local/include/lz4.h" ]]; then
    log "lz4 ${LZ4_VERSION} already installed"
    return 0
  fi
  log "lz4 ${LZ4_VERSION} (built from source; the host has no lz4-devel)"
  fetch "https://github.com/lz4/lz4/releases/download/v${LZ4_VERSION}/lz4-${LZ4_VERSION}.tar.gz" \
    "${ROOT}/dl/lz4.tar.gz" "${LZ4_SHA256}"
  mkdir -p "${ROOT}/src"
  tar -xzf "${ROOT}/dl/lz4.tar.gz" -C "${ROOT}/src"
  make -C "${ROOT}/src/lz4-${LZ4_VERSION}" -j"$(nproc)" PREFIX="${ROOT}/local" install > /dev/null
}

write_env_sh() {
  cat > "${ROOT}/env.sh" << EOF
# Source from ~/.bashrc:  source ${ROOT}/env.sh
export BAZELISK_HOME="${ROOT}/bazelisk"
case ":\${PATH}:" in *":${BIN_DIR}:"*) ;; *) export PATH="${BIN_DIR}:\${PATH}" ;; esac
EOF
}

# The per-user Bazel config. .bazelrc.user is gitignored and imported by the repo's .bazelrc.
bazelrc_contents() {
  cat << EOF
# --- Per-user Bazel config (gitignored); generated by utils/setup_rh8_userspace.sh ---
# One private cache dir, outside the repo. Bazel 9 requires the repo contents cache outside it.
startup --output_user_root=${ROOT}/cache
common --repo_contents_cache=${ROOT}/cache/repo-contents

# clang (conda-forge, wrapped) first on PATH for repo rules and sandboxed actions.
build --repo_env=PATH=${ROOT}/clang19/bin:/usr/local/bin:/usr/bin:/bin
build --action_env=PATH=${ROOT}/clang19/bin:/usr/local/bin:/usr/bin:/bin

# lz4 (Verilator needs lz4.h). Env vars, not -I: Bazel rejects -I paths outside the workspace.
build --action_env=C_INCLUDE_PATH=${ROOT}/local/include
build --action_env=CPLUS_INCLUDE_PATH=${ROOT}/local/include
build --action_env=LIBRARY_PATH=${ROOT}/local/lib
build --action_env=LD_LIBRARY_PATH=${ROOT}/local/lib
build --repo_env=C_INCLUDE_PATH=${ROOT}/local/include
build --repo_env=CPLUS_INCLUDE_PATH=${ROOT}/local/include
build --repo_env=LIBRARY_PATH=${ROOT}/local/lib
build --linkopt=-Wl,-rpath,${ROOT}/local/lib
test --test_env=LD_LIBRARY_PATH=${ROOT}/local/lib

# Keep Bazel's convenience symlinks out of the workspace: bazel-<name> points at execroot/_main, which
# contains a .git link, so editors (VS Code) detect it as a repository with thousands of "changes".
build --symlink_prefix=${ROOT}/links/$(basename "${REPO_DIR}")-

# Bump this to force a rebuild after the clang wrapper changes (Bazel does not hash it). Also run:
#   bazel fetch --force --repo=@@rules_cc++cc_configure_extension+local_config_cc
build --action_env=CLANG_WRAPPER_REV=2
EOF
}

write_bazelrc() {
  local dest="${REPO_DIR}/.bazelrc.user"
  mkdir -p "${ROOT}/links"
  if [[ -e "${dest}" ]]; then
    bazelrc_contents > "${dest}.generated"
    log "${dest} exists; wrote ${dest}.generated instead (compare and merge by hand)"
  else
    bazelrc_contents > "${dest}"
    log "wrote ${dest}"
  fi
}

check() {
  local fail=0 gcc_dir
  gcc_dir="$(gcc_install_dir)"
  for f in "${ROOT}/bin/bazelisk" "${ROOT}/bin/micromamba" "${ROOT}/clang19/bin/clang-19.real" \
    "${ROOT}/local/include/lz4.h" "${gcc_dir}/libstdc++.a" "${gcc_dir}/libatomic.a"; do
    if [[ -e "${f}" ]]; then echo "ok      ${f}"; else
      echo "MISSING ${f}"
      fail=1
    fi
  done
  mkdir -p "${ROOT}/tmp"
  printf '#include <optional>\n#include <iostream>\nint main(){std::optional<int> v=1;std::cout<<*v<<"\\n";}\n' \
    > "${ROOT}/tmp/check.cc"
  if "${ROOT}/clang19/bin/clang++" -std=c++20 "${ROOT}/tmp/check.cc" -o "${ROOT}/tmp/check" \
    && [[ "$("${ROOT}/tmp/check")" == "1" ]]; then
    echo "ok      clang++ -std=c++20 compiles and runs"
  else
    echo "FAILED  clang++ test compile"
    fail=1
  fi
  if ldd "${ROOT}/tmp/check" 2> /dev/null | grep -q libstdc; then
    echo "FAILED  binary links libstdc++ dynamically"
    fail=1
  else
    echo "ok      libstdc++ linked statically"
  fi
  return "${fail}"
}

if [[ "${MODE}" == "check" ]]; then
  check
  exit $?
fi

check_prereqs
mkdir -p "${ROOT}" "${ROOT}/bin"
chmod 700 "${ROOT}"
install_bazelisk
install_micromamba
install_clang
install_wrapper
install_lz4
write_env_sh
if [[ "${WRITE_BAZELRC}" == 1 ]]; then
  write_bazelrc
fi
check
log "done. Add to ~/.bashrc:  source ${ROOT}/env.sh"
log "then write the Bazel config with:  utils/setup_rh8_userspace.sh --write-bazelrc"
