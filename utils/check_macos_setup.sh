#!/usr/bin/env bash
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
# Checks that an Apple Silicon Mac is set up to build and simulate CoralNPU
# from this clone (apple-silicon branch).
#
# Usage:
#   utils/check_macos_setup.sh          # fast checks (a few seconds)
#   utils/check_macos_setup.sh --build  # also build hello world with Bazel
#
# Exits 0 when there are no failures (warnings are allowed), 1 otherwise.
# Written for the macOS system bash (3.2).

set -u

RUN_BUILD=0
for arg in "$@"; do
  case "${arg}" in
    --build) RUN_BUILD=1 ;;
    -h | --help)
      sed -n '16,23p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg} (try --help)" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TC_CONFIG="${REPO_ROOT}/toolchain/cc_toolchain_config.bzl"

if [[ -t 1 ]]; then
  C_OK=$'\033[32m' C_WARN=$'\033[33m' C_FAIL=$'\033[31m' C_DIM=$'\033[2m' C_OFF=$'\033[0m'
else
  C_OK="" C_WARN="" C_FAIL="" C_DIM="" C_OFF=""
fi

FAILS=0
WARNS=0
ok() { printf '  %s[ OK ]%s %s\n' "${C_OK}" "${C_OFF}" "$1"; }
warn() {
  printf '  %s[WARN]%s %s\n' "${C_WARN}" "${C_OFF}" "$1"
  [[ $# -gt 1 ]] && printf '         %s%s%s\n' "${C_DIM}" "$2" "${C_OFF}"
  WARNS=$((WARNS + 1))
}
fail() {
  printf '  %s[FAIL]%s %s\n' "${C_FAIL}" "${C_OFF}" "$1"
  [[ $# -gt 1 ]] && printf '         %sfix: %s%s\n' "${C_DIM}" "$2" "${C_OFF}"
  FAILS=$((FAILS + 1))
}
section() { printf '\n%s\n' "$1"; }

# ---------------------------------------------------------------------------
section "Host"

if [[ "$(uname -s)" != "Darwin" ]]; then
  fail "Not macOS ($(uname -s)). This script is for Apple Silicon Macs." \
    "On Linux, use the regular setup (see utils/coralnpu.dockerfile)."
  echo
  exit 1
fi

# `uname -m` reports x86_64 when the shell itself runs under Rosetta.
if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]]; then
  fail "Not an Apple Silicon Mac. Only arm64 Macs are supported." \
    "Use an x86_64 Linux machine or the Docker image instead."
elif [[ "$(uname -m)" != "arm64" ]]; then
  fail "This shell runs under Rosetta (uname -m = $(uname -m))." \
    "Open a native arm64 terminal; Bazel must run as arm64 to pick the macOS toolchain."
else
  ok "Apple Silicon ($(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo arm64))"
fi

MACOS_VER="$(sw_vers -productVersion)"
MACOS_MAJOR="${MACOS_VER%%.*}"
if [[ "${MACOS_MAJOR}" -lt 11 ]]; then
  fail "macOS ${MACOS_VER} is too old" "Update to macOS 11 or newer."
elif [[ "${MACOS_MAJOR}" -lt 15 ]]; then
  warn "macOS ${MACOS_VER}: the riscv-gnu-toolchain Homebrew bottle targets macOS 15+" \
    "brew will compile the toolchain from source instead (can take an hour or more)."
else
  ok "macOS ${MACOS_VER}"
fi

MEM_GB=$(($(sysctl -n hw.memsize) / 1073741824))
if [[ "${MEM_GB}" -lt 16 ]]; then
  warn "${MEM_GB} GB RAM; the vector-core (RVV) model build may be slow or run out of memory"
else
  ok "${MEM_GB} GB RAM, $(sysctl -n hw.ncpu) CPU cores"
fi

FREE_GB=$(df -g "${HOME}" | awk 'NR==2 {print $4}')
if [[ "${FREE_GB}" -lt 20 ]]; then
  warn "${FREE_GB} GB free disk space; Bazel's cache plus the toolchain need ~20 GB"
else
  ok "${FREE_GB} GB free disk space"
fi

# ---------------------------------------------------------------------------
section "Apple tools"

if xcode-select -p >/dev/null 2>&1 && xcrun --find clang >/dev/null 2>&1; then
  ok "Xcode Command Line Tools ($(xcode-select -p))"
else
  fail "Xcode Command Line Tools not installed" "xcode-select --install"
fi

for tool in make ar; do
  if command -v "${tool}" >/dev/null 2>&1; then
    ok "${tool} ($(command -v "${tool}"))"
  else
    fail "${tool} not found" "xcode-select --install"
  fi
done

if arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
  ok "Rosetta 2 (needed for firtool, which has no arm64 build)"
else
  fail "Rosetta 2 not installed (firtool only ships an x86_64 macOS binary)" \
    "softwareupdate --install-rosetta --agree-to-license"
fi

# ---------------------------------------------------------------------------
section "Homebrew and Bazel"

if command -v brew >/dev/null 2>&1; then
  ok "Homebrew ($(brew --prefix))"
  HAVE_BREW=1
else
  fail "Homebrew not found" "Install from https://brew.sh"
  HAVE_BREW=0
fi

BAZEL=""
if command -v bazelisk >/dev/null 2>&1; then
  BAZEL="bazelisk"
  ok "bazelisk ($(command -v bazelisk))"
elif command -v bazel >/dev/null 2>&1; then
  BAZEL="bazel"
  if bazel --version 2>/dev/null | grep -qi bazelisk || [[ -L "$(command -v bazel)" ]]; then
    ok "bazel ($(command -v bazel))"
  else
    warn "Found 'bazel' but not bazelisk; it must be Bazel $(cat "${REPO_ROOT}/.bazelversion")" \
      "brew install bazelisk  (bazelisk reads .bazelversion automatically)"
  fi
else
  fail "bazelisk not found" "brew install bazelisk"
fi

# ---------------------------------------------------------------------------
section "RISC-V cross toolchain"

EXPECTED_GCC="$(sed -n 's/.*"gcc_version": attr.string(default = "\([0-9.]*\)").*/\1/p' "${TC_CONFIG}")"
MARCH="$(sed -n 's/.* else "\(rv32[a-z0-9_]*\)".*/\1/p' "${TC_CONFIG}" | head -1)"

TC_PREFIX=""
if [[ -n "${CORALNPU_RISCV_TOOLCHAIN:-}" ]]; then
  TC_PREFIX="${CORALNPU_RISCV_TOOLCHAIN}"
  TC_SOURCE="CORALNPU_RISCV_TOOLCHAIN"
elif [[ "${HAVE_BREW}" == 1 ]]; then
  TC_PREFIX="$(brew --prefix riscv-gnu-toolchain 2>/dev/null)"
  TC_SOURCE="Homebrew"
fi
GCC="${TC_PREFIX}/bin/riscv64-unknown-elf-gcc"

if [[ -n "${CORALNPU_RISCV_TOOLCHAIN:-}" && ! -x "${GCC}" ]]; then
  fail "CORALNPU_RISCV_TOOLCHAIN=${CORALNPU_RISCV_TOOLCHAIN} has no bin/riscv64-unknown-elf-gcc" \
    "Point it at the toolchain's install prefix, or unset it to use Homebrew's."
elif [[ -z "${TC_PREFIX}" || ! -x "${GCC}" ]]; then
  fail "riscv-gnu-toolchain not installed" \
    "brew tap riscv-software-src/riscv && brew trust --formula riscv-software-src/riscv/riscv-gnu-toolchain && brew install riscv-software-src/riscv/riscv-gnu-toolchain"
else
  ok "Toolchain from ${TC_SOURCE}: ${TC_PREFIX}"

  GOT_GCC="$("${GCC}" -dumpfullversion 2>/dev/null)"
  if [[ "${GOT_GCC}" == "${EXPECTED_GCC}" ]]; then
    ok "GCC ${GOT_GCC} (matches toolchain/cc_toolchain_config.bzl)"
  else
    fail "GCC ${GOT_GCC}, but toolchain/cc_toolchain_config.bzl expects ${EXPECTED_GCC}" \
      "Its include paths use that version. Install GCC ${EXPECTED_GCC}, or change gcc_version there."
  fi

  missing=""
  for f in \
    "riscv64-unknown-elf/lib/nano.specs" \
    "riscv64-unknown-elf/lib/libc_nano.a" \
    "riscv64-unknown-elf/include/stdio.h" \
    "riscv64-unknown-elf/include/c++/${EXPECTED_GCC}" \
    "lib/gcc/riscv64-unknown-elf/${EXPECTED_GCC}/include"; do
    [[ -e "${TC_PREFIX}/${f}" ]] || missing="${missing} ${f}"
  done
  if [[ -z "${missing}" ]]; then
    ok "newlib-nano, libstdc++ and GCC headers present"
  else
    fail "Toolchain is missing:${missing}" \
      "Use riscv-software-src/riscv/riscv-gnu-toolchain (Homebrew core's riscv64-elf-gcc has no libc)."
  fi

  if [[ -e "${TC_PREFIX}/riscv64-unknown-elf/lib/htif_nano.specs" ]]; then
    ok "libgloss-htif present (semihosting targets supported)"
  else
    warn "No htif_nano.specs: semihosting targets won't link (only tests/npusim_examples and" \
      "tests/verilator_sim:htif_semihosting_test need it)."
  fi

  TMP="$(mktemp -d)"
  trap 'rm -rf "${TMP}"' EXIT
  cat >"${TMP}/t.cc" <<'EOF'
#include <riscv_vector.h>
#include <vector>
int main() {
  std::vector<int> v{1, 2, 3};
  return static_cast<int>(__riscv_vsetvl_e32m1(4)) + v[1];
}
EOF
  if "${TC_PREFIX}/bin/riscv64-unknown-elf-g++" -march="${MARCH}" -mabi=ilp32 \
    -mcmodel=medany -fno-rtti -fno-exceptions -O2 --specs=nano.specs \
    -nostartfiles -Wl,--defsym=_start=0 "${TMP}/t.cc" -o "${TMP}/t.elf" \
    >"${TMP}/cc.log" 2>&1; then
    ok "Cross-compiles C++ with RVV intrinsics for -march=${MARCH}"
  else
    fail "Test compile with CoralNPU's -march failed:" "$(head -3 "${TMP}/cc.log" | tr '\n' ' ')"
  fi
fi

# ---------------------------------------------------------------------------
section "This clone"

BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null)"
if [[ ! -f "${REPO_ROOT}/rules/coralnpu_toolchain.bzl" ]]; then
  fail "Missing the Apple Silicon support (on branch '${BRANCH}')" "git switch apple-silicon"
elif [[ "${BRANCH}" != "apple-silicon" ]]; then
  warn "On branch '${BRANCH}', not apple-silicon (Apple Silicon support is present, so this may be fine)"
else
  ok "On branch apple-silicon"
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain -- MODULE.bazel.lock 2>/dev/null)" ]]; then
  warn "MODULE.bazel.lock has local changes" \
    "Usually harmless (Bazel refreshes it); don't commit an entry for a platform you didn't mean to."
fi

# ---------------------------------------------------------------------------
if [[ "${RUN_BUILD}" == 1 ]]; then
  section "Bazel build (--build)"
  if [[ "${FAILS}" -gt 0 || -z "${BAZEL}" ]]; then
    fail "Skipping the build until the failures above are fixed"
  else
    echo "  Building //examples:coralnpu_v2_hello_world_add_floats (first run downloads"
    echo "  Bazel and dependencies; can take several minutes)..."
    LOG="$(mktemp)"
    if (cd "${REPO_ROOT}" && "${BAZEL}" build //examples:coralnpu_v2_hello_world_add_floats) >"${LOG}" 2>&1; then
      ok "Bazel built a CoralNPU RISC-V binary"
      rm -f "${LOG}"
    else
      fail "Bazel build failed; last lines of ${LOG}:" "$(grep -E 'ERROR|error' "${LOG}" | tail -3 | tr '\n' ' ')"
    fi
  fi
fi

# ---------------------------------------------------------------------------
echo
if [[ "${FAILS}" -gt 0 ]]; then
  printf '%s%d failure(s), %d warning(s).%s Fix the failures above and re-run.\n' \
    "${C_FAIL}" "${FAILS}" "${WARNS}" "${C_OFF}"
  exit 1
fi
printf '%sAll required checks passed%s (%d warning(s)).\n' "${C_OK}" "${C_OFF}" "${WARNS}"
if [[ "${RUN_BUILD}" == 0 ]]; then
  echo "Next: utils/check_macos_setup.sh --build, then"
  echo "      bazelisk test //tests/cocotb:core_mini_axi_sim_cocotb_verilator"
fi
exit 0
