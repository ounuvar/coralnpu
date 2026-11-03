#!/bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
# The script lives at <runfiles>/_main/third_party/llvm-firtool/firtool.sh
# and the @llvm_firtool repo is at <runfiles>/llvm_firtool/.
# Walk three levels up to land at runfiles root, then find it.
ROOTDIR=${SCRIPT_DIR}/../../../

case "$(uname -s)" in
  Darwin) FIRTOOL_PLATFORM=macos-x64 ;;
  *) FIRTOOL_PLATFORM=linux-x64 ;;
esac

FIRTOOL_EXE=$(find "${ROOTDIR}" -name 'firtool' -ipath "*${FIRTOOL_PLATFORM}*" -print -quit)

"${FIRTOOL_EXE}" "$@"

exit $?
