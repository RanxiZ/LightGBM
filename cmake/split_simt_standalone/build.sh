#!/usr/bin/env bash
# Copyright (c) 2026. Licensed under the MIT License.
#
# Configure + build the nested legacy-toolchain project that produces the AI
# Core SIMT best-split finder.
#
#   build.sh <standalone_src_dir> <build_dir> <lightgbm_src_dir> <toolkit_root>
#
# Invoked by the LightGBM top-level CMake through ExternalProject. The build is
# separate because `ascendc.cmake` installs global compile options that are
# incompatible with the new ASC front end used for the histogram kernel.
#
# Two quirks of the legacy device build make the plain `cmake --build` call
# insufficient (both observed in the standalone split package):
#
#   1. `merge_obj_text.sh` relinks the device object in place; running it again
#      on the same tree fails with `ld.lld: ... unknown file type`. Removing the
#      preprocess scratch project forces a clean regeneration.
#   2. That regeneration has a step ordering that is not safe under `make -j`
#      (host-stub generation can start before the preprocess listing exists).
#      A serial run always succeeds.
#
# The device step is therefore skipped when the sources are byte-identical and
# both archives are already present, so an incremental LightGBM rebuild does not
# pay for it.
set -euo pipefail

SRC_DIR="$1"
BUILD_DIR="$2"
REPO_ROOT="$3"
TOOLKIT_ROOT="$4"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${TOOLKIT_ROOT}}"
export ASCEND_HOME="${ASCEND_HOME:-${ASCEND_HOME_PATH}}"
export CMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}/lib64/cmake:${CMAKE_PREFIX_PATH:-}"
export PATH="${ASCEND_HOME_PATH}/bin:${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin:${PATH}"

# The pip build backend (scikit-build-core) exports CMAKE_GENERATOR=Ninja and
# CMAKE_MAKE_PROGRAM into the environment. The legacy ascendc project drives
# `make` from its preprocess step and reads the generated Makefiles, so it
# fails under Ninja (`KeyError: .../preprocess_obj.dir/...o`). Force Makefiles
# for this nested build regardless of the outer generator.
unset CMAKE_GENERATOR CMAKE_MAKE_PROGRAM

SPLIT_INCLUDE="${REPO_ROOT}/src/treelearner"
SPLIT_SOURCE="${SPLIT_INCLUDE}/npu_simt_split"
DEVICE_LIB="${BUILD_DIR}/lib/liblgbm_npu_simt_split.a"
OPERATOR_LIB="${BUILD_DIR}/lib/liblgbm_npu_simt_split_operator.a"

FINGERPRINT="$(
  {
    cat "${SRC_DIR}/CMakeLists.txt"
    cat "${SPLIT_INCLUDE}/npu_simt_split_abi.h"
    cat "${SPLIT_INCLUDE}/npu_simt_split_vf.h"
    cat "${SPLIT_INCLUDE}/npu_simt_split_kernel.cpp"
    cat "${SPLIT_INCLUDE}/npu_simt_split_op.h"
    cat "${SPLIT_INCLUDE}/npu_simt_split_op.cpp"
    cat "${SPLIT_SOURCE}/CMakeLists.txt"
    cat "${BASH_SOURCE[0]}"
  } | sha256sum | cut -d' ' -f1
)"
STAMP="${BUILD_DIR}/.lgbm_split_sources.sha256"
if [[ -f "${STAMP}" && "$(cat "${STAMP}")" == "${FINGERPRINT}" &&
      -f "${DEVICE_LIB}" && -f "${OPERATOR_LIB}" ]]; then
  echo "[split_simt_standalone] sources unchanged; reusing ${BUILD_DIR}"
  exit 0
fi

rm -rf "${BUILD_DIR}/npu_simt_split/lgbm_npu_simt_split_preprocess-prefix"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNPU_TOOLKIT_ROOT="${ASCEND_HOME_PATH}" \
  -DSOC_VERSION=ascend950pr_9579 \
  -DLGBM_SIMT_SPLIT_SOURCE_DIR="${SPLIT_SOURCE}"

cmake --build "${BUILD_DIR}" -j"$(nproc)" || {
  echo "[split_simt_standalone] parallel build failed; retrying serially" >&2
  rm -rf "${BUILD_DIR}/npu_simt_split/lgbm_npu_simt_split_preprocess-prefix"
  cmake --build "${BUILD_DIR}" -j1
}
echo "${FINGERPRINT}" > "${STAMP}"
