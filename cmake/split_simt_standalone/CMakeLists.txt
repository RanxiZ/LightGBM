# Copyright (c) 2026. Licensed under the MIT License.
#
# Nested (standalone) build of the AI Core SIMT best-split finder that is
# integrated into the NPU tree learner.
#
# Why a nested project instead of add_subdirectory() from the LightGBM top
# level:
#   * the histogram SIMT kernel is built with the new ASC front end
#     (`--enable-simt --npu-arch=dav-3510 --asc-aicore-lang`);
#   * the split finder must be built with the legacy `ascendc.cmake` front end
#     (`--cce-aicore-lang --cce-aicore-arch=dav-c310-vec`, SOC_VERSION
#     `ascend950pr_9579`) because only that combination defines `__simt_vf__` /
#     `asc_vf_call` and emits the `aclrtlaunch_*.h` headers on this toolchain.
# Both front ends install global compile options and `include()` global modules,
# so they cannot share one CMake project without contaminating each other. The
# LightGBM top level drives this project through ExternalProject, which gives it
# an independent cache and flag set.
#
# Inputs supplied by the parent:
#   -DNPU_TOOLKIT_ROOT=<CANN toolkit>
#   -DLGBM_SIMT_SPLIT_SOURCE_DIR=<LightGBM>/src/treelearner/npu_simt_split
#   [-DSOC_VERSION=ascend950pr_9579]
cmake_minimum_required(VERSION 3.20)
project(LGBMNPUSimtSplitStandalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT DEFINED NPU_TOOLKIT_ROOT)
  if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(NPU_TOOLKIT_ROOT "$ENV{ASCEND_TOOLKIT_HOME}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(NPU_TOOLKIT_ROOT "$ENV{ASCEND_HOME_PATH}")
  else()
    message(FATAL_ERROR "NPU_TOOLKIT_ROOT is not set.")
  endif()
endif()

if(NOT DEFINED SOC_VERSION)
  set(SOC_VERSION "ascend950pr_9579" CACHE STRING "system on chip type")
endif()

set(ASCENDC_CMAKE_DIR
    "${NPU_TOOLKIT_ROOT}/x86_64-linux/tikcpp/ascendc_kernel_cmake")
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}/ascendc.cmake")
  set(ASCENDC_CMAKE_DIR
      "${NPU_TOOLKIT_ROOT}/compiler/tikcpp/ascendc_kernel_cmake")
endif()
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}/ascendc.cmake")
  message(FATAL_ERROR "ascendc.cmake not found under ${NPU_TOOLKIT_ROOT}")
endif()
# `ascendc.cmake` validates ASCEND_CANN_PACKAGE_PATH before using it, and the
# parent project (new ASC front end) never defines it.
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH "${NPU_TOOLKIT_ROOT}" CACHE PATH
      "CANN package root")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

find_path(
  NPU_INCLUDE_DIR
  NAMES acl/acl.h
  HINTS "${NPU_TOOLKIT_ROOT}/include"
        "${NPU_TOOLKIT_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
  NO_DEFAULT_PATH)
find_library(
  NPU_ASCENDCL_LIBRARY
  NAMES ascendcl
  HINTS "${NPU_TOOLKIT_ROOT}/lib64"
        "${NPU_TOOLKIT_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
  NO_DEFAULT_PATH)
if(NOT NPU_INCLUDE_DIR OR NOT NPU_ASCENDCL_LIBRARY)
  message(FATAL_ERROR "ACL not found under '${NPU_TOOLKIT_ROOT}'.")
endif()

set(NPU_RPATHS
  "${NPU_TOOLKIT_ROOT}/lib64"
  "${NPU_TOOLKIT_ROOT}/runtime/lib64"
  "${NPU_TOOLKIT_ROOT}/lib64/device/lib64")

if(NOT DEFINED LGBM_SIMT_SPLIT_SOURCE_DIR)
  message(FATAL_ERROR "LGBM_SIMT_SPLIT_SOURCE_DIR is not set.")
endif()

add_subdirectory("${LGBM_SIMT_SPLIT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/npu_simt_split")

# One predictable directory so the parent can import both archives by path.
set_target_properties(lgbm_npu_simt_split_operator PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set_target_properties(lgbm_npu_simt_split PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
