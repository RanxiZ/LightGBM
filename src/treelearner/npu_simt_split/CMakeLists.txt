# Copyright (c) 2026. Licensed under the MIT License.
#
# Build unit for the split finder operator (migration guide section 3, file 6).
#
#   lgbm_npu_simt_split       device library: bisheng compiles the AscendC /
#                             SIMT translation unit, `ascendc_library` extracts
#                             the host-side launcher stub and generates
#                             `aclrtlaunch_lgbm_npu_simt_split_kernel.h` and
#                             `aclrtlaunch_lgbm_npu_simt_split_reduce_kernel.h`.
#   lgbm_npu_simt_split_operator  host library: SimtSplitOperator (ACL calls only).
#
# Nothing is hand-written on the launch side: the headers below are produced by
# the toolkit from the kernel signatures.

set(SPLIT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

ascendc_library(lgbm_npu_simt_split STATIC
  "${SPLIT_ROOT}/npu_simt_split_kernel.cpp")
target_include_directories(lgbm_npu_simt_split PRIVATE "${SPLIT_ROOT}")

# Precision contract P1 (migration guide section 5): no FP contraction, no
# fast-math.  bisheng forwards the `OPTIONS` property of `<target>_interface`
# to the device compile line, so this is where the flag has to live.
if(TARGET lgbm_npu_simt_split_interface)
  set_property(TARGET lgbm_npu_simt_split_interface APPEND PROPERTY OPTIONS
               "-ffp-contract=off")
endif()

add_library(lgbm_npu_simt_split_operator STATIC
  "${SPLIT_ROOT}/npu_simt_split_op.cpp")
target_include_directories(lgbm_npu_simt_split_operator PUBLIC
  "${SPLIT_ROOT}"
  "${NPU_INCLUDE_DIR}")
target_link_libraries(lgbm_npu_simt_split_operator PUBLIC
  lgbm_npu_simt_split
  ${NPU_ASCENDCL_LIBRARY})

set_target_properties(lgbm_npu_simt_split_operator PROPERTIES
  BUILD_WITH_INSTALL_RPATH FALSE
  INSTALL_RPATH "${NPU_RPATHS}"
  BUILD_RPATH "${NPU_RPATHS}")

install(TARGETS lgbm_npu_simt_split lgbm_npu_simt_split_operator
        ARCHIVE DESTINATION lib)
install(FILES "${SPLIT_ROOT}/npu_simt_split_abi.h"
              "${SPLIT_ROOT}/npu_simt_split_op.h"
        DESTINATION include/treelearner)
