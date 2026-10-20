/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_FUSED_HISTOGRAM_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_FUSED_HISTOGRAM_H_

#include <stdint.h>

// Host/device ABI for fused histogram construction. All data pointers refer to
// device memory. Each feature record describes one selected source column and
// its destination plane/row in the output histogram.
struct LGBMNPUFusedHistogramFeature {
  int32_t output_index;
  int32_t tile_index;
  int32_t tile_feature;
  int32_t num_bins;
  int32_t offset;
  int32_t reserved;
};

// bisheng -x aicpu emits a device kernel and this host launch wrapper.
#ifndef LGBM_AICPU_DEVICE_SOURCE
extern "C" void LGBMNPUFusedHistogramKernel(
    uint32_t block_dim, void* launch_config, void* stream,
    const int32_t* const* histogram_tiles,
    const int32_t* device_row_indices, const float* gradients,
    const float* hessians, const LGBMNPUFusedHistogramFeature* features,
    double* output, int32_t num_features, int32_t output_features,
    int64_t count, int32_t histogram_rows, int32_t histogram_stride,
    int32_t tile_width, int32_t* completed);

static_assert(sizeof(LGBMNPUFusedHistogramFeature) == 24,
              "fused histogram feature ABI");
#endif  // LGBM_AICPU_DEVICE_SOURCE

#endif  // LIGHTGBM_SRC_TREELEARNER_NPU_FUSED_HISTOGRAM_H_