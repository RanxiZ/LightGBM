/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_FUSED_SPLIT_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_FUSED_SPLIT_H_

#include <stdint.h>

// Shared host/INLINE AICPU ABI. Only POD metadata crosses this interface; all
// histogram, scan, gain and winner-statistic arithmetic stays on the NPU.
struct LGBMNPUFusedSplitFeature {
  int32_t feature;
  int32_t input_index;
  int32_t histogram_index;
  int32_t num_bin;
  int32_t offset;
  int32_t default_bin;
  int32_t missing_type;
  int32_t enabled;
  int32_t randomized;
  int32_t random_threshold;
  double penalty;
};

struct LGBMNPUFusedSplitParameters {
  int64_t histogram_plane;
  int32_t histogram_stride;
  int32_t num_features;
  int32_t num_data;
  int32_t min_data_in_leaf;
  double sum_gradient;
  double sum_hessian;
  double lambda_l1;
  double lambda_l2;
  double min_sum_hessian_in_leaf;
  double min_gain_to_split;
  double max_delta_step;
  double path_smooth;
  double parent_output;
};

struct LGBMNPUFusedSplitWinner {
  int32_t feature;
  uint32_t threshold;
  int32_t default_left;
  int32_t left_count;
  int32_t right_count;
  // Completion is checked because Bisheng's void launch stub only logs
  // synchronous launch errors. A zero-initialized output must fail closed.
  int32_t completed;
  double gain;
  double left_output;
  double right_output;
  double left_gradient;
  double right_gradient;
  double left_hessian;
  double right_hessian;
};

// All pointers except stream address device memory. Features are stable-sorted
// by original feature index; flags are returned in input_index order. The
// caller synchronizes the stream before downloading winner and flags.
// Compile npu_fused_split.aicpu with -ffp-contract=off and without fast-math.
#ifndef LGBM_AICPU_DEVICE_SOURCE
extern "C" void LGBMNPUFusedSplitKernel(
    uint32_t block_dim, void* launch_config, void* stream,
    const double* histogram,
    const LGBMNPUFusedSplitFeature* features,
    const LGBMNPUFusedSplitParameters* parameters,
    LGBMNPUFusedSplitWinner* winner, uint8_t* splittable);

static_assert(sizeof(LGBMNPUFusedSplitFeature) == 48, "fused split feature ABI");
static_assert(sizeof(LGBMNPUFusedSplitParameters) == 96, "fused split parameter ABI");
static_assert(sizeof(LGBMNPUFusedSplitWinner) == 80, "fused split winner ABI");

#endif  // LGBM_AICPU_DEVICE_SOURCE

#endif  // LIGHTGBM_SRC_TREELEARNER_NPU_FUSED_SPLIT_H_