/* Copyright (c) 2026. Licensed under the MIT License. */
//
// Cross-interface ABI for the Ascend SIMT best-split finder.
//
// This header is compiled twice: once by the host compiler (g++) for the
// operator runtime / probe, and once by bisheng for the device kernel.  It must
// therefore stay pure POD: no AscendC headers, no simt_api headers, no
// std::vector, no pointers into either address space.
//
// Semantics are ported from LightGBM's CUDA split finder
// (src/treelearner/cuda/cuda_best_split_finder.{hpp,cpp,cu}) under precision
// contract P1: every accumulation, gain and output is FP32.
//
// Layout rules (migration guide section 5):
//   * fixed-width types + static_assert on every struct size;
//   * tasks are supplied already stably sorted by original feature index, and
//     `input_index` preserves the caller's ordering;
//   * equal-gain ties resolve to the first occurrence in task order;
//   * histogram base pointers, per-task bin offset and bin stride are all
//     explicit -- no implicit row-major assumption;
//   * the merge stage reads `SimtSplitCandidate` only, never recomputes.

#ifndef NPU_SIMT_SPLIT_ABI_H
#define NPU_SIMT_SPLIT_ABI_H

#include <cstdint>

#include <cstddef>

namespace lgbm_npu_simt {

// Mirrors LightGBM's MissingType.
enum SimtMissingType : int32_t {
  kMissingNone = 0,
  kMissingZero = 1,
  kMissingNaN = 2,
};

// Mirrors `kEpsilon` in LightGBM's leaf splits.  Under contract P1 (FP32) this
// addition is a no-op for any realistic hessian, exactly as it is in the CUDA
// `hist_t == float` path; it is kept in the expressions so the port stays
// textually comparable with the reference.
constexpr float kEpsilon = 1e-15f;

// `-inf` without requiring an IEEE inf literal in the device compiler.
// Only finite gains are ever compared against it (a candidate is marked
// `found` explicitly), so this is equivalent to LightGBM's `kMinScore`.
constexpr float kMinScore = -3.4028234663852886e+38F;

// One pass through one feature's histogram in one direction.
// A task is one warp in the scan VF.
struct SimtSplitTask {
  // Original feature index; copied into the result. (CUDA `inner_feature_index`)
  int32_t inner_feature_index;
  // Caller order, preserved verbatim through the merge.
  int32_t input_index;
  // Offset of bin 0 of this feature inside the histogram planes, in bins.
  // Signed 64-bit so large datasets cannot wrap.
  int64_t hist_offset;
  // Stride between consecutive bins of this feature, in bins (usually 1).
  int64_t bin_stride;
  // Number of bins of this feature, including the missing/default bin.
  uint32_t num_bin;
  // Bin that holds the default (most frequent) value.
  uint32_t default_bin;
  // 1 when the most frequent bin is bin 0 (`GetMostFreqBin() == 0`).
  uint32_t mfb_offset;
  // Direction of the pass: 0 = forward (CUDA `reverse == false`).
  uint32_t reverse;
  // Skip `default_bin` while accumulating.
  uint32_t skip_default_bin;
  // Treat missing values as a separate (first) bin.
  uint32_t na_as_missing;
  // Default direction reported for the produced split.
  uint32_t assume_out_default_left;
  // Set when the VF must not handle this task (categorical features are a
  // host-side path in v1); the candidate is left invalid.
  uint32_t is_categorical;
  uint32_t reserved;
  // `extra_trees`: the only threshold value allowed to win, or -1 if unused.
  int32_t rand_threshold;
};

// Leaf-level statistics and regularisation parameters.  Identical for every
// task of one `Run` call.
struct SimtSplitParams {
  int32_t num_tasks;
  // Threads per AI core used by the scan dispatch (informational).
  int32_t threads_per_block;
  // `num_data` of the leaf.
  int32_t num_data;
  // Feature dimensionality, informational (the scan never indexes by it).
  int32_t num_features;
  float sum_gradients;
  float sum_hessians;
  float parent_gain;
  float min_sum_hessian_in_leaf;
  float min_gain_to_split;
  float lambda_l1;
  float lambda_l2;
  float path_smooth;
  float parent_output;
  int32_t min_data_in_leaf;
  // Compile-time specialisation selectors, dispatched outside `asc_vf_call`.
  int32_t use_l1;
  int32_t use_smoothing;
  int32_t use_rand;
};

// Per-task result of the scan phase.  Written by exactly one lane of the
// owning warp; read (not recomputed) by the merge phase.
struct SimtSplitCandidate {
  uint32_t is_valid;
  uint32_t threshold;
  uint32_t inner_feature_index;
  uint32_t assume_out_default_left;
  uint32_t task_index;
  int32_t input_index;
  // `current_gain - min_gain_shift`, i.e. the gain above "do not split".
  float gain;
  float left_sum_gradients;
  float left_sum_hessians;
  float right_sum_gradients;
  float right_sum_hessians;
  float left_value;
  float left_gain;
  float right_value;
  float right_gain;
  int32_t left_count;
  int32_t right_count;
};

// Leaf-level winner: the best candidate in task order.
struct SimtSplitWinner {
  uint32_t is_valid;
  uint32_t threshold;
  uint32_t inner_feature_index;
  uint32_t assume_out_default_left;
  uint32_t task_index;
  uint32_t num_valid_candidates;
  int32_t input_index;
  float gain;
  float left_sum_gradients;
  float left_sum_hessians;
  float right_sum_gradients;
  float right_sum_hessians;
  float left_value;
  float left_gain;
  float right_value;
  float right_gain;
  int32_t left_count;
  int32_t right_count;
};

// Sizes are part of the contract.  Any change here must be mirrored in the
// device translation unit, which is exactly why they are asserted.
static_assert(sizeof(SimtSplitTask) == 64, "SimtSplitTask layout changed");
static_assert(offsetof(SimtSplitTask, hist_offset) == 8,
              "SimtSplitTask.hist_offset moved");
static_assert(sizeof(SimtSplitParams) == 68, "SimtSplitParams layout changed");
static_assert(sizeof(SimtSplitCandidate) == 68,
              "SimtSplitCandidate layout changed");
static_assert(sizeof(SimtSplitWinner) == 72, "SimtSplitWinner layout changed");

// Error codes returned by SimtSplitOperator::Run / Init.
enum SimtSplitStatus : int32_t {
  kSimtSplitOk = 0,
  kSimtSplitInvalidArg = 1,
  kSimtSplitAclError = 2,
  kSimtSplitNotInitialized = 3,
};

}  // namespace lgbm_npu_simt

#endif  // NPU_SIMT_SPLIT_ABI_H
