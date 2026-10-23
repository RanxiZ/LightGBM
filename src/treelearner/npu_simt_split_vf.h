/* Copyright (c) 2026. Licensed under the MIT License. */
//
// SIMT virtual functions for the best-split finder.
//
// Every symbol here is device code.  The header is included by
// npu_simt_split_kernel.cpp, which owns the `__global__` entry points and the
// `asc_vf_call` dispatch.
//
// Ported from
//   * LightGBM/src/treelearner/cuda/cuda_best_split_finder.cu
//       FindBestSplitsForLeafKernelInner   (scan, lines 137-330)
//       ReduceBestGain*                    (merge, lines 20-77)
//   * LightGBM/src/treelearner/cuda/cuda_leaf_splits.hpp
//       ThresholdL1 / CalculateSplittedLeafOutput / GetLeafGainGivenOutput
//       / GetLeafGain / GetSplitGains
//
// Mapping (migration guide section 2):
//   * one task  == one warp;
//   * one lane  == one candidate threshold, 32 bins per chunk, chunks are
//     walked in order carrying the running prefix sums;
//   * `__shfl_up_sync`      -> asc_shfl_up (Hillis-Steele inclusive prefix);
//   * `__shfl_down_sync` + "lowest thread wins" -> asc_shfl_xor over the total
//     order (gain descending, position ascending).  A butterfly is used instead
//     of a down-tree because only lane 0 of a down-tree holds the reduction,
//     while every lane must end with the same ansim;
//   * the merge (`MergeBestVF`) runs the same total order with the task index
//     as the position key, so it keeps the earliest task on equal gains,
//     exactly like CUDA's `ReduceBestGain`.
//
// Precision contract P1: all arithmetic is FP32.  `double` is deliberately
// absent (the SIMT warp primitives have no double overload and the AIV has no
// native FP64).

#ifndef NPU_SIMT_SPLIT_VF_H
#define NPU_SIMT_SPLIT_VF_H

#include "npu_simt_split_abi.h"

#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"

namespace lgbm_npu_simt {

constexpr int32_t kWarpSize = 32;

// ---------------------------------------------------------------------------
// Leaf gain formulas (cuda_leaf_splits.hpp:66-160), FP32.
// ---------------------------------------------------------------------------

__SIMT_DEVICE_FUNCTIONS_DECL__ inline float ThresholdL1(float s, float l1) {
  const float reg_s = fmaxf(0.0f, fabsf(s) - l1);
  return (s >= 0.0f) ? reg_s : -reg_s;
}

template <bool USE_L1, bool USE_SMOOTHING>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline float CalculateSplittedLeafOutput(float sum_gradients,
                                                    float sum_hessians,
                                                    float l1, float l2,
                                                    float path_smooth,
                                                    int32_t num_data,
                                                    float parent_output) {
  float ret;
  if (USE_L1) {
    ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
  } else {
    ret = -sum_gradients / (sum_hessians + l2);
  }
  if (USE_SMOOTHING) {
    const float ratio = static_cast<float>(num_data) / path_smooth;
    ret = ret * ratio / (ratio + 1.0f) + parent_output / (ratio + 1.0f);
  }
  return ret;
}

template <bool USE_L1>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline float GetLeafGainGivenOutput(float sum_gradients,
                                               float sum_hessians, float l1,
                                               float l2, float output) {
  if (USE_L1) {
    const float sg_l1 = ThresholdL1(sum_gradients, l1);
    return -(2.0f * sg_l1 * output + (sum_hessians + l2) * output * output);
  }
  return -(2.0f * sum_gradients * output +
           (sum_hessians + l2) * output * output);
}

template <bool USE_L1, bool USE_SMOOTHING>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline float GetLeafGain(float sum_gradients, float sum_hessians,
                                    float l1, float l2, float path_smooth,
                                    int32_t num_data, float parent_output) {
  if (!USE_SMOOTHING) {
    if (USE_L1) {
      const float sg_l1 = ThresholdL1(sum_gradients, l1);
      return (sg_l1 * sg_l1) / (sum_hessians + l2);
    }
    return (sum_gradients * sum_gradients) / (sum_hessians + l2);
  }
  const float output = CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(
      sum_gradients, sum_hessians, l1, l2, path_smooth, num_data,
      parent_output);
  return GetLeafGainGivenOutput<USE_L1>(sum_gradients, sum_hessians, l1, l2,
                                        output);
}

template <bool USE_L1, bool USE_SMOOTHING>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline float GetSplitGains(float sum_left_gradients,
                                      float sum_left_hessians,
                                      float sum_right_gradients,
                                      float sum_right_hessians, float l1,
                                      float l2, float path_smooth,
                                      int32_t left_count, int32_t right_count,
                                      float parent_output) {
  return GetLeafGain<USE_L1, USE_SMOOTHING>(
             sum_left_gradients, sum_left_hessians, l1, l2, path_smooth,
             left_count, parent_output) +
         GetLeafGain<USE_L1, USE_SMOOTHING>(
             sum_right_gradients, sum_right_hessians, l1, l2, path_smooth,
             right_count, parent_output);
}

// ---------------------------------------------------------------------------
// Warp primitives.
// ---------------------------------------------------------------------------

// Inclusive prefix sum over the 32 lanes, Hillis-Steele doubling.  Matches
// `ShufflePrefixSum` in cuda_algorithms.hpp.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline float WarpPrefixSum(float v) {
  const uint32_t lane = threadIdx.x % static_cast<uint32_t>(kWarpSize);
  for (uint32_t offset = 1u; offset < static_cast<uint32_t>(kWarpSize);
       offset <<= 1) {
    const float other = asc_shfl_up(v, offset, kWarpSize);
    if (lane >= offset) {
      v += other;
    }
  }
  return v;
}

// `(found, gain, position)` reduced over the warp with the total order
//   better  <=>  other.found && (!mine.found || other.gain > mine.gain ||
//                               (other.gain == mine.gain && other.pos < mine.pos))
// Every lane leaves with the same result.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void WarpArgMaxGain(int32_t& found, float& gain,
                                      int32_t& pos) {
  for (uint32_t offset = static_cast<uint32_t>(kWarpSize) / 2u; offset > 0u;
       offset >>= 1) {
    const int32_t other_found =
        asc_shfl_xor(found, static_cast<int32_t>(offset), kWarpSize);
    const float other_gain =
        asc_shfl_xor(gain, static_cast<int32_t>(offset), kWarpSize);
    const int32_t other_pos =
        asc_shfl_xor(pos, static_cast<int32_t>(offset), kWarpSize);
    const bool take = (other_found != 0) &&
                      ((found == 0) || (other_gain > gain) ||
                       (other_gain == gain && other_pos < pos));
    if (take) {
      found = other_found;
      gain = other_gain;
      pos = other_pos;
    }
  }
}

// Same total order, used by the merge stage (order key = task index).
//
// NOTE: an earlier merge VF was sized for the single warp it uses (32
// threads).  On this toolchain (bisheng 15.0.5 / CANN 9.1.0) a module holding
// both a 256-thread scan VF and that 32-thread merge VF makes the scan kernel
// trap in the *merge* VF's entry thunk (aivec error 259, "scalar instruction
// abnormal"): the differently-sized SIMT entry destabilises the module's
// entry layout.  `MergeBestVF` therefore uses the scan's 256-thread dispatch
// and lets the idle warps exit.  See trace.md, deviation D3.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void WarpArgMaxGainIndex(int32_t& found, float& gain,
                                           int32_t& index) {
  for (uint32_t offset = static_cast<uint32_t>(kWarpSize) / 2u; offset > 0u;
       offset >>= 1) {
    const int32_t other_found =
        asc_shfl_xor(found, static_cast<int32_t>(offset), kWarpSize);
    const float other_gain =
        asc_shfl_xor(gain, static_cast<int32_t>(offset), kWarpSize);
    const int32_t other_index =
        asc_shfl_xor(index, static_cast<int32_t>(offset), kWarpSize);
    const bool take = (other_found != 0) &&
                      ((found == 0) || (other_gain > gain) ||
                       (other_gain == gain && other_index < index));
    if (take) {
      found = other_found;
      gain = other_gain;
      index = other_index;
    }
  }
}

// LightGBM's count rounding, `Common::RoundInt` (common.h:911).  The CUDA
// kernel uses `__double2int_rn`; under P1 the argument is FP32 and the two
// rules only diverge on exact .5 ties, which this port resolves the way
// LightGBM's own CPU code does.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int32_t RoundCount(float x) {
  return static_cast<int32_t>(x + 0.5f);
}

// ---------------------------------------------------------------------------
// Scan: one warp, one task->
// ---------------------------------------------------------------------------
//
// Position model, mirroring `FindBestSplitsForLeafKernelInner` exactly:
//
//   t                : position along the sweep, 0-based, one per lane per chunk
//   REVERSE          : t walks bins from the top down, histogram index
//                      nbmf - 1 - t; skip test and threshold use the bin label
//                      num_bin - 1 - t / num_bin - 2 - t
//   FORWARD normal   : t walks bins from the bottom up, histogram index t;
//                      skip test and threshold use t + mfb_offset
//   FORWARD special  : na_as_missing && mfb_offset == 1; lane 0 carries
//                      `sum_gradients - sum(non-default bins)` and the missing
//                      hessian epsilon, lane t >= 1 reads bin t - 1, threshold
//                      is t
//
// `nbmf = num_bin - mfb_offset` is CUDA's `feature_num_bin_minus_offset`.

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int32_t BinIndexForPosition(
    bool reverse, bool special, int32_t num_bin, int32_t mfb, int32_t nbmf,
    int32_t na, int32_t t) {
  if (reverse) {
    if (t < na || t >= nbmf) {
      return -1;
    }
    return nbmf - 1 - t;
  }
  if (special) {
    if (t <= 0 || t >= num_bin) {
      return -1;
    }
    return t - 1;
  }
  if (t < 0 || t >= nbmf) {
    return -1;
  }
  return t;
}

// CUDA's `skip_sum` predicate, which gates both the read and the candidate.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool SkipPosition(
    uint32_t skip_default_bin, uint32_t default_bin, bool reverse,
    int32_t num_bin, int32_t mfb, int32_t t) {
  if (skip_default_bin == 0u) {
    return false;
  }
  const int32_t label = reverse ? (num_bin - 1 - t) : (t + mfb);
  return label == static_cast<int32_t>(default_bin);
}

template <bool REVERSE, bool USE_L1, bool USE_SMOOTHING>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void ScanTaskVF(
    __gm__ float* histogram_gradients, __gm__ float* histogram_hessians,
    __gm__ const SimtSplitTask* tasks, __gm__ const SimtSplitParams* params,
    __gm__ SimtSplitCandidate* candidates) {
  const uint32_t lane = threadIdx.x % static_cast<uint32_t>(kWarpSize);
  const uint32_t warps_per_block =
      blockDim.x / static_cast<uint32_t>(kWarpSize);
  const int32_t warp_index = static_cast<int32_t>(
      blockIdx.x * warps_per_block + threadIdx.x / static_cast<uint32_t>(kWarpSize));
  // One warp per task, grid-strided.  The host caps the grid at the vector
  // core count, so the warps in flight can be fewer than `num_tasks`; without
  // the stride the tasks past `gridDim.x * warps_per_block` would never be
  // scanned and their candidate slots would keep stale values from an earlier
  // launch.
  const int32_t total_warps =
      static_cast<int32_t>(gridDim.x * warps_per_block);
  const int32_t num_tasks = params->num_tasks;

  // Leaf-level values are uniform across the whole launch; read them once per
  // warp instead of once per task.
  const float sum_gradients = params->sum_gradients;
  const float sum_hessians = params->sum_hessians;
  const int32_t num_data = params->num_data;
  const float cnt_factor = static_cast<float>(num_data) / sum_hessians;
  const float min_gain_shift = params->parent_gain + params->min_gain_to_split;
  const float min_hess = params->min_sum_hessian_in_leaf;
  const int32_t min_data = params->min_data_in_leaf;
  const float l1 = params->lambda_l1;
  const float l2 = params->lambda_l2;
  const float path_smooth = params->path_smooth;
  const float parent_output = params->parent_output;
  const int32_t use_rand = params->use_rand;

  for (int32_t task_index = warp_index; task_index < num_tasks;
       task_index += total_warps) {
    __gm__ SimtSplitCandidate* out = candidates + task_index;
    const __gm__ SimtSplitTask* task = tasks + task_index;

    // Direction is a compile-time constant here: the host issues one VF call per
    // direction so no warp ever contains both branches.
    if (task->reverse != (REVERSE ? 1u : 0u)) {
      continue;
    }

    // Categorical features are a host-side path in v1; leave an explicit invalid
    // candidate behind instead of silently skipping the slot.
    if (task->is_categorical != 0u) {
      if (lane == 0u) {
        out->is_valid = 0u;
        out->threshold = 0u;
        out->inner_feature_index =
            static_cast<uint32_t>(task->inner_feature_index);
        out->assume_out_default_left = task->assume_out_default_left;
        out->task_index = static_cast<uint32_t>(task_index);
        out->input_index = task->input_index;
        out->gain = 0.0f;
        out->left_sum_gradients = 0.0f;
        out->left_sum_hessians = 0.0f;
        out->right_sum_gradients = 0.0f;
        out->right_sum_hessians = 0.0f;
        out->left_value = 0.0f;
        out->left_gain = 0.0f;
        out->right_value = 0.0f;
        out->right_gain = 0.0f;
        out->left_count = 0;
        out->right_count = 0;
      }
      continue;
    }

    const int32_t num_bin = static_cast<int32_t>(task->num_bin);
    const int32_t mfb = static_cast<int32_t>(task->mfb_offset);
    const int32_t nbmf = num_bin - mfb;
    const int32_t na = static_cast<int32_t>(task->na_as_missing);
    const uint32_t skip_bin = task->skip_default_bin;
    const uint32_t default_bin = task->default_bin;
    const int32_t rand_threshold = task->rand_threshold;
    const int64_t bin_stride = task->bin_stride;
    const bool special = (!REVERSE) && (task->na_as_missing != 0u) && (mfb == 1);
    // Positions walked by the sweep.  The CUDA block covers `num_bin` positions in
    // the special path: position 0 carries the omitted most-frequent bin and
    // positions 1..num_bin-1 read the num_bin-1 stored entries.  Stopping at
    // `num_bin - 1` would leave the last stored bin out of the pre-pass total and
    // therefore inflate the residual at position 0.  Only positions
    // <= num_bin - 2 can produce a candidate, so the extra position is shift-only.
    const int32_t num_pos = special ? num_bin : nbmf;
    const __gm__ float* grad = histogram_gradients + task->hist_offset;
    const __gm__ float* hess = histogram_hessians + task->hist_offset;

    // --- special forward path: total of the non-default bins, needed before the
    // prefix sweep because position 0 carries `sum_* - sum_non_default` on both
    // planes (the omitted most-frequent bin's mass belongs to the left side).
    float total_grad = 0.0f;
    float total_hess = 0.0f;
    if (special) {
      float acc_g = 0.0f;
      float acc_h = 0.0f;
      for (int32_t base = 0; base < num_pos; base += kWarpSize) {
        const int32_t t = base + static_cast<int32_t>(lane);
        const int32_t bin =
            BinIndexForPosition(false, true, num_bin, mfb, nbmf, na, t);
        if (bin >= 0) {
          const int64_t at = static_cast<int64_t>(bin) * bin_stride;
          acc_g += grad[at];
          acc_h += hess[at];
        }
      }
      // butterfly sum: every lane ends with the block-wide total
      for (uint32_t offset = static_cast<uint32_t>(kWarpSize) / 2u; offset > 0u;
           offset >>= 1) {
        acc_g += asc_shfl_xor(acc_g, static_cast<int32_t>(offset), kWarpSize);
        acc_h += asc_shfl_xor(acc_h, static_cast<int32_t>(offset), kWarpSize);
      }
      total_grad = acc_g;
      total_hess = acc_h;
    }

    // --- prefix sweep, 32 positions per chunk, carry across chunks
    float carry_g = 0.0f;
    float carry_h = 0.0f;

    int32_t best_found = 0;
    float best_gain = kMinScore;
    int32_t best_pos = 0;
    uint32_t best_threshold = 0u;
    float best_left_g = 0.0f;
    float best_left_h = 0.0f;
    float best_right_g = 0.0f;
    float best_right_h = 0.0f;
    int32_t best_left_c = 0;
    int32_t best_right_c = 0;

    for (int32_t base = 0; base < num_pos; base += kWarpSize) {
      const int32_t t = base + static_cast<int32_t>(lane);
      const bool skip = SkipPosition(skip_bin, default_bin, REVERSE, num_bin, mfb, t);

      float g = 0.0f;
      float h = 0.0f;
      if (t < num_pos && !skip) {
        const int32_t bin = BinIndexForPosition(REVERSE, special, num_bin, mfb, nbmf, na, t);
        if (bin >= 0) {
          const int64_t at = static_cast<int64_t>(bin) * bin_stride;
          g = grad[at];
          h = hess[at];
        }
      }
      if (special && t == 0) {
        // Position 0 does not read a bin; it carries the omitted-bin residual of
        // both planes (CUDA: `local_* += sum_* - sum_non_default`).
        g = sum_gradients - total_grad;
        h = sum_hessians - total_hess;
      }
      // CUDA adds kEpsilon to lane 0's hessian before the prefix sum, so every
      // later prefix carries it too.
      if (t == 0) {
        h += kEpsilon;
      }

      float pg = WarpPrefixSum(g);
      float ph = WarpPrefixSum(h);
      const float sum_g = carry_g + pg;
      const float sum_h = carry_h + ph;

      // candidate
      int32_t in_range = 0;
      uint32_t threshold = 0u;
      float left_g = 0.0f;
      float left_h = 0.0f;
      float right_g = 0.0f;
      float right_h = 0.0f;
      int32_t left_c = 0;
      int32_t right_c = 0;
      int32_t rand_label = 0;

      if (!skip && t < num_pos) {
        if (REVERSE) {
          if (t >= na && t <= num_bin - 2) {
            in_range = 1;
            threshold = static_cast<uint32_t>(num_bin - 2 - t);
            rand_label = num_bin - 2 - t;
            right_g = sum_g;
            right_h = sum_h;
            left_g = sum_gradients - right_g;
            left_h = sum_hessians - right_h;
            right_c = RoundCount(right_h * cnt_factor);
            left_c = num_data - right_c;
          }
        } else {
          const int32_t end = special ? (num_bin - 2) : (nbmf - 2);
          if (t <= end) {
            in_range = 1;
            threshold = special ? static_cast<uint32_t>(t)
                                : static_cast<uint32_t>(t + mfb);
            rand_label = t + mfb;
            left_g = sum_g;
            left_h = sum_h;
            right_g = sum_gradients - left_g;
            right_h = sum_hessians - left_h;
            left_c = RoundCount(left_h * cnt_factor);
            right_c = num_data - left_c;
          }
        }
      }

      int32_t lane_found = 0;
      float lane_gain = kMinScore;
      int32_t lane_pos = t;
      if (in_range != 0 && left_h >= min_hess && right_h >= min_hess &&
          left_c >= min_data && right_c >= min_data &&
          (use_rand == 0 || rand_label == rand_threshold)) {
        const float gain = GetSplitGains<USE_L1, USE_SMOOTHING>(
            left_g, left_h, right_g, right_h, l1, l2, path_smooth, left_c,
            right_c, parent_output);
        // gain with the split is worse than without it
        if (gain > min_gain_shift) {
          lane_found = 1;
          lane_gain = gain - min_gain_shift;
          // final statistics, with the epsilon taken back out exactly as the
          // reference does (a no-op under FP32, kept for fidelity)
          float fg = 0.0f;
          float fh = 0.0f;
          if (REVERSE) {
            right_h = sum_h - kEpsilon;
            right_g = sum_g;
            right_c = RoundCount(right_h * cnt_factor);
            left_h = sum_hessians - right_h - kEpsilon;
            left_g = sum_gradients - right_g;
            left_c = num_data - right_c;
          } else {
            left_h = sum_h - kEpsilon;
            left_g = sum_g;
            left_c = RoundCount(left_h * cnt_factor);
            right_h = sum_hessians - left_h - kEpsilon;
            right_g = sum_gradients - left_g;
            right_c = num_data - left_c;
          }
          (void)fg;
          (void)fh;
        }
      }

      // chunk-wide winner under the total order, then broadcast its statistics
      WarpArgMaxGain(lane_found, lane_gain, lane_pos);
      int32_t owner = lane_found != 0 ? (lane_pos - base) : 0;
      owner &= (kWarpSize - 1);
      const int32_t c_threshold =
          static_cast<int32_t>(asc_shfl(static_cast<int32_t>(threshold), owner,
                                        kWarpSize));
      const float c_left_g = asc_shfl(left_g, owner, kWarpSize);
      const float c_left_h = asc_shfl(left_h, owner, kWarpSize);
      const float c_right_g = asc_shfl(right_g, owner, kWarpSize);
      const float c_right_h = asc_shfl(right_h, owner, kWarpSize);
      const int32_t c_left_c = asc_shfl(left_c, owner, kWarpSize);
      const int32_t c_right_c = asc_shfl(right_c, owner, kWarpSize);

      const bool better = (lane_found != 0) &&
                          ((best_found == 0) || (lane_gain > best_gain) ||
                           (lane_gain == best_gain && lane_pos < best_pos));
      if (better) {
        best_found = 1;
        best_gain = lane_gain;
        best_pos = lane_pos;
        best_threshold = static_cast<uint32_t>(c_threshold);
        best_left_g = c_left_g;
        best_left_h = c_left_h;
        best_right_g = c_right_g;
        best_right_h = c_right_h;
        best_left_c = c_left_c;
        best_right_c = c_right_c;
      }

      carry_g += asc_shfl(pg, kWarpSize - 1, kWarpSize);
      carry_h += asc_shfl(ph, kWarpSize - 1, kWarpSize);
    }

    // Every lane holds the same (best_found, best_gain, best_pos, stats); only
    // lane 0 publishes this task's candidate.
    if (lane != 0u) {
      continue;
    }

    out->is_valid = best_found != 0 ? 1u : 0u;
    out->threshold = best_threshold;
    out->inner_feature_index = static_cast<uint32_t>(task->inner_feature_index);
    out->assume_out_default_left = task->assume_out_default_left;
    out->task_index = static_cast<uint32_t>(task_index);
    out->input_index = task->input_index;
    out->gain = best_found != 0 ? best_gain : 0.0f;
    out->left_sum_gradients = best_left_g;
    out->left_sum_hessians = best_left_h;
    out->right_sum_gradients = best_right_g;
    out->right_sum_hessians = best_right_h;
    out->left_count = best_left_c;
    out->right_count = best_right_c;
    if (best_found != 0) {
      const float left_value =
          CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(
              best_left_g, best_left_h, l1, l2, path_smooth, best_left_c,
              parent_output);
      const float right_value =
          CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(
              best_right_g, best_right_h, l1, l2, path_smooth, best_right_c,
              parent_output);
      out->left_value = left_value;
      out->left_gain = GetLeafGainGivenOutput<USE_L1>(
          best_left_g, best_left_h, l1, l2, left_value);
      out->right_value = right_value;
      out->right_gain = GetLeafGainGivenOutput<USE_L1>(
          best_right_g, best_right_h, l1, l2, right_value);
    } else {
      out->left_value = 0.0f;
      out->left_gain = 0.0f;
      out->right_value = 0.0f;
      out->right_gain = 0.0f;
    }
  }
}

// ---------------------------------------------------------------------------
// Merge: deterministic leaf-wide winner over the candidate array.
// ---------------------------------------------------------------------------
//
// Warp 0 strides the candidate array, reduces it under the total order
// (gain descending, task index ascending) and lane 0 publishes the winner by
// *copying* the winning candidate -- the merge never recomputes a split, so it
// cannot introduce rounding of its own.  An equal-gain tie therefore keeps the
// earliest task, which is exactly CUDA's `ReduceBestGain` rule over the task
// order.
//
// The dispatch uses the scan's block shape (256 threads, `LAUNCH_BOUND(256)`)
// rather than the warp it needs.  Two toolchain constraints force that, both
// recorded in trace.md deviation D3:
//   * the entry must contain a SIMT call, otherwise kernel-type inference
//     classifies the reduce entry AIC-only and the vector-core image launch
//     silently runs nothing;
//   * a 32-thread VF in the same module destabilised the SIMT entry layout
//     and made the scan kernel trap (aivec error 259).
// The 224 threads outside warp 0 exit immediately.
template <bool kWarpOnly = true>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void MergeBestVF(
    __gm__ const SimtSplitParams* params,
    __gm__ const SimtSplitCandidate* candidates,
    __gm__ SimtSplitWinner* winner) {
  if (threadIdx.x >= static_cast<uint32_t>(kWarpSize)) {
    return;
  }
  const uint32_t lane = threadIdx.x;
  const int32_t num_tasks = params->num_tasks;

  int32_t found = 0;
  float gain = kMinScore;
  int32_t index = 0;
  int32_t num_valid = 0;
  for (int32_t i = static_cast<int32_t>(lane); i < num_tasks;
       i += kWarpSize) {
    const __gm__ SimtSplitCandidate* c = candidates + i;
    if (c->is_valid == 0u) {
      continue;
    }
    ++num_valid;
    if (found == 0 || c->gain > gain || (c->gain == gain && i < index)) {
      found = 1;
      gain = c->gain;
      index = i;
    }
  }
  // Both reductions are butterflies, so every lane leaves with the same
  // value; the tie rule above already picked the lowest task index per lane
  // and `WarpArgMaxGainIndex` keeps it across lanes.
  WarpArgMaxGainIndex(found, gain, index);
  for (uint32_t offset = static_cast<uint32_t>(kWarpSize) / 2u; offset > 0u;
       offset >>= 1) {
    num_valid += asc_shfl_xor(num_valid, static_cast<int32_t>(offset), kWarpSize);
  }
  if (lane != 0u) {
    return;
  }

  // Every field is written, including the all-zero invalid case, so the host
  // never observes stale values from a previous launch (buffer reuse).
  winner->num_valid_candidates = static_cast<uint32_t>(num_valid);
  if (found == 0) {
    winner->is_valid = 0u;
    winner->threshold = 0u;
    winner->inner_feature_index = 0u;
    winner->assume_out_default_left = 0u;
    winner->task_index = 0u;
    winner->input_index = 0;
    winner->gain = 0.0f;
    winner->left_sum_gradients = 0.0f;
    winner->left_sum_hessians = 0.0f;
    winner->right_sum_gradients = 0.0f;
    winner->right_sum_hessians = 0.0f;
    winner->left_value = 0.0f;
    winner->left_gain = 0.0f;
    winner->right_value = 0.0f;
    winner->right_gain = 0.0f;
    winner->left_count = 0;
    winner->right_count = 0;
    return;
  }
  const __gm__ SimtSplitCandidate* c = candidates + index;
  winner->is_valid = 1u;
  winner->threshold = c->threshold;
  winner->inner_feature_index = c->inner_feature_index;
  winner->assume_out_default_left = c->assume_out_default_left;
  winner->task_index = static_cast<uint32_t>(index);
  winner->input_index = c->input_index;
  winner->gain = c->gain;
  winner->left_sum_gradients = c->left_sum_gradients;
  winner->left_sum_hessians = c->left_sum_hessians;
  winner->right_sum_gradients = c->right_sum_gradients;
  winner->right_sum_hessians = c->right_sum_hessians;
  winner->left_value = c->left_value;
  winner->left_gain = c->left_gain;
  winner->right_value = c->right_value;
  winner->right_gain = c->right_gain;
  winner->left_count = c->left_count;
  winner->right_count = c->right_count;
}

}  // namespace lgbm_npu_simt

#endif  // NPU_SIMT_SPLIT_VF_H
