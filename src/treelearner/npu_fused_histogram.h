/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LGBMNPU_SRC_TREELEARNER_NPU_FUSED_HISTOGRAM_V2_H_
#define LGBMNPU_SRC_TREELEARNER_NPU_FUSED_HISTOGRAM_V2_H_

#include <stdint.h>

// Host/device ABI for the fused histogram construction operator.
//
// The logical contract is identical to the existing AI CPU operator package
// (/workspace/LightGBM_optmize/LGBMNPUFusedHistogram):
//   * sizeof(LGBMNPUFusedHistogramFeature) == 24 bytes.
//   * Bin tiles are feature-major: tile[feature_in_tile * histogram_rows + row]
//     holds `feature_in_tile * histogram_stride + bin - offset` for every
//     stored bin, and `sentinel = tile_width * histogram_stride` for every
//     omitted bin (bin < offset, bin >= num_bins, or bin == most frequent bin).
//   * Gradients/Hessians are FP32.
//   * Output layout: [0, output_features*stride) is the gradient plane and
//     [output_features*stride, 2*output_features*stride) is the Hessian plane.
//
// Differences specific to this AI Core SIMT package (see docs/DESIGN.md):
//   * No `completed` flag: the host synchronises the stream and checks the ACL
//     return code. The AI CPU ABI needed the flag because its launch is
//     fire-and-forget across the AI CPU queue.
//   * The SIMT kernels receive the packed tile matrix as one contiguous buffer
//     (uniform tile_width slots per tile row) instead of an array of per-tile
//     pointers. The AI CPU reference kernel still receives the pointer array;
//     both point at the same packed data.
//   * Accumulation is deterministic int64 fixed point (see simt_fixed_point.h)
//     instead of sequential FP64. The final conversion of the exact integer
//     partial sums to binary64 is done on the device with round-to-nearest-even.
struct LGBMNPUFusedHistogramFeature {
  int32_t output_index;   // Destination feature row in the output planes.
  int32_t tile_index;     // Packed tile holding this feature (feature / 8).
  int32_t tile_feature;   // Row inside the tile (feature % 8).
  int32_t num_bins;       // Logical bin count (including the omitted bin).
  int32_t offset;         // 0 or 1; stored bins are [offset, num_bins).
  int32_t reserved;       // Keeps the ABI at 24 bytes.
};

#if !defined(LGBM_NPU_DEVICE_SOURCE) && !defined(LGBM_AICPU_DEVICE_SOURCE)

#ifdef __cplusplus
extern "C" {
#endif

// Reference AI CPU kernels (not part of the SIMT path). Declared so the probe
// can build an A/B comparison against the sequential-FP64 baseline.
void LGBMNPUFusedHistogramKernel(
    uint32_t block_dim, void* launch_config, void* stream,
    const int32_t* const* histogram_tiles,
    const int32_t* device_row_indices, const float* gradients,
    const float* hessians, const LGBMNPUFusedHistogramFeature* features,
    double* output, int32_t num_features, int32_t output_features,
    int64_t count, int32_t histogram_rows, int32_t histogram_stride,
    int32_t tile_width, int32_t* completed);
void LGBMNPUFusedHistogramReferenceKernel(
    uint32_t block_dim, void* launch_config, void* stream,
    const int32_t* const* histogram_tiles,
    const int32_t* device_row_indices, const float* gradients,
    const float* hessians, const LGBMNPUFusedHistogramFeature* features,
    double* output, int32_t num_features, int32_t output_features,
    int64_t count, int32_t histogram_rows, int32_t histogram_stride,
    int32_t tile_width, int32_t* completed);

// ---------------------------------------------------------------------------
// v2 pipeline: exactly two dependent launches.
//   1. v2_scan_kernel : per-bin |gradient| / |hessian| maxima -> `bin_max`
//   2. v2_hist_kernel : per-bin exponent (derived in-kernel), UB-resident
//                       histogram, one cross-block merge per (feature, bin,
//                       window) -> `partials`, and the last block of each
//                       feature partition converts to binary64 -> `output`
//                       and resets the scratch.
//
// Layout notes that differ from the v1 entry point above:
//   row_bins      : [histogram_rows][num_features] int32. Element (row, feature)
//                   is the *stored* index (bin - offset), or -1 when the bin is
//                   omitted / out of range. Row-major so the kF lanes of one
//                   row read kF contiguous words.
//   feature_index : [num_used] int32, global feature index of each used slot.
//   stored_bins   : [num_features] int32, num_bins - offset.
//   bin_max       : [2][num_used][kernel_stride] uint32 scratch.
//   shift         : [2][num_features][histogram_stride] int32 diagnostic output,
//                   interleaved as shift[2*(feature*stride + bin) + plane].
//   partials      : [6][num_used][kernel_stride] int64 scratch, slot
//                   s = window*2 + plane (see kAccumulatorSlots in the kernel).
//   counters      : [partitions] int32 scratch (one ticket per feature partition).
//
// `kernel_stride` (32/64/128/256) is the compile-time UB tile; it must be >=
// histogram_stride. `shift_gap` is 61 - ceil(log2(count)); the kernel always
// runs three magnitude windows (window2 is retained only for ABI stability).
// ---------------------------------------------------------------------------
void LGBMNFusedHistogramSimtV2Launch(
    void* stream, const int32_t* row_bins, const int32_t* device_rows,
    const float* gradients, const float* hessians,
    const int32_t* feature_index, const int32_t* stored_bins, uint32_t* bin_max,
    int32_t* shift, int64_t* partials, int32_t* counters, double* output,
    int32_t num_features, int32_t num_used, int64_t count,
    int32_t histogram_stride, int32_t partitions, int32_t row_chunks,
    int32_t kernel_stride, int32_t shift_gap, int32_t window2);

// Full pipeline on `stream`, in four dependent passes:
//   1. per-bin |gradient| / |hessian| maxima (sharded int32 bit patterns)
//   2. per-bin quantization exponent                 -> `shift`
//   3. quantize + INT64 atomic accumulation           -> `partials`
//   4. merge shards, int64 -> binary64, reset partials -> `output`
//
//   packed_tiles     : [num_tiles][tile_width][histogram_rows] int32
//   features         : metadata of the used features, length num_features
//   slot_of_feature  : [output_features] int32, index into `features` or -1
//   bin_max          : [num_features][shards][2][histogram_stride] uint32
//                      scratch, zeroed before the first call and re-zeroed by
//                      pass 2 (so the buffer can be reused across calls)
//   shift            : [2][output_features][histogram_stride] int32, filled by
//                      pass 2; element (plane, feature, bin) is the power-of-two
//                      exponent for that bin, and pass 4 negates it when it
//                      converts the int64 total back to binary64
//   partials         : [num_features][shards][2][histogram_stride] int64,
//                      zeroed before the first call and re-zeroed by pass 4
//   stored_bins      : [output_features] int32, num_bins - offset (0 if unused)
//   blocks_per_feature / shards: must match the bin_max/partials layout
//
// Determinism: every value that reaches `output` is an exact integer sum
// (associative and commutative) followed by one round-to-nearest-even
// conversion, so the result does not depend on block or thread scheduling.
void LGBMNFusedHistogramSimtLaunch(
    void* stream, const int32_t* packed_tiles, const int32_t* device_rows,
    const float* gradients, const float* hessians,
    const LGBMNPUFusedHistogramFeature* features, int32_t num_features,
    const int32_t* slot_of_feature, uint32_t* bin_max, int32_t* shift,
    int64_t* partials, const int32_t* stored_bins, double* output,
    int32_t output_features, int64_t count, int32_t histogram_rows,
    int32_t histogram_stride, int32_t tile_width, int32_t blocks_per_feature,
    int32_t shards);

#ifdef __cplusplus
}
#endif

#endif  // !LGBM_NPU_DEVICE_SOURCE && !LGBM_AICPU_DEVICE_SOURCE

#ifdef __cplusplus
static_assert(sizeof(LGBMNPUFusedHistogramFeature) == 24,
              "fused histogram feature ABI");
#endif

#endif  // LGBMNPU_SRC_TREELEARNER_NPU_FUSED_HISTOGRAM_V2_H_
