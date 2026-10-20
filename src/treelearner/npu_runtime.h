/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_RUNTIME_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_RUNTIME_H_

#include <LightGBM/utils/random.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace LightGBM {
// Small CANN-only bridge. All methods throw on device errors; no CPU fallback.
class NPURuntime {
 public:
  explicit NPURuntime(int device);
  ~NPURuntime();
  NPURuntime(const NPURuntime&) = delete;
  NPURuntime& operator=(const NPURuntime&) = delete;

  // Dataset-lifetime feature-major bin matrix. The omitted bin is mapped to a
  // sentinel outside all feature histograms; only stored bins are returned.
  void SetHistogramBins(const std::vector<std::vector<uint32_t>>& bins,
                        const std::vector<int>& offsets,
                        const std::vector<int>& num_bins,
                        const std::vector<int>& most_frequent_bins = {});
  // Update once per boosting tree rather than once per feature/leaf.
  void SetHistogramGradients(const float* gradients, const float* hessians,
                             int64_t num_data, bool constant_hessian);
  struct HistogramResult {
    std::vector<std::vector<double>> gradients;
    std::vector<std::vector<double>> hessians;
  };
  // Build histograms for a leaf in bounded feature tiles on device.
  struct DeviceHistogram;
  using HistogramHandle = std::shared_ptr<DeviceHistogram>;
  // Handles must be released before their runtime. Numeric split search
  // consumes resident FP64 values; categorical search currently downloads
  // them for host-side stable ordering and gain evaluation.
  HistogramHandle BuildHistogram(const int32_t* row_indices, int64_t count,
                                  const std::vector<int8_t>& used);
  // The caller owns device_row_indices and keeps it alive through this call.
  // This entry point lets a future device-side partition supply leaf rows
  // without downloading them or uploading them again for every histogram.
  HistogramHandle BuildHistogramFromDeviceRows(const int32_t* device_row_indices,
                                                int64_t count,
                                                const std::vector<int8_t>& used);
  // Integer-only quantized histogram contract. Input bytes are the CPU
  // discretizer's interleaved [unsigned Hessian, signed gradient] pairs.
  // Quantized handles cannot be passed to the floating histogram/search API.
  // Both accumulation lanes use INT64 (no 16/32-bit compaction yet); no scale
  // or dequantization is applied by build, fix, subtract, or readback.
  void SetQuantizedHistogramGradients(const int8_t* pairs, int64_t num_data);
  struct QuantizedHistogramResult {
    std::vector<std::vector<int64_t>> gradients;
    std::vector<std::vector<int64_t>> hessians;
  };
  struct QuantizedDeviceHistogram;
  using QuantizedHistogramHandle = std::shared_ptr<QuantizedDeviceHistogram>;
  QuantizedHistogramHandle BuildQuantizedHistogram(
      const int32_t* row_indices, int64_t count, const std::vector<int8_t>& used);
  QuantizedHistogramHandle BuildQuantizedHistogramFromDeviceRows(
      const int32_t* device_rows, int64_t count, const std::vector<int8_t>& used);
  void FixQuantizedHistogram(const QuantizedHistogramHandle& histogram,
                             int64_t sum_gradient, int64_t sum_hessian);
  // In-place parent <- parent - smaller; requires fixed histograms with
  // matching dataset/gradient generations. Smaller's feature mask may narrow
  // the parent's mask (feature pruning); reactivation is rejected.
  void SubtractQuantizedHistogram(const QuantizedHistogramHandle& parent,
                                  const QuantizedHistogramHandle& smaller);
  QuantizedHistogramResult ReadQuantizedHistogram(
      const QuantizedHistogramHandle& histogram);
  struct DevicePartition;
  using PartitionHandle = std::shared_ptr<DevicePartition>;
  // Stable numeric split: two contiguous row spans remain on device.
  // Caller keeps input rows alive until SplitDeviceRows returns. The returned
  // handle must be released before the runtime/context is destroyed.
  PartitionHandle SplitDeviceRows(const int32_t* device_rows, int64_t count,
                                  int feature, uint32_t threshold,
                                  uint32_t default_bin, int missing_type,
                                  bool default_left);
  PartitionHandle SplitCategoricalRows(const int32_t* device_rows, int64_t count,
                                       int feature,
                                       const std::vector<uint32_t>& categories,
                                       uint32_t default_bin, int missing_type,
                                       bool default_left);
  PartitionHandle UploadPartitionRows(const int32_t* rows, int64_t count);
  int64_t LeftCount(const PartitionHandle& partition) const;
  const int32_t* PartitionRowsDevice(const PartitionHandle& partition,
                                      bool left) const;
  std::vector<int32_t> ReadPartitionRows(const PartitionHandle& partition);
  // Download just one final leaf, rather than a parent split's entire span.
  std::vector<int32_t> ReadPartitionLeafRows(const PartitionHandle& partition,
                                             bool left);
  struct ScoreLeaf {
    PartitionHandle partition;
    bool left = true;
    double output = 0.0;
  };
  // Update host-owned scores via resident device leaf rows. Rows are never
  // downloaded; host scores are uploaded/downloaded for the CPU objective.
  // Leaves must be disjoint and indices must address [0, num_data).
  void AddPredictionToScore(const std::vector<ScoreLeaf>& leaves,
                            int64_t num_data, double* scores);
  // Discard score buffers when the host score storage or dataset changes.
  void ResetScoreBuffers();
  uint64_t score_update_calls() const { return score_update_calls_; }
  uint64_t score_uploads() const { return score_uploads_; }
  uint64_t score_downloads() const { return score_downloads_; }
  uint64_t partition_rows_downloaded() const { return partition_rows_downloaded_; }
  void FixHistogram(const HistogramHandle& histogram,
                    double sum_gradient, double sum_hessian);
  void SubtractHistogram(const HistogramHandle& parent,
                         const HistogramHandle& smaller);
  HistogramResult ReadHistogram(const HistogramHandle& histogram);
  HistogramResult BatchedHistogram(const int32_t* row_indices, int64_t count,
                                   const std::vector<int8_t>& used);
  // Legacy single-feature entry point retained for independent probes.
  void Histogram(const std::vector<int64_t>& indices,
                 const std::vector<float>& gradients,
                 const std::vector<float>& hessians, int64_t bins,
                 std::vector<float>* out_gradients, std::vector<float>* out_hessians);
  // Each contiguous row is scanned independently on NPU in FP64.
  std::vector<double> PrefixSum(const std::vector<double>& input,
                                int64_t rows, int64_t columns);
  struct Gains {
    std::vector<double> gain;
    std::vector<double> left_output;
    std::vector<double> right_output;
  };
  // Hessians include the caller's epsilon. No constraints or min-gain shift here.
  Gains SplitGains(const std::vector<double>& left_g, const std::vector<double>& left_h,
                   const std::vector<double>& right_g, const std::vector<double>& right_h,
                   double lambda_l1, double lambda_l2);
  // First maximum wins. Caller masks invalid candidates with -infinity.
  int64_t ArgMax(const std::vector<double>& input);

  struct SplitFeature {
    int feature = -1;  // Original feature index, used for deterministic ties.
    int histogram_index = -1;  // Internal feature index in a device histogram.
    int num_bin = 0;
    int offset = 0;
    int default_bin = 0;
    int missing_type = 0;  // MissingType: None=0, Zero=1, NaN=2.
    bool enabled = true;
    // Shared per internal feature across leaves/trees, like FeatureMetainfo::rand.
    // Null selects the exhaustive scanner; node-disabled features still draw.
    Random* extra_tree_random = nullptr;
    double penalty = 1.0;
    bool categorical = false;
    int max_cat_threshold = 32;
    int max_cat_to_onehot = 4;
    int min_data_per_group = 100;
    double cat_smooth = 10.0;
    double cat_l2 = 10.0;
    // Populated only for a categorical winner. Values are logical bin ids.
    std::vector<uint32_t> cat_threshold;
    // Stored bins only: logical bin = vector index + offset.
    std::vector<double> gradients;
    std::vector<double> hessians;
  };
  struct SplitResult {
    int feature = -1;
    uint32_t threshold = 0;
    bool categorical = false;
    std::vector<uint32_t> cat_threshold;
    bool default_left = true;
    double gain = 0.0;
    double left_output = 0.0, right_output = 0.0;
    double left_gradient = 0.0, right_gradient = 0.0;
    double left_hessian = 0.0, right_hessian = 0.0;
    int left_count = 0, right_count = 0;
    // Same order as the input features, including node-disabled features.
    std::vector<uint8_t> splittable;
  };
  // Resident integer candidate preparation, separate from FindBestSplit.
  // This component API does not enable quantized tree training.
  struct QuantizedSplitCandidate {
    int feature = -1;
    uint32_t threshold = 0;
    bool default_left = true;
    int64_t left_gradient = 0, left_hessian = 0;
    int64_t right_gradient = 0, right_hessian = 0;
  };
  struct QuantizedDeviceSplitScan;
  using QuantizedSplitScanHandle = std::shared_ptr<QuantizedDeviceSplitScan>;
  // Exhaustive numerical scans only; categorical and randomized requests are
  // explicitly rejected until their quantized semantics are implemented.
  QuantizedSplitScanHandle PrepareQuantizedSplitScan(
      const std::vector<SplitFeature>& features, int64_t sum_gradient,
      int64_t sum_hessian, int num_data,
      const QuantizedHistogramHandle& histogram);
  // Diagnostic readback for probes, not part of the training hot path.
  std::vector<QuantizedSplitCandidate> ReadQuantizedSplitScan(
      const QuantizedSplitScanHandle& scan);

  struct QuantizedSplitParameters {
    double gradient_scale = 1.0, hessian_scale = 1.0;
    double lambda_l1 = 0.0, lambda_l2 = 0.0;
    int min_data_in_leaf = 1;
    double min_sum_hessian_in_leaf = 0.0, min_gain_to_split = 0.0;
    double max_delta_step = 0.0, path_smooth = 0.0, parent_output = 0.0;
  };
  struct QuantizedSplitEvaluation {
    QuantizedSplitCandidate candidate;
    double left_gradient = 0.0, right_gradient = 0.0;
    double left_hessian = 0.0, right_hessian = 0.0;
    int left_count = 0, right_count = 0;
    int scan_left_count = 0, scan_right_count = 0;
    double left_output = 0.0, right_output = 0.0;
    double current_gain = 0.0, gain_shift = 0.0;
    bool valid = false;
  };
  struct QuantizedDeviceSplitEvaluation;
  using QuantizedSplitEvaluationHandle = std::shared_ptr<QuantizedDeviceSplitEvaluation>;
  // Count/scale/gain/output arithmetic runs on device after INT64 scanning.
  // Invalid candidates expose zero gain/outputs, rather than unsafe division
  // results. The
  // combination min_data=0, min_sum_hessian=0, lambda_l2=0 is rejected:
  // epsilon-only gain validity cannot guarantee a defined raw-H final output.
  QuantizedSplitEvaluationHandle EvaluateQuantizedSplitScan(
      const QuantizedSplitScanHandle& scan, const QuantizedSplitParameters& parameters);
  std::vector<QuantizedSplitEvaluation> ReadQuantizedSplitEvaluation(
      const QuantizedSplitEvaluationHandle& evaluation);
  struct QuantizedSplitSelection {
    SplitResult split;
    QuantizedSplitCandidate integer_statistics;
  };
  // Device-only direction and cross-feature reduction. Downloads only the
  // winning candidate and per-direction splittability flags, never all gains
  // or integer candidates. Requires a current dataset/gradient generation;
  // diagnostic evaluation readback remains a historical snapshot API.
  QuantizedSplitSelection SelectQuantizedSplit(
      const QuantizedSplitEvaluationHandle& evaluation);
  uint64_t quantized_selection_download_bytes() const {
    return quantized_selection_download_bytes_;
  }

  // Numeric candidate arithmetic, validity masks and winner selection run on
  // device. Categorical histograms are device-built, with host split search.
  SplitResult FindBestSplit(const std::vector<SplitFeature>& features,
                            double sum_gradient, double sum_hessian, int num_data,
                            double lambda_l1, double lambda_l2,
                            int min_data_in_leaf, double min_sum_hessian_in_leaf,
                            double min_gain_to_split,
                            const HistogramHandle& histogram = nullptr,
                            double max_delta_step = 0.0,
                            double path_smooth = 0.0,
                            double parent_output = 0.0);
  uint64_t split_calls() const { return split_calls_; }

  uint64_t histogram_calls() const { return histogram_calls_; }
  uint64_t prefix_calls() const { return prefix_calls_; }
  uint64_t gain_calls() const { return gain_calls_; }
  uint64_t argmax_calls() const { return argmax_calls_; }
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  uint64_t histogram_calls_ = 0, prefix_calls_ = 0, gain_calls_ = 0, argmax_calls_ = 0;
  uint64_t split_calls_ = 0;
  uint64_t score_update_calls_ = 0, score_uploads_ = 0, score_downloads_ = 0;
  uint64_t partition_rows_downloaded_ = 0;
  uint64_t quantized_selection_download_bytes_ = 0;
};
}  // namespace LightGBM
#endif