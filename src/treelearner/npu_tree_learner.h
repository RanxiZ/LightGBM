/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_TREE_LEARNER_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_TREE_LEARNER_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "serial_tree_learner.h"

#ifdef USE_NPU

#include <LightGBM/utils/random.h>

#include "npu_runtime.h"
#include "npu_score_updater.h"
#include "npu_best_split_finder.h"
#include "npu_data_partition.h"
#include "npu_gradient_stager.h"
#include "npu_histogram_constructor.h"

namespace LightGBM {
// Experimental NPU backend; device rows drive tree growth and score updates.
// Host rows are materialized only for legacy leaf-renewal/refit consumers.
class NPUTreeLearner : public SerialTreeLearner {
 public:
  explicit NPUTreeLearner(const Config* config);
  ~NPUTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian,
                              bool reset_multi_val_bin) override;
  void ResetConfig(const Config* config) override;
  void SetForcedSplit(const Json* forced_split_json) override;
  Tree* Train(const score_t* gradients, const score_t* hessians, bool is_first_tree) override;
  void AddPredictionToScore(const Tree* tree, double* out_score) const override;
  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj,
                       std::function<double(const label_t*, int)> residual_getter,
                       data_size_t total_num_data, const data_size_t* bag_indices,
                       data_size_t bag_cnt, const double* train_score) const override;
  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients,
                          const score_t* hessians) const override;
  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) const override;

 protected:
  void BeforeTrain() override;
  void InitLeafSplits(LeafSplits* splits, int leaf, double sum_gradient,
                      double sum_hessian, double output) override;
  void InitQuantizedLeafSplits(LeafSplits* splits, int leaf, double sum_gradient,
                               double sum_hessian, int64_t packed, double output) override;
  void RenewQuantizedTreeOutput(Tree* tree) override;
  void PartitionData(int leaf, int feature, const uint32_t* threshold,
                     int num_threshold, bool default_left, int right_leaf) override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used,
                                    bool use_subtract, const Tree* tree) override;
  int SelectBestLeaf() const override;

 private:
  // Only the C++ regression fixture can construct the incomplete quantized
  // path. No parameter, environment variable or public factory enables it.
  friend class NPUQuantizedTreeProbe;
  NPUTreeLearner(const Config* config, bool quantized_probe);
  const bool quantized_probe_;
  void ValidateConfig(const Config* config) const;
  void ValidateDataset(const Dataset* train_data, const Config* config) const;
  void ValidateForcedSplit(const Json* forced_split_json) const;
  void FindBestSplitForLeaf(const std::vector<int8_t>& is_feature_used,
                            const std::vector<int8_t>& node_used_features,
                            const LeafSplits* leaf, FeatureHistogram* histograms,
                            double parent_output);
  void MaterializeHostRows(int num_leaves) const;
  mutable bool host_rows_current_ = true;
  mutable bool device_rows_current_ = false;
  uint64_t tree_rows_downloaded_before_ = 0;
  void CacheBins();
  void ResetExtraTreeRandom();
  // HistogramPool metadata is shared by all leaves and survives BeforeTrain.
  // ResetConfig reseeds it even when extra_seed itself has not changed.
  std::vector<Random> extra_tree_random_;
  void ConstructLeaf(const std::vector<int8_t>& used, const LeafSplits* leaf,
                     FeatureHistogram* histograms);
  std::unique_ptr<NPURuntime> runtime_;
  std::unique_ptr<NPUScoreUpdater> score_updater_;
  std::unique_ptr<NPUBestSplitFinder> best_split_finder_;
  std::unique_ptr<NPUDataPartition> data_partition_npu_;
  std::unique_ptr<NPUGradientStager> gradient_stager_;
  std::unique_ptr<NPUHistogramConstructor> histogram_constructor_;
  // Keys are physical HistogramPool slots; pool moves retain the same slot.
  // Cleared on every tree/reset, and destroyed before the runtime context.
  std::unordered_map<FeatureHistogram*, NPURuntime::HistogramHandle> device_histograms_;
  std::unordered_map<FeatureHistogram*, NPURuntime::QuantizedHistogramHandle> quantized_histograms_;
  uint64_t quantized_histogram_subtractions_ = 0;
  uint64_t quantized_leaf_renewals_ = 0;
  // Logical feature bins decoded via Dataset::FeatureIterator (works with EFB).
  std::vector<std::vector<uint32_t>> bins_;
};
}  // namespace LightGBM

#else  // USE_NPU

namespace LightGBM {
// When NPU support is not compiled in, quit with an error message.
class NPUTreeLearner : public SerialTreeLearner {
 public:
  explicit NPUTreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
    Log::Fatal("NPU Tree Learner was not enabled in this build.\n"
               "Please recompile with CMake option -DUSE_NPU=1");
  }
};
}  // namespace LightGBM

#endif  // USE_NPU
#endif  // LIGHTGBM_SRC_TREELEARNER_NPU_TREE_LEARNER_H_