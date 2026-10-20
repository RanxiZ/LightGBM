/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_BEST_SPLIT_FINDER_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_BEST_SPLIT_FINDER_H_

#include <cstdint>
#include <vector>

#include "npu_runtime.h"

namespace LightGBM {

// Training-facing best-split component. The CPU tree learner remains the
// owner of SplitInfo and tree-growth decisions; this component owns only the
// device candidate-search contract and its objective parameters.
class NPUBestSplitFinder {
 public:
  struct Parameters {
    double lambda_l1 = 0.0;
    double lambda_l2 = 0.0;
    int min_data_in_leaf = 0;
    double min_sum_hessian_in_leaf = 0.0;
    double min_gain_to_split = 0.0;
    double max_delta_step = 0.0;
    double path_smooth = 0.0;
    double parent_output = 0.0;
  };

  explicit NPUBestSplitFinder(NPURuntime* runtime);

  NPUBestSplitFinder(const NPUBestSplitFinder&) = delete;
  NPUBestSplitFinder& operator=(const NPUBestSplitFinder&) = delete;

  NPURuntime::SplitResult Find(const std::vector<NPURuntime::SplitFeature>& features,
                               double sum_gradient, double sum_hessian,
                               int num_data_in_leaf,
                               const Parameters& parameters,
                               const NPURuntime::HistogramHandle& histogram);

  NPURuntime::QuantizedSplitSelection FindQuantized(
      const std::vector<NPURuntime::SplitFeature>& features,
      int64_t sum_gradient, int64_t sum_hessian, int num_data_in_leaf,
      const NPURuntime::QuantizedSplitParameters& parameters,
      const NPURuntime::QuantizedHistogramHandle& histogram);

 private:
  NPURuntime* runtime_ = nullptr;  // non-owning; must outlive this component
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_NPU_BEST_SPLIT_FINDER_H_