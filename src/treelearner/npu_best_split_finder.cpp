/* Copyright (c) 2026. Licensed under the MIT License. */
#include "npu_best_split_finder.h"

#include <LightGBM/utils/common.h>

namespace LightGBM {

NPUBestSplitFinder::NPUBestSplitFinder(NPURuntime* runtime) : runtime_(runtime) {
  CHECK(runtime_ != nullptr);
}

NPURuntime::SplitResult NPUBestSplitFinder::Find(
    const std::vector<NPURuntime::SplitFeature>& features,
    double sum_gradient, double sum_hessian, int num_data_in_leaf,
    const Parameters& parameters, const NPURuntime::HistogramHandle& histogram) {
  CHECK(runtime_ != nullptr);
  return runtime_->FindBestSplit(features, sum_gradient, sum_hessian,
                                 num_data_in_leaf, parameters.lambda_l1,
                                 parameters.lambda_l2,
                                 parameters.min_data_in_leaf,
                                 parameters.min_sum_hessian_in_leaf,
                                 parameters.min_gain_to_split, histogram,
                                 parameters.max_delta_step, parameters.path_smooth,
                                 parameters.parent_output);
}

NPURuntime::QuantizedSplitSelection NPUBestSplitFinder::FindQuantized(
    const std::vector<NPURuntime::SplitFeature>& features,
    int64_t sum_gradient, int64_t sum_hessian, int num_data_in_leaf,
    const NPURuntime::QuantizedSplitParameters& parameters,
    const NPURuntime::QuantizedHistogramHandle& histogram) {
  const auto scan = runtime_->PrepareQuantizedSplitScan(
      features, sum_gradient, sum_hessian, num_data_in_leaf, histogram);
  return runtime_->SelectQuantizedSplit(runtime_->EvaluateQuantizedSplitScan(scan, parameters));
}

}  // namespace LightGBM