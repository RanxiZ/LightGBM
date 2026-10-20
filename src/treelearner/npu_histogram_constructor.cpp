/* Copyright (c) 2026. Licensed under the MIT License. */
#include "npu_histogram_constructor.h"
#include <LightGBM/utils/common.h>
namespace LightGBM {
NPUHistogramConstructor::NPUHistogramConstructor(NPURuntime* runtime) : runtime_(runtime) {
  CHECK(runtime_ != nullptr);
}
NPURuntime::HistogramHandle NPUHistogramConstructor::BuildFromDeviceRows(
    const int32_t* device_rows, int64_t count, const std::vector<int8_t>& used) {
  CHECK(runtime_ != nullptr);
  CHECK_GE(count, 0);
  return runtime_->BuildHistogramFromDeviceRows(device_rows, count, used);
}
void NPUHistogramConstructor::Fix(const NPURuntime::HistogramHandle& histogram,
                                  double sum_gradient, double sum_hessian) {
  CHECK(runtime_ != nullptr);
  runtime_->FixHistogram(histogram, sum_gradient, sum_hessian);
}
NPURuntime::QuantizedHistogramHandle NPUHistogramConstructor::BuildQuantizedFromDeviceRows(
    const int32_t* device_rows, int64_t count, const std::vector<int8_t>& used) {
  CHECK(runtime_ != nullptr);
  CHECK_GE(count, 0);
  return runtime_->BuildQuantizedHistogramFromDeviceRows(device_rows, count, used);
}
void NPUHistogramConstructor::FixQuantized(
    const NPURuntime::QuantizedHistogramHandle& histogram,
    int64_t sum_gradient, int64_t sum_hessian) {
  CHECK(runtime_ != nullptr);
  runtime_->FixQuantizedHistogram(histogram, sum_gradient, sum_hessian);
}
void NPUHistogramConstructor::SubtractQuantized(
    const NPURuntime::QuantizedHistogramHandle& parent,
    const NPURuntime::QuantizedHistogramHandle& smaller) {
  CHECK(runtime_ != nullptr);
  runtime_->SubtractQuantizedHistogram(parent, smaller);
}
void NPUHistogramConstructor::Subtract(const NPURuntime::HistogramHandle& parent,
                                       const NPURuntime::HistogramHandle& smaller) {
  CHECK(runtime_ != nullptr);
  runtime_->SubtractHistogram(parent, smaller);
}
}  // namespace LightGBM