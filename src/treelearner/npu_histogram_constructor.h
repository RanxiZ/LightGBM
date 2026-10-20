/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_HISTOGRAM_CONSTRUCTOR_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_HISTOGRAM_CONSTRUCTOR_H_
#include <cstdint>
#include <vector>
#include "npu_runtime.h"
namespace LightGBM {
// Device histogram lifecycle aligned with CUDAHistogramConstructor.
class NPUHistogramConstructor {
 public:
  explicit NPUHistogramConstructor(NPURuntime* runtime);
  NPUHistogramConstructor(const NPUHistogramConstructor&) = delete;
  NPUHistogramConstructor& operator=(const NPUHistogramConstructor&) = delete;
  NPURuntime::HistogramHandle BuildFromDeviceRows(const int32_t* device_rows,
      int64_t count, const std::vector<int8_t>& used);
  void Fix(const NPURuntime::HistogramHandle& histogram,
           double sum_gradient, double sum_hessian);
  NPURuntime::QuantizedHistogramHandle BuildQuantizedFromDeviceRows(
      const int32_t* device_rows, int64_t count, const std::vector<int8_t>& used);
  void FixQuantized(const NPURuntime::QuantizedHistogramHandle& histogram,
                    int64_t sum_gradient, int64_t sum_hessian);
  void SubtractQuantized(const NPURuntime::QuantizedHistogramHandle& parent,
                         const NPURuntime::QuantizedHistogramHandle& smaller);
  void Subtract(const NPURuntime::HistogramHandle& parent,
                const NPURuntime::HistogramHandle& smaller);
 private:
  NPURuntime* runtime_ = nullptr;
};
}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_TREELEARNER_NPU_HISTOGRAM_CONSTRUCTOR_H_