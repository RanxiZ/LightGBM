/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_GRADIENT_STAGER_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_GRADIENT_STAGER_H_

#include <cstdint>

#include "npu_runtime.h"

namespace LightGBM {

// Training-facing gradient staging component. Objective calculation, bagging,
// and CPU tree scheduling remain outside this class; it only publishes the
// current iteration's host gradient/Hessian arrays to the device histogram
// runtime once per tree.
class NPUGradientStager {
 public:
  explicit NPUGradientStager(NPURuntime* runtime);

  NPUGradientStager(const NPUGradientStager&) = delete;
  NPUGradientStager& operator=(const NPUGradientStager&) = delete;

  void Stage(const float* gradients, const float* hessians,
             int64_t num_data, bool constant_hessian);
  // Forward CPU-discretized dataset-indexed [unsigned Hessian byte, signed
  // gradient byte] pairs without reinterpretation or dequantization. Only the
  // private quantized lifecycle probe uses this during tree training; public
  // quantized NPU requests remain rejected.
  void StageQuantized(const int8_t* pairs, int64_t num_data);

 private:
  NPURuntime* runtime_ = nullptr;  // non-owning; must outlive this component
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_NPU_GRADIENT_STAGER_H_