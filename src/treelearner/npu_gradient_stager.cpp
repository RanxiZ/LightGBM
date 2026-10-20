/* Copyright (c) 2026. Licensed under the MIT License. */
#include "npu_gradient_stager.h"

#include <LightGBM/utils/common.h>

namespace LightGBM {

NPUGradientStager::NPUGradientStager(NPURuntime* runtime) : runtime_(runtime) {
  CHECK(runtime_ != nullptr);
}

void NPUGradientStager::Stage(const float* gradients, const float* hessians,
                              int64_t num_data, bool constant_hessian) {
  CHECK(runtime_ != nullptr);
  CHECK_GE(num_data, 0);
  if (num_data > 0) {
    CHECK(gradients != nullptr);
    CHECK(hessians != nullptr);
  }
  runtime_->SetHistogramGradients(gradients, hessians, num_data, constant_hessian);
}

void NPUGradientStager::StageQuantized(const int8_t* pairs, int64_t num_data) {
  CHECK(runtime_ != nullptr);
  CHECK_GE(num_data, 0);
  if (num_data > 0) {
    CHECK(pairs != nullptr);
  }
  runtime_->SetQuantizedHistogramGradients(pairs, num_data);
}

}  // namespace LightGBM