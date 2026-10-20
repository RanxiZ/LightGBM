/* Copyright (c) 2026. Licensed under the MIT License. */
#include "npu_score_updater.h"

#include <LightGBM/utils/common.h>

namespace LightGBM {

NPUScoreUpdater::NPUScoreUpdater(NPURuntime* runtime) : runtime_(runtime) {
  CHECK(runtime_ != nullptr);
}

void NPUScoreUpdater::Update(const std::vector<NPURuntime::ScoreLeaf>& leaves,
                             int64_t num_data, double* scores) {
  CHECK(runtime_ != nullptr);
  runtime_->AddPredictionToScore(leaves, num_data, scores);
}

uint64_t NPUScoreUpdater::update_calls() const {
  CHECK(runtime_ != nullptr);
  return runtime_->score_update_calls();
}

}  // namespace LightGBM