/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_SCORE_UPDATER_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_SCORE_UPDATER_H_

#include <cstdint>
#include <vector>

#include "npu_runtime.h"

namespace LightGBM {

// Training-facing score update component. The runtime owns ACL resources and
// device partitions; this component owns the contract for applying a tree's
// disjoint leaf outputs to the host score vector without downloading rows.
class NPUScoreUpdater {
 public:
  explicit NPUScoreUpdater(NPURuntime* runtime);

  NPUScoreUpdater(const NPUScoreUpdater&) = delete;
  NPUScoreUpdater& operator=(const NPUScoreUpdater&) = delete;

  void Update(const std::vector<NPURuntime::ScoreLeaf>& leaves,
              int64_t num_data, double* scores);

  uint64_t update_calls() const;

 private:
  NPURuntime* runtime_ = nullptr;  // non-owning; must outlive this component
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_NPU_SCORE_UPDATER_H_