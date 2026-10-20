/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LIGHTGBM_SRC_TREELEARNER_NPU_DATA_PARTITION_H_
#define LIGHTGBM_SRC_TREELEARNER_NPU_DATA_PARTITION_H_

#include <cstdint>
#include <vector>

#include "npu_runtime.h"

namespace LightGBM {

// Device-side leaf row ownership for the NPU tree learner. CPU
// DataPartition remains responsible for counts and host compatibility; this
// component owns only resident row spans and their split lifecycle.
class NPUDataPartition {
 public:
  struct LeafRows {
    NPURuntime::PartitionHandle partition;
    bool left = true;
    int64_t count = 0;
  };

  explicit NPUDataPartition(NPURuntime* runtime);

  NPUDataPartition(const NPUDataPartition&) = delete;
  NPUDataPartition& operator=(const NPUDataPartition&) = delete;

  void Reset(int num_leaves);
  void InitializeRoot(const int32_t* rows, int64_t count);
  int64_t Split(int leaf, int feature, uint32_t threshold,
                uint32_t default_bin, int missing_type, bool default_left,
                int right_leaf);
  int64_t SplitCategorical(int leaf, int feature,
                           const std::vector<uint32_t>& categories,
                           uint32_t default_bin, int missing_type,
                           bool default_left, int right_leaf);

  const LeafRows& Leaf(int leaf) const;
  std::vector<int32_t> ReadLeafRows(int leaf) const;

 private:
  NPURuntime* runtime_ = nullptr;  // non-owning; must outlive this component
  std::vector<LeafRows> leaves_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_NPU_DATA_PARTITION_H_