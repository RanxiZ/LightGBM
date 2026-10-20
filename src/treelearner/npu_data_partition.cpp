/* Copyright (c) 2026. Licensed under the MIT License. */
#include "npu_data_partition.h"

#include <LightGBM/utils/common.h>

namespace LightGBM {

NPUDataPartition::NPUDataPartition(NPURuntime* runtime) : runtime_(runtime) {
  CHECK(runtime_ != nullptr);
}

void NPUDataPartition::Reset(int num_leaves) {
  CHECK_GE(num_leaves, 1);
  leaves_.assign(static_cast<size_t>(num_leaves), LeafRows{});
}

void NPUDataPartition::InitializeRoot(const int32_t* rows, int64_t count) {
  CHECK(!leaves_.empty());
  CHECK_GE(count, 0);
  leaves_[0].partition = runtime_->UploadPartitionRows(rows, count);
  leaves_[0].left = true;
  leaves_[0].count = count;
}

int64_t NPUDataPartition::Split(int leaf, int feature, uint32_t threshold,
                                uint32_t default_bin, int missing_type,
                                bool default_left, int right_leaf) {
  CHECK_GE(leaf, 0);
  CHECK_LT(leaf, static_cast<int>(leaves_.size()));
  CHECK_GE(right_leaf, 0);
  CHECK_LT(right_leaf, static_cast<int>(leaves_.size()));
  const auto& parent = leaves_.at(leaf);
  CHECK(parent.partition != nullptr);
  const int64_t parent_count = parent.count;
  auto partition = runtime_->SplitDeviceRows(
      runtime_->PartitionRowsDevice(parent.partition, parent.left), parent_count,
      feature, threshold, default_bin, missing_type, default_left);
  const int64_t left_count = runtime_->LeftCount(partition);
  leaves_.at(leaf).partition = partition;
  leaves_.at(leaf).left = true;
  leaves_.at(leaf).count = left_count;
  leaves_.at(right_leaf).partition = partition;
  leaves_.at(right_leaf).left = false;
  leaves_.at(right_leaf).count = parent_count - left_count;
  return left_count;
}

int64_t NPUDataPartition::SplitCategorical(
    int leaf, int feature, const std::vector<uint32_t>& categories,
    uint32_t default_bin, int missing_type, bool default_left, int right_leaf) {
  CHECK_GE(leaf, 0);
  CHECK_LT(leaf, static_cast<int>(leaves_.size()));
  CHECK_GE(right_leaf, 0);
  CHECK_LT(right_leaf, static_cast<int>(leaves_.size()));
  const auto& parent = leaves_.at(leaf);
  CHECK(parent.partition != nullptr);
  const int64_t parent_count = parent.count;
  auto partition = runtime_->SplitCategoricalRows(
      runtime_->PartitionRowsDevice(parent.partition, parent.left), parent_count,
      feature, categories, default_bin, missing_type, default_left);
  const int64_t left_count = runtime_->LeftCount(partition);
  leaves_.at(leaf).partition = partition;
  leaves_.at(leaf).left = true;
  leaves_.at(leaf).count = left_count;
  leaves_.at(right_leaf).partition = partition;
  leaves_.at(right_leaf).left = false;
  leaves_.at(right_leaf).count = parent_count - left_count;
  return left_count;
}

const NPUDataPartition::LeafRows& NPUDataPartition::Leaf(int leaf) const {
  CHECK_GE(leaf, 0);
  CHECK_LT(leaf, static_cast<int>(leaves_.size()));
  return leaves_.at(leaf);
}

std::vector<int32_t> NPUDataPartition::ReadLeafRows(int leaf) const {
  const auto& rows = Leaf(leaf);
  CHECK(rows.partition != nullptr);
  return runtime_->ReadPartitionLeafRows(rows.partition, rows.left);
}

}  // namespace LightGBM