/* Copyright (c) 2026. Licensed under the MIT License. */
#include "npu_tree_learner.h"

#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <numeric>
#include <utility>

namespace LightGBM {

namespace {
int64_t PackNPUStatistics(int64_t gradient, int64_t hessian) {
  CHECK_GE(gradient, std::numeric_limits<int32_t>::min());
  CHECK_LE(gradient, std::numeric_limits<int32_t>::max());
  CHECK_GE(hessian, 0);
  CHECK_LE(static_cast<uint64_t>(hessian), std::numeric_limits<uint32_t>::max());
  // Multiplication is defined for negative gradients, unlike a signed shift.
  return gradient * int64_t{4294967296} + hessian;
}
int64_t PackedGradient(int64_t packed) {
  const uint32_t high = static_cast<uint64_t>(packed) >> 32;
  int32_t gradient;
  std::memcpy(&gradient, &high, sizeof(gradient));
  return gradient;
}
int64_t PackedHessian(int64_t packed) { return static_cast<uint32_t>(packed); }
}  // namespace

NPUTreeLearner::NPUTreeLearner(const Config* config) : NPUTreeLearner(config, false) {}

NPUTreeLearner::NPUTreeLearner(const Config* config, bool quantized_probe)
    : SerialTreeLearner(config), quantized_probe_(quantized_probe) {
  train_data_ = nullptr;
  forced_split_json_ = nullptr;
  ValidateConfig(config);
}

NPUTreeLearner::~NPUTreeLearner() {}

// This backend intentionally covers only the plain histogram path. Options
// that change the histogram layout or unsupported split semantics are rejected
// up front rather than silently ignored, so a run either matches the CPU
// semantics or refuses to start.
void NPUTreeLearner::ValidateConfig(const Config* config) const {
  if (config->use_quantized_grad && !quantized_probe_) {
    Log::Fatal("device_type=npu does not support use_quantized_grad");
  }
  if (config->use_quantized_grad) {
    CHECK_GE(config->num_grad_quant_bins, 2);
    CHECK_LE(config->num_grad_quant_bins, 127);
    CHECK_GE(config->min_data_in_leaf, 1);
    if (config->extra_trees) Log::Fatal("NPU quantized probe does not support extra_trees");
  }
  if (config->monotone_constraints.size() > 0) {
    Log::Fatal("device_type=npu does not support monotone_constraints");
  }
  if (config->linear_tree) {
    Log::Fatal("device_type=npu does not support linear_tree");
  }
  if (config->cegb_tradeoff != 1.0 || config->cegb_penalty_split != 0.0 ||
      !config->cegb_penalty_feature_coupled.empty() ||
      !config->cegb_penalty_feature_lazy.empty()) {
    Log::Fatal("device_type=npu does not support CEGB");
  }
  if (!config->forcedsplits_filename.empty()) {
    Log::Fatal("device_type=npu does not support forced splits");
  }
  ValidateForcedSplit(forced_split_json_);
  if (Network::num_machines() > 1) {
    Log::Fatal("device_type=npu does not support distributed training");
  }
}

void NPUTreeLearner::ValidateForcedSplit(const Json* forced_split_json) const {
  if (forced_split_json != nullptr && !forced_split_json->is_null() &&
      !(forced_split_json->is_object() && forced_split_json->object_items().empty())) {
    Log::Fatal("device_type=npu does not support non-empty forced split JSON");
  }
}

void NPUTreeLearner::SetForcedSplit(const Json* forced_split_json) {
  ValidateForcedSplit(forced_split_json);
  // Normalize {} as well as null: Serial's ForceSplits treats any pointer as
  // a forced split request, even an empty object.
  forced_split_json_ = nullptr;
}

void NPUTreeLearner::ValidateDataset(const Dataset* train_data,
                                    const Config* config) const {
  if (config->use_quantized_grad) {
    CHECK_GT(train_data->num_data(), 0);
    CHECK_LE(static_cast<int64_t>(train_data->num_data()) * config->num_grad_quant_bins,
             std::numeric_limits<int32_t>::max());
    for (int f = 0; f < train_data->num_features(); ++f) {
      if (train_data->FeatureBinMapper(f)->bin_type() != BinType::NumericalBin)
        Log::Fatal("NPU quantized probe does not support categorical features");
    }
  }
  if (!config->feature_contri.empty() &&
      config->feature_contri.size() != static_cast<size_t>(train_data->num_total_features())) {
    Log::Fatal("feature_contri should be the same size as feature number");
  }
}

void NPUTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  ValidateConfig(config_);
  ValidateDataset(train_data, config_);
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  ResetExtraTreeRandom();
  device_histograms_.clear();
  quantized_histograms_.clear();
  runtime_.reset(new NPURuntime(config_->gpu_device_id > 0 ? config_->gpu_device_id : 0));
  score_updater_.reset(new NPUScoreUpdater(runtime_.get()));
  best_split_finder_.reset(new NPUBestSplitFinder(runtime_.get()));
  data_partition_npu_.reset(new NPUDataPartition(runtime_.get()));
  gradient_stager_.reset(new NPUGradientStager(runtime_.get()));
  histogram_constructor_.reset(new NPUHistogramConstructor(runtime_.get()));
  CacheBins();
}

void NPUTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
                                            bool is_constant_hessian,
                                            bool reset_multi_val_bin) {
  ValidateConfig(config_);
  ValidateDataset(train_data, config_);
  if (config_->use_quantized_grad)
    Log::Fatal("NPU quantized probe does not support training-data reset or subset-dataset bagging");
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian,
                                            reset_multi_val_bin);
  CacheBins();
}

void NPUTreeLearner::ResetConfig(const Config* config) {
  ValidateConfig(config);
  if (config->use_quantized_grad != config_->use_quantized_grad)
    Log::Fatal("NPU ResetConfig cannot change use_quantized_grad");
  if (config_->use_quantized_grad &&
      (config->num_leaves != config_->num_leaves ||
       config->num_grad_quant_bins != config_->num_grad_quant_bins ||
       config->stochastic_rounding != config_->stochastic_rounding ||
       config->seed != config_->seed || config->num_iterations != config_->num_iterations ||
       config->num_threads != config_->num_threads))
    Log::Fatal("NPU quantized probe cannot reset discretizer dimensions, seed, threads or rounding");
  if (train_data_ != nullptr) {
    ValidateDataset(train_data_, config);
  }
  SerialTreeLearner::ResetConfig(config);
  ResetExtraTreeRandom();
}

void NPUTreeLearner::ResetExtraTreeRandom() {
  // Init uses SetFeatureInfo<true, true>; ResetConfig uses <false, true>.
  // Both reseed unconditionally. ResetTrainingDataInner and BeforeTrain do not.
  extra_tree_random_.clear();
  extra_tree_random_.reserve(num_features_);
  for (int feature = 0; feature < num_features_; ++feature) {
    extra_tree_random_.emplace_back(config_->extra_seed + feature);
  }
}

// Bin values are read once through the public iterator API so this works
// regardless of how features were bundled or sparsely encoded internally.
void NPUTreeLearner::CacheBins() {
  const data_size_t num_data = train_data_->num_data();
  bins_.assign(num_features_, std::vector<uint32_t>());
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(dynamic)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    std::vector<uint32_t> column(num_data, 0);
    std::unique_ptr<BinIterator> iterator(train_data_->FeatureIterator(feature_index));
    iterator->Reset(0);
    for (data_size_t i = 0; i < num_data; ++i) {
      column[i] = iterator->Get(i);
    }
    bins_[feature_index] = std::move(column);
  }
  std::vector<int> offsets(num_features_), num_bins(num_features_), most_frequent(num_features_);
  for (int feature = 0; feature < num_features_; ++feature) {
    most_frequent[feature] = static_cast<int>(train_data_->FeatureBinMapper(feature)->GetMostFreqBin());
    offsets[feature] = most_frequent[feature] == 0 ? 1 : 0;
    num_bins[feature] = train_data_->FeatureNumBin(feature);
  }
  device_histograms_.clear();
  quantized_histograms_.clear();
  host_rows_current_ = true;
  device_rows_current_ = false;
  runtime_->SetHistogramBins(bins_, offsets, num_bins, most_frequent);
  // Device bins are now the training copy; release the decoded host matrix.
  bins_.clear();
  bins_.shrink_to_fit();
  Log::Info("NPU tree learner cached device bins for %d features x %d rows",
            num_features_, static_cast<int>(num_data));
}

void NPUTreeLearner::BeforeTrain() {
  SerialTreeLearner::BeforeTrain();
  data_partition_npu_->Reset(config_->num_leaves);
  data_size_t count = 0;
  const data_size_t* root = data_partition_->GetIndexOnLeaf(0, &count);
  data_partition_npu_->InitializeRoot(root, count);
  host_rows_current_ = true;
  device_rows_current_ = true;
  tree_rows_downloaded_before_ = runtime_->partition_rows_downloaded();
  if (config_->use_quantized_grad) {
    gradient_stager_->StageQuantized(gradient_discretizer_->discretized_gradients_and_hessians(), num_data_);
    const int64_t packed = smaller_leaf_splits_->int_sum_gradients_and_hessians();
    smaller_leaf_splits_->InitQuantizedFromDevice(0, count, smaller_leaf_splits_->sum_gradients(),
                                                  smaller_leaf_splits_->sum_hessians(), packed, 0.0);
  } else {
    smaller_leaf_splits_->InitFromDevice(0, count, smaller_leaf_splits_->sum_gradients(),
                                         smaller_leaf_splits_->sum_hessians(), 0.0);
  }
}

void NPUTreeLearner::InitLeafSplits(LeafSplits* splits, int leaf, double sum_gradient,
                                     double sum_hessian, double output) {
  const auto& rows = data_partition_npu_->Leaf(leaf);
  CHECK_EQ(rows.count, data_partition_->leaf_count(leaf));
  splits->InitFromDevice(leaf, rows.count, sum_gradient, sum_hessian, output);
}

void NPUTreeLearner::InitQuantizedLeafSplits(LeafSplits* splits, int leaf, double sum_gradient,
                                             double sum_hessian, int64_t packed, double output) {
  const auto& rows = data_partition_npu_->Leaf(leaf);
  CHECK_EQ(rows.count, data_partition_->leaf_count(leaf));
  splits->InitQuantizedFromDevice(leaf, rows.count, sum_gradient, sum_hessian, packed, output);
}

void NPUTreeLearner::RenewQuantizedTreeOutput(Tree* tree) {
  // Explicit compatibility path until raw-gradient leaf reductions reside on
  // device. Never let Serial consume stale host partition rows silently.
  MaterializeHostRows(tree->num_leaves());
  Log::Info("NPU quantized leaf renewal: explicit CPU raw-gradient reduction");
  ++quantized_leaf_renewals_;
  SerialTreeLearner::RenewQuantizedTreeOutput(tree);
}

void NPUTreeLearner::PartitionData(int leaf, int feature,
                                   const uint32_t* threshold, int num_threshold,
                                   bool default_left, int right_leaf) {
  CHECK(threshold != nullptr);
  const auto* mapper = train_data_->FeatureBinMapper(feature);
  const auto& parent = data_partition_npu_->Leaf(leaf);
  CHECK(parent.partition != nullptr);
  const data_size_t parent_count = parent.count;
  int64_t left_count = 0;
  if (mapper->bin_type() == BinType::NumericalBin) {
    CHECK_EQ(num_threshold, 1);
    left_count = data_partition_npu_->Split(
        leaf, feature, *threshold, mapper->GetDefaultBin(),
        static_cast<int>(mapper->missing_type()), default_left, right_leaf);
  } else {
    std::vector<uint32_t> categories;
    categories.reserve(static_cast<size_t>(num_threshold) * 32);
    for (int word = 0; word < num_threshold; ++word) {
      for (int bit = 0; bit < 32; ++bit) {
        if ((threshold[word] & (static_cast<uint32_t>(1) << bit)) != 0) {
          categories.push_back(static_cast<uint32_t>(word * 32 + bit));
        }
      }
    }
    CHECK(!categories.empty());
    left_count = data_partition_npu_->SplitCategorical(
        leaf, feature, categories, mapper->GetDefaultBin(),
        static_cast<int>(mapper->missing_type()), default_left, right_leaf);
  }
  // Serial's LeafSplits and tree bookkeeping need counts during training,
  // but the NPU histogram/split path does not consume their host row pointers.
  data_partition_->SetSplitCounts(leaf, right_leaf, parent_count,
                                  static_cast<data_size_t>(left_count));
  host_rows_current_ = false;
}

int NPUTreeLearner::SelectBestLeaf() const {
  if (best_split_per_leaf_.empty()) {
    return 0;
  }
  // SplitInfo compares gain first and the smaller feature index on ties.
  // Sort the candidate indices by that secondary key so device ArgMax's
  // first-equal rule has the same deterministic result as SerialTreeLearner.
  std::vector<int> order(best_split_per_leaf_.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [this](int left, int right) {
    const int left_feature = best_split_per_leaf_[left].feature < 0
                                 ? INT_MAX : best_split_per_leaf_[left].feature;
    const int right_feature = best_split_per_leaf_[right].feature < 0
                                  ? INT_MAX : best_split_per_leaf_[right].feature;
    return left_feature < right_feature;
  });
  std::vector<double> gains;
  gains.reserve(order.size());
  for (const int leaf : order) {
    gains.push_back(best_split_per_leaf_[leaf].gain);
  }
  const int selected = static_cast<int>(runtime_->ArgMax(gains));
  CHECK_GE(selected, 0);
  CHECK_LT(selected, static_cast<int>(order.size()));
  const int best_leaf = order[selected];
  Log::Info("NPU cross-leaf selection: leaf=%d feature=%d gain=%.17g",
            best_leaf, best_split_per_leaf_[best_leaf].feature,
            best_split_per_leaf_[best_leaf].gain);
  return best_leaf;
}

Tree* NPUTreeLearner::Train(const score_t* gradients, const score_t* hessians,
                            bool is_first_tree) {
  ValidateConfig(config_);
  // Also catch a caller mutating an already installed JSON object.
  ValidateForcedSplit(forced_split_json_);
  forced_split_json_ = nullptr;
  const uint64_t split_calls_before = runtime_->split_calls();
  if (is_first_tree) {
    // The caller owns the initial score values. Upload them once, then keep
    // the device copy across subsequent trees in this model.
    runtime_->ResetScoreBuffers();
  }
  device_histograms_.clear();
  quantized_histograms_.clear();
  if (!config_->use_quantized_grad) {
    gradient_stager_->Stage(gradients, hessians, train_data_->num_data(),
                            share_state_->is_constant_hessian);
  } else {
    // The shared CPU discretizer executes inside Serial::Train. Stage its
    // resulting bytes in BeforeTrain, not the preceding iteration's bytes.
    bool positive_hessian = false;
    for (data_size_t i = 0; i < num_data_; ++i) {
      CHECK(std::isfinite(gradients[i]));
      CHECK(std::isfinite(hessians[i]) && hessians[i] >= 0.0f);
      positive_hessian |= hessians[i] > 0.0f;
    }
    CHECK(positive_hessian);
    Log::Info("NPU quantized staging: CPU gradient discretization, device INT64 histogram/search");
  }
  Tree* tree = SerialTreeLearner::Train(gradients, hessians, is_first_tree);
  // Keep the final row indices on device. Legacy consumers request a host
  // mirror explicitly; normal score updates use resident leaf spans directly.
  Log::Info("NPU split search: split_calls=%llu (+%llu this tree), leaves=%d; "
            "runtime winners populate best_split_per_leaf_ used by tree splits",
            static_cast<unsigned long long>(runtime_->split_calls()),
            static_cast<unsigned long long>(runtime_->split_calls() - split_calls_before),
            tree->num_leaves());
  return tree;
}

void NPUTreeLearner::MaterializeHostRows(int num_leaves) const {
  if (host_rows_current_) return;
  CHECK(device_rows_current_);
  CHECK_LE(num_leaves, config_->num_leaves);
  for (int leaf = 0; leaf < num_leaves; ++leaf) {
    const auto& device_leaf = data_partition_npu_->Leaf(leaf);
    CHECK(device_leaf.partition != nullptr);
    CHECK_EQ(device_leaf.count, data_partition_->leaf_count(leaf));
    auto rows = data_partition_npu_->ReadLeafRows(leaf);
    CHECK_EQ(rows.size(), static_cast<size_t>(device_leaf.count));
    data_partition_->SetLeafIndices(leaf, rows.data(), device_leaf.count);
  }
  host_rows_current_ = true;
  Log::Info("NPU compatibility: materialized host rows for leaf renewal/refit");
}

void NPUTreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  if (tree->num_leaves() <= 1) return;  // Match the serial interface.
  if (!device_rows_current_) {
    CHECK(host_rows_current_);  // Explicitly refitted CPU partition.
    SerialTreeLearner::AddPredictionToScore(tree, out_score);
    return;
  }
  CHECK_LE(tree->num_leaves(), config_->num_leaves);
  std::vector<NPURuntime::ScoreLeaf> leaves;
  for (int leaf = 0; leaf < tree->num_leaves(); ++leaf) {
    const auto& device_leaf = data_partition_npu_->Leaf(leaf);
    CHECK_EQ(device_leaf.count, data_partition_->leaf_count(leaf));
    leaves.push_back({device_leaf.partition, device_leaf.left, tree->LeafOutput(leaf)});
  }
  score_updater_->Update(leaves, num_data_, out_score);
  Log::Info("NPU score update: device_calls=%llu, partition_rows_downloaded=%llu this tree",
            static_cast<unsigned long long>(score_updater_->update_calls()),
            static_cast<unsigned long long>(runtime_->partition_rows_downloaded() - tree_rows_downloaded_before_));
}

void NPUTreeLearner::RenewTreeOutput(
    Tree* tree, const ObjectiveFunction* obj,
    std::function<double(const label_t*, int)> residual_getter,
    data_size_t total_num_data, const data_size_t* bag_indices,
    data_size_t bag_cnt, const double* train_score) const {
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    MaterializeHostRows(tree->num_leaves());
  }
  SerialTreeLearner::RenewTreeOutput(tree, obj, residual_getter, total_num_data,
                                     bag_indices, bag_cnt, train_score);
}

Tree* NPUTreeLearner::FitByExistingTree(const Tree* old_tree, const score_t* gradients,
                                       const score_t* hessians) const {
  MaterializeHostRows(old_tree->num_leaves());
  return SerialTreeLearner::FitByExistingTree(old_tree, gradients, hessians);
}

Tree* NPUTreeLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                       const score_t* gradients, const score_t* hessians) const {
  data_partition_->ResetByLeafPred(leaf_pred, old_tree->num_leaves());
  host_rows_current_ = true;
  device_rows_current_ = false;
  return SerialTreeLearner::FitByExistingTree(old_tree, gradients, hessians);
}

void NPUTreeLearner::FindBestSplitsFromHistograms(
    const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree) {
  const auto smaller_node_used_features =
      col_sampler_.GetByNode(tree, smaller_leaf_splits_->leaf_index());
  const bool has_larger = larger_leaf_splits_ != nullptr &&
                          larger_leaf_splits_->leaf_index() >= 0;
  std::vector<int8_t> larger_node_used_features;
  if (has_larger) {
    larger_node_used_features = col_sampler_.GetByNode(tree, larger_leaf_splits_->leaf_index());
  }

  // Match Serial's ordering while keeping histogram values on the device.
  if (config_->use_quantized_grad) {
    auto fix = [&](FeatureHistogram* histograms, const LeafSplits* leaf) {
      const int64_t packed = leaf->int_sum_gradients_and_hessians();
      histogram_constructor_->FixQuantized(quantized_histograms_.at(histograms),
                                            PackedGradient(packed), PackedHessian(packed));
    };
    fix(smaller_leaf_histogram_array_, smaller_leaf_splits_.get());
    FindBestSplitForLeaf(is_feature_used, smaller_node_used_features,
                         smaller_leaf_splits_.get(), smaller_leaf_histogram_array_,
                         GetParentOutput(tree, smaller_leaf_splits_.get()));
    if (has_larger) {
      if (use_subtract) {
        histogram_constructor_->SubtractQuantized(quantized_histograms_.at(larger_leaf_histogram_array_),
                                                   quantized_histograms_.at(smaller_leaf_histogram_array_));
        ++quantized_histogram_subtractions_;
      } else {
        fix(larger_leaf_histogram_array_, larger_leaf_splits_.get());
      }
      FindBestSplitForLeaf(is_feature_used, larger_node_used_features,
                           larger_leaf_splits_.get(), larger_leaf_histogram_array_,
                           GetParentOutput(tree, larger_leaf_splits_.get()));
    }
    return;
  }
  histogram_constructor_->Fix(device_histograms_.at(smaller_leaf_histogram_array_),
                               smaller_leaf_splits_->sum_gradients(),
                               smaller_leaf_splits_->sum_hessians());
  FindBestSplitForLeaf(is_feature_used, smaller_node_used_features,
                       smaller_leaf_splits_.get(), smaller_leaf_histogram_array_,
                       GetParentOutput(tree, smaller_leaf_splits_.get()));

  if (has_larger) {
    if (use_subtract) {
      histogram_constructor_->Subtract(device_histograms_.at(larger_leaf_histogram_array_),
                                       device_histograms_.at(smaller_leaf_histogram_array_));
    } else {
      histogram_constructor_->Fix(device_histograms_.at(larger_leaf_histogram_array_),
                                   larger_leaf_splits_->sum_gradients(),
                                   larger_leaf_splits_->sum_hessians());
    }
    FindBestSplitForLeaf(is_feature_used, larger_node_used_features,
                         larger_leaf_splits_.get(), larger_leaf_histogram_array_,
                         GetParentOutput(tree, larger_leaf_splits_.get()));
  }
}

void NPUTreeLearner::FindBestSplitForLeaf(
    const std::vector<int8_t>& is_feature_used,
    const std::vector<int8_t>& node_used_features, const LeafSplits* leaf,
    FeatureHistogram* histograms, double parent_output) {
  std::vector<NPURuntime::SplitFeature> features;
  std::vector<int> feature_indices;
  features.reserve(num_features_);
  feature_indices.reserve(num_features_);
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used[feature_index]) {
      continue;
    }
    const auto* mapper = train_data_->FeatureBinMapper(feature_index);
    NPURuntime::SplitFeature feature;
    feature.feature = train_data_->RealFeatureIndex(feature_index);
    // Same sources as HistogramPool::SetFeatureInfo<true, true>; FeatureHistogram
    // has no public metadata accessor in this version. feature_contri uses the
    // original feature index, not the bundled/internal histogram index.
    feature.num_bin = train_data_->FeatureNumBin(feature_index);
    feature.offset = mapper->GetMostFreqBin() == 0 ? 1 : 0;
    feature.default_bin = static_cast<int>(mapper->GetDefaultBin());
    feature.missing_type = static_cast<int>(mapper->missing_type());
    feature.categorical = mapper->bin_type() != BinType::NumericalBin;
    feature.max_cat_threshold = config_->max_cat_threshold;
    feature.max_cat_to_onehot = config_->max_cat_to_onehot;
    feature.min_data_per_group = config_->min_data_per_group;
    feature.cat_smooth = config_->cat_smooth;
    feature.cat_l2 = config_->cat_l2;
    feature.penalty = config_->feature_contri.empty()
                          ? 1.0 : config_->feature_contri[feature.feature];
    feature.enabled = node_used_features[feature_index] != 0;
    feature.extra_tree_random = config_->extra_trees ? &extra_tree_random_[feature_index] : nullptr;
    feature.histogram_index = feature_index;
    feature_indices.push_back(feature_index);
    features.push_back(std::move(feature));
  }

  const NPUBestSplitFinder::Parameters split_parameters{
      config_->lambda_l1,
      config_->lambda_l2,
      config_->min_data_in_leaf,
      config_->min_sum_hessian_in_leaf,
      config_->min_gain_to_split,
      config_->max_delta_step,
      config_->path_smooth,
      parent_output};
  NPURuntime::SplitResult result;
  NPURuntime::QuantizedSplitCandidate integer_statistics;
  if (config_->use_quantized_grad) {
    const int64_t packed = leaf->int_sum_gradients_and_hessians();
    // No split can satisfy min_data when the integer Hessian total is zero.
    if (PackedHessian(packed) == 0) {
      result.splittable.assign(features.size(), 0);
    } else {
      NPURuntime::QuantizedSplitParameters parameters;
      parameters.gradient_scale = gradient_discretizer_->grad_scale();
      parameters.hessian_scale = gradient_discretizer_->hess_scale();
      parameters.lambda_l1 = split_parameters.lambda_l1;
      parameters.lambda_l2 = split_parameters.lambda_l2;
      parameters.min_data_in_leaf = split_parameters.min_data_in_leaf;
      parameters.min_sum_hessian_in_leaf = split_parameters.min_sum_hessian_in_leaf;
      parameters.min_gain_to_split = split_parameters.min_gain_to_split;
      parameters.max_delta_step = split_parameters.max_delta_step;
      parameters.path_smooth = split_parameters.path_smooth;
      parameters.parent_output = parent_output;
      const auto selected = best_split_finder_->FindQuantized(
          features, PackedGradient(packed), PackedHessian(packed), leaf->num_data_in_leaf(),
          parameters, quantized_histograms_.at(histograms));
      result = selected.split;
      integer_statistics = selected.integer_statistics;
    }
  } else {
    result = best_split_finder_->Find(
        features, leaf->sum_gradients(), leaf->sum_hessians(), leaf->num_data_in_leaf(),
        split_parameters, device_histograms_.at(histograms));
  }
  CHECK_EQ(result.splittable.size(), features.size());
  for (size_t i = 0; i < features.size(); ++i) {
    // Disabled-at-node features must still be searched for splittability, just
    // as Serial's ComputeBestSplitForFeature filters by-node use only at the end.
    // Features excluded by is_feature_used retain Serial's inherited state.
    histograms[feature_indices[i]].set_is_splittable(result.splittable[i] != 0);
  }

  SplitInfo best;  // Includes feature=-1, gain=kMinScore for no valid candidate.
  if (result.feature >= 0) {
    const int inner_feature_index = train_data_->InnerFeatureIndex(result.feature);
    CHECK_GE(inner_feature_index, 0);
    CHECK_LT(inner_feature_index, num_features_);
    CHECK(is_feature_used[inner_feature_index]);
    CHECK(node_used_features[inner_feature_index]);
    CHECK(histograms[inner_feature_index].is_splittable());
    CHECK(std::isfinite(result.gain));
    best.feature = result.feature;
    best.threshold = result.threshold;
    best.default_left = result.default_left;
    best.gain = result.gain;
    if (result.categorical) {
      best.cat_threshold = result.cat_threshold;
      best.num_cat_threshold = static_cast<int>(result.cat_threshold.size());
    }
    best.left_output = result.left_output;
    best.right_output = result.right_output;
    best.left_sum_gradient = result.left_gradient;
    best.right_sum_gradient = result.right_gradient;
    // Runtime has already removed epsilon, shifted gain by parent/min_gain,
    // and applied feature_contri. Do not adjust these values again here.
    best.left_sum_hessian = result.left_hessian;
    best.right_sum_hessian = result.right_hessian;
    best.left_count = result.left_count;
    best.right_count = result.right_count;
    if (config_->use_quantized_grad) {
      best.left_sum_gradient_and_hessian = PackNPUStatistics(integer_statistics.left_gradient,
                                                             integer_statistics.left_hessian);
      best.right_sum_gradient_and_hessian = PackNPUStatistics(integer_statistics.right_gradient,
                                                              integer_statistics.right_hessian);
    }
  }
  // Serial::Train selects from this array, and Serial::SplitInner uses this
  // exact threshold, direction, outputs and child statistics to build the tree.
  best_split_per_leaf_[leaf->leaf_index()] = best;
}

void NPUTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                                         bool use_subtract) {
  ConstructLeaf(is_feature_used, smaller_leaf_splits_.get(),
                smaller_leaf_histogram_array_);
  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    ConstructLeaf(is_feature_used, larger_leaf_splits_.get(),
                  larger_leaf_histogram_array_);
  }
}

// The CPU partition holds leaf counts/layout; device rows supply histograms.
void NPUTreeLearner::ConstructLeaf(const std::vector<int8_t>& used,
                                   const LeafSplits* leaf,
                                   FeatureHistogram* histograms) {
  const data_size_t count = leaf->num_data_in_leaf();
  CHECK(leaf->data_indices() == nullptr);
  const auto& device_leaf = data_partition_npu_->Leaf(leaf->leaf_index());
  CHECK_EQ(device_leaf.count, count);
  CHECK(device_leaf.partition != nullptr);
  if (config_->use_quantized_grad) {
    quantized_histograms_[histograms] = histogram_constructor_->BuildQuantizedFromDeviceRows(
        runtime_->PartitionRowsDevice(device_leaf.partition, device_leaf.left), count, used);
  } else {
    device_histograms_[histograms] = histogram_constructor_->BuildFromDeviceRows(
        runtime_->PartitionRowsDevice(device_leaf.partition, device_leaf.left), count, used);
  }
}

}  // namespace LightGBM