/* Copyright (c) 2026. Licensed under the MIT License. */
// Internal lifecycle validation only. The public factory/config still reject
// quantized NPU training; this friend fixture is the sole private constructor user.
#include "npu_tree_learner.h"
#include "../../src/boosting/gbdt.h"
#include <LightGBM/metric.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/c_api.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace LightGBM {
class NPUQuantizedTreeProbe {
 public:
  static std::unique_ptr<NPUTreeLearner> Create(const Config* config) {
    return std::unique_ptr<NPUTreeLearner>(new NPUTreeLearner(config, true));
  }
  static uint64_t Downloads(const NPUTreeLearner& learner) {
    return learner.runtime_->partition_rows_downloaded();
  }
  static uint64_t Subtractions(const NPUTreeLearner& learner) {
    return learner.quantized_histogram_subtractions_;
  }
  static uint64_t Renewals(const NPUTreeLearner& learner) {
    return learner.quantized_leaf_renewals_;
  }
  static void CheckResident(const NPUTreeLearner& learner) {
    if (learner.smaller_leaf_splits_->data_indices() != nullptr ||
        (learner.larger_leaf_splits_->leaf_index() >= 0 && learner.larger_leaf_splits_->data_indices() != nullptr))
      throw std::runtime_error("quantized leaves retained stale host row pointers");
  }
};
}  // namespace LightGBM

namespace {
using namespace LightGBM;
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Action> void Reject(Action action, const std::string& message) {
  bool caught = false;
  try { action(); } catch (const std::exception&) { caught = true; }
  Require(caught, message);
}
void Near(double actual, double expected, const std::string& field) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > 1e-10 + 1e-11 * std::fabs(expected)) {
    std::cerr.precision(17);
    std::cerr << field << " actual=" << actual << " expected=" << expected << '\n';
    throw std::runtime_error("quantized tree numerical mismatch: " + field);
  }
}
void CompareJson(const Json& actual, const Json& expected, const std::string& path) {
  Require(actual.type() == expected.type(), path + ": JSON type");
  if (actual.is_number()) Near(actual.number_value(), expected.number_value(), path);
  else if (actual.is_array()) {
    Require(actual.array_items().size() == expected.array_items().size(), path + ": array size");
    for (size_t i = 0; i < actual.array_items().size(); ++i)
      CompareJson(actual[i], expected[i], path + "/" + std::to_string(i));
  } else if (actual.is_object()) {
    Require(actual.object_items().size() == expected.object_items().size(), path + ": object size");
    for (const auto& item : expected.object_items()) {
      Require(actual.object_items().count(item.first) == 1, path + ": missing key");
      CompareJson(actual[item.first], item.second, path + "/" + item.first);
    }
  } else Require(actual == expected, path + ": value mismatch");
}
struct Data {
  DatasetHandle handle = nullptr;
  std::vector<double> values;
  std::vector<float> labels;
  int rows = 192, columns = 5;
  explicit Data(bool zero_missing, bool categorical = false, int n = 192,
                DatasetHandle reference = nullptr, int row_offset = 0) : rows(n) {
    values.resize(rows * columns); labels.resize(rows);
    for (int i = 0; i < rows; ++i) {
      for (int f = 0; f < columns; ++f) {
        const int row = i + row_offset;
        double value = static_cast<double>((row * (13 + 2 * f) + row / (f + 2)) % 47 - 23) / 9.0;
        if ((i + 3 * f) % 11 == 0) value = 0.0;
        if (!zero_missing && (i + 7 * f) % 29 == 0) value = std::nan("");
        if (categorical && f == 0) value = i % 4;
        values[i * columns + f] = value;
      }
      const auto value = [&](int f) { double x = values[i * columns + f]; return std::isnan(x) ? 0.0 : x; };
      labels[i] = static_cast<float>(1.8 * value(0) - 0.7 * value(1) + 0.3 * value(3) + (i % 5) * 0.04);
    }
    std::string parameters = "max_bin=15 min_data_in_bin=1 min_data_in_leaf=2 feature_pre_filter=false num_threads=2";
    if (zero_missing) parameters += " zero_as_missing=true";
    if (categorical) parameters += " categorical_feature=0";
    Require(LGBM_DatasetCreateFromMat(values.data(), C_API_DTYPE_FLOAT64, rows, columns, 1,
                                    parameters.c_str(), reference, &handle) == 0, LGBM_GetLastError());
    Require(LGBM_DatasetSetField(handle, "label", labels.data(), rows, C_API_DTYPE_FLOAT32) == 0, LGBM_GetLastError());
  }
  ~Data() { if (handle) LGBM_DatasetFree(handle); }
  const Dataset* dataset() const { return static_cast<const Dataset*>(handle); }
};
void RootStatisticsOracle() {
  Config config; config.deterministic = true;
  // Deliberately unsigned Hessian bytes, negative gradients, and a reordered
  // non-prefix bag. Independent scalar membership sums, no production packing.
  const int g[] = {1, -2, 3, -4, 5};
  const uint8_t h[] = {1, 255, 3, 128, 5};
  std::vector<int8_t> pairs(10);
  for (int i = 0; i < 5; ++i) {
    pairs[2 * i + 1] = static_cast<int8_t>(g[i]);
    std::memcpy(pairs.data() + 2 * i, h + i, 1);
  }
  for (const auto& rows : std::vector<std::vector<data_size_t>>{{0, 1, 2, 3, 4}, {4, 1, 3}, {3, 1}, {}}) {
    DataPartition partition(5, 3);
    const data_size_t empty_sentinel = 0;
    partition.SetUsedDataIndices(rows.empty() ? &empty_sentinel : rows.data(), rows.size()); partition.Init();
    LeafSplits leaf(5, &config);
    leaf.Init(0, &partition, pairs.data(), 0.5f, 0.25f);
    int64_t sum_g = 0, sum_h = 0;
    for (int row : rows) { sum_g += g[row]; sum_h += h[row]; }
    Require(leaf.int_sum_gradients_and_hessians() == sum_g * int64_t{4294967296} + sum_h,
            "bagged packed root must use selected row membership");
    Near(leaf.sum_gradients(), sum_g * 0.5, "bagged root G");
    Near(leaf.sum_hessians(), sum_h * 0.25, "unsigned bagged root H");
  }
  LeafSplits full(5, &config); full.Init(pairs.data(), 0.5, 0.25);
  Near(full.sum_hessians(), 98.0, "full unsigned Hessian");
  Require(full.int_sum_gradients_and_hessians() == int64_t{3} * 4294967296 + 392, "full packed root");
  std::cout << "QUANTIZED_ROOT_STATISTICS_PASS bags=4 unsigned_hessian=true\n";
}
void HistogramPackingOracle() {
  Config config; config.use_quantized_grad = true;
  FeatureMetainfo meta;
  meta.config = &config; meta.num_bin = 5; meta.offset = 0;
  meta.bin_type = BinType::NumericalBin; meta.missing_type = MissingType::None; meta.default_bin = 0;
  const int gradients[] = {-32768, -1, 0, 1, 32767};
  const int hessians[] = {65535, 17, 0, 1, 32768};
  std::vector<int32_t> compact(5), result(5);
  std::vector<hist_t> parent_storage(10), other_storage(10), result_storage(10);
  std::vector<int64_t> expanded(5), parent(5);
  for (int i = 0; i < 5; ++i) {
    compact[i] = gradients[i] * int32_t{65536} + hessians[i];
    parent[i] = (gradients[i] - int64_t{7}) * int64_t{4294967296} + hessians[i] + 17;
  }
  std::memcpy(parent_storage.data(), parent.data(), parent.size() * sizeof(int64_t));
  FeatureHistogram original, other, narrowed;
  original.Init(parent_storage.data(), reinterpret_cast<int16_t*>(result.data()), &meta);
  other.Init(other_storage.data(), reinterpret_cast<int16_t*>(compact.data()), &meta);
  narrowed.Init(result_storage.data(), reinterpret_cast<int16_t*>(result.data()), &meta);
  other.CopyFromInt16ToInt32(reinterpret_cast<char*>(expanded.data()));
  for (int i = 0; i < 5; ++i)
    Require(expanded[i] == gradients[i] * int64_t{4294967296} + hessians[i], "signed histogram widening");
  // A negative 16-bit gradient must also widen correctly during subtraction.
  std::vector<int32_t> sibling(5, -7 * int32_t{65536} + 17);
  other.Init(other_storage.data(), reinterpret_cast<int16_t*>(sibling.data()), &meta);
  original.Subtract<true, int64_t, int32_t, int64_t, 32, 16, 32>(other);
  std::vector<int64_t> difference(5);
  std::memcpy(difference.data(), parent_storage.data(), difference.size() * sizeof(int64_t));
  Require(difference == expanded, "signed mixed-width histogram subtraction");
  std::memcpy(result_storage.data(), parent.data(), parent.size() * sizeof(int64_t));
  narrowed.Subtract<true, int64_t, int32_t, int32_t, 32, 16, 16>(
      other, reinterpret_cast<const int32_t*>(parent.data()));
  Require(result == compact, "signed histogram narrowing after subtraction");
  std::cout << "QUANTIZED_HISTOGRAM_PACKING_PASS cases=5 signed_boundaries=true\n";
}
void ZeroScaleOracle() {
  Data data(false, false, 32);
  std::vector<score_t> gradients(data.rows, 0.0f), hessians(data.rows, 1.0f);
  int checked = 0;
  for (bool stochastic : {false, true}) for (bool constant : {false, true}) {
    GradientDiscretizer discretizer(16, 3, 37, constant, stochastic);
    discretizer.Init(data.rows, 3, data.dataset()->num_features(), data.dataset());
    for (bool zero_hessian : {false, true}) {
      std::fill(hessians.begin(), hessians.end(), zero_hessian ? 0.0f : 1.0f);
      discretizer.DiscretizeGradients(data.rows, gradients.data(), hessians.data());
      Require(discretizer.grad_scale() == 0.0, "zero-gradient scale");
      Require(discretizer.hess_scale() == (zero_hessian ? 0.0 : (constant ? 1.0 : 0.0625)), "zero-Hessian scale");
      const int8_t* pairs = discretizer.discretized_gradients_and_hessians();
      for (int i = 0; i < data.rows; ++i) {
        Require(pairs[2 * i + 1] == 0, "zero-gradient byte");
        Require(pairs[2 * i] == (constant ? 1 : (zero_hessian ? 0 : 16)), "zero-Hessian byte");
      }
      ++checked;
    }
  }
  std::cout << "QUANTIZED_ZERO_SCALE_PASS cases=" << checked << '\n';
}
void RejectionOracle() {
  Config config;
  config.use_quantized_grad = true;
  for (int change = 0; change < 4; ++change) {
    auto invalid = config;
    if (change == 0) invalid.num_grad_quant_bins = 1;
    if (change == 1) invalid.num_grad_quant_bins = 128;
    if (change == 2) invalid.min_data_in_leaf = 0;
    if (change == 3) invalid.extra_trees = true;
    Reject([&]() { NPUQuantizedTreeProbe::Create(&invalid); }, "unsafe quantized configuration accepted");
  }
  Data categorical(false, true);
  auto npu = NPUQuantizedTreeProbe::Create(&config);
  Reject([&]() { npu->Init(categorical.dataset(), true); }, "quantized categorical dataset accepted");
  std::cout << "QUANTIZED_SUPPORT_REJECTION_PASS cases=5\n";
}
struct Scenario {
  std::string name;
  bool constant = true, bagging = false, renewal = false, stochastic = false, zero_missing = false;
  bool zero_gradient = false;
  double smooth = 0.0, clip = 0.0, pool = -1.0, bynode = 1.0;
};
void Run(const Scenario& scenario) {
  Data data(scenario.zero_missing);
  Config config;
  config.Set({{"objective", "regression"}, {"num_threads", "2"}, {"verbosity", "-1"},
              {"num_leaves", "7"}, {"min_data_in_leaf", "5"}, {"max_bin", "15"},
              {"use_quantized_grad", "true"}, {"num_grad_quant_bins", "16"},
              {"stochastic_rounding", scenario.stochastic ? "true" : "false"},
              {"quant_train_renew_leaf", scenario.renewal ? "true" : "false"},
              {"force_col_wise", "true"}, {"deterministic", "true"}, {"num_iterations", "6"},
              {"feature_fraction", "0.8"}, {"feature_fraction_bynode", std::to_string(scenario.bynode)},
              {"lambda_l1", "0.15"}, {"lambda_l2", "0.7"},
              {"path_smooth", std::to_string(scenario.smooth)}, {"max_delta_step", std::to_string(scenario.clip)},
              {"histogram_pool_size", std::to_string(scenario.pool)}, {"seed", "37"}});
  SerialTreeLearner cpu(&config);
  cpu.SetForcedSplit(nullptr);
  auto npu = NPUQuantizedTreeProbe::Create(&config);
  cpu.Init(data.dataset(), scenario.constant); npu->Init(data.dataset(), scenario.constant);
  if (scenario.name == "constant") {
    for (int change = 0; change < 8; ++change) {
      Config invalid = config;
      if (change == 0) invalid.num_leaves += 4;
      if (change == 1) invalid.num_grad_quant_bins += 2;
      if (change == 2) invalid.stochastic_rounding = !invalid.stochastic_rounding;
      if (change == 3) invalid.seed += 1;
      if (change == 4) invalid.num_iterations += 1;
      if (change == 5) invalid.num_threads += 1;
      if (change == 6) invalid.use_quantized_grad = false;
      if (change == 7) invalid.extra_trees = true;
      Reject([&]() { npu->ResetConfig(&invalid); }, "unsupported quantized config reset accepted");
    }
    Reject([&]() { npu->ResetTrainingDataInner(data.dataset(), false, true); }, "quantized training-data reset accepted");
    Reject([&]() { npu->SetBaggingData(data.dataset(), nullptr, data.rows); }, "quantized subset-dataset bagging accepted");
    std::cout << "QUANTIZED_RESET_REJECTION_PASS cases=10 original_state_retained=true\n";
  }
  std::vector<double> cpu_score(data.rows, 0.0), npu_score(data.rows, 0.0);
  std::vector<score_t> cg(data.rows), ng(data.rows), h(data.rows);
  std::vector<data_size_t> bag;
  int grown = 0;
  for (int round = 0; round < 3; ++round) {
    bag.clear();
    for (int i = 0; i < data.rows; ++i) if (!scenario.bagging || (i + round) % 4 != 0) bag.push_back(i);
    cpu.SetBaggingData(nullptr, bag.data(), bag.size()); npu->SetBaggingData(nullptr, bag.data(), bag.size());
    for (int i = 0; i < data.rows; ++i) {
      h[i] = scenario.constant ? 1.0f : static_cast<float>(0.2 + (i % 7) * 0.13);
      cg[i] = scenario.zero_gradient ? 0.0f : static_cast<float>((cpu_score[i] - data.labels[i]) * h[i]);
      ng[i] = scenario.zero_gradient ? 0.0f : static_cast<float>((npu_score[i] - data.labels[i]) * h[i]);
    }
    const auto before = NPUQuantizedTreeProbe::Downloads(*npu);
    std::unique_ptr<Tree> ct(cpu.Train(cg.data(), h.data(), round == 0));
    std::unique_ptr<Tree> nt(npu->Train(ng.data(), h.data(), round == 0));
    NPUQuantizedTreeProbe::CheckResident(*npu);
    Require(ct->num_leaves() == nt->num_leaves(), scenario.name + ": leaf count");
    if (scenario.zero_gradient) Require(nt->num_leaves() == 1, "zero-gradient tree split");
    grown += nt->num_leaves() > 1;
    std::string error;
    const auto cj = Json::parse("{" + ct->ToJSON() + "}", &error); Require(error.empty(), error);
    const auto nj = Json::parse("{" + nt->ToJSON() + "}", &error); Require(error.empty(), error);
    CompareJson(nj, cj, scenario.name + "/round" + std::to_string(round));
    if (!scenario.renewal) Require(NPUQuantizedTreeProbe::Downloads(*npu) == before, "ordinary quantized training downloaded rows");
    if (nt->num_leaves() > 1) {
      ct->Shrinkage(0.2); nt->Shrinkage(0.2);
      cpu.AddPredictionToScore(ct.get(), cpu_score.data()); npu->AddPredictionToScore(nt.get(), npu_score.data());
      for (int row : bag) {
        Require(ct->PredictLeafIndex(data.values.data() + row * data.columns) ==
                nt->PredictLeafIndex(data.values.data() + row * data.columns), "quantized leaf index mismatch");
        Near(nt->Predict(data.values.data() + row * data.columns), ct->Predict(data.values.data() + row * data.columns), "tree prediction");
      }
    }
    for (int i = 0; i < data.rows; ++i) Near(npu_score[i], cpu_score[i], "cached score");
  }
  if (!scenario.zero_gradient) Require(grown == 3, scenario.name + ": vacuous tree growth");
  if (scenario.renewal) Require(NPUQuantizedTreeProbe::Renewals(*npu) == 3, "missing quantized leaf renewal");
  if (!scenario.zero_gradient) Require(NPUQuantizedTreeProbe::Subtractions(*npu) > 0, "missing quantized histogram subtraction");
  std::cout << "QUANTIZED_TREE_SCENARIO_PASS " << scenario.name << " trees=3 subtractions="
            << NPUQuantizedTreeProbe::Subtractions(*npu) << " row_downloads="
            << NPUQuantizedTreeProbe::Downloads(*npu) << '\n';
}
class QuantizedBoosterProbe : public GBDT {
 public:
  void InitNPU(const Config& config, const Dataset* data, const ObjectiveFunction* objective,
               const std::vector<const Metric*>& metrics) {
    // Use production Booster initialization, sampling, objective and score
    // buffers, replacing only its learner BEFORE the first training iteration.
    // This injection lives solely in the test executable; no public opt-in.
    GBDT::Init(&config, data, objective, metrics);
    config_->device_type = "npu";
    auto learner = NPUQuantizedTreeProbe::Create(config_.get());
    learner->Init(train_data_, is_constant_hessian_);
    learner->SetForcedSplit(&forced_splits_json_);
    tree_learner_ = std::move(learner);
  }
  const std::vector<std::unique_ptr<Tree>>& Trees() const { return models_; }
  uint64_t Downloads() const {
    return NPUQuantizedTreeProbe::Downloads(static_cast<const NPUTreeLearner&>(*tree_learner_));
  }
};
struct BoosterScenario {
  std::string name, objective, metric;
  bool bagging = false, stochastic = false, renewal = true, weighted = true;
};
void SetObjectiveData(Data* data, const BoosterScenario& scenario) {
  std::vector<float> weights(data->rows);
  const bool rank = scenario.objective == "lambdarank" || scenario.objective == "rank_xendcg";
  for (int i = 0; i < data->rows; ++i) {
    const double signal = data->labels[i];
    float label = data->labels[i];
    if (scenario.objective == "binary") label = signal > 0.0;
    else if (scenario.objective == "multiclass" || scenario.objective == "multiclassova")
      label = signal < -1.0 ? 0 : (signal > 1.0 ? 2 : 1);
    else if (scenario.objective == "cross_entropy" || scenario.objective == "cross_entropy_lambda")
      label = static_cast<float>(0.05 + 0.9 / (1.0 + std::exp(-signal)));
    else if (scenario.objective == "poisson" || scenario.objective == "gamma" ||
             scenario.objective == "tweedie" || scenario.objective == "mape")
      label = static_cast<float>(std::exp(0.3 + 0.2 * signal));
    if (rank) label = i % 4;
    data->labels[i] = label;
    weights[i] = static_cast<float>(0.7 + (rank ? (i / 8) % 5 : i % 5) * 0.15);
  }
  Require(LGBM_DatasetSetField(data->handle, "label", data->labels.data(), data->rows, C_API_DTYPE_FLOAT32) == 0,
          LGBM_GetLastError());
  if (scenario.weighted)
    Require(LGBM_DatasetSetField(data->handle, "weight", weights.data(), data->rows, C_API_DTYPE_FLOAT32) == 0,
            LGBM_GetLastError());
  if (rank) {
    Require(data->rows % 8 == 0, "ranking group fixture");
    std::vector<int32_t> groups(data->rows / 8, 8);
    Require(LGBM_DatasetSetField(data->handle, "group", groups.data(), groups.size(), C_API_DTYPE_INT32) == 0,
            LGBM_GetLastError());
  }
}
const PredictionEarlyStopInstance kNoPredictionStop{[](const double*, int) { return false; }, 1};
void CompareBooster(QuantizedBoosterProbe* npu, QuantizedBoosterProbe* cpu,
                    const Data& train, const Data& valid, const std::string& label) {
  Require(npu->GetCurrentIteration() == cpu->GetCurrentIteration(), label + ": iteration");
  Require(npu->NumberOfTotalModel() == cpu->NumberOfTotalModel(), label + ": model count");
  for (size_t i = 0; i < cpu->Trees().size(); ++i) {
    std::string error;
    const auto expected = Json::parse("{" + cpu->Trees()[i]->ToJSON() + "}", &error); Require(error.empty(), error);
    const auto actual = Json::parse("{" + npu->Trees()[i]->ToJSON() + "}", &error); Require(error.empty(), error);
    CompareJson(actual, expected, label + "/tree" + std::to_string(i));
  }
  int64_t nc, nn;
  const auto* cs = cpu->GetTrainingScore(&nc); const auto* ns = npu->GetTrainingScore(&nn);
  Require(nc == nn && nc > 0, "training score lengths");
  for (int64_t i = 0; i < nc; ++i) Near(ns[i], cs[i], label + "/raw_training_score");
  cpu->InitPredict(0, -1, false); npu->InitPredict(0, -1, false);
  for (int data_index : {0, 1}) {
    const auto ce = cpu->GetEvalAt(data_index), ne = npu->GetEvalAt(data_index);
    Require(!ce.empty() && ce.size() == ne.size(), "evaluation metric count");
    for (size_t i = 0; i < ce.size(); ++i) Near(ne[i], ce[i], label + "/metric");
    nc = cpu->GetNumPredictAt(data_index); nn = npu->GetNumPredictAt(data_index);
    Require(nc == nn, "cached prediction length");
    std::vector<double> cp(nc), np(nn);
    cpu->GetPredictAt(data_index, cp.data(), &nc); npu->GetPredictAt(data_index, np.data(), &nn);
    for (int64_t i = 0; i < nc; ++i) Near(np[i], cp[i], label + "/converted_cached_prediction");
    const Data& data = data_index == 0 ? train : valid;
    const int classes = static_cast<int>(nc / data.rows);
    std::vector<double> crow(classes), nrow(classes);
    for (int row = 0; row < data.rows; ++row) {
      const double* values = data.values.data() + row * data.columns;
      cpu->Predict(values, crow.data(), &kNoPredictionStop); npu->Predict(values, nrow.data(), &kNoPredictionStop);
      for (int c = 0; c < classes; ++c) {
        Near(crow[c], cp[c * data.rows + row], label + "/CPU_cache_vs_tree");
        Near(nrow[c], np[c * data.rows + row], label + "/NPU_cache_vs_tree");
      }
    }
  }
  cpu->InitPredict(0, -1, false); npu->InitPredict(0, -1, false);
  const int outputs = cpu->NumPredictOneRow(0, -1, false, false);
  const int leaves = cpu->NumPredictOneRow(0, -1, true, false);
  std::vector<double> cp(outputs), np(outputs), cl(leaves), nl(leaves);
  for (const Data* data : {&train, &valid}) for (int row = 0; row < data->rows; ++row) {
    const double* values = data->values.data() + row * data->columns;
    cpu->Predict(values, cp.data(), &kNoPredictionStop); npu->Predict(values, np.data(), &kNoPredictionStop);
    for (int i = 0; i < outputs; ++i) Near(np[i], cp[i], label + "/row_prediction");
    cpu->PredictLeafIndex(values, cl.data()); npu->PredictLeafIndex(values, nl.data());
    Require(cl == nl, label + ": exact leaf indices");
  }
}
void CheckReload(QuantizedBoosterProbe* model, const Data& valid) {
  const auto text = model->SaveModelToString(0, -1, 0);
  GBDT reloaded;
  Require(reloaded.LoadModelFromString(text.data(), text.size()), "reload trained quantized model");
  model->InitPredict(0, -1, false); reloaded.InitPredict(0, -1, false);
  const int count = model->NumPredictOneRow(0, -1, false, false);
  const int leaves = model->NumPredictOneRow(0, -1, true, false);
  Require(count == reloaded.NumPredictOneRow(0, -1, false, false) &&
          leaves == reloaded.NumPredictOneRow(0, -1, true, false), "reloaded prediction dimensions");
  std::vector<double> before(count), after(count), bl(leaves), al(leaves);
  for (int row = 0; row < valid.rows; ++row) {
    const auto* values = valid.values.data() + row * valid.columns;
    model->Predict(values, before.data(), &kNoPredictionStop); reloaded.Predict(values, after.data(), &kNoPredictionStop);
    for (int i = 0; i < count; ++i) Near(after[i], before[i], "reloaded model prediction");
    model->PredictRaw(values, before.data(), &kNoPredictionStop); reloaded.PredictRaw(values, after.data(), &kNoPredictionStop);
    for (int i = 0; i < count; ++i) Near(after[i], before[i], "reloaded model raw prediction");
    model->PredictLeafIndex(values, bl.data()); reloaded.PredictLeafIndex(values, al.data());
    Require(bl == al, "reloaded leaf indices");
  }
}
void RunBooster(const BoosterScenario& scenario, bool host_only) {
  Data train(false), valid(false, false, 48, train.handle, 301);
  SetObjectiveData(&train, scenario); SetObjectiveData(&valid, scenario);
  Config config;
  config.Set({{"objective", scenario.objective}, {"metric", scenario.metric}, {"num_threads", "2"},
              {"num_leaves", "5"}, {"min_data_in_leaf", "6"}, {"min_data_in_bin", "1"},
              {"max_bin", "15"}, {"feature_pre_filter", "false"}, {"num_iterations", "10"},
              {"use_quantized_grad", "true"}, {"num_grad_quant_bins", "16"},
              {"quant_train_renew_leaf", scenario.renewal ? "true" : "false"},
              {"stochastic_rounding", scenario.stochastic ? "true" : "false"},
              {"force_col_wise", "true"}, {"deterministic", "true"}, {"verbosity", "-1"},
              {"seed", "37"}, {"learning_rate", "0.15"}, {"lambda_l1", "0.03"}, {"lambda_l2", "0.4"},
              {"bagging_fraction", scenario.bagging ? "0.8" : "1.0"}, {"bagging_freq", scenario.bagging ? "1" : "0"},
              {"num_class", scenario.objective.find("multiclass") == 0 ? "3" : "1"}});
  // Separate objective instances: ranking objectives own random state.
  std::unique_ptr<ObjectiveFunction> co(ObjectiveFunction::CreateObjectiveFunction(config.objective, config));
  std::unique_ptr<ObjectiveFunction> no(ObjectiveFunction::CreateObjectiveFunction(config.objective, config));
  Require(co != nullptr && no != nullptr, "missing built-in objective");
  co->Init(train.dataset()->metadata(), train.rows); no->Init(train.dataset()->metadata(), train.rows);
  std::unique_ptr<Metric> tm(Metric::CreateMetric(scenario.metric, config));
  std::unique_ptr<Metric> vm(Metric::CreateMetric(scenario.metric, config));
  Require(tm != nullptr && vm != nullptr, "missing metric");
  tm->Init(train.dataset()->metadata(), train.rows); vm->Init(valid.dataset()->metadata(), valid.rows);
  QuantizedBoosterProbe cpu, npu;
  cpu.Init(&config, train.dataset(), co.get(), {tm.get()});
  if (host_only) npu.Init(&config, train.dataset(), no.get(), {tm.get()});
  else npu.InitNPU(config, train.dataset(), no.get(), {tm.get()});
  cpu.AddValidDataset(valid.dataset(), {vm.get()}); npu.AddValidDataset(valid.dataset(), {vm.get()});
  for (int round = 0; round < 3; ++round) {
    const bool cs = cpu.TrainOneIter(nullptr, nullptr), ns = npu.TrainOneIter(nullptr, nullptr);
    Require(cs == ns && !ns, scenario.name + ": premature stop");
    CompareBooster(&npu, &cpu, train, valid, scenario.name + "/round" + std::to_string(round));
  }
  const bool rollback = scenario.name == "regression" || scenario.name == "multiclass" ||
                        scenario.name == "bagging_regression" || scenario.name == "stochastic_binary";
  if (rollback) {
    cpu.RollbackOneIter(); npu.RollbackOneIter();
    CompareBooster(&npu, &cpu, train, valid, scenario.name + "/rollback_to_two");
    Require(!cpu.TrainOneIter(nullptr, nullptr) && !npu.TrainOneIter(nullptr, nullptr), "retrain after rollback");
    CompareBooster(&npu, &cpu, train, valid, scenario.name + "/retrain_after_rollback");
  }
  const bool rollback_to_zero = scenario.name == "regression" || scenario.name == "multiclass";
  if (rollback_to_zero) {
    for (int left = 2; left >= 0; --left) {
      cpu.RollbackOneIter(); npu.RollbackOneIter();
      CompareBooster(&npu, &cpu, train, valid, scenario.name + "/rollback_to_" + std::to_string(left));
    }
    for (int round = 0; round < 3; ++round) {
      Require(!cpu.TrainOneIter(nullptr, nullptr) && !npu.TrainOneIter(nullptr, nullptr), "retrain after full rollback");
      CompareBooster(&npu, &cpu, train, valid, scenario.name + "/restart_round" + std::to_string(round));
    }
  }
  CheckReload(&cpu, valid); CheckReload(&npu, valid);
  Require(npu.NumberOfTotalModel() == 3 * co->NumModelPerIteration(), "non-vacuous Booster model count");
  if (!host_only) {
    if (scenario.renewal || co->IsRenewTreeOutput())
      Require(npu.Downloads() > 0, "Booster leaf renewal did not materialize row indices");
    else
      Require(npu.Downloads() == 0, "no-renewal Booster downloaded rows");
  }
  std::cout << (host_only ? "QUANTIZED_BOOSTER_HOST_SCENARIO_PASS " : "QUANTIZED_BOOSTER_SCENARIO_PASS ")
            << scenario.name << " rounds=3 trees=" << npu.NumberOfTotalModel();
  if (!host_only) std::cout << " row_downloads=" << npu.Downloads();
  std::cout << " reload=true validation=true rollback=" << (rollback ? "true" : "false")
            << " rollback_to_zero=" << (rollback_to_zero ? "true" : "false") << std::endl;
}
void RunBoosterSuite(const std::string& only, bool host_only) {
  Config rejected; rejected.use_quantized_grad = true;
  Reject([&]() { NPUTreeLearner learner(&rejected); }, "public NPU constructor accepted quantized training");
  Reject([]() { Config c; c.Set({{"device_type", "npu"}, {"use_quantized_grad", "true"}}); },
         "public config accepted quantized NPU");
  std::vector<BoosterScenario> scenarios;
  for (const auto& item : std::vector<std::pair<std::string, std::string>>{
       {"regression", "l2"}, {"regression_l1", "l1"}, {"huber", "huber"}, {"fair", "fair"},
       {"quantile", "quantile"}, {"mape", "mape"}, {"poisson", "poisson"}, {"gamma", "gamma"},
       {"tweedie", "tweedie"}, {"binary", "binary_logloss"}, {"cross_entropy", "cross_entropy"},
       {"cross_entropy_lambda", "cross_entropy_lambda"}, {"multiclass", "multi_logloss"},
       {"multiclassova", "multi_logloss"}, {"lambdarank", "ndcg"}, {"rank_xendcg", "ndcg"}})
    scenarios.push_back({item.first, item.first, item.second});
  scenarios.push_back({"bagging_regression", "regression", "l2", true});
  scenarios.push_back({"stochastic_binary", "binary", "binary_logloss", false, true});
  scenarios.push_back({"no_renew_regression", "regression", "l2", false, false, false});
  scenarios.push_back({"no_renew_multiclass", "multiclass", "multi_logloss", false, false, false});
  // Exercise objective-specific leaf renewal without an earlier quantized
  // renewal having already downloaded the partition's row indices.
  scenarios.push_back({"no_quant_renew_l1", "regression_l1", "l1", false, false, false});
  scenarios.push_back({"no_quant_renew_quantile", "quantile", "quantile", false, false, false});
  scenarios.push_back({"no_quant_renew_mape", "mape", "mape", false, false, false});
  scenarios.push_back({"bagging_no_quant_renew_l1", "regression_l1", "l1", true, false, false});
  scenarios.push_back({"bagging_no_quant_renew_quantile", "quantile", "quantile", true, false, false});
  // Unweighted regression takes the constant-Hessian objective path, unlike
  // the weighted regression scenarios above.
  scenarios.push_back({"unweighted_regression", "regression", "l2", false, false, true, false});
  scenarios.push_back({"unweighted_no_renew_regression", "regression", "l2", false, false, false, false});
  int checked = 0;
  for (const auto& scenario : scenarios)
    if (only.empty() || only == scenario.name) { RunBooster(scenario, host_only); ++checked; }
  Require(checked > 0, "no matching Booster scenario");
  std::cout << (host_only ? "QUANTIZED_BOOSTER_HOST_SUITE_PASS scenarios=" : "QUANTIZED_BOOSTER_SUITE_PASS scenarios=")
            << checked << " public_entry=rejected" << std::endl;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    OMP_SET_NUM_THREADS(2);
    if (argc > 1 && (std::string(argv[1]) == "--booster" || std::string(argv[1]) == "--booster-host")) {
      RunBoosterSuite(argc > 2 ? argv[2] : "", std::string(argv[1]) == "--booster-host");
      return 0;
    }
    RootStatisticsOracle();
    HistogramPackingOracle();
    ZeroScaleOracle();
    if (argc > 1 && std::string(argv[1]) == "--host-only") return 0;
    Config rejected; rejected.use_quantized_grad = true;
    Reject([&]() { NPUTreeLearner learner(&rejected); }, "public NPU constructor accepted quantized training");
    Reject([]() { Config c; c.Set({{"device_type", "npu"}, {"use_quantized_grad", "true"}}); }, "public config accepted quantized NPU");
    RejectionOracle();
    std::vector<Scenario> scenarios;
    scenarios.push_back({"constant"});
    auto s = scenarios.front(); s.name = "variable_hessian"; s.constant = false; scenarios.push_back(s);
    s = scenarios.front(); s.name = "bagging"; s.bagging = true; scenarios.push_back(s);
    s.name = "bagging_variable"; s.constant = false; scenarios.push_back(s);
    s = scenarios.front(); s.name = "renew_leaf"; s.renewal = true; scenarios.push_back(s);
    s.name = "renew_bagging_variable"; s.bagging = true; s.constant = false; scenarios.push_back(s);
    s = scenarios.front(); s.name = "stochastic"; s.stochastic = true; scenarios.push_back(s);
    s.name = "stochastic_variable"; s.constant = false; scenarios.push_back(s);
    s = scenarios.front(); s.name = "smooth_clip"; s.smooth = 3.0; s.clip = 0.1; scenarios.push_back(s);
    s = scenarios.front(); s.name = "node_sampling"; s.bynode = 0.6; scenarios.push_back(s);
    s = scenarios.front(); s.name = "cache_eviction"; s.pool = 0.001; scenarios.push_back(s);
    s = scenarios.front(); s.name = "zero_missing"; s.zero_missing = true; scenarios.push_back(s);
    s = scenarios.front(); s.name = "zero_gradient"; s.zero_gradient = true; scenarios.push_back(s);
    for (const auto& scenario : scenarios) Run(scenario);
    std::cout << "QUANTIZED_TREE_LIFECYCLE_PASS scenarios=" << scenarios.size() << " public_entry=rejected\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n'; return 1;
  }
}