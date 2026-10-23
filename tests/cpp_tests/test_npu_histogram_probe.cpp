/* Copyright (c) 2026. Licensed under the MIT License. */
// Real-device regression for resident bins and tiled weighted histograms.
#include "npu_runtime.h"
#include "npu_gradient_stager.h"
#include "npu_histogram_constructor.h"
#include "npu_data_partition.h"
#include "gradient_discretizer.hpp"
#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Runtime = LightGBM::NPURuntime;

struct QuantizedFixture {
  std::vector<std::vector<uint32_t>> bins;
  std::vector<int> offsets, sizes, most_frequent;
  std::vector<int8_t> used, pairs;
  // The reference values are independent of device packing and readback.
  std::vector<int64_t> gradients, hessians;
};

void PackQuantizedPairs(QuantizedFixture* fixture) {
  fixture->pairs.resize(2 * fixture->gradients.size());
  for (size_t row = 0; row < fixture->gradients.size(); ++row) {
    const uint8_t h = static_cast<uint8_t>(fixture->hessians[row]);
    const int8_t g = static_cast<int8_t>(fixture->gradients[row]);
    // CPU layout is [unsigned H byte, signed G byte]. In particular, never
    // convert H=128/255 to int8_t with an implementation-defined cast.
    std::memcpy(fixture->pairs.data() + 2 * row, &h, sizeof(h));
    std::memcpy(fixture->pairs.data() + 2 * row + 1, &g, sizeof(g));
  }
}

Runtime::QuantizedHistogramResult QuantizedOracle(
    const QuantizedFixture& fixture, const std::vector<int32_t>& rows, bool fixed) {
  Runtime::QuantizedHistogramResult expected;
  expected.gradients.resize(fixture.bins.size());
  expected.hessians.resize(fixture.bins.size());
  for (size_t f = 0; f < fixture.bins.size(); ++f) {
    if (!fixture.used[f]) continue;
    expected.gradients[f].assign(fixture.sizes[f] - fixture.offsets[f], 0);
    expected.hessians[f].assign(expected.gradients[f].size(), 0);
    for (int32_t row : rows) {
      const int bin = static_cast<int>(fixture.bins[f][row]);
      if (bin < fixture.offsets[f] ||
          (!fixed && bin == fixture.most_frequent[f])) continue;
      const size_t stored = static_cast<size_t>(bin - fixture.offsets[f]);
      expected.gradients[f][stored] += fixture.gradients[row];
      expected.hessians[f][stored] += fixture.hessians[row];
    }
  }
  return expected;
}

void CheckQuantized(const Runtime::QuantizedHistogramResult& actual,
                    const Runtime::QuantizedHistogramResult& expected,
                    const char* label) {
  if (actual.gradients.size() != expected.gradients.size() ||
      actual.hessians.size() != expected.hessians.size())
    throw std::runtime_error(std::string(label) + ": feature count");
  for (size_t f = 0; f < expected.gradients.size(); ++f) {
    if (actual.gradients[f].size() != expected.gradients[f].size() ||
        actual.hessians[f].size() != expected.hessians[f].size())
      throw std::runtime_error(std::string(label) + ": stored bin count/mask");
    for (size_t b = 0; b < expected.gradients[f].size(); ++b) {
      // Deliberately no tolerance and no conversion to floating point.
      if (actual.gradients[f][b] != expected.gradients[f][b] ||
          actual.hessians[f][b] != expected.hessians[f][b]) {
        std::cerr << label << " feature=" << f << " bin=" << b
                  << " actual=" << actual.gradients[f][b] << "," << actual.hessians[f][b]
                  << " expected=" << expected.gradients[f][b] << "," << expected.hessians[f][b]
                  << '\n';
        throw std::runtime_error(std::string(label) + ": exact integer mismatch");
      }
    }
  }
}

std::pair<int64_t, int64_t> QuantizedTotals(
    const QuantizedFixture& fixture, const std::vector<int32_t>& rows) {
  std::pair<int64_t, int64_t> totals{0, 0};
  for (int32_t row : rows) {
    totals.first += fixture.gradients[row];
    totals.second += fixture.hessians[row];
  }
  return totals;
}

template <typename Action>
void ExpectQuantizedFailure(Action action, const char* label) {
  bool caught = false;
  try {
    action();
  } catch (const std::exception&) {
    caught = true;
  }
  if (!caught) throw std::runtime_error(std::string(label) + ": expected exception");
}

int64_t PackStats32(int64_t gradient, int64_t hessian) {
  if (gradient < std::numeric_limits<int32_t>::min() ||
      gradient > std::numeric_limits<int32_t>::max() || hessian < 0 ||
      static_cast<uint64_t>(hessian) > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("CPU reference packed statistics overflow");
  const uint64_t bits = (static_cast<uint64_t>(static_cast<uint32_t>(gradient)) << 32) |
      static_cast<uint32_t>(hessian);
  int64_t result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int CheckQuantizedScan(Runtime* runtime, const Runtime::QuantizedSplitScanHandle& scan,
                        const std::vector<Runtime::SplitFeature>& features,
                        const QuantizedFixture& fixture, const std::vector<int32_t>& rows) {
  const auto candidates = runtime->ReadQuantizedSplitScan(scan);
  std::vector<Runtime::QuantizedSplitCandidate> expected_metadata;
  auto ordered_features = features;
  std::stable_sort(ordered_features.begin(), ordered_features.end(),
                   [](const Runtime::SplitFeature& a, const Runtime::SplitFeature& b) {
    return a.feature < b.feature;
  });
  for (const auto& feature : ordered_features) {
    if (feature.num_bin <= 1) continue;
    const bool double_scan = feature.num_bin > 2 && feature.missing_type != 0;
    auto append = [&](int threshold, bool default_left) {
      Runtime::QuantizedSplitCandidate candidate;
      candidate.feature = feature.feature;
      candidate.threshold = static_cast<uint32_t>(threshold);
      candidate.default_left = default_left;
      expected_metadata.push_back(candidate);
    };
    // Derive logical thresholds independently of stored-bin/device indices.
    const int reverse_last_bin = feature.num_bin - 1 -
        static_cast<int>(double_scan && feature.missing_type == 2);
    for (int logical_bin = reverse_last_bin; logical_bin >= 1; --logical_bin) {
      if (double_scan && feature.missing_type == 1 && logical_bin == feature.default_bin) continue;
      append(logical_bin - 1, !(feature.missing_type == 2 && !double_scan));
    }
    if (double_scan) {
      const int first_threshold = feature.missing_type == 2 ? 0 : feature.offset;
      for (int threshold = first_threshold; threshold < feature.num_bin - 1; ++threshold) {
        if (feature.missing_type == 1 && threshold == feature.default_bin) continue;
        append(threshold, false);
      }
    }
  }
  if (candidates.size() != expected_metadata.size())
    throw std::runtime_error("quantized split scan candidate count");
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].feature != expected_metadata[i].feature ||
        candidates[i].threshold != expected_metadata[i].threshold ||
        candidates[i].default_left != expected_metadata[i].default_left)
      throw std::runtime_error("quantized split candidate completeness or scan order");
  }
  int previous_feature = -1;
  for (const auto& candidate : candidates) {
    if (candidate.feature < previous_feature)
      throw std::runtime_error("quantized split feature tie order");
    previous_feature = candidate.feature;
    const auto found = std::find_if(features.begin(), features.end(), [&](const Runtime::SplitFeature& f) {
      return f.feature == candidate.feature;
    });
    if (found == features.end()) throw std::runtime_error("quantized scan unknown feature");
    const auto& feature = *found;
    int64_t left_g = 0, left_h = 0, right_g = 0, right_h = 0;
    for (int row : rows) {
      const uint32_t bin = fixture.bins[feature.histogram_index][row];
      const bool missing = (feature.missing_type == 1 && bin == static_cast<uint32_t>(feature.default_bin)) ||
          (feature.missing_type == 2 && bin == static_cast<uint32_t>(feature.num_bin - 1));
      const bool left = missing ? candidate.default_left : bin <= candidate.threshold;
      (left ? left_g : right_g) += fixture.gradients[row];
      (left ? left_h : right_h) += fixture.hessians[row];
    }
    if (candidate.left_gradient != left_g || candidate.left_hessian != left_h ||
        candidate.right_gradient != right_g || candidate.right_hessian != right_h) {
      std::cerr << "scan mismatch feature=" << feature.feature << " bins=" << feature.num_bin
                << " offset=" << feature.offset << " missing=" << feature.missing_type
                << " default_bin=" << feature.default_bin << " threshold=" << candidate.threshold
                << " default_left=" << candidate.default_left << " actual="
                << candidate.left_gradient << ',' << candidate.left_hessian << ','
                << candidate.right_gradient << ',' << candidate.right_hessian
                << " expected=" << left_g << ',' << left_h << ',' << right_g << ',' << right_h << '\n';
      throw std::runtime_error("quantized split prefix differs from per-row membership oracle");
    }
  }

  // Use the actual CPU quantized scanner with independently packed row-oracle
  // histograms. Every production CPU winner must exist with exact child
  // integer totals in the device candidate scan. This does NOT test NPU gain
  // evaluation or selection, which are not implemented by this API.
  const auto oracle = QuantizedOracle(fixture, rows, true);
  const auto totals = QuantizedTotals(fixture, rows);
  int cpu_winners = 0;
  for (const auto& feature : features) {
    LightGBM::Config config;
    config.use_quantized_grad = true;
    config.min_data_in_leaf = 1;
    config.min_sum_hessian_in_leaf = 0.0;
    config.lambda_l2 = 1.0;
    LightGBM::FeatureMetainfo meta;
    meta.config = &config;
    meta.bin_type = LightGBM::BinType::NumericalBin;
    meta.num_bin = feature.num_bin;
    meta.offset = static_cast<int8_t>(feature.offset);
    meta.default_bin = static_cast<uint32_t>(feature.default_bin);
    meta.missing_type = static_cast<LightGBM::MissingType>(feature.missing_type);
    const int f = feature.histogram_index;
    std::vector<LightGBM::hist_t> storage(oracle.gradients[f].size());
    for (size_t bin = 0; bin < storage.size(); ++bin) {
      const int64_t packed = PackStats32(oracle.gradients[f][bin], oracle.hessians[f][bin]);
      static_assert(sizeof(packed) == sizeof(LightGBM::hist_t), "CPU reference storage width");
      std::memcpy(storage.data() + bin, &packed, sizeof(packed));
    }
    LightGBM::FeatureHistogram histogram;
    histogram.Init(storage.data(), &meta);
    LightGBM::SplitInfo best;
    histogram.FindBestThresholdInt(PackStats32(totals.first, totals.second),
                                   0.125, 0.25, 32, 32, static_cast<int>(rows.size()),
                                   nullptr, 0.0, &best);
    if (best.gain == LightGBM::kMinScore) continue;
    const auto winner = std::find_if(candidates.begin(), candidates.end(), [&](const Runtime::QuantizedSplitCandidate& c) {
      return c.feature == feature.feature && c.threshold == best.threshold && c.default_left == best.default_left;
    });
    if (winner == candidates.end() ||
        PackStats32(winner->left_gradient, winner->left_hessian) != best.left_sum_gradient_and_hessian ||
        PackStats32(winner->right_gradient, winner->right_hessian) != best.right_sum_gradient_and_hessian)
      throw std::runtime_error("CPU quantized winner missing or integer totals differ");
    ++cpu_winners;
  }
  return cpu_winners;
}

void RunQuantizedSplitScanProbe() {
  Runtime runtime(0);
  constexpr int kRows = 257;
  QuantizedFixture fixture;
  fixture.sizes = {2, 3, 7, 5, 9, 2, 3, 6, 4, 8, 1};
  const int count = static_cast<int>(fixture.sizes.size());
  fixture.bins.resize(count);
  fixture.used.assign(count, 1);
  fixture.offsets.resize(count);
  fixture.most_frequent.resize(count);
  fixture.gradients.resize(kRows);
  fixture.hessians.resize(kRows);
  std::vector<int32_t> rows;
  for (int row = kRows - 1; row >= 0; --row) {
    fixture.gradients[row] = row % 17 - 8;
    fixture.hessians[row] = row % 7 == 0 ? 255 : 1 + row % 5;
    if (row % 5 != 2) rows.push_back(row);
  }
  PackQuantizedPairs(&fixture);
  Runtime::QuantizedSplitScanHandle old_scan;
  int cpu_winners = 0;
  for (int epoch = 0; epoch < 3; ++epoch) {
    std::vector<Runtime::SplitFeature> features;
    for (int f = 0; f < count; ++f) {
      fixture.offsets[f] = fixture.sizes[f] == 1 ? 0 : (f + epoch) % 2;
      fixture.most_frequent[f] = fixture.offsets[f] ? 0 : fixture.sizes[f] - 1;
      fixture.bins[f].resize(kRows);
      for (int row = 0; row < kRows; ++row)
        fixture.bins[f][row] = (row * 7 + row / 5 + f + epoch) % fixture.sizes[f];
      Runtime::SplitFeature feature;
      feature.feature = 2 * f + 1;
      feature.histogram_index = f;
      feature.num_bin = fixture.sizes[f];
      feature.offset = fixture.offsets[f];
      // The two-bin zero-missing CPU scanner uses the ordinary reverse scan;
      // keep its default at logical zero as in a binary zero-missing mapper.
      feature.default_bin = fixture.offsets[f] || fixture.sizes[f] <= 2 ? 0 : fixture.sizes[f] / 2;
      feature.missing_type = epoch;
      feature.enabled = f % 3 != 0;  // candidate preparation retains disabled features
      features.push_back(feature);
    }
    std::reverse(features.begin(), features.end());
    runtime.SetHistogramBins(fixture.bins, fixture.offsets, fixture.sizes, fixture.most_frequent);
    runtime.SetQuantizedHistogramGradients(fixture.pairs.data(), kRows);
    auto histogram = runtime.BuildQuantizedHistogram(rows.data(), rows.size(), fixture.used);
    const auto totals = QuantizedTotals(fixture, rows);
    ExpectQuantizedFailure([&]() {
      runtime.PrepareQuantizedSplitScan(features, totals.first, totals.second, rows.size(), histogram);
    }, "quantized scan requires fixed histogram");
    runtime.FixQuantizedHistogram(histogram, totals.first, totals.second);
    const auto downloads = runtime.partition_rows_downloaded();
    auto scan = runtime.PrepareQuantizedSplitScan(features, totals.first, totals.second, rows.size(), histogram);
    cpu_winners += CheckQuantizedScan(&runtime, scan, features, fixture, rows);
    // The overwritten parent must scan identically to a directly built larger
    // child; metadata count follows subtraction as well as integer values.
    std::vector<int32_t> smaller, larger;
    for (int row : rows) (row % 3 == 0 ? smaller : larger).push_back(row);
    auto smaller_histogram = runtime.BuildQuantizedHistogram(smaller.data(), smaller.size(), fixture.used);
    const auto smaller_totals = QuantizedTotals(fixture, smaller);
    runtime.FixQuantizedHistogram(smaller_histogram, smaller_totals.first, smaller_totals.second);
    auto direct_larger = runtime.BuildQuantizedHistogram(larger.data(), larger.size(), fixture.used);
    const auto larger_totals = QuantizedTotals(fixture, larger);
    runtime.FixQuantizedHistogram(direct_larger, larger_totals.first, larger_totals.second);
    auto parent_copy = runtime.BuildQuantizedHistogram(rows.data(), rows.size(), fixture.used);
    runtime.FixQuantizedHistogram(parent_copy, totals.first, totals.second);
    runtime.SubtractQuantizedHistogram(parent_copy, smaller_histogram);
    auto derived_scan = runtime.PrepareQuantizedSplitScan(features, larger_totals.first,
                                                         larger_totals.second, larger.size(), parent_copy);
    CheckQuantizedScan(&runtime, derived_scan, features, fixture, larger);
    const auto direct_scan = runtime.PrepareQuantizedSplitScan(features, larger_totals.first,
                                                              larger_totals.second, larger.size(), direct_larger);
    CheckQuantizedScan(&runtime, direct_scan, features, fixture, larger);
    if (runtime.partition_rows_downloaded() != downloads)
      throw std::runtime_error("quantized scan downloaded partition rows");
    if (!runtime.ReadQuantizedSplitScan(runtime.PrepareQuantizedSplitScan(
        {}, totals.first, totals.second, rows.size(), histogram)).empty())
      throw std::runtime_error("empty feature scan");
    const auto one_bin = std::find_if(features.begin(), features.end(), [](const Runtime::SplitFeature& f) {
      return f.num_bin == 1;
    });
    if (one_bin == features.end() || !runtime.ReadQuantizedSplitScan(runtime.PrepareQuantizedSplitScan(
        {*one_bin}, totals.first, totals.second, rows.size(), histogram)).empty())
      throw std::runtime_error("one-bin-only feature scan");
    if (epoch == 0) {
      Runtime foreign(0);
      ExpectQuantizedFailure([&]() {
        foreign.PrepareQuantizedSplitScan(features, totals.first, totals.second, rows.size(), histogram);
      }, "quantized scan foreign histogram rejection");
      ExpectQuantizedFailure([&]() { foreign.ReadQuantizedSplitScan(scan); },
                             "quantized scan foreign snapshot rejection");
      // Readback must restore the owning context after foreign construction.
      CheckQuantizedScan(&runtime, scan, features, fixture, rows);
    }
    auto empty_histogram = runtime.BuildQuantizedHistogram(nullptr, 0, fixture.used);
    runtime.FixQuantizedHistogram(empty_histogram, 0, 0);
    ExpectQuantizedFailure([&]() {
      runtime.PrepareQuantizedSplitScan(features, 0, 0, 0, empty_histogram);
    }, "quantized empty leaf scan rejection");
    auto invalid = features;
    invalid[0].categorical = true;
    ExpectQuantizedFailure([&]() {
      runtime.PrepareQuantizedSplitScan(invalid, totals.first, totals.second, rows.size(), histogram);
    }, "quantized categorical scan rejection");
    LightGBM::Random random(17);
    invalid = features;
    invalid[0].extra_tree_random = &random;
    ExpectQuantizedFailure([&]() {
      runtime.PrepareQuantizedSplitScan(invalid, totals.first, totals.second, rows.size(), histogram);
    }, "quantized randomized scan rejection");
    ExpectQuantizedFailure([&]() {
      runtime.PrepareQuantizedSplitScan(features, totals.first, totals.second, rows.size() - 1, histogram);
    }, "quantized scan row count mismatch");
    ExpectQuantizedFailure([&]() {
      runtime.PrepareQuantizedSplitScan(features, totals.first, 0, rows.size(), histogram);
    }, "quantized scan zero total Hessian");
    old_scan = scan;
    runtime.SetQuantizedHistogramGradients(fixture.pairs.data(), kRows);
    ExpectQuantizedFailure([&]() {
      runtime.PrepareQuantizedSplitScan(features, totals.first, totals.second, rows.size(), histogram);
    }, "quantized scan stale gradient rejection");
    CheckQuantizedScan(&runtime, old_scan, features, fixture, rows);
    runtime.SetHistogramBins(fixture.bins, fixture.offsets, fixture.sizes, fixture.most_frequent);
    ExpectQuantizedFailure([&]() {
      runtime.PrepareQuantizedSplitScan(features, totals.first, totals.second, rows.size(), histogram);
    }, "quantized scan stale dataset rejection");
    CheckQuantizedScan(&runtime, old_scan, features, fixture, rows);
  }
  if (cpu_winners == 0) throw std::runtime_error("CPU quantized winner reference was vacuous");
  std::cout << "QUANTIZED_SPLIT_SCAN_PASS cpu_winners_checked=" << cpu_winners << '\n';
}

// Independent, scalar host arithmetic: do not call FeatureHistogram's output
// helpers here. Its production integer scanner is a second reference below.
double QuantizedEvaluationOutput(double gradient, double hessian, int count,
                                 const Runtime::QuantizedSplitParameters& p) {
  const double regularized = gradient > p.lambda_l1 ? gradient - p.lambda_l1 :
      (gradient < -p.lambda_l1 ? gradient + p.lambda_l1 : 0.0);
  double output = -regularized / (hessian + p.lambda_l2);
  if (p.max_delta_step > 0.0)
    output = std::max(-p.max_delta_step, std::min(p.max_delta_step, output));
  if (p.path_smooth > LightGBM::kEpsilon) {
    const double weight = count / p.path_smooth;
    output = output * weight / (weight + 1.0) + p.parent_output / (weight + 1.0);
  }
  return output;
}

double QuantizedEvaluationGain(double gradient, double hessian, int count,
                               const Runtime::QuantizedSplitParameters& p) {
  const double regularized = gradient > p.lambda_l1 ? gradient - p.lambda_l1 :
      (gradient < -p.lambda_l1 ? gradient + p.lambda_l1 : 0.0);
  if (p.max_delta_step == 0.0 && p.path_smooth <= LightGBM::kEpsilon)
    return regularized * regularized / (hessian + p.lambda_l2);
  const double output = QuantizedEvaluationOutput(gradient, hessian, count, p);
  return -(2.0 * regularized * output + (hessian + p.lambda_l2) * output * output);
}

void CheckQuantizedEvaluationNear(double actual, double expected, const char* field) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > 1e-11 + 1e-12 * std::fabs(expected)) {
    std::cerr << std::setprecision(17) << field << " actual=" << actual
              << " expected=" << expected << '\n';
    throw std::runtime_error(std::string("quantized evaluation: ") + field);
  }
}

void CheckQuantizedCandidate(const Runtime::QuantizedSplitCandidate& actual,
                             const Runtime::QuantizedSplitCandidate& expected) {
  if (actual.feature != expected.feature || actual.threshold != expected.threshold ||
      actual.default_left != expected.default_left ||
      actual.left_gradient != expected.left_gradient || actual.right_gradient != expected.right_gradient ||
      actual.left_hessian != expected.left_hessian || actual.right_hessian != expected.right_hessian) {
    std::cerr << "candidate actual f/t/d=" << actual.feature << '/' << actual.threshold << '/' << actual.default_left
              << " stats=" << actual.left_gradient << ',' << actual.left_hessian << ','
              << actual.right_gradient << ',' << actual.right_hessian
              << " expected=" << expected.feature << '/' << expected.threshold << '/' << expected.default_left
              << " stats=" << expected.left_gradient << ',' << expected.left_hessian << ','
              << expected.right_gradient << ',' << expected.right_hessian << '\n';
    throw std::runtime_error("quantized evaluation changed candidate metadata/integer totals");
  }
}

Runtime::QuantizedSplitEvaluation QuantizedEvaluationOracle(
    const Runtime::QuantizedSplitCandidate& metadata, const Runtime::SplitFeature& feature,
    const QuantizedFixture& fixture, const std::vector<int32_t>& rows,
    const Runtime::QuantizedSplitParameters& p, bool* per_bin_rounding_differs) {
  Runtime::QuantizedSplitEvaluation result{};
  result.candidate = metadata;
  result.candidate.left_gradient = result.candidate.right_gradient = 0;
  result.candidate.left_hessian = result.candidate.right_hessian = 0;
  // A false default direction means forward only for a real double scan.
  // In particular the binary NaN scanner is reverse with default_left=false.
  const bool reverse = !(feature.num_bin > 2 && feature.missing_type != 0) || metadata.default_left;
  std::vector<int64_t> scanned_bin_hessians(feature.num_bin, 0);
  for (int row : rows) {
    const uint32_t bin = fixture.bins[feature.histogram_index][row];
    const bool missing = (feature.missing_type == 1 && bin == static_cast<uint32_t>(feature.default_bin)) ||
        (feature.missing_type == 2 && bin == static_cast<uint32_t>(feature.num_bin - 1));
    const bool left = missing ? metadata.default_left : bin <= metadata.threshold;
    (left ? result.candidate.left_gradient : result.candidate.right_gradient) += fixture.gradients[row];
    (left ? result.candidate.left_hessian : result.candidate.right_hessian) += fixture.hessians[row];
    if (left != reverse) scanned_bin_hessians[bin] += fixture.hessians[row];
  }
  const auto totals = QuantizedTotals(fixture, rows);
  const int num_data = static_cast<int>(rows.size());
  const double count_factor = static_cast<double>(num_data) / totals.second;
  // All counts are nonnegative: floor(x+0.5) is independent of Common::RoundInt.
  auto count = [&](int64_t h) { return static_cast<int>(std::floor(h * count_factor + 0.5)); };
  result.left_count = count(result.candidate.left_hessian);
  result.right_count = count(result.candidate.right_hessian);
  result.scan_left_count = reverse ? num_data - result.right_count : result.left_count;
  result.scan_right_count = reverse ? result.right_count : num_data - result.left_count;
  int sum_bin_counts = 0;
  for (int64_t h : scanned_bin_hessians) sum_bin_counts += count(h);
  *per_bin_rounding_differs = sum_bin_counts != (reverse ? result.scan_right_count : result.scan_left_count);
  // Convert only after exact integer prefix/complement membership sums.
  result.left_gradient = result.candidate.left_gradient * p.gradient_scale;
  result.right_gradient = result.candidate.right_gradient * p.gradient_scale;
  result.left_hessian = result.candidate.left_hessian * p.hessian_scale;
  result.right_hessian = result.candidate.right_hessian * p.hessian_scale;
  result.gain_shift = QuantizedEvaluationGain(totals.first * p.gradient_scale,
      totals.second * p.hessian_scale, num_data, p) + p.min_gain_to_split;
  const bool bounds = result.scan_left_count >= p.min_data_in_leaf &&
      result.scan_right_count >= p.min_data_in_leaf &&
      result.left_hessian >= p.min_sum_hessian_in_leaf && result.right_hessian >= p.min_sum_hessian_in_leaf;
  if (bounds) {
    const double gain = QuantizedEvaluationGain(result.left_gradient, result.left_hessian + LightGBM::kEpsilon,
        result.scan_left_count, p) + QuantizedEvaluationGain(result.right_gradient,
        result.right_hessian + LightGBM::kEpsilon, result.scan_right_count, p);
    result.valid = gain > result.gain_shift;
    if (result.valid) {
      result.current_gain = gain;
      result.left_output = QuantizedEvaluationOutput(result.left_gradient, result.left_hessian, result.left_count, p);
      result.right_output = QuantizedEvaluationOutput(result.right_gradient, result.right_hessian, result.right_count, p);
    }
  }
  // Invalid gain/outputs intentionally remain defined zero placeholders; do
  // not evaluate a raw-H output for an empty/otherwise invalid leaf.
  return result;
}

struct QuantizedEvaluationCoverage {
  int valid = 0, invalid = 0, independent_counts = 0, cumulative_counts = 0;
  int reverse_nan_binary = 0, forward = 0, clipped_then_smoothed = 0;
};

int CheckQuantizedEvaluation(Runtime* runtime, const Runtime::QuantizedSplitEvaluationHandle& handle,
                             const std::vector<Runtime::QuantizedSplitCandidate>& candidates,
                             const std::vector<Runtime::SplitFeature>& features,
                             const QuantizedFixture& fixture, const std::vector<int32_t>& rows,
                             const Runtime::QuantizedSplitParameters& p,
                             QuantizedEvaluationCoverage* coverage, bool check_selection = false) {
  const auto actual = runtime->ReadQuantizedSplitEvaluation(handle);
  Runtime::QuantizedSplitSelection selection;
  uint64_t downloaded = runtime->quantized_selection_download_bytes();
  if (check_selection) selection = runtime->SelectQuantizedSplit(handle);
  LightGBM::SplitInfo expected_best;
  expected_best.feature = -1;
  std::vector<uint8_t> expected_splittable(features.size(), 0);
  if (actual.size() != candidates.size()) throw std::runtime_error("quantized evaluation candidate count");
  for (size_t i = 0; i < actual.size(); ++i) {
    const auto found = std::find_if(features.begin(), features.end(), [&](const Runtime::SplitFeature& f) {
      return f.feature == candidates[i].feature;
    });
    if (found == features.end()) throw std::runtime_error("quantized evaluation unknown feature");
    bool per_bin_rounding_differs = false;
    const auto expected = QuantizedEvaluationOracle(candidates[i], *found, fixture, rows, p, &per_bin_rounding_differs);
    const auto& value = actual[i];
    CheckQuantizedCandidate(value.candidate, expected.candidate);
    if (value.valid != expected.valid || value.left_count != expected.left_count ||
        value.right_count != expected.right_count || value.scan_left_count != expected.scan_left_count ||
        value.scan_right_count != expected.scan_right_count)
      throw std::runtime_error("quantized evaluation validity/final or scan counts");
    CheckQuantizedEvaluationNear(value.left_gradient, expected.left_gradient, "left gradient");
    CheckQuantizedEvaluationNear(value.right_gradient, expected.right_gradient, "right gradient");
    CheckQuantizedEvaluationNear(value.left_hessian, expected.left_hessian, "left Hessian");
    CheckQuantizedEvaluationNear(value.right_hessian, expected.right_hessian, "right Hessian");
    CheckQuantizedEvaluationNear(value.gain_shift, expected.gain_shift, "parent gain + minimum gain");
    CheckQuantizedEvaluationNear(value.current_gain, expected.current_gain, "current gain");
    CheckQuantizedEvaluationNear(value.left_output, expected.left_output, "left output");
    CheckQuantizedEvaluationNear(value.right_output, expected.right_output, "right output");
    if (!value.valid && (value.current_gain != 0.0 || value.left_output != 0.0 || value.right_output != 0.0))
      throw std::runtime_error("invalid quantized evaluation must have exact zero gain/outputs");
    if (coverage != nullptr) {
      (value.valid ? coverage->valid : coverage->invalid)++;
      coverage->independent_counts += value.left_count != value.scan_left_count || value.right_count != value.scan_right_count;
      coverage->cumulative_counts += per_bin_rounding_differs;
      coverage->reverse_nan_binary += found->num_bin == 2 && found->missing_type == 2 && !value.candidate.default_left;
      coverage->forward += found->num_bin > 2 && found->missing_type != 0 && !value.candidate.default_left;
      coverage->clipped_then_smoothed += value.valid && p.max_delta_step > 0 && p.path_smooth > LightGBM::kEpsilon &&
          (std::fabs(value.left_output) > p.max_delta_step || std::fabs(value.right_output) > p.max_delta_step);
    }
  }

  // A second, production reference uses independent packed per-row histograms,
  // not evaluator/device readback. Compare at each CPU-selected candidate; the
  // evaluator deliberately does not select a winner or apply feature penalty.
  const auto oracle = QuantizedOracle(fixture, rows, true);
  const auto totals = QuantizedTotals(fixture, rows);
  int cpu_winners = 0;
  for (const auto& feature : features) {
    if (feature.num_bin <= 1) continue;
    LightGBM::Config config;
    config.use_quantized_grad = true;
    config.lambda_l1 = p.lambda_l1;
    config.lambda_l2 = p.lambda_l2;
    config.min_data_in_leaf = p.min_data_in_leaf;
    config.min_sum_hessian_in_leaf = p.min_sum_hessian_in_leaf;
    config.min_gain_to_split = p.min_gain_to_split;
    config.max_delta_step = p.max_delta_step;
    config.path_smooth = p.path_smooth;
    LightGBM::FeatureMetainfo meta;
    meta.config = &config;
    meta.bin_type = LightGBM::BinType::NumericalBin;
    meta.num_bin = feature.num_bin;
    meta.offset = static_cast<int8_t>(feature.offset);
    meta.default_bin = static_cast<uint32_t>(feature.default_bin);
    meta.missing_type = static_cast<LightGBM::MissingType>(feature.missing_type);
    meta.penalty = feature.penalty;
    const int f = feature.histogram_index;
    std::vector<LightGBM::hist_t> storage(oracle.gradients[f].size());
    for (size_t b = 0; b < storage.size(); ++b) {
      const int64_t packed = PackStats32(oracle.gradients[f][b], oracle.hessians[f][b]);
      static_assert(sizeof(packed) == sizeof(LightGBM::hist_t), "CPU evaluation reference storage width");
      std::memcpy(storage.data() + b, &packed, sizeof(packed));
    }
    LightGBM::FeatureHistogram histogram;
    histogram.Init(storage.data(), &meta);
    LightGBM::SplitInfo best;
    histogram.FindBestThresholdInt(PackStats32(totals.first, totals.second), p.gradient_scale, p.hessian_scale,
                                   32, 32, static_cast<int>(rows.size()), nullptr, p.parent_output, &best);
    const bool any_valid = std::any_of(actual.begin(), actual.end(), [&](const Runtime::QuantizedSplitEvaluation& e) {
      return e.candidate.feature == feature.feature && e.valid;
    });
    if (histogram.is_splittable() != any_valid)
      throw std::runtime_error("CPU and evaluator disagree on presence of a valid candidate");
    if (!histogram.is_splittable()) continue;
    expected_splittable[&feature - features.data()] = 1;
    best.feature = feature.feature;
    if (feature.enabled && (expected_best.feature < 0 || best.gain > expected_best.gain ||
        (best.gain == expected_best.gain && best.feature < expected_best.feature))) expected_best = best;
    const auto winner = std::find_if(actual.begin(), actual.end(), [&](const Runtime::QuantizedSplitEvaluation& e) {
      return e.candidate.feature == feature.feature && e.candidate.threshold == best.threshold &&
          e.candidate.default_left == best.default_left;
    });
    if (winner == actual.end() || !winner->valid ||
        PackStats32(winner->candidate.left_gradient, winner->candidate.left_hessian) != best.left_sum_gradient_and_hessian ||
        PackStats32(winner->candidate.right_gradient, winner->candidate.right_hessian) != best.right_sum_gradient_and_hessian ||
        winner->left_count != best.left_count || winner->right_count != best.right_count)
      throw std::runtime_error("CPU evaluation winner missing or packed statistics/counts differ");
    CheckQuantizedEvaluationNear(best.gain, (winner->current_gain - winner->gain_shift) * feature.penalty, "CPU penalized gain");
    CheckQuantizedEvaluationNear(winner->left_output, best.left_output, "CPU left output");
    CheckQuantizedEvaluationNear(winner->right_output, best.right_output, "CPU right output");
    CheckQuantizedEvaluationNear(winner->left_gradient, best.left_sum_gradient, "CPU left gradient");
    CheckQuantizedEvaluationNear(winner->right_gradient, best.right_sum_gradient, "CPU right gradient");
    CheckQuantizedEvaluationNear(winner->left_hessian, best.left_sum_hessian, "CPU left Hessian");
    CheckQuantizedEvaluationNear(winner->right_hessian, best.right_sum_hessian, "CPU right Hessian");
    ++cpu_winners;
  }
  if (check_selection) {
    const auto& split = selection.split;
    if (split.splittable != expected_splittable || split.feature != expected_best.feature)
      throw std::runtime_error("quantized device selection feature/splittability differs from CPU");
    // Readback scales with scan directions, not candidate count: at most one
    // byte per direction plus 3 indices/scores, four INT64 and ten FP64 lanes.
    if (runtime->quantized_selection_download_bytes() - downloaded > 2 * features.size() + 136)
      throw std::runtime_error("quantized selection downloaded more than winner and row flags");
    if (split.feature >= 0) {
      if (split.threshold != expected_best.threshold || split.default_left != expected_best.default_left ||
          split.left_count != expected_best.left_count || split.right_count != expected_best.right_count ||
          PackStats32(selection.integer_statistics.left_gradient, selection.integer_statistics.left_hessian) !=
              expected_best.left_sum_gradient_and_hessian ||
          PackStats32(selection.integer_statistics.right_gradient, selection.integer_statistics.right_hessian) !=
              expected_best.right_sum_gradient_and_hessian)
        throw std::runtime_error("quantized device winner threshold/direction/counts/integers differ from CPU");
      CheckQuantizedEvaluationNear(split.gain, expected_best.gain, "selected CPU gain");
      CheckQuantizedEvaluationNear(split.left_output, expected_best.left_output, "selected CPU left output");
      CheckQuantizedEvaluationNear(split.right_output, expected_best.right_output, "selected CPU right output");
      CheckQuantizedEvaluationNear(split.left_gradient, expected_best.left_sum_gradient, "selected CPU left gradient");
      CheckQuantizedEvaluationNear(split.right_gradient, expected_best.right_sum_gradient, "selected CPU right gradient");
      CheckQuantizedEvaluationNear(split.left_hessian, expected_best.left_sum_hessian, "selected CPU left Hessian");
      CheckQuantizedEvaluationNear(split.right_hessian, expected_best.right_sum_hessian, "selected CPU right Hessian");
    }
  }
  return cpu_winners;
}

void RunQuantizedSplitEvaluationProbe(bool check_selection = false) {
  ExpectQuantizedFailure([]() {
    LightGBM::Config config;
    config.Set({{"device_type", "npu"}, {"use_quantized_grad", "true"}});
  }, "NPU quantized training remains unsupported");
  Runtime runtime(0);
  constexpr int kRows = 7;
  QuantizedFixture fixture;
  std::vector<Runtime::SplitFeature> features;
  for (int missing = 0; missing < 3; ++missing) {
    for (int offset = 0; offset <= 1; ++offset) {
      for (int num_bin : {2, 5}) {
        const int f = static_cast<int>(features.size());
        fixture.sizes.push_back(num_bin);
        fixture.offsets.push_back(offset);
        fixture.most_frequent.push_back(offset ? 0 : num_bin - 1);
        fixture.used.push_back(1);
        fixture.bins.emplace_back(kRows);
        for (int row = 0; row < kRows; ++row)
          fixture.bins.back()[row] = num_bin == 2 ? static_cast<uint32_t>(row >= 3) : static_cast<uint32_t>(row % 5);
        Runtime::SplitFeature feature;
        feature.feature = 2 * f + 1;
        feature.histogram_index = f;
        feature.num_bin = num_bin;
        feature.offset = offset;
        // Binary zero-missing mappers have default zero even with offset=0.
        feature.default_bin = offset || num_bin == 2 ? 0 : 2;
        feature.missing_type = missing;
        feature.enabled = f % 3 != 0;  // evaluation retains node-disabled candidates
        feature.penalty = 0.37 + 0.11 * f;
        features.push_back(feature);
      }
    }
  }
  Runtime::SplitFeature one_bin;
  one_bin.feature = 99;
  one_bin.histogram_index = static_cast<int>(fixture.sizes.size());
  one_bin.num_bin = 1;
  features.push_back(one_bin);
  fixture.sizes.push_back(1);
  fixture.offsets.push_back(0);
  fixture.most_frequent.push_back(0);
  fixture.used.push_back(1);
  fixture.bins.emplace_back(kRows, 0);
  std::reverse(features.begin(), features.end());
  std::vector<int32_t> rows{6, 5, 4, 3, 2, 1, 0};
  using Parameters = Runtime::QuantizedSplitParameters;
  Parameters base;
  base.gradient_scale = 0.125;
  base.hessian_scale = 0.25;
  base.lambda_l2 = 0.7;
  std::vector<Parameters> parameters{Parameters{}, base};
  auto p = base;
  p.lambda_l1 = 0.8;
  parameters.push_back(p);
  p = base; p.max_delta_step = 0.09;
  parameters.push_back(p);
  p = base; p.path_smooth = 3.0; p.parent_output = -0.35;
  parameters.push_back(p);
  p = base; p.lambda_l1 = 0.2; p.max_delta_step = 0.11; p.path_smooth = 2.5; p.parent_output = 0.4;
  parameters.push_back(p);
  for (double smooth : {0.5 * LightGBM::kEpsilon, static_cast<double>(LightGBM::kEpsilon)}) {
    p = base; p.path_smooth = smooth; p.parent_output = -1e8;
    parameters.push_back(p);
  }
  p = base; p.min_data_in_leaf = 2;  // final counts 2+6, scan counts 1+6 at binary half-boundary
  parameters.push_back(p);
  p = base; p.min_data_in_leaf = kRows + 1;
  parameters.push_back(p);
  p = base; p.min_sum_hessian_in_leaf = 0.7;
  parameters.push_back(p);
  p = base; p.min_sum_hessian_in_leaf = 1000.0;
  parameters.push_back(p);
  p = base; p.min_gain_to_split = 1e12;
  parameters.push_back(p);
  p = base; p.gradient_scale = 0.0;
  parameters.push_back(p);
  // Make the gain-only epsilon visible, and distinguish raw-H final outputs
  // and parent gain from epsilon-adjusted candidate gain arithmetic.
  p = base; p.gradient_scale = 1e-8; p.hessian_scale = 1e-16; p.lambda_l2 = 2e-15;
  parameters.push_back(p);
  p.min_sum_hessian_in_leaf = 9e-16;  // reject on raw H, before gain-only epsilon
  parameters.push_back(p);
  p = base; p.min_data_in_leaf = 0;
  parameters.push_back(p);
  p = base; p.gradient_scale = 0.13; p.hessian_scale = 0.17;
  parameters.push_back(p);

  QuantizedEvaluationCoverage coverage;
  int cpu_winners = 0;
  for (int variant = 0; variant < 4; ++variant) {
    fixture.gradients = variant == 2 ? std::vector<int64_t>(kRows, 0) :
        std::vector<int64_t>{-12, -10, -8, 9, 11, 7, 4};
    // total H=14 gives count factor 1/2 and simultaneous child half ties.
    fixture.hessians = variant == 1 ? std::vector<int64_t>(kRows, 3) :
        (variant == 3 ? std::vector<int64_t>{0, 0, 0, 1, 2, 2, 9} :
                        std::vector<int64_t>{1, 1, 1, 1, 2, 2, 6});
    PackQuantizedPairs(&fixture);
    runtime.SetHistogramBins(fixture.bins, fixture.offsets, fixture.sizes, fixture.most_frequent);
    runtime.SetQuantizedHistogramGradients(fixture.pairs.data(), kRows);
    const auto totals = QuantizedTotals(fixture, rows);
    auto histogram = runtime.BuildQuantizedHistogram(rows.data(), rows.size(), fixture.used);
    runtime.FixQuantizedHistogram(histogram, totals.first, totals.second);
    auto scan = runtime.PrepareQuantizedSplitScan(features, totals.first, totals.second, kRows, histogram);
    CheckQuantizedScan(&runtime, scan, features, fixture, rows);
    const auto candidates = runtime.ReadQuantizedSplitScan(scan);
    const auto downloads = runtime.partition_rows_downloaded();
    const auto selections = runtime.argmax_calls();
    auto first = runtime.EvaluateQuantizedSplitScan(scan, parameters.front());
    for (size_t index = 0; index < parameters.size(); ++index) {
      const auto& parameter = parameters[index];
      const auto evaluation = runtime.EvaluateQuantizedSplitScan(scan, parameter);
      const int winners = CheckQuantizedEvaluation(&runtime, evaluation, candidates, features, fixture, rows, parameter, &coverage,
                                                   check_selection);
      // These profiles have strictly no winners, independently of fixture.
      if ((parameter.min_data_in_leaf > kRows || parameter.min_sum_hessian_in_leaf == 1000.0 ||
           parameter.min_gain_to_split == 1e12 || parameter.gradient_scale == 0.0) && winners != 0)
        throw std::runtime_error("quantized evaluation expected no CPU winners");
      if (variant < 2 && index <= 5 && winners == 0)
        throw std::runtime_error("quantized evaluation regularized CPU reference was vacuous");
      if (variant == 2 && index == 0 && winners != 0)
        throw std::runtime_error("zero-gradient unregularized scan must have no winners");
      cpu_winners += winners;
    }
    // Evaluating different parameter sets must mutate neither earlier
    // evaluation snapshots nor the resident integer scan.
    CheckQuantizedEvaluation(&runtime, first, candidates, features, fixture, rows, parameters.front(), nullptr);
    const auto unchanged = runtime.ReadQuantizedSplitScan(scan);
    if (unchanged.size() != candidates.size()) throw std::runtime_error("evaluation resized scan");
    for (size_t i = 0; i < candidates.size(); ++i) CheckQuantizedCandidate(unchanged[i], candidates[i]);
    for (const auto& empty_features : std::vector<std::vector<Runtime::SplitFeature>>{{}, {one_bin}}) {
      const auto empty_scan = runtime.PrepareQuantizedSplitScan(empty_features, totals.first, totals.second, kRows, histogram);
      auto empty_evaluation = runtime.EvaluateQuantizedSplitScan(empty_scan, base);
      if (!runtime.ReadQuantizedSplitEvaluation(empty_evaluation).empty())
        throw std::runtime_error("empty/one-bin quantized evaluation");
      if (check_selection && runtime.SelectQuantizedSplit(empty_evaluation).split.feature != -1)
        throw std::runtime_error("empty/one-bin quantized selection");
    }
    if (variant == 0) {
      ExpectQuantizedFailure([&]() { runtime.EvaluateQuantizedSplitScan(nullptr, base); }, "null quantized scan evaluation");
      ExpectQuantizedFailure([&]() { runtime.ReadQuantizedSplitEvaluation(nullptr); }, "null quantized evaluation read");
      double Parameters::* fields[] = {&Parameters::gradient_scale, &Parameters::hessian_scale,
          &Parameters::lambda_l1, &Parameters::lambda_l2, &Parameters::min_sum_hessian_in_leaf,
          &Parameters::min_gain_to_split, &Parameters::max_delta_step, &Parameters::path_smooth, &Parameters::parent_output};
      for (auto field : fields) {
        for (double invalid : {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN(), -1.0}) {
          if (field == &Parameters::parent_output && invalid == -1.0) continue;
          auto invalid_parameters = base;
          invalid_parameters.*field = invalid;
          ExpectQuantizedFailure([&]() { runtime.EvaluateQuantizedSplitScan(scan, invalid_parameters); },
                                 "nonfinite/negative quantized split parameter");
        }
      }
      p = base; p.hessian_scale = 0.0;
      ExpectQuantizedFailure([&]() { runtime.EvaluateQuantizedSplitScan(scan, p); }, "zero Hessian scale");
      p = base; p.min_data_in_leaf = -1;
      ExpectQuantizedFailure([&]() { runtime.EvaluateQuantizedSplitScan(scan, p); }, "negative minimum leaf count");
      p = base; p.min_data_in_leaf = 0; p.min_sum_hessian_in_leaf = 0.0; p.lambda_l2 = 0.0;
      ExpectQuantizedFailure([&]() { runtime.EvaluateQuantizedSplitScan(scan, p); },
                             "unsafe zero-Hessian final output configuration");
      for (auto field : {&Parameters::gradient_scale, &Parameters::hessian_scale}) {
        p = base; p.*field = std::numeric_limits<double>::max();
        ExpectQuantizedFailure([&]() { runtime.EvaluateQuantizedSplitScan(scan, p); }, "scaled integer overflow");
      }
      {
        Runtime foreign(0);
        ExpectQuantizedFailure([&]() { foreign.EvaluateQuantizedSplitScan(scan, base); }, "foreign scan evaluation");
        ExpectQuantizedFailure([&]() { foreign.ReadQuantizedSplitEvaluation(first); }, "foreign evaluation read");
        // Owner calls must restore its context even while foreign is alive.
        auto restored = runtime.EvaluateQuantizedSplitScan(scan, base);
        CheckQuantizedEvaluation(&runtime, restored, candidates, features, fixture, rows, base, nullptr);
      }
      CheckQuantizedEvaluation(&runtime, first, candidates, features, fixture, rows, parameters.front(), nullptr);
    }
    if (runtime.partition_rows_downloaded() != downloads ||
        runtime.argmax_calls() != selections + (check_selection ? parameters.size() : 0))
      throw std::runtime_error("quantized evaluation/selection download or reduction counter mismatch");
    if (check_selection) {
      // Equal penalties expose exact cross-feature ties in duplicate binary
      // features. Input order is reversed; original feature id must win.
      // Also test zero-penalty ties and node-disabled splittability.
      for (int mask = 0; mask < 4; ++mask) {
        auto tied_features = features;
        for (auto& feature : tied_features) {
          feature.penalty = mask == 1 ? 0.0 : 1.0;
          feature.enabled = mask == 2 ? false : (mask == 3 ? feature.feature > 10 : true);
        }
        auto tied_scan = runtime.PrepareQuantizedSplitScan(tied_features, totals.first, totals.second, kRows, histogram);
        auto tied_eval = runtime.EvaluateQuantizedSplitScan(tied_scan, base);
        CheckQuantizedEvaluation(&runtime, tied_eval, runtime.ReadQuantizedSplitScan(tied_scan),
                                 tied_features, fixture, rows, base, nullptr, true);
      }
      ExpectQuantizedFailure([&]() { runtime.SelectQuantizedSplit(nullptr); }, "null quantized selection");
      Runtime foreign(0);
      ExpectQuantizedFailure([&]() { foreign.SelectQuantizedSplit(first); }, "foreign quantized selection");
      runtime.SelectQuantizedSplit(first);  // owner context restoration
    }
    runtime.SetQuantizedHistogramGradients(fixture.pairs.data(), kRows);
    if (check_selection)
      ExpectQuantizedFailure([&]() { runtime.SelectQuantizedSplit(first); }, "stale gradient selection");
    ExpectQuantizedFailure([&]() { runtime.EvaluateQuantizedSplitScan(scan, base); }, "stale gradient scan evaluation");
    CheckQuantizedEvaluation(&runtime, first, candidates, features, fixture, rows, parameters.front(), nullptr);
    // A fresh gradient-generation scan isolates dataset invalidation from the
    // preceding gradient-restage rejection.
    auto fresh_histogram = runtime.BuildQuantizedHistogram(rows.data(), rows.size(), fixture.used);
    runtime.FixQuantizedHistogram(fresh_histogram, totals.first, totals.second);
    auto fresh_scan = runtime.PrepareQuantizedSplitScan(features, totals.first, totals.second, kRows, fresh_histogram);
    auto fresh_evaluation = runtime.EvaluateQuantizedSplitScan(fresh_scan, base);
    runtime.SetHistogramBins(fixture.bins, fixture.offsets, fixture.sizes, fixture.most_frequent);
    ExpectQuantizedFailure([&]() { runtime.EvaluateQuantizedSplitScan(fresh_scan, base); }, "stale dataset scan evaluation");
    if (check_selection)
      ExpectQuantizedFailure([&]() { runtime.SelectQuantizedSplit(fresh_evaluation); }, "stale dataset selection");
    CheckQuantizedEvaluation(&runtime, first, candidates, features, fixture, rows, parameters.front(), nullptr);
    CheckQuantizedEvaluation(&runtime, fresh_evaluation, candidates, features, fixture, rows, base, nullptr);
  }
  if (cpu_winners == 0 || coverage.valid == 0 || coverage.invalid == 0 || coverage.independent_counts == 0 ||
      coverage.cumulative_counts == 0 || coverage.reverse_nan_binary == 0 || coverage.forward == 0 ||
      coverage.clipped_then_smoothed == 0)
    throw std::runtime_error("quantized evaluation CPU/count/direction/smoothing coverage was vacuous");
  std::cout << (check_selection ? "QUANTIZED_SPLIT_SELECTION_PASS cpu_winners_checked=" :
                                  "QUANTIZED_SPLIT_EVALUATION_PASS cpu_winners_checked=") << cpu_winners
            << " valid=" << coverage.valid << " invalid=" << coverage.invalid << '\n';
}

void RunQuantizedSelectionBoundaries() {
  Runtime runtime(0);
  QuantizedFixture fixture;
  fixture.sizes = {5}; fixture.offsets = {0}; fixture.most_frequent = {4}; fixture.used = {1};
  fixture.bins = {{0, 0, 1, 1, 2, 2, 3, 3}};
  fixture.gradients = {-12, -10, -8, 9, 11, 7, 4, 2};
  fixture.hessians.assign(8, 1);
  Runtime::SplitFeature feature;
  feature.feature = 7; feature.histogram_index = 0; feature.num_bin = 5; feature.missing_type = 2;
  std::vector<int32_t> rows{0, 1, 2, 3, 4, 5, 6, 7};
  auto prepare = [&]() {
    PackQuantizedPairs(&fixture);
    runtime.SetHistogramBins(fixture.bins, fixture.offsets, fixture.sizes, fixture.most_frequent);
    runtime.SetQuantizedHistogramGradients(fixture.pairs.data(), rows.size());
    auto histogram = runtime.BuildQuantizedHistogram(rows.data(), rows.size(), fixture.used);
    const auto totals = QuantizedTotals(fixture, rows);
    runtime.FixQuantizedHistogram(histogram, totals.first, totals.second);
    return runtime.PrepareQuantizedSplitScan({feature}, totals.first, totals.second, rows.size(), histogram);
  };
  auto scan = prepare();
  Runtime::QuantizedSplitParameters p;
  p.hessian_scale = 16.0;  // H+epsilon rounds exactly to H; no FP32 intermediate.
  const auto candidates = runtime.ReadQuantizedSplitScan(scan);
  // Both directions have the same threshold/gain, but the restored reverse
  // gain can be equal, one ULP greater, or one ULP smaller. Only the last
  // case must select forward. A direct comparison of unshifted/shifted gains
  // across directions would fail this fixture.
  const double minimum_gains[] = {7.3249496992925085, 1.6038312002323476, 4.916646948843807};
  const int restoration_sign[] = {0, 1, -1};
  for (int i = 0; i < 3; ++i) {
    p.min_gain_to_split = minimum_gains[i];
    Runtime::QuantizedSplitCandidate metadata;
    metadata.feature = feature.feature; metadata.threshold = 0; metadata.default_left = true;
    bool unused = false;
    const auto reference = QuantizedEvaluationOracle(metadata, feature, fixture, rows, p, &unused);
    const double restored = (reference.current_gain - reference.gain_shift) + reference.gain_shift;
    const int sign = (restored > reference.current_gain) - (restored < reference.current_gain);
    if (sign != restoration_sign[i]) throw std::runtime_error("FP64 round-trip fixture lost its boundary");
    auto evaluation = runtime.EvaluateQuantizedSplitScan(scan, p);
    CheckQuantizedEvaluation(&runtime, evaluation, candidates, {feature}, fixture, rows, p, nullptr, true);
    auto selected = runtime.SelectQuantizedSplit(evaluation);
    if (selected.split.default_left != (i != 2) || selected.split.threshold != 0)
      throw std::runtime_error("FP64 quantized direction round-trip rule differs");
    // Mask before multiplying an extreme finite penalty, for both invalid
    // candidates and valid-but-node-disabled candidates.
    auto extreme_features = std::vector<Runtime::SplitFeature>{feature};
    extreme_features[0].penalty = std::numeric_limits<double>::max();
    extreme_features[0].enabled = false;
    const auto totals = QuantizedTotals(fixture, rows);
    auto histogram = runtime.BuildQuantizedHistogram(rows.data(), rows.size(), fixture.used);
    runtime.FixQuantizedHistogram(histogram, totals.first, totals.second);
    auto extreme_scan = runtime.PrepareQuantizedSplitScan(extreme_features, totals.first, totals.second, rows.size(), histogram);
    auto extreme_eval = runtime.EvaluateQuantizedSplitScan(extreme_scan, p);
    if (runtime.SelectQuantizedSplit(extreme_eval).split.feature != -1)
      throw std::runtime_error("disabled huge penalty produced a winner");
    auto rejected = p; rejected.min_data_in_leaf = 9;
    extreme_eval = runtime.EvaluateQuantizedSplitScan(extreme_scan, rejected);
    if (runtime.SelectQuantizedSplit(extreme_eval).split.feature != -1)
      throw std::runtime_error("invalid huge penalty produced a winner");
  }
  // Three equal-gain thresholds in one direction: reverse scanning must keep
  // the highest threshold, rather than the smallest numerical threshold.
  feature.missing_type = 0;
  fixture.bins = {{0, 0, 0, 3, 3, 3, 3, 3}};
  scan = prepare(); p.min_gain_to_split = 0.0;
  auto evaluation = runtime.EvaluateQuantizedSplitScan(scan, p);
  CheckQuantizedEvaluation(&runtime, evaluation, runtime.ReadQuantizedSplitScan(scan),
                           {feature}, fixture, rows, p, nullptr, true);
  if (runtime.SelectQuantizedSplit(evaluation).split.threshold != 2)
    throw std::runtime_error("quantized same-direction equal-gain scan ordering");
  // INT64 winner readback must preserve odd totals beyond FP32's exact range.
  constexpr int kLargeRows = 300003;
  fixture.sizes = {2}; fixture.most_frequent = {1};
  fixture.bins.assign(1, std::vector<uint32_t>(kLargeRows));
  fixture.gradients.resize(kLargeRows); fixture.hessians.assign(kLargeRows, 255);
  rows.resize(kLargeRows); std::iota(rows.begin(), rows.end(), 0);
  for (int i = 0; i < kLargeRows; ++i) {
    fixture.bins[0][i] = i >= 150001;
    fixture.gradients[i] = i < 150001 ? -127 : 127;
  }
  feature.num_bin = 2;
  scan = prepare(); p.hessian_scale = 1.0;
  evaluation = runtime.EvaluateQuantizedSplitScan(scan, p);
  CheckQuantizedEvaluation(&runtime, evaluation, runtime.ReadQuantizedSplitScan(scan),
                           {feature}, fixture, rows, p, nullptr, true);
  const auto selected = runtime.SelectQuantizedSplit(evaluation);
  if (selected.integer_statistics.left_gradient != -int64_t{127} * 150001 ||
      selected.integer_statistics.left_hessian != int64_t{255} * 150001)
    throw std::runtime_error("quantized winner lost large INT64 statistics");
  std::cout << "QUANTIZED_SELECTION_BOUNDARIES_PASS fp64_roundtrip=3 threshold_ties=1 large_integer_winners=1\n";
}

void RunQuantizedHistogramProbe() {
  // Every histogram/device-row owner below is destroyed before this runtime,
  // including when an assertion throws. Old-generation handles stay alive
  // deliberately for the invalid subtraction checks.
  Runtime runtime(0);
  LightGBM::NPUGradientStager quantized_stager(&runtime);
  LightGBM::NPUHistogramConstructor quantized_constructor(&runtime);
  constexpr int kRows = 257, kFeatures = 19;
  QuantizedFixture fixture;
  fixture.bins.assign(kFeatures, std::vector<uint32_t>(kRows));
  fixture.offsets.resize(kFeatures);
  fixture.sizes.resize(kFeatures);
  fixture.most_frequent.resize(kFeatures);
  fixture.used.assign(kFeatures, 1);
  fixture.used[3] = fixture.used[10] = 0;
  fixture.gradients.resize(kRows);
  fixture.hessians.resize(kRows);
  for (int f = 0; f < kFeatures; ++f) {
    // Three tiles of eight features, with a three-feature final tile. CPU
    // metadata omits logical bin zero from storage exactly when MFB is zero.
    fixture.offsets[f] = f % 2;
    fixture.sizes[f] = 3 + f % 13;
    fixture.most_frequent[f] = fixture.offsets[f] ? 0 : 1 + f % (fixture.sizes[f] - 1);
    for (int r = 0; r < kRows; ++r)
      fixture.bins[f][r] = (r * 7 + r / 5 + f * 3) % fixture.sizes[f];
  }
  const int64_t gradient_values[] = {-128, 127, -1, 0, 1, -73, 91};
  const int64_t hessian_values[] = {0, 1, 127, 128, 255};
  for (int r = 0; r < kRows; ++r) {
    fixture.gradients[r] = gradient_values[r % 7];
    fixture.hessians[r] = hessian_values[r % 5];
  }
  PackQuantizedPairs(&fixture);
  runtime.SetHistogramBins(fixture.bins, fixture.offsets, fixture.sizes, fixture.most_frequent);
  ExpectQuantizedFailure([&]() { quantized_stager.StageQuantized(nullptr, kRows); },
                         "stager rejects null nonempty input");
  ExpectQuantizedFailure([&]() { quantized_stager.StageQuantized(nullptr, -1); },
                         "stager rejects negative row count");
  ExpectQuantizedFailure([&]() { quantized_stager.StageQuantized(fixture.pairs.data(), kRows - 1); },
                         "stager rejects dataset row count mismatch");
  quantized_stager.StageQuantized(fixture.pairs.data(), kRows);
  std::vector<int32_t> bag, small, large;
  for (int r = kRows - 1; r >= 0; --r) {
    if (r % 5 == 1 || r % 7 == 2) continue;
    bag.push_back(r);
    (r % 3 == 0 ? small : large).push_back(r);
  }
  auto fix = [&](const Runtime::QuantizedHistogramHandle& handle,
                 const QuantizedFixture& data, const std::vector<int32_t>& rows) {
    const auto totals = QuantizedTotals(data, rows);
    quantized_constructor.FixQuantized(handle, totals.first, totals.second);
  };
  auto verify = [&](const Runtime::QuantizedHistogramHandle& handle,
                    const QuantizedFixture& data, const std::vector<int32_t>& rows,
                    bool fixed, const char* label) {
    CheckQuantized(runtime.ReadQuantizedHistogram(handle), QuantizedOracle(data, rows, fixed), label);
  };
  auto parent = runtime.BuildQuantizedHistogram(bag.data(), bag.size(), fixture.used);
  auto child = runtime.BuildQuantizedHistogram(small.data(), small.size(), fixture.used);
  auto direct_large = runtime.BuildQuantizedHistogram(large.data(), large.size(), fixture.used);
  verify(parent, fixture, bag, false, "raw reversed bag");
  verify(child, fixture, small, false, "raw small child");
  const auto raw = runtime.ReadQuantizedHistogram(parent);
  for (int f = 0; f < kFeatures; ++f) {
    if (!fixture.used[f] || fixture.most_frequent[f] < fixture.offsets[f]) continue;
    const int omitted = fixture.most_frequent[f] - fixture.offsets[f];
    if (raw.gradients[f][omitted] != 0 || raw.hessians[f][omitted] != 0)
      throw std::runtime_error("raw most-frequent bin must be exactly zero before Fix");
  }
  // Supply exactly the same non-contiguous, descending rows through resident
  // device memory; this must not silently fall back to the host-row entry.
  void* raw_rows = nullptr;
  const size_t row_bytes = small.size() * sizeof(int32_t);
  if (aclrtMalloc(&raw_rows, row_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    throw std::runtime_error("allocate quantized device rows");
  std::unique_ptr<void, decltype(&aclrtFree)> device_rows(raw_rows, &aclrtFree);
  if (aclrtMemcpy(device_rows.get(), row_bytes, small.data(), row_bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
    throw std::runtime_error("upload quantized device rows");
  auto resident = quantized_constructor.BuildQuantizedFromDeviceRows(
      static_cast<const int32_t*>(device_rows.get()), small.size(), fixture.used);
  verify(resident, fixture, small, false, "raw resident rows");
  CheckQuantized(runtime.ReadQuantizedHistogram(resident), runtime.ReadQuantizedHistogram(child),
                 "host/resident raw equality");
  const std::vector<int32_t> empty;
  auto empty_host = runtime.BuildQuantizedHistogram(nullptr, 0, fixture.used);
  auto empty_device = runtime.BuildQuantizedHistogramFromDeviceRows(nullptr, 0, fixture.used);
  verify(empty_host, fixture, empty, false, "raw empty host");
  verify(empty_device, fixture, empty, false, "raw empty resident");
  runtime.FixQuantizedHistogram(empty_host, 0, 0);
  runtime.FixQuantizedHistogram(empty_device, 0, 0);
  verify(empty_host, fixture, empty, true, "fixed empty host");
  verify(empty_device, fixture, empty, true, "fixed empty resident");
  fix(parent, fixture, bag);
  fix(child, fixture, small);
  fix(resident, fixture, small);
  fix(direct_large, fixture, large);
  verify(parent, fixture, bag, true, "fixed parent all stored bins");
  verify(child, fixture, small, true, "fixed small all stored bins");
  verify(resident, fixture, small, true, "fixed resident all stored bins");
  verify(direct_large, fixture, large, true, "direct larger child");
  quantized_constructor.SubtractQuantized(parent, child);
  verify(parent, fixture, large, true, "parent overwritten by larger child");
  verify(child, fixture, small, true, "subtraction preserves smaller child");
  CheckQuantized(runtime.ReadQuantizedHistogram(parent), runtime.ReadQuantizedHistogram(direct_large),
                 "subtraction/direct larger equality");
  ExpectQuantizedFailure([&]() { runtime.SubtractQuantizedHistogram(child, child); },
                         "self subtraction");
  verify(child, fixture, small, true, "self rejection preserves child");
  // Tree pruning may remove features between parent and child. Narrow masks
  // safely, but reject reactivation before mutating any histogram contents.
  auto pruned = fixture;
  const auto first_active = std::find(pruned.used.begin(), pruned.used.end(), int8_t{1});
  if (first_active == pruned.used.end()) throw std::runtime_error("pruning fixture has no active feature");
  *first_active = 0;
  auto pruning_parent = runtime.BuildQuantizedHistogram(bag.data(), bag.size(), fixture.used);
  auto pruning_small = runtime.BuildQuantizedHistogram(small.data(), small.size(), pruned.used);
  fix(pruning_parent, fixture, bag); fix(pruning_small, pruned, small);
  runtime.SubtractQuantizedHistogram(pruning_parent, pruning_small);
  verify(pruning_parent, pruned, large, true, "pruned subtraction integer/mask oracle");
  auto forbidden_parent = runtime.BuildQuantizedHistogram(bag.data(), bag.size(), pruned.used);
  fix(forbidden_parent, pruned, bag);
  ExpectQuantizedFailure([&]() { runtime.SubtractQuantizedHistogram(forbidden_parent, child); },
                         "feature mask reactivation");
  verify(forbidden_parent, pruned, bag, true, "mask rejection preserves parent");

  // Reuse the same host pair storage, changing both signed G and unsigned H.
  // Comparing with a fresh host oracle detects stale pointer-keyed uploads.
  for (int r = 0; r < kRows; ++r) {
    fixture.gradients[r] = gradient_values[(r + 3) % 7];
    fixture.hessians[r] = hessian_values[(r + 2) % 5];
  }
  PackQuantizedPairs(&fixture);
  quantized_stager.StageQuantized(fixture.pairs.data(), kRows);
  auto updated = runtime.BuildQuantizedHistogram(bag.data(), bag.size(), fixture.used);
  verify(updated, fixture, bag, false, "updated gradients raw");
  fix(updated, fixture, bag);
  verify(updated, fixture, bag, true, "updated gradients fixed");
  ExpectQuantizedFailure([&]() { runtime.SubtractQuantizedHistogram(updated, child); },
                         "cross gradient epoch subtraction");
  verify(updated, fixture, bag, true, "epoch rejection preserves parent");

  // Reset the dataset with changed bins, offsets and MFBs, not just an
  // identical upload. Keep the preceding fixed handle alive across the reset.
  for (int f = 0; f < kFeatures; ++f) {
    fixture.offsets[f] = 1 - fixture.offsets[f];
    fixture.sizes[f] = 4 + f % 9;
    fixture.most_frequent[f] = fixture.offsets[f] ? 0 : fixture.sizes[f] - 1;
    for (int r = 0; r < kRows; ++r)
      fixture.bins[f][r] = (r * 11 + r / 3 + f + 1) % fixture.sizes[f];
  }
  // Keep used identical for the invalid pair, isolating dataset invalidation.
  runtime.SetHistogramBins(fixture.bins, fixture.offsets, fixture.sizes, fixture.most_frequent);
  quantized_stager.StageQuantized(fixture.pairs.data(), kRows);
  auto reset = runtime.BuildQuantizedHistogram(bag.data(), bag.size(), fixture.used);
  verify(reset, fixture, bag, false, "dataset reset raw");
  fix(reset, fixture, bag);
  verify(reset, fixture, bag, true, "dataset reset fixed");
  ExpectQuantizedFailure([&]() { runtime.SubtractQuantizedHistogram(reset, updated); },
                         "cross dataset subtraction");
  verify(reset, fixture, bag, true, "dataset rejection preserves parent");

  // Exercise the actual float API alongside integer uploads. Fractional FP32
  // inputs cannot pass if either build path accidentally consumes packed G/H
  // or reads the other histogram kind. Handle-type separation is compile-time.
  std::vector<float> float_g(kRows), float_h(kRows);
  for (int r = 0; r < kRows; ++r) {
    float_g[r] = (r % 17 - 8) * 0.125f;
    float_h[r] = (1 + r % 5) * 0.25f;
  }
  runtime.SetHistogramGradients(float_g.data(), float_h.data(), kRows, false);
  auto float_before = runtime.BuildHistogram(bag.data(), bag.size(), fixture.used);
  auto check_float = [&](const Runtime::HistogramHandle& handle) {
    const auto actual = runtime.ReadHistogram(handle);
    if (actual.gradients.size() != kFeatures || actual.hessians.size() != kFeatures)
      throw std::runtime_error("float/quantized regression feature count");
    for (int f = 0; f < kFeatures; ++f) {
      std::vector<double> eg, eh;
      if (fixture.used[f]) {
        eg.assign(fixture.sizes[f] - fixture.offsets[f], 0);
        eh.assign(eg.size(), 0);
        for (int r : bag) {
          const int bin = static_cast<int>(fixture.bins[f][r]);
          if (bin < fixture.offsets[f] || bin == fixture.most_frequent[f]) continue;
          eg[bin - fixture.offsets[f]] += float_g[r];
          eh[bin - fixture.offsets[f]] += float_h[r];
        }
      }
      if (actual.gradients[f] != eg || actual.hessians[f] != eh)
        throw std::runtime_error("float histogram changed by quantized API");
    }
  };
  check_float(float_before);
  fixture.gradients[0] = 127;
  fixture.hessians[0] = 255;
  PackQuantizedPairs(&fixture);
  quantized_stager.StageQuantized(fixture.pairs.data(), kRows);
  auto integer_after_float = runtime.BuildQuantizedHistogram(bag.data(), bag.size(), fixture.used);
  verify(integer_after_float, fixture, bag, false, "quantized after float upload");
  auto float_after = runtime.BuildHistogram(bag.data(), bag.size(), fixture.used);
  check_float(float_before);
  check_float(float_after);
  fix(integer_after_float, fixture, bag);
  verify(integer_after_float, fixture, bag, true, "quantized after float build");

  // Exercise the training-facing components together. CPU discretization is
  // dataset-indexed; a reversed, non-contiguous bag and nested right-child
  // split must gather the original row's bytes, never leaf-ordered bytes.
  LightGBM::NPUDataPartition partition(&runtime);
  for (bool constant : {false, true}) {
    LightGBM::GradientDiscretizer discretizer(16, 4, 17, constant, false);
    // No feature metadata is needed for DiscretizeGradients itself.
    discretizer.Init(kRows, 8, 0, nullptr);
    for (int epoch = 0; epoch < 2; ++epoch) {
      for (int r = 0; r < kRows; ++r) {
        float_g[r] = static_cast<float>((r + epoch) % 17 - 8);
        float_h[r] = constant ? 2.0f : static_cast<float>(1 + (r + epoch) % 4);
        fixture.gradients[r] = static_cast<int64_t>(float_g[r]);
        fixture.hessians[r] = constant ? 1 : static_cast<int64_t>(float_h[r] * 4);
      }
      discretizer.DiscretizeGradients(kRows, float_g.data(), float_h.data());
      const auto* pairs = discretizer.discretized_gradients_and_hessians();
      for (int r = 0; r < kRows; ++r) {
        uint8_t h = 0;
        std::memcpy(&h, pairs + 2 * r, 1);
        if (pairs[2 * r + 1] != fixture.gradients[r] || h != fixture.hessians[r])
          throw std::runtime_error("CPU discretizer byte contract");
      }
      quantized_stager.StageQuantized(pairs, kRows);
      partition.Reset(8);
      partition.InitializeRoot(bag.data(), bag.size());
      const uint64_t downloads_before = runtime.partition_rows_downloaded();
      auto build_leaf = [&](int leaf, const std::vector<int32_t>& expected_rows) {
        const auto& span = partition.Leaf(leaf);
        if (span.count != static_cast<int64_t>(expected_rows.size()))
          throw std::runtime_error("quantized partition leaf count");
        auto histogram = quantized_constructor.BuildQuantizedFromDeviceRows(
            runtime.PartitionRowsDevice(span.partition, span.left), span.count, fixture.used);
        verify(histogram, fixture, expected_rows, false, "component raw histogram");
        fix(histogram, fixture, expected_rows);
        verify(histogram, fixture, expected_rows, true, "component fixed histogram");
        return histogram;
      };
      auto root_histogram = build_leaf(0, bag);
      std::vector<int32_t> left_rows, right_rows, cat_left, cat_right;
      for (int row : bag)
        (fixture.bins[0][row] <= 2 ? left_rows : right_rows).push_back(row);
      partition.Split(0, 0, 2, 0, 0, true, 1);
      auto left_histogram = build_leaf(0, left_rows);
      auto right_histogram = build_leaf(1, right_rows);
      quantized_constructor.SubtractQuantized(root_histogram, left_histogram);
      CheckQuantized(runtime.ReadQuantizedHistogram(root_histogram),
                     runtime.ReadQuantizedHistogram(right_histogram), "component subtraction");
      const auto sibling = partition.Leaf(0).partition;
      for (int row : right_rows)
        (fixture.bins[1][row] == 1 || fixture.bins[1][row] == 3 ? cat_left : cat_right).push_back(row);
      partition.SplitCategorical(1, 1, {1, 3}, 0, 0, false, 2);
      if (partition.Leaf(0).partition != sibling)
        throw std::runtime_error("nested split replaced sibling ownership");
      auto cat_left_histogram = build_leaf(1, cat_left);
      auto cat_right_histogram = build_leaf(2, cat_right);
      quantized_constructor.SubtractQuantized(right_histogram, cat_left_histogram);
      CheckQuantized(runtime.ReadQuantizedHistogram(right_histogram),
                     runtime.ReadQuantizedHistogram(cat_right_histogram), "nested subtraction");
      build_leaf(0, left_rows);
      // Split with a threshold above all bins to create an empty child.
      partition.Split(2, 0, fixture.sizes[0] - 1, 0, 0, true, 3);
      build_leaf(2, cat_right);
      build_leaf(3, {});
      if (runtime.partition_rows_downloaded() != downloads_before)
        throw std::runtime_error("quantized histogram path downloaded leaf rows");
      if (partition.ReadLeafRows(0) != left_rows || partition.ReadLeafRows(1) != cat_left ||
          partition.ReadLeafRows(2) != cat_right || !partition.ReadLeafRows(3).empty())
        throw std::runtime_error("quantized component stable partition order");
      // A fresh staging epoch invalidates old mutation even if row storage is
      // unchanged, while the old histogram remains a readable snapshot.
      quantized_stager.StageQuantized(pairs, kRows);
      ExpectQuantizedFailure([&]() {
        quantized_constructor.SubtractQuantized(root_histogram, left_histogram);
      }, "component stale epoch subtraction");
    }
  }
  std::cout << "QUANTIZED_STAGING_PARTITION_PASS\n";

  // Both final sums are odd and above 2^24, so even FP32 accumulation or a
  // float intermediate/readback cannot pass this integer-only comparison.
  // Feature zero tests accumulation, feature one MFB reconstruction.
  constexpr int kLargeRows = 132105;
  QuantizedFixture big;
  big.bins = {std::vector<uint32_t>(kLargeRows, 2), std::vector<uint32_t>(kLargeRows, 1)};
  big.offsets = {0, 0};
  big.sizes = {3, 3};
  big.most_frequent = {1, 1};
  big.used = {1, 1};
  big.gradients.assign(kLargeRows, 127);
  big.hessians.assign(kLargeRows, 255);
  PackQuantizedPairs(&big);
  std::vector<int32_t> big_rows(kLargeRows);
  std::iota(big_rows.rbegin(), big_rows.rend(), 0);
  runtime.SetHistogramBins(big.bins, big.offsets, big.sizes, big.most_frequent);
  runtime.SetQuantizedHistogramGradients(big.pairs.data(), kLargeRows);
  auto big_histogram = runtime.BuildQuantizedHistogram(big_rows.data(), big_rows.size(), big.used);
  verify(big_histogram, big, big_rows, false, "132105-row raw accumulation");
  fix(big_histogram, big, big_rows);
  verify(big_histogram, big, big_rows, true, "132105-row fixed accumulation");
  std::vector<Runtime::SplitFeature> big_features;
  for (int f = 0; f < 2; ++f) {
    Runtime::SplitFeature feature;
    feature.feature = f;
    feature.histogram_index = f;
    feature.num_bin = big.sizes[f];
    feature.offset = big.offsets[f];
    big_features.push_back(feature);
  }
  auto big_scan = runtime.PrepareQuantizedSplitScan(big_features, int64_t{127} * kLargeRows,
                                                   int64_t{255} * kLargeRows, kLargeRows, big_histogram);
  CheckQuantizedScan(&runtime, big_scan, big_features, big, big_rows);
  const auto big_result = runtime.ReadQuantizedHistogram(big_histogram);
  if (big_result.gradients[0][2] != int64_t{127} * kLargeRows ||
      big_result.hessians[0][2] != int64_t{255} * kLargeRows ||
      big_result.gradients[1][1] != int64_t{127} * kLargeRows ||
      big_result.hessians[1][1] != int64_t{255} * kLargeRows)
    throw std::runtime_error("large-bin integer precision/range regression");
  std::vector<int32_t> big_small(big_rows.begin(), big_rows.begin() + 2);
  std::vector<int32_t> big_large(big_rows.begin() + 2, big_rows.end());
  auto big_child = runtime.BuildQuantizedHistogram(big_small.data(), big_small.size(), big.used);
  auto big_direct = runtime.BuildQuantizedHistogram(big_large.data(), big_large.size(), big.used);
  fix(big_child, big, big_small);
  fix(big_direct, big, big_large);
  runtime.SubtractQuantizedHistogram(big_histogram, big_child);
  verify(big_histogram, big, big_large, true, "large odd-sum subtraction");
  CheckQuantized(runtime.ReadQuantizedHistogram(big_histogram), runtime.ReadQuantizedHistogram(big_direct),
                 "large subtraction/direct equality");

  // Old snapshots carry their own layout metadata and remain readable even
  // after a dataset replacement. They cannot participate in fresh arithmetic.
  const auto old_snapshot = runtime.ReadQuantizedHistogram(big_histogram);
  runtime.SetHistogramBins({}, {}, {});
  quantized_stager.StageQuantized(nullptr, 0);
  auto no_features = runtime.BuildQuantizedHistogram(nullptr, 0, {});
  runtime.FixQuantizedHistogram(no_features, 0, 0);
  CheckQuantized(runtime.ReadQuantizedHistogram(no_features), {}, "zero-feature dataset");
  CheckQuantized(runtime.ReadQuantizedHistogram(big_histogram), old_snapshot, "old snapshot after dataset reset");
  runtime.SetHistogramBins({{}}, {0}, {3}, {1});
  quantized_stager.StageQuantized(nullptr, 0);
  auto zero_rows = runtime.BuildQuantizedHistogramFromDeviceRows(nullptr, 0, {1});
  runtime.FixQuantizedHistogram(zero_rows, 0, 0);
  Runtime::QuantizedHistogramResult zero_expected;
  zero_expected.gradients = {{0, 0, 0}};
  zero_expected.hessians = {{0, 0, 0}};
  CheckQuantized(runtime.ReadQuantizedHistogram(zero_rows), zero_expected, "zero-row dataset");
  runtime.SetHistogramBins(big.bins, big.offsets, big.sizes, big.most_frequent);
  runtime.SetQuantizedHistogramGradients(big.pairs.data(), kLargeRows);
  auto no_mask = runtime.BuildQuantizedHistogram(big_rows.data(), big_rows.size(), {0, 0});
  runtime.FixQuantizedHistogram(no_mask, int64_t{127} * kLargeRows, int64_t{255} * kLargeRows);
  Runtime::QuantizedHistogramResult masked_expected;
  masked_expected.gradients.resize(2);
  masked_expected.hessians.resize(2);
  CheckQuantized(runtime.ReadQuantizedHistogram(no_mask), masked_expected, "all-zero feature mask");

  // Negative odd gradient sums and an omitted-bin NaN forward residual are
  // both above FP32's exact integer range. No dequantized shortcut can pass.
  QuantizedFixture residual_big;
  residual_big.bins = {std::vector<uint32_t>(kLargeRows, 0)};
  residual_big.offsets = {1};
  residual_big.sizes = {4};
  residual_big.most_frequent = {0};
  residual_big.used = {1};
  residual_big.gradients.assign(kLargeRows, -127);
  residual_big.hessians.assign(kLargeRows, 255);
  PackQuantizedPairs(&residual_big);
  runtime.SetHistogramBins(residual_big.bins, residual_big.offsets,
                            residual_big.sizes, residual_big.most_frequent);
  quantized_stager.StageQuantized(residual_big.pairs.data(), kLargeRows);
  auto residual_histogram = runtime.BuildQuantizedHistogram(big_rows.data(), big_rows.size(), residual_big.used);
  fix(residual_histogram, residual_big, big_rows);
  Runtime::SplitFeature residual_feature;
  residual_feature.feature = 0;
  residual_feature.histogram_index = 0;
  residual_feature.num_bin = 4;
  residual_feature.offset = 1;
  residual_feature.missing_type = 2;
  auto residual_scan = runtime.PrepareQuantizedSplitScan({residual_feature},
      -int64_t{127} * kLargeRows, int64_t{255} * kLargeRows, kLargeRows, residual_histogram);
  CheckQuantizedScan(&runtime, residual_scan, {residual_feature}, residual_big, big_rows);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::strcmp(argv[1], "quantized_select") == 0) {
      RunQuantizedSplitEvaluationProbe(true);
      RunQuantizedSelectionBoundaries();
      return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "quantized_eval") == 0) {
      RunQuantizedSplitEvaluationProbe();
      return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "quantized_scan") == 0) {
      RunQuantizedSplitScanProbe();
      return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "quantized") == 0) {
      RunQuantizedHistogramProbe();
      std::cout << "QUANTIZED_HISTOGRAM_PASS\n";
      return 0;
    }
    LightGBM::NPURuntime runtime(0);
    constexpr int rows = 257, features = 19;
    std::vector<std::vector<uint32_t>> bins(features, std::vector<uint32_t>(rows));
    std::vector<int> offsets(features), sizes(features);
    std::vector<int8_t> used(features, 1);
    std::vector<float> g(rows), h(rows);
    for (int f = 0; f < features; ++f) {
      offsets[f] = f % 2;
      sizes[f] = 3 + f % 13;
      for (int r = 0; r < rows; ++r) bins[f][r] = (r * 7 + f) % sizes[f];
    }
    used[3] = used[10] = 0;
    for (int r = 0; r < rows; ++r) {
      g[r] = static_cast<float>(r % 23 - 11) / 8 + static_cast<float>(r) * 0.000137f;
      h[r] = static_cast<float>(1 + r % 7) / 8;
    }
    runtime.SetHistogramBins(bins, offsets, sizes);
    for (int round = 0; round < 4; ++round) {
      const bool constant = round >= 2;
      for (auto& value : g) value += 0.125f;
      if (constant) std::fill(h.begin(), h.end(), round == 2 ? 2.0f : 0.5f);
      runtime.SetHistogramGradients(g.data(), h.data(), rows, constant);
      for (int mode = 0; mode < 3; ++mode) {
        std::vector<int32_t> indices;
        for (int r = rows - 1; r >= 0; --r) {
          if (mode == 0 || (mode == 1 && r % 3 == 0)) indices.push_back(r);
        }
        auto result = runtime.BatchedHistogram(indices.data(), indices.size(), used);
        for (int f = 0; f < features; ++f) {
          if (!used[f]) {
            if (!result.gradients[f].empty()) throw std::runtime_error("feature mask");
            continue;
          }
          std::vector<double> eg(sizes[f] - offsets[f]), eh(eg.size());
          for (int r : indices) {
            int bin = static_cast<int>(bins[f][r]) - offsets[f];
            if (bin >= 0) { eg[bin] += g[r]; eh[bin] += h[r]; }
          }
          for (size_t b = 0; b < eg.size(); ++b) {
            if (std::fabs(eg[b] - result.gradients[f][b]) > 1e-10 ||
                std::fabs(eh[b] - result.hessians[f][b]) > 1e-10) {
              std::cerr << std::setprecision(17) << "round=" << round << " mode=" << mode << " feature=" << f
                        << " bin=" << b << " actual=" << result.gradients[f][b]
                        << "," << result.hessians[f][b] << " expected=" << eg[b] << "," << eh[b] << '\n';
              throw std::runtime_error("histogram mismatch");
            }
          }
        }
      }
    }
    // Dataset reset must invalidate cached constant Hessians and bin tiles.
    runtime.SetHistogramBins(bins, offsets, sizes);
    runtime.SetHistogramGradients(g.data(), h.data(), rows, false);
    // Reconstruct omitted most-frequent bins and subtract a child on device.
    std::vector<int> most_frequent(features);
    for (int f = 0; f < features; ++f) {
      most_frequent[f] = offsets[f] ? 0 : 1 + f % (sizes[f] - 1);
    }
    runtime.SetHistogramBins(bins, offsets, sizes, most_frequent);
    runtime.SetHistogramGradients(g.data(), h.data(), rows, false);
    std::vector<int32_t> all(rows), small, large;
    std::iota(all.begin(), all.end(), 0);
    for (int r : all) (r % 3 == 0 ? small : large).push_back(r);
    auto totals = [&](const std::vector<int32_t>& indices) {
      std::pair<double, double> result{0, 0};
      for (int r : indices) { result.first += g[r]; result.second += h[r]; }
      return result;
    };
    auto parent = runtime.BuildHistogram(all.data(), all.size(), used);
    auto child = runtime.BuildHistogram(small.data(), small.size(), used);
    // A device-side partition will pass this same device pointer directly.
    void* raw_rows = nullptr;
    const size_t row_bytes = small.size() * sizeof(int32_t);
    if (aclrtMalloc(&raw_rows, row_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
      throw std::runtime_error("allocate device leaf rows");
    std::unique_ptr<void, decltype(&aclrtFree)> device_rows(raw_rows, &aclrtFree);
    if (aclrtMemcpy(device_rows.get(), row_bytes, small.data(), row_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
      throw std::runtime_error("upload device leaf rows");
    auto child_from_device = runtime.BuildHistogramFromDeviceRows(
        static_cast<const int32_t*>(device_rows.get()), small.size(), used);
    // Stable numeric partition must preserve the order on both sides, handle
    // the omitted most-frequent bin, and accept the resulting device spans.
    for (int f : {0, 1, 8, 9, 18}) {
      for (int missing_type = 0; missing_type <= 2; ++missing_type) {
        const uint32_t default_bin = most_frequent[f] >= 0
            ? static_cast<uint32_t>(most_frequent[f]) : 0;
        for (bool missing_left : {false, true}) {
          const uint32_t threshold = static_cast<uint32_t>(sizes[f] / 2);
          auto partition = runtime.SplitDeviceRows(
              static_cast<const int32_t*>(device_rows.get()), small.size(), f,
              threshold, default_bin, missing_type, missing_left);
          std::vector<int32_t> expected_left, expected_right;
          for (int32_t row : small) {
            const uint32_t bin = bins[f][row];
            const bool missing = (missing_type == 1 && bin == default_bin) ||
                                 (missing_type == 2 && bin == static_cast<uint32_t>(sizes[f] - 1));
            const bool to_left = missing ? missing_left : bin <= threshold;
            (to_left ? expected_left : expected_right).push_back(row);
          }
          std::vector<int32_t> expected = expected_left;
          expected.insert(expected.end(), expected_right.begin(), expected_right.end());
          if (runtime.LeftCount(partition) != static_cast<int64_t>(expected_left.size()) ||
              runtime.ReadPartitionRows(partition) != expected)
            throw std::runtime_error("device stable partition differs from logical-bin reference");
          if (f == 9 && missing_type == 2 && !missing_left && !expected_left.empty()) {
            std::vector<double> scores(rows), expected_scores(rows);
            for (int r = 0; r < rows; ++r) scores[r] = expected_scores[r] = r * 0.125;
            const double left_output = 0.123456789012345;
            const double right_output = -0.987654321098765;
            const auto downloaded = runtime.partition_rows_downloaded();
            const auto calls = runtime.score_update_calls();
            const auto uploads = runtime.score_uploads();
            const auto score_downloads = runtime.score_downloads();
            for (int repeat = 0; repeat < 2; ++repeat) {
              runtime.AddPredictionToScore({{partition, true, left_output},
                                             {partition, false, right_output}}, rows, scores.data());
              for (int r : expected_left) expected_scores[r] += left_output;
              for (int r : expected_right) expected_scores[r] += right_output;
              for (int r = 0; r < rows; ++r)
                if (std::fabs(scores[r] - expected_scores[r]) > 1e-12)
                  throw std::runtime_error("FP64 device score update mismatch");
            }
            if (runtime.partition_rows_downloaded() != downloaded ||
                runtime.score_update_calls() != calls + 2 ||
                runtime.score_uploads() != uploads + 1 ||
                runtime.score_downloads() != score_downloads + 2)
              throw std::runtime_error("score update did not reuse resident score buffer");
            std::cout << "PASS: FP64 device score updates preserve out-of-bag scores without row downloads\n";
            auto hist = runtime.BuildHistogramFromDeviceRows(
                runtime.PartitionRowsDevice(partition, true), expected_left.size(), used);
            auto actual = runtime.ReadHistogram(hist);
            for (int32_t row : expected_left) {
              const auto bin = bins[0][row];
              if (bin == static_cast<uint32_t>(most_frequent[0])) continue;
              actual.gradients[0][bin] -= g[row];
            }
            for (float value : actual.gradients[0])
              if (std::fabs(value) > 1e-10)
                throw std::runtime_error("partitioned rows cannot build a histogram");
          }
        }
      }
    }
    auto empty_partition = runtime.SplitDeviceRows(nullptr, 0, 0, 0, 0, 0, false);
    if (runtime.LeftCount(empty_partition) != 0 || !runtime.ReadPartitionRows(empty_partition).empty())
      throw std::runtime_error("empty device partition mismatch");
    std::cout << "PASS: stable device rows for numeric/missing splits and resident histogram reuse\n";
    auto empty_from_device = runtime.BuildHistogramFromDeviceRows(nullptr, 0, used);
    auto empty_values = runtime.ReadHistogram(empty_from_device);
    for (int f = 0; f < features; ++f)
      for (auto value : empty_values.gradients[f])
        if (value != 0.0f) throw std::runtime_error("device empty leaf is not zero");
    auto p = totals(all), s = totals(small);
    runtime.FixHistogram(parent, p.first, p.second);
    runtime.FixHistogram(child, s.first, s.second);
    runtime.FixHistogram(child_from_device, s.first, s.second);
    auto verify = [&](const LightGBM::NPURuntime::HistogramHandle& device,
                      const std::vector<int32_t>& indices) {
      auto actual = runtime.ReadHistogram(device);
      std::vector<LightGBM::NPURuntime::SplitFeature> reference;
      for (int f = 0; f < features; ++f) {
        if (!used[f]) continue;
        LightGBM::NPURuntime::SplitFeature meta;
        meta.feature = features - 1 - f;
        meta.histogram_index = f;
        meta.num_bin = sizes[f]; meta.offset = offsets[f];
        meta.missing_type = f % 3; meta.default_bin = f % sizes[f];
        meta.gradients.assign(sizes[f] - offsets[f], 0);
        meta.hessians.assign(meta.gradients.size(), 0);
        for (int r : indices) {
          int b = static_cast<int>(bins[f][r]) - offsets[f];
          if (b >= 0) { meta.gradients[b] += g[r]; meta.hessians[b] += h[r]; }
        }
        for (size_t b = 0; b < meta.gradients.size(); ++b) {
          if (std::fabs(actual.gradients[f][b] - meta.gradients[b]) > 1e-10 ||
              std::fabs(actual.hessians[f][b] - meta.hessians[b]) > 1e-10) {
            std::cerr << "fix/subtract feature=" << f << " bin=" << b
                      << " rows=" << indices.size() << " most_frequent=" << most_frequent[f]
                      << " actual=" << actual.gradients[f][b] << "," << actual.hessians[f][b]
                      << " expected=" << meta.gradients[b] << "," << meta.hessians[b] << '\n';
            throw std::runtime_error("device fix/subtract mismatch");
          }
        }
        reference.push_back(std::move(meta));
      }
      auto total = totals(indices);
      auto expected = runtime.FindBestSplit(reference, total.first, total.second,
                                            indices.size(), 0.2, 0.4, 3, 0.001, 0.0);
      auto metadata = reference;
      for (auto& f : metadata) { f.gradients.clear(); f.hessians.clear(); }
      auto found = runtime.FindBestSplit(metadata, total.first, total.second,
                                         indices.size(), 0.2, 0.4, 3, 0.001, 0.0, device);
      // The AI Core SIMT finder evaluates the split in FP32 (precision class
      // P1) while this reference path accumulates in double. The selected
      // split stays exact; only derived statistics may drift, and the gain is
      // a difference of O(sum^2/hess) terms, so FP32 rounding is amplified to
      // roughly 1e-5 relative. Union-disable the finder for bit-exact checks.
      const bool simt_enabled = std::getenv("LGBM_NPU_DISABLE_SIMT_SPLIT") == nullptr;
      const double stat_tolerance = simt_enabled ? 1e-4 : 1e-8;
      auto close = [stat_tolerance](double a, double b) {
        return std::fabs(a - b) <=
               stat_tolerance * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
      };
      if (found.feature != expected.feature || found.threshold != expected.threshold ||
          found.default_left != expected.default_left || found.splittable != expected.splittable ||
          found.left_count != expected.left_count || found.right_count != expected.right_count ||
          !close(found.gain, expected.gain) ||
          !close(found.left_gradient, expected.left_gradient) ||
          !close(found.left_hessian, expected.left_hessian) ||
          !close(found.left_output, expected.left_output) ||
          !close(found.right_output, expected.right_output))
        throw std::runtime_error("resident split differs from host histogram input");
    };
    verify(parent, all);
    verify(child, small);
    verify(child_from_device, small);
    runtime.SubtractHistogram(parent, child);
    verify(parent, large);
    std::cout << "PASS: device fix, parent-child subtraction and direct split input (all missing types)\n";
    std::cout << "PASS: 12 leaf cases, 19 features, masks, offsets, partial tile, empty leaf, gradient updates and reset\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}