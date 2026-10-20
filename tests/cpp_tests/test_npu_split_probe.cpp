#include "npu_runtime.h"
#include <LightGBM/meta.h>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>
#include <acl/acl.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_square.h>
#include <aclnnop/aclnn_sub.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(aclError status, const std::string& operation) {
  if (status != ACL_SUCCESS) {
    const char* detail = aclGetRecentErrMsg();
    throw std::runtime_error(operation + " error=" + std::to_string(status) +
                             " " + (detail != nullptr ? detail : "no detail"));
  }
}

// Standalone capability checks; no Python dispatcher or host-compute fallback.
struct ProbeResources {
  std::vector<void*> buffers;
  std::vector<aclTensor*> tensors;
  aclScalar* scalar = nullptr;
  aclrtStream stream = nullptr;
  ~ProbeResources() {
    if (stream != nullptr) {
      aclrtSynchronizeStream(stream);
    }
    for (auto tensor : tensors) aclDestroyTensor(tensor);
    if (scalar != nullptr) aclDestroyScalar(scalar);
    for (auto buffer : buffers) aclrtFree(buffer);
    if (stream != nullptr) aclrtDestroyStream(stream);
  }
  void* Allocate(size_t bytes) {
    void* pointer = nullptr;
    Check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
    buffers.push_back(pointer);
    return pointer;
  }
  aclTensor* Tensor(const std::vector<double>& values) {
    void* pointer = Allocate(values.size() * sizeof(double));
    Check(aclrtMemcpy(pointer, values.size() * sizeof(double), values.data(),
                       values.size() * sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE), "upload");
    const int64_t size = static_cast<int64_t>(values.size());
    const int64_t stride = 1;
    auto tensor = aclCreateTensor(&size, 1, ACL_DOUBLE, &stride, 0, ACL_FORMAT_ND, &size, 1, pointer);
    if (tensor == nullptr) throw std::runtime_error("create FP64 tensor");
    tensors.push_back(tensor);
    return tensor;
  }
};

void ProbePrimitive(const std::string& op) {
  ProbeResources resources;
  Check(aclrtCreateStream(&resources.stream), "create stream");
  const auto input = resources.Tensor({3.0, -4.0, 0.5, 2.0});
  const auto second = resources.Tensor({2.0, 3.0, 2.0, 3.0});
  const auto output = resources.Tensor({0.0, 0.0, 0.0, 0.0});
  void* output_buffer = resources.buffers.back();
  double one = 1.0;
  resources.scalar = aclCreateScalar(&one, ACL_DOUBLE);
  if (resources.scalar == nullptr) throw std::runtime_error("create FP64 scalar");
  uint64_t bytes = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status;
  aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream) = nullptr;
  std::vector<double> expected;
  if (op == "abs") {
    status = aclnnAbsGetWorkspaceSize(input, output, &bytes, &executor);
    run = aclnnAbs; expected = {3.0, 4.0, 0.5, 2.0};
  } else if (op == "sign") {
    status = aclnnSignGetWorkspaceSize(input, output, &bytes, &executor);
    run = aclnnSign; expected = {1.0, -1.0, 1.0, 1.0};
  } else if (op == "clamp") {
    status = aclnnClampMinGetWorkspaceSize(input, resources.scalar, output, &bytes, &executor);
    run = aclnnClampMin; expected = {3.0, 1.0, 1.0, 2.0};
  } else if (op == "adds") {
    status = aclnnAddsGetWorkspaceSize(input, resources.scalar, resources.scalar, output, &bytes, &executor);
    run = aclnnAdds; expected = {4.0, -3.0, 1.5, 3.0};
  } else if (op == "subs") {
    status = aclnnSubsGetWorkspaceSize(input, resources.scalar, resources.scalar, output, &bytes, &executor);
    run = aclnnSubs; expected = {2.0, -5.0, -0.5, 1.0};
  } else if (op == "mul") {
    status = aclnnMulGetWorkspaceSize(input, second, output, &bytes, &executor);
    run = aclnnMul; expected = {6.0, -12.0, 1.0, 6.0};
  } else if (op == "div") {
    status = aclnnDivGetWorkspaceSize(input, second, output, &bytes, &executor);
    run = aclnnDiv; expected = {1.5, -4.0 / 3.0, 0.25, 2.0 / 3.0};
  } else if (op == "neg") {
    status = aclnnNegGetWorkspaceSize(input, output, &bytes, &executor);
    run = aclnnNeg; expected = {-3.0, 4.0, -0.5, -2.0};
  } else if (op == "square") {
    status = aclnnSquareGetWorkspaceSize(input, output, &bytes, &executor);
    run = aclnnSquare; expected = {9.0, 16.0, 0.25, 4.0};
  } else {
    throw std::runtime_error("unknown operation " + op);
  }
  Check(status, op + " workspace FP64");
  void* workspace = bytes > 0 ? resources.Allocate(bytes) : nullptr;
  Check(run(workspace, bytes, executor, resources.stream), op + " launch FP64");
  Check(aclrtSynchronizeStream(resources.stream), op + " synchronize FP64");
  std::vector<double> actual(4);
  Check(aclrtMemcpy(actual.data(), 4 * sizeof(double), output_buffer, 4 * sizeof(double),
                     ACL_MEMCPY_DEVICE_TO_HOST), "download");
  for (size_t i = 0; i < actual.size(); ++i) {
    std::cout << op << "[" << i << "]=" << actual[i] << std::endl;
    if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 1e-12) {
      throw std::runtime_error(op + " FP64 value mismatch");
    }
  }
}
void ProbeFusedNumericSplit(LightGBM::NPURuntime* runtime) {
  // Compare the resident fused path with the existing ACL numeric scanner.
  // Host aggregation below is a test oracle only, never a training fallback.
  const std::vector<float> gradients{-3, -3, -2, -2, -1, -1, 1, 1, 2, 2, 3, 3};
  const std::vector<float> hessians(gradients.size(), 1.0f);
  std::vector<uint32_t> bins(gradients.size());
  for (size_t row = 0; row < bins.size(); ++row) bins[row] = static_cast<uint32_t>(row / 3);
  std::vector<int32_t> rows(bins.size());
  std::iota(rows.begin(), rows.end(), 0);
  for (int offset : {0, 1}) {
    for (int missing : {0, 1, 2}) {
      runtime->SetHistogramBins({bins, bins, bins}, {offset, offset, offset}, {4, 4, 4});
      runtime->SetHistogramGradients(gradients.data(), hessians.data(), rows.size(), false);
      auto histogram = runtime->BuildHistogram(rows.data(), rows.size(), {1, 1, 1});
      runtime->FixHistogram(histogram, 0.0, 12.0);
      for (int mode = 0; mode < 5; ++mode) {
        std::vector<LightGBM::NPURuntime::SplitFeature> features(3);
        LightGBM::Random fused_random(193), reference_random(193);
        for (int feature = 0; feature < 3; ++feature) {
          auto& f = features[feature];
          f.feature = feature == 0 ? 7 : (feature == 1 ? 2 : 9);
          f.histogram_index = feature;
          f.num_bin = 4;
          f.offset = offset;
          f.default_bin = 1;
          f.missing_type = missing;
          f.enabled = feature != 2;
          f.penalty = mode == 3 ? 0.0 : 1.0;
          f.gradients.assign(4 - offset, 0.0);
          f.hessians.assign(4 - offset, 0.0);
          for (size_t row = 0; row < bins.size(); ++row) {
            if (bins[row] < static_cast<uint32_t>(offset)) continue;
            f.gradients[bins[row] - offset] += gradients[row];
            f.hessians[bins[row] - offset] += hessians[row];
          }
          if (mode == 2) f.extra_tree_random = &fused_random;
        }
        const double l1 = mode == 1 ? 0.5 : 0.0;
        const double max_delta = mode == 1 ? 0.2 : 0.0;
        const double smoothing = mode == 1 ? 2.0 : 0.0;
        const int minimum_count = mode == 4 ? 7 : 1;
        auto fused = runtime->FindBestSplit(features, 0.0, 12.0, 12, l1, 1.0,
            minimum_count, 1e-3, 0.0, histogram, max_delta, smoothing, 0.3);
        for (auto& f : features) {
          if (mode == 2) f.extra_tree_random = &reference_random;
        }
        auto reference = runtime->FindBestSplit(features, 0.0, 12.0, 12, l1, 1.0,
            minimum_count, 1e-3, 0.0, nullptr, max_delta, smoothing, 0.3);
        if (fused.feature != reference.feature || fused.threshold != reference.threshold ||
            fused.default_left != reference.default_left ||
            fused.left_count != reference.left_count || fused.right_count != reference.right_count ||
            fused.splittable != reference.splittable) {
          throw std::runtime_error("fused numeric metadata differs from ACL reference: offset=" +
              std::to_string(offset) + " missing=" + std::to_string(missing) +
              " mode=" + std::to_string(mode));
        }
        const double actual[]{fused.gain, fused.left_output, fused.right_output,
            fused.left_gradient, fused.right_gradient, fused.left_hessian, fused.right_hessian};
        const double expected[]{reference.gain, reference.left_output, reference.right_output,
            reference.left_gradient, reference.right_gradient,
            reference.left_hessian, reference.right_hessian};
        for (size_t stat = 0; stat < sizeof(actual) / sizeof(actual[0]); ++stat) {
          if (!std::isfinite(actual[stat]) ||
              std::fabs(actual[stat] - expected[stat]) > 1e-12 * (1.0 + std::fabs(expected[stat]))) {
            throw std::runtime_error("fused numeric statistics differ from ACL reference");
          }
        }
        if (mode == 2 && fused_random.NextInt(0, 100000) != reference_random.NextInt(0, 100000)) {
          throw std::runtime_error("fused numeric RNG draw order differs from ACL reference");
        }
      }
    }
  }
  // Exercise the FP64 subtract/add direction decision on the resident path,
  // not just on the no-histogram ACL scanner covered by the "split" probe.
  std::vector<float> direction_gradients(20, 0.0f), direction_hessians(20, 1.0f);
  std::vector<uint32_t> direction_bins(20, 0);
  std::vector<int32_t> direction_rows(20);
  direction_gradients[0] = -30.0f;
  direction_gradients[10] = 7.0f;
  std::fill(direction_bins.begin() + 10, direction_bins.end(), 1);
  std::iota(direction_rows.begin(), direction_rows.end(), 0);
  runtime->SetHistogramBins({direction_bins}, {0}, {3});
  runtime->SetHistogramGradients(direction_gradients.data(), direction_hessians.data(), 20, false);
  auto direction_histogram = runtime->BuildHistogram(direction_rows.data(), 20, {1});
  runtime->FixHistogram(direction_histogram, -23.0, 20.0);
  LightGBM::NPURuntime::SplitFeature direction;
  direction.feature = 0;
  direction.histogram_index = 0;
  direction.num_bin = 3;
  direction.missing_type = 2;
  auto rounded = runtime->FindBestSplit({direction}, -23.0, 20.0, 20,
      0.0, 0.0, 1, 1e-3, 0.0, direction_histogram);
  if (rounded.feature != 0 || rounded.threshold != 0 || rounded.default_left ||
      rounded.left_count != 10 || rounded.right_count != 10) {
    throw std::runtime_error("fused FP64 scan-direction round trip differs from CPU");
  }
  // CPU reconstructs right statistics from total minus stored left, rather
  // than returning the reverse prefix directly. The small right gradient
  // disappears in this round trip when the total spans a large FP32 range.
  const std::vector<float> wide_gradients{1e16f, 1.0f};
  const std::vector<float> wide_hessians{1.0f, 1.0f};
  const int32_t wide_rows[]{0, 1};
  const double wide_sum = static_cast<double>(wide_gradients[0]) + wide_gradients[1];
  runtime->SetHistogramBins({{0, 1}}, {0}, {2});
  runtime->SetHistogramGradients(wide_gradients.data(), wide_hessians.data(), 2, false);
  auto wide_histogram = runtime->BuildHistogram(wide_rows, 2, {1});
  runtime->FixHistogram(wide_histogram, wide_sum, 2.0);
  LightGBM::NPURuntime::SplitFeature wide;
  wide.feature = 0;
  wide.histogram_index = 0;
  wide.num_bin = 2;
  wide.gradients = {static_cast<double>(wide_gradients[0]), 1.0};
  wide.hessians = {1.0, 1.0};
  for (const auto& handle : {wide_histogram, LightGBM::NPURuntime::HistogramHandle{}}) {
    auto best = runtime->FindBestSplit({wide}, wide_sum, 2.0, 2,
        0.0, 0.0, 1, 1e-3, 0.0, handle);
    if (best.feature != 0 || best.threshold != 0 ||
        best.right_gradient != 0.0 || best.right_output != 0.0) {
      throw std::runtime_error("large-range right-statistic round trip differs from CPU");
    }
  }
}
}  // namespace

// Run each operation in a fresh process so unsupported dtype failures are isolated.
int main(int argc, char** argv) {
  try {
    LightGBM::NPURuntime runtime(0);
    const std::string op = argc > 1 ? argv[1] : "prefix";
    if (op == "prefix") {
      auto out = runtime.PrefixSum({1.25, -0.5, 2.0, 0.125, 4.0, -2.0}, 2, 3);
      const std::vector<double> expected{1.25, 0.75, 2.75, 0.125, 4.125, 2.125};
      for (size_t i = 0; i < out.size(); ++i) {
        std::cout << "prefix[" << i << "]=" << out[i] << std::endl;
        if (!std::isfinite(out[i]) || std::fabs(out[i] - expected[i]) > 1e-12) {
          throw std::runtime_error("FP64 prefix mismatch");
        }
      }
    } else if (op == "gain" || op == "gain_l1") {
      const double l1 = op == "gain_l1" ? 1.0 : 0.0;
      auto out = runtime.SplitGains({3.0, -4.0}, {2.0, 3.0}, {-3.0, 4.0}, {2.0, 3.0}, l1, 1.0);
      const std::vector<double> expected{2.0 * (3.0-l1) * (3.0-l1) / 3.0,
                                         2.0 * (4.0-l1) * (4.0-l1) / 4.0};
      for (size_t i = 0; i < expected.size(); ++i) {
        std::cout << "gain[" << i << "]=" << out.gain[i] << std::endl;
        if (!std::isfinite(out.gain[i]) || std::fabs(out.gain[i] - expected[i]) > 1e-12) {
          throw std::runtime_error("FP64 gain mismatch");
        }
      }
    } else if (op == "argmax") {
      auto index = runtime.ArgMax({1.0, 4.0, 4.0, -2.0});
      std::cout << "argmax=" << index << std::endl;
      if (index != 1) throw std::runtime_error("FP64 argmax first-tie mismatch");
    } else if (op == "fused_split") {
      ProbeFusedNumericSplit(&runtime);
    } else if (op == "split") {
      LightGBM::NPURuntime::SplitFeature feature;
      feature.feature = 0;
      feature.num_bin = 4;
      feature.gradients = {-30.0, -20.0, 20.0, 30.0};
      feature.hessians = {10.0, 10.0, 10.0, 10.0};
      auto best = runtime.FindBestSplit({feature}, 0.0, 40.0, 40,
                                        5.0, 1.0, 5, 1e-3, 0.0);
      if (best.feature != 0 || best.threshold != 1 || !best.default_left ||
          best.left_count != 20 || best.right_count != 20 ||
          best.splittable != std::vector<uint8_t>{1} ||
          std::fabs(best.gain - 2.0 * 45.0 * 45.0 / 21.0) > 1e-8) {
        throw std::runtime_error("NPU L1 split differs from CPU reference");
      }
      feature.offset = 1;
      feature.gradients.erase(feature.gradients.begin());
      feature.hessians.erase(feature.hessians.begin());
      auto omitted = runtime.FindBestSplit({feature}, 0.0, 40.0, 40,
                                           5.0, 1.0, 5, 1e-3, 0.0);
      if (omitted.feature != 0 || omitted.threshold != 1 ||
          omitted.left_count != 20 || omitted.right_count != 20) {
        throw std::runtime_error("NPU omitted-bin split mismatch");
      }
      feature.offset = 0;
      feature.gradients = {-30.0, -20.0, 20.0, 30.0};
      feature.hessians = {10.0, 10.0, 10.0, 10.0};
      feature.feature = 7;
      auto smaller = feature;
      smaller.feature = 2;
      auto tied = runtime.FindBestSplit({feature, smaller}, 0.0, 40.0, 40,
                                        5.0, 1.0, 5, 1e-3, 0.0);
      if (tied.feature != 2) throw std::runtime_error("cross-feature tie mismatch");
      // The CPU's forward scan compares current_gain with
      // (reverse_gain - parent_shift) + parent_shift. That FP64 round trip
      // rounds down here, so equal unshifted gains select default_right.
      LightGBM::NPURuntime::SplitFeature rounded_direction;
      rounded_direction.feature = 0;
      rounded_direction.num_bin = 3;
      rounded_direction.missing_type = 2;  // NaN, empty missing bin.
      rounded_direction.gradients = {-30.0, 7.0, 0.0};
      rounded_direction.hessians = {10.0, 10.0, 0.0};
      const double total_h = 20.0 + 2.0 * LightGBM::kEpsilon;
      const double right_h = 10.0 + LightGBM::kEpsilon;
      const double current_gain = 900.0 / (total_h - right_h) + 49.0 / right_h;
      const double parent_shift = 529.0 / total_h;
      if (!(current_gain > (current_gain - parent_shift) + parent_shift)) {
        throw std::runtime_error("direction fixture does not exercise FP64 round trip");
      }
      auto rounded = runtime.FindBestSplit({rounded_direction}, -23.0, 20.0, 20,
                                           0.0, 0.0, 1, 1e-3, 0.0);
      if (rounded.feature != 0 || rounded.threshold != 0 || rounded.default_left ||
          rounded.left_count != 10 || rounded.right_count != 10 ||
          std::fabs(rounded.gain - (current_gain - parent_shift)) > 1e-12) {
        throw std::runtime_error("FP64 scan-direction comparison differs from CPU");
      }
      // A large total swallows epsilon. The empty right bin of an invalid
      // candidate then has G=H=0; vectorized search must not leak its NaN
      // through arithmetic masks and hide the valid split.
      auto empty_residual = rounded_direction;
      empty_residual.num_bin = 3;
      empty_residual.missing_type = 0;
      empty_residual.hessians = {1000.0, 1000.0, 0.0};
      auto safe_empty = runtime.FindBestSplit({empty_residual}, -23.0, 2000.0, 2000,
                                             0.0, 0.0, 1, 1e-3, 0.0);
      if (safe_empty.feature != 0 || safe_empty.threshold != 0 ||
          safe_empty.left_count != 1000 || safe_empty.right_count != 1000 ||
          !std::isfinite(safe_empty.gain)) {
        throw std::runtime_error("empty residual candidate polluted split selection");
      }
      auto impossible = runtime.FindBestSplit({feature}, 0.0, 40.0, 40,
                                              5.0, 1.0, 21, 1e-3, 0.0);
      if (impossible.feature != -1 || impossible.splittable != std::vector<uint8_t>{0}) {
        throw std::runtime_error("invalid split was accepted");
      }
    } else {
      ProbePrimitive(op);
    }
    std::cout << "PROBE_PASS " << op << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PROBE_FAIL " << error.what() << std::endl;
    return 1;
  }
}