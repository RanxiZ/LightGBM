/* Copyright (c) 2026. Licensed under the MIT License. */
// Capability + numerical-fidelity probe for the operators the device-side
// split search needs. Each operation runs in its own process so that an
// unsupported dtype cannot hide a later failure. No host-compute fallback.
#include "npu_runtime.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_gather.h>
#include <aclnnop/aclnn_ge_tensor.h>
#include <aclnnop/aclnn_gt_tensor.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_trunc.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(aclError status, const std::string& what) {
  if (status != ACL_SUCCESS) {
    const char* detail = aclGetRecentErrMsg();
    throw std::runtime_error(what + " error=" + std::to_string(status) + " " +
                             (detail != nullptr ? detail : "no detail"));
  }
}

// Owns every device allocation and handle so a throw cannot leak them.
class Probe {
 public:
  Probe() { Check(aclrtCreateStream(&stream_), "create stream"); }
  ~Probe() {
    aclrtSynchronizeStream(stream_);
    for (auto* t : tensors_) aclDestroyTensor(t);
    for (auto* b : buffers_) aclrtFree(b);
    aclrtDestroyStream(stream_);
  }
  Probe(const Probe&) = delete;
  Probe& operator=(const Probe&) = delete;

  aclrtStream stream() const { return stream_; }

  void* Allocate(size_t bytes) {
    void* pointer = nullptr;
    Check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
    buffers_.push_back(pointer);
    return pointer;
  }

  template <typename T>
  aclTensor* Make(const std::vector<T>& host, aclDataType dtype,
                  std::vector<int64_t> shape, void** out_device = nullptr) {
    const size_t bytes = host.size() * sizeof(T);
    void* device = Allocate(bytes);
    Check(aclrtMemcpy(device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
          "upload");
    if (out_device != nullptr) *out_device = device;
    shapes_.push_back(std::move(shape));
    const std::vector<int64_t>& dims = shapes_.back();
    std::vector<int64_t> strides(dims.size());
    int64_t running = 1;
    for (size_t i = dims.size(); i > 0; --i) {
      strides[i - 1] = running;
      running *= dims[i - 1];
    }
    strides_.push_back(std::move(strides));
    aclTensor* tensor = aclCreateTensor(
        dims.data(), static_cast<uint64_t>(dims.size()), dtype,
        strides_.back().data(), 0, ACL_FORMAT_ND, dims.data(),
        static_cast<uint64_t>(dims.size()), device);
    if (tensor == nullptr) throw std::runtime_error("aclCreateTensor returned null");
    tensors_.push_back(tensor);
    return tensor;
  }

  // Two-stage CANN contract: size, allocate workspace, launch, synchronise.
  template <typename SizeFn, typename RunFn>
  void Launch(SizeFn size_fn, RunFn run_fn, const std::string& name) {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    Check(size_fn(&workspace_size, &executor), name + " workspace");
    void* workspace = workspace_size > 0 ? Allocate(workspace_size) : nullptr;
    Check(run_fn(workspace, workspace_size, executor, stream_), name + " launch");
    Check(aclrtSynchronizeStream(stream_), name + " synchronize");
  }

  template <typename T>
  std::vector<T> Download(void* device, size_t count) {
    std::vector<T> host(count);
    Check(aclrtMemcpy(host.data(), count * sizeof(T), device, count * sizeof(T),
                      ACL_MEMCPY_DEVICE_TO_HOST), "download");
    return host;
  }

 private:
  aclrtStream stream_ = nullptr;
  std::vector<void*> buffers_;
  std::vector<aclTensor*> tensors_;
  std::vector<std::vector<int64_t>> shapes_;
  std::vector<std::vector<int64_t>> strides_;
};

void Expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

int64_t UlpDistance(double a, double b) {
  if (a == b) return 0;
  int64_t ia = 0, ib = 0;
  std::memcpy(&ia, &a, sizeof(double));
  std::memcpy(&ib, &b, sizeof(double));
  if ((ia < 0) != (ib < 0)) return INT64_MAX;
  return ia > ib ? ia - ib : ib - ia;
}

// The split search rests entirely on prefix sums, so compare the device scan
// against the exact accumulation order the CPU tree learner uses.
void ProbeCumsumFidelity(LightGBM::NPURuntime* runtime) {
  const int64_t rows = 3, columns = 512;
  std::mt19937_64 engine(42);
  std::uniform_real_distribution<double> magnitude(-3.0, 3.0);
  std::vector<double> input(static_cast<size_t>(rows * columns));
  for (auto& value : input) {
    // Shape resembles histogram bins: mixed signs, several decades of scale.
    value = magnitude(engine) * std::pow(10.0, magnitude(engine));
  }
  const auto device = runtime->PrefixSum(input, rows, columns);
  Expect(device.size() == input.size(), "prefix size mismatch");

  int64_t worst_ulp = 0, mismatched = 0;
  double worst_relative = 0.0;
  for (int64_t row = 0; row < rows; ++row) {
    double running = 0.0;
    for (int64_t column = 0; column < columns; ++column) {
      const size_t index = static_cast<size_t>(row * columns + column);
      running += input[index];  // Same left-to-right order as the CPU loop.
      const int64_t ulp = UlpDistance(running, device[index]);
      if (ulp != 0) {
        ++mismatched;
        worst_ulp = std::max(worst_ulp, ulp);
        const double scale = std::fabs(running) > 0.0 ? std::fabs(running) : 1.0;
        worst_relative = std::max(worst_relative,
                                  std::fabs(running - device[index]) / scale);
      }
    }
  }
  std::cout << "cumsum_elements=" << input.size() << std::endl;
  std::cout << "cumsum_mismatched=" << mismatched << std::endl;
  std::cout << "cumsum_worst_ulp=" << worst_ulp << std::endl;
  std::cout << "cumsum_worst_relative=" << worst_relative << std::endl;
  std::cout << "cumsum_bitwise_identical=" << (mismatched == 0 ? 1 : 0) << std::endl;
  // A non-zero count is reported, not failed: it decides whether the device
  // search can claim bit-identical thresholds or only numerical equivalence.
  Expect(worst_relative < 1e-12, "device prefix sum is not numerically usable");
}

void ProbeOperator(const std::string& op) {
  Probe probe;
  const std::vector<double> left{3.0, -4.0, 0.5, 2.0};
  const std::vector<double> right{3.0, 1.0, 2.0, -7.5};
  const std::vector<uint8_t> flags_a{1, 0, 1, 0};
  const std::vector<uint8_t> flags_b{1, 1, 0, 0};
  const std::vector<int64_t> shape{4};

  if (op == "ge" || op == "gt") {
    const std::vector<uint8_t> zeros(4, 0);
    void* out_device = nullptr;
    auto* a = probe.Make(left, ACL_DOUBLE, shape);
    auto* b = probe.Make(right, ACL_DOUBLE, shape);
    auto* out = probe.Make(zeros, ACL_BOOL, shape, &out_device);
    if (op == "ge") {
      probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnGeTensorGetWorkspaceSize(a, b, out, size, executor);
      }, aclnnGeTensor, "GeTensor");
    } else {
      probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnGtTensorGetWorkspaceSize(a, b, out, size, executor);
      }, aclnnGtTensor, "GtTensor");
    }
    const auto actual = probe.Download<uint8_t>(out_device, 4);
    const std::vector<uint8_t> expected =
        op == "ge" ? std::vector<uint8_t>{1, 0, 0, 1}
                   : std::vector<uint8_t>{0, 0, 0, 1};
    for (size_t i = 0; i < 4; ++i) {
      std::cout << op << "[" << i << "]=" << static_cast<int>(actual[i]) << std::endl;
      Expect(actual[i] == expected[i], op + " value mismatch");
    }
  } else if (op == "where") {
    const std::vector<double> zeros(4, 0.0);
    void* out_device = nullptr;
    auto* condition = probe.Make(flags_a, ACL_BOOL, shape);
    auto* a = probe.Make(left, ACL_DOUBLE, shape);
    auto* b = probe.Make(right, ACL_DOUBLE, shape);
    auto* out = probe.Make(zeros, ACL_DOUBLE, shape, &out_device);
    probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
      return aclnnSWhereGetWorkspaceSize(condition, a, b, out, size, executor);
    }, aclnnSWhere, "SWhere");
    const auto actual = probe.Download<double>(out_device, 4);
    const std::vector<double> expected{3.0, 1.0, 0.5, -7.5};
    for (size_t i = 0; i < 4; ++i) {
      std::cout << "where[" << i << "]=" << actual[i] << std::endl;
      Expect(actual[i] == expected[i], "where value mismatch");
    }
  }
 else if (op == "and" || op == "or_not") {
    const std::vector<uint8_t> zeros(4, 0);
    void* out_device = nullptr;
    auto* a = probe.Make(flags_a, ACL_BOOL, shape);
    auto* b = probe.Make(flags_b, ACL_BOOL, shape);
    auto* out = probe.Make(zeros, ACL_BOOL, shape, &out_device);
    if (op == "and") {
      probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnLogicalAndGetWorkspaceSize(a, b, out, size, executor);
      }, aclnnLogicalAnd, "LogicalAnd");
    } else {
      probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnLogicalNotGetWorkspaceSize(a, out, size, executor);
      }, aclnnLogicalNot, "LogicalNot");
    }
    const auto actual = probe.Download<uint8_t>(out_device, 4);
    const std::vector<uint8_t> expected =
        op == "and" ? std::vector<uint8_t>{1, 0, 0, 0}
                    : std::vector<uint8_t>{0, 1, 0, 1};
    for (size_t i = 0; i < 4; ++i) {
      std::cout << op << "[" << i << "]=" << static_cast<int>(actual[i]) << std::endl;
      Expect(actual[i] == expected[i], op + " value mismatch");
    }
  } else if (op == "trunc") {
    // Mirrors Common::RoundInt, which truncates hess * cnt_factor + 0.5.
    const std::vector<double> values{2.5, 2.4999999999, -0.5, 19.5000001};
    const std::vector<double> zeros(4, 0.0);
    void* out_device = nullptr;
    auto* input = probe.Make(values, ACL_DOUBLE, shape);
    auto* out = probe.Make(zeros, ACL_DOUBLE, shape, &out_device);
    probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
      return aclnnTruncGetWorkspaceSize(input, out, size, executor);
    }, aclnnTrunc, "Trunc");
    const auto actual = probe.Download<double>(out_device, 4);
    const std::vector<double> expected{2.0, 2.0, -0.0, 19.0};
    for (size_t i = 0; i < 4; ++i) {
      std::cout << "trunc[" << i << "]=" << actual[i] << std::endl;
      Expect(actual[i] == expected[i], "trunc value mismatch");
    }
  } else if (op == "floor" || op == "cast_i32" || op == "cast_i64") {
    // Alternatives to aclnnTrunc, which rejects FP64 on this platform.
    // Common::RoundInt is static_cast<int>(x + 0.5), i.e. truncation toward
    // zero, so only non-negative inputs may use floor.
    const std::vector<double> values{2.5, 2.9999999999, 0.5, 19.5000001};
    void* out_device = nullptr;
    auto* input = probe.Make(values, ACL_DOUBLE, shape);
    const std::vector<double> expected{2.0, 2.0, 0.0, 19.0};
    if (op == "floor") {
      const std::vector<double> zeros(4, 0.0);
      auto* out = probe.Make(zeros, ACL_DOUBLE, shape, &out_device);
      probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnFloorGetWorkspaceSize(input, out, size, executor);
      }, aclnnFloor, "Floor");
      const auto actual = probe.Download<double>(out_device, 4);
      for (size_t i = 0; i < 4; ++i) {
        std::cout << "floor[" << i << "]=" << actual[i] << std::endl;
        Expect(actual[i] == expected[i], "floor value mismatch");
      }
    } else if (op == "cast_i32") {
      const std::vector<int32_t> zeros(4, 0);
      auto* out = probe.Make(zeros, ACL_INT32, shape, &out_device);
      probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnCastGetWorkspaceSize(input, ACL_INT32, out, size, executor);
      }, aclnnCast, "Cast");
      const auto actual = probe.Download<int32_t>(out_device, 4);
      for (size_t i = 0; i < 4; ++i) {
        std::cout << "cast_i32[" << i << "]=" << actual[i] << std::endl;
        Expect(actual[i] == static_cast<int32_t>(expected[i]), "cast value mismatch");
      }
    } else {
      const std::vector<int64_t> zeros(4, 0);
      auto* out = probe.Make(zeros, ACL_INT64, shape, &out_device);
      probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnCastGetWorkspaceSize(input, ACL_INT64, out, size, executor);
      }, aclnnCast, "Cast");
      const auto actual = probe.Download<int64_t>(out_device, 4);
      for (size_t i = 0; i < 4; ++i) {
        std::cout << "cast_i64[" << i << "]=" << actual[i] << std::endl;
        Expect(actual[i] == static_cast<int64_t>(expected[i]), "cast value mismatch");
      }
    }
  } else if (op == "gather") {
    const std::vector<double> table{10.0, 20.0, 30.0, 40.0, 50.0};
    const std::vector<int64_t> index{4, 0, 2, 2};
    const std::vector<double> zeros(4, 0.0);
    void* out_device = nullptr;
    auto* source = probe.Make(table, ACL_DOUBLE, {5});
    auto* indices = probe.Make(index, ACL_INT64, shape);
    auto* out = probe.Make(zeros, ACL_DOUBLE, shape, &out_device);
    probe.Launch([&](uint64_t* size, aclOpExecutor** executor) {
      return aclnnGatherGetWorkspaceSize(source, 0, indices, out, size, executor);
    }, aclnnGather, "Gather");
    const auto actual = probe.Download<double>(out_device, 4);
    const std::vector<double> expected{50.0, 10.0, 30.0, 30.0};
    for (size_t i = 0; i < 4; ++i) {
      std::cout << "gather[" << i << "]=" << actual[i] << std::endl;
      Expect(actual[i] == expected[i], "gather value mismatch");
    }
  } else {
    throw std::runtime_error("unknown operation " + op);
  }
}

// The flattened winner selection relies on argmax returning the earliest
// index on ties, over a row-major buffer viewed as one long vector.
void ProbeFlatFirstMax(Probe* probe) {
  const double neg_inf = -std::numeric_limits<double>::infinity();
  const int64_t rows = 6, width = 64;
  std::vector<double> scores(static_cast<size_t>(rows * width), neg_inf);
  // Equal best gains in two different rows: the earlier row must win.
  scores[2 * width + 10] = 5.5;
  scores[4 * width + 3] = 5.5;
  scores[2 * width + 40] = 5.5;
  scores[1 * width + 7] = 5.0;
  void* device = nullptr;
  auto* flat = probe->Make(scores, ACL_DOUBLE, {rows * width}, &device);
  const std::vector<int64_t> zero{0};
  void* index_device = nullptr;
  auto* index = probe->Make(zero, ACL_INT64, {1}, &index_device);
  probe->Launch([&](uint64_t* size, aclOpExecutor** executor) {
    return aclnnArgMaxGetWorkspaceSize(flat, 0, false, index, size, executor);
  }, aclnnArgMax, "ArgMax");
  const auto actual = probe->Download<int64_t>(index_device, 1);
  std::cout << "flat_first_max=" << actual[0] << std::endl;
  Expect(actual[0] == 2 * width + 10,
         "flattened argmax did not return the earliest tied index");
}

// Invalid candidates are masked with -infinity, so argmax must survive a long
// array that is mostly -infinity and must return the first maximum on ties.
void ProbeMaskedArgMax(LightGBM::NPURuntime* runtime) {
  const double neg_inf = -std::numeric_limits<double>::infinity();
  std::vector<double> scores(4096, neg_inf);
  scores[1500] = 7.25;
  scores[2500] = 7.25;
  scores[900] = 7.0;
  const int64_t index = runtime->ArgMax(scores);
  std::cout << "masked_argmax=" << index << std::endl;
  Expect(index == 1500, "masked argmax did not return the first maximum");

  std::vector<double> all_invalid(1024, neg_inf);
  const int64_t none = runtime->ArgMax(all_invalid);
  std::cout << "all_invalid_argmax=" << none << std::endl;
  Expect(none == 0, "all-invalid argmax should return index 0");
}
}  // namespace

// One operation per process: an unsupported dtype fails loudly on its own.
int main(int argc, char** argv) {
  const std::string op = argc > 1 ? argv[1] : "cumsum";
  try {
    LightGBM::NPURuntime runtime(0);
    if (op == "cumsum") {
      ProbeCumsumFidelity(&runtime);
    } else if (op == "masked_argmax") {
      ProbeMaskedArgMax(&runtime);
    } else {
      ProbeOperator(op);
    }
    std::cout << "PROBE_PASS " << op << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PROBE_FAIL " << op << " " << error.what() << std::endl;
    return 1;
  }
}