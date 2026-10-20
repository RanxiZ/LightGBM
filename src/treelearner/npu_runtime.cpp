/* Copyright (c) 2026. Licensed under the MIT License. */
#include "npu_runtime.h"
#include "npu_fused_split.h"
#include "npu_fused_histogram.h"

#include <LightGBM/utils/log.h>
#include <LightGBM/meta.h>

#include <acl/acl.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_eq_scalar.h>
#include <aclnnop/aclnn_le_scalar.h>
#include <aclnnop/aclnn_logical_or.h>
#include <aclnnop/aclnn_scatter.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_ge_scalar.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_any.h>
#include <aclnnop/aclnn_masked_fill_scalar.h>
#include <aclnnop/aclnn_index_select.h>
#include <aclnnop/aclnn_maximum.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_scatter_add.h>
#include <aclnnop/aclnn_expand.h>
#include <aclnnop/aclnn_bincount.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_square.h>
#include <aclnnop/aclnn_sub.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <cstdlib>
#include <map>
#include <mutex>
#include <numeric>

namespace LightGBM {

namespace {

#define ACL_CHECK(expr)                                                  \
  do {                                                                   \
    aclError code_ = (expr);                                             \
    if (code_ != ACL_SUCCESS) {                                          \
      const char* detail_ = aclGetRecentErrMsg();                        \
      Log::Fatal("NPU call failed (%s) at %s:%d with error %d (%s)",     \
                 #expr, __FILE__, __LINE__, static_cast<int>(code_),     \
                 detail_ != nullptr ? detail_ : "no detail");            \
    }                                                                    \
  } while (0)

// ACL is process-global. Reference-count it so a refit or reset can create a
// second runtime without calling aclInit() while another runtime is alive.
struct AclRuntimeState {
  std::mutex mutex;
  std::map<int, int> device_users;
  bool initialized = false;
};

AclRuntimeState& GetAclRuntimeState() {
  static AclRuntimeState state;
  return state;
}

void AcquireAclRuntime(int device) {
  auto& state = GetAclRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.initialized) {
    ACL_CHECK(aclInit(nullptr));
    state.initialized = true;
  }
  ACL_CHECK(aclrtSetDevice(device));
  ++state.device_users[device];
}

void ReleaseAclRuntime(int device) {
  auto& state = GetAclRuntimeState();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto it = state.device_users.find(device);
  if (it != state.device_users.end()) {
    if (--it->second == 0) {
      state.device_users.erase(it);
      // The caller has already destroyed its context and stream.
      ACL_CHECK(aclrtResetDevice(device));
    }
  }
  if (state.device_users.empty() && state.initialized) {
    ACL_CHECK(aclFinalize());
    state.initialized = false;
  }
}

// Device allocation that frees itself, so a Log::Fatal in the middle of an
// operator sequence cannot leak device memory for the rest of the process.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes_ > 0) {
      ACL_CHECK(aclrtMalloc(&data_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
    }
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      aclrtFree(data_);
    }
  }
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        aclrtFree(data_);
      }
      data_ = other.data_;
      bytes_ = other.bytes_;
      other.data_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data() const { return data_; }
  size_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
};

class TensorHandle {
 public:
  TensorHandle() = default;
  TensorHandle(void* data, const std::vector<int64_t>& shape, aclDataType dtype) {
    strides_.resize(shape.size());
    int64_t running = 1;
    for (size_t i = shape.size(); i > 0; --i) {
      strides_[i - 1] = running;
      running *= shape[i - 1];
    }
    shape_ = shape;
    tensor_ = aclCreateTensor(shape_.data(), static_cast<uint64_t>(shape_.size()),
                              dtype, strides_.data(), 0, ACL_FORMAT_ND,
                              shape_.data(), static_cast<uint64_t>(shape_.size()),
                              data);
    if (tensor_ == nullptr) {
      Log::Fatal("NPU call failed: aclCreateTensor returned null");
    }
  }
  ~TensorHandle() {
    if (tensor_ != nullptr) {
      aclDestroyTensor(tensor_);
    }
  }
  TensorHandle(TensorHandle&& other) noexcept
      : tensor_(other.tensor_), shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)) {
    other.tensor_ = nullptr;
  }
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;

  aclTensor* get() const { return tensor_; }

 private:
  aclTensor* tensor_ = nullptr;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
};

// Scalars are tiny but still need explicit release on every path.
class ScalarHandle {
 public:
  ScalarHandle(double value, aclDataType dtype) {
    // aclCreateScalar reads the bytes in the declared dtype; passing a
    // double address for ACL_FLOAT turns e.g. 1.0 into a float zero.
    if (dtype == ACL_FLOAT) {
      float fp32 = static_cast<float>(value);
      scalar_ = aclCreateScalar(&fp32, dtype);
    } else if (dtype == ACL_DOUBLE) {
      scalar_ = aclCreateScalar(&value, dtype);
    } else {
      Log::Fatal("NPU floating scalar requires ACL_FLOAT or ACL_DOUBLE");
    }
    if (scalar_ == nullptr) {
      Log::Fatal("NPU call failed: aclCreateScalar returned null");
    }
  }
  ~ScalarHandle() {
    if (scalar_ != nullptr) {
      aclDestroyScalar(scalar_);
    }
  }
  ScalarHandle(const ScalarHandle&) = delete;
  ScalarHandle& operator=(const ScalarHandle&) = delete;
  aclScalar* get() const { return scalar_; }

 private:
  aclScalar* scalar_ = nullptr;
};

class IntScalarHandle {
 public:
  explicit IntScalarHandle(int32_t value) {
    scalar_ = aclCreateScalar(&value, ACL_INT32);
    if (scalar_ == nullptr) Log::Fatal("NPU call failed: aclCreateScalar returned null");
  }
  ~IntScalarHandle() { if (scalar_ != nullptr) aclDestroyScalar(scalar_); }
  IntScalarHandle(const IntScalarHandle&) = delete;
  IntScalarHandle& operator=(const IntScalarHandle&) = delete;
  aclScalar* get() const { return scalar_; }
 private:
  aclScalar* scalar_ = nullptr;
};

}  // namespace

struct NPURuntime::DeviceHistogram {
  DeviceBuffer values;  // [gradient/hessian, feature, stored-bin], FP64 on device.
  int features = 0, stride = 0;
  std::vector<int8_t> used;
  bool fixed = false;
};

struct NPURuntime::QuantizedDeviceHistogram {
  DeviceBuffer values;  // [gradient/Hessian, feature, stored-bin], INT64 lanes.
  const NPURuntime* owner = nullptr;
  uint64_t dataset_generation = 0, gradient_generation = 0;
  int features = 0, stride = 0;
  int64_t count = 0;
  std::vector<int8_t> used;
  std::vector<int> offsets, bin_counts, most_frequent;
  bool fixed = false;
};

struct NPURuntime::QuantizedDeviceSplitScan {
  struct Row {
    int feature_index;
    int64_t begin, length;
    bool reverse;
  };
  DeviceBuffer values;  // [left G, left H, right G, right H, candidate], INT64.
  const NPURuntime* owner = nullptr;
  uint64_t dataset_generation = 0, gradient_generation = 0;
  std::vector<QuantizedSplitCandidate> candidates;  // metadata only until readback
  std::vector<SplitFeature> features;
  std::vector<Row> rows;
  int64_t sum_gradient = 0, sum_hessian = 0;
  int num_data = 0;
};

struct NPURuntime::QuantizedDeviceSplitEvaluation {
  // Independent result snapshot. Only the diagnostic readback materializes
  // integer candidates or floating evaluation values on the host.
  QuantizedSplitScanHandle scan;
  const NPURuntime* owner = nullptr;
  // G/H (4), final counts (2), scan counts (2), outputs (2), gain/shift (2).
  DeviceBuffer values;
  DeviceBuffer valid;
};

struct NPURuntime::DevicePartition {
  DeviceBuffer rows;  // [left rows, right rows], stable within each side.
  int64_t count = 0;
  int64_t left_count = 0;
};

struct NPURuntime::Impl {
  int device = 0;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  static constexpr int kHistogramTile = 8;
  int64_t histogram_rows = 0;
  int histogram_stride = 0;
  std::vector<int> histogram_offsets, histogram_bin_counts, histogram_most_frequent;
  std::vector<DeviceBuffer> histogram_tiles;
  DeviceBuffer histogram_tile_pointers;
  DeviceBuffer histogram_gradients, histogram_hessians;
  DeviceBuffer quantized_gradients, quantized_hessians;
  uint64_t dataset_generation = 0, quantized_generation = 0;
  bool quantized_ready = false;
  DeviceBuffer partition_positions;
  DeviceBuffer workspace;

  struct ScoreBuffer {
    int64_t num_data = 0;
    DeviceBuffer device;
    std::vector<double> host_shadow;
  };
  // One resident score vector per host score slice. The pointer is the
  // identity of a LightGBM score slice (including a multiclass slice).
  std::map<const double*, ScoreBuffer> score_buffers;

  // Reuse the largest workspace across launches. Each launch is still
  // synchronized until all temporary tensors have stream-ordered lifetimes.
  template <typename SizeFn, typename RunFn>
  void Launch(SizeFn size_fn, RunFn run_fn) {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(size_fn(&workspace_size, &executor));
    if (workspace.bytes() < workspace_size) workspace = DeviceBuffer(workspace_size);
    ACL_CHECK(run_fn(workspace.data(), workspace_size, executor, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  DeviceBuffer Upload(const void* host, size_t bytes) {
    DeviceBuffer buffer(bytes);
    if (bytes == 0) return buffer;
    ACL_CHECK(aclrtMemcpy(buffer.data(), bytes, host, bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    return buffer;
  }

  void Download(void* host, const DeviceBuffer& buffer, size_t bytes) {
    if (bytes == 0) return;
    ACL_CHECK(aclrtMemcpy(host, bytes, buffer.data(), bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
  }

  DeviceBuffer Zeros(size_t bytes) {
    DeviceBuffer buffer(bytes);
    if (bytes == 0) return buffer;
    ACL_CHECK(aclrtMemset(buffer.data(), bytes, 0, bytes));
    return buffer;
  }
};

NPURuntime::NPURuntime(int device) : impl_(new Impl) {
  impl_->device = device;
  AcquireAclRuntime(device);
  ACL_CHECK(aclrtCreateContext(&impl_->context, device));
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  ACL_CHECK(aclrtCreateStream(&impl_->stream));
  Log::Info("NPU runtime initialised on device %d", device);
}

NPURuntime::~NPURuntime() {
  // Resident allocations must be freed while their device context is alive.
  aclrtSetCurrentContext(impl_->context);
  // Resident score buffers must be released while the context is current.
  impl_->score_buffers.clear();
  impl_->histogram_tile_pointers = DeviceBuffer();
  impl_->histogram_tiles.clear();
  impl_->histogram_gradients = DeviceBuffer();
  impl_->histogram_hessians = DeviceBuffer();
  impl_->quantized_gradients = DeviceBuffer();
  impl_->quantized_hessians = DeviceBuffer();
  impl_->quantized_ready = false;
  impl_->partition_positions = DeviceBuffer();
  impl_->workspace = DeviceBuffer();
  if (impl_->stream != nullptr) {
    aclrtDestroyStream(impl_->stream);
  }
  if (impl_->context != nullptr) {
    aclrtDestroyContext(impl_->context);
  }
  ReleaseAclRuntime(impl_->device);
}

void NPURuntime::SetHistogramBins(const std::vector<std::vector<uint32_t>>& bins,
                                  const std::vector<int>& offsets,
                                  const std::vector<int>& num_bins,
                                  const std::vector<int>& most_frequent_bins) {
  CHECK_EQ(bins.size(), offsets.size());
  CHECK_EQ(bins.size(), num_bins.size());
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  // A new dataset invalidates all host-pointer keyed score vectors and
  // prevents integer histogram subtraction across incompatible layouts.
  ++impl_->dataset_generation;
  impl_->score_buffers.clear();
  impl_->histogram_tile_pointers = DeviceBuffer();
  impl_->histogram_tiles.clear();
  impl_->histogram_gradients = DeviceBuffer();
  impl_->histogram_hessians = DeviceBuffer();
  impl_->quantized_gradients = DeviceBuffer();
  impl_->quantized_hessians = DeviceBuffer();
  impl_->quantized_ready = false;
  impl_->partition_positions = DeviceBuffer();
  impl_->histogram_most_frequent = most_frequent_bins;
  if (impl_->histogram_most_frequent.empty()) {
    impl_->histogram_most_frequent.assign(bins.size(), -1);
  }
  CHECK_EQ(impl_->histogram_most_frequent.size(), bins.size());
  impl_->histogram_rows = bins.empty() ? 0 : static_cast<int64_t>(bins.front().size());
  impl_->histogram_offsets = offsets;
  impl_->histogram_bin_counts = num_bins;
  impl_->histogram_stride = 1;
  for (size_t feature = 0; feature < bins.size(); ++feature) {
    CHECK_EQ(static_cast<int64_t>(bins[feature].size()), impl_->histogram_rows);
    impl_->histogram_stride = std::max(impl_->histogram_stride, num_bins[feature]);
  }
  const int64_t sentinel = Impl::kHistogramTile * impl_->histogram_stride;
  for (size_t first = 0; first < bins.size(); first += Impl::kHistogramTile) {
    const size_t tile = std::min(static_cast<size_t>(Impl::kHistogramTile), bins.size() - first);
    std::vector<int32_t> packed(tile * static_cast<size_t>(impl_->histogram_rows));
    for (size_t feature = 0; feature < tile; ++feature) {
      const size_t global = first + feature;
      const int offset = offsets[global];
      CHECK(offset == 0 || offset == 1);
      for (int64_t row = 0; row < impl_->histogram_rows; ++row) {
        const int64_t bin = bins[global][static_cast<size_t>(row)];
        const int64_t value = bin < offset || bin >= num_bins[global] ||
                                  bin == impl_->histogram_most_frequent[global]
                                  ? sentinel : static_cast<int64_t>(feature) * impl_->histogram_stride + bin - offset;
        CHECK_LE(value, std::numeric_limits<int32_t>::max());
        packed[feature * static_cast<size_t>(impl_->histogram_rows) + static_cast<size_t>(row)] =
            static_cast<int32_t>(value);
      }
    }
    impl_->histogram_tiles.emplace_back(impl_->Upload(packed.data(), packed.size() * sizeof(int32_t)));
  }
  std::vector<const int32_t*> tile_pointers;
  tile_pointers.reserve(impl_->histogram_tiles.size());
  for (const auto& tile : impl_->histogram_tiles) {
    tile_pointers.push_back(static_cast<const int32_t*>(tile.data()));
  }
  impl_->histogram_tile_pointers = impl_->Upload(
      tile_pointers.data(), tile_pointers.size() * sizeof(const int32_t*));
  CHECK_LE(impl_->histogram_rows, std::numeric_limits<int32_t>::max());
  if (impl_->histogram_rows > 0) {
    std::vector<int32_t> positions(static_cast<size_t>(impl_->histogram_rows));
    std::iota(positions.begin(), positions.end(), 0);
    impl_->partition_positions = impl_->Upload(positions.data(), positions.size() * sizeof(int32_t));
  }
}

void NPURuntime::SetHistogramGradients(const float* gradients, const float* hessians,
                                       int64_t num_data, bool constant_hessian) {
  CHECK_EQ(num_data, impl_->histogram_rows);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  const size_t bytes = static_cast<size_t>(num_data) * sizeof(float);
  // A constant Hessian need not equal one (custom objectives and weights).
  // Upload the actual values once per tree for both objective types.
  (void)constant_hessian;
  if (impl_->histogram_gradients.bytes() != bytes) {
    impl_->histogram_gradients = DeviceBuffer(bytes);
    impl_->histogram_hessians = DeviceBuffer(bytes);
  }
  ACL_CHECK(aclrtMemcpy(impl_->histogram_gradients.data(), bytes, gradients, bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(impl_->histogram_hessians.data(), bytes, hessians, bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
}

NPURuntime::HistogramHandle NPURuntime::BuildHistogram(
    const int32_t* row_indices, int64_t count, const std::vector<int8_t>& used) {
  CHECK_EQ(used.size(), impl_->histogram_bin_counts.size());
  CHECK_GE(count, 0);
  CHECK_LE(count, impl_->histogram_rows);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  if (count == 0 || used.empty()) {
    return BuildHistogramFromDeviceRows(nullptr, count, used);
  }
  CHECK(row_indices != nullptr);
  DeviceBuffer d_rows = impl_->Upload(row_indices, static_cast<size_t>(count) * sizeof(int32_t));
  return BuildHistogramFromDeviceRows(static_cast<const int32_t*>(d_rows.data()), count, used);
}

NPURuntime::HistogramHandle NPURuntime::BuildHistogramFromDeviceRows(
    const int32_t* device_row_indices, int64_t count, const std::vector<int8_t>& used) {
  CHECK_EQ(used.size(), impl_->histogram_bin_counts.size());
  CHECK_GE(count, 0);
  CHECK_LE(count, impl_->histogram_rows);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  auto result = std::make_shared<DeviceHistogram>();
  result->features = static_cast<int>(used.size());
  result->stride = impl_->histogram_stride;
  result->used = used;
  const int64_t plane = static_cast<int64_t>(result->features) * result->stride;
  result->values = impl_->Zeros(2 * plane * sizeof(double));
  ++histogram_calls_;
  if (count == 0 || used.empty()) return result;
  CHECK(device_row_indices != nullptr);
  CHECK(impl_->histogram_gradients.data() != nullptr);
  CHECK(impl_->histogram_hessians.data() != nullptr);
  // Default to the fused path validated by the full Alpha158 workflow.
  // It replaces gather/expand/cast/scatter chains with one INLINE AICPU
  // launch; FP64 rows still execute serially within each selected feature.
  // Set LGBM_NPU_DISABLE_FUSED_HISTOGRAM to retain the ACL operator path.
  if (std::getenv("LGBM_NPU_DISABLE_FUSED_HISTOGRAM") == nullptr) {
    std::vector<LGBMNPUFusedHistogramFeature> features;
    for (size_t i = 0; i < used.size(); ++i) {
      if (!used[i]) continue;
      LGBMNPUFusedHistogramFeature feature = {};
      feature.output_index = static_cast<int32_t>(i);
      feature.tile_index = static_cast<int32_t>(i / Impl::kHistogramTile);
      feature.tile_feature = static_cast<int32_t>(i % Impl::kHistogramTile);
      feature.num_bins = impl_->histogram_bin_counts[i];
      feature.offset = impl_->histogram_offsets[i];
      features.push_back(feature);
    }
    if (features.empty()) return result;
    DeviceBuffer metadata = impl_->Upload(features.data(), features.size() * sizeof(features[0]));
    DeviceBuffer completion = impl_->Zeros(sizeof(int32_t));
    LGBMNPUFusedHistogramKernel(1, nullptr, impl_->stream,
        static_cast<const int32_t* const*>(impl_->histogram_tile_pointers.data()),
        device_row_indices, static_cast<const float*>(impl_->histogram_gradients.data()),
        static_cast<const float*>(impl_->histogram_hessians.data()),
        static_cast<const LGBMNPUFusedHistogramFeature*>(metadata.data()),
        static_cast<double*>(result->values.data()), static_cast<int32_t>(features.size()),
        result->features, count, static_cast<int32_t>(impl_->histogram_rows),
        result->stride, Impl::kHistogramTile, static_cast<int32_t*>(completion.data()));
    ACL_CHECK(aclrtSynchronizeStream(impl_->stream));
    int32_t completed = 0;
    impl_->Download(&completed, completion, sizeof(completed));
    if (completed != 1) Log::Fatal("NPU fused histogram kernel did not complete");
    return result;
  }
  TensorHandle t_rows(const_cast<int32_t*>(device_row_indices), {count}, ACL_INT32);
  // Gather objective values in their native FP32 format. Convert weights to
  // FP64 before ScatterAdd: Bincount only accepts FP32 weights on this device,
  // whose rounding can change categorical ordering and equal-gain tie breaks.
  DeviceBuffer selected[2] = {DeviceBuffer(count * sizeof(float)), DeviceBuffer(count * sizeof(float))};
  for (int part = 0; part < 2; ++part) {
    DeviceBuffer& values = part == 0 ? impl_->histogram_gradients : impl_->histogram_hessians;
    TensorHandle t_all(values.data(), {impl_->histogram_rows}, ACL_FLOAT);
    TensorHandle t_selected(selected[part].data(), {count}, ACL_FLOAT);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(t_all.get(), 0, t_rows.get(), t_selected.get(), s, e);
    }, aclnnIndexSelect);
  }
  const int64_t output_length = static_cast<int64_t>(Impl::kHistogramTile) * impl_->histogram_stride + 1;
  for (size_t first = 0; first < used.size(); first += Impl::kHistogramTile) {
    const int64_t tile = static_cast<int64_t>(std::min(static_cast<size_t>(Impl::kHistogramTile), used.size() - first));
    bool needed = false;
    for (int64_t j = 0; j < tile; ++j) needed |= used[first + j] != 0;
    if (!needed) continue;
    const int64_t elements = tile * count;
    DeviceBuffer d_bins(elements * sizeof(int32_t));
    TensorHandle t_bins_in(impl_->histogram_tiles[first / Impl::kHistogramTile].data(),
                           {tile, impl_->histogram_rows}, ACL_INT32);
    TensorHandle t_bins(d_bins.data(), {tile, count}, ACL_INT32);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(t_bins_in.get(), 1, t_rows.get(), t_bins.get(), s, e);
    }, aclnnIndexSelect);
    TensorHandle t_bins_int32(d_bins.data(), {elements}, ACL_INT32);
    DeviceBuffer d_indices(elements * sizeof(int64_t));
    TensorHandle t_bins_flat(d_indices.data(), {elements}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnCastGetWorkspaceSize(t_bins_int32.get(), ACL_INT64, t_bins_flat.get(), s, e);
    }, aclnnCast);
    for (int part = 0; part < 2; ++part) {
      DeviceBuffer d_repeated(elements * sizeof(float));
      TensorHandle t_selected_row(selected[part].data(), {1, count}, ACL_FLOAT);
      TensorHandle t_repeated(d_repeated.data(), {tile, count}, ACL_FLOAT);
      const int64_t dims[] = {tile, count};
      aclIntArray* shape = aclCreateIntArray(dims, 2);
      CHECK(shape != nullptr);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnExpandGetWorkspaceSize(t_selected_row.get(), shape, t_repeated.get(), s, e);
      }, aclnnExpand);
      aclDestroyIntArray(shape);
      TensorHandle t_weights_fp32(d_repeated.data(), {elements}, ACL_FLOAT);
      DeviceBuffer d_weights(elements * sizeof(double));
      TensorHandle t_weights(d_weights.data(), {elements}, ACL_DOUBLE);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnCastGetWorkspaceSize(t_weights_fp32.get(), ACL_DOUBLE, t_weights.get(), s, e);
      }, aclnnCast);
      DeviceBuffer d_zero = impl_->Zeros(static_cast<size_t>(output_length) * sizeof(double));
      DeviceBuffer d_hist(static_cast<size_t>(output_length) * sizeof(double));
      TensorHandle t_zero(d_zero.data(), {output_length}, ACL_DOUBLE);
      TensorHandle t_hist(d_hist.data(), {output_length}, ACL_DOUBLE);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnScatterAddGetWorkspaceSize(t_zero.get(), 0, t_bins_flat.get(),
                                                t_weights.get(), t_hist.get(), s, e);
      }, aclnnScatterAdd);
      auto* dst = static_cast<double*>(result->values.data()) + part * plane + first * result->stride;
      // The extra slot receives omitted bins. Only valid bins are retained.
      const size_t tile_bytes = static_cast<size_t>(tile * result->stride) * sizeof(double);
      ACL_CHECK(aclrtMemcpy(dst, tile_bytes, d_hist.data(), tile_bytes,
                            ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
  }
  return result;
}

void NPURuntime::SetQuantizedHistogramGradients(const int8_t* pairs, int64_t num_data) {
  CHECK_EQ(num_data, impl_->histogram_rows);
  CHECK_GE(num_data, 0);
  CHECK(num_data == 0 || pairs != nullptr);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  impl_->quantized_ready = false;
  ++impl_->quantized_generation;
  if (num_data == 0) {
    impl_->quantized_gradients = DeviceBuffer();
    impl_->quantized_hessians = DeviceBuffer();
    impl_->quantized_ready = true;
    return;
  }
  // Interpret each lane independently: Hessian byte values 128..255 are
  // unsigned, whereas negative gradient bytes must sign-extend on device.
  const size_t bytes = static_cast<size_t>(num_data);
  DeviceBuffer packed = impl_->Upload(pairs, 2 * bytes);
  DeviceBuffer gradient_bytes(bytes), hessian_bytes(bytes);
  TensorHandle source(packed.data(), {num_data, 2}, ACL_INT8);
  int64_t hessian_index = 0, gradient_index = 1;
  DeviceBuffer hi = impl_->Upload(&hessian_index, sizeof(int64_t));
  DeviceBuffer gi = impl_->Upload(&gradient_index, sizeof(int64_t));
  TensorHandle h_index(hi.data(), {1}, ACL_INT64), g_index(gi.data(), {1}, ACL_INT64);
  TensorHandle h_byte(hessian_bytes.data(), {num_data, 1}, ACL_INT8);
  TensorHandle g_byte(gradient_bytes.data(), {num_data, 1}, ACL_INT8);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnIndexSelectGetWorkspaceSize(source.get(), 1, h_index.get(), h_byte.get(), s, e);
  }, aclnnIndexSelect);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnIndexSelectGetWorkspaceSize(source.get(), 1, g_index.get(), g_byte.get(), s, e);
  }, aclnnIndexSelect);
  impl_->quantized_gradients = DeviceBuffer(bytes * sizeof(int64_t));
  impl_->quantized_hessians = DeviceBuffer(bytes * sizeof(int64_t));
  TensorHandle signed_gradient(gradient_bytes.data(), {num_data}, ACL_INT8);
  TensorHandle unsigned_hessian(hessian_bytes.data(), {num_data}, ACL_UINT8);
  TensorHandle g_out(impl_->quantized_gradients.data(), {num_data}, ACL_INT64);
  TensorHandle h_out(impl_->quantized_hessians.data(), {num_data}, ACL_INT64);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnCastGetWorkspaceSize(signed_gradient.get(), ACL_INT64, g_out.get(), s, e);
  }, aclnnCast);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnCastGetWorkspaceSize(unsigned_hessian.get(), ACL_INT64, h_out.get(), s, e);
  }, aclnnCast);
  impl_->quantized_ready = true;
}

NPURuntime::QuantizedHistogramHandle NPURuntime::BuildQuantizedHistogram(
    const int32_t* row_indices, int64_t count, const std::vector<int8_t>& used) {
  CHECK_GE(count, 0);
  CHECK_LE(count, impl_->histogram_rows);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  if (count == 0 || used.empty()) return BuildQuantizedHistogramFromDeviceRows(nullptr, count, used);
  CHECK(row_indices != nullptr);
  for (int64_t i = 0; i < count; ++i) {
    CHECK_GE(row_indices[i], 0);
    CHECK_LT(row_indices[i], impl_->histogram_rows);
  }
  DeviceBuffer rows = impl_->Upload(row_indices, static_cast<size_t>(count) * sizeof(int32_t));
  return BuildQuantizedHistogramFromDeviceRows(static_cast<const int32_t*>(rows.data()), count, used);
}

NPURuntime::QuantizedHistogramHandle NPURuntime::BuildQuantizedHistogramFromDeviceRows(
    const int32_t* device_rows, int64_t count, const std::vector<int8_t>& used) {
  CHECK_EQ(used.size(), impl_->histogram_bin_counts.size());
  CHECK_GE(count, 0);
  CHECK_LE(count, impl_->histogram_rows);
  CHECK(impl_->quantized_ready);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  auto result = std::make_shared<QuantizedDeviceHistogram>();
  result->owner = this;
  result->dataset_generation = impl_->dataset_generation;
  result->gradient_generation = impl_->quantized_generation;
  result->features = static_cast<int>(used.size());
  result->stride = impl_->histogram_stride;
  result->count = count;
  result->used = used;
  result->offsets = impl_->histogram_offsets;
  result->bin_counts = impl_->histogram_bin_counts;
  result->most_frequent = impl_->histogram_most_frequent;
  const int64_t plane = static_cast<int64_t>(result->features) * result->stride;
  result->values = impl_->Zeros(2 * plane * sizeof(int64_t));
  ++histogram_calls_;
  if (count == 0 || used.empty()) return result;
  CHECK(device_rows != nullptr);
  TensorHandle rows(const_cast<int32_t*>(device_rows), {count}, ACL_INT32);
  DeviceBuffer selected[2] = {DeviceBuffer(count * sizeof(int64_t)), DeviceBuffer(count * sizeof(int64_t))};
  for (int part = 0; part < 2; ++part) {
    DeviceBuffer& values = part == 0 ? impl_->quantized_gradients : impl_->quantized_hessians;
    TensorHandle source(values.data(), {impl_->histogram_rows}, ACL_INT64);
    TensorHandle output(selected[part].data(), {count}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(source.get(), 0, rows.get(), output.get(), s, e);
    }, aclnnIndexSelect);
  }
  const int64_t output_length = static_cast<int64_t>(Impl::kHistogramTile) * result->stride + 1;
  for (size_t first = 0; first < used.size(); first += Impl::kHistogramTile) {
    const int64_t tile = static_cast<int64_t>(std::min(static_cast<size_t>(Impl::kHistogramTile), used.size() - first));
    bool needed = false;
    for (int64_t j = 0; j < tile; ++j) needed |= used[first + j] != 0;
    if (!needed) continue;
    const int64_t elements = tile * count;
    DeviceBuffer bins(elements * sizeof(int32_t)), indices(elements * sizeof(int64_t));
    TensorHandle source_bins(impl_->histogram_tiles[first / Impl::kHistogramTile].data(),
                              {tile, impl_->histogram_rows}, ACL_INT32);
    TensorHandle gathered_bins(bins.data(), {tile, count}, ACL_INT32);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(source_bins.get(), 1, rows.get(), gathered_bins.get(), s, e);
    }, aclnnIndexSelect);
    TensorHandle flat_bins(bins.data(), {elements}, ACL_INT32);
    TensorHandle flat_indices(indices.data(), {elements}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnCastGetWorkspaceSize(flat_bins.get(), ACL_INT64, flat_indices.get(), s, e);
    }, aclnnCast);
    for (int part = 0; part < 2; ++part) {
      DeviceBuffer repeated(elements * sizeof(int64_t));
      TensorHandle selected_row(selected[part].data(), {1, count}, ACL_INT64);
      TensorHandle weights(repeated.data(), {tile, count}, ACL_INT64);
      const int64_t dims[] = {tile, count};
      std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> shape(
          aclCreateIntArray(dims, 2), &aclDestroyIntArray);
      CHECK(shape != nullptr);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnExpandGetWorkspaceSize(selected_row.get(), shape.get(), weights.get(), s, e);
      }, aclnnExpand);
      TensorHandle flat_weights(repeated.data(), {elements}, ACL_INT64);
      DeviceBuffer zeros = impl_->Zeros(output_length * sizeof(int64_t));
      DeviceBuffer histogram(output_length * sizeof(int64_t));
      TensorHandle zero(zeros.data(), {output_length}, ACL_INT64);
      TensorHandle out(histogram.data(), {output_length}, ACL_INT64);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnScatterAddGetWorkspaceSize(zero.get(), 0, flat_indices.get(),
                                                flat_weights.get(), out.get(), s, e);
      }, aclnnScatterAdd);
      // Copy only requested feature spans. Padding and disabled features stay
      // zero, including when an active feature shares the same device tile.
      const size_t feature_bytes = result->stride * sizeof(int64_t);
      for (int64_t j = 0; j < tile; ++j) {
        if (!used[first + j]) continue;
        auto* dst = static_cast<int64_t*>(result->values.data()) + part * plane + (first + j) * result->stride;
        const auto* src = static_cast<const int64_t*>(histogram.data()) + j * result->stride;
        ACL_CHECK(aclrtMemcpy(dst, feature_bytes, src, feature_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
      }
    }
  }
  return result;
}

void NPURuntime::FixQuantizedHistogram(const QuantizedHistogramHandle& histogram,
                                       int64_t sum_gradient, int64_t sum_hessian) {
  CHECK(histogram != nullptr);
  CHECK(histogram->owner == this);
  CHECK_EQ(histogram->dataset_generation, impl_->dataset_generation);
  CHECK_EQ(histogram->gradient_generation, impl_->quantized_generation);
  CHECK(!histogram->fixed);
  // Every source byte is bounded. Validate totals before integer subtraction
  // so even a malformed caller cannot cause INT64 overflow in device repair.
  CHECK_GE(sum_gradient, -128 * histogram->count);
  CHECK_LE(sum_gradient, 127 * histogram->count);
  CHECK_GE(sum_hessian, 0);
  CHECK_LE(sum_hessian, 255 * histogram->count);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  const int64_t plane = static_cast<int64_t>(histogram->features) * histogram->stride;
  const int64_t elements = 2 * plane;
  std::vector<int64_t> last_indices, destinations, totals;
  for (int part = 0; part < 2; ++part) {
    for (int feature = 0; feature < histogram->features; ++feature) {
      const int mf = histogram->most_frequent[feature];
      if (!histogram->used[feature] || mf <= 0) continue;
      const int64_t base = part * plane + feature * histogram->stride;
      last_indices.push_back(base + histogram->stride - 1);
      destinations.push_back(base + mf - histogram->offsets[feature]);
      totals.push_back(part == 0 ? sum_gradient : sum_hessian);
    }
  }
  const int64_t fixes = static_cast<int64_t>(last_indices.size());
  if (fixes != 0) {
    DeviceBuffer prefix(elements * sizeof(int64_t));
    TensorHandle source(histogram->values.data(), {2 * histogram->features, histogram->stride}, ACL_INT64);
    TensorHandle output(prefix.data(), {2 * histogram->features, histogram->stride}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnCumsumGetWorkspaceSize(source.get(), 1, ACL_INT64, output.get(), s, e);
    }, aclnnCumsum);
    DeviceBuffer indices = impl_->Upload(last_indices.data(), fixes * sizeof(int64_t));
    DeviceBuffer sums(fixes * sizeof(int64_t)), residuals(fixes * sizeof(int64_t));
    DeviceBuffer parents = impl_->Upload(totals.data(), fixes * sizeof(int64_t));
    TensorHandle flat(prefix.data(), {elements}, ACL_INT64), index(indices.data(), {fixes}, ACL_INT64);
    TensorHandle sum(sums.data(), {fixes}, ACL_INT64), parent(parents.data(), {fixes}, ACL_INT64);
    TensorHandle residual(residuals.data(), {fixes}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(flat.get(), 0, index.get(), sum.get(), s, e);
    }, aclnnIndexSelect);
    IntScalarHandle one(1);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnSubGetWorkspaceSize(parent.get(), sum.get(), one.get(), residual.get(), s, e);
    }, aclnnSub);
    // Commit all repaired bins together. An ACL copy failure leaves the
    // original raw histogram unchanged, so a later retry cannot double-fix it.
    DeviceBuffer repaired(histogram->values.bytes());
    ACL_CHECK(aclrtMemcpy(repaired.data(), repaired.bytes(), histogram->values.data(),
                          histogram->values.bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE));
    for (int64_t i = 0; i < fixes; ++i) {
      ACL_CHECK(aclrtMemcpy(static_cast<int64_t*>(repaired.data()) + destinations[i], sizeof(int64_t),
                            static_cast<const int64_t*>(residuals.data()) + i, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    histogram->values = std::move(repaired);
  }
  histogram->fixed = true;
}

void NPURuntime::SubtractQuantizedHistogram(const QuantizedHistogramHandle& parent,
                                            const QuantizedHistogramHandle& smaller) {
  CHECK(parent != nullptr && smaller != nullptr);
  CHECK(parent != smaller);
  CHECK(parent->owner == this && smaller->owner == this);
  CHECK(parent->fixed && smaller->fixed);
  CHECK_EQ(parent->dataset_generation, impl_->dataset_generation);
  CHECK_EQ(smaller->dataset_generation, impl_->dataset_generation);
  CHECK_EQ(parent->gradient_generation, impl_->quantized_generation);
  CHECK_EQ(smaller->gradient_generation, impl_->quantized_generation);
  CHECK_EQ(parent->features, smaller->features);
  CHECK_EQ(parent->stride, smaller->stride);
  CHECK_EQ(parent->used.size(), smaller->used.size());
  // Tree growth may prune unsplittable features. The smaller histogram may
  // narrow the parent's mask, but can never reactivate an unavailable feature.
  for (size_t f = 0; f < parent->used.size(); ++f)
    CHECK(!smaller->used[f] || parent->used[f]);
  CHECK_GE(parent->count, smaller->count);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  const int64_t elements = static_cast<int64_t>(parent->values.bytes() / sizeof(int64_t));
  if (elements != 0) {
    DeviceBuffer difference(parent->values.bytes());
    TensorHandle a(parent->values.data(), {elements}, ACL_INT64);
    TensorHandle b(smaller->values.data(), {elements}, ACL_INT64);
    TensorHandle out(difference.data(), {elements}, ACL_INT64);
    IntScalarHandle one(1);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnSubGetWorkspaceSize(a.get(), b.get(), one.get(), out.get(), s, e);
    }, aclnnSub);
    parent->values = std::move(difference);
  }
  parent->count -= smaller->count;
  parent->used = smaller->used;
}

NPURuntime::QuantizedHistogramResult NPURuntime::ReadQuantizedHistogram(
    const QuantizedHistogramHandle& histogram) {
  CHECK(histogram != nullptr);
  CHECK(histogram->owner == this);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  QuantizedHistogramResult result;
  result.gradients.resize(histogram->features);
  result.hessians.resize(histogram->features);
  const size_t elements = histogram->values.bytes() / sizeof(int64_t);
  std::vector<int64_t> host(elements);
  if (elements != 0) impl_->Download(host.data(), histogram->values, histogram->values.bytes());
  const int64_t plane = static_cast<int64_t>(histogram->features) * histogram->stride;
  for (int feature = 0; feature < histogram->features; ++feature) {
    if (!histogram->used[feature]) continue;
    const int n = histogram->bin_counts[feature] - histogram->offsets[feature];
    const auto begin = host.begin() + feature * histogram->stride;
    result.gradients[feature].assign(begin, begin + n);
    result.hessians[feature].assign(begin + plane, begin + plane + n);
  }
  return result;
}

NPURuntime::QuantizedSplitScanHandle NPURuntime::PrepareQuantizedSplitScan(
    const std::vector<SplitFeature>& features, int64_t sum_gradient,
    int64_t sum_hessian, int num_data, const QuantizedHistogramHandle& histogram) {
  CHECK(histogram != nullptr);
  CHECK(histogram->owner == this);
  CHECK(histogram->fixed);
  CHECK_EQ(histogram->dataset_generation, impl_->dataset_generation);
  CHECK_EQ(histogram->gradient_generation, impl_->quantized_generation);
  CHECK_EQ(histogram->count, num_data);
  CHECK_GT(num_data, 0);
  CHECK_GE(sum_gradient, -int64_t{128} * num_data);
  CHECK_LE(sum_gradient, int64_t{127} * num_data);
  CHECK_GT(sum_hessian, 0);
  CHECK_LE(sum_hessian, int64_t{255} * num_data);
  auto scan = std::make_shared<QuantizedDeviceSplitScan>();
  scan->owner = this;
  scan->dataset_generation = histogram->dataset_generation;
  scan->gradient_generation = histogram->gradient_generation;
  scan->features = features;
  scan->sum_gradient = sum_gradient;
  scan->sum_hessian = sum_hessian;
  scan->num_data = num_data;
  struct Residual { int64_t candidate; int feature_index; };
  std::vector<Residual> residuals;
  std::vector<int64_t> indices;
  std::vector<size_t> order(features.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return features[a].feature < features[b].feature;
  });
  for (size_t fi : order) {
    const auto& feature = features[fi];
    if (feature.categorical || feature.extra_tree_random != nullptr) {
      Log::Fatal("NPU quantized split scan currently supports exhaustive numerical features only");
    }
    CHECK_GE(feature.feature, 0);
    CHECK_GE(feature.histogram_index, 0);
    CHECK_LT(feature.histogram_index, histogram->features);
    CHECK(histogram->used[feature.histogram_index]);
    CHECK_EQ(feature.num_bin, histogram->bin_counts[feature.histogram_index]);
    CHECK_EQ(feature.offset, histogram->offsets[feature.histogram_index]);
    CHECK(feature.offset == 0 || feature.offset == 1);
    CHECK_GE(feature.default_bin, 0);
    CHECK_LT(feature.default_bin, feature.num_bin);
    CHECK_GE(feature.missing_type, 0);
    CHECK_LE(feature.missing_type, 2);
    CHECK(std::isfinite(feature.penalty) && feature.penalty >= 0.0);
    if (feature.num_bin <= 1) continue;
    const int64_t base = static_cast<int64_t>(feature.histogram_index) * histogram->stride;
    const bool double_scan = feature.num_bin > 2 && feature.missing_type != 0;
    for (int direction = 0; direction < (double_scan ? 2 : 1); ++direction) {
      const bool reverse = direction == 0;
      const int64_t begin = static_cast<int64_t>(indices.size());
      auto append = [&](int stored_bin, uint32_t threshold, bool residual) {
        QuantizedSplitCandidate candidate;
        candidate.feature = feature.feature;
        candidate.threshold = threshold;
        candidate.default_left = reverse && !(feature.missing_type == 2 && !double_scan);
        if (residual) residuals.push_back({static_cast<int64_t>(indices.size()), static_cast<int>(fi)});
        // A residual placeholder is replaced on device before prefix scanning.
        indices.push_back(base + stored_bin);
        scan->candidates.push_back(candidate);
      };
      if (!reverse && feature.missing_type == 2 && feature.offset == 1) {
        append(0, 0, true);
      }
      const int first = reverse ? feature.num_bin - 1 - feature.offset -
          (double_scan && feature.missing_type == 2) : 0;
      const int last = reverse ? 1 - feature.offset : feature.num_bin - 2 - feature.offset;
      for (int t = first; reverse ? t >= last : t <= last; t += reverse ? -1 : 1) {
        if (double_scan && feature.missing_type == 1 && t + feature.offset == feature.default_bin)
          continue;
        append(t, static_cast<uint32_t>(reverse ? t - 1 + feature.offset : t + feature.offset), false);
      }
      const int64_t length = static_cast<int64_t>(indices.size()) - begin;
      if (length != 0) scan->rows.push_back({static_cast<int>(fi), begin, length, reverse});
    }
  }
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  const int64_t n = static_cast<int64_t>(indices.size());
  if (n == 0) return scan;
  const size_t bytes = static_cast<size_t>(n) * sizeof(int64_t);
  scan->values = DeviceBuffer(4 * bytes);
  DeviceBuffer index_buffer = impl_->Upload(indices.data(), bytes);
  TensorHandle index_tensor(index_buffer.data(), {n}, ACL_INT64);
  const int64_t plane = static_cast<int64_t>(histogram->features) * histogram->stride;
  IntScalarHandle one(1);
  for (int part = 0; part < 2; ++part) {
    DeviceBuffer packed(bytes), prefix(bytes), complement(bytes);
    auto* source_pointer = static_cast<int64_t*>(histogram->values.data()) + part * plane;
    TensorHandle source(source_pointer, {plane}, ACL_INT64);
    TensorHandle packed_tensor(packed.data(), {n}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(source.get(), 0, index_tensor.get(), packed_tensor.get(), s, e);
    }, aclnnIndexSelect);
    const int64_t total = part == 0 ? sum_gradient : sum_hessian;
    for (const auto& residual : residuals) {
      const auto& feature = features[residual.feature_index];
      const int64_t length = feature.num_bin - feature.offset;
      DeviceBuffer stored_prefix(length * sizeof(int64_t));
      TensorHandle stored(source_pointer + static_cast<int64_t>(feature.histogram_index) * histogram->stride,
                           {length}, ACL_INT64);
      TensorHandle stored_out(stored_prefix.data(), {length}, ACL_INT64);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnCumsumGetWorkspaceSize(stored.get(), 0, ACL_INT64, stored_out.get(), s, e);
      }, aclnnCumsum);
      DeviceBuffer total_buffer = impl_->Upload(&total, sizeof(total));
      TensorHandle total_tensor(total_buffer.data(), {1}, ACL_INT64);
      TensorHandle last(static_cast<int64_t*>(stored_prefix.data()) + length - 1, {1}, ACL_INT64);
      TensorHandle target(static_cast<int64_t*>(packed.data()) + residual.candidate, {1}, ACL_INT64);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnSubGetWorkspaceSize(total_tensor.get(), last.get(), one.get(), target.get(), s, e);
      }, aclnnSub);
    }
    for (const auto& row : scan->rows) {
      TensorHandle input(static_cast<int64_t*>(packed.data()) + row.begin, {row.length}, ACL_INT64);
      TensorHandle output(static_cast<int64_t*>(prefix.data()) + row.begin, {row.length}, ACL_INT64);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnCumsumGetWorkspaceSize(input.get(), 0, ACL_INT64, output.get(), s, e);
      }, aclnnCumsum);
    }
    std::vector<int64_t> total_values(static_cast<size_t>(n), total);
    DeviceBuffer totals = impl_->Upload(total_values.data(), bytes);
    TensorHandle total_tensor(totals.data(), {n}, ACL_INT64);
    TensorHandle prefix_tensor(prefix.data(), {n}, ACL_INT64);
    TensorHandle complement_tensor(complement.data(), {n}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnSubGetWorkspaceSize(total_tensor.get(), prefix_tensor.get(), one.get(), complement_tensor.get(), s, e);
    }, aclnnSub);
    for (const auto& row : scan->rows) {
      const size_t row_bytes = static_cast<size_t>(row.length) * sizeof(int64_t);
      for (int side = 0; side < 2; ++side) {
        const bool use_prefix = (side == 0) != row.reverse;
        auto* input = static_cast<int64_t*>(use_prefix ? prefix.data() : complement.data()) + row.begin;
        auto* output = static_cast<int64_t*>(scan->values.data()) + (2 * side + part) * n + row.begin;
        ACL_CHECK(aclrtMemcpy(output, row_bytes, input, row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
      }
    }
  }
  return scan;
}

std::vector<NPURuntime::QuantizedSplitCandidate> NPURuntime::ReadQuantizedSplitScan(
    const QuantizedSplitScanHandle& scan) {
  CHECK(scan != nullptr);
  CHECK(scan->owner == this);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  auto result = scan->candidates;
  const size_t n = result.size();
  if (n == 0) return result;
  std::vector<int64_t> values(4 * n);
  impl_->Download(values.data(), scan->values, values.size() * sizeof(int64_t));
  for (size_t i = 0; i < n; ++i) {
    result[i].left_gradient = values[i];
    result[i].left_hessian = values[n + i];
    result[i].right_gradient = values[2 * n + i];
    result[i].right_hessian = values[3 * n + i];
  }
  return result;
}

NPURuntime::QuantizedSplitEvaluationHandle NPURuntime::EvaluateQuantizedSplitScan(
    const QuantizedSplitScanHandle& scan, const QuantizedSplitParameters& p) {
  CHECK(scan != nullptr);
  CHECK(scan->owner == this);
  CHECK_EQ(scan->dataset_generation, impl_->dataset_generation);
  CHECK_EQ(scan->gradient_generation, impl_->quantized_generation);
  CHECK(std::isfinite(p.gradient_scale) && p.gradient_scale >= 0.0);
  CHECK(std::isfinite(p.hessian_scale) && p.hessian_scale > 0.0);
  CHECK(std::isfinite(p.lambda_l1) && p.lambda_l1 >= 0.0);
  CHECK(std::isfinite(p.lambda_l2) && p.lambda_l2 >= 0.0);
  CHECK_GE(p.min_data_in_leaf, 0);
  CHECK(std::isfinite(p.min_sum_hessian_in_leaf) && p.min_sum_hessian_in_leaf >= 0.0);
  CHECK(std::isfinite(p.min_gain_to_split) && p.min_gain_to_split >= 0.0);
  CHECK(std::isfinite(p.max_delta_step) && p.max_delta_step >= 0.0);
  CHECK(std::isfinite(p.path_smooth) && p.path_smooth >= 0.0);
  CHECK(std::isfinite(p.parent_output));
  // A CPU quantized candidate may have zero raw Hessian when all three
  // lower bounds are disabled; its epsilon-based gain can still be valid but
  // its final raw-H output divides by zero. Refuse this unsafe parameter
  // combination explicitly rather than silently changing candidate validity.
  CHECK(p.min_data_in_leaf > 0 || p.min_sum_hessian_in_leaf > 0.0 || p.lambda_l2 > 0.0);
  // Refuse arithmetic ranges that can overflow during candidate or parent
  // evaluation; accepting a finite scale alone is insufficient.
  const double max_g = 128.0 * scan->num_data * p.gradient_scale;
  const double total_h = static_cast<double>(scan->sum_hessian) * p.hessian_scale;
  CHECK(std::isfinite(max_g * max_g));
  CHECK(std::isfinite(total_h + p.lambda_l2) && total_h > 0.0);
  CHECK(std::isfinite((max_g * max_g) / std::min(total_h, static_cast<double>(kEpsilon))));
  const double denominator_bound = p.lambda_l2 > 0.0
      ? std::min(p.lambda_l2, p.hessian_scale) : std::min(total_h, p.hessian_scale);
  const double output_bound = max_g / denominator_bound +
      (p.path_smooth > kEpsilon ? std::fabs(p.parent_output) : 0.0);
  CHECK(std::isfinite(output_bound));
  if (p.path_smooth > kEpsilon || p.max_delta_step > 0.0) {
    CHECK(std::isfinite((total_h + p.lambda_l2) * output_bound * output_bound));
    CHECK(std::isfinite(2.0 * max_g * output_bound));
  }
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  auto evaluation = std::make_shared<QuantizedDeviceSplitEvaluation>();
  evaluation->scan = scan;
  evaluation->owner = this;
  const int64_t n = static_cast<int64_t>(scan->candidates.size());
  if (n == 0) return evaluation;
  const size_t bytes = static_cast<size_t>(n) * sizeof(double);
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<std::unique_ptr<TensorHandle>> tensors;
  auto allocate = [&](aclDataType dtype) {
    buffers.emplace_back(new DeviceBuffer(dtype == ACL_BOOL ? n : bytes));
    tensors.emplace_back(new TensorHandle(buffers.back()->data(), {n}, dtype));
    return static_cast<int>(tensors.size()) - 1;
  };
  auto upload = [&](const std::vector<double>& values) {
    CHECK_EQ(values.size(), static_cast<size_t>(n));
    int output = allocate(ACL_DOUBLE);
    ACL_CHECK(aclrtMemcpy(buffers[output]->data(), bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    return output;
  };
  auto constant = [&](double value) { return upload(std::vector<double>(static_cast<size_t>(n), value)); };
#define Q_UNARY(OP, X) ([&]() { int o = allocate(ACL_DOUBLE); impl_->Launch(\
    [&](uint64_t* s, aclOpExecutor** e) { return aclnn##OP##GetWorkspaceSize(tensors[X]->get(), tensors[o]->get(), s, e); }, aclnn##OP); return o; }())
#define Q_BINARY(OP, X, Y) ([&]() { int o = allocate(ACL_DOUBLE); impl_->Launch(\
    [&](uint64_t* s, aclOpExecutor** e) { return aclnn##OP##GetWorkspaceSize(tensors[X]->get(), tensors[Y]->get(), tensors[o]->get(), s, e); }, aclnn##OP); return o; }())
  auto linear = [&](int x, int y, bool subtract) {
    int o = allocate(ACL_DOUBLE);
    ScalarHandle one(1.0, ACL_DOUBLE);
    if (subtract) impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnSubGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), one.get(), tensors[o]->get(), s, e);
    }, aclnnSub);
    else impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnAddGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), one.get(), tensors[o]->get(), s, e);
    }, aclnnAdd);
    return o;
  };
  auto scalar = [&](int x, double value, bool multiply) {
    int o = allocate(ACL_DOUBLE);
    ScalarHandle v(value, ACL_DOUBLE), one(1.0, ACL_DOUBLE);
    if (multiply) impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnMulsGetWorkspaceSize(tensors[x]->get(), v.get(), tensors[o]->get(), s, e);
    }, aclnnMuls);
    else impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnAddsGetWorkspaceSize(tensors[x]->get(), v.get(), one.get(), tensors[o]->get(), s, e);
    }, aclnnAdds);
    return o;
  };
  auto cast_double = [&](int x) {
    int o = allocate(ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnCastGetWorkspaceSize(tensors[x]->get(), ACL_DOUBLE, tensors[o]->get(), s, e);
    }, aclnnCast);
    return o;
  };
  auto ge = [&](int x, double value, bool strict) {
    int o = allocate(ACL_BOOL);
    ScalarHandle v(value, ACL_DOUBLE);
    if (strict) impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnGtScalarGetWorkspaceSize(tensors[x]->get(), v.get(), tensors[o]->get(), s, e);
    }, aclnnGtScalar);
    else impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnGeScalarGetWorkspaceSize(tensors[x]->get(), v.get(), tensors[o]->get(), s, e);
    }, aclnnGeScalar);
    return o;
  };
  auto logical_and = [&](int x, int y) {
    int o = allocate(ACL_BOOL);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnLogicalAndGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), tensors[o]->get(), s, e);
    }, aclnnLogicalAnd);
    return o;
  };
  auto clamp_min = [&](int x, double lower) {
    int o = allocate(ACL_DOUBLE);
    ScalarHandle value(lower, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnClampMinGetWorkspaceSize(tensors[x]->get(), value.get(), tensors[o]->get(), s, e);
    }, aclnnClampMin);
    return o;
  };
  int raw[4];
  for (int lane = 0; lane < 4; ++lane) {
    int input = allocate(ACL_INT64);
    ACL_CHECK(aclrtMemcpy(buffers[input]->data(), bytes,
        static_cast<int64_t*>(scan->values.data()) + lane * n, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
    raw[lane] = cast_double(input);
  }
  const double count_factor = static_cast<double>(scan->num_data) / static_cast<double>(scan->sum_hessian);
  int left_count = Q_UNARY(Floor, scalar(scalar(raw[1], count_factor, true), 0.5, false));
  int right_count = Q_UNARY(Floor, scalar(scalar(raw[3], count_factor, true), 0.5, false));
  std::vector<double> reverse(static_cast<size_t>(n));
  for (const auto& row : scan->rows)
    std::fill(reverse.begin() + row.begin, reverse.begin() + row.begin + row.length, row.reverse ? 1.0 : 0.0);
  int reverse_mask = upload(reverse);
  int forward_mask = scalar(scalar(reverse_mask, -1.0, true), 1.0, false);
  auto choose = [&](int forward, int backward) {
    return linear(Q_BINARY(Mul, forward, forward_mask), Q_BINARY(Mul, backward, reverse_mask), false);
  };
  int scan_left = choose(left_count, scalar(scalar(right_count, -1.0, true), scan->num_data, false));
  int scan_right = scalar(scalar(scan_left, -1.0, true), scan->num_data, false);
  int left_g = scalar(raw[0], p.gradient_scale, true), right_g = scalar(raw[2], p.gradient_scale, true);
  int left_h = scalar(raw[1], p.hessian_scale, true), right_h = scalar(raw[3], p.hessian_scale, true);
  int bounds = logical_and(ge(scan_left, p.min_data_in_leaf, false), ge(scan_right, p.min_data_in_leaf, false));
  bounds = logical_and(bounds, ge(left_h, p.min_sum_hessian_in_leaf, false));
  bounds = logical_and(bounds, ge(right_h, p.min_sum_hessian_in_leaf, false));
  // Invalid entries are assigned G=0,H=1 before ANY division. Multiplying a
  // NaN by a zero validity flag afterwards would not remove that NaN.
  int bounded = cast_double(bounds);
  int unbounded = scalar(scalar(bounded, -1.0, true), 1.0, false);
  auto safe_g = [&](int x) { return Q_BINARY(Mul, x, bounded); };
  auto safe_h = [&](int x) { return linear(Q_BINARY(Mul, x, bounded), unbounded, false); };
  auto soft = [&](int x) {
    if (p.lambda_l1 <= 0.0) return x;
    return Q_BINARY(Mul, clamp_min(scalar(Q_UNARY(Abs, x), -p.lambda_l1, false), 0.0), Q_UNARY(Sign, x));
  };
  auto output = [&](int g, int h, int count) {
    int value = Q_UNARY(Neg, Q_BINARY(Div, soft(g), scalar(h, p.lambda_l2, false)));
    if (p.max_delta_step > 0.0) {
      value = Q_UNARY(Neg, clamp_min(Q_UNARY(Neg, clamp_min(value, -p.max_delta_step)), -p.max_delta_step));
    }
    if (p.path_smooth > kEpsilon) {
      int ratio = Q_BINARY(Div, count, constant(p.path_smooth));
      int denominator = scalar(ratio, 1.0, false);
      value = linear(Q_BINARY(Div, Q_BINARY(Mul, value, ratio), denominator),
                     Q_BINARY(Div, constant(p.parent_output), denominator), false);
    }
    return value;
  };
  auto gain = [&](int g, int h, int value) {
    if (p.max_delta_step <= 0.0 && p.path_smooth <= kEpsilon) {
      int v = soft(g);
      return Q_BINARY(Div, Q_BINARY(Mul, v, v), scalar(h, p.lambda_l2, false));
    }
    int linear_term = Q_BINARY(Mul, scalar(soft(g), 2.0, true), value);
    int quadratic = Q_BINARY(Mul, Q_BINARY(Mul, scalar(h, p.lambda_l2, false), value), value);
    return Q_UNARY(Neg, linear(linear_term, quadratic, false));
  };
  int lg = safe_g(left_g), rg = safe_g(right_g);
  int lh = safe_h(left_h), rh = safe_h(right_h);
  int gain_lh = scalar(lh, kEpsilon, false), gain_rh = scalar(rh, kEpsilon, false);
  int current = linear(gain(lg, gain_lh, output(lg, gain_lh, scan_left)),
                       gain(rg, gain_rh, output(rg, gain_rh, scan_right)), false);
  int parent_g = scalar(constant(static_cast<double>(scan->sum_gradient)), p.gradient_scale, true);
  int parent_h = scalar(constant(static_cast<double>(scan->sum_hessian)), p.hessian_scale, true);
  int parent_count = constant(scan->num_data);
  int shift = scalar(gain(parent_g, parent_h, output(parent_g, parent_h, parent_count)), p.min_gain_to_split, false);
  int valid = logical_and(bounds, ge(linear(current, shift, true), 0.0, true));
  int valid_double = cast_double(valid);
  // Recompute raw-Hessian final outputs with independently rounded child
  // counts. Candidate gain outputs use H+epsilon and scan-complement counts.
  int invalid_double = scalar(scalar(valid_double, -1.0, true), 1.0, false);
  auto final_g = [&](int x) { return Q_BINARY(Mul, x, valid_double); };
  auto final_h = [&](int x) { return linear(Q_BINARY(Mul, x, valid_double), invalid_double, false); };
  int final_left = Q_BINARY(Mul, output(final_g(left_g), final_h(left_h), left_count), valid_double);
  int final_right = Q_BINARY(Mul, output(final_g(right_g), final_h(right_h), right_count), valid_double);
  int masked_current = Q_BINARY(Mul, current, valid_double);
  const int result_ids[] = {left_g, left_h, right_g, right_h, left_count, right_count,
                            scan_left, scan_right, final_left, final_right, masked_current, shift};
  evaluation->values = DeviceBuffer(12 * bytes);
  for (int lane = 0; lane < 12; ++lane)
    ACL_CHECK(aclrtMemcpy(static_cast<double*>(evaluation->values.data()) + lane * n, bytes,
                          buffers[result_ids[lane]]->data(), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
  evaluation->valid = DeviceBuffer(static_cast<size_t>(n));
  ACL_CHECK(aclrtMemcpy(evaluation->valid.data(), n, buffers[valid]->data(), n, ACL_MEMCPY_DEVICE_TO_DEVICE));
#undef Q_UNARY
#undef Q_BINARY
  return evaluation;
}

std::vector<NPURuntime::QuantizedSplitEvaluation> NPURuntime::ReadQuantizedSplitEvaluation(
    const QuantizedSplitEvaluationHandle& evaluation) {
  CHECK(evaluation != nullptr);
  CHECK(evaluation->owner == this);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  const auto candidates = ReadQuantizedSplitScan(evaluation->scan);
  const size_t n = candidates.size();
  std::vector<QuantizedSplitEvaluation> result(n);
  if (n == 0) return result;
  std::vector<double> values(12 * n);
  std::vector<uint8_t> valid(n);
  impl_->Download(values.data(), evaluation->values, values.size() * sizeof(double));
  impl_->Download(valid.data(), evaluation->valid, n);
  for (size_t i = 0; i < n; ++i) {
    auto& r = result[i];
    r.candidate = candidates[i];
    r.left_gradient = values[i]; r.left_hessian = values[n + i];
    r.right_gradient = values[2 * n + i]; r.right_hessian = values[3 * n + i];
    r.left_count = static_cast<int>(values[4 * n + i]);
    r.right_count = static_cast<int>(values[5 * n + i]);
    r.scan_left_count = static_cast<int>(values[6 * n + i]);
    r.scan_right_count = static_cast<int>(values[7 * n + i]);
    r.left_output = values[8 * n + i]; r.right_output = values[9 * n + i];
    r.current_gain = values[10 * n + i]; r.gain_shift = values[11 * n + i];
    r.valid = valid[i] != 0;
  }
  return result;
}

NPURuntime::QuantizedSplitSelection NPURuntime::SelectQuantizedSplit(
    const QuantizedSplitEvaluationHandle& evaluation) {
  CHECK(evaluation != nullptr);
  CHECK(evaluation->owner == this);
  const auto& scan = evaluation->scan;
  CHECK_EQ(scan->dataset_generation, impl_->dataset_generation);
  CHECK_EQ(scan->gradient_generation, impl_->quantized_generation);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  QuantizedSplitSelection result;
  ++split_calls_;
  result.split.splittable.resize(scan->features.size(), 0);
  const int64_t n = static_cast<int64_t>(scan->candidates.size());
  const int64_t r = static_cast<int64_t>(scan->rows.size());
  if (n == 0) return result;
  CHECK_GT(r, 0);
  const size_t bytes = static_cast<size_t>(r) * sizeof(double);
  DeviceBuffer packed_winners(bytes);
  // Reduce each direction independently on unshifted FP64 gain. Invalid
  // candidates have zero gain; use a finite negative sentinel, not NaN*0 or
  // infinity arithmetic. The first threshold in scan order wins equal gains.
  DeviceBuffer valid_double(n * sizeof(double)), masked(n * sizeof(double));
  TensorHandle valid(evaluation->valid.data(), {n}, ACL_BOOL);
  TensorHandle vd(valid_double.data(), {n}, ACL_DOUBLE);
  TensorHandle current(static_cast<double*>(evaluation->values.data()) + 10 * n, {n}, ACL_DOUBLE);
  TensorHandle masked_tensor(masked.data(), {n}, ACL_DOUBLE);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnCastGetWorkspaceSize(valid.get(), ACL_DOUBLE, vd.get(), s, e);
  }, aclnnCast);
  DeviceBuffer invalid(n * sizeof(double));
  TensorHandle invalid_tensor(invalid.data(), {n}, ACL_DOUBLE);
  ScalarHandle minus_one(-1.0, ACL_DOUBLE), one_double(1.0, ACL_DOUBLE);
  ScalarHandle sentinel(std::numeric_limits<double>::max(), ACL_DOUBLE);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnAddsGetWorkspaceSize(vd.get(), minus_one.get(), one_double.get(), invalid_tensor.get(), s, e);
  }, aclnnAdds);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnMulsGetWorkspaceSize(invalid_tensor.get(), sentinel.get(), masked_tensor.get(), s, e);
  }, aclnnMuls);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnAddGetWorkspaceSize(current.get(), masked_tensor.get(), one_double.get(), invalid_tensor.get(), s, e);
  }, aclnnAdd);
  IntScalarHandle one(1);
  for (int64_t i = 0; i < r; ++i) {
    const auto& row = scan->rows[i];
    TensorHandle source(static_cast<double*>(invalid.data()) + row.begin, {row.length}, ACL_DOUBLE);
    DeviceBuffer local(sizeof(int64_t));
    TensorHandle local_index(local.data(), {1}, ACL_INT64);
    TensorHandle target(static_cast<int64_t*>(packed_winners.data()) + i, {1}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnArgMaxGetWorkspaceSize(source.get(), 0, false, local_index.get(), s, e);
    }, aclnnArgMax);
    // Keep the winning index resident; no per-feature host gain comparison.
    DeviceBuffer begin = impl_->Upload(&row.begin, sizeof(row.begin));
    TensorHandle begin_tensor(begin.data(), {1}, ACL_INT64);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnAddGetWorkspaceSize(local_index.get(), begin_tensor.get(), one.get(), target.get(), s, e);
    }, aclnnAdd);
  }
  TensorHandle packed_indices(packed_winners.data(), {r}, ACL_INT64);
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<std::unique_ptr<TensorHandle>> tensors;
  auto allocate = [&](aclDataType type) {
    buffers.emplace_back(new DeviceBuffer(type == ACL_BOOL ? r : bytes));
    tensors.emplace_back(new TensorHandle(buffers.back()->data(), {r}, type));
    return static_cast<int>(tensors.size()) - 1;
  };
  auto upload = [&](const std::vector<double>& values) {
    int o = allocate(ACL_DOUBLE);
    ACL_CHECK(aclrtMemcpy(buffers[o]->data(), bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    return o;
  };
  auto gather = [&](const TensorHandle& source, const TensorHandle& indices, aclDataType type) {
    int o = allocate(type);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(source.get(), 0, indices.get(), tensors[o]->get(), s, e);
    }, aclnnIndexSelect);
    return o;
  };
  auto cast = [&](int x) {
    int o = allocate(ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnCastGetWorkspaceSize(tensors[x]->get(), ACL_DOUBLE, tensors[o]->get(), s, e);
    }, aclnnCast);
    return o;
  };
  auto linear = [&](int x, int y, bool subtract) {
    int o = allocate(ACL_DOUBLE);
    if (subtract) impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnSubGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), one_double.get(), tensors[o]->get(), s, e);
    }, aclnnSub);
    else impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnAddGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), one_double.get(), tensors[o]->get(), s, e);
    }, aclnnAdd);
    return o;
  };
  auto multiply = [&](int x, int y) {
    int o = allocate(ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnMulGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), tensors[o]->get(), s, e);
    }, aclnnMul);
    return o;
  };
  auto positive = [&](int x) {
    int o = allocate(ACL_BOOL);
    ScalarHandle zero(0.0, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnGtScalarGetWorkspaceSize(tensors[x]->get(), zero.get(), tensors[o]->get(), s, e);
    }, aclnnGtScalar);
    return cast(o);
  };
  int row_current = gather(current, packed_indices, ACL_DOUBLE);
  TensorHandle shift_source(static_cast<double*>(evaluation->values.data()) + 11 * n, {n}, ACL_DOUBLE);
  int shift = gather(shift_source, packed_indices, ACL_DOUBLE);
  int row_valid = cast(gather(valid, packed_indices, ACL_BOOL));
  // Rows are already packed by ascending original feature id, reverse first.
  std::vector<int64_t> reverse_for_feature(scan->features.size(), -1), forward_for_feature(scan->features.size(), -1);
  for (int64_t i = 0; i < r; ++i) {
    const auto& row = scan->rows[i];
    (row.reverse ? reverse_for_feature : forward_for_feature)[row.feature_index] = i;
  }
  std::vector<int64_t> reverse_rows(r), forward_rows(r);
  std::vector<double> has_reverse(r), has_forward(r), reverse_flags(r), enabled(r), penalties(r);
  for (int64_t i = 0; i < r; ++i) {
    const auto& row = scan->rows[i];
    const auto& feature = scan->features[row.feature_index];
    int64_t rev = reverse_for_feature[row.feature_index], fwd = forward_for_feature[row.feature_index];
    reverse_rows[i] = rev < 0 ? i : rev;
    forward_rows[i] = fwd < 0 ? i : fwd;
    has_reverse[i] = rev >= 0; has_forward[i] = fwd >= 0;
    reverse_flags[i] = row.reverse; enabled[i] = feature.enabled; penalties[i] = feature.penalty;
  }
  DeviceBuffer rev_buffer = impl_->Upload(reverse_rows.data(), bytes), fwd_buffer = impl_->Upload(forward_rows.data(), bytes);
  TensorHandle rev_indices(rev_buffer.data(), {r}, ACL_INT64), fwd_indices(fwd_buffer.data(), {r}, ACL_INT64);
  int reverse_best = gather(*tensors[row_current], rev_indices, ACL_DOUBLE);
  int reverse_valid = multiply(gather(*tensors[row_valid], rev_indices, ACL_DOUBLE), upload(has_reverse));
  int forward_best = gather(*tensors[row_current], fwd_indices, ACL_DOUBLE);
  int forward_valid = multiply(gather(*tensors[row_valid], fwd_indices, ACL_DOUBLE), upload(has_forward));
  int ones = upload(std::vector<double>(r, 1.0));
  // CPU's second direction compares against (reverse_gain - shift) + shift,
  // not reverse_gain, nor the reverse shifted score. Preserve both operations.
  int restored_reverse = linear(linear(reverse_best, shift, true), shift, false);
  int better = positive(linear(forward_best, restored_reverse, true));
  int accepted = positive(linear(better, linear(ones, reverse_valid, true), false));
  int replace = multiply(forward_valid, accepted);
  int reverse_mask = upload(reverse_flags);
  int chosen = linear(multiply(reverse_mask, linear(ones, replace, true)),
                      multiply(linear(ones, reverse_mask, true), replace), false);
  int selectable = multiply(multiply(row_valid, chosen), upload(enabled));
  // Mask BEFORE applying the feature penalty: an invalid/disabled row can
  // have a large negative shifted gain whose product overflows. inf * 0 is
  // NaN, not a usable sentinel. Overflow of an actually selectable winning
  // score is rejected by the finite-gain check below.
  int score = multiply(multiply(linear(row_current, shift, true), selectable), upload(penalties));
  // All selectable scores are nonnegative; -1 is an exact invalid sentinel.
  int masked_score = linear(score, linear(selectable, ones, true), false);
  std::vector<uint8_t> row_flags(r);
  int flags = gather(valid, packed_indices, ACL_BOOL);
  impl_->Download(row_flags.data(), *buffers[flags], r);
  quantized_selection_download_bytes_ += r;
  for (int64_t i = 0; i < r; ++i)
    result.split.splittable[scan->rows[i].feature_index] |= row_flags[i];
  DeviceBuffer winner_buffer(sizeof(int64_t));
  TensorHandle winner(winner_buffer.data(), {1}, ACL_INT64);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnArgMaxGetWorkspaceSize(tensors[masked_score]->get(), 0, false, winner.get(), s, e);
  }, aclnnArgMax);
  ++argmax_calls_;
  int64_t row_index;
  impl_->Download(&row_index, winner_buffer, sizeof(row_index));
  quantized_selection_download_bytes_ += sizeof(row_index);
  CHECK_GE(row_index, 0); CHECK_LT(row_index, r);
  auto read = [&](void* target, const void* source, size_t size) {
    ACL_CHECK(aclrtMemcpy(target, size, source, size, ACL_MEMCPY_DEVICE_TO_HOST));
    quantized_selection_download_bytes_ += size;
  };
  double gain;
  read(&gain, static_cast<double*>(buffers[masked_score]->data()) + row_index, sizeof(gain));
  if (gain < 0.0) return result;
  CHECK(std::isfinite(gain));
  int64_t index;
  read(&index, static_cast<int64_t*>(packed_winners.data()) + row_index, sizeof(index));
  CHECK_GE(index, 0); CHECK_LT(index, n);
  result.integer_statistics = scan->candidates[index];
  auto& stats = result.integer_statistics;
  int64_t* integers[] = {&stats.left_gradient, &stats.left_hessian, &stats.right_gradient, &stats.right_hessian};
  for (int lane = 0; lane < 4; ++lane)
    read(integers[lane], static_cast<int64_t*>(scan->values.data()) + lane * n + index, sizeof(int64_t));
  auto& split = result.split;
  split.feature = stats.feature; split.threshold = stats.threshold; split.default_left = stats.default_left; split.gain = gain;
  double values[10];
  for (int lane = 0; lane < 10; ++lane)
    read(values + lane, static_cast<double*>(evaluation->values.data()) + lane * n + index, sizeof(double));
  split.left_gradient = values[0]; split.left_hessian = values[1];
  split.right_gradient = values[2]; split.right_hessian = values[3];
  split.left_count = static_cast<int>(values[4]); split.right_count = static_cast<int>(values[5]);
  split.left_output = values[8]; split.right_output = values[9];
  return result;
}

NPURuntime::PartitionHandle NPURuntime::UploadPartitionRows(const int32_t* rows, int64_t count) {
  CHECK_GE(count, 0);
  CHECK_LE(count, impl_->histogram_rows);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  auto result = std::make_shared<DevicePartition>();
  result->count = count;
  result->left_count = count;
  if (count != 0) {
    CHECK(rows != nullptr);
    result->rows = impl_->Upload(rows, static_cast<size_t>(count) * sizeof(int32_t));
  }
  return result;
}

NPURuntime::PartitionHandle NPURuntime::SplitDeviceRows(
    const int32_t* device_rows, int64_t count, int feature, uint32_t threshold,
    uint32_t default_bin, int missing_type, bool default_left) {
  CHECK_GE(feature, 0);
  CHECK_LT(feature, static_cast<int>(impl_->histogram_bin_counts.size()));
  CHECK_GE(count, 0);
  CHECK_LE(count, impl_->histogram_rows);
  CHECK(missing_type >= 0 && missing_type <= 2);
  CHECK_LT(threshold, static_cast<uint32_t>(impl_->histogram_bin_counts[feature]));
  CHECK_LT(default_bin, static_cast<uint32_t>(impl_->histogram_bin_counts[feature]));
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  auto result = std::make_shared<DevicePartition>();
  result->count = count;
  result->rows = DeviceBuffer(static_cast<size_t>(count) * sizeof(int32_t));
  if (count == 0) return result;
  CHECK(device_rows != nullptr);
  const int32_t stride = impl_->histogram_stride;
  const int32_t local_feature = feature % Impl::kHistogramTile;
  const int32_t offset = impl_->histogram_offsets[feature];
  const int32_t most_frequent = impl_->histogram_most_frequent[feature];
  const int32_t omitted_bin = most_frequent < 0 && offset == 1 ? 0 : most_frequent;
  // Tiles store (tile feature * stride + logical bin - offset); the omitted
  // most-frequent bin (or bin zero when offset=1) uses a common sentinel.
  // This is the exact same resident matrix used by BuildHistogram.
  const int32_t sentinel = Impl::kHistogramTile * stride;
  const int32_t threshold_value = local_feature * stride + static_cast<int32_t>(threshold) - offset;
  const int32_t missing_bin = missing_type == 1 ? static_cast<int32_t>(default_bin)
                                               : impl_->histogram_bin_counts[feature] - 1;
  const int32_t missing_value = missing_bin == omitted_bin
      ? sentinel : local_feature * stride + missing_bin - offset;
  const auto* tile = static_cast<const int32_t*>(
      impl_->histogram_tiles[feature / Impl::kHistogramTile].data());
  const int32_t* column = tile + static_cast<int64_t>(local_feature) * impl_->histogram_rows;
  TensorHandle t_column(const_cast<int32_t*>(column), {impl_->histogram_rows}, ACL_INT32);
  TensorHandle t_rows(const_cast<int32_t*>(device_rows), {count}, ACL_INT32);
  DeviceBuffer gathered(static_cast<size_t>(count) * sizeof(int32_t));
  TensorHandle t_gathered(gathered.data(), {count}, ACL_INT32);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnIndexSelectGetWorkspaceSize(t_column.get(), 0, t_rows.get(), t_gathered.get(), s, e);
  }, aclnnIndexSelect);

  DeviceBuffer base(count), omitted(count), left(count);
  TensorHandle t_base(base.data(), {count}, ACL_BOOL);
  TensorHandle t_omitted(omitted.data(), {count}, ACL_BOOL);
  TensorHandle t_left(left.data(), {count}, ACL_BOOL);
  IntScalarHandle limit(threshold_value), missing(missing_value), sentinel_scalar(sentinel);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnLeScalarGetWorkspaceSize(t_gathered.get(), limit.get(), t_base.get(), s, e);
  }, aclnnLeScalar);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnEqScalarGetWorkspaceSize(t_gathered.get(), sentinel_scalar.get(), t_omitted.get(), s, e);
  }, aclnnEqScalar);
  // In probe-only layouts with no explicit most-frequent bin, offset=1 still
  // omits bin zero. Restore its threshold decision as for a cached omission.
  if (omitted_bin >= 0 && omitted_bin <= static_cast<int32_t>(threshold)) {
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnLogicalOrGetWorkspaceSize(t_base.get(), t_omitted.get(), t_left.get(), s, e);
    }, aclnnLogicalOr);
  } else {
    ACL_CHECK(aclrtMemcpy(left.data(), count, base.data(), count, ACL_MEMCPY_DEVICE_TO_DEVICE));
  }
  // Override missing-value direction, independently of threshold ordering.
  if (missing_type != 0) {
    DeviceBuffer missing_bits(count), adjusted(count);
    TensorHandle t_missing(missing_bits.data(), {count}, ACL_BOOL);
    TensorHandle t_adjusted(adjusted.data(), {count}, ACL_BOOL);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnEqScalarGetWorkspaceSize(t_gathered.get(), missing.get(), t_missing.get(), s, e);
    }, aclnnEqScalar);
    if (default_left) {
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnLogicalOrGetWorkspaceSize(t_left.get(), t_missing.get(), t_adjusted.get(), s, e);
      }, aclnnLogicalOr);
    } else {
      DeviceBuffer not_missing(count);
      TensorHandle t_not_missing(not_missing.data(), {count}, ACL_BOOL);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnLogicalNotGetWorkspaceSize(t_missing.get(), t_not_missing.get(), s, e);
      }, aclnnLogicalNot);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnLogicalAndGetWorkspaceSize(t_left.get(), t_not_missing.get(), t_adjusted.get(), s, e);
      }, aclnnLogicalAnd);
    }
    ACL_CHECK(aclrtMemcpy(left.data(), count, adjusted.data(), count, ACL_MEMCPY_DEVICE_TO_DEVICE));
  }

  const size_t bytes = static_cast<size_t>(count) * sizeof(int32_t);
  DeviceBuffer bits(bytes), prefix(bytes), left_position(bytes), right_position(bytes), shifted(bytes), destination(bytes);
  TensorHandle t_bits(bits.data(), {count}, ACL_INT32);
  TensorHandle t_prefix(prefix.data(), {count}, ACL_INT32);
  TensorHandle t_left_position(left_position.data(), {count}, ACL_INT32);
  TensorHandle t_right_position(right_position.data(), {count}, ACL_INT32);
  TensorHandle t_shifted(shifted.data(), {count}, ACL_INT32);
  TensorHandle t_destination(destination.data(), {count}, ACL_INT32);
  TensorHandle t_positions(impl_->partition_positions.data(), {count}, ACL_INT32);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnCastGetWorkspaceSize(t_left.get(), ACL_INT32, t_bits.get(), s, e);
  }, aclnnCast);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnCumsumGetWorkspaceSize(t_bits.get(), 0, ACL_INT32, t_prefix.get(), s, e);
  }, aclnnCumsum);
  int32_t left_count = 0;
  ACL_CHECK(aclrtMemcpy(&left_count, sizeof(left_count),
                         static_cast<const int32_t*>(prefix.data()) + count - 1,
                         sizeof(left_count), ACL_MEMCPY_DEVICE_TO_HOST));
  result->left_count = left_count;
  CHECK_GE(result->left_count, 0);
  CHECK_LE(result->left_count, count);
  IntScalarHandle one(1), left_size(static_cast<int32_t>(result->left_count));
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnSubsGetWorkspaceSize(t_prefix.get(), one.get(), one.get(), t_left_position.get(), s, e);
  }, aclnnSubs);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnSubGetWorkspaceSize(t_positions.get(), t_prefix.get(), one.get(), t_right_position.get(), s, e);
  }, aclnnSub);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnAddsGetWorkspaceSize(t_right_position.get(), left_size.get(), one.get(), t_shifted.get(), s, e);
  }, aclnnAdds);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnSWhereGetWorkspaceSize(t_left.get(), t_left_position.get(),
                                       t_shifted.get(), t_destination.get(), s, e);
  }, aclnnSWhere);
  DeviceBuffer zeros = impl_->Zeros(bytes);
  TensorHandle t_zeros(zeros.data(), {count}, ACL_INT32);
  TensorHandle t_out(result->rows.data(), {count}, ACL_INT32);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnScatterGetWorkspaceSize(t_zeros.get(), 0, t_destination.get(),
                                        t_rows.get(), 0, t_out.get(), s, e);
  }, aclnnScatter);
  return result;
}

NPURuntime::PartitionHandle NPURuntime::SplitCategoricalRows(
    const int32_t* device_rows, int64_t count, int feature,
    const std::vector<uint32_t>& categories, uint32_t default_bin,
    int missing_type, bool /*default_left*/) {
  CHECK_GE(feature, 0);
  CHECK_LT(feature, static_cast<int>(impl_->histogram_bin_counts.size()));
  CHECK_GE(count, 0);
  CHECK_LE(count, impl_->histogram_rows);
  CHECK(missing_type >= 0 && missing_type <= 2);
  CHECK(!categories.empty());
  for (uint32_t category : categories) {
    CHECK_LT(category, static_cast<uint32_t>(impl_->histogram_bin_counts[feature]));
  }
  CHECK_LT(default_bin, static_cast<uint32_t>(impl_->histogram_bin_counts[feature]));
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  auto result = std::make_shared<DevicePartition>();
  result->count = count;
  result->rows = DeviceBuffer(static_cast<size_t>(count) * sizeof(int32_t));
  if (count == 0) return result;
  CHECK(device_rows != nullptr);
  const int32_t stride = impl_->histogram_stride;
  const int32_t local_feature = feature % Impl::kHistogramTile;
  const int32_t offset = impl_->histogram_offsets[feature];
  const int32_t most_frequent = impl_->histogram_most_frequent[feature];
  const int32_t omitted_bin = most_frequent < 0 && offset == 1 ? 0 : most_frequent;
  const int32_t sentinel = Impl::kHistogramTile * stride;
  const auto* tile = static_cast<const int32_t*>(
      impl_->histogram_tiles[feature / Impl::kHistogramTile].data());
  const int32_t* column = tile + static_cast<int64_t>(local_feature) * impl_->histogram_rows;
  TensorHandle t_column(const_cast<int32_t*>(column), {impl_->histogram_rows}, ACL_INT32);
  TensorHandle t_rows(const_cast<int32_t*>(device_rows), {count}, ACL_INT32);
  DeviceBuffer gathered(static_cast<size_t>(count) * sizeof(int32_t));
  TensorHandle t_gathered(gathered.data(), {count}, ACL_INT32);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnIndexSelectGetWorkspaceSize(t_column.get(), 0, t_rows.get(),
                                            t_gathered.get(), s, e);
  }, aclnnIndexSelect);

  DeviceBuffer left_bits(static_cast<size_t>(count));
  TensorHandle t_left(left_bits.data(), {count}, ACL_BOOL);
  bool first_category = true;
  for (uint32_t category : categories) {
    const int32_t category_value = category == static_cast<uint32_t>(omitted_bin)
        ? sentinel : local_feature * stride + static_cast<int32_t>(category) - offset;
    DeviceBuffer matched(static_cast<size_t>(count));
    TensorHandle t_matched(matched.data(), {count}, ACL_BOOL);
    IntScalarHandle category_scalar(category_value);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnEqScalarGetWorkspaceSize(t_gathered.get(), category_scalar.get(),
                                           t_matched.get(), s, e);
    }, aclnnEqScalar);
    if (first_category) {
      ACL_CHECK(aclrtMemcpy(left_bits.data(), count, matched.data(), count,
                            ACL_MEMCPY_DEVICE_TO_DEVICE));
      first_category = false;
    } else {
      DeviceBuffer combined(static_cast<size_t>(count));
      TensorHandle t_combined(combined.data(), {count}, ACL_BOOL);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnLogicalOrGetWorkspaceSize(t_left.get(), t_matched.get(),
                                              t_combined.get(), s, e);
      }, aclnnLogicalOr);
      ACL_CHECK(aclrtMemcpy(left_bits.data(), count, combined.data(), count,
                            ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
  }
  // Categorical bin 0 represents missing/unknown values and is never selected
  // by the categorical scanner. All other bins follow set membership, even
  // when the mapper reports NaN; numeric last-bin missing rules do not apply.

  const size_t bytes = static_cast<size_t>(count) * sizeof(int32_t);
  DeviceBuffer bits(bytes), prefix(bytes), left_position(bytes), right_position(bytes),
      shifted(bytes), destination(bytes);
  TensorHandle t_bits(bits.data(), {count}, ACL_INT32);
  TensorHandle t_prefix(prefix.data(), {count}, ACL_INT32);
  TensorHandle t_left_position(left_position.data(), {count}, ACL_INT32);
  TensorHandle t_right_position(right_position.data(), {count}, ACL_INT32);
  TensorHandle t_shifted(shifted.data(), {count}, ACL_INT32);
  TensorHandle t_destination(destination.data(), {count}, ACL_INT32);
  TensorHandle t_positions(impl_->partition_positions.data(), {count}, ACL_INT32);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnCastGetWorkspaceSize(t_left.get(), ACL_INT32, t_bits.get(), s, e);
  }, aclnnCast);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnCumsumGetWorkspaceSize(t_bits.get(), 0, ACL_INT32,
                                       t_prefix.get(), s, e);
  }, aclnnCumsum);
  int32_t left_count = 0;
  ACL_CHECK(aclrtMemcpy(&left_count, sizeof(left_count),
                         static_cast<const int32_t*>(prefix.data()) + count - 1,
                         sizeof(left_count), ACL_MEMCPY_DEVICE_TO_HOST));
  result->left_count = left_count;
  IntScalarHandle one(1), left_size(left_count);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnSubsGetWorkspaceSize(t_prefix.get(), one.get(), one.get(),
                                     t_left_position.get(), s, e);
  }, aclnnSubs);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnSubGetWorkspaceSize(t_positions.get(), t_prefix.get(), one.get(),
                                    t_right_position.get(), s, e);
  }, aclnnSub);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnAddsGetWorkspaceSize(t_right_position.get(), left_size.get(), one.get(),
                                     t_shifted.get(), s, e);
  }, aclnnAdds);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnSWhereGetWorkspaceSize(t_left.get(), t_left_position.get(),
                                       t_shifted.get(), t_destination.get(), s, e);
  }, aclnnSWhere);
  DeviceBuffer zeros = impl_->Zeros(bytes);
  TensorHandle t_zeros(zeros.data(), {count}, ACL_INT32);
  TensorHandle t_out(result->rows.data(), {count}, ACL_INT32);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnScatterGetWorkspaceSize(t_zeros.get(), 0, t_destination.get(),
                                        t_rows.get(), 0, t_out.get(), s, e);
  }, aclnnScatter);
  return result;
}

int64_t NPURuntime::LeftCount(const PartitionHandle& partition) const {
  CHECK(partition != nullptr);
  return partition->left_count;
}

const int32_t* NPURuntime::PartitionRowsDevice(const PartitionHandle& partition, bool left) const {
  CHECK(partition != nullptr);
  if (partition->count == 0) return nullptr;
  return static_cast<const int32_t*>(partition->rows.data()) + (left ? 0 : partition->left_count);
}

std::vector<int32_t> NPURuntime::ReadPartitionRows(const PartitionHandle& partition) {
  CHECK(partition != nullptr);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  std::vector<int32_t> rows(static_cast<size_t>(partition->count));
  if (!rows.empty()) impl_->Download(rows.data(), partition->rows, rows.size() * sizeof(int32_t));
  partition_rows_downloaded_ += rows.size();
  return rows;
}

std::vector<int32_t> NPURuntime::ReadPartitionLeafRows(const PartitionHandle& partition,
                                                        bool left) {
  CHECK(partition != nullptr);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  const int64_t count = left ? partition->left_count : partition->count - partition->left_count;
  std::vector<int32_t> rows(static_cast<size_t>(count));
  if (!rows.empty()) {
    ACL_CHECK(aclrtMemcpy(rows.data(), rows.size() * sizeof(int32_t),
                          PartitionRowsDevice(partition, left), rows.size() * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
  }
  partition_rows_downloaded_ += rows.size();
  return rows;
}

void NPURuntime::AddPredictionToScore(const std::vector<ScoreLeaf>& leaves,
                                      int64_t num_data, double* scores) {
  CHECK_GE(num_data, 0);
  CHECK_EQ(num_data, impl_->histogram_rows);
  if (num_data == 0) return;
  CHECK(scores != nullptr);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  const size_t score_bytes = static_cast<size_t>(num_data) * sizeof(double);
  auto score_it = impl_->score_buffers.find(scores);
  if (score_it == impl_->score_buffers.end()) {
    score_it = impl_->score_buffers.emplace(scores, Impl::ScoreBuffer()).first;
    score_it->second.num_data = num_data;
    score_it->second.host_shadow.assign(scores, scores + num_data);
    score_it->second.device = impl_->Upload(scores, score_bytes);
    ++score_uploads_;
  } else {
    // The CPU booster can change out-of-bag entries between trees. Upload only
    // when that host-visible score slice changed; ordinary trees reuse the
    // resident vector without a redundant host-to-device copy.
    if (score_it->second.num_data != num_data ||
        !std::equal(score_it->second.host_shadow.begin(),
                    score_it->second.host_shadow.end(), scores)) {
      score_it->second.device = DeviceBuffer();
      score_it->second.num_data = num_data;
      score_it->second.host_shadow.assign(scores, scores + num_data);
      score_it->second.device = impl_->Upload(scores, score_bytes);
      ++score_uploads_;
    }
  }
  DeviceBuffer& device_scores = score_it->second.device;
  TensorHandle t_scores(device_scores.data(), {num_data}, ACL_DOUBLE);
  for (const auto& leaf : leaves) {
    CHECK(leaf.partition != nullptr);
    const int64_t count = leaf.left ? leaf.partition->left_count
                                    : leaf.partition->count - leaf.partition->left_count;
    CHECK_GE(count, 0);
    CHECK_LE(count, num_data);
    if (count == 0) continue;
    TensorHandle t_rows(const_cast<int32_t*>(PartitionRowsDevice(leaf.partition, leaf.left)),
                         {count}, ACL_INT32);
    ScalarHandle value(leaf.output, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnInplaceScatterValueGetWorkspaceSize(t_scores.get(), 0, t_rows.get(),
                                                       value.get(), 1, s, e);
    }, aclnnInplaceScatterValue);
  }
  impl_->Download(scores, device_scores, static_cast<size_t>(num_data) * sizeof(double));
  score_it->second.host_shadow.assign(scores, scores + num_data);
  ++score_downloads_;
  ++score_update_calls_;
}

void NPURuntime::ResetScoreBuffers() {
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  impl_->score_buffers.clear();
}

NPURuntime::HistogramResult NPURuntime::ReadHistogram(const HistogramHandle& histogram) {
  CHECK(histogram != nullptr);
  HistogramResult result;
  result.gradients.resize(histogram->features);
  result.hessians.resize(histogram->features);
  const size_t value_count = histogram->values.bytes() / sizeof(double);
  std::vector<double> host(value_count);
  impl_->Download(host.data(), histogram->values, histogram->values.bytes());
  const int64_t plane = static_cast<int64_t>(histogram->features) * histogram->stride;
  for (int feature = 0; feature < histogram->features; ++feature) {
    if (!histogram->used[feature]) continue;
    int n = impl_->histogram_bin_counts[feature] - impl_->histogram_offsets[feature];
    const auto begin = host.begin() + feature * histogram->stride;
    result.gradients[feature].assign(begin, begin + n);
    result.hessians[feature].assign(begin + plane, begin + plane + n);
  }
  return result;
}

NPURuntime::HistogramResult NPURuntime::BatchedHistogram(
    const int32_t* row_indices, int64_t count, const std::vector<int8_t>& used) {
  return ReadHistogram(BuildHistogram(row_indices, count, used));
}

void NPURuntime::FixHistogram(const HistogramHandle& histogram,
                              double sum_gradient, double sum_hessian) {
  CHECK(histogram != nullptr);
  CHECK(!histogram->fixed);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  const int64_t plane = static_cast<int64_t>(histogram->features) * histogram->stride;
  const int64_t elements = 2 * plane;
  std::vector<int64_t> last_indices, destinations;
  std::vector<double> totals;
  for (int part = 0; part < 2; ++part) {
    for (int feature = 0; feature < histogram->features; ++feature) {
      const int mf = impl_->histogram_most_frequent[feature];
      if (!histogram->used[feature] || mf <= 0) continue;
      const int64_t base = part * plane + feature * histogram->stride;
      last_indices.push_back(base + histogram->stride - 1);
      const int offset = impl_->histogram_offsets[feature];
      destinations.push_back(base + mf - offset);
      totals.push_back(part == 0 ? sum_gradient : sum_hessian);
    }
  }
  const int64_t fixes = static_cast<int64_t>(last_indices.size());
  if (fixes != 0) {
    DeviceBuffer prefix(elements * sizeof(double));
    TensorHandle input(histogram->values.data(), {2 * histogram->features, histogram->stride}, ACL_DOUBLE);
    TensorHandle output(prefix.data(), {2 * histogram->features, histogram->stride}, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnCumsumGetWorkspaceSize(input.get(), 1, ACL_DOUBLE, output.get(), s, e);
    }, aclnnCumsum);
    DeviceBuffer indices = impl_->Upload(last_indices.data(), fixes * sizeof(int64_t));
    DeviceBuffer sums(fixes * sizeof(double));
    DeviceBuffer parents = impl_->Upload(totals.data(), fixes * sizeof(double));
    DeviceBuffer residuals(fixes * sizeof(double));
    TensorHandle flat(prefix.data(), {elements}, ACL_DOUBLE);
    TensorHandle index(indices.data(), {fixes}, ACL_INT64), sum(sums.data(), {fixes}, ACL_DOUBLE);
    TensorHandle parent(parents.data(), {fixes}, ACL_DOUBLE), residual(residuals.data(), {fixes}, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(flat.get(), 0, index.get(), sum.get(), s, e);
    }, aclnnIndexSelect);
    ScalarHandle one(1.0, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnSubGetWorkspaceSize(parent.get(), sum.get(), one.get(), residual.get(), s, e);
    }, aclnnSub);
    for (int64_t i = 0; i < fixes; ++i) {
      ACL_CHECK(aclrtMemcpy(static_cast<double*>(histogram->values.data()) + destinations[i], sizeof(double),
                            static_cast<double*>(residuals.data()) + i, sizeof(double), ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
  }
  histogram->fixed = true;
}

void NPURuntime::SubtractHistogram(const HistogramHandle& parent,
                                   const HistogramHandle& smaller) {
  CHECK(parent != nullptr && smaller != nullptr);
  CHECK(parent->fixed && smaller->fixed);
  CHECK(parent != smaller);
  CHECK_EQ(parent->features, smaller->features);
  CHECK_EQ(parent->stride, smaller->stride);
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));
  DeviceBuffer difference(parent->values.bytes());
  const int64_t elements = static_cast<int64_t>(parent->values.bytes() / sizeof(double));
  TensorHandle a(parent->values.data(), {elements}, ACL_DOUBLE);
  TensorHandle b(smaller->values.data(), {elements}, ACL_DOUBLE);
  TensorHandle out(difference.data(), {elements}, ACL_DOUBLE);
  ScalarHandle one(1.0, ACL_DOUBLE);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnSubGetWorkspaceSize(a.get(), b.get(), one.get(), out.get(), s, e);
  }, aclnnSub);
  parent->values = std::move(difference);
  parent->used = smaller->used;
}

void NPURuntime::Histogram(const std::vector<int64_t>& indices,
                           const std::vector<float>& gradients,
                           const std::vector<float>& hessians, int64_t bins,
                           std::vector<float>* out_gradients,
                           std::vector<float>* out_hessians) {
  const int64_t count = static_cast<int64_t>(indices.size());
  if (count == 0 || bins == 0) {
    out_gradients->assign(static_cast<size_t>(bins), 0.0f);
    out_hessians->assign(static_cast<size_t>(bins), 0.0f);
    return;
  }
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));

  const size_t value_bytes = static_cast<size_t>(count) * sizeof(float);
  const size_t index_bytes = static_cast<size_t>(count) * sizeof(int64_t);
  const size_t bin_bytes = static_cast<size_t>(bins) * sizeof(float);

  DeviceBuffer d_index = impl_->Upload(indices.data(), index_bytes);
  DeviceBuffer d_grad = impl_->Upload(gradients.data(), value_bytes);
  DeviceBuffer d_hess = impl_->Upload(hessians.data(), value_bytes);
  DeviceBuffer d_zero_grad = impl_->Zeros(bin_bytes);
  DeviceBuffer d_zero_hess = impl_->Zeros(bin_bytes);
  DeviceBuffer d_out_grad = impl_->Zeros(bin_bytes);
  DeviceBuffer d_out_hess = impl_->Zeros(bin_bytes);

  const std::vector<int64_t> value_shape = {count};
  const std::vector<int64_t> bin_shape = {bins};
  TensorHandle t_index(d_index.data(), value_shape, ACL_INT64);
  TensorHandle t_grad(d_grad.data(), value_shape, ACL_FLOAT);
  TensorHandle t_hess(d_hess.data(), value_shape, ACL_FLOAT);
  TensorHandle t_zero_grad(d_zero_grad.data(), bin_shape, ACL_FLOAT);
  TensorHandle t_zero_hess(d_zero_hess.data(), bin_shape, ACL_FLOAT);
  TensorHandle t_out_grad(d_out_grad.data(), bin_shape, ACL_FLOAT);
  TensorHandle t_out_hess(d_out_hess.data(), bin_shape, ACL_FLOAT);

  impl_->Launch(
      [&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnScatterAddGetWorkspaceSize(t_zero_grad.get(), 0, t_index.get(),
                                               t_grad.get(), t_out_grad.get(),
                                               size, executor);
      },
      aclnnScatterAdd);
  impl_->Launch(
      [&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnScatterAddGetWorkspaceSize(t_zero_hess.get(), 0, t_index.get(),
                                               t_hess.get(), t_out_hess.get(),
                                               size, executor);
      },
      aclnnScatterAdd);

  out_gradients->resize(static_cast<size_t>(bins));
  out_hessians->resize(static_cast<size_t>(bins));
  impl_->Download(out_gradients->data(), d_out_grad, bin_bytes);
  impl_->Download(out_hessians->data(), d_out_hess, bin_bytes);
  ++histogram_calls_;
}
std::vector<double> NPURuntime::PrefixSum(const std::vector<double>& input,
                                          int64_t rows, int64_t columns) {
  std::vector<double> output(input.size(), 0.0);
  if (input.empty() || rows == 0 || columns == 0) {
    return output;
  }
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));

  const size_t bytes = input.size() * sizeof(double);
  DeviceBuffer d_in = impl_->Upload(input.data(), bytes);
  DeviceBuffer d_out = impl_->Zeros(bytes);

  // Scanning along the last axis keeps each feature's bins independent.
  const std::vector<int64_t> shape = {rows, columns};
  TensorHandle t_in(d_in.data(), shape, ACL_DOUBLE);
  TensorHandle t_out(d_out.data(), shape, ACL_DOUBLE);

  impl_->Launch(
      [&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnCumsumGetWorkspaceSize(t_in.get(), 1, ACL_DOUBLE, t_out.get(),
                                           size, executor);
      },
      aclnnCumsum);

  impl_->Download(output.data(), d_out, bytes);
  ++prefix_calls_;
  return output;
}

NPURuntime::Gains NPURuntime::SplitGains(const std::vector<double>& left_g,
                                         const std::vector<double>& left_h,
                                         const std::vector<double>& right_g,
                                         const std::vector<double>& right_h,
                                         double lambda_l1, double lambda_l2) {
  Gains result;
  const int64_t count = static_cast<int64_t>(left_g.size());
  result.gain.assign(static_cast<size_t>(count), 0.0);
  result.left_output.assign(static_cast<size_t>(count), 0.0);
  result.right_output.assign(static_cast<size_t>(count), 0.0);
  if (count == 0) {
    return result;
  }
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));

  const size_t bytes = static_cast<size_t>(count) * sizeof(double);
  const std::vector<int64_t> shape = {count};

  DeviceBuffer d_lg = impl_->Upload(left_g.data(), bytes);
  DeviceBuffer d_lh = impl_->Upload(left_h.data(), bytes);
  DeviceBuffer d_rg = impl_->Upload(right_g.data(), bytes);
  DeviceBuffer d_rh = impl_->Upload(right_h.data(), bytes);
  DeviceBuffer d_left_out = impl_->Zeros(bytes);
  DeviceBuffer d_right_out = impl_->Zeros(bytes);
  DeviceBuffer d_left_gain = impl_->Zeros(bytes);
  DeviceBuffer d_right_gain = impl_->Zeros(bytes);
  DeviceBuffer d_scratch_a = impl_->Zeros(bytes);
  DeviceBuffer d_scratch_b = impl_->Zeros(bytes);

  TensorHandle t_lg(d_lg.data(), shape, ACL_DOUBLE);
  TensorHandle t_lh(d_lh.data(), shape, ACL_DOUBLE);
  TensorHandle t_rg(d_rg.data(), shape, ACL_DOUBLE);
  TensorHandle t_rh(d_rh.data(), shape, ACL_DOUBLE);
  TensorHandle t_left_out(d_left_out.data(), shape, ACL_DOUBLE);
  TensorHandle t_right_out(d_right_out.data(), shape, ACL_DOUBLE);
  TensorHandle t_left_gain(d_left_gain.data(), shape, ACL_DOUBLE);
  TensorHandle t_right_gain(d_right_gain.data(), shape, ACL_DOUBLE);
  TensorHandle t_scratch_a(d_scratch_a.data(), shape, ACL_DOUBLE);
  TensorHandle t_scratch_b(d_scratch_b.data(), shape, ACL_DOUBLE);

  // Mirrors CalculateSplittedLeafOutput and GetLeafGainGivenOutput with
  // max_delta_step and path smoothing disabled:
  //   t = -ThresholdL1(g) / (h + l2),  gain = -(2*g*t + (h + l2)*t^2)
  // where ThresholdL1(g) = sign(g) * max(|g| - l1, 0).
  auto threshold_l1 = [&](const TensorHandle& grad, const TensorHandle& out) {
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnAbsGetWorkspaceSize(grad.get(), t_scratch_a.get(), size,
                                          executor);
        },
        aclnnAbs);
    ScalarHandle l1(lambda_l1, ACL_DOUBLE);
    ScalarHandle alpha(1.0, ACL_DOUBLE);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnSubsGetWorkspaceSize(t_scratch_a.get(), l1.get(), alpha.get(),
                                           t_scratch_b.get(), size, executor);
        },
        aclnnSubs);
    ScalarHandle zero(0.0, ACL_DOUBLE);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnClampMinGetWorkspaceSize(t_scratch_b.get(), zero.get(),
                                               t_scratch_a.get(), size, executor);
        },
        aclnnClampMin);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnSignGetWorkspaceSize(grad.get(), t_scratch_b.get(), size,
                                           executor);
        },
        aclnnSign);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnMulGetWorkspaceSize(t_scratch_a.get(), t_scratch_b.get(),
                                          out.get(), size, executor);
        },
        aclnnMul);
  };

  // output = -ThresholdL1(g) / (h + l2)
  auto leaf_output = [&](const TensorHandle& grad, const TensorHandle& hess,
                         const TensorHandle& out) {
    threshold_l1(grad, out);
    ScalarHandle l2(lambda_l2, ACL_DOUBLE);
    ScalarHandle alpha(1.0, ACL_DOUBLE);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnAddsGetWorkspaceSize(hess.get(), l2.get(), alpha.get(),
                                           t_scratch_a.get(), size, executor);
        },
        aclnnAdds);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnDivGetWorkspaceSize(out.get(), t_scratch_a.get(),
                                          t_scratch_b.get(), size, executor);
        },
        aclnnDiv);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnNegGetWorkspaceSize(t_scratch_b.get(), out.get(), size,
                                          executor);
        },
        aclnnNeg);
  };

  leaf_output(t_lg, t_lh, t_left_out);
  leaf_output(t_rg, t_rh, t_right_out);

  // At the L1-regularized optimum gain = (h + l2) * output^2.
  // This includes the L1 term, unlike using the original gradient in the
  // unregularized quadratic. Mul also supports FP64 where Square does not.
  auto leaf_gain = [&](const TensorHandle& hess,
                       const TensorHandle& output, const TensorHandle& gain) {
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnMulGetWorkspaceSize(output.get(), output.get(),
                                          t_scratch_b.get(), size, executor);
        },
        aclnnMul);
    ScalarHandle l2(lambda_l2, ACL_DOUBLE);
    ScalarHandle alpha(1.0, ACL_DOUBLE);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnAddsGetWorkspaceSize(hess.get(), l2.get(), alpha.get(),
                                           gain.get(), size, executor);
        },
        aclnnAdds);
    impl_->Launch(
        [&](uint64_t* size, aclOpExecutor** executor) {
          return aclnnInplaceMulGetWorkspaceSize(gain.get(), t_scratch_b.get(),
                                                 size, executor);
        },
        aclnnInplaceMul);
  };

  leaf_gain(t_lh, t_left_out, t_left_gain);
  leaf_gain(t_rh, t_right_out, t_right_gain);

  ScalarHandle alpha(1.0, ACL_DOUBLE);
  impl_->Launch(
      [&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnInplaceAddGetWorkspaceSize(t_left_gain.get(),
                                               t_right_gain.get(), alpha.get(),
                                               size, executor);
      },
      aclnnInplaceAdd);

  impl_->Download(result.gain.data(), d_left_gain, bytes);
  impl_->Download(result.left_output.data(), d_left_out, bytes);
  impl_->Download(result.right_output.data(), d_right_out, bytes);
  ++gain_calls_;
  return result;
}

int64_t NPURuntime::ArgMax(const std::vector<double>& input) {
  if (input.empty()) {
    return -1;
  }
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));

  const int64_t count = static_cast<int64_t>(input.size());
  DeviceBuffer d_in = impl_->Upload(input.data(), input.size() * sizeof(double));
  DeviceBuffer d_out = impl_->Zeros(sizeof(int64_t));

  const std::vector<int64_t> shape = {count};
  const std::vector<int64_t> scalar_shape = {1};
  TensorHandle t_in(d_in.data(), shape, ACL_DOUBLE);
  TensorHandle t_out(d_out.data(), scalar_shape, ACL_INT64);

  impl_->Launch(
      [&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnArgMaxGetWorkspaceSize(t_in.get(), 0, false, t_out.get(), size,
                                           executor);
      },
      aclnnArgMax);

  int64_t index = 0;
  impl_->Download(&index, d_out, sizeof(int64_t));
  ++argmax_calls_;
  return index;
}

namespace {

double CategoricalThresholdL1(double gradient, double lambda_l1) {
  if (lambda_l1 <= 0.0) return gradient;
  const double magnitude = std::max(std::fabs(gradient) - lambda_l1, 0.0);
  return std::copysign(magnitude, gradient);
}

double CategoricalLeafOutput(double gradient, double hessian, double lambda_l1,
                             double lambda_l2, double max_delta_step,
                             double path_smooth, int count, double parent_output) {
  double output = -CategoricalThresholdL1(gradient, lambda_l1) / (hessian + lambda_l2);
  if (max_delta_step > 0.0) {
    output = std::max(-max_delta_step, std::min(max_delta_step, output));
  }
  if (path_smooth > 0.0) {
    const double ratio = count / path_smooth;
    output = output * ratio / (ratio + 1.0) + parent_output / (ratio + 1.0);
  }
  return output;
}

double CategoricalGainGivenOutput(double gradient, double hessian, double lambda_l1,
                                  double lambda_l2, double output) {
  return -(2.0 * CategoricalThresholdL1(gradient, lambda_l1) * output +
           (hessian + lambda_l2) * output * output);
}

double CategoricalLeafGain(double gradient, double hessian, double lambda_l1,
                           double lambda_l2, double max_delta_step,
                           double path_smooth, int count, double parent_output) {
  if (max_delta_step <= 0.0 && path_smooth <= 0.0) {
    const double thresholded = CategoricalThresholdL1(gradient, lambda_l1);
    return thresholded * thresholded / (hessian + lambda_l2);
  }
  const double output = CategoricalLeafOutput(gradient, hessian, lambda_l1, lambda_l2,
                                              max_delta_step, path_smooth, count,
                                              parent_output);
  return CategoricalGainGivenOutput(gradient, hessian, lambda_l1, lambda_l2, output);
}

}  // namespace

NPURuntime::SplitResult NPURuntime::FindBestSplit(
    const std::vector<SplitFeature>& features,
    double sum_gradient, double sum_hessian, int num_data,
    double lambda_l1, double lambda_l2,
    int min_data_in_leaf, double min_sum_hessian_in_leaf,
    double min_gain_to_split, const HistogramHandle& histogram,
    double max_delta_step, double path_smooth, double parent_output) {
  ++split_calls_;
  SplitResult result;
  result.splittable.resize(features.size(), 0);
  if (features.empty() || num_data <= 0) return result;
  if (!std::isfinite(max_delta_step) || !std::isfinite(path_smooth) ||
      path_smooth < 0.0 || !std::isfinite(parent_output)) {
    Log::Fatal("NPU split received invalid leaf-output parameters");
  }
  ACL_CHECK(aclrtSetCurrentContext(impl_->context));

  // Categorical histograms are still built on NPU. Their ordering and split
  // arithmetic follow FeatureHistogram::FindBestThresholdCategoricalInner;
  // the compact category ordering is decoded here because ACL has no stable
  // device-side sort primitive in the supported FP64 operator set.
  SplitResult categorical_result;
  categorical_result.splittable.resize(features.size(), 0);
  bool has_categorical = false;
  HistogramResult categorical_histogram;
  for (const auto& feature : features) {
    if (feature.categorical) {
      has_categorical = true;
      break;
    }
  }
  if (histogram && !has_categorical &&
      std::getenv("LGBM_NPU_DISABLE_FUSED_SPLIT") == nullptr) {
    // The INLINE AICPU fast path consumes only resident FP64 histogram data.
    // Keep this dispatch before all RNG draws so category/no-histogram calls
    // execute the unchanged path below, with exactly one draw per feature.
    CHECK_LE(features.size(), static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    std::vector<size_t> order(features.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return features[a].feature < features[b].feature;
    });
    std::vector<LGBMNPUFusedSplitFeature> packed;
    packed.reserve(features.size());
    for (size_t fi : order) {
      const auto& f = features[fi];
      if (f.offset != 0 && f.offset != 1) {
        Log::Fatal("NPU split supports only histogram offsets 0 and 1");
      }
      if (f.missing_type < 0 || f.missing_type > 2 ||
          f.default_bin < 0 || f.default_bin >= f.num_bin) {
        Log::Fatal("NPU split received unsupported missing-value metadata");
      }
      if (!std::isfinite(f.penalty) || f.penalty < 0.0 ||
          !std::isfinite(sum_gradient) || !std::isfinite(sum_hessian) ||
          sum_hessian <= 0.0 || !std::isfinite(lambda_l1) || lambda_l1 < 0.0 ||
          !std::isfinite(lambda_l2) || lambda_l2 < 0.0 ||
          !std::isfinite(min_sum_hessian_in_leaf) ||
          !std::isfinite(min_gain_to_split)) {
        Log::Fatal("NPU split received invalid parameters or feature penalty");
      }
      LGBMNPUFusedSplitFeature metadata = {};
      metadata.feature = f.feature;
      metadata.input_index = static_cast<int32_t>(fi);
      metadata.histogram_index = f.histogram_index;
      metadata.num_bin = f.num_bin;
      metadata.offset = f.offset;
      metadata.default_bin = f.default_bin;
      metadata.missing_type = f.missing_type;
      metadata.enabled = f.enabled;
      metadata.randomized = f.extra_tree_random != nullptr;
      metadata.penalty = f.penalty;
      if (f.num_bin > 1) {
        // Match the old scanner's stable feature order, including disabled
        // features. A two-bin feature has a fixed threshold and draws none.
        if (f.extra_tree_random != nullptr && f.num_bin - 2 > 0) {
          metadata.random_threshold = f.extra_tree_random->NextInt(0, f.num_bin - 2);
        }
        CHECK(histogram->fixed);
        CHECK_GE(f.histogram_index, 0);
        CHECK_LT(f.histogram_index, histogram->features);
        CHECK(histogram->used[f.histogram_index]);
        CHECK_EQ(f.num_bin, impl_->histogram_bin_counts[f.histogram_index]);
        CHECK_EQ(f.offset, impl_->histogram_offsets[f.histogram_index]);
      }
      packed.push_back(metadata);
    }
    LGBMNPUFusedSplitParameters parameters = {};
    parameters.histogram_plane = static_cast<int64_t>(histogram->features) * histogram->stride;
    parameters.histogram_stride = histogram->stride;
    parameters.num_features = static_cast<int32_t>(features.size());
    parameters.num_data = num_data;
    parameters.min_data_in_leaf = min_data_in_leaf;
    parameters.sum_gradient = sum_gradient;
    parameters.sum_hessian = sum_hessian;
    parameters.lambda_l1 = lambda_l1;
    parameters.lambda_l2 = lambda_l2;
    parameters.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf;
    parameters.min_gain_to_split = min_gain_to_split;
    parameters.max_delta_step = max_delta_step;
    parameters.path_smooth = path_smooth;
    parameters.parent_output = parent_output;
    DeviceBuffer device_features = impl_->Upload(
        packed.data(), packed.size() * sizeof(LGBMNPUFusedSplitFeature));
    DeviceBuffer device_parameters = impl_->Upload(&parameters, sizeof(parameters));
    DeviceBuffer device_winner = impl_->Zeros(sizeof(LGBMNPUFusedSplitWinner));
    DeviceBuffer device_flags(features.size() * sizeof(uint8_t));
    LGBMNPUFusedSplitKernel(
        1, nullptr, impl_->stream, static_cast<const double*>(histogram->values.data()),
        static_cast<const LGBMNPUFusedSplitFeature*>(device_features.data()),
        static_cast<const LGBMNPUFusedSplitParameters*>(device_parameters.data()),
        static_cast<LGBMNPUFusedSplitWinner*>(device_winner.data()),
        static_cast<uint8_t*>(device_flags.data()));
    ACL_CHECK(aclrtSynchronizeStream(impl_->stream));
    LGBMNPUFusedSplitWinner winner;
    impl_->Download(&winner, device_winner, sizeof(winner));
    if (winner.completed != 1) {
      Log::Fatal("NPU fused split kernel did not complete; launch may have failed");
    }
    impl_->Download(result.splittable.data(), device_flags, result.splittable.size());
    ++argmax_calls_;
    if (winner.feature < 0) return result;
    if (!std::isfinite(winner.gain)) Log::Fatal("NPU split produced a nonfinite winning gain");
    result.feature = winner.feature;
    result.threshold = winner.threshold;
    result.default_left = winner.default_left != 0;
    result.gain = winner.gain;
    result.left_output = winner.left_output;
    result.right_output = winner.right_output;
    result.left_gradient = winner.left_gradient;
    result.right_gradient = winner.right_gradient;
    result.left_hessian = winner.left_hessian;
    result.right_hessian = winner.right_hessian;
    result.left_count = winner.left_count;
    result.right_count = winner.right_count;
    return result;
  }
  if (has_categorical && histogram) {
    categorical_histogram = ReadHistogram(histogram);
  }
  for (size_t feature_index = 0; feature_index < features.size(); ++feature_index) {
    const auto& feature = features[feature_index];
    if (!feature.categorical || feature.num_bin <= 1) continue;
    if (feature.max_cat_threshold <= 0 || feature.max_cat_to_onehot <= 0 ||
        feature.min_data_per_group < 0 || feature.cat_smooth < 0.0 ||
        feature.cat_l2 < 0.0) {
      Log::Fatal("NPU categorical split received invalid category parameters");
    }
    if (feature.offset != 0 && feature.offset != 1) {
      Log::Fatal("NPU categorical split supports only histogram offsets 0 and 1");
    }
    if (feature.missing_type < 0 || feature.missing_type > 2 ||
        feature.default_bin < 0 || feature.default_bin >= feature.num_bin) {
      Log::Fatal("NPU categorical split received unsupported missing-value metadata");
    }
    std::vector<double> gradients, hessians;
    if (histogram) {
      CHECK_GE(feature.histogram_index, 0);
      CHECK_LT(feature.histogram_index, static_cast<int>(categorical_histogram.gradients.size()));
      gradients.assign(categorical_histogram.gradients[feature.histogram_index].begin(),
                       categorical_histogram.gradients[feature.histogram_index].end());
      hessians.assign(categorical_histogram.hessians[feature.histogram_index].begin(),
                      categorical_histogram.hessians[feature.histogram_index].end());
    } else {
      gradients = feature.gradients;
      hessians = feature.hessians;
    }
    const int bin_start = 1 - feature.offset;
    const int bin_end = feature.num_bin - feature.offset;
    CHECK_GE(gradients.size(), static_cast<size_t>(bin_end));
    CHECK_EQ(hessians.size(), gradients.size());
    const double category_total_hessian = sum_hessian + 2.0 * kEpsilon;
    const double count_factor = static_cast<double>(num_data) / category_total_hessian;
    const bool use_onehot = feature.num_bin <= feature.max_cat_to_onehot;
    // Smoothing uses the supplied (already smoothed) parent output. The
    // parent's regularizer is always lambda_l2, never the grouped cat_l2.
    const double parent_gain = path_smooth > 0.0
        ? CategoricalGainGivenOutput(sum_gradient, category_total_hessian,
                                      lambda_l1, lambda_l2, parent_output)
        : CategoricalLeafGain(sum_gradient, category_total_hessian, lambda_l1,
                               lambda_l2, max_delta_step, 0.0, num_data, 0.0);
    const double parent_shift = parent_gain + min_gain_to_split;
    const double category_l2 = lambda_l2 + (use_onehot ? 0.0 : feature.cat_l2);
    double best_gain = -std::numeric_limits<double>::infinity();
    double best_left_gradient = 0.0, best_left_hessian = 0.0;
    int best_left_count = 0, best_threshold = -1, best_direction = 1;
    std::vector<int> sorted_categories;
    auto category_count = [&](int bin) {
      return static_cast<int>(std::floor(hessians[bin] * count_factor + 0.5));
    };
    auto consider = [&](double left_gradient, double left_hessian, int left_count,
                        int threshold, int direction) {
      const int right_count = num_data - left_count;
      const double right_gradient = sum_gradient - left_gradient;
      const double right_hessian = category_total_hessian - left_hessian;
      if (left_count < min_data_in_leaf || right_count < min_data_in_leaf ||
          left_hessian < min_sum_hessian_in_leaf ||
          right_hessian < min_sum_hessian_in_leaf) {
        return;
      }
      const double current_gain =
          CategoricalLeafGain(left_gradient, left_hessian, lambda_l1, category_l2,
                               max_delta_step, path_smooth, left_count, parent_output) +
          CategoricalLeafGain(right_gradient, right_hessian, lambda_l1, category_l2,
                               max_delta_step, path_smooth, right_count, parent_output);
      if (current_gain <= parent_shift) return;
      categorical_result.splittable[feature_index] = 1;
      const double gain = (current_gain - parent_shift) * feature.penalty;
      if (gain > best_gain) {
        best_gain = gain;
        best_left_gradient = left_gradient;
        best_left_hessian = left_hessian;
        best_left_count = left_count;
        best_threshold = threshold;
        best_direction = direction;
      }
    };
    if (use_onehot) {
      int random_threshold = 0;
      if (feature.extra_tree_random != nullptr && bin_end > bin_start) {
        random_threshold = feature.extra_tree_random->NextInt(bin_start, bin_end);
      }
      for (int bin = bin_start; bin < bin_end; ++bin) {
        if (feature.extra_tree_random != nullptr && bin != random_threshold) continue;
        const double gradient = gradients[bin];
        const double hessian = hessians[bin];
        const int count = category_count(bin);
        if (count < min_data_in_leaf || hessian < min_sum_hessian_in_leaf) continue;
        if (num_data - count < min_data_in_leaf ||
            category_total_hessian - hessian - kEpsilon < min_sum_hessian_in_leaf) continue;
        consider(gradient, hessian + kEpsilon, count, bin, 1);
      }
    } else {
      for (int bin = bin_start; bin < bin_end; ++bin) {
        if (category_count(bin) >= feature.cat_smooth) sorted_categories.push_back(bin);
      }
      std::stable_sort(sorted_categories.begin(), sorted_categories.end(),
                       [&](int left, int right) {
        const double left_ctr = gradients[left] /
                                (hessians[left] + feature.cat_smooth);
        const double right_ctr = gradients[right] /
                                 (hessians[right] + feature.cat_smooth);
        return left_ctr < right_ctr;
      });
      const int used_bin = static_cast<int>(sorted_categories.size());
      const int max_num_cat = std::min(feature.max_cat_threshold, (used_bin + 1) / 2);
      const int max_threshold = std::max(std::min(max_num_cat, used_bin) - 1, 0);
      int random_threshold = 0;
      // Draw only after filtering: rare categories must not affect the range.
      // The same sampled prefix index is tested in both sorted directions.
      if (feature.extra_tree_random != nullptr && max_threshold > 0) {
        random_threshold = feature.extra_tree_random->NextInt(0, max_threshold);
      }
      for (int direction : {1, -1}) {
        double left_gradient = 0.0;
        double left_hessian = kEpsilon;
        int left_count = 0;
        int current_group_count = 0;
        for (int i = 0; i < static_cast<int>(sorted_categories.size()) && i < max_num_cat; ++i) {
          const int position = direction == 1 ? i : static_cast<int>(sorted_categories.size()) - 1 - i;
          const int bin = sorted_categories[position];
          const int count = category_count(bin);
          left_gradient += gradients[bin];
          left_hessian += hessians[bin];
          left_count += count;
          current_group_count += count;
          if (left_count < min_data_in_leaf || left_hessian < min_sum_hessian_in_leaf) continue;
          if (num_data - left_count < min_data_in_leaf ||
              num_data - left_count < feature.min_data_per_group ||
              category_total_hessian - left_hessian < min_sum_hessian_in_leaf) break;
          if (current_group_count < feature.min_data_per_group) continue;
          current_group_count = 0;
          if (feature.extra_tree_random != nullptr && i != random_threshold) continue;
          consider(left_gradient, left_hessian, left_count, i, direction);
        }
      }
    }
    if (feature.enabled && categorical_result.splittable[feature_index] &&
        (categorical_result.feature < 0 || best_gain > categorical_result.gain ||
         (best_gain == categorical_result.gain && feature.feature < categorical_result.feature))) {
      categorical_result.feature = feature.feature;
      categorical_result.categorical = true;
      categorical_result.gain = best_gain;
      categorical_result.default_left = false;
      categorical_result.left_gradient = best_left_gradient;
      categorical_result.left_hessian = best_left_hessian - kEpsilon;
      categorical_result.right_gradient = sum_gradient - best_left_gradient;
      categorical_result.right_hessian = category_total_hessian - best_left_hessian - kEpsilon;
      categorical_result.left_count = best_left_count;
      categorical_result.right_count = num_data - best_left_count;
      categorical_result.left_output = CategoricalLeafOutput(
          best_left_gradient, best_left_hessian, lambda_l1, category_l2,
          max_delta_step, path_smooth, best_left_count, parent_output);
      categorical_result.right_output = CategoricalLeafOutput(
          categorical_result.right_gradient, category_total_hessian - best_left_hessian,
          lambda_l1, category_l2, max_delta_step, path_smooth,
          num_data - best_left_count, parent_output);
      if (use_onehot) {
        categorical_result.cat_threshold = {
            static_cast<uint32_t>(best_threshold + feature.offset)};
      } else {
        const int count = best_threshold + 1;
        categorical_result.cat_threshold.resize(count);
        if (best_direction == 1) {
          for (int i = 0; i < count; ++i) {
            categorical_result.cat_threshold[i] =
                static_cast<uint32_t>(sorted_categories[i] + feature.offset);
          }
        } else {
          for (int i = 0; i < count; ++i) {
            categorical_result.cat_threshold[i] = static_cast<uint32_t>(
                sorted_categories[static_cast<int>(sorted_categories.size()) - 1 - i] + feature.offset);
          }
        }
      }
    }
  }

  // The order of the packed rows is the histogram scanner's order: reverse
  // first, then forward (where missing values require a second direction).
  struct Row { size_t feature; int begin, length; bool reverse; };
  struct Residual { int candidate, begin, length; };
  std::vector<Row> rows;
  std::vector<Residual> residuals;
  std::vector<double> gradients, hessians, penalties, enabled_flags;
  std::vector<double> random_candidate_flags;
  bool has_numeric_random = false;
  std::vector<int> feature_indices, thresholds;
  std::vector<int64_t> histogram_indices;
  std::vector<uint8_t> default_left;
  // Device ArgMax keeps the first equal-gain candidate. Pack features in
  // original feature-index order to match SplitInfo's cross-feature tie rule.
  std::vector<size_t> feature_order(features.size());
  for (size_t i = 0; i < features.size(); ++i) feature_order[i] = i;
  std::stable_sort(feature_order.begin(), feature_order.end(),
                   [&](size_t a, size_t b) { return features[a].feature < features[b].feature; });
  for (size_t fi : feature_order) {
    const auto& f = features[fi];
    if (f.categorical) continue;
    if (f.offset != 0 && f.offset != 1) {
      Log::Fatal("NPU split supports only histogram offsets 0 and 1");
    }
    if (f.missing_type < 0 || f.missing_type > 2 ||
        f.default_bin < 0 || f.default_bin >= f.num_bin) {
      Log::Fatal("NPU split received unsupported missing-value metadata");
    }
    if (!std::isfinite(f.penalty) || f.penalty < 0.0 ||
        !std::isfinite(sum_gradient) || !std::isfinite(sum_hessian) ||
        sum_hessian <= 0.0 || !std::isfinite(lambda_l1) || lambda_l1 < 0.0 ||
        !std::isfinite(lambda_l2) || lambda_l2 < 0.0 ||
        !std::isfinite(min_sum_hessian_in_leaf) ||
        !std::isfinite(min_gain_to_split)) {
      Log::Fatal("NPU split received invalid parameters or feature penalty");
    }
    if (f.num_bin <= 1) continue;
    int random_threshold = 0;
    if (f.extra_tree_random != nullptr) {
      has_numeric_random = true;
      if (f.num_bin - 2 > 0) {
        random_threshold = f.extra_tree_random->NextInt(0, f.num_bin - 2);
      }
    }
    if (histogram) {
      CHECK(histogram->fixed);
      CHECK_GE(f.histogram_index, 0);
      CHECK_LT(f.histogram_index, histogram->features);
      CHECK(histogram->used[f.histogram_index]);
      CHECK_EQ(f.num_bin, impl_->histogram_bin_counts[f.histogram_index]);
      CHECK_EQ(f.offset, impl_->histogram_offsets[f.histogram_index]);
    } else {
      if (f.gradients.size() != static_cast<size_t>(f.num_bin - f.offset) ||
          f.hessians.size() != f.gradients.size()) {
        Log::Fatal("NPU split histogram size differs from stored bin count");
      }
      for (size_t bin = 0; bin < f.gradients.size(); ++bin) {
        if (!std::isfinite(f.gradients[bin]) || !std::isfinite(f.hessians[bin]) ||
            f.hessians[bin] < 0.0) {
          Log::Fatal("NPU split requires finite gradients and nonnegative hessians");
        }
      }
    }
    const bool double_scan = f.num_bin > 2 && f.missing_type != 0;
    for (int direction = 0; direction < (double_scan ? 2 : 1); ++direction) {
      const bool reverse = direction == 0;
      const int begin = static_cast<int>(gradients.size());
      // With NaN and an omitted bin 0, the forward scan starts at t=-1:
      // its left side consists only of the implicit missing-bin residual.
      // Reserve that candidate; calculate its sums/count on the device below.
      if (!reverse && f.missing_type == 2 && f.offset == 1) {
        residuals.push_back({static_cast<int>(gradients.size()),
                             static_cast<int>(fi), f.num_bin - f.offset});
        histogram_indices.push_back(histogram ? static_cast<int64_t>(f.histogram_index) * histogram->stride : 0);
        gradients.push_back(0.0);
        hessians.push_back(0.0);
        penalties.push_back(f.penalty);
        enabled_flags.push_back(f.enabled ? 1.0 : 0.0);
        feature_indices.push_back(static_cast<int>(fi));
        thresholds.push_back(0);
        random_candidate_flags.push_back(
            f.extra_tree_random == nullptr || random_threshold == 0 ? 1.0 : 0.0);
        default_left.push_back(false);
      }
      const int first = reverse ? f.num_bin - 1 - f.offset - (double_scan && f.missing_type == 2) : 0;
      const int last = reverse ? 1 - f.offset : f.num_bin - 2 - f.offset;
      for (int t = first; reverse ? t >= last : t <= last; t += reverse ? -1 : 1) {
        if (double_scan && f.missing_type == 1 && t + f.offset == f.default_bin) continue;
        histogram_indices.push_back(histogram ? static_cast<int64_t>(f.histogram_index) * histogram->stride + t : 0);
        gradients.push_back(histogram ? 0.0 : f.gradients[t]);
        hessians.push_back(histogram ? 0.0 : f.hessians[t]);
        penalties.push_back(f.penalty);
        enabled_flags.push_back(f.enabled ? 1.0 : 0.0);
        feature_indices.push_back(static_cast<int>(fi));
        const int threshold = reverse ? t - 1 + f.offset : t + f.offset;
        thresholds.push_back(threshold);
        random_candidate_flags.push_back(
            f.extra_tree_random == nullptr || threshold == random_threshold ? 1.0 : 0.0);
        default_left.push_back(reverse && !(f.missing_type == 2 && !double_scan));
      }
      if (static_cast<int>(gradients.size()) != begin)
        rows.push_back({fi, begin, static_cast<int>(gradients.size()) - begin, reverse});
    }
  }
  const int64_t n = static_cast<int64_t>(gradients.size());
  if (n == 0) return categorical_result;
  const size_t bytes = static_cast<size_t>(n) * sizeof(double);
  std::vector<int64_t> shape = {n};
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<std::unique_ptr<TensorHandle>> tensors;
  auto allocate = [&](aclDataType type) -> int {
    size_t size = static_cast<size_t>(n) * (type == ACL_BOOL ? 1 : sizeof(double));
    buffers.emplace_back(new DeviceBuffer(impl_->Zeros(size)));
    tensors.emplace_back(new TensorHandle(buffers.back()->data(), shape, type));
    return static_cast<int>(tensors.size()) - 1;
  };
  auto upload = [&](const std::vector<double>& values) -> int {
    buffers.emplace_back(new DeviceBuffer(impl_->Upload(values.data(), bytes)));
    tensors.emplace_back(new TensorHandle(buffers.back()->data(), shape, ACL_DOUBLE));
    return static_cast<int>(tensors.size()) - 1;
  };
#define NPU_UNARY(OP, X) ([&]() { int o = allocate(ACL_DOUBLE); impl_->Launch(\
    [&](uint64_t* s, aclOpExecutor** e) { return aclnn##OP##GetWorkspaceSize(tensors[X]->get(), tensors[o]->get(), s, e); }, aclnn##OP); return o; }())
#define NPU_BINARY(OP, X, Y) ([&]() { int o = allocate(ACL_DOUBLE); impl_->Launch(\
    [&](uint64_t* s, aclOpExecutor** e) { return aclnn##OP##GetWorkspaceSize(tensors[X]->get(), tensors[Y]->get(), tensors[o]->get(), s, e); }, aclnn##OP); return o; }())
  auto add = [&](int x, int y) -> int {
    int o = allocate(ACL_DOUBLE); ScalarHandle one(1.0, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnAddGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), one.get(), tensors[o]->get(), s, e);
    }, aclnnAdd); return o;
  };
  auto sub = [&](int x, int y) -> int {
    int o = allocate(ACL_DOUBLE); ScalarHandle one(1.0, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnSubGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), one.get(), tensors[o]->get(), s, e);
    }, aclnnSub); return o;
  };
  auto scalar = [&](int x, double value, bool multiply) -> int {
    int o = allocate(ACL_DOUBLE); ScalarHandle v(value, ACL_DOUBLE), one(1.0, ACL_DOUBLE);
    if (multiply) impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnMulsGetWorkspaceSize(tensors[x]->get(), v.get(), tensors[o]->get(), s, e);
    }, aclnnMuls);
    else impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnAddsGetWorkspaceSize(tensors[x]->get(), v.get(), one.get(), tensors[o]->get(), s, e);
    }, aclnnAdds);
    return o;
  };
  auto ge = [&](int x, double value, bool strict) -> int {
    int o = allocate(ACL_BOOL); ScalarHandle v(value, ACL_DOUBLE);
    if (strict) impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnGtScalarGetWorkspaceSize(tensors[x]->get(), v.get(), tensors[o]->get(), s, e);
    }, aclnnGtScalar);
    else impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnGeScalarGetWorkspaceSize(tensors[x]->get(), v.get(), tensors[o]->get(), s, e);
    }, aclnnGeScalar);
    return o;
  };
  auto logical_and = [&](int x, int y) -> int {
    int o = allocate(ACL_BOOL); impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnLogicalAndGetWorkspaceSize(tensors[x]->get(), tensors[y]->get(), tensors[o]->get(), s, e);
    }, aclnnLogicalAnd); return o;
  };
  // Per-bin RoundInt, not RoundInt of a cumulative hessian. Cumsum is
  // independently launched for each scan row to preserve feature boundaries.
  int g, h;
  if (histogram) {
    g = allocate(ACL_DOUBLE);
    h = allocate(ACL_DOUBLE);
    DeviceBuffer index_buffer = impl_->Upload(histogram_indices.data(), n * sizeof(int64_t));
    TensorHandle indices(index_buffer.data(), {n}, ACL_INT64);
    const int64_t plane = static_cast<int64_t>(histogram->features) * histogram->stride;
    for (int part = 0; part < 2; ++part) {
      TensorHandle source(static_cast<double*>(histogram->values.data()) + part * plane,
                           {plane}, ACL_DOUBLE);
      const int target = part == 0 ? g : h;
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnIndexSelectGetWorkspaceSize(source.get(), 0, indices.get(), tensors[target]->get(), s, e);
      }, aclnnIndexSelect);
    }
  } else {
    g = upload(gradients);
    h = upload(hessians);
  }
  // Common::RoundInt(x) truncates x+0.5 toward zero; hessians are nonnegative.
  // Replace floor(h*factor) by floor(h*factor+0.5).
  // FeatureHistogram::FindBestThreshold adds 2*kEpsilon to the leaf Hessian
  // before its scanner computes per-bin counts and parent/child gains.
  const double scan_total_hessian = sum_hessian + 2.0 * kEpsilon;
  int cnt = NPU_UNARY(Floor, scalar(scalar(h, static_cast<double>(num_data) /
                                    scan_total_hessian, true), 0.5, false));
  // The omitted bin 0 for NaN/offset=1 is a residual, not a stored bin.
  // Subtract the device-computed sums of *all* stored bins and their
  // individually rounded counts before scanning the forward candidates.
  for (const auto& residual : residuals) {
    const auto& feature = features[residual.begin];
    const int64_t len = residual.length;
    const size_t len_bytes = static_cast<size_t>(len) * sizeof(double);
    const std::vector<int64_t> full_shape = {len};
    auto fill = [&](int packed, const std::vector<double>& values, double total,
                    bool round_each_bin, int histogram_part) {
      DeviceBuffer source;
      void* source_pointer;
      if (histogram) {
        const int64_t plane = static_cast<int64_t>(histogram->features) * histogram->stride;
        source_pointer = static_cast<double*>(histogram->values.data()) +
            histogram_part * plane + feature.histogram_index * histogram->stride;
      } else {
        source = impl_->Upload(values.data(), len_bytes);
        source_pointer = source.data();
      }
      DeviceBuffer scaled = impl_->Zeros(len_bytes);
      DeviceBuffer rounded = impl_->Zeros(len_bytes);
      DeviceBuffer summed = impl_->Zeros(len_bytes);
      DeviceBuffer parent = impl_->Upload(&total, sizeof(double));
      DeviceBuffer remainder = impl_->Zeros(sizeof(double));
      TensorHandle t_source(source_pointer, full_shape, ACL_DOUBLE);
      TensorHandle t_scaled(scaled.data(), full_shape, ACL_DOUBLE);
      TensorHandle t_rounded(rounded.data(), full_shape, ACL_DOUBLE);
      TensorHandle t_summed(summed.data(), full_shape, ACL_DOUBLE);
      TensorHandle t_parent(parent.data(), {1}, ACL_DOUBLE);
      TensorHandle t_last(static_cast<char*>(summed.data()) + (len - 1) * sizeof(double),
                          {1}, ACL_DOUBLE);
      TensorHandle t_remainder(remainder.data(), {1}, ACL_DOUBLE);
      aclTensor* scan_input = t_source.get();
      if (round_each_bin) {
        ScalarHandle factor(static_cast<double>(num_data) / scan_total_hessian, ACL_DOUBLE);
        impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
          return aclnnMulsGetWorkspaceSize(t_source.get(), factor.get(),
                                           t_scaled.get(), s, e);
        }, aclnnMuls);
        ScalarHandle half(0.5, ACL_DOUBLE), one(1.0, ACL_DOUBLE);
        impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
          return aclnnAddsGetWorkspaceSize(t_scaled.get(), half.get(), one.get(),
                                           t_rounded.get(), s, e);
        }, aclnnAdds);
        impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
          return aclnnFloorGetWorkspaceSize(t_rounded.get(), t_scaled.get(), s, e);
        }, aclnnFloor);
        scan_input = t_scaled.get();
      }
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnCumsumGetWorkspaceSize(scan_input, 0, ACL_DOUBLE,
                                           t_summed.get(), s, e);
      }, aclnnCumsum);
      ScalarHandle one(1.0, ACL_DOUBLE);
      impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
        return aclnnSubGetWorkspaceSize(t_parent.get(), t_last.get(), one.get(),
                                        t_remainder.get(), s, e);
      }, aclnnSub);
      ACL_CHECK(aclrtMemcpy(static_cast<char*>(buffers[packed]->data()) +
                            residual.candidate * sizeof(double), sizeof(double),
                            remainder.data(), sizeof(double), ACL_MEMCPY_DEVICE_TO_DEVICE));
    };
    fill(g, feature.gradients, sum_gradient, false, 0);
    fill(h, feature.hessians, sum_hessian, false, 1);
    fill(cnt, feature.hessians, static_cast<double>(num_data), true, 1);
  }
  int cg = allocate(ACL_DOUBLE), ch = allocate(ACL_DOUBLE), cc = allocate(ACL_DOUBLE);
  // Batch all feature/direction prefix scans. Repeating the last input in
  // padding cannot affect any real prefix; the padded outputs are discarded.
  int64_t width = 0;
  for (const auto& row : rows) width = std::max(width, static_cast<int64_t>(row.length));
  const int64_t scan_rows = static_cast<int64_t>(rows.size());
  const int64_t padded_count = scan_rows * width;
  std::vector<int64_t> pad_indices(static_cast<size_t>(padded_count));
  std::vector<int64_t> unpad_indices(static_cast<size_t>(n));
  for (int64_t r = 0; r < scan_rows; ++r) {
    const auto& row = rows[r];
    for (int64_t c = 0; c < width; ++c)
      pad_indices[r * width + c] = row.begin + std::min(c, static_cast<int64_t>(row.length - 1));
    for (int c = 0; c < row.length; ++c) unpad_indices[row.begin + c] = r * width + c;
  }
  DeviceBuffer pad_index_buffer = impl_->Upload(pad_indices.data(), padded_count * sizeof(int64_t));
  DeviceBuffer unpad_index_buffer = impl_->Upload(unpad_indices.data(), n * sizeof(int64_t));
  TensorHandle pad_index(pad_index_buffer.data(), {padded_count}, ACL_INT64);
  TensorHandle unpad_index(unpad_index_buffer.data(), {n}, ACL_INT64);
  DeviceBuffer scan_input(padded_count * sizeof(double));
  DeviceBuffer scan_output(padded_count * sizeof(double));
  TensorHandle scan_flat(scan_input.data(), {padded_count}, ACL_DOUBLE);
  TensorHandle scan_matrix(scan_input.data(), {scan_rows, width}, ACL_DOUBLE);
  TensorHandle prefix_matrix(scan_output.data(), {scan_rows, width}, ACL_DOUBLE);
  TensorHandle prefix_flat(scan_output.data(), {padded_count}, ACL_DOUBLE);
  auto scan = [&](int input, int output) {
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(tensors[input]->get(), 0, pad_index.get(), scan_flat.get(), s, e);
    }, aclnnIndexSelect);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnCumsumGetWorkspaceSize(scan_matrix.get(), 1, ACL_DOUBLE, prefix_matrix.get(), s, e);
    }, aclnnCumsum);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(prefix_flat.get(), 0, unpad_index.get(), tensors[output]->get(), s, e);
    }, aclnnIndexSelect);
  };
  // CPU scanners initialize the accumulator with epsilon before adding
  // bins. Adding epsilon after Cumsum changes FP64 rounding and can reverse
  // the missing direction when two scans have almost equal gains.
  std::vector<double> scan_epsilon(static_cast<size_t>(n), 0.0);
  for (const auto& row : rows) scan_epsilon[row.begin] = kEpsilon;
  int hessian_with_epsilon = add(h, upload(scan_epsilon));
  scan(g, cg); scan(hessian_with_epsilon, ch); scan(cnt, cc);
  // Reverse scans accumulate the right side, forward scans the left side.
  std::vector<double> reverse(n), total_g(n, sum_gradient), total_h(n, scan_total_hessian);
  std::vector<double> total_count(n, num_data);
  for (const auto& row : rows)
    std::fill(reverse.begin() + row.begin, reverse.begin() + row.begin + row.length,
              row.reverse ? 1.0 : 0.0);
  int mask_reverse = ge(upload(reverse), 0.5, false);
  auto as_double = [&](int boolean) -> int {
    int output = allocate(ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnCastGetWorkspaceSize(tensors[boolean]->get(), ACL_DOUBLE,
                                       tensors[output]->get(), s, e);
    }, aclnnCast);
    return output;
  };
  int reverse_double = as_double(mask_reverse);
  int forward_double = sub(upload(std::vector<double>(static_cast<size_t>(n), 1.0)),
                           reverse_double);
  auto choose = [&](int forward, int backward) -> int {
    int positive = NPU_BINARY(Mul, forward, forward_double);
    return add(positive, NPU_BINARY(Mul, backward, reverse_double));
  };
  int sum_g = upload(total_g), sum_h = upload(total_h), data_count = upload(total_count);
  int left_g = choose(cg, sub(sum_g,cg));
  int right_g = choose(sub(sum_g,cg), cg);
  int accumulated_h = ch;
  int left_h = choose(accumulated_h, sub(sum_h, accumulated_h));
  int right_h = choose(sub(sum_h, accumulated_h), accumulated_h);
  int left_count = choose(cc, sub(data_count,cc));
  int right_count = sub(data_count,left_count);
  int valid = logical_and(ge(left_count,min_data_in_leaf,false), ge(right_count,min_data_in_leaf,false));
  valid = logical_and(valid, ge(left_h,min_sum_hessian_in_leaf,false));
  valid = logical_and(valid, ge(right_h,min_sum_hessian_in_leaf,false));
  auto soft = [&](int grad) -> int {
    if (lambda_l1 <= 0) return grad;
    int abs_grad = NPU_UNARY(Abs,grad);
    int clipped = allocate(ACL_DOUBLE), sign = NPU_UNARY(Sign, grad);
    ScalarHandle zero(0.0, ACL_DOUBLE);
    int shrunk = scalar(abs_grad,-lambda_l1,false);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnClampMinGetWorkspaceSize(tensors[shrunk]->get(),zero.get(),tensors[clipped]->get(),s,e);
    }, aclnnClampMin);
    return NPU_BINARY(Mul,clipped,sign);
  };
  auto clamp_min = [&](int input, double minimum) -> int {
    int output = allocate(ACL_DOUBLE);
    ScalarHandle lower(minimum, ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnClampMinGetWorkspaceSize(tensors[input]->get(), lower.get(),
                                           tensors[output]->get(), s, e);
    }, aclnnClampMin);
    return output;
  };
  const int smoothing = path_smooth > 0.0
      ? upload(std::vector<double>(static_cast<size_t>(n), path_smooth)) : -1;
  auto output = [&](int grad, int hess, int count) -> int {
    int v = soft(grad);
    int denominator = scalar(hess, lambda_l2, false);
    int quotient = NPU_BINARY(Div, v, denominator);
    int value = NPU_UNARY(Neg, quotient);
    if (max_delta_step > 0.0) {
      // ClampMax is not available for FP64 on all supported devices.
      // min(max(value, -limit), limit) = -max(-max(value, -limit), -limit).
      int lower = clamp_min(value, -max_delta_step);
      int negative = NPU_UNARY(Neg, lower);
      int upper = clamp_min(negative, -max_delta_step);
      value = NPU_UNARY(Neg, upper);
    }
    if (path_smooth > 0.0) {
      // Invalid scan candidates can have negative residual counts after
      // per-bin rounding. Keep their smoothing denominator finite; their
      // original counts still fail the validity mask above.
      int nonnegative_count = clamp_min(count, 0.0);
      int ratio = NPU_BINARY(Div, nonnegative_count, smoothing);
      int divisor = scalar(ratio, 1.0, false);
      int weighted = NPU_BINARY(Mul, value, ratio);
      int child_term = NPU_BINARY(Div, weighted, divisor);
      int parent = upload(std::vector<double>(static_cast<size_t>(n), parent_output));
      int parent_term = NPU_BINARY(Div, parent, divisor);
      value = add(child_term, parent_term);
    }
    return value;
  };
  auto gain_given_output = [&](int grad, int hess, int value) -> int {
    int v = soft(grad);
    int twice_gradient = scalar(v, 2.0, true);
    int linear = NPU_BINARY(Mul, twice_gradient, value);
    int denominator = scalar(hess, lambda_l2, false);
    int quadratic_once = NPU_BINARY(Mul, denominator, value);
    int quadratic = NPU_BINARY(Mul, quadratic_once, value);
    int total = add(linear, quadratic);
    return NPU_UNARY(Neg, total);
  };
  auto leaf = [&](int grad, int hess, int value) -> int {
    if (max_delta_step > 0.0 || path_smooth > 0.0) {
      return gain_given_output(grad, hess, value);
    }
    // Preserve the original exact optimum formula with both rules disabled.
    int v = soft(grad);
    int squared = NPU_BINARY(Mul, v, v);
    int denominator = scalar(hess, lambda_l2, false);
    return NPU_BINARY(Div, squared, denominator);
  };
  // CPU skips candidates that fail count/Hessian bounds before evaluating
  // outputs. Give those entries G=0,H=1 so vectorized division cannot create
  // NaN (e.g. 0/0 in an empty residual bin) that survives a multiply-zero mask.
  int bounded = as_double(valid);
  int unbounded = scalar(scalar(bounded, -1.0, true), 1.0, false);
  int safe_left_g = NPU_BINARY(Mul, left_g, bounded);
  int safe_right_g = NPU_BINARY(Mul, right_g, bounded);
  int safe_left_h = add(NPU_BINARY(Mul, left_h, bounded), unbounded);
  int safe_right_h = add(NPU_BINARY(Mul, right_h, bounded), unbounded);
  int left_out = output(safe_left_g, safe_left_h, left_count);
  int right_out = output(safe_right_g, safe_right_h, right_count);
  int left_gain = leaf(safe_left_g, safe_left_h, left_out);
  int right_gain = leaf(safe_right_g, safe_right_h, right_out);
  int current = add(left_gain, right_gain);
  // Serialization follows CPU's total-minus-left round trip; gain selection
  // above still uses the scan's directly accumulated right statistics.
  int stored_right_g = sub(sum_g, left_g);
  int stored_right_h = sub(sum_h, left_h);
  int stored_right_out = output(NPU_BINARY(Mul, stored_right_g, bounded),
      add(NPU_BINARY(Mul, stored_right_h, bounded), unbounded), right_count);
  // BeforeNumerical uses GetLeafGain (including smoothing of the unsplit
  // output), unlike categorical's GetLeafGainGivenOutput(parent_output).
  int parent_out = output(sum_g, sum_h, data_count);
  int parent_gain = leaf(sum_g, sum_h, parent_out);
  int shift = scalar(parent_gain, min_gain_to_split, false);
  valid = logical_and(valid, ge(sub(current,shift),0.0,true));
  // Keep every histogram/count prefix intact, then restrict BOTH splittability
  // and winner selection to the sampled logical threshold (in either scan).
  if (has_numeric_random) {
    valid = logical_and(valid, ge(upload(random_candidate_flags), 0.5, false));
  }
  int score = NPU_BINARY(Mul,sub(current,shift),upload(penalties));
  // CPU first selects the maximum *unshifted* gain in each scan direction.
  // Its second scan replaces the first only if
  //   forward_gain > (reverse_gain - shift) + shift.
  // The subtract/add round trip is observable in FP64 even for equal gains.
  // Reducing shifted gains across both scans instead changes default_left.
  int valid_double = as_double(valid);
  int invalid_score = scalar(scalar(valid_double, -1.0, false),
                             std::numeric_limits<double>::max(), true);
  int masked_current = add(NPU_BINARY(Mul, current, valid_double), invalid_score);
  DeviceBuffer direction_input(padded_count * sizeof(double));
  DeviceBuffer direction_winners(scan_rows * sizeof(int64_t));
  TensorHandle direction_flat(direction_input.data(), {padded_count}, ACL_DOUBLE);
  TensorHandle direction_matrix(direction_input.data(), {scan_rows, width}, ACL_DOUBLE);
  TensorHandle direction_indices(direction_winners.data(), {scan_rows}, ACL_INT64);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnIndexSelectGetWorkspaceSize(tensors[masked_current]->get(), 0,
                                            pad_index.get(), direction_flat.get(), s, e);
  }, aclnnIndexSelect);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnArgMaxGetWorkspaceSize(direction_matrix.get(), 1, false,
                                       direction_indices.get(), s, e);
  }, aclnnArgMax);
  // Convert per-row local winners into indices in the original packed array.
  std::vector<int64_t> row_begins, own_rows(n), reverse_rows(n);
  std::vector<int64_t> reverse_row_for_feature(features.size(), -1);
  std::vector<double> row_has_reverse(n, 0.0);
  for (int64_t r = 0; r < scan_rows; ++r) {
    row_begins.push_back(rows[r].begin);
    if (rows[r].reverse) reverse_row_for_feature[rows[r].feature] = r;
  }
  for (int64_t r = 0; r < scan_rows; ++r) {
    const auto& row = rows[r];
    const int64_t reverse_row = reverse_row_for_feature[row.feature];
    for (int i = 0; i < row.length; ++i) {
      const int position = row.begin + i;
      own_rows[position] = r;
      reverse_rows[position] = reverse_row >= 0 ? reverse_row : r;
      row_has_reverse[position] = reverse_row >= 0 ? 1.0 : 0.0;
    }
  }
  DeviceBuffer row_begin_buffer = impl_->Upload(row_begins.data(), scan_rows * sizeof(int64_t));
  DeviceBuffer packed_winners(scan_rows * sizeof(int64_t));
  TensorHandle row_begin_tensor(row_begin_buffer.data(), {scan_rows}, ACL_INT64);
  TensorHandle packed_indices(packed_winners.data(), {scan_rows}, ACL_INT64);
  IntScalarHandle one(1);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnAddGetWorkspaceSize(direction_indices.get(), row_begin_tensor.get(),
                                    one.get(), packed_indices.get(), s, e);
  }, aclnnAdd);
  DeviceBuffer own_row_buffer = impl_->Upload(own_rows.data(), n * sizeof(int64_t));
  DeviceBuffer reverse_row_buffer = impl_->Upload(reverse_rows.data(), n * sizeof(int64_t));
  TensorHandle own_row_indices(own_row_buffer.data(), {n}, ACL_INT64);
  TensorHandle reverse_row_indices(reverse_row_buffer.data(), {n}, ACL_INT64);
  auto row_winner_values = [&](int source, const TensorHandle& row_indices) -> int {
    DeviceBuffer values(scan_rows * sizeof(double));
    TensorHandle row_values(values.data(), {scan_rows}, ACL_DOUBLE);
    int output = allocate(ACL_DOUBLE);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(tensors[source]->get(), 0,
                                              packed_indices.get(), row_values.get(), s, e);
    }, aclnnIndexSelect);
    impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
      return aclnnIndexSelectGetWorkspaceSize(row_values.get(), 0,
                                              row_indices.get(), tensors[output]->get(), s, e);
    }, aclnnIndexSelect);
    return output;
  };
  int reverse_best = row_winner_values(current, reverse_row_indices);
  int reverse_valid = row_winner_values(valid_double, reverse_row_indices);
  reverse_valid = NPU_BINARY(Mul, reverse_valid, upload(row_has_reverse));
  int restored_reverse = add(sub(reverse_best, shift), shift);
  // Gather the paired forward winner so both rows make the same decision.
  std::vector<int64_t> forward_rows(n);
  std::vector<int64_t> forward_row_for_feature(features.size(), -1);
  for (int64_t r = 0; r < scan_rows; ++r)
    if (!rows[r].reverse) forward_row_for_feature[rows[r].feature] = r;
  std::vector<double> has_forward(n, 0.0);
  for (int64_t r = 0; r < scan_rows; ++r) {
    const auto& row = rows[r];
    const int64_t forward = forward_row_for_feature[row.feature];
    for (int i = 0; i < row.length; ++i) {
      forward_rows[row.begin + i] = forward >= 0 ? forward : r;
      has_forward[row.begin + i] = forward >= 0 ? 1.0 : 0.0;
    }
  }
  DeviceBuffer forward_row_buffer = impl_->Upload(forward_rows.data(), n * sizeof(int64_t));
  TensorHandle forward_row_indices(forward_row_buffer.data(), {n}, ACL_INT64);
  int forward_best = row_winner_values(current, forward_row_indices);
  int forward_valid = row_winner_values(valid_double, forward_row_indices);
  int forward_better = ge(sub(forward_best, restored_reverse), 0.0, true);
  int forward_accepted = ge(add(as_double(forward_better),
                                scalar(reverse_valid, -1.0, true)), -0.5, true);
  int replace_reverse = logical_and(ge(upload(has_forward), 0.5, false),
      logical_and(ge(forward_valid, 0.5, false), forward_accepted));
  int replace_double = as_double(replace_reverse);
  int chosen_direction = choose(replace_double,
      scalar(scalar(replace_double, -1.0, true), 1.0, false));
  std::vector<double> packed_positions(n);
  std::iota(packed_positions.begin(), packed_positions.end(), 0.0);
  int own_winner_position = row_winner_values(upload(packed_positions), own_row_indices);
  int distance = NPU_UNARY(Abs, sub(upload(packed_positions), own_winner_position));
  int is_row_winner = ge(scalar(distance, -1.0, true), 0.0, false);
  // Splittability is independent of a node's feature-selection mask.
  int selectable = logical_and(valid, logical_and(is_row_winner,
      logical_and(ge(chosen_direction, 0.5, false), ge(upload(enabled_flags), 0.5, false))));
  // Every valid score is nonnegative (gain exceeds its shift and penalty is
  // nonnegative); force invalid candidates to -1 without unsupported FP64
  // masked-fill/where kernels. ArgMax then performs the entire reduction on NPU.
  int selected_double = as_double(selectable);
  int masked_score = add(NPU_BINARY(Mul, score, selected_double),
                         scalar(selected_double, -1.0, false));
  int64_t axis = 1;
  aclIntArray* dims = aclCreateIntArray(&axis, 1);
  if (!dims) Log::Fatal("NPU split aclCreateIntArray failed");
  DeviceBuffer padded_flags(padded_count * sizeof(uint8_t));
  DeviceBuffer reduced_flags(scan_rows * sizeof(uint8_t));
  TensorHandle flags_flat(padded_flags.data(), {padded_count}, ACL_BOOL);
  TensorHandle flags_matrix(padded_flags.data(), {scan_rows, width}, ACL_BOOL);
  TensorHandle flags_reduced(reduced_flags.data(), {scan_rows}, ACL_BOOL);
  // Padding repeats a flag from the same row, so its Any reduction is exact.
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnIndexSelectGetWorkspaceSize(tensors[valid]->get(), 0, pad_index.get(), flags_flat.get(), s, e);
  }, aclnnIndexSelect);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnAnyGetWorkspaceSize(flags_matrix.get(), dims, false, flags_reduced.get(), s, e);
  }, aclnnAny);
  aclDestroyIntArray(dims);
  std::vector<uint8_t> row_flags(static_cast<size_t>(scan_rows));
  impl_->Download(row_flags.data(), reduced_flags, row_flags.size());
  for (int64_t r = 0; r < scan_rows; ++r)
    result.splittable[rows[r].feature] |= row_flags[r];
  DeviceBuffer winner_mem(sizeof(int64_t));
  TensorHandle winner(winner_mem.data(),{1},ACL_INT64);
  impl_->Launch([&](uint64_t* s, aclOpExecutor** e) {
    return aclnnArgMaxGetWorkspaceSize(tensors[masked_score]->get(),0,false,winner.get(),s,e);
  }, aclnnArgMax);
  int64_t index=0; impl_->Download(&index,winner_mem,sizeof(index)); ++argmax_calls_;
  if (index < 0 || index >= n) Log::Fatal("NPU split argmax returned invalid index");
  double gain=0;
  ACL_CHECK(aclrtMemcpy(&gain,sizeof(gain),static_cast<char*>(buffers[masked_score]->data())+index*sizeof(double),sizeof(double),ACL_MEMCPY_DEVICE_TO_HOST));
  if (gain < 0.0) {
    for (size_t i = 0; i < result.splittable.size(); ++i) {
      categorical_result.splittable[i] |= result.splittable[i];
    }
    return categorical_result;
  }
  if (!std::isfinite(gain)) Log::Fatal("NPU split produced a nonfinite winning gain");
  result.feature = features[feature_indices[index]].feature;
  result.threshold = static_cast<uint32_t>(thresholds[index]);
  result.default_left = default_left[index];
  result.gain = gain;
  auto one_value = [&](int id) -> double {
    double value;
    ACL_CHECK(aclrtMemcpy(&value,sizeof(value),static_cast<char*>(buffers[id]->data())+index*sizeof(double),sizeof(double),ACL_MEMCPY_DEVICE_TO_HOST));
    return value;
  };
  result.left_gradient=one_value(left_g); result.right_gradient=one_value(stored_right_g);
  result.left_hessian=one_value(left_h)-kEpsilon;
  result.right_hessian=one_value(stored_right_h)-kEpsilon;
  result.left_count=static_cast<int>(one_value(left_count));
  result.right_count=static_cast<int>(one_value(right_count));
  result.left_output=one_value(left_out); result.right_output=one_value(stored_right_out);
#undef NPU_UNARY
#undef NPU_BINARY
  for (size_t i = 0; i < result.splittable.size(); ++i) {
    result.splittable[i] = static_cast<uint8_t>(result.splittable[i] ||
                                                categorical_result.splittable[i]);
  }
  if (categorical_result.feature >= 0 &&
      (result.feature < 0 || categorical_result.gain > result.gain ||
       (categorical_result.gain == result.gain &&
        categorical_result.feature < result.feature))) {
    categorical_result.splittable = result.splittable;
    result = std::move(categorical_result);
  }
  return result;
}

}  // namespace LightGBM
