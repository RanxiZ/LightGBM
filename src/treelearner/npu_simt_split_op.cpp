/* Copyright (c) 2026. Licensed under the MIT License. */
//
// Host implementation of the Ascend SIMT best-split finder.
//
// Everything here is plain ACL: context + stream, one reusable device buffer per
// operand, one H2D per stage, two kernel launches, one synchronise, one D2H.
// Buffers are grown, never reallocated per call, so a training loop that keeps
// the same task count pays no allocation after the first leaf.
//
// The launch headers are produced by `ascendc_library` from the kernel
// signatures; nothing here hard-codes a kernel symbol.

#include "npu_simt_split_op.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "aclrtlaunch_lgbm_npu_simt_split_kernel.h"
#include "aclrtlaunch_lgbm_npu_simt_split_reduce_kernel.h"

namespace lgbm_npu_simt {

namespace {

// Tasks handled by one scan block (8 warps of 32 lanes, see the kernel).
constexpr int32_t kTasksPerBlock = 8;
// Fallback AIV count when the runtime refuses to report one.  Deliberately
// conservative: with the two-launch structure the grid size only affects
// throughput, never correctness.
constexpr int32_t kFallbackVectorCores = 24;

std::string AclMessage(const char* what, aclError code) {
  const char* detail = aclGetRecentErrMsg();
  char buffer[512];
  std::snprintf(buffer, sizeof(buffer), "%s failed with %d (%s)", what,
                static_cast<int>(code),
                detail != nullptr ? detail : "no detail");
  return std::string(buffer);
}

bool IsOk(aclError code) { return code == ACL_SUCCESS; }

}  // namespace

SimtSplitOperator::SimtSplitOperator() = default;

SimtSplitOperator::~SimtSplitOperator() { Release(); }

int32_t SimtSplitOperator::Fail(int32_t status, const char* what) {
  last_error_ = what != nullptr ? what : "";
  return status;
}

int32_t SimtSplitOperator::Init(int32_t device_id) {
  if (initialized_) {
    return kSimtSplitOk;
  }
  device_id_ = device_id;

  // Tolerate a host application that already initialised ACL / the device.
  const aclError init_ret = aclInit(nullptr);
  if (!IsOk(init_ret)) {
    last_error_ = AclMessage("aclInit (tolerated)", init_ret);
  }
  const aclError device_ret = aclrtSetDevice(device_id);
  if (!IsOk(device_ret)) {
    return Fail(kSimtSplitAclError, AclMessage("aclrtSetDevice", device_ret).c_str());
  }

  aclError code = aclrtCreateContext(&context_, device_id);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtCreateContext", code).c_str());
  }
  code = aclrtSetCurrentContext(context_);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtSetCurrentContext", code).c_str());
  }
  code = aclrtCreateStream(&stream_);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtCreateStream", code).c_str());
  }

  owns_context_ = true;
  return AdoptDevice(device_id);
}

int32_t SimtSplitOperator::AdoptDevice(int32_t device_id) {
  int64_t cores = 0;
  if (IsOk(aclrtGetDeviceInfo(static_cast<uint32_t>(device_id),
                              ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores)) &&
      cores > 0) {
    vector_cores_ = static_cast<int32_t>(cores);
  } else {
    vector_cores_ = kFallbackVectorCores;
  }
  num_blocks_ = 0;
  initialized_ = true;
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::Attach(int32_t device_id, aclrtContext context,
                                  aclrtStream stream) {
  if (initialized_) {
    return kSimtSplitOk;
  }
  if (context == nullptr || stream == nullptr) {
    return Fail(kSimtSplitInvalidArg, "Attach() requires a context and a stream");
  }
  device_id_ = device_id;
  context_ = context;
  stream_ = stream;
  owns_context_ = false;
  const aclError code = aclrtSetCurrentContext(context_);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtSetCurrentContext", code).c_str());
  }
  return AdoptDevice(device_id);
}

void SimtSplitOperator::DropHistogramStorage() {
  // Bound planes belong to the caller and must never be freed here.
  if (!hist_is_bound_) {
    if (device_hist_gradients_ != nullptr) {
      aclrtFree(device_hist_gradients_);
    }
    if (device_hist_hessians_ != nullptr) {
      aclrtFree(device_hist_hessians_);
    }
  }
  device_hist_gradients_ = nullptr;
  device_hist_hessians_ = nullptr;
  hist_capacity_ = 0;
  hist_is_bound_ = false;
  hist_bound_bins_ = 0;
}

void SimtSplitOperator::Reset() {
  if (device_tasks_ != nullptr) {
    aclrtFree(device_tasks_);
    device_tasks_ = nullptr;
  }
  if (device_params_ != nullptr) {
    aclrtFree(device_params_);
    device_params_ = nullptr;
  }
  if (device_candidates_ != nullptr) {
    aclrtFree(device_candidates_);
    device_candidates_ = nullptr;
  }
  if (device_winner_ != nullptr) {
    aclrtFree(device_winner_);
    device_winner_ = nullptr;
  }
  DropHistogramStorage();
  tasks_capacity_ = 0;
  num_tasks_ = 0;
  tasks_staged_ = false;
  hist_staged_ = false;
  staged_hist_bins_ = 0;
}

int64_t SimtSplitOperator::device_bytes() const {
  int64_t total = 0;
  if (device_tasks_ != nullptr) {
    total += tasks_capacity_ * static_cast<int64_t>(sizeof(SimtSplitTask));
  }
  if (device_candidates_ != nullptr) {
    total += tasks_capacity_ * static_cast<int64_t>(sizeof(SimtSplitCandidate));
  }
  if (device_params_ != nullptr) {
    total += static_cast<int64_t>(sizeof(SimtSplitParams));
  }
  if (device_winner_ != nullptr) {
    total += static_cast<int64_t>(sizeof(SimtSplitWinner));
  }
  if (!hist_is_bound_) {
    if (device_hist_gradients_ != nullptr) {
      total += hist_capacity_ * static_cast<int64_t>(sizeof(float));
    }
    if (device_hist_hessians_ != nullptr) {
      total += hist_capacity_ * static_cast<int64_t>(sizeof(float));
    }
  }
  return total;
}

void SimtSplitOperator::Release() {
  Reset();
  if (owns_context_) {
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
    }
    if (context_ != nullptr) {
      aclrtSetCurrentContext(nullptr);
      aclrtDestroyContext(context_);
    }
    if (initialized_ && device_id_ >= 0) {
      aclrtResetDevice(device_id_);
    }
  }
  // Attached handles belong to the caller; dropping our references is all we
  // are allowed to do.
  stream_ = nullptr;
  context_ = nullptr;
  owns_context_ = false;
  initialized_ = false;
  vector_cores_ = 0;
  num_blocks_ = 0;
}

int32_t SimtSplitOperator::EnsureTaskBuffers(int32_t num_tasks) {
  const int64_t wanted = num_tasks;
  if (wanted > tasks_capacity_) {
    if (device_tasks_ != nullptr) {
      aclrtFree(device_tasks_);
      device_tasks_ = nullptr;
    }
    if (device_candidates_ != nullptr) {
      aclrtFree(device_candidates_);
      device_candidates_ = nullptr;
    }
    aclError code =
        aclrtMalloc(&device_tasks_, static_cast<size_t>(wanted * sizeof(SimtSplitTask)),
                    ACL_MEM_MALLOC_HUGE_FIRST);
    if (!IsOk(code)) {
      tasks_capacity_ = 0;
      return Fail(kSimtSplitAclError,
                  AclMessage("aclrtMalloc(tasks)", code).c_str());
    }
    code = aclrtMalloc(
        &device_candidates_,
        static_cast<size_t>(wanted * sizeof(SimtSplitCandidate)),
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (!IsOk(code)) {
      tasks_capacity_ = 0;
      return Fail(kSimtSplitAclError,
                  AclMessage("aclrtMalloc(candidates)", code).c_str());
    }
    tasks_capacity_ = wanted;
  }
  if (device_params_ == nullptr) {
    const aclError code =
        aclrtMalloc(&device_params_, sizeof(SimtSplitParams),
                    ACL_MEM_MALLOC_HUGE_FIRST);
    if (!IsOk(code)) {
      return Fail(kSimtSplitAclError,
                  AclMessage("aclrtMalloc(params)", code).c_str());
    }
  }
  if (device_winner_ == nullptr) {
    const aclError code = aclrtMalloc(&device_winner_, sizeof(SimtSplitWinner),
                                      ACL_MEM_MALLOC_HUGE_FIRST);
    if (!IsOk(code)) {
      return Fail(kSimtSplitAclError,
                  AclMessage("aclrtMalloc(winner)", code).c_str());
    }
  }
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::EnsureHistogramBuffers(int64_t num_bins) {
  if (hist_is_bound_) {
    // The previous run used caller-owned planes; drop the binding before
    // allocating upload buffers, never free the caller's memory.
    DropHistogramStorage();
  }
  if (num_bins <= hist_capacity_ && device_hist_gradients_ != nullptr) {
    return kSimtSplitOk;
  }
  DropHistogramStorage();
  void* gradients = nullptr;
  void* hessians = nullptr;
  aclError code = aclrtMalloc(&gradients,
                              static_cast<size_t>(num_bins * sizeof(float)),
                              ACL_MEM_MALLOC_HUGE_FIRST);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtMalloc(hist_grad)", code).c_str());
  }
  code = aclrtMalloc(&hessians, static_cast<size_t>(num_bins * sizeof(float)),
                     ACL_MEM_MALLOC_HUGE_FIRST);
  if (!IsOk(code)) {
    aclrtFree(gradients);
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtMalloc(hist_hess)", code).c_str());
  }
  device_hist_gradients_ = static_cast<float*>(gradients);
  device_hist_hessians_ = static_cast<float*>(hessians);
  hist_capacity_ = num_bins;
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::ValidateTasks(int32_t num_tasks) const {
  if (num_tasks <= 0) {
    return kSimtSplitInvalidArg;
  }
  if (staged_hist_bins_ <= 0) {
    return kSimtSplitInvalidArg;
  }
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::UploadTasks(const SimtSplitTask* tasks,
                                       int32_t num_tasks,
                                       SimtSplitParams params) {
  if (!initialized_) {
    return Fail(kSimtSplitNotInitialized, "Init() has not been called");
  }
  // The operator may share the process with a second ACL runtime (the v2
  // histogram runtime in the joint probe). Make sure the stream and buffers we
  // touch belong to our own context before any ACL call.
  aclrtSetCurrentContext(context_);
  if (tasks == nullptr || num_tasks <= 0) {
    return Fail(kSimtSplitInvalidArg, "null task list or non-positive count");
  }
  const int32_t status = EnsureTaskBuffers(num_tasks);
  if (status != kSimtSplitOk) {
    return status;
  }

  params.num_tasks = num_tasks;
  params.threads_per_block = 256;

  aclError code = aclrtMemcpyAsync(
      device_tasks_, static_cast<size_t>(num_tasks * sizeof(SimtSplitTask)),
      tasks, static_cast<size_t>(num_tasks * sizeof(SimtSplitTask)),
      ACL_MEMCPY_HOST_TO_DEVICE, stream_);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtMemcpyAsync(tasks)", code).c_str());
  }
  // The copy is asynchronous, so its source must outlive this call: stage the
  // struct in a member instead of copying from the stack frame.
  host_params_ = params;
  code = aclrtMemcpyAsync(device_params_, sizeof(SimtSplitParams), &host_params_,
                          sizeof(SimtSplitParams), ACL_MEMCPY_HOST_TO_DEVICE,
                          stream_);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtMemcpyAsync(params)", code).c_str());
  }
  num_tasks_ = num_tasks;
  tasks_staged_ = true;
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::UploadHistogram(
    const SimtSplitHistogram& host_histogram) {
  if (!initialized_) {
    return Fail(kSimtSplitNotInitialized, "Init() has not been called");
  }
  aclrtSetCurrentContext(context_);
  if (host_histogram.gradients == nullptr || host_histogram.hessians == nullptr ||
      host_histogram.num_bins <= 0) {
    return Fail(kSimtSplitInvalidArg, "empty histogram");
  }
  const int32_t status = EnsureHistogramBuffers(host_histogram.num_bins);
  if (status != kSimtSplitOk) {
    return status;
  }
  hist_is_bound_ = false;
  const size_t bytes = static_cast<size_t>(host_histogram.num_bins * sizeof(float));
  aclError code = aclrtMemcpyAsync(device_hist_gradients_, bytes,
                                   host_histogram.gradients, bytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream_);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtMemcpyAsync(hist_grad)", code).c_str());
  }
  code = aclrtMemcpyAsync(device_hist_hessians_, bytes, host_histogram.hessians,
                          bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream_);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtMemcpyAsync(hist_hess)", code).c_str());
  }
  staged_hist_bins_ = host_histogram.num_bins;
  hist_staged_ = true;
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::BindDeviceHistogram(const float* device_gradients,
                                               const float* device_hessians,
                                               int64_t num_bins) {
  if (!initialized_) {
    return Fail(kSimtSplitNotInitialized, "Init() has not been called");
  }
  aclrtSetCurrentContext(context_);
  if (device_gradients == nullptr || device_hessians == nullptr ||
      num_bins <= 0) {
    return Fail(kSimtSplitInvalidArg, "empty device histogram");
  }
  // Release any upload buffers first: the caller owns the planes we bind.
  DropHistogramStorage();
  hist_is_bound_ = true;
  hist_bound_bins_ = num_bins;
  device_hist_gradients_ = const_cast<float*>(device_gradients);
  device_hist_hessians_ = const_cast<float*>(device_hessians);
  staged_hist_bins_ = num_bins;
  hist_staged_ = true;
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::Launch() {
  const int32_t status = ValidateTasks(num_tasks_);
  if (status != kSimtSplitOk) {
    return Fail(kSimtSplitInvalidArg, "Run prepared with no staged input");
  }
  if (!hist_staged_) {
    return Fail(kSimtSplitInvalidArg, "histogram not staged");
  }
  const int32_t needed_blocks = (num_tasks_ + kTasksPerBlock - 1) / kTasksPerBlock;
  num_blocks_ = std::min(needed_blocks, std::max(1, vector_cores_));

  aclError code = static_cast<aclError>(aclrtlaunch_lgbm_npu_simt_split_kernel(
      static_cast<uint32_t>(num_blocks_), stream_, device_hist_gradients_,
      device_hist_hessians_, device_tasks_, device_params_,
      device_candidates_));
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtlaunch_lgbm_npu_simt_split_kernel", code)
                    .c_str());
  }
  code = static_cast<aclError>(aclrtlaunch_lgbm_npu_simt_split_reduce_kernel(
      1u, stream_, device_params_, device_candidates_, device_winner_));
  if (!IsOk(code)) {
    return Fail(
        kSimtSplitAclError,
        AclMessage("aclrtlaunch_lgbm_npu_simt_split_reduce_kernel", code)
            .c_str());
  }
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::RunPrepared(SimtSplitWinner* result) {
  if (!initialized_) {
    return Fail(kSimtSplitNotInitialized, "Init() has not been called");
  }
  if (result == nullptr) {
    return Fail(kSimtSplitInvalidArg, "null result");
  }
  aclrtSetCurrentContext(context_);
  int32_t status = Launch();
  if (status != kSimtSplitOk) {
    return status;
  }
  const aclError code = aclrtSynchronizeStream(stream_);
  if (!IsOk(code)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtSynchronizeStream", code).c_str());
  }
  const aclError copy = aclrtMemcpy(result, sizeof(SimtSplitWinner),
                                    device_winner_, sizeof(SimtSplitWinner),
                                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (!IsOk(copy)) {
    return Fail(kSimtSplitAclError,
                AclMessage("aclrtMemcpy(winner)", copy).c_str());
  }
  return kSimtSplitOk;
}

int32_t SimtSplitOperator::Run(const SimtSplitTask* tasks, int32_t num_tasks,
                               const SimtSplitParams& params,
                               const SimtSplitHistogram& host_histogram,
                               SimtSplitWinner* result) {
  int32_t status = UploadTasks(tasks, num_tasks, params);
  if (status != kSimtSplitOk) {
    return status;
  }
  status = UploadHistogram(host_histogram);
  if (status != kSimtSplitOk) {
    return status;
  }
  return RunPrepared(result);
}

int32_t SimtSplitOperator::DownloadCandidates(SimtSplitCandidate* out,
                                              int32_t num_tasks) const {
  if (!initialized_ || device_candidates_ == nullptr) {
    return kSimtSplitNotInitialized;
  }
  aclrtSetCurrentContext(context_);
  if (out == nullptr || num_tasks <= 0 || num_tasks > num_tasks_) {
    return kSimtSplitInvalidArg;
  }
  const aclError code = aclrtSynchronizeStream(stream_);
  if (!IsOk(code)) {
    return kSimtSplitAclError;
  }
  const aclError copy = aclrtMemcpy(
      out, static_cast<size_t>(num_tasks * sizeof(SimtSplitCandidate)),
      device_candidates_,
      static_cast<size_t>(num_tasks * sizeof(SimtSplitCandidate)),
      ACL_MEMCPY_DEVICE_TO_HOST);
  return IsOk(copy) ? kSimtSplitOk : kSimtSplitAclError;
}

}  // namespace lgbm_npu_simt
