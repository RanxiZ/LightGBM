/* Copyright (c) 2026. Licensed under the MIT License. */
//
// Host-side operator runtime for the Ascend SIMT best-split finder.
//
// No LightGBM dependency: the operator only sees the POD contract in
// `npu_simt_split_abi.h`.  That is what lets the standalone probe in
// tests/cpp_tests drive it without a training loop.
//
// Counterpart of `CUDABestSplitFinder::FindBestSplitsForLeaf`
// (cuda_best_split_finder.cpp:305): validate the leaf, upload the task list and
// leaf statistics, launch the scan kernel then the merge kernel on one stream,
// download the winner.
//
// Lifecycle:  Init(device) -> [Upload* / Run]* -> Reset() -> Release()
// Errors are returned as `SimtSplitStatus` codes; nothing throws, so the
// operator is safe inside a training loop.

#ifndef NPU_SIMT_SPLIT_OP_H
#define NPU_SIMT_SPLIT_OP_H

#include <cstdint>
#include <string>

#include "npu_simt_split_abi.h"

// ACL handle types, spelled exactly as `acl/acl_base.h` does.  Redeclaring an
// identical typedef is legal, so this header stays free of acl includes while
// remaining compatible with translation units that do include them.
typedef void* aclrtContext;
typedef void* aclrtStream;

namespace lgbm_npu_simt {

// Histogram planes, gradient-major: `gradients[hist_offset + bin*bin_stride]`.
struct SimtSplitHistogram {
  const float* gradients = nullptr;
  const float* hessians = nullptr;
  // Number of bins addressable from bin 0 of the first feature.  The operator
  // validates that every task stays inside this span.
  int64_t num_bins = 0;
};

class SimtSplitOperator {
 public:
  SimtSplitOperator();
  ~SimtSplitOperator();
  SimtSplitOperator(const SimtSplitOperator&) = delete;
  SimtSplitOperator& operator=(const SimtSplitOperator&) = delete;

  // Acquires the device and creates a stream.  `device_id` must match the
  // process' visible device (ASCEND_RT_VISIBLE_DEVICES is already applied by
  // ACL, so this is the logical index).
  int32_t Init(int32_t device_id);

  // LightGBM integration: run on an ACL context and stream that the caller
  // already owns (NPURuntime::Impl).  Nothing is created and, unlike Init(),
  // Release() neither destroys them nor resets the device -- the caller keeps
  // both alive for the operator's lifetime.  This is what lets the operator
  // bind device-resident histogram planes the caller allocated.
  int32_t Attach(int32_t device_id, aclrtContext context, aclrtStream stream);

  // Drops every device buffer but keeps the stream, so the next Run re-uploads.
  void Reset();

  // Destroys the stream, the context and the device reference.  Idempotent.
  void Release();

  // --- staged interface -----------------------------------------------------
  // Tasks and params must be uploaded before RunPrepared; the histogram is
  // either uploaded from host memory or bound from an existing device buffer.
  int32_t UploadTasks(const SimtSplitTask* tasks, int32_t num_tasks,
                      SimtSplitParams params);
  int32_t UploadHistogram(const SimtSplitHistogram& host_histogram);
  // Bind device-resident planes.  The caller keeps them alive across the call;
  // no copy is issued and nothing is freed by Reset().
  int32_t BindDeviceHistogram(const float* device_gradients,
                              const float* device_hessians,
                              int64_t num_bins);

  // Launch scan + merge on the stream, synchronise, download the winner.
  // This is what the kernel-only benchmark loop times.
  int32_t RunPrepared(SimtSplitWinner* result);

  // --- one-shot interface ---------------------------------------------------
  int32_t Run(const SimtSplitTask* tasks, int32_t num_tasks,
              const SimtSplitParams& params,
              const SimtSplitHistogram& host_histogram, SimtSplitWinner* result);

  // Diagnostic: download every per-task candidate of the last Run.
  int32_t DownloadCandidates(SimtSplitCandidate* out, int32_t num_tasks) const;

  // --- introspection --------------------------------------------------------
  int32_t num_blocks() const { return num_blocks_; }
  int32_t vector_cores() const { return vector_cores_; }
  int32_t num_tasks() const { return num_tasks_; }
  int64_t device_bytes() const;
  const char* last_error() const { return last_error_.c_str(); }

 private:
  int32_t Fail(int32_t status, const char* what);
  // Shared tail of Init()/Attach(): read the core count, mark the operator ready.
  int32_t AdoptDevice(int32_t device_id);
  int32_t EnsureTaskBuffers(int32_t num_tasks);
  int32_t EnsureHistogramBuffers(int64_t num_bins);
  void DropHistogramStorage();
  int32_t ValidateTasks(int32_t num_tasks) const;
  int32_t Launch();

  bool initialized_ = false;
  // False when the context/stream were supplied by Attach(); Release() then
  // leaves them (and the device reference) to the caller.
  bool owns_context_ = false;
  int32_t device_id_ = -1;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;
  int32_t vector_cores_ = 0;
  int32_t num_blocks_ = 0;

  // Device buffers.  `*_capacity` is in elements.
  void* device_tasks_ = nullptr;
  void* device_params_ = nullptr;
  void* device_candidates_ = nullptr;
  void* device_winner_ = nullptr;
  float* device_hist_gradients_ = nullptr;
  float* device_hist_hessians_ = nullptr;
  int64_t tasks_capacity_ = 0;
  int64_t hist_capacity_ = 0;
  bool hist_is_bound_ = false;
  int64_t hist_bound_bins_ = 0;

  // Host-side staging for the asynchronous parameter upload.
  SimtSplitParams host_params_{};
  int32_t num_tasks_ = 0;
  bool tasks_staged_ = false;
  bool hist_staged_ = false;
  int64_t staged_hist_bins_ = 0;
  std::string last_error_;
};

}  // namespace lgbm_npu_simt

#endif  // NPU_SIMT_SPLIT_OP_H
