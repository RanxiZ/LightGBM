// Phase 1 gate: verify weighted histogram accumulation on Ascend NPU via aclnnBincount.
// Compares NPU result against CPU reference for gradient/hessian style accumulation.
#include <acl/acl.h>
#include <aclnnop/aclnn_bincount.h>

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#define CHK(expr, msg)                                                   \
  do {                                                                   \
    aclnnStatus _s = (expr);                                             \
    if (_s != ACL_SUCCESS) {                                             \
      std::printf("FAIL %s: status=%d\n", msg, static_cast<int>(_s));     \
      return 1;                                                          \
    }                                                                    \
  } while (0)

namespace {

// Creates a 1-D contiguous aclTensor over already-allocated device memory.
aclTensor* Make1D(void* dev, int64_t n, aclDataType dt) {
  int64_t dims[1] = {n};
  int64_t strides[1] = {1};
  return aclCreateTensor(dims, 1, dt, strides, 0, ACL_FORMAT_ND, dims, 1, dev);
}

}  // namespace

int main() {
  const int64_t num_data = 100000;   // samples in one leaf
  const int64_t num_bins = 256;      // LightGBM max_bin style bucket count

  // Host inputs: bin index per sample, plus float32 weights (gradients).
  std::vector<int32_t> bins(num_data);
  std::vector<float> grads(num_data);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int32_t> bin_dist(0, static_cast<int32_t>(num_bins) - 1);
  std::uniform_real_distribution<float> g_dist(-1.0f, 1.0f);
  for (int64_t i = 0; i < num_data; ++i) {
    bins[i] = bin_dist(rng);
    grads[i] = g_dist(rng);
  }

  // CPU reference accumulation in double for a stable comparison baseline.
  std::vector<double> ref(num_bins, 0.0);
  for (int64_t i = 0; i < num_data; ++i) ref[bins[i]] += static_cast<double>(grads[i]);

  CHK(aclInit(nullptr), "aclInit");
  CHK(aclrtSetDevice(0), "aclrtSetDevice");
  aclrtStream stream = nullptr;
  CHK(aclrtCreateStream(&stream), "aclrtCreateStream");

  void *d_bins = nullptr, *d_grads = nullptr, *d_out = nullptr;
  const size_t bins_bytes = bins.size() * sizeof(int32_t);
  const size_t grads_bytes = grads.size() * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(num_bins) * sizeof(float);
  CHK(aclrtMalloc(&d_bins, bins_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc bins");
  CHK(aclrtMalloc(&d_grads, grads_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc grads");
  CHK(aclrtMalloc(&d_out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc out");
  CHK(aclrtMemcpy(d_bins, bins_bytes, bins.data(), bins_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy bins");
  CHK(aclrtMemcpy(d_grads, grads_bytes, grads.data(), grads_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy grads");

  aclTensor* t_self = Make1D(d_bins, num_data, ACL_INT32);
  aclTensor* t_weights = Make1D(d_grads, num_data, ACL_FLOAT);
  aclTensor* t_out = Make1D(d_out, num_bins, ACL_FLOAT);

  uint64_t ws_size = 0;
  aclOpExecutor* exec = nullptr;
  CHK(aclnnBincountGetWorkspaceSize(t_self, t_weights, num_bins, t_out, &ws_size, &exec),
      "aclnnBincountGetWorkspaceSize");
  void* ws = nullptr;
  if (ws_size > 0) CHK(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
  CHK(aclnnBincount(ws, ws_size, exec, stream), "aclnnBincount");
  CHK(aclrtSynchronizeStream(stream), "sync");

  std::vector<float> out(num_bins, 0.0f);
  CHK(aclrtMemcpy(out.data(), out_bytes, d_out, out_bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy out");

  // Compare every bucket; report the worst absolute and relative deviation.
  double max_abs = 0.0, max_rel = 0.0;
  for (int64_t b = 0; b < num_bins; ++b) {
    const double diff = std::fabs(static_cast<double>(out[b]) - ref[b]);
    const double denom = std::fabs(ref[b]) > 1e-12 ? std::fabs(ref[b]) : 1.0;
    if (diff > max_abs) max_abs = diff;
    if (diff / denom > max_rel) max_rel = diff / denom;
  }
  std::printf("num_data=%lld num_bins=%lld workspace=%llu bytes\n",
              static_cast<long long>(num_data), static_cast<long long>(num_bins),
              static_cast<unsigned long long>(ws_size));
  std::printf("sample bin0: npu=%.6f cpu=%.6f\n", out[0], ref[0]);
  std::printf("max_abs_err=%.6e max_rel_err=%.6e\n", max_abs, max_rel);
  // float32 accumulation over 1e5 values: tolerate small relative drift only.
  std::printf("%s\n", max_rel < 1e-4 ? "RESULT: PASS" : "RESULT: FAIL");

  aclDestroyTensor(t_self);
  aclDestroyTensor(t_weights);
  aclDestroyTensor(t_out);
  if (ws) aclrtFree(ws);
  aclrtFree(d_bins);
  aclrtFree(d_grads);
  aclrtFree(d_out);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}