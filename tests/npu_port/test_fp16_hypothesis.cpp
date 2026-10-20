// Phase 1 gate 7: decisive check on the fp32 bincount kernel's internal accumulation precision.
// Test A: sum of N exact 1.0 values into one bucket. True answer is N.
//   fp32 accumulation is exact up to N = 2^24; fp16 accumulation saturates near 2048.
// Test B: accumulate values far apart in magnitude to expose the working precision.
#include <acl/acl.h>
#include <aclnnop/aclnn_bincount.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static aclTensor* Make1D(void* dev, int64_t n, aclDataType dt) {
  int64_t dims[1] = {n};
  int64_t strides[1] = {1};
  return aclCreateTensor(dims, 1, dt, strides, 0, ACL_FORMAT_ND, dims, 1, dev);
}

// Accumulates `weights` into a single bucket and returns the kernel's result.
static bool SumOneBucket(const std::vector<float>& weights, float* result) {
  const int64_t n = static_cast<int64_t>(weights.size());
  const int64_t buckets = 1;
  std::vector<int32_t> bins(n, 0);  // everything lands in bucket 0

  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_SUCCESS) return false;
  void *d_bins = nullptr, *d_w = nullptr, *d_out = nullptr;
  aclrtMalloc(&d_bins, n * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&d_w, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&d_out, buckets * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(d_bins, n * sizeof(int32_t), bins.data(), n * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(d_w, n * sizeof(float), weights.data(), n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

  aclTensor* t_self = Make1D(d_bins, n, ACL_INT32);
  aclTensor* t_w = Make1D(d_w, n, ACL_FLOAT);
  aclTensor* t_out = Make1D(d_out, buckets, ACL_FLOAT);

  uint64_t ws_size = 0;
  aclOpExecutor* exec = nullptr;
  bool ok = false;
  if (aclnnBincountGetWorkspaceSize(t_self, t_w, buckets, t_out, &ws_size, &exec) == ACL_SUCCESS) {
    void* ws = nullptr;
    if (ws_size > 0) aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclnnBincount(ws, ws_size, exec, stream) == ACL_SUCCESS &&
        aclrtSynchronizeStream(stream) == ACL_SUCCESS) {
      aclrtMemcpy(result, sizeof(float), d_out, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
      ok = true;
    }
    if (ws) aclrtFree(ws);
  }

  aclDestroyTensor(t_self);
  aclDestroyTensor(t_w);
  aclDestroyTensor(t_out);
  aclrtFree(d_bins);
  aclrtFree(d_w);
  aclrtFree(d_out);
  aclrtDestroyStream(stream);
  return ok;
}

int main() {
  aclInit(nullptr);
  aclrtSetDevice(0);

  std::printf("Test A: sum of N ones in one bucket (exact answer = N)\n");
  std::printf("fp32 stays exact to N=16777216; fp16 breaks just past N=2048\n\n");
  std::printf("%10s %16s %16s %14s\n", "N", "expected", "kernel_result", "rel_err");
  for (int64_t n : {100LL, 1000LL, 2048LL, 2049LL, 4096LL, 65536LL, 1000000LL}) {
    std::vector<float> w(n, 1.0f);
    float got = 0.0f;
    if (!SumOneBucket(w, &got)) {
      std::printf("%10lld %16s\n", (long long)n, "FAILED");
      continue;
    }
    const double exp = static_cast<double>(n);
    std::printf("%10lld %16.1f %16.1f %14.3e\n", (long long)n, exp, (double)got,
                std::fabs((double)got - exp) / exp);
  }

  std::printf("\nTest B: 1.0 plus 1000 copies of 0.001 (exact answer = 2.0)\n");
  {
    std::vector<float> w;
    w.push_back(1.0f);
    for (int i = 0; i < 1000; ++i) w.push_back(0.001f);
    float got = 0.0f;
    if (SumOneBucket(w, &got)) {
      std::printf("  expected 2.0  kernel_result %.6f  abs_err %.3e\n", (double)got,
                  std::fabs((double)got - 2.0));
    }
  }

  std::printf("\nTest C: 1e6 plus 1000 copies of 1.0 (exact answer = 1001000)\n");
  {
    std::vector<float> w;
    w.push_back(1e6f);
    for (int i = 0; i < 1000; ++i) w.push_back(1.0f);
    float got = 0.0f;
    if (SumOneBucket(w, &got)) {
      std::printf("  expected 1001000  kernel_result %.1f  rel_err %.3e\n", (double)got,
                  std::fabs((double)got - 1001000.0) / 1001000.0);
    }
  }

  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}