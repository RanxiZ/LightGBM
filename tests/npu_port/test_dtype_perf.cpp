// Phase 1 gate 4: why is fp64 bincount slow at scale? Compare fp32 vs fp64 across sizes,
// and check multi-threaded CPU as the real baseline (LightGBM CPU uses num_threads=20).
#include <acl/acl.h>
#include <aclnnop/aclnn_bincount.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static aclTensor* Make1D(void* dev, int64_t n, aclDataType dt) {
  int64_t dims[1] = {n};
  int64_t strides[1] = {1};
  return aclCreateTensor(dims, 1, dt, strides, 0, ACL_FORMAT_ND, dims, 1, dev);
}

// Returns best-of-3 kernel milliseconds, or -1 if the op is unsupported for this dtype.
static double TimeBincount(const std::vector<int32_t>& packed, int64_t buckets,
                           aclDataType dt, size_t elem_bytes, const void* weights,
                           uint64_t* ws_report) {
  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_SUCCESS) return -1;
  void *d_bins = nullptr, *d_w = nullptr, *d_out = nullptr;
  const int64_t total = static_cast<int64_t>(packed.size());
  const size_t bb = total * sizeof(int32_t);
  const size_t wb = total * elem_bytes;
  const size_t ob = buckets * elem_bytes;
  if (aclrtMalloc(&d_bins, bb, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return -1;
  if (aclrtMalloc(&d_w, wb, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return -1;
  if (aclrtMalloc(&d_out, ob, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return -1;
  aclrtMemcpy(d_bins, bb, packed.data(), bb, ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(d_w, wb, weights, wb, ACL_MEMCPY_HOST_TO_DEVICE);

  aclTensor* t_self = Make1D(d_bins, total, ACL_INT32);
  aclTensor* t_w = Make1D(d_w, total, dt);
  aclTensor* t_out = Make1D(d_out, buckets, dt);

  double best = 1e18;
  for (int r = 0; r < 3; ++r) {
    uint64_t ws_size = 0;
    aclOpExecutor* exec = nullptr;
    if (aclnnBincountGetWorkspaceSize(t_self, t_w, buckets, t_out, &ws_size, &exec) != ACL_SUCCESS) {
      best = -1;
      break;
    }
    *ws_report = ws_size;
    void* ws = nullptr;
    if (ws_size > 0 && aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      best = -1;
      break;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    const aclnnStatus st = aclnnBincount(ws, ws_size, exec, stream);
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ws) aclrtFree(ws);
    if (st != ACL_SUCCESS) { best = -1; break; }
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms < best) best = ms;
  }

  aclDestroyTensor(t_self);
  aclDestroyTensor(t_w);
  aclDestroyTensor(t_out);
  aclrtFree(d_bins);
  aclrtFree(d_w);
  aclrtFree(d_out);
  aclrtDestroyStream(stream);
  return best;
}

int main() {
  const int64_t num_feat = 158;
  const int64_t num_bins = 255;
  const int64_t buckets = num_feat * num_bins;

  aclInit(nullptr);
  aclrtSetDevice(0);
  std::printf("buckets=%lld  omp_max_threads=%d\n", (long long)buckets, omp_get_max_threads());
  std::printf("%10s %12s %12s %12s %12s %10s\n",
              "rows", "entries", "fp64_ms", "fp32_ms", "cpu20_ms", "fp32_vs_cpu");

  std::mt19937 rng(5);
  std::uniform_int_distribution<int32_t> bd(0, static_cast<int32_t>(num_bins) - 1);
  std::uniform_real_distribution<double> gd(-1.0, 1.0);

  for (int64_t rows : {10000LL, 50000LL, 150000LL, 300000LL}) {
    const int64_t total = rows * num_feat;
    std::vector<int32_t> packed(total);
    std::vector<double> wd(total);
    std::vector<float> wf(total);
    for (int64_t f = 0; f < num_feat; ++f) {
      const int64_t base = f * num_bins;
      for (int64_t i = 0; i < rows; ++i) {
        const int64_t idx = f * rows + i;
        packed[idx] = static_cast<int32_t>(base + bd(rng));
        wd[idx] = gd(rng);
        wf[idx] = static_cast<float>(wd[idx]);
      }
    }

    // Multi-threaded CPU baseline with per-thread private histograms,
    // which is how LightGBM's CPU path actually accumulates.
    auto c0 = std::chrono::high_resolution_clock::now();
    std::vector<double> ref(buckets, 0.0);
#pragma omp parallel
    {
      std::vector<double> local(buckets, 0.0);
#pragma omp for schedule(static)
      for (int64_t i = 0; i < total; ++i) local[packed[i]] += wd[i];
#pragma omp critical
      for (int64_t b = 0; b < buckets; ++b) ref[b] += local[b];
    }
    auto c1 = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    uint64_t ws64 = 0, ws32 = 0;
    const double ms64 = TimeBincount(packed, buckets, ACL_DOUBLE, 8, wd.data(), &ws64);
    const double ms32 = TimeBincount(packed, buckets, ACL_FLOAT, 4, wf.data(), &ws32);

    std::printf("%10lld %12lld %12.2f %12.2f %12.2f %10.2fx\n",
                (long long)rows, (long long)total, ms64, ms32, cpu_ms,
                ms32 > 0 ? cpu_ms / ms32 : 0.0);
  }

  std::printf("\nnote: fp64 workspace bytes vs fp32 workspace bytes differ,\n");
  std::printf("which suggests different kernel implementations per dtype.\n");

  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}