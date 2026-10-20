// Phase 1 gate 6: can chunked fp32 recover precision while keeping fp32 speed?
// Accumulate each chunk on NPU in fp32, then reduce the partial histograms in fp64 on host.
#include <acl/acl.h>
#include <aclnnop/aclnn_bincount.h>

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

int main() {
  const int64_t rows = 300000, num_feat = 158, num_bins = 255;
  const int64_t total = rows * num_feat, buckets = num_feat * num_bins;

  std::vector<int32_t> packed(total);
  std::vector<float> gf(total);
  std::mt19937 rng(13);
  std::uniform_int_distribution<int32_t> bd(0, static_cast<int32_t>(num_bins) - 1);
  std::uniform_real_distribution<float> gd(-1.0f, 1.0f);
  for (int64_t f = 0; f < num_feat; ++f) {
    const int64_t base = f * num_bins;
    for (int64_t i = 0; i < rows; ++i) {
      const int64_t k = f * rows + i;
      packed[k] = static_cast<int32_t>(base + bd(rng));
      gf[k] = gd(rng);
    }
  }
  std::vector<double> ref(buckets, 0.0);
  for (int64_t i = 0; i < total; ++i) ref[packed[i]] += static_cast<double>(gf[i]);

  aclInit(nullptr);
  aclrtSetDevice(0);
  aclrtStream stream = nullptr;
  aclrtCreateStream(&stream);

  // Device buffers sized for the whole dataset; chunks are views via offset pointers.
  void *d_bins = nullptr, *d_w = nullptr, *d_out = nullptr;
  aclrtMalloc(&d_bins, total * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&d_w, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&d_out, buckets * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(d_bins, total * sizeof(int32_t), packed.data(), total * sizeof(int32_t),
              ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(d_w, total * sizeof(float), gf.data(), total * sizeof(float),
              ACL_MEMCPY_HOST_TO_DEVICE);

  std::printf("entries=%lld buckets=%lld  avg values per bucket=%.0f\n",
              (long long)total, (long long)buckets, double(total) / double(buckets));
  std::printf("%10s %8s %12s %14s %14s\n", "chunks", "calls", "kernel_ms", "max_rel_err", "d2h_MB");

  for (int64_t nchunk : {1LL, 8LL, 64LL, 512LL}) {
    const int64_t csize = (total + nchunk - 1) / nchunk;
    std::vector<double> acc(buckets, 0.0);
    std::vector<float> part(buckets);
    double kernel_ms = 0;
    int64_t calls = 0;

    for (int64_t c = 0; c < nchunk; ++c) {
      const int64_t off = c * csize;
      const int64_t len = std::min(csize, total - off);
      if (len <= 0) break;
      aclTensor* t_self = Make1D(static_cast<char*>(d_bins) + off * sizeof(int32_t), len, ACL_INT32);
      aclTensor* t_w = Make1D(static_cast<char*>(d_w) + off * sizeof(float), len, ACL_FLOAT);
      aclTensor* t_out = Make1D(d_out, buckets, ACL_FLOAT);

      uint64_t ws_size = 0;
      aclOpExecutor* exec = nullptr;
      if (aclnnBincountGetWorkspaceSize(t_self, t_w, buckets, t_out, &ws_size, &exec) != ACL_SUCCESS) {
        std::printf("  chunk %lld: workspace query failed\n", (long long)c);
        break;
      }
      void* ws = nullptr;
      if (ws_size > 0) aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
      auto t0 = std::chrono::high_resolution_clock::now();
      aclnnBincount(ws, ws_size, exec, stream);
      aclrtSynchronizeStream(stream);
      auto t1 = std::chrono::high_resolution_clock::now();
      kernel_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      ++calls;
      if (ws) aclrtFree(ws);

      // Reduce this chunk's partial histogram into the fp64 accumulator on host.
      aclrtMemcpy(part.data(), buckets * sizeof(float), d_out, buckets * sizeof(float),
                  ACL_MEMCPY_DEVICE_TO_HOST);
      for (int64_t b = 0; b < buckets; ++b) acc[b] += static_cast<double>(part[b]);

      aclDestroyTensor(t_self);
      aclDestroyTensor(t_w);
      aclDestroyTensor(t_out);
    }

    double maxrel = 0;
    for (int64_t b = 0; b < buckets; ++b) {
      const double den = std::fabs(ref[b]) > 1e-12 ? std::fabs(ref[b]) : 1.0;
      maxrel = std::max(maxrel, std::fabs(acc[b] - ref[b]) / den);
    }
    std::printf("%10lld %8lld %12.2f %14.3e %14.1f\n",
                (long long)nchunk, (long long)calls, kernel_ms, maxrel,
                calls * buckets * 4.0 / 1e6);
  }

  std::printf("\nreference: fp64 single call was ~181 ms, int32 exact ~192 ms\n");

  aclrtFree(d_bins);
  aclrtFree(d_w);
  aclrtFree(d_out);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}