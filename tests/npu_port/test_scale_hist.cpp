// Phase 1 gate 3: realistic-scale histogram. Packs all features into one bincount call
// (bucket = feature_index * num_bins + local_bin) and compares NPU time vs CPU time.
#include <acl/acl.h>
#include <aclnnop/aclnn_bincount.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define CHK(expr, msg)                                                \
  do {                                                                \
    aclnnStatus _s = (expr);                                          \
    if (_s != ACL_SUCCESS) {                                          \
      std::printf("FAIL %s status=%d\n", msg, static_cast<int>(_s));  \
      return 1;                                                       \
    }                                                                 \
  } while (0)

static aclTensor* Make1D(void* dev, int64_t n, aclDataType dt) {
  int64_t dims[1] = {n};
  int64_t strides[1] = {1};
  return aclCreateTensor(dims, 1, dt, strides, 0, ACL_FORMAT_ND, dims, 1, dev);
}

int main() {
  // Alpha158-like training shape for CSI300.
  const int64_t num_data = 300000;
  const int64_t num_feat = 158;
  const int64_t num_bins = 255;
  const int64_t total = num_data * num_feat;
  const int64_t buckets = num_feat * num_bins;

  std::printf("num_data=%lld num_feat=%lld num_bins=%lld\n",
              (long long)num_data, (long long)num_feat, (long long)num_bins);
  std::printf("packed entries=%lld  int32=%.2f GB  fp64 weights=%.2f GB  buckets=%lld\n",
              (long long)total, total * 4.0 / 1e9, total * 8.0 / 1e9, (long long)buckets);

  // Host side: packed bucket ids and the gradient repeated for each feature column.
  std::vector<int32_t> packed(total);
  std::vector<double> wrep(total);
  std::vector<double> grad(num_data);
  std::mt19937 rng(11);
  std::uniform_int_distribution<int32_t> bd(0, static_cast<int32_t>(num_bins) - 1);
  std::uniform_real_distribution<double> gd(-1.0, 1.0);
  for (int64_t i = 0; i < num_data; ++i) grad[i] = gd(rng);
  for (int64_t f = 0; f < num_feat; ++f) {
    const int64_t base = f * num_bins;
    for (int64_t i = 0; i < num_data; ++i) {
      const int64_t idx = f * num_data + i;
      packed[idx] = static_cast<int32_t>(base + bd(rng));
      wrep[idx] = grad[i];
    }
  }

  // CPU reference: same accumulation, single-threaded, for a correctness and time baseline.
  std::vector<double> ref(buckets, 0.0);
  auto c0 = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < total; ++i) ref[packed[i]] += wrep[i];
  auto c1 = std::chrono::high_resolution_clock::now();
  const double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

  aclInit(nullptr);
  aclrtSetDevice(0);
  aclrtStream stream = nullptr;
  CHK(aclrtCreateStream(&stream), "create stream");

  void *d_bins = nullptr, *d_w = nullptr, *d_out = nullptr;
  const size_t bb = total * sizeof(int32_t);
  const size_t wb = total * sizeof(double);
  const size_t ob = buckets * sizeof(double);
  CHK(aclrtMalloc(&d_bins, bb, ACL_MEM_MALLOC_HUGE_FIRST), "malloc bins");
  CHK(aclrtMalloc(&d_w, wb, ACL_MEM_MALLOC_HUGE_FIRST), "malloc weights");
  CHK(aclrtMalloc(&d_out, ob, ACL_MEM_MALLOC_HUGE_FIRST), "malloc out");

  // Time the host-to-device transfer separately: it is a real per-iteration cost
  // if gradients do not stay resident on the device.
  auto h0 = std::chrono::high_resolution_clock::now();
  CHK(aclrtMemcpy(d_bins, bb, packed.data(), bb, ACL_MEMCPY_HOST_TO_DEVICE), "copy bins");
  auto h1 = std::chrono::high_resolution_clock::now();
  CHK(aclrtMemcpy(d_w, wb, wrep.data(), wb, ACL_MEMCPY_HOST_TO_DEVICE), "copy weights");
  auto h2 = std::chrono::high_resolution_clock::now();
  const double bins_copy_ms = std::chrono::duration<double, std::milli>(h1 - h0).count();
  const double w_copy_ms = std::chrono::duration<double, std::milli>(h2 - h1).count();

  aclTensor* t_self = Make1D(d_bins, total, ACL_INT32);
  aclTensor* t_w = Make1D(d_w, total, ACL_DOUBLE);
  aclTensor* t_out = Make1D(d_out, buckets, ACL_DOUBLE);

  double best_ms = 1e18;
  for (int r = 0; r < 4; ++r) {
    uint64_t ws_size = 0;
    aclOpExecutor* exec = nullptr;
    CHK(aclnnBincountGetWorkspaceSize(t_self, t_w, buckets, t_out, &ws_size, &exec), "ws");
    void* ws = nullptr;
    if (ws_size > 0) CHK(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");
    auto t0 = std::chrono::high_resolution_clock::now();
    CHK(aclnnBincount(ws, ws_size, exec, stream), "bincount");
    CHK(aclrtSynchronizeStream(stream), "sync");
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r > 0 && ms < best_ms) best_ms = ms;  // skip first (warm-up) iteration
    if (ws) aclrtFree(ws);
  }

  std::vector<double> out(buckets, 0.0);
  CHK(aclrtMemcpy(out.data(), ob, d_out, ob, ACL_MEMCPY_DEVICE_TO_HOST), "copy out");

  double max_rel = 0.0;
  int64_t worst = -1;
  for (int64_t b = 0; b < buckets; ++b) {
    const double den = std::fabs(ref[b]) > 1e-12 ? std::fabs(ref[b]) : 1.0;
    const double rel = std::fabs(out[b] - ref[b]) / den;
    if (rel > max_rel) { max_rel = rel; worst = b; }
  }

  std::printf("\ncorrectness: max_rel_err=%.3e at bucket %lld\n", max_rel, (long long)worst);
  std::printf("npu kernel     : %.2f ms\n", best_ms);
  std::printf("cpu 1-thread   : %.2f ms\n", cpu_ms);
  std::printf("h2d bins copy  : %.2f ms\n", bins_copy_ms);
  std::printf("h2d weight copy: %.2f ms\n", w_copy_ms);
  std::printf("kernel speedup vs 1-thread cpu : %.2fx\n", cpu_ms / best_ms);
  std::printf("with-transfer  vs 1-thread cpu : %.2fx\n",
              cpu_ms / (best_ms + bins_copy_ms + w_copy_ms));
  std::printf("%s\n", max_rel < 1e-12 ? "RESULT: PASS (exact)" : "RESULT: CHECK PRECISION");

  aclDestroyTensor(t_self);
  aclDestroyTensor(t_w);
  aclDestroyTensor(t_out);
  aclrtFree(d_bins);
  aclrtFree(d_w);
  aclrtFree(d_out);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}