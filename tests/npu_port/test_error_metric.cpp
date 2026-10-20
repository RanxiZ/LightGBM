// Phase 1 gate 8: measure fp32 histogram error with a metric that is meaningful under
// cancellation. Relative error against a near-zero bucket sum is misleading, so we also
// normalize by the sum of absolute values in the bucket (the actual rounding scale),
// and we check what matters for LightGBM: whether split gains / argmax decisions change.
#include <acl/acl.h>
#include <aclnnop/aclnn_bincount.h>

#include <algorithm>
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
  // Gradient distribution closer to a real boosting iteration: centered but skewed,
  // so bucket sums are small but not exactly zero.
  std::normal_distribution<float> gd(0.05f, 0.4f);
  for (int64_t f = 0; f < num_feat; ++f) {
    const int64_t base = f * num_bins;
    for (int64_t i = 0; i < rows; ++i) {
      const int64_t k = f * rows + i;
      packed[k] = static_cast<int32_t>(base + bd(rng));
      gf[k] = gd(rng);
    }
  }

  // Exact fp64 reference plus the per-bucket absolute scale.
  std::vector<double> ref(buckets, 0.0), absscale(buckets, 0.0);
  for (int64_t i = 0; i < total; ++i) {
    ref[packed[i]] += static_cast<double>(gf[i]);
    absscale[packed[i]] += std::fabs(static_cast<double>(gf[i]));
  }

  aclInit(nullptr);
  aclrtSetDevice(0);
  aclrtStream stream = nullptr;
  aclrtCreateStream(&stream);

  void *d_bins = nullptr, *d_w = nullptr, *d_out = nullptr;
  aclrtMalloc(&d_bins, total * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&d_w, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&d_out, buckets * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(d_bins, total * sizeof(int32_t), packed.data(), total * sizeof(int32_t),
              ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(d_w, total * sizeof(float), gf.data(), total * sizeof(float),
              ACL_MEMCPY_HOST_TO_DEVICE);

  aclTensor* t_self = Make1D(d_bins, total, ACL_INT32);
  aclTensor* t_w = Make1D(d_w, total, ACL_FLOAT);
  aclTensor* t_out = Make1D(d_out, buckets, ACL_FLOAT);
  uint64_t ws_size = 0;
  aclOpExecutor* exec = nullptr;
  aclnnBincountGetWorkspaceSize(t_self, t_w, buckets, t_out, &ws_size, &exec);
  void* ws = nullptr;
  if (ws_size > 0) aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnBincount(ws, ws_size, exec, stream);
  aclrtSynchronizeStream(stream);
  std::vector<float> got(buckets);
  aclrtMemcpy(got.data(), buckets * sizeof(float), d_out, buckets * sizeof(float),
              ACL_MEMCPY_DEVICE_TO_HOST);

  double max_abs = 0, max_norm = 0, max_rel = 0;
  for (int64_t b = 0; b < buckets; ++b) {
    const double e = std::fabs(static_cast<double>(got[b]) - ref[b]);
    max_abs = std::max(max_abs, e);
    if (absscale[b] > 0) max_norm = std::max(max_norm, e / absscale[b]);
    const double den = std::fabs(ref[b]) > 1e-12 ? std::fabs(ref[b]) : 1.0;
    max_rel = std::max(max_rel, e / den);
  }

  std::printf("fp32 single-call bincount, rows=%lld feat=%lld\n", (long long)rows, (long long)num_feat);
  std::printf("  max_abs_err                    = %.3e\n", max_abs);
  std::printf("  max_err / sum|g| in bucket     = %.3e   <- true rounding scale\n", max_norm);
  std::printf("  max_err / |bucket sum|         = %.3e   <- inflated by cancellation\n", max_rel);
  std::printf("  typical |bucket sum|           = %.3e\n", std::fabs(ref[0]));
  std::printf("  typical sum|g| per bucket      = %.3e\n", absscale[0]);

  // What LightGBM actually consumes: cumulative prefix sums drive split gain.
  // Check whether the best split bin chosen per feature changes between fp32 and fp64.
  int64_t changed = 0;
  for (int64_t f = 0; f < num_feat; ++f) {
    double c64 = 0, c32 = 0, best64 = -1e300, best32 = -1e300;
    int64_t arg64 = -1, arg32 = -1;
    for (int64_t b = 0; b < num_bins; ++b) {
      const int64_t k = f * num_bins + b;
      c64 += ref[k];
      c32 += static_cast<double>(got[k]);
      // Simplified gain proxy: squared cumulative gradient, which is what the
      // real gain formula is monotone in for a fixed count split.
      if (c64 * c64 > best64) { best64 = c64 * c64; arg64 = b; }
      if (c32 * c32 > best32) { best32 = c32 * c32; arg32 = b; }
    }
    if (arg64 != arg32) ++changed;
  }
  std::printf("\n  features whose argmax split bin changed: %lld / %lld\n",
              (long long)changed, (long long)num_feat);
  std::printf("  %s\n", changed == 0 ? "RESULT: fp32 preserves split decisions on this data"
                                     : "RESULT: fp32 alters some split decisions");

  if (ws) aclrtFree(ws);
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