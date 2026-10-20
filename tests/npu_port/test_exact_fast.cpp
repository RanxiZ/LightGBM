// Phase 1 gate 5: find a histogram accumulation mode that is both exact and fast.
// Candidates: int32 weights (LightGBM's int_hist_t / use_quantized_grad path),
// int64 weights, and fp32 with Kahan-style split accumulation.
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

struct Result {
  bool supported = false;
  double ms = -1;
  uint64_t ws = 0;
};

// Runs bincount with the given weight dtype and copies the raw output bytes back.
static Result Run(const std::vector<int32_t>& packed, int64_t buckets, aclDataType dt,
                  size_t elem_bytes, const void* weights, void* out_host) {
  Result r;
  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_SUCCESS) return r;
  const int64_t total = static_cast<int64_t>(packed.size());
  void *d_bins = nullptr, *d_w = nullptr, *d_out = nullptr;
  const size_t bb = total * sizeof(int32_t);
  const size_t wb = total * elem_bytes;
  const size_t ob = buckets * elem_bytes;
  if (aclrtMalloc(&d_bins, bb, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return r;
  if (aclrtMalloc(&d_w, wb, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return r;
  if (aclrtMalloc(&d_out, ob, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return r;
  aclrtMemcpy(d_bins, bb, packed.data(), bb, ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(d_w, wb, weights, wb, ACL_MEMCPY_HOST_TO_DEVICE);

  aclTensor* t_self = Make1D(d_bins, total, ACL_INT32);
  aclTensor* t_w = Make1D(d_w, total, dt);
  aclTensor* t_out = Make1D(d_out, buckets, dt);

  double best = 1e18;
  bool ok = false;
  for (int i = 0; i < 3; ++i) {
    uint64_t ws_size = 0;
    aclOpExecutor* exec = nullptr;
    if (aclnnBincountGetWorkspaceSize(t_self, t_w, buckets, t_out, &ws_size, &exec) != ACL_SUCCESS) break;
    r.ws = ws_size;
    void* ws = nullptr;
    if (ws_size > 0 && aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) break;
    auto t0 = std::chrono::high_resolution_clock::now();
    const aclnnStatus st = aclnnBincount(ws, ws_size, exec, stream);
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ws) aclrtFree(ws);
    if (st != ACL_SUCCESS) break;
    ok = true;
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  if (ok) {
    aclrtMemcpy(out_host, ob, d_out, ob, ACL_MEMCPY_DEVICE_TO_HOST);
    r.supported = true;
    r.ms = best;
  }
  aclDestroyTensor(t_self);
  aclDestroyTensor(t_w);
  aclDestroyTensor(t_out);
  aclrtFree(d_bins);
  aclrtFree(d_w);
  aclrtFree(d_out);
  aclrtDestroyStream(stream);
  return r;
}

int main() {
  const int64_t rows = 300000, num_feat = 158, num_bins = 255;
  const int64_t total = rows * num_feat, buckets = num_feat * num_bins;

  std::vector<int32_t> packed(total);
  std::vector<float> gf(total);
  std::vector<int32_t> gi32(total);
  std::vector<int64_t> gi64(total);

  // Quantization matching LightGBM's quantized-gradient idea: map float gradient
  // onto a fixed integer grid so accumulation is exact.
  const double kScale = 1 << 20;
  std::mt19937 rng(13);
  std::uniform_int_distribution<int32_t> bd(0, static_cast<int32_t>(num_bins) - 1);
  std::uniform_real_distribution<float> gd(-1.0f, 1.0f);
  for (int64_t f = 0; f < num_feat; ++f) {
    const int64_t base = f * num_bins;
    for (int64_t i = 0; i < rows; ++i) {
      const int64_t k = f * rows + i;
      packed[k] = static_cast<int32_t>(base + bd(rng));
      gf[k] = gd(rng);
      gi32[k] = static_cast<int32_t>(std::lround(gf[k] * 1024.0));  // small scale to avoid int32 overflow
      gi64[k] = static_cast<int64_t>(std::llround(static_cast<double>(gf[k]) * kScale));
    }
  }

  // Exact references computed on host.
  std::vector<double> ref_f32(buckets, 0.0);     // exact sum of the fp32 gradient values
  std::vector<int64_t> ref_i32(buckets, 0);
  std::vector<int64_t> ref_i64(buckets, 0);
  for (int64_t i = 0; i < total; ++i) {
    ref_f32[packed[i]] += static_cast<double>(gf[i]);
    ref_i32[packed[i]] += gi32[i];
    ref_i64[packed[i]] += gi64[i];
  }

  aclInit(nullptr);
  aclrtSetDevice(0);
  std::printf("rows=%lld entries=%lld buckets=%lld\n\n",
              (long long)rows, (long long)total, (long long)buckets);

  // --- INT32 weights ---
  std::vector<int32_t> o32(buckets, 0);
  Result r32 = Run(packed, buckets, ACL_INT32, 4, gi32.data(), o32.data());
  std::printf("[INT32 weights] ");
  if (r32.supported) {
    int64_t bad = 0, maxdiff = 0;
    for (int64_t b = 0; b < buckets; ++b) {
      const int64_t d = std::llabs(static_cast<int64_t>(o32[b]) - ref_i32[b]);
      if (d != 0) ++bad;
      maxdiff = std::max(maxdiff, d);
    }
    std::printf("SUPPORTED %.2f ms  ws=%llu  mismatched_buckets=%lld max_abs_diff=%lld %s\n",
                r32.ms, (unsigned long long)r32.ws, (long long)bad, (long long)maxdiff,
                bad == 0 ? "EXACT" : "INEXACT");
  } else {
    std::printf("NOT SUPPORTED\n");
  }

  // --- INT64 weights ---
  std::vector<int64_t> o64(buckets, 0);
  Result r64 = Run(packed, buckets, ACL_INT64, 8, gi64.data(), o64.data());
  std::printf("[INT64 weights] ");
  if (r64.supported) {
    int64_t bad = 0;
    for (int64_t b = 0; b < buckets; ++b) if (o64[b] != ref_i64[b]) ++bad;
    std::printf("SUPPORTED %.2f ms  ws=%llu  mismatched_buckets=%lld %s\n",
                r64.ms, (unsigned long long)r64.ws, (long long)bad, bad == 0 ? "EXACT" : "INEXACT");
  } else {
    std::printf("NOT SUPPORTED\n");
  }

  // --- FP32 weights, for the precision/speed tradeoff record ---
  std::vector<float> of(buckets, 0.0f);
  Result rf = Run(packed, buckets, ACL_FLOAT, 4, gf.data(), of.data());
  std::printf("[FP32  weights] ");
  if (rf.supported) {
    double maxrel = 0;
    for (int64_t b = 0; b < buckets; ++b) {
      const double den = std::fabs(ref_f32[b]) > 1e-12 ? std::fabs(ref_f32[b]) : 1.0;
      maxrel = std::max(maxrel, std::fabs(static_cast<double>(of[b]) - ref_f32[b]) / den);
    }
    std::printf("SUPPORTED %.2f ms  ws=%llu  max_rel_err=%.3e\n",
                rf.ms, (unsigned long long)rf.ws, maxrel);
  } else {
    std::printf("NOT SUPPORTED\n");
  }

  std::printf("\nint32 overflow headroom check: max |bucket sum| = ");
  int64_t mx = 0;
  for (int64_t b = 0; b < buckets; ++b) {
    const int64_t a = static_cast<int64_t>(std::llabs(ref_i32[b]));
    if (a > mx) mx = a;
  }
  std::printf("%lld (int32 limit 2147483647)\n", (long long)mx);

  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}