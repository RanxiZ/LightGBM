// Phase 1 gate 2: check dtype support (fp32/fp64) and the multi-feature offset trick,
// where bin_global = feature_index * num_bins + local_bin lets one call cover all features.
#include <acl/acl.h>
#include <aclnnop/aclnn_bincount.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define CHK(expr, msg)                                               \
  do {                                                               \
    aclnnStatus _s = (expr);                                         \
    if (_s != ACL_SUCCESS) {                                         \
      std::printf("  %s -> status=%d\n", msg, static_cast<int>(_s)); \
      return _s;                                                     \
    }                                                                \
  } while (0)

namespace {

aclTensor* Make1D(void* dev, int64_t n, aclDataType dt) {
  int64_t dims[1] = {n};
  int64_t strides[1] = {1};
  return aclCreateTensor(dims, 1, dt, strides, 0, ACL_FORMAT_ND, dims, 1, dev);
}

size_t DtSize(aclDataType dt) { return dt == ACL_DOUBLE ? 8 : (dt == ACL_FLOAT ? 4 : 2); }

// Runs weighted bincount on device for the given weight dtype; returns status and timing.
template <typename WT>
aclnnStatus RunBincount(const std::vector<int32_t>& bins, const std::vector<WT>& w,
                        int64_t num_buckets, aclDataType dt, std::vector<WT>* out,
                        double* ms, uint64_t* ws_out) {
  aclrtStream stream = nullptr;
  CHK(aclrtCreateStream(&stream), "create stream");
  void *d_bins = nullptr, *d_w = nullptr, *d_out = nullptr;
  const size_t bb = bins.size() * sizeof(int32_t);
  const size_t wb = w.size() * DtSize(dt);
  const size_t ob = static_cast<size_t>(num_buckets) * DtSize(dt);
  CHK(aclrtMalloc(&d_bins, bb, ACL_MEM_MALLOC_HUGE_FIRST), "malloc bins");
  CHK(aclrtMalloc(&d_w, wb, ACL_MEM_MALLOC_HUGE_FIRST), "malloc weights");
  CHK(aclrtMalloc(&d_out, ob, ACL_MEM_MALLOC_HUGE_FIRST), "malloc out");
  CHK(aclrtMemcpy(d_bins, bb, bins.data(), bb, ACL_MEMCPY_HOST_TO_DEVICE), "copy bins");
  CHK(aclrtMemcpy(d_w, wb, w.data(), wb, ACL_MEMCPY_HOST_TO_DEVICE), "copy weights");

  aclTensor* t_self = Make1D(d_bins, static_cast<int64_t>(bins.size()), ACL_INT32);
  aclTensor* t_w = Make1D(d_w, static_cast<int64_t>(w.size()), dt);
  aclTensor* t_out = Make1D(d_out, num_buckets, dt);

  uint64_t ws_size = 0;
  aclOpExecutor* exec = nullptr;
  aclnnStatus st = aclnnBincountGetWorkspaceSize(t_self, t_w, num_buckets, t_out, &ws_size, &exec);
  if (st != ACL_SUCCESS) {
    std::printf("  GetWorkspaceSize -> status=%d\n", static_cast<int>(st));
    return st;
  }
  *ws_out = ws_size;
  void* ws = nullptr;
  if (ws_size > 0) CHK(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");

  // One warm-up pass, then timed repeats so dispatch overhead is not counted once.
  CHK(aclnnBincount(ws, ws_size, exec, stream), "aclnnBincount warmup");
  CHK(aclrtSynchronizeStream(stream), "sync warmup");
  const int reps = 5;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; ++r) {
    aclOpExecutor* e2 = nullptr;
    uint64_t w2 = 0;
    CHK(aclnnBincountGetWorkspaceSize(t_self, t_w, num_buckets, t_out, &w2, &e2), "ws repeat");
    void* ws2 = nullptr;
    if (w2 > 0) CHK(aclrtMalloc(&ws2, w2, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws repeat");
    CHK(aclnnBincount(ws2, w2, e2, stream), "aclnnBincount repeat");
    CHK(aclrtSynchronizeStream(stream), "sync repeat");
    if (ws2) aclrtFree(ws2);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  *ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

  out->assign(static_cast<size_t>(num_buckets), WT(0));
  CHK(aclrtMemcpy(out->data(), ob, d_out, ob, ACL_MEMCPY_DEVICE_TO_HOST), "copy out");

  aclDestroyTensor(t_self);
  aclDestroyTensor(t_w);
  aclDestroyTensor(t_out);
  if (ws) aclrtFree(ws);
  aclrtFree(d_bins);
  aclrtFree(d_w);
  aclrtFree(d_out);
  aclrtDestroyStream(stream);
  return ACL_SUCCESS;
}

}  // namespace

int main() {
  // Alpha158-like shape: many rows, 158 features, 255 bins per feature.
  const int64_t num_data = 400000;
  const int64_t num_feat = 158;
  const int64_t num_bins = 255;
  const int64_t total = num_data * num_feat;      // one entry per (sample, feature)
  const int64_t buckets = num_feat * num_bins;    // offset-packed bucket space

  std::printf("shape: num_data=%lld num_feat=%lld num_bins=%lld\n",
              (long long)num_data, (long long)num_feat, (long long)num_bins);
  std::printf("packed bins array = %lld entries (%.2f GB int32)\n",
              (long long)total, total * 4.0 / 1e9);

  aclInit(nullptr);
  aclrtSetDevice(0);

  // Small-scale dtype probe first: is fp64 accepted at all?
  const int64_t n_small = 50000;
  std::vector<int32_t> bins(n_small);
  std::mt19937 rng(7);
  std::uniform_int_distribution<int32_t> bd(0, static_cast<int32_t>(num_bins) - 1);
  for (auto& b : bins) b = bd(rng);
  std::vector<double> wd(n_small);
  std::vector<float> wf(n_small);
  std::uniform_real_distribution<double> gd(-1.0, 1.0);
  for (int64_t i = 0; i < n_small; ++i) {
    wd[i] = gd(rng);
    wf[i] = static_cast<float>(wd[i]);
  }
  std::vector<double> ref(num_bins, 0.0);
  for (int64_t i = 0; i < n_small; ++i) ref[bins[i]] += wd[i];

  double ms = 0;
  uint64_t ws = 0;
  std::printf("\n[dtype ACL_DOUBLE]\n");
  std::vector<double> od;
  aclnnStatus st_d = RunBincount<double>(bins, wd, num_bins, ACL_DOUBLE, &od, &ms, &ws);
  if (st_d == ACL_SUCCESS) {
    double mr = 0;
    for (int64_t b = 0; b < num_bins; ++b) {
      double den = std::fabs(ref[b]) > 1e-12 ? std::fabs(ref[b]) : 1.0;
      mr = std::max(mr, std::fabs(od[b] - ref[b]) / den);
    }
    std::printf("  SUPPORTED  max_rel_err=%.3e  %.3f ms  ws=%llu\n", mr, ms, (unsigned long long)ws);
  } else {
    std::printf("  NOT SUPPORTED\n");
  }

  std::printf("\n[dtype ACL_FLOAT]\n");
  std::vector<float> of;
  aclnnStatus st_f = RunBincount<float>(bins, wf, num_bins, ACL_FLOAT, &of, &ms, &ws);
  if (st_f == ACL_SUCCESS) {
    double mr = 0;
    for (int64_t b = 0; b < num_bins; ++b) {
      double den = std::fabs(ref[b]) > 1e-12 ? std::fabs(ref[b]) : 1.0;
      mr = std::max(mr, std::fabs(static_cast<double>(of[b]) - ref[b]) / den);
    }
    std::printf("  SUPPORTED  max_rel_err=%.3e  %.3f ms  ws=%llu\n", mr, ms, (unsigned long long)ws);
  } else {
    std::printf("  NOT SUPPORTED\n");
  }

  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}