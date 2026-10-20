/* Copyright (c) 2026. Licensed under the MIT License. */
// Lifecycle regression for process-global ACL state and resident score buffers.
#include "npu_runtime.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void Configure(LightGBM::NPURuntime* runtime) {
  runtime->SetHistogramBins({{0, 1}}, {0}, {2});
  std::vector<float> gradients{1.0f, -1.0f};
  std::vector<float> hessians{1.0f, 1.0f};
  runtime->SetHistogramGradients(gradients.data(), hessians.data(), 2, false);
}

void Update(LightGBM::NPURuntime* runtime, double* scores) {
  auto rows = runtime->UploadPartitionRows(std::vector<int32_t>{0, 1}.data(), 2);
  runtime->AddPredictionToScore({{rows, true, 0.25}}, 2, scores);
}

void CheckScores(const double* scores, double first, double second) {
  if (std::fabs(scores[0] - first) > 1e-12 || std::fabs(scores[1] - second) > 1e-12) {
    throw std::runtime_error("score buffer lifecycle value mismatch");
  }
}

}  // namespace

int main() {
  try {
    // Two live runtimes on one device exercise ACL process-level reference
    // counting and context switching while both own device allocations.
    LightGBM::NPURuntime first(0);
    LightGBM::NPURuntime second(0);
    Configure(&first);
    Configure(&second);

    double first_scores[2] = {1.0, 2.0};
    double second_scores[2] = {-1.0, 4.0};
    Update(&first, first_scores);
    Update(&second, second_scores);
    CheckScores(first_scores, 1.25, 2.25);
    CheckScores(second_scores, -0.75, 4.25);
    if (first.score_uploads() != 1 || second.score_uploads() != 1) {
      throw std::runtime_error("initial score upload count mismatch");
    }

    // A second call with unchanged host scores reuses the resident allocation.
    Update(&first, first_scores);
    CheckScores(first_scores, 1.5, 2.5);
    if (first.score_uploads() != 1 || first.score_downloads() != 2) {
      throw std::runtime_error("resident score buffer was not reused");
    }

    // Explicit reset forces the next call to upload the current host values.
    first.ResetScoreBuffers();
    Update(&first, first_scores);
    CheckScores(first_scores, 1.75, 2.75);
    if (first.score_uploads() != 2) {
      throw std::runtime_error("score buffer reset did not invalidate allocation");
    }

    // Dataset replacement also invalidates score buffers before old device
    // allocations are released by the runtime.
    Configure(&first);
    Update(&first, first_scores);
    CheckScores(first_scores, 2.0, 3.0);
    if (first.score_uploads() != 3) {
      throw std::runtime_error("dataset reset did not invalidate score allocation");
    }

    // A host-side score mutation, as performed by bagging/out-of-bag logic,
    // must invalidate the resident copy before applying the next update.
    first_scores[0] += 10.0;
    Update(&first, first_scores);
    CheckScores(first_scores, 12.25, 3.25);
    if (first.score_uploads() != 4) {
      throw std::runtime_error("external host score mutation was not uploaded");
    }

    std::cout << "PASS: multiple NPU runtimes, context reuse, score reset, host mutation and dataset reset\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}