/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_LEAF_SPLITS_HPP_
#define LIGHTGBM_SRC_TREELEARNER_LEAF_SPLITS_HPP_

#include <LightGBM/config.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>

#include <limits>
#include <vector>

#include "data_partition.hpp"

namespace LightGBM {

/*!
* \brief used to find split candidates for a leaf
*/
class LeafSplits {
 public:
  LeafSplits(data_size_t num_data, const Config* config)
      : deterministic_(false),
        num_data_in_leaf_(num_data),
        num_data_(num_data),
    data_indices_(nullptr), weight_(0) {
    if (config != nullptr) {
      deterministic_ = config->deterministic;
    }
  }
  void ResetNumData(data_size_t num_data) {
    num_data_ = num_data;
    num_data_in_leaf_ = num_data;
  }
  ~LeafSplits() {
  }

  // Initialize device-backed leaf statistics without retaining a host index
  // pointer. The backend supplies row membership to histogram construction.
  void InitFromDevice(int leaf, data_size_t count, double sum_gradients,
                      double sum_hessians, double weight) {
    leaf_index_ = leaf;
    num_data_in_leaf_ = count;
    data_indices_ = nullptr;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
    weight_ = weight;
    int_sum_gradients_and_hessians_ = 0;
  }
  void InitQuantizedFromDevice(int leaf, data_size_t count, double sum_gradients,
                               double sum_hessians, int64_t packed, double weight) {
    InitFromDevice(leaf, count, sum_gradients, sum_hessians, weight);
    int_sum_gradients_and_hessians_ = packed;
  }

  /*!
  * \brief Init split on current leaf on partial data.
  * \param leaf Index of current leaf
  * \param data_partition current data partition
  * \param sum_gradients
  * \param sum_hessians
  * \param weight
  */
  void Init(int leaf, const DataPartition* data_partition, double sum_gradients,
            double sum_hessians, double weight) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
    weight_ = weight;
  }

  /*!
  * \brief Init split on current leaf on partial data.
  * \param leaf Index of current leaf
  * \param data_partition current data partition
  * \param sum_gradients
  * \param sum_hessians
  * \param sum_gradients_and_hessians
  * \param weight
  */
  void Init(int leaf, const DataPartition* data_partition, double sum_gradients,
            double sum_hessians, int64_t sum_gradients_and_hessians, double weight) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
    int_sum_gradients_and_hessians_ = sum_gradients_and_hessians;
    weight_ = weight;
  }

  /*!
  * \brief Init split on current leaf on partial data.
  * \param leaf Index of current leaf
  * \param sum_gradients
  * \param sum_hessians
  */
  void Init(int leaf, double sum_gradients, double sum_hessians) {
    leaf_index_ = leaf;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
  }

  /*!
   * \brief Init splits on the current leaf, it will traverse all data to sum up the results
   * \param gradients
   * \param hessians
   */
  void Init(const score_t* gradients, const score_t* hessians) {
    num_data_in_leaf_ = num_data_;
    leaf_index_ = 0;
    data_indices_ = nullptr;
    double tmp_sum_gradients = 0.0f;
    double tmp_sum_hessians = 0.0f;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) reduction(+:tmp_sum_gradients, tmp_sum_hessians) if (num_data_in_leaf_ >= 1024 && !deterministic_)
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      tmp_sum_gradients += gradients[i];
      tmp_sum_hessians += hessians[i];
    }
    sum_gradients_ = tmp_sum_gradients;
    sum_hessians_ = tmp_sum_hessians;
  }


  /*!
   * \brief Init splits on the current leaf, it will traverse all data to sum up the results
   * \param int_gradients_and_hessians Discretized gradients and hessians
   * \param grad_scale Scaling factor to recover original gradients from discretized gradients
   * \param hess_scale Scaling factor to recover original hessians from discretized hessians
   */
  void Init(const int8_t* int_gradients_and_hessians,
    const double grad_scale, const double hess_scale) {
    num_data_in_leaf_ = num_data_;
    leaf_index_ = 0;
    data_indices_ = nullptr;
    double tmp_sum_gradients = 0.0f;
    double tmp_sum_hessians = 0.0f;
    int64_t tmp_sum_gradients_and_hessians = 0;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) reduction(+:tmp_sum_gradients, tmp_sum_hessians, tmp_sum_gradients_and_hessians) if (num_data_in_leaf_ >= 1024 && !deterministic_)
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      tmp_sum_gradients += int_gradients_and_hessians[2 * i + 1] * grad_scale;
      tmp_sum_hessians += static_cast<uint8_t>(int_gradients_and_hessians[2 * i]) * hess_scale;
      const int64_t packed_long_int_grad_and_hess =
          static_cast<int64_t>(int_gradients_and_hessians[2 * i + 1]) * int64_t{4294967296} +
          static_cast<uint8_t>(int_gradients_and_hessians[2 * i]);
      tmp_sum_gradients_and_hessians += packed_long_int_grad_and_hess;
    }
    sum_gradients_ = tmp_sum_gradients;
    sum_hessians_ = tmp_sum_hessians;
    int_sum_gradients_and_hessians_ = tmp_sum_gradients_and_hessians;
  }


  /*!
   * \brief Init splits on current leaf of partial data.
   * \param leaf Index of current leaf
   * \param data_partition current data partition
   * \param gradients
   * \param hessians
   */
  void Init(int leaf, const DataPartition* data_partition,
            const score_t* gradients, const score_t* hessians) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    double tmp_sum_gradients = 0.0f;
    double tmp_sum_hessians = 0.0f;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) reduction(+:tmp_sum_gradients, tmp_sum_hessians) if (num_data_in_leaf_ >= 1024 && !deterministic_)
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      const data_size_t idx = data_indices_[i];
      tmp_sum_gradients += gradients[idx];
      tmp_sum_hessians += hessians[idx];
    }
    sum_gradients_ = tmp_sum_gradients;
    sum_hessians_ = tmp_sum_hessians;
  }


  /*!
   * \brief Init splits on current leaf of partial data.
   * \param leaf Index of current leaf
   * \param data_partition current data partition
   * \param int_gradients_and_hessians Discretized gradients and hessians
   * \param grad_scale Scaling factor to recover original gradients from discretized gradients
   * \param hess_scale Scaling factor to recover original hessians from discretized hessians
   */
  void Init(int leaf, const DataPartition* data_partition,
            const int8_t* int_gradients_and_hessians,
            const score_t grad_scale, const score_t hess_scale) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    double tmp_sum_gradients = 0.0f;
    double tmp_sum_hessians = 0.0f;
    int64_t tmp_sum_gradients_and_hessians = 0;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) reduction(+:tmp_sum_gradients, tmp_sum_hessians, tmp_sum_gradients_and_hessians) if (num_data_in_leaf_ >= 1024 && !deterministic_)
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      const data_size_t idx = data_indices_[i];
      tmp_sum_gradients += int_gradients_and_hessians[2 * idx + 1] * grad_scale;
      tmp_sum_hessians += static_cast<uint8_t>(int_gradients_and_hessians[2 * idx]) * hess_scale;
      const int64_t packed_long_int_grad_and_hess =
          static_cast<int64_t>(int_gradients_and_hessians[2 * idx + 1]) * int64_t{4294967296} +
          static_cast<uint8_t>(int_gradients_and_hessians[2 * idx]);
      tmp_sum_gradients_and_hessians += packed_long_int_grad_and_hess;
    }
    sum_gradients_ = tmp_sum_gradients;
    sum_hessians_ = tmp_sum_hessians;
    int_sum_gradients_and_hessians_ = tmp_sum_gradients_and_hessians;
  }


  /*!
  * \brief Init splits on current leaf, only update sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  */
  void Init(double sum_gradients, double sum_hessians) {
    leaf_index_ = 0;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
  }

  /*!
  * \brief Init splits on current leaf, only update sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \param int_sum_gradients_and_hessians
  */
  void Init(double sum_gradients, double sum_hessians, int64_t int_sum_gradients_and_hessians) {
    leaf_index_ = 0;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
    int_sum_gradients_and_hessians_ = int_sum_gradients_and_hessians;
  }

  /*!
  * \brief Init splits on current leaf
  */
  void Init() {
    leaf_index_ = -1;
    data_indices_ = nullptr;
    num_data_in_leaf_ = 0;
  }


  /*! \brief Get current leaf index */
  int leaf_index() const { return leaf_index_; }

  /*! \brief Get number of data in current leaf */
  data_size_t num_data_in_leaf() const { return num_data_in_leaf_; }

  /*! \brief Get sum of gradients of current leaf */
  double sum_gradients() const { return sum_gradients_; }

  /*! \brief Get sum of Hessians of current leaf */
  double sum_hessians() const { return sum_hessians_; }

  /*! \brief Get sum of discretized gradients and Hessians of current leaf */
  int64_t int_sum_gradients_and_hessians() const { return int_sum_gradients_and_hessians_; }

  /*! \brief Get indices of data of current leaf */
  const data_size_t* data_indices() const { return data_indices_; }

  /*! \brief Get weight of current leaf */
  double weight() const { return weight_; }



 private:
  bool deterministic_;
  /*! \brief current leaf index */
  int leaf_index_;
  /*! \brief number of data on current leaf */
  data_size_t num_data_in_leaf_;
  /*! \brief number of all training data */
  data_size_t num_data_;
  /*! \brief sum of gradients of current leaf */
  double sum_gradients_;
  /*! \brief sum of Hessians of current leaf */
  double sum_hessians_;
  /*! \brief sum of discretized gradients and Hessians of current leaf */
  int64_t int_sum_gradients_and_hessians_;
  /*! \brief indices of data of current leaf */
  const data_size_t* data_indices_;
  /*! \brief weight of current leaf */
  double weight_;
};

}  // namespace LightGBM
#endif   // LIGHTGBM_SRC_TREELEARNER_LEAF_SPLITS_HPP_
