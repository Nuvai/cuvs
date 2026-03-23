/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_distance.hpp"

#include <cuvs/distance/distance.hpp>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct binary_adc_descriptor_spec : public instance_spec<DataT, IndexT, DistanceT> {
  using base_type = instance_spec<DataT, IndexT, DistanceT>;
  using typename base_type::data_type;
  using typename base_type::distance_type;
  using typename base_type::host_type;
  using typename base_type::index_type;

  template <typename DatasetT>
  constexpr static inline auto accepts_dataset()
    -> std::enable_if_t<is_binary_dataset_v<DatasetT>, bool>
  {
    return true;
  }

  template <typename DatasetT>
  constexpr static inline auto accepts_dataset()
    -> std::enable_if_t<!is_binary_dataset_v<DatasetT>, bool>
  {
    return false;
  }

  template <typename DatasetT>
  static auto init(const cagra::search_params& params,
                   const DatasetT& dataset,
                   cuvs::distance::DistanceType metric,
                   const DistanceT* dataset_norms = nullptr) -> host_type
  {
    return init_(params,
                 dataset.data_handle(),
                 dataset.packed_dim(),
                 IndexT(dataset.n_rows()),
                 dataset.dim());
  }

  template <typename DatasetT>
  static auto priority(const cagra::search_params& params,
                       const DatasetT& dataset,
                       cuvs::distance::DistanceType metric) -> double
  {
    // If explicit team_size is specified and doesn't match the instance, discard it
    if (params.team_size != 0 && TeamSize != params.team_size) { return -1.0; }
    // Binary ADC computes -dot(query, binary_vec). This is a correct ranking proxy for:
    //   - L2Expanded:    ||q-b||^2 = sum(q^2) - 2*dot(q,b) + popcount(b).
    //                    sum(q^2) is per-query constant; popcount(b) is dropped (small noise).
    //                    Ranking by -dot(q,b) ≈ ranking by L2. Postprocess: no-op for float.
    //   - InnerProduct:  kernel returns -dot; postprocess multiplies by -1 → user gets +dot.
    //
    // Reject everything else:
    //   - CosineExpanded: requires dataset norms (binary_dataset has none).
    //   - BitwiseHamming: expects uint8 queries and symmetric Hamming, not float ADC.
    //   - L1:             true L1 = sum(|q-b|), NOT a dot product. Would give wrong rankings.
    //   - L2Sqrt*:        postprocess takes sqrt() of our negative output → NaN.
    if (metric != cuvs::distance::DistanceType::L2Expanded &&
        metric != cuvs::distance::DistanceType::InnerProduct) {
      return -1.0;
    }
    // Favor the closest dimensionality for kernel unrolling efficiency.
    return 1.0 / (0.1 + std::abs(double(dataset.dim()) - double(DatasetBlockDim)));
  }

 private:
  static dataset_descriptor_host<DataT, IndexT, DistanceT> init_(
    const cagra::search_params& params,
    const uint8_t* packed_data_ptr,
    uint32_t packed_dim,
    IndexT size,
    uint32_t dim);
};

}  // namespace cuvs::neighbors::cagra::detail
