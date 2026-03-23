/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

#include <cmath>

namespace cuvs::neighbors::cagra {

void search_with_reranking(
  raft::resources const& res,
  const search_params& params,
  const index<float, uint32_t>& idx,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  raft::device_matrix_view<const float, int64_t, raft::row_major> original_dataset,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::distance::DistanceType metric,
  float oversample_factor)
{
  auto n_queries = queries.extent(0);
  auto k         = neighbors.extent(1);
  RAFT_EXPECTS(oversample_factor >= 1.0f, "oversample_factor must be >= 1.0");
  RAFT_EXPECTS(k > 0, "k must be > 0");

  auto k_oversample =
    static_cast<int64_t>(std::ceil(static_cast<float>(k) * oversample_factor));
  // Clamp to dataset size
  if (k_oversample > static_cast<int64_t>(idx.size())) {
    k_oversample = static_cast<int64_t>(idx.size());
  }
  // Ensure at least k candidates for refine
  if (k_oversample < k) { k_oversample = k; }

  // Phase 1: Approximate search with oversampled k
  auto candidates =
    raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k_oversample);
  auto approx_distances =
    raft::make_device_matrix<float, int64_t>(res, n_queries, k_oversample);
  search(res, params, idx, queries, candidates.view(), approx_distances.view());

  // Widen candidate indices uint32_t → int64_t for refine()
  auto candidates_i64 =
    raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k_oversample);
  {
    auto src_view = raft::make_device_vector_view<const uint32_t, int64_t>(
      candidates.data_handle(), n_queries * k_oversample);
    auto dst_view = raft::make_device_vector_view<int64_t, int64_t>(
      candidates_i64.data_handle(), n_queries * k_oversample);
    raft::linalg::map(res, dst_view, raft::cast_op<int64_t>{}, src_view);
  }

  // Phase 2: Refine — compute exact float distances against original dataset, return top-k.
  // Always runs (even when k_oversample == k) to guarantee exact distances in the output.
  cuvs::neighbors::refine(res,
                           original_dataset,
                           queries,
                           raft::make_const_mdspan(candidates_i64.view()),
                           neighbors,
                           distances,
                           metric);
}

}  // namespace cuvs::neighbors::cagra
