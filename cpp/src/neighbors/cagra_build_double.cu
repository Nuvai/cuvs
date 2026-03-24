/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

// NOTE: build_knn_graph with ivf_pq_params is NOT instantiated for double because
// ivf_pq::build does not support double-precision data. Only the build() overloads
// are instantiated here.

#define RAFT_INST_CAGRA_BUILD(T, IdxT)                                                    \
  auto build(raft::resources const& handle,                                               \
             const cuvs::neighbors::cagra::index_params& params,                          \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)         \
    -> cuvs::neighbors::cagra::index<T, IdxT>                                             \
  {                                                                                       \
    return cuvs::neighbors::cagra::build<T, IdxT>(handle, params, dataset);               \
  }                                                                                       \
                                                                                          \
  auto build(raft::resources const& handle,                                               \
             const cuvs::neighbors::cagra::index_params& params,                          \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)           \
    -> cuvs::neighbors::cagra::index<T, IdxT>                                             \
  {                                                                                       \
    return cuvs::neighbors::cagra::build<T, IdxT>(handle, params, dataset);               \
  }

RAFT_INST_CAGRA_BUILD(double, uint32_t);

#undef RAFT_INST_CAGRA_BUILD

}  // namespace cuvs::neighbors::cagra
