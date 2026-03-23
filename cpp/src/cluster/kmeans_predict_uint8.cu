/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>

#include <thrust/transform.h>

namespace cuvs::cluster::kmeans {

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const uint8_t, int64_t> X,
             std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia)
{
  auto X_float = raft::make_device_matrix<float, int64_t>(handle, X.extent(0), X.extent(1));
  thrust::transform(raft::resource::get_thrust_policy(handle),
                    X.data_handle(),
                    X.data_handle() + X.size(),
                    X_float.data_handle(),
                    [] __device__(uint8_t v) { return static_cast<float>(v); });
  cuvs::cluster::kmeans::predict(
    handle, params, raft::make_const_mdspan(X_float.view()), sample_weight, centroids, labels, normalize_weight, inertia);
}

}  // namespace cuvs::cluster::kmeans
