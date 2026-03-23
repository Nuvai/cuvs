/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <memory>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/vamana.h>
#include <cuvs/neighbors/vamana.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

template <typename T>
void* _build(cuvsResources_t res, cuvsVamanaIndexParams* params, DLManagedTensorVersioned* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto dataset = dataset_tensor->dl_tensor;
  cuvs::neighbors::vamana::index_params index_params;
  index_params.metric            = static_cast<cuvs::distance::DistanceType>((int)params->metric);
  index_params.graph_degree      = params->graph_degree;
  index_params.visited_size      = params->visited_size;
  index_params.vamana_iters      = params->vamana_iters;
  index_params.alpha             = params->alpha;
  index_params.max_fraction      = params->max_fraction;
  index_params.batch_base        = params->batch_base;
  index_params.queue_size        = params->queue_size;
  index_params.reverse_batchsize = params->reverse_batchsize;
  auto index = std::make_unique<cuvs::neighbors::vamana::index<T, uint32_t>>(*res_ptr);

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *index            = cuvs::neighbors::vamana::build(*res_ptr, index_params, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *index            = cuvs::neighbors::vamana::build(*res_ptr, index_params, mds);
  }

  return index.release();
}

template <typename T>
void _serialize(cuvsResources_t res,
                const char* filename,
                cuvsVamanaIndex_t index,
                bool include_dataset)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::vamana::index<T, uint32_t>*>(index->addr);

  cuvs::neighbors::vamana::serialize(*res_ptr, std::string(filename), *index_ptr, include_dataset);
}

template <typename T>
void _deserialize(cuvsResources_t res, const char* filename, cuvsVamanaIndex_t index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = std::make_unique<cuvs::neighbors::vamana::index<T, uint32_t>>(*res_ptr);

  cuvs::neighbors::vamana::deserialize(*res_ptr, std::string(filename), index_ptr.get());

  index->addr = reinterpret_cast<uintptr_t>(index_ptr.release());
}

}  // namespace

extern "C" cuvsError_t cuvsVamanaIndexCreate(cuvsVamanaIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] {
    *index          = new cuvsVamanaIndex{};
    (*index)->addr  = 0;
    (*index)->dtype = {};
  });
}

extern "C" cuvsError_t cuvsVamanaIndexDestroy(cuvsVamanaIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if (index.addr != 0) {
      if (index.dtype.code == kDLFloat && index.dtype.bits == 32) {
        delete reinterpret_cast<cuvs::neighbors::vamana::index<float, uint32_t>*>(index.addr);
      } else if (index.dtype.code == kDLInt && index.dtype.bits == 8) {
        delete reinterpret_cast<cuvs::neighbors::vamana::index<int8_t, uint32_t>*>(index.addr);
      } else if (index.dtype.code == kDLUInt && index.dtype.bits == 8) {
        delete reinterpret_cast<cuvs::neighbors::vamana::index<uint8_t, uint32_t>*>(index.addr);
      }
    }

    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsVamanaIndexGetDims(cuvsVamanaIndex_t index, int* dim)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index->addr != 0, "index has not been built or deserialized");
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::vamana::index<float, uint32_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::vamana::index<int8_t, uint32_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::vamana::index<uint8_t, uint32_t>*>(index->addr);
      *dim = index_ptr->dim();
    }
  });
}

extern "C" cuvsError_t cuvsVamanaBuild(cuvsResources_t res,
                                       cuvsVamanaIndexParams_t params,
                                       DLManagedTensorVersioned* dataset_tensor,
                                       cuvsVamanaIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;

    void* addr = nullptr;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      addr = _build<float>(res, params, dataset_tensor);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      addr = _build<int8_t>(res, params, dataset_tensor);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      addr = _build<uint8_t>(res, params, dataset_tensor);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
    // Commit both fields only after successful build
    index->dtype = dataset.dtype;
    index->addr  = reinterpret_cast<uintptr_t>(addr);
  });
}

extern "C" cuvsError_t cuvsVamanaSerialize(cuvsResources_t res,
                                           const char* filename,
                                           cuvsVamanaIndex_t index,
                                           bool include_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize<float>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize<int8_t>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize<uint8_t>(res, filename, index, include_dataset);
    } else {
      RAFT_FAIL(
        "Unsupported index DLtensor dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsVamanaIndexParamsCreate(cuvsVamanaIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params                      = new cuvsVamanaIndexParams{};
    (*params)->metric            = CUVS_DISTANCE_L2_EXPANDED;
    (*params)->graph_degree      = 32;
    (*params)->visited_size      = 64;
    (*params)->vamana_iters      = 1;
    (*params)->alpha             = 1.2f;
    (*params)->max_fraction      = 0.06f;
    (*params)->batch_base        = 2.0f;
    (*params)->queue_size        = 127;
    (*params)->reverse_batchsize = 1000000;
  });
}

extern "C" cuvsError_t cuvsVamanaIndexParamsDestroy(cuvsVamanaIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsVamanaDeserialize(cuvsResources_t res,
                                              const char* filename,
                                              cuvsVamanaIndex_t index,
                                              DLDataType dtype)
{
  return cuvs::core::translate_exceptions([=] {
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      _deserialize<float>(res, filename, index);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      _deserialize<int8_t>(res, filename, index);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      _deserialize<uint8_t>(res, filename, index);
    } else {
      RAFT_FAIL("Unsupported DLtensor dtype: %d and bits: %d", dtype.code, dtype.bits);
    }
    // Commit dtype only after successful deserialize (addr already set by _deserialize)
    index->dtype = dtype;
  });
}
