/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/interop.hpp"

namespace cuvs::core {

/**
 * @defgroup interop Interoperability between `mdspan` and `DLManagedTensorVersioned`
 * @{
 */

// ---------------------------------------------------------------------------
// Device / host compatibility checks
// ---------------------------------------------------------------------------

/**
 * @brief Check if DLTensor has device accessible memory.
 *        This function returns true for `DLDeviceType` of values
 *        `kDLCUDA`, `kDLCUDAHost`, or `kDLCUDAManaged`
 *
 * @param[in] tensor DLTensor object to check underlying memory type
 * @return bool
 */
inline bool is_dlpack_device_compatible(DLTensor tensor)
{
  return detail::is_dlpack_device_compatible(tensor);
}

/**
 * @brief Check if DLTensor has host accessible memory.
 *        This function returns true for `DLDeviceType` of values
 *        `kDLCPU`, `kDLCUDAHost`, or `kDLCUDAManaged`
 *
 * @param tensor DLTensor object to check underlying memory type
 * @return bool
 */
inline bool is_dlpack_host_compatible(DLTensor tensor)
{
  return detail::is_dlpack_host_compatible(tensor);
}

// ---------------------------------------------------------------------------
// Contiguity checks
// ---------------------------------------------------------------------------

/**
 * @brief Check if DLManagedTensorVersioned has a row-major (c-contiguous) layout
 *
 * @param tensor DLManagedTensorVersioned object to check
 * @return bool
 */
inline bool is_c_contiguous(DLManagedTensorVersioned* tensor)
{
  return detail::is_c_contiguous(tensor);
}

/**
 * @brief Check if DLManagedTensorVersioned has a col-major (f-contiguous) layout
 *
 * @param tensor DLManagedTensorVersioned object to check
 * @return bool
 */
inline bool is_f_contiguous(DLManagedTensorVersioned* tensor)
{
  return detail::is_f_contiguous(tensor);
}

// ---------------------------------------------------------------------------
// from_dlpack
// ---------------------------------------------------------------------------

/**
 * @brief Convert a DLManagedTensorVersioned to a mdspan
 * NOTE: This function only supports compact row-major and col-major layouts.
 *
 * @tparam MdspanType
 * @param[in] managed_tensor
 * @return MdspanType
 */
template <typename MdspanType, typename = raft::is_mdspan_t<MdspanType>>
inline MdspanType from_dlpack(DLManagedTensorVersioned* managed_tensor)
{
  return detail::from_dlpack<MdspanType>(managed_tensor);
}

// ---------------------------------------------------------------------------
// to_dlpack
// ---------------------------------------------------------------------------

/**
 * @brief Convert a mdspan to a DLManagedTensorVersioned
 *
 * Converts a mdspan to a DLManagedTensorVersioned object with version and flags set.
 * This lets us pass non-owning views from C++ to C code without copying.
 * Note that the returned tensor is a non-owning view, and doesn't ensure
 * that the underlying memory stays valid.
 */
template <typename MdspanType, typename = raft::is_mdspan_t<MdspanType>>
void to_dlpack(MdspanType src, DLManagedTensorVersioned* dst)
{
  return detail::to_dlpack(src, dst);
}

/**
 * @}
 */

}  // namespace cuvs::core
