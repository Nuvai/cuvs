/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuda_bf16.h>
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {
CUVS_INST_CAGRA_MERGE(nv_bfloat16, uint32_t);
}  // namespace cuvs::neighbors::cagra
