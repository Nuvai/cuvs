/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra_serialize.cuh"

#include <cuda_bf16.h>

namespace cuvs::neighbors::cagra {

CUVS_INST_CAGRA_SERIALIZE(nv_bfloat16);

}  // namespace cuvs::neighbors::cagra
