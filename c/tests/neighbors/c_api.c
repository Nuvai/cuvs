/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/all_neighbors.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/tiered_index.h>

#include <dlpack/dlpack.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_compile_cagra()
{
  // simple smoke test to make sure that we can compile the cagra.h API
  // using a c compiler. This isn't aiming to be a full test, just checking
  // that the exposed C-API is valid C code and doesn't contain C++ features
  assert(!"test_compile_cagra is not meant to be run");

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);
  cuvsCagraIndexDestroy(index);
}

void test_compile_tiered_index()
{
  // Smoke test to ensure that the tiered_index.h API compiles correctly
  // using a c compiler. Not a full test.
  assert(!"test_compile_tiered_index is not meant to be run");

  cuvsTieredIndex_t tiered_index;
  cuvsTieredIndexCreate(&tiered_index);
  cuvsTieredIndexDestroy(tiered_index);

  cuvsTieredIndexParams_t index_params;
  cuvsResources_t resources;
  cuvsFilter prefilter;
  DLManagedTensorVersioned dataset, neighbors, distances;
  memset(&dataset, 0, sizeof(dataset));
  dataset.version.major = DLPACK_MAJOR_VERSION;
  dataset.version.minor = DLPACK_MINOR_VERSION;
  dataset.flags         = 0;
  memset(&neighbors, 0, sizeof(neighbors));
  neighbors.version.major = DLPACK_MAJOR_VERSION;
  neighbors.version.minor = DLPACK_MINOR_VERSION;
  neighbors.flags         = 0;
  memset(&distances, 0, sizeof(distances));
  distances.version.major = DLPACK_MAJOR_VERSION;
  distances.version.minor = DLPACK_MINOR_VERSION;
  distances.flags         = 0;
  cuvsTieredIndexParamsCreate(&index_params);
  cuvsTieredIndexParamsDestroy(index_params);
  cuvsTieredIndexBuild(resources, index_params, &dataset, tiered_index);
  cuvsTieredIndexSearch(resources, NULL, tiered_index, &dataset, &neighbors, &distances, prefilter);
  cuvsTieredIndexExtend(resources, &dataset, tiered_index);
}

void test_compile_all_neighbors()
{
  // Smoke test to ensure that the all_neighbors.h API compiles correctly
  // using a c compiler. Not a full test.
  assert(!"test_compile_all_neighbors is not meant to be run");

  cuvsAllNeighborsIndexParams_t params;
  cuvsResources_t resources;
  DLManagedTensorVersioned dataset, indices, distances, core_distances;
  memset(&dataset, 0, sizeof(dataset));
  dataset.version.major = DLPACK_MAJOR_VERSION;
  dataset.version.minor = DLPACK_MINOR_VERSION;
  dataset.flags         = 0;
  memset(&indices, 0, sizeof(indices));
  indices.version.major = DLPACK_MAJOR_VERSION;
  indices.version.minor = DLPACK_MINOR_VERSION;
  indices.flags         = 0;
  memset(&distances, 0, sizeof(distances));
  distances.version.major = DLPACK_MAJOR_VERSION;
  distances.version.minor = DLPACK_MINOR_VERSION;
  distances.flags         = 0;
  memset(&core_distances, 0, sizeof(core_distances));
  core_distances.version.major = DLPACK_MAJOR_VERSION;
  core_distances.version.minor = DLPACK_MINOR_VERSION;
  core_distances.flags         = 0;
  cuvsAllNeighborsIndexParamsCreate(&params);
  cuvsAllNeighborsIndexParamsDestroy(params);
  cuvsAllNeighborsBuild(resources, params, &dataset, &indices, &distances, &core_distances, 1.0f);
}

int main()
{
  // These are smoke tests that check that the C-APIs compile with a C compiler.
  // These are not meant to be run.
  test_compile_cagra();
  test_compile_tiered_index();
  test_compile_all_neighbors();

  return 0;
}
