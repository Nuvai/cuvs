/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup filters Filters APIs
 * @brief APIs related to filter functionality.
 * @{
 */

/**
 * @brief Enum to denote filter type.
 */
enum cuvsFilterType {
  /* No filter */
  CUVS_FILTER_NONE   = 0,
  /* Filter an index with a bitset */
  CUVS_FILTER_BITSET = 1,
  /* Filter an index with a bitmap */
  CUVS_FILTER_BITMAP = 2
};

/* Backward-compatible aliases — opt in with CUVS_ENABLE_DEPRECATED_ENUM_ALIASES */
#ifdef CUVS_ENABLE_DEPRECATED_ENUM_ALIASES
#define NO_FILTER CUVS_FILTER_NONE
#define BITSET    CUVS_FILTER_BITSET
#define BITMAP    CUVS_FILTER_BITMAP
#endif

/**
 * @brief Struct to hold address of cuvs::neighbors::prefilter and its type
 *
 */
typedef struct {
  uintptr_t addr;
  enum cuvsFilterType type;
} cuvsFilter;

/**
 * @}
 */

/**
 * @defgroup index_merge Index Merge
 * @brief Common definitions related to index merging.
 * @{
 */

/**
 * @brief Strategy for merging indices.
 */
typedef enum {
  CUVS_MERGE_STRATEGY_PHYSICAL = 0,  ///< Merge indices physically
  CUVS_MERGE_STRATEGY_LOGICAL  = 1   ///< Merge indices logically
} cuvsMergeStrategy;

/* Backward-compatible aliases — opt in with CUVS_ENABLE_DEPRECATED_ENUM_ALIASES */
#ifdef CUVS_ENABLE_DEPRECATED_ENUM_ALIASES
#define MERGE_STRATEGY_PHYSICAL CUVS_MERGE_STRATEGY_PHYSICAL
#define MERGE_STRATEGY_LOGICAL  CUVS_MERGE_STRATEGY_LOGICAL
#endif

/**
 * @}
 */
#ifdef __cplusplus
}
#endif
