/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** enum to tell how to compute distance */
typedef enum {

  /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
  CUVS_DISTANCE_L2_EXPANDED = 0,
  /** same as above, but inside the epilogue, perform square root operation */
  CUVS_DISTANCE_L2_SQRT_EXPANDED = 1,
  /** cosine distance */
  CUVS_DISTANCE_COSINE_EXPANDED = 2,
  /** L1 distance */
  CUVS_DISTANCE_L1 = 3,
  /** evaluate as dist_ij += (x_ik - y-jk)^2 */
  CUVS_DISTANCE_L2_UNEXPANDED = 4,
  /** same as above, but inside the epilogue, perform square root operation */
  CUVS_DISTANCE_L2_SQRT_UNEXPANDED = 5,
  /** basic inner product **/
  CUVS_DISTANCE_INNER_PRODUCT = 6,
  /** Chebyshev (Linf) distance **/
  CUVS_DISTANCE_LINF = 7,
  /** Canberra distance **/
  CUVS_DISTANCE_CANBERRA = 8,
  /** Generalized Minkowski distance **/
  CUVS_DISTANCE_LP_UNEXPANDED = 9,
  /** Correlation distance **/
  CUVS_DISTANCE_CORRELATION_EXPANDED = 10,
  /** Jaccard distance **/
  CUVS_DISTANCE_JACCARD_EXPANDED = 11,
  /** Hellinger distance **/
  CUVS_DISTANCE_HELLINGER_EXPANDED = 12,
  /** Haversine distance **/
  CUVS_DISTANCE_HAVERSINE = 13,
  /** Bray-Curtis distance **/
  CUVS_DISTANCE_BRAY_CURTIS = 14,
  /** Jensen-Shannon distance**/
  CUVS_DISTANCE_JENSEN_SHANNON = 15,
  /** Hamming distance **/
  CUVS_DISTANCE_HAMMING_UNEXPANDED = 16,
  /** KLDivergence **/
  CUVS_DISTANCE_KL_DIVERGENCE = 17,
  /** RusselRao **/
  CUVS_DISTANCE_RUSSEL_RAO_EXPANDED = 18,
  /** Dice-Sorensen distance **/
  CUVS_DISTANCE_DICE_EXPANDED = 19,
  /** Bitstring Hamming distance **/
  CUVS_DISTANCE_BITWISE_HAMMING = 20,
  /** Precomputed (special value) **/
  CUVS_DISTANCE_PRECOMPUTED = 100
} cuvsDistanceType;

/* Backward-compatible aliases — opt in with CUVS_ENABLE_DEPRECATED_ENUM_ALIASES */
#ifdef CUVS_ENABLE_DEPRECATED_ENUM_ALIASES
#define L2Expanded            CUVS_DISTANCE_L2_EXPANDED
#define L2SqrtExpanded        CUVS_DISTANCE_L2_SQRT_EXPANDED
#define CosineExpanded        CUVS_DISTANCE_COSINE_EXPANDED
#define L1                    CUVS_DISTANCE_L1
#define L2Unexpanded          CUVS_DISTANCE_L2_UNEXPANDED
#define L2SqrtUnexpanded      CUVS_DISTANCE_L2_SQRT_UNEXPANDED
#define InnerProduct          CUVS_DISTANCE_INNER_PRODUCT
#define Linf                  CUVS_DISTANCE_LINF
#define Canberra              CUVS_DISTANCE_CANBERRA
#define LpUnexpanded          CUVS_DISTANCE_LP_UNEXPANDED
#define CorrelationExpanded   CUVS_DISTANCE_CORRELATION_EXPANDED
#define JaccardExpanded       CUVS_DISTANCE_JACCARD_EXPANDED
#define HellingerExpanded     CUVS_DISTANCE_HELLINGER_EXPANDED
#define Haversine             CUVS_DISTANCE_HAVERSINE
#define BrayCurtis            CUVS_DISTANCE_BRAY_CURTIS
#define JensenShannon         CUVS_DISTANCE_JENSEN_SHANNON
#define HammingUnexpanded     CUVS_DISTANCE_HAMMING_UNEXPANDED
#define KLDivergence          CUVS_DISTANCE_KL_DIVERGENCE
#define RusselRaoExpanded     CUVS_DISTANCE_RUSSEL_RAO_EXPANDED
#define DiceExpanded          CUVS_DISTANCE_DICE_EXPANDED
#define BitwiseHamming        CUVS_DISTANCE_BITWISE_HAMMING
#define Precomputed           CUVS_DISTANCE_PRECOMPUTED
#endif

#ifdef __cplusplus
}
#endif
