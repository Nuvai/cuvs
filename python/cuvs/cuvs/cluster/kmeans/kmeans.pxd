#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensorVersioned
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/cluster/kmeans.h" nogil:
    ctypedef enum cuvsKMeansInitMethod:
        KMeansPlusPlus
        Random
        Array

    ctypedef enum cuvsKMeansType:
        CUVS_KMEANS_TYPE_KMEANS
        CUVS_KMEANS_TYPE_KMEANS_BALANCED

    ctypedef struct cuvsKMeansParams:
        cuvsDistanceType metric,
        int n_clusters,
        cuvsKMeansInitMethod init,
        int max_iter,
        double tol,
        int n_init,
        double oversampling_factor,
        int batch_samples,
        int batch_centroids,
        bool inertia_check,
        bool hierarchical,
        int hierarchical_n_iters

    ctypedef cuvsKMeansParams* cuvsKMeansParams_t

    cuvsError_t cuvsKMeansParamsCreate(cuvsKMeansParams_t* index)

    cuvsError_t cuvsKMeansParamsDestroy(cuvsKMeansParams_t index)

    cuvsError_t cuvsKMeansFit(cuvsResources_t res,
                              cuvsKMeansParams_t params,
                              DLManagedTensorVersioned* X,
                              DLManagedTensorVersioned* sample_weight,
                              DLManagedTensorVersioned * centroids,
                              double * inertia,
                              int * n_iter) except +

    cuvsError_t cuvsKMeansPredict(cuvsResources_t res,
                                  cuvsKMeansParams_t params,
                                  DLManagedTensorVersioned* X,
                                  DLManagedTensorVersioned* sample_weight,
                                  DLManagedTensorVersioned * centroids,
                                  DLManagedTensorVersioned * labels,
                                  bool normalize_weight,
                                  double * inertia)

    cuvsError_t cuvsKMeansClusterCost(cuvsResources_t res,
                                      DLManagedTensorVersioned* X,
                                      DLManagedTensorVersioned* centroids,
                                      double* cost)
