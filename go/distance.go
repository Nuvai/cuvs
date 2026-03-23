package cuvs

// #include <cuvs/distance/pairwise_distance.h>
import "C"

import (
	"errors"
	"unsafe"
)

type Distance int

// Supported distance metrics
const (
	DistanceL2 Distance = iota
	DistanceSQEuclidean
	DistanceEuclidean
	DistanceL1
	DistanceCityblock
	DistanceInnerProduct
	DistanceChebyshev
	DistanceCanberra
	DistanceCosine
	DistanceLp
	DistanceCorrelation
	DistanceJaccard
	DistanceHellinger
	DistanceBrayCurtis
	DistanceJensenShannon
	DistanceHamming
	DistanceKLDivergence
	DistanceMinkowski
	DistanceRusselRao
	DistanceDice
	DistanceBitwiseHamming
)

// Maps cuvs Go distances to C distances
var CDistances = map[Distance]int{
	DistanceL2:            C.CUVS_DISTANCE_L2_SQRT_EXPANDED,
	DistanceSQEuclidean:   C.CUVS_DISTANCE_L2_EXPANDED,
	DistanceEuclidean:     C.CUVS_DISTANCE_L2_SQRT_EXPANDED,
	DistanceL1:            C.CUVS_DISTANCE_L1,
	DistanceCityblock:     C.CUVS_DISTANCE_L1,
	DistanceInnerProduct:  C.CUVS_DISTANCE_INNER_PRODUCT,
	DistanceChebyshev:     C.CUVS_DISTANCE_LINF,
	DistanceCanberra:      C.CUVS_DISTANCE_CANBERRA,
	DistanceCosine:        C.CUVS_DISTANCE_COSINE_EXPANDED,
	DistanceLp:            C.CUVS_DISTANCE_LP_UNEXPANDED,
	DistanceCorrelation:   C.CUVS_DISTANCE_CORRELATION_EXPANDED,
	DistanceJaccard:       C.CUVS_DISTANCE_JACCARD_EXPANDED,
	DistanceHellinger:     C.CUVS_DISTANCE_HELLINGER_EXPANDED,
	DistanceBrayCurtis:    C.CUVS_DISTANCE_BRAY_CURTIS,
	DistanceJensenShannon: C.CUVS_DISTANCE_JENSEN_SHANNON,
	DistanceHamming:       C.CUVS_DISTANCE_HAMMING_UNEXPANDED,
	DistanceKLDivergence:  C.CUVS_DISTANCE_KL_DIVERGENCE,
	DistanceMinkowski:     C.CUVS_DISTANCE_LP_UNEXPANDED,
	DistanceRusselRao:     C.CUVS_DISTANCE_RUSSEL_RAO_EXPANDED,
	DistanceDice:           C.CUVS_DISTANCE_DICE_EXPANDED,
	DistanceBitwiseHamming: C.CUVS_DISTANCE_BITWISE_HAMMING,
}

// Computes the pairwise distance between two vectors.
func PairwiseDistance[T any](Resources Resource, x *Tensor[T], y *Tensor[T], distances *Tensor[float32], metric Distance, metric_arg float32) error {
	CMetric, exists := CDistances[metric]

	if !exists {
		return errors.New("cuvs: invalid distance metric")
	}

	return CheckCuvs(CuvsError(C.cuvsPairwiseDistance(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(x.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(y.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), C.cuvsDistanceType(CMetric), C.float(metric_arg))))
}
