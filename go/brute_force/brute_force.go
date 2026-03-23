package brute_force

// #include <stdlib.h>
// #include <cuvs/neighbors/brute_force.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

// Brute Force KNN Index
type BruteForceIndex struct {
	index   C.cuvsBruteForceIndex_t
	trained bool
}

// Creates a new empty Brute Force KNN Index
func CreateIndex() (*BruteForceIndex, error) {
	var index C.cuvsBruteForceIndex_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &BruteForceIndex{index: index, trained: false}, nil
}

// Destroys the Brute Force KNN Index
func (index *BruteForceIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Builds a new Brute Force KNN Index from the dataset for efficient search.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `Dataset` - A row-major matrix on either the host or device to index
// * `metric` - Distance type to use for building the index
// * `metric_arg` - Value of `p` for Minkowski distances - set to 2.0 if not applicable
func BuildIndex[T any](Resources cuvs.Resource, Dataset *cuvs.Tensor[T], metric cuvs.Distance, metric_arg float32, index *BruteForceIndex) error {
	CMetric, exists := cuvs.CDistances[metric]

	if !exists {
		return errors.New("cuvs: invalid distance metric")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceBuild(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensorVersioned)(unsafe.Pointer(Dataset.C_tensor)), C.cuvsDistanceType(CMetric), C.float(metric_arg), index.index)))
	if err != nil {
		return err
	}
	index.trained = true

	return nil
}

// Serialize a Brute Force index to file.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `index` - BruteForceIndex to serialize
// * `filename` - Path to save the index
func SerializeIndex(Resources cuvs.Resource, index *BruteForceIndex, filename string) error {
	if !index.trained {
		return errors.New("index needs to be built before calling serialize")
	}
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceSerialize(
		C.cuvsResources_t(Resources.Resource),
		cFilename,
		index.index,
	)))
}

// Deserialize a Brute Force index from file.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `filename` - Path to load the index from
func DeserializeIndex(Resources cuvs.Resource, filename string) (*BruteForceIndex, error) {
	index, err := CreateIndex()
	if err != nil {
		return nil, err
	}
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceDeserialize(
		C.cuvsResources_t(Resources.Resource),
		cFilename,
		index.index,
	)))
	if err != nil {
		index.Close()
		return nil, err
	}
	index.trained = true
	return index, nil
}

// Perform a Nearest Neighbors search on the Index
//
// # Arguments
//
// * `Resources` - Resources to use
// * `queries` - Tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
func SearchIndex[T any](resources cuvs.Resource, index BruteForceIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[float32]) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	prefilter := C.cuvsFilter{
		addr:  0,
		_type: C.CUVS_FILTER_NONE,
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceSearch(C.ulong(resources.Resource), index.index, (*C.DLManagedTensorVersioned)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensorVersioned)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensorVersioned)(unsafe.Pointer(distances.C_tensor)), prefilter)))

	return err
}
