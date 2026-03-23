package cagra

// #include <stdlib.h>
// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

// Cagra ANN Index
type CagraIndex struct {
	index   C.cuvsCagraIndex_t
	trained bool
}

// Creates a new empty Cagra Index
func CreateIndex() (*CagraIndex, error) {
	var index C.cuvsCagraIndex_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &CagraIndex{index: index}, nil
}

// Builds a new Index from the dataset for efficient search.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for building the index
// * `dataset` - A row-major Tensor on either the host or device to index
// * `index` - CagraIndex to build
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *CagraIndex) error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuild(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensorVersioned)(unsafe.Pointer(dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

// CagraBuildHandle is an opaque handle for an in-progress async build.
type CagraBuildHandle struct {
	handle C.cuvsCagraBuildHandle_t
}

// BuildIndexAsync starts building a CAGRA index in the background and returns
// immediately. Call Await on the returned handle to block until the build
// finishes and retrieve the index.
//
// Params are deep-copied and may be destroyed immediately after this call.
// The dataset must remain valid until Await returns.
func BuildIndexAsync[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T]) (*CagraBuildHandle, error) {
	var handle C.cuvsCagraBuildHandle_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuildAsync(
		C.ulong(Resources.Resource),
		params.params,
		(*C.DLManagedTensorVersioned)(unsafe.Pointer(dataset.C_tensor)),
		&handle)))
	if err != nil {
		return nil, err
	}
	return &CagraBuildHandle{handle: handle}, nil
}

// Await blocks until the async build completes and stores the result in index.
// The handle is consumed and must not be used again.
func (h *CagraBuildHandle) Await(index *CagraIndex) error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuildAwait(h.handle, index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

// Close destroys the async build handle. If the build is still in progress,
// it blocks until completion and discards the result.
func (h *CagraBuildHandle) Close() error {
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuildHandleDestroy(h.handle)))
}

// Extends the index with additional data
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for extending the index
// * `additional_dataset` - A row-major Tensor on the device to extend the index with
// * `index` - CagraIndex to extend
func ExtendIndex[T any](Resources cuvs.Resource, params *ExtendParams, additional_dataset *cuvs.Tensor[T], index *CagraIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before calling extend")
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraExtend(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensorVersioned)(unsafe.Pointer(additional_dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Destroys the Cagra Index
func (index *CagraIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Perform a Approximate Nearest Neighbors search on the Index
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters to use in searching the index
// * `queries` - A tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
// * `allowList` - List of indices to allow in the search, if nil, no filtering is applied
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *CagraIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[uint32], distances *cuvs.Tensor[T], allowList []uint32) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	var filter C.cuvsFilter
	bitset := createBitset(allowList)
	allowListTensor, err := cuvs.NewVector[uint32](bitset)
	if err != nil {
		return err
	}
	defer allowListTensor.Close()
	_, err = allowListTensor.ToDevice(&Resources)
	if err != nil {
		return err
	}
	if allowList == nil {
		filter = C.cuvsFilter{
			_type: C.CUVS_FILTER_NONE,
			addr:  C.uintptr_t(0),
		}
	} else {
		filter = C.cuvsFilter{
			_type: C.CUVS_FILTER_BITSET,
			addr:  C.uintptr_t(uintptr(unsafe.Pointer(allowListTensor.C_tensor))),
		}
	}
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensorVersioned)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensorVersioned)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensorVersioned)(unsafe.Pointer(distances.C_tensor)), filter)))
}

// Serialize a CAGRA index to file.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `index` - CagraIndex to serialize
// * `filename` - Path to save the index
// * `includeDataset` - Whether to include the dataset in the serialized index
func SerializeIndex(Resources cuvs.Resource, index *CagraIndex, filename string, includeDataset bool) error {
	if !index.trained {
		return errors.New("index needs to be built before calling serialize")
	}
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSerialize(
		C.cuvsResources_t(Resources.Resource),
		cFilename,
		index.index,
		C.bool(includeDataset),
	)))
}

// Deserialize a CAGRA index from file.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `filename` - Path to load the index from
func DeserializeIndex(Resources cuvs.Resource, filename string) (*CagraIndex, error) {
	index, err := CreateIndex()
	if err != nil {
		return nil, err
	}
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraDeserialize(
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

func createBitset(allowList []uint32) []uint32 {
	// Calculate size needed for the bitset array
	// Each uint32 handles 32 bits, so we divide the max ID by 32 (shift right by 5)
	maxID := uint32(0)
	for _, id := range allowList {
		if id > maxID {
			maxID = id
		}
	}
	size := (maxID >> 5) + 1 // Division by 32, add 1 to handle remainder
	bitset := make([]uint32, size)
	for _, id := range allowList {
		// Calculate which uint32 in our array (divide by 32)
		arrayIndex := id >> 5
		// Calculate bit position within that uint32 (mod 32)
		bitPosition := id & 31 // equivalent to id % 32
		// Set the bit
		bitset[arrayIndex] |= 1 << bitPosition
	}
	return bitset
}
