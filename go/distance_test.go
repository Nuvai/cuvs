package cuvs_test

import (
	"math/rand"
	"testing"
	"time"

	cuvs "github.com/rapidsai/cuvs/go"
)

func TestDistance(t *testing.T) {
	resource, _ := cuvs.NewResource(nil)

	rand.Seed(time.Now().UnixNano())

	NDataPoints := 256
	NFeatures := 16

	TestDataset := make([][]float32, NDataPoints)
	for i := range TestDataset {
		TestDataset[i] = make([]float32, NFeatures)
		for j := range TestDataset[i] {
			TestDataset[i][j] = rand.Float32()
		}
	}

	dataset, _ := cuvs.NewTensor(TestDataset)

	DistancesDataset := make([][]float32, NDataPoints)
	for i := range DistancesDataset {
		DistancesDataset[i] = make([]float32, NDataPoints)
	}

	distances, _ := cuvs.NewTensor(DistancesDataset)

	distances.ToDevice(&resource)
	dataset.ToDevice(&resource)

	cuvs.PairwiseDistance(resource, &dataset, &dataset, &distances, cuvs.DistanceL2, 0.0)

	distances.ToHost(&resource)

	resource.Sync()

	arr, _ := distances.Slice()
	if arr[0][0] != 0.0 {
		t.Error("wrong distance, expected", 0.0, "got", arr[0][0])
	}
}

func TestDistanceBitwiseHammingConstant(t *testing.T) {
	// Verify DistanceBitwiseHamming is defined and maps to the correct C constant
	cDist, exists := cuvs.CDistances[cuvs.DistanceBitwiseHamming]
	if !exists {
		t.Fatal("DistanceBitwiseHamming not found in CDistances map")
	}
	// CUVS_DISTANCE_BITWISE_HAMMING = 20
	if cDist != 20 {
		t.Errorf("DistanceBitwiseHamming maps to %d, expected 20", cDist)
	}
}
