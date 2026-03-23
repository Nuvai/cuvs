/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{stderr, Write};
use std::marker::PhantomData;

use crate::cagra::{IndexParams, SearchParams};
use crate::dlpack::{DatasetOwnership, ManagedTensor};
use crate::error::{check_cuvs, Result};
use crate::filters::{Filter, NoFilter};
use crate::resources::Resources;

/// CAGRA ANN Index
///
/// The lifetime parameter `'a` ensures the dataset outlives the index when built
/// with [`Index::build`]. Use [`Index::build_owned`] for a self-contained index
/// that owns its dataset (e.g., after [`ManagedTensor::to_device`]).
///
/// # Examples
///
/// ## Borrowed dataset (compiler enforces lifetime)
///
/// ```no_run
/// # use cuvs::{ManagedTensor, Resources};
/// # use cuvs::cagra::{Index, IndexParams};
/// let res = Resources::new().unwrap();
/// let arr = ndarray::Array::<f32, _>::zeros((256, 16));
/// let params = IndexParams::new().unwrap();
/// let tensor = ManagedTensor::from(&arr);
/// let index = Index::build(&res, &params, &tensor).unwrap();
/// // arr and tensor must remain alive while index is in use
/// ```
///
/// ## Owned dataset ('static lifetime)
///
/// ```no_run
/// # use cuvs::{ManagedTensor, Resources};
/// # use cuvs::cagra::{Index, IndexParams};
/// let res = Resources::new().unwrap();
/// let arr = ndarray::Array::<f32, _>::zeros((256, 16));
/// let params = IndexParams::new().unwrap();
/// let device_tensor = ManagedTensor::from(&arr).to_device(&res).unwrap();
/// let index = Index::build_owned(&res, &params, device_tensor).unwrap();
/// drop(arr); // Fine — index owns the device copy
/// ```
#[derive(Debug)]
pub struct Index<'a> {
    inner: ffi::cuvsCagraIndex_t,
    _data: DatasetOwnership<'a>,
}

impl<'a> Index<'a> {
    /// Creates a new FFI index handle.
    fn create_handle() -> Result<ffi::cuvsCagraIndex_t> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsCagraIndex_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexCreate(index.as_mut_ptr()))?;
            Ok(index.assume_init())
        }
    }

    /// Builds a new CAGRA Index from a borrowed dataset.
    ///
    /// The compiler enforces that `dataset` outlives the returned index,
    /// preventing use-after-free when the C library dereferences its
    /// internal view of the data.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build(
        res: &Resources,
        params: &IndexParams,
        dataset: &'a ManagedTensor,
    ) -> Result<Index<'a>> {
        let index = Index {
            inner: Self::create_handle()?,
            _data: DatasetOwnership::Borrowed(PhantomData),
        };
        unsafe {
            check_cuvs(ffi::cuvsCagraBuild(
                res.0,
                params.0,
                dataset.as_ptr(),
                index.inner,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index<'a>> {
        Ok(Index {
            inner: Self::create_handle()?,
            _data: DatasetOwnership::Borrowed(PhantomData),
        })
    }

    /// Perform a Approximate Nearest Neighbors search on the Index
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters to use in searching the index
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    /// * `filter` - Optional filter to apply to the search (defaults to NoFilter if not provided)
    pub fn search(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
        filter: Option<&dyn Filter>,
    ) -> Result<()> {
        unsafe {
            let filter_ffi = filter
                .map(|f| f.into_ffi())
                .unwrap_or_else(|| NoFilter.into_ffi());

            check_cuvs(ffi::cuvsCagraSearch(
                res.0,
                params.0,
                self.inner,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                filter_ffi,
            ))
        }
    }

    /// Save the CAGRA index to file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file name for saving the index
    /// * `include_dataset` - Whether to write out the dataset to the file
    pub fn serialize(&self, res: &Resources, filename: &str, include_dataset: bool) -> Result<()> {
        let c_filename = CString::new(filename).expect("filename contains null byte");
        unsafe {
            check_cuvs(ffi::cuvsCagraSerialize(
                res.0,
                c_filename.as_ptr(),
                self.inner,
                include_dataset,
            ))
        }
    }

    /// Save the CAGRA index to file in hnswlib format.
    ///
    /// NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
    /// as the serialization format is not compatible with the original hnswlib.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file name for saving the index
    pub fn serialize_to_hnswlib(&self, res: &Resources, filename: &str) -> Result<()> {
        let c_filename = CString::new(filename).expect("filename contains null byte");
        unsafe {
            check_cuvs(ffi::cuvsCagraSerializeToHnswlib(
                res.0,
                c_filename.as_ptr(),
                self.inner,
            ))
        }
    }

    /// Load a CAGRA index from file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The name of the file that stores the index
    pub fn deserialize(res: &Resources, filename: &str) -> Result<Index<'a>> {
        let c_filename = CString::new(filename).expect("filename contains null byte");
        let index = Index {
            inner: Self::create_handle()?,
            _data: DatasetOwnership::Borrowed(PhantomData),
        };
        unsafe {
            check_cuvs(ffi::cuvsCagraDeserialize(
                res.0,
                c_filename.as_ptr(),
                index.inner,
            ))?;
        }
        Ok(index)
    }
}

impl Index<'static> {
    /// Builds a new CAGRA Index from an owned dataset.
    ///
    /// The index takes ownership of `dataset`, making it self-contained
    /// with a `'static` lifetime. This is useful when the dataset is a
    /// device copy (from [`ManagedTensor::to_device`]) that should live
    /// as long as the index.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index
    /// * `dataset` - A row-major matrix to index (ownership transferred to the index)
    pub fn build_owned(
        res: &Resources,
        params: &IndexParams,
        dataset: ManagedTensor,
    ) -> Result<Index<'static>> {
        // Wrap handle immediately so Drop runs on error
        let mut index = Index {
            inner: Self::create_handle()?,
            _data: DatasetOwnership::Borrowed(PhantomData),
        };
        unsafe {
            check_cuvs(ffi::cuvsCagraBuild(
                res.0,
                params.0,
                dataset.as_ptr(),
                index.inner,
            ))?;
        }
        // Transfer dataset ownership only after successful build
        index._data = DatasetOwnership::Owned(dataset);
        Ok(index)
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexDestroy(self.inner) }) {
            write!(stderr(), "failed to call cagraIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn test_cagra(build_params: IndexParams) {
        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // build the cagra index (borrowed — dataset tensor must outlive index)
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to create cagra index");

        // use the first 4 points from the dataset as queries : will test that we get them back
        // as their own nearest neighbor
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);

        let k = 10;

        // CAGRA search API requires queries and outputs to be on device memory
        // copy query data over, and allocate new device memory for the distances/ neighbors
        // outputs
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host)
            .to_device(&res)
            .unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        let search_params = SearchParams::new().unwrap();

        index
            .search(&res, &search_params, &queries, &neighbors, &distances, None)
            .unwrap();

        // Copy back to host memory
        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        // nearest neighbors should be themselves, since queries are from the
        // dataset
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    #[test]
    fn test_cagra_index() {
        let build_params = IndexParams::new().unwrap();
        test_cagra(build_params);
    }

    #[test]
    fn test_cagra_compression() {
        use crate::cagra::CompressionParams;
        let build_params = IndexParams::new()
            .unwrap()
            .set_compression(CompressionParams::new().unwrap());
        test_cagra(build_params);
    }

    /// Test that an index can be searched multiple times without rebuilding.
    /// This validates that search() takes &self instead of self.
    #[test]
    fn test_cagra_multiple_searches() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();

        // Create a random dataset
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // Build the index once (borrowed)
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to create cagra index");

        let search_params = SearchParams::new().unwrap();
        let k = 5;

        // Perform multiple searches on the same index
        for search_iter in 0..3 {
            let n_queries = 4;
            let queries = dataset.slice(s![0..n_queries, ..]);
            let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();

            let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
            let neighbors = ManagedTensor::from(&neighbors_host)
                .to_device(&res)
                .unwrap();

            let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
            let distances = ManagedTensor::from(&distances_host)
                .to_device(&res)
                .unwrap();

            // This should work on every iteration because search() takes &self
            index
                .search(&res, &search_params, &queries, &neighbors, &distances, None)
                .unwrap_or_else(|e| panic!("search iteration {} failed: {}", search_iter, e));

            // Copy back to host memory
            distances.to_host(&res, &mut distances_host).unwrap();
            neighbors.to_host(&res, &mut neighbors_host).unwrap();

            // Verify results are consistent across searches
            assert_eq!(
                neighbors_host[[0, 0]],
                0,
                "iteration {}: first query should find itself",
                search_iter
            );
        }
    }

    #[test]
    fn test_cagra_serialize_deserialize() {
        let res = Resources::new().unwrap();

        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        let build_params = IndexParams::new().unwrap();
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to create cagra index");

        let dir = std::env::temp_dir();
        let filepath = dir.join("test_cagra_index.bin");
        let filepath_str = filepath.to_str().unwrap();
        index
            .serialize(&res, filepath_str, true)
            .expect("failed to serialize cagra index");

        assert!(filepath.exists());
        assert!(std::fs::metadata(&filepath).unwrap().len() > 0);

        let loaded_index: Index<'_> =
            Index::deserialize(&res, filepath_str).expect("failed to deserialize cagra index");

        let n_queries = 4;
        let k = 10;
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host)
            .to_device(&res)
            .unwrap();
        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        let search_params = SearchParams::new().unwrap();
        loaded_index
            .search(&res, &search_params, &queries, &neighbors, &distances, None)
            .expect("failed to search deserialized index");

        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_to_hnswlib() {
        let res = Resources::new().unwrap();

        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        let build_params = IndexParams::new().unwrap();
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to create cagra index");

        let dir = std::env::temp_dir();
        let filepath = dir.join("test_cagra_index_hnsw.bin");
        let filepath_str = filepath.to_str().unwrap();
        index
            .serialize_to_hnswlib(&res, filepath_str)
            .expect("failed to serialize to hnswlib format");

        assert!(filepath.exists());
        assert!(std::fs::metadata(&filepath).unwrap().len() > 0);

        let _ = std::fs::remove_file(&filepath);
    }

    /// Test that an index built with build_owned can be searched after the
    /// original host data is dropped.
    #[test]
    fn test_cagra_build_owned() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();

        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // Build with owned device memory
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();
        let index = Index::build_owned(&res, &build_params, dataset_device)
            .expect("failed to create cagra index");

        let search_params = SearchParams::new().unwrap();
        let k = 5;
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();

        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host)
            .to_device(&res)
            .unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        index
            .search(&res, &search_params, &queries, &neighbors, &distances, None)
            .unwrap();

        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        assert_eq!(neighbors_host[[0, 0]], 0);
    }
}
