# Upstream Issues Priority Plan

**Date:** 2026-03-23
**Source:** https://github.com/rapidsai/cuvs/issues (open issues)
**Excludes:** Issues already addressed by our cherry-picked PRs (#1802 OOB fix, #1859 validated builder, and related fixes/features in our PR #1)

---

## Resolved Issues

### PR #2 — `nuvai/priority1-critical-bugs` (merged)

| # | Issue | Fix summary |
|---|-------|-------------|
| **#1875** | [BUG] IVF-PQ IP distance assumes normalized vectors | Normalize a copy of the trainset for InnerProduct metric; re-derive unnormalized centers for correct search. (`ivf_pq_build.cuh`) |
| **#1632** | [BUG] Brute Force returns wrong distances | Removed flawed self-neighbor detection in L2 that falsely zeroed distances between distinct points with identical norms; replaced with simple clamp-to-zero. (`l2_exp.cuh`) |
| **#1765** | [BUG] CAGRA build with IVF-PQ build algorithm | Increased `n_probes` heuristic from `sqrt(n)/20+4` to `sqrt(n)` and `refinement_rate` from 1→2 for better self-inclusion recall; fixed self-inclusion counter double-counting. (`cagra_build.cuh`, `ivf_pq.hpp`) |
| **#1622** | [BUG] Brute force crash after IVF build (Go) | Fixed use-after-free and double-free in Go `Tensor.Close`/`ToHost`/`Expand` by tracking data ownership via deleter; fixed shape pointer dangling. (`go/dlpack.go`) |
| **#1777** | [BUG] CagraQ compression kmeans defaults silently change | `fill_missing_params_heuristics` now respects `max_train_points_per_vq_cluster`/`pq_code` when explicitly set by user. (`vpq_dataset.cuh`) |
| — | C-API UB in resource destroy | `cuvsResourcesDestroy` now deletes through `raft::device_resources*` (derived type) to avoid UB from non-virtual destructor. (`c_api.cpp`) |
| — | C interop DLPack metadata leak | `to_dlpack` now invokes prior deleter to prevent metadata leak. (`interop.hpp`) |

### PR #3 — `nuvai/priority1-2-fixes` (merged)

| # | Issue | Fix summary |
|---|-------|-------------|
| **#1863** | [BUG] Python `cuvs.Resources` segfault with explicit stream | Added `cuvsResourcesCreateWithStream()` to construct `raft::device_resources` with the user's stream from the start, avoiding the create-then-set pattern that left internal state bound to the default stream. Includes `cudaStreamQuery` validation to fail fast on invalid handles. (`c_api.h`, `c_api.cpp`, `c_api.pxd`, `resources.pyx`) |
| **#1860** | [FEA] Async CAGRA build | Added `cuvsCagraBuildAsync`/`cuvsCagraBuildAwait`/`cuvsCagraBuildHandleDestroy`. Thread-safe: each async build owns its own `raft::device_resources`, deep-copies params via `convert_c_index_params` on the calling thread, pins CUDA device for multi-GPU safety, syncs stream before returning. Typed delete in `HandleDestroy` prevents GPU memory leaks. Go bindings: `BuildIndexAsync`, `Await`, `Close`. (`cagra.h`, `cagra.cpp`, `go/cagra/cagra.go`) |
| **#1717** | [FEA] C enum rename to avoid symbol collisions | All unprefixed C enum values renamed with `CUVS_` prefix across 9 enums (distance, build algo, search algo, hash mode, filter type, merge strategy, HNSW hierarchy, KMeans init, binary quantizer threshold). Backward-compat `#define` aliases gated behind `CUVS_ENABLE_DEPRECATED_ENUM_ALIASES`. All internal C source (17 files), Go bindings (6 files), and Rust bindings (4 files) updated to use new names directly. |
| **#1744** | [SBIN] SOVERSION always enabled | Removed `if(PROJECT_IS_TOP_LEVEL)` guard around SOVERSION setup in `c/CMakeLists.txt`. SOVERSION is now set unconditionally so `libcuvs_c.so` gets proper ABI versioning even when consumed via `add_subdirectory()`. |
| **#1745** | [SBIN] Symbol visibility controls | Created `cuvs_export.h` with `CUVS_API` macro (`visibility("default")` on GCC/Clang, `dllexport`/`dllimport` on Windows). Set `C_VISIBILITY_PRESET hidden`, `CXX_VISIBILITY_PRESET hidden`, `VISIBILITY_INLINES_HIDDEN ON` on the `cuvs_c` target. Added `PRIVATE CUVS_C_EXPORTS` compile definition. Annotated all 132 public C function declarations with `CUVS_API` across 19 headers. Internal C++ helpers no longer leak into the public ABI surface. |
| **#1629** | [FEA] Add Disk API (C/Rust) for Vamana | Implemented `deserialize()` for Vamana in C++, C, and Rust. C++ detail reads the DiskANN graph format (24-byte header + variable-length adjacency lists), optionally loads `.data` file, constructs `index<T>` via host mdspans. Macro instantiation for float/int8/uint8. C API `cuvsVamanaDeserialize()` takes explicit `DLDataType` (format doesn't encode dtype). Rust `Index::deserialize::<T>()` uses `IntoDtype` trait for compile-time type safety. Added `update_medoid()` public setter to `vamana.hpp`. Also hardened: `unique_ptr` exception safety in `_build`/`_deserialize`, `RAFT_EXPECTS(addr!=0)` guard in `GetDims`, atomic commit of `(dtype, addr)` on success. |
| **#1592** | [FEA] Go/Rust index serialization | **Go:** Added `SerializeIndex`/`DeserializeIndex` for CAGRA, Brute Force, IVF-Flat, IVF-PQ. Proper `C.CString`+`defer C.free`, error-path `index.Close()`. **Rust:** Added `serialize`/`deserialize` for Brute Force. Added `IntoDtype` impls for `i8`/`u8`, re-exported `IntoDtype` from crate root. Fixed pre-existing FFI handle leak on error across all 6 Rust index types (CAGRA, BF, IVF-Flat, IVF-PQ build/build_owned/deserialize) by wrapping handles in `Index` before fallible C calls. |

### PR #4 — `nuvai/priority3-perf-quality` (merged)

| # | Issue | Fix summary |
|---|-------|-------------|
| **#1756** | [FEA] `float64` support for CAGRA | Full `double` support across all CAGRA operations. **Template instantiations:** `cagra_{build,search,extend,serialize,merge}_double.cu`, `iface_cagra_double_uint32_t.cu`. **Search kernels:** 18 `compute_distance_standard/vpq` `.cu` files for double (4 metrics × 3 team sizes + 6 VPQ variants), plus `search_{multi,single}_cta_double_uint32.cu`. **C-API:** dispatch branches for `kDLFloat/64` in all 14 dispatch sites. **Header:** declarations in `cagra.hpp` for build/search/extend/serialize/deserialize/merge/build_knn_graph. **Infra fixes:** `utils::size_of<double>()` specialization (required by `get_vlen` in distance kernels), `config<double>` already existed in `ann_utils.cuh`. **Build path:** IVF-PQ and NN-Descent graph builders are only instantiated for float/half/int8/uint8 — added `if constexpr` float-cast (thrust::transform T→float on device) so double CAGRA build works with the default IVF-PQ algorithm. |
| **#1685** | [FEA] bf16 support | Full `nv_bfloat16` support — same scope as double. **Template instantiations:** 7 `.cu` files (same pattern). **Search kernels:** 18 `compute_distance` `.cu` files + 2 search CTA files. **C-API:** dispatch via `kDLBfloat/16`. **Infra fixes:** `config<nv_bfloat16>` (value_t=float, kDivisor=1.0) in `ann_utils.cuh`, `utils::size_of<nv_bfloat16>()` in `utils.hpp`, `#include <cuda_bf16.h>` in 7 headers (`cagra.hpp`, `ann_utils.cuh`, `utils.hpp`, `compute_distance_{standard,vpq}.hpp`, `search_{multi,single}_cta_inst.cuh`). **Serialization:** bf16 has no numpy dtype — serialize writes `"\|V2"` (void, 2 bytes); deserialize matches on `kind=='V' && itemsize==2`. **Build path:** same `if constexpr` float-cast as double. **Math note:** search kernels promote bf16→float (QUERY_T=float) via `mapping<float>{}()` which uses CUDA's native `__nv_bfloat16`→float cast; distance computation is always float-precision. |
| **#1595** | [BUG] CAGRA extend doesn't support float16 | Already supported — `cagra_extend_half.cu` exists with proper instantiation. No change needed. |
| **#1773** | [BUG] Single-GPU KMeans 2x memory | Investigated — upstream confirmed the 2x overhead is in Dask/multi-GPU data chunking, not single-GPU KMeans (which ran 30GB on 32GB V100). No fix needed. |
| **#1672** | [FEA] Remove CAGRA IndexWrapper | Already removed upstream (PR #1792). The `cuvsCagraIndex` C struct is the expected type-erased handle pattern for C APIs. No change needed. |

### PR #5 — Priority 5 Bug Fixes & KMeans int8/uint8 Support

| # | Issue | Fix summary |
|---|-------|-------------|
| **#1761** | [BUG] KMeans `reduce_cols_by_key` crash at scale | Zero-initialized `wtInCluster` buffer before `reduce_cols_by_key` (which uses atomic adds and requires zeroed output). One-line fix in `kmeans.cuh` and `kmeans_mg.cuh`. |
| **#1637** | [BUG] Iterative CAGRA build fails with mmap'd memory | Added `cudaHostRegister`/`cudaHostUnregister` in `mmap_owner` (cagra_build.cuh) so mmap'd memory is properly registered with CUDA for device↔host transfers and correct `cudaPointerGetAttributes` behavior on UVM/HMM systems. |
| **#1819** | [FEA] KMeans uint8/int8 support | Added `fit`, `predict`, `fit_predict` overloads for `int8_t` and `uint8_t` input data. Thin wrappers cast input to float via `thrust::transform` then delegate to existing float KMeans. Float centroids/inertia (integer centroids are meaningless after averaging). 6 new `.cu` files + header declarations. |

### PR #7 — Binary Dataset + ADC Search for CAGRA (Issue #5)

| # | Issue | Fix summary |
|---|-------|-------------|
| **#1906** | [FEA] Binary dataset support in CAGRA | **binary_dataset class** (`common.hpp`): `binary_dataset<IdxT>` follows `vpq_dataset` pattern — packed `uint8_t` data in `dataset<IdxT>` subclass, `dim()` returns original bit dimensionality, constructor validates `packed_dim >= ceil(dim/8)`. Type traits `is_binary_dataset_v`. **ADC kernel** (`compute_distance_binary_adc-impl.cuh`): computes `-dot(query, binary_vec)` via branchless `float(bit) * query_val`. Per-byte `ld.global.cg.u8` loads (alignment-safe for arbitrary `packed_dim`). Adjacent team threads read adjacent bytes for coalescing. Supports L2Expanded and InnerProduct only (L1 semantically wrong, L2Sqrt would NaN on negative output, Cosine needs norms, BitwiseHamming needs uint8 queries). **Descriptor spec** (`compute_distance_binary_adc.hpp`): `binary_adc_descriptor_spec` with metric validation in `priority()`. 3 generated kernel `.cu` files (TeamSize 8/16/32). **Search integration** (`cagra_search.cuh`): `dynamic_cast<binary_dataset>` branch with `RAFT_EXPECTS` metric guard. **Descriptor cache** (`factory.cuh`): `make_key` overload for `binary_dataset`. **Workflow:** `cagra::build(float, L2)` → `binary::train + transform` → `index.update_dataset(binary_dataset)` → `cagra::search(float queries)`. No new public API functions. |
| — | [BUG] Go missing `DistanceBitwiseHamming` | Added `DistanceBitwiseHamming` constant in iota block + mapping to `C.CUVS_DISTANCE_BITWISE_HAMMING`. (`go/distance.go`) |
| — | [FEA] Rust `BITWISE_HAMMING` re-export | Added `pub const BITWISE_HAMMING: DistanceType` friendly constant. (`rust/cuvs/src/distance_type.rs`) |

---

## Priority 1 — Critical Bugs (correctness / crashes)

| # | Issue | Status |
|---|-------|--------|
| **#1875** | [BUG] IVF-PQ IP distance assumes normalized vectors | ✅ Fixed in PR #2 |
| **#1863** | [BUG] Python `cuvs.Resources` segfault with explicit stream | ✅ Fixed in PR #3 |
| **#1632** | [BUG] Brute Force returns wrong distances | ✅ Fixed in PR #2 |
| **#1622** | [BUG] Brute force crash after IVF build (Go) | ✅ Fixed in PR #2 |
| **#1765** | [BUG] CAGRA build with IVF-PQ build algorithm | ✅ Fixed in PR #2 |
| **#1777** | [BUG] CagraQ compression kmeans defaults silently change | ✅ Fixed in PR #2 |
| **#1829** | [BUG] Multi-GPU KMeans convergence degradation | ⬜ Open — lower urgency, multi-GPU specific |

## Priority 2 — High-Value Features (directly aligned with our Rust/C-API focus)

| # | Issue | Status |
|---|-------|--------|
| **#1860** | [Rust/C] `cuvsCagraBuild` blocks thread — add async variant | ✅ Fixed in PR #3 |
| **#1717** | [FEA] Rename C enums to avoid symbol collisions | ✅ Fixed in PR #3 |
| **#1629** | [FEA] Add Disk API (C/Rust) for Vamana | ✅ Fixed in PR #3 |
| **#1745** | [SBIN] Symbol visibility controls in `cuvs_c` | ✅ Fixed in PR #3 |
| **#1744** | [SBIN] SOVERSION always enabled in `cuvs_c` | ✅ Fixed in PR #3 |
| **#1592** | [FEA] Go/Rust: index serialization back to CPU | ✅ Fixed in PR #3 |

## Priority 3 — Performance & Quality

| # | Issue | Status |
|---|-------|--------|
| **#1595** | [BUG] CAGRA extend doesn't support float16 | ✅ Already supported (cagra_extend_half.cu exists) |
| **#1756** | [FEA] `float64` support for CAGRA | ✅ Fixed in PR #4 — full double support (build/search/extend/serialize/merge) |
| **#1685** | [FEA] bf16 support | ✅ Fixed in PR #4 — full nv_bfloat16 support (build/search/extend/serialize/merge) |
| **#1773** | [BUG] Single-GPU KMeans 2x memory footprint | ⬜ Upstream confirmed issue is in Dask/multi-GPU chunking, not single-GPU. Single-GPU KMeans runs 30GB on 32GB V100 fine. |
| **#1672** | [FEA] Remove CAGRA `IndexWrapper` | ⬜ Already removed upstream (PR #1792). The C struct `cuvsCagraIndex` (type-erased handle) is the expected C-API pattern. |
| **#1906** | [FEA] Binary dataset support in CAGRA | ✅ Fixed in PR #7 — `binary_dataset` + ADC kernel + Go/Rust binding fixes. Chose Option C (hybrid): fresh impl following `vpq_dataset` pattern, no dependency on upstream PR #1846. |

## Priority 4 — ABI Stability & Packaging (important for distribution)

| # | Issue | Why it matters |
|---|-------|----------------|
| **#1739–#1743** | SBIN tracker (ABI check CI, code-owners, docs, bot, release job) | Full ABI stability infrastructure — important if we ship binary packages. |
| **#1684** | [FEA] CI checker for ABI stability | Related to the SBIN series above. |
| **#1683** | [DOC] ABI stability guidelines | |
| **#1643** | [DOC] Enterprise packaging & ABI implications | |

## Priority 5 — Nice-to-Have / Lower Urgency

| # | Issue | Why it matters |
|---|-------|----------------|
| **#1870–#1873** | UDF architecture for JIT + CAGRA filtering | We already added filtering (`778c47fc`); the UDF framework is the upstream generalization. Worth tracking. |
| **#1761** | [BUG] KMeans `reduce_cols_by_key` error | ✅ Fixed in PR #5 — zero-init wtInCluster buffer |
| **#1637** | [BUG] Iterative CAGRA build fails with mmap | ✅ Fixed in PR #5 — cudaHostRegister for mmap'd memory |
| **#1541** | NNDescent Dask perf depends on import order | Python-specific, 4 comments. |
| **#1819** | [FEA] KMeans uint8/int8 support | ✅ Fixed in PR #5 — float-promoted fit/predict/fit_predict overloads |

---

## Recommended Next Steps

1. **Priorities 1–3 and 5 are fully resolved.** Only multi-GPU issues (#1829, #1773 Dask path) and Python-specific #1541 remain open.
2. **Priority 4** (ABI stability infrastructure) is the next natural focus if binary distribution is planned.
3. Track the UDF architecture issues (#1870–#1873) as they generalize the filtering support we already added.
4. **Binary ADC follow-ups** (not blocking, can be fast follows):
   - Oversampling + re-ranking utility (`search_with_reranking`) for >90% recall (Plan Part 3)
   - Serialization support for `binary_dataset` (currently not persisted across serialize/deserialize)
   - C-API dispatch for binary dataset (currently C++ only)
