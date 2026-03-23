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

### PR #3 — `nuvai/priority1-2-fixes` (current branch)

| # | Issue | Fix summary |
|---|-------|-------------|
| **#1863** | [BUG] Python `cuvs.Resources` segfault with explicit stream | Added `cuvsResourcesCreateWithStream()` to construct `raft::device_resources` with the user's stream from the start, avoiding the create-then-set pattern that left internal state bound to the default stream. Includes `cudaStreamQuery` validation to fail fast on invalid handles. (`c_api.h`, `c_api.cpp`, `c_api.pxd`, `resources.pyx`) |
| **#1860** | [FEA] Async CAGRA build | Added `cuvsCagraBuildAsync`/`cuvsCagraBuildAwait`/`cuvsCagraBuildHandleDestroy`. Thread-safe: each async build owns its own `raft::device_resources`, deep-copies params via `convert_c_index_params` on the calling thread, pins CUDA device for multi-GPU safety, syncs stream before returning. Typed delete in `HandleDestroy` prevents GPU memory leaks. Go bindings: `BuildIndexAsync`, `Await`, `Close`. (`cagra.h`, `cagra.cpp`, `go/cagra/cagra.go`) |
| **#1717** | [FEA] C enum rename to avoid symbol collisions | All unprefixed C enum values renamed with `CUVS_` prefix across 9 enums (distance, build algo, search algo, hash mode, filter type, merge strategy, HNSW hierarchy, KMeans init, binary quantizer threshold). Backward-compat `#define` aliases gated behind `CUVS_ENABLE_DEPRECATED_ENUM_ALIASES`. All internal C source (17 files), Go bindings (6 files), and Rust bindings (4 files) updated to use new names directly. |
| **#1744** | [SBIN] SOVERSION always enabled | Removed `if(PROJECT_IS_TOP_LEVEL)` guard around SOVERSION setup in `c/CMakeLists.txt`. SOVERSION is now set unconditionally so `libcuvs_c.so` gets proper ABI versioning even when consumed via `add_subdirectory()`. |
| **#1745** | [SBIN] Symbol visibility controls | Created `cuvs_export.h` with `CUVS_API` macro (`visibility("default")` on GCC/Clang, `dllexport`/`dllimport` on Windows). Set `C_VISIBILITY_PRESET hidden`, `CXX_VISIBILITY_PRESET hidden`, `VISIBILITY_INLINES_HIDDEN ON` on the `cuvs_c` target. Added `PRIVATE CUVS_C_EXPORTS` compile definition. Annotated all 132 public C function declarations with `CUVS_API` across 19 headers. Internal C++ helpers no longer leak into the public ABI surface. |
| **#1629** | [FEA] Add Disk API (C/Rust) for Vamana | Implemented `deserialize()` for Vamana in C++, C, and Rust. C++ detail reads the DiskANN graph format (24-byte header + variable-length adjacency lists), optionally loads `.data` file, constructs `index<T>` via host mdspans. Macro instantiation for float/int8/uint8. C API `cuvsVamanaDeserialize()` takes explicit `DLDataType` (format doesn't encode dtype). Rust `Index::deserialize::<T>()` uses `IntoDtype` trait for compile-time type safety. Added `update_medoid()` public setter to `vamana.hpp`. Also hardened: `unique_ptr` exception safety in `_build`/`_deserialize`, `RAFT_EXPECTS(addr!=0)` guard in `GetDims`, atomic commit of `(dtype, addr)` on success. |
| **#1592** | [FEA] Go/Rust index serialization | **Go:** Added `SerializeIndex`/`DeserializeIndex` for CAGRA, Brute Force, IVF-Flat, IVF-PQ. Proper `C.CString`+`defer C.free`, error-path `index.Close()`. **Rust:** Added `serialize`/`deserialize` for Brute Force. Added `IntoDtype` impls for `i8`/`u8`, re-exported `IntoDtype` from crate root. Fixed pre-existing FFI handle leak on error across all 6 Rust index types (CAGRA, BF, IVF-Flat, IVF-PQ build/build_owned/deserialize) by wrapping handles in `Index` before fallible C calls. |

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

| # | Issue | Why it matters |
|---|-------|----------------|
| **#1906** | [FEA] Binary dataset support in CAGRA | Opens CAGRA to binary embeddings — expanding use cases. |
| **#1595** | [BUG] CAGRA extend doesn't support float16 | Blocks FP16 workflows from using incremental index updates. |
| **#1756** | [FEA] `float64` support for CAGRA | Requested by scientific computing users. |
| **#1685** | [FEA] bf16 support | 4 comments — high demand for transformer-native precision. |
| **#1773** | [BUG] Single-GPU KMeans 2x memory footprint | Memory efficiency issue at scale. |
| **#1672** | [FEA] Remove CAGRA `IndexWrapper` | Code simplification that would ease our maintenance burden. |

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
| **#1761** | [BUG] KMeans `reduce_cols_by_key` error | Edge-case crash during training. |
| **#1637** | [BUG] Iterative CAGRA build fails with mmap | Affects large-dataset mmap workflows. |
| **#1541** | NNDescent Dask perf depends on import order | Python-specific, 4 comments. |
| **#1819** | [FEA] KMeans uint8/int8 support | Quantized clustering. |

---

## Recommended Next Steps

1. **Priority 1 is nearly clear** — only #1829 (multi-GPU KMeans) remains; defer unless multi-GPU is a near-term target.
2. **Priority 2 is now fully resolved** — all 6 issues addressed in PRs #2 and #3.
3. **Priority 3 candidates** for next iteration:
   - **#1906** (binary dataset CAGRA) and **#1685** (bf16 support) — expand data type coverage.
   - **#1595** (CAGRA extend float16) — unblocks FP16 incremental updates.
   - **#1773** (single-GPU KMeans memory) — production scaling concern.
4. Track the UDF architecture issues (#1870–#1873) as they generalize the filtering support we already added.
