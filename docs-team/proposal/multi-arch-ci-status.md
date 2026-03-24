# Multi-Architecture CI Status

**Date:** 2026-03-24
**Status:** Implemented (commit b84e555c)

---

## Overview

CI workflows now build and test all language bindings across both `x86_64` (amd64) and `aarch64` (arm64) architectures.

## Matrix: Language x Architecture x CUDA

| Language | amd64 | arm64 | CUDA Versions | Workflow Jobs |
|----------|-------|-------|---------------|---------------|
| C++ (conda) | Yes | Yes (via shared-workflows) | 12.9, 13.1 | `cpp-build` |
| C (standalone) | Yes | Yes | 12.9.1, 13.1.1 | `rocky8-clib-standalone-build` |
| Python (conda) | Yes | Yes (via shared-workflows) | 12.9, 13.1 | `python-build` |
| Python (wheel) | Yes | Yes (via matrix_filter) | 12.9, 13.1 | `wheel-build-*` |
| **Rust** | Yes | **Yes (new)** | 12.9.1, 13.1.1 | `rust-build` |
| **Go** | Yes | **Yes (new)** | 12.9.1, 13.1.1 | `go-build` |
| **Java** | Yes | **Yes (new)** | 12.9.1, 13.1.1 | `java-build` / `conda-java-build-and-tests` |
| Docs | Yes | No (not needed) | latest | `docs-build` |

## Changes Made

### .github/workflows/build.yaml
- `rust-build`: Added `arch: [amd64, arm64]` to matrix, changed `arch:` from hardcoded `"amd64"` to `"${{ matrix.arch }}"`
- `go-build`: Same change
- `java-build`: Same change + updated `artifact-name` to include `_${{ matrix.arch }}` suffix

### .github/workflows/pr.yaml
- `rust-build`: Same matrix change as build.yaml
- `go-build`: Same matrix change as build.yaml
- `conda-java-build-and-tests`: Same matrix change + artifact name fix

## Architecture-Specific Notes

### DiskANN
DiskANN is **x86_64 only** (excluded on aarch64 in conda recipe). This means some index types may not be available on arm64. The Rust/Go/Java bindings should handle this gracefully (features behind conditional compilation or runtime detection).

### Build Scripts
All build scripts (`ci/build_rust.sh`, `ci/build_go.sh`, `ci/build_java.sh`) already use `arch=$(arch)` for conda environment selection. No script changes were required.

### Sccache
Cache keys include architecture: `cuvs-rs/${RAPIDS_CONDA_ARCH}/cuda${CUDA_MAJOR}`. Separate caches for amd64 and arm64 are maintained automatically.

## Job Count Impact

Each language binding goes from **2 jobs** (2 CUDA versions) to **4 jobs** (2 CUDA x 2 arch):

| Before | After |
|--------|-------|
| rust-build: 2 jobs | rust-build: 4 jobs |
| go-build: 2 jobs | go-build: 4 jobs |
| java-build: 2 jobs | java-build: 4 jobs |
| **Total: +6 jobs per workflow run** | |
