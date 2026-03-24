# Proposal: Pre-built Binary Distribution for Rust cuVS Bindings

**Date:** 2026-03-24
**Status:** Draft
**Scope:** Rust crate (`cuvs` / `cuvs-sys`) binary distribution across platforms

---

## Problem

The `cuvs-sys` crate wraps `libcuvs_c`, a CUDA C++ shared library. Publishing to crates.io (or any Cargo registry) only distributes source. Users who `cargo add cuvs` today must have:

- A full CUDA toolkit installed
- CMake and a C++ compiler
- The entire cuVS C++ source tree available for the CMake fallback build

This makes onboarding painful and builds slow (the C++ compilation can take 30+ minutes).

## Goal

Enable `cargo add cuvs && cargo build` to just work by having `build.rs` automatically download a pre-built `libcuvs_c` for the user's platform.

---

## Current Build Flow

```
cargo build
  -> cuvs-sys/build.rs
       -> cmake::Config::new(".").build()
            -> find_package(cuvs) OR build C++ from source
       -> bindgen generates Rust FFI bindings
       -> cargo:rustc-link-lib=dylib=cuvs_c
```

## Proposed Build Flow

```
cargo build
  -> cuvs-sys/build.rs
       1. Check CUVS_LIBRARY_PATH env var         -> use if set (manual override)
       2. Check system (pkg-config / find_package) -> use if found (conda/system)
       3. Download pre-built from artifact store   -> based on target triple + CUDA version
       4. Fall back to CMake build from source     -> existing behavior
       -> bindgen generates Rust FFI bindings
       -> cargo:rustc-link-lib=dylib=cuvs_c
```

---

## Artifact Store Options

| Option | Pros | Cons |
|--------|------|------|
| **GitHub Releases** | Free, simple, works with public/private repos | Manual upload or CI scripting needed |
| **GitHub Packages** | Integrated with repo permissions | More complex API |
| **S3 bucket** | Already used for sccache in CI | Requires auth setup for users |
| **Artifactory/Nexus** | Enterprise-grade, access control | Infra overhead |

### Recommendation

**GitHub Releases** for simplicity. The CI already builds `libcuvs_c` for multiple platforms. Add a step to attach the built `.tar.gz` to the release when a tag is pushed.

For private/enterprise use, make the download URL configurable via `CUVS_BINARY_URL` env var so teams can point at their own artifact store (S3, Artifactory, etc).

---

## Artifact Layout

```
{base_url}/v{version}/
  libcuvs_c-x86_64-linux-cuda12.tar.gz
  libcuvs_c-x86_64-linux-cuda13.tar.gz
  libcuvs_c-aarch64-linux-cuda12.tar.gz
  libcuvs_c-aarch64-linux-cuda13.tar.gz
```

Each tarball contains:
```
lib/
  libcuvs_c.so
  libcuvs_c.so.26
include/
  cuvs/
    *.h
```

### Naming Convention

```
libcuvs_c-{arch}-{os}-cuda{major}.tar.gz
```

Where:
- `arch`: `x86_64`, `aarch64`
- `os`: `linux` (only supported platform currently)
- `cuda major`: `12`, `13`

---

## build.rs Changes

### Detection Logic

```rust
fn find_prebuilt_library() -> Option<PathBuf> {
    // 1. Explicit path override
    if let Ok(path) = env::var("CUVS_LIBRARY_PATH") {
        return Some(PathBuf::from(path));
    }

    // 2. System/conda install
    if let Some(path) = find_system_library() {
        return Some(path);
    }

    // 3. Download pre-built binary
    if let Some(path) = download_prebuilt() {
        return Some(path);
    }

    None // fall back to CMake
}
```

### Platform Detection

```rust
fn artifact_name() -> String {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap(); // x86_64, aarch64
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();     // linux
    let cuda_major = detect_cuda_major_version();           // 12, 13
    format!("libcuvs_c-{}-{}-cuda{}.tar.gz", arch, os, cuda_major)
}
```

### CUDA Version Detection

Priority order:
1. `CUVS_CUDA_VERSION` env var (explicit override)
2. `CUDA_VERSION` env var (set by NVIDIA containers)
3. Parse output of `nvcc --version`
4. Default to `12` (most common)

### Download URL Resolution

```rust
fn download_url() -> String {
    let base = env::var("CUVS_BINARY_URL")
        .unwrap_or_else(|_| "https://github.com/Nuvai/cuvs/releases/download".into());
    let version = env!("CARGO_PKG_VERSION"); // e.g., "26.6.0"
    let tag = format!("v{}", version);
    format!("{}/{}/{}", base, tag, artifact_name())
}
```

---

## CI Changes

### build.yaml: Add artifact upload to releases

```yaml
release-rust-binaries:
  needs: [rocky8-clib-standalone-build]
  if: startsWith(github.ref, 'refs/tags/v')
  runs-on: ubuntu-latest
  steps:
    - uses: actions/download-artifact@v4
      with:
        path: artifacts/
        pattern: libcuvs_c_*
    - name: Rename artifacts to standard convention
      run: |
        for f in artifacts/libcuvs_c_*.tar.gz; do
          # Map from CI naming to distribution naming
          # libcuvs_c_12.9.1_amd64.tar.gz -> libcuvs_c-x86_64-linux-cuda12.tar.gz
          ...
        done
    - uses: softprops/action-gh-release@v2
      with:
        files: dist/*.tar.gz
```

### Existing CI jobs already produce the binaries

The `rocky8-clib-standalone-build` job builds `libcuvs_c` for:
- `{12.9.1, 13.1.1}` x `{amd64, arm64}`

These just need to be attached to GitHub Releases on tag push.

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CUVS_LIBRARY_PATH` | Path to pre-built libcuvs_c directory | (none, skip download) |
| `CUVS_BINARY_URL` | Base URL for downloading pre-built binaries | GitHub Releases URL |
| `CUVS_CUDA_VERSION` | Override CUDA major version for download | Auto-detect from nvcc |
| `CUVS_BUILD_FROM_SOURCE` | Set to `1` to skip download and always build from source | `0` |

---

## User Experience

### Default (zero-config)
```bash
cargo add cuvs
cargo build
# build.rs detects x86_64-linux, CUDA 12 via nvcc
# downloads libcuvs_c-x86_64-linux-cuda12.tar.gz from GitHub Releases
# links and builds
```

### Conda/system install
```bash
conda install -c rapidsai cuvs
cargo build
# build.rs finds libcuvs_c via system search, no download needed
```

### Custom artifact store
```bash
export CUVS_BINARY_URL="https://artifacts.internal.example.com/cuvs"
cargo build
# downloads from internal mirror
```

### Force source build
```bash
export CUVS_BUILD_FROM_SOURCE=1
cargo build
# existing CMake behavior, requires full toolchain
```

---

## Platform Support Matrix

| Arch | OS | CUDA 12 | CUDA 13 | Status |
|------|----|---------|---------|--------|
| x86_64 | Linux | Yes | Yes | CI already builds |
| aarch64 | Linux | Yes | Yes | CI already builds (as of b84e555c) |
| x86_64 | macOS | No | No | CUDA not supported on macOS |
| aarch64 | macOS | No | No | CUDA not supported on macOS |
| x86_64 | Windows | No | No | Future consideration |

---

## Implementation Plan

1. **CI: Attach libcuvs_c artifacts to GitHub Releases** - Add release job to build.yaml
2. **build.rs: Add download fallback** - Implement detection + download logic before CMake fallback
3. **Testing** - Verify on clean machine with no CUDA toolkit that download path works
4. **Documentation** - Update README with env var table and install instructions

---

## Dependencies

- `ureq` or `reqwest` (blocking) as build-dependency for HTTP downloads in build.rs
- `flate2` + `tar` for extracting downloaded archives
- Or: shell out to `curl` + `tar` to avoid adding build deps

### Recommendation

Shell out to `curl` and `tar` in build.rs to avoid adding crate dependencies:

```rust
fn download_and_extract(url: &str, dest: &Path) -> Result<(), Box<dyn Error>> {
    let status = Command::new("curl")
        .args(["-sL", "--fail", url])
        .stdout(Stdio::piped())
        .spawn()?
        .stdout
        .take()
        .map(|stdout| {
            Command::new("tar")
                .args(["xzf", "-", "-C"])
                .arg(dest)
                .stdin(stdout)
                .status()
        });
    // ...
}
```

This keeps the build dependency footprint minimal and works on all Linux CI environments.
