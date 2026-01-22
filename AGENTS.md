# cblas-trampoline Agent Guidelines

This document provides guidelines for AI agents working on cblas-trampoline.

## Project Overview

cblas-trampoline provides CBLAS-compatible functions backed by Fortran BLAS function pointers
registered at runtime. This enables interoperability with BLAS libraries from Python (SciPy)
and Julia (libblastrampoline).

**Key directories:**
```
cblas-trampoline/
├── src/
│   ├── lib.rs           # Public exports
│   ├── backend.rs       # Function pointer storage (OnceLock per function)
│   ├── types.rs         # CBLAS enums, blasint type
│   ├── autoregister.rs  # Auto-registration for `openblas` feature
│   ├── blas1/           # BLAS Level 1 (vector operations)
│   │   ├── mod.rs
│   │   ├── dot.rs       # dot, nrm2, asum, amax
│   │   ├── vector.rs    # swap, copy, axpy, scal
│   │   └── rotation.rs  # rot, rotg, rotm, rotmg
│   └── blas3/           # BLAS Level 3 (matrix-matrix operations)
│       ├── mod.rs
│       ├── gemm.rs      # General matrix multiply
│       ├── symm.rs      # Symmetric matrix multiply
│       └── ...
├── ctest/               # OpenBLAS CBLAS test suite (ported)
│   ├── Makefile
│   ├── common.h         # Minimal header for tests
│   ├── c_sblat1.c       # Single precision BLAS1 tests
│   ├── c_dblat1.c       # Double precision BLAS1 tests
│   ├── c_cblat1.c       # Single complex BLAS1 tests
│   └── c_zblat1.c       # Double complex BLAS1 tests
├── tests/               # Rust integration tests
│   └── gemm.rs          # GEMM comparison tests
├── extern/
│   ├── OpenBLAS/        # Git submodule (reference implementation)
│   └── libblastrampoline/  # Git submodule (ABI handling reference)
└── Cargo.toml
```

**Cargo features:**
- `default` - LP64 (32-bit integers)
- `ilp64` - ILP64 (64-bit integers)
- `openblas` - Auto-register OpenBLAS functions at library load (uses `ctor` crate)

## Critical: Fortran Complex Return Value ABI

### The Problem

Fortran `COMPLEX FUNCTION` return values have **two different ABIs**:

1. **Return Value Convention**: Complex result returned via register
2. **Hidden Argument Convention**: Complex result written to a hidden first pointer argument

```c
// Return value convention (OpenBLAS, MKL intel, BLIS)
Complex64 zdotu_(const int *n, ...);

// Hidden argument convention (gfortran default, MKL gf)
void zdotu_(Complex64 *ret, const int *n, ...);
```

### Affected Functions (4 only)

| Function | Type | Description |
|----------|------|-------------|
| `cdotu` | Complex32 | Dot product (unconjugated) |
| `cdotc` | Complex32 | Dot product (conjugated) |
| `zdotu` | Complex64 | Dot product (unconjugated) |
| `zdotc` | Complex64 | Dot product (conjugated) |

All other BLAS functions are unaffected (return void, real, or integer).

**Note:** LAPACK also has 2 affected functions (`cladiv`, `zladiv`), but these are not
currently in scope for cblas-trampoline.

### Library Support Matrix

| Library | Convention | Notes |
|---------|------------|-------|
| OpenBLAS | Return value | ARM64/x86_64 |
| MKL (intel) | Return value | `mkl_intel_lp64` |
| MKL (gfortran) | Hidden argument | `mkl_gf_lp64` |
| BLIS | Both | Returns in both places |
| Apple Accelerate | N/A | CBLAS only |
| libblastrampoline | Both | Runtime detection |

### Implementation Plan

**Phase 1** (current): Return value convention only
- Works with OpenBLAS, MKL intel, BLIS

**Phase 2** (planned): Dual convention support
1. Add `ComplexReturnStyle` enum and detection API
2. Use raw pointer registration for 4 affected functions
3. Unified call interface that branches on convention

```rust
// Planned API
pub unsafe fn detect_complex_convention(zdotu_ptr: *const ()) -> ComplexReturnStyle;
pub fn set_complex_convention(style: ComplexReturnStyle);
pub unsafe fn register_zdotu(ptr: *const ());  // Requires convention set first
```

**Phase 3** (planned): Hidden convention tests
- Rust reference implementations for testing both conventions

## Critical: CBLAS Complex Scalar Passing

CBLAS passes complex scalars as **pointers**, not by value:

```c
// Correct (CBLAS standard)
void cblas_caxpy(int n, const void *alpha, ...);

// WRONG - do not use value passing
void cblas_caxpy(int n, Complex32 alpha, ...);  // INCORRECT
```

Affected functions: `cblas_caxpy`, `cblas_zaxpy`, `cblas_cscal`, `cblas_zscal`,
`cblas_cgemm`, `cblas_zgemm`, and all complex BLAS functions with scalar parameters.

## Row-Major Conversion

CBLAS supports row-major layout, but Fortran BLAS is column-major only.

**Design Rule:** Follow OpenBLAS's `interface/` implementation exactly.

| Function | Reference |
|----------|-----------|
| gemm | `OpenBLAS/interface/gemm.c` |
| symm | `OpenBLAS/interface/symm.c` |
| syrk | `OpenBLAS/interface/syrk.c` |

Each source file implementing row-major conversion must include:
```rust
// Row-major conversion logic derived from OpenBLAS
// Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
// https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/{function}.c
```

## Implementation Status

### BLAS Level 3 (22 functions)

| Function | s | d | c | z |
|----------|---|---|---|---|
| gemm | Y | Y | Y | Y |
| symm | - | Y | - | - |
| syrk | - | Y | - | - |
| syr2k | - | Y | - | - |
| trmm | - | Y | - | - |
| trsm | - | Y | - | - |
| hemm | - | - | - | - |
| herk | - | - | - | - |
| her2k | - | - | - | - |

### BLAS Level 2 (52 functions)

Not started. Functions to implement:

**General matrix-vector (4 precisions each):**
- `gemv`, `gbmv` - General matrix-vector multiply
- `trmv`, `tbmv`, `tpmv` - Triangular matrix-vector multiply
- `trsv`, `tbsv`, `tpsv` - Triangular solve

**Symmetric (real only: s/d):**
- `symv`, `sbmv`, `spmv` - Symmetric matrix-vector multiply
- `ger` - Rank-1 update
- `syr`, `spr`, `syr2`, `spr2` - Symmetric rank-1/2 updates

**Hermitian (complex only: c/z):**
- `hemv`, `hbmv`, `hpmv` - Hermitian matrix-vector multiply
- `geru`, `gerc` - Complex rank-1 updates
- `her`, `hpr`, `her2`, `hpr2` - Hermitian rank-1/2 updates

### BLAS Level 1 (38 functions)

All implemented and tested.

| Category | Functions |
|----------|-----------|
| Dot products | sdot, ddot, cdotu_sub, cdotc_sub, zdotu_sub, zdotc_sub, sdsdot, dsdot |
| Norms | snrm2, dnrm2, scnrm2, dznrm2 |
| Sums | sasum, dasum, scasum, dzasum |
| Index | isamax, idamax, icamax, izamax |
| Vector ops | sswap/dswap/cswap/zswap, scopy/dcopy/ccopy/zcopy |
| | saxpy/daxpy/caxpy/zaxpy, sscal/dscal/cscal/zscal/csscal/zdscal |
| Rotations | srot/drot, srotg/drotg, srotm/drotm, srotmg/drotmg |

## Testing

### Design Principle: Minimize Test Code Copies

**Do NOT copy test files from OpenBLAS.** Instead:
1. Reference files directly from `extern/OpenBLAS/ctest/`
2. Create only minimal adapter files in `ctest/`
3. Use symlinks or include paths where possible

This ensures:
- Tests stay in sync with OpenBLAS updates
- Less maintenance burden
- Clear provenance of test code

### OpenBLAS ctest (C tests)

```bash
cd ctest
make
./xscblat1 && ./xdcblat1 && ./xccblat1 && ./xzcblat1
```

Tests BLAS Level 1 functions. All should pass.

**Test input files** are in `extern/OpenBLAS/ctest/`:
- `sin1`, `din1`, `cin1`, `zin1` - BLAS1 parameters
- `sin2`, `din2`, `cin2`, `zin2` - BLAS2 parameters
- `sin3`, `din3`, `cin3`, `zin3` - BLAS3 parameters

### Rust integration tests

```bash
cargo test
cargo test --test gemm  # GEMM-specific tests
```

### Adding Tests for New Functions

```rust
fn setup_function() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        cblas_inject::register_function(fortran_function_);
    });
}

#[test]
fn test_function_exhaustive() {
    setup_function();
    // Test parameter ranges from OpenBLAS input files (din3, sin3, etc.)
    // Compare cblas_inject vs OpenBLAS cblas_* results
}
```

### Needed: Hidden Convention Tests

Create `tests/complex_dot_convention.rs`:
- Reference implementations for both conventions
- Test detection logic
- Verify both conventions produce same results

## Adding New CBLAS Functions

1. **Check OpenBLAS interface**: `extern/OpenBLAS/interface/{function}.c`
2. **Add Fortran function pointer type** in `src/backend.rs`
3. **Add registration function** in `src/backend.rs`
4. **Implement CBLAS wrapper** in `src/blas{1,2,3}/{function}.rs`
5. **Add tests** comparing against OpenBLAS
6. **Update implementation status** in this document

### Function Pointer Type Convention

```rust
/// Fortran {function} function pointer type
pub type {Function}FnPtr = unsafe extern "C" fn(
    // Parameters match Fortran signature (all pointers)
    param1: *const blasint,
    param2: *const f64,
    // ...
) -> ReturnType;  // or no return for void
```

### backend.rs Design Pattern

Each function uses `OnceLock` for thread-safe one-time registration:

```rust
static DGEMM: OnceLock<DgemmFnPtr> = OnceLock::new();

pub unsafe fn register_dgemm(f: DgemmFnPtr) {
    DGEMM.set(f).expect("dgemm already registered");
}

pub(crate) fn get_dgemm() -> DgemmFnPtr {
    *DGEMM.get().expect("dgemm not registered")
}
```

### autoregister.rs (openblas feature)

When `--features openblas` is enabled, functions are auto-registered at library load:

```rust
#[ctor::ctor]
fn register_all_blas() {
    unsafe {
        register_dgemm(std::mem::transmute(dgemm_ as *const ()));
        // ...
    }
}
```

This requires linking against OpenBLAS and is used for ctest.

## LP64 vs ILP64

| Mode | Feature | blasint type |
|------|---------|--------------|
| LP64 (default) | - | i32 |
| ILP64 | `ilp64` | i64 |

```bash
cargo build                      # LP64
cargo build --features ilp64     # ILP64
```

## Code Style

- `cargo fmt --all` before committing
- `cargo clippy --workspace` for linting
- Avoid `unwrap()`/`expect()` in library code (use proper error handling)

## Current Priority Tasks

1. **Phase 2: Dual ABI Support** - Add hidden argument convention support for cdotu/cdotc/zdotu/zdotc
2. **Hidden Convention Tests** - Create `tests/complex_dot_convention.rs`
3. **BLAS Level 2** - Implement gemv and other Level 2 functions

## References

- [libblastrampoline complex adapters](extern/libblastrampoline/src/complex_return_style_adapters.c)
- [SciPy g77 ABI wrappers](extern/scipy/) (if available)
- [OpenBLAS ctest](extern/OpenBLAS/ctest/)
- [OpenBLAS interface files](extern/OpenBLAS/interface/) - Row-major conversion reference
- [CBLAS reference](https://www.netlib.org/blas/blast-forum/cblas.tgz)
