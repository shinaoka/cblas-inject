# Testing Strategy for cblas-trampoline

## Overview

cblas-trampoline provides CBLAS-compatible functions backed by user-registered Fortran BLAS
function pointers. Testing verifies that:

1. Row-major/column-major conversion is correct
2. All parameter combinations work correctly
3. Results match a reference CBLAS implementation (OpenBLAS)

## Testing Approach

### Comparison-based Testing

We compare cblas-trampoline results against OpenBLAS's native CBLAS implementation.
This approach:

- Uses OpenBLAS as the reference implementation (linked via `blas-src`)
- Tests all parameter combinations systematically
- Achieves high coverage without manual expected-value calculation

### Test Parameters from OpenBLAS

We use the same parameter ranges as OpenBLAS's official CBLAS tests.
OpenBLAS test parameters are defined in input files:

| File | Description | Reference |
|------|-------------|-----------|
| `din3` | DBLAT3 parameters (dgemm, dsymm, etc.) | [OpenBLAS/ctest/din3](https://github.com/OpenMathLib/OpenBLAS/blob/develop/ctest/din3) |
| `zin3` | ZBLAT3 parameters (zgemm, etc.) | [OpenBLAS/ctest/zin3](https://github.com/OpenMathLib/OpenBLAS/blob/develop/ctest/zin3) |
| `din2` | DBLAT2 parameters (dgemv, etc.) | [OpenBLAS/ctest/din2](https://github.com/OpenMathLib/OpenBLAS/blob/develop/ctest/din2) |

Example from `din3`:
```
7                 NUMBER OF VALUES OF N
1 2 3 5 7 9 35    VALUES OF N
3                 NUMBER OF VALUES OF ALPHA
0.0 1.0 0.7       VALUES OF ALPHA
3                 NUMBER OF VALUES OF BETA
0.0 1.0 1.3       VALUES OF BETA
```

### Current Coverage

| Function | Test File | Test Cases |
|----------|-----------|------------|
| cblas_dgemm | `tests/gemm.rs` | 13,824 (2 orders × 2 transA × 2 transB × 6 dims³ × 4 alphas × 4 betas) |
| cblas_zgemm | `tests/gemm.rs` | 10,368 (2 orders × 3 transA × 3 transB × 4 dims³ × 3 alphas × 3 betas) |

## OpenBLAS Test System Architecture

For reference, OpenBLAS's official CBLAS test system has this structure:

```
c_dblat3.f (Fortran test driver)
    ↓ f2c
c_dblat3c.c (C version of test driver)
    ↓ calls
F77_dgemm() in c_dblas3.c (C wrapper)
    ↓ calls
cblas_dgemm() (CBLAS implementation under test)
```

The Fortran test drivers originated from LAPACK's BLAS test suite and provide
comprehensive coverage of edge cases, error conditions, and numerical accuracy.

### Why Not Port OpenBLAS Tests Directly?

We considered porting OpenBLAS's f2c-converted test drivers, but:

1. **f2c runtime dependency**: The converted C code requires libf2c
2. **Fortran I/O complexity**: The test drivers use Fortran I/O (READ, WRITE, FORMAT)
   which f2c converts to complex runtime calls
3. **Maintenance burden**: Large amount of f2c-generated code is hard to maintain

Instead, we:
- Extract test parameters from OpenBLAS input files
- Write native Rust tests that cover the same parameter space
- Compare against OpenBLAS's cblas_* functions directly

This gives us the same coverage with maintainable code.

## Adding Tests for New Functions

When adding a new CBLAS function, follow this pattern:

```rust
// In tests/new_function.rs

fn setup_function() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        cblas_trampoline::register_function(fortran_function_);
    });
}

#[test]
fn test_function_exhaustive() {
    setup_function();

    // Use parameter ranges from OpenBLAS input files
    let dims = [1, 2, 3, 5, 7, 9];
    let alphas = [0.0, 1.0, 0.7];
    // ...

    for &param1 in &params1 {
        for &param2 in &params2 {
            // Call cblas_trampoline function
            // Call OpenBLAS cblas function
            // Compare results
        }
    }
}
```

## Running Tests

```bash
# Run all tests
cargo test

# Run GEMM tests only
cargo test --test gemm

# Run with output
cargo test -- --nocapture
```

## Future Improvements

1. **Macro-based test generation**: Create macros to reduce boilerplate for
   testing multiple precision variants (s/d/c/z)

2. **Property-based testing**: Use proptest/quickcheck for random parameter
   exploration beyond fixed grid

3. **Benchmark suite**: Add criterion benchmarks comparing cblas-trampoline
   overhead vs direct Fortran calls
