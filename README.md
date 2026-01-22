# cblas-trampoline

CBLAS/LAPACKE compatible interface backed by Fortran BLAS/LAPACK function pointers.

## Overview

This crate provides CBLAS-style functions (with row-major support) that internally call Fortran BLAS/LAPACK functions via runtime-registered function pointers. This is useful for integrating Rust code with:

- **Python** (scipy): Access BLAS via `scipy.linalg.cython_blas.__pyx_capi__`
- **Julia** (libblastrampoline): Access BLAS via `LinearAlgebra.BLAS.lbt_get_forward()`

## Features

- **Row-major support**: Handles row-major (C-style) data without memory copy for BLAS operations
- **Partial registration**: Only register the functions you need
- **Zero runtime overhead**: Uses `OnceLock` for minimal overhead (~0.5ns per call)
- **LP64/ILP64 support**: Compile with `ilp64` feature for 64-bit integers

## Usage

```rust
use cblas_trampoline::{
    register_dgemm, cblas_dgemm,
    CblasRowMajor, CblasNoTrans,
};

fn main() {
    // Register Fortran dgemm pointer (obtained from Python/Julia)
    unsafe {
        register_dgemm(dgemm_ptr);
    }

    // Use CBLAS-style interface
    let m = 2;
    let n = 3;
    let k = 4;
    let alpha = 1.0;
    let beta = 0.0;

    let a: Vec<f64> = vec![/* m x k matrix, row-major */];
    let b: Vec<f64> = vec![/* k x n matrix, row-major */];
    let mut c: Vec<f64> = vec![0.0; m * n];

    unsafe {
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m as i32, n as i32, k as i32,
            alpha,
            a.as_ptr(), k as i32,  // lda = k for row-major NoTrans
            b.as_ptr(), n as i32,  // ldb = n for row-major NoTrans
            beta,
            c.as_mut_ptr(), n as i32,  // ldc = n for row-major
        );
    }
}
```

## Row-Major Handling

For BLAS operations (GEMM, etc.), row-major data is handled via argument swapping without memory copy, following the same approach as [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c#L489-L537):

```
C = A × B  (row-major)
⟺ C^T = B^T × A^T  (column-major)
```

For LAPACK operations (SVD, etc.), explicit transpose copies are required, following [LAPACKE's approach](https://github.com/OpenMathLib/OpenBLAS/blob/develop/lapack-netlib/LAPACKE/src/lapacke_dgesvd_work.c#L49-L127).

## Supported Functions

### BLAS Level 3

| Function | Description |
|----------|-------------|
| `cblas_sgemm` | Single precision matrix multiply |
| `cblas_dgemm` | Double precision matrix multiply |
| `cblas_cgemm` | Single precision complex matrix multiply |
| `cblas_zgemm` | Double precision complex matrix multiply |

More functions will be added as needed.

## License

MIT OR Apache-2.0

## Acknowledgments

The row-major conversion logic is based on [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) (BSD-3-Clause license).
