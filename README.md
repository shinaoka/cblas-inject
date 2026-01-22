# cblas-runtime

CBLAS compatible interface backed by Fortran BLAS function pointers.

## Overview

This crate provides CBLAS-style functions (with row-major support) that internally call Fortran BLAS functions via runtime-registered function pointers. This is useful for integrating Rust code with:

- **Python** (scipy): Access BLAS via `scipy.linalg.cython_blas.__pyx_capi__`
- **Julia** (libblastrampoline): Access BLAS via `LinearAlgebra.BLAS.lbt_get_forward()`

## Features

- **cblas-sys compatible**: All 120 functions from cblas-sys are implemented
- **Row-major support**: Handles row-major (C-style) data without memory copy for BLAS operations
- **Partial registration**: Only register the functions you need
- **Zero runtime overhead**: Uses `OnceLock` for minimal overhead (~0.5ns per call)
- **LP64/ILP64 support**: Compile with `ilp64` feature for 64-bit integers

## Usage

### Basic Usage

```rust
use cblas_runtime::{
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

### Julia Example

See [examples/julia/dgemm_example.jl](examples/julia/dgemm_example.jl) for a complete working example.

```julia
using LinearAlgebra, Libdl

# Build with ILP64 for modern Julia: cargo build --release --features ilp64
lib = dlopen("path/to/libcblas_runtime.dylib")

# Get dgemm_ pointer from libblastrampoline
dgemm_ptr = LinearAlgebra.BLAS.lbt_get_forward("dgemm_", :ilp64)

# Register with cblas-runtime
register_dgemm = dlsym(lib, :register_dgemm)
ccall(register_dgemm, Cvoid, (Ptr{Cvoid},), dgemm_ptr)

# Now use cblas_dgemm
cblas_dgemm = dlsym(lib, :cblas_dgemm)
# ... call cblas_dgemm via ccall
```

### Python (scipy) Example

See [examples/python/dgemm_example.py](examples/python/dgemm_example.py) for a complete working example.

```python
import ctypes
import scipy.linalg.cython_blas as blas

# Build: cargo build --release
lib = ctypes.CDLL("path/to/libcblas_runtime.dylib")

# Extract dgemm pointer from scipy's PyCapsule
capsule = blas.__pyx_capi__['dgemm']
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
capsule_name = ctypes.pythonapi.PyCapsule_GetName(capsule)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
dgemm_ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, capsule_name)

# Register with cblas-runtime
lib.register_dgemm(dgemm_ptr)

# Now use cblas_dgemm
# ... call lib.cblas_dgemm with ctypes
```

## Row-Major Handling

For BLAS operations (GEMM, etc.), row-major data is handled via argument swapping without memory copy, following the same approach as [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c#L489-L537):

```
C = A × B  (row-major)
⟺ C^T = B^T × A^T  (column-major)
```

## Supported Functions

All functions from [cblas-sys](https://crates.io/crates/cblas-sys) are implemented:

### BLAS Level 1 (38 functions)

- Dot products: `sdot`, `ddot`, `cdotu_sub`, `cdotc_sub`, `zdotu_sub`, `zdotc_sub`, `sdsdot`, `dsdot`
- Norms: `snrm2`, `dnrm2`, `scnrm2`, `dznrm2`
- Absolute sums: `sasum`, `dasum`, `scasum`, `dzasum`
- Index of max: `isamax`, `idamax`, `icamax`, `izamax`
- Vector operations: `sswap`, `dswap`, `cswap`, `zswap`, `scopy`, `dcopy`, `ccopy`, `zcopy`, `saxpy`, `daxpy`, `caxpy`, `zaxpy`
- Scaling: `sscal`, `dscal`, `cscal`, `zscal`, `csscal`, `zdscal`
- Rotations: `srot`, `drot`, `srotg`, `drotg`, `srotm`, `drotm`, `srotmg`, `drotmg`
- Complex utilities: `scabs1`, `dcabs1`

### BLAS Level 2 (52 functions)

- General matrix-vector: `sgemv`, `dgemv`, `cgemv`, `zgemv`, `sgbmv`, `dgbmv`, `cgbmv`, `zgbmv`
- Triangular: `strmv`, `dtrmv`, `ctrmv`, `ztrmv`, `stbmv`, `dtbmv`, `ctbmv`, `ztbmv`, `stpmv`, `dtpmv`, `ctpmv`, `ztpmv`
- Triangular solve: `strsv`, `dtrsv`, `ctrsv`, `ztrsv`, `stbsv`, `dtbsv`, `ctbsv`, `ztbsv`, `stpsv`, `dtpsv`, `ctpsv`, `ztpsv`
- Symmetric/Hermitian: `ssymv`, `dsymv`, `chemv`, `zhemv`, `ssbmv`, `dsbmv`, `chbmv`, `zhbmv`, `sspmv`, `dspmv`, `chpmv`, `zhpmv`
- Rank updates: `sger`, `dger`, `cgeru`, `cgerc`, `zgeru`, `zgerc`, `ssyr`, `dsyr`, `cher`, `zher`, `sspr`, `dspr`, `chpr`, `zhpr`, `ssyr2`, `dsyr2`, `cher2`, `zher2`, `sspr2`, `dspr2`, `chpr2`, `zhpr2`

### BLAS Level 3 (30 functions)

- General matrix multiply: `sgemm`, `dgemm`, `cgemm`, `zgemm`
- Symmetric/Hermitian multiply: `ssymm`, `dsymm`, `csymm`, `zsymm`, `chemm`, `zhemm`
- Rank-k update: `ssyrk`, `dsyrk`, `csyrk`, `zsyrk`, `cherk`, `zherk`
- Rank-2k update: `ssyr2k`, `dsyr2k`, `csyr2k`, `zsyr2k`, `cher2k`, `zher2k`
- Triangular multiply: `strmm`, `dtrmm`, `ctrmm`, `ztrmm`
- Triangular solve: `strsm`, `dtrsm`, `ctrsm`, `ztrsm`

### Error Handling

- `cblas_xerbla`: Error handler (simplified non-variadic version)

## License

MIT OR Apache-2.0

## Acknowledgments

The row-major conversion logic is based on [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) (BSD-3-Clause license).
