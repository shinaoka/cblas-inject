# cblas-inject

CBLAS compatible interface backed by Fortran BLAS function pointers.

## Overview

This crate provides CBLAS-style functions (with row-major support) that internally call Fortran BLAS functions via runtime-registered function pointers. This is useful for integrating Rust code with:

- **Python** (scipy): Access BLAS via `scipy.linalg.cython_blas.__pyx_capi__`
- **Julia** (libblastrampoline): Access BLAS via `LinearAlgebra.BLAS.lbt_get_forward()`

## Features

- **cblas-sys compatible by default**: All 120 functions from cblas-sys are implemented. The default build exports LP64 `cblas_*` symbols that can serve as a drop-in CBLAS provider for crates that depend on cblas-sys (see [below](#use-with-cblas-sys))
- **Row-major support**: Handles row-major (C-style) data without memory copy for BLAS operations
- **Partial registration**: Only register the functions you need
- **Zero runtime overhead**: Uses `OnceLock` for minimal overhead (~0.5ns per call)
- **LP64/ILP64 GEMM provider support**: Register LP64 and ILP64 `dgemm`/`zgemm` Fortran BLAS providers explicitly at runtime
- **ILP64 CBLAS extensions**: `cblas_dgemm_64` and `cblas_zgemm_64` accept 64-bit CBLAS dimensions independent of the default LP64 ABI
- **Complex return style**: Configurable ABI for complex dot products (cdotc, cdotu, zdotc, zdotu)

## Usage

### Basic Usage

This example uses the default LP64 CBLAS ABI with an LP64 Fortran provider.

```rust
use cblas_inject::{
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

## LP64, ILP64, and the C API

BLAS libraries commonly expose one of two integer ABIs:

- **LP64**: Fortran BLAS integer arguments are `int32_t`.
- **ILP64**: Fortran BLAS integer arguments are `int64_t`.

The default build keeps unprefixed `cblas_*` symbols LP64-compatible, which is
the build to use with `cblas-sys`. The `ilp64` Cargo feature is transitional:
it changes the crate's `blasint` alias and unprefixed `cblas_*` ABI to 64-bit,
so it is not `cblas-sys` compatible. `openblas` and `ilp64` cannot be enabled
together because the `openblas` feature auto-registers LP64 OpenBLAS symbols.

For new C or FFI integrations, use the stable prefixed registration API:

```c
#include "cblas_inject.h"

void *dgemm_ilp64 = /* address of an ILP64 Fortran dgemm provider */;

int status = cblas_inject_register_dgemm_ilp64(dgemm_ilp64);
if (status != CBLAS_INJECT_STATUS_OK) {
    /* handle CBLAS_INJECT_STATUS_NULL_POINTER or
       CBLAS_INJECT_STATUS_ALREADY_REGISTERED */
}
```

The header is provided at [`include/cblas_inject.h`](include/cblas_inject.h). It
declares exact LP64 and ILP64 provider function pointer types, status codes,
capability queries, and the `_64` CBLAS entry points. If you have a typed C
function pointer instead of an address returned by `dlsym` or a host runtime,
pass it with an explicit platform-appropriate cast to `const void *`. The
registration suffix must match the provider pointer ABI.

Available registration entry points:

- `cblas_inject_register_dgemm_lp64`
- `cblas_inject_register_dgemm_ilp64`
- `cblas_inject_register_zgemm_lp64`
- `cblas_inject_register_zgemm_ilp64`

Capability queries:

- `cblas_inject_blas_int_width()` returns the integer width of unprefixed
  `cblas_*` symbols in the loaded library instance: `32` for the default build,
  `64` for the transitional `--features ilp64` build.
- `cblas_inject_supports_lp64_registration()` and
  `cblas_inject_supports_ilp64_registration()` report whether the loaded build
  accepts those explicit provider registrations.

True ILP64 CBLAS calls are exposed as `cblas_dgemm_64` and `cblas_zgemm_64`.
They always take `int64_t` dimensions and leading dimensions. Their
order/transpose arguments use standard CBLAS numeric values, or the
`CBLAS_INJECT_*` constants from `include/cblas_inject.h`. If only an LP64
provider is registered, `_64` calls dispatch only when all BLAS integer
arguments fit in `int32_t`; otherwise they call `cblas_xerbla` and return.

The older `register_*` symbols, such as `register_dgemm`, are compatibility
entry points whose ABI follows the current Rust build. New C integrations
should prefer the explicit `cblas_inject_register_*_{lp64,ilp64}` API.

Registration and CBLAS calls must use the same loaded `libcblas_inject`
instance. If a host program `dlopen`s one path but a downstream shared library
links a different copy, the provider registry is not shared between those
instances.

### Julia Example

See [examples/julia/dgemm_example.jl](examples/julia/dgemm_example.jl) for a complete working example.

```julia
using LinearAlgebra, Libdl

# Recommended: build the default LP64 CBLAS-compatible library.
# Register whichever BLAS integer ABI Julia selected at runtime.
lib = dlopen("path/to/libcblas_inject.dylib")

blas64 = LinearAlgebra.BLAS.USE_BLAS64
interface = blas64 ? :ilp64 : :lp64

# Get dgemm_ pointer from libblastrampoline.
dgemm_ptr = LinearAlgebra.BLAS.lbt_get_forward("dgemm_", interface)

register_name = if blas64
    :cblas_inject_register_dgemm_ilp64
else
    :cblas_inject_register_dgemm_lp64
end
status = ccall(dlsym(lib, register_name), Cint, (Ptr{Cvoid},), dgemm_ptr)
status == 0 || error("cblas-inject registration failed: $status")

# Now use cblas_dgemm for LP64 CBLAS calls, or cblas_dgemm_64
# when the caller needs an ILP64 CBLAS ABI.
```

### Python (scipy) Example

See [examples/python/dgemm_example.py](examples/python/dgemm_example.py) for a complete working example.

```python
import ctypes
import scipy.linalg.cython_blas as blas

# Build: cargo build --release
lib = ctypes.CDLL("path/to/libcblas_inject.dylib")

# Extract dgemm pointer from scipy's PyCapsule
capsule = blas.__pyx_capi__['dgemm']
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
capsule_name = ctypes.pythonapi.PyCapsule_GetName(capsule)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
dgemm_ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, capsule_name)

# Register with cblas-inject. SciPy wheels commonly expose LP64 BLAS.
lib.cblas_inject_register_dgemm_lp64.argtypes = [ctypes.c_void_p]
lib.cblas_inject_register_dgemm_lp64.restype = ctypes.c_int
status = lib.cblas_inject_register_dgemm_lp64(dgemm_ptr)
if status != 0:
    raise RuntimeError(f"cblas-inject registration failed: {status}")

# Now use cblas_dgemm
# ... call lib.cblas_dgemm with ctypes
```

### Use with cblas-sys

cblas-inject's default build can serve as the CBLAS implementation for crates that depend on
[cblas-sys](https://crates.io/crates/cblas-sys). Since cblas-inject exports
all CBLAS functions as `#[no_mangle] pub extern "C"` symbols, the linker
resolves cblas-sys's `extern "C"` declarations to cblas-inject's implementations
automatically.

```toml
[dependencies]
cblas-sys = "0.1"
cblas-inject = "0.1"
```

```rust
use cblas_inject::register_dgemm;

// Register an LP64 Fortran BLAS pointer.
unsafe { register_dgemm(dgemm_ptr); }

// Now cblas_sys::cblas_dgemm calls cblas-inject's implementation
unsafe {
    cblas_sys::cblas_dgemm(
        cblas_sys::CblasRowMajor,
        cblas_sys::CblasNoTrans,
        cblas_sys::CblasNoTrans,
        m, n, k, alpha,
        a.as_ptr(), lda,
        b.as_ptr(), ldb,
        beta,
        c.as_mut_ptr(), ldc,
    );
}
```

This is useful when a third-party crate depends on cblas-sys and you want
cblas-inject to provide the underlying CBLAS implementation.

**Notes:**

- Use the default build for `cblas-sys`. The `ilp64` feature changes the
  unprefixed `cblas_*` ABI to 64-bit and is not compatible with cblas-sys.
- Do not link another native CBLAS library (e.g., via `openblas-src`) at the
  same time, as this would cause duplicate symbol errors.

## Row-Major Handling

For BLAS operations (GEMM, etc.), row-major data is handled via argument swapping without memory copy, following the same approach as [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c#L489-L537):

```
C = A × B  (row-major)
⟺ C^T = B^T × A^T  (column-major)
```

## Complex Return Style

Fortran complex functions (`cdotu`, `cdotc`, `zdotu`, `zdotc`) have two calling conventions:

- **ReturnValue (0)**: Complex returned via register (OpenBLAS, MKL intel, BLIS)
- **HiddenArgument (1)**: Complex written to first pointer argument (gfortran default, MKL gf)

Set the convention **before** registering complex dot product functions:

```rust
use cblas_inject::{set_complex_return_style, ComplexReturnStyle};

unsafe {
    set_complex_return_style(ComplexReturnStyle::ReturnValue);
    register_zdotc(zdotc_ptr);
}
```

See [examples/julia/zdotc_example.jl](examples/julia/zdotc_example.jl) and [examples/python/zdotc_example.py](examples/python/zdotc_example.py) for complete examples.

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

Additional ILP64 CBLAS extension symbols:

- `cblas_dgemm_64`
- `cblas_zgemm_64`

### Error Handling

- `cblas_xerbla`: CBLAS error handler (simplified non-variadic version). Its
  parameter-number argument is a C `int`, independent of the LP64/ILP64 BLAS
  integer ABI.

## License

MIT OR Apache-2.0. This project also includes portions derived from OpenBLAS
under the BSD-3-Clause license. See `THIRD_PARTY_NOTICES.md`.

## Acknowledgments

The row-major conversion logic is based on [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) (BSD-3-Clause license).
