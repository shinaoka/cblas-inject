# OpenBLAS CBLAS Test Suite Porting Plan

## Overview

This document describes the plan to port OpenBLAS's CBLAS test suite to validate
cblas-trampoline's CBLAS implementation.

## Implementation Scope

**cblas-trampoline covers the same CBLAS functions as cblas-sys (crates.io/crates/cblas-sys).**

This ensures API compatibility - users can switch from cblas-sys to cblas-trampoline
by changing imports while keeping the same function signatures.

### Function Count Summary

| Level | Functions | Notes |
|-------|-----------|-------|
| BLAS Level 1 | 38 | dot, nrm2, asum, amax, swap, copy, axpy, scal, rot, rotg, rotm, rotmg |
| BLAS Level 2 | 52 | gemv, gbmv, trmv, tbmv, tpmv, trsv, tbsv, tpsv, symv, sbmv, spmv, ger, syr, spr, syr2, spr2, hemv, hbmv, hpmv, geru, gerc, her, hpr, her2, hpr2 |
| BLAS Level 3 | 22 | gemm, symm, syrk, syr2k, trmm, trsm, hemm, herk, her2k |
| **Total** | **112** | All s/d/c/z variants |

### Complete Function List

#### BLAS Level 1 (38 functions)

**Dot products:**
- `cblas_sdot`, `cblas_ddot` - real dot product
- `cblas_cdotu_sub`, `cblas_zdotu_sub` - complex dot product (unconjugated)
- `cblas_cdotc_sub`, `cblas_zdotc_sub` - complex dot product (conjugated)
- `cblas_sdsdot`, `cblas_dsdot` - extended precision dot product

**Norms and sums:**
- `cblas_snrm2`, `cblas_dnrm2`, `cblas_scnrm2`, `cblas_dznrm2` - Euclidean norm
- `cblas_sasum`, `cblas_dasum`, `cblas_scasum`, `cblas_dzasum` - sum of absolute values

**Index of max:**
- `cblas_isamax`, `cblas_idamax`, `cblas_icamax`, `cblas_izamax`

**Vector operations:**
- `cblas_sswap`, `cblas_dswap`, `cblas_cswap`, `cblas_zswap` - swap vectors
- `cblas_scopy`, `cblas_dcopy`, `cblas_ccopy`, `cblas_zcopy` - copy vectors
- `cblas_saxpy`, `cblas_daxpy`, `cblas_caxpy`, `cblas_zaxpy` - y = αx + y
- `cblas_sscal`, `cblas_dscal`, `cblas_cscal`, `cblas_zscal` - scale vector
- `cblas_csscal`, `cblas_zdscal` - scale complex by real

**Rotations (real only):**
- `cblas_srot`, `cblas_drot` - apply Givens rotation
- `cblas_srotg`, `cblas_drotg` - generate Givens rotation
- `cblas_srotm`, `cblas_drotm` - apply modified Givens rotation
- `cblas_srotmg`, `cblas_drotmg` - generate modified Givens rotation

**Auxiliary:**
- `cblas_dcabs1`, `cblas_scabs1` - absolute value of complex

#### BLAS Level 2 (52 functions)

**General matrix-vector (4 precisions each):**
- `cblas_{s,d,c,z}gemv` - general matrix-vector multiply
- `cblas_{s,d,c,z}gbmv` - general banded matrix-vector multiply
- `cblas_{s,d,c,z}trmv` - triangular matrix-vector multiply
- `cblas_{s,d,c,z}tbmv` - triangular banded matrix-vector multiply
- `cblas_{s,d,c,z}tpmv` - triangular packed matrix-vector multiply
- `cblas_{s,d,c,z}trsv` - triangular solve
- `cblas_{s,d,c,z}tbsv` - triangular banded solve
- `cblas_{s,d,c,z}tpsv` - triangular packed solve

**Symmetric matrix-vector (real only):**
- `cblas_{s,d}symv` - symmetric matrix-vector multiply
- `cblas_{s,d}sbmv` - symmetric banded matrix-vector multiply
- `cblas_{s,d}spmv` - symmetric packed matrix-vector multiply

**Hermitian matrix-vector (complex only):**
- `cblas_{c,z}hemv` - Hermitian matrix-vector multiply
- `cblas_{c,z}hbmv` - Hermitian banded matrix-vector multiply
- `cblas_{c,z}hpmv` - Hermitian packed matrix-vector multiply

**Rank-1/2 updates:**
- `cblas_{s,d}ger` - real rank-1 update
- `cblas_{c,z}geru`, `cblas_{c,z}gerc` - complex rank-1 update
- `cblas_{s,d}syr`, `cblas_{s,d}spr` - symmetric rank-1 update
- `cblas_{s,d}syr2`, `cblas_{s,d}spr2` - symmetric rank-2 update
- `cblas_{c,z}her`, `cblas_{c,z}hpr` - Hermitian rank-1 update
- `cblas_{c,z}her2`, `cblas_{c,z}hpr2` - Hermitian rank-2 update

#### BLAS Level 3 (22 functions)

**All precisions (s/d/c/z):**
- `cblas_{s,d,c,z}gemm` - general matrix multiply ✅ implemented
- `cblas_{s,d,c,z}symm` - symmetric matrix multiply (dsymm ✅)
- `cblas_{s,d,c,z}syrk` - symmetric rank-k update (dsyrk ✅)
- `cblas_{s,d,c,z}syr2k` - symmetric rank-2k update (dsyr2k ✅)
- `cblas_{s,d,c,z}trmm` - triangular matrix multiply (dtrmm ✅)
- `cblas_{s,d,c,z}trsm` - triangular solve (dtrsm ✅)

**Complex only (c/z):**
- `cblas_{c,z}hemm` - Hermitian matrix multiply
- `cblas_{c,z}herk` - Hermitian rank-k update
- `cblas_{c,z}her2k` - Hermitian rank-2k update

#### Auxiliary

- `cblas_xerbla` - error handler

## Row-Major to Column-Major Conversion

CBLAS supports row-major layout, but Fortran BLAS is column-major only.

**Design Rule:** cblas-trampoline's row-major conversion logic follows OpenBLAS's
`interface/` implementation exactly. For each CBLAS function, refer to the
corresponding file in OpenBLAS:

| Function | Reference |
|----------|-----------|
| gemm | https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c |
| symm | https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symm.c |
| syrk | https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syrk.c |
| syr2k | https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2k.c |
| trmm | https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trmm.c |
| trsm | https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trsm.c |

When implementing a new function, copy the row-major handling logic from the
corresponding OpenBLAS interface file. This ensures compatibility with OpenBLAS's
well-tested behavior.

**Copyright Notice:** The conversion logic is derived from OpenBLAS, which is
licensed under BSD-3-Clause. Each source file that implements this logic must
include the following copyright notice:

```rust
// Row-major conversion logic derived from OpenBLAS
// Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
// https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/{function}.c
```

## OpenBLAS Test Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Test Driver (f2c converted)                      │
│  c_dblat3c.c - Main test logic, parameter iteration, result checking   │
│  (Originally c_dblat3.f, converted by f2c to pure C)                   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ calls F77_dgemm(), F77_dsymm(), etc.
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        C Wrappers (c_dblas3.c)                          │
│  F77_dgemm() → cblas_dgemm()                                           │
│  F77_dsymm() → cblas_dsymm()                                           │
│  ... handles row-major test matrix transposition                       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ calls cblas_*()
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CBLAS Implementation (under test)                    │
│  Currently: OpenBLAS's cblas_dgemm, etc.                               │
│  Target: cblas-trampoline's cblas_dgemm, etc.                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Files to Port from OpenBLAS/ctest/

### Core Files (copy as-is or with minimal changes)

| File | Description | Changes Needed |
|------|-------------|----------------|
| `c_dblat3c.c` | f2c-converted BLAS3 test driver | Replace `common.h` with minimal header |
| `c_sblat3c.c` | Single precision BLAS3 tests | Same |
| `c_zblat3c.c` | Complex BLAS3 tests | Same |
| `c_cblat3c.c` | Single complex BLAS3 tests | Same |
| `c_dblat2c.c` | BLAS2 test driver | Same |
| `c_dblat1c.c` | BLAS1 test driver | Same |
| (and s/c/z variants) | | |

### Wrapper Files (need modification)

| File | Description | Changes Needed |
|------|-------------|----------------|
| `c_dblas3.c` | F77_dgemm() etc. wrappers | Include cblas-trampoline headers |
| `c_dblas2.c` | BLAS2 wrappers | Same |
| `c_dblas1.c` | BLAS1 wrappers | Same |
| (and s/c/z variants) | | |

### Support Files (copy as-is)

| File | Description |
|------|-------------|
| `auxiliary.c` | `get_transpose_type()`, `get_uplo_type()`, etc. |
| `c_xerbla.c` | Error handler for testing |
| `constant.c` | Global variables `CBLAS_CallFromC`, `RowMajorStrg` |

### Input Files (copy as-is)

| File | Description |
|------|-------------|
| `din3` | DBLAT3 test parameters |
| `sin3` | SBLAT3 test parameters |
| `cin3` | CBLAT3 test parameters |
| `zin3` | ZBLAT3 test parameters |
| `din2`, `sin2`, `cin2`, `zin2` | BLAS2 parameters |

### Header Modifications

**cblas_test.h** - Copy and simplify:
- Remove OpenBLAS-specific `#include "cblas.h"`
- Add cblas-trampoline includes instead
- Keep `#define ADD_` for symbol naming (or choose one convention)
- Keep helper struct definitions

**common.h** - Create minimal replacement:
```c
// ctest/common.h - Minimal replacement for cblas-trampoline tests
#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// blasint type matching cblas-trampoline
#ifdef USE64BITINT
typedef int64_t blasint;
#else
typedef int32_t blasint;
#endif

// Fortran name mangling (pick one, match your BLAS library)
#define BLASFUNC(x) x##_

#endif
```

## Build System

### Option 1: Makefile (standalone)

```makefile
# ctest/Makefile
CC = cc
CFLAGS = -DADD_ -I../include

# Link against cblas-trampoline (built as C library) and a Fortran BLAS
LIBS = -L../target/release -lcblas_trampoline -lopenblas

OBJS = auxiliary.o c_xerbla.o constant.o

xdcblat3: c_dblat3c.o c_dblas3.o c_d3chke.o $(OBJS)
	$(CC) -o $@ $^ $(LIBS)

test: xdcblat3
	./xdcblat3 < din3
```

### Option 2: Cargo build.rs integration

```rust
// build.rs
fn main() {
    cc::Build::new()
        .file("ctest/c_dblat3c.c")
        .file("ctest/c_dblas3.c")
        .file("ctest/c_d3chke.c")
        .file("ctest/auxiliary.c")
        .file("ctest/c_xerbla.c")
        .file("ctest/constant.c")
        .define("ADD_", None)
        .include("ctest")
        .compile("dblat3");
}
```

## Implementation Steps

### Phase 1: Minimal Setup (BLAS3 only)

1. Create `ctest/` directory in cblas-trampoline
2. Copy from OpenBLAS/ctest/:
   - `c_dblat3c.c`, `c_dblas3.c`, `c_d3chke.c`
   - `auxiliary.c`, `c_xerbla.c`, `constant.c`
   - `din3`
   - `cblas_test.h`
3. Create minimal `common.h`
4. Modify `c_dblas3.c`:
   - Replace `#include "cblas.h"` with cblas-trampoline header
5. Create build system (Makefile or build.rs)
6. Test: `./xdcblat3 < din3`

### Phase 2: All Precisions

Repeat for s/c/z variants:
- `c_sblat3c.c`, `c_sblas3.c`, `c_s3chke.c`, `sin3`
- `c_cblat3c.c`, `c_cblas3.c`, `c_c3chke.c`, `cin3`
- `c_zblat3c.c`, `c_zblas3.c`, `c_z3chke.c`, `zin3`

### Phase 3: BLAS2

- `c_dblat2c.c`, `c_dblas2.c`, `c_d2chke.c`, `din2`
- (and s/c/z variants)

### Phase 4: BLAS1

- `c_dblat1c.c`, `c_dblas1.c`
- (and s/c/z variants)

## CBLAS Functions Tested

### BLAS Level 3 (c_dblas3.c)

| Function | Wrapper | Description |
|----------|---------|-------------|
| cblas_dgemm | F77_dgemm | General matrix multiply |
| cblas_dsymm | F77_dsymm | Symmetric matrix multiply |
| cblas_dtrmm | F77_dtrmm | Triangular matrix multiply |
| cblas_dtrsm | F77_dtrsm | Triangular solve |
| cblas_dsyrk | F77_dsyrk | Symmetric rank-k update |
| cblas_dsyr2k | F77_dsyr2k | Symmetric rank-2k update |

Complex variants add:
| cblas_zhemm | F77_zhemm | Hermitian matrix multiply |
| cblas_zherk | F77_zherk | Hermitian rank-k update |
| cblas_zher2k | F77_zher2k | Hermitian rank-2k update |

### BLAS Level 2 (c_dblas2.c)

| Function | Description |
|----------|-------------|
| cblas_dgemv | General matrix-vector multiply |
| cblas_dgbmv | General banded matrix-vector multiply |
| cblas_dsymv | Symmetric matrix-vector multiply |
| cblas_dsbmv | Symmetric banded matrix-vector multiply |
| cblas_dspmv | Symmetric packed matrix-vector multiply |
| cblas_dtrmv | Triangular matrix-vector multiply |
| cblas_dtbmv | Triangular banded matrix-vector multiply |
| cblas_dtpmv | Triangular packed matrix-vector multiply |
| cblas_dtrsv | Triangular solve (vector) |
| cblas_dtbsv | Triangular banded solve |
| cblas_dtpsv | Triangular packed solve |
| cblas_dger | Rank-1 update |
| cblas_dsyr | Symmetric rank-1 update |
| cblas_dspr | Symmetric packed rank-1 update |
| cblas_dsyr2 | Symmetric rank-2 update |
| cblas_dspr2 | Symmetric packed rank-2 update |

### BLAS Level 1 (c_dblas1.c)

| Function | Description |
|----------|-------------|
| cblas_drotg | Generate Givens rotation |
| cblas_drotmg | Generate modified Givens rotation |
| cblas_drot | Apply Givens rotation |
| cblas_drotm | Apply modified Givens rotation |
| cblas_dswap | Swap vectors |
| cblas_dscal | Scale vector |
| cblas_dcopy | Copy vector |
| cblas_daxpy | y = alpha*x + y |
| cblas_ddot | Dot product |
| cblas_dnrm2 | Euclidean norm |
| cblas_dasum | Sum of absolute values |
| cblas_idamax | Index of max absolute value |

## cblas-trampoline Implementation Status

Track which CBLAS functions are implemented in cblas-trampoline:

### BLAS Level 3 (22 functions)

| Function | s | d | c | z | Notes |
|----------|---|---|---|---|-------|
| gemm | ✅ | ✅ | ✅ | ✅ | All precisions done |
| symm | ❌ | ✅ | ❌ | ❌ | Only dsymm |
| syrk | ❌ | ✅ | ❌ | ❌ | Only dsyrk |
| syr2k | ❌ | ✅ | ❌ | ❌ | Only dsyr2k |
| trmm | ❌ | ✅ | ❌ | ❌ | Only dtrmm |
| trsm | ❌ | ✅ | ❌ | ❌ | Only dtrsm |
| hemm | - | - | ❌ | ❌ | Complex only |
| herk | - | - | ❌ | ❌ | Complex only |
| her2k | - | - | ❌ | ❌ | Complex only |

### BLAS Level 2 (52 functions)

| Status | Notes |
|--------|-------|
| ❌ | Not started |

### BLAS Level 1 (38 functions)

| Status | Notes |
|--------|-------|
| ❌ | Not started |

### Summary

| Level | Total | Implemented | Remaining |
|-------|-------|-------------|-----------|
| BLAS3 | 22 | 10 | 12 |
| BLAS2 | 52 | 0 | 52 |
| BLAS1 | 38 | 0 | 38 |
| **Total** | **112** | **10** | **102** |

## ILP64 vs LP64 Support

### Standard CBLAS vs OpenBLAS Extension

The **standard CBLAS** (Netlib reference) uses `int` for all dimensions and strides:

```c
// Standard CBLAS (Netlib)
void cblas_dgemm(..., const int M, const int N, const int K, ...);
```

**OpenBLAS extends** this with `blasint` to support 64-bit integers:

```c
// OpenBLAS
void cblas_dgemm(..., const blasint M, const blasint N, const blasint K, ...);
// where blasint = int (LP64) or long (ILP64, with USE64BITINT)
```

### cblas-trampoline Approach

cblas-trampoline follows OpenBLAS's approach, supporting both LP64 and ILP64 via the
`ilp64` Cargo feature. The OpenBLAS test suite also supports this via the `USE64BITINT` macro.

### Integer Type Mapping

| Mode | cblas-trampoline | OpenBLAS ctest | blasint type |
|------|------------------|----------------|--------------|
| LP64 (default) | `default-features` | (no flag) | `i32` / `int` |
| ILP64 | `features = ["ilp64"]` | `-DUSE64BITINT` | `i64` / `long` |

### Build Configuration

**LP64 build:**
```makefile
CFLAGS = -DADD_
```

**ILP64 build:**
```makefile
CFLAGS = -DADD_ -DUSE64BITINT
```

### CI Testing Matrix

To ensure both modes work correctly, CI should test:

```yaml
strategy:
  matrix:
    include:
      - name: LP64
        cargo_features: ""
        cflags: "-DADD_"
      - name: ILP64
        cargo_features: "--features ilp64"
        cflags: "-DADD_ -DUSE64BITINT"
```

### Linking Considerations

The Fortran BLAS library must match the integer size:

| Mode | OpenBLAS build | Symbol suffix |
|------|----------------|---------------|
| LP64 | Default build | `dgemm_` |
| ILP64 | `INTERFACE64=1` | `dgemm_64_` (or `dgemm_` with 64-bit ints) |

**Note:** OpenBLAS with `INTERFACE64=1` uses 64-bit integers but keeps the same symbol names.
Some distributions (e.g., Intel MKL) use `_64` suffix for ILP64 symbols.

### Test Matrix for Complete Coverage

| Precision | Integer Mode | Test Binary | Input |
|-----------|--------------|-------------|-------|
| Double | LP64 | xdcblat3 | din3 |
| Double | ILP64 | xdcblat3_64 | din3 |
| Single | LP64 | xscblat3 | sin3 |
| Single | ILP64 | xscblat3_64 | sin3 |
| Complex | LP64 | xzcblat3 | zin3 |
| Complex | ILP64 | xzcblat3_64 | zin3 |
| Complex Single | LP64 | xccblat3 | cin3 |
| Complex Single | ILP64 | xccblat3_64 | cin3 |

Total: 8 configurations for BLAS3 alone (× 3 levels = 24 configurations for full BLAS).

### Makefile Example with Both Modes

```makefile
# Common settings
CC = cc
COMMON_CFLAGS = -DADD_

# LP64 build
LP64_CFLAGS = $(COMMON_CFLAGS)
LP64_LIBS = -L../target/release -lcblas_trampoline -lopenblas

# ILP64 build
ILP64_CFLAGS = $(COMMON_CFLAGS) -DUSE64BITINT
ILP64_LIBS = -L../target/release -lcblas_trampoline_ilp64 -lopenblas64

# Targets
xdcblat3: c_dblat3c.c c_dblas3.c ...
	$(CC) $(LP64_CFLAGS) -o $@ $^ $(LP64_LIBS)

xdcblat3_64: c_dblat3c.c c_dblas3.c ...
	$(CC) $(ILP64_CFLAGS) -o $@ $^ $(ILP64_LIBS)

test-lp64: xdcblat3
	./xdcblat3 < din3

test-ilp64: xdcblat3_64
	./xdcblat3_64 < din3

test: test-lp64 test-ilp64
```

## References

- OpenBLAS ctest: https://github.com/OpenMathLib/OpenBLAS/tree/develop/ctest
- CBLAS reference: https://www.netlib.org/blas/blast-forum/cblas.tgz
- Original Fortran test suite: https://www.netlib.org/blas/
- OpenBLAS ILP64: https://github.com/OpenMathLib/OpenBLAS/wiki/User-Manual#building-openblas-64-bit-integers
