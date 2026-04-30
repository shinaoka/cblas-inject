//! General matrix multiply (GEMM) - CBLAS interface.
//!
//! Computes: C = alpha * op(A) * op(B) + beta * C
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c>
//!
//! For row-major layout, we swap A↔B, m↔n, lda↔ldb, TransA↔TransB.
//! The transpose flags are NOT inverted, just swapped.

use std::ffi::c_char;

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_cgemm, get_dgemm_for_current_cblas, get_sgemm, get_zgemm_for_current_cblas, BlasInt32,
    BlasInt64, DgemmProvider, ZgemmProvider,
};
use crate::types::{
    blasint, transpose_to_char, CblasColMajor, CblasRowMajor, CBLAS_ORDER, CBLAS_TRANSPOSE,
};

#[cfg(feature = "ilp64")]
#[inline]
fn to_lp64(function: &str, name: &str, value: blasint) -> BlasInt32 {
    BlasInt32::try_from(value).unwrap_or_else(|_| {
        panic!("{function}: {name}={value} cannot be represented by an LP64 BLAS provider")
    })
}

#[cfg(not(feature = "ilp64"))]
#[inline]
fn to_lp64(_function: &str, _name: &str, value: blasint) -> BlasInt32 {
    value
}

#[cfg(feature = "ilp64")]
#[inline]
fn to_ilp64(value: blasint) -> BlasInt64 {
    value
}

#[cfg(not(feature = "ilp64"))]
#[inline]
fn to_ilp64(value: blasint) -> BlasInt64 {
    BlasInt64::from(value)
}

#[allow(clippy::too_many_arguments)]
unsafe fn call_dgemm_provider(
    provider: DgemmProvider,
    transa: c_char,
    transb: c_char,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: f64,
    a: *const f64,
    lda: blasint,
    b: *const f64,
    ldb: blasint,
    beta: f64,
    c: *mut f64,
    ldc: blasint,
) {
    match provider {
        DgemmProvider::Lp64(dgemm) => {
            let m = to_lp64("cblas_dgemm", "m", m);
            let n = to_lp64("cblas_dgemm", "n", n);
            let k = to_lp64("cblas_dgemm", "k", k);
            let lda = to_lp64("cblas_dgemm", "lda", lda);
            let ldb = to_lp64("cblas_dgemm", "ldb", ldb);
            let ldc = to_lp64("cblas_dgemm", "ldc", ldc);
            unsafe {
                dgemm(
                    &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                );
            }
        }
        DgemmProvider::Ilp64(dgemm) => {
            let m = to_ilp64(m);
            let n = to_ilp64(n);
            let k = to_ilp64(k);
            let lda = to_ilp64(lda);
            let ldb = to_ilp64(ldb);
            let ldc = to_ilp64(ldc);
            unsafe {
                dgemm(
                    &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn call_zgemm_provider(
    provider: ZgemmProvider,
    transa: c_char,
    transb: c_char,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: blasint,
    b: *const Complex64,
    ldb: blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: blasint,
) {
    match provider {
        ZgemmProvider::Lp64(zgemm) => {
            let m = to_lp64("cblas_zgemm", "m", m);
            let n = to_lp64("cblas_zgemm", "n", n);
            let k = to_lp64("cblas_zgemm", "k", k);
            let lda = to_lp64("cblas_zgemm", "lda", lda);
            let ldb = to_lp64("cblas_zgemm", "ldb", ldb);
            let ldc = to_lp64("cblas_zgemm", "ldc", ldc);
            unsafe {
                zgemm(
                    &transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                );
            }
        }
        ZgemmProvider::Ilp64(zgemm) => {
            let m = to_ilp64(m);
            let n = to_ilp64(n);
            let k = to_ilp64(k);
            let lda = to_ilp64(lda);
            let ldb = to_ilp64(ldb);
            let ldc = to_ilp64(ldc);
            unsafe {
                zgemm(
                    &transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                );
            }
        }
    }
}

/// Double precision general matrix multiply.
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dgemm must be registered via `register_dgemm`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_dgemm(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: f64,
    a: *const f64,
    lda: blasint,
    b: *const f64,
    ldb: blasint,
    beta: f64,
    c: *mut f64,
    ldc: blasint,
) {
    let dgemm = get_dgemm_for_current_cblas();

    match order {
        CblasColMajor => {
            // Column-major: call Fortran directly
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            call_dgemm_provider(
                dgemm,
                transa_char,
                transb_char,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: swap A↔B, m↔n, lda↔ldb, TransA↔TransB
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c#L489-L537
            let transa_char = transpose_to_char(transb); // TransB becomes transa
            let transb_char = transpose_to_char(transa); // TransA becomes transb
            call_dgemm_provider(
                dgemm,
                transa_char,
                transb_char,
                n, // swapped: m -> n
                m, // swapped: n -> m
                k,
                alpha,
                b,   // swapped: a -> b
                ldb, // swapped: lda -> ldb
                a,   // swapped: b -> a
                lda, // swapped: ldb -> lda
                beta,
                c,
                ldc,
            );
        }
    }
}

/// Single precision general matrix multiply.
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - sgemm must be registered via `register_sgemm`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_sgemm(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: f32,
    a: *const f32,
    lda: blasint,
    b: *const f32,
    ldb: blasint,
    beta: f32,
    c: *mut f32,
    ldc: blasint,
) {
    let sgemm = get_sgemm();

    match order {
        CblasColMajor => {
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            sgemm(
                &transa_char,
                &transb_char,
                &m,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            let transa_char = transpose_to_char(transb);
            let transb_char = transpose_to_char(transa);
            sgemm(
                &transa_char,
                &transb_char,
                &n,
                &m,
                &k,
                &alpha,
                b,
                &ldb,
                a,
                &lda,
                &beta,
                c,
                &ldc,
            );
        }
    }
}

/// Double precision complex general matrix multiply.
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgemm must be registered via `register_zgemm`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_zgemm(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: blasint,
    b: *const Complex64,
    ldb: blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: blasint,
) {
    let zgemm = get_zgemm_for_current_cblas();

    match order {
        CblasColMajor => {
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            call_zgemm_provider(
                zgemm,
                transa_char,
                transb_char,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            );
        }
        CblasRowMajor => {
            let transa_char = transpose_to_char(transb);
            let transb_char = transpose_to_char(transa);
            call_zgemm_provider(
                zgemm,
                transa_char,
                transb_char,
                n,
                m,
                k,
                alpha,
                b,
                ldb,
                a,
                lda,
                beta,
                c,
                ldc,
            );
        }
    }
}

/// Single precision complex general matrix multiply.
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cgemm must be registered via `register_cgemm`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_cgemm(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: blasint,
    b: *const Complex32,
    ldb: blasint,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: blasint,
) {
    let cgemm = get_cgemm();

    match order {
        CblasColMajor => {
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            cgemm(
                &transa_char,
                &transb_char,
                &m,
                &n,
                &k,
                alpha,
                a,
                &lda,
                b,
                &ldb,
                beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            let transa_char = transpose_to_char(transb);
            let transb_char = transpose_to_char(transa);
            cgemm(
                &transa_char,
                &transb_char,
                &n,
                &m,
                &k,
                alpha,
                b,
                &ldb,
                a,
                &lda,
                beta,
                c,
                &ldc,
            );
        }
    }
}
