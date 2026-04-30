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

use std::ffi::{c_char, c_int};

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_cgemm, get_dgemm_for_current_cblas, get_dgemm_for_ilp64_cblas, get_sgemm,
    get_zgemm_for_current_cblas, get_zgemm_for_ilp64_cblas, BlasInt32, BlasInt64, DgemmProvider,
    ZgemmProvider,
};
use crate::types::{
    blasint, transpose_to_char, CblasColMajor, CblasRowMajor, CBLAS_ORDER, CBLAS_TRANSPOSE,
};
use crate::xerbla::cblas_xerbla;

const CBLAS_DGEMM_ROUTINE: &[u8] = b"cblas_dgemm\0";
const CBLAS_ZGEMM_ROUTINE: &[u8] = b"cblas_zgemm\0";
const CBLAS_DGEMM_64_ROUTINE: &[u8] = b"cblas_dgemm_64\0";
const CBLAS_ZGEMM_64_ROUTINE: &[u8] = b"cblas_zgemm_64\0";

#[cfg(not(feature = "ilp64"))]
#[inline]
fn to_lp64(_routine: &[u8], _param: blasint, value: blasint) -> Option<BlasInt32> {
    Some(value)
}

#[cfg(feature = "ilp64")]
#[inline]
fn to_lp64(routine: &[u8], param: blasint, value: blasint) -> Option<BlasInt32> {
    match BlasInt32::try_from(value) {
        Ok(value) => Some(value),
        Err(_) => {
            unsafe {
                cblas_xerbla(param as c_int, routine.as_ptr().cast(), std::ptr::null());
            }
            None
        }
    }
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

#[inline]
fn to_lp64_i64(routine: &[u8], param: c_int, value: i64) -> Option<BlasInt32> {
    match BlasInt32::try_from(value) {
        Ok(value) => Some(value),
        Err(_) => {
            unsafe {
                cblas_xerbla(param, routine.as_ptr().cast(), std::ptr::null());
            }
            None
        }
    }
}

#[inline]
fn check_lp64_gemm_i64(
    routine: &[u8],
    m: i64,
    n: i64,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
) -> bool {
    to_lp64_i64(routine, 4, m).is_some()
        && to_lp64_i64(routine, 5, n).is_some()
        && to_lp64_i64(routine, 6, k).is_some()
        && to_lp64_i64(routine, 9, lda).is_some()
        && to_lp64_i64(routine, 11, ldb).is_some()
        && to_lp64_i64(routine, 14, ldc).is_some()
}

#[inline]
fn unchecked_lp64_i64(value: i64) -> BlasInt32 {
    debug_assert!(BlasInt32::try_from(value).is_ok());
    value as BlasInt32
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
) -> bool {
    match provider {
        DgemmProvider::Lp64(dgemm) => {
            let Some(m) = to_lp64(CBLAS_DGEMM_ROUTINE, 4, m) else {
                return false;
            };
            let Some(n) = to_lp64(CBLAS_DGEMM_ROUTINE, 5, n) else {
                return false;
            };
            let Some(k) = to_lp64(CBLAS_DGEMM_ROUTINE, 6, k) else {
                return false;
            };
            let Some(lda) = to_lp64(CBLAS_DGEMM_ROUTINE, 9, lda) else {
                return false;
            };
            let Some(ldb) = to_lp64(CBLAS_DGEMM_ROUTINE, 11, ldb) else {
                return false;
            };
            let Some(ldc) = to_lp64(CBLAS_DGEMM_ROUTINE, 14, ldc) else {
                return false;
            };
            unsafe {
                dgemm(
                    &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                );
            }
            true
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
            true
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
) -> bool {
    match provider {
        ZgemmProvider::Lp64(zgemm) => {
            let Some(m) = to_lp64(CBLAS_ZGEMM_ROUTINE, 4, m) else {
                return false;
            };
            let Some(n) = to_lp64(CBLAS_ZGEMM_ROUTINE, 5, n) else {
                return false;
            };
            let Some(k) = to_lp64(CBLAS_ZGEMM_ROUTINE, 6, k) else {
                return false;
            };
            let Some(lda) = to_lp64(CBLAS_ZGEMM_ROUTINE, 9, lda) else {
                return false;
            };
            let Some(ldb) = to_lp64(CBLAS_ZGEMM_ROUTINE, 11, ldb) else {
                return false;
            };
            let Some(ldc) = to_lp64(CBLAS_ZGEMM_ROUTINE, 14, ldc) else {
                return false;
            };
            unsafe {
                zgemm(
                    &transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                );
            }
            true
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
            true
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn call_dgemm_provider_i64(
    provider: DgemmProvider,
    transa: c_char,
    transb: c_char,
    m: i64,
    n: i64,
    k: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    b: *const f64,
    ldb: i64,
    beta: f64,
    c: *mut f64,
    ldc: i64,
) {
    match provider {
        DgemmProvider::Lp64(dgemm) => {
            let m = unchecked_lp64_i64(m);
            let n = unchecked_lp64_i64(n);
            let k = unchecked_lp64_i64(k);
            let lda = unchecked_lp64_i64(lda);
            let ldb = unchecked_lp64_i64(ldb);
            let ldc = unchecked_lp64_i64(ldc);
            unsafe {
                dgemm(
                    &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                );
            }
        }
        DgemmProvider::Ilp64(dgemm) => unsafe {
            dgemm(
                &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
            );
        },
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn call_zgemm_provider_i64(
    provider: ZgemmProvider,
    transa: c_char,
    transb: c_char,
    m: i64,
    n: i64,
    k: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    b: *const Complex64,
    ldb: i64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i64,
) {
    match provider {
        ZgemmProvider::Lp64(zgemm) => {
            let m = unchecked_lp64_i64(m);
            let n = unchecked_lp64_i64(n);
            let k = unchecked_lp64_i64(k);
            let lda = unchecked_lp64_i64(lda);
            let ldb = unchecked_lp64_i64(ldb);
            let ldc = unchecked_lp64_i64(ldc);
            unsafe {
                zgemm(
                    &transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                );
            }
        }
        ZgemmProvider::Ilp64(zgemm) => unsafe {
            zgemm(
                &transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc,
            );
        },
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

/// Double precision general matrix multiply with ILP64 CBLAS integer ABI.
///
/// This symbol always accepts 64-bit BLAS integer arguments, independent of the
/// crate's `ilp64` feature.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dgemm must be registered via `register_dgemm` or the C registration API
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_dgemm_64(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: i64,
    n: i64,
    k: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    b: *const f64,
    ldb: i64,
    beta: f64,
    c: *mut f64,
    ldc: i64,
) {
    let dgemm = get_dgemm_for_ilp64_cblas();

    if matches!(dgemm, DgemmProvider::Lp64(_))
        && !check_lp64_gemm_i64(CBLAS_DGEMM_64_ROUTINE, m, n, k, lda, ldb, ldc)
    {
        return;
    }

    match order {
        CblasColMajor => {
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            call_dgemm_provider_i64(
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
            let transa_char = transpose_to_char(transb);
            let transb_char = transpose_to_char(transa);
            call_dgemm_provider_i64(
                dgemm,
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

/// Double precision complex general matrix multiply with ILP64 CBLAS integer ABI.
///
/// This symbol always accepts 64-bit BLAS integer arguments, independent of the
/// crate's `ilp64` feature. Complex scalar parameters follow CBLAS and remain
/// pointer arguments.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgemm must be registered via `register_zgemm` or the C registration API
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_zgemm_64(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: i64,
    n: i64,
    k: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    b: *const Complex64,
    ldb: i64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i64,
) {
    let zgemm = get_zgemm_for_ilp64_cblas();

    if matches!(zgemm, ZgemmProvider::Lp64(_))
        && !check_lp64_gemm_i64(CBLAS_ZGEMM_64_ROUTINE, m, n, k, lda, ldb, ldc)
    {
        return;
    }

    match order {
        CblasColMajor => {
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            call_zgemm_provider_i64(
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
            call_zgemm_provider_i64(
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
