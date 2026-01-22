//! Symmetric rank-2k update (SYR2K) - CBLAS interface.
//!
//! Computes: C = alpha * A * B^T + alpha * B * A^T + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^T * B + alpha * B^T * A + beta * C  (Trans=Trans)
//! where C is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2k.c>

use crate::backend::{get_csyr2k, get_dsyr2k, get_ssyr2k, get_zsyr2k};
use crate::types::{
    blasint, transpose_to_char, uplo_to_char, CblasColMajor, CblasLower, CblasNoTrans,
    CblasRowMajor, CblasTrans, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Double precision symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyr2k must be registered via `register_dsyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
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
    let dsyr2k = get_dsyr2k();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            dsyr2k(
                &uplo_char,
                &trans_char,
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
            // Row-major: invert trans, invert uplo (same as syrk)
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2k.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            dsyr2k(
                &uplo_char,
                &trans_char,
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
    }
}

/// Single precision symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyr2k must be registered via `register_ssyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
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
    let ssyr2k = get_ssyr2k();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            ssyr2k(
                &uplo_char,
                &trans_char,
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
            // Row-major: invert trans, invert uplo (same as syrk)
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2k.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            ssyr2k(
                &uplo_char,
                &trans_char,
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
    }
}

/// Single precision complex symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csyr2k must be registered via `register_csyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
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
    let csyr2k = get_csyr2k();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            csyr2k(
                &uplo_char,
                &trans_char,
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
            // Row-major: invert trans, invert uplo
            // For complex symmetric, Trans stays Trans (not ConjTrans)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            csyr2k(
                &uplo_char,
                &trans_char,
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
    }
}

/// Double precision complex symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsyr2k must be registered via `register_zsyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
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
    let zsyr2k = get_zsyr2k();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            zsyr2k(
                &uplo_char,
                &trans_char,
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
            // Row-major: invert trans, invert uplo
            // For complex symmetric, Trans stays Trans (not ConjTrans)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            zsyr2k(
                &uplo_char,
                &trans_char,
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
    }
}
