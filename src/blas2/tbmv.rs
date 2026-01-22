//! Triangular band matrix-vector multiply (TBMV) - CBLAS interface.
//!
//! Computes: x = op(A) * x
//! where A is an n x n triangular band matrix with k super/sub-diagonals and x is a vector.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbmv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{get_ctbmv, get_dtbmv, get_stbmv, get_ztbmv};
use crate::types::{
    blasint, diag_to_char, normalize_transpose_real, transpose_to_char, uplo_to_char, CblasColMajor,
    CblasConjNoTrans, CblasConjTrans, CblasLower, CblasNoTrans, CblasRowMajor, CblasTrans,
    CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Single precision triangular band matrix-vector multiply.
///
/// Computes x = op(A) * x where A is a triangular band matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - stbmv must be registered via `register_stbmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_stbmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    k: blasint,
    a: *const f32,
    lda: blasint,
    x: *mut f32,
    incx: blasint,
) {
    let stbmv = get_stbmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            stbmv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbmv.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match normalize_transpose_real(trans) {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                _ => unreachable!(),
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            stbmv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
    }
}

/// Double precision triangular band matrix-vector multiply.
///
/// Computes x = op(A) * x where A is a triangular band matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtbmv must be registered via `register_dtbmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_dtbmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    k: blasint,
    a: *const f64,
    lda: blasint,
    x: *mut f64,
    incx: blasint,
) {
    let dtbmv = get_dtbmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            dtbmv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match normalize_transpose_real(trans) {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                _ => unreachable!(),
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            dtbmv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
    }
}

/// Single precision complex triangular band matrix-vector multiply.
///
/// Computes x = op(A) * x where A is a triangular band matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ctbmv must be registered via `register_ctbmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ctbmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    k: blasint,
    a: *const Complex32,
    lda: blasint,
    x: *mut Complex32,
    incx: blasint,
) {
    let ctbmv = get_ctbmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctbmv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // For complex: flip transpose with conjugation preserved (OpenBLAS)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjNoTrans => CblasConjTrans,
                CblasConjTrans => CblasConjNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ctbmv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
    }
}

/// Double precision complex triangular band matrix-vector multiply.
///
/// Computes x = op(A) * x where A is a triangular band matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ztbmv must be registered via `register_ztbmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ztbmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    k: blasint,
    a: *const Complex64,
    lda: blasint,
    x: *mut Complex64,
    incx: blasint,
) {
    let ztbmv = get_ztbmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztbmv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // For complex: flip transpose with conjugation preserved (OpenBLAS)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjNoTrans => CblasConjTrans,
                CblasConjTrans => CblasConjNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ztbmv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
    }
}
