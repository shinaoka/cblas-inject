//! Triangular band solve (TBSV) - CBLAS interface.
//!
//! Solves: op(A) * x = b
//! where A is an n x n triangular band matrix with k super/sub-diagonals
//! and x is the solution vector. The vector x overwrites b.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbsv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{get_ctbsv, get_dtbsv, get_stbsv, get_ztbsv};
use crate::types::{
    blasint, diag_to_char, transpose_to_char, uplo_to_char, CblasColMajor, CblasConjTrans,
    CblasLower, CblasNoTrans, CblasRowMajor, CblasTrans, CblasUpper, CBLAS_DIAG, CBLAS_ORDER,
    CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Single precision triangular band solve.
///
/// Solves op(A) * x = b where A is a triangular band matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - stbsv must be registered via `register_stbsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_stbsv(
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
    let stbsv = get_stbsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            stbsv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbsv.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => CblasNoTrans, // For real types, ConjTrans = Trans
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            stbsv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
    }
}

/// Double precision triangular band solve.
///
/// Solves op(A) * x = b where A is a triangular band matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtbsv must be registered via `register_dtbsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_dtbsv(
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
    let dtbsv = get_dtbsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            dtbsv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => CblasNoTrans, // For real types, ConjTrans = Trans
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            dtbsv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
    }
}

/// Single precision complex triangular band solve.
///
/// Solves op(A) * x = b where A is a triangular band matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ctbsv must be registered via `register_ctbsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ctbsv(
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
    let ctbsv = get_ctbsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctbsv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // For complex: ConjTrans stays ConjTrans (conjugate is preserved)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => CblasConjTrans, // Conjugate transpose stays as conj trans for complex
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ctbsv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
    }
}

/// Double precision complex triangular band solve.
///
/// Solves op(A) * x = b where A is a triangular band matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ztbsv must be registered via `register_ztbsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ztbsv(
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
    let ztbsv = get_ztbsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztbsv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // For complex: ConjTrans stays ConjTrans (conjugate is preserved)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => CblasConjTrans, // Conjugate transpose stays as conj trans for complex
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ztbsv(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
        }
    }
}
