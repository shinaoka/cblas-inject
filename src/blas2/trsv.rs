//! Triangular solve (TRSV) - CBLAS interface.
//!
//! Solves: op(A) * x = b
//! where A is an n x n triangular matrix and x is the solution vector.
//! The vector x overwrites b.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trsv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{get_ctrsv, get_dtrsv, get_strsv, get_ztrsv};
use crate::types::{
    blasint, diag_to_char, normalize_transpose_real, transpose_to_char, uplo_to_char,
    CblasColMajor, CblasConjNoTrans, CblasConjTrans, CblasLower, CblasNoTrans, CblasRowMajor,
    CblasTrans, CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Single precision triangular solve.
///
/// Solves op(A) * x = b where A is triangular.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - strsv must be registered via `register_strsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_strsv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    a: *const f32,
    lda: blasint,
    x: *mut f32,
    incx: blasint,
) {
    let strsv = get_strsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            strsv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trsv.c
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
            strsv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
}

/// Double precision triangular solve.
///
/// Solves op(A) * x = b where A is triangular.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtrsv must be registered via `register_dtrsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_dtrsv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    a: *const f64,
    lda: blasint,
    x: *mut f64,
    incx: blasint,
) {
    let dtrsv = get_dtrsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            dtrsv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            dtrsv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
}

/// Single precision complex triangular solve.
///
/// Solves op(A) * x = b where A is triangular.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ctrsv must be registered via `register_ctrsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ctrsv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    a: *const Complex32,
    lda: blasint,
    x: *mut Complex32,
    incx: blasint,
) {
    let ctrsv = get_ctrsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctrsv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
                CblasConjNoTrans => CblasConjTrans,
                CblasConjTrans => CblasConjNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ctrsv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
}

/// Double precision complex triangular solve.
///
/// Solves op(A) * x = b where A is triangular.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ztrsv must be registered via `register_ztrsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ztrsv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    a: *const Complex64,
    lda: blasint,
    x: *mut Complex64,
    incx: blasint,
) {
    let ztrsv = get_ztrsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztrsv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
                CblasConjNoTrans => CblasConjTrans,
                CblasConjTrans => CblasConjNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ztrsv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
}
