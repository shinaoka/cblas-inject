//! Triangular packed matrix-vector multiply (TPMV) and solve (TPSV) - CBLAS interface.
//!
//! TPMV computes: x = op(A) * x
//! TPSV solves: op(A) * x = b
//! where A is an n x n triangular matrix stored in packed format.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tpmv.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tpsv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_ctpmv, get_ctpsv, get_dtpmv, get_dtpsv, get_stpmv, get_stpsv, get_ztpmv, get_ztpsv,
};
use crate::types::{
    blasint, diag_to_char, transpose_to_char, uplo_to_char, CblasColMajor, CblasConjTrans,
    CblasLower, CblasNoTrans, CblasRowMajor, CblasTrans, CblasUpper, CBLAS_DIAG, CBLAS_ORDER,
    CBLAS_TRANSPOSE, CBLAS_UPLO,
};

// =============================================================================
// TPMV: Triangular Packed Matrix-Vector Multiply
// =============================================================================

/// Single precision triangular packed matrix-vector multiply.
///
/// Computes x = op(A) * x where A is triangular in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - stpmv must be registered via `register_stpmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_stpmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    ap: *const f32,
    x: *mut f32,
    incx: blasint,
) {
    let stpmv = get_stpmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
            stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
    }
}

/// Double precision triangular packed matrix-vector multiply.
///
/// Computes x = op(A) * x where A is triangular in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - dtpmv must be registered via `register_dtpmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_dtpmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    ap: *const f64,
    x: *mut f64,
    incx: blasint,
) {
    let dtpmv = get_dtpmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
            dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
    }
}

/// Single precision complex triangular packed matrix-vector multiply.
///
/// Computes x = op(A) * x where A is triangular in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - ctpmv must be registered via `register_ctpmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ctpmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: blasint,
) {
    let ctpmv = get_ctpmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // For complex: ConjTrans stays ConjTrans
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => CblasConjTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
    }
}

/// Double precision complex triangular packed matrix-vector multiply.
///
/// Computes x = op(A) * x where A is triangular in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - ztpmv must be registered via `register_ztpmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ztpmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: blasint,
) {
    let ztpmv = get_ztpmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // For complex: ConjTrans stays ConjTrans
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => CblasConjTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
    }
}

// =============================================================================
// TPSV: Triangular Packed Solve
// =============================================================================

/// Single precision triangular packed solve.
///
/// Solves op(A) * x = b where A is triangular in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - stpsv must be registered via `register_stpsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_stpsv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    ap: *const f32,
    x: *mut f32,
    incx: blasint,
) {
    let stpsv = get_stpsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
            stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
    }
}

/// Double precision triangular packed solve.
///
/// Solves op(A) * x = b where A is triangular in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - dtpsv must be registered via `register_dtpsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_dtpsv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    ap: *const f64,
    x: *mut f64,
    incx: blasint,
) {
    let dtpsv = get_dtpsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
            dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
    }
}

/// Single precision complex triangular packed solve.
///
/// Solves op(A) * x = b where A is triangular in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - ctpsv must be registered via `register_ctpsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ctpsv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: blasint,
) {
    let ctpsv = get_ctpsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // For complex: ConjTrans stays ConjTrans
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => CblasConjTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
    }
}

/// Double precision complex triangular packed solve.
///
/// Solves op(A) * x = b where A is triangular in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - ztpsv must be registered via `register_ztpsv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ztpsv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: blasint,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: blasint,
) {
    let ztpsv = get_ztpsv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // For complex: ConjTrans stays ConjTrans
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => CblasConjTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            let diag_char = diag_to_char(diag);
            ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
        }
    }
}
