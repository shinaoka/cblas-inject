//! Triangular matrix-vector multiply (TRMV) - CBLAS interface.
//!
//! Computes: x = op(A) * x
//! where A is an n x n triangular matrix and x is a vector.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trmv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_ctrmv_for_lp64_cblas, get_ctrmv_for_ilp64_cblas, CtrmvProvider,
    get_dtrmv_for_lp64_cblas, get_dtrmv_for_ilp64_cblas, DtrmvProvider,
    get_strmv_for_lp64_cblas, get_strmv_for_ilp64_cblas, StrmvProvider,
    get_ztrmv_for_lp64_cblas, get_ztrmv_for_ilp64_cblas, ZtrmvProvider,
};
use crate::types::{
    blasint, diag_to_char, normalize_transpose_real, transpose_to_char, uplo_to_char,
    CblasColMajor, CblasConjNoTrans, CblasConjTrans, CblasLower, CblasNoTrans, CblasRowMajor,
    CblasTrans, CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Single precision triangular matrix-vector multiply.
///
/// Computes x = op(A) * x where A is triangular.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - strmv must be registered via `register_strmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_strmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i32,
    a: *const f32,
    lda: i32,
    x: *mut f32,
    incx: i32,
) {
    let p = get_strmv_for_lp64_cblas();
    match p {
        StrmvProvider::Lp64(strmv) => {

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            strmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trmv.c
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
            strmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
        StrmvProvider::Ilp64(strmv) => {
            let n = n as i64; let lda = lda as i64; let incx = incx as i64;

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            strmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trmv.c
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
            strmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_strmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    a: *const f32,
    lda: i64,
    x: *mut f32,
    incx: i64,
) {
    let p = get_strmv_for_ilp64_cblas();
    match p {
        StrmvProvider::Ilp64(strmv) => {

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            strmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trmv.c
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
            strmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
        StrmvProvider::Lp64(strmv) => {
            let n = n as i32; let lda = lda as i32; let incx = incx as i32;

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            strmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
        CblasRowMajor => {
            // Row-major: invert uplo and trans
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trmv.c
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
            strmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
    }
}

/// Double precision triangular matrix-vector multiply.
///
/// Computes x = op(A) * x where A is triangular.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtrmv must be registered via `register_dtrmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_dtrmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i32,
    a: *const f64,
    lda: i32,
    x: *mut f64,
    incx: i32,
) {
    let p = get_dtrmv_for_lp64_cblas();
    match p {
        DtrmvProvider::Lp64(dtrmv) => {

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            dtrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            dtrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
        DtrmvProvider::Ilp64(dtrmv) => {
            let n = n as i64; let lda = lda as i64; let incx = incx as i64;

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            dtrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            dtrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_dtrmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    a: *const f64,
    lda: i64,
    x: *mut f64,
    incx: i64,
) {
    let p = get_dtrmv_for_ilp64_cblas();
    match p {
        DtrmvProvider::Ilp64(dtrmv) => {

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            dtrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            dtrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
        DtrmvProvider::Lp64(dtrmv) => {
            let n = n as i32; let lda = lda as i32; let incx = incx as i32;

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(normalize_transpose_real(trans));
            let diag_char = diag_to_char(diag);
            dtrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            dtrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
    }
}

/// Single precision complex triangular matrix-vector multiply.
///
/// Computes x = op(A) * x where A is triangular.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ctrmv must be registered via `register_ctrmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ctrmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i32,
    a: *const Complex32,
    lda: i32,
    x: *mut Complex32,
    incx: i32,
) {
    let p = get_ctrmv_for_lp64_cblas();
    match p {
        CtrmvProvider::Lp64(ctrmv) => {

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            ctrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
        CtrmvProvider::Ilp64(ctrmv) => {
            let n = n as i64; let lda = lda as i64; let incx = incx as i64;

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            ctrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ctrmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    a: *const Complex32,
    lda: i64,
    x: *mut Complex32,
    incx: i64,
) {
    let p = get_ctrmv_for_ilp64_cblas();
    match p {
        CtrmvProvider::Ilp64(ctrmv) => {

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            ctrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
        CtrmvProvider::Lp64(ctrmv) => {
            let n = n as i32; let lda = lda as i32; let incx = incx as i32;

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ctrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            ctrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
    }
}

/// Double precision complex triangular matrix-vector multiply.
///
/// Computes x = op(A) * x where A is triangular.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ztrmv must be registered via `register_ztrmv`
#[no_mangle]
pub unsafe extern "C" fn cblas_ztrmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i32,
    a: *const Complex64,
    lda: i32,
    x: *mut Complex64,
    incx: i32,
) {
    let p = get_ztrmv_for_lp64_cblas();
    match p {
        ZtrmvProvider::Lp64(ztrmv) => {

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            ztrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
        ZtrmvProvider::Ilp64(ztrmv) => {
            let n = n as i64; let lda = lda as i64; let incx = incx as i64;

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            ztrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ztrmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    a: *const Complex64,
    lda: i64,
    x: *mut Complex64,
    incx: i64,
) {
    let p = get_ztrmv_for_ilp64_cblas();
    match p {
        ZtrmvProvider::Ilp64(ztrmv) => {

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            ztrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
        ZtrmvProvider::Lp64(ztrmv) => {
            let n = n as i32; let lda = lda as i32; let incx = incx as i32;

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            ztrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
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
            ztrmv(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
        }
    }
        }
    }
}
