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
    get_ctpmv_for_ilp64_cblas, get_ctpmv_for_lp64_cblas, get_ctpsv_for_ilp64_cblas,
    get_ctpsv_for_lp64_cblas, get_dtpmv_for_ilp64_cblas, get_dtpmv_for_lp64_cblas,
    get_dtpsv_for_ilp64_cblas, get_dtpsv_for_lp64_cblas, get_stpmv_for_ilp64_cblas,
    get_stpmv_for_lp64_cblas, get_stpsv_for_ilp64_cblas, get_stpsv_for_lp64_cblas,
    get_ztpmv_for_ilp64_cblas, get_ztpmv_for_lp64_cblas, get_ztpsv_for_ilp64_cblas,
    get_ztpsv_for_lp64_cblas, CtpmvProvider, CtpsvProvider, DtpmvProvider, DtpsvProvider,
    StpmvProvider, StpsvProvider, ZtpmvProvider, ZtpsvProvider,
};
use crate::types::{
    diag_to_char, normalize_transpose_real, transpose_to_char, uplo_to_char, CblasColMajor,
    CblasConjNoTrans, CblasConjTrans, CblasLower, CblasNoTrans, CblasRowMajor, CblasTrans,
    CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
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
    n: i32,
    ap: *const f32,
    x: *mut f32,
    incx: i32,
) {
    let p = get_stpmv_for_lp64_cblas();
    match p {
        StpmvProvider::Lp64(stpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        StpmvProvider::Ilp64(stpmv) => {
            let n = n as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_stpmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    ap: *const f32,
    x: *mut f32,
    incx: i64,
) {
    let p = get_stpmv_for_ilp64_cblas();
    if matches!(p, StpmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_stpmv_64\0", [(5, n), (8, incx)]).is_none()
    {
        return;
    }

    match p {
        StpmvProvider::Ilp64(stpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        StpmvProvider::Lp64(stpmv) => {
            let n = n as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    stpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
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
    n: i32,
    ap: *const f64,
    x: *mut f64,
    incx: i32,
) {
    let p = get_dtpmv_for_lp64_cblas();
    match p {
        DtpmvProvider::Lp64(dtpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        DtpmvProvider::Ilp64(dtpmv) => {
            let n = n as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_dtpmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    ap: *const f64,
    x: *mut f64,
    incx: i64,
) {
    let p = get_dtpmv_for_ilp64_cblas();
    if matches!(p, DtpmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_dtpmv_64\0", [(5, n), (8, incx)]).is_none()
    {
        return;
    }

    match p {
        DtpmvProvider::Ilp64(dtpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        DtpmvProvider::Lp64(dtpmv) => {
            let n = n as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    dtpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
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
    n: i32,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: i32,
) {
    let p = get_ctpmv_for_lp64_cblas();
    match p {
        CtpmvProvider::Lp64(ctpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        CtpmvProvider::Ilp64(ctpmv) => {
            let n = n as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ctpmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: i64,
) {
    let p = get_ctpmv_for_ilp64_cblas();
    if matches!(p, CtpmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_ctpmv_64\0", [(5, n), (8, incx)]).is_none()
    {
        return;
    }

    match p {
        CtpmvProvider::Ilp64(ctpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        CtpmvProvider::Lp64(ctpmv) => {
            let n = n as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ctpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
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
    n: i32,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: i32,
) {
    let p = get_ztpmv_for_lp64_cblas();
    match p {
        ZtpmvProvider::Lp64(ztpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        ZtpmvProvider::Ilp64(ztpmv) => {
            let n = n as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ztpmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: i64,
) {
    let p = get_ztpmv_for_ilp64_cblas();
    if matches!(p, ZtpmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_ztpmv_64\0", [(5, n), (8, incx)]).is_none()
    {
        return;
    }

    match p {
        ZtpmvProvider::Ilp64(ztpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        ZtpmvProvider::Lp64(ztpmv) => {
            let n = n as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ztpmv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
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
    n: i32,
    ap: *const f32,
    x: *mut f32,
    incx: i32,
) {
    let p = get_stpsv_for_lp64_cblas();
    match p {
        StpsvProvider::Lp64(stpsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        StpsvProvider::Ilp64(stpsv) => {
            let n = n as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_stpsv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    ap: *const f32,
    x: *mut f32,
    incx: i64,
) {
    let p = get_stpsv_for_ilp64_cblas();
    if matches!(p, StpsvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_stpsv_64\0", [(5, n), (8, incx)]).is_none()
    {
        return;
    }

    match p {
        StpsvProvider::Ilp64(stpsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        StpsvProvider::Lp64(stpsv) => {
            let n = n as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    stpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
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
    n: i32,
    ap: *const f64,
    x: *mut f64,
    incx: i32,
) {
    let p = get_dtpsv_for_lp64_cblas();
    match p {
        DtpsvProvider::Lp64(dtpsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        DtpsvProvider::Ilp64(dtpsv) => {
            let n = n as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_dtpsv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    ap: *const f64,
    x: *mut f64,
    incx: i64,
) {
    let p = get_dtpsv_for_ilp64_cblas();
    if matches!(p, DtpsvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_dtpsv_64\0", [(5, n), (8, incx)]).is_none()
    {
        return;
    }

    match p {
        DtpsvProvider::Ilp64(dtpsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        DtpsvProvider::Lp64(dtpsv) => {
            let n = n as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    dtpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
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
    n: i32,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: i32,
) {
    let p = get_ctpsv_for_lp64_cblas();
    match p {
        CtpsvProvider::Lp64(ctpsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        CtpsvProvider::Ilp64(ctpsv) => {
            let n = n as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ctpsv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: i64,
) {
    let p = get_ctpsv_for_ilp64_cblas();
    if matches!(p, CtpsvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_ctpsv_64\0", [(5, n), (8, incx)]).is_none()
    {
        return;
    }

    match p {
        CtpsvProvider::Ilp64(ctpsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        CtpsvProvider::Lp64(ctpsv) => {
            let n = n as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ctpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
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
    n: i32,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: i32,
) {
    let p = get_ztpsv_for_lp64_cblas();
    match p {
        ZtpsvProvider::Lp64(ztpsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        ZtpsvProvider::Ilp64(ztpsv) => {
            let n = n as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ztpsv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: i64,
) {
    let p = get_ztpsv_for_ilp64_cblas();
    if matches!(p, ZtpsvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_ztpsv_64\0", [(5, n), (8, incx)]).is_none()
    {
        return;
    }

    match p {
        ZtpsvProvider::Ilp64(ztpsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
        ZtpsvProvider::Lp64(ztpsv) => {
            let n = n as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
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
                    ztpsv(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
                }
            }
        }
    }
}
