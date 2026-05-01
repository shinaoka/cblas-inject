//! Triangular band matrix-vector multiply (TBMV) - CBLAS interface.
//!
//! Computes: x = op(A) * x
//! where A is an n x n triangular band matrix with k super/sub-diagonals and x is a vector.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbmv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_ctbmv_for_ilp64_cblas, get_ctbmv_for_lp64_cblas, get_dtbmv_for_ilp64_cblas,
    get_dtbmv_for_lp64_cblas, get_stbmv_for_ilp64_cblas, get_stbmv_for_lp64_cblas,
    get_ztbmv_for_ilp64_cblas, get_ztbmv_for_lp64_cblas, CtbmvProvider, DtbmvProvider,
    StbmvProvider, ZtbmvProvider,
};
use crate::types::{
    blasint, diag_to_char, normalize_transpose_real, transpose_to_char, uplo_to_char,
    CblasColMajor, CblasConjNoTrans, CblasConjTrans, CblasLower, CblasNoTrans, CblasRowMajor,
    CblasTrans, CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
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
    n: i32,
    k: i32,
    a: *const f32,
    lda: i32,
    x: *mut f32,
    incx: i32,
) {
    let p = get_stbmv_for_lp64_cblas();
    match p {
        StbmvProvider::Lp64(stbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    stbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
        StbmvProvider::Ilp64(stbmv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    stbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_stbmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    k: i64,
    a: *const f32,
    lda: i64,
    x: *mut f32,
    incx: i64,
) {
    let p = get_stbmv_for_ilp64_cblas();
    if matches!(p, StbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_stbmv_64\0",
            [(5, n), (6, k), (8, lda), (10, incx)],
        )
        .is_none()
    {
        return;
    }

    match p {
        StbmvProvider::Ilp64(stbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    stbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
        StbmvProvider::Lp64(stbmv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    stbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
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
    n: i32,
    k: i32,
    a: *const f64,
    lda: i32,
    x: *mut f64,
    incx: i32,
) {
    let p = get_dtbmv_for_lp64_cblas();
    match p {
        DtbmvProvider::Lp64(dtbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    dtbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
        DtbmvProvider::Ilp64(dtbmv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    dtbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_dtbmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    k: i64,
    a: *const f64,
    lda: i64,
    x: *mut f64,
    incx: i64,
) {
    let p = get_dtbmv_for_ilp64_cblas();
    if matches!(p, DtbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dtbmv_64\0",
            [(5, n), (6, k), (8, lda), (10, incx)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DtbmvProvider::Ilp64(dtbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    dtbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
        DtbmvProvider::Lp64(dtbmv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    dtbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
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
    n: i32,
    k: i32,
    a: *const Complex32,
    lda: i32,
    x: *mut Complex32,
    incx: i32,
) {
    let p = get_ctbmv_for_lp64_cblas();
    match p {
        CtbmvProvider::Lp64(ctbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    ctbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
        CtbmvProvider::Ilp64(ctbmv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    ctbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ctbmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    k: i64,
    a: *const Complex32,
    lda: i64,
    x: *mut Complex32,
    incx: i64,
) {
    let p = get_ctbmv_for_ilp64_cblas();
    if matches!(p, CtbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ctbmv_64\0",
            [(5, n), (6, k), (8, lda), (10, incx)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CtbmvProvider::Ilp64(ctbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    ctbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
        CtbmvProvider::Lp64(ctbmv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    ctbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
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
    n: i32,
    k: i32,
    a: *const Complex64,
    lda: i32,
    x: *mut Complex64,
    incx: i32,
) {
    let p = get_ztbmv_for_lp64_cblas();
    match p {
        ZtbmvProvider::Lp64(ztbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    ztbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
        ZtbmvProvider::Ilp64(ztbmv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    ztbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ztbmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    n: i64,
    k: i64,
    a: *const Complex64,
    lda: i64,
    x: *mut Complex64,
    incx: i64,
) {
    let p = get_ztbmv_for_ilp64_cblas();
    if matches!(p, ZtbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ztbmv_64\0",
            [(5, n), (6, k), (8, lda), (10, incx)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZtbmvProvider::Ilp64(ztbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    ztbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
        ZtbmvProvider::Lp64(ztbmv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
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
                    ztbmv(
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &k,
                        a,
                        &lda,
                        x,
                        &incx,
                    );
                }
            }
        }
    }
}
