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

use crate::backend::{
    get_ctbsv_for_ilp64_cblas, get_ctbsv_for_lp64_cblas, get_dtbsv_for_ilp64_cblas,
    get_dtbsv_for_lp64_cblas, get_stbsv_for_ilp64_cblas, get_stbsv_for_lp64_cblas,
    get_ztbsv_for_ilp64_cblas, get_ztbsv_for_lp64_cblas, CtbsvProvider, DtbsvProvider,
    StbsvProvider, ZtbsvProvider,
};
use crate::types::{
    diag_to_char, normalize_transpose_real, transpose_to_char, uplo_to_char, CblasColMajor,
    CblasConjNoTrans, CblasConjTrans, CblasLower, CblasNoTrans, CblasRowMajor, CblasTrans,
    CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
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
    n: i32,
    k: i32,
    a: *const f32,
    lda: i32,
    x: *mut f32,
    incx: i32,
) {
    let p = get_stbsv_for_lp64_cblas();
    match p {
        StbsvProvider::Lp64(stbsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stbsv(
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
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbsv.c
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
                    stbsv(
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
        StbsvProvider::Ilp64(stbsv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stbsv(
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
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbsv.c
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
                    stbsv(
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
pub unsafe extern "C" fn cblas_stbsv_64(
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
    let p = get_stbsv_for_ilp64_cblas();
    if matches!(p, StbsvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_stbsv_64\0",
            [(5, n), (6, k), (8, lda), (10, incx)],
        )
        .is_none()
    {
        return;
    }

    match p {
        StbsvProvider::Ilp64(stbsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stbsv(
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
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbsv.c
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
                    stbsv(
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
        StbsvProvider::Lp64(stbsv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    stbsv(
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
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/tbsv.c
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
                    stbsv(
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
    n: i32,
    k: i32,
    a: *const f64,
    lda: i32,
    x: *mut f64,
    incx: i32,
) {
    let p = get_dtbsv_for_lp64_cblas();
    match p {
        DtbsvProvider::Lp64(dtbsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtbsv(
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
                    dtbsv(
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
        DtbsvProvider::Ilp64(dtbsv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtbsv(
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
                    dtbsv(
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
pub unsafe extern "C" fn cblas_dtbsv_64(
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
    let p = get_dtbsv_for_ilp64_cblas();
    if matches!(p, DtbsvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dtbsv_64\0",
            [(5, n), (6, k), (8, lda), (10, incx)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DtbsvProvider::Ilp64(dtbsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtbsv(
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
                    dtbsv(
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
        DtbsvProvider::Lp64(dtbsv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    let diag_char = diag_to_char(diag);
                    dtbsv(
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
                    dtbsv(
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
    n: i32,
    k: i32,
    a: *const Complex32,
    lda: i32,
    x: *mut Complex32,
    incx: i32,
) {
    let p = get_ctbsv_for_lp64_cblas();
    match p {
        CtbsvProvider::Lp64(ctbsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctbsv(
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
                    ctbsv(
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
        CtbsvProvider::Ilp64(ctbsv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctbsv(
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
                    ctbsv(
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
pub unsafe extern "C" fn cblas_ctbsv_64(
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
    let p = get_ctbsv_for_ilp64_cblas();
    if matches!(p, CtbsvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ctbsv_64\0",
            [(5, n), (6, k), (8, lda), (10, incx)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CtbsvProvider::Ilp64(ctbsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctbsv(
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
                    ctbsv(
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
        CtbsvProvider::Lp64(ctbsv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ctbsv(
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
                    ctbsv(
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
    n: i32,
    k: i32,
    a: *const Complex64,
    lda: i32,
    x: *mut Complex64,
    incx: i32,
) {
    let p = get_ztbsv_for_lp64_cblas();
    match p {
        ZtbsvProvider::Lp64(ztbsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztbsv(
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
                    ztbsv(
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
        ZtbsvProvider::Ilp64(ztbsv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztbsv(
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
                    ztbsv(
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
pub unsafe extern "C" fn cblas_ztbsv_64(
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
    let p = get_ztbsv_for_ilp64_cblas();
    if matches!(p, ZtbsvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ztbsv_64\0",
            [(5, n), (6, k), (8, lda), (10, incx)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZtbsvProvider::Ilp64(ztbsv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztbsv(
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
                    ztbsv(
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
        ZtbsvProvider::Lp64(ztbsv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    ztbsv(
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
                    ztbsv(
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
