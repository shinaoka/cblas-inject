//! Triangular solve (TRSM) - CBLAS interface.
//!
//! Solves: op(A) * X = alpha * B  (Side=Left)
//!     or: X * op(A) = alpha * B  (Side=Right)
//! where A is triangular, and overwrites B with X.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trsm.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_ctrsm_for_ilp64_cblas, get_ctrsm_for_lp64_cblas, get_dtrsm_for_ilp64_cblas,
    get_dtrsm_for_lp64_cblas, get_strsm_for_ilp64_cblas, get_strsm_for_lp64_cblas,
    get_ztrsm_for_ilp64_cblas, get_ztrsm_for_lp64_cblas, CtrsmProvider, DtrsmProvider,
    StrsmProvider, ZtrsmProvider,
};
use crate::types::{
    diag_to_char, side_to_char, transpose_to_char, uplo_to_char, CblasColMajor, CblasLeft,
    CblasLower, CblasRight, CblasRowMajor, CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_SIDE,
    CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Double precision triangular solve.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtrsm must be registered via `register_dtrsm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dtrsm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) {
    let p = get_dtrsm_for_lp64_cblas();
    match p {
        DtrsmProvider::Lp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &m,
                    &n,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
            CblasRowMajor => {
                let new_side = match side {
                    CblasLeft => CblasRight,
                    CblasRight => CblasLeft,
                };
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let side_char = side_to_char(new_side);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &n,
                    &m,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
        },
        DtrsmProvider::Ilp64(f) => {
            let m = m as i64;
            let n = n as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &m,
                        &n,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
                CblasRowMajor => {
                    let new_side = match side {
                        CblasLeft => CblasRight,
                        CblasRight => CblasLeft,
                    };
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let side_char = side_to_char(new_side);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &m,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
            }
        }
    }
}

/// Double precision triangular solve with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtrsm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dtrsm_64(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i64,
    n: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    b: *mut f64,
    ldb: i64,
) {
    let p = get_dtrsm_for_ilp64_cblas();
    if matches!(p, DtrsmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dtrsm_64\0",
            [(6, m), (7, n), (10, lda), (12, ldb)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DtrsmProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &m,
                    &n,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
            CblasRowMajor => {
                let new_side = match side {
                    CblasLeft => CblasRight,
                    CblasRight => CblasLeft,
                };
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let side_char = side_to_char(new_side);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &n,
                    &m,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
        },
        DtrsmProvider::Lp64(f) => {
            let m = m as i32;
            let n = n as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &m,
                        &n,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
                CblasRowMajor => {
                    let new_side = match side {
                        CblasLeft => CblasRight,
                        CblasRight => CblasLeft,
                    };
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let side_char = side_to_char(new_side);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &m,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
            }
        }
    }
}

/// Single precision triangular solve.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - strsm must be registered via `register_strsm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_strsm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *mut f32,
    ldb: i32,
) {
    let p = get_strsm_for_lp64_cblas();
    match p {
        StrsmProvider::Lp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &m,
                    &n,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
            CblasRowMajor => {
                let new_side = match side {
                    CblasLeft => CblasRight,
                    CblasRight => CblasLeft,
                };
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let side_char = side_to_char(new_side);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &n,
                    &m,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
        },
        StrsmProvider::Ilp64(f) => {
            let m = m as i64;
            let n = n as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &m,
                        &n,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
                CblasRowMajor => {
                    let new_side = match side {
                        CblasLeft => CblasRight,
                        CblasRight => CblasLeft,
                    };
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let side_char = side_to_char(new_side);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &m,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
            }
        }
    }
}

/// Single precision triangular solve with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - strsm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_strsm_64(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i64,
    n: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    b: *mut f32,
    ldb: i64,
) {
    let p = get_strsm_for_ilp64_cblas();
    if matches!(p, StrsmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_strsm_64\0",
            [(6, m), (7, n), (10, lda), (12, ldb)],
        )
        .is_none()
    {
        return;
    }

    match p {
        StrsmProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &m,
                    &n,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
            CblasRowMajor => {
                let new_side = match side {
                    CblasLeft => CblasRight,
                    CblasRight => CblasLeft,
                };
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let side_char = side_to_char(new_side);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &n,
                    &m,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
        },
        StrsmProvider::Lp64(f) => {
            let m = m as i32;
            let n = n as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &m,
                        &n,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
                CblasRowMajor => {
                    let new_side = match side {
                        CblasLeft => CblasRight,
                        CblasRight => CblasLeft,
                    };
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let side_char = side_to_char(new_side);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &m,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
            }
        }
    }
}

/// Single precision complex triangular solve.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ctrsm must be registered via `register_ctrsm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ctrsm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *mut Complex32,
    ldb: i32,
) {
    let p = get_ctrsm_for_lp64_cblas();
    match p {
        CtrsmProvider::Lp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &m,
                    &n,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
            CblasRowMajor => {
                let new_side = match side {
                    CblasLeft => CblasRight,
                    CblasRight => CblasLeft,
                };
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let side_char = side_to_char(new_side);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &n,
                    &m,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
        },
        CtrsmProvider::Ilp64(f) => {
            let m = m as i64;
            let n = n as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &m,
                        &n,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
                CblasRowMajor => {
                    let new_side = match side {
                        CblasLeft => CblasRight,
                        CblasRight => CblasLeft,
                    };
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let side_char = side_to_char(new_side);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &m,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
            }
        }
    }
}

/// Single precision complex triangular solve with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ctrsm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ctrsm_64(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i64,
    n: i64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i64,
    b: *mut Complex32,
    ldb: i64,
) {
    let p = get_ctrsm_for_ilp64_cblas();
    if matches!(p, CtrsmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ctrsm_64\0",
            [(6, m), (7, n), (10, lda), (12, ldb)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CtrsmProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &m,
                    &n,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
            CblasRowMajor => {
                let new_side = match side {
                    CblasLeft => CblasRight,
                    CblasRight => CblasLeft,
                };
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let side_char = side_to_char(new_side);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &n,
                    &m,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
        },
        CtrsmProvider::Lp64(f) => {
            let m = m as i32;
            let n = n as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &m,
                        &n,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
                CblasRowMajor => {
                    let new_side = match side {
                        CblasLeft => CblasRight,
                        CblasRight => CblasLeft,
                    };
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let side_char = side_to_char(new_side);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &m,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
            }
        }
    }
}

/// Double precision complex triangular solve.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ztrsm must be registered via `register_ztrsm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ztrsm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *mut Complex64,
    ldb: i32,
) {
    let p = get_ztrsm_for_lp64_cblas();
    match p {
        ZtrsmProvider::Lp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &m,
                    &n,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
            CblasRowMajor => {
                let new_side = match side {
                    CblasLeft => CblasRight,
                    CblasRight => CblasLeft,
                };
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let side_char = side_to_char(new_side);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &n,
                    &m,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
        },
        ZtrsmProvider::Ilp64(f) => {
            let m = m as i64;
            let n = n as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &m,
                        &n,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
                CblasRowMajor => {
                    let new_side = match side {
                        CblasLeft => CblasRight,
                        CblasRight => CblasLeft,
                    };
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let side_char = side_to_char(new_side);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &m,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
            }
        }
    }
}

/// Double precision complex triangular solve with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ztrsm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ztrsm_64(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i64,
    n: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    b: *mut Complex64,
    ldb: i64,
) {
    let p = get_ztrsm_for_ilp64_cblas();
    if matches!(p, ZtrsmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ztrsm_64\0",
            [(6, m), (7, n), (10, lda), (12, ldb)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZtrsmProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &m,
                    &n,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
            CblasRowMajor => {
                let new_side = match side {
                    CblasLeft => CblasRight,
                    CblasRight => CblasLeft,
                };
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let side_char = side_to_char(new_side);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(trans);
                let diag_char = diag_to_char(diag);
                f(
                    &side_char,
                    &uplo_char,
                    &trans_char,
                    &diag_char,
                    &n,
                    &m,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                );
            }
        },
        ZtrsmProvider::Lp64(f) => {
            let m = m as i32;
            let n = n as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &m,
                        &n,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
                CblasRowMajor => {
                    let new_side = match side {
                        CblasLeft => CblasRight,
                        CblasRight => CblasLeft,
                    };
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let side_char = side_to_char(new_side);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(trans);
                    let diag_char = diag_to_char(diag);
                    f(
                        &side_char,
                        &uplo_char,
                        &trans_char,
                        &diag_char,
                        &n,
                        &m,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                    );
                }
            }
        }
    }
}
