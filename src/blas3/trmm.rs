//! Triangular matrix multiply (TRMM) - CBLAS interface.
//!
//! Computes: B = alpha * op(A) * B  (Side=Left)
//!       or: B = alpha * B * op(A)  (Side=Right)
//! where A is triangular.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trmm.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_ctrmm_for_ilp64_cblas, get_ctrmm_for_lp64_cblas, get_dtrmm_for_ilp64_cblas,
    get_dtrmm_for_lp64_cblas, get_strmm_for_ilp64_cblas, get_strmm_for_lp64_cblas,
    get_ztrmm_for_ilp64_cblas, get_ztrmm_for_lp64_cblas, CtrmmProvider, DtrmmProvider,
    StrmmProvider, ZtrmmProvider,
};
use crate::types::{
    diag_to_char, side_to_char, transpose_to_char, uplo_to_char, CblasColMajor, CblasLeft,
    CblasLower, CblasRight, CblasRowMajor, CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_SIDE,
    CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Double precision triangular matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtrmm must be registered via `register_dtrmm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dtrmm(
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
    let p = get_dtrmm_for_lp64_cblas();
    match p {
        DtrmmProvider::Lp64(f) => match order {
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
        DtrmmProvider::Ilp64(f) => {
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

/// Double precision triangular matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtrmm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dtrmm_64(
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
    let p = get_dtrmm_for_ilp64_cblas();
    if matches!(p, DtrmmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dtrmm_64\0",
            [(6, m), (7, n), (10, lda), (12, ldb)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DtrmmProvider::Ilp64(f) => match order {
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
        DtrmmProvider::Lp64(f) => {
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

/// Single precision triangular matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - strmm must be registered via `register_strmm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_strmm(
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
    let p = get_strmm_for_lp64_cblas();
    match p {
        StrmmProvider::Lp64(f) => match order {
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
        StrmmProvider::Ilp64(f) => {
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

/// Single precision triangular matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - strmm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_strmm_64(
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
    let p = get_strmm_for_ilp64_cblas();
    if matches!(p, StrmmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_strmm_64\0",
            [(6, m), (7, n), (10, lda), (12, ldb)],
        )
        .is_none()
    {
        return;
    }

    match p {
        StrmmProvider::Ilp64(f) => match order {
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
        StrmmProvider::Lp64(f) => {
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

/// Single precision complex triangular matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ctrmm must be registered via `register_ctrmm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ctrmm(
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
    let p = get_ctrmm_for_lp64_cblas();
    match p {
        CtrmmProvider::Lp64(f) => match order {
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
        CtrmmProvider::Ilp64(f) => {
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

/// Single precision complex triangular matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ctrmm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ctrmm_64(
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
    let p = get_ctrmm_for_ilp64_cblas();
    if matches!(p, CtrmmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ctrmm_64\0",
            [(6, m), (7, n), (10, lda), (12, ldb)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CtrmmProvider::Ilp64(f) => match order {
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
        CtrmmProvider::Lp64(f) => {
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

/// Double precision complex triangular matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ztrmm must be registered via `register_ztrmm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ztrmm(
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
    let p = get_ztrmm_for_lp64_cblas();
    match p {
        ZtrmmProvider::Lp64(f) => match order {
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
        ZtrmmProvider::Ilp64(f) => {
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

/// Double precision complex triangular matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ztrmm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ztrmm_64(
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
    let p = get_ztrmm_for_ilp64_cblas();
    if matches!(p, ZtrmmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ztrmm_64\0",
            [(6, m), (7, n), (10, lda), (12, ldb)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZtrmmProvider::Ilp64(f) => match order {
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
        ZtrmmProvider::Lp64(f) => {
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
