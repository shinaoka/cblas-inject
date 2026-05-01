//! Symmetric rank-2k update (SYR2K) - CBLAS interface.
//!
//! Computes: C = alpha * A * B^T + alpha * B * A^T + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^T * B + alpha * B^T * A + beta * C  (Trans=Trans)
//! where C is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2k.c>

use crate::backend::{
    get_csyr2k_for_ilp64_cblas, get_csyr2k_for_lp64_cblas, get_dsyr2k_for_ilp64_cblas,
    get_dsyr2k_for_lp64_cblas, get_ssyr2k_for_ilp64_cblas, get_ssyr2k_for_lp64_cblas,
    get_zsyr2k_for_ilp64_cblas, get_zsyr2k_for_lp64_cblas, Csyr2kProvider, Dsyr2kProvider,
    Ssyr2kProvider, Zsyr2kProvider,
};
use crate::types::{
    transpose_to_char, uplo_to_char, CblasColMajor, CblasLower, CblasNoTrans, CblasRowMajor,
    CblasTrans, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Double precision symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyr2k must be registered via `register_dsyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    let p = get_dsyr2k_for_lp64_cblas();
    match p {
        Dsyr2kProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    &beta,
                    c,
                    &ldc,
                );
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    &beta,
                    c,
                    &ldc,
                );
            }
        },
        Dsyr2kProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        &beta,
                        c,
                        &ldc,
                    );
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        &beta,
                        c,
                        &ldc,
                    );
                }
            }
        }
    }
}

/// Double precision symmetric rank-2k update with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyr2k must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr2k_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    b: *const f64,
    ldb: i64,
    beta: f64,
    c: *mut f64,
    ldc: i64,
) {
    let p = get_dsyr2k_for_ilp64_cblas();
    if matches!(p, Dsyr2kProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dsyr2k_64\0",
            [(4, n), (5, k), (8, lda), (10, ldb), (13, ldc)],
        )
        .is_none()
    {
        return;
    }

    match p {
        Dsyr2kProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    &beta,
                    c,
                    &ldc,
                );
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    &beta,
                    c,
                    &ldc,
                );
            }
        },
        Dsyr2kProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        &beta,
                        c,
                        &ldc,
                    );
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        &beta,
                        c,
                        &ldc,
                    );
                }
            }
        }
    }
}

/// Single precision symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyr2k must be registered via `register_ssyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    let p = get_ssyr2k_for_lp64_cblas();
    match p {
        Ssyr2kProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    &beta,
                    c,
                    &ldc,
                );
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    &beta,
                    c,
                    &ldc,
                );
            }
        },
        Ssyr2kProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        &beta,
                        c,
                        &ldc,
                    );
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        &beta,
                        c,
                        &ldc,
                    );
                }
            }
        }
    }
}

/// Single precision symmetric rank-2k update with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyr2k must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr2k_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    b: *const f32,
    ldb: i64,
    beta: f32,
    c: *mut f32,
    ldc: i64,
) {
    let p = get_ssyr2k_for_ilp64_cblas();
    if matches!(p, Ssyr2kProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ssyr2k_64\0",
            [(4, n), (5, k), (8, lda), (10, ldb), (13, ldc)],
        )
        .is_none()
    {
        return;
    }

    match p {
        Ssyr2kProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    &beta,
                    c,
                    &ldc,
                );
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    &alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    &beta,
                    c,
                    &ldc,
                );
            }
        },
        Ssyr2kProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        &beta,
                        c,
                        &ldc,
                    );
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        &alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        &beta,
                        c,
                        &ldc,
                    );
                }
            }
        }
    }
}

/// Single precision complex symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csyr2k must be registered via `register_csyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *const Complex32,
    ldb: i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i32,
) {
    let p = get_csyr2k_for_lp64_cblas();
    match p {
        Csyr2kProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    beta,
                    c,
                    &ldc,
                );
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    beta,
                    c,
                    &ldc,
                );
            }
        },
        Csyr2kProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        beta,
                        c,
                        &ldc,
                    );
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        beta,
                        c,
                        &ldc,
                    );
                }
            }
        }
    }
}

/// Single precision complex symmetric rank-2k update with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csyr2k must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csyr2k_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i64,
    b: *const Complex32,
    ldb: i64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i64,
) {
    let p = get_csyr2k_for_ilp64_cblas();
    if matches!(p, Csyr2kProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_csyr2k_64\0",
            [(4, n), (5, k), (8, lda), (10, ldb), (13, ldc)],
        )
        .is_none()
    {
        return;
    }

    match p {
        Csyr2kProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    beta,
                    c,
                    &ldc,
                );
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    beta,
                    c,
                    &ldc,
                );
            }
        },
        Csyr2kProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        beta,
                        c,
                        &ldc,
                    );
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        beta,
                        c,
                        &ldc,
                    );
                }
            }
        }
    }
}

/// Double precision complex symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsyr2k must be registered via `register_zsyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *const Complex64,
    ldb: i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i32,
) {
    let p = get_zsyr2k_for_lp64_cblas();
    match p {
        Zsyr2kProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    beta,
                    c,
                    &ldc,
                );
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    beta,
                    c,
                    &ldc,
                );
            }
        },
        Zsyr2kProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        beta,
                        c,
                        &ldc,
                    );
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        beta,
                        c,
                        &ldc,
                    );
                }
            }
        }
    }
}

/// Double precision complex symmetric rank-2k update with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsyr2k must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsyr2k_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    b: *const Complex64,
    ldb: i64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i64,
) {
    let p = get_zsyr2k_for_ilp64_cblas();
    if matches!(p, Zsyr2kProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zsyr2k_64\0",
            [(4, n), (5, k), (8, lda), (10, ldb), (13, ldc)],
        )
        .is_none()
    {
        return;
    }

    match p {
        Zsyr2kProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    beta,
                    c,
                    &ldc,
                );
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(
                    &uplo_char,
                    &trans_char,
                    &n,
                    &k,
                    alpha,
                    a,
                    &lda,
                    b,
                    &ldb,
                    beta,
                    c,
                    &ldc,
                );
            }
        },
        Zsyr2kProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        beta,
                        c,
                        &ldc,
                    );
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(
                        &uplo_char,
                        &trans_char,
                        &n,
                        &k,
                        alpha,
                        a,
                        &lda,
                        b,
                        &ldb,
                        beta,
                        c,
                        &ldc,
                    );
                }
            }
        }
    }
}
