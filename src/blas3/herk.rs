//! Hermitian rank-k update (HERK) - CBLAS interface.
//!
//! Computes: C = alpha * A * A^H + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^H * A + beta * C  (Trans=ConjTrans)
//! where C is Hermitian.
//!
//! Note: alpha and beta are REAL (not complex) for HERK operations.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/herk.c>

use crate::backend::{
    get_cherk_for_ilp64_cblas, get_cherk_for_lp64_cblas, get_zherk_for_ilp64_cblas,
    get_zherk_for_lp64_cblas, CherkProvider, ZherkProvider,
};
use crate::types::{
    transpose_to_char, uplo_to_char, CblasColMajor, CblasConjTrans, CblasLower, CblasNoTrans,
    CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Single precision complex Hermitian rank-k update.
///
/// Computes: C = alpha * A * A^H + beta * C  (Trans=NoTrans)
///       or: C = alpha * A^H * A + beta * C  (Trans=ConjTrans)
///
/// Note: alpha and beta are real (f32), not complex.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cherk must be registered via `register_cherk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cherk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const Complex32,
    lda: i32,
    beta: f32,
    c: *mut Complex32,
    ldc: i32,
) {
    let p = get_cherk_for_lp64_cblas();
    match p {
        CherkProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasConjTrans,
                    CblasConjTrans => CblasNoTrans,
                    _ => CblasConjTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
        },
        CherkProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasNoTrans,
                        _ => CblasConjTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Single precision complex Hermitian rank-k update with ILP64 CBLAS integer ABI.
///
/// Note: alpha and beta are real (f32), not complex.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cherk must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cherk_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const Complex32,
    lda: i64,
    beta: f32,
    c: *mut Complex32,
    ldc: i64,
) {
    let p = get_cherk_for_ilp64_cblas();
    match p {
        CherkProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasConjTrans,
                    CblasConjTrans => CblasNoTrans,
                    _ => CblasConjTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
        },
        CherkProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasNoTrans,
                        _ => CblasConjTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Double precision complex Hermitian rank-k update.
///
/// Computes: C = alpha * A * A^H + beta * C  (Trans=NoTrans)
///       or: C = alpha * A^H * A + beta * C  (Trans=ConjTrans)
///
/// Note: alpha and beta are real (f64), not complex.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zherk must be registered via `register_zherk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zherk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const Complex64,
    lda: i32,
    beta: f64,
    c: *mut Complex64,
    ldc: i32,
) {
    let p = get_zherk_for_lp64_cblas();
    match p {
        ZherkProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasConjTrans,
                    CblasConjTrans => CblasNoTrans,
                    _ => CblasConjTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
        },
        ZherkProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasNoTrans,
                        _ => CblasConjTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Double precision complex Hermitian rank-k update with ILP64 CBLAS integer ABI.
///
/// Note: alpha and beta are real (f64), not complex.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zherk must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zherk_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: f64,
    a: *const Complex64,
    lda: i64,
    beta: f64,
    c: *mut Complex64,
    ldc: i64,
) {
    let p = get_zherk_for_ilp64_cblas();
    match p {
        ZherkProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasConjTrans,
                    CblasConjTrans => CblasNoTrans,
                    _ => CblasConjTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
        },
        ZherkProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasNoTrans,
                        _ => CblasConjTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
            }
        }
    }
}
