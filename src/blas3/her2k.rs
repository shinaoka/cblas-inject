//! Hermitian rank-2k update (HER2K) - CBLAS interface.
//!
//! Computes: C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^H * B + conj(alpha) * B^H * A + beta * C  (Trans=ConjTrans)
//! where C is Hermitian.
//!
//! Note: For HER2K, alpha is complex but beta is REAL.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/her2k.c>

use crate::backend::{
    get_cher2k_for_ilp64_cblas, get_cher2k_for_lp64_cblas, get_zher2k_for_ilp64_cblas,
    get_zher2k_for_lp64_cblas, Cher2kProvider, Zher2kProvider,
};
use crate::types::{
    transpose_to_char, uplo_to_char, CblasColMajor, CblasConjTrans, CblasLower, CblasNoTrans,
    CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Single precision complex Hermitian rank-2k update.
///
/// Note: alpha is complex, but beta is real (f32).
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cher2k must be registered via `register_cher2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cher2k(
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
    beta: f32,
    c: *mut Complex32,
    ldc: i32,
) {
    let p = get_cher2k_for_lp64_cblas();
    match p {
        Cher2kProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasConjTrans,
                    CblasConjTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let alpha_val = *alpha;
                let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &conj_alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
        },
        Cher2kProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let alpha_val = *alpha;
                    let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &conj_alpha, a, &lda, b, &ldb, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Single precision complex Hermitian rank-2k update with ILP64 CBLAS integer ABI.
///
/// Note: alpha is complex, but beta is real (f32).
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cher2k must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cher2k_64(
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
    beta: f32,
    c: *mut Complex32,
    ldc: i64,
) {
    let p = get_cher2k_for_ilp64_cblas();
    match p {
        Cher2kProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasConjTrans,
                    CblasConjTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let alpha_val = *alpha;
                let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &conj_alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
        },
        Cher2kProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let alpha_val = *alpha;
                    let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &conj_alpha, a, &lda, b, &ldb, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Double precision complex Hermitian rank-2k update.
///
/// Note: alpha is complex, but beta is real (f64).
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zher2k must be registered via `register_zher2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zher2k(
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
    beta: f64,
    c: *mut Complex64,
    ldc: i32,
) {
    let p = get_zher2k_for_lp64_cblas();
    match p {
        Zher2kProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasConjTrans,
                    CblasConjTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let alpha_val = *alpha;
                let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &conj_alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
        },
        Zher2kProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let alpha_val = *alpha;
                    let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &conj_alpha, a, &lda, b, &ldb, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Double precision complex Hermitian rank-2k update with ILP64 CBLAS integer ABI.
///
/// Note: alpha is complex, but beta is real (f64).
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zher2k must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zher2k_64(
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
    beta: f64,
    c: *mut Complex64,
    ldc: i64,
) {
    let p = get_zher2k_for_ilp64_cblas();
    match p {
        Zher2kProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
            CblasRowMajor => {
                let new_uplo = match uplo {
                    CblasUpper => CblasLower,
                    CblasLower => CblasUpper,
                };
                let new_trans = match trans {
                    CblasNoTrans => CblasConjTrans,
                    CblasConjTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let alpha_val = *alpha;
                let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &conj_alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
        },
        Zher2kProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, b, &ldb, &beta, c, &ldc);
                }
                CblasRowMajor => {
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let new_trans = match trans {
                        CblasNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let alpha_val = *alpha;
                    let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &conj_alpha, a, &lda, b, &ldb, &beta, c, &ldc);
                }
            }
        }
    }
}
