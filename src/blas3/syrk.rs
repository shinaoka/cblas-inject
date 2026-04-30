//! Symmetric rank-k update (SYRK) - CBLAS interface.
//!
//! Computes: C = alpha * A * A^T + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^T * A + beta * C  (Trans=Trans)
//! where C is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syrk.c>

use crate::backend::{
    get_csyrk_for_ilp64_cblas, get_csyrk_for_lp64_cblas, get_dsyrk_for_ilp64_cblas,
    get_dsyrk_for_lp64_cblas, get_ssyrk_for_ilp64_cblas, get_ssyrk_for_lp64_cblas,
    get_zsyrk_for_ilp64_cblas, get_zsyrk_for_lp64_cblas, CsyrkProvider, DsyrkProvider,
    SsyrkProvider, ZsyrkProvider,
};
use crate::types::{
    transpose_to_char, uplo_to_char, CblasColMajor, CblasLower, CblasNoTrans, CblasRowMajor,
    CblasTrans, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Double precision symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyrk must be registered via `register_dsyrk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    let p = get_dsyrk_for_lp64_cblas();
    match p {
        DsyrkProvider::Lp64(f) => match order {
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
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
        },
        DsyrkProvider::Ilp64(f) => {
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
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Double precision symmetric rank-k update with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyrk must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyrk_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    beta: f64,
    c: *mut f64,
    ldc: i64,
) {
    let p = get_dsyrk_for_ilp64_cblas();
    match p {
        DsyrkProvider::Ilp64(f) => match order {
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
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
        },
        DsyrkProvider::Lp64(f) => {
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
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Single precision symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyrk must be registered via `register_ssyrk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    let p = get_ssyrk_for_lp64_cblas();
    match p {
        SsyrkProvider::Lp64(f) => match order {
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
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
        },
        SsyrkProvider::Ilp64(f) => {
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
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Single precision symmetric rank-k update with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyrk must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyrk_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    beta: f32,
    c: *mut f32,
    ldc: i64,
) {
    let p = get_ssyrk_for_ilp64_cblas();
    match p {
        SsyrkProvider::Ilp64(f) => match order {
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
                    CblasNoTrans => CblasTrans,
                    CblasTrans => CblasNoTrans,
                    _ => CblasNoTrans,
                };
                let uplo_char = uplo_to_char(new_uplo);
                let trans_char = transpose_to_char(new_trans);
                f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
            }
        },
        SsyrkProvider::Lp64(f) => {
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
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        _ => CblasNoTrans,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    let trans_char = transpose_to_char(new_trans);
                    f(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
                }
            }
        }
    }
}

/// Single precision complex symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csyrk must be registered via `register_csyrk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i32,
) {
    let p = get_csyrk_for_lp64_cblas();
    match p {
        CsyrkProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
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
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
            }
        },
        CsyrkProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
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
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
                }
            }
        }
    }
}

/// Single precision complex symmetric rank-k update with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csyrk must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csyrk_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i64,
) {
    let p = get_csyrk_for_ilp64_cblas();
    match p {
        CsyrkProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
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
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
            }
        },
        CsyrkProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
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
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
                }
            }
        }
    }
}

/// Double precision complex symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsyrk must be registered via `register_zsyrk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i32,
    k: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i32,
) {
    let p = get_zsyrk_for_lp64_cblas();
    match p {
        ZsyrkProvider::Lp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
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
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
            }
        },
        ZsyrkProvider::Ilp64(f) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
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
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
                }
            }
        }
    }
}

/// Double precision complex symmetric rank-k update with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsyrk must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsyrk_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: i64,
    k: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i64,
) {
    let p = get_zsyrk_for_ilp64_cblas();
    match p {
        ZsyrkProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let uplo_char = uplo_to_char(uplo);
                let trans_char = transpose_to_char(trans);
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
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
                f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
            }
        },
        ZsyrkProvider::Lp64(f) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    let trans_char = transpose_to_char(trans);
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
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
                    f(&uplo_char, &trans_char, &n, &k, alpha, a, &lda, beta, c, &ldc);
                }
            }
        }
    }
}
