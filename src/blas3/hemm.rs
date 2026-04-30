//! Hermitian matrix multiply (HEMM) - CBLAS interface.
//!
//! Computes: C = alpha * A * B + beta * C  (Side=Left)
//!       or: C = alpha * B * A + beta * C  (Side=Right)
//! where A is Hermitian.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/hemm.c>

use crate::backend::{
    get_chemm_for_ilp64_cblas, get_chemm_for_lp64_cblas, get_zhemm_for_ilp64_cblas,
    get_zhemm_for_lp64_cblas, ChemmProvider, ZhemmProvider,
};
use crate::types::{
    side_to_char, uplo_to_char, CblasColMajor, CblasLeft, CblasLower, CblasRight, CblasRowMajor,
    CblasUpper, CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Single precision complex Hermitian matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - chemm must be registered via `register_chemm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chemm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    b: *const Complex32,
    ldb: i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i32,
) {
    let p = get_chemm_for_lp64_cblas();
    match p {
        ChemmProvider::Lp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                f(
                    &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
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
                f(
                    &side_char, &uplo_char, &n, &m, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                );
            }
        },
        ChemmProvider::Ilp64(f) => {
            let m = m as i64;
            let n = n as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    f(
                        &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
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
                    f(
                        &side_char, &uplo_char, &n, &m, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                    );
                }
            }
        }
    }
}

/// Single precision complex Hermitian matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - chemm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chemm_64(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: i64,
    n: i64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i64,
    b: *const Complex32,
    ldb: i64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: i64,
) {
    let p = get_chemm_for_ilp64_cblas();
    match p {
        ChemmProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                f(
                    &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
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
                f(
                    &side_char, &uplo_char, &n, &m, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                );
            }
        },
        ChemmProvider::Lp64(f) => {
            let m = m as i32;
            let n = n as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    f(
                        &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
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
                    f(
                        &side_char, &uplo_char, &n, &m, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                    );
                }
            }
        }
    }
}

/// Double precision complex Hermitian matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zhemm must be registered via `register_zhemm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhemm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    b: *const Complex64,
    ldb: i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i32,
) {
    let p = get_zhemm_for_lp64_cblas();
    match p {
        ZhemmProvider::Lp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                f(
                    &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
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
                f(
                    &side_char, &uplo_char, &n, &m, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                );
            }
        },
        ZhemmProvider::Ilp64(f) => {
            let m = m as i64;
            let n = n as i64;
            let lda = lda as i64;
            let ldb = ldb as i64;
            let ldc = ldc as i64;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    f(
                        &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
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
                    f(
                        &side_char, &uplo_char, &n, &m, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                    );
                }
            }
        }
    }
}

/// Double precision complex Hermitian matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zhemm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhemm_64(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: i64,
    n: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    b: *const Complex64,
    ldb: i64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: i64,
) {
    let p = get_zhemm_for_ilp64_cblas();
    match p {
        ZhemmProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                f(
                    &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
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
                f(
                    &side_char, &uplo_char, &n, &m, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                );
            }
        },
        ZhemmProvider::Lp64(f) => {
            let m = m as i32;
            let n = n as i32;
            let lda = lda as i32;
            let ldb = ldb as i32;
            let ldc = ldc as i32;
            match order {
                CblasColMajor => {
                    let side_char = side_to_char(side);
                    let uplo_char = uplo_to_char(uplo);
                    f(
                        &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
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
                    f(
                        &side_char, &uplo_char, &n, &m, alpha, a, &lda, b, &ldb, beta, c, &ldc,
                    );
                }
            }
        }
    }
}
