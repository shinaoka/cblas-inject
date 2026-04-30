//! Symmetric matrix multiply (SYMM) - CBLAS interface.
//!
//! Computes: C = alpha * A * B + beta * C  (Side=Left)
//!       or: C = alpha * B * A + beta * C  (Side=Right)
//! where A is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symm.c>

use crate::backend::{
    get_csymm_for_ilp64_cblas, get_csymm_for_lp64_cblas, get_dsymm_for_ilp64_cblas,
    get_dsymm_for_lp64_cblas, get_ssymm_for_ilp64_cblas, get_ssymm_for_lp64_cblas,
    get_zsymm_for_ilp64_cblas, get_zsymm_for_lp64_cblas, CsymmProvider, DsymmProvider,
    SsymmProvider, ZsymmProvider,
};
use crate::types::{
    side_to_char, uplo_to_char, CblasColMajor, CblasLeft, CblasLower, CblasRight, CblasRowMajor,
    CblasUpper, CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Double precision symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsymm must be registered via `register_dsymm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsymm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    let p = get_dsymm_for_lp64_cblas();
    match p {
        DsymmProvider::Lp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                f(
                    &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
                    &side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                );
            }
        },
        DsymmProvider::Ilp64(f) => {
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
                        &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
                        &side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                    );
                }
            }
        }
    }
}

/// Double precision symmetric matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsymm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsymm_64(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: i64,
    n: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    b: *const f64,
    ldb: i64,
    beta: f64,
    c: *mut f64,
    ldc: i64,
) {
    let p = get_dsymm_for_ilp64_cblas();
    match p {
        DsymmProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                f(
                    &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
                    &side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                );
            }
        },
        DsymmProvider::Lp64(f) => {
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
                        &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
                        &side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                    );
                }
            }
        }
    }
}

/// Single precision symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssymm must be registered via `register_ssymm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssymm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    let p = get_ssymm_for_lp64_cblas();
    match p {
        SsymmProvider::Lp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                f(
                    &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
                    &side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                );
            }
        },
        SsymmProvider::Ilp64(f) => {
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
                        &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
                        &side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                    );
                }
            }
        }
    }
}

/// Single precision symmetric matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssymm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssymm_64(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: i64,
    n: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    b: *const f32,
    ldb: i64,
    beta: f32,
    c: *mut f32,
    ldc: i64,
) {
    let p = get_ssymm_for_ilp64_cblas();
    match p {
        SsymmProvider::Ilp64(f) => match order {
            CblasColMajor => {
                let side_char = side_to_char(side);
                let uplo_char = uplo_to_char(uplo);
                f(
                    &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
                    &side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                );
            }
        },
        SsymmProvider::Lp64(f) => {
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
                        &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
                        &side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                    );
                }
            }
        }
    }
}

/// Single precision complex symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csymm must be registered via `register_csymm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csymm(
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
    let p = get_csymm_for_lp64_cblas();
    match p {
        CsymmProvider::Lp64(f) => match order {
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
        CsymmProvider::Ilp64(f) => {
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

/// Single precision complex symmetric matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csymm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csymm_64(
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
    let p = get_csymm_for_ilp64_cblas();
    match p {
        CsymmProvider::Ilp64(f) => match order {
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
        CsymmProvider::Lp64(f) => {
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

/// Double precision complex symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsymm must be registered via `register_zsymm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsymm(
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
    let p = get_zsymm_for_lp64_cblas();
    match p {
        ZsymmProvider::Lp64(f) => match order {
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
        ZsymmProvider::Ilp64(f) => {
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

/// Double precision complex symmetric matrix multiply with ILP64 CBLAS integer ABI.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsymm must be registered
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsymm_64(
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
    let p = get_zsymm_for_ilp64_cblas();
    match p {
        ZsymmProvider::Ilp64(f) => match order {
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
        ZsymmProvider::Lp64(f) => {
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
