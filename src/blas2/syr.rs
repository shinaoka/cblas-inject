//! Symmetric/Hermitian rank-1 and rank-2 updates (SYR, HER, SYR2, HER2) - CBLAS interface.
//!
//! SYR:  A = alpha * x * x^T + A  (symmetric rank-1 update)
//! HER:  A = alpha * x * conj(x)^T + A  (hermitian rank-1 update)
//! SYR2: A = alpha * x * y^T + alpha * y * x^T + A  (symmetric rank-2 update)
//! HER2: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A  (hermitian rank-2 update)
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zher.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zher2.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_cher2_for_ilp64_cblas, get_cher2_for_lp64_cblas, get_cher_for_ilp64_cblas,
    get_cher_for_lp64_cblas, get_dsyr2_for_ilp64_cblas, get_dsyr2_for_lp64_cblas,
    get_dsyr_for_ilp64_cblas, get_dsyr_for_lp64_cblas, get_ssyr2_for_ilp64_cblas,
    get_ssyr2_for_lp64_cblas, get_ssyr_for_ilp64_cblas, get_ssyr_for_lp64_cblas,
    get_zher2_for_ilp64_cblas, get_zher2_for_lp64_cblas, get_zher_for_ilp64_cblas,
    get_zher_for_lp64_cblas, Cher2Provider, CherProvider, Dsyr2Provider, DsyrProvider,
    Ssyr2Provider, SsyrProvider, Zher2Provider, ZherProvider,
};
use crate::types::{
    uplo_to_char, CblasColMajor, CblasLower, CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_UPLO,
};

// =============================================================================
// Real SYR: A = alpha * x * x^T + A
// =============================================================================

/// Single precision symmetric rank-1 update: A = alpha * x * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyr must be registered via `register_ssyr`
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    a: *mut f32,
    lda: i32,
) {
    let p = get_ssyr_for_lp64_cblas();
    match p {
        SsyrProvider::Lp64(ssyr) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
        SsyrProvider::Ilp64(ssyr) => {
            let n = n as i64;
            let incx = incx as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f32,
    x: *const f32,
    incx: i64,
    a: *mut f32,
    lda: i64,
) {
    let p = get_ssyr_for_ilp64_cblas();
    if matches!(p, SsyrProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_ssyr_64\0", [(3, n), (6, incx), (8, lda)])
            .is_none()
    {
        return;
    }

    match p {
        SsyrProvider::Ilp64(ssyr) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
        SsyrProvider::Lp64(ssyr) => {
            let n = n as i32;
            let incx = incx as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
    }
}

/// Double precision symmetric rank-1 update: A = alpha * x * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyr must be registered via `register_dsyr`
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    a: *mut f64,
    lda: i32,
) {
    let p = get_dsyr_for_lp64_cblas();
    match p {
        DsyrProvider::Lp64(dsyr) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
        DsyrProvider::Ilp64(dsyr) => {
            let n = n as i64;
            let incx = incx as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f64,
    x: *const f64,
    incx: i64,
    a: *mut f64,
    lda: i64,
) {
    let p = get_dsyr_for_ilp64_cblas();
    if matches!(p, DsyrProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_dsyr_64\0", [(3, n), (6, incx), (8, lda)])
            .is_none()
    {
        return;
    }

    match p {
        DsyrProvider::Ilp64(dsyr) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
        DsyrProvider::Lp64(dsyr) => {
            let n = n as i32;
            let incx = incx as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
    }
}

// =============================================================================
// Complex HER: A = alpha * x * conj(x)^T + A
// =============================================================================

/// Single precision complex hermitian rank-1 update: A = alpha * x * conj(x)^T + A
///
/// Note: alpha is real for HER operations.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cher must be registered via `register_cher`
#[no_mangle]
pub unsafe extern "C" fn cblas_cher(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f32,
    x: *const Complex32,
    incx: i32,
    a: *mut Complex32,
    lda: i32,
) {
    let p = get_cher_for_lp64_cblas();
    match p {
        CherProvider::Lp64(cher) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    // The Hermitian property A = A^H means A^T = conj(A)
                    // So row-major upper triangle = col-major lower triangle (conjugated)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
        CherProvider::Ilp64(cher) => {
            let n = n as i64;
            let incx = incx as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    // The Hermitian property A = A^H means A^T = conj(A)
                    // So row-major upper triangle = col-major lower triangle (conjugated)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_cher_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f32,
    x: *const Complex32,
    incx: i64,
    a: *mut Complex32,
    lda: i64,
) {
    let p = get_cher_for_ilp64_cblas();
    if matches!(p, CherProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_cher_64\0", [(3, n), (6, incx), (8, lda)])
            .is_none()
    {
        return;
    }

    match p {
        CherProvider::Ilp64(cher) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    // The Hermitian property A = A^H means A^T = conj(A)
                    // So row-major upper triangle = col-major lower triangle (conjugated)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
        CherProvider::Lp64(cher) => {
            let n = n as i32;
            let incx = incx as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    // The Hermitian property A = A^H means A^T = conj(A)
                    // So row-major upper triangle = col-major lower triangle (conjugated)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
    }
}

/// Double precision complex hermitian rank-1 update: A = alpha * x * conj(x)^T + A
///
/// Note: alpha is real for HER operations.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zher must be registered via `register_zher`
#[no_mangle]
pub unsafe extern "C" fn cblas_zher(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f64,
    x: *const Complex64,
    incx: i32,
    a: *mut Complex64,
    lda: i32,
) {
    let p = get_zher_for_lp64_cblas();
    match p {
        ZherProvider::Lp64(zher) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
        ZherProvider::Ilp64(zher) => {
            let n = n as i64;
            let incx = incx as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_zher_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f64,
    x: *const Complex64,
    incx: i64,
    a: *mut Complex64,
    lda: i64,
) {
    let p = get_zher_for_ilp64_cblas();
    if matches!(p, ZherProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_zher_64\0", [(3, n), (6, incx), (8, lda)])
            .is_none()
    {
        return;
    }

    match p {
        ZherProvider::Ilp64(zher) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
        ZherProvider::Lp64(zher) => {
            let n = n as i32;
            let incx = incx as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
                }
            }
        }
    }
}

// =============================================================================
// Real SYR2: A = alpha * x * y^T + alpha * y * x^T + A
// =============================================================================

/// Single precision symmetric rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyr2 must be registered via `register_ssyr2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
    a: *mut f32,
    lda: i32,
) {
    let p = get_ssyr2_for_lp64_cblas();
    match p {
        Ssyr2Provider::Lp64(ssyr2) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
            }
        }
        Ssyr2Provider::Ilp64(ssyr2) => {
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr2_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f32,
    x: *const f32,
    incx: i64,
    y: *const f32,
    incy: i64,
    a: *mut f32,
    lda: i64,
) {
    let p = get_ssyr2_for_ilp64_cblas();
    if matches!(p, Ssyr2Provider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ssyr2_64\0",
            [(3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        Ssyr2Provider::Ilp64(ssyr2) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
            }
        }
        Ssyr2Provider::Lp64(ssyr2) => {
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
            }
        }
    }
}

/// Double precision symmetric rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyr2 must be registered via `register_dsyr2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    a: *mut f64,
    lda: i32,
) {
    let p = get_dsyr2_for_lp64_cblas();
    match p {
        Dsyr2Provider::Lp64(dsyr2) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
            }
        }
        Dsyr2Provider::Ilp64(dsyr2) => {
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr2_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f64,
    x: *const f64,
    incx: i64,
    y: *const f64,
    incy: i64,
    a: *mut f64,
    lda: i64,
) {
    let p = get_dsyr2_for_ilp64_cblas();
    if matches!(p, Dsyr2Provider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dsyr2_64\0",
            [(3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        Dsyr2Provider::Ilp64(dsyr2) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
            }
        }
        Dsyr2Provider::Lp64(dsyr2) => {
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: invert uplo (Upper <-> Lower)
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
            }
        }
    }
}

// =============================================================================
// Complex HER2: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
// =============================================================================

/// Single precision complex hermitian rank-2 update
///
/// Computes: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cher2 must be registered via `register_cher2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cher2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    a: *mut Complex32,
    lda: i32,
) {
    let p = get_cher2_for_lp64_cblas();
    match p {
        Cher2Provider::Lp64(cher2) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    cher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    // For HER2 in row-major, we also need to swap x and y
                    // and use conjugate of alpha (handled by the property of HER2)
                    cher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        Cher2Provider::Ilp64(cher2) => {
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    cher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    // For HER2 in row-major, we also need to swap x and y
                    // and use conjugate of alpha (handled by the property of HER2)
                    cher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cher2_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i64,
    y: *const Complex32,
    incy: i64,
    a: *mut Complex32,
    lda: i64,
) {
    let p = get_cher2_for_ilp64_cblas();
    if matches!(p, Cher2Provider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_cher2_64\0",
            [(3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        Cher2Provider::Ilp64(cher2) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    cher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    // For HER2 in row-major, we also need to swap x and y
                    // and use conjugate of alpha (handled by the property of HER2)
                    cher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        Cher2Provider::Lp64(cher2) => {
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    cher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    // For HER2 in row-major, we also need to swap x and y
                    // and use conjugate of alpha (handled by the property of HER2)
                    cher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

/// Double precision complex hermitian rank-2 update
///
/// Computes: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zher2 must be registered via `register_zher2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zher2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    a: *mut Complex64,
    lda: i32,
) {
    let p = get_zher2_for_lp64_cblas();
    match p {
        Zher2Provider::Lp64(zher2) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo and swap x<->y
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    zher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        Zher2Provider::Ilp64(zher2) => {
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo and swap x<->y
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    zher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zher2_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i64,
    y: *const Complex64,
    incy: i64,
    a: *mut Complex64,
    lda: i64,
) {
    let p = get_zher2_for_ilp64_cblas();
    if matches!(p, Zher2Provider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zher2_64\0",
            [(3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        Zher2Provider::Ilp64(zher2) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo and swap x<->y
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    zher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        Zher2Provider::Lp64(zher2) => {
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: invert uplo and swap x<->y
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    zher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}
