//! General rank-1 update (GER) - CBLAS interface.
//!
//! Computes: A = alpha * x * y^T + A  (for real types)
//!       or: A = alpha * x * y^T + A  (GERU, unconjugated)
//!       or: A = alpha * x * conj(y)^T + A  (GERC, conjugated)
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/ger.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_cgerc_for_ilp64_cblas, get_cgerc_for_lp64_cblas, get_cgeru_for_ilp64_cblas,
    get_cgeru_for_lp64_cblas, get_dger_for_ilp64_cblas, get_dger_for_lp64_cblas,
    get_sger_for_ilp64_cblas, get_sger_for_lp64_cblas, get_zgerc_for_ilp64_cblas,
    get_zgerc_for_lp64_cblas, get_zgeru_for_ilp64_cblas, get_zgeru_for_lp64_cblas, CgercProvider,
    CgeruProvider, DgerProvider, SgerProvider, ZgercProvider, ZgeruProvider,
};
use crate::types::{CblasColMajor, CblasRowMajor, CBLAS_ORDER};

// =============================================================================
// Real GER: A = alpha * x * y^T + A
// =============================================================================

/// Single precision rank-1 update: A = alpha * x * y^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - sger must be registered via `register_sger`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_sger(
    order: CBLAS_ORDER,
    m: i32,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
    a: *mut f32,
    lda: i32,
) {
    let p = get_sger_for_lp64_cblas();
    match p {
        SgerProvider::Lp64(sger) => {
            match order {
                CblasColMajor => {
                    sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    // A(m x n) in row-major = A^T(n x m) in col-major
                    // x * y^T in row-major = y * x^T in col-major
                    sger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        SgerProvider::Ilp64(sger) => {
            let m = m as i64;
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    // A(m x n) in row-major = A^T(n x m) in col-major
                    // x * y^T in row-major = y * x^T in col-major
                    sger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_sger_64(
    order: CBLAS_ORDER,
    m: i64,
    n: i64,
    alpha: f32,
    x: *const f32,
    incx: i64,
    y: *const f32,
    incy: i64,
    a: *mut f32,
    lda: i64,
) {
    let p = get_sger_for_ilp64_cblas();
    if matches!(p, SgerProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_sger_64\0",
            [(2, m), (3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        SgerProvider::Ilp64(sger) => {
            match order {
                CblasColMajor => {
                    sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    // A(m x n) in row-major = A^T(n x m) in col-major
                    // x * y^T in row-major = y * x^T in col-major
                    sger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        SgerProvider::Lp64(sger) => {
            let m = m as i32;
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    // A(m x n) in row-major = A^T(n x m) in col-major
                    // x * y^T in row-major = y * x^T in col-major
                    sger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

/// Double precision rank-1 update: A = alpha * x * y^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dger must be registered via `register_dger`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dger(
    order: CBLAS_ORDER,
    m: i32,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    a: *mut f64,
    lda: i32,
) {
    let p = get_dger_for_lp64_cblas();
    match p {
        DgerProvider::Lp64(dger) => {
            match order {
                CblasColMajor => {
                    dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    dger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        DgerProvider::Ilp64(dger) => {
            let m = m as i64;
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    dger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dger_64(
    order: CBLAS_ORDER,
    m: i64,
    n: i64,
    alpha: f64,
    x: *const f64,
    incx: i64,
    y: *const f64,
    incy: i64,
    a: *mut f64,
    lda: i64,
) {
    let p = get_dger_for_ilp64_cblas();
    if matches!(p, DgerProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dger_64\0",
            [(2, m), (3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DgerProvider::Ilp64(dger) => {
            match order {
                CblasColMajor => {
                    dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    dger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        DgerProvider::Lp64(dger) => {
            let m = m as i32;
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    dger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

// =============================================================================
// Complex GERU: A = alpha * x * y^T + A (unconjugated)
// =============================================================================

/// Single precision complex unconjugated rank-1 update: A = alpha * x * y^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cgeru must be registered via `register_cgeru`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cgeru(
    order: CBLAS_ORDER,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    a: *mut Complex32,
    lda: i32,
) {
    let p = get_cgeru_for_lp64_cblas();
    match p {
        CgeruProvider::Lp64(cgeru) => {
            match order {
                CblasColMajor => {
                    cgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    cgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        CgeruProvider::Ilp64(cgeru) => {
            let m = m as i64;
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    cgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    cgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cgeru_64(
    order: CBLAS_ORDER,
    m: i64,
    n: i64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i64,
    y: *const Complex32,
    incy: i64,
    a: *mut Complex32,
    lda: i64,
) {
    let p = get_cgeru_for_ilp64_cblas();
    if matches!(p, CgeruProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_cgeru_64\0",
            [(2, m), (3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CgeruProvider::Ilp64(cgeru) => {
            match order {
                CblasColMajor => {
                    cgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    cgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        CgeruProvider::Lp64(cgeru) => {
            let m = m as i32;
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    cgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    cgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

/// Double precision complex unconjugated rank-1 update: A = alpha * x * y^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgeru must be registered via `register_zgeru`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zgeru(
    order: CBLAS_ORDER,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    a: *mut Complex64,
    lda: i32,
) {
    let p = get_zgeru_for_lp64_cblas();
    match p {
        ZgeruProvider::Lp64(zgeru) => {
            match order {
                CblasColMajor => {
                    zgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    zgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        ZgeruProvider::Ilp64(zgeru) => {
            let m = m as i64;
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    zgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    zgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zgeru_64(
    order: CBLAS_ORDER,
    m: i64,
    n: i64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i64,
    y: *const Complex64,
    incy: i64,
    a: *mut Complex64,
    lda: i64,
) {
    let p = get_zgeru_for_ilp64_cblas();
    if matches!(p, ZgeruProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zgeru_64\0",
            [(2, m), (3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZgeruProvider::Ilp64(zgeru) => {
            match order {
                CblasColMajor => {
                    zgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    zgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        ZgeruProvider::Lp64(zgeru) => {
            let m = m as i32;
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    zgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major: swap m<->n, swap x<->y, swap incx<->incy
                    zgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

// =============================================================================
// Complex GERC: A = alpha * x * conj(y)^T + A (conjugated)
// =============================================================================

/// Single precision complex conjugated rank-1 update: A = alpha * x * conj(y)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cgerc must be registered via `register_cgerc`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cgerc(
    order: CBLAS_ORDER,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    a: *mut Complex32,
    lda: i32,
) {
    let p = get_cgerc_for_lp64_cblas();
    match p {
        CgercProvider::Lp64(cgerc) => {
            match order {
                CblasColMajor => {
                    cgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for GERC: swap m<->n, but also need to use conjugate of alpha
                    // and swap to GERC(y, x) pattern. Following OpenBLAS logic.
                    // A = alpha * x * conj(y)^T becomes A^T = conj(alpha) * conj(y) * x^H
                    // which is equivalent to calling GERC with swapped vectors
                    cgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        CgercProvider::Ilp64(cgerc) => {
            let m = m as i64;
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    cgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for GERC: swap m<->n, but also need to use conjugate of alpha
                    // and swap to GERC(y, x) pattern. Following OpenBLAS logic.
                    // A = alpha * x * conj(y)^T becomes A^T = conj(alpha) * conj(y) * x^H
                    // which is equivalent to calling GERC with swapped vectors
                    cgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cgerc_64(
    order: CBLAS_ORDER,
    m: i64,
    n: i64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i64,
    y: *const Complex32,
    incy: i64,
    a: *mut Complex32,
    lda: i64,
) {
    let p = get_cgerc_for_ilp64_cblas();
    if matches!(p, CgercProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_cgerc_64\0",
            [(2, m), (3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CgercProvider::Ilp64(cgerc) => {
            match order {
                CblasColMajor => {
                    cgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for GERC: swap m<->n, but also need to use conjugate of alpha
                    // and swap to GERC(y, x) pattern. Following OpenBLAS logic.
                    // A = alpha * x * conj(y)^T becomes A^T = conj(alpha) * conj(y) * x^H
                    // which is equivalent to calling GERC with swapped vectors
                    cgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        CgercProvider::Lp64(cgerc) => {
            let m = m as i32;
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    cgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for GERC: swap m<->n, but also need to use conjugate of alpha
                    // and swap to GERC(y, x) pattern. Following OpenBLAS logic.
                    // A = alpha * x * conj(y)^T becomes A^T = conj(alpha) * conj(y) * x^H
                    // which is equivalent to calling GERC with swapped vectors
                    cgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

/// Double precision complex conjugated rank-1 update: A = alpha * x * conj(y)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgerc must be registered via `register_zgerc`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zgerc(
    order: CBLAS_ORDER,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    a: *mut Complex64,
    lda: i32,
) {
    let p = get_zgerc_for_lp64_cblas();
    match p {
        ZgercProvider::Lp64(zgerc) => {
            match order {
                CblasColMajor => {
                    zgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for GERC: swap m<->n, swap x<->y
                    zgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        ZgercProvider::Ilp64(zgerc) => {
            let m = m as i64;
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;
            let lda = lda as i64;

            match order {
                CblasColMajor => {
                    zgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for GERC: swap m<->n, swap x<->y
                    zgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zgerc_64(
    order: CBLAS_ORDER,
    m: i64,
    n: i64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i64,
    y: *const Complex64,
    incy: i64,
    a: *mut Complex64,
    lda: i64,
) {
    let p = get_zgerc_for_ilp64_cblas();
    if matches!(p, ZgercProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zgerc_64\0",
            [(2, m), (3, n), (6, incx), (8, incy), (10, lda)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZgercProvider::Ilp64(zgerc) => {
            match order {
                CblasColMajor => {
                    zgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for GERC: swap m<->n, swap x<->y
                    zgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
        ZgercProvider::Lp64(zgerc) => {
            let m = m as i32;
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;
            let lda = lda as i32;

            match order {
                CblasColMajor => {
                    zgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
                }
                CblasRowMajor => {
                    // Row-major for GERC: swap m<->n, swap x<->y
                    zgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
                }
            }
        }
    }
}
