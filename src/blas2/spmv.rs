//! Symmetric/Hermitian packed matrix-vector multiply (SPMV/HPMV) - CBLAS interface.
//!
//! Computes: y = alpha * A * x + beta * y
//! where A is a symmetric (SPMV) or Hermitian (HPMV) matrix stored in packed format.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/spmv.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zhpmv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_chpmv_for_ilp64_cblas, get_chpmv_for_lp64_cblas, get_dspmv_for_ilp64_cblas,
    get_dspmv_for_lp64_cblas, get_sspmv_for_ilp64_cblas, get_sspmv_for_lp64_cblas,
    get_zhpmv_for_ilp64_cblas, get_zhpmv_for_lp64_cblas, ChpmvProvider, DspmvProvider,
    SspmvProvider, ZhpmvProvider,
};
use crate::types::{
    uplo_to_char, CblasColMajor, CblasLower, CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_UPLO,
};

/// Single precision symmetric packed matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a symmetric matrix stored in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - sspmv must be registered via `register_sspmv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_sspmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f32,
    ap: *const f32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    let p = get_sspmv_for_lp64_cblas();
    match p {
        SspmvProvider::Lp64(sspmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    sspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    sspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
            }
        }
        SspmvProvider::Ilp64(sspmv) => {
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    sspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    sspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_sspmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f32,
    ap: *const f32,
    x: *const f32,
    incx: i64,
    beta: f32,
    y: *mut f32,
    incy: i64,
) {
    let p = get_sspmv_for_ilp64_cblas();
    if matches!(p, SspmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_sspmv_64\0",
            [(3, n), (7, incx), (10, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        SspmvProvider::Ilp64(sspmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    sspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    sspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
            }
        }
        SspmvProvider::Lp64(sspmv) => {
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    sspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    sspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
            }
        }
    }
}

/// Double precision symmetric packed matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a symmetric matrix stored in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - dspmv must be registered via `register_dspmv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dspmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f64,
    ap: *const f64,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    let p = get_dspmv_for_lp64_cblas();
    match p {
        DspmvProvider::Lp64(dspmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
            }
        }
        DspmvProvider::Ilp64(dspmv) => {
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dspmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f64,
    ap: *const f64,
    x: *const f64,
    incx: i64,
    beta: f64,
    y: *mut f64,
    incy: i64,
) {
    let p = get_dspmv_for_ilp64_cblas();
    if matches!(p, DspmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dspmv_64\0",
            [(3, n), (7, incx), (10, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DspmvProvider::Ilp64(dspmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
            }
        }
        DspmvProvider::Lp64(dspmv) => {
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dspmv(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
                }
            }
        }
    }
}

/// Single precision complex Hermitian packed matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a Hermitian matrix stored in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - chpmv must be registered via `register_chpmv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chpmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: *const Complex32,
    ap: *const Complex32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    let p = get_chpmv_for_lp64_cblas();
    match p {
        ChpmvProvider::Lp64(chpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chpmv(&uplo_char, &n, alpha, ap, x, &incx, beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);

                    // Conjugate alpha and beta
                    let alpha_val = *alpha;
                    let beta_val = *beta;
                    let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
                    let conj_beta = Complex32::new(beta_val.re, -beta_val.im);

                    if n > 0 {
                        // Conjugate y (in-place before operation)
                        let abs_incy = if incy < 0 { -incy } else { incy };
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex32::new(val.re, -val.im);
                        }

                        // Create conjugated copy of x
                        let abs_incx = if incx < 0 { -incx } else { incx };
                        let mut x_conj = vec![Complex32::new(0.0, 0.0); n as usize];
                        for i in 0..n {
                            let idx = if incx < 0 {
                                ((n - 1 - i) * abs_incx) as isize
                            } else {
                                (i * abs_incx) as isize
                            };
                            let val = *x.offset(idx);
                            x_conj[i as usize] = Complex32::new(val.re, -val.im);
                        }

                        // Call Fortran HPMV with conjugated values
                        chpmv(
                            &uplo_char,
                            &n,
                            &conj_alpha,
                            ap,
                            x_conj.as_ptr(),
                            &1,
                            &conj_beta,
                            y,
                            &incy,
                        );

                        // Conjugate y (in-place after operation)
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex32::new(val.re, -val.im);
                        }
                    }
                }
            }
        }
        ChpmvProvider::Ilp64(chpmv) => {
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chpmv(&uplo_char, &n, alpha, ap, x, &incx, beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);

                    // Conjugate alpha and beta
                    let alpha_val = *alpha;
                    let beta_val = *beta;
                    let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
                    let conj_beta = Complex32::new(beta_val.re, -beta_val.im);

                    if n > 0 {
                        // Conjugate y (in-place before operation)
                        let abs_incy = if incy < 0 { -incy } else { incy };
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex32::new(val.re, -val.im);
                        }

                        // Create conjugated copy of x
                        let abs_incx = if incx < 0 { -incx } else { incx };
                        let mut x_conj = vec![Complex32::new(0.0, 0.0); n as usize];
                        for i in 0..n {
                            let idx = if incx < 0 {
                                ((n - 1 - i) * abs_incx) as isize
                            } else {
                                (i * abs_incx) as isize
                            };
                            let val = *x.offset(idx);
                            x_conj[i as usize] = Complex32::new(val.re, -val.im);
                        }

                        // Call Fortran HPMV with conjugated values
                        chpmv(
                            &uplo_char,
                            &n,
                            &conj_alpha,
                            ap,
                            x_conj.as_ptr(),
                            &1,
                            &conj_beta,
                            y,
                            &incy,
                        );

                        // Conjugate y (in-place after operation)
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex32::new(val.re, -val.im);
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chpmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: *const Complex32,
    ap: *const Complex32,
    x: *const Complex32,
    incx: i64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i64,
) {
    let p = get_chpmv_for_ilp64_cblas();
    if matches!(p, ChpmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_chpmv_64\0",
            [(3, n), (7, incx), (10, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ChpmvProvider::Ilp64(chpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chpmv(&uplo_char, &n, alpha, ap, x, &incx, beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);

                    // Conjugate alpha and beta
                    let alpha_val = *alpha;
                    let beta_val = *beta;
                    let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
                    let conj_beta = Complex32::new(beta_val.re, -beta_val.im);

                    if n > 0 {
                        // Conjugate y (in-place before operation)
                        let abs_incy = if incy < 0 { -incy } else { incy };
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex32::new(val.re, -val.im);
                        }

                        // Create conjugated copy of x
                        let abs_incx = if incx < 0 { -incx } else { incx };
                        let mut x_conj = vec![Complex32::new(0.0, 0.0); n as usize];
                        for i in 0..n {
                            let idx = if incx < 0 {
                                ((n - 1 - i) * abs_incx) as isize
                            } else {
                                (i * abs_incx) as isize
                            };
                            let val = *x.offset(idx);
                            x_conj[i as usize] = Complex32::new(val.re, -val.im);
                        }

                        // Call Fortran HPMV with conjugated values
                        chpmv(
                            &uplo_char,
                            &n,
                            &conj_alpha,
                            ap,
                            x_conj.as_ptr(),
                            &1,
                            &conj_beta,
                            y,
                            &incy,
                        );

                        // Conjugate y (in-place after operation)
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex32::new(val.re, -val.im);
                        }
                    }
                }
            }
        }
        ChpmvProvider::Lp64(chpmv) => {
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chpmv(&uplo_char, &n, alpha, ap, x, &incx, beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);

                    // Conjugate alpha and beta
                    let alpha_val = *alpha;
                    let beta_val = *beta;
                    let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
                    let conj_beta = Complex32::new(beta_val.re, -beta_val.im);

                    if n > 0 {
                        // Conjugate y (in-place before operation)
                        let abs_incy = if incy < 0 { -incy } else { incy };
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex32::new(val.re, -val.im);
                        }

                        // Create conjugated copy of x
                        let abs_incx = if incx < 0 { -incx } else { incx };
                        let mut x_conj = vec![Complex32::new(0.0, 0.0); n as usize];
                        for i in 0..n {
                            let idx = if incx < 0 {
                                ((n - 1 - i) * abs_incx) as isize
                            } else {
                                (i * abs_incx) as isize
                            };
                            let val = *x.offset(idx);
                            x_conj[i as usize] = Complex32::new(val.re, -val.im);
                        }

                        // Call Fortran HPMV with conjugated values
                        chpmv(
                            &uplo_char,
                            &n,
                            &conj_alpha,
                            ap,
                            x_conj.as_ptr(),
                            &1,
                            &conj_beta,
                            y,
                            &incy,
                        );

                        // Conjugate y (in-place after operation)
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex32::new(val.re, -val.im);
                        }
                    }
                }
            }
        }
    }
}

/// Double precision complex Hermitian packed matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a Hermitian matrix stored in packed format.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - zhpmv must be registered via `register_zhpmv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhpmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: *const Complex64,
    ap: *const Complex64,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    let p = get_zhpmv_for_lp64_cblas();
    match p {
        ZhpmvProvider::Lp64(zhpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhpmv(&uplo_char, &n, alpha, ap, x, &incx, beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);

                    // Conjugate alpha and beta
                    let alpha_val = *alpha;
                    let beta_val = *beta;
                    let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
                    let conj_beta = Complex64::new(beta_val.re, -beta_val.im);

                    if n > 0 {
                        // Conjugate y (in-place before operation)
                        let abs_incy = if incy < 0 { -incy } else { incy };
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex64::new(val.re, -val.im);
                        }

                        // Create conjugated copy of x
                        let abs_incx = if incx < 0 { -incx } else { incx };
                        let mut x_conj = vec![Complex64::new(0.0, 0.0); n as usize];
                        for i in 0..n {
                            let idx = if incx < 0 {
                                ((n - 1 - i) * abs_incx) as isize
                            } else {
                                (i * abs_incx) as isize
                            };
                            let val = *x.offset(idx);
                            x_conj[i as usize] = Complex64::new(val.re, -val.im);
                        }

                        // Call Fortran HPMV with conjugated values
                        zhpmv(
                            &uplo_char,
                            &n,
                            &conj_alpha,
                            ap,
                            x_conj.as_ptr(),
                            &1,
                            &conj_beta,
                            y,
                            &incy,
                        );

                        // Conjugate y (in-place after operation)
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex64::new(val.re, -val.im);
                        }
                    }
                }
            }
        }
        ZhpmvProvider::Ilp64(zhpmv) => {
            let n = n as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhpmv(&uplo_char, &n, alpha, ap, x, &incx, beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);

                    // Conjugate alpha and beta
                    let alpha_val = *alpha;
                    let beta_val = *beta;
                    let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
                    let conj_beta = Complex64::new(beta_val.re, -beta_val.im);

                    if n > 0 {
                        // Conjugate y (in-place before operation)
                        let abs_incy = if incy < 0 { -incy } else { incy };
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex64::new(val.re, -val.im);
                        }

                        // Create conjugated copy of x
                        let abs_incx = if incx < 0 { -incx } else { incx };
                        let mut x_conj = vec![Complex64::new(0.0, 0.0); n as usize];
                        for i in 0..n {
                            let idx = if incx < 0 {
                                ((n - 1 - i) * abs_incx) as isize
                            } else {
                                (i * abs_incx) as isize
                            };
                            let val = *x.offset(idx);
                            x_conj[i as usize] = Complex64::new(val.re, -val.im);
                        }

                        // Call Fortran HPMV with conjugated values
                        zhpmv(
                            &uplo_char,
                            &n,
                            &conj_alpha,
                            ap,
                            x_conj.as_ptr(),
                            &1,
                            &conj_beta,
                            y,
                            &incy,
                        );

                        // Conjugate y (in-place after operation)
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex64::new(val.re, -val.im);
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhpmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: *const Complex64,
    ap: *const Complex64,
    x: *const Complex64,
    incx: i64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i64,
) {
    let p = get_zhpmv_for_ilp64_cblas();
    if matches!(p, ZhpmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zhpmv_64\0",
            [(3, n), (7, incx), (10, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZhpmvProvider::Ilp64(zhpmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhpmv(&uplo_char, &n, alpha, ap, x, &incx, beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);

                    // Conjugate alpha and beta
                    let alpha_val = *alpha;
                    let beta_val = *beta;
                    let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
                    let conj_beta = Complex64::new(beta_val.re, -beta_val.im);

                    if n > 0 {
                        // Conjugate y (in-place before operation)
                        let abs_incy = if incy < 0 { -incy } else { incy };
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex64::new(val.re, -val.im);
                        }

                        // Create conjugated copy of x
                        let abs_incx = if incx < 0 { -incx } else { incx };
                        let mut x_conj = vec![Complex64::new(0.0, 0.0); n as usize];
                        for i in 0..n {
                            let idx = if incx < 0 {
                                ((n - 1 - i) * abs_incx) as isize
                            } else {
                                (i * abs_incx) as isize
                            };
                            let val = *x.offset(idx);
                            x_conj[i as usize] = Complex64::new(val.re, -val.im);
                        }

                        // Call Fortran HPMV with conjugated values
                        zhpmv(
                            &uplo_char,
                            &n,
                            &conj_alpha,
                            ap,
                            x_conj.as_ptr(),
                            &1,
                            &conj_beta,
                            y,
                            &incy,
                        );

                        // Conjugate y (in-place after operation)
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex64::new(val.re, -val.im);
                        }
                    }
                }
            }
        }
        ZhpmvProvider::Lp64(zhpmv) => {
            let n = n as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhpmv(&uplo_char, &n, alpha, ap, x, &incx, beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);

                    // Conjugate alpha and beta
                    let alpha_val = *alpha;
                    let beta_val = *beta;
                    let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
                    let conj_beta = Complex64::new(beta_val.re, -beta_val.im);

                    if n > 0 {
                        // Conjugate y (in-place before operation)
                        let abs_incy = if incy < 0 { -incy } else { incy };
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex64::new(val.re, -val.im);
                        }

                        // Create conjugated copy of x
                        let abs_incx = if incx < 0 { -incx } else { incx };
                        let mut x_conj = vec![Complex64::new(0.0, 0.0); n as usize];
                        for i in 0..n {
                            let idx = if incx < 0 {
                                ((n - 1 - i) * abs_incx) as isize
                            } else {
                                (i * abs_incx) as isize
                            };
                            let val = *x.offset(idx);
                            x_conj[i as usize] = Complex64::new(val.re, -val.im);
                        }

                        // Call Fortran HPMV with conjugated values
                        zhpmv(
                            &uplo_char,
                            &n,
                            &conj_alpha,
                            ap,
                            x_conj.as_ptr(),
                            &1,
                            &conj_beta,
                            y,
                            &incy,
                        );

                        // Conjugate y (in-place after operation)
                        for i in 0..n {
                            let idx = if incy < 0 {
                                ((n - 1 - i) * abs_incy) as isize
                            } else {
                                (i * abs_incy) as isize
                            };
                            let y_ptr = y.offset(idx);
                            let val = *y_ptr;
                            *y_ptr = Complex64::new(val.re, -val.im);
                        }
                    }
                }
            }
        }
    }
}
