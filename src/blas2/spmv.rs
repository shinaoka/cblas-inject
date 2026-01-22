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

use crate::backend::{get_chpmv, get_dspmv, get_sspmv, get_zhpmv};
use crate::types::{
    blasint, uplo_to_char, CblasColMajor, CblasLower, CblasRowMajor, CblasUpper, CBLAS_ORDER,
    CBLAS_UPLO,
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
    n: blasint,
    alpha: f32,
    ap: *const f32,
    x: *const f32,
    incx: blasint,
    beta: f32,
    y: *mut f32,
    incy: blasint,
) {
    let sspmv = get_sspmv();

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
    n: blasint,
    alpha: f64,
    ap: *const f64,
    x: *const f64,
    incx: blasint,
    beta: f64,
    y: *mut f64,
    incy: blasint,
) {
    let dspmv = get_dspmv();

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
    n: blasint,
    alpha: *const Complex32,
    ap: *const Complex32,
    x: *const Complex32,
    incx: blasint,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: blasint,
) {
    let chpmv = get_chpmv();

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
    n: blasint,
    alpha: *const Complex64,
    ap: *const Complex64,
    x: *const Complex64,
    incx: blasint,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: blasint,
) {
    let zhpmv = get_zhpmv();

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
