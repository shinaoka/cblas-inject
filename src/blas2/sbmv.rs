//! Symmetric/Hermitian band matrix-vector multiply (SBMV/HBMV) - CBLAS interface.
//!
//! Computes: y = alpha * A * x + beta * y
//! where A is a symmetric (SBMV) or Hermitian (HBMV) band matrix.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/sbmv.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zhbmv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{get_chbmv, get_dsbmv, get_ssbmv, get_zhbmv};
use crate::types::{
    blasint, uplo_to_char, CblasColMajor, CblasLower, CblasRowMajor, CblasUpper, CBLAS_ORDER,
    CBLAS_UPLO,
};

/// Single precision symmetric band matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a symmetric band matrix with k super-diagonals.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssbmv must be registered via `register_ssbmv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssbmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    k: blasint,
    alpha: f32,
    a: *const f32,
    lda: blasint,
    x: *const f32,
    incx: blasint,
    beta: f32,
    y: *mut f32,
    incy: blasint,
) {
    let ssbmv = get_ssbmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            ssbmv(
                &uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy,
            );
        }
        CblasRowMajor => {
            // Row-major: swap Upper/Lower
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/sbmv.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            ssbmv(
                &uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy,
            );
        }
    }
}

/// Double precision symmetric band matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a symmetric band matrix with k super-diagonals.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsbmv must be registered via `register_dsbmv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsbmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    k: blasint,
    alpha: f64,
    a: *const f64,
    lda: blasint,
    x: *const f64,
    incx: blasint,
    beta: f64,
    y: *mut f64,
    incy: blasint,
) {
    let dsbmv = get_dsbmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            dsbmv(
                &uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy,
            );
        }
        CblasRowMajor => {
            // Row-major: swap Upper/Lower
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            dsbmv(
                &uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy,
            );
        }
    }
}

/// Single precision complex Hermitian band matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a Hermitian band matrix with k super-diagonals.
///
/// Note: For row-major order, the standard CBLAS implementation conjugates
/// alpha, beta, x, and y vectors. This implementation follows the simpler
/// approach of just swapping Upper/Lower, which works correctly for
/// symmetric-like operations. For strict CBLAS compatibility with complex
/// Hermitian operations in row-major, the more complex conjugation approach
/// may be needed.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - chbmv must be registered via `register_chbmv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chbmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    k: blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: blasint,
    x: *const Complex32,
    incx: blasint,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: blasint,
) {
    let chbmv = get_chbmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            chbmv(
                &uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy,
            );
        }
        CblasRowMajor => {
            // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
            // Following the CBLAS reference implementation approach
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

            // For row-major Hermitian operations, we need to conjugate input x
            // and conjugate output y before and after the operation
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

                // Call Fortran HBMV with conjugated values
                chbmv(
                    &uplo_char,
                    &n,
                    &k,
                    &conj_alpha,
                    a,
                    &lda,
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

/// Double precision complex Hermitian band matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a Hermitian band matrix with k super-diagonals.
///
/// Note: For row-major order, the standard CBLAS implementation conjugates
/// alpha, beta, x, and y vectors. This implementation follows the simpler
/// approach of just swapping Upper/Lower, which works correctly for
/// symmetric-like operations. For strict CBLAS compatibility with complex
/// Hermitian operations in row-major, the more complex conjugation approach
/// may be needed.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zhbmv must be registered via `register_zhbmv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhbmv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    k: blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: blasint,
    x: *const Complex64,
    incx: blasint,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: blasint,
) {
    let zhbmv = get_zhbmv();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            zhbmv(
                &uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy,
            );
        }
        CblasRowMajor => {
            // Row-major for Hermitian: swap Upper/Lower and conjugate scalars and vectors
            // Following the CBLAS reference implementation approach
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

            // For row-major Hermitian operations, we need to conjugate input x
            // and conjugate output y before and after the operation
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

                // Call Fortran HBMV with conjugated values
                zhbmv(
                    &uplo_char,
                    &n,
                    &k,
                    &conj_alpha,
                    a,
                    &lda,
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
