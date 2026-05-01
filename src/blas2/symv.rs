//! Symmetric/Hermitian matrix-vector multiply (SYMV/HEMV) - CBLAS interface.
//!
//! Computes: y = alpha * A * x + beta * y
//! where A is symmetric (SYMV) or Hermitian (HEMV).
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symv.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zhemv.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_chemv_for_ilp64_cblas, get_chemv_for_lp64_cblas, get_dsymv_for_ilp64_cblas,
    get_dsymv_for_lp64_cblas, get_ssymv_for_ilp64_cblas, get_ssymv_for_lp64_cblas,
    get_zhemv_for_ilp64_cblas, get_zhemv_for_lp64_cblas, ChemvProvider, DsymvProvider,
    SsymvProvider, ZhemvProvider,
};
use crate::types::{
    uplo_to_char, CblasColMajor, CblasLower, CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_UPLO,
};

/// Single precision symmetric matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a symmetric matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssymv must be registered via `register_ssymv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssymv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    let p = get_ssymv_for_lp64_cblas();
    match p {
        SsymvProvider::Lp64(ssymv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symv.c
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
            }
        }
        SsymvProvider::Ilp64(ssymv) => {
            let n = n as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symv.c
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssymv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    x: *const f32,
    incx: i64,
    beta: f32,
    y: *mut f32,
    incy: i64,
) {
    let p = get_ssymv_for_ilp64_cblas();
    if matches!(p, SsymvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ssymv_64\0",
            [(3, n), (6, lda), (8, incx), (11, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        SsymvProvider::Ilp64(ssymv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symv.c
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
            }
        }
        SsymvProvider::Lp64(ssymv) => {
            let n = n as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    ssymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symv.c
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    ssymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
            }
        }
    }
}

/// Double precision symmetric matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a symmetric matrix.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsymv must be registered via `register_dsymv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsymv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    let p = get_dsymv_for_lp64_cblas();
    match p {
        DsymvProvider::Lp64(dsymv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
            }
        }
        DsymvProvider::Ilp64(dsymv) => {
            let n = n as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsymv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    x: *const f64,
    incx: i64,
    beta: f64,
    y: *mut f64,
    incy: i64,
) {
    let p = get_dsymv_for_ilp64_cblas();
    if matches!(p, DsymvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dsymv_64\0",
            [(3, n), (6, lda), (8, incx), (11, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DsymvProvider::Ilp64(dsymv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
            }
        }
        DsymvProvider::Lp64(dsymv) => {
            let n = n as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    dsymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
                CblasRowMajor => {
                    // Row-major: swap Upper/Lower
                    let new_uplo = match uplo {
                        CblasUpper => CblasLower,
                        CblasLower => CblasUpper,
                    };
                    let uplo_char = uplo_to_char(new_uplo);
                    dsymv(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
                }
            }
        }
    }
}

/// Single precision complex Hermitian matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a Hermitian matrix.
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
/// - chemv must be registered via `register_chemv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chemv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    let p = get_chemv_for_lp64_cblas();
    match p {
        ChemvProvider::Lp64(chemv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chemv(&uplo_char, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
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

                        // Call Fortran HEMV with conjugated values
                        chemv(
                            &uplo_char,
                            &n,
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
        ChemvProvider::Ilp64(chemv) => {
            let n = n as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chemv(&uplo_char, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
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

                        // Call Fortran HEMV with conjugated values
                        chemv(
                            &uplo_char,
                            &n,
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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chemv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i64,
    x: *const Complex32,
    incx: i64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i64,
) {
    let p = get_chemv_for_ilp64_cblas();
    if matches!(p, ChemvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_chemv_64\0",
            [(3, n), (6, lda), (8, incx), (11, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ChemvProvider::Ilp64(chemv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chemv(&uplo_char, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
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

                        // Call Fortran HEMV with conjugated values
                        chemv(
                            &uplo_char,
                            &n,
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
        ChemvProvider::Lp64(chemv) => {
            let n = n as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chemv(&uplo_char, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
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

                        // Call Fortran HEMV with conjugated values
                        chemv(
                            &uplo_char,
                            &n,
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
    }
}

/// Double precision complex Hermitian matrix-vector multiply.
///
/// Computes: y = alpha * A * x + beta * y
/// where A is a Hermitian matrix.
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
/// - zhemv must be registered via `register_zhemv`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhemv(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    let p = get_zhemv_for_lp64_cblas();
    match p {
        ZhemvProvider::Lp64(zhemv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhemv(&uplo_char, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
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

                        // Call Fortran HEMV with conjugated values
                        zhemv(
                            &uplo_char,
                            &n,
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
        ZhemvProvider::Ilp64(zhemv) => {
            let n = n as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhemv(&uplo_char, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
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

                        // Call Fortran HEMV with conjugated values
                        zhemv(
                            &uplo_char,
                            &n,
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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhemv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    x: *const Complex64,
    incx: i64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i64,
) {
    let p = get_zhemv_for_ilp64_cblas();
    if matches!(p, ZhemvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zhemv_64\0",
            [(3, n), (6, lda), (8, incx), (11, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZhemvProvider::Ilp64(zhemv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhemv(&uplo_char, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
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

                        // Call Fortran HEMV with conjugated values
                        zhemv(
                            &uplo_char,
                            &n,
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
        ZhemvProvider::Lp64(zhemv) => {
            let n = n as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhemv(&uplo_char, &n, alpha, a, &lda, x, &incx, beta, y, &incy);
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

                        // Call Fortran HEMV with conjugated values
                        zhemv(
                            &uplo_char,
                            &n,
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
    }
}
