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

use crate::backend::{
    get_chbmv_for_ilp64_cblas, get_chbmv_for_lp64_cblas, get_dsbmv_for_ilp64_cblas,
    get_dsbmv_for_lp64_cblas, get_ssbmv_for_ilp64_cblas, get_ssbmv_for_lp64_cblas,
    get_zhbmv_for_ilp64_cblas, get_zhbmv_for_lp64_cblas, ChbmvProvider, DsbmvProvider,
    SsbmvProvider, ZhbmvProvider,
};
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
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    let p = get_ssbmv_for_lp64_cblas();
    match p {
        SsbmvProvider::Lp64(ssbmv) => {
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
        SsbmvProvider::Ilp64(ssbmv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssbmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    x: *const f32,
    incx: i64,
    beta: f32,
    y: *mut f32,
    incy: i64,
) {
    let p = get_ssbmv_for_ilp64_cblas();
    if matches!(p, SsbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ssbmv_64\0",
            [(3, n), (4, k), (7, lda), (9, incx), (12, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        SsbmvProvider::Ilp64(ssbmv) => {
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
        SsbmvProvider::Lp64(ssbmv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

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
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    let p = get_dsbmv_for_lp64_cblas();
    match p {
        DsbmvProvider::Lp64(dsbmv) => {
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
        DsbmvProvider::Ilp64(dsbmv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsbmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    k: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    x: *const f64,
    incx: i64,
    beta: f64,
    y: *mut f64,
    incy: i64,
) {
    let p = get_dsbmv_for_ilp64_cblas();
    if matches!(p, DsbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dsbmv_64\0",
            [(3, n), (4, k), (7, lda), (9, incx), (12, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DsbmvProvider::Ilp64(dsbmv) => {
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
        DsbmvProvider::Lp64(dsbmv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

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
    n: i32,
    k: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    let p = get_chbmv_for_lp64_cblas();
    match p {
        ChbmvProvider::Lp64(chbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chbmv(&uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy);
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
        ChbmvProvider::Ilp64(chbmv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chbmv(&uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy);
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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chbmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    k: i64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i64,
    x: *const Complex32,
    incx: i64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i64,
) {
    let p = get_chbmv_for_ilp64_cblas();
    if matches!(p, ChbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_chbmv_64\0",
            [(3, n), (4, k), (7, lda), (9, incx), (12, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ChbmvProvider::Ilp64(chbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chbmv(&uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy);
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
        ChbmvProvider::Lp64(chbmv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    chbmv(&uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy);
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
    n: i32,
    k: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    let p = get_zhbmv_for_lp64_cblas();
    match p {
        ZhbmvProvider::Lp64(zhbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhbmv(&uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy);
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
        ZhbmvProvider::Ilp64(zhbmv) => {
            let n = n as i64;
            let k = k as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhbmv(&uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy);
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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhbmv_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    k: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    x: *const Complex64,
    incx: i64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i64,
) {
    let p = get_zhbmv_for_ilp64_cblas();
    if matches!(p, ZhbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zhbmv_64\0",
            [(3, n), (4, k), (7, lda), (9, incx), (12, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZhbmvProvider::Ilp64(zhbmv) => {
            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhbmv(&uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy);
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
        ZhbmvProvider::Lp64(zhbmv) => {
            let n = n as i32;
            let k = k as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let uplo_char = uplo_to_char(uplo);
                    zhbmv(&uplo_char, &n, &k, alpha, a, &lda, x, &incx, beta, y, &incy);
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
    }
}
