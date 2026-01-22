//! Pure-Rust layout tests for GEMV (complex types).
//!
//! Policy:
//! - Do NOT modify existing OpenBLAS-derived tests.
//! - Add additional tests that validate row-major conversion logic by comparing
//!   `order=RowMajor` vs `order=ColMajor` results for the *same logical matrix*.
//!
//! Notes:
//! - These tests still use OpenBLAS as the backend BLAS (via `blas_src`) for the
//!   registered Fortran symbols, but the *test oracle* is purely internal:
//!   RowMajor result must equal ColMajor result after providing equivalent inputs.

extern crate blas_src;

use cblas_inject::{
    blasint, register_cgemv, register_zgemv, CblasColMajor, CblasConjTrans, CblasNoTrans,
    CblasRowMajor, CblasTrans,
};
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

#[macro_use]
mod common;
use common::{assert_complex32_eq, assert_complex64_eq, x_len_gemv, y_len_gemv, Layout, Matrix};

// Fortran BLAS function declarations (provided by linked OpenBLAS)
extern "C" {
    fn cgemv_(
        trans: *const c_char,
        m: *const blasint,
        n: *const blasint,
        alpha: *const Complex32,
        a: *const Complex32,
        lda: *const blasint,
        x: *const Complex32,
        incx: *const blasint,
        beta: *const Complex32,
        y: *mut Complex32,
        incy: *const blasint,
    );

    fn zgemv_(
        trans: *const c_char,
        m: *const blasint,
        n: *const blasint,
        alpha: *const Complex64,
        a: *const Complex64,
        lda: *const blasint,
        x: *const Complex64,
        incx: *const blasint,
        beta: *const Complex64,
        y: *mut Complex64,
        incy: *const blasint,
    );
}

setup_once!(setup_cgemv, register_cgemv, cgemv_);
setup_once!(setup_zgemv, register_zgemv, zgemv_);

#[test]
fn cgemv_row_vs_col_agree() {
    setup_cgemv();

    let cases = [
        (1usize, 1usize),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
        (5, 7),
    ];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let alpha = Complex32::new(0.7, 0.3);
    let beta = Complex32::new(1.3, -0.5);

    for &(m, n) in &cases {
        for &trans in &transposes {
            // Build the same logical matrix in both layouts, with padding LDA.
            let a_row = Matrix::new_row_major(m, n, n + 2, |i, j| {
                Complex32::new(((i + 3 * j) as f32 * 0.1).sin(), ((7 * i + j) as f32 * 0.2).cos())
            });
            let a_col = a_row.to_layout(Layout::ColMajor, m + 2);

            let xl = x_len_gemv(trans, m, n);
            let yl = y_len_gemv(trans, m, n);
            let x: Vec<Complex32> = (0..xl)
                .map(|k| Complex32::new(((k + 11) as f32 * 0.15).cos(), ((k + 5) as f32 * 0.25).sin()))
                .collect();
            let y0: Vec<Complex32> = (0..yl)
                .map(|k| Complex32::new(((k + 17) as f32 * 0.05).sin(), ((k + 19) as f32 * 0.07).cos()))
                .collect();

            let mut y_row = y0.clone();
            let mut y_col = y0.clone();

            unsafe {
                cblas_inject::cblas_cgemv(
                    CblasRowMajor,
                    trans,
                    m as blasint,
                    n as blasint,
                    &alpha,
                    a_row.as_ptr(),
                    a_row.lda_blasint(),
                    x.as_ptr(),
                    1,
                    &beta,
                    y_row.as_mut_ptr(),
                    1,
                );
                cblas_inject::cblas_cgemv(
                    CblasColMajor,
                    trans,
                    m as blasint,
                    n as blasint,
                    &alpha,
                    a_col.as_ptr(),
                    a_col.lda_blasint(),
                    x.as_ptr(),
                    1,
                    &beta,
                    y_col.as_mut_ptr(),
                    1,
                );
            }

            let context = format!("cgemv row-vs-col: m={}, n={}, trans={:?}", m, n, trans);
            assert_complex32_eq(&y_row, &y_col, 1e-5, &context);
        }
    }
}

#[test]
fn zgemv_row_vs_col_agree() {
    setup_zgemv();

    let cases = [
        (1usize, 1usize),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
        (5, 7),
    ];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let alpha = Complex64::new(0.7, 0.3);
    let beta = Complex64::new(1.3, -0.5);

    for &(m, n) in &cases {
        for &trans in &transposes {
            let a_row = Matrix::new_row_major(m, n, n + 2, |i, j| {
                Complex64::new(((i + 3 * j) as f64 * 0.1).sin(), ((7 * i + j) as f64 * 0.2).cos())
            });
            let a_col = a_row.to_layout(Layout::ColMajor, m + 2);

            let xl = x_len_gemv(trans, m, n);
            let yl = y_len_gemv(trans, m, n);
            let x: Vec<Complex64> = (0..xl)
                .map(|k| Complex64::new(((k + 11) as f64 * 0.15).cos(), ((k + 5) as f64 * 0.25).sin()))
                .collect();
            let y0: Vec<Complex64> = (0..yl)
                .map(|k| Complex64::new(((k + 17) as f64 * 0.05).sin(), ((k + 19) as f64 * 0.07).cos()))
                .collect();

            let mut y_row = y0.clone();
            let mut y_col = y0.clone();

            unsafe {
                cblas_inject::cblas_zgemv(
                    CblasRowMajor,
                    trans,
                    m as blasint,
                    n as blasint,
                    &alpha,
                    a_row.as_ptr(),
                    a_row.lda_blasint(),
                    x.as_ptr(),
                    1,
                    &beta,
                    y_row.as_mut_ptr(),
                    1,
                );
                cblas_inject::cblas_zgemv(
                    CblasColMajor,
                    trans,
                    m as blasint,
                    n as blasint,
                    &alpha,
                    a_col.as_ptr(),
                    a_col.lda_blasint(),
                    x.as_ptr(),
                    1,
                    &beta,
                    y_col.as_mut_ptr(),
                    1,
                );
            }

            let context = format!("zgemv row-vs-col: m={}, n={}, trans={:?}", m, n, trans);
            assert_complex64_eq(&y_row, &y_col, 1e-12, &context);
        }
    }
}

