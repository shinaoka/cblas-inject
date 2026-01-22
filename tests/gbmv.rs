//! Pure-Rust layout tests for GBMV (complex types).
//!
//! Policy:
//! - Do NOT modify existing OpenBLAS-derived tests.
//! - Add additional tests that validate row-major conversion logic by comparing
//!   `order=RowMajor` vs `order=ColMajor` results for the *same logical band matrix*.
//!
//! Notes:
//! - Band matrix storage format: ColMajor uses kl+ku+1 rows, RowMajor swaps m/n and kl/ku.
//! - These tests still use OpenBLAS as the backend BLAS (via `blas_src`) for the
//!   registered Fortran symbols, but the *test oracle* is purely internal:
//!   RowMajor result must equal ColMajor result after providing equivalent inputs.

extern crate blas_src;

use cblas_inject::{
    blasint, register_cgbmv, register_zgbmv, CblasColMajor, CblasConjTrans, CblasNoTrans,
    CblasRowMajor, CblasTrans,
};
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

#[macro_use]
mod common;
use common::{
    assert_complex32_eq, assert_complex64_eq, calc_lda_gbmv, create_band_matrix_col,
    create_band_matrix_row, x_len_gemv, y_len_gemv,
};

// Fortran BLAS function declarations (provided by linked OpenBLAS)
extern "C" {
    fn cgbmv_(
        trans: *const c_char,
        m: *const blasint,
        n: *const blasint,
        kl: *const blasint,
        ku: *const blasint,
        alpha: *const Complex32,
        a: *const Complex32,
        lda: *const blasint,
        x: *const Complex32,
        incx: *const blasint,
        beta: *const Complex32,
        y: *mut Complex32,
        incy: *const blasint,
    );

    fn zgbmv_(
        trans: *const c_char,
        m: *const blasint,
        n: *const blasint,
        kl: *const blasint,
        ku: *const blasint,
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

setup_once!(setup_cgbmv, register_cgbmv, cgbmv_);
setup_once!(setup_zgbmv, register_zgbmv, zgbmv_);

#[test]
fn cgbmv_row_vs_col_agree() {
    setup_cgbmv();

    let cases = [
        (3usize, 4usize, 1usize, 1usize), // m, n, kl, ku
        (4, 3, 1, 1),
        (5, 5, 2, 1),
        (6, 4, 1, 2),
    ];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let alpha = Complex32::new(0.7, 0.3);
    let beta = Complex32::new(1.3, -0.5);

    for &(m, n, kl, ku) in &cases {
        for &trans in &transposes {
            // Create the same logical band matrix in both layouts
            let a_col = create_band_matrix_col(m, n, kl, ku, |i, j| {
                Complex32::new(((i + 3 * j) as f32 * 0.1).sin(), ((7 * i + j) as f32 * 0.2).cos())
            });
            let a_row = create_band_matrix_row(m, n, kl, ku, |i, j| {
                Complex32::new(((i + 3 * j) as f32 * 0.1).sin(), ((7 * i + j) as f32 * 0.2).cos())
            });

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

            let lda_col = calc_lda_gbmv(kl, ku);
            let lda_row = calc_lda_gbmv(ku, kl); // swapped kl/ku for RowMajor

            unsafe {
                cblas_inject::cblas_cgbmv(
                    CblasRowMajor,
                    trans,
                    m as blasint,
                    n as blasint,
                    kl as blasint,
                    ku as blasint,
                    &alpha,
                    a_row.as_ptr(),
                    lda_row,
                    x.as_ptr(),
                    1,
                    &beta,
                    y_row.as_mut_ptr(),
                    1,
                );
                cblas_inject::cblas_cgbmv(
                    CblasColMajor,
                    trans,
                    m as blasint,
                    n as blasint,
                    kl as blasint,
                    ku as blasint,
                    &alpha,
                    a_col.as_ptr(),
                    lda_col,
                    x.as_ptr(),
                    1,
                    &beta,
                    y_col.as_mut_ptr(),
                    1,
                );
            }

            let context = format!("cgbmv row-vs-col: m={}, n={}, kl={}, ku={}, trans={:?}", m, n, kl, ku, trans);
            assert_complex32_eq(&y_row, &y_col, 1e-5, &context);
        }
    }
}

#[test]
fn zgbmv_row_vs_col_agree() {
    setup_zgbmv();

    let cases = [
        (3usize, 4usize, 1usize, 1usize),
        (4, 3, 1, 1),
        (5, 5, 2, 1),
        (6, 4, 1, 2),
    ];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let alpha = Complex64::new(0.7, 0.3);
    let beta = Complex64::new(1.3, -0.5);

    for &(m, n, kl, ku) in &cases {
        for &trans in &transposes {
            let a_col = create_band_matrix_col(m, n, kl, ku, |i, j| {
                Complex64::new(((i + 3 * j) as f64 * 0.1).sin(), ((7 * i + j) as f64 * 0.2).cos())
            });
            let a_row = create_band_matrix_row(m, n, kl, ku, |i, j| {
                Complex64::new(((i + 3 * j) as f64 * 0.1).sin(), ((7 * i + j) as f64 * 0.2).cos())
            });

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

            let lda_col = calc_lda_gbmv(kl, ku);
            let lda_row = calc_lda_gbmv(ku, kl);

            unsafe {
                cblas_inject::cblas_zgbmv(
                    CblasRowMajor,
                    trans,
                    m as blasint,
                    n as blasint,
                    kl as blasint,
                    ku as blasint,
                    &alpha,
                    a_row.as_ptr(),
                    lda_row,
                    x.as_ptr(),
                    1,
                    &beta,
                    y_row.as_mut_ptr(),
                    1,
                );
                cblas_inject::cblas_zgbmv(
                    CblasColMajor,
                    trans,
                    m as blasint,
                    n as blasint,
                    kl as blasint,
                    ku as blasint,
                    &alpha,
                    a_col.as_ptr(),
                    lda_col,
                    x.as_ptr(),
                    1,
                    &beta,
                    y_col.as_mut_ptr(),
                    1,
                );
            }

            let context = format!("zgbmv row-vs-col: m={}, n={}, kl={}, ku={}, trans={:?}", m, n, kl, ku, trans);
            assert_complex64_eq(&y_row, &y_col, 1e-12, &context);
        }
    }
}
