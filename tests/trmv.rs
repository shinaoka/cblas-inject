//! Pure-Rust layout tests for TRMV (complex types).
//!
//! Policy:
//! - Do NOT modify existing OpenBLAS-derived tests.
//! - Add additional tests that validate row-major conversion logic by comparing
//!   `order=RowMajor` vs `order=ColMajor` results for the *same logical triangular matrix*.

extern crate blas_src;

use cblas_inject::{
    blasint, register_ctrmv, register_ztrmv, CblasColMajor, CblasConjTrans, CblasLower,
    CblasNoTrans, CblasNonUnit, CblasRowMajor, CblasTrans, CblasUpper,
};
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

#[macro_use]
mod common;
use common::{assert_complex32_eq, assert_complex64_eq, Layout, Matrix};

// Fortran BLAS function declarations (provided by linked OpenBLAS)
extern "C" {
    fn ctrmv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blasint,
        a: *const Complex32,
        lda: *const blasint,
        x: *mut Complex32,
        incx: *const blasint,
    );

    fn ztrmv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blasint,
        a: *const Complex64,
        lda: *const blasint,
        x: *mut Complex64,
        incx: *const blasint,
    );
}

setup_once!(setup_ctrmv, register_ctrmv, ctrmv_);
setup_once!(setup_ztrmv, register_ztrmv, ztrmv_);

#[test]
fn ctrmv_row_vs_col_agree() {
    setup_ctrmv();

    let cases = [1usize, 2, 3, 4, 5];
    let uplos = [CblasUpper, CblasLower];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let diags = [CblasNonUnit, cblas_inject::CblasUnit];

    for &n in &cases {
        for &uplo in &uplos {
            for &trans in &transposes {
                for &diag in &diags {
                    // Create triangular matrix
                    let a_row = Matrix::new_row_major(n, n, n + 2, |i, j| {
                        let is_triangular = match uplo {
                            CblasUpper => i <= j,
                            CblasLower => i >= j,
                        };
                        if is_triangular && (diag != cblas_inject::CblasUnit || i != j) {
                            Complex32::new(((i + 3 * j) as f32 * 0.1).sin(), ((7 * i + j) as f32 * 0.2).cos())
                        } else {
                            Complex32::new(0.0, 0.0)
                        }
                    });
                    let a_col = a_row.to_layout(Layout::ColMajor, n + 2);

                    let x0: Vec<Complex32> = (0..n)
                        .map(|k| Complex32::new(((k + 11) as f32 * 0.15).cos(), ((k + 5) as f32 * 0.25).sin()))
                        .collect();

                    let mut x_row = x0.clone();
                    let mut x_col = x0.clone();

                    unsafe {
                        cblas_inject::cblas_ctrmv(
                            CblasRowMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            a_row.as_ptr(),
                            a_row.lda_blasint(),
                            x_row.as_mut_ptr(),
                            1,
                        );
                        cblas_inject::cblas_ctrmv(
                            CblasColMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            a_col.as_ptr(),
                            a_col.lda_blasint(),
                            x_col.as_mut_ptr(),
                            1,
                        );
                    }

                    let context = format!("ctrmv row-vs-col: n={}, uplo={:?}, trans={:?}, diag={:?}", n, uplo, trans, diag);
                    assert_complex32_eq(&x_row, &x_col, 1e-5, &context);
                }
            }
        }
    }
}

#[test]
fn ztrmv_row_vs_col_agree() {
    setup_ztrmv();

    let cases = [1usize, 2, 3, 4, 5];
    let uplos = [CblasUpper, CblasLower];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let diags = [CblasNonUnit, cblas_inject::CblasUnit];

    for &n in &cases {
        for &uplo in &uplos {
            for &trans in &transposes {
                for &diag in &diags {
                    let a_row = Matrix::new_row_major(n, n, n + 2, |i, j| {
                        let is_triangular = match uplo {
                            CblasUpper => i <= j,
                            CblasLower => i >= j,
                        };
                        if is_triangular && (diag != cblas_inject::CblasUnit || i != j) {
                            Complex64::new(((i + 3 * j) as f64 * 0.1).sin(), ((7 * i + j) as f64 * 0.2).cos())
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    });
                    let a_col = a_row.to_layout(Layout::ColMajor, n + 2);

                    let x0: Vec<Complex64> = (0..n)
                        .map(|k| Complex64::new(((k + 11) as f64 * 0.15).cos(), ((k + 5) as f64 * 0.25).sin()))
                        .collect();

                    let mut x_row = x0.clone();
                    let mut x_col = x0.clone();

                    unsafe {
                        cblas_inject::cblas_ztrmv(
                            CblasRowMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            a_row.as_ptr(),
                            a_row.lda_blasint(),
                            x_row.as_mut_ptr(),
                            1,
                        );
                        cblas_inject::cblas_ztrmv(
                            CblasColMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            a_col.as_ptr(),
                            a_col.lda_blasint(),
                            x_col.as_mut_ptr(),
                            1,
                        );
                    }

                    let context = format!("ztrmv row-vs-col: n={}, uplo={:?}, trans={:?}, diag={:?}", n, uplo, trans, diag);
                    assert_complex64_eq(&x_row, &x_col, 1e-12, &context);
                }
            }
        }
    }
}
