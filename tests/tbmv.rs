//! Pure-Rust layout tests for TBMV (complex types).
//!
//! Policy:
//! - Do NOT modify existing OpenBLAS-derived tests.
//! - Add additional tests that validate row-major conversion logic by comparing
//!   `order=RowMajor` vs `order=ColMajor` results for the *same logical triangular band matrix*.
//!
//! Key insight for band storage RowMajor conversion:
//! - `RowMajor + Upper + NoTrans` internally becomes `ColMajor + Lower + Trans`
//! - For the same logical matrix M, RowMajor storage must contain M^T with swapped uplo

extern crate blas_src;

use cblas_inject::{
    blasint, register_ctbmv, register_ztbmv, CblasColMajor, CblasConjTrans, CblasLower,
    CblasNoTrans, CblasNonUnit, CblasRowMajor, CblasTrans, CblasUpper,
};
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

#[macro_use]
mod common;
use common::{
    assert_complex32_eq, assert_complex64_eq, create_triangular_band_matrix_col,
    create_triangular_band_matrix_row,
};

// Fortran BLAS function declarations (provided by linked OpenBLAS)
extern "C" {
    fn ctbmv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blasint,
        k: *const blasint,
        a: *const Complex32,
        lda: *const blasint,
        x: *mut Complex32,
        incx: *const blasint,
    );

    fn ztbmv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blasint,
        k: *const blasint,
        a: *const Complex64,
        lda: *const blasint,
        x: *mut Complex64,
        incx: *const blasint,
    );
}

setup_once!(setup_ctbmv, register_ctbmv, ctbmv_);
setup_once!(setup_ztbmv, register_ztbmv, ztbmv_);

#[test]
fn ctbmv_row_vs_col_agree() {
    setup_ctbmv();

    let cases = [(3usize, 1usize), (4, 1), (5, 2)]; // (n, k)
    let uplos = [CblasUpper, CblasLower];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let diags = [CblasNonUnit, cblas_inject::CblasUnit];

    for &(n, k) in &cases {
        for &uplo in &uplos {
            for &trans in &transposes {
                for &diag in &diags {
                    // Create the same logical matrix M for both layouts
                    let fill = |i: usize, j: usize| -> Complex32 {
                        let is_triangular = match uplo {
                            CblasUpper => i <= j,
                            CblasLower => i >= j,
                        };
                        if is_triangular && (diag != cblas_inject::CblasUnit || i != j) {
                            Complex32::new(
                                ((i + 3 * j) as f32 * 0.1).sin(),
                                ((7 * i + j) as f32 * 0.2).cos(),
                            )
                        } else if i == j {
                            Complex32::new(1.0, 0.0) // diagonal for unit case
                        } else {
                            Complex32::new(0.0, 0.0)
                        }
                    };

                    let a_col = create_triangular_band_matrix_col(n, k, uplo, fill);
                    let a_row = create_triangular_band_matrix_row(n, k, uplo, fill);

                    let x0: Vec<Complex32> = (0..n)
                        .map(|k| {
                            Complex32::new(
                                ((k + 11) as f32 * 0.15).cos(),
                                ((k + 5) as f32 * 0.25).sin(),
                            )
                        })
                        .collect();

                    let mut x_row = x0.clone();
                    let mut x_col = x0.clone();

                    let lda = (k + 1) as blasint;

                    unsafe {
                        cblas_inject::cblas_ctbmv(
                            CblasRowMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            k as blasint,
                            a_row.as_ptr(),
                            lda,
                            x_row.as_mut_ptr(),
                            1,
                        );
                        cblas_inject::cblas_ctbmv(
                            CblasColMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            k as blasint,
                            a_col.as_ptr(),
                            lda,
                            x_col.as_mut_ptr(),
                            1,
                        );
                    }

                    let context = format!(
                        "ctbmv row-vs-col: n={}, k={}, uplo={:?}, trans={:?}, diag={:?}",
                        n, k, uplo, trans, diag
                    );
                    assert_complex32_eq(&x_row, &x_col, 1e-5, &context);
                }
            }
        }
    }
}

#[test]
fn ztbmv_row_vs_col_agree() {
    setup_ztbmv();

    let cases = [(3usize, 1usize), (4, 1), (5, 2)];
    let uplos = [CblasUpper, CblasLower];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let diags = [CblasNonUnit, cblas_inject::CblasUnit];

    for &(n, k) in &cases {
        for &uplo in &uplos {
            for &trans in &transposes {
                for &diag in &diags {
                    let fill = |i: usize, j: usize| -> Complex64 {
                        let is_triangular = match uplo {
                            CblasUpper => i <= j,
                            CblasLower => i >= j,
                        };
                        if is_triangular && (diag != cblas_inject::CblasUnit || i != j) {
                            Complex64::new(
                                ((i + 3 * j) as f64 * 0.1).sin(),
                                ((7 * i + j) as f64 * 0.2).cos(),
                            )
                        } else if i == j {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    };

                    let a_col = create_triangular_band_matrix_col(n, k, uplo, fill);
                    let a_row = create_triangular_band_matrix_row(n, k, uplo, fill);

                    let x0: Vec<Complex64> = (0..n)
                        .map(|k| {
                            Complex64::new(
                                ((k + 11) as f64 * 0.15).cos(),
                                ((k + 5) as f64 * 0.25).sin(),
                            )
                        })
                        .collect();

                    let mut x_row = x0.clone();
                    let mut x_col = x0.clone();

                    let lda = (k + 1) as blasint;

                    unsafe {
                        cblas_inject::cblas_ztbmv(
                            CblasRowMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            k as blasint,
                            a_row.as_ptr(),
                            lda,
                            x_row.as_mut_ptr(),
                            1,
                        );
                        cblas_inject::cblas_ztbmv(
                            CblasColMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            k as blasint,
                            a_col.as_ptr(),
                            lda,
                            x_col.as_mut_ptr(),
                            1,
                        );
                    }

                    let context = format!(
                        "ztbmv row-vs-col: n={}, k={}, uplo={:?}, trans={:?}, diag={:?}",
                        n, k, uplo, trans, diag
                    );
                    assert_complex64_eq(&x_row, &x_col, 1e-12, &context);
                }
            }
        }
    }
}
