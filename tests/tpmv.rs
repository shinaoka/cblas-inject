//! Pure-Rust layout tests for TPMV and TPSV (complex types).
//!
//! Policy:
//! - Do NOT modify existing OpenBLAS-derived tests.
//! - Add additional tests that validate row-major conversion logic by comparing
//!   `order=RowMajor` vs `order=ColMajor` results for the *same logical triangular packed matrix*.
//!
//! Key insight for packed storage RowMajor conversion:
//! - `RowMajor + Upper + NoTrans` internally becomes `ColMajor + Lower + Trans`
//! - For the same logical matrix M, RowMajor storage must contain M^T with swapped uplo

extern crate blas_src;

use cblas_inject::{
    blasint, register_ctpmv, register_ctpsv, register_ztpmv, register_ztpsv, CblasColMajor,
    CblasConjTrans, CblasLower, CblasNoTrans, CblasNonUnit, CblasRowMajor, CblasTrans, CblasUpper,
};
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

#[macro_use]
mod common;
use common::{
    assert_complex32_eq, assert_complex64_eq, create_triangular_packed_matrix_col,
    create_triangular_packed_matrix_row,
};

// Fortran BLAS function declarations (provided by linked OpenBLAS)
extern "C" {
    fn ctpmv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blasint,
        ap: *const Complex32,
        x: *mut Complex32,
        incx: *const blasint,
    );

    fn ztpmv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blasint,
        ap: *const Complex64,
        x: *mut Complex64,
        incx: *const blasint,
    );

    fn ctpsv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blasint,
        ap: *const Complex32,
        x: *mut Complex32,
        incx: *const blasint,
    );

    fn ztpsv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blasint,
        ap: *const Complex64,
        x: *mut Complex64,
        incx: *const blasint,
    );
}

setup_once!(setup_ctpmv, register_ctpmv, ctpmv_);
setup_once!(setup_ztpmv, register_ztpmv, ztpmv_);
setup_once!(setup_ctpsv, register_ctpsv, ctpsv_);
setup_once!(setup_ztpsv, register_ztpsv, ztpsv_);

#[test]
fn ctpmv_row_vs_col_agree() {
    setup_ctpmv();

    let cases = [1usize, 2, 3, 4, 5];
    let uplos = [CblasUpper, CblasLower];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let diags = [CblasNonUnit, cblas_inject::CblasUnit];

    for &n in &cases {
        for &uplo in &uplos {
            for &trans in &transposes {
                for &diag in &diags {
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
                            Complex32::new(1.0, 0.0)
                        } else {
                            Complex32::new(0.0, 0.0)
                        }
                    };

                    let ap_col = create_triangular_packed_matrix_col(n, uplo, fill);
                    let ap_row = create_triangular_packed_matrix_row(n, uplo, fill);

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

                    unsafe {
                        cblas_inject::cblas_ctpmv(
                            CblasRowMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            ap_row.as_ptr(),
                            x_row.as_mut_ptr(),
                            1,
                        );
                        cblas_inject::cblas_ctpmv(
                            CblasColMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            ap_col.as_ptr(),
                            x_col.as_mut_ptr(),
                            1,
                        );
                    }

                    let context = format!(
                        "ctpmv row-vs-col: n={}, uplo={:?}, trans={:?}, diag={:?}",
                        n, uplo, trans, diag
                    );
                    assert_complex32_eq(&x_row, &x_col, 1e-5, &context);
                }
            }
        }
    }
}

#[test]
fn ztpmv_row_vs_col_agree() {
    setup_ztpmv();

    let cases = [1usize, 2, 3, 4, 5];
    let uplos = [CblasUpper, CblasLower];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let diags = [CblasNonUnit, cblas_inject::CblasUnit];

    for &n in &cases {
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

                    let ap_col = create_triangular_packed_matrix_col(n, uplo, fill);
                    let ap_row = create_triangular_packed_matrix_row(n, uplo, fill);

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

                    unsafe {
                        cblas_inject::cblas_ztpmv(
                            CblasRowMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            ap_row.as_ptr(),
                            x_row.as_mut_ptr(),
                            1,
                        );
                        cblas_inject::cblas_ztpmv(
                            CblasColMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            ap_col.as_ptr(),
                            x_col.as_mut_ptr(),
                            1,
                        );
                    }

                    let context = format!(
                        "ztpmv row-vs-col: n={}, uplo={:?}, trans={:?}, diag={:?}",
                        n, uplo, trans, diag
                    );
                    assert_complex64_eq(&x_row, &x_col, 1e-12, &context);
                }
            }
        }
    }
}

#[test]
fn ctpsv_row_vs_col_agree() {
    setup_ctpsv();

    let cases = [1usize, 2, 3, 4, 5];
    let uplos = [CblasUpper, CblasLower];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let diags = [CblasNonUnit, cblas_inject::CblasUnit];

    for &n in &cases {
        for &uplo in &uplos {
            for &trans in &transposes {
                for &diag in &diags {
                    let fill = |i: usize, j: usize| -> Complex32 {
                        let is_triangular = match uplo {
                            CblasUpper => i <= j,
                            CblasLower => i >= j,
                        };
                        if is_triangular {
                            if diag == cblas_inject::CblasUnit && i == j {
                                Complex32::new(1.0, 0.0)
                            } else {
                                let val = if i == j {
                                    2.0 + ((i + j) as f32 * 0.1)
                                } else {
                                    ((i + 3 * j) as f32 * 0.1).sin()
                                };
                                Complex32::new(val, ((7 * i + j) as f32 * 0.2).cos())
                            }
                        } else {
                            Complex32::new(0.0, 0.0)
                        }
                    };

                    let ap_col = create_triangular_packed_matrix_col(n, uplo, fill);
                    let ap_row = create_triangular_packed_matrix_row(n, uplo, fill);

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

                    unsafe {
                        cblas_inject::cblas_ctpsv(
                            CblasRowMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            ap_row.as_ptr(),
                            x_row.as_mut_ptr(),
                            1,
                        );
                        cblas_inject::cblas_ctpsv(
                            CblasColMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            ap_col.as_ptr(),
                            x_col.as_mut_ptr(),
                            1,
                        );
                    }

                    let context = format!(
                        "ctpsv row-vs-col: n={}, uplo={:?}, trans={:?}, diag={:?}",
                        n, uplo, trans, diag
                    );
                    assert_complex32_eq(&x_row, &x_col, 1e-5, &context);
                }
            }
        }
    }
}

#[test]
fn ztpsv_row_vs_col_agree() {
    setup_ztpsv();

    let cases = [1usize, 2, 3, 4, 5];
    let uplos = [CblasUpper, CblasLower];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let diags = [CblasNonUnit, cblas_inject::CblasUnit];

    for &n in &cases {
        for &uplo in &uplos {
            for &trans in &transposes {
                for &diag in &diags {
                    let fill = |i: usize, j: usize| -> Complex64 {
                        let is_triangular = match uplo {
                            CblasUpper => i <= j,
                            CblasLower => i >= j,
                        };
                        if is_triangular {
                            if diag == cblas_inject::CblasUnit && i == j {
                                Complex64::new(1.0, 0.0)
                            } else {
                                let val = if i == j {
                                    2.0 + ((i + j) as f64 * 0.1)
                                } else {
                                    ((i + 3 * j) as f64 * 0.1).sin()
                                };
                                Complex64::new(val, ((7 * i + j) as f64 * 0.2).cos())
                            }
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    };

                    let ap_col = create_triangular_packed_matrix_col(n, uplo, fill);
                    let ap_row = create_triangular_packed_matrix_row(n, uplo, fill);

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

                    unsafe {
                        cblas_inject::cblas_ztpsv(
                            CblasRowMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            ap_row.as_ptr(),
                            x_row.as_mut_ptr(),
                            1,
                        );
                        cblas_inject::cblas_ztpsv(
                            CblasColMajor,
                            uplo,
                            trans,
                            diag,
                            n as blasint,
                            ap_col.as_ptr(),
                            x_col.as_mut_ptr(),
                            1,
                        );
                    }

                    let context = format!(
                        "ztpsv row-vs-col: n={}, uplo={:?}, trans={:?}, diag={:?}",
                        n, uplo, trans, diag
                    );
                    assert_complex64_eq(&x_row, &x_col, 1e-12, &context);
                }
            }
        }
    }
}
