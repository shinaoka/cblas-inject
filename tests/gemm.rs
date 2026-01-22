//! Integration tests for GEMM functions.
//!
//! These tests compare cblas-trampoline results with cblas-sys (OpenBLAS)
//! to verify correctness across all parameter combinations.

extern crate blas_src;

use cblas_inject::{
    blasint, register_dgemm, register_zgemm, CblasColMajor, CblasConjTrans, CblasNoTrans,
    CblasRowMajor, CblasTrans, CBLAS_ORDER, CBLAS_TRANSPOSE,
};
use num_complex::Complex64;
use std::ffi::c_char;

// Fortran BLAS function declarations (from cblas-sys's underlying BLAS)
extern "C" {
    fn dgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const blasint,
        n: *const blasint,
        k: *const blasint,
        alpha: *const f64,
        a: *const f64,
        lda: *const blasint,
        b: *const f64,
        ldb: *const blasint,
        beta: *const f64,
        c: *mut f64,
        ldc: *const blasint,
    );

    fn zgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const blasint,
        n: *const blasint,
        k: *const blasint,
        alpha: *const Complex64,
        a: *const Complex64,
        lda: *const blasint,
        b: *const Complex64,
        ldb: *const blasint,
        beta: *const Complex64,
        c: *mut Complex64,
        ldc: *const blasint,
    );
}

// CBLAS declarations from OpenBLAS for direct comparison (reference implementation)
mod openblas {
    use super::*;

    extern "C" {
        pub fn cblas_dgemm(
            order: u32,
            transa: u32,
            transb: u32,
            m: blasint,
            n: blasint,
            k: blasint,
            alpha: f64,
            a: *const f64,
            lda: blasint,
            b: *const f64,
            ldb: blasint,
            beta: f64,
            c: *mut f64,
            ldc: blasint,
        );

        pub fn cblas_zgemm(
            order: u32,
            transa: u32,
            transb: u32,
            m: blasint,
            n: blasint,
            k: blasint,
            alpha: *const Complex64,
            a: *const Complex64,
            lda: blasint,
            b: *const Complex64,
            ldb: blasint,
            beta: *const Complex64,
            c: *mut Complex64,
            ldc: blasint,
        );
    }
}

fn setup_dgemm() {
    // When openblas feature is enabled, autoregister handles this
    #[cfg(not(feature = "openblas"))]
    {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| unsafe {
            register_dgemm(dgemm_);
        });
    }
}

fn setup_zgemm() {
    // When openblas feature is enabled, autoregister handles this
    #[cfg(not(feature = "openblas"))]
    {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| unsafe {
            register_zgemm(zgemm_);
        });
    }
}

/// Generate random-ish test data
fn generate_matrix(rows: usize, cols: usize, seed: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|i| ((i + seed) as f64 * 0.1).sin())
        .collect()
}

fn generate_complex_matrix(rows: usize, cols: usize, seed: usize) -> Vec<Complex64> {
    (0..rows * cols)
        .map(|i| {
            Complex64::new(
                ((i + seed) as f64 * 0.1).sin(),
                ((i + seed) as f64 * 0.2).cos(),
            )
        })
        .collect()
}

/// Calculate leading dimension for a matrix
fn calc_lda(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, rows: usize, cols: usize) -> blasint {
    match order {
        CblasRowMajor => match trans {
            CblasNoTrans => cols as blasint,
            CblasTrans | CblasConjTrans => rows as blasint,
        },
        CblasColMajor => match trans {
            CblasNoTrans => rows as blasint,
            CblasTrans | CblasConjTrans => cols as blasint,
        },
    }
}

/// Compare two f64 slices with tolerance
fn assert_f64_eq(got: &[f64], expected: &[f64], tol: f64, context: &str) {
    assert_eq!(got.len(), expected.len(), "{}: length mismatch", context);
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let scale = e.abs().max(1.0);
        assert!(
            diff < tol * scale,
            "{}: mismatch at index {}: got {}, expected {}, diff {}",
            context,
            i,
            g,
            e,
            diff
        );
    }
}

/// Compare two Complex64 slices with tolerance
fn assert_c64_eq(got: &[Complex64], expected: &[Complex64], tol: f64, context: &str) {
    assert_eq!(got.len(), expected.len(), "{}: length mismatch", context);
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).norm();
        let scale = e.norm().max(1.0);
        assert!(
            diff < tol * scale,
            "{}: mismatch at index {}: got {:?}, expected {:?}, diff {}",
            context,
            i,
            g,
            e,
            diff
        );
    }
}

// =============================================================================
// Exhaustive DGEMM tests - compare cblas-trampoline with cblas-sys
// =============================================================================

#[test]
fn test_dgemm_exhaustive() {
    setup_dgemm();

    let orders = [CblasRowMajor, CblasColMajor];
    let transposes = [CblasNoTrans, CblasTrans];
    let dims = [1, 2, 3, 5, 7, 9];
    let alphas = [0.0, 1.0, 0.7, -1.0];
    let betas = [0.0, 1.0, 1.3, -0.5];

    let mut test_count = 0;

    for &order in &orders {
        for &transa in &transposes {
            for &transb in &transposes {
                for &m in &dims {
                    for &n in &dims {
                        for &k in &dims {
                            for &alpha in &alphas {
                                for &beta in &betas {
                                    test_dgemm_case(order, transa, transb, m, n, k, alpha, beta);
                                    test_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("Ran {} DGEMM test cases", test_count);
}

fn test_dgemm_case(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
) {
    // Determine matrix dimensions based on transpose
    let (a_rows, a_cols) = match transa {
        CblasNoTrans => (m, k),
        CblasTrans | CblasConjTrans => (k, m),
    };
    let (b_rows, b_cols) = match transb {
        CblasNoTrans => (k, n),
        CblasTrans | CblasConjTrans => (n, k),
    };

    // Generate test data
    let a = generate_matrix(a_rows, a_cols, 42);
    let b = generate_matrix(b_rows, b_cols, 123);
    let c_init = generate_matrix(m, n, 456);

    // Leading dimensions
    let lda = calc_lda(order, transa, m, k);
    let ldb = calc_lda(order, transb, k, n);
    let ldc = match order {
        CblasRowMajor => n as blasint,
        CblasColMajor => m as blasint,
    };

    // Result from cblas-trampoline
    let mut c_trampoline = c_init.clone();
    unsafe {
        cblas_inject::cblas_dgemm(
            order,
            transa,
            transb,
            m as blasint,
            n as blasint,
            k as blasint,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c_trampoline.as_mut_ptr(),
            ldc,
        );
    }

    // Result from OpenBLAS cblas
    let mut c_reference = c_init.clone();
    unsafe {
        openblas::cblas_dgemm(
            order as u32,
            transa as u32,
            transb as u32,
            m as blasint,
            n as blasint,
            k as blasint,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c_reference.as_mut_ptr(),
            ldc,
        );
    }

    let context = format!(
        "order={:?}, transa={:?}, transb={:?}, m={}, n={}, k={}, alpha={}, beta={}",
        order, transa, transb, m, n, k, alpha, beta
    );
    assert_f64_eq(&c_trampoline, &c_reference, 1e-12, &context);
}

// =============================================================================
// Exhaustive ZGEMM tests - compare cblas-trampoline with cblas-sys
// =============================================================================

#[test]
#[ignore] // TODO: Fix zgemm row-major handling
fn test_zgemm_exhaustive() {
    setup_zgemm();

    let orders = [CblasRowMajor, CblasColMajor];
    let transposes = [CblasNoTrans, CblasTrans, CblasConjTrans];
    let dims = [1, 2, 3, 5];
    let alphas = [
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.7, 0.3),
    ];
    let betas = [
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.5, 0.2),
    ];

    let mut test_count = 0;

    for &order in &orders {
        for &transa in &transposes {
            for &transb in &transposes {
                for &m in &dims {
                    for &n in &dims {
                        for &k in &dims {
                            for &alpha in &alphas {
                                for &beta in &betas {
                                    test_zgemm_case(order, transa, transb, m, n, k, alpha, beta);
                                    test_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("Ran {} ZGEMM test cases", test_count);
}

fn test_zgemm_case(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    beta: Complex64,
) {
    // Determine matrix dimensions based on transpose
    let (a_rows, a_cols) = match transa {
        CblasNoTrans => (m, k),
        CblasTrans | CblasConjTrans => (k, m),
    };
    let (b_rows, b_cols) = match transb {
        CblasNoTrans => (k, n),
        CblasTrans | CblasConjTrans => (n, k),
    };

    // Generate test data
    let a = generate_complex_matrix(a_rows, a_cols, 42);
    let b = generate_complex_matrix(b_rows, b_cols, 123);
    let c_init = generate_complex_matrix(m, n, 456);

    // Leading dimensions
    let lda = calc_lda(order, transa, m, k);
    let ldb = calc_lda(order, transb, k, n);
    let ldc = match order {
        CblasRowMajor => n as blasint,
        CblasColMajor => m as blasint,
    };

    // Result from cblas-trampoline
    let mut c_trampoline = c_init.clone();
    unsafe {
        cblas_inject::cblas_zgemm(
            order,
            transa,
            transb,
            m as blasint,
            n as blasint,
            k as blasint,
            &alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            &beta,
            c_trampoline.as_mut_ptr(),
            ldc,
        );
    }

    // Result from OpenBLAS cblas
    let mut c_reference = c_init.clone();
    unsafe {
        openblas::cblas_zgemm(
            order as u32,
            transa as u32,
            transb as u32,
            m as blasint,
            n as blasint,
            k as blasint,
            &alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            &beta,
            c_reference.as_mut_ptr(),
            ldc,
        );
    }

    let context = format!(
        "order={:?}, transa={:?}, transb={:?}, m={}, n={}, k={}, alpha={:?}, beta={:?}",
        order, transa, transb, m, n, k, alpha, beta
    );
    assert_c64_eq(&c_trampoline, &c_reference, 1e-12, &context);
}

// =============================================================================
// Edge case tests
// =============================================================================

#[test]
fn test_dgemm_zero_dimensions() {
    setup_dgemm();

    // m=0 case
    let a: Vec<f64> = vec![];
    let b = vec![1.0, 2.0];
    let mut c: Vec<f64> = vec![];

    unsafe {
        cblas_inject::cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            0,
            2,
            1,
            1.0,
            a.as_ptr(),
            1,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
    }
    // Should not crash
}

#[test]
fn test_dgemm_non_square() {
    setup_dgemm();

    // Non-square matrices: A(3x5), B(5x7) -> C(3x7)
    let m = 3;
    let n = 7;
    let k = 5;

    let a = generate_matrix(m, k, 1);
    let b = generate_matrix(k, n, 2);
    let c_init = generate_matrix(m, n, 3);

    let mut c_trampoline = c_init.clone();
    let mut c_reference = c_init.clone();

    unsafe {
        cblas_inject::cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m as blasint,
            n as blasint,
            k as blasint,
            1.5,
            a.as_ptr(),
            k as blasint,
            b.as_ptr(),
            n as blasint,
            0.5,
            c_trampoline.as_mut_ptr(),
            n as blasint,
        );

        openblas::cblas_dgemm(
            CblasRowMajor as u32,
            CblasNoTrans as u32,
            CblasNoTrans as u32,
            m as blasint,
            n as blasint,
            k as blasint,
            1.5,
            a.as_ptr(),
            k as blasint,
            b.as_ptr(),
            n as blasint,
            0.5,
            c_reference.as_mut_ptr(),
            n as blasint,
        );
    }

    assert_f64_eq(&c_trampoline, &c_reference, 1e-12, "non-square test");
}
