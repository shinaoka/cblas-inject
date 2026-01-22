//! Integration tests for GEMM functions.
//!
//! These tests use cblas-sys (linked BLAS) as the backend to verify
//! that cblas-trampoline produces correct results.

extern crate blas_src;

use cblas_trampoline::{
    blasint, cblas_dgemm, cblas_zgemm, register_dgemm, register_zgemm, CblasColMajor, CblasNoTrans,
    CblasRowMajor, CblasTrans,
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

fn setup_dgemm() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        register_dgemm(dgemm_);
    });
}

fn setup_zgemm() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        register_zgemm(zgemm_);
    });
}

/// Naive matrix multiply for reference (row-major)
fn naive_matmul(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

#[test]
fn test_dgemm_row_major_no_trans() {
    setup_dgemm();

    let m = 2usize;
    let n = 3usize;
    let k = 4usize;

    // A: m x k matrix (row-major)
    let a: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
    // B: k x n matrix (row-major)
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.5).collect();

    // Expected result using naive multiply
    let mut expected = vec![0.0; m * n];
    naive_matmul(&a, &b, &mut expected, m, n, k, 1.0, 0.0);

    // Result using cblas_trampoline
    let mut c = vec![0.0; m * n];
    unsafe {
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m as blasint,
            n as blasint,
            k as blasint,
            1.0,
            a.as_ptr(),
            k as blasint,
            b.as_ptr(),
            n as blasint,
            0.0,
            c.as_mut_ptr(),
            n as blasint,
        );
    }

    for (i, (got, exp)) in c.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_dgemm_col_major_no_trans() {
    setup_dgemm();

    let m = 2usize;
    let n = 3usize;
    let k = 4usize;

    // A: m x k matrix (column-major)
    // Column-major storage: A[i,j] at index i + j*m
    let mut a = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            a[i + j * m] = (i * k + j) as f64;
        }
    }

    // B: k x n matrix (column-major)
    let mut b = vec![0.0; k * n];
    for i in 0..k {
        for j in 0..n {
            b[i + j * k] = (i * n + j) as f64 * 0.5;
        }
    }

    // Result using cblas_trampoline
    let mut c = vec![0.0; m * n];
    unsafe {
        cblas_dgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            m as blasint,
            n as blasint,
            k as blasint,
            1.0,
            a.as_ptr(),
            m as blasint, // lda = m for col-major NoTrans
            b.as_ptr(),
            k as blasint, // ldb = k for col-major NoTrans
            0.0,
            c.as_mut_ptr(),
            m as blasint, // ldc = m for col-major
        );
    }

    // Convert to row-major for comparison
    let mut c_row_major = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            c_row_major[i * n + j] = c[i + j * m];
        }
    }

    // Compute expected with naive (row-major)
    let a_row: Vec<f64> = (0..m * k).map(|i| (i / k * k + i % k) as f64).collect();
    let b_row: Vec<f64> = (0..k * n)
        .map(|i| (i / n * n + i % n) as f64 * 0.5)
        .collect();
    let mut expected = vec![0.0; m * n];
    naive_matmul(&a_row, &b_row, &mut expected, m, n, k, 1.0, 0.0);

    for (i, (got, exp)) in c_row_major.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_dgemm_row_major_trans_a() {
    setup_dgemm();

    let m = 2usize;
    let n = 3usize;
    let k = 4usize;

    // A: k x m matrix (row-major), will be transposed to m x k
    let a: Vec<f64> = (0..k * m).map(|i| i as f64).collect();
    // B: k x n matrix (row-major)
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.5).collect();

    // Compute A^T * B naively
    // A^T[i,j] = A[j,i] (row-major of A means A[j,i] = A[j*m + i])
    let mut expected = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                let a_val = a[l * m + i]; // A^T[i,l] = A[l,i]
                let b_val = b[l * n + j];
                sum += a_val * b_val;
            }
            expected[i * n + j] = sum;
        }
    }

    let mut c = vec![0.0; m * n];
    unsafe {
        cblas_dgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            m as blasint,
            n as blasint,
            k as blasint,
            1.0,
            a.as_ptr(),
            m as blasint, // lda = m for row-major Trans
            b.as_ptr(),
            n as blasint,
            0.0,
            c.as_mut_ptr(),
            n as blasint,
        );
    }

    for (i, (got, exp)) in c.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_dgemm_alpha_beta() {
    setup_dgemm();

    let m = 2usize;
    let n = 2usize;
    let k = 2usize;

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![1.0, 1.0, 1.0, 1.0];

    let alpha = 2.0;
    let beta = 3.0;

    // expected = alpha * A * B + beta * C
    let mut expected = vec![1.0, 1.0, 1.0, 1.0];
    naive_matmul(&a, &b, &mut expected, m, n, k, alpha, beta);

    unsafe {
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m as blasint,
            n as blasint,
            k as blasint,
            alpha,
            a.as_ptr(),
            k as blasint,
            b.as_ptr(),
            n as blasint,
            beta,
            c.as_mut_ptr(),
            n as blasint,
        );
    }

    for (i, (got, exp)) in c.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "Mismatch at index {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_zgemm_row_major() {
    setup_zgemm();

    let m = 2usize;
    let n = 2usize;
    let k = 2usize;

    let a: Vec<Complex64> = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, -1.0),
    ];
    let b: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 0.0),
    ];

    // Naive complex matmul
    let mut expected = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            expected[i * n + j] = sum;
        }
    }

    let mut c = vec![Complex64::new(0.0, 0.0); m * n];
    let alpha = Complex64::new(1.0, 0.0);
    let beta = Complex64::new(0.0, 0.0);

    unsafe {
        cblas_zgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m as blasint,
            n as blasint,
            k as blasint,
            alpha,
            a.as_ptr(),
            k as blasint,
            b.as_ptr(),
            n as blasint,
            beta,
            c.as_mut_ptr(),
            n as blasint,
        );
    }

    for (i, (got, exp)) in c.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).norm() < 1e-10,
            "Mismatch at index {}: got {:?}, expected {:?}",
            i,
            got,
            exp
        );
    }
}
