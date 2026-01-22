//! Common test utilities for CBLAS tests.
#![allow(dead_code)]
//!
//! Provides data generation, comparison functions, and helper utilities
//! for testing cblas-inject against OpenBLAS reference implementation.

use cblas_inject::{
    blasint, CblasColMajor, CblasRowMajor, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// CBLAS enum values for OpenBLAS FFI (u32)
pub const CBLAS_ROW_MAJOR: u32 = 101;
pub const CBLAS_COL_MAJOR: u32 = 102;
pub const CBLAS_NO_TRANS: u32 = 111;
pub const CBLAS_TRANS: u32 = 112;
pub const CBLAS_CONJ_TRANS: u32 = 113;
pub const CBLAS_CONJ_NO_TRANS: u32 = 114;

/// Convert CBLAS_ORDER to u32 for OpenBLAS FFI
pub fn order_to_u32(order: CBLAS_ORDER) -> u32 {
    match order {
        CblasRowMajor => CBLAS_ROW_MAJOR,
        CblasColMajor => CBLAS_COL_MAJOR,
    }
}

/// Convert CBLAS_TRANSPOSE to u32 for OpenBLAS FFI
pub fn transpose_to_u32(trans: CBLAS_TRANSPOSE) -> u32 {
    match trans {
        cblas_inject::CblasNoTrans => CBLAS_NO_TRANS,
        cblas_inject::CblasTrans => CBLAS_TRANS,
        cblas_inject::CblasConjTrans => CBLAS_CONJ_TRANS,
        cblas_inject::CblasConjNoTrans => CBLAS_CONJ_NO_TRANS,
    }
}

/// Generate test matrix data (real)
#[allow(dead_code)]
pub fn generate_matrix_f64(rows: usize, cols: usize, seed: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|i| ((i + seed) as f64 * 0.1).sin())
        .collect()
}

/// Generate test matrix data (complex)
pub fn generate_matrix_complex64(rows: usize, cols: usize, seed: usize) -> Vec<Complex64> {
    (0..rows * cols)
        .map(|i| {
            Complex64::new(
                ((i + seed) as f64 * 0.1).sin(),
                ((i + seed) as f64 * 0.2).cos(),
            )
        })
        .collect()
}

/// Generate test matrix data (complex32)
pub fn generate_matrix_complex32(rows: usize, cols: usize, seed: usize) -> Vec<Complex32> {
    (0..rows * cols)
        .map(|i| {
            Complex32::new(
                ((i + seed) as f32 * 0.1).sin(),
                ((i + seed) as f32 * 0.2).cos(),
            )
        })
        .collect()
}

/// Generate test vector data (real)
#[allow(dead_code)]
pub fn generate_vector_f64(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i + seed) as f64 * 0.15).cos())
        .collect()
}

/// Generate test vector data (complex64)
pub fn generate_vector_complex64(len: usize, seed: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            Complex64::new(
                ((i + seed) as f64 * 0.15).cos(),
                ((i + seed) as f64 * 0.25).sin(),
            )
        })
        .collect()
}

/// Generate test vector data (complex32)
pub fn generate_vector_complex32(len: usize, seed: usize) -> Vec<Complex32> {
    (0..len)
        .map(|i| {
            Complex32::new(
                ((i + seed) as f32 * 0.15).cos(),
                ((i + seed) as f32 * 0.25).sin(),
            )
        })
        .collect()
}

/// Calculate leading dimension for a matrix (GEMV-like operations)
///
/// OpenBLAS CBLAS semantics for GEMV:
/// - ColMajor: `lda` is the leading dimension of the *original* A(m×n) and must satisfy `lda >= max(1, m)`,
///            independent of `trans`.
/// - RowMajor: OpenBLAS swaps `m/n` internally and checks `lda >= max(1, swapped_m)` where `swapped_m = n`,
///            so `lda >= max(1, n)` independent of `trans`.
pub fn calc_lda_gemv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: usize,
    n: usize,
) -> blasint {
    match order {
        // RowMajor: lda >= n (after OpenBLAS swap)
        CblasRowMajor => n as blasint,
        // ColMajor: lda >= m (independent of trans)
        CblasColMajor => {
            let _ = trans; // keep signature symmetric; trans does not affect lda for CBLAS GEMV
            m as blasint
        }
    }
}

/// Calculate leading dimension for a band matrix (GBMV-like operations)
#[allow(dead_code)]
pub fn calc_lda_gbmv(kl: usize, ku: usize) -> blasint {
    (kl + ku + 1) as blasint
}

/// Calculate vector length for GEMV output
pub fn calc_output_len_gemv(trans: CBLAS_TRANSPOSE, m: usize, n: usize) -> usize {
    match trans {
        cblas_inject::CblasNoTrans | cblas_inject::CblasConjNoTrans => m,
        cblas_inject::CblasTrans | cblas_inject::CblasConjTrans => n,
    }
}

/// Calculate input vector length for GEMV/GBMV based on transpose.
pub fn x_len_gemv(trans: CBLAS_TRANSPOSE, m: usize, n: usize) -> usize {
    match trans {
        cblas_inject::CblasNoTrans => n,
        cblas_inject::CblasTrans | cblas_inject::CblasConjTrans => m,
        #[allow(unreachable_patterns)]
        _ => n,
    }
}

/// Calculate output vector length for GEMV/GBMV based on transpose.
pub fn y_len_gemv(trans: CBLAS_TRANSPOSE, m: usize, n: usize) -> usize {
    match trans {
        cblas_inject::CblasNoTrans => m,
        cblas_inject::CblasTrans | cblas_inject::CblasConjTrans => n,
        #[allow(unreachable_patterns)]
        _ => m,
    }
}

/// Calculate vector storage size given length and stride
/// For BLAS, storage size is: 1 + (n - 1) * abs(inc)
pub fn calc_vector_storage_size(len: usize, inc: blasint) -> usize {
    if len == 0 {
        0
    } else {
        1 + (len - 1) * (inc.abs() as usize)
    }
}

/// Compare two f64 slices with tolerance
#[allow(dead_code)]
pub fn assert_f64_eq(got: &[f64], expected: &[f64], tol: f64, context: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{}: length mismatch: got {}, expected {}",
        context,
        got.len(),
        expected.len()
    );
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
pub fn assert_complex64_eq(
    got: &[Complex64],
    expected: &[Complex64],
    tol: f64,
    context: &str,
) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{}: length mismatch: got {}, expected {}",
        context,
        got.len(),
        expected.len()
    );
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

/// Compare two Complex32 slices with tolerance
pub fn assert_complex32_eq(
    got: &[Complex32],
    expected: &[Complex32],
    tol: f32,
    context: &str,
) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{}: length mismatch: got {}, expected {}",
        context,
        got.len(),
        expected.len()
    );
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

/// Compare two Complex32 vectors with stride (for testing with inc != 1)
pub fn assert_complex32_eq_strided(
    got: &[Complex32],
    expected: &[Complex32],
    len: usize,
    inc: blasint,
    tol: f32,
    context: &str,
) {
    let inc_abs = inc.abs() as usize;
    for i in 0..len {
        let got_idx = i * inc_abs;
        let exp_idx = i * inc_abs;
        if got_idx >= got.len() || exp_idx >= expected.len() {
            panic!(
                "{}: index out of bounds: got_idx={}, exp_idx={}, got_len={}, exp_len={}",
                context,
                got_idx,
                exp_idx,
                got.len(),
                expected.len()
            );
        }
        let g = got[got_idx];
        let e = expected[exp_idx];
        let diff = (g - e).norm();
        let scale = e.norm().max(1.0);
        assert!(
            diff < tol * scale,
            "{}: mismatch at element {} (index {}): got {:?}, expected {:?}, diff {}",
            context,
            i,
            got_idx,
            g,
            e,
            diff
        );
    }
}

// =============================================================================
// Macro helpers
// =============================================================================

/// Define a one-time setup function for registering a BLAS symbol.
#[macro_export]
macro_rules! setup_once {
    ($name:ident, $register:path, $symbol:ident) => {
        fn $name() {
            #[cfg(not(feature = "openblas"))]
            {
                static INIT: std::sync::Once = std::sync::Once::new();
                INIT.call_once(|| unsafe { $register($symbol) });
            }
        }
    };
}

// ============================================================================
// Pure Rust Matrix/Vector abstractions for layout conversion tests
// ============================================================================

/// Matrix layout (RowMajor or ColMajor)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
}

/// Minimal matrix wrapper for pure Rust layout conversion tests.
///
/// - RowMajor index: `i*lda + j`, requires `lda >= cols`
/// - ColMajor index: `i + j*lda`, requires `lda >= rows`
pub struct Matrix<T> {
    layout: Layout,
    rows: usize,
    cols: usize,
    lda: usize,
    data: Vec<T>,
}

impl<T: Copy> Matrix<T> {
    /// Create a new row-major matrix with padding.
    ///
    /// # Panics
    /// Panics if `lda < cols`.
    pub fn new_row_major(rows: usize, cols: usize, lda: usize, fill: impl Fn(usize, usize) -> T) -> Self {
        assert!(lda >= cols, "RowMajor: lda ({}) must be >= cols ({})", lda, cols);
        let mut data = vec![fill(0, 0); rows * lda];
        for i in 0..rows {
            for j in 0..cols {
                data[i * lda + j] = fill(i, j);
            }
        }
        Self {
            layout: Layout::RowMajor,
            rows,
            cols,
            lda,
            data,
        }
    }

    /// Create a new column-major matrix with padding.
    ///
    /// # Panics
    /// Panics if `lda < rows`.
    pub fn new_col_major(rows: usize, cols: usize, lda: usize, fill: impl Fn(usize, usize) -> T) -> Self {
        assert!(lda >= rows, "ColMajor: lda ({}) must be >= rows ({})", lda, rows);
        let mut data = vec![fill(0, 0); lda * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i + j * lda] = fill(i, j);
            }
        }
        Self {
            layout: Layout::ColMajor,
            rows,
            cols,
            lda,
            data,
        }
    }

    /// Get element at position (i, j).
    ///
    /// # Panics
    /// Panics if `i >= rows` or `j >= cols`.
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.rows && j < self.cols, "Index out of bounds: ({}, {}) for matrix {}x{}", i, j, self.rows, self.cols);
        match self.layout {
            Layout::RowMajor => self.data[i * self.lda + j],
            Layout::ColMajor => self.data[i + j * self.lda],
        }
    }

    /// Convert matrix to a different layout with a new LDA.
    pub fn to_layout(&self, layout: Layout, lda: usize) -> Matrix<T> {
        match layout {
            Layout::RowMajor => Matrix::new_row_major(self.rows, self.cols, lda, |i, j| self.get(i, j)),
            Layout::ColMajor => Matrix::new_col_major(self.rows, self.cols, lda, |i, j| self.get(i, j)),
        }
    }

    /// Get raw pointer to matrix data.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer to matrix data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Get LDA as blasint.
    pub fn lda_blasint(&self) -> blasint {
        self.lda as blasint
    }

    /// Get number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }
}

/// Strided vector wrapper for pure Rust layout conversion tests.
pub struct StridedVec<T> {
    data: Vec<T>,
    len: usize,
    inc: blasint,
}

impl<T: Copy + Default> StridedVec<T> {
    /// Create a new strided vector.
    ///
    /// # Panics
    /// Panics if `inc == 0`.
    pub fn new(len: usize, inc: blasint, fill: impl Fn(usize) -> T) -> Self {
        assert!(inc != 0, "Stride cannot be zero");
        let storage_size = calc_vector_storage_size(len, inc);
        let mut data = vec![T::default(); storage_size];
        let inc_abs = inc.abs() as usize;
        for i in 0..len {
            let idx = if inc < 0 {
                (len - 1 - i) * inc_abs
            } else {
                i * inc_abs
            };
            data[idx] = fill(i);
        }
        Self { data, len, inc }
    }

    /// Get raw pointer to vector data.
    pub fn as_ptr(&self) -> *const T {
        if self.inc < 0 && self.len > 0 {
            // For negative stride, BLAS expects pointer to the last element
            let inc_abs = (-self.inc) as usize;
            unsafe { self.data.as_ptr().add((self.len - 1) * inc_abs) }
        } else {
            self.data.as_ptr()
        }
    }

    /// Get mutable raw pointer to vector data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if self.inc < 0 && self.len > 0 {
            // For negative stride, BLAS expects pointer to the last element
            let inc_abs = (-self.inc) as usize;
            unsafe { self.data.as_mut_ptr().add((self.len - 1) * inc_abs) }
        } else {
            self.data.as_mut_ptr()
        }
    }

    /// Get stride.
    pub fn inc(&self) -> blasint {
        self.inc
    }

    /// Get logical length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Extract logical elements as a Vec (for comparison).
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len);
        let inc_abs = self.inc.abs() as usize;
        for i in 0..self.len {
            let idx = if self.inc < 0 {
                (self.len - 1 - i) * inc_abs
            } else {
                i * inc_abs
            };
            result.push(self.data[idx]);
        }
        result
    }
}

// =============================================================================
// Band/Packed matrix storage helpers
// =============================================================================

/// Create band matrix in ColMajor band storage format.
///
/// Band storage format (ColMajor):
/// - LDA >= kl + ku + 1
/// - Element (i, j) of the original matrix is stored at A[ku + i - j, j]
/// - Only elements within the band are stored
pub fn create_band_matrix_col<T: Copy + Default>(
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    fill: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let lda = calc_lda_gbmv(kl, ku) as usize;
    let mut data = vec![T::default(); lda * n];
    for j in 0..n {
        for i in 0..m {
            // Check if (i, j) is within the band: j - ku <= i <= j + kl
            if i >= j.saturating_sub(ku) && i <= j + kl {
                let band_row = ku as isize + i as isize - j as isize;
                if band_row >= 0 && (band_row as usize) < lda {
                    data[band_row as usize + j * lda] = fill(i, j);
                }
            }
        }
    }
    data
}

/// Create band matrix in RowMajor band storage format.
///
/// For RowMajor, we swap m/n and kl/ku, then use the same band storage format.
pub fn create_band_matrix_row<T: Copy + Default>(
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    fill: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    // RowMajor: swap m/n and kl/ku
    create_band_matrix_col(n, m, ku, kl, |i, j| fill(j, i))
}

/// Create triangular band matrix in ColMajor band storage format.
///
/// Band storage format (ColMajor):
/// - LDA >= k + 1
/// - For Upper: element (i, j) is stored at A[k + i - j, j] where j-k <= i <= j
/// - For Lower: element (i, j) is stored at A[i - j, j] where j <= i <= j+k
pub fn create_triangular_band_matrix_col<T: Copy + Default>(
    n: usize,
    k: usize,
    uplo: CBLAS_UPLO,
    fill: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let lda = k + 1;
    let mut data = vec![T::default(); lda * n];
    for j in 0..n {
        match uplo {
            cblas_inject::CblasUpper => {
                let i_start = j.saturating_sub(k);
                let i_end = j;
                for i in i_start..=i_end {
                    let band_row = k + i - j;
                    data[band_row + j * lda] = fill(i, j);
                }
            }
            cblas_inject::CblasLower => {
                let i_start = j;
                let i_end = (j + k).min(n - 1);
                for i in i_start..=i_end {
                    let band_row = i - j;
                    data[band_row + j * lda] = fill(i, j);
                }
            }
        }
    }
    data
}

/// Create triangular band matrix for RowMajor CBLAS call.
pub fn create_triangular_band_matrix_row<T: Copy + Default>(
    n: usize,
    k: usize,
    uplo: CBLAS_UPLO,
    fill: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let swapped_uplo = match uplo {
        cblas_inject::CblasUpper => cblas_inject::CblasLower,
        cblas_inject::CblasLower => cblas_inject::CblasUpper,
    };
    create_triangular_band_matrix_col(n, k, swapped_uplo, |i, j| fill(j, i))
}

/// Create triangular packed matrix in ColMajor packed format.
pub fn create_triangular_packed_matrix_col<T: Copy + Default>(
    n: usize,
    uplo: CBLAS_UPLO,
    fill: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let size = n * (n + 1) / 2;
    let mut data = vec![T::default(); size];
    let mut idx = 0;
    match uplo {
        cblas_inject::CblasUpper => {
            for j in 0..n {
                for i in 0..=j {
                    data[idx] = fill(i, j);
                    idx += 1;
                }
            }
        }
        cblas_inject::CblasLower => {
            for j in 0..n {
                for i in j..n {
                    data[idx] = fill(i, j);
                    idx += 1;
                }
            }
        }
    }
    data
}

/// Create triangular packed matrix for RowMajor CBLAS call.
pub fn create_triangular_packed_matrix_row<T: Copy + Default>(
    n: usize,
    uplo: CBLAS_UPLO,
    fill: impl Fn(usize, usize) -> T,
) -> Vec<T> {
    let swapped_uplo = match uplo {
        cblas_inject::CblasUpper => cblas_inject::CblasLower,
        cblas_inject::CblasLower => cblas_inject::CblasUpper,
    };
    create_triangular_packed_matrix_col(n, swapped_uplo, |i, j| fill(j, i))
}
