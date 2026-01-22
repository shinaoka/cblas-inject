//! # cblas-trampoline
//!
//! CBLAS/LAPACKE compatible interface backed by Fortran BLAS/LAPACK function pointers.
//!
//! This crate allows you to use CBLAS-style functions (with row-major support) while
//! the actual computation is performed by Fortran BLAS/LAPACK functions provided at runtime.
//! This is useful for integrating with Python (scipy) or Julia (libblastrampoline) where
//! Fortran BLAS pointers are available.
//!
//! ## Usage
//!
//! ```ignore
//! use cblas_trampoline::{register_dgemm, cblas_dgemm, CblasRowMajor, CblasNoTrans};
//!
//! // Register Fortran dgemm pointer (e.g., from scipy or Julia)
//! unsafe {
//!     register_dgemm(dgemm_ptr);
//! }
//!
//! // Use CBLAS-style interface with row-major support
//! unsafe {
//!     cblas_dgemm(
//!         CblasRowMajor, CblasNoTrans, CblasNoTrans,
//!         m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
//!     );
//! }
//! ```
//!
//! ## Row-Major Handling
//!
//! For BLAS operations (GEMM, etc.), row-major data is handled via argument swapping
//! without memory copy, following the same approach as OpenBLAS:
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c#L489-L537>
//!
//! For LAPACK operations (SVD, etc.), explicit transpose copies are required,
//! following LAPACKE's approach:
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/lapack-netlib/LAPACKE/src/lapacke_dgesvd_work.c#L49-L127>

mod backend;
mod types;

pub mod blas3;

pub use backend::*;
pub use types::*;

// Re-export commonly used functions at crate root
pub use blas3::gemm::{cblas_cgemm, cblas_dgemm, cblas_sgemm, cblas_zgemm};
