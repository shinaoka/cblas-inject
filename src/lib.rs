//! # cblas-inject
//!
//! CBLAS compatible interface backed by Fortran BLAS function pointers.
//!
//! This crate allows you to use CBLAS-style functions (with row-major support) while
//! the actual computation is performed by Fortran BLAS functions provided at runtime.
//! This is useful for integrating with Python (scipy) or Julia (libblastrampoline) where
//! Fortran BLAS pointers are available.
//!
//! ## Usage
//!
//! ```ignore
//! use cblas_inject::{register_dgemm, cblas_dgemm, CblasRowMajor, CblasNoTrans};
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

mod backend;
mod types;
mod xerbla;

#[cfg(feature = "openblas")]
mod autoregister;

pub mod blas1;
pub mod blas2;
pub mod blas3;

pub use backend::*;
pub use types::*;

// Re-export commonly used functions at crate root
// BLAS Level 1
pub use blas1::dot::{
    cblas_cdotc_sub, cblas_cdotu_sub, cblas_dasum, cblas_ddot, cblas_dnrm2, cblas_dsdot,
    cblas_dzasum, cblas_dznrm2, cblas_icamax, cblas_idamax, cblas_isamax, cblas_izamax,
    cblas_sasum, cblas_scasum, cblas_scnrm2, cblas_sdot, cblas_sdsdot, cblas_snrm2,
    cblas_zdotc_sub, cblas_zdotu_sub,
};
pub use blas1::rot::{
    cblas_dcabs1, cblas_drot, cblas_drotg, cblas_drotm, cblas_drotmg, cblas_scabs1, cblas_srot,
    cblas_srotg, cblas_srotm, cblas_srotmg,
};
pub use blas1::vector::{
    cblas_caxpy, cblas_ccopy, cblas_cscal, cblas_csscal, cblas_cswap, cblas_daxpy, cblas_dcopy,
    cblas_dscal, cblas_dswap, cblas_saxpy, cblas_scopy, cblas_sscal, cblas_sswap, cblas_zaxpy,
    cblas_zcopy, cblas_zdscal, cblas_zscal, cblas_zswap,
};

// BLAS Level 2
pub use blas2::gbmv::{cblas_cgbmv, cblas_dgbmv, cblas_sgbmv, cblas_zgbmv};
pub use blas2::gemv::{cblas_cgemv, cblas_dgemv, cblas_sgemv, cblas_zgemv};
pub use blas2::ger::{cblas_cgerc, cblas_cgeru, cblas_dger, cblas_sger, cblas_zgerc, cblas_zgeru};
pub use blas2::sbmv::{cblas_chbmv, cblas_dsbmv, cblas_ssbmv, cblas_zhbmv};
pub use blas2::symv::{cblas_chemv, cblas_dsymv, cblas_ssymv, cblas_zhemv};
pub use blas2::syr::{
    cblas_cher, cblas_cher2, cblas_dsyr, cblas_dsyr2, cblas_ssyr, cblas_ssyr2, cblas_zher,
    cblas_zher2,
};
pub use blas2::spmv::{cblas_chpmv, cblas_dspmv, cblas_sspmv, cblas_zhpmv};
pub use blas2::spr::{
    cblas_chpr, cblas_chpr2, cblas_dspr, cblas_dspr2, cblas_sspr, cblas_sspr2, cblas_zhpr,
    cblas_zhpr2,
};
pub use blas2::tbmv::{cblas_ctbmv, cblas_dtbmv, cblas_stbmv, cblas_ztbmv};
pub use blas2::tbsv::{cblas_ctbsv, cblas_dtbsv, cblas_stbsv, cblas_ztbsv};
pub use blas2::tpmv::{
    cblas_ctpmv, cblas_ctpsv, cblas_dtpmv, cblas_dtpsv, cblas_stpmv, cblas_stpsv, cblas_ztpmv,
    cblas_ztpsv,
};
pub use blas2::trmv::{cblas_ctrmv, cblas_dtrmv, cblas_strmv, cblas_ztrmv};
pub use blas2::trsv::{cblas_ctrsv, cblas_dtrsv, cblas_strsv, cblas_ztrsv};

// BLAS Level 3
pub use blas3::gemm::{cblas_cgemm, cblas_dgemm, cblas_sgemm, cblas_zgemm};
pub use blas3::hemm::{cblas_chemm, cblas_zhemm};
pub use blas3::her2k::{cblas_cher2k, cblas_zher2k};
pub use blas3::herk::{cblas_cherk, cblas_zherk};
pub use blas3::symm::{cblas_csymm, cblas_dsymm, cblas_ssymm, cblas_zsymm};
pub use blas3::syr2k::{cblas_csyr2k, cblas_dsyr2k, cblas_ssyr2k, cblas_zsyr2k};
pub use blas3::syrk::{cblas_csyrk, cblas_dsyrk, cblas_ssyrk, cblas_zsyrk};
pub use blas3::trmm::{cblas_ctrmm, cblas_dtrmm, cblas_strmm, cblas_ztrmm};
pub use blas3::trsm::{cblas_ctrsm, cblas_dtrsm, cblas_strsm, cblas_ztrsm};

// Error handling
pub use xerbla::cblas_xerbla;
