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

#![allow(clippy::missing_safety_doc)]

#[cfg(all(feature = "openblas", feature = "ilp64"))]
compile_error!(
    "the openblas feature auto-registers LP64 OpenBLAS symbols and cannot be used with ilp64; \
     disable openblas and register ILP64 providers explicitly"
);

mod backend;
mod int_convert;
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
    cblas_cdotc_sub, cblas_cdotc_sub_64, cblas_cdotu_sub, cblas_cdotu_sub_64, cblas_dasum,
    cblas_dasum_64, cblas_ddot, cblas_ddot_64, cblas_dnrm2, cblas_dnrm2_64, cblas_dsdot,
    cblas_dsdot_64, cblas_dzasum, cblas_dzasum_64, cblas_dznrm2, cblas_dznrm2_64, cblas_icamax,
    cblas_icamax_64, cblas_idamax, cblas_idamax_64, cblas_isamax, cblas_isamax_64, cblas_izamax,
    cblas_izamax_64, cblas_sasum, cblas_sasum_64, cblas_scasum, cblas_scasum_64, cblas_scnrm2,
    cblas_scnrm2_64, cblas_sdot, cblas_sdot_64, cblas_sdsdot, cblas_sdsdot_64, cblas_snrm2,
    cblas_snrm2_64, cblas_zdotc_sub, cblas_zdotc_sub_64, cblas_zdotu_sub, cblas_zdotu_sub_64,
};
pub use blas1::rot::{
    cblas_dcabs1, cblas_drot, cblas_drot_64, cblas_drotg, cblas_drotm, cblas_drotm_64,
    cblas_drotmg, cblas_scabs1, cblas_srot, cblas_srot_64, cblas_srotg, cblas_srotm,
    cblas_srotm_64, cblas_srotmg,
};
pub use blas1::vector::{
    cblas_caxpy, cblas_caxpy_64, cblas_ccopy, cblas_ccopy_64, cblas_cscal, cblas_cscal_64,
    cblas_csscal, cblas_csscal_64, cblas_cswap, cblas_cswap_64, cblas_daxpy, cblas_daxpy_64,
    cblas_dcopy, cblas_dcopy_64, cblas_dscal, cblas_dscal_64, cblas_dswap, cblas_dswap_64,
    cblas_saxpy, cblas_saxpy_64, cblas_scopy, cblas_scopy_64, cblas_sscal, cblas_sscal_64,
    cblas_sswap, cblas_sswap_64, cblas_zaxpy, cblas_zaxpy_64, cblas_zcopy, cblas_zcopy_64,
    cblas_zdscal, cblas_zdscal_64, cblas_zscal, cblas_zscal_64, cblas_zswap, cblas_zswap_64,
};

// BLAS Level 2
pub use blas2::gbmv::{
    cblas_cgbmv, cblas_cgbmv_64, cblas_dgbmv, cblas_dgbmv_64, cblas_sgbmv, cblas_sgbmv_64,
    cblas_zgbmv, cblas_zgbmv_64,
};
pub use blas2::gemv::{
    cblas_cgemv, cblas_cgemv_64, cblas_dgemv, cblas_dgemv_64, cblas_sgemv, cblas_sgemv_64,
    cblas_zgemv, cblas_zgemv_64,
};
pub use blas2::ger::{
    cblas_cgerc, cblas_cgerc_64, cblas_cgeru, cblas_cgeru_64, cblas_dger, cblas_dger_64,
    cblas_sger, cblas_sger_64, cblas_zgerc, cblas_zgerc_64, cblas_zgeru, cblas_zgeru_64,
};
pub use blas2::sbmv::{
    cblas_chbmv, cblas_chbmv_64, cblas_dsbmv, cblas_dsbmv_64, cblas_ssbmv, cblas_ssbmv_64,
    cblas_zhbmv, cblas_zhbmv_64,
};
pub use blas2::spmv::{
    cblas_chpmv, cblas_chpmv_64, cblas_dspmv, cblas_dspmv_64, cblas_sspmv, cblas_sspmv_64,
    cblas_zhpmv, cblas_zhpmv_64,
};
pub use blas2::spr::{
    cblas_chpr, cblas_chpr2, cblas_chpr2_64, cblas_chpr_64, cblas_dspr, cblas_dspr2,
    cblas_dspr2_64, cblas_dspr_64, cblas_sspr, cblas_sspr2, cblas_sspr2_64, cblas_sspr_64,
    cblas_zhpr, cblas_zhpr2, cblas_zhpr2_64, cblas_zhpr_64,
};
pub use blas2::symv::{
    cblas_chemv, cblas_chemv_64, cblas_dsymv, cblas_dsymv_64, cblas_ssymv, cblas_ssymv_64,
    cblas_zhemv, cblas_zhemv_64,
};
pub use blas2::syr::{
    cblas_cher, cblas_cher2, cblas_cher2_64, cblas_cher_64, cblas_dsyr, cblas_dsyr2,
    cblas_dsyr2_64, cblas_dsyr_64, cblas_ssyr, cblas_ssyr2, cblas_ssyr2_64, cblas_ssyr_64,
    cblas_zher, cblas_zher2, cblas_zher2_64, cblas_zher_64,
};
pub use blas2::tbmv::{
    cblas_ctbmv, cblas_ctbmv_64, cblas_dtbmv, cblas_dtbmv_64, cblas_stbmv, cblas_stbmv_64,
    cblas_ztbmv, cblas_ztbmv_64,
};
pub use blas2::tbsv::{
    cblas_ctbsv, cblas_ctbsv_64, cblas_dtbsv, cblas_dtbsv_64, cblas_stbsv, cblas_stbsv_64,
    cblas_ztbsv, cblas_ztbsv_64,
};
pub use blas2::tpmv::{
    cblas_ctpmv, cblas_ctpmv_64, cblas_ctpsv, cblas_ctpsv_64, cblas_dtpmv, cblas_dtpmv_64,
    cblas_dtpsv, cblas_dtpsv_64, cblas_stpmv, cblas_stpmv_64, cblas_stpsv, cblas_stpsv_64,
    cblas_ztpmv, cblas_ztpmv_64, cblas_ztpsv, cblas_ztpsv_64,
};
pub use blas2::trmv::{
    cblas_ctrmv, cblas_ctrmv_64, cblas_dtrmv, cblas_dtrmv_64, cblas_strmv, cblas_strmv_64,
    cblas_ztrmv, cblas_ztrmv_64,
};
pub use blas2::trsv::{
    cblas_ctrsv, cblas_ctrsv_64, cblas_dtrsv, cblas_dtrsv_64, cblas_strsv, cblas_strsv_64,
    cblas_ztrsv, cblas_ztrsv_64,
};

// BLAS Level 3
pub use blas3::gemm::{
    cblas_cgemm, cblas_cgemm_64, cblas_dgemm, cblas_dgemm_64, cblas_sgemm, cblas_sgemm_64,
    cblas_zgemm, cblas_zgemm_64,
};
pub use blas3::hemm::{cblas_chemm, cblas_chemm_64, cblas_zhemm, cblas_zhemm_64};
pub use blas3::her2k::{cblas_cher2k, cblas_cher2k_64, cblas_zher2k, cblas_zher2k_64};
pub use blas3::herk::{cblas_cherk, cblas_cherk_64, cblas_zherk, cblas_zherk_64};
pub use blas3::symm::{
    cblas_csymm, cblas_csymm_64, cblas_dsymm, cblas_dsymm_64, cblas_ssymm, cblas_ssymm_64,
    cblas_zsymm, cblas_zsymm_64,
};
pub use blas3::syr2k::{
    cblas_csyr2k, cblas_csyr2k_64, cblas_dsyr2k, cblas_dsyr2k_64, cblas_ssyr2k, cblas_ssyr2k_64,
    cblas_zsyr2k, cblas_zsyr2k_64,
};
pub use blas3::syrk::{
    cblas_csyrk, cblas_csyrk_64, cblas_dsyrk, cblas_dsyrk_64, cblas_ssyrk, cblas_ssyrk_64,
    cblas_zsyrk, cblas_zsyrk_64,
};
pub use blas3::trmm::{
    cblas_ctrmm, cblas_ctrmm_64, cblas_dtrmm, cblas_dtrmm_64, cblas_strmm, cblas_strmm_64,
    cblas_ztrmm, cblas_ztrmm_64,
};
pub use blas3::trsm::{
    cblas_ctrsm, cblas_ctrsm_64, cblas_dtrsm, cblas_dtrsm_64, cblas_strsm, cblas_strsm_64,
    cblas_ztrsm, cblas_ztrsm_64,
};

// Error handling
pub use xerbla::cblas_xerbla;
