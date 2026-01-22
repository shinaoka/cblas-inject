//! CBLAS type definitions.
//!
//! These types are compatible with the CBLAS standard as defined in:
//! <https://www.netlib.org/blas/blast-forum/cblas.tgz>
//!
//! The enum values match those used by OpenBLAS:
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/cblas.h>

#![allow(non_camel_case_types)]

use std::ffi::c_char;

/// Matrix layout/order for CBLAS functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CBLAS_ORDER {
    /// Row-major order (C-style)
    CblasRowMajor = 101,
    /// Column-major order (Fortran-style)
    CblasColMajor = 102,
}

pub use CBLAS_ORDER::{CblasColMajor, CblasRowMajor};

/// Alias for CBLAS_ORDER (used by some CBLAS implementations)
pub type CBLAS_LAYOUT = CBLAS_ORDER;

/// Transpose operation for CBLAS functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CBLAS_TRANSPOSE {
    /// No transpose
    CblasNoTrans = 111,
    /// Transpose
    CblasTrans = 112,
    /// Conjugate transpose (Hermitian)
    CblasConjTrans = 113,
}

pub use CBLAS_TRANSPOSE::{CblasConjTrans, CblasNoTrans, CblasTrans};

/// Upper/Lower triangle selector for symmetric/triangular operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CBLAS_UPLO {
    /// Upper triangle
    CblasUpper = 121,
    /// Lower triangle
    CblasLower = 122,
}

pub use CBLAS_UPLO::{CblasLower, CblasUpper};

/// Diagonal type for triangular operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CBLAS_DIAG {
    /// Non-unit diagonal
    CblasNonUnit = 131,
    /// Unit diagonal (diagonal elements assumed to be 1)
    CblasUnit = 132,
}

pub use CBLAS_DIAG::{CblasNonUnit, CblasUnit};

/// Side selector (left/right multiplication).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CBLAS_SIDE {
    /// Multiply from left: op(A) * B
    CblasLeft = 141,
    /// Multiply from right: B * op(A)
    CblasRight = 142,
}

pub use CBLAS_SIDE::{CblasLeft, CblasRight};

/// Integer type for BLAS operations (LP64: 32-bit).
#[cfg(not(feature = "ilp64"))]
pub type blasint = i32;

/// Integer type for BLAS operations (ILP64: 64-bit).
#[cfg(feature = "ilp64")]
pub type blasint = i64;

/// Convert CBLAS_TRANSPOSE to Fortran character.
#[inline]
pub(crate) fn transpose_to_char(trans: CBLAS_TRANSPOSE) -> c_char {
    match trans {
        CblasNoTrans => b'N' as c_char,
        CblasTrans => b'T' as c_char,
        CblasConjTrans => b'C' as c_char,
    }
}

/// Convert CBLAS_UPLO to Fortran character.
#[inline]
#[allow(dead_code)]
pub(crate) fn uplo_to_char(uplo: CBLAS_UPLO) -> c_char {
    match uplo {
        CblasUpper => b'U' as c_char,
        CblasLower => b'L' as c_char,
    }
}

/// Convert CBLAS_DIAG to Fortran character.
#[inline]
#[allow(dead_code)]
pub(crate) fn diag_to_char(diag: CBLAS_DIAG) -> c_char {
    match diag {
        CblasNonUnit => b'N' as c_char,
        CblasUnit => b'U' as c_char,
    }
}

/// Convert CBLAS_SIDE to Fortran character.
#[inline]
#[allow(dead_code)]
pub(crate) fn side_to_char(side: CBLAS_SIDE) -> c_char {
    match side {
        CblasLeft => b'L' as c_char,
        CblasRight => b'R' as c_char,
    }
}

/// Complex return value ABI convention for Fortran BLAS functions.
///
/// Fortran complex function return values have two ABIs:
/// - **ReturnValue**: Complex returned via register (OpenBLAS, MKL intel, BLIS)
/// - **HiddenArgument**: Complex written to first pointer argument (gfortran default, MKL gf)
///
/// Only 4 BLAS functions are affected: `cdotu`, `cdotc`, `zdotu`, `zdotc`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComplexReturnStyle {
    /// Complex returned via register (OpenBLAS, MKL intel, BLIS)
    #[default]
    ReturnValue = 0,
    /// Complex written to first pointer argument (gfortran default, MKL gf)
    HiddenArgument = 1,
}

/// Index type returned by iamax functions
pub type CBLAS_INDEX = blasint;
