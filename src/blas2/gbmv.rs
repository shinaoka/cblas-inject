//! General band matrix-vector multiply (GBMV) - CBLAS interface.
//!
//! Computes: y = alpha * op(A) * x + beta * y
//! where A is a band matrix stored in band storage format.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gbmv.c>
//!
//! For row-major layout:
//! - Swap m and n
//! - Swap kl and ku (sub-diagonals <-> super-diagonals)
//! - Flip the transpose operation (NoTrans <-> Trans, ConjNoTrans <-> ConjTrans)

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_cgbmv_for_ilp64_cblas, get_cgbmv_for_lp64_cblas, get_dgbmv_for_ilp64_cblas,
    get_dgbmv_for_lp64_cblas, get_sgbmv_for_ilp64_cblas, get_sgbmv_for_lp64_cblas,
    get_zgbmv_for_ilp64_cblas, get_zgbmv_for_lp64_cblas, CgbmvProvider, DgbmvProvider,
    SgbmvProvider, ZgbmvProvider,
};
use crate::types::{
    blasint, normalize_transpose_real, transpose_to_char, CblasColMajor, CblasConjNoTrans,
    CblasConjTrans, CblasNoTrans, CblasRowMajor, CblasTrans, CBLAS_ORDER, CBLAS_TRANSPOSE,
};

/// Flip transpose operation for row-major conversion (real-valued operations).
///
/// For real types, conjugation is a no-op, so we normalize to {NoTrans, Trans}.
#[inline]
fn flip_transpose_real(trans: CBLAS_TRANSPOSE) -> CBLAS_TRANSPOSE {
    match normalize_transpose_real(trans) {
        CblasNoTrans => CblasTrans,
        CblasTrans => CblasNoTrans,
        _ => unreachable!(),
    }
}

/// Single precision general band matrix-vector multiply.
///
/// Computes: y = alpha * op(A) * x + beta * y
/// where A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - sgbmv must be registered via `register_sgbmv`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_sgbmv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
) {
    let p = get_sgbmv_for_lp64_cblas();
    match p {
        SgbmvProvider::Lp64(sgbmv) => {
            match order {
                CblasColMajor => {
                    // Column-major: call Fortran directly
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    sgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // Row-major: swap m/n, kl/ku and flip transpose
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gbmv.c
                    let trans_char = transpose_to_char(flip_transpose_real(trans));
                    sgbmv(
                        &trans_char,
                        &n,  // swapped: m -> n
                        &m,  // swapped: n -> m
                        &ku, // swapped: kl -> ku
                        &kl, // swapped: ku -> kl
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
            }
        }
        SgbmvProvider::Ilp64(sgbmv) => {
            let m = m as i64;
            let n = n as i64;
            let kl = kl as i64;
            let ku = ku as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    // Column-major: call Fortran directly
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    sgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // Row-major: swap m/n, kl/ku and flip transpose
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gbmv.c
                    let trans_char = transpose_to_char(flip_transpose_real(trans));
                    sgbmv(
                        &trans_char,
                        &n,  // swapped: m -> n
                        &m,  // swapped: n -> m
                        &ku, // swapped: kl -> ku
                        &kl, // swapped: ku -> kl
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
            }
        }
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_sgbmv_64(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: i64,
    n: i64,
    kl: i64,
    ku: i64,
    alpha: f32,
    a: *const f32,
    lda: i64,
    x: *const f32,
    incx: i64,
    beta: f32,
    y: *mut f32,
    incy: i64,
) {
    let p = get_sgbmv_for_ilp64_cblas();
    if matches!(p, SgbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_sgbmv_64\0",
            [
                (3, m),
                (4, n),
                (5, kl),
                (6, ku),
                (9, lda),
                (11, incx),
                (14, incy),
            ],
        )
        .is_none()
    {
        return;
    }

    match p {
        SgbmvProvider::Ilp64(sgbmv) => {
            match order {
                CblasColMajor => {
                    // Column-major: call Fortran directly
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    sgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // Row-major: swap m/n, kl/ku and flip transpose
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gbmv.c
                    let trans_char = transpose_to_char(flip_transpose_real(trans));
                    sgbmv(
                        &trans_char,
                        &n,  // swapped: m -> n
                        &m,  // swapped: n -> m
                        &ku, // swapped: kl -> ku
                        &kl, // swapped: ku -> kl
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
            }
        }
        SgbmvProvider::Lp64(sgbmv) => {
            let m = m as i32;
            let n = n as i32;
            let kl = kl as i32;
            let ku = ku as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    // Column-major: call Fortran directly
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    sgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // Row-major: swap m/n, kl/ku and flip transpose
                    // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gbmv.c
                    let trans_char = transpose_to_char(flip_transpose_real(trans));
                    sgbmv(
                        &trans_char,
                        &n,  // swapped: m -> n
                        &m,  // swapped: n -> m
                        &ku, // swapped: kl -> ku
                        &kl, // swapped: ku -> kl
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
            }
        }
    }
}

/// Double precision general band matrix-vector multiply.
///
/// Computes: y = alpha * op(A) * x + beta * y
/// where A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dgbmv must be registered via `register_dgbmv`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_dgbmv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    let p = get_dgbmv_for_lp64_cblas();
    match p {
        DgbmvProvider::Lp64(dgbmv) => match order {
            CblasColMajor => {
                let trans_char = transpose_to_char(normalize_transpose_real(trans));
                dgbmv(
                    &trans_char,
                    &m,
                    &n,
                    &kl,
                    &ku,
                    &alpha,
                    a,
                    &lda,
                    x,
                    &incx,
                    &beta,
                    y,
                    &incy,
                );
            }
            CblasRowMajor => {
                let trans_char = transpose_to_char(flip_transpose_real(trans));
                dgbmv(
                    &trans_char,
                    &n,
                    &m,
                    &ku,
                    &kl,
                    &alpha,
                    a,
                    &lda,
                    x,
                    &incx,
                    &beta,
                    y,
                    &incy,
                );
            }
        },
        DgbmvProvider::Ilp64(dgbmv) => {
            let m = m as i64;
            let n = n as i64;
            let kl = kl as i64;
            let ku = ku as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    dgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    let trans_char = transpose_to_char(flip_transpose_real(trans));
                    dgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
            }
        }
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_dgbmv_64(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: i64,
    n: i64,
    kl: i64,
    ku: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    x: *const f64,
    incx: i64,
    beta: f64,
    y: *mut f64,
    incy: i64,
) {
    let p = get_dgbmv_for_ilp64_cblas();
    if matches!(p, DgbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dgbmv_64\0",
            [
                (3, m),
                (4, n),
                (5, kl),
                (6, ku),
                (9, lda),
                (11, incx),
                (14, incy),
            ],
        )
        .is_none()
    {
        return;
    }

    match p {
        DgbmvProvider::Ilp64(dgbmv) => match order {
            CblasColMajor => {
                let trans_char = transpose_to_char(normalize_transpose_real(trans));
                dgbmv(
                    &trans_char,
                    &m,
                    &n,
                    &kl,
                    &ku,
                    &alpha,
                    a,
                    &lda,
                    x,
                    &incx,
                    &beta,
                    y,
                    &incy,
                );
            }
            CblasRowMajor => {
                let trans_char = transpose_to_char(flip_transpose_real(trans));
                dgbmv(
                    &trans_char,
                    &n,
                    &m,
                    &ku,
                    &kl,
                    &alpha,
                    a,
                    &lda,
                    x,
                    &incx,
                    &beta,
                    y,
                    &incy,
                );
            }
        },
        DgbmvProvider::Lp64(dgbmv) => {
            let m = m as i32;
            let n = n as i32;
            let kl = kl as i32;
            let ku = ku as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(normalize_transpose_real(trans));
                    dgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    let trans_char = transpose_to_char(flip_transpose_real(trans));
                    dgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        &alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        &beta,
                        y,
                        &incy,
                    );
                }
            }
        }
    }
}

/// Single precision complex general band matrix-vector multiply.
///
/// Computes: y = alpha * op(A) * x + beta * y
/// where A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cgbmv must be registered via `register_cgbmv`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_cgbmv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    let p = get_cgbmv_for_lp64_cblas();
    match p {
        CgbmvProvider::Lp64(cgbmv) => {
            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(trans);
                    cgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // For complex, row-major requires flipping transpose with conjugation preserved:
                    // NoTrans <-> Trans, ConjNoTrans <-> ConjTrans (OpenBLAS)
                    let flipped_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        CblasConjNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasConjNoTrans,
                    };
                    let trans_char = transpose_to_char(flipped_trans);
                    cgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
            }
        }
        CgbmvProvider::Ilp64(cgbmv) => {
            let m = m as i64;
            let n = n as i64;
            let kl = kl as i64;
            let ku = ku as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(trans);
                    cgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // For complex, row-major requires flipping transpose with conjugation preserved:
                    // NoTrans <-> Trans, ConjNoTrans <-> ConjTrans (OpenBLAS)
                    let flipped_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        CblasConjNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasConjNoTrans,
                    };
                    let trans_char = transpose_to_char(flipped_trans);
                    cgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
            }
        }
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_cgbmv_64(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: i64,
    n: i64,
    kl: i64,
    ku: i64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i64,
    x: *const Complex32,
    incx: i64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i64,
) {
    let p = get_cgbmv_for_ilp64_cblas();
    if matches!(p, CgbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_cgbmv_64\0",
            [
                (3, m),
                (4, n),
                (5, kl),
                (6, ku),
                (9, lda),
                (11, incx),
                (14, incy),
            ],
        )
        .is_none()
    {
        return;
    }

    match p {
        CgbmvProvider::Ilp64(cgbmv) => {
            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(trans);
                    cgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // For complex, row-major requires flipping transpose with conjugation preserved:
                    // NoTrans <-> Trans, ConjNoTrans <-> ConjTrans (OpenBLAS)
                    let flipped_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        CblasConjNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasConjNoTrans,
                    };
                    let trans_char = transpose_to_char(flipped_trans);
                    cgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
            }
        }
        CgbmvProvider::Lp64(cgbmv) => {
            let m = m as i32;
            let n = n as i32;
            let kl = kl as i32;
            let ku = ku as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(trans);
                    cgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // For complex, row-major requires flipping transpose with conjugation preserved:
                    // NoTrans <-> Trans, ConjNoTrans <-> ConjTrans (OpenBLAS)
                    let flipped_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        CblasConjNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasConjNoTrans,
                    };
                    let trans_char = transpose_to_char(flipped_trans);
                    cgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
            }
        }
    }
}

/// Double precision complex general band matrix-vector multiply.
///
/// Computes: y = alpha * op(A) * x + beta * y
/// where A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgbmv must be registered via `register_zgbmv`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_zgbmv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: i32,
    n: i32,
    kl: i32,
    ku: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    let p = get_zgbmv_for_lp64_cblas();
    match p {
        ZgbmvProvider::Lp64(zgbmv) => {
            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(trans);
                    zgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // Same handling as cgbmv (OpenBLAS row-major transpose flip for complex)
                    let flipped_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        CblasConjNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasConjNoTrans,
                    };
                    let trans_char = transpose_to_char(flipped_trans);
                    zgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
            }
        }
        ZgbmvProvider::Ilp64(zgbmv) => {
            let m = m as i64;
            let n = n as i64;
            let kl = kl as i64;
            let ku = ku as i64;
            let lda = lda as i64;
            let incx = incx as i64;
            let incy = incy as i64;

            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(trans);
                    zgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // Same handling as cgbmv (OpenBLAS row-major transpose flip for complex)
                    let flipped_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        CblasConjNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasConjNoTrans,
                    };
                    let trans_char = transpose_to_char(flipped_trans);
                    zgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
            }
        }
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_zgbmv_64(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: i64,
    n: i64,
    kl: i64,
    ku: i64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i64,
    x: *const Complex64,
    incx: i64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i64,
) {
    let p = get_zgbmv_for_ilp64_cblas();
    if matches!(p, ZgbmvProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zgbmv_64\0",
            [
                (3, m),
                (4, n),
                (5, kl),
                (6, ku),
                (9, lda),
                (11, incx),
                (14, incy),
            ],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZgbmvProvider::Ilp64(zgbmv) => {
            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(trans);
                    zgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // Same handling as cgbmv (OpenBLAS row-major transpose flip for complex)
                    let flipped_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        CblasConjNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasConjNoTrans,
                    };
                    let trans_char = transpose_to_char(flipped_trans);
                    zgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
            }
        }
        ZgbmvProvider::Lp64(zgbmv) => {
            let m = m as i32;
            let n = n as i32;
            let kl = kl as i32;
            let ku = ku as i32;
            let lda = lda as i32;
            let incx = incx as i32;
            let incy = incy as i32;

            match order {
                CblasColMajor => {
                    let trans_char = transpose_to_char(trans);
                    zgbmv(
                        &trans_char,
                        &m,
                        &n,
                        &kl,
                        &ku,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
                CblasRowMajor => {
                    // Same handling as cgbmv (OpenBLAS row-major transpose flip for complex)
                    let flipped_trans = match trans {
                        CblasNoTrans => CblasTrans,
                        CblasTrans => CblasNoTrans,
                        CblasConjNoTrans => CblasConjTrans,
                        CblasConjTrans => CblasConjNoTrans,
                    };
                    let trans_char = transpose_to_char(flipped_trans);
                    zgbmv(
                        &trans_char,
                        &n,
                        &m,
                        &ku,
                        &kl,
                        alpha,
                        a,
                        &lda,
                        x,
                        &incx,
                        beta,
                        y,
                        &incy,
                    );
                }
            }
        }
    }
}
