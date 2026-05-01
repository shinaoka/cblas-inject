use std::ffi::c_int;

use crate::backend::{BlasInt32, BlasInt64};
use crate::xerbla::cblas_xerbla;

#[inline]
pub(crate) fn to_lp64_i64(routine: &[u8], param: c_int, value: i64) -> Option<BlasInt32> {
    match BlasInt32::try_from(value) {
        Ok(value) => Some(value),
        Err(_) => {
            unsafe {
                cblas_xerbla(param, routine.as_ptr().cast(), std::ptr::null());
            }
            None
        }
    }
}

#[inline]
pub(crate) fn to_lp64_blasint(
    routine: &[u8],
    param: crate::blasint,
    value: crate::blasint,
) -> Option<BlasInt32> {
    to_lp64_i64(routine, param as c_int, value as i64)
}

#[inline]
pub(crate) fn to_ilp64_i32(value: i32) -> BlasInt64 {
    BlasInt64::from(value)
}

#[inline]
pub(crate) fn unchecked_lp64_i64(value: i64) -> BlasInt32 {
    debug_assert!(BlasInt32::try_from(value).is_ok());
    value as BlasInt32
}
