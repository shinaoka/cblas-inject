use std::ffi::c_int;

use crate::backend::BlasInt32;
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
pub(crate) fn to_lp64_array_i64<const N: usize>(
    routine: &[u8],
    values: [(c_int, i64); N],
) -> Option<[BlasInt32; N]> {
    let mut converted = [0 as BlasInt32; N];
    for (idx, (param, value)) in values.into_iter().enumerate() {
        converted[idx] = to_lp64_i64(routine, param, value)?;
    }
    Some(converted)
}

#[inline]
#[cfg(feature = "ilp64")]
pub(crate) fn to_lp64_blasint(
    routine: &[u8],
    param: crate::blasint,
    value: crate::blasint,
) -> Option<BlasInt32> {
    to_lp64_i64(routine, param as c_int, value as i64)
}

#[inline]
pub(crate) fn unchecked_lp64_i64(value: i64) -> BlasInt32 {
    debug_assert!(BlasInt32::try_from(value).is_ok());
    value as BlasInt32
}
