//! CBLAS error handler (xerbla).
//!
//! Note: The standard cblas_xerbla is variadic, but Rust stable doesn't support
//! C variadic functions. We provide a non-variadic version that handles the
//! common case.

use crate::types::blasint;
use std::ffi::c_char;

/// CBLAS error handler.
///
/// This function is called when an illegal parameter is detected.
/// It prints an error message to stderr.
///
/// Note: This is a simplified version that ignores the format string and
/// variadic arguments. The standard signature is:
/// `void cblas_xerbla(int p, char *rout, char *form, ...)`
///
/// # Safety
///
/// - `rout` must be a valid null-terminated C string or null
/// - `_form` is ignored in this implementation
#[no_mangle]
pub unsafe extern "C" fn cblas_xerbla(p: blasint, rout: *const c_char, _form: *const c_char) {
    let routine = if rout.is_null() {
        "<unknown>"
    } else {
        std::ffi::CStr::from_ptr(rout)
            .to_str()
            .unwrap_or("<invalid>")
    };
    eprintln!(
        "** On entry to {} parameter number {} had an illegal value",
        routine, p
    );
}
