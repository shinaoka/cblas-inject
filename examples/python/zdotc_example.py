#!/usr/bin/env python3
"""
Example: Complex dot product with return style configuration

This demonstrates how to configure the complex return ABI for
cdotc, cdotu, zdotc, zdotu functions.

Fortran complex functions have two calling conventions:
- ReturnValue (0): Complex returned via register (OpenBLAS, MKL intel, BLIS)
- HiddenArgument (1): Complex written to first pointer arg (gfortran default, MKL gf)

Prerequisites:
  pip install scipy numpy
  cargo build --release

Usage:
  python examples/python/zdotc_example.py
"""

import ctypes
import sys
import os
import numpy as np

import scipy.linalg.cython_blas as cython_blas

# Complex return style constants (must match Rust enum)
ComplexReturnStyle_ReturnValue = 0
ComplexReturnStyle_HiddenArgument = 1


def get_blas_func_ptr(name: str) -> int:
    """Extract function pointer from scipy's PyCapsule."""
    capsule = cython_blas.__pyx_capi__[name]
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
    capsule_name = ctypes.pythonapi.PyCapsule_GetName(capsule)
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.pythonapi.PyCapsule_GetPointer(capsule, capsule_name)


def main():
    # Determine library path based on platform
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(script_dir, "../../target/release")

    if sys.platform == "darwin":
        lib_path = os.path.join(lib_dir, "libcblas_inject.dylib")
    elif sys.platform == "linux":
        lib_path = os.path.join(lib_dir, "libcblas_inject.so")
    elif sys.platform == "win32":
        lib_path = os.path.join(lib_dir, "cblas_inject.dll")
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    # Load cblas-inject library
    lib = ctypes.CDLL(lib_path)
    print(f"Loaded cblas-inject library from {lib_path}")

    # Get zdotc function pointer from scipy
    zdotc_ptr = get_blas_func_ptr("zdotc")
    print(f"Got zdotc pointer: {hex(zdotc_ptr)}")

    # Set the complex return style BEFORE registering
    # scipy's OpenBLAS uses ReturnValue convention
    lib.set_complex_return_style.argtypes = [ctypes.c_int]
    lib.set_complex_return_style.restype = None
    lib.set_complex_return_style(ComplexReturnStyle_ReturnValue)
    print("Set complex return style to ReturnValue")

    # Register zdotc
    lib.register_zdotc.argtypes = [ctypes.c_void_p]
    lib.register_zdotc.restype = None
    lib.register_zdotc(zdotc_ptr)
    print("Registered zdotc")

    # Define cblas_zdotc_sub signature
    # void cblas_zdotc_sub(int n, const void *x, int incx, const void *y, int incy, void *result)
    lib.cblas_zdotc_sub.argtypes = [
        ctypes.c_int,                    # n
        ctypes.c_void_p,                 # x
        ctypes.c_int,                    # incx
        ctypes.c_void_p,                 # y
        ctypes.c_int,                    # incy
        ctypes.c_void_p,                 # result (output)
    ]
    lib.cblas_zdotc_sub.restype = None

    # Test vectors
    n = 4
    x = np.array([1+2j, 3+4j, 5+6j, 7+8j], dtype=np.complex128)
    y = np.array([1-1j, 2-2j, 3-3j, 4-4j], dtype=np.complex128)
    result = np.array([0+0j], dtype=np.complex128)

    # Call cblas_zdotc_sub
    # zdotc computes: conj(x)' * y = sum(conj(x[i]) * y[i])
    lib.cblas_zdotc_sub(
        n,
        x.ctypes.data_as(ctypes.c_void_p),
        1,
        y.ctypes.data_as(ctypes.c_void_p),
        1,
        result.ctypes.data_as(ctypes.c_void_p),
    )

    print(f"\nzdotc(x, y) where:")
    print(f"  x = {x}")
    print(f"  y = {y}")
    print(f"  result = {result[0]}")

    # Verify with numpy
    # zdotc = conj(x)' * y = np.vdot(x, y)
    expected = np.vdot(x, y)
    print(f"  expected (numpy vdot(x,y)) = {expected}")

    if np.isclose(result[0], expected):
        print("\n✓ Results match!")
    else:
        print("\n✗ Results don't match!")
        sys.exit(1)


if __name__ == "__main__":
    main()
