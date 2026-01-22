#!/usr/bin/env python3
"""
Example: Using cblas-runtime from Python (scipy)

This demonstrates how to:
1. Get Fortran BLAS function pointers from scipy
2. Register them with cblas-runtime
3. Call CBLAS-style functions

Prerequisites:
  pip install scipy numpy
  cargo build --release

Usage:
  python examples/python/dgemm_example.py
"""

import ctypes
import sys
import os
import numpy as np

# scipy.linalg.cython_blas provides PyCapsule objects with BLAS function pointers
import scipy.linalg.cython_blas as cython_blas


def get_blas_func_ptr(name: str) -> int:
    """Extract function pointer from scipy's PyCapsule."""
    capsule = cython_blas.__pyx_capi__[name]
    # Get the capsule's name (it encodes the function signature)
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
    capsule_name = ctypes.pythonapi.PyCapsule_GetName(capsule)
    # PyCapsule_GetPointer returns the void* stored in the capsule
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.pythonapi.PyCapsule_GetPointer(capsule, capsule_name)


def main():
    # Determine library path based on platform
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(script_dir, "../../target/release")

    if sys.platform == "darwin":
        lib_path = os.path.join(lib_dir, "libcblas_runtime.dylib")
    elif sys.platform == "linux":
        lib_path = os.path.join(lib_dir, "libcblas_runtime.so")
    elif sys.platform == "win32":
        lib_path = os.path.join(lib_dir, "cblas_runtime.dll")
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    # Load cblas-runtime library
    lib = ctypes.CDLL(lib_path)
    print(f"Loaded cblas-runtime library from {lib_path}")

    # Get Fortran BLAS function pointer from scipy
    dgemm_ptr = get_blas_func_ptr("dgemm")
    print(f"Got dgemm pointer: {hex(dgemm_ptr)}")

    # Register with cblas-runtime
    lib.register_dgemm.argtypes = [ctypes.c_void_p]
    lib.register_dgemm.restype = None
    lib.register_dgemm(dgemm_ptr)
    print("Registered dgemm")

    # Define cblas_dgemm signature
    lib.cblas_dgemm.argtypes = [
        ctypes.c_int,     # Order
        ctypes.c_int,     # TransA
        ctypes.c_int,     # TransB
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_double,  # alpha
        ctypes.POINTER(ctypes.c_double),  # A
        ctypes.c_int,     # lda
        ctypes.POINTER(ctypes.c_double),  # B
        ctypes.c_int,     # ldb
        ctypes.c_double,  # beta
        ctypes.POINTER(ctypes.c_double),  # C
        ctypes.c_int,     # ldc
    ]
    lib.cblas_dgemm.restype = None

    # CBLAS constants
    CblasRowMajor = 101
    CblasNoTrans = 111

    # Matrix dimensions
    m, n, k = 3, 4, 2
    alpha = 1.0
    beta = 0.0

    # Row-major matrices (C-style)
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)  # 3×2
    B = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)  # 2×4
    C = np.zeros((m, n), dtype=np.float64)

    # Call cblas_dgemm
    lib.cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k,  # lda = k for row-major NoTrans
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,  # ldb = n for row-major NoTrans
        beta,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,  # ldc = n for row-major
    )

    print("\nResult C = A × B:")
    print(f"A = \n{A}")
    print(f"\nB = \n{B}")
    print(f"\nC = \n{C}")

    # Verify with numpy
    expected = A @ B
    print(f"\nExpected (numpy A @ B):\n{expected}")

    if np.allclose(C, expected):
        print("\n✓ Results match!")
    else:
        print("\n✗ Results don't match!")
        sys.exit(1)


if __name__ == "__main__":
    main()
