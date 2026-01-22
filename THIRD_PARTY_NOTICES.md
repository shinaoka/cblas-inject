# Third-party notices

This project includes or is derived from third-party software. The third-party
components listed below remain under their respective licenses.

## OpenBLAS (BSD-3-Clause)

Portions of this project are derived from OpenBLAS, including:

- Row-major conversion logic derived from OpenBLAS interface implementations:
  - `src/blas2/gbmv.rs`
  - `src/blas2/gemv.rs`
  - `src/blas2/ger.rs`
  - `src/blas2/sbmv.rs`
  - `src/blas2/spmv.rs`
  - `src/blas2/spr.rs`
  - `src/blas2/symv.rs`
  - `src/blas2/syr.rs`
  - `src/blas2/tbmv.rs`
  - `src/blas2/tbsv.rs`
  - `src/blas2/tpmv.rs`
  - `src/blas2/trmv.rs`
  - `src/blas2/trsv.rs`
  - `src/blas3/gemm.rs`
  - `src/blas3/hemm.rs`
  - `src/blas3/her2k.rs`
  - `src/blas3/herk.rs`
  - `src/blas3/symm.rs`
  - `src/blas3/syr2k.rs`
  - `src/blas3/syrk.rs`
  - `src/blas3/trmm.rs`
  - `src/blas3/trsm.rs`
- An OpenBLAS CBLAS header copy for local C test builds:
  - `ctest/cblas.h`
- The OpenBLAS source tree is included as a git submodule under `extern/OpenBLAS/`.

OpenBLAS project: `https://github.com/OpenMathLib/OpenBLAS`

### OpenBLAS license text

Copyright (c) 2011-2014, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

