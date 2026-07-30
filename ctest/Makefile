# Makefile for cblas-inject ctest
# Uses test files from extern/OpenBLAS/ctest/

OPENBLAS_DIR = ../extern/OpenBLAS
OPENBLAS_CTEST = $(OPENBLAS_DIR)/ctest
OPENBLAS_COMMIT = 29703e52ad07310bf61407a86b1d5f25821fff44
VPATH = $(OPENBLAS_CTEST)

# Fetch the known-compatible OpenBLAS ctest sources into a temporary checkout,
# then publish it only after the commit has been verified.
clone-openblas:
	set -eu; \
	if test -d "$(OPENBLAS_DIR)/.git" && test "$$(git -C "$(OPENBLAS_DIR)" rev-parse HEAD 2>/dev/null)" = "$(OPENBLAS_COMMIT)" && test -z "$$(git -C "$(OPENBLAS_DIR)" status --porcelain --untracked-files=all -- ctest)"; then exit 0; fi; \
	mkdir -p "../extern"; \
	tmp="$$(mktemp -d "../extern/OpenBLAS.tmp.XXXXXX")"; \
	old="$$(mktemp -d "../extern/OpenBLAS.old.XXXXXX")"; rmdir "$$old"; \
	published=0; restored=0; \
	cleanup() { status=$$?; if test "$$published" -eq 0 && test -d "$$old" && test ! -e "$(OPENBLAS_DIR)"; then if mv "$$old" "$(OPENBLAS_DIR)"; then restored=1; else echo "failed to restore previous OpenBLAS checkout" >&2; status=1; fi; fi; rm -rf "$$tmp"; if test "$$published" -eq 1 || test "$$restored" -eq 1; then rm -rf "$$old"; fi; exit "$$status"; }; \
	trap cleanup 0; \
	trap 'trap - HUP INT TERM; exit 129' HUP; \
	trap 'trap - HUP INT TERM; exit 130' INT; \
	trap 'trap - HUP INT TERM; exit 143' TERM; \
	git -C "$$tmp" init -q; \
	git -C "$$tmp" remote add origin https://github.com/OpenMathLib/OpenBLAS.git; \
	git -C "$$tmp" fetch -q --depth 1 origin "$(OPENBLAS_COMMIT)"; \
	git -C "$$tmp" checkout -q --detach FETCH_HEAD; \
	test "$$(git -C "$$tmp" rev-parse HEAD)" = "$(OPENBLAS_COMMIT)"; \
	if test -e "$(OPENBLAS_DIR)"; then mv "$(OPENBLAS_DIR)" "$$old"; fi; \
	mv "$$tmp" "$(OPENBLAS_DIR)"; \
	published=1; rm -rf "$$old"; trap - 0 HUP INT TERM

CC = cc
CFLAGS = -DADD_ -DCBLAS -I. -I$(OPENBLAS_CTEST)

# Detect OS and set OpenBLAS path
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS (Homebrew)
    OPENBLAS_LIB ?= /opt/homebrew/opt/openblas/lib
else
    # Linux
    OPENBLAS_LIB ?= /usr/lib/x86_64-linux-gnu/openblas-pthread
endif

# Link cblas-inject first (provides cblas_* symbols)
# Then OpenBLAS for Fortran BLAS (drot_, etc.) reference
# -lm is needed on Linux for math functions (sqrt, etc.)
LDFLAGS = -L../target/release -lcblas_inject -L$(OPENBLAS_LIB) -lopenblas -lm

# BLAS Level 1 test objects
stestl1o = c_sblas1.o
dtestl1o = c_dblas1.o
ctestl1o = c_cblas1.o
ztestl1o = c_zblas1.o

# BLAS Level 2 test objects
stestl2o = c_sblas2.o c_s2chke.o auxiliary.o c_xerbla.o constant.o
dtestl2o = c_dblas2.o c_d2chke.o auxiliary.o c_xerbla.o constant.o
ctestl2o = c_cblas2.o c_c2chke.o auxiliary.o c_xerbla.o constant.o
ztestl2o = c_zblas2.o c_z2chke.o auxiliary.o c_xerbla.o constant.o

# BLAS Level 3 test objects
stestl3o = c_sblas3.o c_s3chke.o auxiliary.o c_xerbla.o constant.o
dtestl3o = c_dblas3.o c_d3chke.o auxiliary.o c_xerbla.o constant.o
ctestl3o = c_cblas3.o c_c3chke.o auxiliary.o c_xerbla.o constant.o
ztestl3o = c_zblas3.o c_z3chke.o auxiliary.o c_xerbla.o constant.o

# Test drivers
stestl1_driver = c_sblat1c.o
dtestl1_driver = c_dblat1c.o
ctestl1_driver = c_cblat1c.o
ztestl1_driver = c_zblat1c.o

stestl2_driver = c_sblat2c.o
dtestl2_driver = c_dblat2c.o
ctestl2_driver = c_cblat2c.o
ztestl2_driver = c_zblat2c.o

stestl3_driver = c_sblat3c.o
dtestl3_driver = c_dblat3c.o
ctestl3_driver = c_cblat3c.o
ztestl3_driver = c_zblat3c.o

.PHONY: all clean test test1 test2 test3 build-trampoline clone-openblas \
       _run-test1 _run-test2 _run-test3

all: clone-openblas build-trampoline xscblat1 xdcblat1 xccblat1 xzcblat1 xscblat2 xdcblat2 xccblat2 xzcblat2 xscblat3 xdcblat3 xccblat3 xzcblat3

build-trampoline:
	cd .. && cargo build --release --features openblas

# Compile sources
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# BLAS Level 1 test executables
xscblat1: $(stestl1_driver) $(stestl1o)
	$(CC) -o $@ $^ $(LDFLAGS)

xdcblat1: $(dtestl1_driver) $(dtestl1o)
	$(CC) -o $@ $^ $(LDFLAGS)

xccblat1: $(ctestl1_driver) $(ctestl1o)
	$(CC) -o $@ $^ $(LDFLAGS)

xzcblat1: $(ztestl1_driver) $(ztestl1o)
	$(CC) -o $@ $^ $(LDFLAGS)

# BLAS Level 2 test executables
xscblat2: $(stestl2_driver) $(stestl2o)
	$(CC) -o $@ $^ $(LDFLAGS)

xdcblat2: $(dtestl2_driver) $(dtestl2o)
	$(CC) -o $@ $^ $(LDFLAGS)

xccblat2: $(ctestl2_driver) $(ctestl2o)
	$(CC) -o $@ $^ $(LDFLAGS)

xzcblat2: $(ztestl2_driver) $(ztestl2o)
	$(CC) -o $@ $^ $(LDFLAGS)

# BLAS Level 3 test executables
xscblat3: $(stestl3_driver) $(stestl3o)
	$(CC) -o $@ $^ $(LDFLAGS)

xdcblat3: $(dtestl3_driver) $(dtestl3o)
	$(CC) -o $@ $^ $(LDFLAGS)

xccblat3: $(ctestl3_driver) $(ctestl3o)
	$(CC) -o $@ $^ $(LDFLAGS)

xzcblat3: $(ztestl3_driver) $(ztestl3o)
	$(CC) -o $@ $^ $(LDFLAGS)

# Use recursive make so that VPATH can find .c files after OpenBLAS is cloned.
# Make resolves the full dependency graph before executing any commands, so
# VPATH directories must exist at parse time for pattern rules to work.
test1: clone-openblas build-trampoline
	$(MAKE) _run-test1

_run-test1: xscblat1 xdcblat1 xccblat1 xzcblat1
	./xscblat1
	./xdcblat1
	./xccblat1
	./xzcblat1

test2: clone-openblas build-trampoline
	$(MAKE) _run-test2

_run-test2: xscblat2 xdcblat2 xccblat2 xzcblat2
	./xscblat2 < $(OPENBLAS_CTEST)/sin2
	./xdcblat2 < $(OPENBLAS_CTEST)/din2
	./xccblat2 < $(OPENBLAS_CTEST)/cin2
	./xzcblat2 < $(OPENBLAS_CTEST)/zin2

test3: clone-openblas build-trampoline
	$(MAKE) _run-test3

_run-test3: xscblat3 xdcblat3 xccblat3 xzcblat3
	./xscblat3 < $(OPENBLAS_CTEST)/sin3
	./xdcblat3 < $(OPENBLAS_CTEST)/din3
	./xccblat3 < $(OPENBLAS_CTEST)/cin3
	./xzcblat3 < $(OPENBLAS_CTEST)/zin3

clean:
	rm -f xscblat1 xdcblat1 xccblat1 xzcblat1
	rm -f xscblat2 xdcblat2 xccblat2 xzcblat2
	rm -f xscblat3 xdcblat3 xccblat3 xzcblat3
	rm -f *.o
