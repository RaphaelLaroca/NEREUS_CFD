#!/bin/bash
# Builds NEREUS from ../src into tests/work/, then runs every test suite and the headless check. Nothing is written
# outside tests/: objects, binaries and every file a test writes stay under tests/work/.
#
#   tests/run_all.sh                 every suite, then the headless check
#   tests/run_all.sh flip units      the named suites (the headless check runs with "io")
#   tests/run_all.sh flip "3D"       one suite, only the cases whose label contains "3D"
#
# Make variables pass through, e.g.  NVIDIA_GPU=0 tests/run_all.sh  for a CPU-only build.
# KEEP_WORK=1 keeps tests/work/run (what the cases wrote) for inspection.
set -u
TESTS="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$TESTS/.." && pwd)"
WORK="$TESTS/work"
ALL="adi_obstacle taylorgreen exact3d pressure flip sph transport io setup_ui units"

SUITES="$ALL"; FILTER=""
if [ $# -ge 1 ]; then
    if [ $# -eq 2 ] && [[ " $ALL " == *" $1 "* ]] && [[ " $ALL " != *" $2 "* ]]; then SUITES="$1"; FILTER="$2"
    else SUITES="$*"; fi
fi

# ---- 1. build the library objects and the headless program from src/, into tests/work ----
MAKEVARS="HEADLESS=1 BUILD_DIR=$WORK/build BIN_DIR=$WORK/bin"
[ -n "${NVIDIA_GPU:-}" ] && MAKEVARS="$MAKEVARS NVIDIA_GPU=$NVIDIA_GPU"
echo "================ building NEREUS into tests/work ================"
make -C "$ROOT" -j"$(nproc)" $MAKEVARS || { echo "FAILED: the build"; exit 1; }

cd "$ROOT"
INC="-Isrc -Isrc/headers -Isrc/headers/Solvers -Itests -Iinclude/eigen-3.3.7 -Iinclude/tinyexpr -idirafter include/amgcl"
LIBS="-fopenmp -lpthread -ldl"
GPUFLAG=""
if ls "$WORK"/build/.mode_*_gpu1_* >/dev/null 2>&1; then   # the build's mode stamp says whether it has the GPU solver
    AMGX_PATH="${AMGX_PATH:-$HOME/SDK/AMGX}"; CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
    GPUFLAG="-DNVIDIA_GPU"
    INC="$INC -I$CUDA_PATH/include -I$AMGX_PATH/include -I$AMGX_PATH/build"
    LIBS="-L$CUDA_PATH/lib64 -L$AMGX_PATH/build -lamgx -lamgxsh -lmpi -lcudart -lcublas -lcusparse -lcusolver -lrt $LIBS"
fi
OBJS="$(ls "$WORK"/build/*.o | grep -v '/main.o$') $WORK/build/tinyexpr/tinyexpr.o"

# ---- 2. the directory every case runs in ----
RUN="$WORK/run"
rm -rf "$RUN"; mkdir -p "$RUN"
cp "$TESTS/configs/solver_pressure_config.txt" "$RUN/"
ln -s ../.. "$RUN/tests"                     # the fixtures, read as tests/scenes/... and tests/configs/...
export SUITE_WORK="$RUN"
export NEREUS_ROOT="$ROOT"                   # read only: the setup-screen suite validates the shipped presets
[ "${KEEP_WORK:-0}" = 1 ] || trap 'rm -rf "$RUN"' EXIT

# ---- 3. the suites ----
FAILED=""
for s in $SUITES; do
    [ -f "tests/suite_$s.cpp" ] || { echo "unknown suite '$s' (one of: $ALL)"; FAILED="$FAILED $s"; continue; }
    echo; echo "================ building suite $s ================"
    if ! g++ -std=c++17 -O2 -fopenmp $GPUFLAG $INC "tests/suite_$s.cpp" $OBJS -o "$WORK/bin/suite_$s" $LIBS; then
        FAILED="$FAILED $s"; continue
    fi
    ok=1
    "$WORK/bin/suite_$s" $FILTER || ok=0
    rm -f "$WORK/bin/suite_$s"
    if [ "$s" = io ] && [ -z "$FILTER" ]; then
        "$TESTS/run_headless_check.sh" "$WORK/bin/nereus_headless" || ok=0
    fi
    [ $ok = 1 ] || FAILED="$FAILED $s"
done

echo
if [ -z "$FAILED" ]; then echo "ALL SUITES PASSED ($SUITES)"; exit 0; fi
echo "FAILED:$FAILED"; exit 1
