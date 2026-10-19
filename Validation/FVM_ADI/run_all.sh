#!/usr/bin/env bash
# The overnight set: lid-driven cavity at Re 1000 on GRID_SIZE 128 and 256, on the GPU, to t = 60
# -- the cases Validation/OpenFOAM/results already holds at the same end time, so each run here
# has a counterpart to difference against.
#
# A failed case is logged and the batch moves on. Anything run_ldc.sh accepts can be overridden
# (THREADS, COURANT, END_TIME, P_TOL, GPU, BIN). Progress goes to results/batch.log.
#
#   ./run_all.sh                    the two cases above
#   CASES="1000:128" ./run_all.sh   just one
#   GPU=0 ./run_all.sh              the CPU (Eigen) path instead
#
# Re 100 is not in the default set: OpenFOAM ran it to t = 30, so add it as "100:128 100:256"
# and run_ldc.sh picks the matching end time on its own.

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
mkdir -p "$HERE/results"
LOG="$HERE/results/batch.log"

CASES=${CASES:-"1000:128 1000:256"}
export THREADS=${THREADS:-6}
export GPU=${GPU:-1}
export DIM=${DIM:-2}          # DIM=3 runs the 3D cavity

BIN=${BIN:-$ROOT/bin/nereus_headless}
# AMGX is linked statically, so CUDA is the marker for a GPU build (see run_ldc.sh).
if [ ! -x "$BIN" ] || { [ "$GPU" = "1" ] && ! ldd "$BIN" 2>/dev/null | grep -q libcudart; }; then
    echo "Build the GPU headless binary first:" | tee -a "$LOG"
    echo "  make -C $ROOT NVIDIA_GPU=1 HEADLESS=1 -j\$(nproc)" | tee -a "$LOG"
    exit 1
fi

echo "batch started $(date)" | tee -a "$LOG"
for C in $CASES; do
    RE=${C%%:*}
    N=${C##*:}
    if "$HERE/run_ldc.sh" "$N" "$RE" >> "$LOG" 2>&1; then
        echo "  re${RE}_n${N}$( [ "$DIM" = 3 ] && echo _3d ): ok" | tee -a "$LOG"
    else
        echo "  re${RE}_n${N}$( [ "$DIM" = 3 ] && echo _3d ): FAILED (see results/re${RE}_n${N}/log.run)" | tee -a "$LOG"
    fi
done
echo "batch finished $(date)" | tee -a "$LOG"
