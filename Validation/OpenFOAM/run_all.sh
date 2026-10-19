#!/usr/bin/env bash
# The overnight set: lid-driven cavity at Re 100 and 1000, OpenFOAM (icoFoam) and NEREUS (ADI, headless) on the
# same cases, one after the other so they never share the cores. A failed case is logged and the batch moves on.
# Progress goes to results/batch.log.
#
#   OpenFOAM   run_ldc.sh N RE -> results/re<RE>_n<N>/ (timings.csv, summary.txt, grid-final.vtr)
#   NEREUS  OpenFOAM_Presets/2d_OpenFOAM_re<RE>_n<N>_<cpu|gpu>.cfg, run from the repo root by bin/nereus_headless;
#              the preset exports to results/2d_OpenFOAM_re<RE>_n<N>_<dev>/<N>_2D_re<RE>/ (telemetry.csv with one
#              <step>_timing column per step, the final .vtr), and this script adds log.run and summary.txt there.
#
# The presets match run_ldc.sh: the same fluid cells (NEREUS's N x N fluid cells on [0,1]^2, its ring outside, are
# OpenFOAM's N^2 mesh), dt = COURANT/N = 0.5/N fixed, end time 30 (Re 100) / 60 (Re 1000), central convection, a pressure
# tolerance of 1e-6 relative, incremental projection, warm start on. Edit a preset and run_ldc.sh together.
#
# Environment overrides (defaults in brackets):
#   RES ["100 1000"]   NS ["64 128 256 512 1024"]   DEVICES ["cpu gpu"]   CORES [6]
#   RUN_FOAM [1]       RUN_NEREUS [1]        BUILD [1]  (make NVIDIA_GPU=1 HEADLESS=1 first)
#   RES3D ["100"]      NS3D ["128"]   the 3D cube, after the 2D set: DIM=3 run_ldc.sh -> results/re<RE>_n<N>_3d/, and
#                      OpenFOAM_Presets/3d_OpenFOAM_re<RE>_n<N>_<dev>.cfg -> results/3d_OpenFOAM_.../<N>_3D_re<RE>/.
#                      RES3D="" skips it; RES="" runs the 3D case alone.
#   anything run_ldc.sh reads (COURANT, END_TIME, P_TOL, ...) -- but the presets do not follow it

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PRESETS="$HERE/OpenFOAM_Presets"
BIN="$ROOT/bin/nereus_headless"
mkdir -p "$HERE/results"
LOG="$HERE/results/batch.log"

RES=${RES:-"100 1000"}
NS=${NS:-"64 128 256 512"}
RES3D=${RES3D-"100"}
NS3D=${NS3D:-"128"}   # 1024: icoFoam ~5 s/step, ~87 h to t = 30 (10-09); NS="1024" to run it anyway
DEVICES=${DEVICES:-"cpu gpu"}
RUN_FOAM=${RUN_FOAM:-1}
RUN_NEREUS=${RUN_NEREUS:-1}
BUILD=${BUILD:-1}
export CORES=${CORES:-6}
export DIM=2

if [ "$RUN_NEREUS" = 1 ]; then
    if [ "$BUILD" = 1 ]; then
        echo "building $BIN" | tee -a "$LOG"
        if ! make -C "$ROOT" NVIDIA_GPU=1 HEADLESS=1 -j"$(nproc)" >> "$LOG" 2>&1; then
            echo "build FAILED, see $LOG" | tee -a "$LOG"; exit 1
        fi
    fi
    # Without NVIDIA_GPU the reader forces GPU_ACCELERATION off and the gpu presets would silently run on the CPU.
    if [[ " $DEVICES " == *" gpu "* ]] && ! "$BIN" --build-info 2>/dev/null | grep -q '^gpu 1'; then
        echo "$BIN is a CPU-only build; the gpu presets would run on the CPU. Rebuild with NVIDIA_GPU=1." | tee -a "$LOG"
        exit 1
    fi
fi

# One NEREUS case: time the whole run, keep the log and a summary beside the preset's own exports. D is 2 or 3.
run_nereus() {
    local RE=$1 N=$2 DEV=$3 D=${4:-2}
    local NAME="${D}d_OpenFOAM_re${RE}_n${N}_${DEV}"
    local CFG="$PRESETS/$NAME.cfg"
    local OUT="$HERE/results/$NAME/${N}_${D}D_re${RE}"
    [ -f "$CFG" ] || { echo "  $NAME: no preset $CFG" | tee -a "$LOG"; return 1; }
    rm -rf "$HERE/results/$NAME"
    mkdir -p "$OUT"
    local T0 T1 RC
    T0=$(date +%s.%N)
    (cd "$ROOT" && "$BIN" --headless "$CFG" --progress 1000) > "$OUT/log.run" 2>&1
    RC=$?
    T1=$(date +%s.%N)
    local DT END STEPS WALL
    DT=$(awk -F= '/^TIME_STEP/ { gsub(/ /, "", $2); print $2 }' "$CFG")
    END=$(awk -F= '/^STOP_TIME/ { gsub(/ /, "", $2); print $2 }' "$CFG")
    STEPS=$(awk '/^steps / { print $2 }' "$OUT/log.run")
    WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN { printf "%.3f", b - a }')
    cp "$CFG" "$OUT/"
    cat > "$OUT/summary.txt" <<EOF
code                 NEREUS ADI (incremental, central flux form), headless
dimension            $D
device               $DEV
grid_size            $N
reynolds_nominal     $RE
dt                   $DT
end_time             $END
steps                $STEPS
threads              $CORES
pressure             $([ "$DEV" = gpu ] && echo "AMGX PCG + aggregation AMG (SIZE_2, block Jacobi), RELATIVE_INI_CORE 1e-6" || echo "AMGCL CG + smoothed aggregation, relative 1e-6")
exit_code            $RC
total_wall_s         $WALL
total_wall_per_step_s $(awk -v s="$WALL" -v n="$STEPS" 'BEGIN { if (n > 0) printf "%.6g", s / n; else print "nan" }')
EOF
    [ "$RC" = 0 ]
}

echo "batch started $(date)" | tee -a "$LOG"
for RE in $RES; do
    for N in $NS; do
        if [ "$RUN_FOAM" = 1 ]; then
            if "$HERE/run_ldc.sh" "$N" "$RE" >> "$LOG" 2>&1; then
                echo "  openfoam re${RE}_n${N}: ok" | tee -a "$LOG"
            else
                echo "  openfoam re${RE}_n${N}: FAILED (see results/re${RE}_n${N}/log.*)" | tee -a "$LOG"
            fi
        fi
        if [ "$RUN_NEREUS" = 1 ]; then
            for DEV in $DEVICES; do
                if run_nereus "$RE" "$N" "$DEV"; then
                    echo "  nereus re${RE}_n${N}_${DEV}: ok ($(awk '/^total_wall_s/ { print $2 }' \
                        "$HERE/results/2d_OpenFOAM_re${RE}_n${N}_${DEV}/${N}_2D_re${RE}/summary.txt") s)" | tee -a "$LOG"
                else
                    echo "  nereus re${RE}_n${N}_${DEV}: FAILED (see results/2d_OpenFOAM_re${RE}_n${N}_${DEV}/*/log.run)" | tee -a "$LOG"
                fi
            done
        fi
    done
done
# The 3D cube. The OpenFOAM side is the same script with DIM=3 (frontAndBack become no-slip walls).
for RE in $RES3D; do
    for N in $NS3D; do
        if [ "$RUN_FOAM" = 1 ]; then
            if DIM=3 "$HERE/run_ldc.sh" "$N" "$RE" >> "$LOG" 2>&1; then
                echo "  openfoam re${RE}_n${N}_3d: ok" | tee -a "$LOG"
            else
                echo "  openfoam re${RE}_n${N}_3d: FAILED (see results/re${RE}_n${N}_3d/log.*)" | tee -a "$LOG"
            fi
        fi
        if [ "$RUN_NEREUS" = 1 ]; then
            for DEV in $DEVICES; do
                SUM="$HERE/results/3d_OpenFOAM_re${RE}_n${N}_${DEV}/${N}_3D_re${RE}/summary.txt"
                if run_nereus "$RE" "$N" "$DEV" 3; then
                    echo "  nereus re${RE}_n${N}_3d_${DEV}: ok ($(awk '/^total_wall_s/ { print $2 }' "$SUM") s)" | tee -a "$LOG"
                else
                    echo "  nereus re${RE}_n${N}_3d_${DEV}: FAILED (see results/3d_OpenFOAM_re${RE}_n${N}_${DEV}/*/log.run)" | tee -a "$LOG"
                fi
            done
        fi
    done
done
echo "batch finished $(date)" | tee -a "$LOG"
