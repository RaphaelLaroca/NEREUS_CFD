#!/usr/bin/env bash
# The Eulerian (FVM-ADI) benchmarks of the paper, headless, one run after another (never in parallel: every run
# uses all the OpenMP threads its preset asks for). Each writes its final field (STOP_EXPORT) and telemetry under
# Validation/FVM_ADI/results/<preset name>/, and its console log under results/logs/.
#
#   ghia     configs/Ghia/2d_ghia_re{100,1000,10000}.cfg        -> plot_ghia.py
#   schafer  configs/Schafer/2d_schafer_2d-1_sym_l{1,2,3}.cfg           -> plot_schafer.py
#   es       configs/ES/es_3d_space_n{8..64}, es_3d_time_n64_*  -> plot_es_convergence.py
#   tgv      configs/TGV/tgv_2d_{space,time}_*                  -> plot_tgv_convergence.py (not in the default set)
#
#   ./Validation/FVM_ADI/run_fvm_benchmarks.sh                 ghia, schafer, es
#   SETS="schafer" ./Validation/FVM_ADI/run_fvm_benchmarks.sh  one set
#   SETS="ghia schafer es tgv" ...                             everything
#   MAX_STEPS=400000 ...                                       the cap on a run stopped by its residual (default)
#   BIN="bin/nereus_headless" ...                             the binary (default: the GUI build, run --headless)
#
# A failed or capped run is logged and the batch moves on.

cd "$(dirname "$0")/../.." || exit 1

SETS=${SETS:-"ghia schafer es"}
MAX_STEPS=${MAX_STEPS:-400000}
BIN=${BIN:-"bin/nereus --headless"}
LOGS=Validation/FVM_ADI/results/logs
mkdir -p "$LOGS"

run() {   # run <config> [extra args...]
    local cfg=$1; shift
    local name
    name=$(basename "$cfg" .cfg)
    echo "[$(date +%H:%M:%S)] $name"
    $BIN "$cfg" --progress 1000 "$@" > "$LOGS/$name.log" 2>&1
    local code=$?
    case $code in
        0) grep -E "^(steps|time|wall) " "$LOGS/$name.log" | sed 's/^/    /' ;;
        *) echo "    exited with $code, see $LOGS/$name.log" ;;
    esac
}

for set in $SETS; do
    case $set in
        ghia)
            for re in 100 1000 10000; do run "configs/Ghia/2d_ghia_re$re.cfg" --steps "$MAX_STEPS"; done ;;
        schafer)
            for l in 1 2 3; do run "configs/Schafer/2d_schafer_2d-1_sym_l$l.cfg" --steps "$MAX_STEPS"; done ;;
        es)
            for n in 8 16 32 64; do run "configs/ES/es_3d_space_n$n.cfg"; done
            for dt in 0p01 0p005 0p0025 0p00125; do run "configs/ES/es_3d_time_n64_dt$dt.cfg"; done ;;
        tgv)
            for n in 16 32 64 128 256; do run "configs/TGV/tgv_2d_space_n$n.cfg"; run "configs/TGV/tgv_2d_space_n${n}_upwind.cfg"; done
            for dt in 0p02 0p01 0p005 0p0025; do run "configs/TGV/tgv_2d_time_n256_dt$dt.cfg"; done ;;
        *) echo "unknown set '$set' (ghia, schafer, es, tgv)" ;;
    esac
done
echo "[$(date +%H:%M:%S)] done"
