#!/usr/bin/env bash
# 2D Taylor-Green convergence study, headless, one run after another.
#   space: N = 16..256 at a fixed dt = 5e-4, so the time error sits below the spatial one
#   time:  N = 256 at dt = 0.02, 0.01, 0.005, 0.0025
# The presets are configs/TGV/tgv_2d_space_n*.cfg and configs/TGV/tgv_2d_time_n256_dt*.cfg; each writes its final field
# (STOP_EXPORT) under Validation/FVM_ADI/results/<preset name>/. Run from anywhere: ./Validation/FVM_ADI/run_tgv.sh

cd "$(dirname "$0")/../.." || exit 1

BIN=bin/nereus_headless
[ -x "$BIN" ] || BIN="bin/nereus --headless"

RUNS="tgv_2d_space_n16 tgv_2d_space_n32 tgv_2d_space_n64 tgv_2d_space_n128 tgv_2d_space_n256
      tgv_2d_time_n256_dt0p02 tgv_2d_time_n256_dt0p01 tgv_2d_time_n256_dt0p005 tgv_2d_time_n256_dt0p0025"

mkdir -p Validation/FVM_ADI/results/logs
for name in $RUNS; do
    echo "[$(date +%H:%M:%S)] $name"
    $BIN "configs/TGV/$name.cfg" --progress 1000 > "Validation/FVM_ADI/results/logs/$name.log" 2>&1
    code=$?
    [ $code -eq 0 ] || echo "    exited with $code, see Validation/FVM_ADI/results/logs/$name.log"
done
echo "[$(date +%H:%M:%S)] done"
