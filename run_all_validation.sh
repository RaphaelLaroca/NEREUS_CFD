#!/usr/bin/env bash
# Every validation run of the paper, headless, one after another -- cheapest first, the long ones last, so a problem
# shows up early and whatever is left when you stop it is the expensive tail. The OpenFOAM batch is not here
# (Validation/OpenFOAM/run_all.sh).
#
#   ./run_all_validation.sh                  everything, in order
#   ONLY="ghia|schafer" ./run_all_validation.sh   only the runs whose preset path matches the regex
#   FORCE=1 ...                              rerun runs already marked done (default: skip them, so an interrupted
#                                            batch resumes where it stopped)
#   BUILD=0 ...                              do not run make first
#   BIN="bin/nereus_headless" ...           the binary (default: the GUI build, run --headless)
#   NEW_ONLY=1 ...                           only the entries flagged new: (below)
#   IGNORE_STALE=1 ...                       trust a .done marker even when the preset is newer than it
#   CPU_BIN="bin/nereus_headless_cpu" ...   the CPU-only binary the Scalability CPU runs use (built with BUILD=1)
#
# A preset with GPU_ACCELERATION = 1 runs only when BIN was built with NVIDIA_GPU (its --build-info says gpu 1) and
# nvidia-smi sees a device. Otherwise it is skipped and listed as such: a CPU-only build would quietly run it on the
# CPU, and its timings and memory would land in the results as GPU numbers.
#
# An entry written `new:<preset>` is flagged UNFINISHED: its results are missing or no longer belong to the current
# code or preset. The flag only marks it (and NEW_ONLY picks those); a run is skipped when its .done marker exists AND is
# newer than the preset, so a new or edited preset runs on the next batch either way. Drop the flag once its results
# are in the paper.
#
# Each run writes its results where its preset says (Validation/FVM_ADI/results, Validation/FREE_SURFACE/results), its
# console log to Validation/logs/<preset>.log, and Validation/logs/<preset>.done when it finished cleanly. Runs are
# NEVER started in parallel: each uses every OpenMP thread its preset asks for, and the GPU ones share the card.
# A failed run is logged and the batch moves on; the summary at the end lists what failed.
#
# Afterwards: Validation/FVM_ADI/plot_{tgv_convergence,es_convergence,ghia,schafer}.py and
# Validation/FREE_SURFACE/plot_{martin_moyce,lobovsky}.py. SPHERIC Test 2 (Kleefsman) was dropped on 2026-09-30.
#
# Scalability (configs/Scalability/, Validation/Scalability/plot_scalability.py): the CPU presets run on the CPU-only
# binary, so the memory they report is the solver's and not the CUDA/AMGX libraries'; the _gpu presets run on BIN.
# Each is wrapped in /usr/bin/time -v (peak host memory, "Maximum resident set size" in the log), and a _gpu run has
# its device memory sampled with nvidia-smi every 0.2 s, written to the log as "GPU peak memory (MiB): <n>".

cd "$(dirname "$0")" || exit 1

ONLY=${ONLY:-}
FORCE=${FORCE:-0}
BUILD=${BUILD:-1}
NEW_ONLY=${NEW_ONLY:-0}
IGNORE_STALE=${IGNORE_STALE:-0}
BIN=${BIN:-"bin/nereus --headless"}
CPU_BIN=${CPU_BIN:-"bin/nereus_headless_cpu"}
LOGS=Validation/logs
mkdir -p "$LOGS"

if [ "$BUILD" = 1 ]; then
    echo "[$(date +%H:%M:%S)] make"
    make -j"$(nproc)" > "$LOGS/make.log" 2>&1 || { echo "make failed, see $LOGS/make.log"; exit 1; }
    echo "[$(date +%H:%M:%S)] make (CPU-only binary, $CPU_BIN)"
    make NVIDIA_GPU=0 HEADLESS=1 BUILD_DIR=build_cpu TARGET="$CPU_BIN" -j"$(nproc)" > "$LOGS/make_cpu.log" 2>&1 ||
        { echo "make failed, see $LOGS/make_cpu.log"; exit 1; }
fi

# Can BIN run the GPU presets? The build must have AMGX and the machine a visible NVIDIA device.
GPU_OK=0
GPU_WHY=""
if ! $BIN --build-info 2>/dev/null | grep -q '^gpu 1'; then
    GPU_WHY="$BIN was built without NVIDIA_GPU"
elif ! nvidia-smi -L > /dev/null 2>&1; then
    GPU_WHY="nvidia-smi sees no device"
else
    GPU_OK=1
fi
[ "$GPU_OK" = 1 ] || echo "[$(date +%H:%M:%S)] GPU presets will be skipped: $GPU_WHY"

# Peak device memory of the running program (every process whose name holds bin/nereus), written to $1 as it rises.
gpu_sampler() {
    local peak=0 m
    echo 0 > "$1"
    while :; do
        m=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null |
            awk -F', ' '$1 ~ /bin\/nereus/ {s += $2} END {print s + 0}')
        if [ "${m:-0}" -gt "$peak" ]; then peak=$m; echo "$peak" > "$1"; fi
        sleep 0.2
    done
}

# ---- the runs, cheapest first -------------------------------------------------------------------------------------
# Rough cost on this machine, from earlier runs of the same sizes: seconds for the first block, minutes for the middle,
# hours at the end (Ghia Re 10000, Schaefer l3, the 3D-1Q/1Z n84 runs, the 3D viscous dam break).
RUNS=(
    # Taylor-Green 2D, space: N = 16..128, central and upwind
    configs/TGV/tgv_2d_space_n16.cfg        configs/TGV/tgv_2d_space_n16_upwind.cfg
    configs/TGV/tgv_2d_space_n32.cfg        configs/TGV/tgv_2d_space_n32_upwind.cfg
    configs/TGV/tgv_2d_space_n64.cfg        configs/TGV/tgv_2d_space_n64_upwind.cfg
    # the same with the non-incremental (Chorin) projection, central only: the incremental/Chorin comparison
    new:configs/TGV/tgv_2d_space_n16_chorin.cfg  new:configs/TGV/tgv_2d_space_n32_chorin.cfg
    new:configs/TGV/tgv_2d_space_n64_chorin.cfg
    # Ethier-Steinman 3D, space: N = 8..32
    configs/ES/es_3d_space_n8.cfg           configs/ES/es_3d_space_n16.cfg
    configs/ES/es_3d_space_n32.cfg
    # The practical example: the same cavity file (Re 100, 64^2, 1 thread) under the explicit method and ADI, the two
    # configs differing only in TYPE. Scored by Validation/FVM_ADI/compare_explicit_adi.py
    configs/Explicit/ldc_2d_re100_n64_explicit.cfg
    configs/Explicit/ldc_2d_re100_n64_adi.cfg
    # Scalability, 3D cavity Re 100: memory and step time against N = 16..128 (6 threads, 100 steps), then OpenMP scaling at 128^3
    # (1, 2, 3, 4, 6, 8 threads; 8 is past the 6 physical cores). CPU runs on CPU_BIN, _gpu on BIN (AMGX pressure).
    new:configs/Scalability/cavity3d_re100_n16.cfg      new:configs/Scalability/cavity3d_re100_n16_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n32.cfg      new:configs/Scalability/cavity3d_re100_n32_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n48.cfg      new:configs/Scalability/cavity3d_re100_n48_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n64.cfg      new:configs/Scalability/cavity3d_re100_n64_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n96.cfg      new:configs/Scalability/cavity3d_re100_n96_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n128.cfg     new:configs/Scalability/cavity3d_re100_n128_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n128_t1.cfg  new:configs/Scalability/cavity3d_re100_n128_t1_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n128_t2.cfg  new:configs/Scalability/cavity3d_re100_n128_t2_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n128_t3.cfg  new:configs/Scalability/cavity3d_re100_n128_t3_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n128_t4.cfg  new:configs/Scalability/cavity3d_re100_n128_t4_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n128_t6.cfg  new:configs/Scalability/cavity3d_re100_n128_t6_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n128_t8.cfg  new:configs/Scalability/cavity3d_re100_n128_t8_gpu.cfg
    # The same thread sweep at 64^3 (the pressure fits in the 96 MB L3 and scales) and 96^3 (it no longer does): the
    # cache falloff of the bandwidth-bound pressure solve.
    new:configs/Scalability/cavity3d_re100_n64_t1.cfg   new:configs/Scalability/cavity3d_re100_n64_t1_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n64_t2.cfg   new:configs/Scalability/cavity3d_re100_n64_t2_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n64_t3.cfg   new:configs/Scalability/cavity3d_re100_n64_t3_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n64_t4.cfg   new:configs/Scalability/cavity3d_re100_n64_t4_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n64_t6.cfg   new:configs/Scalability/cavity3d_re100_n64_t6_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n64_t8.cfg   new:configs/Scalability/cavity3d_re100_n64_t8_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n96_t1.cfg   new:configs/Scalability/cavity3d_re100_n96_t1_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n96_t2.cfg   new:configs/Scalability/cavity3d_re100_n96_t2_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n96_t3.cfg   new:configs/Scalability/cavity3d_re100_n96_t3_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n96_t4.cfg   new:configs/Scalability/cavity3d_re100_n96_t4_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n96_t6.cfg   new:configs/Scalability/cavity3d_re100_n96_t6_gpu.cfg
    new:configs/Scalability/cavity3d_re100_n96_t8.cfg   new:configs/Scalability/cavity3d_re100_n96_t8_gpu.cfg
    # Martin & Moyce FLIP, a/24 and a/48
    new:configs/MartinMoyce/martin_moyce_2d_n2_g74.cfg
    new:configs/MartinMoyce/martin_moyce_2d_n2_g148.cfg
    configs/TGV/tgv_2d_space_n128.cfg       configs/TGV/tgv_2d_space_n128_upwind.cfg
    new:configs/TGV/tgv_2d_space_n128_chorin.cfg
    # Taylor-Green 2D, time: N = 256, dt 0.02 .. 0.0025, incremental then Chorin
    configs/TGV/tgv_2d_time_n256_dt0p02.cfg configs/TGV/tgv_2d_time_n256_dt0p01.cfg
    configs/TGV/tgv_2d_time_n256_dt0p005.cfg configs/TGV/tgv_2d_time_n256_dt0p0025.cfg
    new:configs/TGV/tgv_2d_time_n256_dt0p02_chorin.cfg  new:configs/TGV/tgv_2d_time_n256_dt0p01_chorin.cfg
    new:configs/TGV/tgv_2d_time_n256_dt0p005_chorin.cfg new:configs/TGV/tgv_2d_time_n256_dt0p0025_chorin.cfg
    # Ghia Re 100, Schaefer 2D-1 level 1, Lobovsky FLIP at 3.33 mm (all Lobovsky: rerun in the 1.2 m open-top domain)
    new:configs/Ghia/2d_ghia_re100.cfg
    new:configs/Schafer/2d_schafer_2d-1_sym_l1.cfg
    new:configs/Lobovsky/lobovsky_2d_h300_g150_dc.cfg
    # viscous dam break (FLIP, Ancey et al. 2009), 2D at 6 mm (4 m flume, FLIP_VISCOUS_PRESSURE, dt 0.01)
    configs/Ancey/viscous_dambreak_2d_g50.cfg
    # the same with PIC (ALPHA = 0), the FLIP / PIC comparison of the viscous dam break
    configs/Ancey/viscous_dambreak_2d_g50_pic.cfg
    # Schaefer 3D-1Q (square) and 3D-1Z (circle), 10 cells across; rerun for the 3D control-volume force (cd_cv)
    configs/Schafer/3d_schafer1q_n43.cfg
    configs/Schafer/3d_schafer_3d-1z_n43.cfg
    # Martin & Moyce WCSPH, a/96
    new:configs/MartinMoyce/martin_moyce_2d_n2_sph.cfg
    # Taylor-Green 2D, N = 256 at dt 5e-4 (also the time study's reference)
    configs/TGV/tgv_2d_space_n256.cfg       configs/TGV/tgv_2d_space_n256_upwind.cfg
    new:configs/TGV/tgv_2d_space_n256_chorin.cfg
    # Ethier-Steinman 3D, N = 64: time (dt 0.01 .. 0.00125), then the dt 5e-4 run (space study, time reference)
    configs/ES/es_3d_time_n64_dt0p01.cfg    configs/ES/es_3d_time_n64_dt0p005.cfg
    configs/ES/es_3d_time_n64_dt0p0025.cfg  configs/ES/es_3d_time_n64_dt0p00125.cfg
    configs/ES/es_3d_space_n64.cfg
    # Schaefer 2D-1 level 2, Ghia Re 1000, Martin & Moyce FLIP a/96
    configs/Schafer/2d_schafer_2d-1_sym_l2.cfg
    configs/Ghia/2d_ghia_re1000.cfg
    new:configs/MartinMoyce/martin_moyce_2d_n2_g296.cfg
    # Lobovsky WCSPH, then FLIP at 1.11 mm
    new:configs/Lobovsky/lobovsky_2d_h300_sph.cfg
    new:configs/Lobovsky/lobovsky_2d_h300_g525_dc.cfg
    # Lobovsky 3D, FLIP with the render meshes (~500 PLY frames: mind the disk); the 2026-09-30 run had the lid
    new:configs/Lobovsky/lobovsky_3d_h300_flip.cfg
    # viscous dam break, 2D at 3 mm (~35 min)
    configs/Ancey/viscous_dambreak_2d_g100.cfg
    # Schaefer 3D-1Q and 3D-1Z, the refined n42 base (40 cells across; 1Q's grid moved with the anchored BuildAxis)
    configs/Schafer/3d_schafer1q_n42_r4.cfg
    configs/Schafer/3d_schafer_3d-1z_n42_r4.cfg
    # Lobovsky 3D, WCSPH
    new:configs/Lobovsky/lobovsky_3d_h300_sph.cfg
    # the long tail
    configs/Schafer/2d_schafer_2d-1_sym_l3.cfg
    configs/Schafer/3d_schafer1q_n84.cfg
    configs/Schafer/3d_schafer_3d-1z_n84.cfg
    new:configs/Ghia/2d_ghia_re10000.cfg
    configs/Ancey/viscous_dambreak_3d_g50.cfg          # ran 2026-09-30, ~10 h
    configs/Ancey/viscous_dambreak_3d_g50_pic.cfg  # ALPHA = 0 (PIC): the FLIP run threw stray particles off the surface at the side walls; ~10 h
    configs/Schafer/3d_schafer1q_n84_r2.cfg
)

declare -a FAILED=() DONE=() SKIPPED=() NOGPU=()
BATCH_START=$(date +%s)
TOTAL=${#RUNS[@]}
n=0

for entry in "${RUNS[@]}"; do
    n=$((n + 1))
    cfg=${entry#new:}
    name=$(basename "$cfg" .cfg)
    if [ "$NEW_ONLY" = 1 ] && [ "$cfg" = "$entry" ]; then continue; fi
    if [ -n "$ONLY" ] && ! [[ "$cfg" =~ $ONLY ]]; then continue; fi
    if [ ! -f "$cfg" ]; then
        echo "[$(date +%H:%M:%S)] ($n/$TOTAL) $name: preset not found, skipped"
        FAILED+=("$name (missing preset)")
        continue
    fi
    if [ "$FORCE" != 1 ] && [ -f "$LOGS/$name.done" ] &&
       { [ "$IGNORE_STALE" = 1 ] || [ "$LOGS/$name.done" -nt "$cfg" ]; }; then
        SKIPPED+=("$name")
        continue
    fi
    if [ "$GPU_OK" != 1 ] && grep -Eq '^[[:space:]]*GPU_ACCELERATION[[:space:]]*=[[:space:]]*1' "$cfg"; then
        echo "[$(date +%H:%M:%S)] ($n/$TOTAL) $name: needs the GPU, skipped ($GPU_WHY)"
        NOGPU+=("$name")
        continue
    fi
    rm -f "$LOGS/$name.done"


    echo "[$(date +%H:%M:%S)] ($n/$TOTAL) $name"
    t0=$(date +%s)
    if [[ "$cfg" == configs/Scalability/* ]]; then
        # Peak memory: host from /usr/bin/time -v, device from the sampler. A CPU preset runs on the CPU-only binary.
        run=$BIN
        [[ "$name" == *_gpu ]] || run=$CPU_BIN
        sampler=""
        if [[ "$name" == *_gpu ]]; then
            gpu_sampler "$LOGS/$name.gpumem" &
            sampler=$!
        fi
        /usr/bin/time -v $run "$cfg" --progress 1000 > "$LOGS/$name.log" 2>&1
        code=$?
        if [ -n "$sampler" ]; then
            kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
            echo "GPU peak memory (MiB): $(cat "$LOGS/$name.gpumem")" >> "$LOGS/$name.log"
            rm -f "$LOGS/$name.gpumem"
        fi
    else
        $BIN "$cfg" --progress 1000 > "$LOGS/$name.log" 2>&1
        code=$?
    fi
    dt=$(( $(date +%s) - t0 ))
    if [ $code -eq 0 ]; then
        printf '    done in %dh%02dm%02ds\n' $((dt / 3600)) $((dt % 3600 / 60)) $((dt % 60))
        date > "$LOGS/$name.done"
        DONE+=("$name")
    else
        printf '    exited with %d after %ds, see %s\n' "$code" "$dt" "$LOGS/$name.log"
        FAILED+=("$name (exit $code)")
    fi
done

total=$(( $(date +%s) - BATCH_START ))
echo
printf '[%s] batch finished in %dh%02dm: %d run, %d skipped (already done), %d failed\n' "$(date +%H:%M:%S)" \
       $((total / 3600)) $((total % 3600 / 60)) ${#DONE[@]} ${#SKIPPED[@]} ${#FAILED[@]}
for f in "${FAILED[@]}"; do echo "    FAILED: $f"; done
if [ ${#NOGPU[@]} -gt 0 ]; then
    echo "    ${#NOGPU[@]} GPU preset(s) not run ($GPU_WHY):"
    for f in "${NOGPU[@]}"; do echo "        $f"; done
fi
[ ${#FAILED[@]} -eq 0 ]
