#!/usr/bin/env bash
# One NEREUS lid-driven cavity run, laid out to match Validation/OpenFOAM/run_ldc.sh case for
# case so the two can be differenced directly. It writes the config itself -- the config IS the
# job, so nothing here is set interactively and the run is reproducible from this file alone.
#
# usage: ./run_ldc.sh N RE          (DIM=3 for the 3D cavity)
#   N   GRID_SIZE (128, 256). The domain is [0,1]^2 with a one-cell solid ring, so the fluid is the
#       (N-2)x(N-2) interior on [1/N, 1-1/N]^2 -- exactly the region OpenFOAM meshes, which is what
#       makes every cell centre coincide and the fields comparable without interpolation.
#   RE  nu = 1/RE with lid speed 1.
#
# WHAT IS MATCHED TO OPENFOAM, and what is not:
#   matched  domain and cell centres; uniform mesh; dt = COURANT/N, held FIXED (icoFoam runs a
#            fixed deltaT, so ADAPTATIVE_TIMESTEP is off here -- the shipped cavity preset has it
#            on, and leaving it on would compare different step counts);
#            end time (60 for Re > 100, 30 otherwise, as in run_ldc.sh);
#            central convection (icoFoam's `div(phi,U) Gauss linear`);
#            the requested pressure tolerance P_TOL.
#   NOT      the pressure solve itself. OpenFOAM runs PISO with two correctors and GAMG; this runs
#            one projection per step with AMGX PCG preconditioned by classical AMG. That is the
#            comparison, not a discrepancy to remove.
#   CAVEAT   the two codes do not define the residual the same way. OpenFOAM's `tolerance` is on a
#            normalised residual; AMGX `convergence=ABSOLUTE` is not normalised. P_TOL therefore
#            matches the requested NUMBER, not the achieved accuracy. Read div_sum in
#            timings.csv for what was actually converged -- that is the honest comparison.
#
# The shipped preset configs/cavity_2d_adi.cfg is NOT used: it carries REFINE_X/REFINE_Y (a
# stretched grid) and an adaptive timestep, neither of which OpenFOAM has.
#
# Environment overrides (defaults in brackets):
#   DIM [2]      THREADS [6]  COURANT [0.5]  END_TIME [30 for Re <= 100, 60 otherwise]
#   P_TOL [1e-6]  P_MAXITER [10000]  GPU [1]  PROGRESS [500]
#   WARM_START [1]     start each pressure solve from the previous step's field (OpenFOAM always does)
#   EIGEN_SOLVER [AMG] the CPU (GPU=0) solve: AMG (AMGCL CG + smoothed aggregation), CG or BICGSTAB
#   EIGEN_PRECON [DIAGONAL] for CG / BICGSTAB; ignored by AMG
#   PROJECTION [CHORIN] CHORIN or INCREMENTAL (the pressure correction)
#   FORM [ADVECTIVE]   ADVECTIVE or FLUX, the momentum convection form
#   SUFFIX []          appended to the result directory, so variants of one case do not overwrite
#   BIN [<repo>/bin/nereus_headless]
#
# Writes to results/re<RE>_n<N>[_3d]/, mirroring the OpenFOAM side:
#   grid-final.vtr   final field, VTK RectilinearGrid, N x N, ring included
#   timings.csv      the run's telemetry CSV (per step: time, divergence, CFL, residual, per-phase
#                    wall time) -- the <phase>_timing columns are the analogue of OpenFOAM's
#                    per-step solver time
#   summary.txt      settings and wall times, same keys as the OpenFOAM summary where they apply
#   re<RE>_n<N>.cfg  the exact config that was run
#   log.run          stdout and stderr of the run

set -eo pipefail

N=${1:?usage: run_ldc.sh N RE}
RE=${2:?usage: run_ldc.sh N RE}
DIM=${DIM:-2}
THREADS=${THREADS:-6}
COURANT=${COURANT:-0.5}
if [ -z "${END_TIME}" ]; then
    if [ "$RE" -le 100 ]; then END_TIME=30; else END_TIME=60; fi
fi
P_TOL=${P_TOL:-1e-6}
P_MAXITER=${P_MAXITER:-10000}
GPU=${GPU:-1}
PROGRESS=${PROGRESS:-500}
WARM_START=${WARM_START:-1}
EIGEN_SOLVER=${EIGEN_SOLVER:-AMG}
EIGEN_PRECON=${EIGEN_PRECON:-DIAGONAL}
SUFFIX=${SUFFIX:-}
PROJECTION=${PROJECTION:-CHORIN}
FORM=${FORM:-ADVECTIVE}

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BIN=${BIN:-$ROOT/bin/nereus_headless}

TAG="re${RE}_n${N}"; [ "$DIM" = "3" ] && TAG="${TAG}_3d"
TAG="${TAG}${SUFFIX}"
OUT="$HERE/results/$TAG"
CFG="$OUT/${TAG}.cfg"

NU=$(awk -v re="$RE" 'BEGIN { printf "%.15g", 1.0 / re }')
DT=$(awk -v c="$COURANT" -v n="$N" 'BEGIN { printf "%.15g", c / n }')
STEPS_EXPECTED=$(awk -v t="$END_TIME" -v dt="$DT" 'BEGIN { printf "%d", (t / dt) + 0.5 }')

# ---- pre-flight: the two ways this wastes a night -------------------------------------------
if [ ! -x "$BIN" ]; then
    echo "No headless binary at $BIN" >&2
    echo "  build it with:  make -C $ROOT NVIDIA_GPU=1 HEADLESS=1 -j\$(nproc)" >&2
    exit 1
fi
if [ "$GPU" = "1" ] && ! ldd "$BIN" 2>/dev/null | grep -q libcudart; then
    # Without -DNVIDIA_GPU the reader forces GPU_ACCELERATION off and every solve silently falls
    # back to Eigen on the CPU -- the run would finish and measure the wrong thing. AMGX is linked
    # statically here, so it does not show in ldd; CUDA is dynamic and is the reliable marker (a
    # CPU-only build links no CUDA, MPI or AMGX at all).
    echo "$BIN links no CUDA, so it is a CPU-only build and GPU_ACCELERATION would be forced off." >&2
    echo "  rebuild with:   make -C $ROOT NVIDIA_GPU=1 HEADLESS=1 -j\$(nproc)" >&2
    echo "  or run CPU:     GPU=0 $0 $N $RE" >&2
    exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"

# ---- the config ------------------------------------------------------------------------------
cat > "$CFG" <<EOF
# NEREUS lid-driven cavity, written by Validation/NEREUS/run_ldc.sh to match the OpenFOAM
# case of the same name. Re $RE, GRID_SIZE $N, dt $DT fixed, to t = $END_TIME.

DIMENSION = $DIM
TYPE = ADI
CONVECTION_SCHEME = CENTRAL
PROJECTION_SCHEME = $PROJECTION
CONVECTION_FORM = $FORM
VORTICITY_CONFINEMENT = 0
GPU_ACCELERATION = $GPU
THREAD_COUNT = $THREADS

# icoFoam runs a fixed deltaT, so the step count is matched rather than the instantaneous CFL.
ADAPTATIVE_TIMESTEP = 0
MAX_CFL = $COURANT
TIME_STEP = $DT

DOMAIN_X0 = 0
DOMAIN_XF = 1
DOMAIN_Y0 = 0
DOMAIN_YF = 1
DOMAIN_Z0 = 0
DOMAIN_ZF = 1

# Uniform: the OpenFOAM mesh is uniform, and the shipped preset's refinement would compare two
# different discretisations.
NON_UNIFORM_GRID = 0
GRID_SIZE = $N
VISCOSITY = $NU
DENSITY = 1

Fu = 0
Fv = 0
Fw = 0

STOP_CONDITION = TIME
STOP_TIME = $END_TIME
STOP_RESIDUAL = 1e-06
STOP_EXPORT = 1

# EXPORT_GRID = 0 writes no intermediate frames; Exports::WriteFinal exports the grid
# unconditionally when the rule fires, which is the one field wanted here.
EXPORT_BASE_PATH = $OUT/raw
EXPORT_ENABLED = 1
EXPORT_TELEMETRY_INTERVAL = 1
EXPORT_GRID_INTERVAL = 100
EXPORT_FRAME_MODE = ITERATIONS
EXPORT_GRID = 0
EXPORT_PARTICLES = 0
EXPORT_SURFACE = 0
EXPORT_VOLUMETRIC = 0
EXPORT_AERODYNAMICS = 0
EXPORT_DAMBREAK = 0
EXPORT_COLUMN_TIME = 1
EXPORT_COLUMN_DIVERGENCE = 1
EXPORT_COLUMN_CFL = 1
EXPORT_COLUMN_RESIDUAL = 1
EXPORT_COLUMN_STEP_TIMINGS = 1
EXPORT_COLUMN_FLOW_INTEGRALS = 1
CHECKPOINT_AUTOSAVE = 0

REYNOLDS_NUMBER = $RE
CHARACTERISTIC_LENGTH = 1
MEAN_VELOCITY = 1

PRESSURE_BEGIN
# A closed box is pure Neumann, so the pressure system is singular and the mean must be removed.
NEEDS_COMPATIBILITY_CONDITION = 1
PRESSURE_WARM_START = $WARM_START
config_version=2
solver=PCG
max_iters=$P_MAXITER
norm=L2
convergence=ABSOLUTE
monitor_residual=1
relaxation_factor=1.0
tolerance=$P_TOL
preconditioner(amg_solver)=AMG
amg_solver:algorithm=CLASSICAL
amg_solver:selector=PMIS
amg_solver:max_iters=1
amg_solver:presweeps=1
amg_solver:postsweeps=1
amg_solver:cycle=V
print_solve_stats=0
print_grid_stats=0
obtain_timings=0
eigen_solver=$EIGEN_SOLVER
eigen_preconditioner=$EIGEN_PRECON
eigen_tolerance=$P_TOL
eigen_max_iters=$P_MAXITER
PRESSURE_END

SCENE_BEGIN
# Last match wins, so the lid is listed last and beats the side walls at the two top corners --
# the same corner treatment as OpenFOAM's movingWall patch.
sampling center
boundary xmin noslip
boundary xmax noslip
boundary ymin noslip
boundary zmin noslip
boundary zmax noslip
boundary ymax moving u="1.0" v="0"
SCENE_END
EOF

# ---- run -------------------------------------------------------------------------------------
echo "[$TAG] ${DIM}D, N $N, nu $NU, dt $DT, end $END_TIME (~$STEPS_EXPECTED steps), $( [ "$GPU" = 1 ] && echo GPU || echo CPU )"

cd "$OUT"
T0=$(date +%s.%N)
set +e
"$BIN" "$CFG" --progress "$PROGRESS" > "$OUT/log.run" 2>&1
RC=$?
set -e
T1=$(date +%s.%N)
cd "$HERE"

PROCESS=$(awk -v a="$T0" -v b="$T1" 'BEGIN { printf "%.3f", b - a }')

if [ "$RC" -ne 0 ]; then
    echo "[$TAG] FAILED, exit $RC -- see $OUT/log.run" >&2
    # 1 usage, 2 the field went non-finite, 3 interrupted.
    exit "$RC"
fi

# ---- collect, into the OpenFOAM side's layout -------------------------------------------------
RAW="$OUT/raw"
VTR=$(find "$RAW" -name 'grid-*.vtr' -print 2>/dev/null | sort | tail -1)
CSV=$(find "$RAW" -name 'telemetry.csv' -print 2>/dev/null | head -1)
[ -n "$VTR" ] && cp "$VTR" "$OUT/grid-final.vtr" || echo "  warning: no .vtr was written" >&2
[ -n "$CSV" ] && cp "$CSV" "$OUT/timings.csv"    || echo "  warning: no telemetry.csv was written" >&2

STEPS=$(awk '/^steps[[:space:]]/ { print $2; exit }' "$OUT/log.run")
STEPS=${STEPS:-0}
STEP_WALL=$(awk '/^wall[[:space:]]/ { gsub(",", "", $2); print $2; exit }' "$OUT/log.run")
STEP_WALL=${STEP_WALL:-nan}
PERSTEP=$(awk -v s="$STEP_WALL" -v n="$STEPS" \
    'BEGIN { if (n > 0 && s == s + 0) printf "%.6g", s / n; else print "nan" }')

cat > "$OUT/summary.txt" <<EOF
code                 NEREUS ADI (locally one-dimensional, one projection per step)
dimension            $DIM
grid_size            $N
fluid_cells          $( [ "$DIM" = 3 ] && echo "$((N-2))x$((N-2))x$((N-2))" || echo "$((N-2))x$((N-2))" )
cavity               [0,1]^$DIM with a one-cell solid ring; fluid on [1/$N, 1-1/$N]^$DIM, lid u = 1
reynolds_nominal     $RE
nu                   $NU
courant              $COURANT
dt                   $DT
adaptive_timestep    off (fixed dt, to match icoFoam)
end_time             $END_TIME
steps                $STEPS
steps_expected       $STEPS_EXPECTED
threads              $THREADS
device               $( [ "$GPU" = 1 ] && echo "GPU (NVIDIA AMGX)" || echo "CPU (Eigen)" )
p_solver             $( [ "$GPU" = 1 ] && echo "AMGX PCG + classical AMG (PMIS), convergence ABSOLUTE, tolerance $P_TOL" || echo "Eigen path: eigen_solver=$EIGEN_SOLVER ($EIGEN_PRECON), relative ||r||/||b|| < $P_TOL" )
warm_start           $WARM_START
projection           $PROJECTION
convection           central
convection_form      $FORM
solver_wall_s        $STEP_WALL
solver_wall_per_step_s $PERSTEP
process_wall_s       $PROCESS
note                 solver_wall_s is the stepping time the run reports itself, which is what
                     OpenFOAM's solver_wall_s measures too. process_wall_s adds AMGX start-up,
                     the grid build and the final export. Per-phase costs (momentum, pressure,
                     projection) are the <name>_timing columns of timings.csv.
note                 OpenFOAM's tolerance is on a normalised residual and AMGX ABSOLUTE is not,
                     so $P_TOL is the same requested number, not the same achieved accuracy.
EOF

echo "[$TAG] done: $STEPS steps, ${STEP_WALL}s stepping (${PERSTEP}s/step), ${PROCESS}s total -> $OUT"
