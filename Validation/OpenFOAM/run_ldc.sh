#!/usr/bin/env bash
# One OpenFOAM (icoFoam, PISO) lid-driven cavity run, laid out to match NEREUS's cavity preset.
#
# usage: ./run_ldc.sh N RE          (DIM=3 for the 3D cavity)
#   N   NEREUS GRID_SIZE (64 .. 1024). NEREUS's fluid is the N x N cells on [0,1]^2 (its one-cell solid ring
#       lies outside the domain since 2026-09-28). This mesh is exactly that region, so every OpenFOAM cell centre
#       coincides with a NEREUS interior cell centre and the two fields can be differenced directly.
#   RE  nu = 1/RE with lid speed 1, as in the preset (VISCOSITY = 1/RE, MEAN_VELOCITY = 1).
#
# Environment overrides (defaults in brackets):
#   DIM [2]    CORES [6]  COURANT [0.5]  END_TIME [30 for Re <= 100, 60 otherwise]
#
# DIM=3 is a genuine case change, not a flag: the block gains CELLS cells in z over the same
# extent, and frontAndBack stops being an `empty` pair and becomes no-slip walls. Everything else
# -- schemes, solvers, Courant number, end time -- is untouched, so the 2D and 3D cases differ
# only in the geometry, exactly as NEREUS's DIMENSION = 3 does.
#   P_TOL [1e-6]  P_RELTOL [0.05]  U_TOL [1e-5]
#   FOAM_BASHRC [/usr/lib/openfoam/openfoam2412/etc/bashrc]
#
# Writes to results/re<RE>_n<N>/:
#   grid-final.vtr   final field in NEREUS's VTK RectilinearGrid layout ((N+2)^2 [or ^3], ring
#                    included). 3D writes it only when foam_to_vtr3d.py is present.
#   timings.csv      per step: time, max Courant, pressure iterations/residuals, cumulative solver CPU time
#   summary.txt      settings, wall times (mesh + decomposition, solver), steps, seconds per step
#   log.*            OpenFOAM logs
# The case directory itself is deleted at the end to save disk.

set -eo pipefail

N=${1:?usage: run_ldc.sh N RE}
RE=${2:?usage: run_ldc.sh N RE}
DIM=${DIM:-2}
CORES=${CORES:-6}
COURANT=${COURANT:-0.5}
if [ -z "${END_TIME}" ]; then
    if [ "$RE" -le 100 ]; then END_TIME=30; else END_TIME=60; fi
fi
P_TOL=${P_TOL:-1e-6}
P_RELTOL=${P_RELTOL:-0.05}
U_TOL=${U_TOL:-1e-5}
FOAM_BASHRC=${FOAM_BASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

HERE=$(cd "$(dirname "$0")" && pwd)
TAG="re${RE}_n${N}"; [ "$DIM" = "3" ] && TAG="${TAG}_3d"
OUT="$HERE/results/$TAG"
CASE="$OUT/case"

CELLS=$N
H=$(awk -v n="$N" 'BEGIN { printf "%.15g", 1.0 / n }')
X0=0
X1=1
if [ "$DIM" = "3" ]; then
    Z0=$X0; Z1=$X1; ZCELLS=$CELLS
    FB_TYPE=wall; FB_U="type noSlip;"; FB_P="type zeroGradient;"
else
    Z0=0;   Z1=$H;  ZCELLS=1
    FB_TYPE=empty; FB_U="type empty;"; FB_P="type empty;"
fi
NU=$(awk -v re="$RE" 'BEGIN { printf "%.15g", 1.0 / re }')
DT=$(awk -v c="$COURANT" -v h="$H" 'BEGIN { printf "%.15g", c * h }')

rm -rf "$OUT"
mkdir -p "$CASE/0" "$CASE/constant" "$CASE/system"

header() {
    cat <<EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       $1;
    object      $2;
}
EOF
}

{ header dictionary blockMeshDict; cat <<EOF
scale 1;

vertices
(
    ($X0 $X0 $Z0)
    ($X1 $X0 $Z0)
    ($X1 $X1 $Z0)
    ($X0 $X1 $Z0)
    ($X0 $X0 $Z1)
    ($X1 $X0 $Z1)
    ($X1 $X1 $Z1)
    ($X0 $X1 $Z1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ($CELLS $CELLS $ZCELLS) simpleGrading (1 1 1)
);

edges ();

boundary
(
    movingWall   { type wall;  faces ((3 7 6 2)); }
    fixedWalls   { type wall;  faces ((0 4 7 3) (2 6 5 1) (1 5 4 0)); }
    frontAndBack { type $FB_TYPE; faces ((0 3 2 1) (4 5 6 7)); }
);
EOF
} > "$CASE/system/blockMeshDict"

{ header dictionary controlDict; cat <<EOF
application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         $END_TIME;
deltaT          $DT;
writeControl    runTime;
writeInterval   $END_TIME;
purgeWrite      0;
writeFormat     ascii;
writePrecision  12;
writeCompression off;
timeFormat      general;
timePrecision   10;
runTimeModifiable false;
EOF
} > "$CASE/system/controlDict"

{ header dictionary fvSchemes; cat <<EOF
ddtSchemes           { default Euler; }
gradSchemes          { default Gauss linear; grad(p) Gauss linear; }
divSchemes           { default none; div(phi,U) Gauss linear; }
laplacianSchemes     { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes        { default orthogonal; }
EOF
} > "$CASE/system/fvSchemes"

{ header dictionary fvSolution; cat <<EOF
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       $P_TOL;
        relTol          $P_RELTOL;
    }
    pFinal
    {
        \$p;
        relTol          0;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       $U_TOL;
        relTol          0;
    }
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
EOF
} > "$CASE/system/fvSolution"

{ header dictionary decomposeParDict; cat <<EOF
numberOfSubdomains $CORES;
method          scotch;
EOF
} > "$CASE/system/decomposeParDict"

{ header dictionary transportProperties; cat <<EOF
nu              $NU;
EOF
} > "$CASE/constant/transportProperties"

{ header volVectorField U; cat <<EOF
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    movingWall   { type fixedValue; value uniform (1 0 0); }
    fixedWalls   { type noSlip; }
    frontAndBack { $FB_U }
}
EOF
} > "$CASE/0/U"

{ header volScalarField p; cat <<EOF
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    movingWall   { type zeroGradient; }
    fixedWalls   { type zeroGradient; }
    frontAndBack { $FB_P }
}
EOF
} > "$CASE/0/p"

# The OpenFOAM environment script is not written for `set -e`.
set +e
# shellcheck disable=SC1090
source "$FOAM_BASHRC"
set -e

now() { date +%s.%N; }
cd "$CASE"

echo "[$TAG] ${DIM}D mesh ${CELLS}x${CELLS}$( [ "$DIM" = 3 ] && echo "x${CELLS}" ), nu $NU, dt $DT, end $END_TIME, $CORES cores"
T0=$(now)
blockMesh > "$OUT/log.blockMesh" 2>&1
if [ "$CORES" -gt 1 ]; then decomposePar > "$OUT/log.decomposePar" 2>&1; fi
T1=$(now)
if [ "$CORES" -gt 1 ]; then
    mpirun --bind-to core -np "$CORES" icoFoam -parallel > "$OUT/log.icoFoam" 2>&1
else
    icoFoam > "$OUT/log.icoFoam" 2>&1
fi
T2=$(now)
if [ "$CORES" -gt 1 ]; then reconstructPar -latestTime > "$OUT/log.reconstructPar" 2>&1; fi
postProcess -func writeCellCentres -latestTime > "$OUT/log.writeCellCentres" 2>&1

if [ "$DIM" = "3" ]; then
    if [ -f "$HERE/foam_to_vtr3d.py" ]; then
        python3 "$HERE/foam_to_vtr3d.py" "$CASE" "$N" "$OUT/grid-final.vtr"
    else
        echo "  no foam_to_vtr3d.py: the field was left in $CASE (timings are still written)" >&2
        KEEP_CASE=1
    fi
else
    python3 "$HERE/foam_to_vtr.py" "$CASE" "$N" "$OUT/grid-final.vtr"
fi
python3 "$HERE/foam_log_timings.py" "$OUT/log.icoFoam" "$OUT/timings.csv" > "$OUT/steps.txt"

STEPS=$(cat "$OUT/steps.txt"); rm -f "$OUT/steps.txt"
PRE=$(awk -v a="$T0" -v b="$T1" 'BEGIN { printf "%.3f", b - a }')
SOLVE=$(awk -v a="$T1" -v b="$T2" 'BEGIN { printf "%.3f", b - a }')
PERSTEP=$(awk -v s="$SOLVE" -v n="$STEPS" 'BEGIN { if (n > 0) printf "%.6g", s / n; else print "nan" }')

cat > "$OUT/summary.txt" <<EOF
code                 OpenFOAM v2412 icoFoam (PISO, 2 correctors)
dimension            $DIM
grid_size            $N
fluid_cells          ${CELLS}x${CELLS}$( [ "$DIM" = 3 ] && echo "x${CELLS}" )
cavity               [$X0, $X1]^$DIM, lid u = 1
reynolds_nominal     $RE
nu                   $NU
courant              $COURANT
dt                   $DT
end_time             $END_TIME
steps                $STEPS
cores                $CORES
p_solver             GAMG GaussSeidel, tolerance $P_TOL, relTol $P_RELTOL (pFinal relTol 0)
u_solver             smoothSolver symGaussSeidel, tolerance $U_TOL
mesh_decompose_wall_s $PRE
solver_wall_s        $SOLVE
solver_wall_per_step_s $PERSTEP
EOF

cd "$HERE"
[ -z "$KEEP_CASE" ] && rm -rf "$CASE"
echo "[$TAG] done: $STEPS steps, solver ${SOLVE}s (${PERSTEP}s/step)"
