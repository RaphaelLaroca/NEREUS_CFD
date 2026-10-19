// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The flip suite: flip2d_dambreak_check, flip3d_dambreak_check, flip2d_hydrostatic_check, flip3d_hydrostatic_check, flip2d_obstacle_check, flip3d_obstacle_check, flip_solidflux_check, flip_density_check, diffusion2d_check, diffusion3d_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "Functions.h"
#include "Methods/FLIP/FLIP2D.h"
#include "Methods/PPESolver/PressureSolver2D.h"
#include "Methods/FLIP/DiffusionSolver2D.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include "Methods/FLIP/FLIP3D.h"
#include "Methods/PPESolver/PressureSolver3D.h"
#include "Methods/FLIP/DiffusionSolver3D.h"
#include <fstream>
#include "headers/Core/Scene.h"
#include "headers/Core/Particles.h"
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// flip2d_dambreak_check
// ------------------------------------------------------------------------------------------------------------------
namespace flip2d_dambreak_check
{
// Headless dam break. Drives the FLIP pipeline the same way SimulationManager does,
// with the CPU pressure solver so no GUI or GPU solve is involved, and reports the
// things that go wrong when the surface handling is off: particles leaving the domain,
// NaNs, leftover divergence, and the gap between the lowest particle and the floor.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail)
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok)
        failures++;
}

struct Result
{
    bool finite = true;
    bool inDomain = true;
    double maxDiv = 0.0;
    double floorGapCells = 1e9; // lowest particle, in cells above the wall surface
    double maxSpeed = 0.0;
};

static Result RunDamBreak(bool physicalDiffusion, DIFFUSION_SOLVERS solver, int steps)
{
    const int N = 48;

    DIMENSION = 2;
    SelectMethod("FLIP");
    THREAD_COUNT = 1;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 0.6, 0.0, 0.6, 0.0, 1.0};
    SIMULATION.GRID_SIZE = N;
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-2;   // deliberately viscous: this is the regime that misbehaved
    SIMULATION.dt = 5e-4;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = 4;
    FLIP_SETTINGS.alpha = 0.95;
    FLIP_SETTINGS.physicalDiffusion = physicalDiffusion;
    FLIP_SETTINGS.diffusionSolver = solver;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    SIMULATION.VelocityBoundaryFunction2D = DAMBREAK_BORDER_2D;
    SIMULATION.PressureBoundaryFunction2D = DAMBREAK_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = DAMBREAK_SOLID_MASK_2D;

    static MAC gridSol, gridAnt;
    gridSol = MAC();
    gridAnt = MAC();
    gridSol.InitializeGrid(SIMULATION.domain, true);
    gridAnt.InitializeGrid(SIMULATION.domain, true);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    gridSol.SetSolidMask(DAMBREAK_SOLID_MASK_2D);
    gridAnt.SetSolidMask(DAMBREAK_SOLID_MASK_2D);

    PressureSolver2D::InitializePressureSolver(&gridSol);
    LoadReservoirScene(2, gridSol.dh);
    FLIP2D::InitializeFLIP(&gridSol, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    gridAnt.EnforceBoundaryConditions(0);

    Result r;
    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();

        FLIP2D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, step * SIMULATION.dt);
        PressureSolver2D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver2D::Project(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        FLIP2D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);

        gridAnt.CopyGrid(gridSol);
    }

    // Post-run inspection
    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx; j++)
            if (gridSol.GetSolid(i, j) == FLUID_CELL)
                r.maxDiv = std::max(r.maxDiv, std::abs(gridSol.GetDivergencyAt(i, j)));

    double wall = SIMULATION.domain.y0;   // the floor is the domain face: the border ring lies outside it
    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        const auto &pt = SIMULATION.particles.data[p];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
            !std::isfinite(pt.u) || !std::isfinite(pt.v))
        {
            r.finite = false;
            continue;
        }
        if (pt.x < SIMULATION.domain.x0 || pt.x > SIMULATION.domain.xf ||
            pt.y < SIMULATION.domain.y0 || pt.y > SIMULATION.domain.yf)
            r.inDomain = false;

        r.floorGapCells = std::min(r.floorGapCells, (pt.y - wall) / SIMULATION.dh);
        r.maxSpeed = std::max(r.maxSpeed, std::sqrt(pt.u * pt.u + pt.v * pt.v));
    }

#ifndef NO_SOLVER_TEARDOWN
    PressureSolver2D::DestroyPressureSolver();
#endif
    return r;
}

// Optional argv[1] selects a single case, so the harness can also be pointed at an
// older build of the solvers where re-initialising in one process is not safe.
int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int only = (argc > 1) ? std::atoi(argv[1]) : -1;
    struct Case { const char *name; bool diff; DIFFUSION_SOLVERS solver; };
    Case cases[] = {
        {"no physical diffusion", false, DIFFUSION_SOLVERS::ADI},
        {"physical diffusion, CG", true, DIFFUSION_SOLVERS::IMPLICIT},
        {"physical diffusion, ADI", true, DIFFUSION_SOLVERS::ADI},
    };

    for (int ci = 0; ci < 3; ci++)
    {
        if (only >= 0 && ci != only)
            continue;
        const Case &c = cases[ci];
        std::printf("\n== %s ==\n", c.name);
        Result r = RunDamBreak(c.diff, c.solver, 400);
        std::printf("     lowest particle sits %.3f cells above the wall\n", r.floorGapCells);
        std::printf("     max |div| in fluid = %.3e, max particle speed = %.3f\n", r.maxDiv, r.maxSpeed);

        check(r.finite, "particles stay finite", "");
        check(r.inDomain, "particles stay inside the domain", "");
        check(r.maxDiv < 1e-6, "projected field is divergence free",
              "max|div| = " + std::to_string(r.maxDiv));
        // The clamp puts a settled particle exactly on the wall. A persistent gap of a
        // whole cell is the hovering artifact.
        check(r.floorGapCells < 0.5, "fluid actually reaches the floor",
              "gap = " + std::to_string(r.floorGapCells) + " cells");
        check(r.maxSpeed < 50.0, "no runaway velocities",
              "max speed = " + std::to_string(r.maxSpeed));
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace flip2d_dambreak_check

// ------------------------------------------------------------------------------------------------------------------
// flip3d_dambreak_check
// ------------------------------------------------------------------------------------------------------------------
namespace flip3d_dambreak_check
{
// Headless 3D dam break - the 3D twin of tests/flip2d_dambreak_check.cpp. Also the direct
// check that the body force exists at all: FLIP3D had AddAcceleration commented out, so
// unless the ADI diffusion was selected (which smuggled gravity into one sweep's RHS)
// nothing ever fell.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail)
{
    std::printf("     [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}

static void Run(const char *label, bool physicalDiffusion, DIFFUSION_SOLVERS solver, int steps,
                double density = 1.0)
{
    DIMENSION = 3;
    SelectMethod("FLIP");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 0.6, 0.0, 0.3, 0.0, 0.3};
    SIMULATION.GRID_SIZE = 30;              // 60x30x30, inside the 60^3 ceiling
    SIMULATION.RHO = density;
    SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.002;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = 8;
    FLIP_SETTINGS.alpha = 0.95;
    FLIP_SETTINGS.physicalDiffusion = physicalDiffusion;
    FLIP_SETTINGS.diffusionSolver = solver;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    SIMULATION.VelocityBoundaryFunction = DAMBREAK_BORDER;
    SIMULATION.PressureBoundaryFunction = DAMBREAK_PRESSURE;
    SIMULATION.SolidMaskFunction = DAMBREAK_SOLID_MASK;

    static MAC gridSol, gridAnt;
    gridSol.InitializeGrid(SIMULATION.domain, false);
    gridAnt.InitializeGrid(SIMULATION.domain, false);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.Nz = gridSol.Nz;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    gridSol.SetSolidMask(DAMBREAK_SOLID_MASK);
    gridAnt.SetSolidMask(DAMBREAK_SOLID_MASK);

    PressureSolver3D::InitializePressureSolver(&gridSol);
    if (physicalDiffusion) DiffusionSolver3D::InitializeDiffusionSolver(&gridSol);
    LoadReservoirScene(3, gridSol.dh);
    FLIP3D::InitializeFLIP(&gridSol, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    gridAnt.EnforceBoundaryConditions(0);

    double tParticles = 0.0, tPressure = 0.0, tDiffusion = 0.0, tTotal = 0.0;
    for (int step = 0; step < steps; step++)
    {
        double t0 = GetWallTime();
        gridSol.ClearFlowFields();
        FLIP3D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, step * SIMULATION.dt);
        const double t1 = GetWallTime();
        PressureSolver3D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        const double t2 = GetWallTime();
        PressureSolver3D::Project(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        FLIP3D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        const double t3 = GetWallTime();
        gridAnt.CopyGrid(gridSol);
        tTotal += GetWallTime() - t0;
        tDiffusion += t1 - t0;
        tPressure += t2 - t1;
        tParticles += t3 - t2;
    }
    std::printf("     time: total %.2fs | momentum %.2fs | pressure %.2fs | projection %.2fs | other %.2fs\n",
                tTotal, tDiffusion, tPressure, tParticles,
                tTotal - tParticles - tPressure - tDiffusion);

    double maxDiv = 0.0;
    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx; j++)
            for (int k = 0; k < gridSol.Nz; k++)
                if (gridSol.GetSolid(i, j, k) == FLUID_CELL)
                    maxDiv = std::max(maxDiv, std::abs(gridSol.GetDivergencyAt(i, j, k)));

    bool finite = true, inDomain = true;
    double maxSpeed = 0.0, meanV = 0.0, frontX = 0.0;
    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        const auto &q = SIMULATION.particles.data[p];
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) ||
            !std::isfinite(q.u) || !std::isfinite(q.v) || !std::isfinite(q.w))
        { finite = false; continue; }
        if (q.x < SIMULATION.domain.x0 || q.x > SIMULATION.domain.xf ||
            q.y < SIMULATION.domain.y0 || q.y > SIMULATION.domain.yf ||
            q.z < SIMULATION.domain.z0 || q.z > SIMULATION.domain.zf) inDomain = false;
        maxSpeed = std::max(maxSpeed, std::sqrt(q.u * q.u + q.v * q.v + q.w * q.w));
        meanV += q.v;
        frontX = std::max(frontX, q.x);
    }
    meanV /= SIMULATION.particles.count;

    std::printf("  %-26s rho=%-7.0f particles=%d  maxSpeed=%.4f  meanV=%+.4f  front x=%.4f  max|div|=%.2e\n",
                label, density, SIMULATION.particles.count, maxSpeed, meanV, frontX, maxDiv);
    check(finite, "particles stay finite", "");
    check(inDomain, "particles stay inside the domain", "");
    check(maxDiv < 1e-6, "projected field is divergence free", "max|div| = " + std::to_string(maxDiv));
    // Gravity has to actually do something: a released column must be moving.
    check(maxSpeed > 0.05, "the body force is applied (fluid is moving)",
          "maxSpeed = " + std::to_string(maxSpeed));
    // The projection must divide by density. Without it the velocity is over-corrected by
    // a factor of rho every step and the run diverges outright.
    check(maxSpeed < 20.0, "no runaway velocities", "maxSpeed = " + std::to_string(maxSpeed));

#ifndef NO_SOLVER_TEARDOWN
    PressureSolver3D::DestroyPressureSolver();
#endif
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int steps = (argc > 1) ? std::atoi(argv[1]) : 60;
    int only = (argc > 2) ? std::atoi(argv[2]) : -1;
    std::printf("3D dam break, %d steps\n", steps);
    if (only < 0 || only == 0) Run("no physical diffusion", false, DIFFUSION_SOLVERS::ADI, steps);
    if (only < 0 || only == 1) Run("physical diffusion, CG", true, DIFFUSION_SOLVERS::IMPLICIT, steps);
    if (only < 0 || only == 2) Run("physical diffusion, ADI", true, DIFFUSION_SOLVERS::ADI, steps);
    // Density must not change stability, only the scale of the pressure field.
    if (only < 0 || only == 3) Run("water density, ADI", true, DIFFUSION_SOLVERS::ADI, steps, 1000.0);
    if (only < 0 || only == 4) Run("water density, CG", true, DIFFUSION_SOLVERS::IMPLICIT, steps, 1000.0);
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace flip3d_dambreak_check

// ------------------------------------------------------------------------------------------------------------------
// flip2d_hydrostatic_check
// ------------------------------------------------------------------------------------------------------------------
namespace flip2d_hydrostatic_check
{
// Hydrostatic tank. A block of water at rest under gravity in a closed box: after it
// settles every column should read ~0 vertical velocity, and the field must be symmetric
// left-to-right. Any systematic upward velocity in the columns next to the side walls is
// the artifact, with no dam-break dynamics to hide it.


static void SeedFullTank()
{
    double dh = SIMULATION.dh;
    int n = (int)std::sqrt((double)FLIP_SETTINGS.particlesPerCell);
    std::vector<double> off(n);
    for (int k = 0; k < n; k++)
        off[k] = (k + 0.5) / (double)n;

    SIMULATION.particles.count = 0;
    // Fill the lower half of the interior, wall to wall.
    for (int i = 1; i < SIMULATION.Ny / 2; i++)
        for (int j = 1; j < SIMULATION.Nx - 1; j++)
            for (int io = 0; io < n; io++)
                for (int jo = 0; jo < n; jo++)
                {
                    Particle p;
                    p.x = SIMULATION.spacing.xf[0] + j * dh + off[jo] * dh;
                    p.y = SIMULATION.spacing.yf[0] + i * dh + off[io] * dh;
                    p.u = 0.0; p.v = 0.0; p.isActive = true;
                    SIMULATION.particles.data[SIMULATION.particles.count++] = p;
                }
}

static int failures = 0;

static void Report(const char *label, bool physicalDiffusion, DIFFUSION_SOLVERS solver, int steps)
{
    DIMENSION = 2;
    SelectMethod("FLIP");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 0.5, 0.0, 0.5, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 50;
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.005;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = 4;
    FLIP_SETTINGS.alpha = 0.98;
    FLIP_SETTINGS.physicalDiffusion = physicalDiffusion;
    FLIP_SETTINGS.diffusionSolver = solver;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    SIMULATION.VelocityBoundaryFunction2D = DAMBREAK_BORDER_2D;
    SIMULATION.PressureBoundaryFunction2D = DAMBREAK_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = DAMBREAK_SOLID_MASK_2D;

    static MAC gridSol, gridAnt;
    gridSol.InitializeGrid(SIMULATION.domain, true);
    gridAnt.InitializeGrid(SIMULATION.domain, true);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    gridSol.SetSolidMask(DAMBREAK_SOLID_MASK_2D);
    gridAnt.SetSolidMask(DAMBREAK_SOLID_MASK_2D);

    PressureSolver2D::InitializePressureSolver(&gridSol);
    FLIP2D::InitializeFLIP(&gridSol, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    SeedFullTank();
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    gridAnt.EnforceBoundaryConditions(0);

    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        FLIP2D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, step * SIMULATION.dt);
        PressureSolver2D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver2D::Project(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        FLIP2D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        gridAnt.CopyGrid(gridSol);
    }

    // Mean particle v per column band.
    int Nx = SIMULATION.Nx;
    double dh = SIMULATION.dh;
    std::vector<double> colV(Nx, 0.0);
    std::vector<int> colN(Nx, 0);
    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        int j = (int)((SIMULATION.particles.data[p].x - SIMULATION.domain.x0 + dh) / dh);
        if (j < 0 || j >= Nx) continue;
        colV[j] += SIMULATION.particles.data[p].v;
        colN[j]++;
    }
    for (int j = 0; j < Nx; j++)
        if (colN[j]) colV[j] /= colN[j];

    std::printf("  %s\n", label);
    std::printf("     mean particle v by column:  ");
    for (int j : {1, 2, 3, 4, Nx / 2, Nx - 5, Nx - 4, Nx - 3, Nx - 2})
        std::printf("j=%d:%+.4f ", j, colV[j]);
    std::printf("\n");

    // Grid-level view: the V faces in the columns adjacent to each side wall.
    double leftFace = 0.0, rightFace = 0.0, midFace = 0.0;
    int ln = 0, rn = 0, mn = 0;
    for (int i = 2; i < SIMULATION.Ny - 1; i++)
    {
        if (gridSol.GetSolid(i, 1) == FLUID_CELL)      { leftFace  += gridSol.GetV(i, 1);      ln++; }
        if (gridSol.GetSolid(i, Nx - 2) == FLUID_CELL) { rightFace += gridSol.GetV(i, Nx - 2); rn++; }
        if (gridSol.GetSolid(i, Nx / 2) == FLUID_CELL) { midFace   += gridSol.GetV(i, Nx / 2); mn++; }
    }
    double lv = ln ? leftFace / ln : 0.0;
    double rv = rn ? rightFace / rn : 0.0;
    double mv = mn ? midFace / mn : 0.0;
    std::printf("     grid V in fluid:  left col=%+.5f   interior col=%+.5f   right col=%+.5f\n", lv, mv, rv);

    // Nothing is moving, so every column must read ~0. A body force that is allowed
    // through the implicit viscous solve leaves an upward residue in the wall columns
    // that FLIP accumulates; that showed up here as ~+0.026 against both side walls.
    // Faces deep inside the air above the water, past the reach of the extrapolation
    // sweep. Nothing should be moving there: the body force belongs on faces that carry
    // fluid. A uniform -g*dt draped over the whole air region is the artifact.
    const int R = FLIP2D::EXTRAPOLATION_LAYERS + 2;
    double maxAirV = 0.0;
    int airFaces = 0;
    for (int i = 2; i < SIMULATION.Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            bool nearNonAir = false;
            for (int di = -R; di <= R && !nearNonAir; di++)
                for (int dj = -R; dj <= R && !nearNonAir; dj++)
                {
                    int a = i + di, b = j + dj;
                    if (a < 0 || a >= SIMULATION.Ny || b < 0 || b >= Nx) continue;
                    if (gridSol.GetSolid(a, b) != EMPTY_CELL) nearNonAir = true;
                }
            if (nearNonAir) continue;
            maxAirV = std::max(maxAirV, std::abs(gridSol.GetV(i, j)));
            airFaces++;
        }
    std::printf("     deep-air V faces: %d, max|v| there = %.6f  (g*dt = %.6f)\n",
                airFaces, maxAirV, 9.8 * SIMULATION.dt);

    const double TOL = 1e-3;
    bool ok = true;
    for (int j = 1; j < Nx - 1; j++)
        if (colN[j] && std::abs(colV[j]) > TOL) ok = false;
    if (std::abs(lv) > TOL || std::abs(rv) > TOL || std::abs(mv) > TOL) ok = false;
    std::printf("     [%s] tank at rest stays at rest (max column |v| tolerance %.0e)\n",
                ok ? "PASS" : "FAIL", TOL);
    if (!ok) failures++;

    bool airOk = (airFaces > 0) && (maxAirV < 1e-9);
    std::printf("     [%s] no body force on faces with no fluid (%d faces sampled)\n",
                airOk ? "PASS" : "FAIL", airFaces);
    if (!airOk) failures++;

#ifndef NO_SOLVER_TEARDOWN
    PressureSolver2D::DestroyPressureSolver();
#endif
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int steps = (argc > 1) ? std::atoi(argv[1]) : 100;
    int only = (argc > 2) ? std::atoi(argv[2]) : -1;
    std::printf("hydrostatic tank at rest, %d steps (everything should settle to ~0, symmetric)\n", steps);
    if (only < 0 || only == 0) Report("no physical diffusion", false, DIFFUSION_SOLVERS::ADI, steps);
    if (only < 0 || only == 1) Report("physical diffusion, CG", true, DIFFUSION_SOLVERS::IMPLICIT, steps);
    if (only < 0 || only == 2) Report("physical diffusion, ADI", true, DIFFUSION_SOLVERS::ADI, steps);
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace flip2d_hydrostatic_check

// ------------------------------------------------------------------------------------------------------------------
// flip3d_hydrostatic_check
// ------------------------------------------------------------------------------------------------------------------
namespace flip3d_hydrostatic_check
{
// 3D hydrostatic tank - the 3D twin of tests/flip2d_hydrostatic_check.cpp.
//
// A block of water at rest under gravity in a closed box. After it settles every column
// must read ~0 vertical velocity and the field must be symmetric across the domain. Any
// systematic upward velocity in the columns next to the side walls is the artifact that
// a body force smuggled through the implicit viscous solve produces.


static int failures = 0;

static void SeedFullTank()
{
    double dh = SIMULATION.dh;
    int n = (int)std::cbrt((double)FLIP_SETTINGS.particlesPerCell);
    if (n < 1) n = 1;
    std::vector<double> off(n);
    for (int t = 0; t < n; t++) off[t] = (t + 0.5) / (double)n;

    SIMULATION.particles.count = 0;
    for (int i = 1; i < SIMULATION.Ny / 2; i++)
        for (int j = 1; j < SIMULATION.Nx - 1; j++)
            for (int k = 1; k < SIMULATION.Nz - 1; k++)
                for (int io = 0; io < n; io++)
                    for (int jo = 0; jo < n; jo++)
                        for (int ko = 0; ko < n; ko++)
                        {
                            Particle p;
                            p.x = SIMULATION.spacing.xf[0] + j * dh + off[jo] * dh;
                            p.y = SIMULATION.spacing.yf[0] + i * dh + off[io] * dh;
                            p.z = SIMULATION.spacing.zf[0] + k * dh + off[ko] * dh;
                            p.u = p.v = p.w = 0.0;
                            p.isActive = true;
                            SIMULATION.particles.data[SIMULATION.particles.count++] = p;
                        }
}

static void Report(const char *label, bool physicalDiffusion, DIFFUSION_SOLVERS solver, int steps)
{
    DIMENSION = 3;
    SelectMethod("FLIP");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 0.3, 0.0, 0.3, 0.0, 0.3};
    SIMULATION.GRID_SIZE = 24;              // 24^3, inside the 60^3 ceiling
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.004;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = 8;
    FLIP_SETTINGS.alpha = 0.95;
    FLIP_SETTINGS.physicalDiffusion = physicalDiffusion;
    FLIP_SETTINGS.diffusionSolver = solver;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    SIMULATION.VelocityBoundaryFunction = DAMBREAK_BORDER;
    SIMULATION.PressureBoundaryFunction = DAMBREAK_PRESSURE;
    SIMULATION.SolidMaskFunction = DAMBREAK_SOLID_MASK;

    static MAC gridSol, gridAnt;
    gridSol.InitializeGrid(SIMULATION.domain, false);
    gridAnt.InitializeGrid(SIMULATION.domain, false);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.Nz = gridSol.Nz;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    gridSol.SetSolidMask(DAMBREAK_SOLID_MASK);
    gridAnt.SetSolidMask(DAMBREAK_SOLID_MASK);

    PressureSolver3D::InitializePressureSolver(&gridSol);
    if (physicalDiffusion) DiffusionSolver3D::InitializeDiffusionSolver(&gridSol);
    FLIP3D::InitializeFLIP(&gridSol, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    SeedFullTank();
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    gridAnt.EnforceBoundaryConditions(0);

    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        FLIP3D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, step * SIMULATION.dt);
        PressureSolver3D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver3D::Project(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        FLIP3D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        gridAnt.CopyGrid(gridSol);
    }

    int Nx = SIMULATION.Nx;
    double dh = SIMULATION.dh;
    std::vector<double> colV(Nx, 0.0);
    std::vector<int> colN(Nx, 0);
    double meanAll = 0.0;
    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        int j = (int)((SIMULATION.particles.data[p].x - SIMULATION.domain.x0 + dh) / dh);
        if (j < 0 || j >= Nx) continue;
        colV[j] += SIMULATION.particles.data[p].v;
        colN[j]++;
        meanAll += SIMULATION.particles.data[p].v;
    }
    for (int j = 0; j < Nx; j++) if (colN[j]) colV[j] /= colN[j];
    meanAll /= SIMULATION.particles.count;

    std::printf("  %s\n", label);
    std::printf("     mean particle v by x-column:  ");
    for (int j : {1, 2, 3, Nx / 2, Nx - 4, Nx - 3, Nx - 2})
        std::printf("j=%d:%+.4f ", j, colV[j]);
    std::printf("\n     mean v over all particles: %+.6f\n", meanAll);

    // Faces deep inside the air above the water, far enough out that the extrapolation
    // sweep cannot reach them. Nothing should be moving there: the body force belongs on
    // faces that carry fluid, and the band just outside the surface belongs to the
    // extrapolation. A uniform -g*dt sitting across the whole air region is the artifact.
    const int R = FLIP3D::EXTRAPOLATION_LAYERS + 2;
    double maxAirV = 0.0;
    int airFaces = 0;
    for (int i = 2; i < SIMULATION.Ny - 1; i++)
        for (int j = 1; j < SIMULATION.Nx - 1; j++)
            for (int k = 1; k < SIMULATION.Nz - 1; k++)
            {
                bool nearNonAir = false;
                for (int di = -R; di <= R && !nearNonAir; di++)
                    for (int dj = -R; dj <= R && !nearNonAir; dj++)
                        for (int dk = -R; dk <= R && !nearNonAir; dk++)
                        {
                            int a = i + di, b = j + dj, c = k + dk;
                            if (a < 0 || a >= SIMULATION.Ny || b < 0 || b >= SIMULATION.Nx ||
                                c < 0 || c >= SIMULATION.Nz) continue;
                            if (gridSol.GetSolid(a, b, c) != EMPTY_CELL) nearNonAir = true;
                        }
                if (nearNonAir) continue;
                maxAirV = std::max(maxAirV, std::abs(gridSol.GetV(i, j, k)));
                airFaces++;
            }
    std::printf("     deep-air V faces: %d, max|v| there = %.6f  (g*dt = %.6f)\n",
                airFaces, maxAirV, 9.8 * SIMULATION.dt);

    const double TOL = 2e-3;
    bool ok = std::abs(meanAll) <= TOL;
    for (int j = 1; j < Nx - 1; j++)
        if (colN[j] && std::abs(colV[j]) > TOL) ok = false;
    std::printf("     [%s] tank at rest stays at rest (tolerance %.0e)\n", ok ? "PASS" : "FAIL", TOL);
    if (!ok) failures++;

    bool airOk = (airFaces > 0) && (maxAirV < 1e-9);
    std::printf("     [%s] no body force on faces with no fluid (%d faces sampled)\n",
                airOk ? "PASS" : "FAIL", airFaces);
    if (!airOk) failures++;

#ifndef NO_SOLVER_TEARDOWN
    PressureSolver3D::DestroyPressureSolver();
#endif
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int steps = (argc > 1) ? std::atoi(argv[1]) : 60;
    int only = (argc > 2) ? std::atoi(argv[2]) : -1;
    std::printf("3D hydrostatic tank at rest, %d steps (everything should settle to ~0)\n", steps);
    if (only < 0 || only == 0) Report("no physical diffusion", false, DIFFUSION_SOLVERS::ADI, steps);
    if (only < 0 || only == 1) Report("physical diffusion, CG", true, DIFFUSION_SOLVERS::IMPLICIT, steps);
    if (only < 0 || only == 2) Report("physical diffusion, ADI", true, DIFFUSION_SOLVERS::ADI, steps);
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace flip3d_hydrostatic_check

// ------------------------------------------------------------------------------------------------------------------
// flip2d_obstacle_check
// ------------------------------------------------------------------------------------------------------------------
namespace flip2d_obstacle_check
{
// Particles versus interior solid geometry.
//
// Advection interpolates a velocity field that knows nothing about obstacles, so nothing
// in the pipeline stops a particle entering one; before Core/Particles.h carried a contact test the
// only thing a particle was ever tested against was the domain box. This drives the same
// FLIP pipeline SimulationManager does, with a cylinder standing in the path of a
// collapsing column, and asserts that no particle ends up inside it -- by the voxel mask,
// and within the voxelisation error by the analytic surface -- while still sliding past
// rather than sticking to it.//
// Note the contract these assert. The **voxel mask** is what decides where a particle may
// be, because it is what the pressure solve sees; the analytic surface only supplies the
// push direction. The two can be up to a cell apart (with `sampling corner` a whole cell
// on the low-index faces), so the hard invariant is "no particle in a SOLID_CELL" and the
// analytic penetration is merely *bounded* by the voxelisation error. Asserting zero
// analytic penetration instead is what produced the -x/-y/-z conveyor: it made the
// mismatch shell a trap the grid drove particles into and the collision evicted them from,
// every step.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok)
        failures++;
}

// The obstacle, in domain-local metres. Kept here so the assertions and the scene file
// cannot drift apart.
static const double CYL_X = 0.34, CYL_Y = 0.10, CYL_R = 0.07;

static void WriteScene(const char *path)
{
    std::ofstream f(path);
    f << "sampling center\n"
      << "boundary xmin noslip\n"
      << "boundary xmax noslip\n"
      << "boundary ymin noslip\n"
      << "boundary ymax noslip\n"
      << "region fluid box min=(0,0,0) max=(0.20,0.40,1)\n"
      << "region solid cylinder axis=z center=(" << CYL_X << "," << CYL_Y << ") radius="
      << CYL_R << "\n";
}

struct Result
{
    int inside = 0;         // particles inside the analytic surface
    int inSolidCell = 0;    // particles inside a cell the solver calls solid
    double worstDepth = 0.0;// deepest penetration, in cells
    int nearSurface = 0;    // particles within 1.5 cells of the surface
    int pastObstacle = 0;   // particles downstream of it, so the flow is not merely blocked
    double maxTangential = 0.0;
    double maxSpeed = 0.0;
    bool finite = true;
    int particles = 0;
};

static Result RunObstacle(const char *sceneFile, double dt, int steps)
{
    const int N = 64;

    DIMENSION = 2;
    SelectMethod("FLIP");
    THREAD_COUNT = 1;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 0.6, 0.0, 0.6, 0.0, 1.0};
    SIMULATION.GRID_SIZE = N;
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-3;
    SIMULATION.dt = dt;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = 4;
    FLIP_SETTINGS.alpha = 0.95;
    FLIP_SETTINGS.physicalDiffusion = false;
    FLIP_SETTINGS.diffusionSolver = DIFFUSION_SOLVERS::ADI;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    static MAC gridSol, gridAnt;
    gridSol = MAC();
    gridAnt = MAC();
    gridSol.InitializeGrid(SIMULATION.domain, true);
    gridAnt.InitializeGrid(SIMULATION.domain, true);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    // Load after dh/Nx/Ny are live: the evaluator reads them.
    if (!SCENE.Load(sceneFile))
    {
        check(false, "scene loads", SCENE.error);
        return Result();
    }
    SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = SCENE_SOLID_MASK_2D;

    gridSol.SetSolidMask(SCENE_SOLID_MASK_2D);
    gridAnt.SetSolidMask(SCENE_SOLID_MASK_2D);

    PressureSolver2D::InitializePressureSolver(&gridSol);
    FLIP2D::InitializeFLIP(&gridSol, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    gridAnt.EnforceBoundaryConditions(0);

    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        FLIP2D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, step * SIMULATION.dt);
        PressureSolver2D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver2D::Project(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        FLIP2D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        gridAnt.CopyGrid(gridSol);
    }

    Result r;
    r.particles = SIMULATION.particles.count;
    const double dh = SIMULATION.dh;

    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        const auto &pt = SIMULATION.particles.data[p];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y))
        {
            r.finite = false;
            continue;
        }

        const double lx = pt.x - SIMULATION.domain.x0;
        const double ly = pt.y - SIMULATION.domain.y0;
        const double dx = lx - CYL_X, dy = ly - CYL_Y;
        const double dist = std::sqrt(dx * dx + dy * dy);
        const double phi = dist - CYL_R; // negative inside

        if (phi < 0.0)
        {
            r.inside++;
            r.worstDepth = std::max(r.worstDepth, -phi / dh);
        }
        if (Particles::InSolidCell(&gridSol, pt.x, pt.y, 0.0))
            r.inSolidCell++;
        if (phi >= 0.0 && phi < 1.5 * dh)
        {
            r.nearSurface++;
            // Tangential component, to show contact is not gluing particles down.
            const double nx = dx / (dist > 0 ? dist : 1.0), ny = dy / (dist > 0 ? dist : 1.0);
            const double vn = pt.u * nx + pt.v * ny;
            const double tx = pt.u - vn * nx, ty = pt.v - vn * ny;
            r.maxTangential = std::max(r.maxTangential, std::sqrt(tx * tx + ty * ty));
        }
        if (lx > CYL_X + CYL_R)
            r.pastObstacle++;

        r.maxSpeed = std::max(r.maxSpeed, std::sqrt(pt.u * pt.u + pt.v * pt.v));
    }

    PressureSolver2D::DestroyPressureSolver();
    return r;
}

int run(int, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char *sceneFile = "obstacle_collision.scene";
    WriteScene(sceneFile);

    // ---- 1. the collision primitives, in isolation ----
    std::printf("\n== push-out and velocity response ==\n");
    {
        DIMENSION = 2;
        SIMULATION.domain = Domain{0.0, 0.6, 0.0, 0.6, 0.0, 1.0};
        SIMULATION.GRID_SIZE = 64;

        static MAC g;
        g = MAC();
        g.InitializeGrid(SIMULATION.domain, true);
        SIMULATION.dh = g.dh;
        SIMULATION.Nx = g.Nx;
        SIMULATION.Ny = g.Ny;

        check(SCENE.Load(sceneFile), "scene loads", SCENE.error);
        g.SetSolidMask(SCENE_SOLID_MASK_2D);
        check(SCENE.HasSolidRegions(), "the scene reports a solid region");

        // Straight at the cylinder from the left, ending up well inside it.
        Particle before;
        before.x = CYL_X - 2.0 * CYL_R;
        before.y = CYL_Y;

        Particle part = before;
        part.x = CYL_X - 0.25 * CYL_R;
        part.u = 1.0;
        part.v = 0.3;

        Particles::Resolve(&g, before, part);
        const double x = part.x, y = part.y, u = part.u, v = part.v;

        const double dist = std::sqrt((x - CYL_X) * (x - CYL_X) + (y - CYL_Y) * (y - CYL_Y));
        char buf[256];
        std::snprintf(buf, sizeof(buf), "phi = %+.5f m (radius %.3f)", dist - CYL_R, CYL_R);
        check(dist >= CYL_R, "a particle inside the surface is pushed back out", buf);
        check(!Particles::InSolidCell(&g, x, y, 0.0),
              "and out of every cell the solver calls solid");

        // It arrived moving +x into the cylinder from the left, so the outward normal is
        // -x there: the inward part of u must be gone, the tangential v untouched.
        std::snprintf(buf, sizeof(buf), "u = %+.4f, v = %+.4f (was +1.0000, +0.3000)", u, v);
        check(u <= 1e-9, "the inward normal component is removed", buf);
        check(std::abs(v - 0.3) < 1e-9, "the tangential component is left alone", buf);

        // A particle already leaving must not be touched.
        Particle leaving = before;
        leaving.x = CYL_X - 0.25 * CYL_R;
        leaving.u = -1.0;
        Particles::Resolve(&g, before, leaving);
        std::snprintf(buf, sizeof(buf), "u = %+.4f", leaving.u);
        check(std::abs(leaving.u + 1.0) < 1e-9, "a particle already moving away keeps its velocity", buf);
    }

    // ---- 1b. the three contact responses ----
    std::printf("\n== contact response ==\n");
    {
        static MAC g;
        g = MAC();
        g.InitializeGrid(SIMULATION.domain, true);
        SIMULATION.dh = g.dh; SIMULATION.Nx = g.Nx; SIMULATION.Ny = g.Ny;
        SCENE.Load(sceneFile);
        g.SetSolidMask(SCENE_SOLID_MASK_2D);

        Particle before;
        before.x = CYL_X - 2.0 * CYL_R; before.y = CYL_Y;

        // Straight in from the left: the outward normal is -x, so u is the inward
        // component and v is tangential.
        auto hit = [&](int mode, double e)
        {
            SIMULATION.particleOptions.contact = mode;
            SIMULATION.particleOptions.restitution = e;
            Particle p = before;
            p.x = CYL_X - 0.25 * CYL_R; p.u = 1.0; p.v = 0.3;
            Particles::Resolve(&g, before, p);
            return p;
        };

        char buf[256];
        Particle slip = hit(CONTACT_SLIP, 0.0);
        std::snprintf(buf, sizeof(buf), "u %+.4f, v %+.4f", slip.u, slip.v);
        check(std::abs(slip.u) < 1e-9 && std::abs(slip.v - 0.3) < 1e-9,
              "free slip removes the inward motion and keeps the tangential", buf);

        Particle noslip = hit(CONTACT_NOSLIP, 0.0);
        std::snprintf(buf, sizeof(buf), "u %+.4f, v %+.4f", noslip.u, noslip.v);
        check(std::abs(noslip.u) < 1e-12 && std::abs(noslip.v) < 1e-12,
              "no slip stops the particle dead", buf);

        Particle bounce = hit(CONTACT_BOUNCE, 0.5);
        std::snprintf(buf, sizeof(buf), "u %+.4f (expected -0.5000), v %+.4f", bounce.u, bounce.v);
        check(std::abs(bounce.u + 0.5) < 1e-9 && std::abs(bounce.v - 0.3) < 1e-9,
              "bounce reflects the normal component scaled by the restitution", buf);

        Particle zeroE = hit(CONTACT_BOUNCE, 0.0);
        std::snprintf(buf, sizeof(buf), "u %+.4f, v %+.4f", zeroE.u, zeroE.v);
        check(std::abs(zeroE.u - slip.u) < 1e-12 && std::abs(zeroE.v - slip.v) < 1e-12,
              "restitution 0 collapses to exactly free slip", buf);

        SIMULATION.particleOptions.contact = CONTACT_SLIP;
    }

    // ---- 2. substepping ----
    std::printf("\n== advection substeps ==\n");
    {
        const double dh = SIMULATION.dh;
        char buf[128];
        const int slow = Particles::SubstepCount(0.1 * dh / 1.0, 1.0);
        const int fast = Particles::SubstepCount(3.4 * dh / 1.0, 1.0);
        const int wild = Particles::SubstepCount(1e6 * dh, 1.0);
        std::snprintf(buf, sizeof(buf), "0.1 cells -> %d, 3.4 cells -> %d, 1e6 cells -> %d",
                      slow, fast, wild);
        check(slow == 1, "a sub-cell step is not split", buf);
        check(fast == 4, "a multi-cell step is split to about one cell each", buf);
        check(wild == Particles::MaxSubsteps(), "the split is capped", buf);
    }

    // ---- 3. the obstacle in a running dam break ----
    std::printf("\n== dam break against a cylinder ==\n");
    {
        Result r = RunObstacle(sceneFile, 5e-4, 700);
        char buf[256];
        std::printf("     %d particles | %d near the surface | %d past the obstacle | maxSpeed %.4f\n",
                    r.particles, r.nearSurface, r.pastObstacle, r.maxSpeed);

        check(r.finite, "particles stay finite");
        std::snprintf(buf, sizeof(buf), "%d particles, worst depth %.3f cells",
                      r.inside, r.worstDepth);
        check(r.worstDepth < 1.0, "analytic penetration stays within the voxelisation error", buf);
        std::snprintf(buf, sizeof(buf), "%d particles", r.inSolidCell);
        check(r.inSolidCell == 0, "no particle is inside a solid cell", buf);

        // Guards against a vacuous pass: the water has to actually get there, and get past.
        std::snprintf(buf, sizeof(buf), "%d particles within 1.5 cells", r.nearSurface);
        check(r.nearSurface > 20, "the water actually reaches the obstacle", buf);
        std::snprintf(buf, sizeof(buf), "%d particles beyond x = %.3f", r.pastObstacle,
                      CYL_X + CYL_R);
        check(r.pastObstacle > 20, "and flows past it rather than being dammed", buf);

        std::snprintf(buf, sizeof(buf), "max |v_tangential| = %.4f m/s", r.maxTangential);
        check(r.maxTangential > 0.05, "contact does not glue particles to the surface", buf);
    }

    // ---- 4. a step big enough to jump the obstacle in one go ----
    //         Without substepping the contact test only sees the endpoint, so a particle
    //         moving further than the obstacle is thick crosses it undetected.
    std::printf("\n== high CFL (tunnelling) ==\n");
    {
        Result r = RunObstacle(sceneFile, 4e-3, 120);
        char buf[256];
        std::printf("     %d particles | %d near the surface | maxSpeed %.4f | ~%.1f cells per step\n",
                    r.particles, r.nearSurface, r.maxSpeed, r.maxSpeed * 4e-3 / SIMULATION.dh);
        check(r.finite, "particles stay finite");
        std::snprintf(buf, sizeof(buf), "%d particles, worst depth %.3f cells",
                      r.inside, r.worstDepth);
        check(r.worstDepth < 1.0, "nothing tunnels past the voxelisation error", buf);
        std::snprintf(buf, sizeof(buf), "%d particles", r.inSolidCell);
        check(r.inSolidCell == 0, "no particle tunnels into a solid cell", buf);
    }

    std::remove(sceneFile);
    std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
} // namespace flip2d_obstacle_check

// ------------------------------------------------------------------------------------------------------------------
// flip3d_obstacle_check
// ------------------------------------------------------------------------------------------------------------------
namespace flip3d_obstacle_check
{
// The 3D half of the particle / solid-geometry contact check. Same assertions as
// tests/flip2d_obstacle_check.cpp, on a sphere standing in the path of a collapsing
// column, so the z handling in Core/Particles.h is exercised rather than assumed.//
// Note the contract these assert. The **voxel mask** is what decides where a particle may
// be, because it is what the pressure solve sees; the analytic surface only supplies the
// push direction. The two can be up to a cell apart (with `sampling corner` a whole cell
// on the low-index faces), so the hard invariant is "no particle in a SOLID_CELL" and the
// analytic penetration is merely *bounded* by the voxelisation error. Asserting zero
// analytic penetration instead is what produced the -x/-y/-z conveyor: it made the
// mismatch shell a trap the grid drove particles into and the collision evicted them from,
// every step.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok)
        failures++;
}

static const double SPH_X = 0.32, SPH_Y = 0.08, SPH_Z = 0.15, SPH_R = 0.06;

static void WriteScene(const char *path)
{
    std::ofstream f(path);
    f << "sampling center\n"
      << "boundary xmin noslip\nboundary xmax noslip\n"
      << "boundary ymin noslip\nboundary ymax noslip\n"
      << "boundary zmin noslip\nboundary zmax noslip\n"
      << "region fluid box min=(0,0,0) max=(0.18,0.20,0.3)\n"
      << "region solid sphere center=(" << SPH_X << "," << SPH_Y << "," << SPH_Z
      << ") radius=" << SPH_R << "\n";
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int steps = (argc > 1) ? std::atoi(argv[1]) : 250;
    const char *sceneFile = "obstacle_collision_3d.scene";
    WriteScene(sceneFile);

    DIMENSION = 3;
    SelectMethod("FLIP");
    THREAD_COUNT = 1;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 0.6, 0.0, 0.3, 0.0, 0.3};
    SIMULATION.GRID_SIZE = 30; // 60x30x30, inside the 60^3 ceiling
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.002;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = 8;
    FLIP_SETTINGS.alpha = 0.95;
    FLIP_SETTINGS.physicalDiffusion = false;
    FLIP_SETTINGS.diffusionSolver = DIFFUSION_SOLVERS::ADI;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    static MAC gridSol, gridAnt;
    gridSol.InitializeGrid(SIMULATION.domain, false);
    gridAnt.InitializeGrid(SIMULATION.domain, false);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.Nz = gridSol.Nz;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    check(SCENE.Load(sceneFile), "scene loads", SCENE.error);
    check(SCENE.HasSolidRegions(), "the scene reports a solid region");

    SIMULATION.VelocityBoundaryFunction = SCENE_VELOCITY;
    SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
    SIMULATION.SolidMaskFunction = SCENE_SOLID_MASK;

    gridSol.SetSolidMask(SCENE_SOLID_MASK);
    gridAnt.SetSolidMask(SCENE_SOLID_MASK);

    PressureSolver3D::InitializePressureSolver(&gridSol);
    FLIP3D::InitializeFLIP(&gridSol, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    gridAnt.EnforceBoundaryConditions(0);

    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        FLIP3D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, step * SIMULATION.dt);
        PressureSolver3D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver3D::Project(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        FLIP3D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        gridAnt.CopyGrid(gridSol);
    }

    int inside = 0, inSolidCell = 0, nearSurface = 0, past = 0;
    double worstDepth = 0.0, maxTangential = 0.0, maxSpeed = 0.0;
    bool finite = true;
    const double dh = SIMULATION.dh;

    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        const auto &q = SIMULATION.particles.data[p];
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z))
        {
            finite = false;
            continue;
        }
        const double dx = (q.x - SIMULATION.domain.x0) - SPH_X;
        const double dy = (q.y - SIMULATION.domain.y0) - SPH_Y;
        const double dz = (q.z - SIMULATION.domain.z0) - SPH_Z;
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double phi = dist - SPH_R;

        if (phi < 0.0)
        {
            inside++;
            worstDepth = std::max(worstDepth, -phi / dh);
        }
        if (Particles::InSolidCell(&gridSol, q.x, q.y, q.z))
            inSolidCell++;
        if (phi >= 0.0 && phi < 1.5 * dh)
        {
            nearSurface++;
            const double inv = (dist > 0.0) ? 1.0 / dist : 0.0;
            const double nx = dx * inv, ny = dy * inv, nz = dz * inv;
            const double vn = q.u * nx + q.v * ny + q.w * nz;
            const double tx = q.u - vn * nx, ty = q.v - vn * ny, tz = q.w - vn * nz;
            maxTangential = std::max(maxTangential, std::sqrt(tx * tx + ty * ty + tz * tz));
        }
        if ((q.x - SIMULATION.domain.x0) > SPH_X + SPH_R)
            past++;
        maxSpeed = std::max(maxSpeed, std::sqrt(q.u * q.u + q.v * q.v + q.w * q.w));
    }

    std::printf("\n== dam break against a sphere (%d steps) ==\n", steps);
    std::printf("     %d particles | %d near the surface | %d past it | maxSpeed %.4f\n",
                SIMULATION.particles.count, nearSurface, past, maxSpeed);

    char buf[256];
    check(finite, "particles stay finite");
    std::snprintf(buf, sizeof(buf), "%d particles, worst depth %.3f cells", inside, worstDepth);
    check(worstDepth < 1.0, "analytic penetration stays within the voxelisation error", buf);
    std::snprintf(buf, sizeof(buf), "%d particles", inSolidCell);
    check(inSolidCell == 0, "no particle is inside a solid cell", buf);
    std::snprintf(buf, sizeof(buf), "%d particles within 1.5 cells", nearSurface);
    check(nearSurface > 20, "the water actually reaches the obstacle", buf);
    std::snprintf(buf, sizeof(buf), "%d particles beyond x = %.3f", past, SPH_X + SPH_R);
    check(past > 20, "and flows past it rather than being dammed", buf);
    std::snprintf(buf, sizeof(buf), "max |v_tangential| = %.4f m/s", maxTangential);
    check(maxTangential > 0.05, "contact does not glue particles to the surface", buf);

    PressureSolver3D::DestroyPressureSolver();
    std::remove(sceneFile);
    std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
} // namespace flip3d_obstacle_check

// ------------------------------------------------------------------------------------------------------------------
// flip_solidflux_check
// ------------------------------------------------------------------------------------------------------------------
namespace flip_solidflux_check
{
// Diagnostic: what the solver does at an interior obstacle.
//
// Symptom being chased: in 3D, particles drift toward a solid obstacle and pile against
// it, sliding along the surface, as though the obstacle were a sink. Three candidate
// causes, all measurable:
//
//   A. spurious fluid cells -- UpdateFluidCells' hole filling counts SOLID neighbours as
//      "non-empty", so an air cell hugging the obstacle can be promoted to FLUID with no
//      particles in it. The pressure solve then has to fill it, which pulls fluid in.
//   B. flux through the obstacle -- particle-to-grid writes a particle-weighted value onto
//      faces that separate FLUID from SOLID. ExtrapolateVelocity treats those faces as
//      pinned and keeps them; ProjectPressure skips them. Nothing imposes u.n = 0, and
//      SetBorder only pins the domain ring, not interior geometry. Any residue is a mass
//      source or sink stuck to the obstacle.
//   C. neither, and it is downstream of the pressure solve.
//
// Reports both dimensions so the 2D/3D asymmetry the author reports can be checked rather
// than assumed. Measured before the fix: max|u.n| reached 1.16 m/s on the sphere and 1.00
// on the cylinder, where the correct value is exactly zero, and the count of particles
// hugging the obstacle rose monotonically (1866 -> 2582 -> 2806 -> 2981 in 3D) instead of
// settling. Cause A contributed 205 no-particle FLUID cells in 3D.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok)
        failures++;
}

static void WriteScene2D(const char *path)
{
    std::ofstream f(path);
    f << "sampling center\n"
      << "boundary xmin noslip\nboundary xmax noslip\n"
      << "boundary ymin noslip\nboundary ymax noslip\n"
      << "region fluid box min=(0,0,0) max=(0.20,0.40,1)\n"
      << "region solid cylinder axis=z center=(0.34,0.10) radius=0.07\n";
}
static void WriteScene3D(const char *path)
{
    std::ofstream f(path);
    f << "sampling center\n"
      << "boundary xmin noslip\nboundary xmax noslip\n"
      << "boundary ymin noslip\nboundary ymax noslip\n"
      << "boundary zmin noslip\nboundary zmax noslip\n"
      << "region fluid box min=(0,0,0) max=(0.18,0.20,0.3)\n"
      << "region solid sphere center=(0.32,0.08,0.15) radius=0.06\n";
}

struct Diag
{
    int fluidCells = 0;
    int emptyFluidCells = 0;   // FLUID with no particle in them  (cause A)
    double maxSolidFaceVel = 0.0; // max |u.n| on a FLUID|SOLID face   (cause B)
    double netInflux = 0.0;       // signed flux through obstacle faces, positive = into it
    int nearObstacle = 0;         // particles within 1 cell of the surface (accumulation)
};

// ---------------------------------------------------------------- 3D
static void Report3D(int steps)
{
    const char *sceneFile = "solidflux_3d.scene";
    WriteScene3D(sceneFile);

    DIMENSION = 3; SelectMethod("FLIP"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 0.6, 0.0, 0.3, 0.0, 0.3};
    SIMULATION.GRID_SIZE = 30;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 1e-3; SIMULATION.dt = 0.002;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = 8; FLIP_SETTINGS.alpha = 0.95;
    FLIP_SETTINGS.physicalDiffusion = false; FLIP_SETTINGS.diffusionSolver = DIFFUSION_SOLVERS::ADI;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    static MAC gridSol, gridAnt;
    gridSol.InitializeGrid(SIMULATION.domain, false);
    gridAnt.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx; SIMULATION.Ny = gridSol.Ny; SIMULATION.Nz = gridSol.Nz;
    SIMULATION.GRID_SOL = &gridSol; SIMULATION.GRID_ANT = &gridAnt;

    if (!SCENE.Load(sceneFile)) { std::printf("scene failed: %s\n", SCENE.error.c_str()); return; }
    SIMULATION.VelocityBoundaryFunction = SCENE_VELOCITY;
    SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
    SIMULATION.SolidMaskFunction = SCENE_SOLID_MASK;
    gridSol.SetSolidMask(SCENE_SOLID_MASK);
    gridAnt.SetSolidMask(SCENE_SOLID_MASK);

    PressureSolver3D::InitializePressureSolver(&gridSol);
    FLIP3D::InitializeFLIP(&gridSol, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    gridAnt.EnforceBoundaryConditions(0);

    const double dh = SIMULATION.dh;
    double worstFace = 0.0; int peakNear = 0, finalNear = 0;
    std::printf("\n== 3D, sphere obstacle, %d steps ==\n", steps);
    std::printf("  %6s %8s %10s %14s %14s %10s\n", "step", "fluid", "no-particle",
                "max|u.n| solid", "net influx", "near");

    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        FLIP3D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, step * SIMULATION.dt);
        PressureSolver3D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver3D::Project(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        FLIP3D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        gridAnt.CopyGrid(gridSol);

        if (step % (steps / 5) != 0 && step != steps - 1) continue;

        Diag d;
        // particles per cell
        std::vector<int> count((size_t)gridSol.Nx * gridSol.Ny * gridSol.Nz, 0);
        for (int p = 0; p < SIMULATION.particles.count; p++)
        {
            const auto &q = SIMULATION.particles.data[p];
            int j = (int)((q.x - SIMULATION.domain.x0 + dh) / dh);
            int i = (int)((q.y - SIMULATION.domain.y0 + dh) / dh);
            int k = (int)((q.z - SIMULATION.domain.z0 + dh) / dh);
            if (i < 0 || i >= gridSol.Ny || j < 0 || j >= gridSol.Nx || k < 0 || k >= gridSol.Nz) continue;
            count[(size_t)(i * gridSol.Nx + j) * gridSol.Nz + k]++;

            const double dx = (q.x - SIMULATION.domain.x0) - 0.32;
            const double dy = (q.y - SIMULATION.domain.y0) - 0.08;
            const double dz = (q.z - SIMULATION.domain.z0) - 0.15;
            const double phi = std::sqrt(dx*dx + dy*dy + dz*dz) - 0.06;
            if (phi >= 0.0 && phi < dh) d.nearObstacle++;
        }

        for (int i = 1; i < gridSol.Ny - 1; i++)
            for (int j = 1; j < gridSol.Nx - 1; j++)
                for (int k = 1; k < gridSol.Nz - 1; k++)
                {
                    if (gridSol.GetSolid(i, j, k) != FLUID_CELL) continue;
                    d.fluidCells++;
                    if (count[(size_t)(i * gridSol.Nx + j) * gridSol.Nz + k] == 0) d.emptyFluidCells++;

                    // faces of this fluid cell that touch an interior solid (not the domain ring)
                    auto interiorSolid = [&](int a, int b, int c)
                    {
                        if (a <= 0 || a >= gridSol.Ny - 1 || b <= 0 || b >= gridSol.Nx - 1 ||
                            c <= 0 || c >= gridSol.Nz - 1) return false;
                        return gridSol.GetSolid(a, b, c) == SOLID_CELL;
                    };
                    if (interiorSolid(i, j - 1, k))
                    { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetU(i, j, k)));
                      d.netInflux += gridSol.GetU(i, j, k) * -1.0; }
                    if (interiorSolid(i, j + 1, k))
                    { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetU(i, j + 1, k)));
                      d.netInflux += gridSol.GetU(i, j + 1, k); }
                    if (interiorSolid(i - 1, j, k))
                    { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetV(i, j, k)));
                      d.netInflux += gridSol.GetV(i, j, k) * -1.0; }
                    if (interiorSolid(i + 1, j, k))
                    { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetV(i + 1, j, k)));
                      d.netInflux += gridSol.GetV(i + 1, j, k); }
                    if (interiorSolid(i, j, k - 1))
                    { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetW(i, j, k)));
                      d.netInflux += gridSol.GetW(i, j, k) * -1.0; }
                    if (interiorSolid(i, j, k + 1))
                    { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetW(i, j, k + 1)));
                      d.netInflux += gridSol.GetW(i, j, k + 1); }
                }

        std::printf("  %6d %8d %10d %14.6f %14.6f %10d\n", step, d.fluidCells,
                    d.emptyFluidCells, d.maxSolidFaceVel, d.netInflux, d.nearObstacle);
        worstFace = std::max(worstFace, d.maxSolidFaceVel);
        peakNear = std::max(peakNear, d.nearObstacle);
        finalNear = d.nearObstacle;
    }

    char buf[128];
    std::snprintf(buf, sizeof(buf), "max over the run = %.3e", worstFace);
    check(worstFace < 1e-12, "3D: no flux through an obstacle face", buf);
    std::snprintf(buf, sizeof(buf), "peak %d, final %d", peakNear, finalNear);
    check(finalNear <= peakNear, "3D: particles at the surface settle rather than pile up", buf);

    PressureSolver3D::DestroyPressureSolver();
    std::remove(sceneFile);
}

// ---------------------------------------------------------------- 2D
static void Report2D(int steps)
{
    const char *sceneFile = "solidflux_2d.scene";
    WriteScene2D(sceneFile);

    DIMENSION = 2; SelectMethod("FLIP"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 0.6, 0.0, 0.6, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 64;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 1e-3; SIMULATION.dt = 5e-4;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = 4; FLIP_SETTINGS.alpha = 0.95;
    FLIP_SETTINGS.physicalDiffusion = false; FLIP_SETTINGS.diffusionSolver = DIFFUSION_SOLVERS::ADI;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    static MAC gridSol, gridAnt;
    gridSol = MAC(); gridAnt = MAC();
    gridSol.InitializeGrid(SIMULATION.domain, true);
    gridAnt.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx; SIMULATION.Ny = gridSol.Ny;
    SIMULATION.GRID_SOL = &gridSol; SIMULATION.GRID_ANT = &gridAnt;

    if (!SCENE.Load(sceneFile)) { std::printf("scene failed: %s\n", SCENE.error.c_str()); return; }
    SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = SCENE_SOLID_MASK_2D;
    gridSol.SetSolidMask(SCENE_SOLID_MASK_2D);
    gridAnt.SetSolidMask(SCENE_SOLID_MASK_2D);

    PressureSolver2D::InitializePressureSolver(&gridSol);
    FLIP2D::InitializeFLIP(&gridSol, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    gridAnt.EnforceBoundaryConditions(0);

    const double dh = SIMULATION.dh;
    double worstFace = 0.0;
    std::printf("\n== 2D, cylinder obstacle, %d steps ==\n", steps);
    std::printf("  %6s %8s %10s %14s %14s %10s\n", "step", "fluid", "no-particle",
                "max|u.n| solid", "net influx", "near");

    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        FLIP2D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, step * SIMULATION.dt);
        PressureSolver2D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver2D::Project(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        FLIP2D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, 0.0);
        gridAnt.CopyGrid(gridSol);

        if (step % (steps / 5) != 0 && step != steps - 1) continue;

        Diag d;
        std::vector<int> count((size_t)gridSol.Nx * gridSol.Ny, 0);
        for (int p = 0; p < SIMULATION.particles.count; p++)
        {
            const auto &q = SIMULATION.particles.data[p];
            int j = (int)((q.x - SIMULATION.domain.x0 + dh) / dh);
            int i = (int)((q.y - SIMULATION.domain.y0 + dh) / dh);
            if (i < 0 || i >= gridSol.Ny || j < 0 || j >= gridSol.Nx) continue;
            count[(size_t)i * gridSol.Nx + j]++;
            const double dx = (q.x - SIMULATION.domain.x0) - 0.34;
            const double dy = (q.y - SIMULATION.domain.y0) - 0.10;
            const double phi = std::sqrt(dx*dx + dy*dy) - 0.07;
            if (phi >= 0.0 && phi < dh) d.nearObstacle++;
        }

        for (int i = 1; i < gridSol.Ny - 1; i++)
            for (int j = 1; j < gridSol.Nx - 1; j++)
            {
                if (gridSol.GetSolid(i, j) != FLUID_CELL) continue;
                d.fluidCells++;
                if (count[(size_t)i * gridSol.Nx + j] == 0) d.emptyFluidCells++;

                auto interiorSolid = [&](int a, int b)
                {
                    if (a <= 0 || a >= gridSol.Ny - 1 || b <= 0 || b >= gridSol.Nx - 1) return false;
                    return gridSol.GetSolid(a, b) == SOLID_CELL;
                };
                if (interiorSolid(i, j - 1))
                { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetU(i, j)));
                  d.netInflux += -gridSol.GetU(i, j); }
                if (interiorSolid(i, j + 1))
                { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetU(i, j + 1)));
                  d.netInflux += gridSol.GetU(i, j + 1); }
                if (interiorSolid(i - 1, j))
                { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetV(i, j)));
                  d.netInflux += -gridSol.GetV(i, j); }
                if (interiorSolid(i + 1, j))
                { d.maxSolidFaceVel = std::max(d.maxSolidFaceVel, std::abs(gridSol.GetV(i + 1, j)));
                  d.netInflux += gridSol.GetV(i + 1, j); }
            }

        // packing: the cells touching the obstacle against the domain mean
        int maxAtBox = 0; double sumPC = 0; int nPC = 0;
        for (int i = 1; i < gridSol.Ny - 1; i++)
            for (int j = 1; j < gridSol.Nx - 1; j++)
            {
                const int c = count[(size_t)i * gridSol.Nx + j];
                if (gridSol.GetSolid(i, j) == FLUID_CELL) { sumPC += c; nPC++; }
                bool touches = false;
                for (int di = -1; di <= 1 && !touches; di++)
                    for (int dj = -1; dj <= 1 && !touches; dj++)
                    {
                        const int a = i + di, b = j + dj;
                        if (a <= 0 || a >= gridSol.Ny - 1 || b <= 0 || b >= gridSol.Nx - 1) continue;
                        if (gridSol.GetSolid(a, b) == SOLID_CELL) touches = true;
                    }
                if (touches) maxAtBox = std::max(maxAtBox, c);
            }

        std::printf("  %6d %8d %10d %14.6f %14.6f %10d   ppc: max@box=%d mean=%.2f (seed %d)\n",
                    step, d.fluidCells, d.emptyFluidCells, d.maxSolidFaceVel, d.netInflux,
                    d.nearObstacle, maxAtBox, nPC ? sumPC / nPC : 0.0,
                    FLIP_SETTINGS.particlesPerCell);
        worstFace = std::max(worstFace, d.maxSolidFaceVel);
    }

    char buf[128];
    std::snprintf(buf, sizeof(buf), "max over the run = %.3e", worstFace);
    check(worstFace < 1e-12, "2D: no flux through an obstacle face", buf);

    PressureSolver2D::DestroyPressureSolver();
    std::remove(sceneFile);
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int s = (argc > 1) ? std::atoi(argv[1]) : 300;
    Report3D(s);
    Report2D(s * 2);
    std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
} // namespace flip_solidflux_check

// ------------------------------------------------------------------------------------------------------------------
// flip_density_check
// ------------------------------------------------------------------------------------------------------------------
namespace flip_density_check
{
// FLIP density correction (Kugelstadt et al. 2019), FLIP_DENSITY_CORRECTION. Drives the FLIP pipeline as the
// manager does, on the CPU pressure path, and checks: the particle density reads exactly the rest density on a
// lattice at rest, walls included; a tank at rest is left at rest; and the liquid volume a dam break drifts to,
// with the correction off and on, in 2D and 3D.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail)
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}

static std::string Num(double v)
{
    char b[64];
    std::snprintf(b, sizeof(b), "%.3e", v);
    return b;
}

static MAC gridSol, gridAnt;

static void Setup(int dim, int N, const std::string &region, double gravity, double dt, bool correct)
{
    DIMENSION = dim;
    SelectMethod("FLIP");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 0.6, 0.0, 0.6, 0.0, 0.6};
    SIMULATION.GRID_SIZE = N;
    SIMULATION.RHO = 1000.0;
    SIMULATION.EPS = 1e-6;
    SIMULATION.dt = dt;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    FLIP_SETTINGS.particlesPerCell = dim == 2 ? 4 : 8;
    FLIP_SETTINGS.alpha = 1.0;
    FLIP_SETTINGS.physicalDiffusion = false;
    FLIP_SETTINGS.densityCorrection = correct;
    FLIP_SETTINGS.densitySurfaceClamp = true;
    SIMULATION.f = Vec3{0.0, gravity, 0.0};

    if (!SCENE.LoadText("region fluid box " + region + "\n"))
    {
        std::printf("scene error: %s\n", SCENE.error.c_str());
        std::exit(2);
    }

    gridSol = MAC();
    gridAnt = MAC();
    if (dim == 2)
    {
        SIMULATION.VelocityBoundaryFunction2D = DAMBREAK_BORDER_2D;
        SIMULATION.PressureBoundaryFunction2D = DAMBREAK_PRESSURE_2D;
        SIMULATION.SolidMaskFunction2D = DAMBREAK_SOLID_MASK_2D;
        gridSol.InitializeGrid(SIMULATION.domain, true);
        gridAnt.InitializeGrid(SIMULATION.domain, true);
    }
    else
    {
        SIMULATION.VelocityBoundaryFunction = DAMBREAK_BORDER;
        SIMULATION.PressureBoundaryFunction = DAMBREAK_PRESSURE;
        SIMULATION.SolidMaskFunction = DAMBREAK_SOLID_MASK;
        gridSol.InitializeGrid(SIMULATION.domain, false);
        gridAnt.InitializeGrid(SIMULATION.domain, false);
    }
    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.Nz = gridSol.Nz;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    if (dim == 2)
    {
        gridSol.SetSolidMask(DAMBREAK_SOLID_MASK_2D);
        gridAnt.SetSolidMask(DAMBREAK_SOLID_MASK_2D);
        PressureSolver2D::InitializePressureSolver(&gridSol);
        FLIP2D::InitializeFLIP(&gridSol, &SIMULATION.particles, dt, 1.0);
        gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
        gridAnt.EnforceBoundaryConditions(0);
    }
    else
    {
        gridSol.SetSolidMask(DAMBREAK_SOLID_MASK);
        gridAnt.SetSolidMask(DAMBREAK_SOLID_MASK);
        PressureSolver3D::InitializePressureSolver(&gridSol);
        FLIP3D::InitializeFLIP(&gridSol, &SIMULATION.particles, dt, 1.0);
        gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
        gridAnt.EnforceBoundaryConditions(0);
    }
}

// One step in the manager's order: momentum, pressure, projection (with G2P), then the correction.
static void Step(int dim, int step)
{
    const double t = step * SIMULATION.dt;
    if (dim == 2)
    {
        gridSol.ClearFlowFields();
        FLIP2D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, t);
        PressureSolver2D::SolvePressure(nullptr, &gridSol, nullptr, t);
        PressureSolver2D::Project(&gridAnt, &gridSol, &SIMULATION.particles, t);
        FLIP2D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, t);
        if (FLIP_SETTINGS.densityCorrection)
            FLIP2D::DensityCorrection(&gridAnt, &gridSol, &SIMULATION.particles, t);
    }
    else
    {
        gridSol.ClearFlowFields();
        FLIP3D::FLIP_Momentum(&gridAnt, &gridSol, &SIMULATION.particles, t);
        PressureSolver3D::SolvePressure(nullptr, &gridSol, nullptr, t);
        PressureSolver3D::Project(&gridAnt, &gridSol, &SIMULATION.particles, t);
        FLIP3D::FLIP_GridToParticle(&gridAnt, &gridSol, &SIMULATION.particles, t);
        if (FLIP_SETTINGS.densityCorrection)
            FLIP3D::DensityCorrection(&gridAnt, &gridSol, &SIMULATION.particles, t);
    }
    gridAnt.CopyGrid(gridSol);
}

static void Teardown(int dim)
{
    if (dim == 2) PressureSolver2D::DestroyPressureSolver();
    else PressureSolver3D::DestroyPressureSolver();
}

static double CellVolume(int dim) { return std::pow(SIMULATION.dh, dim); }
static double ParticleVolume(int dim) { return dim == 2 ? FLIP2D::particleVolume : FLIP3D::particleVolume; }
static const std::vector<double> &Density(int dim) { return dim == 2 ? FLIP2D::density : FLIP3D::density; }

static bool BesideAir(int dim, int i, int j, int k)
{
    const int di[6] = {1, -1, 0, 0, 0, 0}, dj[6] = {0, 0, 1, -1, 0, 0}, dk[6] = {0, 0, 0, 0, 1, -1};
    for (int n = 0; n < (dim == 2 ? 4 : 6); n++)
    {
        const int t = dim == 2 ? gridSol.GetSolid(i + di[n], j + dj[n]) : gridSol.GetSolid(i + di[n], j + dj[n], k + dk[n]);
        if (t == EMPTY_CELL) return true;
    }
    return false;
}

// 1. At rest on a lattice, every fluid cell not beside air reads rho/rho0 = 1, the ones against walls included,
//    and the cells together hold exactly the seeded volume.
static void RestDensity(int dim)
{
    std::printf("\n== %dD: rest density on the seeding lattice ==\n", dim);
    const int N = dim == 2 ? 48 : 24;
    Setup(dim, N, "min=(0,0,0) max=(0.6,0.3,0.6)", 0.0, 1e-3, true);
    Step(dim, 0);   // no gravity: nothing moves, and the step marks the fluid cells
    if (dim == 2) FLIP2D::ComputeDensity(&gridSol, &SIMULATION.particles);
    else FLIP3D::ComputeDensity(&gridSol, &SIMULATION.particles);

    const std::vector<double> &d = Density(dim);
    double worst = 0.0, worstWall = 0.0, total = 0.0;
    int n = 0;
    for (size_t c = 0; c < d.size(); c++) total += d[c];
    for (int i = 1; i < gridSol.Ny - 1; i++)
        for (int j = 1; j < gridSol.Nx - 1; j++)
            for (int k = (dim == 2 ? 0 : 1); k < (dim == 2 ? 1 : gridSol.Nz - 1); k++)
            {
                const int t = dim == 2 ? gridSol.GetSolid(i, j) : gridSol.GetSolid(i, j, k);
                if (t != FLUID_CELL || BesideAir(dim, i, j, k)) continue;
                const size_t c = dim == 2 ? (size_t)i * gridSol.Nx + j : ((size_t)i * gridSol.Nx + j) * gridSol.Nz + k;
                const double e = std::abs(d[c] - 1.0);
                worst = std::max(worst, e);
                const bool wall = i == 1 || j == 1 || j == gridSol.Nx - 2 || (dim == 3 && (k == 1 || k == gridSol.Nz - 2));
                if (wall) worstWall = std::max(worstWall, e);
                n++;
            }
    const double seeded = SIMULATION.particles.count * ParticleVolume(dim);
    const double relTotal = std::abs(total * CellVolume(dim) - seeded) / seeded;
    std::printf("     %d interior cells, worst |rho/rho0 - 1| = %s (against a wall %s)\n", n, Num(worst).c_str(),
                Num(worstWall).c_str());
    check(n > 0 && worst < 1e-12, "rest density exact, walls included", Num(worst));
    check(relTotal < 1e-12, "the cells hold exactly the seeded volume", Num(relTotal));
    Teardown(dim);
}

// 2. A tank at rest stays at rest: under gravity the correction must find nothing to correct.
static void TankAtRest(int dim)
{
    std::printf("\n== %dD: tank at rest, correction on ==\n", dim);
    const int N = dim == 2 ? 48 : 24;
    Setup(dim, N, "min=(0,0,0) max=(0.6,0.3,0.6)", -9.81, 1e-3, true);
    std::vector<Particle> start(SIMULATION.particles.data.begin(),
                                SIMULATION.particles.data.begin() + SIMULATION.particles.count);
    const int steps = dim == 2 ? 300 : 100;
    for (int s = 0; s < steps; s++) Step(dim, s);
    double moved = 0.0;
    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        const Particle &a = start[p], &b = SIMULATION.particles.data[p];
        moved = std::max(moved, std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)));
    }
    std::printf("     largest particle displacement after %d steps: %.3e cells\n", steps, moved / SIMULATION.dh);
    check(moved < 1e-3 * SIMULATION.dh, "the tank does not move", Num(moved / SIMULATION.dh) + " cells");
    Teardown(dim);
}

// 3. A dam break, off and on: the liquid volume the grid sees at the end, against the seeded volume.
struct DriftResult { double drift; double interior; bool finite; bool outOfSolid; };

static DriftResult DamBreak(int dim, bool correct, int steps)
{
    const int N = dim == 2 ? 48 : 24;
    Setup(dim, N, "min=(0,0,0) max=(0.2,0.4,0.6)", -9.81, dim == 2 ? 1e-3 : 2e-3, correct);
    const double seeded = SIMULATION.particles.count * ParticleVolume(dim);
    for (int s = 0; s < steps; s++) Step(dim, s);
    DriftResult r{0.0, 0.0, true, true};
    r.drift = gridSol.GetFluidCellCount() * CellVolume(dim) / seeded - 1.0;
    // The mean rho/rho0 over fluid cells not beside air: what the particles actually do, free of the surface
    // cells the cell count takes as full.
    if (dim == 2) FLIP2D::ComputeDensity(&gridSol, &SIMULATION.particles);
    else FLIP3D::ComputeDensity(&gridSol, &SIMULATION.particles);
    int n = 0;
    for (int i = 1; i < gridSol.Ny - 1; i++)
        for (int j = 1; j < gridSol.Nx - 1; j++)
            for (int k = (dim == 2 ? 0 : 1); k < (dim == 2 ? 1 : gridSol.Nz - 1); k++)
            {
                const int t = dim == 2 ? gridSol.GetSolid(i, j) : gridSol.GetSolid(i, j, k);
                if (t != FLUID_CELL || BesideAir(dim, i, j, k)) continue;
                r.interior += Density(dim)[dim == 2 ? (size_t)i * gridSol.Nx + j
                                                     : ((size_t)i * gridSol.Nx + j) * gridSol.Nz + k];
                n++;
            }
    if (n > 0) r.interior /= n;
    const double dh = SIMULATION.dh;
    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        const Particle &q = SIMULATION.particles.data[p];
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z)) { r.finite = false; continue; }
        const int i = (int)std::floor((q.y - gridSol.FacePos(MAC::Y, 0)) / dh);
        const int j = (int)std::floor((q.x - gridSol.FacePos(MAC::X, 0)) / dh);
        const int k = dim == 2 ? 0 : (int)std::floor((q.z - gridSol.FacePos(MAC::Z, 0)) / dh);
        const int t = dim == 2 ? gridSol.GetSolid(i, j) : gridSol.GetSolid(i, j, k);
        if (t == SOLID_CELL) r.outOfSolid = false;
    }
    Teardown(dim);
    return r;
}

static void Drift(int dim)
{
    const int steps = dim == 2 ? 2000 : 500;
    std::printf("\n== %dD: dam break, %d steps (%.1f s) ==\n", dim, steps, steps * (dim == 2 ? 1e-3 : 2e-3));
    const DriftResult off = DamBreak(dim, false, steps);
    const DriftResult on = DamBreak(dim, true, steps);
    std::printf("     fluid volume drift: off %+.2f%%, on %+.2f%%\n", 100.0 * off.drift, 100.0 * on.drift);
    std::printf("     mean rho/rho0 away from the surface: off %.4f, on %.4f\n", off.interior, on.interior);
    check(on.finite && off.finite, "particles stay finite", "");
    check(on.outOfSolid, "no particle ends in a solid cell with the correction", "");
    check(std::abs(on.drift) < std::abs(off.drift), "the correction reduces the drift", "");
    check(std::abs(on.interior - 1.0) < std::abs(off.interior - 1.0), "the correction holds the interior density", "");
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    RegisterBuiltinMethods();
    RestDensity(2);
    RestDensity(3);
    TankAtRest(2);
    TankAtRest(3);
    Drift(2);
    Drift(3);
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace flip_density_check

// ------------------------------------------------------------------------------------------------------------------
// diffusion2d_check
// ------------------------------------------------------------------------------------------------------------------
namespace diffusion2d_check
{
// Headless check for the 2D viscous solvers.
//
// Builds a MAC grid with a one-cell solid ring, drives it with a known velocity
// field, and runs the two diffusion operators from identical input so they can be
// compared directly. Link against the project objects (minus main.o):
//
//   see tests/run_diffusion_check.sh


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail)
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok)
        failures++;
}

// u(x,y) = sin(pi x) sin(pi y) sampled at the staggered face positions.
static void SeedField(MAC *g)
{
    double dh = g->dh;
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx + 1; j++)
            g->SetU(i, j, std::sin(M_PI * g->FacePos(MAC::X, j)) * std::sin(M_PI * g->CentrePos(MAC::Y, i)));
    for (int i = 0; i < g->Ny + 1; i++)
        for (int j = 0; j < g->Nx; j++)
            g->SetV(i, j, std::sin(M_PI * g->CentrePos(MAC::X, j)) * std::sin(M_PI * g->FacePos(MAC::Y, i)));
}

static void MarkInteriorFluid(MAC *g)
{
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx; j++)
            g->SetSolid(i, j, (i == 0 || j == 0 || i == g->Ny - 1 || j == g->Nx - 1) ? SOLID_CELL : FLUID_CELL);
}

// Carve an air pocket so the free-surface branch of the stencil is exercised.
static void CarveAirRegion(MAC *g)
{
    for (int i = g->Ny / 2; i < g->Ny - 1; i++)
        for (int j = 1; j < g->Nx - 1; j++)
            g->SetSolid(i, j, EMPTY_CELL);
}

static double MaxDiffU(MAC *a, MAC *b)
{
    double m = 0.0;
    for (int i = 0; i < a->Ny; i++)
        for (int j = 0; j < a->Nx + 1; j++)
            m = std::max(m, std::abs(a->GetU(i, j) - b->GetU(i, j)));
    return m;
}

static double MaxDiffV(MAC *a, MAC *b)
{
    double m = 0.0;
    for (int i = 0; i < a->Ny + 1; i++)
        for (int j = 0; j < a->Nx; j++)
            m = std::max(m, std::abs(a->GetV(i, j) - b->GetV(i, j)));
    return m;
}

static bool Finite(MAC *g)
{
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx + 1; j++)
            if (!std::isfinite(g->GetU(i, j)))
                return false;
    for (int i = 0; i < g->Ny + 1; i++)
        for (int j = 0; j < g->Nx; j++)
            if (!std::isfinite(g->GetV(i, j)))
                return false;
    return true;
}

static double MaxAbs(MAC *g)
{
    double m = 0.0;
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx + 1; j++)
            m = std::max(m, std::abs(g->GetU(i, j)));
    for (int i = 0; i < g->Ny + 1; i++)
        for (int j = 0; j < g->Nx; j++)
            m = std::max(m, std::abs(g->GetV(i, j)));
    return m;
}

int run(int, char **)
{
    const int N = 32;

    DIMENSION = 2;
    SelectMethod("FLIP");
    THREAD_COUNT = 1;

    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = N;
    SIMULATION.RHO = 1.0;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0}; // no body force, so the two paths are comparable

    MAC gridIn, gridCG, gridADI;
    gridIn.InitializeGrid(SIMULATION.domain, true);
    gridCG.InitializeGrid(SIMULATION.domain, true);
    gridADI.InitializeGrid(SIMULATION.domain, true);

    SIMULATION.dh = gridIn.dh;
    SIMULATION.Nx = gridIn.Nx;
    SIMULATION.Ny = gridIn.Ny;
    SIMULATION.GRID_SOL = &gridCG;
    SIMULATION.GRID_ANT = &gridIn;

    DiffusionSolver2D::InitializeDiffusionSolver(&gridIn);

    struct Case { const char *name; bool air; };
    Case cases[] = {{"closed box", false}, {"free surface", true}};

    for (const Case &c : cases)
    {
        std::printf("\n== %s ==\n", c.name);

        // -------- ADI and CG must converge to each other as nu*dt/dh^2 -> 0 --------
        double prev = -1.0;
        bool shrinking = true;
        for (double beta : {1.0, 0.25, 0.0625, 0.015625})
        {
            SIMULATION.EPS = 1e-2;
            SIMULATION.dt = beta * SIMULATION.dh * SIMULATION.dh / SIMULATION.EPS;

            MarkInteriorFluid(&gridIn);
            if (c.air) CarveAirRegion(&gridIn);
            SeedField(&gridIn);
            gridCG.CopyGrid(gridIn);
            gridADI.CopyGrid(gridIn);

            DiffusionSolver2D::SolveDiffusion_Eigen(&gridIn, &gridCG);
            DiffusionSolver2D::SolveDiffusion_ADI(&gridIn, &gridADI);

            double d = std::max(MaxDiffU(&gridCG, &gridADI), MaxDiffV(&gridCG, &gridADI));
            std::printf("     nu*dt/dh^2 = %-9.4g  max|ADI - CG| = %.3e\n", beta, d);
            if (prev >= 0.0 && d > prev * 0.75)
                shrinking = false;
            prev = d;
        }
        check(shrinking, "ADI converges to CG as the splitting parameter shrinks",
              "each refinement must cut the gap by at least 25%");
        check(prev < 5e-4, "ADI and CG agree at small nu*dt/dh^2",
              "gap = " + std::to_string(prev));

        // -------- both stay bounded and finite at a viscosity that a FLIP run hits --------
        for (double beta : {1.0, 10.0, 100.0})
        {
            SIMULATION.EPS = 1e-2;
            SIMULATION.dt = beta * SIMULATION.dh * SIMULATION.dh / SIMULATION.EPS;

            MarkInteriorFluid(&gridIn);
            if (c.air) CarveAirRegion(&gridIn);
            SeedField(&gridIn);
            double before = MaxAbs(&gridIn);
            gridCG.CopyGrid(gridIn);
            gridADI.CopyGrid(gridIn);

            DiffusionSolver2D::SolveDiffusion_Eigen(&gridIn, &gridCG);
            DiffusionSolver2D::SolveDiffusion_ADI(&gridIn, &gridADI);

            std::string tag = " at nu*dt/dh^2 = " + std::to_string((int)beta);
            check(Finite(&gridCG), std::string("CG stays finite") + tag, "");
            check(Finite(&gridADI), std::string("ADI stays finite") + tag, "");
            // Diffusion is a contraction: no face may grow past the starting maximum.
            check(MaxAbs(&gridCG) <= before * 1.0000001,
                  std::string("CG does not amplify") + tag,
                  std::to_string(MaxAbs(&gridCG)) + " vs " + std::to_string(before));
            check(MaxAbs(&gridADI) <= before * 1.0000001,
                  std::string("ADI does not amplify") + tag,
                  std::to_string(MaxAbs(&gridADI)) + " vs " + std::to_string(before));
        }

        // -------- boundary and non-fluid faces must survive untouched --------
        SIMULATION.EPS = 1e-2;
        SIMULATION.dt = 0.25 * SIMULATION.dh * SIMULATION.dh / SIMULATION.EPS;
        MarkInteriorFluid(&gridIn);
        if (c.air) CarveAirRegion(&gridIn);
        SeedField(&gridIn);

        // Stamp a recognisable value on every face the solvers must not write.
        const double SENTINEL = -7.25;
        for (int i = 0; i < gridIn.Ny; i++)
        {
            gridIn.SetU(i, 0, SENTINEL);
            gridIn.SetU(i, 1, SENTINEL);
            gridIn.SetU(i, gridIn.Nx - 1, SENTINEL);
            gridIn.SetU(i, gridIn.Nx, SENTINEL);
        }
        for (int j = 0; j < gridIn.Nx; j++)
        {
            gridIn.SetV(0, j, SENTINEL);
            gridIn.SetV(1, j, SENTINEL);
            gridIn.SetV(gridIn.Ny - 1, j, SENTINEL);
            gridIn.SetV(gridIn.Ny, j, SENTINEL);
        }
        gridCG.CopyGrid(gridIn);
        gridADI.CopyGrid(gridIn);

        DiffusionSolver2D::SolveDiffusion_Eigen(&gridIn, &gridCG);
        DiffusionSolver2D::SolveDiffusion_ADI(&gridIn, &gridADI);

        bool cgKept = true, adiKept = true;
        for (int i = 0; i < gridIn.Ny; i++)
            for (int j : {0, 1, gridIn.Nx - 1, gridIn.Nx})
            {
                if (gridCG.GetU(i, j) != SENTINEL) cgKept = false;
                if (gridADI.GetU(i, j) != SENTINEL) adiKept = false;
            }
        for (int j = 0; j < gridIn.Nx; j++)
            for (int i : {0, 1, gridIn.Ny - 1, gridIn.Ny})
            {
                if (gridCG.GetV(i, j) != SENTINEL) cgKept = false;
                if (gridADI.GetV(i, j) != SENTINEL) adiKept = false;
            }
        check(cgKept, "CG leaves the pinned boundary faces alone", "");
        check(adiKept, "ADI leaves the pinned boundary faces alone", "");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace diffusion2d_check

// ------------------------------------------------------------------------------------------------------------------
// diffusion3d_check
// ------------------------------------------------------------------------------------------------------------------
namespace diffusion3d_check
{
// Headless check for the 3D viscous solvers - the 3D twin of tests/diffusion2d_check.cpp.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail)
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}

static void SeedField(MAC *g)
{
    double dh = g->dh;
    auto f = [](double a, double b, double c) {
        return std::sin(M_PI * a) * std::sin(M_PI * b) * std::sin(M_PI * c);
    };
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx + 1; j++)
            for (int k = 0; k < g->Nz; k++)
                g->SetU(i, j, k, f(g->FacePos(MAC::X, j), g->CentrePos(MAC::Y, i), g->CentrePos(MAC::Z, k)));
    for (int i = 0; i < g->Ny + 1; i++)
        for (int j = 0; j < g->Nx; j++)
            for (int k = 0; k < g->Nz; k++)
                g->SetV(i, j, k, f(g->CentrePos(MAC::X, j), g->FacePos(MAC::Y, i), g->CentrePos(MAC::Z, k)));
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx; j++)
            for (int k = 0; k < g->Nz + 1; k++)
                g->SetW(i, j, k, f(g->CentrePos(MAC::X, j), g->CentrePos(MAC::Y, i), g->FacePos(MAC::Z, k)));
}

static void MarkInteriorFluid(MAC *g, bool air)
{
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx; j++)
            for (int k = 0; k < g->Nz; k++)
            {
                bool border = (i == 0 || j == 0 || k == 0 ||
                               i == g->Ny - 1 || j == g->Nx - 1 || k == g->Nz - 1);
                int t = border ? SOLID_CELL : FLUID_CELL;
                if (!border && air && i >= g->Ny / 2) t = EMPTY_CELL;
                g->SetSolid(i, j, k, t);
            }
}

static double MaxDiff(MAC *a, MAC *b)
{
    double m = 0.0;
    for (int i = 0; i < a->Ny; i++)
        for (int j = 0; j < a->Nx + 1; j++)
            for (int k = 0; k < a->Nz; k++)
                m = std::max(m, std::abs(a->GetU(i, j, k) - b->GetU(i, j, k)));
    for (int i = 0; i < a->Ny + 1; i++)
        for (int j = 0; j < a->Nx; j++)
            for (int k = 0; k < a->Nz; k++)
                m = std::max(m, std::abs(a->GetV(i, j, k) - b->GetV(i, j, k)));
    for (int i = 0; i < a->Ny; i++)
        for (int j = 0; j < a->Nx; j++)
            for (int k = 0; k < a->Nz + 1; k++)
                m = std::max(m, std::abs(a->GetW(i, j, k) - b->GetW(i, j, k)));
    return m;
}

static double MaxAbs(MAC *g)
{
    double m = 0.0;
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx + 1; j++)
            for (int k = 0; k < g->Nz; k++) m = std::max(m, std::abs(g->GetU(i, j, k)));
    for (int i = 0; i < g->Ny + 1; i++)
        for (int j = 0; j < g->Nx; j++)
            for (int k = 0; k < g->Nz; k++) m = std::max(m, std::abs(g->GetV(i, j, k)));
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx; j++)
            for (int k = 0; k < g->Nz + 1; k++) m = std::max(m, std::abs(g->GetW(i, j, k)));
    return m;
}

static bool Finite(MAC *g) { return std::isfinite(MaxAbs(g)); }

int run(int, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int N = 24; // 24^3, well inside the 60^3 ceiling

    DIMENSION = 3;
    SelectMethod("FLIP");
    THREAD_COUNT = 4;

    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = N;
    SIMULATION.RHO = 1.0;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    static MAC gridIn, gridCG, gridADI;
    gridIn.InitializeGrid(SIMULATION.domain, false);
    gridCG.InitializeGrid(SIMULATION.domain, false);
    gridADI.InitializeGrid(SIMULATION.domain, false);

    SIMULATION.dh = gridIn.dh;
    SIMULATION.Nx = gridIn.Nx;
    SIMULATION.Ny = gridIn.Ny;
    SIMULATION.Nz = gridIn.Nz;
    SIMULATION.GRID_SOL = &gridCG;
    SIMULATION.GRID_ANT = &gridIn;
    std::printf("grid %dx%dx%d\n", SIMULATION.Nx, SIMULATION.Ny, SIMULATION.Nz);

    MarkInteriorFluid(&gridIn, false);
    DiffusionSolver3D::InitializeDiffusionSolver(&gridIn);

    struct Case { const char *name; bool air; };
    for (const Case &c : {Case{"closed box", false}, Case{"free surface", true}})
    {
        std::printf("\n== %s ==\n", c.name);

        double prev = -1.0;
        bool shrinking = true;
        for (double beta : {1.0, 0.25, 0.0625, 0.015625})
        {
            SIMULATION.EPS = 1e-2;
            SIMULATION.dt = beta * SIMULATION.dh * SIMULATION.dh / SIMULATION.EPS;

            MarkInteriorFluid(&gridIn, c.air);
            SeedField(&gridIn);
            gridCG.CopyGrid(gridIn);
            gridADI.CopyGrid(gridIn);

            DiffusionSolver3D::SolveDiffusion_Eigen(&gridIn, &gridCG);
            DiffusionSolver3D::SolveDiffusion_ADI(&gridIn, &gridADI);

            double d = MaxDiff(&gridCG, &gridADI);
            std::printf("     nu*dt/dh^2 = %-9.4g  max|ADI - CG| = %.3e\n", beta, d);
            if (prev >= 0.0 && d > prev * 0.75) shrinking = false;
            prev = d;
        }
        check(shrinking, "ADI converges to CG as the splitting parameter shrinks", "");
        check(prev < 5e-3, "ADI and CG agree at small nu*dt/dh^2", "gap = " + std::to_string(prev));

        for (double beta : {1.0, 10.0, 100.0})
        {
            SIMULATION.EPS = 1e-2;
            SIMULATION.dt = beta * SIMULATION.dh * SIMULATION.dh / SIMULATION.EPS;
            MarkInteriorFluid(&gridIn, c.air);
            SeedField(&gridIn);
            double before = MaxAbs(&gridIn);
            gridCG.CopyGrid(gridIn);
            gridADI.CopyGrid(gridIn);
            DiffusionSolver3D::SolveDiffusion_Eigen(&gridIn, &gridCG);
            DiffusionSolver3D::SolveDiffusion_ADI(&gridIn, &gridADI);

            std::string tag = " at nu*dt/dh^2 = " + std::to_string((int)beta);
            check(Finite(&gridCG), "CG stays finite" + tag, "");
            check(Finite(&gridADI), "ADI stays finite" + tag, "");
            check(MaxAbs(&gridCG) <= before * 1.0000001, "CG does not amplify" + tag,
                  std::to_string(MaxAbs(&gridCG)) + " vs " + std::to_string(before));
            check(MaxAbs(&gridADI) <= before * 1.0000001, "ADI does not amplify" + tag,
                  std::to_string(MaxAbs(&gridADI)) + " vs " + std::to_string(before));
        }

        // Boundary faces SetBorder owns must come back untouched.
        SIMULATION.EPS = 1e-2;
        SIMULATION.dt = 0.25 * SIMULATION.dh * SIMULATION.dh / SIMULATION.EPS;
        MarkInteriorFluid(&gridIn, c.air);
        SeedField(&gridIn);
        const double S = -7.25;
        int Nx = gridIn.Nx, Ny = gridIn.Ny, Nz = gridIn.Nz;
        for (int i = 0; i < Ny; i++) for (int k = 0; k < Nz; k++)
            for (int j : {0, 1, Nx - 1, Nx}) gridIn.SetU(i, j, k, S);
        for (int j = 0; j < Nx; j++) for (int k = 0; k < Nz; k++)
            for (int i : {0, 1, Ny - 1, Ny}) gridIn.SetV(i, j, k, S);
        for (int i = 0; i < Ny; i++) for (int j = 0; j < Nx; j++)
            for (int k : {0, 1, Nz - 1, Nz}) gridIn.SetW(i, j, k, S);
        gridCG.CopyGrid(gridIn);
        gridADI.CopyGrid(gridIn);
        DiffusionSolver3D::SolveDiffusion_Eigen(&gridIn, &gridCG);
        DiffusionSolver3D::SolveDiffusion_ADI(&gridIn, &gridADI);

        bool cgKept = true, adiKept = true;
        for (int i = 0; i < Ny; i++) for (int k = 0; k < Nz; k++)
            for (int j : {0, 1, Nx - 1, Nx}) {
                if (gridCG.GetU(i, j, k) != S) cgKept = false;
                if (gridADI.GetU(i, j, k) != S) adiKept = false; }
        for (int j = 0; j < Nx; j++) for (int k = 0; k < Nz; k++)
            for (int i : {0, 1, Ny - 1, Ny}) {
                if (gridCG.GetV(i, j, k) != S) cgKept = false;
                if (gridADI.GetV(i, j, k) != S) adiKept = false; }
        for (int i = 0; i < Ny; i++) for (int j = 0; j < Nx; j++)
            for (int k : {0, 1, Nz - 1, Nz}) {
                if (gridCG.GetW(i, j, k) != S) cgKept = false;
                if (gridADI.GetW(i, j, k) != S) adiKept = false; }
        check(cgKept, "CG leaves the pinned boundary faces alone", "");
        check(adiKept, "ADI leaves the pinned boundary faces alone", "");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace diffusion3d_check

static const SuiteCase CASES[] = {
    {"FLIP 2D dam break", flip2d_dambreak_check::run, {}, false},
    {"FLIP 3D dam break", flip3d_dambreak_check::run, {"60"}, false},
    {"FLIP 2D tank at rest", flip2d_hydrostatic_check::run, {"100"}, false},
    {"FLIP 3D tank at rest", flip3d_hydrostatic_check::run, {"60"}, false},
    {"FLIP 2D obstacle contact", flip2d_obstacle_check::run, {}, false},
    {"FLIP 3D obstacle contact", flip3d_obstacle_check::run, {"250"}, false},
    {"FLIP solid faces", flip_solidflux_check::run, {"300"}, false},
    {"FLIP density correction", flip_density_check::run, {}, false},
    {"viscous solve 2D", diffusion2d_check::run, {}, false},
    {"viscous solve 3D", diffusion3d_check::run, {}, false},
};

int main(int argc, char **argv) { return RunSuite("flip", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
