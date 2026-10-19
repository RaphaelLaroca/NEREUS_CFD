// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The adi_obstacle suite: adi2d_obstacle_check, adi3d_obstacle_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "Functions.h"
#include "Methods/ADI/ADI2D.h"
#include "Methods/PPESolver/PressureSolver2D.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include "Methods/ADI/ADI3D.h"
#include "Methods/PPESolver/PressureSolver3D.h"
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// adi2d_obstacle_check
// ------------------------------------------------------------------------------------------------------------------
namespace adi2d_obstacle_check
{
// Regression guard for the validated Eulerian path. Runs the 2D ADI obstacle case
// through the real momentum/pressure/projection pipeline on the GPU solver and prints
// field checksums, so a build before and after a pressure-solver change can be compared.


int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int steps = (argc > 1) ? std::atoi(argv[1]) : 50;

    DIMENSION = 2;
    SelectMethod("ADI");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 1;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 1.1, 0.0, 0.41, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 50;
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.002;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    SIMULATION.CHARACTERISTIC_LENGTH = 0.041;
    SIMULATION.MEAN_VELOCITY = 0.2;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    SIMULATION.VelocityBoundaryFunction2D = OBSTACLE_FLOW_2D;
    SIMULATION.PressureBoundaryFunction2D = OBSTACLE_FLOW_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = OBSTACLE_SOLID_MASK_2D;

    static MAC gridSol, gridAnt;
    gridSol.InitializeGrid(SIMULATION.domain, true);
    gridAnt.InitializeGrid(SIMULATION.domain, true);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    gridSol.SetSolidMask(OBSTACLE_SOLID_MASK_2D);
    gridAnt.SetSolidMask(OBSTACLE_SOLID_MASK_2D);

    ADI2D::InitializeADI2D(&gridSol, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction2D,
                            SIMULATION.PressureBoundaryFunction2D);
    PressureSolver2D::InitializePressureSolver(&gridSol, ADIPressureScheme());
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    gridAnt.EnforceBoundaryConditions(0);

    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        ADI2D::SolveADIStep(&gridAnt, &gridSol, nullptr, step * SIMULATION.dt);
        // AMGX (GPU_ACCELERATION = 1 at init), the backend the application binds on the GPU.
        PressureSolver2D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver2D::ProjectPressure(&gridSol);
        gridAnt.CopyGrid(gridSol);
    }

    double divSum = gridSol.GetDivSum();
    double maxVel = gridSol.GetMaxVelocity();
    double pSum = 0.0, uSum = 0.0, vSum = 0.0;
    // Interior only. The border ring's pressure is a display value -- the matrix excludes
    // the ring and the projection skips solid-adjacent faces -- and at a corner cell it
    // depends on the order the border passes run in, so it is not a solution quantity.
    double pInterior = 0.0;
    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx; j++)
        {
            pSum += gridSol.GetP(i, j);
            if (i > 0 && j > 0 && i < gridSol.Ny - 1 && j < gridSol.Nx - 1)
                pInterior += gridSol.GetP(i, j);
        }
    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx + 1; j++)
            uSum += std::abs(gridSol.GetU(i, j));
    for (int i = 0; i < gridSol.Ny + 1; i++)
        for (int j = 0; j < gridSol.Nx; j++)
            vSum += std::abs(gridSol.GetV(i, j));

    std::printf("steps=%d  divSum=%.12e  maxVel=%.12f\n", steps, divSum, maxVel);
    std::printf("sum|u|=%.10f  sum|v|=%.10f  sum p=%.10f  sum p interior=%.10f\n", uSum, vSum, pSum, pInterior);
    return 0;
}
} // namespace adi2d_obstacle_check

// ------------------------------------------------------------------------------------------------------------------
// adi3d_obstacle_check
// ------------------------------------------------------------------------------------------------------------------
namespace adi3d_obstacle_check
{
// Regression guard for the validated 3D Eulerian path. Runs the 3D ADI obstacle case
// through the real momentum/pressure/projection pipeline and prints field checksums, so a
// build before and after a change to the shared pressure solver can be diffed.


int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int steps = (argc > 1) ? std::atoi(argv[1]) : 30;

    DIMENSION = 3;
    SelectMethod("ADI");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 1;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 0.5, 0.0, 0.5};
    SIMULATION.GRID_SIZE = 24;              // gives 48x24x24, inside the 60^3 ceiling
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.002;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    SIMULATION.CHARACTERISTIC_LENGTH = 0.041;
    SIMULATION.MEAN_VELOCITY = 0.2;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    SIMULATION.VelocityBoundaryFunction = OBSTACLE_FLOW;
    SIMULATION.PressureBoundaryFunction = OBSTACLE_FLOW_PRESSURE;
    SIMULATION.SolidMaskFunction = OBSTACLE_SOLID_MASK;

    static MAC gridSol, gridAnt;
    gridSol.InitializeGrid(SIMULATION.domain, false);
    gridAnt.InitializeGrid(SIMULATION.domain, false);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.Nz = gridSol.Nz;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;
    std::printf("grid %dx%dx%d  dh=%.5f\n", SIMULATION.Nx, SIMULATION.Ny, SIMULATION.Nz, SIMULATION.dh);

    gridSol.SetSolidMask(OBSTACLE_SOLID_MASK);
    gridAnt.SetSolidMask(OBSTACLE_SOLID_MASK);

    ADI3D::InitializeADI(&gridSol, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction,
                          SIMULATION.PressureBoundaryFunction);
    PressureSolver3D::InitializePressureSolver(&gridSol, ADIPressureScheme());
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    gridAnt.EnforceBoundaryConditions(0);

    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        ADI3D::SolveADIStep(&gridAnt, &gridSol, nullptr, step * SIMULATION.dt);
        PressureSolver3D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver3D::ProjectPressure(&gridSol);
        gridAnt.CopyGrid(gridSol);
    }

    double uSum = 0.0, vSum = 0.0, wSum = 0.0, pSum = 0.0;
    // Interior only; see the 2D check for why the ring is not a solution quantity.
    double pInterior = 0.0;
    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx + 1; j++)
            for (int k = 0; k < gridSol.Nz; k++)
                uSum += std::abs(gridSol.GetU(i, j, k));
    for (int i = 0; i < gridSol.Ny + 1; i++)
        for (int j = 0; j < gridSol.Nx; j++)
            for (int k = 0; k < gridSol.Nz; k++)
                vSum += std::abs(gridSol.GetV(i, j, k));
    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx; j++)
            for (int k = 0; k < gridSol.Nz + 1; k++)
                wSum += std::abs(gridSol.GetW(i, j, k));
    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx; j++)
            for (int k = 0; k < gridSol.Nz; k++)
            {
                pSum += gridSol.GetP(i, j, k);
                if (i > 0 && j > 0 && k > 0 && i < gridSol.Ny - 1 &&
                    j < gridSol.Nx - 1 && k < gridSol.Nz - 1)
                    pInterior += gridSol.GetP(i, j, k);
            }

    std::printf("steps=%d  divSum=%.12e  maxVel=%.12f\n", steps, gridSol.GetDivSum(), gridSol.GetMaxVelocity());
    std::printf("sum|u|=%.10f  sum|v|=%.10f  sum|w|=%.10f  sum p=%.10f  sum p interior=%.10f\n", uSum, vSum, wSum, pSum, pInterior);
    return 0;
}
} // namespace adi3d_obstacle_check

static const SuiteCase CASES[] = {
    {"2D obstacle checksums", adi2d_obstacle_check::run, {"50"}, true},
    {"3D obstacle checksums", adi3d_obstacle_check::run, {"50"}, true},
};

int main(int argc, char **argv) { return RunSuite("adi_obstacle", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
