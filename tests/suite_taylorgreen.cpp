// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The taylorgreen suite: taylorgreen_check, nonuniform_tgv_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "Functions.h"
#include "headers/Core/Scene.h"
#include "Methods/ADI/ADI2D.h"
#include "Methods/PPESolver/PressureSolver2D.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// taylorgreen_check
// ------------------------------------------------------------------------------------------------------------------
namespace taylorgreen_check
{
// Taylor-Green vortex: the whole boundary pipeline against an exact solution.
//
// This is the case the prescribed-pressure work was for. Every face carries the analytic
// velocity AND the analytic pressure, so the interior error is the discretisation error
// alone -- there is no boundary approximation mixed into it. It exercises, end to end:
// physical (non-zero-origin) coordinates, expression-driven velocity and pressure
// boundaries, PRESSURE_CELL as a non-zero Dirichlet condition in the matrix, and the ADI
// momentum step.
//
//   u = -cos(x) sin(y) exp(-2 t / Re)
//   v =  sin(x) cos(y) exp(-2 t / Re)
//   p = -(1/4)(cos 2x + cos 2y) exp(-4 t / Re)      nu = 1/Re, rho = 1


static int failures = 0;
static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}

static const double RE = 100.0;
static const double PI = 3.14159265358979323846;

static double ExactU(double x, double y, double t) { return -std::cos(x) * std::sin(y) * std::exp(-2.0 * t / RE); }
static double ExactV(double x, double y, double t) { return  std::sin(x) * std::cos(y) * std::exp(-2.0 * t / RE); }
static double ExactP(double x, double y, double t)
{ return -0.25 * (std::cos(2 * x) + std::cos(2 * y)) * std::exp(-4.0 * t / RE); }

struct Result { double errU = 0, errV = 0, errP = 0, dh = 0, worstU = 0; int Nx = 0, Ny = 0, wi = 0, wj = 0; };

static Result Run(int gridSize, int steps, double dt)
{
    DIMENSION = 2; SelectMethod("ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = gridSize;
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1.0 / RE;          // nu = 1/Re, matching the scene's exponents
    SIMULATION.dt = dt; SIMULATION.dtConfigured = dt; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = true;    // velocity on every face: pure Neumann, p= sets the level
    // PROJECTION=incremental in the environment selects the incremental pressure correction.
    ADI_SETTINGS.projectionScheme = (std::getenv("PROJECTION") && std::string(std::getenv("PROJECTION")) == "incremental")
        ? PROJECTION_INCREMENTAL : PROJECTION_CHORIN;
    ADI_SETTINGS.convectionForm = (std::getenv("FORM") && std::string(std::getenv("FORM")) == "flux")
        ? CONVECTION_FLUX : CONVECTION_ADVECTIVE;
    PRESSURE_SETTINGS.warmStart = false;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    static MAC gridSol, gridAnt;
    gridSol = MAC(); gridAnt = MAC();
    gridSol.InitializeGrid(SIMULATION.domain, true);
    gridAnt.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = gridSol.dh; SIMULATION.Nx = gridSol.Nx; SIMULATION.Ny = gridSol.Ny;
    SIMULATION.GRID_SOL = &gridSol; SIMULATION.GRID_ANT = &gridAnt;

    if (!SCENE.Load("tests/scenes/taylorgreen_2d.scene"))
    { check(false, "scene loads", SCENE.error); return Result(); }
    SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = SCENE_SOLID_MASK_2D;
    gridSol.SetSolidMask(SCENE_SOLID_MASK_2D);
    gridAnt.SetSolidMask(SCENE_SOLID_MASK_2D);

    // The grid's own origin: the border ring lies one cell outside the domain.
    const double dh = gridSol.dh, x0 = gridSol.FacePos(MAC::X, 0), y0 = gridSol.FacePos(MAC::Y, 0);

    // Initial condition: the analytic field at t = 0, at each quantity's own location.
    for (int i = 0; i < gridAnt.Ny; i++)
        for (int j = 0; j < gridAnt.Nx + 1; j++)
            gridAnt.SetU(i, j, ExactU(x0 + j * dh, y0 + (i + 0.5) * dh, 0.0));
    for (int i = 0; i < gridAnt.Ny + 1; i++)
        for (int j = 0; j < gridAnt.Nx; j++)
            gridAnt.SetV(i, j, ExactV(x0 + (j + 0.5) * dh, y0 + i * dh, 0.0));
    for (int i = 0; i < gridAnt.Ny; i++)
        for (int j = 0; j < gridAnt.Nx; j++)
            gridAnt.SetP(i, j, ExactP(x0 + (j + 0.5) * dh, y0 + (i + 0.5) * dh, 0.0));

    ADI2D::InitializeADI2D(&gridSol, dt, SIMULATION.VelocityBoundaryFunction2D,
                            SIMULATION.PressureBoundaryFunction2D);
    // EIGEN=cg | amg in the environment appends eigen_solver to the pressure settings (the last key wins).
    if (std::getenv("EIGEN"))
    {
        std::string text = PressureConfigText();
        text += std::string("\neigen_solver=") + std::getenv("EIGEN") + "\n";
        PRESSURE_SETTINGS.text = text;
    }
    PressureSolver2D::InitializePressureSolver(&gridSol, ADIPressureScheme());
    gridAnt.SetBoundaryFunctions(SCENE_VELOCITY_2D, SCENE_PRESSURE_2D);
    gridAnt.EnforceBoundaryConditions(0.0);

    for (int step = 0; step < steps; step++)
    {
        const double t = step * dt;
        // What the manager does: the border conditions belong to the field being solved
        // for, so they are evaluated at t + dt.
        SIMULATION.time = t + dt;
        gridSol.ClearFlowFields();
        ADI2D::SolveADIStep(&gridAnt, &gridSol, nullptr, t);
        PressureSolver2D::SolvePressure(&gridAnt, &gridSol, nullptr, 0.0);
        PressureSolver2D::Project(&gridAnt, &gridSol, nullptr, t);   // + p^n under INCREMENTAL
        gridAnt.CopyGrid(gridSol);
    }

    const double T = steps * dt;
    Result r; r.dh = dh; r.Nx = gridSol.Nx; r.Ny = gridSol.Ny;
    // Worst cell, so a boundary-only failure is distinguishable from a bulk one.
    double worst = 0.0; int wi = -1, wj = -1;
    for (int i = 1; i < gridSol.Ny - 1; i++)
        for (int j = 2; j < gridSol.Nx - 1; j++)
        {
            const double e = std::abs(gridSol.GetU(i, j) - ExactU(x0 + j * dh, y0 + (i + 0.5) * dh, T));
            if (e > worst) { worst = e; wi = i; wj = j; }
        }
    r.worstU = worst; r.wi = wi; r.wj = wj;
    long n = 0;
    // Interior only: the border carries the exact answer by construction.
    for (int i = 1; i < gridSol.Ny - 1; i++)
        for (int j = 2; j < gridSol.Nx - 1; j++)
        {
            const double e = gridSol.GetU(i, j) - ExactU(x0 + j * dh, y0 + (i + 0.5) * dh, T);
            r.errU += e * e; n++;
        }
    r.errU = std::sqrt(r.errU / (double)(n ? n : 1));
    n = 0;
    for (int i = 2; i < gridSol.Ny - 1; i++)
        for (int j = 1; j < gridSol.Nx - 1; j++)
        {
            const double e = gridSol.GetV(i, j) - ExactV(x0 + (j + 0.5) * dh, y0 + i * dh, T);
            r.errV += e * e; n++;
        }
    r.errV = std::sqrt(r.errV / (double)(n ? n : 1));
    // The pressure of a projection method is defined up to a constant only when the system
    // is pure Neumann; here it is pinned, so compare it directly.
    n = 0;
    for (int i = 1; i < gridSol.Ny - 1; i++)
        for (int j = 1; j < gridSol.Nx - 1; j++)
        {
            const double e = gridSol.GetP(i, j) - ExactP(x0 + (j + 0.5) * dh, y0 + (i + 0.5) * dh, T);
            r.errP += e * e; n++;
        }
    r.errP = std::sqrt(r.errP / (double)(n ? n : 1));

    PressureSolver2D::DestroyPressureSolver();
    return r;
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int steps = (argc > 1) ? std::atoi(argv[1]) : 100;
    const double dt = 0.005;

    std::printf("\n== Taylor-Green vortex on [0,1]^2, Re = %.0f, t = %.3f ==\n", RE, steps * dt);
    std::printf("  %6s %8s %12s %12s %12s\n", "N", "dh", "L2 err u", "L2 err v", "L2 err p");

    // dt is halved with dh so the CFL is held fixed. Refining space alone leaves the ADI
    // splitting error -- which is a function of dt, not dh -- as a floor, and the observed
    // spatial order comes out around 1 for the velocity because of it rather than because
    // of the discretisation.
    Result coarse = Run(32, steps, dt);
    std::printf("  %6d %8.4f %12.3e %12.3e %12.3e   worst|u| %.3e at (%d,%d)\n", coarse.Nx, coarse.dh,
                coarse.errU, coarse.errV, coarse.errP, coarse.worstU, coarse.wi, coarse.wj);
    Result fine = Run(64, steps * 2, dt / 2.0);
    std::printf("  %6d %8.4f %12.3e %12.3e %12.3e   worst|u| %.3e at (%d,%d)\n", fine.Nx, fine.dh,
                fine.errU, fine.errV, fine.errP, fine.worstU, fine.wi, fine.wj);

    const double orderU = std::log(coarse.errU / fine.errU) / std::log(2.0);
    const double orderV = std::log(coarse.errV / fine.errV) / std::log(2.0);
    const double orderP = std::log(coarse.errP / fine.errP) / std::log(2.0);
    std::printf("  observed order:  u %.2f   v %.2f   p %.2f\n", orderU, orderV, orderP);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "u %.3e, v %.3e at N = %d", fine.errU, fine.errV, fine.Nx);
    check(fine.errU < 5e-3 && fine.errV < 5e-3, "the velocity tracks the exact solution", buf);

    std::snprintf(buf, sizeof(buf), "coarse %.3e -> fine %.3e", coarse.errU, fine.errU);
    check(fine.errU < coarse.errU, "and refining the grid reduces the error", buf);

    std::snprintf(buf, sizeof(buf), "p error %.3e -- a prescribed border cannot leave p adrift",
                  fine.errP);
    check(fine.errP < 5e-2, "the pressure is anchored, not merely defined up to a constant", buf);

    std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
} // namespace taylorgreen_check

// ------------------------------------------------------------------------------------------------------------------
// nonuniform_tgv_check
// ------------------------------------------------------------------------------------------------------------------
namespace nonuniform_tgv_check
{
// Taylor-Green vortex on a STRETCHED grid.
//
// The exact solution is the only honest way to check a rectilinear implementation: a
// wrong metric factor still produces a smooth, plausible-looking field, and only a
// comparison against the closed form catches it. Coordinates come from the grid tables
// (Xf/Xc/Yf/Yc) rather than x0 + j*dh, because on a stretched grid those differ -- which
// is the whole point.
//
// This is the case the prescribed-pressure work was for. Every face carries the analytic
// velocity AND the analytic pressure, so the interior error is the discretisation error
// alone -- there is no boundary approximation mixed into it. It exercises, end to end:
// physical (non-zero-origin) coordinates, expression-driven velocity and pressure
// boundaries, PRESSURE_CELL as a non-zero Dirichlet condition in the matrix, and the ADI
// momentum step.
//
//   u = -cos(x) sin(y) exp(-2 t / Re)
//   v =  sin(x) cos(y) exp(-2 t / Re)
//   p = -(1/4)(cos 2x + cos 2y) exp(-4 t / Re)      nu = 1/Re, rho = 1


static int failures = 0;
static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}

static const double RE = 100.0;
static const double PI = 3.14159265358979323846;

static double ExactU(double x, double y, double t) { return -std::cos(x) * std::sin(y) * std::exp(-2.0 * t / RE); }
static double ExactV(double x, double y, double t) { return  std::sin(x) * std::cos(y) * std::exp(-2.0 * t / RE); }
static double ExactP(double x, double y, double t)
{ return -0.25 * (std::cos(2 * x) + std::cos(2 * y)) * std::exp(-4.0 * t / RE); }

struct Result { double errU = 0, errV = 0, errP = 0, dh = 0, worstU = 0; int Nx = 0, Ny = 0, wi = 0, wj = 0; };

static Result Run(int gridSize, int steps, double dt, bool refine, double a = 0.0, double b = 0.0, int f = 1, int mode = REFINE_HARD)
{
    DIMENSION = 2; SelectMethod("ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = gridSize;
    SIMULATION.grid.nonUniform = refine;
    SIMULATION.grid.refineX.clear();
    SIMULATION.grid.refineY.clear();
    SIMULATION.grid.refineZ.clear();
    if (refine)
        SIMULATION.grid.refineX.push_back(GridRefinement{true, a, b, f, mode});
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1.0 / RE;          // nu = 1/Re, matching the scene's exponents
    SIMULATION.dt = dt; SIMULATION.dtConfigured = dt; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = true;    // velocity on every face: pure Neumann, p= sets the level
    PRESSURE_SETTINGS.warmStart = false;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    static MAC gridSol, gridAnt;
    gridSol = MAC(); gridAnt = MAC();
    gridSol.InitializeGrid(SIMULATION.domain, true);
    gridAnt.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = gridSol.dh; SIMULATION.Nx = gridSol.Nx; SIMULATION.Ny = gridSol.Ny;
    SIMULATION.GRID_SOL = &gridSol; SIMULATION.GRID_ANT = &gridAnt;

    if (!SCENE.Load("tests/scenes/taylorgreen_2d.scene"))
    { check(false, "scene loads", SCENE.error); return Result(); }
    SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = SCENE_SOLID_MASK_2D;
    gridSol.SetSolidMask(SCENE_SOLID_MASK_2D);
    gridAnt.SetSolidMask(SCENE_SOLID_MASK_2D);

    const double dh = gridSol.dh, x0 = SIMULATION.domain.x0, y0 = SIMULATION.domain.y0;

    // Initial condition: the analytic field at t = 0, at each quantity's own location.
    for (int i = 0; i < gridAnt.Ny; i++)
        for (int j = 0; j < gridAnt.Nx + 1; j++)
            gridAnt.SetU(i, j, ExactU(gridAnt.FacePos(MAC::X, j), gridAnt.CentrePos(MAC::Y, i), 0.0));
    for (int i = 0; i < gridAnt.Ny + 1; i++)
        for (int j = 0; j < gridAnt.Nx; j++)
            gridAnt.SetV(i, j, ExactV(gridAnt.CentrePos(MAC::X, j), gridAnt.FacePos(MAC::Y, i), 0.0));
    for (int i = 0; i < gridAnt.Ny; i++)
        for (int j = 0; j < gridAnt.Nx; j++)
            gridAnt.SetP(i, j, ExactP(gridAnt.CentrePos(MAC::X, j), gridAnt.CentrePos(MAC::Y, i), 0.0));

    ADI2D::InitializeADI2D(&gridSol, dt, SIMULATION.VelocityBoundaryFunction2D,
                            SIMULATION.PressureBoundaryFunction2D);
    PressureSolver2D::InitializePressureSolver(&gridSol, ADIPressureScheme());
    gridAnt.SetBoundaryFunctions(SCENE_VELOCITY_2D, SCENE_PRESSURE_2D);
    gridAnt.EnforceBoundaryConditions(0.0);

    for (int step = 0; step < steps; step++)
    {
        const double t = step * dt;
        // What the manager does: the border conditions belong to the field being solved
        // for, so they are evaluated at t + dt.
        SIMULATION.time = t + dt;
        gridSol.ClearFlowFields();
        ADI2D::SolveADIStep(&gridAnt, &gridSol, nullptr, t);
        PressureSolver2D::SolvePressure(nullptr, &gridSol, nullptr, 0.0);
        PressureSolver2D::ProjectPressure(&gridSol);
        gridAnt.CopyGrid(gridSol);
    }

    const double T = steps * dt;
    Result r; r.dh = dh; r.Nx = gridSol.Nx; r.Ny = gridSol.Ny;
    // Worst cell, so a boundary-only failure is distinguishable from a bulk one.
    double worst = 0.0; int wi = -1, wj = -1;
    for (int i = 1; i < gridSol.Ny - 1; i++)
        for (int j = 2; j < gridSol.Nx - 1; j++)
        {
            const double e = std::abs(gridSol.GetU(i, j) - ExactU(gridSol.FacePos(MAC::X, j), gridSol.CentrePos(MAC::Y, i), T));
            if (e > worst) { worst = e; wi = i; wj = j; }
        }
    r.worstU = worst; r.wi = wi; r.wj = wj;
    if (getenv("PROFILE"))
    {
        std::printf("    column profile (j, x, dx, max|err u|):\n");
        for (int j = 2; j < gridSol.Nx - 1; j++)
        {
            double m = 0.0;
            for (int i = 1; i < gridSol.Ny - 1; i++)
                m = std::max(m, std::abs(gridSol.GetU(i,j) - ExactU(gridSol.FacePos(MAC::X, j), gridSol.CentrePos(MAC::Y, i), T)));
            std::printf("      %3d  %7.4f  %7.5f  %.3e\n", j, gridSol.FacePos(MAC::X, j), gridSol.Width(MAC::X, j), m);
        }
    }
    long n = 0;
    // Interior only: the border carries the exact answer by construction.
    for (int i = 1; i < gridSol.Ny - 1; i++)
        for (int j = 2; j < gridSol.Nx - 1; j++)
        {
            const double e = gridSol.GetU(i, j) - ExactU(gridSol.FacePos(MAC::X, j), gridSol.CentrePos(MAC::Y, i), T);
            r.errU += e * e; n++;
        }
    r.errU = std::sqrt(r.errU / (double)(n ? n : 1));
    n = 0;
    for (int i = 2; i < gridSol.Ny - 1; i++)
        for (int j = 1; j < gridSol.Nx - 1; j++)
        {
            const double e = gridSol.GetV(i, j) - ExactV(gridSol.CentrePos(MAC::X, j), gridSol.FacePos(MAC::Y, i), T);
            r.errV += e * e; n++;
        }
    r.errV = std::sqrt(r.errV / (double)(n ? n : 1));
    // The pressure of a projection method is defined up to a constant only when the system
    // is pure Neumann; here it is pinned, so compare it directly.
    n = 0;
    for (int i = 1; i < gridSol.Ny - 1; i++)
        for (int j = 1; j < gridSol.Nx - 1; j++)
        {
            const double e = gridSol.GetP(i, j) - ExactP(gridSol.CentrePos(MAC::X, j), gridSol.CentrePos(MAC::Y, i), T);
            r.errP += e * e; n++;
        }
    r.errP = std::sqrt(r.errP / (double)(n ? n : 1));

    PressureSolver2D::DestroyPressureSolver();
    return r;
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int steps = (argc > 1) ? std::atoi(argv[1]) : 100;
    const double dt = 1.0 / 3200.0;

    std::printf("\n== Taylor-Green on [0,1]^2, uniform vs stretched ==\n");
    Result uni = Run(32, steps, dt, false);
    std::printf("  uniform    %3dx%-3d dh=%.5f  L2 u=%.3e v=%.3e p=%.3e\n",
                uni.Nx, uni.Ny, uni.dh, uni.errU, uni.errV, uni.errP);

    // Isolation: every non-uniform branch active, but the spacing still equal. Any
    // difference from uniform here is a defect in a branch, not in the stretching.
    Result flat = Run(32, steps, dt, true, 0.0, 1.0, 1);
    std::printf("  nu-flat    %3dx%-3d dh_min=%.5f  L2 u=%.3e v=%.3e p=%.3e\n",
                flat.Nx, flat.Ny, flat.dh, flat.errU, flat.errV, flat.errP);

    // Refine the middle band in x by 2. Same physics, same exact solution.
    Result nu = Run(32, steps, dt, true, 0.35, 0.65, 2);
    std::printf("  stretched  %3dx%-3d dh_min=%.5f  L2 u=%.3e v=%.3e p=%.3e\n",
                nu.Nx, nu.Ny, nu.dh, nu.errU, nu.errV, nu.errP);

    // Same refinement at a finer base, to see the error actually fall.
    Result nu2 = Run(64, steps * 2, dt / 2.0, true, 0.35, 0.65, 2);
    std::printf("  stretched  %3dx%-3d dh_min=%.5f  L2 u=%.3e v=%.3e p=%.3e\n",
                nu2.Nx, nu2.Ny, nu2.dh, nu2.errU, nu2.errV, nu2.errP);

    // Graded: the same interval, but the size ramps to the target instead of jumping.
    Result gr = Run(32, steps, dt, true, 0.35, 0.65, 2, REFINE_SYMMETRIC);
    std::printf("  graded     %3dx%-3d dh_min=%.5f  L2 u=%.3e v=%.3e p=%.3e\n",
                gr.Nx, gr.Ny, gr.dh, gr.errU, gr.errV, gr.errP);

    std::printf("\n== checks ==\n");
    {
        char b2[200];
        std::snprintf(b2, sizeof(b2), "uniform %.3e vs nu-flat %.3e", uni.errU, flat.errU);
        check(std::abs(flat.errU - uni.errU) < 1e-12 * std::max(1.0, uni.errU),
              "with equal spacing the rectilinear paths reproduce the uniform answer", b2);
    }
    check(std::isfinite(nu.errU) && std::isfinite(nu.errV) && std::isfinite(nu.errP),
          "the stretched run stays finite");
    check(nu.Nx > uni.Nx && nu.Ny == uni.Ny,
          "refinement added columns and left the rows alone",
          std::to_string(uni.Nx) + "x" + std::to_string(uni.Ny) + " -> " +
          std::to_string(nu.Nx) + "x" + std::to_string(nu.Ny));

    // The real test. A wrong metric factor anywhere -- diffusion, convection, a boundary
    // row, the pressure coefficient, the interpolation -- shows up here as an error that
    // is orders of magnitude worse than uniform, not as a crash.
    char buf[200];
    std::snprintf(buf, sizeof(buf), "uniform %.3e vs stretched %.3e (ratio %.2f)",
                  uni.errU, nu.errU, nu.errU / uni.errU);
    check(nu.errU < 3.0 * uni.errU, "stretched velocity error stays the same order as uniform", buf);
    std::snprintf(buf, sizeof(buf), "uniform %.3e vs stretched %.3e (ratio %.2f)",
                  uni.errP, nu.errP, nu.errP / uni.errP);
    check(nu.errP < 3.0 * uni.errP, "stretched pressure error stays the same order as uniform", buf);

    check(std::isfinite(gr.errU) && gr.errU < 3.0 * uni.errU,
          "a graded interval solves as well as a hard one",
          "graded " + std::to_string(gr.errU) + " vs uniform " + std::to_string(uni.errU));

    const double rate = std::log(nu.errU / nu2.errU) / std::log(2.0);
    std::snprintf(buf, sizeof(buf), "%.3e -> %.3e, observed order %.2f", nu.errU, nu2.errU, rate);
    check(nu2.errU < nu.errU, "refining the stretched grid reduces the error", buf);

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace nonuniform_tgv_check

static const SuiteCase CASES[] = {
    {"Taylor-Green, uniform", taylorgreen_check::run, {"100"}, false},
    {"Taylor-Green, stretched", nonuniform_tgv_check::run, {}, false},
};

int main(int argc, char **argv) { return RunSuite("taylorgreen", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
