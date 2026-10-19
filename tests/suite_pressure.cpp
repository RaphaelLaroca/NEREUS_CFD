// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The pressure suite: pressure_dirichlet_check, pressure_warmstart_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <algorithm>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "Functions.h"
#include "headers/Core/Scene.h"
#include "Methods/PPESolver/PressureSolver2D.h"
#include "Methods/PPESolver/PressureSolver3D.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include <vector>
#include "Methods/ADI/ADI2D.h"
#include "Methods/ADI/ADI3D.h"
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// pressure_dirichlet_check
// ------------------------------------------------------------------------------------------------------------------
namespace pressure_dirichlet_check
{
// Prescribed (non-zero Dirichlet) pressure boundaries.
//
// EMPTY_CELL has always given a Dirichlet condition, but only p = 0 -- the free surface,
// and an open outflow. PRESSURE_CELL is the general case: a cell whose pressure is known
// and need not be zero. That is what a pressure-driven inlet needs, and what a convergence
// study against an analytic solution (Taylor-Green) needs on the border.
//
// The test is sharp because a LINEAR field is harmonic and reproduced exactly: the interior second difference of it
// is zero, and the boundary value sits on the boundary face, half a cell from the first centre, where the half-cell
// difference of a linear field is exact. (A quadratic such as x^2 - y^2 is not: that difference is its slope a
// quarter cell in, an O(h^2) error.) With zero velocity the right-hand side is zero apart from the Dirichlet term, so
// anything other than machine precision here is a defect in that term -- its sign, its half-cell distance, or its
// scaling against whichever way that solve path scales its matrix.


static int failures = 0;
static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}

// p = 0.3 + 1.1 x - 0.7 y : harmonic in 2D and in 3D.
#define EXACT_EXPR "0.3 + 1.1*x - 0.7*y"
static double Exact(double x, double y) { return 0.3 + 1.1 * x - 0.7 * y; }

// Where the value of cell (i, j[, k]) lives: a ring cell's on the boundary face it borders (MAC::PrescribedPressure),
// any other cell's at its centre.
static double RingPos(const MAC &g, MAC::Axis a, int n, int N)
{
    return n == 0 ? g.FacePos(a, 1) : (n == N - 1 ? g.FacePos(a, N - 1) : g.CentrePos(a, n));
}

static void WriteScene2D(const char *path, bool prescribe)
{
    std::ofstream f(path);
    f << "sampling corner\n";
    for (const char *face : {"xmin", "xmax", "ymin", "ymax"})
    {
        if (prescribe) f << "boundary " << face << " pressure p=\"" EXACT_EXPR "\"\n";
        else           f << "boundary " << face << " pressure_outlet\n";   // the p = 0 control
    }
}

// gpu: 0 = CPU (Eigen), 1 = AMGX with FLIP the active method, 2 = AMGX with ADI. One solve path since the
// pressure rework; the two AMGX modes differ only in the method the solver is initialised under.
static void Run2D(bool prescribe, int gpu, const char *label)
{
    const char *sceneFile = "dirichlet_2d.scene";
    WriteScene2D(sceneFile, prescribe);

    DIMENSION = 2; SelectMethod(gpu == 1 ? "FLIP" : "ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = (gpu != 0); ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 40;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 1e-3; SIMULATION.dt = 1e-3;
    SIMULATION.dtConfigured = 1e-3; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    PRESSURE_SETTINGS.warmStart = false;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    static MAC grid;
    grid = MAC();
    grid.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = grid.dh; SIMULATION.Nx = grid.Nx; SIMULATION.Ny = grid.Ny;
    SIMULATION.GRID_SOL = &grid; SIMULATION.GRID_ANT = &grid;

    if (!SCENE.Load(sceneFile)) { check(false, "scene loads", SCENE.error); return; }
    SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = SCENE_SOLID_MASK_2D;
    grid.SetSolidMask(SCENE_SOLID_MASK_2D);

    // Zero velocity everywhere: the divergence, and therefore the whole right-hand side
    // apart from the Dirichlet term, is exactly zero.
    grid.ClearFlowFields();
    grid.SetBoundaryFunctions(SCENE_VELOCITY_2D, SCENE_PRESSURE_2D);
    grid.EnforceBoundaryConditions(0.0); // writes the ring's p

    PressureSolver2D::InitializePressureSolver(&grid, ADIPressureScheme());

    // Solve twice and compare. The Dirichlet value used to be produced by evaluating the
    // scene expression inside the OpenMP right-hand side loop, and tinyexpr binds its
    // variables to shared addresses -- so the right-hand side itself came out different on
    // every run while the solver converged tidily to each wrong answer. Bit-identical
    // repeats are the direct guard against that coming back.
    std::vector<double> first;
    for (int pass = 0; pass < 2; pass++)
    {
        PressureSolver2D::SolvePressure(nullptr, &grid, nullptr, 0.0);

        if (pass == 0)
        {
            first.reserve((size_t)grid.Nx * grid.Ny);
            for (int i = 0; i < grid.Ny; i++)
                for (int j = 0; j < grid.Nx; j++)
                    first.push_back(grid.GetP(i, j));
        }
        else
        {
            size_t n = 0; double drift = 0.0;
            for (int i = 0; i < grid.Ny; i++)
                for (int j = 0; j < grid.Nx; j++, n++)
                    drift = std::max(drift, std::abs(grid.GetP(i, j) - first[n]));
            char db[128];
            std::snprintf(db, sizeof(db), "max drift between two identical solves = %.3e", drift);
            check(drift == 0.0, std::string(label) + ": repeating the solve is deterministic", db);
        }
    }

    double worst = 0.0, maxAbs = 0.0; int wi = -1, wj = -1; int bad = 0;
    for (int i = 1; i < grid.Ny - 1; i++)
        for (int j = 1; j < grid.Nx - 1; j++)
        {
            const double x = grid.CentrePos(MAC::X, j);
            const double y = grid.CentrePos(MAC::Y, i);
            const double want = prescribe ? Exact(x, y) : 0.0;
            const double e = std::abs(grid.GetP(i, j) - want);
            if (e > 1e-6) bad++;
            if (e > worst) { worst = e; wi = i; wj = j; }
            maxAbs = std::max(maxAbs, std::abs(grid.GetP(i, j)));
        }
    if (worst > 1e-6)
        std::printf("     worst at (%d,%d) of %dx%d: got %.6f want %.6f | %d of %d cells off\n",
                    wi, wj, grid.Ny, grid.Nx, grid.GetP(wi, wj),
                    Exact(grid.CentrePos(MAC::X, wj), grid.CentrePos(MAC::Y, wi)),
                    bad, (grid.Ny - 2) * (grid.Nx - 2));

    char buf[256];
    // The 2D Eigen path stops at 1e-9 relative, which leaves ~5e-8 here (AMGX, at its tighter configured tolerance,
    // lands on the field to ~3e-13, and 3D Eigen at 1e-12 to ~3e-12: the discretisation is exact). The AMGX bar is
    // loose because its tolerance is solver_pressure_config.txt's, the user's to change. The determinism check above
    // is the sharp one.
    const double bar = (gpu == 0) ? 1e-7 : 1e-5;
    std::snprintf(buf, sizeof(buf), "max|p - exact| = %.3e   (max|p| = %.4f, bar %.0e)",
                  worst, maxAbs, bar);
    check(worst < bar, std::string(label) + ": interior matches the prescribed field", buf);
    if (prescribe)
        check(maxAbs > 0.1, std::string(label) + ": and the field is genuinely non-zero", buf);

    PressureSolver2D::DestroyPressureSolver();
    std::remove(sceneFile);
}

static void Run3D()
{
    const char *sceneFile = "dirichlet_3d.scene";
    {
        std::ofstream f(sceneFile);
        f << "sampling corner\n";
        for (const char *face : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
            f << "boundary " << face << " pressure p=\"" EXACT_EXPR "\"\n";
    }

    DIMENSION = 3; SelectMethod("ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 20;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 1e-3; SIMULATION.dt = 1e-3;
    SIMULATION.dtConfigured = 1e-3; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    PRESSURE_SETTINGS.warmStart = false;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    static MAC grid;
    grid = MAC();
    grid.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.dh = grid.dh;
    SIMULATION.Nx = grid.Nx; SIMULATION.Ny = grid.Ny; SIMULATION.Nz = grid.Nz;
    SIMULATION.GRID_SOL = &grid; SIMULATION.GRID_ANT = &grid;

    if (!SCENE.Load(sceneFile)) { check(false, "3D scene loads", SCENE.error); return; }
    SIMULATION.VelocityBoundaryFunction = SCENE_VELOCITY;
    SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
    SIMULATION.SolidMaskFunction = SCENE_SOLID_MASK;
    grid.SetSolidMask(SCENE_SOLID_MASK);

    grid.ClearFlowFields();
    grid.SetBoundaryFunctions(SCENE_VELOCITY, SCENE_PRESSURE);
    grid.EnforceBoundaryConditions(0.0);

    PressureSolver3D::InitializePressureSolver(&grid, ADIPressureScheme());
    PressureSolver3D::SolvePressure(nullptr, &grid, nullptr, 0.0);

    double worst = 0.0, maxAbs = 0.0;
    for (int i = 1; i < grid.Ny - 1; i++)
        for (int j = 1; j < grid.Nx - 1; j++)
            for (int k = 1; k < grid.Nz - 1; k++)
            {
                const double x = grid.CentrePos(MAC::X, j);
                const double y = grid.CentrePos(MAC::Y, i);
                worst = std::max(worst, std::abs(grid.GetP(i, j, k) - Exact(x, y)));
                maxAbs = std::max(maxAbs, std::abs(grid.GetP(i, j, k)));
            }

    char buf[256];
    std::snprintf(buf, sizeof(buf), "max|p - exact| = %.3e   (max|p| = %.4f)", worst, maxAbs);
    check(worst < 1e-8, "3D Eigen: interior matches the prescribed field", buf);
    check(maxAbs > 0.1, "3D Eigen: and the field is genuinely non-zero", buf);

    PressureSolver3D::DestroyPressureSolver();
    std::remove(sceneFile);
}

// Incremental projection with prescribed-pressure faces (CPC plan A.9). The unknown is phi = p^{n+1} - p^n, so on a
// PRESSURE_CELL phi's known value is p_b(t^{n+1}) - p_b(t^n). The boundary is p_b = (1+t)(0.3 + 1.1x - 0.7y), harmonic at
// every t: starting from the exact p at t = 0, one solve + projection at t1 must land on (1+t1)(0.3 + 1.1x - 0.7y) exactly --
// interior and ring. Before the fix the solve saw p_b(t1) itself as phi's boundary and the face drifted by p_b.
static double ExactT(double x, double y, double z, double t) { return (1.0 + t) * Exact(x, y); }

static void RunIncremental(int dim, bool gpu, const char *label)
{
    const char *sceneFile = "dirichlet_inc.scene";
    {
        std::ofstream f(sceneFile);
        f << "sampling corner\n";
        const char *faces2[] = {"xmin", "xmax", "ymin", "ymax"};
        const char *faces3[] = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        for (int q = 0; q < 2 * dim; q++)
            f << "boundary " << (dim == 2 ? faces2[q] : faces3[q]) << " pressure p=\"(1+t)*(" EXACT_EXPR ")\"\n";
    }

    DIMENSION = dim; SelectMethod("ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = gpu; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = (dim == 2) ? 40 : 20;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 1e-3; SIMULATION.dt = 1e-3;
    SIMULATION.dtConfigured = 1e-3; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    PRESSURE_SETTINGS.warmStart = false;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    static MAC ga, gs;
    ga = MAC(); gs = MAC();
    ga.InitializeGrid(SIMULATION.domain, dim == 2);
    gs.InitializeGrid(SIMULATION.domain, dim == 2);
    SIMULATION.dh = ga.dh; SIMULATION.Nx = ga.Nx; SIMULATION.Ny = ga.Ny; SIMULATION.Nz = ga.Nz;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &ga;

    if (!SCENE.Load(sceneFile)) { check(false, std::string(label) + ": scene loads", SCENE.error); return; }
    SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.VelocityBoundaryFunction = SCENE_VELOCITY;
    SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
    const int nz = (dim == 2) ? 1 : ga.Nz;
    for (MAC *g : {&ga, &gs})
    {
        if (dim == 2) { g->SetSolidMask(SCENE_SOLID_MASK_2D); g->SetBoundaryFunctions(SCENE_VELOCITY_2D, SCENE_PRESSURE_2D); }
        else          { g->SetSolidMask(SCENE_SOLID_MASK);    g->SetBoundaryFunctions(SCENE_VELOCITY, SCENE_PRESSURE); }
        g->ClearFlowFields();
    }
    // p^n: the exact field at t = 0, ring included.
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
            for (int k = 0; k < nz; k++)
            {
                const double x = RingPos(ga, MAC::X, j, ga.Nx), y = RingPos(ga, MAC::Y, i, ga.Ny);
                if (dim == 2) ga.SetP(i, j, ExactT(x, y, 0.0, 0.0));
                else          ga.SetP(i, j, k, ExactT(x, y, 0.0, 0.0));
            }
    gs.CopyGrid(ga);

    const double t1 = 0.5;
    SIMULATION.time = t1;
    if (dim == 2)
    {
        PressureSolver2D::InitializePressureSolver(&gs, PRESSURE_INCREMENTAL);
        PressureSolver2D::SolvePressure(&ga, &gs, nullptr, 0.0);
        PressureSolver2D::Project(&ga, &gs, nullptr, 0.0);
    }
    else
    {
        PressureSolver3D::InitializePressureSolver(&gs, PRESSURE_INCREMENTAL);
        PressureSolver3D::SolvePressure(&ga, &gs, nullptr, 0.0);
        PressureSolver3D::Project(&ga, &gs, nullptr, 0.0);
    }

    double worst = 0.0, worstRing = 0.0;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            for (int k = 0; k < nz; k++)
            {
                const int t = (dim == 2) ? gs.GetSolid(i, j) : gs.GetSolid(i, j, k);
                const double p = (dim == 2) ? gs.GetP(i, j) : gs.GetP(i, j, k);
                const double e = std::abs(p - ExactT(RingPos(gs, MAC::X, j, gs.Nx), RingPos(gs, MAC::Y, i, gs.Ny), 0.0, t1));
                if (t == FLUID_CELL) worst = std::max(worst, e);
                else if (t == PRESSURE_CELL) worstRing = std::max(worstRing, e);
            }
    const double bar = gpu ? 1e-5 : 1e-7;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "max|p - exact(t1)| = %.3e (bar %.0e), ring %.3e", worst, bar, worstRing);
    check(worst < bar, std::string(label) + ": p^n + phi lands on the prescribed field at t1", buf);
    check(worstRing < 1e-12, std::string(label) + ": and the ring is back at p_b(t1)", buf);

    if (dim == 2) PressureSolver2D::DestroyPressureSolver(); else PressureSolver3D::DestroyPressureSolver();
    std::remove(sceneFile);
}

// AMGX does not survive initialize -> finalize -> initialize in one process ("Incorrect C
// API mode"), and every case here stands up its own solver. So the runner invokes this once
// per case and each process touches AMGX at most once.
int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int only = (argc > 1) ? std::atoi(argv[1]) : -1;
    auto want = [&](int c) { return only < 0 || only == c; };

    if (want(0) || want(1) || want(2) || want(3) || want(4))
        std::printf("\n== prescribed pressure, p = x^2 - y^2 (harmonic: exact for the stencil) ==\n");
    if (want(0)) Run2D(true, 0, "2D Eigen");
    if (want(1)) Run2D(true, 1, "2D AMGX (FLIP)");
    if (want(2)) Run2D(true, 2, "2D AMGX (ADI)");
    if (want(3)) Run3D();
    if (want(4)) Run2D(false, 1, "2D AMGX (FLIP), outflow (p must be exactly 0)");
    if (want(5) || want(6) || want(7))
        std::printf("\n== incremental: prescribed pressure p = (1+t)(x^2 - y^2), one step from the exact p^n ==\n");
    if (want(5)) RunIncremental(2, false, "2D Eigen, incremental");
    if (want(6)) RunIncremental(2, true, "2D AMGX, incremental");
    if (want(7)) RunIncremental(3, false, "3D Eigen, incremental");
    if (only >= 0 && only <= 7)
    {
        std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
        return failures ? 1 : 0;
    }

    std::printf("\n== the shipped Taylor-Green scene parses ==\n");
    {
        DIMENSION = 2;
        SIMULATION.domain = Domain{0.0, 6.283185307179586, 0.0, 6.283185307179586, 0.0, 1.0};
        SIMULATION.GRID_SIZE = 32;
        static MAC g; g = MAC();
        g.InitializeGrid(SIMULATION.domain, true);
        SIMULATION.dh = g.dh; SIMULATION.Nx = g.Nx; SIMULATION.Ny = g.Ny;

        const bool ok = SCENE.Load("tests/scenes/taylorgreen_2d.scene");
        check(ok, "tests/scenes/taylorgreen_2d.scene loads", ok ? "" : SCENE.error);
        if (ok)
        {
            // every face prescribes the velocity (with p= as the pressure level), so the ring is inflow
            int ring = 0, wrong = 0;
            for (int i = 0; i < g.Ny; i++)
                for (int j = 0; j < g.Nx; j++)
                {
                    if (i && j && i < g.Ny - 1 && j < g.Nx - 1) continue;
                    ring++;
                    if (SCENE.CellTypeAt(i, j, 0) != INFLOW_CELL) wrong++;
                }
            char b[128];
            std::snprintf(b, sizeof(b), "%d ring cells, %d not INFLOW_CELL", ring, wrong);
            check(wrong == 0, "its border classifies as prescribed velocity", b);

            // and the value is the analytic one, non-zero. The point has to sit ON a face (or beyond it):
            // PressureAt only answers there. This one is on the xmin face.
            const double x = SIMULATION.domain.x0, y = 0.7;
            const double want = -0.25 * (std::cos(2 * x) + std::cos(2 * y));
            const double got = SCENE.PressureAt(x, y, 0.0, 0.0);
            std::snprintf(b, sizeof(b), "p(x0, 0.7, t=0) = %.6f, expected %.6f", got, want);
            check(std::abs(got - want) < 1e-9, "the pressure expression evaluates correctly", b);
        }
    }

    std::printf("\n== control: outflow faces still mean p = 0 ==\n");
    Run2D(false, 0, "2D Eigen, outflow");

    std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
} // namespace pressure_dirichlet_check
#undef EXACT_EXPR

// ------------------------------------------------------------------------------------------------------------------
// pressure_warmstart_check
// ------------------------------------------------------------------------------------------------------------------
namespace pressure_warmstart_check
{
// Warm-starting the pressure solve.
//
// Every solve used to start from zero. The pressure field between consecutive steps of an
// Eulerian run barely changes -- the fluid region is fixed and the field is dominated by a
// slowly evolving smooth part -- so the previous solution is a far better starting point,
// and as a run settles toward steady state it becomes nearly exact.
//
// Two things are measured here. That it does not change the answer (an initial guess must
// not: the solver converges to the same solution either way, to within the tolerance), and
// how many iterations it actually saves. The second is the whole point; the first is what
// makes the second worth having.
//
// Eigen only. The CPU path measures ||r||/||b||, which does not depend on the guess, so it
// needs no change of stopping criterion -- which makes it the honest place to find out
// whether the idea pays before touching AMGX's RELATIVE_INI_CORE trap.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok)
        failures++;
}

struct Run
{
    long long iterations = 0;
    int firstIters = 0, lastIters = 0;
    double seconds = 0.0;         // whole step loop, momentum included
    double pressureSeconds = 0.0; // the solve alone, which is what this changes
    double uSum = 0.0, vSum = 0.0, pSum = 0.0;
    double divSum = 0.0, maxVel = 0.0;
};

// ---------------------------------------------------------------------------
// 2D lid-driven cavity: closed box, so the system is singular and the zero-mean
// handling in the guess is exercised. Also the case that settles fastest, which is
// where a warm start should look best.
// ---------------------------------------------------------------------------
static Run RunCavity2D(bool warmStart, int steps)
{
    DIMENSION = 2;
    SelectMethod("ADI");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 64;
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-2;
    SIMULATION.dt = 0.002;
    PRESSURE_SETTINGS.needsCompatibilityCondition = true; // closed box: pure Neumann
    PRESSURE_SETTINGS.warmStart = warmStart;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    SIMULATION.VelocityBoundaryFunction2D = LID_CAVITY_FLOW_2D;
    SIMULATION.PressureBoundaryFunction2D = LID_CAVITY_FLOW_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = LID_CAVITY_SOLID_MASK_2D;

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

    gridSol.SetSolidMask(LID_CAVITY_SOLID_MASK_2D);
    gridAnt.SetSolidMask(LID_CAVITY_SOLID_MASK_2D);

    ADI2D::InitializeADI2D(&gridSol, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction2D,
                            SIMULATION.PressureBoundaryFunction2D);
    PressureSolver2D::InitializePressureSolver(&gridSol, ADIPressureScheme());
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    gridAnt.EnforceBoundaryConditions(0);

    Run r;
    const double t0 = omp_get_wtime();
    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        ADI2D::SolveADIStep(&gridAnt, &gridSol, nullptr, step * SIMULATION.dt);
        const double tSolve = GetWallTime();
        PressureSolver2D::SolvePressure(&gridAnt, &gridSol, nullptr, 0.0);
        r.pressureSeconds += GetWallTime() - tSolve;
        PressureSolver2D::ProjectPressure(&gridSol);
        gridAnt.CopyGrid(gridSol);

        r.iterations += PRESSURE_TELEMETRY.lastIterations;
        if (step == 0) r.firstIters = PRESSURE_TELEMETRY.lastIterations;
        r.lastIters = PRESSURE_TELEMETRY.lastIterations;
    }
    r.seconds = omp_get_wtime() - t0;

    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx; j++)
        {
            r.uSum += std::abs(gridSol.GetU(i, j));
            r.vSum += std::abs(gridSol.GetV(i, j));
            r.pSum += gridSol.GetP(i, j);
        }
    r.divSum = gridSol.GetDivSum();
    r.maxVel = gridSol.GetMaxVelocity();

    PressureSolver2D::DestroyPressureSolver();
    return r;
}

// ---------------------------------------------------------------------------
// 3D lid-driven cavity, small enough to stay well inside the 60^3 ceiling.
// ---------------------------------------------------------------------------
static Run RunCavity3D(bool warmStart, int steps)
{
    DIMENSION = 3;
    SelectMethod("ADI");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;

    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 24;
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-2;
    SIMULATION.dt = 0.002;
    PRESSURE_SETTINGS.needsCompatibilityCondition = true;
    PRESSURE_SETTINGS.warmStart = warmStart;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    SIMULATION.VelocityBoundaryFunction = LID_CAVITY_FLOW;
    SIMULATION.PressureBoundaryFunction = LID_CAVITY_FLOW_PRESSURE;
    SIMULATION.SolidMaskFunction = LID_CAVITY_SOLID_MASK;

    static MAC gridSol, gridAnt;
    gridSol = MAC();
    gridAnt = MAC();
    gridSol.InitializeGrid(SIMULATION.domain, false);
    gridAnt.InitializeGrid(SIMULATION.domain, false);

    SIMULATION.dh = gridSol.dh;
    SIMULATION.Nx = gridSol.Nx;
    SIMULATION.Ny = gridSol.Ny;
    SIMULATION.Nz = gridSol.Nz;
    SIMULATION.GRID_SOL = &gridSol;
    SIMULATION.GRID_ANT = &gridAnt;

    gridSol.SetSolidMask(LID_CAVITY_SOLID_MASK);
    gridAnt.SetSolidMask(LID_CAVITY_SOLID_MASK);

    ADI3D::InitializeADI(&gridSol, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction,
                            SIMULATION.PressureBoundaryFunction);
    PressureSolver3D::InitializePressureSolver(&gridSol, ADIPressureScheme());
    gridAnt.SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    gridAnt.EnforceBoundaryConditions(0);

    Run r;
    const double t0 = omp_get_wtime();
    for (int step = 0; step < steps; step++)
    {
        gridSol.ClearFlowFields();
        ADI3D::SolveADIStep(&gridAnt, &gridSol, nullptr, step * SIMULATION.dt);
        const double tSolve = GetWallTime();
        PressureSolver3D::SolvePressure(&gridAnt, &gridSol, nullptr, 0.0);
        r.pressureSeconds += GetWallTime() - tSolve;
        PressureSolver3D::ProjectPressure(&gridSol);
        gridAnt.CopyGrid(gridSol);

        r.iterations += PRESSURE_TELEMETRY.lastIterations;
        if (step == 0) r.firstIters = PRESSURE_TELEMETRY.lastIterations;
        r.lastIters = PRESSURE_TELEMETRY.lastIterations;
    }
    r.seconds = omp_get_wtime() - t0;

    for (int i = 0; i < gridSol.Ny; i++)
        for (int j = 0; j < gridSol.Nx; j++)
            for (int k = 0; k < gridSol.Nz; k++)
            {
                r.uSum += std::abs(gridSol.GetU(i, j, k));
                r.vSum += std::abs(gridSol.GetV(i, j, k));
                r.pSum += gridSol.GetP(i, j, k);
            }
    r.divSum = gridSol.GetDivSum();
    r.maxVel = gridSol.GetMaxVelocity();

    PressureSolver3D::DestroyPressureSolver();
    return r;
}

static double relDiff(double a, double b)
{
    const double s = (std::max)(std::abs(a), std::abs(b));
    return (s > 1e-30) ? std::abs(a - b) / s : 0.0;
}

static void Report(const char *label, const Run &cold, const Run &warm, int steps)
{
    std::printf("\n== %s (%d steps) ==\n", label, steps);
    std::printf("  %-6s %10s %8s %8s %14s %12s\n", "", "iters", "step 1", "last",
                "solve (s)", "step (s)");
    std::printf("  %-6s %10lld %8d %8d %14.3f %12.3f\n", "cold", cold.iterations,
                cold.firstIters, cold.lastIters, cold.pressureSeconds, cold.seconds);
    std::printf("  %-6s %10lld %8d %8d %14.3f %12.3f\n", "warm", warm.iterations,
                warm.firstIters, warm.lastIters, warm.pressureSeconds, warm.seconds);
    if (warm.iterations > 0)
        std::printf("  iterations %.2fx fewer | solve %.2fx faster | whole step %.2fx faster\n",
                    (double)cold.iterations / (double)warm.iterations,
                    cold.pressureSeconds / (warm.pressureSeconds > 0 ? warm.pressureSeconds : 1),
                    cold.seconds / (warm.seconds > 0 ? warm.seconds : 1));

    char buf[256];
    // An initial guess must not change where the solver lands. The tolerance is loose
    // relative to machine precision and tight relative to the solve tolerance (1e-12
    // relative residual), which is the band a legitimate difference has to sit in.
    const double du = relDiff(cold.uSum, warm.uSum);
    const double dv = relDiff(cold.vSum, warm.vSum);
    const double dm = relDiff(cold.maxVel, warm.maxVel);
    std::snprintf(buf, sizeof(buf), "sum|u| %.3e, sum|v| %.3e, maxVel %.3e", du, dv, dm);
    check(du < 1e-8 && dv < 1e-8 && dm < 1e-8, "warm start does not change the answer", buf);

    // Both must be converged. They are not expected to agree digit for digit: a warm start
    // stops at a different point inside the same tolerance ball, so once the divergence is
    // down at solver-noise level the two runs differ by a few percent OF that noise. The
    // statement that matters -- that the answer is the same -- is the field check above.
    std::snprintf(buf, sizeof(buf), "cold %.6e, warm %.6e", cold.divSum, warm.divSum);
    check(cold.divSum < 1e-5 && warm.divSum < 1e-5,
          "the projected field is divergence free in both runs", buf);

    std::snprintf(buf, sizeof(buf), "%lld vs %lld", warm.iterations, cold.iterations);
    check(warm.iterations < cold.iterations, "and it takes fewer iterations", buf);
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int steps = (argc > 1) ? std::atoi(argv[1]) : 120;

    {
        Run cold = RunCavity2D(false, steps);
        Run warm = RunCavity2D(true, steps);
        Report("2D lid-driven cavity, Eigen", cold, warm, steps);
    }
    {
        const int s3 = steps / 2;
        Run cold = RunCavity3D(false, s3);
        Run warm = RunCavity3D(true, s3);
        Report("3D lid-driven cavity, Eigen", cold, warm, s3);
    }

    std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
} // namespace pressure_warmstart_check

static const SuiteCase CASES[] = {
    {"Dirichlet pressure, mode 0", pressure_dirichlet_check::run, {"0"}, false},
    {"Dirichlet pressure, mode 1", pressure_dirichlet_check::run, {"1"}, false},
    {"Dirichlet pressure, mode 2", pressure_dirichlet_check::run, {"2"}, false},
    {"Dirichlet pressure, mode 3", pressure_dirichlet_check::run, {"3"}, false},
    {"Dirichlet pressure, mode 5", pressure_dirichlet_check::run, {"5"}, false},
    {"Dirichlet pressure, mode 6", pressure_dirichlet_check::run, {"6"}, false},
    {"Dirichlet pressure, mode 7", pressure_dirichlet_check::run, {"7"}, false},
    {"Dirichlet pressure, mode 9", pressure_dirichlet_check::run, {"9"}, false},
    {"warm start", pressure_warmstart_check::run, {"120"}, false},
};

int main(int argc, char **argv) { return RunSuite("pressure", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
