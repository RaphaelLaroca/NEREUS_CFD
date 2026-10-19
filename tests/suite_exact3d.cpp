// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The exact3d suite: exact3d_check, nonuniform3d_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include "exact_solutions.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "Functions.h"
#include "headers/Core/Scene.h"
#include "Methods/ADI/ADI3D.h"
#include "Methods/PPESolver/PressureSolver3D.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include <vector>
#include <algorithm>
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// exact3d_check
// ------------------------------------------------------------------------------------------------------------------
namespace exact3d_check
{
// The 3D exact solution (Ethier-Steinman 1994) and the scene `initial` field.
//
// Two things are being pinned, and only one of them is about the formulas:
//
//  - The scene's expressions must agree with EthierSteinmanU/V/W/P to machine precision.
//    They are generated from the same maths, but a transcription slip in one of the six
//    pressure cross terms is invisible in the velocity and would show up only as a slowly
//    growing scored error -- which reads as "the solver is first order" rather than "the
//    reference is wrong".
//  - `initial` must actually seed the grid. Without it the run starts from rest, and since
//    Ethier-Steinman is O(1) at t = 0 the panel would be scoring the startup transient.

static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(),
                d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}

static MAC gs, ga;

static void Setup(int N)
{
    DIMENSION = 3; SelectMethod("ADI"); THREAD_COUNT = 6;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0; MAX_CFL = 0.5;
    SIMULATION.domain = Domain{-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    SIMULATION.GRID_SIZE = N;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 0.05;
    SIMULATION.dt = 0.002; SIMULATION.dtConfigured = 0.002; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = true;    // velocity on every face: pure Neumann, p= sets the level
    PRESSURE_SETTINGS.warmStart = false;
    SIMULATION.grid.nonUniform = false;
    SIMULATION.grid.refineX.clear(); SIMULATION.grid.refineY.clear(); SIMULATION.grid.refineZ.clear();
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    gs.InitializeGrid(SIMULATION.domain, false);
    ga.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.dh = gs.dh; SIMULATION.Nx = gs.Nx; SIMULATION.Ny = gs.Ny; SIMULATION.Nz = gs.Nz;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &ga;

    if (!SCENE.Load("tests/scenes/ethier_steinman_3d.scene"))
    { std::printf("scene fail: %s\n", SCENE.error.c_str()); std::exit(1); }
    SIMULATION.VelocityBoundaryFunction = SCENE_VELOCITY;
    SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
    SIMULATION.SolidMaskFunction = SCENE_SOLID_MASK;
    gs.SetSolidMask(SCENE_SOLID_MASK);
    ga.SetSolidMask(SCENE_SOLID_MASK);
}

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int steps = (argc > 1) ? std::atoi(argv[1]) : 20;
    const double nu = 0.05;

    // ---- 1. the scene expressions ARE the closed form ----
    Setup(16);
    double worstV = 0.0, worstP = 0.0;
    // ON THE FACES. SCENE_VELOCITY/SCENE_PRESSURE are BOUNDARY functions -- they answer
    // through FacesAtPoint and return zero anywhere no face matches, so sampling the
    // interior compares the exact solution against nothing and "fails" spectacularly.
    for (double t : {0.0, 0.05, 0.1})
        for (double x : {-1.0, 1.0, 0.3})
            for (double y : {-1.0, 1.0, -0.4})
                for (double z : {-1.0, 1.0, 0.7})
                {
                    const bool onFace = (std::abs(std::abs(x) - 1.0) < 1e-12) ||
                                        (std::abs(std::abs(y) - 1.0) < 1e-12) ||
                                        (std::abs(std::abs(z) - 1.0) < 1e-12);
                    if (!onFace) continue;
                    double u, v, w;
                    SIMULATION.time = t;
                    Vec3 b = SCENE_VELOCITY(x, y, z, t);
                    u = b.u; v = b.v; w = b.w;
                    worstV = std::max(worstV, std::abs(u - EthierSteinmanU(x, y, z, t, nu)));
                    worstV = std::max(worstV, std::abs(v - EthierSteinmanV(x, y, z, t, nu)));
                    worstV = std::max(worstV, std::abs(w - EthierSteinmanW(x, y, z, t, nu)));
                    worstP = std::max(worstP, std::abs(SCENE_PRESSURE(x, y, z, t)
                                                       - EthierSteinmanP(x, y, z, t, nu)));
                }
    std::printf("\n  scene vs closed form: worst |velocity| %.3e, worst |pressure| %.3e\n",
                worstV, worstP);
    check(worstV < 1e-12, "the scene's velocity expressions match EthierSteinmanU/V/W",
          "worst " + std::to_string(worstV));
    check(worstP < 1e-10, "the scene's pressure expression matches EthierSteinmanP",
          "worst " + std::to_string(worstP));

    // ---- 2. `initial` seeds the grid ----
    Setup(16);
    check(gs.GetMaxVelocity() < 1e-15, "the grid starts empty before ApplyInitialField");
    ga.ApplyInitialField();
    gs.ApplyInitialField();
    const double seeded = gs.GetMaxVelocity();
    std::printf("  after ApplyInitialField: maxVel %.4f\n", seeded);
    check(seeded > 0.5, "the scene's `initial` line seeds an O(1) field",
          "maxVel " + std::to_string(seeded));

    double worstSeed = 0.0;
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 2; j < gs.Nx - 1; j++)
            for (int k = 1; k < gs.Nz - 1; k++)
                worstSeed = std::max(worstSeed, std::abs(gs.GetU(i, j, k) -
                    EthierSteinmanU(gs.FacePos(MAC::X, j), gs.CentrePos(MAC::Y, i), gs.CentrePos(MAC::Z, k), 0.0, nu)));
    check(worstSeed < 1e-12, "and seeds it at the right staggered points",
          "worst U error at t=0: " + std::to_string(worstSeed));

    // ---- 3. the run converges against the prescribed field ----
    //
    // This used to guard an open question: the error saturated around 0.21-0.25 and got
    // WORSE with refinement, concentrated at the wall. The cause is found and fixed --
    // MAC::SetBorder omitted `omega.z0` on twelve of its coordinate expressions, so on a
    // domain not starting at the origin every z-face (and the far x/y pressure faces)
    // sampled the boundary function at the wrong place and got zero back. See the note on
    // SetBorder in MAC.cpp. Measured here at N = 16, 20 steps: L2 in u went 2.1e-01 ->
    // 2.05e-03, a hundredfold, and `run_exact3d_convergence.sh` now reports a clean first
    // order (0.92 -> 1.00 in u as N goes 12 -> 24 -> 48), which is what this scheme is.
    //
    // This harness stays the cheap guard -- the scene matches the closed form, `initial`
    // seeds exactly, the run is bounded and divergence free, and the error is now a small
    // FRACTION of the field rather than a sizeable part of it. The observed order lives in
    // run_exact3d_convergence.sh, which needs three resolutions and is too slow for this one.
    Setup(16);
    ga.ApplyInitialField(); gs.ApplyInitialField();
    ga.SetBoundaryFunctions(SCENE_VELOCITY, SCENE_PRESSURE);
    ga.EnforceBoundaryConditions(0.0);
    ADI3D::InitializeADI(&gs, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction,
                          SIMULATION.PressureBoundaryFunction);
    PressureSolver3D::InitializePressureSolver(&gs, ADIPressureScheme());
    for (int s2 = 0; s2 < steps; s2++)
    {
        SIMULATION.time = (s2 + 1) * SIMULATION.dt;
        ADI3D::SolveADIStep(&ga, &gs, nullptr, s2 * SIMULATION.dt);
        PressureSolver3D::SolvePressure(nullptr, &gs, nullptr, 0.0);
        PressureSolver3D::ProjectPressure(&gs);
        ga.CopyGrid(gs);
    }
    const double tEnd = steps * SIMULATION.dt;
    double su = 0.0, sI = 0.0, sW = 0.0, ref = 0.0, maxDiv = 0.0;
    long n = 0, nI = 0, nW = 0;
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 2; j < gs.Nx - 1; j++)
            for (int k = 1; k < gs.Nz - 1; k++)
            {
                const double ex = EthierSteinmanU(gs.FacePos(MAC::X, j), gs.CentrePos(MAC::Y, i), gs.CentrePos(MAC::Z, k), tEnd, nu);
                const double e = gs.GetU(i, j, k) - ex;
                su += e * e; n++; ref = std::max(ref, std::abs(ex));
                const bool wall = (i <= 2 || i >= gs.Ny - 3 || j <= 3 || j >= gs.Nx - 3 ||
                                   k <= 2 || k >= gs.Nz - 3);
                if (wall) { sW += e * e; nW++; } else { sI += e * e; nI++; }
            }
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 1; j < gs.Nx - 1; j++)
            for (int k = 1; k < gs.Nz - 1; k++)
                if (gs.GetSolid(i, j, k) == FLUID_CELL)
                    maxDiv = std::max(maxDiv, std::abs(gs.GetDivergencyAt(i, j, k)));

    const double l2 = std::sqrt(su / (double)(n ? n : 1));
    const double l2I = std::sqrt(sI / (double)(nI ? nI : 1));
    const double l2W = std::sqrt(sW / (double)(nW ? nW : 1));
    std::printf("  after %d steps (t = %.3f, N = 16):  L2 %.4e   interior %.4e   wall %.4e\n",
                steps, tEnd, l2, l2I, l2W);
    std::printf("  |u|max = %.3f, maxDiv = %.3e\n\n", ref, maxDiv);

    check(std::isfinite(l2) && l2 < ref, "the run stays finite and bounded against the reference",
          "L2 " + std::to_string(l2) + " against |u|max " + std::to_string(ref));
    check(maxDiv < 1e-9, "and divergence free",
          "maxDiv " + std::to_string(maxDiv));
    // The threshold that matters: 1% of the field. Before the z0 fix this was 8% and rising
    // with refinement, so this is the assertion that would have caught it.
    check(l2 < 0.01 * ref,
          "the error is a small fraction of the field, not a sizeable part of it",
          "L2 " + std::to_string(l2) + " = " +
          std::to_string(100.0 * l2 / ref) + "% of |u|max");

    // The wall band is still somewhat worse than the interior, which is ordinary for a
    // problem driven entirely by its boundary -- it is no longer the 3x gap that signalled a
    // defect, and it no longer grows with refinement. Recorded rather than asserted tightly.
    check(l2W < 5.0 * l2I,
          "and the wall is no longer far worse than the interior",
          "wall " + std::to_string(l2W) + " vs interior " + std::to_string(l2I));

    std::printf("%s (%d failure%s)\n", fails ? "FAILED" : "ALL CHECKS PASSED",
                fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace exact3d_check

// ------------------------------------------------------------------------------------------------------------------
// nonuniform3d_check
// ------------------------------------------------------------------------------------------------------------------
namespace nonuniform3d_check
{
// The rectilinear 3D Eulerian grid, against the Ethier-Steinman exact solution.
//
// The middle step is the one that matters and it is why this harness exists: **the
// rectilinear paths with EQUAL spacing must reproduce the uniform answer.** Setting
// NON_UNIFORM_GRID with no active interval routes every coefficient through the new
// finite-volume branch while the spacing tables stay uniform, so a disagreement there
// separates "a branch is wrong" from "the stretching is wrong" -- which are very different
// bugs and look identical from the error norm alone.


static int fails = 0;
static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) fails++;
}

static MAC gs, ga;
static const double NU_VISC = 0.05;

struct Res { int Nx, Ny, Nz; double l2u, l2p, maxDiv, maxVel, netFlux, divBeyondFlux; bool finite; };

static Res Run(int N, bool rect, const std::vector<GridRefinement> &rx, double tFinal, double dt,
               bool gpu = false)
{
    DIMENSION = 3; SelectMethod("ADI"); THREAD_COUNT = 6;
    GPU_ACCELERATION = gpu ? 1 : 0; ADAPTATIVE_TIMESTEP = 0; MAX_CFL = 0.5;
    SIMULATION.domain = Domain{-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    SIMULATION.GRID_SIZE = N;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = NU_VISC;
    SIMULATION.dt = dt; SIMULATION.dtConfigured = dt; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = true;    // velocity on every face: pure Neumann, p= sets the level
    PRESSURE_SETTINGS.warmStart = false;
    SIMULATION.grid.nonUniform = rect;
    SIMULATION.grid.refineX = rx;
    SIMULATION.grid.refineY.clear(); SIMULATION.grid.refineZ.clear();
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};

    gs.InitializeGrid(SIMULATION.domain, false);
    ga.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.dh = gs.dh; SIMULATION.Nx = gs.Nx; SIMULATION.Ny = gs.Ny; SIMULATION.Nz = gs.Nz;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &ga;

    if (!SCENE.Load("tests/scenes/ethier_steinman_3d.scene"))
    { std::printf("  scene failed: %s\n", SCENE.error.c_str()); std::exit(1); }
    SIMULATION.VelocityBoundaryFunction = SCENE_VELOCITY;
    SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
    SIMULATION.SolidMaskFunction = SCENE_SOLID_MASK;
    gs.SetSolidMask(SCENE_SOLID_MASK);
    ga.SetSolidMask(SCENE_SOLID_MASK);

    ga.ApplyInitialField(); gs.ApplyInitialField();
    ga.SetBoundaryFunctions(SCENE_VELOCITY, SCENE_PRESSURE);
    ga.EnforceBoundaryConditions(0.0);
    ADI3D::InitializeADI(&gs, dt, SCENE_VELOCITY,  SCENE_PRESSURE);
    PressureSolver3D::InitializePressureSolver(&gs, ADIPressureScheme());

    const int steps = (int)std::llround(tFinal / dt);
    for (int s = 0; s < steps; s++)
    {
        SIMULATION.time = (s + 1) * dt;
        ADI3D::SolveADIStep(&ga, &gs, nullptr, s * dt);
        PressureSolver3D::SolvePressure(nullptr, &gs, nullptr, 0.0);
        PressureSolver3D::ProjectPressure(&gs);
        ga.CopyGrid(gs);
    }
    const double t = steps * dt;

    Res r; r.Nx = gs.Nx; r.Ny = gs.Ny; r.Nz = gs.Nz;
    double su = 0, mx = 0, md = 0; long n = 0;
    // Coordinates come from the TABLES, never from x0 + j*dh: on a stretched grid the two
    // differ, and a harness that measures at the wrong points looks exactly like a broken
    // solver -- smooth, O(1) and non-converging.
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 2; j < gs.Nx - 1; j++)
            for (int k = 1; k < gs.Nz - 1; k++)
            {
                const double e = gs.GetU(i, j, k) -
                    EthierSteinmanU(gs.FacePos(MAC::X, j), gs.CentrePos(MAC::Y, i), gs.CentrePos(MAC::Z, k), t, NU_VISC);
                su += e * e; n++;
            }
    double sp = 0, mp = 0, me = 0; long np = 0;
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 1; j < gs.Nx - 1; j++)
            for (int k = 1; k < gs.Nz - 1; k++)
            {
                mp += gs.GetP(i, j, k);
                me += EthierSteinmanP(gs.CentrePos(MAC::X, j), gs.CentrePos(MAC::Y, i), gs.CentrePos(MAC::Z, k), t, NU_VISC);
                np++;
            }
    const double off = np ? (mp - me) / (double)np : 0.0;
    // Velocity on every face makes the pressure pure Neumann, so the projection can only bring each cell's
    // vol*div to the net discrete boundary flux shared equally (the compatibility condition removes that
    // mean). The sampled exact field does not sum to exactly zero flux on every grid; what must vanish is
    // the divergence BEYOND that share.
    double flux = 0.0;
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 1; j < gs.Nx - 1; j++)
            for (int k = 1; k < gs.Nz - 1; k++)
                flux += gs.GetDivergencyAt(i, j, k) * gs.Width(MAC::X, j) * gs.Width(MAC::Y, i) * gs.Width(MAC::Z, k);
    const double share = np ? flux / (double)np : 0.0;
    double beyond = 0.0;
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 1; j < gs.Nx - 1; j++)
            for (int k = 1; k < gs.Nz - 1; k++)
            {
                const double e = (gs.GetP(i, j, k) - off) -
                    EthierSteinmanP(gs.CentrePos(MAC::X, j), gs.CentrePos(MAC::Y, i), gs.CentrePos(MAC::Z, k), t, NU_VISC);
                sp += e * e;
                md = std::max(md, std::abs(gs.GetDivergencyAt(i, j, k)));
                const double vol = gs.Width(MAC::X, j) * gs.Width(MAC::Y, i) * gs.Width(MAC::Z, k);
                beyond = std::max(beyond, std::abs(gs.GetDivergencyAt(i, j, k) * vol - share) / vol);
            }
    mx = gs.GetMaxVelocity();
    r.l2u = std::sqrt(su / (double)(n ? n : 1));
    r.l2p = std::sqrt(sp / (double)(np ? np : 1));
    r.maxDiv = md; r.maxVel = mx; r.netFlux = flux; r.divBeyondFlux = beyond;
    // A NaN loses every comparison, so max() would leave a running maximum at zero and a
    // "maxDiv < tol" test would pass vacuously on a field that has blown up.
    r.finite = std::isfinite(r.l2u) && std::isfinite(r.l2p) &&
               std::isfinite(r.maxDiv) && std::isfinite(r.maxVel);
    return r;
}

static GridRefinement Band(double a, double b, int f, int m)
{ GridRefinement g; g.active = true; g.start = a; g.end = b; g.factor = f; g.mode = m; return g; }

int run(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const double T = (argc > 1) ? std::atof(argv[1]) : 0.02;
    const double dt = 0.002;

    std::printf("\n=== 3D rectilinear Eulerian, Ethier-Steinman on [-1,1]^3 ===\n\n");
    std::printf("  %-42s %6s %12s %12s %11s\n", "case", "cells", "L2 u", "L2 p", "maxDiv");

    Res uni  = Run(24, false, {},  T, dt);
    std::printf("  %-42s %6d %12.5e %12.5e %11.2e\n", "uniform (scalar dh path)",
                uni.Nx, uni.l2u, uni.l2p, uni.maxDiv);

    Res eq   = Run(24, true,  {},  T, dt);
    std::printf("  %-42s %6d %12.5e %12.5e %11.2e\n", "rectilinear, EQUAL spacing",
                eq.Nx, eq.l2u, eq.l2p, eq.maxDiv);

    // ---- the separator: equal spacing must reproduce the uniform answer ----
    const double du = std::abs(eq.l2u - uni.l2u) / std::max(uni.l2u, 1e-30);
    const double dp = std::abs(eq.l2p - uni.l2p) / std::max(uni.l2p, 1e-30);
    char b[220];
    // Velocity is asserted an order tighter than pressure, and the split is deliberate.
    // The two branches solve the SAME system scaled differently -- uniform multiplies
    // through by 1/dh^2, rectilinear leaves it unscaled and weights the divergence by the
    // cell volume -- so BiCGSTAB at tol = 1e-12 lands on slightly different points. The
    // measured pressure gap is 2.0e-10 RELATIVE, which against an L2 norm of 1.9e-02 is
    // 3.8e-12 absolute: the solver's own tolerance, not a disagreement between branches.
    // The velocity, which is what the projection actually produces, agrees to 4.5e-13.
    std::snprintf(b, sizeof(b), "L2 u differs by %.2e (abs %.2e), L2 p by %.2e (abs %.2e)",
                  du, du * uni.l2u, dp, dp * uni.l2p);
    check(du < 1e-11 && dp * uni.l2p < 1e-10,
          "the rectilinear branch at EQUAL spacing reproduces the uniform answer", b);

    // ---- and now actually stretched ----
    Res hard = Run(24, true, std::vector<GridRefinement>{ Band(-0.3, 0.3, 2, REFINE_HARD)   }, T, dt);
    std::printf("  %-42s %6d %12.5e %12.5e %11.2e\n", "x refined 2x on [-0.3,0.3], hard",
                hard.Nx, hard.l2u, hard.l2p, hard.maxDiv);
    Res grad = Run(24, true, std::vector<GridRefinement>{ Band(-0.3, 0.3, 2, REFINE_SYMMETRIC) }, T, dt);
    std::printf("  %-42s %6d %12.5e %12.5e %11.2e\n", "the same, graded",
                grad.Nx, grad.l2u, grad.l2p, grad.maxDiv);
    Res fine = Run(48, true, std::vector<GridRefinement>{ Band(-0.3, 0.3, 2, REFINE_SYMMETRIC) }, T, dt / 2.0);
    std::printf("  %-42s %6d %12.5e %12.5e %11.2e\n", "the same, finer base (N=48, dt/2)",
                fine.Nx, fine.l2u, fine.l2p, fine.maxDiv);

    std::printf("\n");
    for (const Res *r : {&hard, &grad, &fine})
    {
        std::snprintf(b, sizeof(b), "%dx%dx%d  L2 u %.4e  maxVel %.4f",
                      r->Nx, r->Ny, r->Nz, r->l2u, r->maxVel);
        check(r->finite, "the stretched run stays finite", b);
        std::snprintf(b, sizeof(b), "maxDiv %.3e, net boundary flux %.3e, divergence beyond its share %.3e on %dx%dx%d",
                      r->maxDiv, r->netFlux, r->divBeyondFlux, r->Nx, r->Ny, r->Nz);
        check(r->divBeyondFlux < 1e-9, "and divergence free up to the boundary's own net flux", b);
    }

    // The stretched grid adds cells, so it must not be WORSE than uniform at the same base.
    std::snprintf(b, sizeof(b), "uniform %.4e, hard %.4e, graded %.4e", uni.l2u, hard.l2u, grad.l2u);
    check(hard.l2u < 1.5 * uni.l2u && grad.l2u < 1.5 * uni.l2u,
          "refining does not degrade the solution", b);

    // ---- the AMGX path must solve the SAME system as Eigen ----
    //
    // This is where the rectilinear conversion first broke on the GPU and the CPU was fine:
    // the Eigen copy of the matrix and the CSR array AMGX uploads are scaled at TWO separate
    // sites, and only one of them had been converted. The stretched coefficients already
    // carry A_f/d_f, so the surviving hardcoded 1/dh^2 multiplied them by 1/dh_min^2 and the
    // divergence came out in the thousands. Nothing in a CPU-only harness can see that,
    // which is why both paths are driven here.
    Res gpuU = Run(24, true, {}, T, dt, true);
    Res gpuS = Run(24, true, std::vector<GridRefinement>{ Band(-0.3, 0.3, 2, REFINE_HARD) }, T, dt, true);
    std::printf("  %-42s %6d %12.5e %12.5e %11.2e\n", "AMGX, rectilinear EQUAL spacing",
                gpuU.Nx, gpuU.l2u, gpuU.l2p, gpuU.maxDiv);
    std::printf("  %-42s %6d %12.5e %12.5e %11.2e\n", "AMGX, x refined 2x hard",
                gpuS.Nx, gpuS.l2u, gpuS.l2p, gpuS.maxDiv);
    std::printf("\n");
    for (auto pr : { std::make_pair(&eq, &gpuU), std::make_pair(&hard, &gpuS) })
    {
        const Res *cpu = pr.first, *g = pr.second;
        std::snprintf(b, sizeof(b), "Eigen %.5e vs AMGX %.5e (%.2e relative), maxDiv %.2e",
                      cpu->l2u, g->l2u,
                      std::abs(g->l2u - cpu->l2u) / std::max(cpu->l2u, 1e-30), g->maxDiv);
        check(g->finite && g->maxDiv < 1e-8 &&
              std::abs(g->l2u - cpu->l2u) / std::max(cpu->l2u, 1e-30) < 1e-4,
              "AMGX agrees with Eigen on the rectilinear grid", b);
    }

    const double order = std::log(grad.l2u / fine.l2u) / std::log(2.0);
    std::snprintf(b, sizeof(b), "L2 u %.4e -> %.4e, observed order %.2f", grad.l2u, fine.l2u, order);
    check(order > 0.6, "the stretched grid converges under refinement", b);

    std::printf("\n%s (%d failure%s)\n", fails ? "CHECKS FAILED" : "ALL CHECKS PASSED",
                fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace nonuniform3d_check

static const SuiteCase CASES[] = {
    {"Ethier-Steinman", exact3d_check::run, {}, false},
    {"Ethier-Steinman, stretched", nonuniform3d_check::run, {"0.02"}, false},
};

int main(int argc, char **argv) { return RunSuite("exact3d", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
