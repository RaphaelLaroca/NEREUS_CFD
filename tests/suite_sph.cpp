// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The sph suite: wcsph_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "Functions.h"
#include "headers/Core/Scene.h"
#include "Methods/SPH/SPH2D.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include "Methods/SPH/SPH3D.h"
#include <algorithm>
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// wcsph_check
// ------------------------------------------------------------------------------------------------------------------
namespace wcsph_check
{
// WCSPH, 2D and 3D: a tank at rest must stay at rest with the hydrostatic pressure, and a
// dam break must stay bounded, keep its density within the weakly compressible band and advance its front
// close to where Martin & Moyce measured it.

static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(), d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}

static std::string Num(double v)
{
    char b[64];
    std::snprintf(b, sizeof(b), "%.4g", v);
    return b;
}

static MAC gs, ga;

static void Setup(double xf, double yf, const std::string &fluid, double h)
{
    DIMENSION = 2; SelectMethod("SPH"); THREAD_COUNT = 6;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0; MAX_CFL = 1.0;
    SIMULATION.domain = Domain{0.0, xf, 0.0, yf, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 120;
    SIMULATION.dt = 1e-3; SIMULATION.dtConfigured = 1e-3; SIMULATION.time = 0.0;
    SIMULATION.f = Vec3{0.0, -9.81, 0.0};
    SIMULATION.particleOptions.contact = 0;
    SIMULATION.particleOptions.restitution = 0.0;
    SIMULATION.RHO = 1000.0;
    SIMULATION.EPS = 1e-6;
    SPH_SETTINGS.kernelRadius = h;
    SPH_SETTINGS.particleSpacing = 0.0;          // h/2.6
    SPH_SETTINGS.soundSpeed = 20.0;
    SPH_SETTINGS.delta = 0.1;
    SPH_SETTINGS.alpha = 0.02;

    gs.InitializeGrid(SIMULATION.domain, true);
    ga.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = gs.dh; SIMULATION.Nx = gs.Nx; SIMULATION.Ny = gs.Ny;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &ga;
    if (!SCENE.LoadText("sampling center\nregion fluid box " + fluid + "\n"))
    {
        std::printf("scene fail: %s\n", SCENE.error.c_str());
        std::exit(1);
    }
    SPH2D::InitializeSPH2D(&SIMULATION.particles, 400000);
}

static void Setup3D(double xf, double yf, double zf, const std::string &fluid, double h)
{
    DIMENSION = 3; SelectMethod("SPH"); THREAD_COUNT = 6;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0; MAX_CFL = 1.0;
    SIMULATION.domain = Domain{0.0, xf, 0.0, yf, 0.0, zf};
    SIMULATION.GRID_SIZE = 16;
    SIMULATION.dt = 1e-3; SIMULATION.dtConfigured = 1e-3; SIMULATION.time = 0.0;
    SIMULATION.f = Vec3{0.0, -9.81, 0.0};
    SIMULATION.particleOptions.contact = 0;
    SIMULATION.RHO = 1000.0;
    SIMULATION.EPS = 1e-6;
    SPH_SETTINGS.kernelRadius = h;
    SPH_SETTINGS.particleSpacing = 0.0;
    SPH_SETTINGS.soundSpeed = 20.0;
    SPH_SETTINGS.delta = 0.1;
    SPH_SETTINGS.alpha = 0.02;
    gs.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &gs;
    if (!SCENE.LoadText("sampling center\nregion fluid box " + fluid + "\n"))
    {
        std::printf("scene fail: %s\n", SCENE.error.c_str());
        std::exit(1);
    }
    SPH3D::InitializeSPH3D(&SIMULATION.particles, 400000);
}

static void Run3D(double tEnd)
{
    const int n = (int)std::lround(tEnd / SIMULATION.dt);
    for (int s = 0; s < n; s++)
    {
        SPH3D::AdvanceSPH(nullptr, &gs, &SIMULATION.particles, s * SIMULATION.dt);
        SIMULATION.time = (s + 1) * SIMULATION.dt;
    }
}

static bool AllFinite3D()
{
    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        const Particle &q = SIMULATION.particles.data[p];
        if (!std::isfinite(q.x + q.y + q.z + q.u + q.v + q.w)) return false;
    }
    return true;
}

static void Run(double tEnd)
{
    const int n = (int)std::lround(tEnd / SIMULATION.dt);
    for (int s = 0; s < n; s++)
    {
        SPH2D::AdvanceSPH(nullptr, &gs, &SIMULATION.particles, s * SIMULATION.dt);
        SIMULATION.time = (s + 1) * SIMULATION.dt;
    }
}

static bool AllFinite()
{
    for (int p = 0; p < SIMULATION.particles.count; p++)
    {
        const Particle &q = SIMULATION.particles.data[p];
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.u) || !std::isfinite(q.v)) return false;
    }
    return true;
}

int run(int, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // ---- 1. a tank at rest ----
    std::printf("\n== tank at rest: 0.4 m of water, h = 0.03 m, c0 = 20 m/s, 2 s ==\n");
    const double H = 0.4;
    Setup(0.6, 0.6, "min=(0,0,-1) max=(0.6,0.4,1)", 0.03);
    const int N = SIMULATION.particles.count;
    Run(1.8);
    // A WCSPH pressure carries acoustic noise, so the bed pressure is averaged over the last 0.2 s.
    const double yBed = 0.5 * SPH2D::spacing + SPH2D::h * 0.5;
    double pBed = 0.0;
    for (int s = 0; s < 200; s++)
    {
        SPH2D::AdvanceSPH(nullptr, &gs, &SIMULATION.particles, 1.8 + s * SIMULATION.dt);
        pBed += SPH2D::GetPressureAt(0.3, yBed) / 200.0;
    }
    double vmax = 0.0, vmean = 0.0, dmin = 1e30, dmax = 0.0, top = 0.0;
    for (int p = 0; p < N; p++)
    {
        const Particle &q = SIMULATION.particles.data[p];
        vmax = std::max(vmax, std::sqrt(q.u * q.u + q.v * q.v));
        vmean += std::sqrt(q.u * q.u + q.v * q.v) / N;
        dmin = std::min(dmin, SIMULATION.particles.density[p]);
        dmax = std::max(dmax, SIMULATION.particles.density[p]);
        top = std::max(top, q.y);
    }
    check(AllFinite(), "the field stays finite");
    // The seeding lattice is an unstable equilibrium: near t = 1.1 s it rearranges into a disordered one, a
    // one-time event that sets particles moving at up to ~4% of sqrt(gH) and then decays.
    check(vmax < 0.05 * std::sqrt(9.81 * H) && vmean < 0.01 * std::sqrt(9.81 * H),
          "the liquid stays at rest (peak under 5%, mean under 1% of sqrt(gH))",
          "max |u| " + Num(vmax) + ", mean " + Num(vmean) + " m/s");
    const double pExact = 1000.0 * 9.81 * (H - yBed);
    // A weakly compressible pressure is c0^2 (rho - rho0): 140 Pa off is a density 0.035% off, well inside the 1% that
    // c0 = 20 m/s allows. Measured: +70 to +140 Pa depending on how the lattice rearranged, so the band is 5%.
    check(std::abs(pBed - pExact) < 0.05 * 1000.0 * 9.81 * H,
          "the pressure near the bed, averaged over 0.2 s, is hydrostatic (5% of rho g H)",
          "p " + Num(pBed) + " Pa against " + Num(pExact));
    check(dmax / 1000.0 - 1.0 < 0.02 && dmin > 999.0, "the density stays within the weakly compressible band",
          "rho in [" + Num(dmin) + ", " + Num(dmax) + "]");
    check(std::abs(top - (H - 0.5 * SPH2D::spacing)) < 0.5 * SPH2D::spacing, "the free surface stays where it was",
          "top particle at " + Num(top) + " m");
    // The grid SPH writes for telemetry: a level probe must read the column, a pressure probe the hydrostatic value.
    SPH2D::AdvanceSPH(nullptr, &gs, &SIMULATION.particles, 2.0);
    Probe lv; lv.kind = ProbeQuantity::Level; lv.at[0] = 0.3;
    Probe pr; pr.kind = ProbeQuantity::Pressure; pr.at[0] = 0.3; pr.at[1] = 0.1;
    const double level = EvaluateProbe(&gs, &SIMULATION.particles, lv);
    const double pGrid = EvaluateProbe(&gs, &SIMULATION.particles, pr);
    check(std::abs(level - H) < 1.5 * gs.dh, "a level probe on the deposited grid reads the column (within 1.5 cells)",
          Num(level) + " m against " + Num(H) + ", cell " + Num(gs.dh));
    check(std::abs(pGrid - 1000.0 * 9.81 * (H - 0.1)) < 0.05 * 1000.0 * 9.81 * H,
          "a pressure probe on the deposited grid reads the hydrostatic pressure",
          Num(pGrid) + " Pa against " + Num(1000.0 * 9.81 * (H - 0.1)));

    // ---- 2. dam break, Martin & Moyce n^2 = 2 (a = 0.2 m, H = 0.4 m) ----
    std::printf("\n== dam break, a = 0.2 m, H = 0.4 m, h = 0.02 m ==\n");
    const double a = 0.2;
    Setup(1.6, 0.6, "min=(0,0,-1) max=(0.2,0.4,1)", 0.02);
    const int M = SIMULATION.particles.count;
    const double T = 2.0 * std::sqrt(a / (2.0 * 9.81));   // T = t sqrt(2g/a) = 2
    Run(T);
    double front = 0.0, dmn = 1e30, dmx = 0.0;
    int outside = 0;
    for (int p = 0; p < M; p++)
    {
        const Particle &q = SIMULATION.particles.data[p];
        front = std::max(front, q.x);
        dmn = std::min(dmn, SIMULATION.particles.density[p]);
        dmx = std::max(dmx, SIMULATION.particles.density[p]);
        if (q.x < 0.0 || q.x > 1.6 || q.y < 0.0 || q.y > 0.6) outside++;
    }
    check(AllFinite() && outside == 0, "the field stays finite and every particle in the box",
          std::to_string(outside) + " outside");
    check(dmx < 1030.0 && dmn > 970.0, "the density stays within 3% of rho0", "[" + Num(dmn) + ", " + Num(dmx) + "]");
    // Martin & Moyce (1952), Table 2: Z = x/a = 2.35 at T = n t sqrt(g/a) = 2 (between 2.33 at 1.98 and 2.56 at
    // 2.20). Models run ahead of it -- no gate release, no bed friction, no surface tension -- so a front up to
    // 25% ahead is accepted; behind it would be wrong.
    const double Z = front / a;
    check(Z > 2.2 && Z < 1.25 * 2.35, "the front is at or a little ahead of Martin & Moyce at T = 2 (Z = 2.35)",
          "Z = " + Num(Z));

    // ---- 3. 3D: a tank at rest, and a dam break against its own 2D twin ----
    // h = 0.039 m makes the pitch h/2.6 = 15 mm, which divides the 0.3 m box: with a pitch that does not, the
    // lattice leaves a slot against the far walls and the column collapses into it -- a dam break, not a tank.
    std::printf("\n== 3D tank at rest: 0.21 m of water, h = 0.039 m, 1 s ==\n");
    Setup3D(0.3, 0.3, 0.3, "min=(0,0,0) max=(0.3,0.21,0.3)", 0.039);
    const int K = SIMULATION.particles.count;
    Run3D(1.0);
    double v3 = 0.0, d3n = 1e30, d3x = 0.0;
    for (int p = 0; p < K; p++)
    {
        const Particle &q = SIMULATION.particles.data[p];
        v3 = std::max(v3, std::sqrt(q.u * q.u + q.v * q.v + q.w * q.w));
        d3n = std::min(d3n, SIMULATION.particles.density[p]);
        d3x = std::max(d3x, SIMULATION.particles.density[p]);
    }
    check(AllFinite3D(), "the 3D field stays finite", std::to_string(K) + " particles");
    check(v3 < 0.05 * std::sqrt(9.81 * 0.21), "the 3D liquid stays at rest (under 5% of sqrt(gH))", "max |u| " + Num(v3));
    check(d3n > 999.0 && d3x < 1010.0, "the 3D density stays within the weakly compressible band",
          "[" + Num(d3n) + ", " + Num(d3x) + "]");

    std::printf("\n== 3D dam break, a = 0.2 m, H = 0.4 m, 0.1 m deep, h = 0.03 m ==\n");
    Setup3D(1.0, 0.5, 0.1, "min=(0,0,0) max=(0.2,0.4,0.1)", 0.03);
    const int L = SIMULATION.particles.count;
    Run3D(T);
    double f3 = 0.0;
    int out3 = 0;
    for (int p = 0; p < L; p++)
    {
        const Particle &q = SIMULATION.particles.data[p];
        f3 = std::max(f3, q.x);
        if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 0.5 || q.z < 0.0 || q.z > 0.1) out3++;
    }
    check(AllFinite3D() && out3 == 0, "the 3D dam break stays finite and in the box", std::to_string(out3) + " outside");
    check(f3 / a > 2.2 && f3 / a < 1.25 * 2.35, "its front is at or a little ahead of Martin & Moyce at T = 2",
          "Z = " + Num(f3 / a));

    std::printf("\n%s (%d failures)\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", fails);
    return fails ? 1 : 0;
}
} // namespace wcsph_check

static const SuiteCase CASES[] = {
    {"WCSPH", wcsph_check::run, {}, false},
};

int main(int argc, char **argv) { return RunSuite("sph", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
