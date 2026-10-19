// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The io suite: checkpoint_check, export_check, restart_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cmath>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "Functions.h"
#include "headers/Core/Scene.h"
#include "headers/Core/Checkpoint.h"
#include "Methods/ADI/ADI2D.h"
#include "Methods/FLIP/FLIP3D.h"
#include "Methods/SPH/SPH3D.h"
#include "Methods/PPESolver/PressureSolver3D.h"
#include "Methods/PPESolver/PressureSolver2D.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// checkpoint_check
// ------------------------------------------------------------------------------------------------------------------
namespace checkpoint_check
{
// Checkpoint / restart: a run stopped and resumed must be the SAME computation.
//
// The instrument is bit-identity, not closeness. Run the obstacle case 60 steps in one go;
// separately, run 30, checkpoint, throw the state away, rebuild from the config, load, and run
// the remaining 30. Every field checksum must match to the last digit. Anything that is state
// and was not saved -- the previous-step grid, the masks, the clock -- shows up here as a
// difference in the last digits, which is why the threshold is zero rather than a tolerance.
//
// The ADI case runs with warm start ON: the guess is read from GRID_ANT's pressure, which the
// checkpoint carries, so no solver holds state and a resumed run lands on the same bits.


static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(), d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}

static MAC gs, ga;

struct Sums { double u = 0, v = 0, p = 0, div = 0, au = 0, av = 0, ap = 0; double time = 0; unsigned long long it = 0; };

static void StandUp(int gridSize = 48)
{
    DIMENSION = 2; SelectMethod("ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0; MAX_CFL = 1.0;
    SIMULATION.domain = Domain{0.0, 1.1, 0.0, 0.41, 0.0, 1.0};
    SIMULATION.GRID_SIZE = gridSize;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.002; SIMULATION.dtConfigured = 0.002; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    PRESSURE_SETTINGS.warmStart = true;    // the guess comes from GRID_ANT, which is checkpointed
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};
    SIMULATION.grid.nonUniform = false;

    gs.InitializeGrid(SIMULATION.domain, true);
    ga.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = gs.dh; SIMULATION.Nx = gs.Nx; SIMULATION.Ny = gs.Ny;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &ga;

    if (!SCENE.Load("ckpt.scene")) { std::printf("scene: %s\n", SCENE.error.c_str()); std::exit(1); }
    SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = SCENE_SOLID_MASK_2D;
    gs.SetSolidMask(SCENE_SOLID_MASK_2D);
    ga.SetSolidMask(SCENE_SOLID_MASK_2D);
    ADI2D::InitializeADI2D(&gs, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction2D, 
                           SIMULATION.PressureBoundaryFunction2D);
    PressureSolver2D::InitializePressureSolver(&gs, ADIPressureScheme());
    ga.SetBoundaryFunctions(SCENE_VELOCITY_2D, SCENE_PRESSURE_2D);
    ga.EnforceBoundaryConditions(0.0);
}

// Sums of the CURRENT grid, for bisecting a step into its phases.
static double SumU() { double t = 0; for (int i = 0; i < gs.Ny; i++) for (int j = 0; j < gs.Nx + 1; j++) t += std::fabs(gs.GetU(i, j)); return t; }
static double SumP() { double t = 0; for (int i = 0; i < gs.Ny; i++) for (int j = 0; j < gs.Nx; j++) t += gs.GetP(i, j); return t; }
static double probeAfterADI = 0.0, probeAfterSolve = 0.0;

static void Step(unsigned long long step, double &clock)
{
    SIMULATION.time = (step + 1) * SIMULATION.dt;
    gs.ClearFlowFields();
    ADI2D::SolveADIStep(&ga, &gs, nullptr, step * SIMULATION.dt);
    probeAfterADI = SumU();
    PressureSolver2D::SolvePressure(&ga, &gs, nullptr, 0.0);
    probeAfterSolve = SumP();
    PressureSolver2D::ProjectPressure(&gs);
    ga.CopyGrid(gs);
    clock = SIMULATION.time;
}

static Sums Measure(double clock, unsigned long long it)
{
    Sums s;
    s.time = clock; s.it = it;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx + 1; j++) s.u += std::fabs(gs.GetU(i, j));
    for (int i = 0; i < gs.Ny + 1; i++)
        for (int j = 0; j < gs.Nx; j++) s.v += std::fabs(gs.GetV(i, j));
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 1; j < gs.Nx - 1; j++)
        {
            s.p += gs.GetP(i, j);
            if (gs.GetSolid(i, j) == FLUID_CELL) s.div += std::fabs(gs.GetDivergencyAt(i, j));
        }
    // GRID_ANT is what the ADI sweep reads as the previous field, so it is state too.
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx + 1; j++) s.au += std::fabs(ga.GetU(i, j));
    for (int i = 0; i < ga.Ny + 1; i++)
        for (int j = 0; j < ga.Nx; j++) s.av += std::fabs(ga.GetV(i, j));
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++) s.ap += ga.GetP(i, j);
    return s;
}


// ---------------------------------------------------------------------------
// The particle solvers. Same instrument as above -- run N steps two ways and demand the
// last digit -- but the quantity that matters is the CLOUD, so it is summed separately
// from the grid.
// ---------------------------------------------------------------------------
struct Cloud
{
    int n = 0;
    double x = 0, y = 0, z = 0, u = 0, v = 0, w = 0;  // the cloud
    double gu = 0, gp = 0;                            // the grid it transfers to
};

static MAC pgs, pga;

template <typename P>
static Cloud SumCloud(const P *p, int n, MAC &g)
{
    Cloud s;
    s.n = n;
    for (int i = 0; i < n; i++)
    {
        s.x += std::fabs(p[i].x); s.y += std::fabs(p[i].y); s.z += std::fabs(p[i].z);
        s.u += std::fabs(p[i].u); s.v += std::fabs(p[i].v); s.w += std::fabs(p[i].w);
    }
    for (int i = 0; i < g.Ny; i++)
        for (int j = 0; j < g.Nx + 1; j++)
            for (int k = 0; k < g.Nz; k++) s.gu += std::fabs(g.GetU(i, j, k));
    for (int i = 0; i < g.Ny; i++)
        for (int j = 0; j < g.Nx; j++)
            for (int k = 0; k < g.Nz; k++) s.gp += g.GetP(i, j, k);
    return s;
}

static void StandUpFLIP()
{
    DIMENSION = 3; SelectMethod("FLIP"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 0.3, 0.0, 0.3, 0.0, 0.3};
    SIMULATION.GRID_SIZE = 20;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.002; SIMULATION.dtConfigured = 0.002; SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    PRESSURE_SETTINGS.warmStart = false;   // FLIP never warm starts
    FLIP_SETTINGS.particlesPerCell = 8;
    FLIP_SETTINGS.alpha = 0.95;
    FLIP_SETTINGS.physicalDiffusion = false;
    SIMULATION.particleOptions.seedJitter = 0.0;
    SIMULATION.grid.nonUniform = false;
    SIMULATION.f = Vec3{0.0, -9.8, 0.0};

    pgs.InitializeGrid(SIMULATION.domain, false);
    pga.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.dh = pgs.dh; SIMULATION.Nx = pgs.Nx; SIMULATION.Ny = pgs.Ny; SIMULATION.Nz = pgs.Nz;
    SIMULATION.GRID_SOL = &pgs; SIMULATION.GRID_ANT = &pga;

    if (!SCENE.Load("ckpt_flip3.scene")) { std::printf("scene: %s\n", SCENE.error.c_str()); std::exit(1); }
    SIMULATION.VelocityBoundaryFunction = SCENE_VELOCITY;
    SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
    SIMULATION.SolidMaskFunction = SCENE_SOLID_MASK;
    pgs.SetSolidMask(SCENE_SOLID_MASK);
    pga.SetSolidMask(SCENE_SOLID_MASK);
    PressureSolver3D::InitializePressureSolver(&pgs);
    FLIP3D::InitializeFLIP(&pgs, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    pga.SetBoundaryFunctions(SCENE_VELOCITY, SCENE_PRESSURE);
    pga.EnforceBoundaryConditions(0);
}

static void StepFLIP(unsigned long long step)
{
    pgs.ClearFlowFields();
    FLIP3D::FLIP_Momentum(&pga, &pgs, &SIMULATION.particles, step * SIMULATION.dt);
    PressureSolver3D::SolvePressure(nullptr, &pgs, nullptr, 0.0);
    PressureSolver3D::Project(&pga, &pgs, &SIMULATION.particles, 0.0);
    FLIP3D::FLIP_GridToParticle(&pga, &pgs, &SIMULATION.particles, 0.0);
    pga.CopyGrid(pgs);
}

// `split` runs half, checkpoints, tears the solver down and rebuilds it from the config,
// loads, and finishes -- which is what a real restart does.
static Cloud RunFLIP(unsigned long long total, bool split)
{
    const unsigned long long half = total / 2;
    StandUpFLIP();
    for (unsigned long long s = 0; s < (split ? half : total); s++) StepFLIP(s);
    if (split)
    {
        std::string e;
        if (!Checkpoint::Save("ckpt_flip.ckpt", half * SIMULATION.dt, half, e))
        { std::printf("  FLIP save failed: %s\n", e.c_str()); std::exit(1); }
        StandUpFLIP();
        double t = 0; uint64_t it = 0;
        if (!Checkpoint::Load("ckpt_flip.ckpt", t, it, e))
        { std::printf("  FLIP load failed: %s\n", e.c_str()); std::exit(1); }
        for (unsigned long long s = half; s < total; s++) StepFLIP(s);
    }
    return SumCloud(SIMULATION.particles.data.data(), SIMULATION.particles.count, pgs);
}

static void StandUpSPH()
{
    DIMENSION = 3; SelectMethod("SPH"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 16;
    SIMULATION.dt = 0.0005; SIMULATION.dtConfigured = 0.0005; SIMULATION.time = 0.0;
    SIMULATION.f = Vec3{0.0, -9.81, 0.0};
    SIMULATION.particleOptions.contact = 0; SIMULATION.particleOptions.restitution = 0.0;
    SIMULATION.grid.nonUniform = false;
    SIMULATION.RHO = 1000.0; SIMULATION.EPS = 3.5e-3;
    SPH_SETTINGS.kernelRadius = 0.10;
    SPH_SETTINGS.particleSpacing = 0.04;
    SPH_SETTINGS.soundSpeed = 20.0;
    SPH_SETTINGS.delta = 0.1;
    SPH_SETTINGS.alpha = 0.02;

    pgs.InitializeGrid(SIMULATION.domain, false);
    pga.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.dh = pgs.dh; SIMULATION.Nx = pgs.Nx; SIMULATION.Ny = pgs.Ny; SIMULATION.Nz = pgs.Nz;
    SIMULATION.GRID_SOL = &pgs; SIMULATION.GRID_ANT = &pga;
    if (!SCENE.Load("ckpt_sph3.scene")) { std::printf("scene: %s\n", SCENE.error.c_str()); std::exit(1); }
    SPH3D::InitializeSPH3D(&SIMULATION.particles, 200000);
}

// `withFlag` off resumes as if the clock were at zero, so the first resumed substep takes leapfrog's
// opening half kick -- what the old `leapfrogPrimed` flag guarded against.
static Cloud RunSPH(unsigned long long total, bool split, bool withFlag)
{
    const unsigned long long half = total / 2;
    StandUpSPH();
    for (unsigned long long s = 0; s < (split ? half : total); s++)
        SPH3D::AdvanceSPH(&pga, &pgs, &SIMULATION.particles, s * SIMULATION.dt);
    if (split)
    {
        std::string e;
        if (!Checkpoint::Save("ckpt_sph.ckpt", half * SIMULATION.dt, half, e))
        { std::printf("  SPH save failed: %s\n", e.c_str()); std::exit(1); }
        StandUpSPH();
        double t = 0; uint64_t it = 0;
        if (!Checkpoint::Load("ckpt_sph.ckpt", t, it, e))
        { std::printf("  SPH load failed: %s\n", e.c_str()); std::exit(1); }
        for (unsigned long long s = half; s < total; s++)
            SPH3D::AdvanceSPH(&pga, &pgs, &SIMULATION.particles,
                              (withFlag || s != half) ? s * SIMULATION.dt : 0.0);
    }
    return SumCloud(SIMULATION.particles.data.data(), SIMULATION.particles.count, pgs);
}

static double firstAfterHalfADI = 0.0, firstAfterHalfSolve = 0.0;

int run(int, char **)
{
    std::printf("checkpoint_check: a resumed run must be the same computation\n");
    {
        std::ofstream f("ckpt.scene");
        f << "sampling center\n"
             "boundary xmin inflow u=\"1.5 - (1.5/(0.205*0.205))*(y-0.205)^2\" clamp_negative=1\n"
             "boundary xmax pressure_outlet\nboundary ymin noslip\nboundary ymax noslip\n"
             "region solid cylinder axis=z center=(0.2,0.2,0.5) radius=0.05\n";
    }
    {
        std::ofstream f("ckpt_flip3.scene");
        f << "sampling center\n"
             "boundary xmin noslip\nboundary xmax noslip\nboundary ymin noslip\n"
             "boundary ymax noslip\nboundary zmin noslip\nboundary zmax noslip\n"
             "region fluid box min=(0,0,0) max=(0.15,0.2,0.3)\n";
        std::ofstream g("ckpt_sph3.scene");
        g << "sampling center\n"
             "boundary xmin noslip\nboundary xmax noslip\nboundary ymin noslip\n"
             "boundary ymax noslip\nboundary zmin noslip\nboundary zmax noslip\n"
             "region fluid box min=(0,0,0) max=(0.4,0.5,1.0)\n";
    }
    const unsigned long long half = 30, total = 60;

    // ---- 1. straight through ----
    StandUp();
    double clock = 0.0;
    Sums wholeAtHalf;
    for (unsigned long long s = 0; s < total; s++)
    {
        Step(s, clock);
        if (s + 1 == half) wholeAtHalf = Measure(clock, half);
        if (s == half) { firstAfterHalfADI = probeAfterADI; firstAfterHalfSolve = probeAfterSolve; }
    }
    const Sums whole = Measure(clock, total);
    std::printf("  reference: %llu steps, t = %.6f, sum|u| = %.12f\n", total, whole.time, whole.u);

    // ---- 1b. the same 60 steps, but with the SOLVERS re-initialised at step 30 and the
    //          grids left alone. This separates "the checkpoint lost field state" from "the
    //          solvers carry state across steps that a restart cannot have".
    StandUp();
    clock = 0.0;
    for (unsigned long long s = 0; s < half; s++) Step(s, clock);
    ADI2D::InitializeADI2D(&gs, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction2D, 
                           SIMULATION.PressureBoundaryFunction2D);
    PressureSolver2D::InitializePressureSolver(&gs, ADIPressureScheme());
    for (unsigned long long s = half; s < total; s++) Step(s, clock);
    const Sums reinit = Measure(clock, total);
    std::printf("  solver re-init at step %llu: sum|u| = %.12f  (straight through %.12f)\n",
                half, reinit.u, whole.u);

    // ---- 2. half, save ----
    StandUp();
    clock = 0.0;
    for (unsigned long long s = 0; s < half; s++) Step(s, clock);
    const Sums atHalf = Measure(clock, half);
    // Two runs in one process: a second StandUp has to be a clean slate, or the comparison
    // below is measuring re-initialisation rather than the checkpoint.
    check(atHalf.u == wholeAtHalf.u && atHalf.p == wholeAtHalf.p,
          "a second run in the same process reproduces the first",
          "d(sum|u|) = " + std::to_string(atHalf.u - wholeAtHalf.u) +
              ", d(sum p) = " + std::to_string(atHalf.p - wholeAtHalf.p));
    std::string err;
    check(Checkpoint::Save("run.ckpt", clock, half, err), "Save writes a checkpoint", err);
    std::string desc;
    check(Checkpoint::Describe("run.ckpt", desc), "Describe reads the header back", desc);

    // ---- 3. rebuild from nothing, load, finish ----
    // StandUp rebuilds the grids from the config and the scene, exactly as the app does at
    // startup, so the load has to restore everything the first 30 steps produced.
    StandUp();
    double loadedClock = 0.0;
    uint64_t loadedIt = 0;
    check(Checkpoint::Load("run.ckpt", loadedClock, loadedIt, err), "Load restores it", err);
    check(loadedIt == half && std::fabs(loadedClock - atHalf.time) == 0.0,
          "the clock comes back exactly", "t = " + std::to_string(loadedClock) +
              ", step " + std::to_string(loadedIt));
    const Sums restored = Measure(loadedClock, loadedIt);
    // The pressure is not saved: every step recomputes it before anything reads it.
    check(restored.u == atHalf.u && restored.v == atHalf.v,
          "GRID_SOL velocity comes back bit-identical",
          "sum|u| " + std::to_string(restored.u) + " against " + std::to_string(atHalf.u));
    check(restored.au == atHalf.au && restored.av == atHalf.av,
          "GRID_ANT velocity comes back bit-identical",
          "d(sum|u|) = " + std::to_string(restored.au - atHalf.au));

    // Byte-for-byte: save again straight after loading. If the two files are identical, every
    // field, mask and spacing table came back exactly, and any later divergence is state that
    // lives outside the grids.
    check(Checkpoint::Save("run2.ckpt", loadedClock, loadedIt, err), "save again after load", err);
    {
        std::ifstream a("run.ckpt", std::ios::binary), b("run2.ckpt", std::ios::binary);
        std::string da((std::istreambuf_iterator<char>(a)), std::istreambuf_iterator<char>());
        std::string db((std::istreambuf_iterator<char>(b)), std::istreambuf_iterator<char>());
        size_t firstDiff = std::string::npos;
        for (size_t n = 0; n < std::min(da.size(), db.size()); n++)
            if (da[n] != db[n]) { firstDiff = n; break; }
        check(da.size() == db.size() && firstDiff == std::string::npos,
              "the saved state round trips byte for byte",
              da.size() == db.size() ? ("first differing byte " + std::to_string(firstDiff) +
                                        " of " + std::to_string(da.size()))
                                     : "sizes differ");
    }

    // The first resumed step, bisected: if the post-ADI field already differs, the missing
    // state is the momentum solver's; if only the post-solve pressure differs, it is the
    // pressure solver's.
    Step(loadedIt, clock);
    check(probeAfterADI == firstAfterHalfADI, "first resumed step: ADI output identical",
          "d = " + std::to_string(probeAfterADI - firstAfterHalfADI));
    check(probeAfterSolve == firstAfterHalfSolve, "first resumed step: pressure solve identical",
          "d = " + std::to_string(probeAfterSolve - firstAfterHalfSolve));
    for (unsigned long long s = loadedIt + 1; s < total; s++) Step(s, clock);
    const Sums resumed = Measure(clock, total);
    std::printf("  resumed:   %llu steps, t = %.6f, sum|u| = %.12f\n", total, resumed.time, resumed.u);
    check(resumed.u == whole.u && resumed.v == whole.v && resumed.p == whole.p,
          "30 + 30 == 60, to the last digit",
          "d(sum|u|) = " + std::to_string(resumed.u - whole.u) +
              ", d(sum p) = " + std::to_string(resumed.p - whole.p));
    // The ADI sweep grids hold no state between steps, so standing the solvers up again mid-run
    // changes nothing: that is what lets the checkpoint carry no solver block for ADI.
    check(reinit.u == whole.u && reinit.v == whole.v && reinit.p == whole.p,
          "re-initialising the solvers mid-run changes nothing (no hidden solver state)",
          "d(sum|u|) = " + std::to_string(reinit.u - whole.u) +
              ", d(sum p) = " + std::to_string(reinit.p - whole.p));
    check(resumed.div == whole.div, "divergence identical too",
          std::to_string(resumed.div) + " against " + std::to_string(whole.div));

    // ---- 4. a checkpoint from another grid is refused, not half-applied ----
    StandUp(32);
    double t2 = 0; uint64_t i2 = 0;
    const bool refused = !Checkpoint::Load("run.ckpt", t2, i2, err);
    check(refused && err.find("this run is") != std::string::npos,
          "a checkpoint from a different grid is refused with a readable error", err);

    // ---- 5. garbage is refused ----
    { std::ofstream f("bad.ckpt", std::ios::binary); f << "not a checkpoint at all, really"; }
    check(!Checkpoint::Load("bad.ckpt", t2, i2, err), "a non-checkpoint file is refused", err);

    // ---- 6. the auto-save POLICY (the pure predicate the step loop uses) ----
    {
        CheckpointSettings c;
        c.autoSave = true; c.interval = 250;
        check(!ShouldAutoSaveCheckpoint(0, c), "step 0 is not a save point (nothing has run yet)");
        check(ShouldAutoSaveCheckpoint(250, c) && ShouldAutoSaveCheckpoint(500, c),
              "every interval-th step is");
        check(!ShouldAutoSaveCheckpoint(249, c) && !ShouldAutoSaveCheckpoint(251, c),
              "and only those");
        c.autoSave = false;
        check(!ShouldAutoSaveCheckpoint(250, c), "off means off");
        c.autoSave = true; c.interval = 0;
        check(!ShouldAutoSaveCheckpoint(250, c), "a zero interval cannot divide by zero");
    }

    // ---- 7. auto-save OVERWRITES: one file, the newest state, nothing accumulated ----
    {
        StandUp();
        double c2 = 0.0;
        for (unsigned long long s = 0; s < 5; s++) Step(s, c2);
        std::string e;
        Checkpoint::Save("over.ckpt", c2, 5, e);
        const Sums five = Measure(c2, 5);
        for (unsigned long long s = 5; s < 10; s++) Step(s, c2);
        Checkpoint::Save("over.ckpt", c2, 10, e);

        StandUp();
        double t3 = 0; uint64_t i3 = 0;
        check(Checkpoint::Load("over.ckpt", t3, i3, e), "the overwritten file loads", e);
        const Sums back = Measure(t3, i3);
        check(i3 == 10 && back.u != five.u,
              "it holds the LATEST state, not the first one or a series",
              "step " + std::to_string(i3));
        std::remove("over.ckpt");
        check(!std::ifstream("over.ckpt.tmp").good(), "no .tmp left behind");
    }

    // ---- 8. a checkpoint with more particles than the run allocated is refused ----
    {
        std::string e;
        SIMULATION.particles.data.assign(4, Particle());
        SIMULATION.particles.count = 4;
        check(Checkpoint::Save("flip.ckpt", 0.0, 1, e), "a checkpoint with 4 particles saves", e);
        SIMULATION.particles.data.assign(2, Particle());
        SIMULATION.particles.count = 0;
        double t = 0; uint64_t it = 0;
        const bool refused = !Checkpoint::Load("flip.ckpt", t, it, e);
        check(refused && e.find("room for 2") != std::string::npos,
              "loading it into a run with room for 2 is refused", e);
        SIMULATION.particles = ParticleSet();
        std::remove("flip.ckpt");
    }

    // ---- 9. FLIP: the particles ARE the state ----
    //
    // Same instrument as the Eulerian case: 30 + 30 must equal 60 to the last digit, with
    // the solver torn down and rebuilt from the config in between. The grids alone cannot
    // carry a FLIP run -- the particle velocity is what the blend accumulates, and the
    // FLUID/EMPTY split is re-derived from the positions -- so this is what says the block
    // holds everything.
    std::printf("\n== FLIP3D: 30 + 30 == 60 ==\n");
    {
        const Cloud pWhole = RunFLIP(60, false);
        const Cloud pSplit = RunFLIP(60, true);
        std::printf("  60 steps straight: %d particles, sum|x| = %.12f, sum|u| = %.12f\n",
                    pWhole.n, pWhole.x, pWhole.u);
        std::printf("  30 + restart + 30: %d particles, sum|x| = %.12f, sum|u| = %.12f\n",
                    pSplit.n, pSplit.x, pSplit.u);
        check(pSplit.n == pWhole.n, "the same particle count comes back",
              std::to_string(pSplit.n) + " against " + std::to_string(pWhole.n));
        check(pSplit.x == pWhole.x && pSplit.y == pWhole.y && pSplit.z == pWhole.z,
              "every position is bit-identical",
              "d(sum|x|) = " + std::to_string(pSplit.x - pWhole.x));
        check(pSplit.u == pWhole.u && pSplit.v == pWhole.v && pSplit.w == pWhole.w,
              "and every velocity is too",
              "d(sum|u|) = " + std::to_string(pSplit.u - pWhole.u));
        check(pSplit.gu == pWhole.gu && pSplit.gp == pWhole.gp,
              "so is the grid the particles transferred to",
              "d(sum|u| grid) = " + std::to_string(pSplit.gu - pWhole.gu));
    }

    // ---- 10. SPH: positions, velocities, and the leapfrog flag ----
    std::printf("\n== SPH3D: 30 + 30 == 60 ==\n");
    {
        const Cloud sWhole = RunSPH(60, false, true);
        const Cloud sSplit = RunSPH(60, true, true);
        std::printf("  60 steps straight: %d particles, sum|x| = %.12f, sum|u| = %.12f\n",
                    sWhole.n, sWhole.x, sWhole.u);
        std::printf("  30 + restart + 30: %d particles, sum|x| = %.12f, sum|u| = %.12f\n",
                    sSplit.n, sSplit.x, sSplit.u);
        check(sSplit.n == sWhole.n && sSplit.x == sWhole.x && sSplit.y == sWhole.y &&
              sSplit.z == sWhole.z && sSplit.u == sWhole.u && sSplit.v == sWhole.v &&
              sSplit.w == sWhole.w,
              "a resumed SPH run is bit-identical",
              "d(sum|x|) = " + std::to_string(sSplit.x - sWhole.x) +
              ", d(sum|u|) = " + std::to_string(sSplit.u - sWhole.u));

        // The flag on its own: restore without it and the first resumed substep kicks by
        // dt/2 instead of dt. That is a small, plausible-looking difference -- exactly the
        // kind this file exists to catch -- so it is asserted separately.
        const Cloud noFlag = RunSPH(60, true, false);
        check(noFlag.u != sWhole.u,
              "and the opening half kick is real: resuming with the clock at zero changes the answer",
              "sum|u| " + std::to_string(noFlag.u) + " against " + std::to_string(sWhole.u));
    }

    std::remove("run.ckpt"); std::remove("bad.ckpt"); std::remove("ckpt.scene");
    std::remove("ckpt_flip.ckpt"); std::remove("ckpt_flip3.scene");
    std::remove("ckpt_sph.ckpt"); std::remove("ckpt_sph3.scene");
    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace checkpoint_check

// ------------------------------------------------------------------------------------------------------------------
// export_check
// ------------------------------------------------------------------------------------------------------------------
namespace export_check
{
// Grid export carries the registered scalars, and the volumetric export writes them on their own.
//
// Writes a 2D uniform, a 3D uniform and a 2D rectilinear grid (all .vtr), each with two scalars filled with a pattern
// that encodes the cell index, then reads every file back: each scalar array under its name with every value on its own
// point, the coordinates equal to the cell centres, and each .npy (header, shape, float32 values in [k, i, j] order).
// Checking the text of the file would not catch a wrong ordering or a malformed array, which is the failure that matters.


static MAC grid;

// A value that says which cell it came from, and which field: a misplaced value cannot match its pattern.
static int FAILS = 0;
static void Expect(bool ok, const std::string &what)
{
    std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) FAILS++;
}

// The numbers of one ascii <DataArray Name="name"> of a .vtr, in file order.
static std::vector<double> ReadArray(const std::string &text, const std::string &name)
{
    std::vector<double> out;
    const size_t at = text.find("Name=\"" + name + "\"");
    if (at == std::string::npos) return out;
    const size_t open = text.find('>', at), close = text.find("</DataArray>", at);
    std::istringstream in(text.substr(open + 1, close - open - 1));
    double v;
    while (in >> v) out.push_back(v);
    return out;
}

// A .npy file as written by WriteNpyFloat32: the header's dtype and shape, and the float32 values.
static bool ReadNpy(const std::string &path, std::string &header, std::vector<float> &data)
{
    std::ifstream f(path, std::ios::binary);
    char magic[10];
    if (!f.read(magic, 10) || std::string(magic + 1, 5) != "NUMPY") return false;
    const unsigned len = (unsigned char)magic[8] | ((unsigned char)magic[9] << 8);
    header.assign(len, ' ');
    if (!f.read(&header[0], len)) return false;
    data.clear();
    float x;
    while (f.read(reinterpret_cast<char *>(&x), sizeof(x))) data.push_back(x);
    return true;
}
static double Pattern(int field, int i, int j, int k)
{
    const double v = 400.0 * k + 20.0 * i + j;
    return (field == 0) ? v : -v - 0.5;
}

static void StandUp(int dimension, int gridSize, bool refine)
{
    DIMENSION = dimension; SelectMethod("ADI"); THREAD_COUNT = 1;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = gridSize;
    SIMULATION.RE = 100;
    SIMULATION.grid.nonUniform = refine;
    SIMULATION.grid.refineX.clear(); SIMULATION.grid.refineY.clear(); SIMULATION.grid.refineZ.clear();
    if (refine)
        SIMULATION.grid.refineX.push_back(GridRefinement{true, 0.3, 0.6, 2, REFINE_HARD});

    grid.ClearScalars();
    grid.InitializeGrid(SIMULATION.domain, dimension == 2);
    grid.RegisterScalar("d", TransportScheme::Upwind);
    grid.RegisterScalar("T", TransportScheme::Central);

    const int nz = (dimension == 2) ? 1 : grid.Nz;
    for (int h = 0; h < 2; h++)
        for (int k = 0; k < nz; k++)
            for (int i = 0; i < grid.Ny; i++)
                for (int j = 0; j < grid.Nx; j++)
                {
                    if (dimension == 2) grid.SetS(h, i, j, Pattern(h, i, j, 0));
                    else grid.SetS(h, i, j, k, Pattern(h, i, j, k));
                }
    // One cell carries a value with more digits than the writer keeps, to measure it.
    if (dimension == 2) grid.SetS(0, 1, 1, 1.0 / 3.0); else grid.SetS(0, 1, 1, 1, 1.0 / 3.0);
    std::printf("case %dD %s: %d x %d x %d\n", dimension, refine ? "rectilinear" : "uniform",
                grid.Nx, grid.Ny, nz);
    Exports::WriteGrid(grid, 1);
    std::string message;
    Expect(Exports::WriteVolumetric(grid, 1, 0.25, message), "volumetric export: " + message);

    // ---- read the .vtr back ----
    std::ifstream f(ExportDirectory("VTK") + "grid-0001.vtr");
    std::stringstream buf;
    buf << f.rdbuf();
    const std::string text = buf.str();
    Expect(!text.empty() && text.find("RectilinearGrid") != std::string::npos, "the .vtr exists and is a RectilinearGrid");
    const size_t n = (size_t)nz * grid.Ny * grid.Nx;
    for (int h = 0; h < 2; h++)
    {
        const std::vector<double> a = ReadArray(text, grid.ScalarName(h));
        bool placed = a.size() == n;
        double third = 0.0;
        for (int k = 0, q = 0; placed && k < nz; k++)
            for (int i = 0; i < grid.Ny; i++)
                for (int j = 0; j < grid.Nx; j++, q++)
                {
                    if (h == 0 && i == 1 && j == 1 && k == (dimension == 2 ? 0 : 1)) { third = a[(size_t)q]; continue; }
                    if (a[(size_t)q] != Pattern(h, i, j, dimension == 2 ? 0 : k)) placed = false;
                }
        Expect(placed, "scalar '" + grid.ScalarName(h) + "': " + std::to_string(a.size()) + " values, each on its point");
        if (h == 0) Expect(std::abs(third - 1.0 / 3.0) < 1e-11, "a value with more digits survives to 1e-11");
    }
    double worst = 0.0;
    const std::vector<double> X = ReadArray(text, "X"), Y = ReadArray(text, "Y");
    bool sized = (int)X.size() == grid.Nx && (int)Y.size() == grid.Ny;
    for (int j = 0; sized && j < grid.Nx; j++) worst = std::max(worst, std::abs(X[j] - grid.CentrePos(MAC::X, j)));
    for (int i = 0; sized && i < grid.Ny; i++) worst = std::max(worst, std::abs(Y[i] - grid.CentrePos(MAC::Y, i)));
    if (dimension == 3)
    {
        const std::vector<double> Z = ReadArray(text, "Z");
        sized = sized && (int)Z.size() == grid.Nz;
        for (int k = 0; sized && k < grid.Nz; k++) worst = std::max(worst, std::abs(Z[k] - grid.CentrePos(MAC::Z, k)));
    }
    Expect(sized && worst < 1e-11, "the coordinates are the cell centres (worst " + std::to_string(worst) + ")");

    // ---- read the .npy files back ----
    const std::string vol = ExportDirectory("Volumetric");
    const std::string shape = dimension == 2
        ? "(" + std::to_string(grid.Ny) + ", " + std::to_string(grid.Nx) + ")"
        : "(" + std::to_string(nz) + ", " + std::to_string(grid.Ny) + ", " + std::to_string(grid.Nx) + ")";
    for (int h = 0; h < 2; h++)
    {
        std::string header;
        std::vector<float> data;
        const bool read = ReadNpy(vol + grid.ScalarName(h) + "_00001.npy", header, data);
        Expect(read && header.find("<f4") != std::string::npos && header.find("False") != std::string::npos &&
                   header.find(shape) != std::string::npos,
               "'" + grid.ScalarName(h) + "' .npy header: float32, C order, shape " + shape);
        bool same = read && data.size() == n;
        for (int k = 0, q = 0; same && k < nz; k++)
            for (int i = 0; i < grid.Ny; i++)
                for (int j = 0; j < grid.Nx; j++, q++)
                    if (data[(size_t)q] != (float)(dimension == 2 ? grid.GetS(h, i, j) : grid.GetS(h, i, j, k))) same = false;
        Expect(same, "'" + grid.ScalarName(h) + "' .npy values in [k, i, j] order");
    }
    Expect(std::ifstream(vol + "frame_00001.json").good(), "the frame's .json is written");
}

int run(int, char **)
{
    StandUp(2, 20, false);   // Exports/ADI/20_2D_re100/VTK/grid-0001.vtr
    StandUp(3, 12, false);   // Exports/ADI/12_3D_re100/VTK/grid-0001.vtr
    StandUp(2, 16, true);    // Exports/ADI/16_2D_re100/VTK/grid-0001.vtr
    std::printf("%s (%d failure%s)\n", FAILS ? "EXPORT CHECK FAILED" : "ALL CHECKS PASSED", FAILS, FAILS == 1 ? "" : "s");
    return FAILS ? 1 : 0;
}
} // namespace export_check


// ------------------------------------------------------------------------------------------------------------------
// restart_check
// ------------------------------------------------------------------------------------------------------------------
namespace restart_check
{
// Restart: SimulationManager::DisposeSimulation must leave nothing behind. Four small runs, one per method (ADI 2D,
// FLIP 3D, SPH 2D, EXPLICIT 2D), are each run once in a fresh process, then all four twice in a row in this one,
// disposed between runs. Every run must reproduce its fresh-process fingerprint to the last bit.

static int fails = 0;

static const char *PRESSURE_BLOCK =
    "PRESSURE_BEGIN\nNEEDS_COMPATIBILITY_CONDITION = 1\nPRESSURE_WARM_START = 0\neigen_solver=AMG\n"
    "eigen_tolerance=1e-09\neigen_max_iters=2000\nPRESSURE_END\n";

static void WriteConfigs()
{
    const std::string common = "GPU_ACCELERATION = 0\nTHREAD_COUNT = 2\nADAPTATIVE_TIMESTEP = 0\nDENSITY = 1000\n"
                               "EXPORT_ENABLED = 0\nSTOP_CONDITION = NEVER\n";
    std::ofstream("restart_adi.cfg")
        << "DIMENSION = 2\nTYPE = ADI\n" << common
        << "DOMAIN_X0 = 0\nDOMAIN_XF = 1\nDOMAIN_Y0 = 0\nDOMAIN_YF = 1\nGRID_SIZE = 32\nTIME_STEP = 0.005\n"
           "VISCOSITY = 0.01\n" << PRESSURE_BLOCK
        << "SCENE_BEGIN\nsampling center\nboundary ymax moving u=\"1\"\nSCENE_END\n";
    std::ofstream("restart_flip.cfg")
        << "DIMENSION = 3\nTYPE = FLIP\n" << common
        << "DOMAIN_X0 = 0\nDOMAIN_XF = 1\nDOMAIN_Y0 = 0\nDOMAIN_YF = 0.5\nDOMAIN_Z0 = 0\nDOMAIN_ZF = 0.5\n"
           "GRID_SIZE = 12\nTIME_STEP = 0.002\nVISCOSITY = 1e-06\nFv = -9.81\n" << PRESSURE_BLOCK
        << "SCENE_BEGIN\nsampling center\nregion fluid box min=(0,0,0) max=(0.3,0.3,0.5)\nSCENE_END\n";
    std::ofstream("restart_sph.cfg")
        << "DIMENSION = 2\nTYPE = SPH\n" << common
        << "DOMAIN_X0 = 0\nDOMAIN_XF = 0.6\nDOMAIN_Y0 = 0\nDOMAIN_YF = 0.4\nGRID_SIZE = 24\nTIME_STEP = 0.0005\n"
           "VISCOSITY = 1e-06\nFv = -9.81\nSPH_BEGIN\nSPH_KERNEL_RADIUS = 0.03\nSPH_END\n"
        << "SCENE_BEGIN\nsampling center\nregion fluid box min=(0,0,0) max=(0.15,0.3,1)\nSCENE_END\n";
    std::ofstream("restart_explicit.cfg")
        << "DIMENSION = 2\nTYPE = EXPLICIT\n" << common
        << "DOMAIN_X0 = 0\nDOMAIN_XF = 1\nDOMAIN_Y0 = 0\nDOMAIN_YF = 1\nGRID_SIZE = 24\nTIME_STEP = 0.002\n"
           "VISCOSITY = 0.01\n"
        << "SCENE_BEGIN\nsampling center\nboundary ymax moving u=\"1\"\nSCENE_END\n";
}

// Runs `config` for 20 steps and returns the fields and particles as one line of full-precision sums.
static std::string Run(const char *config)
{
    SimulationManager::InitializeSimulation(config);
    SIMULATION.running = true;
    for (int s = 0; s < 20; s++) SimulationManager::StepSimulation();
    MAC &g = *SIMULATION.GRID_SOL;
    double su = 0, sv = 0, sw = 0, sp = 0, sx = 0, sq = 0;
    const int nz = (DIMENSION == 3) ? g.Nz : 1;
    for (int k = 0; k < nz; k++)
        for (int i = 0; i < g.Ny; i++)
            for (int j = 0; j < g.Nx; j++)
            {
                if (DIMENSION == 3)
                {
                    su += std::abs(g.GetU(i, j, k)); sv += std::abs(g.GetV(i, j, k));
                    sw += std::abs(g.GetW(i, j, k)); sp += g.GetP(i, j, k);
                }
                else
                {
                    su += std::abs(g.GetU(i, j)); sv += std::abs(g.GetV(i, j)); sp += g.GetP(i, j);
                }
            }
    for (int k = 0; k < SIMULATION.particles.count; k++)
    {
        const Particle &q = SIMULATION.particles.data[k];
        sx += q.x + q.y + q.z;
        sq += std::abs(q.u) + std::abs(q.v) + std::abs(q.w);
    }
    char line[400];
    std::snprintf(line, sizeof line, "%.17g %.17g %.17g %.17g %.17g %.17g n=%d it=%llu t=%.17g", su, sv, sw, sp, sx, sq,
                  SIMULATION.particles.count, (unsigned long long)SIMULATION.telemetry.iteration,
                  SIMULATION.telemetry.clock);
    return line;
}

int run(int, char **)
{
    std::printf("restart_check: a run after DisposeSimulation is the run a fresh process makes\n");
    WriteConfigs();
    RegisterBuiltinMethods();
    const char *configs[] = {"restart_adi.cfg", "restart_flip.cfg", "restart_sph.cfg", "restart_explicit.cfg"};
    std::string fresh[4];
    for (int c = 0; c < 4; c++)
    {
        const std::string out = std::string(configs[c]) + ".fresh";
        const pid_t pid = fork();
        if (pid == 0)
        {
            std::ofstream(out) << Run(configs[c]);
            std::_Exit(0);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        std::ifstream in(out);
        std::getline(in, fresh[c]);
    }
    for (int round = 0; round < 2; round++)
        for (int c = 0; c < 4; c++)
        {
            const std::string again = Run(configs[c]);
            SimulationManager::DisposeSimulation();
            const bool ok = !fresh[c].empty() && again == fresh[c];
            std::printf("  [%s] %s, run %d in one process -- %s\n", ok ? "PASS" : "FAIL", configs[c],
                        round * 4 + c + 1, again.c_str());
            if (!ok) { std::printf("         fresh process: %s\n", fresh[c].c_str()); fails++; }
        }
    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace restart_check

static const SuiteCase CASES[] = {
    {"checkpoint", checkpoint_check::run, {}, true},
    {"export", export_check::run, {}, true},
    {"restart", restart_check::run, {}, true},
};

int main(int argc, char **argv) { return RunSuite("io", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
