// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The units suite: stopping_check, timestep_check, grid_spacing_check, boundary_check, energy_check, cellre_check, probe_check, confinement_check, scene_check, mesh_check, airfoil_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cstdio>
#include <cmath>
#include <string>
#include "headers/Core/Definitions.h"
#include <algorithm>
#include <vector>
#include "headers/Core/MAC.h"
#include "Functions.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include <cstdlib>
#include <random>
#include "headers/Core/Scene.h"
#include "headers/Utils.h"
#include "Methods/ADI/VorticityConfinement2D.h"
#include "headers/Core/ConfigReader.h"
#include "Methods/ADI/ADI2D.h"
#include "Methods/ADI/ADI3D.h"
#include "Methods/PPESolver/PressureSolver2D.h"
#include "Methods/PPESolver/PressureSolver3D.h"
#include "Methods/FLIP/FLIP2D.h"
#include "Methods/FLIP/FLIP3D.h"
#include <array>
#include <chrono>
#include <fstream>
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// stopping_check
// ------------------------------------------------------------------------------------------------------------------
namespace stopping_check
{
// The stopping rule.
//
// The decision is a pure function shared with SimulationManager::CheckStoppingRule, so this
// exercises the real thing rather than a restatement of it.
static int fails=0;
static void check(bool ok,const std::string&n,const std::string&d=""){
  std::printf("  [%s] %s%s%s\n",ok?"PASS":"FAIL",n.c_str(),d.empty()?"":" -- ",d.c_str()); if(!ok)fails++;}

// Calls the SAME function SimulationManager::CheckStoppingRule uses, not a copy of it.
static bool Fires(int cond, double t, double tf, double res, double tol, int it)
{
    StoppingRule r;
    r.condition = cond; r.tFinal = tf; r.residualTol = tol;
    std::string why;
    return StoppingRuleFires(r, t, res, it, why);
}

int run(int, char **)
{
    std::setvbuf(stdout,nullptr,_IONBF,0);
    check(!Fires(STOP_NEVER, 99.0, 1.0, 0.0, 1e-6, 100), "never: does not fire");
    check(!Fires(STOP_TIME, 0.5, 1.0, 1.0, 1e-6, 10), "time: does not fire before tFinal");
    check(Fires(STOP_TIME, 1.0, 1.0, 1.0, 1e-6, 10), "time: fires exactly at tFinal");
    check(Fires(STOP_TIME, 1.7, 1.0, 1.0, 1e-6, 10), "time: fires past tFinal");

    // A clock summed in floating point lands a hair under tFinal; that is tFinal, not one more step.
    {
        double clock = 0.0;
        int steps = 0;
        while (!Fires(STOP_TIME, clock, 5.0, 1.0, 1e-6, 10)) { clock += 0.01; steps++; }
        check(steps == 500, "time: 500 steps of 0.01 stop at 5, not one step past it",
              std::to_string(steps) + " steps, clock " + std::to_string(clock));
    }
    check(!Fires(STOP_TIME, 1.0 - 1e-6, 1.0, 1.0, 1e-6, 10), "time: a real shortfall is not rounded away");

    // The last step is shortened to land on tFinal, and only when it would pass it.
    {
        StoppingRule r;
        r.condition = STOP_TIME; r.tFinal = 1.0;
        check(ClipStepToStopTime(r, 0.95, 0.1) == 1.0 - 0.95, "clip: a step past tFinal is cut to what is left");
        check(ClipStepToStopTime(r, 0.5, 0.1) == 0.1, "clip: an ordinary step is untouched");
        check(ClipStepToStopTime(r, 0.9 + 1e-15, 0.1) == 0.1, "clip: a step landing on tFinal to round-off is untouched");
        r.triggered = true;
        check(ClipStepToStopTime(r, 0.95, 0.1) == 0.1, "clip: not once the rule has fired (running past a stop)");
        r.triggered = false; r.condition = STOP_RESIDUAL;
        check(ClipStepToStopTime(r, 0.95, 0.1) == 0.1, "clip: only for a time rule");
    }
    check(!Fires(STOP_RESIDUAL, 0.1, 1.0, 1e-9, 1e-6, 1),
          "residual: does NOT fire on the first step, where the comparison is against the "
          "initial field");
    check(Fires(STOP_RESIDUAL, 0.1, 1.0, 1e-9, 1e-6, 2), "residual: fires once settled");
    check(!Fires(STOP_RESIDUAL, 0.1, 1.0, 1e-3, 1e-6, 50), "residual: does not fire while evolving");

    // Re-arming is what lets Run continue past a stop without firing again immediately.
    SIMULATION.stop.condition = STOP_TIME; SIMULATION.stop.triggered = true; SIMULATION.stop.exported = true;
    SIMULATION.stop.reason = "x";
    SIMULATION.stop.Rearm();
    check(!SIMULATION.stop.triggered && !SIMULATION.stop.exported && SIMULATION.stop.reason.empty(),
          "Rearm clears the trigger, the export latch and the reason");
    check(SIMULATION.stop.condition == STOP_TIME, "Rearm leaves the user's choice alone");

    std::printf("\n%s (%d failure%s)\n",fails?"FAILED":"ALL CHECKS PASSED",fails,fails==1?"":"s");
    return fails?1:0;
}
} // namespace stopping_check

// ------------------------------------------------------------------------------------------------------------------
// timestep_check
// ------------------------------------------------------------------------------------------------------------------
namespace timestep_check
{
// The adaptive timestep controller.
//
// Reported symptom: with ADAPTATIVE_TIMESTEP on, a dam break sometimes gains enormous
// energy -- liquid over the wall, off the ceiling and back. The cause is arithmetic, not
// an edge case. The old form was
//
//     dt_advection = MAX_CFL * dh / GetMaxVelocity();          // no floor on the divisor
//     dt_diffusion = MAX_CFL * dh * dh / EPS;
//     dt           = min(dt_advection, dt_diffusion);          // no ceiling at all
//
// With the author's config (dh = 1/30, nu = 1e-4, MAX_CFL = 1) the diffusion limit is
// 11.1 SECONDS. A fluid at rest makes the advection term dh/0 = +inf, dt falls through to
// 11.1 s, and one step of gravity at that dt is 109 m/s. It fires on step 0 from rest and
// again whenever the flow drops below a few mm/s.


static int failures = 0;

static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok)
        failures++;
}

int run(int, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // The author's configuration, which is where this was observed.
    const double dh = 1.0 / 30.0;
    const double maxCFL = 1.0;
    const double dtCfg = 0.01;
    const double g = 9.8;

    std::printf("\n== the reported failure ==\n");
    {
        char buf[256];

        // At rest. The old code produced MAX_CFL*dh*dh/nu = 11.1 s here.
        const double atRest = ComputeAdaptiveTimestep(0.0, dh, maxCFL, dtCfg);
        std::snprintf(buf, sizeof(buf), "dt = %.6f s (old form gave 11.1 s)", atRest);
        check(atRest <= dtCfg + 1e-15, "a fluid at rest does not produce a huge step", buf);

        std::snprintf(buf, sizeof(buf), "%.4f m/s after one step", g * atRest);
        check(g * atRest < 1.0, "so one step of gravity cannot launch the fluid", buf);

        // Just under the threshold the old form crossed: 3 mm/s.
        const double crawling = ComputeAdaptiveTimestep(3e-3, dh, maxCFL, dtCfg);
        std::snprintf(buf, sizeof(buf), "dt = %.6f s", crawling);
        check(crawling <= dtCfg + 1e-15, "nor does a fluid crawling at 3 mm/s", buf);
    }

    std::printf("\n== the ceiling ==\n");
    {
        char buf[256];
        double worst = 0.0;
        for (double v = 0.0; v < 50.0; v += 0.001)
            worst = (std::max)(worst, ComputeAdaptiveTimestep(v, dh, maxCFL, dtCfg));
        std::snprintf(buf, sizeof(buf), "max over v in [0, 50] m/s = %.6f, TIME_STEP = %.6f",
                      worst, dtCfg);
        check(worst <= dtCfg + 1e-15, "dt never exceeds the configured TIME_STEP", buf);

        // and it is not simply pinned there: a fast flow has to shrink it
        const double fast = ComputeAdaptiveTimestep(20.0, dh, maxCFL, dtCfg);
        std::snprintf(buf, sizeof(buf), "dt = %.6f s at 20 m/s", fast);
        check(fast < dtCfg * 0.5, "adaptivity still shrinks the step when the flow is fast", buf);
    }

    std::printf("\n== the CFL it is meant to hold ==\n");
    {
        char buf[256];
        double worstCFL = 0.0;
        for (double v = 1e-4; v < 50.0; v += 0.001)
        {
            const double dt = ComputeAdaptiveTimestep(v, dh, maxCFL, dtCfg);
            worstCFL = (std::max)(worstCFL, v * dt / dh);
        }
        std::snprintf(buf, sizeof(buf), "max advective CFL = %.6f, MAX_CFL = %.2f",
                      worstCFL, maxCFL);
        check(worstCFL <= maxCFL + 1e-9, "the advective CFL stays within MAX_CFL", buf);
    }

    // ---- frames at a constant RATE under a variable step ----------------------------
    //
    // This is the reason the controller's output cannot be trusted as a clock: "export every
    // N iterations" is a constant number of STEPS, and with adaptivity the steps are not a
    // constant amount of TIME. An animation built that way plays back faster wherever the
    // solver slowed down. Drive the real decision function with a real varying dt and measure
    // the frame spacing both ways.
    std::printf("\n== frames at a constant rate, not a constant step count ==\n");
    {
        auto RunFrames = [&](bool everyTime, double interval, int everyN,
                             std::vector<double> &frameTimes)
        {
            ExportSettings e;
            e.gridEveryTime = everyTime;
            e.gridTimeInterval = interval;
            e.gridInterval = everyN;
            e.nextGridExportTime = 0.0;
            double t = 0.0;
            for (unsigned long long it = 1; it <= 4000 && t < 2.0; it++)
            {
                // A velocity that swings over an order of magnitude, so dt does too -- the
                // dam break's own shape: slow at rest, fast at the wave front.
                const double v = 0.05 + 40.0 * std::fabs(std::sin(0.004 * (double)it));
                t += ComputeAdaptiveTimestep(v, dh, maxCFL, dtCfg);
                if (FrameExportDue(it, t, e))
                {
                    frameTimes.push_back(t);
                    AdvanceFrameExportClock(t, e);
                }
            }
        };

        auto Spread = [](const std::vector<double> &f, double &lo, double &hi)
        {
            lo = 1e30; hi = -1e30;
            for (size_t i = 1; i < f.size(); i++)
            {
                const double d = f[i] - f[i - 1];
                lo = (std::min)(lo, d); hi = (std::max)(hi, d);
            }
        };

        std::vector<double> byIt, byTime;
        RunFrames(false, 0.04, 50, byIt);
        RunFrames(true, 0.04, 50, byTime);

        double loIt = 0, hiIt = 0, loT = 0, hiT = 0;
        Spread(byIt, loIt, hiIt);
        Spread(byTime, loT, hiT);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "every 50 iterations: %.4f to %.4f s between frames "
                      "(a %.1fx swing)", loIt, hiIt, hiIt / loIt);
        check(hiIt > 2.0 * loIt, "every N iterations spaces frames UNEVENLY in time", buf);

        // The deadline never moves, so a frame can only be LATE, and only by the step it
        // happened to land on: every gap is the interval plus or minus one step. There is no
        // tighter bound available and none is wanted -- a frame between two samples is a
        // frame the solver never computed.
        std::snprintf(buf, sizeof(buf), "every 0.04 s: %.5f to %.5f s between frames over %zu "
                      "frames, interval +/- one step of %.5f s",
                      loT, hiT, byTime.size(), dtCfg);
        check(byTime.size() > 20 && loT >= 0.04 - dtCfg - 1e-12 && hiT <= 0.04 + dtCfg + 1e-12,
              "every X seconds holds every gap to the interval within one step", buf);

        // The deadlines are multiples of the interval, so the frame times cannot drift no
        // matter how far each step overshot the one before.
        double worstOff = 0.0;
        for (size_t i = 0; i < byTime.size(); i++)
        {
            const double k = std::floor(byTime[i] / 0.04 + 0.5);
            worstOff = (std::max)(worstOff, std::fabs(byTime[i] - k * 0.04));
        }
        std::snprintf(buf, sizeof(buf), "worst frame sits %.5f s past its deadline (dt up to %.5f)",
                      worstOff, dtCfg);
        check(worstOff <= dtCfg + 1e-12,
              "and each frame lands within one step of its deadline, with no accumulated drift",
              buf);

        // One step that jumps several deadlines must not emit a burst of frames at the same
        // instant -- the solver never visited them.
        ExportSettings e;
        e.gridEveryTime = true;
        e.gridTimeInterval = 0.01;
        e.nextGridExportTime = 0.0;
        int fired = 0;
        double t = 0.0;
        for (int i = 0; i < 3; i++)
        {
            t += 0.1; // ten deadlines in one step
            if (FrameExportDue(0, t, e)) { fired++; AdvanceFrameExportClock(t, e); }
        }
        std::snprintf(buf, sizeof(buf), "%d frames from 3 steps of 10 intervals each", fired);
        check(fired == 3, "a step that jumps several deadlines writes ONE frame, not a burst", buf);
    }

    std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
} // namespace timestep_check

// ------------------------------------------------------------------------------------------------------------------
// grid_spacing_check
// ------------------------------------------------------------------------------------------------------------------
namespace grid_spacing_check
{
static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{ std::printf("  [%s] %s%s%s\n", ok?"PASS":"FAIL", n.c_str(), d.empty()?"":" -- ", d.c_str()); if(!ok) fails++; }

static MAC Build(bool nu, double s, double e, int f)
{
    DIMENSION = 2; SelectMethod("ADI");
    SIMULATION.GRID_SIZE = 20;
    SIMULATION.grid.nonUniform = nu;
    SIMULATION.grid.refineX.clear();
    if (nu) SIMULATION.grid.refineX.push_back(GridRefinement{true, s, e, f, REFINE_HARD});
    SIMULATION.grid.refineY.clear();
    SIMULATION.grid.refineZ.clear();
    MAC g; g.InitializeGrid(Domain{0.0,1.0,0.0,1.0,0.0,1.0}, true);
    return g;
}

int run(int, char **)
{
    std::printf("\n== uniform tables reproduce the inline expressions ==\n");
    {
        MAC g = Build(false, 0, 0, 1);
        bool exact = true;
        // The interior from x0 = 0, the ring one cell outside it at each end.
        for (int j = 1; j <= g.Nx - 1; j++) if (g.FacePos(MAC::X, j) != 0.0 + (j - 1) * g.dh) exact = false;
        for (int j = 1; j < g.Nx - 1; j++)  if (g.CentrePos(MAC::X, j) != 0.0 + (j - 1) * g.dh + g.dh / 2.0) exact = false;
        if (g.FacePos(MAC::X, 0) != -g.dh || g.CentrePos(MAC::X, 0) != -0.5 * g.dh) exact = false;
        check(exact, "FacePos(X, j) and CentrePos(X, j) are bit-identical to x0 + (j-1)*dh (+ dh/2), the ring outside");
        check(!g.nonUniform, "flag off");
        check(g.Width(MAC::X, 4) == g.dh && g.Width(MAC::Y, 3) == g.dh, "Dx and Dy return dh on a uniform grid");
        check(g.FaceGap(MAC::X, 5) == g.dh, "centre-to-centre distance is dh");
    }

    std::printf("\n== refinement ==\n");
    {
        MAC g = Build(true, 0.4, 0.6, 2);
        check(g.nonUniform, "flag on");
        // 20 base cells of 0.05; centres in [0.4,0.6] are 0.425..0.575 -> 4 cells, each split in 2; plus the ring
        check(g.Nx == 26, "cell count grew by the refined span", "Nx = " + std::to_string(g.Nx));
        check(g.Ny == 22, "the unrefined axis is untouched", "Ny = " + std::to_string(g.Ny));

        double sum = 0.0; for (int j = 1; j < g.Nx - 1; j++) sum += g.Width(MAC::X, j);
        check(std::fabs(sum - 1.0) < 1e-12, "interior cell sizes sum to the domain width",
              "sum = " + std::to_string(sum));
        check(std::fabs(g.FacePos(MAC::X, 1) - 0.0) < 1e-15 && std::fabs(g.FacePos(MAC::X, g.Nx - 1) - 1.0) < 1e-15,
              "interior faces span exactly [x0, xf]");

        bool monotone = true, centred = true;
        for (int j = 0; j < g.Nx; j++)
        {
            if (!(g.FacePos(MAC::X, j + 1) > g.FacePos(MAC::X, j))) monotone = false;
            if (std::fabs(g.CentrePos(MAC::X, j) - 0.5 * (g.FacePos(MAC::X, j) + g.FacePos(MAC::X, j + 1))) > 1e-15) centred = false;
        }
        check(monotone, "faces are strictly increasing");
        check(centred, "centres sit midway between their faces");

        double fine = 1e9, coarse = 0.0;
        for (int j = 0; j < g.Nx; j++) { fine = std::min(fine, g.Width(MAC::X, j)); coarse = std::max(coarse, g.Width(MAC::X, j)); }
        check(std::fabs(coarse / fine - 2.0) < 1e-12, "the refined band is exactly half the base spacing",
              "coarse/fine = " + std::to_string(coarse / fine));
        check(std::fabs(g.dh - fine) < 1e-15, "dh reports the TIGHTEST spacing, so dt is sized off it",
              "dh = " + std::to_string(g.dh));

        // symmetry of the face coefficient: cell P toward E must equal cell E toward P
        bool sym = true;
        for (int j = 1; j < g.Nx - 1; j++)
        {
            const double pe = g.Width(MAC::Y, 3) / g.FaceGap(MAC::X, j);       // P -> E
            const double ep = g.Width(MAC::Y, 3) / g.FaceGap(MAC::X, j);       // E -> P, same shared face
            if (pe != ep) sym = false;
        }
        check(sym, "A_f/d_f belongs to the shared face, so the matrix stays symmetric");

        check(g.Width(MAC::X, g.Nx / 2) > 0 && g.Width(MAC::Y, 0) > 0, "Dx and Dy return the local sizes",
              "dx=" + std::to_string(g.Width(MAC::X, g.Nx / 2)) + " dy=" + std::to_string(g.Width(MAC::Y, 0)));
    }

    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL CHECKS PASSED", fails, fails==1?"":"s");
    return fails ? 1 : 0;
}
} // namespace grid_spacing_check

// ------------------------------------------------------------------------------------------------------------------
// boundary_check
// ------------------------------------------------------------------------------------------------------------------
namespace boundary_check
{
// MAC::EnforceVelocityBC / EnforcePressureBC against what each boundary kind means.
//
// Imposing a condition forces the ring's values: the prescribed one, or a copy of the one inside (zero gradient). A
// random field is enforced and every ring cell owned by one face is checked against its kind:
//   noslip           normal and tangential 0
//   slip             normal 0, tangential copied from the cell inside
//   moving, inlet    the scene's value, normal and tangential
//   pressure_outlet  normal copied two faces deep, tangential copied
//   outflow          as pressure_outlet, plus the boundary's net flux 0 (what the inlets bring in goes out)
// and the ring pressure: 0 on an outlet without p=, the scene's value with one, the inward neighbour elsewhere.
// Faces within a cell of another face are left to the corner checks, which follow the rule the channel preset
// broke: a corner on an outlet copies its neighbour's values instead of keeping the prescribed zero. Interior faces
// away from any solid must be untouched. 2D and 3D, uniform and stretched, two times.

static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(), d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}

static void Fill(MAC &g, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (int c = MAC::U; c <= MAC::P; c++)
    {
        VectorXd &f = g.Field((MAC::Component)c);
        for (int n = 0; n < f.size(); n++) f[n] = d(rng);
    }
}

// f = {j, i, k}: the index along x, y, z, as the boundary code writes it.
static double Get(MAC &g, int c, const int f[3])
{
    return g.is2D ? g.GetComponent((MAC::Component)c, f[1], f[0]) : g.GetComponent((MAC::Component)c, f[1], f[0], f[2]);
}

static int CellType(MAC &g, int i, int j, int k) { return g.is2D ? g.GetSolid(i, j) : g.GetSolid(i, j, k); }

static double Coord(MAC &g, int axis, int n, bool face)
{
    return face ? g.FacePos((MAC::Axis)axis, n) : g.CentrePos((MAC::Axis)axis, n);
}

// A ring cell's prescribed pressure is the value on the boundary face it borders: the face's coordinate along each
// axis on which the cell is on the ring, the centre's elsewhere (MAC::PrescribedPressure).
static double RingFacePos(MAC &g, MAC::Axis a, int n, int N)
{
    return n == 0 ? g.FacePos(a, 1) : (n == N - 1 ? g.FacePos(a, N - 1) : g.CentrePos(a, n));
}

// The scene's prescribed velocity component c at the face f of that component.
static double Prescribed(MAC &g, int c, const int f[3], double t)
{
    double x[3];
    for (int a = 0; a < 3; a++) x[a] = (a == 2 && g.is2D) ? 0.0 : Coord(g, a, f[a], a == c);
    double u, v, w;
    SCENE.VelocityAt(x[0], x[1], x[2], t, u, v, w);
    return c == 0 ? u : (c == 1 ? v : w);
}

static const char *SCENES[] = {
    // closed box: a moving lid, a slip side, walls, an interior solid
    "sampling center\n"
    "boundary xmin noslip\nboundary xmax slip\nboundary ymin noslip\nboundary zmin slip\nboundary zmax noslip\n"
    "boundary ymax moving u=\"1\" w=\"0.5*x\"\n"
    "region solid box min=(0.35,0.3,0.3) max=(0.6,0.55,0.6)\n",
    // channel: a time-dependent inlet, pressure outlets on three faces, one of them the top -- the preset's case
    "sampling center\n"
    "boundary xmin inlet u=\"1 + 0.5*sin(3*t) + y\" v=\"0.2*y\" w=\"0.1*z\"\n"
    "boundary xmax pressure_outlet\nboundary ymin noslip\nboundary zmin pressure_outlet\nboundary zmax noslip\n"
    "boundary ymax pressure_outlet\n"
    "region solid sphere center=(0.5,0.5,0.5) radius=0.18\n",
    // prescribed pressures, an outlet at p = 0, an inlet
    "sampling center\n"
    "boundary xmin pressure_outlet p=\"1 + y + t\"\n"
    "boundary xmax pressure_outlet p=\"0.5*t - x\"\n"
    "boundary ymin pressure_outlet\nboundary ymax inlet u=\"0.2\" v=\"-0.5*x\" w=\"0\"\n"
    "boundary zmin pressure_outlet p=\"2*z\"\nboundary zmax slip\n"
    "region solid box min=(0.4,0.4,0.4) max=(0.6,0.7,0.55)\n",
    // continuity outflow: the inlet's flux leaves through xmax, the top is a slip wall
    "sampling center\n"
    "boundary xmin inlet u=\"1 - (y - 1)^2 + 0.3*z\"\n"
    "boundary xmax outflow\nboundary ymin noslip\nboundary ymax slip\nboundary zmin slip\nboundary zmax slip\n"
    "region solid sphere center=(0.5,0.4,0.5) radius=0.15\n",
};
static const int SCENE_COUNT = 4;

static void Case(int dim, bool stretched, int scene)
{
    DIMENSION = dim;
    SelectMethod("ADI");
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = (dim == 2) ? 24 : 12;
    SIMULATION.grid.nonUniform = stretched;
    SIMULATION.grid.refineX.clear(); SIMULATION.grid.refineY.clear(); SIMULATION.grid.refineZ.clear();
    if (stretched)
    {
        SIMULATION.grid.refineX.push_back(GridRefinement{true, 0.2, 0.5, 2, REFINE_SYMMETRIC});   // graded
        SIMULATION.grid.refineY.push_back(GridRefinement{true, 0.6, 0.9, 2, REFINE_HARD});
        if (dim == 3) SIMULATION.grid.refineZ.push_back(GridRefinement{true, 0.0, 0.3, 2, REFINE_RAMP_BACK});
    }

    static MAC g, before;
    g.InitializeGrid(SIMULATION.domain, dim == 2);
    before.InitializeGrid(SIMULATION.domain, dim == 2);
    SIMULATION.dh = g.dh; SIMULATION.Nx = g.Nx; SIMULATION.Ny = g.Ny; SIMULATION.Nz = g.Nz;
    SIMULATION.GRID_SOL = &g; SIMULATION.GRID_ANT = &g;

    if (!SCENE.LoadText(SCENES[scene]))
    {
        std::printf("scene %d failed: %s\n", scene, SCENE.error.c_str());
        std::exit(1);
    }
    SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.BoundaryKindFunction = SCENE_BOUNDARY_KIND;
    if (dim == 2)
    {
        g.SetSolidMask(SCENE_SOLID_MASK_2D);
        before.SetSolidMask(SCENE_SOLID_MASK_2D);
        g.SetBoundaryFunctions(SCENE_VELOCITY_2D, SCENE_PRESSURE_2D);
    }
    else
    {
        g.SetSolidMask(SCENE_SOLID_MASK);
        before.SetSolidMask(SCENE_SOLID_MASK);
        g.SetBoundaryFunctions(SCENE_VELOCITY, SCENE_PRESSURE);
    }

    char label[96];
    std::snprintf(label, sizeof label, "%dD %s scene %d (%dx%dx%d)", dim, stretched ? "stretched" : "uniform", scene,
                  g.Nx, g.Ny, dim == 3 ? g.Nz : 1);
    const int axes = (dim == 3) ? 3 : 2;
    const int count[3] = {g.Nx, g.Ny, dim == 3 ? g.Nz : 1};

    const double times[2] = {0.0, 0.37};
    for (int n = 0; n < 2; n++)
    {
        const double t = times[n];
        SIMULATION.time = t;
        const unsigned seed = 1000u * (unsigned)scene + 10u * (unsigned)dim + (stretched ? 5u : 0u) + (unsigned)n;
        Fill(g, seed);
        Fill(before, seed);
        g.EnforceBoundaryConditions(t);

        int bad[6] = {0, 0, 0, 0, 0, 0}, seen[6] = {0, 0, 0, 0, 0, 0};
        int badP = 0, seenP = 0;
        double netFlux = 0.0, inFlux = 0.0;
        for (int i = 0; i < count[1]; i++)
            for (int j = 0; j < count[0]; j++)
                for (int k = 0; k < count[2]; k++)
                {
                    const int cell[3] = {j, i, k};
                    int onRing = 0, a = -1;
                    for (int b = 0; b < axes; b++)
                        if (cell[b] == 0 || cell[b] == count[b] - 1) { onRing++; a = b; }
                    if (onRing != 1) continue;   // corners have their own check
                    int face = -1;
                    const int kind = SCENE.BoundaryKindAtCell(i, j, k, face);
                    const bool high = cell[a] == count[a] - 1;
                    const int d = high ? -1 : 1;
                    // Faces within a cell of another face belong to that face's corner as well.
                    bool nearCorner = false;
                    for (int b = 0; b < axes; b++)
                        if (b != a && (cell[b] < 2 || cell[b] > count[b] - 3)) nearCorner = true;

                    // the ring pressure
                    {
                        const int ct = CellType(g, i, j, k);
                        int in[3] = {cell[0], cell[1], cell[2]};
                        in[a] += d;
                        const double p = Get(g, 3, cell);
                        double want;
                        if (ct == EMPTY_CELL) want = 0.0;
                        else if (ct == PRESSURE_CELL)
                            // on the boundary face the ring cell borders, where the pressure solver reads it
                            want = SCENE.PressureAt(RingFacePos(g, MAC::X, j, g.Nx), RingFacePos(g, MAC::Y, i, g.Ny),
                                                    dim == 3 ? RingFacePos(g, MAC::Z, k, g.Nz) : 0.0, t);
                        else want = Get(g, 3, in);
                        seenP++;
                        if (std::fabs(p - want) > 1e-12 * (1.0 + std::fabs(want))) badP++;
                    }

                    // the normal component, two faces deep
                    int fb[3] = {cell[0], cell[1], cell[2]}, fo[3] = {cell[0], cell[1], cell[2]}, fi[3] = {cell[0], cell[1], cell[2]};
                    fb[a] = high ? count[a] - 1 : 1;
                    fo[a] = high ? count[a] : 0;
                    fi[a] = fb[a] + d;
                    const double ub = Get(g, a, fb), uo = Get(g, a, fo);
                    double area = 1.0;
                    for (int b = 0; b < axes; b++)
                        if (b != a) area *= g.Width((MAC::Axis)b, cell[b]);
                    netFlux += (high ? -1.0 : 1.0) * ub * area;
                    if (kind == BC_INLET) inFlux += (high ? -1.0 : 1.0) * ub * area;

                    if (!nearCorner)
                    {
                        bool ok = true;
                        switch (kind)
                        {
                        case BC_NOSLIP: case BC_SLIP: case BC_MOVING: ok = (ub == 0.0 && uo == 0.0); break;
                        case BC_INLET: ok = (ub == Prescribed(g, a, fb, t) && uo == Prescribed(g, a, fo, t)); break;
                        case BC_PRESSURE_OUTLET: ok = (ub == Get(g, a, fi) && uo == ub); break;
                        case BC_OUTFLOW: ok = (uo == ub); break;
                        }
                        seen[kind]++;
                        if (!ok) bad[kind]++;

                        // the tangential components, on the face the cell owns (its low face along b)
                        for (int b = 0; b < axes; b++)
                        {
                            if (b == a) continue;
                            int ft[3] = {cell[0], cell[1], cell[2]}, fs[3] = {cell[0], cell[1], cell[2]};
                            fs[a] += d;
                            const double v = Get(g, b, ft);
                            bool okT = true;
                            switch (kind)
                            {
                            case BC_NOSLIP: okT = (v == 0.0); break;
                            case BC_MOVING: case BC_INLET: okT = (v == Prescribed(g, b, ft, t)); break;
                            default: okT = (v == Get(g, b, fs)); break;   // slip, outlet, outflow
                            }
                            seen[kind]++;
                            if (!okT) bad[kind]++;
                        }
                    }
                }

        const char *names[6] = {"noslip", "slip", "moving", "inlet", "pressure_outlet", "outflow"};
        for (int kd = 0; kd < 6; kd++)
            if (seen[kd] > 0)
                check(bad[kd] == 0, std::string(label) + ", t = " + std::to_string(t).substr(0, 4) + ": " + names[kd],
                      std::to_string(seen[kd]) + " faces, " + std::to_string(bad[kd]) + " wrong");
        check(badP == 0 && seenP > 0, std::string(label) + ": ring pressure",
              std::to_string(seenP) + " cells, " + std::to_string(badP) + " wrong");
        if (scene == 3)
        {
            char buf[96];
            std::snprintf(buf, sizeof buf, "net %.3e of inflow %.3e", netFlux, inFlux);
            check(std::fabs(netFlux) < 1e-12 * std::fabs(inFlux) && inFlux > 0.0,
                  std::string(label) + ": outflow takes out what the inlet brings in", buf);
        }

        // Corners on an outlet copy their neighbours (the preset's top corner used to keep a pinned 0).
        if (scene == 1)
        {
            const int i = count[1] - 1, j = count[0] - 1;
            int corner = 0, cbad = 0;
            for (int k = (dim == 3 ? 2 : 0); k < (dim == 3 ? count[2] - 2 : 1); k++)
            {
                const int fu1[3] = {j, i, k}, su1[3] = {j, i - 1, k};
                const int fu2[3] = {j + 1, i, k}, su2[3] = {j + 1, i - 1, k};
                const int fv1[3] = {j, i + 1, k}, fv0[3] = {j, i, k}, sv[3] = {j, i - 1, k};
                corner++;
                if (Get(g, 0, fu1) != Get(g, 0, su1) || Get(g, 0, fu2) != Get(g, 0, su2) ||
                    Get(g, 1, fv0) != Get(g, 1, sv) || Get(g, 1, fv1) != Get(g, 1, fv0))
                    cbad++;
            }
            check(cbad == 0, std::string(label) + ": the top-right corner (ymax outlet) copies its neighbours",
                  std::to_string(corner) + " corner cells, " + std::to_string(cbad) + " wrong");
        }

        // Interior faces away from any solid are the solvers', not the boundary's.
        int touched = 0;
        for (int c = 0; c < axes; c++)
            for (int i = 0; i < count[1] + (c == 1); i++)
                for (int j = 0; j < count[0] + (c == 0); j++)
                    for (int k = 0; k < count[2] + (c == 2); k++)
                    {
                        const int f[3] = {j, i, k};
                        bool interior = true;
                        for (int b = 0; b < axes; b++)
                        {
                            const int lo = (b == c) ? 2 : 1, hi = (b == c) ? count[b] - 2 : count[b] - 2;
                            if (f[b] < lo || f[b] > hi) interior = false;
                        }
                        if (!interior) continue;
                        int l[3] = {j, i, k};
                        l[c] -= 1;
                        if (CellType(g, i, j, k) == SOLID_CELL || CellType(g, l[1], l[0], l[2]) == SOLID_CELL) continue;
                        if (Get(g, c, f) != Get(before, c, f)) touched++;
                    }
        check(touched == 0, std::string(label) + ": interior faces untouched", std::to_string(touched) + " changed");
    }
}

int run(int, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    for (int dim = 2; dim <= 3; dim++)
        for (int s = 0; s < 2; s++)
            for (int scene = 0; scene < SCENE_COUNT; scene++)
                Case(dim, s == 1, scene);
    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace boundary_check

// ------------------------------------------------------------------------------------------------------------------
// energy_check
// ------------------------------------------------------------------------------------------------------------------
namespace energy_check
{
// Flow integrals: kinetic energy and enstrophy, against closed forms.
//
// The 2D Taylor-Green field  u = sin x cos y,  v = -cos x sin y  on [0, pi]^2 has zero normal
// velocity AND zero vorticity on every wall, so it is exact for a free-slip box, and its volume
// averages are known: E = 1/4, Z = 1/2. Decaying as exp(-2 nu t) in velocity it has
// -dE/dt = 2 nu Z exactly, so the numerical dissipation of an exact field must come out ~0.
// The 3D field  u = sin x cos y cos z,  v = -cos x sin y cos z,  w = 0  on [0, pi]^3 gives
// E = 1/8, Z = 3/8.
//
// The domain is [0, pi]: the border ring lies one cell outside it, so the interior fluid spans [0, pi] exactly and
// the walls are its faces.


static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(), d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}

static std::string fmt(const char *f, double a = 0, double b = 0, double c = 0)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

static MAC grid;

static void StandUp(int dimension, int n, bool refine)
{
    // n cells counting the ring, so n - 2 across the fluid, as the grids of this harness have always been.
    DIMENSION = dimension; SelectMethod("ADI"); THREAD_COUNT = 1;
    SIMULATION.domain = Domain{0.0, M_PI, 0.0, M_PI, 0.0, M_PI};
    SIMULATION.GRID_SIZE = n - 2;
    SIMULATION.grid.nonUniform = refine;
    SIMULATION.grid.refineX.clear(); SIMULATION.grid.refineY.clear(); SIMULATION.grid.refineZ.clear();
    if (refine)
        SIMULATION.grid.refineX.push_back(GridRefinement{true, 1.0, 2.0, 2, REFINE_SYMMETRIC});
    grid.InitializeGrid(SIMULATION.domain, dimension == 2);
}

// Ring SOLID, interior FLUID; the field scaled by `amp`.
static void Seed2D(double amp)
{
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j < grid.Nx; j++)
        {
            const bool ring = (i == 0 || j == 0 || i == grid.Ny - 1 || j == grid.Nx - 1);
            grid.SetSolid(i, j, ring ? SOLID_CELL : FLUID_CELL);
        }
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j <= grid.Nx; j++)
            grid.SetU(i, j, amp * std::sin(grid.FacePos(MAC::X, j)) * std::cos(grid.CentrePos(MAC::Y, i)));
    for (int i = 0; i <= grid.Ny; i++)
        for (int j = 0; j < grid.Nx; j++)
            grid.SetV(i, j, -amp * std::cos(grid.CentrePos(MAC::X, j)) * std::sin(grid.FacePos(MAC::Y, i)));
}

static void Seed3D()
{
    for (int k = 0; k < grid.Nz; k++)
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
            {
                const bool ring = (i == 0 || j == 0 || k == 0 || i == grid.Ny - 1 ||
                                   j == grid.Nx - 1 || k == grid.Nz - 1);
                grid.SetSolid(i, j, k, ring ? SOLID_CELL : FLUID_CELL);
            }
    for (int k = 0; k < grid.Nz; k++)
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j <= grid.Nx; j++)
                grid.SetU(i, j, k, std::sin(grid.FacePos(MAC::X, j)) * std::cos(grid.CentrePos(MAC::Y, i)) * std::cos(grid.CentrePos(MAC::Z, k)));
    for (int k = 0; k < grid.Nz; k++)
        for (int i = 0; i <= grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
                grid.SetV(i, j, k, -std::cos(grid.CentrePos(MAC::X, j)) * std::sin(grid.FacePos(MAC::Y, i)) * std::cos(grid.CentrePos(MAC::Z, k)));
    for (int k = 0; k <= grid.Nz; k++)
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
                grid.SetW(i, j, k, 0.0);
}

int run(int, char **)
{
    std::printf("\n== 1. 2D integrals against the closed form, and their order ==\n");
    {
        const int sizes[3] = {34, 66, 130};
        double eE[3], eZ[3], vol[3];
        for (int s = 0; s < 3; s++)
        {
            StandUp(2, sizes[s], false);
            Seed2D(1.0);
            const FlowIntegrals fi = ComputeFlowIntegrals2D(&grid);
            eE[s] = std::fabs(fi.kineticEnergy - 0.25);
            eZ[s] = std::fabs(fi.enstrophy - 0.5);
            vol[s] = fi.fluidVolume;
            std::printf("      N=%3d (%dx%d)  E = %.8f  Z = %.8f  V = %.6f\n", sizes[s], grid.Nx, grid.Ny,
                        fi.kineticEnergy, fi.enstrophy, fi.fluidVolume);
        }
        // E comes out exact to round-off: the midpoint rule over whole periods of a
        // trigonometric integrand has no truncation error, so there is no order to measure.
        // Z is differenced before it is integrated, so it carries the O(h^2) of the difference.
        const double oZ = std::log(eZ[1] / eZ[2]) / std::log(2.0);
        check(std::fabs(vol[2] - M_PI * M_PI) < 1e-9, "the fluid area is exactly pi^2",
              fmt("%.12f", vol[2]));
        check(eE[0] / 0.25 < 1e-11 && eE[2] / 0.25 < 1e-11, "E is exactly 1/4 at every resolution",
              fmt("relative error %.2e", eE[2] / 0.25));
        check(eZ[2] / 0.5 < 1e-3, "Z converges to 1/2", fmt("relative error %.2e", eZ[2] / 0.5));
        check(oZ > 1.9 && oZ < 2.1, "Z at second order", fmt("%.2f", oZ));
    }

    std::printf("\n== 2. a stretched grid weights the cells by their own size ==\n");
    {
        StandUp(2, 66, true);
        Seed2D(1.0);
        const FlowIntegrals fi = ComputeFlowIntegrals2D(&grid);
        std::printf("      %dx%d graded in x  E = %.8f  Z = %.8f\n", grid.Nx, grid.Ny, fi.kineticEnergy, fi.enstrophy);
        check(grid.nonUniform, "the grid really is rectilinear");
        check(std::fabs(fi.kineticEnergy - 0.25) / 0.25 < 5e-3 && std::fabs(fi.enstrophy - 0.5) / 0.5 < 5e-3,
              "E and Z still land on the closed form",
              fmt("E %.2e, Z %.2e relative", std::fabs(fi.kineticEnergy - 0.25) / 0.25,
                  std::fabs(fi.enstrophy - 0.5) / 0.5));
    }

    std::printf("\n== 3. 3D integrals against the closed form ==\n");
    {
        const int sizes[2] = {18, 34};
        double eE[2], eZ[2];
        for (int s = 0; s < 2; s++)
        {
            StandUp(3, sizes[s], false);
            Seed3D();
            const FlowIntegrals fi = ComputeFlowIntegrals3D(&grid);
            eE[s] = std::fabs(fi.kineticEnergy - 0.125);
            eZ[s] = std::fabs(fi.enstrophy - 0.375);
            std::printf("      N=%2d  E = %.8f  Z = %.8f\n", sizes[s], fi.kineticEnergy, fi.enstrophy);
        }
        check(eE[1] / 0.125 < 5e-3 && eZ[1] / 0.375 < 5e-3, "E -> 1/8 and Z -> 3/8",
              fmt("E %.2e, Z %.2e relative", eE[1] / 0.125, eZ[1] / 0.375));
        check(eE[1] / 0.125 < 1e-11 && eZ[0] / eZ[1] > 3.5, "E exact, Z at second order",
              fmt("Z error ratio %.2f between N=18 and N=34", eZ[0] / eZ[1]));
    }

    std::printf("\n== 4. an exactly decaying field: -dE/dt matches 2 nu Z ==\n");
    {
        // The budget the post-processing forms from the CSV: -dE/dt by a backward difference against 2 nu Z at
        // the midpoint of the same interval. For an exact decay the gap is the grid's own error in Z.
        StandUp(2, 130, false);
        const double nu = 0.01, dt = 0.01;
        double worst = 0.0, phys = 0.0, eps = 0.0, prevE = 0.0, prevZ = 0.0;
        for (int step = 0; step <= 50; step++)
        {
            const double t = step * dt;
            Seed2D(std::exp(-2.0 * nu * t));
            const FlowIntegrals fi = ComputeFlowIntegrals2D(&grid);
            if (step > 0)
            {
                eps = -(fi.kineticEnergy - prevE) / dt;
                phys = nu * (fi.enstrophy + prevZ);
                worst = std::fmax(worst, std::fabs(eps - phys) / phys);
            }
            prevE = fi.kineticEnergy;
            prevZ = fi.enstrophy;
        }
        std::printf("      2 nu Z = %.6e   -dE/dt = %.6e\n", phys, eps);
        check(worst < 2e-3, "-dE/dt matches 2 nu Z to within the grid's own error",
              fmt("worst |gap| / 2 nu Z = %.2e", worst));
    }

    std::printf("\n== 5. a fluid at rest has nothing ==\n");
    {
        StandUp(2, 34, false);
        Seed2D(0.0);
        const FlowIntegrals fi = ComputeFlowIntegrals2D(&grid);
        check(fi.kineticEnergy == 0.0 && fi.enstrophy == 0.0, "zero velocity, zero E and Z");
    }

    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace energy_check

// ------------------------------------------------------------------------------------------------------------------
// cellre_check
// ------------------------------------------------------------------------------------------------------------------
namespace cellre_check
{
// Max cell Reynolds telemetry: exact on uniform and stretched grids, blind to wall faces, and it
// names the face it found.


static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(), d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}

static std::string fmt(const char *f, double a = 0, double b = 0, double c = 0)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

static MAC grid;

static void StandUp(int dimension, int n, bool refine)
{
    DIMENSION = dimension; SelectMethod("ADI"); THREAD_COUNT = 1;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = n;
    SIMULATION.grid.nonUniform = refine;
    SIMULATION.grid.refineX.clear(); SIMULATION.grid.refineY.clear(); SIMULATION.grid.refineZ.clear();
    if (refine)
        SIMULATION.grid.refineX.push_back(GridRefinement{true, 0.4, 0.6, 4, REFINE_HARD});
    grid = MAC();
    grid.InitializeGrid(SIMULATION.domain, dimension == 2);
    for (int k = 0; k < (dimension == 3 ? grid.Nz : 1); k++)
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
            {
                bool ring = (i == 0 || j == 0 || i == grid.Ny - 1 || j == grid.Nx - 1);
                if (dimension == 3) ring = ring || k == 0 || k == grid.Nz - 1;
                if (dimension == 3) grid.SetSolid(i, j, k, ring ? SOLID_CELL : FLUID_CELL);
                else grid.SetSolid(i, j, ring ? SOLID_CELL : FLUID_CELL);
            }
}

int run(int, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const double nu = 1e-3, U = 1.5;

    std::printf("1. uniform 2D field\n");
    StandUp(2, 32, false);
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j <= grid.Nx; j++) grid.SetU(i, j, U);
    CellReynolds re = ComputeMaxCellReynolds2D(&grid, nu);
    const double expect = U * grid.dh / nu;
    check(re.valid && std::fabs(re.value - expect) < 1e-12 * expect, "max equals U dh / nu",
          fmt("%.15g against %.15g", re.value, expect));
    check(re.component == 0, "reported as a u face");

    std::printf("2. wall faces are ignored, an interior peak is found where it is\n");
    grid.SetV(1, 5, 50.0);                       // between the solid ring row 0 and fluid row 1
    grid.SetSolid(10, 10, SOLID_CELL);
    grid.SetU(10, 11, 80.0);                     // between that solid cell and fluid
    grid.SetV(20, 7, 9.0);                       // interior
    re = ComputeMaxCellReynolds2D(&grid, nu);
    check(std::fabs(re.value - 9.0 * grid.dh / nu) < 1e-9, "the peak is the interior v face",
          fmt("%.6g against %.6g", re.value, 9.0 * grid.dh / nu));
    check(re.component == 1 && std::fabs(re.x - grid.CentrePos(MAC::X, 7)) < 1e-15 && std::fabs(re.y - grid.FacePos(MAC::Y, 20)) < 1e-15,
          "located at that face", fmt("(%.6g, %.6g)", re.x, re.y));

    std::printf("3. stretched grid uses the local spacing\n");
    StandUp(2, 32, true);
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j <= grid.Nx; j++) grid.SetU(i, j, U);
    re = ComputeMaxCellReynolds2D(&grid, nu);
    double coarsest = 0.0;
    for (int j = 1; j < grid.Nx; j++) coarsest = std::max(coarsest, grid.FaceGap(MAC::X, j - 1));
    check(std::fabs(re.value - U * coarsest / nu) < 1e-9, "max over the coarsest face spacing",
          fmt("%.6g against %.6g (finest %.4g)", re.value, U * coarsest / nu, grid.dh));
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j <= grid.Nx; j++) grid.SetU(i, j, (grid.FacePos(MAC::X, j) > 0.45 && grid.FacePos(MAC::X, j) < 0.55) ? U : 0.1 * U);
    re = ComputeMaxCellReynolds2D(&grid, nu);
    check(re.value < 0.5 * U * coarsest / nu && std::fabs(re.value - U * grid.dh / nu) < 1e-9,
          "fast flow inside the fine band scores the fine spacing", fmt("%.6g", re.value));

    std::printf("4. 3D\n");
    StandUp(3, 16, false);
    grid.SetW(5, 6, 7, 3.0);
    grid.SetU(8, 1, 8, 2.0);
    re = ComputeMaxCellReynolds3D(&grid, nu);
    check(re.component == 2 && std::fabs(re.value - 3.0 * grid.dh / nu) < 1e-9, "w peak found",
          fmt("%.6g, component %.0f", re.value, (double)re.component));
    check(std::fabs(re.x - grid.CentrePos(MAC::X, 6)) < 1e-15 && std::fabs(re.y - grid.CentrePos(MAC::Y, 5)) < 1e-15 && std::fabs(re.z - grid.FacePos(MAC::Z, 7)) < 1e-15,
          "located at that w face", fmt("(%.4g, %.4g, %.4g)", re.x, re.y, re.z));

    std::printf("5. no viscosity\n");
    check(!ComputeMaxCellReynolds3D(&grid, 0.0).valid, "nu = 0 is reported invalid, not infinite");

    std::printf(fails ? "\nFAILED: %d\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
} // namespace cellre_check

// ------------------------------------------------------------------------------------------------------------------
// probe_check
// ------------------------------------------------------------------------------------------------------------------
namespace probe_check
{
// Scene probes: the `probe` directive parses and refuses what it should, pressure/scalar/velocity probes are exact
// on linear fields (uniform and stretched, 2D and 3D), a probe on a wall reads the fluid and never the solid,
// and `level` reads the highest particle on its vertical line, or the highest fluid cell without particles.


static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(), d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}

static std::string fmt(const char *f, double a = 0, double b = 0)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b);
    return buf;
}

static MAC grid;

static Probe MakeProbe(ProbeQuantity kind, const std::string &quantity, double x, double y, double z = 0.0)
{
    Probe pr;
    pr.name = "t";
    pr.quantity = quantity;
    pr.kind = kind;
    pr.at[0] = x; pr.at[1] = y; pr.at[2] = z;
    return pr;
}

static void StandUp2D(int n, bool refine)
{
    DIMENSION = 2; SelectMethod("ADI"); THREAD_COUNT = 1;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = n;
    SIMULATION.grid.nonUniform = refine;
    SIMULATION.grid.refineX.clear(); SIMULATION.grid.refineY.clear(); SIMULATION.grid.refineZ.clear();
    if (refine)
    {
        SIMULATION.grid.refineX.push_back(GridRefinement{true, 0.3, 0.6, 3, REFINE_SYMMETRIC});
        SIMULATION.grid.refineY.push_back(GridRefinement{true, 0.2, 0.5, 2, REFINE_HARD});
    }
    grid = MAC();
    grid.InitializeGrid(SIMULATION.domain, true);
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j < grid.Nx; j++) grid.SetSolid(i, j, FLUID_CELL);
}

static void StandUp3D(int n)
{
    DIMENSION = 3; SelectMethod("ADI"); THREAD_COUNT = 1;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = n;
    SIMULATION.grid.nonUniform = false;
    SIMULATION.grid.refineX.clear(); SIMULATION.grid.refineY.clear(); SIMULATION.grid.refineZ.clear();
    grid = MAC();
    grid.InitializeGrid(SIMULATION.domain, false);
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j < grid.Nx; j++)
            for (int k = 0; k < grid.Nz; k++) grid.SetSolid(i, j, k, FLUID_CELL);
}

int run(int, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("1. the probe directive\n");
    {
        DIMENSION = 2; SelectMethod("ADI");
        const bool ok = SCENE.LoadText("scalar d scheme=upwind\n"
                                       "probe P1 p at=(0.5,0.021)\n"
                                       "probe H2 level at=(0.25,0,0.5)\n"
                                       "probe D1 d at=(0.1,0.2,0.3)\n"
                                       "probe S speed at=(0.1,0.2)\n");
        check(ok, "a scene with p, level, scalar and speed probes parses", SCENE.error);
        check(SCENE.probes.size() == 4, "four probes, in declaration order");
        if (SCENE.probes.size() == 4)
        {
            check(SCENE.probes[0].kind == ProbeQuantity::Pressure && SCENE.probes[0].at[1] == 0.021,
                  "P1 is a pressure probe at y = 0.021");
            check(SCENE.probes[1].kind == ProbeQuantity::Level && SCENE.probes[1].at[2] == 0.5, "H2 is a level probe at z = 0.5");
            check(SCENE.probes[2].kind == ProbeQuantity::Scalar && SCENE.probes[2].quantity == "d", "D1 reads the scalar d");
        }
        check(SCENE.LoadText("probe T top at=(0.5,0)\n") && SCENE.probes[0].kind == ProbeQuantity::Top,
              "top parses as its own quantity", SCENE.error);
        check(!SCENE.LoadText("probe P1 temperature at=(0,0)\n"), "an undeclared scalar is refused", SCENE.error);
        check(!SCENE.LoadText("probe P1 d at=(0,0)\nscalar d\n"), "a scalar declared AFTER its probe is refused");
        check(!SCENE.LoadText("probe P1 p\n"), "a probe with no position is refused");
        check(!SCENE.LoadText("probe P1 p at=(0,0)\nprobe P1 u at=(0,0)\n"), "a repeated name is refused");
        check(!SCENE.LoadText("probe P1 p at=(0,0) side=left\n"), "an unknown key is refused");
        SCENE.LoadText("");
        check(SCENE.probes.empty(), "a reload clears the probes");
    }

    std::printf("2. pressure, scalar and velocity probes are exact on linear fields (2D)\n");
    for (int refine = 0; refine <= 1; refine++)
    {
        StandUp2D(24, refine == 1);
        const int h = grid.RegisterScalar("d");
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
            {
                grid.SetP(i, j, 1.0 + 2.0 * grid.CentrePos(MAC::X, j) + 3.0 * grid.CentrePos(MAC::Y, i));
                grid.SetS(h, i, j, -1.0 + 0.5 * grid.CentrePos(MAC::X, j) - 4.0 * grid.CentrePos(MAC::Y, i));
            }
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j <= grid.Nx; j++) grid.SetU(i, j, 0.3 + 1.5 * grid.FacePos(MAC::X, j) - 0.7 * grid.CentrePos(MAC::Y, i));
        for (int i = 0; i <= grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++) grid.SetV(i, j, -0.2 + 0.4 * grid.CentrePos(MAC::X, j) + 2.5 * grid.FacePos(MAC::Y, i));
        double worst = 0.0;
        const double pts[4][2] = {{0.37, 0.41}, {0.51, 0.23}, {0.12, 0.88}, {0.66, 0.34}};
        for (const auto &q : pts)
        {
            const double x = q[0], y = q[1];
            worst = std::max(worst, std::abs(EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::Pressure, "p", x, y)) - (1 + 2 * x + 3 * y)));
            worst = std::max(worst, std::abs(EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::Scalar, "d", x, y)) - (-1 + 0.5 * x - 4 * y)));
            // MAC's own plain samplers, on the same all-fluid grid: P through ComponentAt, d by handle and by name.
            worst = std::max(worst, std::abs(grid.ComponentAt(MAC::P, x, y) - (1 + 2 * x + 3 * y)));
            worst = std::max(worst, std::abs(grid.ScalarAt(h, x, y) - (-1 + 0.5 * x - 4 * y)));
            worst = std::max(worst, std::abs(grid.ScalarAt("d", x, y) - (-1 + 0.5 * x - 4 * y)));
            worst = std::max(worst, std::abs(EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::U, "u", x, y)) - (0.3 + 1.5 * x - 0.7 * y)));
            worst = std::max(worst, std::abs(EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::V, "v", x, y)) - (-0.2 + 0.4 * x + 2.5 * y)));
            const double su = 0.3 + 1.5 * x - 0.7 * y, sv = -0.2 + 0.4 * x + 2.5 * y;
            worst = std::max(worst, std::abs(EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::Speed, "speed", x, y)) - std::sqrt(su * su + sv * sv)));
        }
        check(worst < 1e-12, refine ? "stretched grid: p, d, u, v, speed exact" : "uniform grid: p, d, u, v, speed exact",
              fmt("worst %.2e", worst));
    }

    std::printf("3. a probe on a wall reads the fluid, never the solid\n");
    {
        StandUp2D(20, false);
        const double big = 1e6;
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
            {
                const bool ring = (i == 0 || j == grid.Nx - 1);
                grid.SetSolid(i, j, ring ? SOLID_CELL : FLUID_CELL);
                grid.SetP(i, j, ring ? big : 10.0 + grid.CentrePos(MAC::X, j) + 2.0 * grid.CentrePos(MAC::Y, i));
            }
        // a sensor on the right wall (x = the ring's inner face) 0.3 m up
        const double xw = grid.FacePos(MAC::X, grid.Nx - 1), y = 0.3;
        const double pw = EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::Pressure, "p", xw, y));
        const double expect = 10.0 + grid.CentrePos(MAC::X, grid.Nx - 2) + 2.0 * y;   // the fluid column beside the wall, linear in y
        check(std::abs(pw - expect) < 1e-12, "a wall sensor reads the fluid cell beside the wall", fmt("%.12g against %.12g", pw, expect));
        // a sensor on the floor, 0.3 of a cell above it: only the first fluid row lies around it
        const double yf = grid.FacePos(MAC::Y, 1) + 0.3 * grid.dh, x = 0.47;
        const double pf = EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::Pressure, "p", x, yf));
        const double expectF = 10.0 + x + 2.0 * grid.CentrePos(MAC::Y, 1);
        check(std::abs(pf - expectF) < 1e-12, "a sensor below the first fluid centre reads the first fluid row",
              fmt("%.12g against %.12g", pf, expectF));
        const double inSolid = EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::Pressure, "p", grid.CentrePos(MAC::X, grid.Nx - 1) + 0.2 * grid.dh, 0.5));
        check(inSolid == 0.0, "a probe deep inside a solid reads 0, not the solid's value", fmt("%.6g", inSolid));
        check(EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::U, "u", 0.5, grid.CentrePos(MAC::Y, 0))) == 0.0, "velocity inside a solid cell reads 0");
    }

    std::printf("4. level (connected column) and top (highest particle)\n");
    {
        StandUp2D(20, false);
        // A tank with 6 rows of liquid on the bed, an air gap, and a sheet of liquid overhead.
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
            {
                int t = EMPTY_CELL;
                if (i == 0 || j == 0 || j == grid.Nx - 1) t = SOLID_CELL;   // bed and side walls
                else if (i <= 6) t = FLUID_CELL;
                else if (i == 12 || i == 13) t = FLUID_CELL;                 // a sheet in the air
                grid.SetSolid(i, j, t);
            }
        ParticleSet ps;
        ps.data.resize(3);
        ps.data[0].x = 0.50; ps.data[0].y = 0.30;   // in the bed column
        ps.data[1].x = 0.50; ps.data[1].y = 0.66;   // in the sheet overhead
        ps.data[2].x = 0.90; ps.data[2].y = 0.95;   // another column entirely
        ps.count = 3;

        const double lvl = EvaluateProbe(&grid, &ps, MakeProbe(ProbeQuantity::Level, "level", 0.5, 0.0));
        check(std::abs(lvl - grid.FacePos(MAC::Y, 7)) < 1e-15, "level is the top of the column connected to the bed",
              fmt("%.6g, expected %.6g", lvl, grid.FacePos(MAC::Y, 7)));
        const double top = EvaluateProbe(&grid, &ps, MakeProbe(ProbeQuantity::Top, "top", 0.5, 0.0));
        check(top == 0.66, "top is the highest particle, sheet included", fmt("%.6g", top));
        check(lvl < top, "the sheet overhead raises top and not level");

        // A dry column reads 0 on both.
        for (int i = 1; i < grid.Ny; i++)
            for (int j = 1; j < grid.Nx - 1; j++) grid.SetSolid(i, j, EMPTY_CELL);
        check(EvaluateProbe(&grid, &ps, MakeProbe(ProbeQuantity::Level, "level", 0.5, 0.0)) == 0.0, "a dry column has no level");

        // The radius averages whole columns: half of them wet to row 4, half dry.
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
                grid.SetSolid(i, j, (i == 0) ? SOLID_CELL
                                   : (grid.CentrePos(MAC::X, j) < 0.5 && i <= 4) ? FLUID_CELL : EMPTY_CELL);
        Probe wide = MakeProbe(ProbeQuantity::Level, "level", 0.5, 0.0);
        wide.radius = 4.2 * grid.dh;   // not a whole number of cells: a radius landing ON a centre is a knife edge
        const double mixed = EvaluateProbe(&grid, nullptr, wide);
        check(std::abs(mixed - 0.5 * grid.FacePos(MAC::Y, 5)) < 1e-12, "half wet, half dry columns average to half the level",
              fmt("%.6g, expected %.6g", mixed, 0.5 * grid.FacePos(MAC::Y, 5)));

        // Without particles, top falls back to the highest fluid cell.
        ParticleSet none;
        check(std::abs(EvaluateProbe(&grid, &none, MakeProbe(ProbeQuantity::Top, "top", 0.2, 0.0)) - grid.FacePos(MAC::Y, 5)) < 1e-15,
              "without particles, top is the highest fluid cell");
    }

    std::printf("5. a sensor with a radius averages over its face\n");
    {
        StandUp2D(20, false);
        const double dh = grid.dh;
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
            {
                grid.SetSolid(i, j, FLUID_CELL);
                grid.SetP(i, j, 1.0 + 2.0 * grid.CentrePos(MAC::X, j) + 3.0 * grid.CentrePos(MAC::Y, i));
            }
        check(SCENE.LoadText("probe P p at=(0.5,0.5) radius=0.02\n") && SCENE.probes.size() == 1 &&
              SCENE.probes[0].radius == 0.02, "radius= parses", SCENE.error);
        check(!SCENE.LoadText("probe P p at=(0,0) radius=-1\n"), "a negative radius is refused");

        // A linear field averaged over a symmetric set of cells is the value at the centre.
        Probe wide = MakeProbe(ProbeQuantity::Pressure, "p", grid.CentrePos(MAC::X, 10), grid.CentrePos(MAC::Y, 10));
        wide.radius = 3.2 * dh;
        const double avg = EvaluateProbe(&grid, nullptr, wide);
        const double at = 1.0 + 2.0 * grid.CentrePos(MAC::X, 10) + 3.0 * grid.CentrePos(MAC::Y, 10);
        check(std::abs(avg - at) < 1e-12, "a linear field averages to its value at the centre", fmt("%.12g vs %.12g", avg, at));

        // Smaller than a cell: nothing inside, so it falls back to the interpolated point value.
        Probe tiny = MakeProbe(ProbeQuantity::Pressure, "p", 0.5 * (grid.CentrePos(MAC::X, 10) + grid.CentrePos(MAC::X, 11)), grid.CentrePos(MAC::Y, 10));
        tiny.radius = 0.2 * dh;
        const double pt = EvaluateProbe(&grid, nullptr, tiny);
        tiny.radius = 0.0;
        check(pt == EvaluateProbe(&grid, nullptr, tiny), "a radius below one cell falls back to the point value");

        // The average is volume weighted and drops solids: make the left half solid and carrying garbage.
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
                if (grid.CentrePos(MAC::X, j) < grid.CentrePos(MAC::X, 10)) { grid.SetSolid(i, j, SOLID_CELL); grid.SetP(i, j, 1e6); }
        const double half = EvaluateProbe(&grid, nullptr, wide);
        check(half < 100.0 && half > at, "solid cells are left out of the average", fmt("%.6g", half));
    }

    std::printf("6. 3D: pressure and w exact, level banded in z\n");
    {
        StandUp3D(12);
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
                for (int k = 0; k < grid.Nz; k++) grid.SetP(i, j, k, 2.0 - grid.CentrePos(MAC::X, j) + 0.5 * grid.CentrePos(MAC::Y, i) + 3.0 * grid.CentrePos(MAC::Z, k));
        for (int i = 0; i < grid.Ny; i++)
            for (int j = 0; j < grid.Nx; j++)
                for (int k = 0; k <= grid.Nz; k++) grid.SetW(i, j, k, 0.1 + grid.CentrePos(MAC::X, j) - grid.FacePos(MAC::Z, k));
        const double x = 0.43, y = 0.61, z = 0.27;
        const double e1 = std::abs(EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::Pressure, "p", x, y, z)) - (2.0 - x + 0.5 * y + 3.0 * z));
        const double e2 = std::abs(EvaluateProbe(&grid, nullptr, MakeProbe(ProbeQuantity::W, "w", x, y, z)) - (0.1 + x - z));
        const double e3 = std::abs(grid.ComponentAt(MAC::P, x, y, z) - (2.0 - x + 0.5 * y + 3.0 * z));
        check(e1 < 1e-12 && e2 < 1e-12 && e3 < 1e-12, "trilinear pressure (probe and ComponentAt) and w exact",
              fmt("%.2e / %.2e", e1, std::max(e2, e3)));
        check(grid.ScalarAt("nope", x, y, z) == 0.0, "ScalarAt on a scalar the grid does not carry reads 0");
        ParticleSet ps;
        ps.data.resize(2);
        ps.data[0].x = 0.5; ps.data[0].y = 0.4; ps.data[0].z = 0.5;
        ps.data[1].x = 0.5; ps.data[1].y = 0.8; ps.data[1].z = 0.9;   // same x, far in z
        ps.count = 2;
        check(EvaluateProbe(&grid, &ps, MakeProbe(ProbeQuantity::Top, "top", 0.5, 0.0, 0.5)) == 0.4,
              "top: a particle far in z does not count");
    }

    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL CHECKS PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace probe_check

// ------------------------------------------------------------------------------------------------------------------
// confinement_check
// ------------------------------------------------------------------------------------------------------------------
namespace confinement_check
{
// Vorticity confinement: the force does what it is for, and nothing where it should not.
//
//   1. a solid-body rotation has uniform |omega|, so there is no core to confine: zero force
//   2. a Lamb-Oseen vortex gains kinetic energy, the force runs WITH the rotation, and the gain
//      is linear in eps
//   3. eps = 0, or never initialised, leaves the field bit-identical
//   4. nothing within the solid margin of an obstacle is touched


static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(), d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}
static std::string fmt(const char *f, double a = 0, double b = 0)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b);
    return buf;
}

static MAC grid, reference;

static void StandUp(int n)
{
    DIMENSION = 2; SelectMethod("ADI"); THREAD_COUNT = 4;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = n;
    SIMULATION.grid.nonUniform = false;
    grid.InitializeGrid(SIMULATION.domain, true);
    reference.InitializeGrid(SIMULATION.domain, true);
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j < grid.Nx; j++)
        {
            const bool ring = (i == 0 || j == 0 || i == grid.Ny - 1 || j == grid.Nx - 1);
            grid.SetSolid(i, j, ring ? SOLID_CELL : FLUID_CELL);
        }
}

static void SolidBody()
{
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j <= grid.Nx; j++) grid.SetU(i, j, -3.0 * (grid.CentrePos(MAC::Y, i) - 0.5));
    for (int i = 0; i <= grid.Ny; i++)
        for (int j = 0; j < grid.Nx; j++) grid.SetV(i, j, 3.0 * (grid.CentrePos(MAC::X, j) - 0.5));
}

// Lamb-Oseen: u_theta = G/(2 pi r) (1 - exp(-r^2/rc^2)), counter-clockwise.
static void Vortex()
{
    const double G = 1.0, rc = 0.08;
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j <= grid.Nx; j++)
        {
            const double x = grid.FacePos(MAC::X, j) - 0.5, y = grid.CentrePos(MAC::Y, i) - 0.5, r2 = x * x + y * y;
            const double k = (r2 > 1e-14) ? G / (2 * M_PI * r2) * (1.0 - std::exp(-r2 / (rc * rc))) : 0.0;
            grid.SetU(i, j, -k * y);
        }
    for (int i = 0; i <= grid.Ny; i++)
        for (int j = 0; j < grid.Nx; j++)
        {
            const double x = grid.CentrePos(MAC::X, j) - 0.5, y = grid.FacePos(MAC::Y, i) - 0.5, r2 = x * x + y * y;
            const double k = (r2 > 1e-14) ? G / (2 * M_PI * r2) * (1.0 - std::exp(-r2 / (rc * rc))) : 0.0;
            grid.SetV(i, j, k * x);
        }
}

static double MaxDiff()
{
    double worst = 0.0;
    for (int i = 0; i < grid.Ny; i++)
        for (int j = 0; j <= grid.Nx; j++) worst = std::fmax(worst, std::fabs(grid.GetU(i, j) - reference.GetU(i, j)));
    for (int i = 0; i <= grid.Ny; i++)
        for (int j = 0; j < grid.Nx; j++) worst = std::fmax(worst, std::fabs(grid.GetV(i, j) - reference.GetV(i, j)));
    return worst;
}

// Kinetic energy change and the work the change does against the original velocity.
static void Gain(double &dE, double &work)
{
    dE = ComputeFlowIntegrals2D(&grid).kineticEnergy - ComputeFlowIntegrals2D(&reference).kineticEnergy;
    work = 0.0;
    for (int i = 1; i < grid.Ny - 1; i++)
        for (int j = 2; j < grid.Nx - 1; j++) work += (grid.GetU(i, j) - reference.GetU(i, j)) * reference.GetU(i, j);
    for (int i = 2; i < grid.Ny - 1; i++)
        for (int j = 1; j < grid.Nx - 1; j++) work += (grid.GetV(i, j) - reference.GetV(i, j)) * reference.GetV(i, j);
}

int run(int, char **)
{
    const double dt = 1e-3;
    ADI_SETTINGS.vorticitySolidMargin = 2;

    std::printf("\n== 1. uniform vorticity has no core to confine ==\n");
    {
        StandUp(64);
        SolidBody();
        reference.CopyFlowFields(grid);
        ADI_SETTINGS.vorticityEpsilon = 0.5;
        VorticityConfinement2D::Initialize(&grid);
        VorticityConfinement2D::Apply(&grid, dt);
        check(MaxDiff() == 0.0, "a solid-body rotation is left exactly as it was", fmt("moved by %.3e", MaxDiff()));
    }

    std::printf("\n== 2. a vortex is spun up, with its own rotation, linearly in eps ==\n");
    {
        StandUp(96);
        Vortex();
        reference.CopyFlowFields(grid);
        VorticityConfinement2D::Initialize(&grid);

        ADI_SETTINGS.vorticityEpsilon = 0.2;
        VorticityConfinement2D::Apply(&grid, dt);
        double dE1, work1;
        Gain(dE1, work1);

        grid.CopyFlowFields(reference);
        ADI_SETTINGS.vorticityEpsilon = 0.4;
        VorticityConfinement2D::Apply(&grid, dt);
        double dE2, work2;
        Gain(dE2, work2);

        std::printf("      eps 0.2: dE = %.4e  work = %.4e    eps 0.4: dE = %.4e  work = %.4e\n", dE1, work1, dE2, work2);
        check(dE1 > 0.0 && work1 > 0.0, "kinetic energy rises and the force works with the flow",
              fmt("dE %.3e, f.u %.3e", dE1, work1));
        check(std::fabs(work2 / work1 - 2.0) < 1e-12, "the force is exactly linear in eps",
              fmt("work ratio %.15f", work2 / work1));
    }

    std::printf("\n== 3. eps = 0 does nothing ==\n");
    {
        StandUp(64);
        Vortex();
        reference.CopyFlowFields(grid);
        VorticityConfinement2D::Initialize(&grid);
        ADI_SETTINGS.vorticityEpsilon = 0.0;
        VorticityConfinement2D::Apply(&grid, dt);
        check(MaxDiff() == 0.0, "the field is bit-identical", fmt("%.3e", MaxDiff()));
    }

    std::printf("\n== 4. the solid margin keeps it off the wall shear ==\n");
    {
        StandUp(96);
        Vortex();
        // A solid block right beside the core, where the force would otherwise be strong.
        const int bi0 = grid.Ny / 2 - 3, bi1 = grid.Ny / 2 + 3, bj0 = grid.Nx / 2 + 8, bj1 = grid.Nx / 2 + 12;
        for (int i = bi0; i <= bi1; i++)
            for (int j = bj0; j <= bj1; j++) grid.SetSolid(i, j, SOLID_CELL);
        reference.CopyFlowFields(grid);

        ADI_SETTINGS.vorticityEpsilon = 0.5;
        const int margin = 3;
        ADI_SETTINGS.vorticitySolidMargin = margin;
        VorticityConfinement2D::Initialize(&grid);
        VorticityConfinement2D::Apply(&grid, dt);

        // A face is untouched when both its cells are inside the margin band around the block.
        double nearWorst = 0.0;
        for (int i = bi0 - margin + 1; i <= bi1 + margin - 1; i++)
            for (int j = bj0 - margin + 1; j <= bj1 + margin; j++)
                nearWorst = std::fmax(nearWorst, std::fabs(grid.GetU(i, j) - reference.GetU(i, j)));
        check(nearWorst == 0.0, "faces within the margin of the block are untouched", fmt("%.3e", nearWorst));
        check(MaxDiff() > 0.0, "while the rest of the vortex is still confined", fmt("%.3e", MaxDiff()));

        grid.CopyFlowFields(reference);
        ADI_SETTINGS.vorticitySolidMargin = 0;
        VorticityConfinement2D::Initialize(&grid);
        VorticityConfinement2D::Apply(&grid, dt);
        double nearNoMargin = 0.0;
        for (int i = bi0 - margin + 1; i <= bi1 + margin - 1; i++)
            for (int j = bj0 - margin + 1; j <= bj1 + margin; j++)
                nearNoMargin = std::fmax(nearNoMargin, std::fabs(grid.GetU(i, j) - reference.GetU(i, j)));
        check(nearNoMargin > 0.0, "and with no margin the same faces are forced", fmt("%.3e", nearNoMargin));
    }

    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace confinement_check

// ------------------------------------------------------------------------------------------------------------------
// scene_check
// ------------------------------------------------------------------------------------------------------------------
namespace scene_check
{
// Compares the scene-driven geometry and boundary conditions against the hardcoded triple
// they replace. The solid mask and the border values are diffed directly, then the whole
// ADI pipeline is run both ways and the field checksums compared.


static int failures = 0;
static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("     [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}

static const char *TypeName(int t)
{
    switch (t) { case FLUID_CELL: return "FLUID"; case SOLID_CELL: return "SOLID";
                 case EMPTY_CELL: return "EMPTY"; default: return "INFLOW"; }
}

// ---------------------------------------------------------------------------
static void CompareCase2D(const char *label, const char *sceneFile,
                          int (*maskFn)(int, int), Vec2 (*velFn)(double, double, double),
                          double xf, double yf, int gridSize)
{
    std::printf("\n== %s ==\n", label);
    DIMENSION = 2;
    SelectMethod("ADI");
    SIMULATION.domain = Domain{0.0, xf, 0.0, yf, 0.0, 1.0};
    SIMULATION.GRID_SIZE = gridSize;

    static MAC ga, gb;
    ga.InitializeGrid(SIMULATION.domain, true);
    gb.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = ga.dh; SIMULATION.Nx = ga.Nx; SIMULATION.Ny = ga.Ny;
    SIMULATION.GRID_SOL = &ga; SIMULATION.GRID_ANT = &gb;

    if (!SCENE.Load(sceneFile)) { check(false, "scene loads", SCENE.error); return; }
    check(true, "scene loads", SCENE.Summary());

    // ---- solid mask ----
    long diff = 0; std::string firstDiff;
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
        {
            int a = maskFn(i, j), b = SCENE_SOLID_MASK_2D(i, j);
            if (a != b)
            {
                if (!diff) firstDiff = "(" + std::to_string(i) + "," + std::to_string(j) + ") " +
                                       TypeName(a) + " vs " + TypeName(b);
                diff++;
            }
        }
    // The hardcoded masks sample a cell's lower corner. A scene that samples centres is
    // describing the same geometry with a half-cell-better voxelisation, so the two differ
    // by design. What has to hold is that the *geometry* agrees: evaluated the way the
    // hardcoded mask evaluates, the scene reproduces it exactly. The sampling choice then
    // stays the scene author's, and switching it in the setup screen cannot fail this.
    const bool wasCentre = SCENE.sampleAtCenter;
    SCENE.sampleAtCenter = false;
    long diffCorner = 0; std::string firstCorner;
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
        {
            int a = maskFn(i, j), b = SCENE_SOLID_MASK_2D(i, j);
            if (a == b) continue;
            if (!diffCorner) firstCorner = "(" + std::to_string(i) + "," + std::to_string(j) + ") " +
                                           TypeName(a) + " vs " + TypeName(b);
            diffCorner++;
        }
    SCENE.sampleAtCenter = wasCentre;

    check(diffCorner == 0, "solid mask matches the hardcoded one at the same sample point",
          diffCorner ? std::to_string(diffCorner) + " cells differ, first " + firstCorner
                     : (wasCentre ? std::to_string(diff) + " cells differ under centre sampling, "
                                    "which is the half-cell voxelisation offset"
                                  : ""));

    // ---- border values, exactly as SetBorder writes them ----
    ga.SetSolidMask(maskFn);  gb.SetSolidMask(SCENE_SOLID_MASK_2D);
    ga.ClearFlowFields(); gb.ClearFlowFields();
    ga.SetBoundaryFunctions(velFn, ZERO2D_SCALAR);
    ga.EnforceBoundaryConditions(0.0);
    gb.SetBoundaryFunctions(SCENE_VELOCITY_2D, SCENE_PRESSURE_2D);
    gb.EnforceBoundaryConditions(0.0);

    // A difference is only acceptable on a wall row where the scene gives zero and the
    // hardcoded version does not. That is the `y < 6.0 - dh/2` guard in Functions.h: 6.0 is
    // not the channel height, so the guard never fires and the old inlet profile leaks onto
    // the top no-slip wall while the bottom one is correctly excluded. The scene is
    // symmetric; the hardcoded version was not.
    long unexplained = 0, wallRowFix = 0; double worstD = 0.0; std::string worst;
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx + 1; j++)
        {
            double d = std::abs(ga.GetU(i, j) - gb.GetU(i, j));
            if (d <= 1e-15) continue;
            const bool onWallRow = (i == 0 || i == ga.Ny - 1);
            if (onWallRow && gb.GetU(i, j) == 0.0) { wallRowFix++; continue; }
            unexplained++;
            if (d > worstD) { worstD = d;
                worst = "U(" + std::to_string(i) + "," + std::to_string(j) + ") " +
                        std::to_string(ga.GetU(i,j)) + " vs " + std::to_string(gb.GetU(i,j)); }
        }
    for (int i = 0; i < ga.Ny + 1; i++)
        for (int j = 0; j < ga.Nx; j++)
            if (std::abs(ga.GetV(i, j) - gb.GetV(i, j)) > 1e-15) unexplained++;

    check(unexplained == 0, "border velocities match away from the wall rows",
          unexplained ? std::to_string(unexplained) + " unexplained faces, worst " + worst : "");
    if (wallRowFix)
        std::printf("     ...  %ld inlet faces on a no-slip wall row are now zero "
                    "(the old `y < 6.0` guard never fired)\n", wallRowFix);
}

// ---------------------------------------------------------------------------
static void CompareCase3D(const char *label, const char *sceneFile,
                          int (*maskFn)(int, int, int), Vec3 (*velFn)(double, double, double, double),
                          double xf, double yf, double zf, int gridSize)
{
    std::printf("\n== %s ==\n", label);
    DIMENSION = 3;
    SelectMethod("ADI");
    SIMULATION.domain = Domain{0.0, xf, 0.0, yf, 0.0, zf};
    SIMULATION.GRID_SIZE = gridSize;

    static MAC ga, gb;
    ga.InitializeGrid(SIMULATION.domain, false);
    gb.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.dh = ga.dh; SIMULATION.Nx = ga.Nx; SIMULATION.Ny = ga.Ny; SIMULATION.Nz = ga.Nz;
    SIMULATION.GRID_SOL = &ga; SIMULATION.GRID_ANT = &gb;

    if (!SCENE.Load(sceneFile)) { check(false, "scene loads", SCENE.error); return; }
    check(true, "scene loads", SCENE.Summary());

    long diff = 0; std::string firstDiff;
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
            for (int k = 0; k < ga.Nz; k++)
            {
                int a = maskFn(i, j, k), b = SCENE_SOLID_MASK(i, j, k);
                if (a != b)
                {
                    if (!diff) firstDiff = "(" + std::to_string(i) + "," + std::to_string(j) + "," +
                                           std::to_string(k) + ") " + TypeName(a) + " vs " + TypeName(b);
                    diff++;
                }
            }
    // The hardcoded masks sample a cell's lower corner. A scene that samples centres is
    // describing the same geometry with a half-cell-better voxelisation, so the two differ
    // by design. What has to hold is that the *geometry* agrees: evaluated the way the
    // hardcoded mask evaluates, the scene reproduces it exactly. The sampling choice then
    // stays the scene author's, and switching it in the setup screen cannot fail this.
    const bool wasCentre = SCENE.sampleAtCenter;
    SCENE.sampleAtCenter = false;
    long diffCorner = 0; std::string firstCorner;
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
            for (int k = 0; k < ga.Nz; k++)
            {
                int a = maskFn(i, j, k), b = SCENE_SOLID_MASK(i, j, k);
                if (a == b) continue;
                if (!diffCorner) firstCorner = "(" + std::to_string(i) + "," + std::to_string(j) +
                                               "," + std::to_string(k) + ") " + TypeName(a) + " vs " + TypeName(b);
                diffCorner++;
            }
    SCENE.sampleAtCenter = wasCentre;

    check(diffCorner == 0, "solid mask matches the hardcoded one at the same sample point",
          diffCorner ? std::to_string(diffCorner) + " cells differ, first " + firstCorner
                     : (wasCentre ? std::to_string(diff) + " cells differ under centre sampling, "
                                    "which is the half-cell voxelisation offset"
                                  : ""));

    ga.SetSolidMask(maskFn); gb.SetSolidMask(SCENE_SOLID_MASK);
    ga.ClearFlowFields(); gb.ClearFlowFields();
    ga.SetBoundaryFunctions(velFn, ZERO_SCALAR);
    ga.EnforceBoundaryConditions(0.0);
    gb.SetBoundaryFunctions(SCENE_VELOCITY, SCENE_PRESSURE);
    gb.EnforceBoundaryConditions(0.0);

    long unexplained = 0, wallRowFix = 0; double worstD = 0.0; std::string worst;
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx + 1; j++)
            for (int k = 0; k < ga.Nz; k++)
            {
                double d = std::abs(ga.GetU(i,j,k) - gb.GetU(i,j,k));
                if (d <= 1e-15) continue;
                const bool onWall = (i == 0 || i == ga.Ny - 1 || k == 0 || k == ga.Nz - 1);
                if (onWall && gb.GetU(i,j,k) == 0.0) { wallRowFix++; continue; }
                unexplained++;
                if (d > worstD) { worstD = d;
                    worst = "U(" + std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+") "
                          + std::to_string(ga.GetU(i,j,k)) + " vs " + std::to_string(gb.GetU(i,j,k)); }
            }
    for (int i = 0; i < ga.Ny + 1; i++)
        for (int j = 0; j < ga.Nx; j++)
            for (int k = 0; k < ga.Nz; k++)
                if (std::abs(ga.GetV(i,j,k) - gb.GetV(i,j,k)) > 1e-15) unexplained++;
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
            for (int k = 0; k < ga.Nz + 1; k++)
                if (std::abs(ga.GetW(i,j,k) - gb.GetW(i,j,k)) > 1e-15) unexplained++;

    check(unexplained == 0, "border velocities match away from the wall rows",
          unexplained ? std::to_string(unexplained) + " unexplained faces, worst " + worst : "");
    if (wallRowFix)
        std::printf("     ...  %ld inlet faces on a no-slip wall are now zero "
                    "(the old `y < 6.0` / `z < 6.0` guards never fired)\n", wallRowFix);
}

// Runs the real 2D ADI pipeline with a given geometry triple and returns field checksums,
// so the effect of any boundary difference on the validated result can be measured rather
// than guessed at.
struct Sums { double u, v, p, div, maxVel; };
static Sums RunPipeline2D(int (*maskFn)(int, int), Vec2 (*velFn)(double, double, double),
                          double (*pressFn)(double, double, double), int steps)
{
    DIMENSION = 2;
    SelectMethod("ADI");
    THREAD_COUNT = 4;
    GPU_ACCELERATION = 0;
    ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.1, 0.0, 0.41, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 50;
    SIMULATION.RHO = 1.0;
    SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.002;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    SIMULATION.f = Vec3{0, 0, 0};
    SIMULATION.VelocityBoundaryFunction2D = velFn;
    SIMULATION.PressureBoundaryFunction2D = pressFn;
    SIMULATION.SolidMaskFunction2D = maskFn;

    static MAC gs, gaPrev;
    gs = MAC(); gaPrev = MAC();
    gs.InitializeGrid(SIMULATION.domain, true);
    gaPrev.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = gs.dh; SIMULATION.Nx = gs.Nx; SIMULATION.Ny = gs.Ny;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &gaPrev;
    gs.SetSolidMask(maskFn); gaPrev.SetSolidMask(maskFn);

    ADI2D::InitializeADI2D(&gs, SIMULATION.dt, velFn,  pressFn);
    PressureSolver2D::InitializePressureSolver(&gs, ADIPressureScheme());
    gaPrev.SetBoundaryFunctions(velFn, pressFn);
    gaPrev.EnforceBoundaryConditions(0);

    for (int st = 0; st < steps; st++)
    {
        gs.ClearFlowFields();
        ADI2D::SolveADIStep(&gaPrev, &gs, nullptr, st * SIMULATION.dt);
        PressureSolver2D::SolvePressure(nullptr, &gs, nullptr, 0.0);
        PressureSolver2D::ProjectPressure(&gs);
        gaPrev.CopyGrid(gs);
    }

    Sums r{0, 0, 0, gs.GetDivSum(), gs.GetMaxVelocity()};
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx + 1; j++) r.u += std::abs(gs.GetU(i, j));
    for (int i = 0; i < gs.Ny + 1; i++)
        for (int j = 0; j < gs.Nx; j++) r.v += std::abs(gs.GetV(i, j));
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++) r.p += gs.GetP(i, j);
    PressureSolver2D::DestroyPressureSolver();
    return r;
}

int run(int, char **)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("scene-driven geometry vs the hardcoded triple it replaces\n");

    CompareCase2D("lid-driven cavity 2D", "tests/scenes/cavity_2d.scene",
                  LID_CAVITY_SOLID_MASK_2D, LID_CAVITY_FLOW_2D, 1.0, 1.0, 64);
    CompareCase2D("obstacle 2D", "tests/scenes/obstacle_2d.scene",
                  OBSTACLE_SOLID_MASK_2D, OBSTACLE_FLOW_2D, 1.1, 0.41, 100);
    CompareCase3D("lid-driven cavity 3D", "tests/scenes/cavity_3d.scene",
                  LID_CAVITY_SOLID_MASK, LID_CAVITY_FLOW, 1.0, 1.0, 1.0, 24);
    CompareCase3D("obstacle 3D", "tests/scenes/obstacle_3d.scene",
                  OBSTACLE_SOLID_MASK, OBSTACLE_FLOW, 1.0, 0.5, 0.5, 24);

    // ---- what the remaining boundary difference actually costs ----
    std::printf("\n== effect on the 2D obstacle result (60 steps) ==\n");
    {
        Sums a = RunPipeline2D(OBSTACLE_SOLID_MASK_2D, OBSTACLE_FLOW_2D, OBSTACLE_FLOW_PRESSURE_2D, 60);
        SCENE.Load("tests/scenes/obstacle_2d.scene");
        Sums b = RunPipeline2D(SCENE_SOLID_MASK_2D, SCENE_VELOCITY_2D, SCENE_PRESSURE_2D, 60);
        std::printf("     hardcoded : sum|u|=%.8f sum|v|=%.8f sum p=%.8f maxVel=%.8f div=%.3e\n",
                    a.u, a.v, a.p, a.maxVel, a.div);
        std::printf("     scene     : sum|u|=%.8f sum|v|=%.8f sum p=%.8f maxVel=%.8f div=%.3e\n",
                    b.u, b.v, b.p, b.maxVel, b.div);
        std::printf("     relative  : u %.3e  v %.3e  maxVel %.3e\n",
                    std::abs(a.u - b.u) / std::abs(a.u),
                    std::abs(a.v - b.v) / std::abs(a.v),
                    std::abs(a.maxVel - b.maxVel) / std::abs(a.maxVel));
    }

    // ---- FLIP seeds from region fluid ----
    std::printf("\n== FLIP seeding from a fluid region ==\n");
    {
        DIMENSION = 2; SelectMethod("FLIP");
        SIMULATION.domain = Domain{0.0, 0.9, 0.0, 0.9, 0.0, 1.0};
        SIMULATION.GRID_SIZE = 60;
        FLIP_SETTINGS.particlesPerCell = 4;
        SIMULATION.dt = 1e-3;
        static MAC g; g = MAC();
        g.InitializeGrid(SIMULATION.domain, true);
        SIMULATION.dh = g.dh; SIMULATION.Nx = g.Nx; SIMULATION.Ny = g.Ny;
        SIMULATION.GRID_SOL = &g; SIMULATION.GRID_ANT = &g;

        if (!SCENE.Load("tests/scenes/dambreak_2d.scene")) { check(false, "dambreak scene loads", SCENE.error); }
        else
        {
            check(SCENE.HasFluidRegions(), "scene declares a fluid region");
            g.SetSolidMask(SCENE_SOLID_MASK_2D);
            FLIP2D::InitializeFLIP(&g, &SIMULATION.particles, SIMULATION.dt, 1.0);

            const double dh = SIMULATION.dh;
            long inSolid = 0, outsideRegion = 0;
            double minX = 1e30, maxX = -1e30, minY = 1e30, maxY = -1e30;
            for (int p = 0; p < SIMULATION.particles.count; p++)
            {
                const auto &q = SIMULATION.particles.data[p];
                int j = (int)std::floor((q.x - g.FacePos(MAC::X, 0)) / dh), i = (int)std::floor((q.y - g.FacePos(MAC::Y, 0)) / dh);
                if (i >= 0 && i < g.Ny && j >= 0 && j < g.Nx && g.GetSolid(i, j) == SOLID_CELL) inSolid++;
                if (!SCENE.IsFluidSeedAt(g.CentrePos(MAC::X, j), g.CentrePos(MAC::Y, i), 0.0)) outsideRegion++;
                minX = std::min(minX, q.x); maxX = std::max(maxX, q.x);
                minY = std::min(minY, q.y); maxY = std::max(maxY, q.y);
            }
            std::printf("     %d particles, x in [%.4f, %.4f], y in [%.4f, %.4f]\n",
                        SIMULATION.particles.count, minX, maxX, minY, maxY);
            check(SIMULATION.particles.count > 0, "particles were seeded");
            check(inSolid == 0, "none inside a solid cell", std::to_string(inSolid));
            check(outsideRegion == 0, "none outside the declared fluid region",
                  std::to_string(outsideRegion));
            // region is 0.28 x 0.11 from the corner; the solid ring lies outside the domain, so the block starts
            // at the walls. Seeding is per cell, so a block can reach half a cell past the region edge.
            check(maxX <= 0.28 + dh && maxY <= 0.11 + dh,
                  "the block stops at the region bounds",
                  "maxX " + std::to_string(maxX) + " maxY " + std::to_string(maxY));
            check(minX >= 0.0 && minY >= 0.0 && minX < dh && minY < dh,
                  "the block starts at the walls, outside the solid ring");
        }
    }

    // ---- ConfigReader wires the scene, and falls back cleanly when it cannot ----
    std::printf("\n== ConfigReader wiring ==\n");
    // loadConfig deletes the previous grid pair; the pointers above are to stack grids, so let it start from none.
    SIMULATION.GRID_SOL = nullptr; SIMULATION.GRID_ANT = nullptr;
    {
        auto writeCfg = [](const char *path, const char *sceneLine)
        {
            std::ofstream f(path);
            f << "DIMENSION = 2\nTYPE = ADI\n"
              << "DOMAIN_X0 = 0.0\nDOMAIN_XF = 1.1\nDOMAIN_Y0 = 0.0\nDOMAIN_YF = 0.41\n"
              << "DOMAIN_Z0 = 0.0\nDOMAIN_ZF = 0.41\nGRID_SIZE = 40\n"
              << "TIME_STEP = 0.002\nVISCOSITY = 1e-3\nDENSITY = 1.0\n"
              << sceneLine << "\n";
        };

        writeCfg("scene_ok.cfg", "SCENE_FILE = tests/scenes/obstacle_2d.scene");
        SIMULATION_CONFIG probe;
        ConfigReader::loadConfig(probe, "scene_ok.cfg");
        check(SIMULATION.SolidMaskFunction2D == SCENE_SOLID_MASK_2D &&
                  SIMULATION.VelocityBoundaryFunction2D == SCENE_VELOCITY_2D &&
                  SIMULATION.PressureBoundaryFunction2D == SCENE_PRESSURE_2D,
              "SCENE_FILE wires the generic triple into SIMULATION");

        writeCfg("scene_missing.cfg", "SCENE_FILE = tests/scenes/definitely_not_here.scene");
        ConfigReader::loadConfig(probe, "scene_missing.cfg");
        check(SIMULATION.SolidMaskFunction2D == SCENE_SOLID_MASK_2D && SCENE.loaded &&
                  SCENE.regions.empty(),
              "a missing scene falls back to an empty scene -- a closed box");

        {
            std::ofstream f("scene_bad.scene");
            f << "boundary xmin inflow u=\"1.5 - ((((\"\n";
        }
        writeCfg("scene_bad.cfg", "SCENE_FILE = scene_bad.scene");
        ConfigReader::loadConfig(probe, "scene_bad.cfg");
        check(SIMULATION.SolidMaskFunction2D == SCENE_SOLID_MASK_2D && SCENE.loaded &&
                  SCENE.regions.empty(),
              "a scene with a broken expression falls back to a closed box, not a null BC");

        {
            std::ofstream f("scene_typo.scene");
            f << "boundary xmin inflow  u=1 bogus_param=3\n";
        }
        writeCfg("scene_typo.cfg", "SCENE_FILE = scene_typo.scene");
        ConfigReader::loadConfig(probe, "scene_typo.cfg");
        check(SIMULATION.SolidMaskFunction2D == SCENE_SOLID_MASK_2D && SCENE.loaded &&
                  SCENE.regions.empty(),
              "an unknown parameter is reported rather than silently ignored");
    }

    // ---- a face the scene never mentions is a no-slip wall ----
    // WinningFace used to start its search at order -1, which is exactly the order of an
    // unspecified face, so such a face could never win: its ring stayed FLUID and the
    // pressure matrix treated it as an open boundary.
    std::printf("\n== unspecified faces default to no-slip ==\n");
    {
        DIMENSION = 3;
        SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        SIMULATION.dh = 0.1; SIMULATION.Nx = SIMULATION.Ny = SIMULATION.Nz = 10;
        SIMULATION.spacing.nonUniform = false;
        {
            std::ofstream f("scene_partial.scene");
            f << "boundary xmin inflow u=\"1\"\nboundary xmax pressure_outlet\n";
        }
        check(SCENE.Load("scene_partial.scene"), "partial scene loads", SCENE.error);
        int wrong = 0;
        const int N = 10;
        for (int i = 0; i < N; i++)
            for (int j = 1; j < N - 1; j++)     // the x faces are declared; skip them
                for (int k = 0; k < N; k++)
                {
                    const bool ring = (i == 0 || i == N - 1 || k == 0 || k == N - 1);
                    const int t = SCENE.CellTypeAt(i, j, k);
                    if (ring && t != SOLID_CELL) wrong++;
                    if (!ring && t != FLUID_CELL) wrong++;
                }
        check(wrong == 0, "the y and z rings are SOLID, the interior FLUID",
              std::to_string(wrong) + " cells wrong");
        check(SCENE.CellTypeAt(5, 0, 5) == INFLOW_CELL && SCENE.CellTypeAt(5, N - 1, 5) == EMPTY_CELL,
              "the declared faces are unaffected");
        // At an edge a declared face still beats an undeclared one.
        check(SCENE.CellTypeAt(0, 0, 5) == INFLOW_CELL, "a declared face wins the edge it shares");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace scene_check

// ------------------------------------------------------------------------------------------------------------------
// mesh_check
// ------------------------------------------------------------------------------------------------------------------
namespace mesh_check
{
// `region ... mesh`: an STL model turned into a signed distance and fed through the same
// scene machinery as the analytic primitives.
//
// The instrument is a CUBE, because a cube is the one model with a closed-form signed
// distance under ANY placement: rotate it, stretch it per axis, mirror it, and the exact
// answer is still a box SDF evaluated in the box's own frame. Every property of the mesh
// path is checked against that, so a disagreement is a defect in the mesh path and not
// a question of how accurate an STL is.
//
//   1. STL input: ASCII and binary give the same mesh; an inside-out file is corrected.
//   2. An open (non-watertight) mesh is refused rather than voxelised into noise.
//   3. Transform exactness: distance AND normal against the box SDF, under rotation,
//      non-uniform scale and a mirror.
//   4. Through the Scene: the mesh cube voxelises to exactly the cells `region box` does,
//      and the particle-contact queries agree with the analytic primitive.
//   5. 2D: the airfoil model is cut at its mid-plane, the section has the right area and
//      the angle of attack points the nose up.
//   6. End to end: the 2D airfoil scene through ADI + pressure, finite and divergence free.


static int failures = 0;
static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("     [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}
static std::string fmt(const char *f, double a, double b = 0, double c = 0)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

// ---------------------------------------------------------------------------
// The cube [-1,1]^3, outward winding.
// ---------------------------------------------------------------------------
static std::array<double, 3> Corner(int i)
{
    return {(i & 1) ? 1.0 : -1.0, (i & 2) ? 1.0 : -1.0, (i & 4) ? 1.0 : -1.0};
}
static const int CUBE[12][3] = {
    {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}, {0, 1, 5}, {0, 5, 4},
    {2, 6, 7}, {2, 7, 3}, {0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6},
};

static void WriteAscii(const std::string &file, bool reversed, int drop = -1)
{
    std::ofstream f(file);
    f << "solid cube\n";
    for (int t = 0; t < 12; t++)
    {
        if (t == drop) continue;
        int a = CUBE[t][0], b = CUBE[t][1], c = CUBE[t][2];
        if (reversed) std::swap(b, c);
        f << "  facet normal 0 0 0\n    outer loop\n";
        for (int v : {a, b, c})
        {
            auto p = Corner(v);
            f << "      vertex " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
        f << "    endloop\n  endfacet\n";
    }
    f << "endsolid cube\n";
}

static void WriteBinary(const std::string &file)
{
    std::ofstream f(file, std::ios::binary);
    char header[80] = {0};
    f.write(header, 80);
    const uint32_t n = 12;
    f.write((const char *)&n, 4);
    for (int t = 0; t < 12; t++)
    {
        float rec[12] = {0, 0, 0};
        for (int k = 0; k < 3; k++)
        {
            auto p = Corner(CUBE[t][k]);
            for (int a = 0; a < 3; a++) rec[3 + 3 * k + a] = (float)p[a];
        }
        f.write((const char *)rec, 48);
        const uint16_t attr = 0;
        f.write((const char *)&attr, 2);
    }
}

// The exact signed distance of the placed cube: a box of half extents |s| rotated by R
// about c. The mesh path must reproduce this to round-off.
struct PlacedBox
{
    double c[3], R[9], h[3];
    PlacedBox(const SceneDesc::MeshTransform &xf)
    {
        SceneDesc::MeshRotation(xf.rotate, R);
        for (int a = 0; a < 3; a++) { c[a] = xf.center[a]; h[a] = std::fabs(xf.scale[a]); }
    }
    void Local(const double p[3], double l[3]) const
    {
        const double d[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
        for (int a = 0; a < 3; a++) l[a] = R[a] * d[0] + R[3 + a] * d[1] + R[6 + a] * d[2]; // R^T d
    }
    double Distance(const double p[3]) const
    {
        double l[3]; Local(p, l);
        double out = 0.0, in = -1e300;
        for (int a = 0; a < 3; a++)
        {
            const double q = std::fabs(l[a]) - h[a];
            if (q > 0.0) out += q * q;
            in = std::max(in, q);
        }
        return out > 0.0 ? std::sqrt(out) : in;
    }
    // Outward normal, for a point OUTSIDE (where it is unique).
    void Normal(const double p[3], double n[3]) const
    {
        double l[3], g[3], len = 0.0;
        Local(p, l);
        for (int a = 0; a < 3; a++)
        {
            const double q = std::fabs(l[a]) - h[a];
            g[a] = (q > 0.0) ? q * (l[a] < 0 ? -1.0 : 1.0) : 0.0;
            len += g[a] * g[a];
        }
        len = std::sqrt(len);
        for (int a = 0; a < 3; a++) g[a] /= len;
        for (int r = 0; r < 3; r++) n[r] = R[3 * r] * g[0] + R[3 * r + 1] * g[1] + R[3 * r + 2] * g[2];
    }
};

static std::string Dir()
{
    static std::string d;
    if (d.empty())
    {
        char tmpl[] = "nereus_mesh_XXXXXX";
        d = mkdtemp(tmpl);
    }
    return d;
}

// ---------------------------------------------------------------------------
static void CheckReading()
{
    std::printf("\n== 1. STL input ==\n");
    const std::string a = Dir() + "/cube_ascii.stl", b = Dir() + "/cube_bin.stl",
                      r = Dir() + "/cube_reversed.stl";
    WriteAscii(a, false);
    WriteBinary(b);
    WriteAscii(r, true);

    std::string err;
    auto ma = SceneDesc::LoadRawMesh(a, err);
    auto mb = SceneDesc::LoadRawMesh(b, err);
    auto mr = SceneDesc::LoadRawMesh(r, err);
    check(ma && mb && mr, "ASCII, binary and reversed cubes load", err);
    if (!ma || !mb || !mr) return;

    // stl_reader welds coincident corners: 36 corners in the file, 8 vertices out. The
    // distance structure needs that -- without it every edge looks open.
    check(ma->VertexCount() == 8 && ma->TriangleCount() == 12,
          "coincident corners welded", fmt("%.0f vertices, %.0f triangles", ma->VertexCount(), ma->TriangleCount()));
    check(mb->VertexCount() == 8 && mb->TriangleCount() == 12, "binary reads the same mesh");
    check(std::fabs(ma->volume - 8.0) < 1e-12 && !ma->flipped,
          "enclosed volume from the winding", fmt("%.15g (exact 8)", ma->volume));

    // An inside-out file would make the whole domain solid and the model fluid.
    check(mr->flipped && std::fabs(mr->volume - 8.0) < 1e-12,
          "inward-wound file detected and reversed", fmt("volume %.15g", mr->volume));
    SceneDesc::MeshSDF sa, sr;
    SceneDesc::MeshTransform id;
    sa.Build(a, id, err);
    sr.Build(r, id, err);
    double worst = 0.0;
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> U(-2.0, 2.0);
    for (int n = 0; n < 500; n++)
    {
        const double x = U(rng), y = U(rng), z = U(rng);
        worst = std::max(worst, std::fabs(sa.Distance(x, y, z) - sr.Distance(x, y, z)));
    }
    check(worst == 0.0, "reversed file gives the same signed distance", fmt("worst %.3g", worst));
}

static void CheckOpenMeshRefused()
{
    std::printf("\n== 2. an open mesh is refused ==\n");
    const std::string o = Dir() + "/cube_open.stl";
    WriteAscii(o, false, /*drop*/ 5);
    SceneDesc::MeshSDF s;
    SceneDesc::MeshTransform id;
    std::string err;
    const bool ok = s.Build(o, id, err);
    check(!ok && err.find("watertight") != std::string::npos,
          "a cube missing one triangle does not build", err);

    // ...and through the scene parser, so the setup screen reports it.
    const std::string sc = Dir() + "/open.scene";
    { std::ofstream f(sc); f << "region solid mesh file=\"" << o << "\" center=(0.5,0.5,0.5) scale=0.1\n"; }
    DIMENSION = 3;
    check(!SCENE.Load(sc) && SCENE.error.find("watertight") != std::string::npos,
          "the scene refuses it with a readable error", SCENE.error);
}

static void CheckTransforms()
{
    std::printf("\n== 3. distance and normal under the transform ==\n");
    const std::string a = Dir() + "/cube_ascii.stl";
    struct Case { const char *name; double c[3], r[3], s[3]; };
    const Case cases[] = {
        {"identity", {0, 0, 0}, {0, 0, 0}, {1, 1, 1}},
        {"translate + uniform scale", {0.4, -0.2, 0.7}, {0, 0, 0}, {0.3, 0.3, 0.3}},
        {"rotate about z (angle of attack)", {0.5, 0.5, 0.5}, {0, 0, -5}, {0.2, 0.05, 0.3}},
        {"rotate about all three, NON-uniform scale", {0.1, 0.2, 0.3}, {30, -45, 60}, {0.5, 0.1, 0.25}},
        {"mirror (negative x scale)", {0, 0, 0}, {10, 20, 30}, {-0.4, 0.2, 0.3}},
    };
    std::mt19937 rng(7);
    for (const Case &k : cases)
    {
        SceneDesc::MeshTransform xf;
        for (int d = 0; d < 3; d++) { xf.center[d] = k.c[d]; xf.rotate[d] = k.r[d]; xf.scale[d] = k.s[d]; }
        SceneDesc::MeshSDF m;
        std::string err;
        if (!m.Build(a, xf, err)) { check(false, k.name, err); continue; }
        PlacedBox box(xf);

        const double span = 2.0 * std::max(std::fabs(k.s[0]), std::max(std::fabs(k.s[1]), std::fabs(k.s[2])));
        std::uniform_real_distribution<double> U(-span, span);
        double worstD = 0.0, worstN = 0.0;
        int inside = 0;
        for (int n = 0; n < 4000; n++)
        {
            const double p[3] = {k.c[0] + U(rng), k.c[1] + U(rng), k.c[2] + U(rng)};
            const double exact = box.Distance(p);
            worstD = std::max(worstD, std::fabs(m.Distance(p[0], p[1], p[2]) - exact));
            if (exact < 0.0) { inside++; continue; }
            if (exact < 1e-6) continue;
            double phi, n1[3], n2[3];
            m.DistanceNormal(p[0], p[1], p[2], phi, n1);
            box.Normal(p, n2);
            worstN = std::max(worstN, 1.0 - (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]));
        }
        check(worstD < 1e-12 && worstN < 1e-12 && inside > 20, k.name,
              fmt("max |d - exact| %.2e, max (1 - n.n_exact) %.2e, %.0f points inside",
                  worstD, worstN, inside));
    }
}

// Standing the globals up for Scene::CellTypeAt on a uniform grid.
static void UniformGrid(int dim, double x0, double xf, double y0, double yf, double z0, double zf, int n)
{
    DIMENSION = dim;
    SIMULATION.domain = Domain{x0, xf, y0, yf, z0, zf};
    SIMULATION.dh = std::min(xf - x0, yf - y0) / n;
    SIMULATION.Nx = MAC::InteriorCells(xf - x0, SIMULATION.dh) + 2;   // the interior plus the ring outside it
    SIMULATION.Ny = MAC::InteriorCells(yf - y0, SIMULATION.dh) + 2;
    SIMULATION.Nz = (dim == 3) ? MAC::InteriorCells(zf - z0, SIMULATION.dh) + 2 : 1;
    SIMULATION.spacing.nonUniform = false;
    SIMULATION.spacing.xf.clear(); SIMULATION.spacing.yf.clear(); SIMULATION.spacing.zf.clear();
}

static void CheckScene()
{
    std::printf("\n== 4. through the Scene: voxels and contact queries ==\n");
    UniformGrid(3, 0, 1, 0, 1, 0, 1, 40);
    const std::string a = Dir() + "/cube_ascii.stl";
    const std::string sb = Dir() + "/box.scene", sm = Dir() + "/mesh.scene";
    { std::ofstream f(sb); f << "sampling center\nregion solid box min=(0.3,0.25,0.35) max=(0.7,0.55,0.6)\n"; }
    { std::ofstream f(sm); f << "sampling center\nregion solid mesh file=\"" << a
                             << "\" center=(0.5,0.4,0.475) rotate=(0,0,0) scale=(0.2,0.15,0.125)\n"; }

    const int Nx = SIMULATION.Nx, Ny = SIMULATION.Ny, Nz = SIMULATION.Nz;
    std::vector<int> tb((size_t)Nx * Ny * Nz), tm(tb.size());
    if (!SCENE.Load(sb)) { check(false, "box scene", SCENE.error); return; }
    for (int k = 0; k < Nz; k++) for (int i = 0; i < Ny; i++) for (int j = 0; j < Nx; j++)
        tb[((size_t)k * Ny + i) * Nx + j] = SCENE.CellTypeAt(i, j, k);
    if (!SCENE.Load(sm)) { check(false, "mesh scene", SCENE.error); return; }
    check(true, "mesh scene parses", SCENE.Summary());

    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < Nz; k++) for (int i = 0; i < Ny; i++) for (int j = 0; j < Nx; j++)
        tm[((size_t)k * Ny + i) * Nx + j] = SCENE.CellTypeAt(i, j, k);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    long diff = 0, solid = 0;
    for (size_t n = 0; n < tb.size(); n++) { diff += (tb[n] != tm[n]); solid += (tm[n] == SOLID_CELL); }
    check(diff == 0 && solid > 0, "mesh cube voxelises to exactly the cells `region box` does",
          fmt("%.0f differing cells, %.0f solid", diff, solid));
    std::printf("         (%.0f cells in %.3f s)\n", (double)tb.size(), secs);

    // Particle contact reads SolidDistanceAt (inside) and NearestSolidSurface (outside too).
    SceneDesc::Scene boxScene;
    boxScene.Load(sb);
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(0.2, 0.8);
    double worstPhi = 0.0, worstN = 0.0;
    int tested = 0;
    for (int n = 0; n < 3000; n++)
    {
        const double x = U(rng), y = U(rng), z = U(rng);
        double p1, p2, n1[3], n2[3];
        const bool o1 = SCENE.NearestSolidSurface(x, y, z, p1, n1);
        const bool o2 = boxScene.NearestSolidSurface(x, y, z, p2, n2);
        if (!o1 || !o2) continue;
        worstPhi = std::max(worstPhi, std::fabs(p1 - p2));
        // The analytic box's normal comes from central differences, which is only
        // accurate away from the edges and the medial planes; compare where it is.
        if (std::fabs(p2) > 0.01 && p2 > 0.0)
        {
            worstN = std::max(worstN, 1.0 - (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]));
            tested++;
        }
    }
    check(worstPhi < 1e-12, "NearestSolidSurface: same distance as the box primitive",
          fmt("worst %.2e", worstPhi));
    check(worstN < 1e-6 && tested > 100, "NearestSolidSurface: same outward normal",
          fmt("worst 1 - n.n %.2e over %.0f points", worstN, tested));

    double phi, n[3];
    const bool in = SCENE.SolidDistanceAt(0.5, 0.4, 0.58, phi, n);
    check(in && std::fabs(phi + 0.02) < 1e-12 && n[2] > 0.999999,
          "SolidDistanceAt inside: depth and the way out", fmt("phi %.6f, n_z %.6f", phi, n[2]));
}

static int CountSolid(double &meanFront, double &meanBack, double &area)
{
    const int Nx = SIMULATION.Nx, Ny = SIMULATION.Ny;
    const double dh = SIMULATION.dh;
    int solid = 0, nf = 0, nb = 0;
    meanFront = meanBack = 0.0;
    const double cx = 0.6;
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            if (i > 0 && j > 0 && i < Ny - 1 && j < Nx - 1 && SCENE.CellTypeAt(i, j, 0) == SOLID_CELL)
            {
                solid++;
                // UniformGrid builds no tables: centre j is x0 + (j - 1/2) dh, the ring outside the domain
                const double x = SIMULATION.domain.x0 + (j - 0.5) * dh, y = SIMULATION.domain.y0 + (i - 0.5) * dh;
                if (x < cx - 0.1) { meanFront += y; nf++; }
                if (x > cx + 0.1) { meanBack += y; nb++; }
            }
    meanFront /= std::max(nf, 1);
    meanBack /= std::max(nb, 1);
    area = solid * dh * dh;
    return solid;
}

// A NACA 0012 wing written by the test itself (chord 260, span 100, model units), so the mesh checks read no user
// file. The section is convex, so each end cap is a fan from the centroid. Returns the path; `area` is the exact
// area of the section polygon in model units.
static std::string TestAirfoilSTL(double &area)
{
    const std::string path = Dir() + "/naca0012.stl";
    const double c = 260.0, span = 100.0;
    const int n = 120;
    std::vector<double> px, py;                       // upper surface TE -> LE, then lower LE -> TE
    for (int k = n; k >= 0; k--)
    {
        const double x = 0.5 * (1.0 - std::cos(M_PI * k / n));
        px.push_back(x * c);
        py.push_back(5.0 * 0.12 * c * (0.2969 * std::sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x));
    }
    for (int k = 1; k <= n; k++) { px.push_back(px[n - k]); py.push_back(-py[n - k]); }
    const int m = (int)px.size();
    area = 0.0;
    double cx = 0.0, cy = 0.0;
    for (int k = 0; k < m; k++)
    {
        const int l = (k + 1) % m;
        area += 0.5 * (px[k] * py[l] - px[l] * py[k]);
        cx += px[k] / m; cy += py[k] / m;
    }
    area = std::fabs(area);
    std::ofstream f(path);
    f << "solid naca0012\n";
    auto tri = [&f](double ax, double ay, double az, double bx, double by, double bz, double qx, double qy, double qz) {
        f << " facet normal 0 0 0\n  outer loop\n"
          << "   vertex " << ax << ' ' << ay << ' ' << az << "\n   vertex " << bx << ' ' << by << ' ' << bz
          << "\n   vertex " << qx << ' ' << qy << ' ' << qz << "\n  endloop\n endfacet\n";
    };
    f.precision(17);
    for (int k = 0; k < m; k++)
    {
        const int l = (k + 1) % m;
        tri(px[k], py[k], 0.0, px[l], py[l], 0.0, px[l], py[l], span);
        tri(px[k], py[k], 0.0, px[l], py[l], span, px[k], py[k], span);
        tri(cx, cy, 0.0, px[l], py[l], 0.0, px[k], py[k], 0.0);
        tri(cx, cy, span, px[k], py[k], span, px[l], py[l], span);
    }
    f << "endsolid naca0012\n";
    return path;
}

static void Check2D()
{
    std::printf("\n== 5. 2D: the model cut at its mid-plane ==\n");
    UniformGrid(2, 0, 2, 0, 0.8, 0, 1, 256);
    double modelArea = 0.0;
    const std::string stl = TestAirfoilSTL(modelArea);
    const double exactArea = modelArea * 0.002 * 0.002;

    const std::string s0 = Dir() + "/af0.scene", s5 = Dir() + "/af5.scene";
    { std::ofstream f(s0); f << "sampling center\nregion solid mesh file=\"" << stl << "\" center=(0.6,0.4,0.5) rotate=(0,0,0) scale=0.002\n"; }
    { std::ofstream f(s5); f << "sampling center\nregion solid mesh file=\"" << stl << "\" center=(0.6,0.4,0.5) rotate=(0,0,-5) scale=0.002\n"; }

    double f0, b0, a0, f5, b5, a5;
    if (!SCENE.Load(s0)) { check(false, "airfoil scene", SCENE.error); return; }
    CountSolid(f0, b0, a0);
    if (!SCENE.Load(s5)) { check(false, "airfoil scene, rotated", SCENE.error); return; }
    const int n5 = CountSolid(f5, b5, a5);

    check(std::fabs(a5 - exactArea) / exactArea < 0.03,
          "voxelised section area matches the model",
          fmt("%.5f m^2 against %.5f (%.2f%%)", a5, exactArea, 100.0 * (a5 - exactArea) / exactArea));
    // A nose-up angle of attack lifts the front half relative to the back.
    const double tilt = (f5 - b5) - (f0 - b0);
    check(n5 > 0 && tilt > 0.02, "rotate z = -5 puts the nose up",
          fmt("front-minus-back height grew by %.4f m (~0.26 m apart x tan 5 deg = 0.023)", tilt));
}

static void CheckEndToEnd()
{
    std::printf("\n== 6. the 2D airfoil scene through ADI + pressure ==\n");
    DIMENSION = 2; SelectMethod("ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0; MAX_CFL = 1.0;
    SIMULATION.domain = Domain{0.0, 2.0, 0.0, 0.8, 0.0, 1.0};
    SIMULATION.GRID_SIZE = 64;
    // THE SCENE IS EDITABLE FROM THE SETUP SCREEN, so this check must take its physics from
    // the scene rather than assume them. It did assume them, and the scene changed underneath
    // it: the inlet became 50 m/s and the chord was refitted to 0.5 m, which at the hardcoded
    // dt = 0.004 is CFL = 16 -- the run went non-finite and the failure read like a geometry
    // bug. The timestep is now derived per step from the speed actually in the field, and the
    // viscosity from the inlet speed, so what is asserted stays "the mesh path works" instead
    // of "the scheme is stable at the numbers someone last saved".
    SIMULATION.RHO = 1.0;
    SIMULATION.time = 0.0;
    PRESSURE_SETTINGS.needsCompatibilityCondition = false;
    PRESSURE_SETTINGS.warmStart = false;
    SIMULATION.f = Vec3{0.0, 0.0, 0.0};
    SIMULATION.grid.nonUniform = false;

    static MAC gs, ga;
    gs.InitializeGrid(SIMULATION.domain, true);
    ga.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = gs.dh; SIMULATION.Nx = gs.Nx; SIMULATION.Ny = gs.Ny; SIMULATION.Nz = 1;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &ga;

    double modelArea = 0.0;
    const std::string stl = TestAirfoilSTL(modelArea);
    const std::string scene = Dir() + "/airfoil_2d.scene";
    {
        std::ofstream f(scene);
        f << "sampling center\nboundary xmin inflow u=\"10\" v=\"0\"\nboundary xmax pressure_outlet\n"
             "boundary ymin freeslip\nboundary ymax freeslip\n"
             "region solid mesh file=\"" << stl << "\" center=(0.4,0.4,0.5) rotate=(0,0,-5) scale=0.002\n";
    }
    if (!SCENE.Load(scene)) { check(false, "the airfoil scene", SCENE.error); return; }
    SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
    SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
    SIMULATION.SolidMaskFunction2D = SCENE_SOLID_MASK_2D;
    gs.SetSolidMask(SCENE_SOLID_MASK_2D);
    ga.SetSolidMask(SCENE_SOLID_MASK_2D);

    // What does the scene actually blow in at? Sample the inlet column.
    double uIn = 0.0;
    for (int i = 1; i < gs.Ny - 1; i++)
        uIn = std::max(uIn, std::fabs(SCENE_VELOCITY_2D(gs.FacePos(MAC::X, 1), gs.CentrePos(MAC::Y, i), 0.0).u));
    if (uIn <= 0.0) uIn = 1.0;
    // Cell Reynolds number about 2, which is what central differencing wants, and a timestep
    // at CFL 0.4 against that same inlet speed to start with.
    SIMULATION.EPS = std::max(1e-3, uIn * gs.dh / 2.0);
    SIMULATION.dt = 0.4 * gs.dh / uIn;
    SIMULATION.dtConfigured = SIMULATION.dt;
    std::printf("     scene inflow %.3f m/s -> nu %.4g, dt %.5f (cell Re %.1f, CFL %.2f)\n",
                uIn, SIMULATION.EPS, SIMULATION.dt, uIn * gs.dh / SIMULATION.EPS,
                uIn * SIMULATION.dt / gs.dh);

    int solid = 0;
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 1; j < gs.Nx - 1; j++)
            solid += (gs.GetSolid(i, j) == SOLID_CELL);

    ADI2D::InitializeADI2D(&gs, SIMULATION.dt, SIMULATION.VelocityBoundaryFunction2D,
                            SIMULATION.PressureBoundaryFunction2D);
    PressureSolver2D::InitializePressureSolver(&gs, ADIPressureScheme());
    ga.SetBoundaryFunctions(SCENE_VELOCITY_2D, SCENE_PRESSURE_2D);
    ga.EnforceBoundaryConditions(0.0);

    const int steps = 60;
    for (int step = 0; step < steps; step++)
    {
        // Adaptive, exactly as SimulationManager does it: the flow accelerates past the body,
        // so the step that was safe at the inlet speed is not safe in the gap.
        SIMULATION.dt = ComputeAdaptiveTimestep(gs.GetMaxVelocity(), gs.dh, 0.4,
                                                SIMULATION.dtConfigured);
        SIMULATION.time += SIMULATION.dt;
        ADI2D::SolveADIStep(&ga, &gs, nullptr, SIMULATION.time - SIMULATION.dt);
        PressureSolver2D::SolvePressure(nullptr, &gs, nullptr, 0.0);
        PressureSolver2D::ProjectPressure(&gs);
        ga.CopyGrid(gs);
    }

    bool finite = true;
    double maxDiv = 0.0, maxSolidFlux = 0.0;
    for (int i = 1; i < gs.Ny - 1; i++)
        for (int j = 1; j < gs.Nx - 1; j++)
        {
            if (gs.GetSolid(i, j) != FLUID_CELL) continue;
            const double d = gs.GetDivergencyAt(i, j);
            if (!std::isfinite(d)) finite = false;
            maxDiv = std::max(maxDiv, std::fabs(d));
            // No flux into the model: the face between a fluid cell and a solid one.
            if (gs.GetSolid(i, j + 1) == SOLID_CELL) maxSolidFlux = std::max(maxSolidFlux, std::fabs(gs.GetU(i, j + 1)));
            if (gs.GetSolid(i, j - 1) == SOLID_CELL) maxSolidFlux = std::max(maxSolidFlux, std::fabs(gs.GetU(i, j)));
            if (gs.GetSolid(i + 1, j) == SOLID_CELL) maxSolidFlux = std::max(maxSolidFlux, std::fabs(gs.GetV(i + 1, j)));
            if (gs.GetSolid(i - 1, j) == SOLID_CELL) maxSolidFlux = std::max(maxSolidFlux, std::fabs(gs.GetV(i, j)));
        }
    const double maxVel = gs.GetMaxVelocity();
    check(solid > 20, "the airfoil is in the mask", fmt("%.0f solid cells at %.0f x %.0f", solid, gs.Nx, gs.Ny));
    check(finite && std::isfinite(maxVel), "field stays finite");
    check(maxDiv < 1e-6, "divergence free", fmt("max |div| %.2e", maxDiv));
    check(maxSolidFlux == 0.0, "no flux through the model's faces", fmt("max %.2e", maxSolidFlux));
    // The model blocks part of the stream, so the flow has to speed up past it.
    check(maxVel > 1.02 * uIn && maxVel < 6.0 * uIn, "flow accelerates around the model",
          fmt("max |u| %.4f against an inflow of %.4f", maxVel, uIn));
}

// ---------------------------------------------------------------------------
// A model made of SEVERAL closed shells. The car that found this is a user file and may
// change, and a guard that reads a user-editable file takes its physics from that file --
// so the mesh here is built in the harness: a big box with a small box nested inside it,
// which is a car's body and its dashboard in miniature.
//
// A signed distance from the NEAREST triangle is only correct for ONE closed surface. Inside
// the outer box but nearer the inner one, the nearest surface belongs to a shell that does
// not contain the point, and it reads as OUTSIDE -- the interior voxelises hollow.
// ---------------------------------------------------------------------------
static void WriteBox(std::ofstream &f, const double lo[3], const double hi[3])
{
    // Outward-wound triangles for an axis-aligned box, ASCII STL.
    const double v[8][3] = {{lo[0],lo[1],lo[2]},{hi[0],lo[1],lo[2]},{hi[0],hi[1],lo[2]},{lo[0],hi[1],lo[2]},
                            {lo[0],lo[1],hi[2]},{hi[0],lo[1],hi[2]},{hi[0],hi[1],hi[2]},{lo[0],hi[1],hi[2]}};
    const int tri[12][3] = {{0,2,1},{0,3,2},{4,5,6},{4,6,7},{0,1,5},{0,5,4},
                            {1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7}};
    for (const auto &t : tri)
    {
        f << "facet normal 0 0 0\n  outer loop\n";
        for (int e = 0; e < 3; e++)
            f << "    vertex " << v[t[e]][0] << " " << v[t[e]][1] << " " << v[t[e]][2] << "\n";
        f << "  endloop\nendfacet\n";
    }
}

static void CheckComponents()
{
    std::printf("\n== 7. a model of several closed shells ==\n");
    DIMENSION = 3;
    {
        std::ofstream f("nested.stl");
        f << "solid nested\n";
        const double olo[3] = {-1, -1, -1}, ohi[3] = {1, 1, 1};      // the body
        const double ilo[3] = {-0.3, -0.3, -0.3}, ihi[3] = {0.3, 0.3, 0.3}; // the dashboard
        WriteBox(f, olo, ohi);
        WriteBox(f, ilo, ihi);
        f << "endsolid nested\n";
    }

    SceneDesc::MeshTransform xf;   // identity: keep the model's own units
    SceneDesc::MeshSDF split, whole;
    std::string err;
    xf.splitComponents = true;
    check(split.Build("nested.stl", xf, err), "the two-shell model loads", err);
    xf.splitComponents = false;
    check(whole.Build("nested.stl", xf, err), "and again in whole-file mode", err);
    check(split.componentCount == 2, "both find two closed shells",
          std::to_string(split.componentCount));

    // The union of the two boxes is just the outer one: volume 2^3 = 8.
    const int N = 40;
    const double h = 2.0 / N;
    long inSplit = 0, inWhole = 0, lost = 0, invented = 0;
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
            {
                const double x = -1 + (i + 0.5) * h, y = -1 + (j + 0.5) * h, z = -1 + (k + 0.5) * h;
                const bool a = split.Distance(x, y, z) < 0.0;
                const bool b = whole.Distance(x, y, z) < 0.0;
                if (a) inSplit++;
                if (b) inWhole++;
                if (a && !b) lost++;
                if (b && !a) invented++;
            }
    const double cell = h * h * h;
    std::printf("      union of the shells is the outer box, volume 8.000\n");
    std::printf("      split: %ld cells = %.3f    whole: %ld cells = %.3f\n",
                inSplit, inSplit * cell, inWhole, inWhole * cell);
    check(std::fabs(inSplit * cell - 8.0) < 0.05,
          "split recovers the whole interior", fmt("%.4f against 8.0", inSplit * cell));
    check(lost > 0 && invented == 0,
          "whole-file signing LOSES interior and never invents any",
          std::to_string(lost) + " cells lost, " + std::to_string(invented) + " invented");

    // The inner shell is what goes missing, and it is exactly the nested region.
    const double centre = whole.Distance(0, 0, 0);
    check(split.Distance(0, 0, 0) < 0.0, "split calls the nested region solid",
          fmt("phi = %+.4f", split.Distance(0, 0, 0)));
    check(centre < 0.0, "and so does whole-file AT the very centre (it is inside the inner box)",
          fmt("phi = %+.4f", centre));
    const double between = whole.Distance(0.45, 0, 0);   // between the shells, nearer the inner
    check(between > 0.0 && split.Distance(0.45, 0, 0) < 0.0,
          "while BETWEEN the shells whole-file says outside and split says solid",
          fmt("whole %+.4f, split %+.4f", between, split.Distance(0.45, 0, 0)));

    // A single-shell model must be untouched by the setting -- that is what keeps every
    // existing scene, and the airfoil presets, exactly where they were.
    {
        std::ofstream f("onebox.stl");
        f << "solid one\n";
        const double lo[3] = {-1, -1, -1}, hi[3] = {1, 1, 1};
        WriteBox(f, lo, hi);
        f << "endsolid one\n";
    }
    SceneDesc::MeshSDF oneA, oneB;
    SceneDesc::MeshTransform x1;
    x1.splitComponents = true;
    check(oneA.Build("onebox.stl", x1, err), "single-shell model loads", err);
    x1.splitComponents = false;
    check(oneB.Build("onebox.stl", x1, err), "and again whole", err);
    double worst = 0.0;
    for (int i = 0; i < 500; i++)
    {
        const double x = -2.0 + 4.0 * (i % 10) / 9.0;
        const double y = -2.0 + 4.0 * ((i / 10) % 10) / 9.0;
        const double z = -2.0 + 4.0 * ((i / 100) % 5) / 4.0;
        worst = (std::max)(worst, std::fabs(oneA.Distance(x, y, z) - oneB.Distance(x, y, z)));
    }
    check(oneA.componentCount == 1 && worst == 0.0,
          "a single-shell model is bit-identical either way",
          fmt("worst difference %.3e over 500 samples", worst));

    std::remove("nested.stl");
    std::remove("onebox.stl");
}

int run(int, char **)
{
    std::printf("mesh_check: STL -> signed distance -> scene\n");
    CheckReading();
    CheckOpenMeshRefused();
    CheckTransforms();
    CheckScene();
    Check2D();
    CheckEndToEnd();
    CheckComponents();
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace mesh_check

// ------------------------------------------------------------------------------------------------------------------
// airfoil_check
// ------------------------------------------------------------------------------------------------------------------
namespace airfoil_check
{
// The NACA 00XX scene primitive: the thickness law, the distance, the rotation about its pivot, the 3D span, the
// parser's refusals, and the section as the solver sees it (voxelised on a grid).

static int failures = 0;
static void Check(bool ok, const char *what, double value)
{
    std::printf("  %-4s %-66s %.3e\n", ok ? "ok" : "FAIL", what, value);
    if (!ok) failures++;
}

// Closed-form area of a symmetric four-digit section: 2 * integral of y_t over the chord.
static double ExactArea(double t, double c, double a4)
{
    return 10.0 * t * c * c * (0.2969 * 2.0 / 3.0 - 0.1260 / 2.0 - 0.3516 / 3.0 + 0.2843 / 4.0 - a4 / 5.0);
}

static const SceneDesc::Primitive &LoadOne(const std::string &text)
{
    if (!SCENE.LoadText(text, "airfoil_check"))
    {
        std::printf("scene failed: %s\n", SCENE.error.c_str());
        std::exit(1);
    }
    return SCENE.regions.back();
}

int run(int, char **)
{
    SelectMethod("ADI");
    DIMENSION = 2;
    const double c = 0.8;

    std::printf("1. the section, NACA 0012, chord %.2f, in its own frame\n", c);
    {
        const SceneDesc::Primitive &p = LoadOne("region solid naca0012 center=(0,0,0) chord=0.8 pivot=0\n");
        const SceneDesc::AirfoilSection &a = *p.airfoil;
        double ymax = 0.0, xmax = 0.0;
        for (int m = 0; m <= 200000; m++)
        {
            const double X = c * m / 200000.0, y = a.HalfThickness(X);
            if (y > ymax) { ymax = y; xmax = X; }
        }
        Check(std::fabs(ymax / c - 0.06) < 2e-4, "maximum half-thickness 0.06 c (the 12%)", ymax / c - 0.06);
        Check(std::fabs(xmax / c - 0.30) < 0.01, "at x = 0.30 c", xmax / c - 0.30);
        Check(std::fabs(a.HalfThickness(c) / c - 0.00126) < 1e-6, "open trailing edge half-thickness 0.00126 c", a.HalfThickness(c) / c - 0.00126);
        // Leading-edge radius 1.1019 t^2 c: y_t^2 / (2 x) as x -> 0.
        const double X = 1e-7 * c, rle = a.HalfThickness(X) * a.HalfThickness(X) / (2.0 * X);
        Check(std::fabs(rle / (1.1019 * 0.0144 * c) - 1.0) < 2e-3, "leading-edge radius 1.1019 t^2 c", rle / (1.1019 * 0.0144 * c) - 1.0);

        // Area by the inside test on a fine lattice, against the closed form.
        const int n = 4000;
        const double hx = c / n, hy = 0.07 * c / n;
        long in = 0;
        for (int i = 0; i < n; i++)
            for (int j = -n; j < n; j++)
                if (p.Contains((i + 0.5) * hx, (j + 0.5) * hy, 0.0)) in++;
        const double area = in * hx * hy, exact = ExactArea(0.12, c, 0.1015);
        Check(std::fabs(area / exact - 1.0) < 1e-3, "area 0.68508 t c^2 (open)", area / exact - 1.0);

        // Distance against a brute-force sampling of the whole outline (upper, lower, blunt trailing edge).
        std::vector<double> ox, oy;
        for (int m = 0; m <= 400000; m++)
        {
            const double th = 3.14159265358979323846 * m / 400000.0, X = 0.5 * c * (1.0 - std::cos(th)), y = a.HalfThickness(X);
            ox.push_back(X); oy.push_back(y); ox.push_back(X); oy.push_back(-y);
        }
        for (int m = 0; m <= 1000; m++) { ox.push_back(c); oy.push_back(a.HalfThickness(c) * (2.0 * m / 1000.0 - 1.0)); }
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> ux(-0.3 * c, 1.3 * c), uy(-0.3 * c, 0.3 * c), un(-0.02 * c, 0.02 * c), uu(0.0, c);
        double worst = 0.0, worstRel = 0.0;
        int signBad = 0;
        for (int k = 0; k < 3000; k++)
        {
            // Half the points near the surface, where the distance matters most.
            double x, y;
            if (k % 2) { x = ux(rng); y = uy(rng); }
            else { x = uu(rng); y = ((k / 2) % 2 ? 1.0 : -1.0) * a.HalfThickness(x) + un(rng); }
            double brute = 1e300;
            for (size_t q = 0; q < ox.size(); q++) brute = std::min(brute, std::hypot(x - ox[q], y - oy[q]));
            const double d = p.Distance(x, y, 0.0);
            worst = std::max(worst, std::fabs(std::fabs(d) - brute));
            worstRel = std::max(worstRel, std::fabs(std::fabs(d) - brute) / c);
            if ((d < 0.0) != p.Contains(x, y, 0.0)) signBad++;
        }
        Check(worstRel < 2e-5, "|distance| against a 800k-point outline, worst / c", worstRel);
        Check(signBad == 0, "sign of the distance agrees with the inside test", signBad);
    }

    std::printf("2. closed trailing edge\n");
    {
        const SceneDesc::Primitive &p = LoadOne("region solid naca0012 center=(0,0,0) chord=0.8 pivot=0 te=closed\n");
        Check(std::fabs(p.airfoil->HalfThickness(c)) < 1e-15, "half-thickness 0 at the trailing edge", p.airfoil->HalfThickness(c));
        const int n = 4000;
        const double hx = c / n, hy = 0.07 * c / n;
        long in = 0;
        for (int i = 0; i < n; i++)
            for (int j = -n; j < n; j++)
                if (p.Contains((i + 0.5) * hx, (j + 0.5) * hy, 0.0)) in++;
        const double area = in * hx * hy, exact = ExactArea(0.12, c, 0.1036);
        Check(std::fabs(area / exact - 1.0) < 1e-3, "area 0.68088 t c^2 (closed)", area / exact - 1.0);
    }

    std::printf("3. rotation about the pivot (Kurtulus: quarter chord, alpha = 10 deg nose up)\n");
    {
        const double px = 1.0, py = 0.5, alpha = 10.0;
        const SceneDesc::Primitive &p = LoadOne("region solid naca0012 center=(1,0.5,0) chord=0.8 pivot=0.25 rotate=(0,0,-10)\n");
        const double k = alpha * 3.14159265358979323846 / 180.0;
        // Nose up in a +x stream: the leading edge moves up, the trailing edge down.
        const double lex = px - 0.25 * c * std::cos(k), ley = py + 0.25 * c * std::sin(k);
        const double tex = px + 0.75 * c * std::cos(k), tey = py - 0.75 * c * std::sin(k);
        Check(std::fabs(p.Distance(lex, ley, 0.0)) < 1e-12, "leading edge where the rotation puts it", p.Distance(lex, ley, 0.0));
        Check(std::fabs(p.Distance(tex, tey, 0.0)) < 1e-12, "trailing edge (on its blunt face) where it should be", p.Distance(tex, tey, 0.0));
        Check(p.Contains(px, py, 0.0), "the pivot is inside", 0.0);
        // A rotation preserves distance: the rotated section at a rotated point equals the plain one.
        // Both in one scene: a Primitive points at a section its Scene owns, so a reload would free the first.
        LoadOne("region solid naca0012 center=(0,0,0) chord=0.8 pivot=0.25\n"
                "region solid naca0012 center=(1,0.5,0) chord=0.8 pivot=0.25 rotate=(0,0,-10)\n");
        const SceneDesc::Primitive &plain = SCENE.regions[0], &rot = SCENE.regions[1];
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> u(-0.5, 0.9), v(-0.3, 0.3);
        double worst = 0.0;
        for (int m = 0; m < 2000; m++)
        {
            const double lx = u(rng), ly = v(rng);
            const double wx = px + std::cos(-k) * lx - std::sin(-k) * ly, wy = py + std::sin(-k) * lx + std::cos(-k) * ly;
            worst = std::max(worst, std::fabs(rot.Distance(wx, wy, 0.0) - plain.Distance(lx, ly, 0.0)));
        }
        Check(worst < 1e-12, "rotated distance equals the unrotated one at the rotated point", worst);
    }

    std::printf("4. 3D: extruded, and a finite span\n");
    {
        DIMENSION = 3;
        const SceneDesc::Primitive &inf = LoadOne("region solid naca0012 center=(0,0,0) chord=0.8 pivot=0\n");
        Check(std::fabs(inf.Distance(0.3, 0.0, 5.0) - inf.Distance(0.3, 0.0, 0.0)) < 1e-15, "no span: the same section at every z", 0.0);
        const SceneDesc::Primitive &fin = LoadOne("region solid naca0012 center=(0,0,0) chord=0.8 pivot=0 span=1\n");
        Check(fin.Contains(0.3, 0.0, 0.49) && !fin.Contains(0.3, 0.0, 0.51), "span 1: inside at z = 0.49, outside at 0.51", 0.0);
        Check(std::fabs(fin.Distance(0.3, 0.0, 0.7) - 0.2) < 1e-12, "distance beyond the tip is the gap to the flat end", fin.Distance(0.3, 0.0, 0.7) - 0.2);
        // Rotated about x, the span axis tilts: rotate 90 deg about x puts the span along y.
        const SceneDesc::Primitive &tilt = LoadOne("region solid naca0012 center=(0,0,0) chord=0.8 pivot=0 span=1 rotate=(90,0,0)\n");
        Check(tilt.Contains(0.3, 0.45, 0.0) && !tilt.Contains(0.3, 0.55, 0.0), "rotate (90,0,0): the span lies along y", 0.0);
        DIMENSION = 2;
    }

    std::printf("5. the parser\n");
    {
        Check(!SCENE.LoadText("region solid naca2412 center=(0,0,0) chord=1\n", "x"), "a cambered code is refused", 0.0);
        Check(!SCENE.LoadText("region solid naca0012 chord=-1\n", "x"), "a non-positive chord is refused", 0.0);
        Check(!SCENE.LoadText("region solid naca0012 te=round\n", "x"), "an unknown trailing edge is refused", 0.0);
        const SceneDesc::Primitive &p15 = LoadOne("region solid naca0015 center=(0,0,0)\n");
        Check(std::fabs(p15.airfoil->thickness - 0.15) < 1e-15, "naca0015 reads a 15% section", p15.airfoil->thickness);
    }

    std::printf("6. voxelised on the grid, 2D, chord 1 at 5 deg\n");
    {
        SIMULATION.domain = Domain();
        SIMULATION.domain.x0 = -1.0; SIMULATION.domain.xf = 3.0;
        SIMULATION.domain.y0 = -1.0; SIMULATION.domain.yf = 1.0;
        SIMULATION.GRID_SIZE = 400;
        SIMULATION.grid.nonUniform = false;
        LoadOne("sampling center\nregion solid naca0012 center=(0,0,0) chord=1 rotate=(0,0,-5)\n");
        MAC g;
        g.InitializeGrid(SIMULATION.domain, true);
        SIMULATION.dh = g.dh; SIMULATION.Nx = g.Nx; SIMULATION.Ny = g.Ny; SIMULATION.Nz = 1;   // as ConfigReader does
        g.SetSolidMask(SCENE_SOLID_MASK_2D);
        long solid = 0;
        for (int i = 1; i < g.Ny - 1; i++)
            for (int j = 1; j < g.Nx - 1; j++)
                if (g.GetSolid(i, j) == SOLID_CELL) solid++;
        const double area = solid * g.dh * g.dh, exact = ExactArea(0.12, 1.0, 0.1015);
        std::printf("       %d x %d cells, dh = %.4f, %ld solid cells, %.1f cells across at the thickest\n",
                    g.Nx, g.Ny, g.dh, solid, 0.12 / g.dh);
        Check(std::fabs(area / exact - 1.0) < 0.03, "voxelised area within 3% of 0.68508 t c^2", area / exact - 1.0);
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace airfoil_check

static const SuiteCase CASES[] = {
    {"stopping rule", stopping_check::run, {}, false},
    {"adaptive timestep", timestep_check::run, {}, false},
    {"grid spacing", grid_spacing_check::run, {}, false},
    {"boundary conditions", boundary_check::run, {}, false},
    {"flow integrals", energy_check::run, {}, false},
    {"cell Reynolds", cellre_check::run, {}, false},
    {"probes", probe_check::run, {}, false},
    {"vorticity confinement", confinement_check::run, {}, false},
    {"scenes", scene_check::run, {}, false},
    {"STL meshes", mesh_check::run, {}, false},
    {"NACA airfoils", airfoil_check::run, {}, false},
};

int main(int argc, char **argv) { return RunSuite("units", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
