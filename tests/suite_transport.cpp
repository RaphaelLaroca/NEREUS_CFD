// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The transport suite: transport_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "Functions.h"
#include "headers/Core/Scene.h"
#include "headers/Core/Checkpoint.h"
#include "headers/Solvers/TransportSolver2D.h"
#include "headers/Solvers/TransportSolver3D.h"
#include "Methods/ADI/ADI3D.h"
#include "Methods/ADI/ADI2D.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// transport_check
// ------------------------------------------------------------------------------------------------------------------
namespace transport_check
{
// Registered cell-centred scalars on the MAC grid: the plumbing, before any solver exists.
//
// Stage 1 of the scalar transport work.
// There is deliberately NO transport here yet -- a field that advects with nothing, so that
// every hazard below is proven on its own and cannot later be blamed on the physics.
//
// The hazards are all of one kind: four places hand-write the list of fields a grid carries,
// and a new field is SILENTLY DROPPED by each of them until someone remembers. CopyGrid alone
// runs three times a step on the FLIP path, so a dropped scalar would look like a transport
// bug rather than a plumbing one.
//
//   1. a scalar is indexed exactly as the pressure is        (same cell, same stride)
//   2. registration is idempotent, names are the identity, the SCHEME is per field
//   3. CopyGrid carries it -- by NAME, not by position
//   4. a rebuilt grid brings its fields back at the new size
//   5. the checkpoint round trips it bit for bit, on BOTH grids
//   6. and refuses a file whose fields do not match this run
// Sections 7-12 advect in 2D (TransportSolver2D); 13-17 are the 3D twin (TransportSolver3D); 18-24 are
// Stage 5: the two forms, diffusion, reaction and the source, against exact solutions.


static int fails = 0;
static void check(bool ok, const std::string &n, const std::string &d = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", n.c_str(), d.empty() ? "" : " -- ", d.c_str());
    if (!ok) fails++;
}

static std::string fmt(const char *f, double a = 0, double b = 0)
{ char buf[256]; std::snprintf(buf, sizeof(buf), f, a, b); return buf; }

static MAC gs, ga;

static void StandUp(int gridSize = 16)
{
    DIMENSION = 3; SelectMethod("ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = gridSize;
    SIMULATION.RHO = 1.0; SIMULATION.EPS = 1e-3;
    SIMULATION.dt = 0.002; SIMULATION.dtConfigured = 0.002; SIMULATION.time = 0.0;
    SIMULATION.grid.nonUniform = false;
    PRESSURE_SETTINGS.warmStart = false;

    // The registry deliberately SURVIVES InitializeGrid -- a restart at a different
    // resolution must keep its fields and resize them, which is what check 4 pins. So a
    // harness that stands the same static grids up again has to clear them explicitly, or
    // each section inherits the last one's fields.
    gs.ClearScalars();
    ga.ClearScalars();

    gs.InitializeGrid(SIMULATION.domain, false);
    ga.InitializeGrid(SIMULATION.domain, false);
    SIMULATION.dh = gs.dh; SIMULATION.Nx = gs.Nx; SIMULATION.Ny = gs.Ny; SIMULATION.Nz = gs.Nz;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &ga;
}

// A field with a distinct value in every cell, so a copy that loses or shuffles anything shows.

// ---- stage 2 helpers. Named functions, not lambdas. ----------------------
static void StandUp2D(int gridSize)
{
    DIMENSION = 2; SelectMethod("ADI"); THREAD_COUNT = 4;
    GPU_ACCELERATION = 0; ADAPTATIVE_TIMESTEP = 0;
    SIMULATION.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    SIMULATION.GRID_SIZE = gridSize;
    SIMULATION.dt = 0.002; SIMULATION.dtConfigured = 0.002; SIMULATION.time = 0.0;
    SIMULATION.grid.nonUniform = false;

    gs.ClearScalars();
    ga.ClearScalars();
    gs.InitializeGrid(SIMULATION.domain, true);
    ga.InitializeGrid(SIMULATION.domain, true);
    SIMULATION.dh = gs.dh; SIMULATION.Nx = gs.Nx; SIMULATION.Ny = gs.Ny;
    SIMULATION.GRID_SOL = &gs; SIMULATION.GRID_ANT = &ga;
}

static void UniformFlow(double u, double v)
{
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx + 1; j++) gs.SetU(i, j, u);
    for (int i = 0; i < gs.Ny + 1; i++)
        for (int j = 0; j < gs.Nx; j++) gs.SetV(i, j, v);
}

// u = d(psi)/dy, v = -d(psi)/dx for psi = sin(pi x) sin(pi y). The normal component is zero on
// every wall, so a closed box really is closed.
static void CellularFlow()
{
    const double pi = M_PI;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx + 1; j++)
            gs.SetU(i, j, pi * std::sin(pi * gs.FacePos(MAC::X, j)) * std::cos(pi * gs.CentrePos(MAC::Y, i)));
    for (int i = 0; i < gs.Ny + 1; i++)
        for (int j = 0; j < gs.Nx; j++)
            gs.SetV(i, j, -pi * std::cos(pi * gs.CentrePos(MAC::X, j)) * std::sin(pi * gs.FacePos(MAC::Y, i)));
}

static void StepProfile(int h)
{
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
            ga.SetS(h, i, j, (j < ga.Nx / 3) ? 1.0 : 0.0);
}

static void BlobProfile(int h)
{
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
        {
            const double dx = ga.CentrePos(MAC::X, j) - 0.3, dy = ga.CentrePos(MAC::Y, i) - 0.5;
            ga.SetS(h, i, j, (dx * dx + dy * dy < 0.15 * 0.15) ? 1.0 : 0.0);
        }
}

static double TotalMass(int h)
{
    double t = 0.0;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            t += ga.GetS(h, i, j) * ga.Width(MAC::X, j) * ga.Width(MAC::Y, i);
    return t;
}

// The step loop the manager will run: transport reads GRID_ANT, writes GRID_SOL, and the copy
// at the end carries it forward. The velocity is held fixed here, which is what isolates the
// transport from everything else.
static void RunSteps(int steps, double &lo, double &hi)
{
    for (int s = 0; s < steps; s++)
    {
        TransportSolver2D::SolveTransportStep(&ga, &gs, nullptr, 0.0);
        for (int i = 0; i < gs.Ny; i++)
            for (int j = 0; j < gs.Nx; j++)
                ga.SetS(0, i, j, gs.GetS(0, i, j));
    }
    lo = 1e300; hi = -1e300;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
        {
            const double v = gs.GetS(0, i, j);
            lo = (std::min)(lo, v); hi = (std::max)(hi, v);
        }
}

// The furthest x at which the field is still above `level` -- the leading edge.
static double FrontPosition(int h, double level)
{
    double x = 0.0;
    const int i = gs.Ny / 2;
    for (int j = 0; j < gs.Nx; j++)
        if (gs.GetS(h, i, j) >= level) x = gs.CentrePos(MAC::X, j);
    return x;
}

// ---- stage 4 helpers: the 3D twin. -------------------------------------------------------
static void UniformFlow3D(double u, double v, double w)
{
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx + 1; j++)
            for (int k = 0; k < gs.Nz; k++) gs.SetU(i, j, k, u);
    for (int i = 0; i < gs.Ny + 1; i++)
        for (int j = 0; j < gs.Nx; j++)
            for (int k = 0; k < gs.Nz; k++) gs.SetV(i, j, k, v);
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            for (int k = 0; k < gs.Nz + 1; k++) gs.SetW(i, j, k, w);
}

// Divergence free, with the normal component zero on all six walls:
// u = pi sin(pi x) cos(pi y), v = -pi cos(pi x) sin(pi y) + pi sin(pi y) cos(pi z), w = -pi cos(pi y) sin(pi z).
static void CellularFlow3D()
{
    const double pi = M_PI;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx + 1; j++)
            for (int k = 0; k < gs.Nz; k++)
                gs.SetU(i, j, k, pi * std::sin(pi * gs.FacePos(MAC::X, j)) * std::cos(pi * gs.CentrePos(MAC::Y, i)));
    for (int i = 0; i < gs.Ny + 1; i++)
        for (int j = 0; j < gs.Nx; j++)
            for (int k = 0; k < gs.Nz; k++)
                gs.SetV(i, j, k, -pi * std::cos(pi * gs.CentrePos(MAC::X, j)) * std::sin(pi * gs.FacePos(MAC::Y, i)) +
                                  pi * std::sin(pi * gs.FacePos(MAC::Y, i)) * std::cos(pi * gs.CentrePos(MAC::Z, k)));
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            for (int k = 0; k < gs.Nz + 1; k++)
                gs.SetW(i, j, k, -pi * std::cos(pi * gs.CentrePos(MAC::Y, i)) * std::sin(pi * gs.FacePos(MAC::Z, k)));
}

// 1 in the first third along `axis` (0 x, 1 y, 2 z), 0 elsewhere.
static void StepProfile3D(int h, int axis)
{
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++)
            for (int k = 0; k < ga.Nz; k++)
            {
                const int along = (axis == 0) ? j : (axis == 1 ? i : k);
                const int n = (axis == 0) ? ga.Nx : (axis == 1 ? ga.Ny : ga.Nz);
                ga.SetS(h, i, j, k, (along < n / 3) ? 1.0 : 0.0);
            }
}

static double TotalMass3D(int h)
{
    double t = 0.0;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            for (int k = 0; k < gs.Nz; k++)
                t += ga.GetS(h, i, j, k) * ga.Width(MAC::X, j) * ga.Width(MAC::Y, i) * ga.Width(MAC::Z, k);
    return t;
}

static void RunSteps3D(int steps, double &lo, double &hi)
{
    for (int s = 0; s < steps; s++)
    {
        TransportSolver3D::SolveTransportStep(&ga, &gs, nullptr, 0.0);
        for (int i = 0; i < gs.Ny; i++)
            for (int j = 0; j < gs.Nx; j++)
                for (int k = 0; k < gs.Nz; k++) ga.SetS(0, i, j, k, gs.GetS(0, i, j, k));
    }
    lo = 1e300; hi = -1e300;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            for (int k = 0; k < gs.Nz; k++)
            {
                const double v = gs.GetS(0, i, j, k);
                lo = (std::min)(lo, v); hi = (std::max)(hi, v);
            }
}

static const char *SchemeLabel(TransportScheme s) { return (s == TransportScheme::Central) ? "central" : "upwind"; }

// ---- stage 5 helpers: diffusion, reaction, source, and the two forms. ----------------------------
static ScalarTransport Transport(TransportScheme scheme, TransportForm form, double epsilon, double gamma)
{
    ScalarTransport t;
    t.scheme = scheme; t.form = form; t.epsilon = epsilon; t.gamma = gamma;
    return t;
}

static void RegisterBoth(const std::string &name, const ScalarTransport &t)
{
    gs.RegisterScalar(name, t);
    ga.RegisterScalar(name, t);
}

static void Fill2D(int h, double value)
{
    for (int i = 0; i < ga.Ny; i++)
        for (int j = 0; j < ga.Nx; j++) ga.SetS(h, i, j, value);
}

// The one-cell ring of non-fluid cells every real run has. Without it the first fluid cell touches the
// domain face, and a y-face query half a cell from xmin lands in the scene's xmin band -- a corner the
// solver never sees in a run, because there the ring cell occupies it.
static void MakeRing2D()
{
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            if (i == 0 || j == 0 || i == gs.Ny - 1 || j == gs.Nx - 1) { gs.SetSolid(i, j, SOLID_CELL); ga.SetSolid(i, j, SOLID_CELL); }
}

static void MakeRing3D()
{
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            for (int k = 0; k < gs.Nz; k++)
                if (i == 0 || j == 0 || k == 0 || i == gs.Ny - 1 || j == gs.Nx - 1 || k == gs.Nz - 1)
                { gs.SetSolid(i, j, k, SOLID_CELL); ga.SetSolid(i, j, k, SOLID_CELL); }
}

// The fluid cells' extent along x on row i (the harness grids carry a non-fluid ring).
static void FluidRangeX(int i, int &jlo, int &jhi)
{
    jlo = -1; jhi = -1;
    for (int j = 0; j < gs.Nx; j++)
        if (gs.GetSolid(i, j) == FLUID_CELL) { if (jlo < 0) jlo = j; jhi = j; }
}

static void FluidRangeY(int j, int &ilo, int &ihi)
{
    ilo = -1; ihi = -1;
    for (int i = 0; i < gs.Ny; i++)
        if (gs.GetSolid(i, j) == FLUID_CELL) { if (ilo < 0) ilo = i; ihi = i; }
}

static double MaxDeviationFluid2D(int h, double value)
{
    double worst = 0.0;
    for (int i = 0; i < gs.Ny; i++)
        for (int j = 0; j < gs.Nx; j++)
            if (gs.GetSolid(i, j) == FLUID_CELL) worst = (std::max)(worst, std::fabs(gs.GetS(h, i, j) - value));
    return worst;
}

// Steady 1D advection-diffusion between phi = 0 at the inflow face and phi = 1 at the outflow face,
// against (exp(Pe xi) - 1)/(exp(Pe) - 1). v = 0 and the field is uniform in y, so the y sweep is the
// identity and the steady state carries no splitting error. Returns the RMS error over the fluid cells.
static double SteadyAdvectionDiffusionError(int N, TransportScheme scheme, double &pe)
{
    StandUp2D(N);
    MakeRing2D();
    const char *text = "sampling center\n"
                       "scalar c epsilon=0.1\n"
                       "boundary ymin noslip\nboundary ymax noslip\n"
                       "boundary xmin inflow u=\"1\" c=\"0\"\n"
                       "boundary xmax pressure_outlet c=\"1\"\n";
    if (!SCENE.LoadText(text)) { std::printf("      scene: %s\n", SCENE.error.c_str()); return 1e300; }
    RegisterBoth("c", Transport(scheme, TransportForm::Flux, 0.1, 0.0));
    UniformFlow(1.0, 0.0);
    Fill2D(0, 0.0);
    SIMULATION.dt = 1.0;
    TransportSolver2D::InitializeTransportSolver2D(&gs);
    double lo = 0.0, hi = 0.0;
    RunSteps(80, lo, hi);

    const int i = gs.Ny / 2;
    int jlo = 0, jhi = 0;
    FluidRangeX(i, jlo, jhi);
    const double x0 = gs.FacePos(MAC::X, jlo), L = gs.FacePos(MAC::X, jhi + 1) - x0;
    pe = 1.0 * L / 0.1;
    double sum = 0.0;
    for (int j = jlo; j <= jhi; j++)
    {
        const double xi = (gs.CentrePos(MAC::X, j) - x0) / L;
        const double exact = (std::exp(pe * xi) - 1.0) / (std::exp(pe) - 1.0);
        sum += (gs.GetS(0, i, j) - exact) * (gs.GetS(0, i, j) - exact);
    }
    SCENE.loaded = false;
    return std::sqrt(sum / (jhi - jlo + 1));
}

static double Pattern(int i, int j, int k) { return 1.0 + i * 100.0 + j * 7.0 + k * 0.25; }

int run(int, char **)
{
    std::printf("transport_check: registered scalars on the grid\n");

    std::printf("\n== 1. a scalar is the pressure's own indexing ==\n");
    StandUp();
    const int d = gs.RegisterScalar("d");
    check(d == 0 && gs.ScalarCount() == 1, "registering gives handle 0",
          "handle " + std::to_string(d));
    {
        // Write through SetS, read back through the pressure's index arithmetic. If the two
        // disagree the field is a different cell than it claims and every later comparison
        // is meaningless.
        for (int k = 0; k < gs.Nz; k++)
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                    gs.SetS(d, i, j, k, Pattern(i, j, k));
        long wrong = 0;
        for (int k = 0; k < gs.Nz; k++)
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                {
                    const long idx = i * ((long)gs.Nx * gs.Nz) + (long)j * gs.Nz + k;
                    if (gs.ScalarField(0)[idx] != Pattern(i, j, k)) wrong++;
                }
        check(wrong == 0, "GetS/SetS address the same cell the pressure does",
              std::to_string(wrong) + " of " + std::to_string((long)gs.Nx * gs.Ny * gs.Nz) + " wrong");
        check((long)gs.ScalarField(0).size() == (long)gs.Nx * gs.Ny * gs.Nz,
              "and it is exactly the size of the pressure field",
              std::to_string(gs.ScalarField(0).size()));
    }

    std::printf("\n== 2. names are the identity ==\n");
    {
        const int again = gs.RegisterScalar("d");
        check(again == d && gs.ScalarCount() == 1, "registering the same name twice is one field",
              "handle " + std::to_string(again) + ", count " + std::to_string(gs.ScalarCount()));
        const int t = gs.RegisterScalar("T");
        check(t == 1 && gs.ScalarCount() == 2, "a second name is a second field");
        check(gs.ScalarHandle("d") == 0 && gs.ScalarHandle("T") == 1,
              "and lookups come back by name");
        check(gs.ScalarHandle("nope") == -1, "an unregistered name is -1, not a crash");
    }

    std::printf("\n== 2b. the scheme is a property of the FIELD ==\n");
    {
        // Two schemes, chosen per field: k and omega want the bounded one, a density field
        // being rendered wants the accurate one. The default is the bounded one, so a field
        // that neglects to say cannot produce a negative value.
        MAC g;
        g.InitializeGrid(SIMULATION.domain, false);
        const int dd = g.RegisterScalar("d", TransportScheme::Central);
        const int kk = g.RegisterScalar("k", TransportScheme::Upwind);
        const int ww = g.RegisterScalar("w");            // no scheme given
        check(g.ScalarTransportOf(dd).scheme == TransportScheme::Central &&
              g.ScalarTransportOf(kk).scheme == TransportScheme::Upwind,
              "each field keeps the scheme it asked for",
              std::string(TransportSchemeName(g.ScalarTransportOf(dd).scheme)) + " / " +
              TransportSchemeName(g.ScalarTransportOf(kk).scheme));
        check(g.ScalarTransportOf(ww).scheme == TransportScheme::Upwind,
              "and the default is upwind -- the one that cannot go negative",
              TransportSchemeName(g.ScalarTransportOf(ww).scheme));

        // Asking for the same name with a different scheme is two parts of the code
        // disagreeing. The first registration wins, so the answer does not depend on which
        // caller ran last, and it is reported at registration rather than once per copy.
        const long before = MAC::scalarSchemeConflicts;
        const int again = g.RegisterScalar("d", TransportScheme::Upwind);
        check(again == dd && g.ScalarTransportOf(dd).scheme == TransportScheme::Central,
              "a conflicting re-registration keeps the FIRST scheme",
              TransportSchemeName(g.ScalarTransportOf(dd).scheme));
        check(MAC::scalarSchemeConflicts == before + 1,
              "and says so", std::to_string(MAC::scalarSchemeConflicts - before) + " conflict");
        check(g.ScalarCount() == 3, "without adding a field",
              std::to_string(g.ScalarCount()));
    }

    std::printf("\n== 3. CopyGrid: handles are the fast path, names are the check ==\n");
    {
        // First the ordinary case -- both grids registered from the same place, so the handles
        // agree and the name is one comparison. Nothing should be reported.
        {
            MAC a, b;
            a.InitializeGrid(SIMULATION.domain, false);
            b.InitializeGrid(SIMULATION.domain, false);
            a.RegisterScalar("d"); a.RegisterScalar("T");
            b.RegisterScalar("d"); b.RegisterScalar("T");
            for (int k = 0; k < b.Nz; k++)
                for (int i = 0; i < b.Ny; i++)
                    for (int j = 0; j < b.Nx; j++) b.SetS(0, i, j, k, Pattern(i, j, k));
            const long before = MAC::scalarOrderMismatches;
            a.CopyGrid(b);
            check(MAC::scalarOrderMismatches == before,
                  "matching orders take the handle path and report nothing",
                  std::to_string(MAC::scalarOrderMismatches - before) + " mismatches");
            check(a.GetS(0, 1, 2, 3) == Pattern(1, 2, 3), "and the copy is right");
        }

        // Now register in the OPPOSITE order on the destination. Copying by position would put
        // the density into the temperature, so this must be caught -- and SAID, because it
        // means two parts of the code disagree about what the grid holds.
        ga.RegisterScalar("T");
        ga.RegisterScalar("d");
        const long beforeMismatch = MAC::scalarOrderMismatches;
        check(ga.ScalarHandle("d") == 1 && gs.ScalarHandle("d") == 0,
              "the two grids hold 'd' at different handles, on purpose");
        for (int k = 0; k < gs.Nz; k++)
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                    gs.SetS(gs.ScalarHandle("T"), i, j, k, -Pattern(i, j, k));

        ga.CopyGrid(gs);
        long wrongD = 0, wrongT = 0;
        for (int k = 0; k < gs.Nz; k++)
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                {
                    if (ga.GetS(ga.ScalarHandle("d"), i, j, k) != Pattern(i, j, k)) wrongD++;
                    if (ga.GetS(ga.ScalarHandle("T"), i, j, k) != -Pattern(i, j, k)) wrongT++;
                }
        check(wrongD == 0 && wrongT == 0, "every cell of both fields arrives, in the right field",
              std::to_string(wrongD) + " density cells and " + std::to_string(wrongT) +
              " temperature cells wrong");
        check(MAC::scalarOrderMismatches > beforeMismatch,
              "and the disagreement was REPORTED, not silently worked around",
              std::to_string(MAC::scalarOrderMismatches - beforeMismatch) + " mismatches counted");
    }

    std::printf("\n== 4. a rebuilt grid brings its fields back at the new size ==\n");
    {
        // InitializeGrid is what a restart at a different resolution runs. A field registered
        // before it must not come back at the old length -- that is an Eigen bounds assertion
        // on the first write, a long way from the cause.
        MAC g;
        g.InitializeGrid(SIMULATION.domain, false);
        g.RegisterScalar("d");
        const long before = g.ScalarField(0).size();
        SIMULATION.GRID_SIZE = 24;
        g.InitializeGrid(SIMULATION.domain, false);
        const long after = g.ScalarField(0).size();
        check(g.ScalarCount() == 1 && after == (long)g.Nx * g.Ny * g.Nz && after != before,
              "the field survives the rebuild and is resized with the grid",
              std::to_string(before) + " -> " + std::to_string(after));
        SIMULATION.GRID_SIZE = 16;
    }

    std::printf("\n== 5. the checkpoint round trips the scalars ==\n");
    {
        // A checkpoint is taken at a step boundary, where GRID_SOL == GRID_ANT, so it stores GRID_ANT's
        // fields and Load copies them into GRID_SOL. GRID_SOL is filled with something else here to show
        // that it is overwritten, not left behind.
        StandUp();
        gs.RegisterScalar("d"); ga.RegisterScalar("d");
        for (int k = 0; k < gs.Nz; k++)
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                {
                    ga.SetS(0, i, j, k, Pattern(i, j, k));
                    gs.SetS(0, i, j, k, -1.0);
                }
        std::string err;
        check(Checkpoint::Save("scalar.ckpt", 1.25, 7, err), "saved", err);

        StandUp();                       // a fresh pair of grids, as a restart gives
        gs.RegisterScalar("d"); ga.RegisterScalar("d");
        double t = 0; uint64_t it = 0;
        check(Checkpoint::Load("scalar.ckpt", t, it, err), "loaded", err);
        long wrongSol = 0, wrongAnt = 0;
        for (int k = 0; k < gs.Nz; k++)
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                {
                    if (ga.GetS(0, i, j, k) != Pattern(i, j, k)) wrongAnt++;
                    if (gs.GetS(0, i, j, k) != Pattern(i, j, k)) wrongSol++;
                }
        check(wrongSol == 0 && wrongAnt == 0,
              "GRID_ANT's scalar comes back bit-identical, and GRID_SOL receives the same",
              std::to_string(wrongAnt) + " / " + std::to_string(wrongSol) + " cells wrong");
    }

    std::printf("\n== 6. a checkpoint whose fields do not match is refused ==\n");
    {
        StandUp();                       // no scalars registered at all this time
        double t = 0; uint64_t it = 0;
        std::string err;
        const bool refused = !Checkpoint::Load("scalar.ckpt", t, it, err);
        check(refused && err.find("scalar") != std::string::npos,
              "a run with no scalars refuses a file that has one", err);

        StandUp();
        gs.RegisterScalar("smoke"); ga.RegisterScalar("smoke");
        std::string err2;
        const bool refusedName = !Checkpoint::Load("scalar.ckpt", t, it, err2);
        // The message names the field the FILE holds, not the one the run registered, which
        // is the more useful of the two: it says what the checkpoint actually contains.
        check(refusedName && err2.find("'d'") != std::string::npos,
              "and a run with the WRONG name refuses it too, naming the file's field", err2);
        std::remove("scalar.ckpt");
    }

    // ---------------------------------------------------------------------
    // Stage 2: the solver itself. 2D, advection only.
    // ---------------------------------------------------------------------
    std::printf("\n== 7. advection: a step stays monotone under upwind ==\n");
    {
        StandUp2D(64);
        const int h = gs.RegisterScalar("d", TransportScheme::Upwind);
        ga.RegisterScalar("d", TransportScheme::Upwind);
        UniformFlow(1.0, 0.0);
        StepProfile(h);
        TransportSolver2D::InitializeTransportSolver2D(&gs);

        double lo = 0.0, hi = 0.0;
        RunSteps(120, lo, hi);
        std::printf("      after 120 steps: min %.6f  max %.6f\n", lo, hi);
        check(lo > -1e-12, "never undershoots below zero", fmt("min %.3e", lo));
        check(hi < 1.0 + 1e-12, "never overshoots above the initial maximum", fmt("max %.9f", hi));
    }

    std::printf("\n== 8. central differencing does overshoot, which is why both exist ==\n");
    {
        StandUp2D(64);
        const int h = gs.RegisterScalar("d", TransportScheme::Central);
        ga.RegisterScalar("d", TransportScheme::Central);
        UniformFlow(1.0, 0.0);
        StepProfile(h);
        TransportSolver2D::InitializeTransportSolver2D(&gs);

        double lo = 0.0, hi = 0.0;
        RunSteps(120, lo, hi);
        std::printf("      after 120 steps: min %.6f  max %.6f\n", lo, hi);
        check(lo < -1e-6 || hi > 1.0 + 1e-6,
              "central leaves the initial range on a sharp front",
              fmt("min %.4f, max %.4f", lo, hi));
    }

    std::printf("\n== 9. a closed box conserves the total exactly ==\n");
    {
        // A cellular flow from the stream function sin(pi x) sin(pi y): divergence free, and
        // its normal component vanishes on all four walls, so nothing can leave. The flux form
        // telescopes, so the total is conserved whatever the field does in between.
        StandUp2D(48);
        const int h = gs.RegisterScalar("d", TransportScheme::Upwind);
        ga.RegisterScalar("d", TransportScheme::Upwind);
        CellularFlow();
        BlobProfile(h);
        TransportSolver2D::InitializeTransportSolver2D(&gs);

        const double before = TotalMass(h);
        double lo = 0.0, hi = 0.0;
        RunSteps(200, lo, hi);
        const double after = TotalMass(h);
        std::printf("      total %.14f -> %.14f\n", before, after);
        check(std::fabs(after - before) < 1e-10 * (std::fabs(before) + 1.0),
              "mass is conserved to round-off", fmt("relative change %.3e",
              std::fabs(after - before) / (std::fabs(before) + 1e-300)));
        check(lo > -1e-12, "and the field never goes negative", fmt("min %.3e", lo));
    }

    std::printf("\n== 10. an inflow face pins the scalar, a wall does not ==\n");
    {
        {
            std::ofstream f("transport.scene");
            f << "sampling center\n"
                 "scalar d scheme=upwind\n"
                 "boundary xmin inflow u=\"1\" d=\"1\"\n"
                 "boundary xmax pressure_outlet\n"
                 "boundary ymin noslip\nboundary ymax noslip\n";
        }
        StandUp2D(48);
        if (!SCENE.Load("transport.scene"))
        { check(false, "scene loads", SCENE.error); }
        else
        {
            check(SCENE.ScalarIndex("d") == 0, "the scene declares the scalar");
            const int h = gs.RegisterScalar("d", TransportScheme::Upwind);
            ga.RegisterScalar("d", TransportScheme::Upwind);
            UniformFlow(1.0, 0.0);
            for (int i = 0; i < ga.Ny; i++)
                for (int j = 0; j < ga.Nx; j++) ga.SetS(h, i, j, 0.0);
            TransportSolver2D::InitializeTransportSolver2D(&gs);

            // Empty domain, an inflow that pins d = 1. The scalar has to arrive, stay
            // bounded, and reach the far side at about the flow speed.
            double lo = 0.0, hi = 0.0;
            RunSteps(100, lo, hi);           // 100 * 0.002 * 1.0 = 0.2 m of travel
            const double frontAt02 = FrontPosition(h, 0.5);
            RunSteps(150, lo, hi);           // 0.5 m in total
            const double frontAt05 = FrontPosition(h, 0.5);
            std::printf("      front at t=0.2: x = %.3f   at t=0.5: x = %.3f\n",
                        frontAt02, frontAt05);

            check(hi > 0.9 && hi <= 1.0 + 1e-12,
                  "the inflow value arrives and is not exceeded", fmt("max %.6f", hi));
            check(lo > -1e-12, "and nothing goes negative", fmt("min %.3e", lo));
            check(frontAt02 > 0.10 && frontAt02 < 0.30 &&
                  frontAt05 > 0.35 && frontAt05 < 0.65,
                  "the front travels at about the flow speed",
                  fmt("%.3f then %.3f, against 0.2 and 0.5", frontAt02, frontAt05));

            // A wall pins nothing, so the y walls must stay zero-flux: no scalar should have
            // leaked across them, and the profile is uniform in y.
            double spread = 0.0;
            const int jm = gs.Nx / 4;
            double c0 = gs.GetS(h, 1, jm);
            for (int i = 1; i < gs.Ny - 1; i++)
                spread = (std::max)(spread, std::fabs(gs.GetS(h, i, jm) - c0));
            check(spread < 1e-12, "and a wall leaves the field uniform along it",
                  fmt("varies by %.3e across the channel", spread));

        }
        std::remove("transport.scene");
    }

    std::printf("\n== 11. a scalar's prescribed value can be clamped at zero, per face ==\n");
    {
        {
            std::ofstream f("clamp.scene");
            f << "sampling center\n"
                 "scalar d scheme=upwind\n"
                 "boundary xmin inflow u=\"1\" d=\"1 - 4*y\" d_clamp_negative=1\n"
                 "boundary xmax pressure_outlet d=\"1 - 4*y\"\n"
                 "boundary ymin noslip\nboundary ymax noslip\n";
        }
        StandUp2D(48);
        if (!SCENE.Load("clamp.scene"))
        { check(false, "clamp scene loads", SCENE.error); }
        else
        {
            const double y = 0.75;   // 1 - 4y = -2 here
            const double yc = 0.10;  // 1 - 4y = 0.6, below the root
            // This section follows one whose scene pinned d = 1 on the same face, so it also
            // guards that a reload does not keep the previous scene's scalar values.
            double lo = 0.0, hi = 0.0, below = 0.0;
            const bool pinnedLo = SCENE.ScalarDirichletAt("d", 0.0, y, 0.0, 0.0, lo);
            const bool pinnedHi = SCENE.ScalarDirichletAt("d", 1.0, y, 0.0, 0.0, hi);
            SCENE.ScalarDirichletAt("d", 0.0, yc, 0.0, 0.0, below);
            check(pinnedLo && lo == 0.0, "the clamped face returns zero past the root",
                  fmt("%.3f", lo));
            check(pinnedHi && hi < -1.9, "an unclamped face keeps the negative value",
                  fmt("%.3f", hi));
            check(std::fabs(below - 0.6) < 1e-12, "and a positive value is untouched",
                  fmt("%.6f", below));
        }
        std::remove("clamp.scene");
    }

    std::printf("\n== 12. the ADI write-back copies only the flow fields ==\n");
    {
        // ADI2D's sweep grids carry no scalars. Its write-back used to be a whole-grid
        // CopyGrid, which asked Y_STEP_SOL for 'd' on every step and tripped MatchScalar's
        // warning. A real ADI step with a scalar declared must now report no mismatch.
        {
            std::ofstream f("adicopy.scene");
            f << "sampling center\n"
                 "scalar d scheme=upwind\n"
                 "boundary xmin inflow u=\"1\" d=\"1\"\n"
                 "boundary xmax pressure_outlet\n"
                 "boundary ymin noslip\nboundary ymax noslip\n";
        }
        StandUp2D(32);
        SIMULATION.EPS = 1e-2; SIMULATION.RHO = 1.0;
        if (!SCENE.Load("adicopy.scene"))
        { check(false, "scene loads", SCENE.error); }
        else
        {
            gs.RegisterScalar("d", TransportScheme::Upwind);
            ga.RegisterScalar("d", TransportScheme::Upwind);
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++) { gs.SetS(0, i, j, 0.25); ga.SetS(0, i, j, 0.25); }
            ADI2D::InitializeADI2D(&gs, SIMULATION.dt, SCENE_VELOCITY_2D,  SCENE_PRESSURE_2D);
            // As the manager does: the sweeps read the turbulent viscosity field even when the
            // model is off, so it has to exist (zeroed).

            const long before = MAC::scalarOrderMismatches;
            for (int step = 0; step < 3; step++)
                ADI2D::SolveADIStep(&ga, &gs, nullptr, SIMULATION.dt * step);
            check(MAC::scalarOrderMismatches == before,
                  "an ADI step with a scalar declared reports no mismatch",
                  std::to_string(MAC::scalarOrderMismatches - before) + " new");

            double worst = 0.0;
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                    worst = (std::max)(worst, std::fabs(gs.GetS(0, i, j) - 0.25));
            check(worst == 0.0, "and leaves the scalar where the transport stage will find it",
                  fmt("moved by %.3e", worst));

            // CopyGrid is still everything.
            ga.SetS(0, 3, 3, 0.75);
            gs.CopyGrid(ga);
            check(gs.GetS(0, 3, 3) == 0.75 && gs.GetU(2, 2) == ga.GetU(2, 2),
                  "CopyGrid still copies the flow fields and the scalars");
        }
        std::remove("adicopy.scene");
    }

    std::printf("\n== 13. 3D: every z-slice of a z-uniform run IS the 2D run ==\n");
    {
        // With w = 0 the z sweep is the identity to the bit (diagonal 1, no coupling), so the 3D
        // solver must reproduce TransportSolver2D slice by slice -- including the inflow pin and the
        // closed walls, since the scene answers the same at every z.
        {
            std::ofstream f("t3d.scene");
            f << "sampling center\n"
                 "scalar d scheme=upwind\n"
                 "boundary xmin inflow u=\"1\" d=\"1\"\n"
                 "boundary xmax pressure_outlet\n"
                 "boundary ymin noslip\nboundary ymax noslip\n";
        }
        const TransportScheme schemes[2] = {TransportScheme::Upwind, TransportScheme::Central};
        for (TransportScheme scheme : schemes)
        {
            const int N = 24, steps = 60;
            std::vector<double> ref;
            StandUp2D(N);
            if (!SCENE.Load("t3d.scene")) { check(false, "scene loads", SCENE.error); break; }
            gs.RegisterScalar("d", scheme); ga.RegisterScalar("d", scheme);
            UniformFlow(1.0, 0.3);
            BlobProfile(0);
            TransportSolver2D::InitializeTransportSolver2D(&gs);
            double lo = 0.0, hi = 0.0;
            RunSteps(steps, lo, hi);
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++) ref.push_back(gs.GetS(0, i, j));

            StandUp(N);
            gs.RegisterScalar("d", scheme); ga.RegisterScalar("d", scheme);
            UniformFlow3D(1.0, 0.3, 0.0);
            for (int i = 0; i < ga.Ny; i++)
                for (int j = 0; j < ga.Nx; j++)
                    for (int k = 0; k < ga.Nz; k++)
                    {
                        const double dx = ga.CentrePos(MAC::X, j) - 0.3, dy = ga.CentrePos(MAC::Y, i) - 0.5;
                        ga.SetS(0, i, j, k, (dx * dx + dy * dy < 0.15 * 0.15) ? 1.0 : 0.0);
                    }
            TransportSolver3D::InitializeTransportSolver3D(&gs);
            RunSteps3D(steps, lo, hi);
            double worst = 0.0;
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                    for (int k = 0; k < gs.Nz; k++)
                        worst = (std::max)(worst, std::fabs(gs.GetS(0, i, j, k) - ref[(size_t)(i * gs.Nx + j)]));
            check(worst == 0.0, std::string(SchemeLabel(scheme)) + ": 3D slices equal the 2D field, bit for bit",
                  fmt("worst %.3e", worst));
        }
        std::remove("t3d.scene");
        SCENE.loaded = false;
    }

    std::printf("\n== 14. 3D: x, y and z are the same solver ==\n");
    {
        // A front carried along each axis in turn. Only one velocity component is non-zero, so
        // the other two sweeps are the identity and the three runs are exact permutations.
        const TransportScheme schemes[2] = {TransportScheme::Upwind, TransportScheme::Central};
        for (TransportScheme scheme : schemes)
        {
            const int N = 20, steps = 80;
            std::vector<double> run[3];
            double lo[3], hi[3];
            for (int axis = 0; axis < 3; axis++)
            {
                StandUp(N);
                gs.RegisterScalar("d", scheme); ga.RegisterScalar("d", scheme);
                UniformFlow3D(axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0, axis == 2 ? 1.0 : 0.0);
                StepProfile3D(0, axis);
                TransportSolver3D::InitializeTransportSolver3D(&gs);
                RunSteps3D(steps, lo[axis], hi[axis]);
                run[axis].assign(gs.ScalarField(0).data(), gs.ScalarField(0).data() + gs.ScalarField(0).size());
            }
            double worstY = 0.0, worstZ = 0.0;
            const int M = gs.Nx;   // N interior cells plus the ring on each side
            for (int i = 0; i < M; i++)
                for (int j = 0; j < M; j++)
                    for (int k = 0; k < M; k++)
                    {
                        const double x = run[0][(size_t)((i * M + j) * M + k)];
                        worstY = (std::max)(worstY, std::fabs(x - run[1][(size_t)((j * M + i) * M + k)]));
                        worstZ = (std::max)(worstZ, std::fabs(x - run[2][(size_t)((i * M + k) * M + j)]));
                    }
            std::printf("      %s: min %.6f max %.6f (x)\n", SchemeLabel(scheme), lo[0], hi[0]);
            check(worstY == 0.0 && worstZ == 0.0,
                  std::string(SchemeLabel(scheme)) + ": the y and z runs are the x run, permuted, bit for bit",
                  fmt("worst %.3e (y), %.3e (z)", worstY, worstZ));
            if (scheme == TransportScheme::Upwind)
                check(lo[2] > -1e-12 && hi[2] < 1.0 + 1e-12, "upwind stays in [0, 1] along z",
                      fmt("min %.3e, max %.9f", lo[2], hi[2]));
            else
                check(lo[2] < -1e-6 || hi[2] > 1.0 + 1e-6, "central leaves [0, 1] along z, as in 2D",
                      fmt("min %.4f, max %.4f", lo[2], hi[2]));
        }
    }

    std::printf("\n== 15. 3D: a closed box conserves the total exactly ==\n");
    {
        StandUp(28);
        gs.RegisterScalar("d", TransportScheme::Upwind); ga.RegisterScalar("d", TransportScheme::Upwind);
        CellularFlow3D();
        for (int i = 0; i < ga.Ny; i++)
            for (int j = 0; j < ga.Nx; j++)
                for (int k = 0; k < ga.Nz; k++)
                {
                    const double dx = ga.CentrePos(MAC::X, j) - 0.3, dy = ga.CentrePos(MAC::Y, i) - 0.5, dz = ga.CentrePos(MAC::Z, k) - 0.6;
                    ga.SetS(0, i, j, k, (dx * dx + dy * dy + dz * dz < 0.15 * 0.15) ? 1.0 : 0.0);
                }
        TransportSolver3D::InitializeTransportSolver3D(&gs);
        const double before = TotalMass3D(0);
        double lo = 0.0, hi = 0.0;
        RunSteps3D(150, lo, hi);
        const double after = TotalMass3D(0);
        std::printf("      total %.14f -> %.14f\n", before, after);
        check(std::fabs(after - before) < 1e-10 * (std::fabs(before) + 1.0), "mass is conserved to round-off",
              fmt("relative change %.3e", std::fabs(after - before) / (std::fabs(before) + 1e-300)));
        check(lo > -1e-12, "and the field never goes negative", fmt("min %.3e", lo));
    }

    std::printf("\n== 16. 3D: a zmin inflow pins the scalar and the front travels along z ==\n");
    {
        {
            std::ofstream f("t3dz.scene");
            f << "sampling center\n"
                 "scalar d scheme=upwind\n"
                 "boundary zmin inflow w=\"1\" d=\"1\"\n"
                 "boundary zmax pressure_outlet\n"
                 "boundary xmin noslip\nboundary xmax noslip\nboundary ymin noslip\nboundary ymax noslip\n";
        }
        StandUp(32);
        if (!SCENE.Load("t3dz.scene")) { check(false, "scene loads", SCENE.error); }
        else
        {
            gs.RegisterScalar("d", TransportScheme::Upwind); ga.RegisterScalar("d", TransportScheme::Upwind);
            UniformFlow3D(0.0, 0.0, 1.0);
            TransportSolver3D::InitializeTransportSolver3D(&gs);
            double lo = 0.0, hi = 0.0;
            const int ic = gs.Ny / 2, jc = gs.Nx / 2;
            RunSteps3D(100, lo, hi);
            double front02 = 0.0;
            for (int k = 0; k < gs.Nz; k++) if (gs.GetS(0, ic, jc, k) >= 0.5) front02 = gs.CentrePos(MAC::Z, k);
            RunSteps3D(150, lo, hi);
            double front05 = 0.0;
            for (int k = 0; k < gs.Nz; k++) if (gs.GetS(0, ic, jc, k) >= 0.5) front05 = gs.CentrePos(MAC::Z, k);
            std::printf("      front at t=0.2: z = %.3f   at t=0.5: z = %.3f\n", front02, front05);
            check(hi > 0.9 && hi <= 1.0 + 1e-12, "the inflow value arrives and is not exceeded", fmt("max %.6f", hi));
            check(lo > -1e-12, "and nothing goes negative", fmt("min %.3e", lo));
            check(front02 > 0.10 && front02 < 0.30 && front05 > 0.35 && front05 < 0.65,
                  "the front travels at about the flow speed", fmt("%.3f then %.3f, against 0.2 and 0.5", front02, front05));
            double spread = 0.0;
            const int km = gs.Nz / 4;
            const double c0 = gs.GetS(0, 1, 1, km);
            for (int i = 1; i < gs.Ny - 1; i++)
                for (int j = 1; j < gs.Nx - 1; j++)
                    spread = (std::max)(spread, std::fabs(gs.GetS(0, i, j, km) - c0));
            check(spread < 1e-12, "the side walls leave the field uniform across the channel", fmt("varies by %.3e", spread));
        }
        std::remove("t3dz.scene");
    }

    std::printf("\n== 17. 3D: the ADI write-back copies only the flow fields ==\n");
    {
        {
            std::ofstream f("adicopy3d.scene");
            f << "sampling center\n"
                 "scalar d scheme=upwind\n"
                 "boundary xmin inflow u=\"1\" d=\"1\"\n"
                 "boundary xmax pressure_outlet\n"
                 "boundary ymin noslip\nboundary ymax noslip\nboundary zmin noslip\nboundary zmax noslip\n";
        }
        StandUp(16);
        SIMULATION.EPS = 1e-2;
        if (!SCENE.Load("adicopy3d.scene")) { check(false, "scene loads", SCENE.error); }
        else
        {
            gs.RegisterScalar("d", TransportScheme::Upwind); ga.RegisterScalar("d", TransportScheme::Upwind);
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                    for (int k = 0; k < gs.Nz; k++) { gs.SetS(0, i, j, k, 0.25); ga.SetS(0, i, j, k, 0.25); }
            ADI3D::InitializeADI(&gs, SIMULATION.dt, SCENE_VELOCITY,  SCENE_PRESSURE);
            const long before = MAC::scalarOrderMismatches;
            for (int step = 0; step < 3; step++) ADI3D::SolveADIStep(&ga, &gs, nullptr, SIMULATION.dt * step);
            check(MAC::scalarOrderMismatches == before, "an ADI3D step with a scalar declared reports no mismatch",
                  std::to_string(MAC::scalarOrderMismatches - before) + " new");
            double worst = 0.0;
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                    for (int k = 0; k < gs.Nz; k++) worst = (std::max)(worst, std::fabs(gs.GetS(0, i, j, k) - 0.25));
            check(worst == 0.0, "and leaves the scalar where the transport stage will find it", fmt("moved by %.3e", worst));
        }
        std::remove("adicopy3d.scene");
    }

    // ---------------------------------------------------------------------
    // Stage 5: the two forms, diffusion, reaction and the source.
    // ---------------------------------------------------------------------
    std::printf("\n== 18. the advective form keeps a uniform field uniform; the flux form does not ==\n");
    SCENE.loaded = false;   // section 17 leaves its scene loaded
    {
        // The cellular flow is discretely divergence free, but each sweep sees only du/dx or dv/dy.
        // Flux form: the split's (1+a)(1-a) != 1 leaves peaks. Advective: the row sums are exactly one.
        double dev[2] = {0.0, 0.0};
        for (int f = 0; f < 2; f++)
        {
            StandUp2D(48);
            SIMULATION.dt = 4e-3;
            const TransportForm form = (f == 0) ? TransportForm::Flux : TransportForm::Advective;
            RegisterBoth("d", Transport(TransportScheme::Upwind, form, 0.0, 0.0));
            CellularFlow();
            Fill2D(0, 1.0);
            TransportSolver2D::InitializeTransportSolver2D(&gs);
            double lo = 0.0, hi = 0.0;
            RunSteps(250, lo, hi);
            dev[f] = MaxDeviationFluid2D(0, 1.0);
        }
        std::printf("      max |phi - 1| after 1 s: flux %.4e, advective %.4e\n", dev[0], dev[1]);
        check(dev[0] > 1e-3, "flux form: the split distorts a uniform field (the recorded overshoot)",
              fmt("%.3e", dev[0]));
        check(dev[1] < 1e-12, "advective form: it stays uniform to round-off", fmt("%.3e", dev[1]));

        StandUp2D(48);
        SIMULATION.dt = 4e-3;
        RegisterBoth("d", Transport(TransportScheme::Upwind, TransportForm::Advective, 0.0, 0.0));
        CellularFlow();
        BlobProfile(0);
        TransportSolver2D::InitializeTransportSolver2D(&gs);
        const double before = TotalMass(0);
        double lo = 0.0, hi = 0.0;
        RunSteps(250, lo, hi);
        const double drift = std::fabs(TotalMass(0) - before) / before;
        std::printf("      advective blob: min %.3e max %.9f, total drift %.3e\n", lo, hi, drift);
        check(lo > -1e-12 && hi < 1.0 + 1e-12, "advective upwind stays inside [0, 1]", fmt("min %.3e, max %.9f", lo, hi));
        check(drift > 1e-10, "and pays for it in the total, which is no longer exact", fmt("relative %.3e", drift));

        // In a uniform flow du/dx = 0, so the two forms are the same row to the bit.
        std::vector<double> ref;
        for (int f = 0; f < 2; f++)
        {
            StandUp2D(32);
            const TransportForm form = (f == 0) ? TransportForm::Flux : TransportForm::Advective;
            RegisterBoth("d", Transport(TransportScheme::Central, form, 0.0, 0.0));
            UniformFlow(1.0, 0.4);
            BlobProfile(0);
            TransportSolver2D::InitializeTransportSolver2D(&gs);
            RunSteps(80, lo, hi);
            if (f == 0) ref.assign(gs.ScalarField(0).data(), gs.ScalarField(0).data() + gs.ScalarField(0).size());
            else
            {
                double worst = 0.0;
                for (size_t q = 0; q < ref.size(); q++) worst = (std::max)(worst, std::fabs(ref[q] - gs.ScalarField(0)[(long)q]));
                check(worst == 0.0, "in a uniform flow the two forms are bit for bit the same", fmt("worst %.3e", worst));
            }
        }
    }

    std::printf("\n== 19. diffusion: a cosine mode decays at its exact discrete rate ==\n");
    {
        // Zero-flux walls on the faces, u = 0. cos(pi (x - x0)/L) sampled at the cell centres is an exact
        // eigenvector of the cell-centred Laplacian with eigenvalue (4 eps/h^2) sin^2(pi h/(2L)), so each
        // implicit sweep divides the amplitude by 1 + dt*lambda -- exactly, to round-off. In 2D the mode
        // cos x * cos y decays by the product of the two sweeps' factors, which is the LOD split itself.
        const double eps = 0.05, A = 0.3;
        const int N = 32, steps = 200;
        StandUp2D(N);
        RegisterBoth("d", Transport(TransportScheme::Upwind, TransportForm::Flux, eps, 0.0));
        UniformFlow(0.0, 0.0);
        const int im = gs.Ny / 2, jm = gs.Nx / 2;
        int jlo = 0, jhi = 0, ilo = 0, ihi = 0;
        FluidRangeX(im, jlo, jhi);
        FluidRangeY(jm, ilo, ihi);
        const double x0 = gs.FacePos(MAC::X, jlo), Lx = gs.FacePos(MAC::X, jhi + 1) - x0, y0 = gs.FacePos(MAC::Y, ilo), Ly = gs.FacePos(MAC::Y, ihi + 1) - y0;
        for (int i = 0; i < ga.Ny; i++)
            for (int j = 0; j < ga.Nx; j++)
                ga.SetS(0, i, j, 1.0 + A * std::cos(M_PI * (ga.CentrePos(MAC::X, j) - x0) / Lx) * std::cos(M_PI * (ga.CentrePos(MAC::Y, i) - y0) / Ly));
        TransportSolver2D::InitializeTransportSolver2D(&gs);
        const double before = TotalMass(0);
        double lo = 0.0, hi = 0.0;
        RunSteps(steps, lo, hi);
        const double h = gs.dh, dt = SIMULATION.dt;
        const double lx = 4.0 * eps / (h * h) * std::pow(std::sin(M_PI * h / (2.0 * Lx)), 2);
        const double ly = 4.0 * eps / (h * h) * std::pow(std::sin(M_PI * h / (2.0 * Ly)), 2);
        const double amp = A * std::pow(1.0 / (1.0 + dt * lx), steps) * std::pow(1.0 / (1.0 + dt * ly), steps);
        double worst = 0.0;
        for (int i = ilo; i <= ihi; i++)
            for (int j = jlo; j <= jhi; j++)
            {
                const double exact = 1.0 + amp * std::cos(M_PI * (gs.CentrePos(MAC::X, j) - x0) / Lx) * std::cos(M_PI * (gs.CentrePos(MAC::Y, i) - y0) / Ly);
                worst = (std::max)(worst, std::fabs(gs.GetS(0, i, j) - exact));
            }
        const double cont = A * std::exp(-eps * M_PI * M_PI * (1.0 / (Lx * Lx) + 1.0 / (Ly * Ly)) * dt * steps);
        const double shape = std::cos(M_PI * (gs.CentrePos(MAC::X, jlo) - x0) / Lx) * std::cos(M_PI * (gs.CentrePos(MAC::Y, ilo) - y0) / Ly);
        std::printf("      amplitude %.10f (discrete exact %.10f, continuum %.10f)\n",
                    (gs.GetS(0, ilo, jlo) - 1.0) / shape, amp, cont);
        check(worst < 1e-12, "the field is the exact discrete decay, to round-off", fmt("worst %.3e", worst));
        check(std::fabs(TotalMass(0) - before) < 1e-12 * before, "zero-flux walls conserve the total",
              fmt("relative %.3e", std::fabs(TotalMass(0) - before) / before));
    }

    std::printf("\n== 20. diffusion: Dirichlet walls give the exact linear profile, uniform and stretched ==\n");
    {
        // Finite volume with the boundary value half a cell from the first centre is exact for a linear
        // profile, on any spacing. On a stretched grid that pins the centre-to-centre pairing (DxFace).
        const char *text = "sampling center\n"
                           "scalar T epsilon=0.1\n"
                           "boundary ymin noslip\nboundary ymax noslip\n"
                           "boundary xmin noslip T=\"1\"\n"
                           "boundary xmax noslip T=\"0\"\n";
        for (int stretched = 0; stretched < 2; stretched++)
        {
            StandUp2D(32);
            if (stretched)
            {
                SIMULATION.grid.nonUniform = true;
                GridRefinement r; r.active = true; r.start = 0.2; r.end = 0.5; r.factor = 3; r.mode = REFINE_SYMMETRIC;
                SIMULATION.grid.refineX.assign(1, r);
                gs.InitializeGrid(SIMULATION.domain, true);
                ga.InitializeGrid(SIMULATION.domain, true);
                SIMULATION.Nx = gs.Nx;
            }
            MakeRing2D();
            if (!SCENE.LoadText(text)) { check(false, "scene loads", SCENE.error); break; }
            RegisterBoth("T", Transport(TransportScheme::Upwind, TransportForm::Flux, 0.1, 0.0));
            UniformFlow(0.0, 0.0);
            Fill2D(0, 0.5);
            SIMULATION.dt = 5.0;
            TransportSolver2D::InitializeTransportSolver2D(&gs);
            double lo = 0.0, hi = 0.0;
            RunSteps(200, lo, hi);
            int jlo = 0, jhi = 0, ilo = 0, ihi = 0;
            FluidRangeX(gs.Ny / 2, jlo, jhi);
            FluidRangeY(gs.Nx / 2, ilo, ihi);
            const double x0 = gs.FacePos(MAC::X, jlo), L = gs.FacePos(MAC::X, jhi + 1) - x0;
            double worst = 0.0;
            for (int i = ilo; i <= ihi; i++)
                for (int j = jlo; j <= jhi; j++)
                    worst = (std::max)(worst, std::fabs(gs.GetS(0, i, j) - (1.0 - (gs.CentrePos(MAC::X, j) - x0) / L)));
            std::printf("      %s (%d cells across): worst %.3e\n", stretched ? "stretched" : "uniform", jhi - jlo + 1, worst);
            check(worst < 1e-10, std::string(stretched ? "stretched" : "uniform") + ": the steady profile is the exact line",
                  fmt("worst %.3e", worst));
            SIMULATION.grid.nonUniform = false;
            SIMULATION.grid.refineX.clear();
            SCENE.loaded = false;
        }
    }

    std::printf("\n== 21. reaction and source: the implicit factor is exact, and never goes negative ==\n");
    {
        // u = 0 and epsilon = 0, so both sweeps are the identity and only the reaction factor acts:
        // (1 + dt gamma) phi_{n+1} = phi_n + dt S, whose n-step solution is known in closed form.
        const char *text = "sampling center\n"
                           "scalar a gamma=5\n"
                           "scalar b gamma=5 source=\"3\"\n"
                           "scalar c source=\"x*y\"\n"
                           "scalar g gamma=-2\n"
                           "scalar s gamma=1e8\n"
                           "boundary xmin noslip\nboundary xmax noslip\nboundary ymin noslip\nboundary ymax noslip\n";
        StandUp2D(24);
        if (!SCENE.LoadText(text)) check(false, "scene loads", SCENE.error);
        else
        {
            const char *names[5] = {"a", "b", "c", "g", "s"};
            for (int q = 0; q < 5; q++)
            {
                const SceneDesc::Scene::ScalarSpec &sp = SCENE.scalars[(size_t)SCENE.ScalarIndex(names[q])];
                RegisterBoth(names[q], sp.transport);
            }
            UniformFlow(0.0, 0.0);
            Fill2D(0, 2.0); Fill2D(1, 0.0); Fill2D(2, 0.0); Fill2D(3, 1.0); Fill2D(4, 1.0);
            SIMULATION.dt = 0.01;
            TransportSolver2D::InitializeTransportSolver2D(&gs);
            const int steps = 100;
            for (int s = 0; s < steps; s++)
            {
                TransportSolver2D::SolveTransportStep(&ga, &gs, nullptr, 0.0);
                for (int h = 0; h < 5; h++)
                    for (int i = 0; i < gs.Ny; i++)
                        for (int j = 0; j < gs.Nx; j++) ga.SetS(h, i, j, gs.GetS(h, i, j));
            }
            const double dt = SIMULATION.dt;
            const double g = 1.0 / (1.0 + dt * 5.0);
            const double aExact = 2.0 * std::pow(g, steps);
            const double bExact = 0.6 * (1.0 - std::pow(g, steps));       // S/gamma * (1 - g^n)
            const double gExact = std::pow(1.0 + dt * 2.0, steps);
            const double devA = MaxDeviationFluid2D(0, aExact), devB = MaxDeviationFluid2D(1, bExact);
            const double devG = MaxDeviationFluid2D(3, gExact);
            double devC = 0.0, minS = 1e300;
            for (int i = 0; i < gs.Ny; i++)
                for (int j = 0; j < gs.Nx; j++)
                    if (gs.GetSolid(i, j) == FLUID_CELL)
                    {
                        devC = (std::max)(devC, std::fabs(gs.GetS(2, i, j) - steps * dt * gs.CentrePos(MAC::X, j) * gs.CentrePos(MAC::Y, i)));
                        minS = (std::min)(minS, gs.GetS(4, i, j));
                    }
            std::printf("      decay %.12f (exact %.12f), decay+source %.12f (exact %.12f)\n",
                        gs.GetS(0, gs.Ny / 2, gs.Nx / 2), aExact, gs.GetS(1, gs.Ny / 2, gs.Nx / 2), bExact);
            check(devA < 1e-14 * 2.0, "decay: phi = phi0 / (1 + dt gamma)^n", fmt("worst %.3e", devA));
            check(devB < 1e-13, "decay with a source: phi = (S/gamma)(1 - g^n)", fmt("worst %.3e", devB));
            check(devC < 1e-13, "a spatial source lands at the cell centres: phi = n dt x y", fmt("worst %.3e", devC));
            check(devG < 1e-12 * gExact, "negative gamma grows explicitly: phi = (1 + dt|gamma|)^n", fmt("worst %.3e", devG));
            check(minS >= 0.0, "gamma dt = 1e6 decays to a positive value, never below zero", fmt("min %.3e", minS));
            check(SCENE.ScalarSource(SCENE.ScalarIndex("a")) == nullptr && SCENE.ScalarSource(SCENE.ScalarIndex("c")) != nullptr,
                  "a scalar without `source` carries none");
        }

        // Refusals: a negative diffusivity, an unknown form, a source that does not parse.
        check(!SCENE.LoadText("scalar d epsilon=-1\n"), "a negative epsilon is refused", SCENE.error);
        check(!SCENE.LoadText("scalar d form=conservative\n"), "an unknown form is refused", SCENE.error);
        check(!SCENE.LoadText("scalar d source=\"x*(\"\n"), "a source that does not parse is refused", SCENE.error);
        SCENE.loaded = false;
    }

    std::printf("\n== 22. advection-diffusion: the steady exponential layer, and each scheme's order ==\n");
    {
        const int Ns[3] = {24, 48, 96};
        const TransportScheme schemes[2] = {TransportScheme::Central, TransportScheme::Upwind};
        for (TransportScheme scheme : schemes)
        {
            double e[3], pe = 0.0;
            for (int n = 0; n < 3; n++) e[n] = SteadyAdvectionDiffusionError(Ns[n], scheme, pe);
            const double o1 = std::log(e[0] / e[1]) / std::log(2.0), o2 = std::log(e[1] / e[2]) / std::log(2.0);
            std::printf("      %s, Pe %.2f: RMS %.3e %.3e %.3e, order %.2f %.2f\n", SchemeLabel(scheme), pe, e[0], e[1], e[2], o1, o2);
            if (scheme == TransportScheme::Central)
                check(o2 > 1.8, "central converges at second order", fmt("%.2f", o2));
            else
                check(o2 > 0.85 && o2 < 1.3, "upwind converges at first order", fmt("%.2f", o2));
        }
        SIMULATION.dt = 0.002;
    }

    std::printf("\n== 23. 3D: the new terms keep the z-slices equal to the 2D run ==\n");
    {
        // Same as section 13 with every Stage 5 term on. Diffusion makes the z sweep a real tridiagonal
        // solve on a z-uniform field, which returns the constant only to round-off -- so with epsilon > 0
        // the bound is 1e-13, while reaction, source and form alone stay bit for bit.
        const char *base = "sampling center\n"
                           "boundary xmin inflow u=\"1\" d=\"1\"\n"
                           "boundary xmax pressure_outlet\n"
                           "boundary ymin noslip\nboundary ymax noslip\n";
        const TransportForm forms[2] = {TransportForm::Flux, TransportForm::Advective};
        const double epsilons[2] = {0.0, 0.004};
        for (TransportForm form : forms)
            for (double eps : epsilons)
            {
                std::string text = std::string("scalar d gamma=0.7 source=\"0.2*y\" epsilon=") + std::to_string(eps) +
                                   (form == TransportForm::Advective ? " form=advective\n" : "\n") + base;
                const int N = 24, steps = 60;
                StandUp2D(N);
                MakeRing2D();
                if (!SCENE.LoadText(text)) { check(false, "scene loads", SCENE.error); break; }
                const ScalarTransport tr = SCENE.scalars[0].transport;
                RegisterBoth("d", tr);
                CellularFlow();
                BlobProfile(0);
                TransportSolver2D::InitializeTransportSolver2D(&gs);
                double lo = 0.0, hi = 0.0;
                RunSteps(steps, lo, hi);
                std::vector<double> ref;
                for (int i = 0; i < gs.Ny; i++)
                    for (int j = 0; j < gs.Nx; j++) ref.push_back(gs.GetS(0, i, j));

                StandUp(N);
                MakeRing3D();
                RegisterBoth("d", tr);
                const double pi = M_PI;
                UniformFlow3D(0.0, 0.0, 0.0);
                for (int i = 0; i < gs.Ny; i++)
                    for (int j = 0; j < gs.Nx + 1; j++)
                        for (int k = 0; k < gs.Nz; k++) gs.SetU(i, j, k, pi * std::sin(pi * gs.FacePos(MAC::X, j)) * std::cos(pi * gs.CentrePos(MAC::Y, i)));
                for (int i = 0; i < gs.Ny + 1; i++)
                    for (int j = 0; j < gs.Nx; j++)
                        for (int k = 0; k < gs.Nz; k++) gs.SetV(i, j, k, -pi * std::cos(pi * gs.CentrePos(MAC::X, j)) * std::sin(pi * gs.FacePos(MAC::Y, i)));
                for (int i = 0; i < ga.Ny; i++)
                    for (int j = 0; j < ga.Nx; j++)
                        for (int k = 0; k < ga.Nz; k++)
                        {
                            const double dx = ga.CentrePos(MAC::X, j) - 0.3, dy = ga.CentrePos(MAC::Y, i) - 0.5;
                            ga.SetS(0, i, j, k, (dx * dx + dy * dy < 0.15 * 0.15) ? 1.0 : 0.0);
                        }
                TransportSolver3D::InitializeTransportSolver3D(&gs);
                RunSteps3D(steps, lo, hi);
                double worst = 0.0;   // the k = 0 and k = Nz-1 slices are the ring, which no sweep touches
                for (int i = 0; i < gs.Ny; i++)
                    for (int j = 0; j < gs.Nx; j++)
                        for (int k = 1; k < gs.Nz - 1; k++)
                            worst = (std::max)(worst, std::fabs(gs.GetS(0, i, j, k) - ref[(size_t)(i * gs.Nx + j)]));
                const std::string label = std::string(TransportFormName(form)) + (eps > 0.0 ? ", diffusion" : "") +
                                          ", reaction, source";
                if (eps > 0.0) check(worst < 1e-13, label + ": 3D slices equal the 2D field to round-off", fmt("worst %.3e", worst));
                else           check(worst == 0.0, label + ": 3D slices equal the 2D field, bit for bit", fmt("worst %.3e", worst));
            }
        SCENE.loaded = false;
    }

    std::printf("\n== 24. 3D: diffusion along x, y and z is the same operator ==\n");
    {
        const int N = 20, steps = 60;
        std::vector<double> run[3];
        for (int axis = 0; axis < 3; axis++)
        {
            StandUp(N);
            RegisterBoth("d", Transport(TransportScheme::Upwind, TransportForm::Advective, 0.02, 0.0));
            UniformFlow3D(axis == 0 ? 0.5 : 0.0, axis == 1 ? 0.5 : 0.0, axis == 2 ? 0.5 : 0.0);
            StepProfile3D(0, axis);
            TransportSolver3D::InitializeTransportSolver3D(&gs);
            double lo = 0.0, hi = 0.0;
            RunSteps3D(steps, lo, hi);
            run[axis].assign(gs.ScalarField(0).data(), gs.ScalarField(0).data() + gs.ScalarField(0).size());
        }
        double worstY = 0.0, worstZ = 0.0;
        const int M = gs.Nx;   // N interior cells plus the ring on each side
        for (int i = 0; i < M; i++)
            for (int j = 0; j < M; j++)
                for (int k = 0; k < M; k++)
                {
                    const double x = run[0][(size_t)((i * M + j) * M + k)];
                    worstY = (std::max)(worstY, std::fabs(x - run[1][(size_t)((j * M + i) * M + k)]));
                    worstZ = (std::max)(worstZ, std::fabs(x - run[2][(size_t)((i * M + k) * M + j)]));
                }
        check(worstY < 1e-13 && worstZ < 1e-13, "the y and z runs are the x run, permuted, to round-off",
              fmt("worst %.3e (y), %.3e (z)", worstY, worstZ));
    }

    std::printf("\n== 26. solver=none: the grid holds the field and sets its ring, nothing moves it ==\n");
    {
        StandUp2D(24);
        MakeRing2D();
        if (!SCENE.LoadText("sampling center\nscalar k solver=none gamma=0.5 source=\"3*x\"\nscalar d scheme=upwind\n"
                            "boundary xmin inflow u=\"1\" k=\"2\" d=\"1\"\nboundary xmax pressure_outlet\n"
                            "boundary ymin noslip\nboundary ymax noslip\n"))
        { check(false, "scene loads", SCENE.error); }
        else
        {
            const ScalarTransport &spec = SCENE.scalars[0].transport;
            check(spec.solver == ScalarSolver::None, "the scene reads solver=none");
            check(SCENE.scalars[1].transport.solver == ScalarSolver::Adi, "a scalar with no solver key is ADI");
            check(spec.gamma == 0.5, "it keeps its parameters");
            RegisterBoth("k", spec);
            RegisterBoth("d", SCENE.scalars[1].transport);
            UniformFlow(1.0, 0.0);
            for (int i = 0; i < ga.Ny; i++)
                for (int j = 0; j < ga.Nx; j++)
                {
                    const double v = 0.3 + 0.01 * i + 0.001 * j;
                    ga.SetS(0, i, j, v); gs.SetS(0, i, j, v);
                    ga.SetS(1, i, j, 0.0); gs.SetS(1, i, j, 0.0);
                }
            TransportSolver2D::InitializeTransportSolver2D(&gs);
            for (int s = 0; s < 20; s++)
            {
                TransportSolver2D::SolveTransportStep(&ga, &gs, nullptr, 0.0);
                ga.CopyGrid(gs);
            }
            double moved = 0.0, dMax = 0.0;
            for (int i = 1; i < gs.Ny - 1; i++)
                for (int j = 1; j < gs.Nx - 1; j++)
                {
                    moved = (std::max)(moved, std::fabs(gs.GetS(0, i, j) - (0.3 + 0.01 * i + 0.001 * j)));
                    dMax = (std::max)(dMax, gs.GetS(1, i, j));
                }
            check(moved == 0.0, "the interior of a solver=none field is untouched, source and gamma included",
                  fmt("max change %.3e", moved));
            check(gs.ScalarSourceAt(0, 0.25, 0.5, 0.0, 0.0) == 0.75 && gs.ScalarSourceAt(1, 0.25, 0.5, 0.0, 0.0) == 0.0,
                  "ScalarSourceAt evaluates its source for the method, and 0 for a field without one");
            check(dMax > 0.1, "an ADI scalar beside it is still transported", fmt("max d %.3f", dMax));
            const int mid = gs.Ny / 2;
            check(gs.GetS(0, mid, 0) == 2.0, "its inflow ring carries the scene's value", fmt("%.6f", gs.GetS(0, mid, 0)));
            check(gs.GetS(0, mid, gs.Nx - 1) == gs.GetS(0, mid, gs.Nx - 2),
                  "and a face that pins nothing copies the cell inside");
        }
    }

    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASSED", fails,
                fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
} // namespace transport_check

static const SuiteCase CASES[] = {
    {"scalar transport", transport_check::run, {}, true},
};

int main(int argc, char **argv) { return RunSuite("transport", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
