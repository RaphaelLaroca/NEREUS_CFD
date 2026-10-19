// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef FLIP_SETTINGS_H
#define FLIP_SETTINGS_H

// FLIP's settings, shared by FLIP2D and FLIP3D: the struct, its one instance and its table. The contact
// settings it shares with SPH are ParticleOptions (Definitions.h).

#include <cmath>
#include <cstddef>
#include <vector>
#include <Eigen/Dense>
#include "../../headers/Core/Definitions.h"
#include "../../headers/Utils.h"
#ifdef NEREUS_GUI
#include "../../headers/Core/SettingsPanel.h"
#endif

//for FLIP
enum class DIFFUSION_SOLVERS : int {
    ADI,
    IMPLICIT,
};

struct FLIPSettings
{
    int particlesPerCell = 4;
    double alpha = 1.0;                  // 1 = pure FLIP, 0 = PIC
    bool physicalDiffusion = false;
    DIFFUSION_SOLVERS diffusionSolver = DIFFUSION_SOLVERS::IMPLICIT;   // the ADI split, or one implicit solve
    // The body force and -grad p^n / rho inside the viscous solve's right-hand side (grad p^n added back after it),
    // so viscosity balances the net driving force. Off: the body force is added after the solve, undamped.
    bool viscousPressure = false;
    // Kugelstadt et al. (2019) density projection: a second Poisson solve per step that moves the particles
    // (not their velocities) toward the rest density, against FLIP's volume drift. Kugelstadt et al., IEEE TVCG 27(4), 2021.
    bool densityCorrection = false;
    bool densitySurfaceClamp = true;     // a cell beside air never reads below rest density
};

inline FLIPSettings FLIP_SETTINGS;

inline const char *const FLIP_DIFFUSION_KEYS[] = {"ADI", "IMPLICIT"};
inline const char *const FLIP_DIFFUSION_LABELS[] = {"Directional split (ADI)", "Single implicit solve (CG)"};

inline const SettingField FLIP_FIELDS[] = {
    IntSetting("PARTICLES_PER_CELL", "Particles per cell",
               "Particles seeded per cell, best as a square (cube in 3D)",
               offsetof(FLIPSettings, particlesPerCell), 4, 1, 27),
    DoubleSetting("ALPHA", "FLIP / PIC blend",
                  "1 is pure FLIP, 0 is pure PIC",
                  offsetof(FLIPSettings, alpha), 1.0, 0.0, 1.0),
    BoolSetting("PHYSICAL_DIFFUSION", "Physical viscosity",
                "Solve the viscous term. Off is inviscid",
                offsetof(FLIPSettings, physicalDiffusion), false),
    EnumSetting("DIFFUSION_SOLVER", "Viscous solve",
                "Directional split is cheaper. A single implicit solve has no splitting error",
                offsetof(FLIPSettings, diffusionSolver), (int)DIFFUSION_SOLVERS::IMPLICIT,
                FLIP_DIFFUSION_KEYS, FLIP_DIFFUSION_LABELS, 2),
    BoolSetting("FLIP_VISCOUS_PRESSURE", "Body force and grad p in the viscous solve",
                "For very viscous liquids, balances gravity and pressure inside the viscous solve",
                offsetof(FLIPSettings, viscousPressure), false),
    BoolSetting("FLIP_DENSITY_CORRECTION", "Density correction",
                "Moves particles toward the rest density each step. Stops volume drift",
                offsetof(FLIPSettings, densityCorrection), false),
    BoolSetting("FLIP_DENSITY_SURFACE_CLAMP", "Clamp density at the free surface",
                "Keeps surface cells from reading under-dense",
                offsetof(FLIPSettings, densitySurfaceClamp), true),
};
inline constexpr int FLIP_FIELD_COUNT = (int)(sizeof(FLIP_FIELDS) / sizeof(FLIP_FIELDS[0]));

// The setup screen's FLIP section: the table's fields, with the seeding layout, the PIC viscosity
// and the viscous-solver choice shown only where they apply.
static inline void DrawFLIPSettings(const SettingsContext &ctx)
{
#ifdef NEREUS_GUI
    FLIPSettings &s = FLIP_SETTINGS;
    DrawSetting(FLIP_FIELDS, FLIP_FIELD_COUNT, &s, "PARTICLES_PER_CELL");
    if (ctx.dimension == 3)
    {
        const int n = (int)std::cbrt((double)s.particlesPerCell);
        ImGui::TextDisabled("Seeded as a %dx%dx%d block per cell (%d actually placed).", n, n, n, n * n * n);
    }
    else
    {
        const int n = (int)std::sqrt((double)s.particlesPerCell);
        ImGui::TextDisabled("Seeded as a %dx%d block per cell (%d actually placed).", n, n, n * n);
    }

    DrawSetting(FLIP_FIELDS, FLIP_FIELD_COUNT, &s, "ALPHA");
    ImGui::TextDisabled("1.0 = pure FLIP (no damping), 0.0 = pure PIC (very dissipative).");
    // Bridson's PIC-fraction relation alpha_PIC = 6 dt nu / dx^2, with alpha_PIC = 1 - ALPHA here.
    if (ctx.dh > 0.0 && ctx.timeStep > 0.0)
    {
        const double nuPic = (1.0 - s.alpha) * ctx.dh * ctx.dh / (6.0 * ctx.timeStep);
        if (nuPic > 0.0)
        {
            ImGui::Text("PIC numerical viscosity  (1 - alpha) dx^2 / (6 dt) = %.3g m²/s", nuPic);
            if (ctx.viscosity > 0.0)
                ImGui::TextDisabled("%.3gx the physical nu (%.3g m²/s), at dx = %.4g m, dt = %.3g s.",
                                    nuPic / ctx.viscosity, ctx.viscosity, ctx.dh, ctx.timeStep);
            else
                ImGui::TextDisabled("At dx = %.4g m, dt = %.3g s.", ctx.dh, ctx.timeStep);
            if (ctx.adaptiveTimestep)
                ImGui::TextDisabled("Adaptive dt only shrinks the step, so this is the minimum.");
        }
        else
            ImGui::TextDisabled("PIC numerical viscosity: none (pure FLIP).");
    }

    ImGui::Spacing();
    DrawSetting(FLIP_FIELDS, FLIP_FIELD_COUNT, &s, "PHYSICAL_DIFFUSION");
    if (s.physicalDiffusion)
    {
        ImGui::Indent();
        DrawSetting(FLIP_FIELDS, FLIP_FIELD_COUNT, &s, "DIFFUSION_SOLVER");
        DrawSetting(FLIP_FIELDS, FLIP_FIELD_COUNT, &s, "FLIP_VISCOUS_PRESSURE");
        if (ctx.dh > 0.0 && ctx.timeStep > 0.0 && ctx.viscosity > 0.0)
            ImGui::TextDisabled("nu dt / dx^2 = %.3g", ctx.viscosity * ctx.timeStep / (ctx.dh * ctx.dh));
        ImGui::Unindent();
    }
    else
        ImGui::TextDisabled("Off: the flow is inviscid apart from numerical dissipation.");

    ImGui::Spacing();
    DrawSetting(FLIP_FIELDS, FLIP_FIELD_COUNT, &s, "FLIP_DENSITY_CORRECTION");
    if (s.densityCorrection)
    {
        ImGui::Indent();
        DrawSetting(FLIP_FIELDS, FLIP_FIELD_COUNT, &s, "FLIP_DENSITY_SURFACE_CLAMP");
        ImGui::Unindent();
    }
    else
        ImGui::TextDisabled("Off: the particle volume drifts on long runs (+87%% over 6 s on Kleefsman at 2 cm).");
#endif
}

static inline void CheckFLIPSettings(const SettingsContext &ctx, std::vector<std::string> &problems,
                                     std::vector<std::string> &notes)
{
    const FLIPSettings &s = FLIP_SETTINGS;
    if (s.physicalDiffusion && ctx.dh > 0.0 && !ctx.adaptiveTimestep &&
        s.diffusionSolver == DIFFUSION_SOLVERS::ADI && ctx.viscosity * ctx.timeStep / (ctx.dh * ctx.dh) > 0.5)
        notes.push_back("nu*dt/dh^2 is large, so the directional split will visibly "
                        "disagree with the single implicit solve. Lower the timestep or "
                        "pick the CG solve if that matters.");
    if (s.alpha > 0.99)
        notes.push_back("A blend of 1.0 is pure FLIP: nothing damps particle noise.");
}

// ---- FLIP's viscous solve: the pencil DiffusionSolver2D/3D fill row by row and solve run by run ----

// One row of a sweep's tridiagonal system; unsolvable rows are pinned and left alone.
struct PencilRow
{
    double lower;
    double diag;
    double upper;
    double rhs;
    double x;
    bool solvable;
};

// A line of rows plus the TDMA scratch, sized once and reused across pencils.
struct PencilLine
{
    std::vector<PencilRow> rows;
    Eigen::VectorXd lower, diag, upper, rhs, sol;
};

// Grows only, so a static line allocates on its first pencil and never again.
inline void ResizePencilLine(PencilLine &line, int n)
{
    if ((int)line.rows.size() >= n) return;
    line.rows.resize(n);
    line.lower.resize(n); line.diag.resize(n); line.upper.resize(n);
    line.rhs.resize(n); line.sol.resize(n);
}

// Solves each run of consecutive solvable rows as its own tridiagonal system; unsolvable rows are left alone.
inline void SolvePencilLine(PencilLine &line, int n)
{
    int a = 0;
    while (a < n)
    {
        if (!line.rows[a].solvable) { a++; continue; }
        int b = a;
        while (b + 1 < n && line.rows[b + 1].solvable) b++;
        const int len = b - a + 1;

        for (int t = 0; t < len; t++)
        {
            const PencilRow &r = line.rows[a + t];
            line.diag(t) = r.diag;
            line.lower(t) = r.lower;
            line.upper(t) = r.upper;
            line.rhs(t) = r.rhs;
            line.sol(t) = 0.0;
        }
        TDMA_VECTORIZED_N(line.lower, line.diag, line.upper, line.rhs, line.sol, len);
        for (int t = 0; t < len; t++)
            line.rows[a + t].x = line.sol(t);

        a = b + 1;
    }
}

// ---- the method: the row main() registers with RegisterFLIP(), and its initialisation ----
// Below the settings because the solvers include this header; static inline, so each program compiles its own panel.
#include "FLIP2D.h"
#include "FLIP3D.h"
#include "DiffusionSolver2D.h"
#include "DiffusionSolver3D.h"
#include "../PPESolver/PressureSolver2D.h"
#include "../PPESolver/PressureSolver3D.h"

// The pressure matrix is rebuilt whenever the fluid region moves; FLIP seeds the particles and registers its steps.
static inline void InitializeFLIP()
{
    if (DIMENSION == 2)
    {
        PressureSolver2D::InitializePressureSolver(SIMULATION.GRID_SOL);
        FLIP2D::InitializeFLIP(SIMULATION.GRID_SOL, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    }
    else
    {
        PressureSolver3D::InitializePressureSolver(SIMULATION.GRID_SOL);
        FLIP3D::InitializeFLIP(SIMULATION.GRID_SOL, &SIMULATION.particles, SIMULATION.dt, FLIP_SETTINGS.alpha);
    }
    MAC::InitializeSimulationGrids();
}

// Frees what InitializeFLIP built, both dimensions.
static inline void DisposeFLIP()
{
    FLIP2D::Dispose();
    FLIP3D::Dispose();
    DiffusionSolver2D::Dispose();
    DiffusionSolver3D::Dispose();
    PressureSolver2D::ReleasePressureSolver();
    PressureSolver3D::ReleasePressureSolver();
}

// A grid-based cloud is seeded per cell.
static inline double FLIPParticleSpacing()
{
    const int ppc = (FLIP_SETTINGS.particlesPerCell > 0) ? FLIP_SETTINGS.particlesPerCell : 8;
    return SIMULATION.dh / std::cbrt((double)ppc);
}

static inline void RegisterFLIP()
{
    RegisterSimulationMethod({"FLIP", "FLIP (hybrid)",
        "Particle velocities with a grid projection, for free-surface liquids",
        MethodFamily::HYBRID, true, true,
        CAP_GRID | CAP_PARTICLES | CAP_BOUNDARY_CELLS | CAP_PRESSURE_SOLVER,
        InitializeFLIP, DisposeFLIP,
        &FLIP_SETTINGS, FLIP_FIELDS, FLIP_FIELD_COUNT, DrawFLIPSettings, CheckFLIPSettings,
        nullptr, nullptr, nullptr, FLIPParticleSpacing});
}

#endif
