// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPH_SETTINGS_H
#define SPH_SETTINGS_H

// SPH's settings, shared by SPH2D and SPH3D: WCSPH (continuity density with delta-SPH, p = c0^2 (rho - rho0),
// Monaghan viscosity, Wendland C2). h is the kernel SUPPORT radius; rho0 comes from DENSITY.

#include <cmath>
#include <cstddef>
#include <algorithm>
#include "../../headers/Core/Definitions.h"
#ifdef NEREUS_GUI
#include "../../headers/Core/SettingsPanel.h"
#endif

struct SPHSettings
{
    double kernelRadius = 0.05;      // h, the kernel support radius
    double particleSpacing = 0.0;    // seeding lattice pitch; <= 0 means h/2.6
    double soundSpeed = 20.0;        // c0: rule of thumb >= 10 max(sqrt(g H), u_max), ~1% density variation
    double delta = 0.1;              // delta-SPH density diffusion coefficient
    double alpha = 0.02;             // Monaghan artificial viscosity coefficient
};

inline SPHSettings SPH_SETTINGS;

// Phase timings, wall seconds for the whole OUTER step with every substep summed, so they add up to
// totalTime -- the `sph` step's time seen from inside. Telemetry, not settings.
struct SPHTimings
{
    double neighbourTime = 0.0;   // BuildGhosts + BuildCellList
    double densityTime = 0.0;     // LoadDensityPressure + SyncGhostFields
    double forceTime = 0.0;       // ComputeForces
    double integrateTime = 0.0;   // Integrate, contacts included
    double totalTime = 0.0;
    int ghosts = 0;
};

inline SPHTimings SPH_TIMINGS;

inline const SettingField SPH_FIELDS[] = {
    WithUnit(DoubleSetting("SPH_KERNEL_RADIUS", "Kernel radius h",
                           "Kernel support radius, the resolution of the method",
                           offsetof(SPHSettings, kernelRadius), 0.05, 1e-6), "m"),
    WithUnit(DoubleSetting("SPH_PARTICLE_SPACING", "Particle spacing",
                           "Seeding lattice pitch. 0 uses h/2.6",
                           offsetof(SPHSettings, particleSpacing), 0.0, 0.0), "m"),
    WithUnit(DoubleSetting("SPH_SOUND_SPEED", "Sound speed c0",
                           "Artificial sound speed. At least 10 times the fastest flow speed",
                           offsetof(SPHSettings, soundSpeed), 20.0, 1e-6), "m/s"),
    DoubleSetting("SPH_DELTA", "Density diffusion",
                  "delta-SPH coefficient. 0.1 is standard, 0 turns it off",
                  offsetof(SPHSettings, delta), 0.1, 0.0),
    DoubleSetting("SPH_ALPHA", "Artificial viscosity",
                  "Artificial viscosity, the only viscosity. 0.01 to 0.02 for water",
                  offsetof(SPHSettings, alpha), 0.02, 0.0),
};
inline constexpr int SPH_FIELD_COUNT = (int)(sizeof(SPH_FIELDS) / sizeof(SPH_FIELDS[0]));

// The lattice pitch a run seeds with: h/2.6, i.e. a smoothing length (h/2) of 1.3 spacings, the usual Wendland choice.
inline double SPHSeedSpacing(const SPHSettings &s)
{
    if (s.particleSpacing > 0.0) return s.particleSpacing;
    return s.kernelRadius / 2.6;
}

// The setup screen's SPH section: the fields, with what they imply -- mass, neighbours per kernel, and the
// density variation g H / c0^2.
static inline void DrawSPHSettings(const SettingsContext &ctx)
{
#ifdef NEREUS_GUI
    SPHSettings &s = SPH_SETTINGS;
    ImGui::TextWrapped(
        "Weakly compressible SPH: continuity density with delta-SPH diffusion, p = c0^2 (rho - rho0), "
        "the symmetric pressure force, Monaghan artificial viscosity and a Wendland C2 kernel. The mass is "
        "rho0 * spacing^dim and the column starts hydrostatic.");
    ImGui::Separator();

    ImGui::TextUnformatted("Resolution");
    DrawSetting(SPH_FIELDS, SPH_FIELD_COUNT, &s, "SPH_KERNEL_RADIUS");
    DrawSetting(SPH_FIELDS, SPH_FIELD_COUNT, &s, "SPH_PARTICLE_SPACING");

    const double spacing = SPHSeedSpacing(s);
    const double mass = ctx.density * spacing * spacing;
    const double perKernel = M_PI * s.kernelRadius * s.kernelRadius / (spacing * spacing);
    const double area = (std::max)(ctx.xf - ctx.x0, 0.0) * (std::max)(ctx.yf - ctx.y0, 0.0);
    const long estN = (long)(area / (spacing * spacing));
    ImGui::TextDisabled("spacing %.5f m   mass %.6g kg   ~%.0f neighbours per kernel", spacing, mass, perKernel);
    ImGui::TextDisabled("up to ~%ld particles if the whole domain were fluid", estN);
    // Too few neighbours and the density sum is noise; too many and it is slow.
    if (perKernel < 12.0)
        ImGui::TextColored(ToneColour(UiTone::Note),
            "Only ~%.0f neighbours per kernel -- the density estimate will be noisy. "
            "Raise h or lower the spacing.", perKernel);
    else if (perKernel > 80.0)
        ImGui::TextColored(ToneColour(UiTone::Warn),
            "~%.0f neighbours per kernel is a lot of work per particle.", perKernel);

    ImGui::Spacing();
    ImGui::TextUnformatted("Physical parameters");
    ImGui::Text("Rest density  rho0 = %.1f kg/m³", ctx.density);
    ImGui::TextDisabled("from DENSITY in the Fluid section -- one fluid, one density");
    DrawSetting(SPH_FIELDS, SPH_FIELD_COUNT, &s, "SPH_SOUND_SPEED");
    DrawSetting(SPH_FIELDS, SPH_FIELD_COUNT, &s, "SPH_DELTA");
    DrawSetting(SPH_FIELDS, SPH_FIELD_COUNT, &s, "SPH_ALPHA");
    const double g = std::abs(ctx.fv), H = (std::max)(ctx.yf - ctx.y0, 0.0);
    ImGui::TextDisabled("The artificial viscosity is the only viscosity: VISCOSITY is not used by SPH.");
    if (g > 0.0 && H > 0.0)
    {
        ImGui::TextDisabled("10 sqrt(g H) = %.1f m/s over the domain height; density variation ~%.2f%%",
                            10.0 * std::sqrt(g * H), 100.0 * g * H / (s.soundSpeed * s.soundSpeed));
        if (s.soundSpeed < 10.0 * std::sqrt(g * H))
            ImGui::TextColored(ToneColour(UiTone::Warn), "c0 is under 10 sqrt(g H): the liquid will compress.");
    }
    ImGui::TextDisabled("acoustic CFL %.2f at dt = %g (substeps are taken to keep it at 0.25 of h/2)",
                        s.soundSpeed * ctx.timeStep / (0.5 * s.kernelRadius), ctx.timeStep);
#endif
}

// ---- the method: the row main() registers with RegisterSPH(), and its initialisation ----
// Below the settings because the solvers include this header; static inline, so each program compiles its own panel.
#include "SPH2D.h"
#include "SPH3D.h"

static inline void InitializeSPH()
{
    if (DIMENSION == 2) SPH2D::InitializeSPH2D(&SIMULATION.particles, 1000000);
    else                SPH3D::InitializeSPH3D(&SIMULATION.particles, 2000000);
}

static inline void DisposeSPH()
{
    SPH2D::Dispose();
    SPH3D::Dispose();
}

static inline double SPHStableTimestep()
{
    return (DIMENSION == 2) ? SPH2D::StableTimestep(&SIMULATION.particles) : SPH3D::StableTimestep(&SIMULATION.particles);
}

static inline double SPHMaxVelocity()
{
    return (DIMENSION == 2) ? SPH2D::GetMaxVelocity(&SIMULATION.particles) : SPH3D::GetMaxVelocity(&SIMULATION.particles);
}

static inline double SPHLength() { return (DIMENSION == 2) ? SPH2D::h : SPH3D::h; }

static inline double SPHParticleSpacing() { return SPHSeedSpacing(SPH_SETTINGS); }

static inline void RegisterSPH()
{
    RegisterSimulationMethod({"SPH", "SPH (Lagrangian)",
        "Smoothed particle hydrodynamics, no grid",
        MethodFamily::LAGRANGIAN, true, true,
        CAP_PARTICLES | CAP_FIXED_REGION,
        InitializeSPH, DisposeSPH,
        &SPH_SETTINGS, SPH_FIELDS, SPH_FIELD_COUNT, DrawSPHSettings, nullptr,
        SPHStableTimestep, SPHMaxVelocity, SPHLength, SPHParticleSpacing});
}

#endif
