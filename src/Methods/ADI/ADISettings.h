// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef ADI_SETTINGS_H
#define ADI_SETTINGS_H

// ADI's settings, shared by ADI2D, ADI3D and the 2D vorticity confinement it applies: the struct, its
// one instance and the table the config and the setup screen walk.

#include <cstddef>
#include "../../headers/Core/Definitions.h"
#ifdef NEREUS_GUI
#include "../../headers/Core/SettingsPanel.h"
#endif

// How the ADI momentum sweeps difference the convective term (the transport solver has its own, per scalar).
// Upwind adds a numerical viscosity |u| h / 2.
enum ConvectionScheme { CONVECTION_CENTRAL = 0, CONVECTION_UPWIND = 1 };
// How the ADI step couples to the pressure. Chorin: the predictor carries no pressure and the solve returns p.
// Incremental: the predictor carries grad p^n and the solve returns the increment, so a steady state is dt-free.
enum ProjectionScheme { PROJECTION_CHORIN = 0, PROJECTION_INCREMENTAL = 1 };

// Advective: u.grad(u) at the node, a finite difference. Flux: div(u u) over the velocity's own staggered
// control volume, a finite volume that conserves momentum discretely.
enum ConvectionForm { CONVECTION_ADVECTIVE = 0, CONVECTION_FLUX = 1 };

struct ADISettings
{
    int convectionScheme = CONVECTION_CENTRAL;
    int projectionScheme = PROJECTION_CHORIN;
    int convectionForm = CONVECTION_FLUX;   // the default: every validation case runs the finite-volume form
    // Vorticity confinement (VorticityConfinement2D), 2D only: a visual control, not physics. eps is
    // NOT dimensionless -- the force is eps*h*(N x omega), so eps carries 1/s.
    bool vorticityConfinement = false;
    double vorticityEpsilon = 10.0;
    int vorticitySolidMargin = 2;    // cells of clearance from any solid, to leave wall shear alone
};

inline ADISettings ADI_SETTINGS;

inline const char *const ADI_CONVECTION_KEYS[] = {"CENTRAL", "UPWIND"};
inline const char *const ADI_CONVECTION_LABELS[] = {"Central difference", "Upwind (first order)"};
inline const char *const ADI_PROJECTION_KEYS[] = {"CHORIN", "INCREMENTAL"};
inline const char *const ADI_FORM_KEYS[] = {"ADVECTIVE", "FLUX"};
inline const char *const ADI_FORM_LABELS[] = {"Advective (finite difference)", "Flux (finite volume)"};
inline const char *const ADI_PROJECTION_LABELS[] = {"Chorin", "Incremental"};

inline const SettingField ADI_FIELDS[] = {
    EnumSetting("CONVECTION_SCHEME", "Convection scheme",
                "Central is second order but oscillates above cell Re 2. Upwind is bounded but diffusive",
                offsetof(ADISettings, convectionScheme), CONVECTION_CENTRAL,
                ADI_CONVECTION_KEYS, ADI_CONVECTION_LABELS, 2),
    EnumSetting("PROJECTION_SCHEME", "Projection scheme",
                "Incremental gives a steady state independent of dt. Chorin does not",
                offsetof(ADISettings, projectionScheme), PROJECTION_CHORIN,
                ADI_PROJECTION_KEYS, ADI_PROJECTION_LABELS, 2),
    EnumSetting("CONVECTION_FORM", "Convection form",
                "Flux conserves momentum. Advective keeps a uniform field uniform",
                offsetof(ADISettings, convectionForm), CONVECTION_FLUX,
                ADI_FORM_KEYS, ADI_FORM_LABELS, 2),
    BoolSetting("VORTICITY_CONFINEMENT", "Vorticity confinement",
                "Restores small vortices lost to numerical dissipation. Visual only, adds energy",
                offsetof(ADISettings, vorticityConfinement), false),
    WithUnit(DoubleSetting("VORTICITY_EPSILON", "Confinement strength",
                           "Around 10 is visible, 100 is too strong",
                           offsetof(ADISettings, vorticityEpsilon), 10.0, 0.0), "1/s"),
    WithUnit(IntSetting("VORTICITY_SOLID_MARGIN", "Solid margin",
                        "No confinement within this many cells of a solid",
                        offsetof(ADISettings, vorticitySolidMargin), 2, 0), "cells"),
};
inline constexpr int ADI_FIELD_COUNT = (int)(sizeof(ADI_FIELDS) / sizeof(ADI_FIELDS[0]));

// The setup screen's ADI section: the two schemes side by side, the cell Reynolds number they have to
// be judged against, and confinement where it runs (2D).
static inline void DrawADISettings(const SettingsContext &ctx)
{
#ifdef NEREUS_GUI
    ADISettings &s = ADI_SETTINGS;
    DrawSetting(ADI_FIELDS, ADI_FIELD_COUNT, &s, "CONVECTION_SCHEME");
    DrawSetting(ADI_FIELDS, ADI_FIELD_COUNT, &s, "CONVECTION_FORM");
    DrawSetting(ADI_FIELDS, ADI_FIELD_COUNT, &s, "PROJECTION_SCHEME");

    // Cell Reynolds and, under upwind, the numerical viscosity it adds, at the mean velocity.
    if (ctx.dh > 0.0 && ctx.viscosity > 0.0 && ctx.meanVelocity > 0.0)
    {
        const double cellRe = ctx.meanVelocity * ctx.dh / ctx.viscosity;
        ImGui::Spacing();
        ImGui::Text("Cell Reynolds  U h / nu = %.3g", cellRe);
        ImGui::TextDisabled("U = %.3g m/s (mean velocity, Export section), h = %.4g m (base spacing)",
                            ctx.meanVelocity, ctx.dh);
        if (s.convectionScheme == CONVECTION_UPWIND)
        {
            const double nuNum = 0.5 * ctx.meanVelocity * ctx.dh;
            const double nuEff = ctx.viscosity + nuNum;
            ImGui::Text("Upwind numerical viscosity  U h / 2 = %.3g m²/s  (%.3gx nu)", nuNum, nuNum / ctx.viscosity);
            ImGui::Text("Effective viscosity %.3g m²/s, effective Re %.4g (physical %.4g)",
                        nuEff, ctx.meanVelocity * ctx.characteristicLength / nuEff,
                        ctx.meanVelocity * ctx.characteristicLength / ctx.viscosity);
            ImGui::TextDisabled("It scales with the local |u| and h, so it is largest in the fastest, coarsest cells.");
        }
        else if (cellRe > 2.0)
            ImGui::TextColored(ToneColour(UiTone::Warn),
                "Above 2, central convection is not damped by the viscosity and can oscillate or diverge.");
    }

    ImGui::Spacing();
    if (ctx.dimension == 2)
    {
        DrawSetting(ADI_FIELDS, ADI_FIELD_COUNT, &s, "VORTICITY_CONFINEMENT");
        if (s.vorticityConfinement)
        {
            ImGui::Indent();
            DrawSetting(ADI_FIELDS, ADI_FIELD_COUNT, &s, "VORTICITY_EPSILON");
            DrawSetting(ADI_FIELDS, ADI_FIELD_COUNT, &s, "VORTICITY_SOLID_MARGIN");
            ImGui::Unindent();
        }
    }
    else
    {
        // Off on a path that cannot run it, rather than writing a setting the solver ignores.
        s.vorticityConfinement = false;
        ImGui::TextDisabled("Vorticity confinement: 2D only for now.");
    }
#endif
}

// ---- shared by the ADI2D and ADI3D sweeps: the convective part of a tridiagonal row ----
// One flux-form control volume along one axis: width h, face transport velocities Vm/Vp, face weights am/ap.
struct FluxGeom { double h, Vm, Vp, am, ap; };

// Component c at the face whose indices are ordered by axis, q = {j, i, k}.
inline double FluxRead(MAC *g, MAC::Component c, const int q[3], bool threeD)
{
    return threeD ? g->GetComponent(c, q[1], q[0], q[2]) : g->GetComponent(c, q[1], q[0]);
}

// Along the component's own axis the volume runs centre to centre, transport velocity the mean of its faces;
// along another axis it is one cell wide, transport velocity interpolated, the unknown weighted by distance.
inline FluxGeom FluxGeometry(MAC *g, MAC::Component c, MAC::Axis a, const int p[3], bool threeD)
{
    FluxGeom f;
    int q[3] = {p[0], p[1], p[2]};
    const int n = p[a];
    const MAC::Axis own = MAC::AxisOf(c);
    if (a == own)
    {
        f.h = g->FaceGap(a, n - 1);
        q[a] = n - 1; const double behind = FluxRead(g, c, q, threeD);
        q[a] = n;     const double here = FluxRead(g, c, q, threeD);
        q[a] = n + 1; const double ahead = FluxRead(g, c, q, threeD);
        f.Vm = 0.5 * (behind + here);
        f.Vp = 0.5 * (here + ahead);
        f.am = 0.5;
        f.ap = 0.5;
        return f;
    }
    const MAC::Component transport = MAC::ComponentAlong(a);
    const int m = p[own];
    const double wBehind = 0.5 * g->Width(own, m) / g->FaceGap(own, m - 1);   // cell m-1's share at face m
    f.h = g->Width(a, n);
    q[a] = n;     q[own] = m - 1; const double lowBehind = FluxRead(g, transport, q, threeD);
                  q[own] = m;     const double lowHere = FluxRead(g, transport, q, threeD);
    q[a] = n + 1; q[own] = m - 1; const double highBehind = FluxRead(g, transport, q, threeD);
                  q[own] = m;     const double highHere = FluxRead(g, transport, q, threeD);
    f.Vm = wBehind * lowBehind + (1.0 - wBehind) * lowHere;
    f.Vp = wBehind * highBehind + (1.0 - wBehind) * highHere;
    f.am = 0.5 * g->Width(a, n) / g->FaceGap(a, n - 1);
    f.ap = 0.5 * g->Width(a, n + 1) / g->FaceGap(a, n);
    return f;
}

// The flux-form geometry of component c's control volume at face (i, j[, k]) along axis a, with transport velocities
// from grid g.
inline FluxGeom FluxGeomAt(MAC *g, MAC::Component c, MAC::Axis a, int i, int j, int k)
{
    const int p[3] = {j, i, k};
    return FluxGeometry(g, c, a, p, true);
}

inline FluxGeom FluxGeomAt(MAC *g, MAC::Component c, MAC::Axis a, int i, int j)
{
    const int p[3] = {j, i, 0};
    return FluxGeometry(g, c, a, p, false);
}

// One tridiagonal row's share of an implicit convection term (upwind, or flux form).
struct UpwindRow { double low, mid, up; };

// First-order upwind a du/ds over one sweep, with cB = dt_s/h- and cF = dt_s/h+.
// Implicit: the row coefficients. Explicit: the term that goes to the right-hand side.
inline UpwindRow UpwindCoefficients(double a, double cB, double cF)
{
    if (a > 0.0) return UpwindRow{-cB * a, cB * a, 0.0};
    return UpwindRow{0.0, -cF * a, cF * a};
}

inline double UpwindDifference(double a, double uM, double u, double uP, double cB, double cF)
{
    return (a > 0.0) ? cB * a * (u - uM) : cF * a * (uP - u);
}

// Flux-form convection over one control volume, dts/h * (F+ - F-) with F = V phi on each face. Central
// puts phi on a face by the linear weights, upwind takes the value the flow comes from.
inline UpwindRow FluxCoefficients(double dts, const FluxGeom &g, bool upwind)
{
    const double C = dts / g.h;
    if (upwind)
    {
        const double pPos = (g.Vp > 0.0) ? g.Vp : 0.0, pNeg = (g.Vp < 0.0) ? g.Vp : 0.0;
        const double mPos = (g.Vm > 0.0) ? g.Vm : 0.0, mNeg = (g.Vm < 0.0) ? g.Vm : 0.0;
        return UpwindRow{-C * mPos, C * (pPos - mNeg), C * pNeg};
    }
    return UpwindRow{-C * g.Vm * g.am, C * (g.Vp * g.ap - g.Vm * (1.0 - g.am)), C * g.Vp * (1.0 - g.ap)};
}

inline double FluxDifference(double dts, const FluxGeom &g, bool upwind, double phiM, double phi, double phiP)
{
    const UpwindRow r = FluxCoefficients(dts, g, upwind);
    return r.low * phiM + r.mid * phi + r.up * phiP;
}

// A no-slip or moving wall across a tangential diffusion link: the wall velocity sits half a cell away, so the
// link is scaled by f (MAC::WallLinkFactor). These add what that changes; zero without a wall.
inline double WallExplicit(const MAC *g, MAC::Component c, MAC::Axis a, int i, int j, int k,
                           double aLo, double aUp, double uLo, double u, double uUp)
{
    const double fLo = g->WallLinkFactor(c, a, i, j, k, -1), fUp = g->WallLinkFactor(c, a, i, j, k, +1);
    if (fLo == 1.0 && fUp == 1.0) return 0.0;
    return (fLo - 1.0) * aLo * (uLo - u) + (fUp - 1.0) * aUp * (uUp - u);
}
// Implicit row: the diagonal gains (f - 1) a per side. The neighbour's value then enters with f a: an interior solid
// face is pinned at 0 and drops out; a ring face's value moves to the right-hand side (WallRing).
inline double WallDiagonal(const MAC *g, MAC::Component c, MAC::Axis a, int i, int j, int k, double aLo, double aUp)
{
    const double fLo = g->WallLinkFactor(c, a, i, j, k, -1), fUp = g->WallLinkFactor(c, a, i, j, k, +1);
    if (fLo == 1.0 && fUp == 1.0) return 0.0;
    return (fLo - 1.0) * aLo + (fUp - 1.0) * aUp;
}
// The extra right-hand side of a pencil's end row whose known neighbour (value uRing) is a wall face: (f - 1) a uRing.
inline double WallRing(const MAC *g, MAC::Component c, MAC::Axis a, int i, int j, int k, int s, double aLink, double uRing)
{
    const double f = g->WallLinkFactor(c, a, i, j, k, s);
    return f == 1.0 ? 0.0 : (f - 1.0) * aLink * uRing;
}

// ---- the method: the row main() registers with RegisterADI(), and its initialisation ----
// Below the settings because the solvers include this header; static inline, so each program compiles its own panel.
#include "ADI2D.h"
#include "ADI3D.h"
#include "VorticityConfinement2D.h"
#include "../PPESolver/PressureSolver2D.h"
#include "../PPESolver/PressureSolver3D.h"
#include "../../headers/Solvers/TransportSolver2D.h"
#include "../../headers/Solvers/TransportSolver3D.h"

// The pressure solver's unknown under the projection this run uses.
inline PressureScheme ADIPressureScheme()
{
    return ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL ? PRESSURE_INCREMENTAL : PRESSURE_CHORIN;
}

// Builds the momentum, pressure and (with scalars declared) transport solvers; each registers its own steps.
static inline void InitializeADI()
{
    if (DIMENSION == 2)
    {
        ADI2D::InitializeADI2D(SIMULATION.GRID_SOL, SIMULATION.dt,
                               SIMULATION.VelocityBoundaryFunction2D,
                               SIMULATION.PressureBoundaryFunction2D);
        PressureSolver2D::InitializePressureSolver(SIMULATION.GRID_SOL, ADIPressureScheme());
        MAC::InitializeSimulationGrids();
        if (SIMULATION.GRID_SOL->ScalarCount() > 0)
            TransportSolver2D::InitializeTransportSolver2D(SIMULATION.GRID_SOL);
    }
    else
    {
        ADI3D::InitializeADI(SIMULATION.GRID_SOL, SIMULATION.dt,
                             SIMULATION.VelocityBoundaryFunction,
                             SIMULATION.PressureBoundaryFunction);
        PressureSolver3D::InitializePressureSolver(SIMULATION.GRID_SOL, ADIPressureScheme());
        MAC::InitializeSimulationGrids();
        if (SIMULATION.GRID_SOL->ScalarCount() > 0)
            TransportSolver3D::InitializeTransportSolver3D(SIMULATION.GRID_SOL);
    }
}

// Frees the solvers InitializeADI built, both dimensions (a solver never built has nothing to free).
static inline void DisposeADI()
{
    ADI2D::Dispose();
    ADI3D::Dispose();
    VorticityConfinement2D::Dispose();
    PressureSolver2D::ReleasePressureSolver();
    PressureSolver3D::ReleasePressureSolver();
    TransportSolver2D::Dispose();
    TransportSolver3D::Dispose();
}

static inline void RegisterADI()
{
    RegisterSimulationMethod({"ADI", "ADI (Eulerian)",
        "Implicit ADI momentum with a pressure projection",
        MethodFamily::EULERIAN, true, true,
        CAP_GRID | CAP_RECTILINEAR | CAP_BOUNDARY_CELLS | CAP_FIXED_REGION | CAP_SCALARS | CAP_PRESSURE_SOLVER,
        InitializeADI, DisposeADI,
        &ADI_SETTINGS, ADI_FIELDS, ADI_FIELD_COUNT, DrawADISettings, nullptr,
        nullptr, nullptr, nullptr, nullptr});
}

#endif
