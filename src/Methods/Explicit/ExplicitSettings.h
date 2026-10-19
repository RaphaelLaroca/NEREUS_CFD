// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef EXPLICIT_SETTINGS_H
#define EXPLICIT_SETTINGS_H

#include <cstddef>
#include "../../headers/Core/Definitions.h"

struct ExplicitSettings
{
    double relaxation = 1.7;
    double tolerance = 1e-6;
    int maxIterations = 10000;
};

inline ExplicitSettings EXPLICIT_SETTINGS;

inline const SettingField EXPLICIT_FIELDS[] = {
    DoubleSetting("EXPLICIT_RELAXATION", "SOR relaxation", "SOR over-relaxation factor, 0.5 to 2",
                  offsetof(ExplicitSettings, relaxation), 1.7, 0.5, 2.0),
    DoubleSetting("EXPLICIT_TOLERANCE", "Pressure tolerance", "Largest residual relative to the largest right-hand side",
                  offsetof(ExplicitSettings, tolerance), 1e-6, 1e-14, 1.0),
    IntSetting("EXPLICIT_MAX_ITERATIONS", "Max sweeps", "", offsetof(ExplicitSettings, maxIterations), 10000, 1, 1e7),
};

#include "Explicit2D.h"

static inline void InitializeExplicit()
{
    MAC::InitializeSimulationGrids();
    RegisterSimulationStep(Explicit2D::Momentum, "momentum", 0);
    RegisterSimulationStep(Explicit2D::Pressure, "pressure", 4);
    RegisterSimulationStep(Explicit2D::Projection, "projection", 6);
}

static inline void RegisterExplicit()
{
    RegisterSimulationMethod({"EXPLICIT", "Explicit (Eulerian)",
        "Forward Euler momentum and an SOR pressure solve, 2D uniform grids",
        MethodFamily::EULERIAN, true, false, CAP_GRID | CAP_BOUNDARY_CELLS | CAP_FIXED_REGION,
        InitializeExplicit, nullptr,
        &EXPLICIT_SETTINGS, EXPLICIT_FIELDS, (int)(sizeof(EXPLICIT_FIELDS) / sizeof(EXPLICIT_FIELDS[0])), nullptr,
        nullptr, Explicit2D::StableTimestep, nullptr, nullptr, nullptr});
}

#endif
