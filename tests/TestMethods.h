// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_METHODS_H
#define TEST_METHODS_H

// For the harnesses, which drive solvers without main(): every shipped method registered, as main() does.
#include "Methods/ADI/ADISettings.h"
#include "Methods/FLIP/FLIPSettings.h"
#include "Methods/SPH/SPHSettings.h"
#include "Methods/Explicit/ExplicitSettings.h"
#include <cstdio>

// Starts the registry from empty, so calling it twice registers each method once.
static inline void RegisterBuiltinMethods()
{
    METHOD_COUNT = 0;
    RegisterADI();
    RegisterFLIP();
    RegisterSPH();
    RegisterExplicit();
}

// The shipped methods registered and `key` made the active one.
static inline void SelectMethod(const char *key)
{
    RegisterBuiltinMethods();
    SIM_TYPE = MethodFromKey(key);
}

// The dam-break block the FLIP harnesses fill, as a scene `region fluid`: 0.28 x 0.11 m in 2D, 0.51 x 0.258 x 0.3 m in
// 3D, from the domain's corner and rounded down to whole cells, so the particles land where they always have.
static inline void LoadReservoirScene(int dim, double dh)
{
    const Domain &d = SIMULATION.domain;
    char text[256];
    if (dim == 2)
        std::snprintf(text, sizeof(text), "region fluid box min=(%.17g,%.17g,0) max=(%.17g,%.17g,1)\n",
                      d.x0, d.y0, d.x0 + (int)(0.28 / dh) * dh, d.y0 + (int)(0.11 / dh) * dh);
    else
        std::snprintf(text, sizeof(text), "region fluid box min=(%.17g,%.17g,%.17g) max=(%.17g,%.17g,%.17g)\n",
                      d.x0, d.y0, d.z0, d.x0 + (int)(0.51 / dh) * dh, d.y0 + (int)(0.258 / dh) * dh,
                      d.z0 + (int)(0.3 / dh) * dh);
    SCENE.LoadText(text);
}

#endif
