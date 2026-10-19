// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef EXPLICIT2D_H
#define EXPLICIT2D_H

#include "../../headers/Core/Definitions.h"

class Explicit2D
{
public:
    static void Momentum(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);
    static void Pressure(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);
    static void Projection(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);
    static double StableTimestep();
};

#endif
