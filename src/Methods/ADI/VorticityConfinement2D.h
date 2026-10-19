// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef VORTICITYCONFINEMENT2D_H
#define VORTICITYCONFINEMENT2D_H

#include "../../headers/Core/MAC.h"
#include "../../headers/Core/Definitions.h"

#include <Eigen/Dense>
#include <vector>

// Vorticity confinement (Steinhoff & Underhill 1994; Fedkiw, Stam & Jensen 2001): a body force that
// puts back, around existing vortex cores, the rotation the grid's numerical dissipation removes.
class VorticityConfinement2D
{
public:
    // Sizes the scratch and builds the mask of cells far enough from any solid. The solid mask
    // of an Eulerian run does not change, so this runs once.
    static void Initialize(MAC *grid);
    static void Dispose();   // frees what Initialize allocated

    // Adds dt * f to the interior faces between two active cells. A no-op until Initialize.
    static void Apply(MAC *grid, double dt);

private:
    static Eigen::VectorXd corner;   // omega at corner (i, j), (Ny+1) x (Nx+1)
    static Eigen::VectorXd forceX;   // f at the cell centres
    static Eigen::VectorXd forceY;
    static std::vector<char> active; // the force is computed here, zero elsewhere
    static bool ready;
};

#endif
