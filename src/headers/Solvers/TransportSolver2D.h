// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TRANSPORTSOLVER2D_H
#define TRANSPORTSOLVER2D_H

#include "../Core/MAC.h"
#include "../Core/Definitions.h"
#include "Utils.h"

#include <Eigen/Dense>
#include <string>

namespace SceneDesc { struct Expr; }

// d(phi)/dt + conv(u, phi) = div(epsilon grad phi) - gamma phi + S for the registered scalars: a point-
// implicit reaction factor, then one tridiagonal sweep per axis, each the full dt.
class TransportSolver2D
{
public:
    static void InitializeTransportSolver2D(MAC *grid);
    static void Dispose();   // frees what Initialize allocated

    // Reads the scalars from gridAnt and the velocity from gridSol -- which must be the PROJECTED
    // field, so this runs after the pressure correction -- and writes the new scalars into gridSol.
    static void SolveTransportStep(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);

private:
    static Eigen::VectorXd low, mid, up, rhs, sol;
    static Eigen::VectorXd interm;   // the x sweep's output, read by the y sweep
    static Eigen::VectorXd staged;   // the reaction factor's output, read by the x sweep

    static void React(MAC *gridAnt, MAC *gridSol, int src, const ScalarTransport &tr, const SceneDesc::Expr *source);
    static void SweepX(MAC *gridAnt, MAC *gridSol, int handle, const ScalarTransport &tr,
                       const std::string &name, bool fromStaged);
    static void SweepY(MAC *gridSol, int handle, const ScalarTransport &tr, const std::string &name);

    static void FillRow(int t, double w, double velLow, double velHigh, TransportScheme scheme,
                        bool hasLow, bool hasHigh);
    static void CloseLowFace(double w, double vel, double dBoundary, TransportScheme scheme,
                             const std::string &name, double x, double y, int t);
    static void CloseHighFace(double w, double vel, double dBoundary, TransportScheme scheme,
                              const std::string &name, double x, double y, int t);
};

#endif
