// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TRANSPORTSOLVER3D_H
#define TRANSPORTSOLVER3D_H

#include "../Core/MAC.h"
#include "../Core/Definitions.h"
#include "Utils.h"

#include <Eigen/Dense>
#include <string>

namespace SceneDesc { struct Expr; }

// The 3D twin of TransportSolver2D: advection-diffusion-reaction of the registered cell-centred
// scalars, a point-implicit reaction factor and one tridiagonal sweep per axis, each advancing the full dt.
class TransportSolver3D
{
public:
    static void InitializeTransportSolver3D(MAC *grid);
    static void Dispose();   // frees what Initialize allocated

    // Reads the scalars from gridAnt and the velocity from gridSol -- which must be the PROJECTED
    // field, so this runs after the pressure correction -- and writes the new scalars into gridSol.
    static void SolveTransportStep(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);

private:
    static Eigen::VectorXd low, mid, up, rhs, sol;
    static Eigen::VectorXd intermX;   // the x sweep's output, read by the y sweep
    static Eigen::VectorXd intermY;   // the y sweep's output, read by the z sweep
    static Eigen::VectorXd staged;    // the reaction factor's output, read by the x sweep

    static void React(MAC *gridAnt, MAC *gridSol, int src, const ScalarTransport &tr, const SceneDesc::Expr *source);
    static void SweepX(MAC *gridAnt, MAC *gridSol, int handle, const ScalarTransport &tr,
                       const std::string &name, bool fromStaged);
    static void SweepY(MAC *gridSol, const ScalarTransport &tr, const std::string &name);
    static void SweepZ(MAC *gridSol, int handle, const ScalarTransport &tr, const std::string &name);

    static void CloseLowFace(double w, double vel, double dBoundary, TransportScheme scheme,
                             const std::string &name, double x, double y, double z, int t);
    static void CloseHighFace(double w, double vel, double dBoundary, TransportScheme scheme,
                              const std::string &name, double x, double y, double z, int t);
    static void FillRow(int t, double w, double velLow, double velHigh, TransportScheme scheme,
                        bool hasLow, bool hasHigh);
};

#endif
