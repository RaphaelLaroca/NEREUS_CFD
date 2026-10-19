// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef ADI2D_H
#define ADI2D_H

#include "../../headers/Core/MAC.h"
#include "../../headers/Core/Definitions.h"
#include "../../headers/Utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class ADI2D
{
private:

    static MAC X_STEP_SOL;
    static MAC Y_STEP_SOL;

    static double dh;

    static Vec2(*VelocityBorderFunction)(double, double, double);
    static double(*PressureFunction)(double, double, double);

// OPENMP functions
    static void SolveADI_X_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

public:
    static void InitializeADI2D(MAC* grid,double dt,Vec2(*VelocityBorderFunction)(double, double, double),double(*PressureFunction)(double, double, double));
    static void Dispose();   // frees what Initialize allocated

    static void SolveADIStep(MAC* gridAnt,MAC* gridSol,ParticleSet *particles, double time);

};
#endif
