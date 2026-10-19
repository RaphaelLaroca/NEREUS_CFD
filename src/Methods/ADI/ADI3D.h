// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef ADI3D_H
#define ADI3D_H

#include "../../headers/Core/MAC.h"
#include "../../headers/Core/Definitions.h"
#include "../../headers/Utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class ADI3D
{
private:

    static MAC X_STEP_SOL;
    static MAC Y_STEP_SOL;
    static MAC Z_STEP_SOL;

    static double dh;

    //internal mirrors of the bc functions, used for the internal grids!
    static Vec3(*VelocityBorderFunction)(double, double, double,double);
    static double(*PressureFunction)(double, double, double,double);

// OPENMP functions
    //the reason these are programmed like this is because this is a copy and paste from the 2023 code
    //they work and this also makes parallel calls easy, although it is quite bug prone in case things change
    static void SolveADI_X_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_U_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_V_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

    static void SolveADI_X_W_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Y_W_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);
    static void SolveADI_Z_W_Step_OPENMP(MAC* gridAnt,MAC* gridSol,MAC* velocityField,double time);

public:
    static void InitializeADI(MAC* grid,double dt,Vec3(*VelocityBorderFunction)(double, double, double,double),double(*PressureFunction)(double, double, double,double));
    static void Dispose();   // frees what Initialize allocated

    static void SolveADIStep(MAC* gridAnt,MAC* gridSol,ParticleSet *particles, double time);

};
#endif
