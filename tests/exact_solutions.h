// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TESTS_EXACT_SOLUTIONS_H
#define TESTS_EXACT_SOLUTIONS_H

// Closed-form Navier-Stokes solutions the harnesses score against. Not part of the application.

#include <cmath>

// Taylor-Green vortex, with nu = SIMULATION.EPS and rho = 1.
inline double TaylorGreenU(double x, double y, double t, double nu)
{ return -std::cos(x) * std::sin(y) * std::exp(-2.0 * nu * t); }
inline double TaylorGreenV(double x, double y, double t, double nu)
{ return  std::sin(x) * std::cos(y) * std::exp(-2.0 * nu * t); }
inline double TaylorGreenP(double x, double y, double t, double nu)
{ return -0.25 * (std::cos(2.0 * x) + std::cos(2.0 * y)) * std::exp(-4.0 * nu * t); }

// Ethier & Steinman, "Exact fully 3D Navier-Stokes solutions for benchmarking", International
// Journal for Numerical Methods in Fluids 19(5):369-375, 1994.
inline double ES_A = M_PI / 4.0;   // the paper's benchmark values
inline double ES_D = M_PI / 2.0;

inline double EthierSteinmanU(double x, double y, double z, double t, double nu)
{
    const double a = ES_A, d = ES_D;
    return -a * (std::exp(a * x) * std::sin(a * y + d * z) +
                 std::exp(a * z) * std::cos(a * x + d * y)) * std::exp(-nu * d * d * t);
}
inline double EthierSteinmanV(double x, double y, double z, double t, double nu)
{
    const double a = ES_A, d = ES_D;
    return -a * (std::exp(a * y) * std::sin(a * z + d * x) +
                 std::exp(a * x) * std::cos(a * y + d * z)) * std::exp(-nu * d * d * t);
}
inline double EthierSteinmanW(double x, double y, double z, double t, double nu)
{
    const double a = ES_A, d = ES_D;
    return -a * (std::exp(a * z) * std::sin(a * x + d * y) +
                 std::exp(a * y) * std::cos(a * z + d * x)) * std::exp(-nu * d * d * t);
}
inline double EthierSteinmanP(double x, double y, double z, double t, double nu)
{
    const double a = ES_A, d = ES_D;
    return -(a * a / 2.0) * (std::exp(2 * a * x) + std::exp(2 * a * y) + std::exp(2 * a * z)
        + 2.0 * std::sin(a * x + d * y) * std::cos(a * z + d * x) * std::exp(a * (y + z))
        + 2.0 * std::sin(a * y + d * z) * std::cos(a * x + d * y) * std::exp(a * (z + x))
        + 2.0 * std::sin(a * z + d * x) * std::cos(a * y + d * z) * std::exp(a * (x + y)))
        * std::exp(-2.0 * nu * d * d * t);
}

#endif
