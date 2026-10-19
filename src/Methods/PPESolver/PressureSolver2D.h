// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "../../headers/Core/MAC.h"
#include "../../headers/Core/Definitions.h"
#include "../../headers/Utils.h"
#include "PressureSettings.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>

#ifndef PRESSURE_SOLVER2D_H
#define PRESSURE_SOLVER2D_H
using Eigen::VectorXd;

// The pressure Poisson solve on a 2D MAC grid, shared by every grid method: the method picks the scheme
// (Chorin or Incremental), GPU_ACCELERATION the backend; the matrix is reused while the layout stands.
class PressureSolver2D
{
private:
    static int Nx, Ny;
    // Matrix row of each cell, -1 for a cell that is not an unknown.
    static std::vector<int> IDP;
    static PressureSystem SYSTEM;

    // (Re)builds IDP and the matrix when the grid's cell layout differs from the one the matrix was built for.
    static void EnsureSystem(MAC *grid);
    static void AssembleSystem(MAC *grid);
    // Chorin and Incremental are this with a different stopping rule and guess.
    static void SolveScheme(MAC *gridAnt, MAC *grid, bool incremental);

    static double DirichletSum(MAC *grid, MAC *old, int i, int j);
    // Incremental projection: ||b + A p^n||, the right-hand side the full pressure would have. The solve's
    // tolerance is taken against it, since b itself (the increment's) vanishes at steady state.
    static double IncrementReferenceNorm(MAC *gridAnt, const VectorXd &b);

    // The constant level of a solution vector (indexed like the matrix): the one place it is set.
    static void LevelPressure(MAC *grid, VectorXd &p);
    // Incremental: the solve returned phi = p^{n+1} - p^n; set p = p^n + phi, levelled.
    static void AccumulatePressureIncrement(MAC *gridAnt, MAC *grid);

    static bool WarmStartEnabled(const MAC *grid);
    // Warm start: x0 is the previous step's pressure, read from gridAnt, so the solver keeps no state.
    // Chorin only (under Incremental the unknown is the increment, which no grid holds); false means solve from zero.
    static bool BuildWarmStartGuess(MAC *gridAnt, VectorXd &x0);

public:
    // Chooses the backend, registers SolveChorin or SolveIncremental as the "pressure" step and Project as the
    // "projection" step, and assembles the matrix.
    static void InitializePressureSolver(MAC *grid, PressureScheme scheme = PRESSURE_CHORIN);
    static void DestroyPressureSolver();
    static void ReleasePressureSolver();   // frees the system and the AMGX handle, leaves the AMGX library up

    // The two solves: p, or the increment phi = p^{n+1} - p^n (gridAnt carries p^n).
    static void SolveChorin(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);
    static void SolveIncremental(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);
    // Whichever of the two InitializePressureSolver chose, for a caller that steps the solver itself.
    static void SolvePressure(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);

    // Solves L x = f on the grid's fluid cells (indexed like the pressure, x = 0 at air and pressure cells, no
    // flux at solids) with the pressure matrix, rebuilt only if the layout moved. FLIP's density correction.
    static bool SolveLaplacian(MAC *grid, const VectorXd &f, VectorXd &x);
    // Rows of the current matrix, i.e. the length of a vector SolveLaplacian takes.
    static int SystemSize() { return (int)SYSTEM.A.rows(); }
    // The pressure step's counter: iterations of the last solve, read by the manager after the step.
    static int LastIterations() { return PRESSURE_TELEMETRY.lastIterations; }

    // The projection step, the same for every grid method: ProjectPressure, and under Incremental p = p^n + phi.
    static void Project(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);
    // u -= (dt/rho) grad p on every face that is not a wall, with the pressure on the grid.
    static void ProjectPressure(MAC *grid);

    static inline int GetIDP(int i, int j) { return IDP[i * Nx + j]; }
};

#endif
