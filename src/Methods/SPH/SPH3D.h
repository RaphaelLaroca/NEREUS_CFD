// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "../../headers/Core/Definitions.h"
#include "../../headers/Core/Particles.h"
#include <algorithm>
#include <iosfwd>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef SPH3D_H
#define SPH3D_H

// SPH3D, weakly compressible SPH, the 3D twin of SPH2D.
// The most basic and simple SPH implementation, it handles shocks badly, but it does produce good
// results in gravity-oriented flow
class SPH3D {
public:

    // The kernel buffer: the real particles' kinematics copied from the shared ParticleSet, then the
    // mirror ghosts after them, with SPH's own fields. Rebuilt every substep by BuildGhosts.
    struct KernelParticle
    {
        double x = 0.0, y = 0.0, z = 0.0;
        double u = 0.0, v = 0.0, w = 0.0;
        double d = 0.0;   // density
        double p = 0.0;   // pressure
        double fx = 0.0;  // force DENSITY: a = f / rho
        double fy = 0.0;
        double fz = 0.0;
        double drho = 0.0; // continuity rate d(rho)/dt
    };
    static std::vector<KernelParticle> kernel;

    static double restDensity;   // rho_0, from DENSITY
    static double h;             // kernel support radius
    static double mass;          // m, identical for every particle (derived)
    static double spacing;       // seeding lattice pitch
    static int lastSubsteps;

    static void InitializeSPH3D(ParticleSet* particles, int maxP);
    static void Dispose();   // frees what Initialize allocated
    static void AdvanceSPH(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time);



    static double GetMaxVelocity(const ParticleSet* particles);

    // Public for the same reason as SPH2D's: SimulationManager's ADAPTATIVE_TIMESTEP branch
    // sizes the outer step with it so it agrees with AdvanceSPH's substepping.
    static double StableTimestep(const ParticleSet* particles);

private:
    static void EmitGhost(int count, double gx, double gy, double gz, double gu, double gv, double gw, int src);
    static int cellsX, cellsY, cellsZ;
    static std::vector<int> cellStart, cellEnd, particleIndex, cellOf;
    static void BuildCellList(int n);

    static int ghostCount;
    static std::vector<int> ghostSource;
    static void BuildGhosts(const ParticleSet* particles);
    static void SyncGhostFields(int count);

    static double Wendland(double r);          // Wendland C2, support h
    static double WendlandGradMag(double r);   // dW/dr, negative
    static void HydrostaticDensity(ParticleSet* particles);   // fills particles->density
    static void LoadDensityPressure(const ParticleSet* particles);
    static void ComputeDensityRate(int count);
    static void ComputeForces(int count);
    static void DensityPhase(const ParticleSet* particles);        // density and pressure, ghosts synced
    static void ForcePhase(int count);          // forces
    static void DepositToGrid(MAC* grid);       // pressure and FLUID/EMPTY onto the run's grid, for telemetry
    static void Integrate(ParticleSet* particles, double dt, bool firstKick);

};

#endif
