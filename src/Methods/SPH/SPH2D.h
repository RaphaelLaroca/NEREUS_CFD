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
#ifndef SPH2D_H
#define SPH2D_H

// SPH2D -- weakly compressible SPH (Monaghan 1994): density carried by continuity with delta-SPH diffusion,
// p = c0^2 (rho - rho0), the symmetric pressure gradient, Monaghan's artificial viscosity, a Wendland C2 kernel.
class SPH2D {
public:

    // The kernel buffer: the real particles' kinematics copied from the shared ParticleSet, then the
    // mirror ghosts after them, with SPH's own fields. Rebuilt every substep by BuildGhosts.
    struct KernelParticle
    {
        double x = 0.0, y = 0.0;
        double u = 0.0, v = 0.0;
        double d = 0.0;   // density
        double p = 0.0;   // pressure
        double fx = 0.0;  // force DENSITY: a = f / rho
        double fy = 0.0;
        double drho = 0.0; // continuity rate d(rho)/dt
    };
    static std::vector<KernelParticle> kernel;

    static double restDensity;   // rho_0, from DENSITY
    static double h;             // kernel support radius
    static double mass;          // m, identical for every particle (derived)
    static double spacing;       // seeding lattice pitch
    static int lastSubsteps;     // substeps taken inside the last AdvanceSPH, for telemetry

    static void InitializeSPH2D(ParticleSet* particles, int maxP);
    static void Dispose();   // frees what Initialize allocated
    static void AdvanceSPH(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time);

    static double GetPressureAt(double x, double y);


    static double GetMaxVelocity(const ParticleSet* particles);

    // Largest step this state is stable at (Monaghan): an acoustic+advective CFL on c0 plus the fastest
    // particle, and a force limit.
    static double StableTimestep(const ParticleSet* particles);

private:
    static void EmitGhost(int count, double gx, double gy, double gu, double gv, int src);
    // Uniform cell list of side h: neighbours can only lie in a particle's own cell and the eight around it.
    static int cellsX, cellsY;
    static std::vector<int> cellStart, cellEnd, particleIndex, cellOf;
    static void BuildCellList(int n);

    // ---- mirror ghost particles at the domain walls and the solids ----
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

    // Leapfrog carries velocity at half steps: the first substep of a run (step time 0) kicks by dt/2,
    // every later one by dt. Keyed on the clock, so a restored run needs no flag.
};

#endif
