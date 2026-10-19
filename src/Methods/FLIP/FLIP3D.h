// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "../../headers/Core/MAC.h"
#include <iosfwd>
#include "../../headers/Core/Particles.h"
#include "../../headers/Utils.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "math.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef FLIP3D_H
#define FLIP3D_H
// this desperately needs gpu paralelization for the particle update, to do!
// Textbook bridson implementation with density correction


class FLIP3D{
public:

    static int particlePerCell;
    static double alpha;
    static std::vector<std::vector<std::vector<std::vector<int>>>> SPACE_HASH; // 4D vector for 3D spatial hashing
    // Snapshot taken right after particle-to-grid, before diffusion and projection.
    // GridToParticle differences against this, so it is the FLIP delta base.
    static MAC gridAnt;
    // The viscous solve's input, the grid after particle-to-grid. Kept here so the steps never write the manager's
    // previous grid, which the residual compares against.
    static MAC gridWork;

    // Number of cells the velocity field is pushed outward past the fluid surface.
    static const int EXTRAPOLATION_LAYERS = 3;

    // Transfer kernel support radius, in cells.
    static constexpr double KERNEL_SUPPORT_CELLS = 1.0;
    static constexpr int KERNEL_CELLS = (int)(KERNEL_SUPPORT_CELLS + 0.9999);

public:
    static void InitializeFLIP(MAC* grid, ParticleSet* particles, double dt, double alpha = 1.0);
    static void Dispose();   // frees what Initialize allocated
    static void FLIP_StepBeforeProjection(MAC* advect, MAC* grid, ParticleSet* particles, double dt);

    // Wrapper functions for the manager
    static void FLIP_Advection(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time);
    static void FLIP_Diffusion(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time);
    static void FLIP_Momentum(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time);
    // After the pressure solver's projection (order 6): extrapolate the projected field into the air and
    // transfer it to the particles (order 7).
    static void FLIP_GridToParticle(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time);

    // Density correction (Kugelstadt et al. 2019), registered after the projection when FLIP_DENSITY_CORRECTION
    // is on: the particle density on the cells, a Poisson solve on the pressure matrix, and a position shift.
    static void DensityCorrection(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time);
    static void ComputeDensity(MAC* grid, const ParticleSet* particles);
    // The liquid volume one particle carries (seeded volume / particle count), and the relative density
    // rho/rho0 per cell, indexed (i*Nx + j)*Nz + k.
    static double particleVolume;
    static std::vector<double> density;
    static MAC gridShift;               // the position correction, -grad phi, on the faces

private:
    inline static double k(double x, double y, double z);
    inline static double h(double r);
    static void UpdateSpaceHash(const ParticleSet* particles);
    static void UpdateFluidCells(MAC* grid, const ParticleSet* particles);
    static void UpdateParticles(ParticleSet* particles, double dt, MAC* grid, MAC* advect);

    static void ExtrapolateVelocity(MAC* grid, int layers);
    static void ExtrapolateComponent(MAC* grid, MAC::Component component, int layers);

    // Scratch for ExtrapolateVelocity, sized once in InitializeFLIP and reused.
    static std::vector<double> exField, exSrc;
    static std::vector<char> exValid, exPinned, exCur;
    static void ParticleToGrid(MAC* grid, const ParticleSet* particles);
    static void GridToParticle(MAC* grid, ParticleSet* particles);

    // This function is only used if another part of the code must add force to the simulation
};

#endif