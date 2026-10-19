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
#ifndef FLIP2D_H
#define FLIP2D_H

class FLIP2D{

    public:

    static int particlePerCell;
    static double alpha;
    static std::vector<std::vector<std::vector<int>>> SPACE_HASH; //this will be a bit inneficient for my sanity, there is no way im using a linked list for this while maintaining this whole code.
    // Snapshot taken right after particle-to-grid, before diffusion and projection.
    // GridToParticle differences against this, so it is the FLIP delta base.
    static MAC gridAnt;
    // The viscous solve's input, the grid after particle-to-grid. Kept here so the steps never write the manager's
    // previous grid, which the residual compares against.
    static MAC gridWork;


    // Number of cells the velocity field is pushed outward past the fluid surface.
    static const int EXTRAPOLATION_LAYERS = 3;

    public:
    static void InitializeFLIP(MAC* grid, ParticleSet* particles, double dt, double alpha = 1.0);
    static void Dispose();   // frees what Initialize allocated
    static void FLIP_StepBeforeProjection(MAC* advect, MAC* grid, ParticleSet* particles, double dt);
    // wrapper functions for the manager
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
    // The liquid area one particle carries (seeded area / particle count), and the relative density rho/rho0
    // per cell, indexed i*Nx + j.
    static double particleVolume;
    static std::vector<double> density;
    static MAC gridShift;               // the position correction, -grad phi, on the faces

    private:
    static double k(double x,double y);
    static double h(double r);

    static void UpdateSpaceHash(const ParticleSet* particles);
    static void UpdateFluidCells(MAC* grid, const ParticleSet* particles);
    static void UpdateParticles(ParticleSet* particles, double dt, MAC* grid, MAC* advect);


    // Fills the faces that carry no fluid with a value swept out from the ones that do.
    static void ExtrapolateVelocity(MAC* grid, int layers);

    static void ParticleToGrid(MAC* grid, const ParticleSet* particles);
    static void GridToParticle(MAC* grid, ParticleSet* particles);

    // this function is only used it another part of the code must add force to the simulation.


};

#endif