// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "FLIP2D.h"
#include "FLIPSettings.h"
#include "../../headers/Core/Scene.h"
#include <algorithm>
#include <random>
#include <iostream>
#include "../PPESolver/PressureSolver2D.h"
#include "DiffusionSolver2D.h"
int FLIP2D::particlePerCell = 0.0;
std::vector<std::vector<std::vector<int>>> FLIP2D::SPACE_HASH = std::vector<std::vector<std::vector<int>>>();
MAC FLIP2D::gridAnt = MAC();
MAC FLIP2D::gridWork = MAC();
double FLIP2D::alpha = 0.0;
double FLIP2D::particleVolume = 0.0;
std::vector<double> FLIP2D::density;
MAC FLIP2D::gridShift = MAC();

// A seeding offset inside the particle's own sub-cell; zero, and no random draw, when jitter is off.
static double SeedJitter(std::mt19937 &rng, std::uniform_real_distribution<double> &dist, double amp)
{
    return (amp > 0.0) ? amp * dist(rng) : 0.0;
}

void FLIP2D::InitializeFLIP(MAC *grid, ParticleSet *particles, double dt, double alpha)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    double dh = SIMULATION.dh;
    FLIP2D::alpha = alpha;
    const int maxParticle = SIMULATION.Nx * SIMULATION.Ny * FLIP_SETTINGS.particlesPerCell;
    FLIP2D::SPACE_HASH.clear();
    particles->data.assign(maxParticle, Particle());
    Particle *ps = particles->data.data();
    int particleCount = 0;
    int seededCells = 0;

    grid->SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    gridAnt.InitializeFromGrid(*grid, SIMULATION.time);
    gridWork.InitializeFromGrid(*grid, SIMULATION.time);

    double *offset;
    int n = sqrt(FLIP_SETTINGS.particlesPerCell);
    FLIP2D::particlePerCell = FLIP_SETTINGS.particlesPerCell;
    offset = (double *)malloc(sizeof(double) * n);

    // Evenly spaced offsets within the cell
    for (int i = 0; i < n; i++)
    {
        offset[i] = (i + 0.5) / (double)n; // Centers particles in subdivided cell regions
    }

    // Seeding on an exact lattice leaves the liquid LAYERED, and the layers survive wherever the
    // flow is slow: a settled film keeps them.
    std::mt19937 seedRng(20260912u);
    std::uniform_real_distribution<double> seedJitterDist(-0.5, 0.5);
    const double seedJitterAmp =
        std::min(std::max(SIMULATION.particleOptions.seedJitter, 0.0), 1.0) * (dh / (double)n);

    // The scene's `region fluid` entries decide where the liquid starts; without one nothing is seeded.
    if (SCENE.loaded && SCENE.HasFluidRegions())
    {
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
            {
                if (grid->GetSolid(i, j) == SOLID_CELL)
                    continue;
                // Sample the cell centre: a particle block belongs to the cell it fills,
                // not to whichever corner happens to fall inside the region.
                if (!SCENE.IsFluidSeedAt(grid->FacePos(MAC::X, 0) + (j + 0.5) * dh,
                                         grid->FacePos(MAC::Y, 0) + (i + 0.5) * dh, 0.0))
                    continue;
                seededCells++;

                for (int io = 0; io < n; io++)
                    for (int jo = 0; jo < n; jo++)
                    {
                        if (particleCount >= maxParticle)
                        {
                            std::cerr << "WARNING: FLIP2D seeding hit maxParticle ("
                                      << maxParticle << "), truncating" << std::endl;
                            break;
                        }
                        Particle p;
                        p.x = grid->FacePos(MAC::X, 0) + j * dh + offset[jo] * dh + SeedJitter(seedRng, seedJitterDist, seedJitterAmp);
                        p.y = grid->FacePos(MAC::Y, 0) + i * dh + offset[io] * dh + SeedJitter(seedRng, seedJitterDist, seedJitterAmp);
                        p.isActive = true;
                        ps[particleCount] = p;
                        particleCount++;
                    }
            }

        std::cout << "FLIP2D: seeded " << particleCount
                  << " particles from the scene's fluid regions" << std::endl;
    }
    else
        std::cerr << "FLIP2D: the scene has no `region fluid`, so no particles were seeded" << std::endl;

    particles->count = particleCount;
    // The liquid each particle stands for, from what was seeded: rho/rho0 is then exactly 1 on the lattice.
    particleVolume = particleCount > 0 ? seededCells * dh * dh / particleCount : 0.0;
    density.assign((size_t)Nx * Ny, 0.0);
    if (FLIP_SETTINGS.densityCorrection)
        gridShift.InitializeGrid(SIMULATION.domain, true);
    // initialize the space hash structure
    for (int i = 0; i < Ny; i++)
    {
        SPACE_HASH.push_back(std::vector<std::vector<int>>());
        for (int j = 0; j < Nx; j++)
        {
            SPACE_HASH.at(i).push_back(std::vector<int>(FLIP_SETTINGS.particlesPerCell)); // preallocatigon of some values
        }
    }

    UpdateSpaceHash(particles);

    // falling block for now allocate the space hash create all particles
    free(offset);

    DiffusionSolver2D::InitializeDiffusionSolver(grid);
    // Split so the telemetry separates the two halves: the particle phase (advection, transfer,
    // extrapolation) from the grid phase (the viscous solve and the body force).
    RegisterSimulationStep(FLIP2D::FLIP_Advection, "particle_advection", 0);
    RegisterSimulationStep(FLIP2D::FLIP_Diffusion, "grid_diffusion", 2);
    RegisterSimulationStep(FLIP2D::FLIP_GridToParticle, "grid_to_particle", 7);
    if (FLIP_SETTINGS.densityCorrection)
        RegisterSimulationStep(FLIP2D::DensityCorrection, "density_correction", 8);
}

// p^n, kept for FLIP_VISCOUS_PRESSURE: gridAnt holds it at the start of the step and loses it to the copy
// in FLIP_Advection (the particle half: advect, mark, transfer, extrapolate).
static std::vector<double> PREV_PRESSURE;

// False on a step with no previous pressure (the first, or a restart from rest): gravity alone inside the viscous
// solve would carve a boundary layer out of it at the walls, so that step adds the body force after the solve.
static bool HAS_PREV_PRESSURE = false;

static void SavePreviousPressure(MAC *g)
{
    PREV_PRESSURE.assign((size_t)g->Nx * g->Ny, 0.0);
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx; j++)
            PREV_PRESSURE[(size_t)i * g->Nx + j] = g->GetP(i, j);
    HAS_PREV_PRESSURE = false;
    for (double p : PREV_PRESSURE)
        if (p != 0.0) { HAS_PREV_PRESSURE = true; break; }
}

// p^n on this step's cells with the air at 0, the projection's own Dirichlet value, so the projection removes
// its gradient exactly.
static inline double PreviousPressure(MAC *g, int i, int j)
{
    return g->GetSolid(i, j) == EMPTY_CELL ? 0.0 : PREV_PRESSURE[(size_t)i * g->Nx + j];
}

// u += c grad p^n on every face the projection corrects.
static void AddPreviousPressureGradient(MAC *g, double c)
{
    for (int i = 1; i < g->Ny - 1; i++)
        for (int j = 1; j < g->Nx; j++)
            if (g->FaceGate(MAC::U, i, j) != GateState::WALL)
                g->SetU(i, j, g->GetU(i, j) + c * (PreviousPressure(g, i, j) - PreviousPressure(g, i, j - 1)) /
                                                  g->FaceGap(MAC::X, j - 1));
    for (int i = 1; i < g->Ny; i++)
        for (int j = 1; j < g->Nx - 1; j++)
            if (g->FaceGate(MAC::V, i, j) != GateState::WALL)
                g->SetV(i, j, g->GetV(i, j) + c * (PreviousPressure(g, i, j) - PreviousPressure(g, i - 1, j)) /
                                                  g->FaceGap(MAC::Y, i - 1));
}

void FLIP2D::FLIP_Advection(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    if (FLIP_SETTINGS.physicalDiffusion && FLIP_SETTINGS.viscousPressure)
        SavePreviousPressure(gridAnt);
    FLIP2D::FLIP_StepBeforeProjection(gridAnt, gridSol, particles, SIMULATION.dt);
    gridWork.CopyGrid(*gridSol);
}

// The GRID half: the viscous solve (optional) and the body force, which must be applied whether or not the
// viscous solve ran -- and after it, so the no-slip walls cannot smooth the gravity impulse.
void FLIP2D::FLIP_Diffusion(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    gridSol->SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    // The net driving force, body force minus grad p^n, inside the solve (which reads gridWork), grad p^n added back.
    if (FLIP_SETTINGS.physicalDiffusion && FLIP_SETTINGS.viscousPressure && HAS_PREV_PRESSURE)
    {
        const double c = SIMULATION.dt / SIMULATION.RHO;
        gridSol->AddForce(time, SIMULATION.dt);
        AddPreviousPressureGradient(gridSol, -c);
        gridSol->EnforceVelocityBC(SIMULATION.time);
        gridWork.CopyFlowFields(*gridSol);
        if (FLIP_SETTINGS.diffusionSolver == DIFFUSION_SOLVERS::IMPLICIT)
            DiffusionSolver2D::SolveDiffusion_Eigen(&gridWork, gridSol);
        else
            DiffusionSolver2D::SolveDiffusion_ADI(&gridWork, gridSol);
        AddPreviousPressureGradient(gridSol, c);
        gridSol->EnforceVelocityBC(SIMULATION.time);
        return;
    }
    if (FLIP_SETTINGS.physicalDiffusion)
    {
        gridSol->EnforceVelocityBC(SIMULATION.time);
        if(FLIP_SETTINGS.diffusionSolver == DIFFUSION_SOLVERS::IMPLICIT){
            DiffusionSolver2D::SolveDiffusion_Eigen(&gridWork, gridSol);
        }
        else if(FLIP_SETTINGS.diffusionSolver == DIFFUSION_SOLVERS::ADI){
            DiffusionSolver2D::SolveDiffusion_ADI(&gridWork, gridSol);
        }
        gridSol->EnforceVelocityBC(SIMULATION.time);
    }

    // The body force is applied here in BOTH branches, after any viscous solve.
    gridSol->AddForce(time, SIMULATION.dt);

    gridSol->EnforceVelocityBC(SIMULATION.time);
}

// Both halves in order, for harnesses that drive a whole momentum step.
void FLIP2D::FLIP_Momentum(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    FLIP2D::FLIP_Advection(gridAnt, gridSol, particles, time);
    FLIP2D::FLIP_Diffusion(gridAnt, gridSol, particles, time);
}



// Sweeps a scalar face field outward from the cells flagged valid.
static void ExtrapolateField(std::vector<std::vector<double>> &f,
                             std::vector<std::vector<char>> &valid,
                             int rows, int cols, int layers)
{
    for (int layer = 0; layer < layers; layer++)
    {
        std::vector<std::vector<double>> src = f;
        std::vector<std::vector<char>> cur = valid;
        bool filledAny = false;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (cur[i][j])
                    continue;

                double sum = 0.0;
                int n = 0;
                if (i > 0 && cur[i - 1][j])        { sum += src[i - 1][j]; n++; }
                if (i < rows - 1 && cur[i + 1][j]) { sum += src[i + 1][j]; n++; }
                if (j > 0 && cur[i][j - 1])        { sum += src[i][j - 1]; n++; }
                if (j < cols - 1 && cur[i][j + 1]) { sum += src[i][j + 1]; n++; }

                if (n > 0)
                {
                    f[i][j] = sum / (double)n;
                    valid[i][j] = 1;
                    filledAny = true;
                }
            }
        }

        if (!filledAny)
            break;
    }
}

void FLIP2D::ExtrapolateVelocity(MAC *grid, int layers)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;

    // A face is a real value when it touches fluid, or when it is pinned against a
    // solid. Everything else is air and gets swept into from those.
    std::vector<std::vector<double>> u(Ny, std::vector<double>(Nx + 1, 0.0));
    std::vector<std::vector<char>> uValid(Ny, std::vector<char>(Nx + 1, 0));
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            u[i][j] = grid->GetU(i, j);
            int right = (j < Nx) ? grid->GetSolid(i, j) : SOLID_CELL;
            int left = (j > 0) ? grid->GetSolid(i, j - 1) : SOLID_CELL;
            uValid[i][j] = (right == FLUID_CELL || left == FLUID_CELL ||
                            right == SOLID_CELL || left == SOLID_CELL)
                               ? 1
                               : 0;
        }
    }

    std::vector<std::vector<double>> v(Ny + 1, std::vector<double>(Nx, 0.0));
    std::vector<std::vector<char>> vValid(Ny + 1, std::vector<char>(Nx, 0));
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            v[i][j] = grid->GetV(i, j);
            int above = (i < Ny) ? grid->GetSolid(i, j) : SOLID_CELL;
            int below = (i > 0) ? grid->GetSolid(i - 1, j) : SOLID_CELL;
            vValid[i][j] = (above == FLUID_CELL || below == FLUID_CELL ||
                            above == SOLID_CELL || below == SOLID_CELL)
                               ? 1
                               : 0;
        }
    }

    ExtrapolateField(u, uValid, Ny, Nx + 1, layers);
    ExtrapolateField(v, vValid, Ny + 1, Nx, layers);

    // Only the air faces are written back; the ones that already carried fluid or a
    // wall condition keep exactly what they had.
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx + 1; j++)
        {
            int right = (j < Nx) ? grid->GetSolid(i, j) : SOLID_CELL;
            int left = (j > 0) ? grid->GetSolid(i, j - 1) : SOLID_CELL;
            if (right != FLUID_CELL && left != FLUID_CELL &&
                right != SOLID_CELL && left != SOLID_CELL)
                grid->SetU(i, j, u[i][j]);
        }

    for (int i = 0; i < Ny + 1; i++)
        for (int j = 0; j < Nx; j++)
        {
            int above = (i < Ny) ? grid->GetSolid(i, j) : SOLID_CELL;
            int below = (i > 0) ? grid->GetSolid(i - 1, j) : SOLID_CELL;
            if (above != FLUID_CELL && below != FLUID_CELL &&
                above != SOLID_CELL && below != SOLID_CELL)
                grid->SetV(i, j, v[i][j]);
        }
}

void FLIP2D::ParticleToGrid(MAC *grid, const ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = grid->FacePos(MAC::X, 0), oy = grid->FacePos(MAC::Y, 0);
    const Particle *ps = particles->data.data();
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    double dh = SIMULATION.dh;

    // Create data structures for u component
    std::vector<std::vector<double>> uW(Ny, std::vector<double>(Nx + 1, 0.0));
    std::vector<std::vector<double>> uWeights(Ny, std::vector<double>(Nx + 1, 0.0));

    // Create data structures for v component
    std::vector<std::vector<double>> vW(Ny + 1, std::vector<double>(Nx, 0.0));
    std::vector<std::vector<double>> vWeights(Ny + 1, std::vector<double>(Nx, 0.0));

// Main parallel region to avoid thread creation/destruction overhead
#pragma omp parallel
    {
// Process u component (horizontal velocity)
#pragma omp for
        for (int i = 1; i < Ny - 1; i++)
        {
            for (int j = 1; j < Nx; j++)
            {
                std::vector<int> closeParticles;

                // Gather nearby particles
                for (const auto &p : SPACE_HASH[i][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i][j - 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i + 1][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i + 1][j - 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j - 1])
                    closeParticles.push_back(p);

                // Calculate weights and weighted velocities
                for (const auto &p : closeParticles)
                {
                    double weight = k((ps[p].x - ox) - j * dh, (ps[p].y - oy) - (dh * i + dh / 2.0));
                    uWeights[i][j] += weight;
                    uW[i][j] += ps[p].u * weight;
                }
            }
        }

// Process v component (vertical velocity)
#pragma omp for
        for (int i = 1; i < Ny; i++)
        {
            for (int j = 1; j < Nx - 1; j++)
            {
                std::vector<int> closeParticles;

                // Gather nearby particles
                for (const auto &p : SPACE_HASH[i][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i][j - 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j - 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i][j + 1])
                    closeParticles.push_back(p);
                for (const auto &p : SPACE_HASH[i - 1][j + 1])
                    closeParticles.push_back(p);

                // Calculate weights and weighted velocities
                for (const auto &p : closeParticles)
                {
                    double weight = k((ps[p].x - ox) - (dh * j + dh / 2.0), (ps[p].y - oy) - (dh * i));
                    vWeights[i][j] += weight;
                    vW[i][j] += ps[p].v * weight;
                }
            }
        }

// Normalize and set u velocity values
#pragma omp for
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx + 1; j++)
            {
                // Set values below tolerance to zero
                if (std::abs(uW[i][j]) < 1e-12)
                    uW[i][j] = 0.0;
                if (std::abs(uWeights[i][j]) < 1e-12)
                    uWeights[i][j] = 0.0;

                // Divide velocity by weight where weight is not zero
                if (uWeights[i][j] != 0.0)
                {
                    grid->SetU(i, j, uW[i][j] / uWeights[i][j]);
                }
                else
                {
                    grid->SetU(i, j, 0.0);
                }
            }
        }

// Normalize and set v velocity values
#pragma omp for
        for (int i = 0; i < Ny + 1; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                // Set values below tolerance to zero
                if (std::abs(vW[i][j]) < 1e-12)
                    vW[i][j] = 0.0;
                if (std::abs(vWeights[i][j]) < 1e-12)
                    vWeights[i][j] = 0.0;

                // Divide velocity by weight where weight is not zero
                if (vWeights[i][j] != 0.0)
                {
                    grid->SetV(i, j, vW[i][j] / vWeights[i][j]);
                }
                else
                {
                    grid->SetV(i, j, 0.0);
                }
            }
        }
    } // End of parallel region
}

void FLIP2D::GridToParticle(MAC *grid, ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = grid->FacePos(MAC::X, 0), oy = grid->FacePos(MAC::Y, 0);
    Particle *ps = particles->data.data();
    const int particleCount = particles->count;
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    double dh = SIMULATION.dh;

    // Calculate the velocity difference LATER
    std::vector<std::vector<double>> du(Ny, std::vector<double>(Nx + 1, 0.0));
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            du[i][j] = grid->GetU(i, j) - gridAnt.GetU(i, j);
        }
    }

    // u velocity component
    std::vector<double> uW(particleCount, 0.0);  // weight times the velocity
    std::vector<double> duW(particleCount, 0.0); // weight times the velocity difference
    std::vector<double> wu(particleCount, 0.0);  // weight for each particle

    // Each iteration only touches its own particle and its own accumulator slot, so there is
    // nothing to guard here.
#pragma omp parallel for
    for (int p = 0; p < particleCount; p++)
    {
        // we check only our nearby faces, for u, it is the to
        int pJ = (ps[p].x - ox) / dh; // by the logic we are using, this should never segfault, but if it does, it is this 100%
        int pI = ((ps[p].y - oy) + (dh / 2.0)) / dh;
        // Both ends have to be clamped: the hat kernel reaches one cell back, so a
        // particle in the first row would otherwise index row -1.
        int pIbeg = std::max(pI - 1, 0);
        int pJbeg = std::max(pJ - 1, 0);
        int pIend = std::min(pI + 2, Ny);
        int pJend = std::min(pJ + 2, Nx + 1);

        for (int i = pIbeg; i < pIend; i++)
        {
            for (int j = pJbeg; j < pJend; j++)
            {
                double weight = k((ps[p].x - ox) - j * dh, (ps[p].y - oy) - (i * dh + dh / 2.0));
                wu[p] += weight;
                uW[p] += grid->GetU(i, j) * weight;
                duW[p] += du[i][j] * weight;
            }
        }

        // Set values below tolerance to zero
        if (std::abs(uW[p]) < 1e-12)
            uW[p] = 0.0;
        if (std::abs(wu[p]) < 1e-12)
            wu[p] = 0.0;

        // Update particle velocity using FLIP2D/PIC blend
        if (wu[p] != 0.0)
        {
            ps[p].u = (uW[p] / wu[p]) * (1.0 - alpha) + (ps[p].u + (duW[p] / wu[p])) * alpha;
        }
    }

    // Calculate the velocity difference for v component
    std::vector<std::vector<double>> dv(Ny + 1, std::vector<double>(Nx, 0.0));
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            dv[i][j] = grid->GetV(i, j) - gridAnt.GetV(i, j);
        }
    }

    // v velocity component
    std::vector<double> vW(particleCount, 0.0);  // weight times the velocity
    std::vector<double> dvW(particleCount, 0.0); // weight times the velocity difference
    std::vector<double> wv(particleCount, 0.0);  // weight for each particle

#pragma omp parallel for
    for (int p = 0; p < particleCount; p++)
    {

        int pJ = ((ps[p].x - ox) + (dh / 2.0)) / dh; // by the logic we are using, this should never segfault, but if it does, it is this 100%
        int pI = ((ps[p].y - oy)) / dh;

        int pIbeg = std::max(pI - 1, 0);
        int pJbeg = std::max(pJ - 1, 0);
        int pIend = std::min(pI + 2, Ny + 1);
        int pJend = std::min(pJ + 2, Nx);

        for (int i = pIbeg; i < pIend; i++)
        {
            for (int j = pJbeg; j < pJend; j++)
            {
                double weight = k((ps[p].x - ox) - (j * dh + dh / 2.0), (ps[p].y - oy) - (i * dh));
                wv[p] += weight;
                vW[p] += grid->GetV(i, j) * weight;
                dvW[p] += dv[i][j] * weight;
            }
        }

        // Set values below tolerance to zero
        if (std::abs(vW[p]) < 1e-12)
            vW[p] = 0.0;
        if (std::abs(wv[p]) < 1e-12)
            wv[p] = 0.0;

        // Update particle velocity using FLIP2D/PIC blend
        if (wv[p] != 0.0)
        {
            ps[p].v = (vW[p] / wv[p]) * (1.0 - alpha) + (ps[p].v + (dvW[p] / wv[p])) * alpha;
        }
    }
}

void FLIP2D::UpdateParticles(ParticleSet *particles, double dt, MAC *grid, MAC *advect)
{
    Particle *ps = particles->data.data();
    const int particleCount = particles->count;
    for (int p = 0; p < particleCount; p++)
    {
        Particle &part = ps[p];

        // Substepped so that no single move covers much more than a cell.
        const int steps = Particles::SubstepCount(Particles::Speed(part), dt);
        const double h = dt / steps;

        for (int s = 0; s < steps; s++)
        {
            const Particle before = part;

            // First substep integrates from the particle's own velocity, exactly as the single-step
            // form did.
            const double k1x = (s == 0) ? part.u : advect->ComponentAt(MAC::U, part.x, part.y);
            const double k1y = (s == 0) ? part.v : advect->ComponentAt(MAC::V, part.x, part.y);

            const double midx = part.x + k1x * h / 2.0;
            const double midy = part.y + k1y * h / 2.0;
            part.x += advect->ComponentAt(MAC::U, midx, midy) * h;
            part.y += advect->ComponentAt(MAC::V, midx, midy) * h;

            Particles::ResolveContacts(grid, before, part);
        }
    }
}

void FLIP2D::UpdateSpaceHash(const ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = gridAnt.FacePos(MAC::X, 0), oy = gridAnt.FacePos(MAC::Y, 0);
    const Particle *ps = particles->data.data();
    const int particleCount = particles->count;
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    // cleaning up
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            SPACE_HASH.at(i).at(j).clear();
        }
    }

    int i, j;
    double dh = SIMULATION.dh;
    for (int p = 0; p < particleCount; p++)
    {
        i = (ps[p].y - oy) / dh;
        j = (ps[p].x - ox) / dh; // this implicit downmcast takes care of thhe floor

        // Clamp, the way UpdateFluidCells already does.
        i = std::min(std::max(i, 0), Ny - 1);
        j = std::min(std::max(j, 0), Nx - 1);

        SPACE_HASH[i][j].push_back(p);

        // pushes to the list
    }
}
void FLIP2D::UpdateFluidCells(MAC *grid, const ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = grid->FacePos(MAC::X, 0), oy = grid->FacePos(MAC::Y, 0);
    const Particle *ps = particles->data.data();
    const int particleCount = particles->count;
    grid->ResetFluidCells();

    int i, j;
    double dh = SIMULATION.dh;
    for (int p = 0; p < particleCount; p++)
    {
        i = (ps[p].y - oy) / dh;
        j = (ps[p].x - ox) / dh; // this implicit downmcast takes care of thhe floor
        i = i > (SIMULATION.Ny - 1) ? SIMULATION.Ny - 1 : i;
        j = j > (SIMULATION.Nx - 1) ? SIMULATION.Nx - 1 : j;

        if (grid->GetSolid(i, j) == EMPTY_CELL)
        {
            grid->SetSolid(i, j, FLUID_CELL);
        }

        // check all neightboors to no have problems
    }
    // check for holes, not needed usually if we have high particle per cell count > 16

    for (int i = 1; i < grid->Ny - 1; i++)
    {
        for (int j = 1; j < grid->Nx - 1; j++)
        {

            // FLUID neighbours only.
            int fluidNeighbors = 0;
            if (grid->GetSolid(i, j + 1) == FLUID_CELL)
                fluidNeighbors++;
            if (grid->GetSolid(i, j - 1) == FLUID_CELL)
                fluidNeighbors++;
            if (grid->GetSolid(i + 1, j) == FLUID_CELL)
                fluidNeighbors++;
            if (grid->GetSolid(i - 1, j) == FLUID_CELL)
                fluidNeighbors++;

            if (grid->GetSolid(i, j) == EMPTY_CELL && fluidNeighbors >= 3)
            {
                grid->SetSolid(i, j, FLUID_CELL);
            }

            // update the grid solid mask for the pressure step, rewrite a bit of thhe pressure so
            // that it remakes the pressure matrix each step
        }
    }
}

double FLIP2D::h(double r)
{
    r = std::abs(r); // Kernel is symmetric

    if (r < 1.0)
    {
        return 1.0 - r; // Linear (Hat) kernel
    }
    else
    {
        return 0.0;
    }
}

// 2D kernel using tensor product of 1D kernels
double FLIP2D::k(double x, double y)
{
    return h(x / SIMULATION.dh) * h(y / SIMULATION.dh);
}

void FLIP2D::FLIP_StepBeforeProjection(MAC *advect, MAC *grid, ParticleSet *particles, double dt)
{
    grid->SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    // impose boundary
    // change the external force later
    grid->EnforceVelocityBC(SIMULATION.time);

    UpdateParticles(particles, dt, grid, advect);

    UpdateFluidCells(grid, particles);
    UpdateSpaceHash(particles);

    ParticleToGrid(grid, particles); // copy the grid here for flip

    // Particle-to-grid leaves a hard zero on every face that no particle reached. Sweep the field
    // outward before anything reads it.
    ExtrapolateVelocity(grid, EXTRAPOLATION_LAYERS);

    // The boundary conditions, and no flux through an obstacle face (particle-to-grid writes one there).
    grid->EnforceVelocityBC(SIMULATION.time);

    gridAnt.CopyGrid(*grid);

}
void FLIP2D::FLIP_GridToParticle(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    // The projection only touches faces that border fluid, so without this sweep the air band would still hold
    // pre-projection values when the next step advects through it.
    ExtrapolateVelocity(gridSol, EXTRAPOLATION_LAYERS);
    GridToParticle(gridSol, particles);
}

// ---- density correction (Kugelstadt et al. 2019) ---------------------------------------------------------

static inline bool DensityWall2D(MAC *g, int i, int j)
{
    if (i < 0 || j < 0 || i >= g->Ny || j >= g->Nx) return true;
    return g->GetSolid(i, j) == SOLID_CELL;
}

// The share of a particle that lands in a solid cell is mirrored back across the wall, so a fluid cell against a
// wall reads the rest density at rest instead of losing the part of the kernel inside the solid.
static void DepositDensity2D(MAC *g, std::vector<double> &d, int ic, int jc, int ti, int tj, double w)
{
    if (DensityWall2D(g, ti, tj))
    {
        if (ti != ic && DensityWall2D(g, ti, jc)) ti = ic;
        if (tj != jc && DensityWall2D(g, ic, tj)) tj = jc;
        if (DensityWall2D(g, ti, tj)) { ti = ic; tj = jc; }
    }
    d[(size_t)ti * g->Nx + tj] += w;
}

// rho/rho0 on the cell centres: each particle spreads its area over the four centres around it with the hat
// kernel, which sums to one, so the total over the cells is the seeded area exactly. Serial: it is a scatter.
void FLIP2D::ComputeDensity(MAC *grid, const ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = grid->FacePos(MAC::X, 0), oy = grid->FacePos(MAC::Y, 0);
    const double dh = grid->dh;
    const double share = particleVolume / (dh * dh);
    std::fill(density.begin(), density.end(), 0.0);
    const Particle *ps = particles->data.data();
    for (int p = 0; p < particles->count; p++)
    {
        const double sx = (ps[p].x - ox) / dh - 0.5, sy = (ps[p].y - oy) / dh - 0.5;
        const int j0 = (int)std::floor(sx), i0 = (int)std::floor(sy);
        const double fx = sx - j0, fy = sy - i0;
        const int jc = std::min(std::max((int)((ps[p].x - ox) / dh), 0), grid->Nx - 1);
        const int ic = std::min(std::max((int)((ps[p].y - oy) / dh), 0), grid->Ny - 1);
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
            {
                const double w = (a ? fy : 1.0 - fy) * (b ? fx : 1.0 - fx);
                if (w > 0.0)
                    DepositDensity2D(grid, density, ic, jc, i0 + a, j0 + b, share * w);
            }
    }
}

static inline double CellPhi2D(const VectorXd &phi, int i, int j)
{
    const int id = PressureSolver2D::GetIDP(i, j);
    return id >= 0 ? phi(id) : 0.0;
}

// A face carries no shift when either side is a solid wall or a prescribed inflow (no flux through either).
static inline bool ShiftBlocked2D(MAC *g, int i, int j)
{
    const int t = g->GetSolid(i, j);
    return t == SOLID_CELL || IsVelocityBoundaryCell(t);
}

// L phi = 1 - rho/rho0 on the fluid cells, then each particle moves by -grad phi (div = rho/rho0 - 1):
// an over-dense cell spreads, an under-dense one closes up. Positions only.
void FLIP2D::DensityCorrection(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    if (particleVolume <= 0.0) return;
    const int Nx = grid->Nx, Ny = grid->Ny;
    const double dh = grid->dh;
    ComputeDensity(grid, particles);

    VectorXd f = VectorXd::Zero(PressureSolver2D::SystemSize());
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const int id = PressureSolver2D::GetIDP(i, j);
            if (id < 0) continue;
            double d = density[(size_t)i * Nx + j];
            // Beside air the particle sum is incomplete and reads low; the clamp keeps the surface from being
            // pulled inward by a deficit that is only the sampling.
            if (FLIP_SETTINGS.densitySurfaceClamp &&
                (grid->GetSolid(i + 1, j) == EMPTY_CELL || grid->GetSolid(i - 1, j) == EMPTY_CELL ||
                 grid->GetSolid(i, j + 1) == EMPTY_CELL || grid->GetSolid(i, j - 1) == EMPTY_CELL))
                d = std::max(d, 1.0);
            // Kugelstadt et al. clamp rho/rho0 to [0.5, 1.5]: the solve sees only what one step can safely correct.
            f(id) = 1.0 - std::min(std::max(d, 0.5), 1.5);
        }

    VectorXd phi;
    if (!PressureSolver2D::SolveLaplacian(grid, f, phi))
    {
        static bool warned = false;
        if (!warned) std::cerr << "FLIP2D: density correction skipped, no usable pressure matrix" << std::endl;
        warned = true;
        return;
    }

    for (int i = 0; i < Ny; i++)
        for (int j = 0; j <= Nx; j++)
        {
            double s = 0.0;
            if (j > 0 && j < Nx && !ShiftBlocked2D(grid, i, j - 1) && !ShiftBlocked2D(grid, i, j))
                s = -(CellPhi2D(phi, i, j) - CellPhi2D(phi, i, j - 1)) / dh;
            gridShift.SetU(i, j, s);
        }
    for (int i = 0; i <= Ny; i++)
        for (int j = 0; j < Nx; j++)
        {
            double s = 0.0;
            if (i > 0 && i < Ny && !ShiftBlocked2D(grid, i - 1, j) && !ShiftBlocked2D(grid, i, j))
                s = -(CellPhi2D(phi, i, j) - CellPhi2D(phi, i - 1, j)) / dh;
            gridShift.SetV(i, j, s);
        }

    // At most one cell per step: the correction is a nudge, and a longer jump could step across a thin solid.
    Particle *ps = particles->data.data();
    const int count = particles->count;
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < count; p++)
    {
        Particle &part = ps[p];
        const Particle before = part;
        const double dx = std::min(std::max(gridShift.ComponentAt(MAC::U, part.x, part.y), -dh), dh);
        const double dy = std::min(std::max(gridShift.ComponentAt(MAC::V, part.x, part.y), -dh), dh);
        part.x += dx;
        part.y += dy;
        Particles::ResolveContacts(grid, before, part);
    }
}

void FLIP2D::Dispose()
{
    std::vector<std::vector<std::vector<int>>>().swap(SPACE_HASH);
    gridAnt = MAC(); gridWork = MAC(); gridShift = MAC();
    std::vector<double>().swap(density);
}
