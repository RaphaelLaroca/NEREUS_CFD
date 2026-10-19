// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "FLIP3D.h"
#include "FLIPSettings.h"
#include "../../headers/Core/Scene.h"
#include <algorithm>
#include <random>
#include <iostream>
#include "../PPESolver/PressureSolver3D.h"
#include "DiffusionSolver3D.h"

int FLIP3D::particlePerCell = 0.0;
std::vector<std::vector<std::vector<std::vector<int>>>> FLIP3D::SPACE_HASH = std::vector<std::vector<std::vector<std::vector<int>>>>();
MAC FLIP3D::gridAnt = MAC();
MAC FLIP3D::gridWork = MAC();
double FLIP3D::alpha = 0.0;

std::vector<double> FLIP3D::exField, FLIP3D::exSrc;
std::vector<char> FLIP3D::exValid, FLIP3D::exPinned, FLIP3D::exCur;
double FLIP3D::particleVolume = 0.0;
std::vector<double> FLIP3D::density;
MAC FLIP3D::gridShift = MAC();

// A seeding offset inside the particle's own sub-cell; zero, and no random draw, when jitter is off.
static double SeedJitter(std::mt19937 &rng, std::uniform_real_distribution<double> &dist, double amp)
{
    return (amp > 0.0) ? amp * dist(rng) : 0.0;
}

void FLIP3D::InitializeFLIP(MAC *grid, ParticleSet *particles, double dt, double alpha)
{
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    int Nz = SIMULATION.Nz;
    double dh = SIMULATION.dh;
    FLIP3D::alpha = alpha;
    const int maxParticle = SIMULATION.Nx * SIMULATION.Ny * SIMULATION.Nz * FLIP_SETTINGS.particlesPerCell;
    FLIP3D::SPACE_HASH.clear();
    particles->data.assign(maxParticle, Particle());
    Particle *ps = particles->data.data();
    int particleCount = 0;
    int seededCells = 0;

    grid->SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    gridAnt.InitializeFromGrid(*grid, SIMULATION.time);
    gridWork.InitializeFromGrid(*grid, SIMULATION.time);

    // Largest of the three staggered face arrays; the extrapolation reuses one set of
    // buffers for U, V and W in turn.
    const size_t faceMax = (size_t)(SIMULATION.Nx + 1) * (SIMULATION.Ny + 1) * (SIMULATION.Nz + 1);
    exField.assign(faceMax, 0.0);
    exSrc.assign(faceMax, 0.0);
    exValid.assign(faceMax, 0);
    exPinned.assign(faceMax, 0);
    exCur.assign(faceMax, 0);

    // Position particles in initial volume
    double *offset;
    int n = cbrt(FLIP_SETTINGS.particlesPerCell); // Use cube root for 3D
    FLIP3D::particlePerCell = FLIP_SETTINGS.particlesPerCell;
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
                for (int k = 1; k < Nz - 1; k++)
                {
                    if (grid->GetSolid(i, j, k) == SOLID_CELL)
                        continue;
                    if (!SCENE.IsFluidSeedAt(grid->FacePos(MAC::X, 0) + (j + 0.5) * dh,
                                             grid->FacePos(MAC::Y, 0) + (i + 0.5) * dh,
                                             grid->FacePos(MAC::Z, 0) + (k + 0.5) * dh))
                        continue;
                    seededCells++;

                    for (int io = 0; io < n; io++)
                        for (int jo = 0; jo < n; jo++)
                            for (int ko = 0; ko < n; ko++)
                            {
                                if (particleCount >= maxParticle)
                                {
                                    std::cerr << "WARNING: FLIP3D seeding hit maxParticle ("
                                              << maxParticle << "), truncating" << std::endl;
                                    break;
                                }
                                Particle p;
                                p.x = grid->FacePos(MAC::X, 0) + j * dh + offset[jo] * dh + SeedJitter(seedRng, seedJitterDist, seedJitterAmp);
                                p.y = grid->FacePos(MAC::Y, 0) + i * dh + offset[io] * dh + SeedJitter(seedRng, seedJitterDist, seedJitterAmp);
                                p.z = grid->FacePos(MAC::Z, 0) + k * dh + offset[ko] * dh + SeedJitter(seedRng, seedJitterDist, seedJitterAmp);
                                p.isActive = true;
                                ps[particleCount] = p;
                                particleCount++;
                            }
                }

        std::cout << "FLIP3D: seeded " << particleCount
                  << " particles from the scene's fluid regions" << std::endl;
    }
    else
        std::cerr << "FLIP3D: the scene has no `region fluid`, so no particles were seeded" << std::endl;

    particles->count = particleCount;
    // The liquid each particle stands for, from what was seeded: rho/rho0 is then exactly 1 on the lattice.
    particleVolume = particleCount > 0 ? seededCells * dh * dh * dh / particleCount : 0.0;
    density.assign((size_t)Nx * Ny * Nz, 0.0);
    if (FLIP_SETTINGS.densityCorrection)
        gridShift.InitializeGrid(SIMULATION.domain, false);

    // Initialize 3D space hash structure
    for (int i = 0; i < Ny; i++)
    {
        SPACE_HASH.push_back(std::vector<std::vector<std::vector<int>>>());
        for (int j = 0; j < Nx; j++)
        {
            SPACE_HASH.at(i).push_back(std::vector<std::vector<int>>());
            for (int k = 0; k < Nz; k++)
            {
                SPACE_HASH.at(i).at(j).push_back(std::vector<int>(FLIP_SETTINGS.particlesPerCell));
            }
        }
    }

    DiffusionSolver3D::InitializeDiffusionSolver(grid);

    UpdateSpaceHash(particles);
    free(offset);
    // Split so the telemetry separates the two halves: the particle phase (advection, transfer,
    // extrapolation) from the grid phase (the viscous solve and the body force).
    RegisterSimulationStep(FLIP3D::FLIP_Advection, "particle_advection", 0);
    RegisterSimulationStep(FLIP3D::FLIP_Diffusion, "grid_diffusion", 2);
    RegisterSimulationStep(FLIP3D::FLIP_GridToParticle, "grid_to_particle", 7);
    if (FLIP_SETTINGS.densityCorrection)
        RegisterSimulationStep(FLIP3D::DensityCorrection, "density_correction", 8);
}

// p^n, kept for FLIP_VISCOUS_PRESSURE: gridAnt holds it at the start of the step and loses it to the copy
// in FLIP_Advection (the particle half: advect, mark, transfer, extrapolate).
static std::vector<double> PREV_PRESSURE;

static inline size_t PrevIndex(MAC *g, int i, int j, int k) { return ((size_t)i * g->Nx + j) * g->Nz + k; }

// False on a step with no previous pressure (the first, or a restart from rest): gravity alone inside the viscous
// solve would carve a boundary layer out of it at the walls, so that step adds the body force after the solve.
static bool HAS_PREV_PRESSURE = false;

static void SavePreviousPressure(MAC *g)
{
    PREV_PRESSURE.assign((size_t)g->Nx * g->Ny * g->Nz, 0.0);
    for (int i = 0; i < g->Ny; i++)
        for (int j = 0; j < g->Nx; j++)
            for (int k = 0; k < g->Nz; k++)
                PREV_PRESSURE[PrevIndex(g, i, j, k)] = g->GetP(i, j, k);
    HAS_PREV_PRESSURE = false;
    for (double p : PREV_PRESSURE)
        if (p != 0.0) { HAS_PREV_PRESSURE = true; break; }
}

// p^n on this step's cells with the air at 0, the projection's own Dirichlet value, so the projection removes
// its gradient exactly.
static inline double PreviousPressure(MAC *g, int i, int j, int k)
{
    return g->GetSolid(i, j, k) == EMPTY_CELL ? 0.0 : PREV_PRESSURE[PrevIndex(g, i, j, k)];
}

// u += c grad p^n on every face the projection corrects.
static void AddPreviousPressureGradient(MAC *g, double c)
{
    const int nx = g->Nx, ny = g->Ny, nz = g->Nz;
    #pragma omp parallel for
    for (int i = 1; i < ny - 1; i++)
        for (int j = 1; j < nx; j++)
            for (int k = 1; k < nz - 1; k++)
                if (g->FaceGate(MAC::U, i, j, k) != GateState::WALL)
                    g->SetU(i, j, k, g->GetU(i, j, k) + c * (PreviousPressure(g, i, j, k) -
                                     PreviousPressure(g, i, j - 1, k)) / g->FaceGap(MAC::X, j - 1));
    #pragma omp parallel for
    for (int i = 1; i < ny; i++)
        for (int j = 1; j < nx - 1; j++)
            for (int k = 1; k < nz - 1; k++)
                if (g->FaceGate(MAC::V, i, j, k) != GateState::WALL)
                    g->SetV(i, j, k, g->GetV(i, j, k) + c * (PreviousPressure(g, i, j, k) -
                                     PreviousPressure(g, i - 1, j, k)) / g->FaceGap(MAC::Y, i - 1));
    #pragma omp parallel for
    for (int i = 1; i < ny - 1; i++)
        for (int j = 1; j < nx - 1; j++)
            for (int k = 1; k < nz; k++)
                if (g->FaceGate(MAC::W, i, j, k) != GateState::WALL)
                    g->SetW(i, j, k, g->GetW(i, j, k) + c * (PreviousPressure(g, i, j, k) -
                                     PreviousPressure(g, i, j, k - 1)) / g->FaceGap(MAC::Z, k - 1));
}

void FLIP3D::FLIP_Advection(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    if (FLIP_SETTINGS.physicalDiffusion && FLIP_SETTINGS.viscousPressure)
        SavePreviousPressure(gridAnt);
    FLIP3D::FLIP_StepBeforeProjection(gridAnt, gridSol, particles, SIMULATION.dt);
    gridWork.CopyGrid(*gridSol);
}

// The GRID half: the viscous solve (optional) and the body force, which must be applied whether or not the
// viscous solve ran -- and after it, so the no-slip walls cannot smooth the gravity impulse.
void FLIP3D::FLIP_Diffusion(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    gridSol->SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    // The net driving force, body force minus grad p^n, inside the solve (which reads gridWork), grad p^n added back.
    if (FLIP_SETTINGS.physicalDiffusion && FLIP_SETTINGS.viscousPressure && HAS_PREV_PRESSURE)
    {
        const double c = SIMULATION.dt / SIMULATION.RHO;
        gridSol->AddForce(time, SIMULATION.dt);
        AddPreviousPressureGradient(gridSol, -c);
        gridSol->EnforceVelocityBC(SIMULATION.time);
        gridWork.CopyFlowFields(*gridSol);
        if (FLIP_SETTINGS.diffusionSolver == DIFFUSION_SOLVERS::IMPLICIT)
            DiffusionSolver3D::SolveDiffusion_Eigen(&gridWork, gridSol);
        else
            DiffusionSolver3D::SolveDiffusion_ADI(&gridWork, gridSol);
        AddPreviousPressureGradient(gridSol, c);
        gridSol->EnforceVelocityBC(SIMULATION.time);
        return;
    }
    if (FLIP_SETTINGS.physicalDiffusion)
    {
        gridSol->EnforceVelocityBC(SIMULATION.time);
        if(FLIP_SETTINGS.diffusionSolver == DIFFUSION_SOLVERS::IMPLICIT){
            DiffusionSolver3D::SolveDiffusion_Eigen(&gridWork, gridSol);
        }
        else if(FLIP_SETTINGS.diffusionSolver == DIFFUSION_SOLVERS::ADI){
            DiffusionSolver3D::SolveDiffusion_ADI(&gridWork, gridSol);
        }
        gridSol->EnforceVelocityBC(SIMULATION.time);
    }

    // The body force is applied here in BOTH branches, after any viscous solve.
    gridSol->AddForce(time, SIMULATION.dt);
    gridSol->EnforceVelocityBC(SIMULATION.time);
}

// Both halves in order, for harnesses that drive a whole momentum step.
void FLIP3D::FLIP_Momentum(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    FLIP3D::FLIP_Advection(gridAnt, gridSol, particles, time);
    FLIP3D::FLIP_Diffusion(gridAnt, gridSol, particles, time);
}



// Sweeps a scalar face field outward from the entries flagged valid.
static void ExtrapolateField3D(std::vector<double> &f, std::vector<char> &valid,
                               std::vector<double> &src, std::vector<char> &cur,
                               int ni, int nj, int nk, int layers)
{
    const size_t n = (size_t)ni * nj * nk;

    for (int layer = 0; layer < layers; layer++)
    {
        std::copy(f.begin(), f.begin() + n, src.begin());
        std::copy(valid.begin(), valid.begin() + n, cur.begin());
        int filledAny = 0;

        for (int i = 0; i < ni; i++)
        {
            for (int j = 0; j < nj; j++)
            {
                for (int k = 0; k < nk; k++)
                {
                    const size_t id = (size_t)(i * nj + j) * nk + k;
                    if (cur[id])
                        continue;

                    double sum = 0.0;
                    int c = 0;
                    if (i > 0      && cur[id - (size_t)nj * nk]) { sum += src[id - (size_t)nj * nk]; c++; }
                    if (i < ni - 1 && cur[id + (size_t)nj * nk]) { sum += src[id + (size_t)nj * nk]; c++; }
                    if (j > 0      && cur[id - (size_t)nk])      { sum += src[id - (size_t)nk];      c++; }
                    if (j < nj - 1 && cur[id + (size_t)nk])      { sum += src[id + (size_t)nk];      c++; }
                    if (k > 0      && cur[id - 1])               { sum += src[id - 1];               c++; }
                    if (k < nk - 1 && cur[id + 1])               { sum += src[id + 1];               c++; }

                    if (c > 0)
                    {
                        f[id] = sum / (double)c;
                        valid[id] = 1;
                        filledAny++;
                    }
                }
            }
        }

        if (filledAny == 0)
            break;
    }
}

// A face keeps its value when it touches fluid or is pinned against a solid; the rest is swept into.
void FLIP3D::ExtrapolateComponent(MAC *grid, MAC::Component component, int layers)
{
    const int Nx = SIMULATION.Nx, Ny = SIMULATION.Ny, Nz = SIMULATION.Nz;
    const int ni = (component == MAC::V) ? Ny + 1 : Ny;
    const int nj = (component == MAC::U) ? Nx + 1 : Nx;
    const int nk = (component == MAC::W) ? Nz + 1 : Nz;

    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++)
            for (int k = 0; k < nk; k++)
            {
                const size_t id = (size_t)(i * nj + j) * nk + k;
                exField[id] = grid->GetComponent(component, i, j, k);
                int a, b;
                if (component == MAC::U)
                {
                    a = (j < Nx) ? grid->GetSolid(i, j, k) : SOLID_CELL;
                    b = (j > 0) ? grid->GetSolid(i, j - 1, k) : SOLID_CELL;
                }
                else if (component == MAC::V)
                {
                    a = (i < Ny) ? grid->GetSolid(i, j, k) : SOLID_CELL;
                    b = (i > 0) ? grid->GetSolid(i - 1, j, k) : SOLID_CELL;
                }
                else
                {
                    a = (k < Nz) ? grid->GetSolid(i, j, k) : SOLID_CELL;
                    b = (k > 0) ? grid->GetSolid(i, j, k - 1) : SOLID_CELL;
                }
                char keep = (a == FLUID_CELL || b == FLUID_CELL || a == SOLID_CELL || b == SOLID_CELL) ? 1 : 0;
                exValid[id] = keep;
                exPinned[id] = keep;
            }

    ExtrapolateField3D(exField, exValid, exSrc, exCur, ni, nj, nk, layers);

    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++)
            for (int k = 0; k < nk; k++)
            {
                const size_t id = (size_t)(i * nj + j) * nk + k;
                if (exPinned[id]) continue;
                grid->SetComponent(component, i, j, k, exField[id]);
            }
}

void FLIP3D::ExtrapolateVelocity(MAC *grid, int layers)
{
    ExtrapolateComponent(grid, MAC::U, layers);
    ExtrapolateComponent(grid, MAC::V, layers);
    ExtrapolateComponent(grid, MAC::W, layers);
}

// The space-hash bucket of a cell, or null outside the grid.
static const std::vector<int> *HashBucket(int i, int j, int k)
{
    if (i < 0 || i >= SIMULATION.Ny || j < 0 || j >= SIMULATION.Nx || k < 0 || k >= SIMULATION.Nz) return nullptr;
    return &FLIP3D::SPACE_HASH[i][j][k];
}

void FLIP3D::ParticleToGrid(MAC *grid, const ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = grid->FacePos(MAC::X, 0), oy = grid->FacePos(MAC::Y, 0), oz = grid->FacePos(MAC::Z, 0);
    const Particle *ps = particles->data.data();
    const int Nx = SIMULATION.Nx;
    const int Ny = SIMULATION.Ny;
    const int Nz = SIMULATION.Nz;
    const double dh = SIMULATION.dh;

    // --- U COMPONENT (Grid Centric) ---
    // Staggered X: defined at vertical faces
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx + 1; j++) {
            for (int k = 0; k < Nz; k++) {

                double sum_val = 0.0;
                double sum_wt = 0.0;
                double grid_x = j * dh;
                double grid_y = (i * dh) + (dh * 0.5);
                double grid_z = (k * dh) + (dh * 0.5);

                // Search neighborhood in Spatial Hash U-face is between cell (j-1) and cell (j). We
                // check buckets [j-1, j] and surrounding neighbors.
                for (int di = -KERNEL_CELLS; di <= KERNEL_CELLS; di++) {
                    for (int dj = -KERNEL_CELLS; dj <= KERNEL_CELLS - 1; dj++) { // U-face sits between these cols
                        for (int dk = -KERNEL_CELLS; dk <= KERNEL_CELLS; dk++) {

                            const auto* bucket = HashBucket(i + di, j + dj, k + dk);
                            if (!bucket) continue;

                            for (int p : *bucket) {
                                double dx = (ps[p].x - ox) - grid_x;
                                double dy = (ps[p].y - oy) - grid_y;
                                double dz = (ps[p].z - oz) - grid_z;

                                // Inline or Macro this distance check for speed
                                if (std::abs(dx) > KERNEL_SUPPORT_CELLS * dh || std::abs(dy) > KERNEL_SUPPORT_CELLS * dh ||
                                    std::abs(dz) > KERNEL_SUPPORT_CELLS * dh) 
                                    continue;

                                double w = FLIP3D::k(dx, dy, dz);
                                if (w > 1e-12) {
                                    sum_val += ps[p].u * w;
                                    sum_wt  += w;
                                }
                            }
                        }
                    }
                }

                grid->SetU(i, j, k, (sum_wt > 1e-12) ? (sum_val / sum_wt) : 0.0);
            }
        }
    }

    // --- V COMPONENT (Grid Centric) ---
    // Staggered Y: defined at horizontal faces
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < Ny + 1; i++) {
        for (int j = 0; j < Nx; j++) {
            for (int k = 0; k < Nz; k++) {

                double sum_val = 0.0;
                double sum_wt = 0.0;
                double grid_x = (j * dh) + (dh * 0.5);
                double grid_y = i * dh;
                double grid_z = (k * dh) + (dh * 0.5);

                for (int di = -KERNEL_CELLS; di <= KERNEL_CELLS - 1; di++) { // V-face sits between these rows
                    for (int dj = -KERNEL_CELLS; dj <= KERNEL_CELLS; dj++) {
                        for (int dk = -KERNEL_CELLS; dk <= KERNEL_CELLS; dk++) {

                            const auto* bucket = HashBucket(i + di, j + dj, k + dk);
                            if (!bucket) continue;

                            for (int p : *bucket) {
                                double dx = (ps[p].x - ox) - grid_x;
                                double dy = (ps[p].y - oy) - grid_y;
                                double dz = (ps[p].z - oz) - grid_z;

                                if (std::abs(dx) > KERNEL_SUPPORT_CELLS * dh || std::abs(dy) > KERNEL_SUPPORT_CELLS * dh ||
                                    std::abs(dz) > KERNEL_SUPPORT_CELLS * dh) 
                                    continue;

                                double w = FLIP3D::k(dx, dy, dz);
                                if (w > 1e-12) {
                                    sum_val += ps[p].v * w;
                                    sum_wt  += w;
                                }
                            }
                        }
                    }
                }
                grid->SetV(i, j, k, (sum_wt > 1e-12) ? (sum_val / sum_wt) : 0.0);
            }
        }
    }

    // --- W COMPONENT (Grid Centric) ---
    // Staggered Z: defined at depth faces
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            for (int k = 0; k < Nz + 1; k++) {

                double sum_val = 0.0;
                double sum_wt = 0.0;
                double grid_x = (j * dh) + (dh * 0.5);
                double grid_y = (i * dh) + (dh * 0.5);
                double grid_z = k * dh;

                for (int di = -KERNEL_CELLS; di <= KERNEL_CELLS; di++) {
                    for (int dj = -KERNEL_CELLS; dj <= KERNEL_CELLS; dj++) {
                        for (int dk = -KERNEL_CELLS; dk <= KERNEL_CELLS - 1; dk++) { // W-face sits between these slabs

                            const auto* bucket = HashBucket(i + di, j + dj, k + dk);
                            if (!bucket) continue;

                            for (int p : *bucket) {
                                double dx = (ps[p].x - ox) - grid_x;
                                double dy = (ps[p].y - oy) - grid_y;
                                double dz = (ps[p].z - oz) - grid_z;

                                if (std::abs(dx) > KERNEL_SUPPORT_CELLS * dh || std::abs(dy) > KERNEL_SUPPORT_CELLS * dh ||
                                    std::abs(dz) > KERNEL_SUPPORT_CELLS * dh) 
                                    continue;

                                double w = FLIP3D::k(dx, dy, dz);
                                if (w > 1e-12) {
                                    sum_val += ps[p].w * w;
                                    sum_wt  += w;
                                }
                            }
                        }
                    }
                }
                grid->SetW(i, j, k, (sum_wt > 1e-12) ? (sum_val / sum_wt) : 0.0);
            }
        }
    }
}

void FLIP3D::GridToParticle(MAC *grid, ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = grid->FacePos(MAC::X, 0), oy = grid->FacePos(MAC::Y, 0), oz = grid->FacePos(MAC::Z, 0);
    Particle *ps = particles->data.data();
    const int particleCount = particles->count;
    const int Nx = SIMULATION.Nx;
    const int Ny = SIMULATION.Ny;
    const int Nz = SIMULATION.Nz;
    const double dh = SIMULATION.dh;

    // Get raw pointers for direct array access (eliminates function call overhead)
    const double* u_data = grid->Field(MAC::U).data();
    const double* v_data = grid->Field(MAC::V).data();
    const double* w_data = grid->Field(MAC::W).data();

    const double* u_ant = gridAnt.Field(MAC::U).data();
    const double* v_ant = gridAnt.Field(MAC::V).data();
    const double* w_ant = gridAnt.Field(MAC::W).data();

    // Pre-calculate strides for flat indexing
    const int stride_x_u = Nz;             // For U grid (Ny, Nx+1, Nz)
    const int stride_y_u = (Nx + 1) * Nz;

    const int stride_x_v = Nz;             // For V grid (Ny+1, Nx, Nz)
    const int stride_y_v = Nx * Nz;

    const int stride_x_w = (Nz + 1);       // For W grid (Ny, Nx, Nz+1)
    const int stride_y_w = Nx * (Nz + 1);

    // Single unified particle update loop
    // Compute velocity differences on-the-fly to reduce memory traffic and eliminate barriers
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < particleCount; p++)
    {
        // --- U COMPONENT UPDATE ---
        {
            // Calculate base grid cell for particle (staggered U grid)
            int pJ = static_cast<int>((ps[p].x - ox) / dh);
            int pI = static_cast<int>(((ps[p].y - oy) + (dh * 0.5)) / dh);
            int pK = static_cast<int>(((ps[p].z - oz) + (dh * 0.5)) / dh);

            // Clamp to valid range (4-point stencil: -1 to +2 around base cell)
            int pI_start = (std::max)(pI - KERNEL_CELLS, 0);
            int pI_end = (std::min)(pI + KERNEL_CELLS + 1, Ny);
            int pJ_start = (std::max)(pJ - KERNEL_CELLS, 0);
            int pJ_end = (std::min)(pJ + KERNEL_CELLS + 1, Nx + 1);
            int pK_start = (std::max)(pK - KERNEL_CELLS, 0);
            int pK_end = (std::min)(pK + KERNEL_CELLS + 1, Nz);

            double uW_local = 0.0;      // Weighted velocity sum
            double duW_local = 0.0;     // Weighted velocity difference sum
            double wu_local = 0.0;      // Weight sum

            for (int i = pI_start; i < pI_end; i++) {
                double dy = (ps[p].y - oy) - (i * dh + dh * 0.5);
                int y_offset = i * stride_y_u;

                for (int j = pJ_start; j < pJ_end; j++) {
                    double dx = (ps[p].x - ox) - j * dh;
                    int yx_offset = y_offset + j * stride_x_u;

                    // Vectorize innermost loop
                    #pragma omp simd reduction(+:wu_local,uW_local,duW_local)
                    for (int k = pK_start; k < pK_end; k++) {
                        double dz = (ps[p].z - oz) - (k * dh + dh * 0.5);
                        double weight = FLIP3D::k(dx, dy, dz);

                        int idx = yx_offset + k;
                        double u_curr = u_data[idx];
                        double u_prev = u_ant[idx];

                        wu_local += weight;
                        uW_local += u_curr * weight;
                        duW_local += (u_curr - u_prev) * weight;
                    }
                }
            }

            // Update particle velocity using PIC/FLIP blend
            if (wu_local > 1e-12) {
                double pic_u = uW_local / wu_local;
                double flip_u = ps[p].u + (duW_local / wu_local);
                ps[p].u = pic_u * (1.0 - alpha) + flip_u * alpha;
            }
        }

        // --- V COMPONENT UPDATE ---
        {
            // Calculate base grid cell for particle (staggered V grid)
            int pJ = static_cast<int>(((ps[p].x - ox) + (dh * 0.5)) / dh);
            int pI = static_cast<int>((ps[p].y - oy) / dh);
            int pK = static_cast<int>(((ps[p].z - oz) + (dh * 0.5)) / dh);

            // Clamp to valid range
            int pI_start = (std::max)(pI - KERNEL_CELLS, 0);
            int pI_end = (std::min)(pI + KERNEL_CELLS + 1, Ny + 1);
            int pJ_start = (std::max)(pJ - KERNEL_CELLS, 0);
            int pJ_end = (std::min)(pJ + KERNEL_CELLS + 1, Nx);
            int pK_start = (std::max)(pK - KERNEL_CELLS, 0);
            int pK_end = (std::min)(pK + KERNEL_CELLS + 1, Nz);

            double vW_local = 0.0;
            double dvW_local = 0.0;
            double wv_local = 0.0;

            for (int i = pI_start; i < pI_end; i++) {
                double dy = (ps[p].y - oy) - i * dh;
                int y_offset = i * stride_y_v;

                for (int j = pJ_start; j < pJ_end; j++) {
                    double dx = (ps[p].x - ox) - (j * dh + dh * 0.5);
                    int yx_offset = y_offset + j * stride_x_v;

                    // Vectorize innermost loop
                    #pragma omp simd reduction(+:wv_local,vW_local,dvW_local)
                    for (int k = pK_start; k < pK_end; k++) {
                        double dz = (ps[p].z - oz) - (k * dh + dh * 0.5);
                        double weight = FLIP3D::k(dx, dy, dz);

                        int idx = yx_offset + k;
                        double v_curr = v_data[idx];
                        double v_prev = v_ant[idx];

                        wv_local += weight;
                        vW_local += v_curr * weight;
                        dvW_local += (v_curr - v_prev) * weight;
                    }
                }
            }

            // Update particle velocity using PIC/FLIP blend
            if (wv_local > 1e-12) {
                double pic_v = vW_local / wv_local;
                double flip_v = ps[p].v + (dvW_local / wv_local);
                ps[p].v = pic_v * (1.0 - alpha) + flip_v * alpha;
            }
        }

        // --- W COMPONENT UPDATE ---
        {
            // Calculate base grid cell for particle (staggered W grid)
            int pJ = static_cast<int>(((ps[p].x - ox) + (dh * 0.5)) / dh);
            int pI = static_cast<int>(((ps[p].y - oy) + (dh * 0.5)) / dh);
            int pK = static_cast<int>((ps[p].z - oz) / dh);

            // Clamp to valid range
            int pI_start = (std::max)(pI - KERNEL_CELLS, 0);
            int pI_end = (std::min)(pI + KERNEL_CELLS + 1, Ny);
            int pJ_start = (std::max)(pJ - KERNEL_CELLS, 0);
            int pJ_end = (std::min)(pJ + KERNEL_CELLS + 1, Nx);
            int pK_start = (std::max)(pK - KERNEL_CELLS, 0);
            int pK_end = (std::min)(pK + KERNEL_CELLS + 1, Nz + 1);

            double wW_local = 0.0;
            double dwW_local = 0.0;
            double ww_local = 0.0;

            for (int i = pI_start; i < pI_end; i++) {
                double dy = (ps[p].y - oy) - (i * dh + dh * 0.5);
                int y_offset = i * stride_y_w;

                for (int j = pJ_start; j < pJ_end; j++) {
                    double dx = (ps[p].x - ox) - (j * dh + dh * 0.5);
                    int yx_offset = y_offset + j * stride_x_w;

                    // Vectorize innermost loop
                    #pragma omp simd reduction(+:ww_local,wW_local,dwW_local)
                    for (int k = pK_start; k < pK_end; k++) {
                        double dz = (ps[p].z - oz) - k * dh;
                        double weight = FLIP3D::k(dx, dy, dz);

                        int idx = yx_offset + k;
                        double w_curr = w_data[idx];
                        double w_prev = w_ant[idx];

                        ww_local += weight;
                        wW_local += w_curr * weight;
                        dwW_local += (w_curr - w_prev) * weight;
                    }
                }
            }

            // Update particle velocity using PIC/FLIP blend
            if (ww_local > 1e-12) {
                double pic_w = wW_local / ww_local;
                double flip_w = ps[p].w + (dwW_local / ww_local);
                ps[p].w = pic_w * (1.0 - alpha) + flip_w * alpha;
            }
        }
    }
}
void FLIP3D::UpdateParticles(ParticleSet *particles, double dt, MAC *grid, MAC *advect)
{
    Particle *ps = particles->data.data();
    const int particleCount = particles->count;
    #pragma omp parallel for schedule(static)
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
            const double k1x = (s == 0) ? part.u : advect->ComponentAt(MAC::U, part.x, part.y, part.z);
            const double k1y = (s == 0) ? part.v : advect->ComponentAt(MAC::V, part.x, part.y, part.z);
            const double k1z = (s == 0) ? part.w : advect->ComponentAt(MAC::W, part.x, part.y, part.z);

            const double midx = part.x + k1x * h * 0.5;
            const double midy = part.y + k1y * h * 0.5;
            const double midz = part.z + k1z * h * 0.5;
            part.x += advect->ComponentAt(MAC::U, midx, midy, midz) * h;
            part.y += advect->ComponentAt(MAC::V, midx, midy, midz) * h;
            part.z += advect->ComponentAt(MAC::W, midx, midy, midz) * h;

            Particles::ResolveContacts(grid, before, part);
        }
    }
}

void FLIP3D::UpdateSpaceHash(const ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = gridAnt.FacePos(MAC::X, 0), oy = gridAnt.FacePos(MAC::Y, 0), oz = gridAnt.FacePos(MAC::Z, 0);
    const Particle *ps = particles->data.data();
    const int particleCount = particles->count;
    int Nx = SIMULATION.Nx;
    int Ny = SIMULATION.Ny;
    int Nz = SIMULATION.Nz;

    // Clear existing hash
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                SPACE_HASH.at(i).at(j).at(k).clear();
            }
        }
    }

    int i, j, k;
    double dh = SIMULATION.dh;
    for (int p = 0; p < particleCount; p++)
    {
        i = (ps[p].y - oy) / dh;
        j = (ps[p].x - ox) / dh;
        k = (ps[p].z - oz) / dh;

        // Clamp, the way UpdateFluidCells already does.
        i = std::min(std::max(i, 0), Ny - 1);
        j = std::min(std::max(j, 0), Nx - 1);
        k = std::min(std::max(k, 0), Nz - 1);

        SPACE_HASH[i][j][k].push_back(p);
    }
}

void FLIP3D::UpdateFluidCells(MAC *grid, const ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = grid->FacePos(MAC::X, 0), oy = grid->FacePos(MAC::Y, 0), oz = grid->FacePos(MAC::Z, 0);
    const Particle *ps = particles->data.data();
    const int particleCount = particles->count;
    grid->ResetFluidCells();

    int i, j, k;
    double dh = SIMULATION.dh;
    for (int p = 0; p < particleCount; p++)
    {
        i = (ps[p].y - oy) / dh;
        j = (ps[p].x - ox) / dh;
        k = (ps[p].z - oz) / dh;

        i = i > (SIMULATION.Ny - 1) ? SIMULATION.Ny - 1 : i;
        j = j > (SIMULATION.Nx - 1) ? SIMULATION.Nx - 1 : j;
        k = k > (SIMULATION.Nz - 1) ? SIMULATION.Nz - 1 : k;

        if (grid->GetSolid(i, j, k) == EMPTY_CELL)
        {
            grid->SetSolid(i, j, k, FLUID_CELL);
        }
    }

    // Check for holes and fill them
    for (int i = 1; i < grid->Ny - 1; i++)
    {
        for (int j = 1; j < grid->Nx - 1; j++)
        {
            for (int k = 1; k < grid->Nz - 1; k++)
            {
                // FLUID neighbours only.
                int fluidNeighbors = 0;
                if (grid->GetSolid(i, j + 1, k) == FLUID_CELL) fluidNeighbors++;
                if (grid->GetSolid(i, j - 1, k) == FLUID_CELL) fluidNeighbors++;
                if (grid->GetSolid(i + 1, j, k) == FLUID_CELL) fluidNeighbors++;
                if (grid->GetSolid(i - 1, j, k) == FLUID_CELL) fluidNeighbors++;
                if (grid->GetSolid(i, j, k + 1) == FLUID_CELL) fluidNeighbors++;
                if (grid->GetSolid(i, j, k - 1) == FLUID_CELL) fluidNeighbors++;

                if (grid->GetSolid(i, j, k) == EMPTY_CELL && fluidNeighbors >= 5)
                {
                    grid->SetSolid(i, j, k, FLUID_CELL);
                }
            }
        }
    }
}

// Linear (hat) kernel, the same one FLIP2D uses. It reaches zero at KERNEL_SUPPORT_CELLS, which is
// what every transfer neighbourhood is sized from.
double FLIP3D::h(double r)
{
    r = std::abs(r); // Kernel is symmetric

    if (r < FLIP3D::KERNEL_SUPPORT_CELLS)
    {
        return 1.0 - r / FLIP3D::KERNEL_SUPPORT_CELLS;
    }
    else
    {
        return 0.0;
    }
}

// 3D kernel using tensor product of 1D kernels
double FLIP3D::k(double x, double y, double z)
{
    return h(x / SIMULATION.dh) * h(y / SIMULATION.dh) * h(z / SIMULATION.dh);
}

void FLIP3D::FLIP_StepBeforeProjection(MAC *advect, MAC *grid, ParticleSet *particles, double dt)
{
    grid->SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    grid->EnforceVelocityBC(SIMULATION.time);

    UpdateParticles(particles, dt, grid, advect);

    UpdateFluidCells(grid, particles);
    UpdateSpaceHash(particles);

    ParticleToGrid(grid, particles);

    // Particle-to-grid leaves a hard zero on every face that no particle reached. Sweep the field
    // outward before anything reads it.
    ExtrapolateVelocity(grid, EXTRAPOLATION_LAYERS);

    // The boundary conditions, and no flux through an obstacle face (particle-to-grid writes one there).
    grid->EnforceVelocityBC(SIMULATION.time);

    gridAnt.CopyGrid(*grid);

}

void FLIP3D::FLIP_GridToParticle(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    // The projection only touches faces that border fluid, so without this sweep the air band would still hold
    // pre-projection values when the next step advects through it.
    ExtrapolateVelocity(gridSol, EXTRAPOLATION_LAYERS);
    GridToParticle(gridSol, particles);
}

// ---- density correction (Kugelstadt et al. 2019) ---------------------------------------------------------

static inline bool DensityWall3D(MAC *g, int i, int j, int k)
{
    if (i < 0 || j < 0 || k < 0 || i >= g->Ny || j >= g->Nx || k >= g->Nz) return true;
    return g->GetSolid(i, j, k) == SOLID_CELL;
}

// The share of a particle that lands in a solid cell is mirrored back across the wall, so a fluid cell against a
// wall reads the rest density at rest instead of losing the part of the kernel inside the solid.
static void DepositDensity3D(MAC *g, std::vector<double> &d, int ic, int jc, int kc, int ti, int tj, int tk, double w)
{
    if (DensityWall3D(g, ti, tj, tk))
    {
        if (ti != ic && DensityWall3D(g, ti, jc, kc)) ti = ic;
        if (tj != jc && DensityWall3D(g, ic, tj, kc)) tj = jc;
        if (tk != kc && DensityWall3D(g, ic, jc, tk)) tk = kc;
        if (DensityWall3D(g, ti, tj, tk)) { ti = ic; tj = jc; tk = kc; }
    }
    d[((size_t)ti * g->Nx + tj) * g->Nz + tk] += w;
}

// rho/rho0 on the cell centres: each particle spreads its volume over the eight centres around it with the hat
// kernel, which sums to one, so the total over the cells is the seeded volume exactly. Serial: it is a scatter.
void FLIP3D::ComputeDensity(MAC *grid, const ParticleSet *particles)
{
    // Grid-local coordinates: the grid's first face lies one ring cell outside the domain.
    const double ox = grid->FacePos(MAC::X, 0), oy = grid->FacePos(MAC::Y, 0), oz = grid->FacePos(MAC::Z, 0);
    const double dh = grid->dh;
    const double share = particleVolume / (dh * dh * dh);
    std::fill(density.begin(), density.end(), 0.0);
    const Particle *ps = particles->data.data();
    for (int p = 0; p < particles->count; p++)
    {
        const double sx = (ps[p].x - ox) / dh - 0.5, sy = (ps[p].y - oy) / dh - 0.5, sz = (ps[p].z - oz) / dh - 0.5;
        const int j0 = (int)std::floor(sx), i0 = (int)std::floor(sy), k0 = (int)std::floor(sz);
        const double fx = sx - j0, fy = sy - i0, fz = sz - k0;
        const int jc = std::min(std::max((int)((ps[p].x - ox) / dh), 0), grid->Nx - 1);
        const int ic = std::min(std::max((int)((ps[p].y - oy) / dh), 0), grid->Ny - 1);
        const int kc = std::min(std::max((int)((ps[p].z - oz) / dh), 0), grid->Nz - 1);
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 2; c++)
                {
                    const double w = (a ? fy : 1.0 - fy) * (b ? fx : 1.0 - fx) * (c ? fz : 1.0 - fz);
                    if (w > 0.0)
                        DepositDensity3D(grid, density, ic, jc, kc, i0 + a, j0 + b, k0 + c, share * w);
                }
    }
}

static inline double CellPhi3D(const VectorXd &phi, int i, int j, int k)
{
    const int id = PressureSolver3D::GetIDP(i, j, k);
    return id >= 0 ? phi(id) : 0.0;
}

// A face carries no shift when either side is a solid wall or a prescribed inflow (no flux through either).
static inline bool ShiftBlocked3D(MAC *g, int i, int j, int k)
{
    const int t = g->GetSolid(i, j, k);
    return t == SOLID_CELL || IsVelocityBoundaryCell(t);
}

// L phi = 1 - rho/rho0 on the fluid cells, then each particle moves by -grad phi (div = rho/rho0 - 1):
// an over-dense cell spreads, an under-dense one closes up. Positions only.
void FLIP3D::DensityCorrection(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    if (particleVolume <= 0.0) return;
    const int Nx = grid->Nx, Ny = grid->Ny, Nz = grid->Nz;
    const double dh = grid->dh;
    ComputeDensity(grid, particles);

    VectorXd f = VectorXd::Zero(PressureSolver3D::SystemSize());
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                const int id = PressureSolver3D::GetIDP(i, j, k);
                if (id < 0) continue;
                double d = density[((size_t)i * Nx + j) * Nz + k];
                // Beside air the particle sum is incomplete and reads low; the clamp keeps the surface from
                // being pulled inward by a deficit that is only the sampling.
                if (FLIP_SETTINGS.densitySurfaceClamp &&
                    (grid->GetSolid(i + 1, j, k) == EMPTY_CELL || grid->GetSolid(i - 1, j, k) == EMPTY_CELL ||
                     grid->GetSolid(i, j + 1, k) == EMPTY_CELL || grid->GetSolid(i, j - 1, k) == EMPTY_CELL ||
                     grid->GetSolid(i, j, k + 1) == EMPTY_CELL || grid->GetSolid(i, j, k - 1) == EMPTY_CELL))
                    d = std::max(d, 1.0);
                // Kugelstadt et al. clamp rho/rho0 to [0.5, 1.5]: the solve sees only what one step can safely correct.
                f(id) = 1.0 - std::min(std::max(d, 0.5), 1.5);
            }

    VectorXd phi;
    if (!PressureSolver3D::SolveLaplacian(grid, f, phi))
    {
        static bool warned = false;
        if (!warned) std::cerr << "FLIP3D: density correction skipped, no usable pressure matrix" << std::endl;
        warned = true;
        return;
    }

    for (int i = 0; i < Ny; i++)
        for (int j = 0; j <= Nx; j++)
            for (int k = 0; k < Nz; k++)
            {
                double s = 0.0;
                if (j > 0 && j < Nx && !ShiftBlocked3D(grid, i, j - 1, k) && !ShiftBlocked3D(grid, i, j, k))
                    s = -(CellPhi3D(phi, i, j, k) - CellPhi3D(phi, i, j - 1, k)) / dh;
                gridShift.SetU(i, j, k, s);
            }
    for (int i = 0; i <= Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < Nz; k++)
            {
                double s = 0.0;
                if (i > 0 && i < Ny && !ShiftBlocked3D(grid, i - 1, j, k) && !ShiftBlocked3D(grid, i, j, k))
                    s = -(CellPhi3D(phi, i, j, k) - CellPhi3D(phi, i - 1, j, k)) / dh;
                gridShift.SetV(i, j, k, s);
            }
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k <= Nz; k++)
            {
                double s = 0.0;
                if (k > 0 && k < Nz && !ShiftBlocked3D(grid, i, j, k - 1) && !ShiftBlocked3D(grid, i, j, k))
                    s = -(CellPhi3D(phi, i, j, k) - CellPhi3D(phi, i, j, k - 1)) / dh;
                gridShift.SetW(i, j, k, s);
            }

    // At most one cell per step: the correction is a nudge, and a longer jump could step across a thin solid.
    Particle *ps = particles->data.data();
    const int count = particles->count;
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < count; p++)
    {
        Particle &part = ps[p];
        const Particle before = part;
        const double dx = std::min(std::max(gridShift.ComponentAt(MAC::U, part.x, part.y, part.z), -dh), dh);
        const double dy = std::min(std::max(gridShift.ComponentAt(MAC::V, part.x, part.y, part.z), -dh), dh);
        const double dz = std::min(std::max(gridShift.ComponentAt(MAC::W, part.x, part.y, part.z), -dh), dh);
        part.x += dx;
        part.y += dy;
        part.z += dz;
        Particles::ResolveContacts(grid, before, part);
    }
}

void FLIP3D::Dispose()
{
    std::vector<std::vector<std::vector<std::vector<int>>>>().swap(SPACE_HASH);
    gridAnt = MAC(); gridWork = MAC(); gridShift = MAC();
    std::vector<double>().swap(density);
    std::vector<double>().swap(exField); std::vector<double>().swap(exSrc);
    std::vector<char>().swap(exValid); std::vector<char>().swap(exPinned); std::vector<char>().swap(exCur);
}
