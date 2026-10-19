// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "SPH2D.h"
#include "SPHSettings.h"
#include <fstream>
#include <iostream>

std::vector<SPH2D::KernelParticle> SPH2D::kernel;

double SPH2D::restDensity = 1000.0;
double SPH2D::h           = 0.05;
double SPH2D::mass        = 0.0;
double SPH2D::spacing     = 0.0;
int    SPH2D::lastSubsteps = 1;
int    SPH2D::ghostCount   = 0;
std::vector<int> SPH2D::ghostSource;

int SPH2D::cellsX = 0;
int SPH2D::cellsY = 0;
std::vector<int> SPH2D::cellStart;
std::vector<int> SPH2D::cellEnd;
std::vector<int> SPH2D::particleIndex;
std::vector<int> SPH2D::cellOf;

// Wendland C2 with support h: W = 7/(pi h^2) (1 - s)^4 (1 + 4s), s = r/h.
double SPH2D::Wendland(double r)
{
    if (r < 0.0 || r >= h) return 0.0;
    const double s = r / h, a = 1.0 - s;
    return (7.0 / (M_PI * h * h)) * a * a * a * a * (1.0 + 4.0 * s);
}

double SPH2D::WendlandGradMag(double r)
{
    if (r <= 0.0 || r >= h) return 0.0;
    const double s = r / h, a = 1.0 - s;
    return -(7.0 / (M_PI * h * h)) * 20.0 * s * a * a * a / h;
}

// Uniform cell list of side h, as a counting sort.
void SPH2D::BuildCellList(int n)
{
    const double x0 = SIMULATION.domain.x0, y0 = SIMULATION.domain.y0;
    const int nCells = cellsX * cellsY;

    std::fill(cellStart.begin(), cellStart.end(), 0);

    for (int p = 0; p < n; p++)
    {
        int j = (int)((kernel[p].x - x0) / h);
        int i = (int)((kernel[p].y - y0) / h);
        j = std::min(std::max(j, 0), cellsX - 1);
        i = std::min(std::max(i, 0), cellsY - 1);
        const int c = i * cellsX + j;
        cellOf[p] = c;
        cellStart[c]++;
    }

    int run = 0;                                   // exclusive scan -> start offsets
    for (int c = 0; c < nCells; c++)
    {
        const int n = cellStart[c];
        cellStart[c] = run;
        cellEnd[c]   = run;
        run += n;
    }
    for (int p = 0; p < n; p++)                    // scatter; cellEnd ends up as the end
        particleIndex[cellEnd[cellOf[p]]++] = p;
}

// Mirror every particle within h of a domain wall across that wall, plus the diagonal mirror at a
// corner so the corner sum is complete too.
void SPH2D::EmitGhost(int count, double gx, double gy, double gu, double gv, int src)
{
    const int idx = count + ghostCount;
    if (idx >= (int)kernel.size()) return;
    KernelParticle &g = kernel[idx];
    g.x = gx; g.y = gy;
    g.u = gu; g.v = gv;
    ghostSource[ghostCount] = src;
    ghostCount++;
}

void SPH2D::BuildGhosts(const ParticleSet* particles)
{
    ghostCount = 0;
    const int maxParticles = (int)kernel.size();
    if ((int)ghostSource.size() < maxParticles) ghostSource.assign(maxParticles, 0);
    const int particleCount = particles->count;
    const Particle *ps = particles->data.data();
    for (int p = 0; p < particleCount; p++)
    {
        kernel[p].x = ps[p].x; kernel[p].y = ps[p].y;
        kernel[p].u = ps[p].u; kernel[p].v = ps[p].v;
    }

    const double x0 = SIMULATION.domain.x0, xf = SIMULATION.domain.xf;
    const double y0 = SIMULATION.domain.y0, yf = SIMULATION.domain.yf;
    const bool noSlip = (SIMULATION.particleOptions.contact == 1);

    for (int p = 0; p < particleCount; p++)
    {
        const double px = ps[p].x, py = ps[p].y;
        const double pu = ps[p].u, pv = ps[p].v;
        const bool nx0 = (px - x0) < h, nx1 = (xf - px) < h;
        const bool ny0 = (py - y0) < h, ny1 = (yf - py) < h;

        const double tu = noSlip ? -pu : pu;   // tangential, when the wall is an x wall
        const double tv = noSlip ? -pv : pv;   // tangential, when the wall is a y wall

        if (nx0) EmitGhost(particleCount, 2.0 * x0 - px, py, -pu, tv, p);
        if (nx1) EmitGhost(particleCount, 2.0 * xf - px, py, -pu, tv, p);
        if (ny0) EmitGhost(particleCount, px, 2.0 * y0 - py, tu, -pv, p);
        if (ny1) EmitGhost(particleCount, px, 2.0 * yf - py, tu, -pv, p);

        // corners: both normals negated
        if (nx0 && ny0) EmitGhost(particleCount, 2.0 * x0 - px, 2.0 * y0 - py, -pu, -pv, p);
        if (nx0 && ny1) EmitGhost(particleCount, 2.0 * x0 - px, 2.0 * yf - py, -pu, -pv, p);
        if (nx1 && ny0) EmitGhost(particleCount, 2.0 * xf - px, 2.0 * y0 - py, -pu, -pv, p);
        if (nx1 && ny1) EmitGhost(particleCount, 2.0 * xf - px, 2.0 * yf - py, -pu, -pv, p);

        // Interior solids, mirrored across the ANALYTIC surface -- no voxelisation, so the ghost
        // wall of a cylinder is a cylinder.
        if (SCENE.loaded)
        {
            double phi = 0.0, sn[3];
            if (SCENE.NearestSolidSurface(px, py, 0.0, phi, sn) && phi > 0.0 && phi < h)
            {
                // reflect the position across the surface, and the normal velocity with it
                const double vn = pu * sn[0] + pv * sn[1];
                const double gu = pu - (noSlip ? 2.0 * pu : 2.0 * vn * sn[0]);
                const double gv = pv - (noSlip ? 2.0 * pv : 2.0 * vn * sn[1]);
                EmitGhost(particleCount, px - 2.0 * phi * sn[0], py - 2.0 * phi * sn[1], gu, gv, p);
            }
        }
    }
}

// The ghosts take the density and pressure of the particle they mirror, corrected hydrostatically.
void SPH2D::SyncGhostFields(int count)
{
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < ghostCount; g++)
    {
        KernelParticle &gp = kernel[count + g];
        const KernelParticle &src = kernel[ghostSource[g]];
        // The mirror sits deeper (or higher) than its source: give it the hydrostatic density there, so the wall
        // supports the column and the delta term sees no jump across it.
        const double c2 = SPH_SETTINGS.soundSpeed * SPH_SETTINGS.soundSpeed;
        gp.d = src.d + restDensity * (SIMULATION.f.u * (gp.x - src.x) + SIMULATION.f.v * (gp.y - src.y)) / c2;
        gp.p = c2 * (gp.d - restDensity);
    }
}

// The 3x3 bucket walk every loop below shares. `body(q)` is called once per candidate.
#define SPH_FOR_NEIGHBOURS(px, py, body)                                                  \
    do {                                                                                  \
        int jc = (int)(((px) - SIMULATION.domain.x0) / h);                                \
        int ic = (int)(((py) - SIMULATION.domain.y0) / h);                                \
        jc = std::min(std::max(jc, 0), cellsX - 1);                                       \
        ic = std::min(std::max(ic, 0), cellsY - 1);                                       \
        for (int di = -1; di <= 1; di++)                                                  \
            for (int dj = -1; dj <= 1; dj++) {                                            \
                const int ii = ic + di, jj = jc + dj;                                     \
                if (ii < 0 || ii >= cellsY || jj < 0 || jj >= cellsX) continue;           \
                const int cc = ii * cellsX + jj;                                          \
                for (int kk = cellStart[cc]; kk < cellEnd[cc]; kk++) {                    \
                    const int q = particleIndex[kk];                                      \
                    body                                                                  \
                }                                                                         \
            }                                                                             \
    } while (0)

// The density is the particle's own, carried by continuity; the pressure is the linear equation of state.
void SPH2D::LoadDensityPressure(const ParticleSet* particles)
{
    const int count = particles->count;
    const double *rho = particles->density.data();
    const double c2 = SPH_SETTINGS.soundSpeed * SPH_SETTINGS.soundSpeed;
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < count; p++)
    {
        kernel[p].d = rho[p];
        kernel[p].p = c2 * (rho[p] - restDensity);
    }
}

// d(rho_i)/dt = sum_j m v_ij . grad W_ij + the delta-SPH diffusion with its hydrostatic part removed
// (Fourtakas et al. 2019), so a column at rest has nothing to diffuse.
void SPH2D::ComputeDensityRate(int count)
{
    const double c0 = SPH_SETTINGS.soundSpeed, c2 = c0 * c0;
    const double hs = 0.5 * h, D = SPH_SETTINGS.delta * hs * c0;
    const double gx = SIMULATION.f.u, gy = SIMULATION.f.v;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int p = 0; p < count; p++)
    {
        const double px = kernel[p].x, py = kernel[p].y;
        const double pu = kernel[p].u, pv = kernel[p].v, dp = kernel[p].d;
        double rate = 0.0;

        SPH_FOR_NEIGHBOURS(px, py, {
            if (q == p) continue;
            const double dx = px - kernel[q].x;
            const double dy = py - kernel[q].y;
            const double r2 = dx * dx + dy * dy;
            if (r2 <= 0.0 || r2 >= h * h) continue;
            const double r = std::sqrt(r2);
            const double dWr = WendlandGradMag(r) / r;         // grad_i W = dWr * (r_i - r_j)
            rate += mass * ((pu - kernel[q].u) * dx + (pv - kernel[q].v) * dy) * dWr;
            const double jump = (kernel[q].d - dp) + restDensity * (gx * dx + gy * dy) / c2;
            rate += D * (mass / kernel[q].d) * (-2.0 * jump) * dWr;
        });
        kernel[p].drho = rate;
    }
}

// a_i = -sum_j m (p_i/rho_i^2 + p_j/rho_j^2) grad W + artificial viscosity (Monaghan 1992) + g; the artificial viscosity
// is the only viscosity (Monaghan 1994). Stored as the force DENSITY rho_i a_i.
void SPH2D::ComputeForces(int count)
{
    const double c0 = SPH_SETTINGS.soundSpeed, alpha = SPH_SETTINGS.alpha;
    const double hs = 0.5 * h, eta2 = 0.01 * hs * hs;
    const double gx = SIMULATION.f.u, gy = SIMULATION.f.v;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int p = 0; p < count; p++)
    {
        const double px = kernel[p].x, py = kernel[p].y;
        const double pu = kernel[p].u, pv = kernel[p].v;
        const double dp = kernel[p].d, Pp = kernel[p].p / (dp * dp);
        double ax = 0.0, ay = 0.0;

        SPH_FOR_NEIGHBOURS(px, py, {
            if (q == p) continue;
            const double dx = px - kernel[q].x;
            const double dy = py - kernel[q].y;
            const double r2 = dx * dx + dy * dy;
            if (r2 <= 0.0 || r2 >= h * h) continue;
            const double r = std::sqrt(r2);
            const double dWr = WendlandGradMag(r) / r;
            const double dq = kernel[q].d;
            const double ux = pu - kernel[q].u;
            const double uy = pv - kernel[q].v;
            const double vr = ux * dx + uy * dy;

            double c = -(Pp + kernel[q].p / (dq * dq));             // pressure
            if (vr < 0.0)                                            // artificial viscosity, approaching only
                c += alpha * c0 * (hs * vr / (r2 + eta2)) / (0.5 * (dp + dq));
            ax += mass * c * dWr * dx;
            ay += mass * c * dWr * dy;

        });
        kernel[p].fx = dp * (ax + gx);
        kernel[p].fy = dp * (ay + gy);
    }
}

void SPH2D::DensityPhase(const ParticleSet* particles)
{
    LoadDensityPressure(particles);
    SyncGhostFields(particles->count);
}

void SPH2D::ForcePhase(int count)
{
    ComputeForces(count);    // the density rate is taken in Integrate, after the kick
}

// The column starts hydrostatic, rho = rho0 + rho0 g depth / c0^2, so the run does not open with an acoustic
// wave. Depth is measured from the highest particle in the same column of the seeding lattice; gravity along y.
void SPH2D::HydrostaticDensity(ParticleSet* particles)
{
    const int count = particles->count;
    const Particle *ps = particles->data.data();
    std::vector<double> &rho = particles->density;
    rho.assign(particles->data.size(), restDensity);
    const double g = -SIMULATION.f.v, c2 = SPH_SETTINGS.soundSpeed * SPH_SETTINGS.soundSpeed;
    if (g <= 0.0) return;
    const double x0 = SIMULATION.domain.x0;
    const int nCol = (int)((SIMULATION.domain.xf - x0) / spacing) + 2;
    std::vector<double> top(nCol, -1e300);
    for (int p = 0; p < count; p++)
    {
        const int c = std::min(std::max((int)((ps[p].x - x0) / spacing), 0), nCol - 1);
        top[c] = std::max(top[c], ps[p].y + 0.5 * spacing);
    }
    for (int p = 0; p < count; p++)
    {
        const int c = std::min(std::max((int)((ps[p].x - x0) / spacing), 0), nCol - 1);
        rho[p] = restDensity + restDensity * g * (top[c] - ps[p].y) / c2;
    }
}

// Leapfrog: second order, forces evaluated once per step. v(t + dt/2) = v(t -
// dt/2) + a(t) dt x(t + dt) = x(t) + v(t + dt/2) dt The stored velocity is the HALF-STEP one.
void SPH2D::Integrate(ParticleSet* particles, double dt, bool firstKick)
{
    const double kick   = firstKick ? 0.5 * dt : dt;
    const double margin = 0.5 * spacing;
    const int count = particles->count;
    Particle *ps = particles->data.data();

    // Kick, then the density rate from the KICKED velocities, then drift: advancing the density with the old
    // velocity makes the acoustic coupling forward Euler, and it grows without bound.
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < count; p++)
    {
        ps[p].u += (kernel[p].fx / kernel[p].d) * kick;
        ps[p].v += (kernel[p].fy / kernel[p].d) * kick;
    }
    BuildGhosts(particles);   // same positions, so the same ghosts and cell list, with the new velocities
    ComputeDensityRate(count);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < count; p++)
    {
        particles->density[p] += kernel[p].drho * dt;
        ps[p].x += ps[p].u * dt;
        ps[p].y += ps[p].v * dt;
        Particles::ResolveContactsAnalytic(ps[p], margin);
    }
}

// Monaghan's acoustic and force limits, on the smoothing length (half the support). There is no physical viscosity.
double SPH2D::StableTimestep(const ParticleSet* particles)
{
    const double c = SPH_SETTINGS.soundSpeed;
    const double L = 0.5 * h;
    double vmax = 0.0, amax = 0.0;
    const int count = std::min(particles->count, (int)kernel.size());
    const Particle *ps = particles->data.data();
    #pragma omp parallel for schedule(static) reduction(max:vmax) reduction(max:amax)
    for (int p = 0; p < count; p++)
    {
        const double d = (kernel[p].d > 0.0) ? kernel[p].d : restDensity;
        vmax = std::max(vmax, std::sqrt(ps[p].u * ps[p].u +
                                        ps[p].v * ps[p].v));
        amax = std::max(amax, std::sqrt(kernel[p].fx * kernel[p].fx +
                                        kernel[p].fy * kernel[p].fy) / d);
    }
    const double big = 1e30;
    const double dtCFL   = 0.25 * L / (c + vmax);                       // acoustic + advective
    const double dtForce = (amax > 0.0) ? 0.25 * std::sqrt(L / amax) : big;
    return std::min(dtCFL, dtForce);
}

void SPH2D::AdvanceSPH(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time)
{
    restDensity = SIMULATION.RHO;

    // Substep SIMULATION.dt down to something stable.
    const int cap = 64;

    // Phase timing. The whole outer step is measured, every substep summed, so the four parts add
    // up to lastSPHTotalTime.
    const double tStepStart = GetWallTime();
    double tNb = 0.0, tDen = 0.0, tFor = 0.0, tInt = 0.0, a;

    a = GetWallTime();
    BuildGhosts(particles);
    BuildCellList(particles->count + ghostCount);
    tNb += GetWallTime() - a;

    a = GetWallTime();
    DensityPhase(particles);
    tDen += GetWallTime() - a;

    a = GetWallTime();
    ForcePhase(particles->count);
    tFor += GetWallTime() - a;

    int n = 1;
    const double dtStable = StableTimestep(particles);
    if (dtStable > 0.0 && SIMULATION.dt > dtStable)
        n = (int)std::ceil(SIMULATION.dt / dtStable);
    if (n < 1) n = 1;
    if (n > cap) n = cap;
    lastSubsteps = n;

    const double dtSub = SIMULATION.dt / n;
    a = GetWallTime();
    Integrate(particles, dtSub, time == 0.0);   // reuse the forces already computed above
    tInt += GetWallTime() - a;

    for (int i = 1; i < n; i++)
    {
        a = GetWallTime();
        BuildGhosts(particles);
        BuildCellList(particles->count + ghostCount);
        tNb += GetWallTime() - a;

        a = GetWallTime();
        DensityPhase(particles);
        tDen += GetWallTime() - a;

        a = GetWallTime();
        ForcePhase(particles->count);
        tFor += GetWallTime() - a;

        a = GetWallTime();
        Integrate(particles, dtSub, false);
        tInt += GetWallTime() - a;
    }

    SPH_TIMINGS.neighbourTime = tNb;
    SPH_TIMINGS.densityTime = tDen;
    SPH_TIMINGS.forceTime = tFor;
    SPH_TIMINGS.integrateTime = tInt;
    SPH_TIMINGS.totalTime = GetWallTime() - tStepStart;
    SPH_TIMINGS.ghosts = ghostCount;
    DepositToGrid(gridSol);   // outside the timings: telemetry, not the step
}


void SPH2D::InitializeSPH2D(ParticleSet* particles, int maxP)
{
    h           = (SPH_SETTINGS.kernelRadius > 0.0) ? SPH_SETTINGS.kernelRadius : 0.05;
    restDensity = SIMULATION.RHO;                    // rho_0 IS the fluid density
    spacing     = SPHSeedSpacing(SPH_SETTINGS);
    mass        = restDensity * spacing * spacing;   // the density is carried by continuity

    const double x0 = SIMULATION.domain.x0, xf = SIMULATION.domain.xf;
    const double y0 = SIMULATION.domain.y0, yf = SIMULATION.domain.yf;

    const int maxParticles = maxP;
    particles->data.assign(maxParticles, Particle());
    kernel.assign(maxParticles, KernelParticle());
    Particle *ps = particles->data.data();
    int particleCount = 0;

    // Seed from the scene's `region fluid`, exactly as FLIP does -- the geometry is the scene's
    // business, not the solver's.
    const bool useScene = SCENE.loaded && SCENE.HasFluidRegions();

    for (double y = y0 + 0.5 * spacing; y < yf && particleCount < maxParticles; y += spacing)
        for (double x = x0 + 0.5 * spacing; x < xf && particleCount < maxParticles; x += spacing)
        {
            bool seed = false;
            if (useScene)
            {
                seed = SCENE.IsFluidSeedAt(x, y, 0.0);
                double phi = 0.0, n[3];
                // Not just "outside the solid" -- at least the standoff clear of it.
                if (seed && SCENE.NearestSolidSurface(x, y, 0.0, phi, n) && phi < 0.5 * spacing)
                    seed = false;
            }

            if (!seed) continue;
            ps[particleCount].isActive = true;
            ps[particleCount].x = x;
            ps[particleCount].y = y;
            particleCount++;
        }
    particles->count = particleCount;
    HydrostaticDensity(particles);

    cellsX = std::max(1, (int)((xf - x0) / h) + 1);
    cellsY = std::max(1, (int)((yf - y0) / h) + 1);
    cellStart.assign((size_t)cellsX * cellsY, 0);
    cellEnd.assign((size_t)cellsX * cellsY, 0);
    particleIndex.assign(maxParticles, 0);
    cellOf.assign(maxParticles, 0);

    BuildGhosts(particles);
    BuildCellList(particles->count + ghostCount);
    DensityPhase(particles);
    ForcePhase(particles->count);

    std::cout << "SPH2D: " << particleCount << " particles, h=" << h
              << " spacing=" << spacing << " mass=" << mass
              << " cells " << cellsX << "x" << cellsY
              << (useScene ? " (seeded from the scene)" : " -- no `region fluid` in the scene, so nothing was seeded") << std::endl;
    RegisterSimulationStep(SPH2D::AdvanceSPH, "sph", 0);
}

// The particles' pressure (Shepard mean) and liquid fraction at each cell centre, for the probes and exports;
// FLUID where the kernel volume is at least half. Reads the last substep's kernel buffer.
void SPH2D::DepositToGrid(MAC* grid)
{
    if (grid == nullptr || grid->Nx <= 0 || kernel.empty() || cellsX <= 0) return;
    const int Nx = grid->Nx, Ny = grid->Ny;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
        {
            if (grid->GetSolid(i, j) == SOLID_CELL) continue;
            const double x = grid->CentrePos(MAC::X, j), y = grid->CentrePos(MAC::Y, i);
            double vol = 0.0, pw = 0.0;
            SPH_FOR_NEIGHBOURS(x, y, {
                const double dx = x - kernel[q].x;
                const double dy = y - kernel[q].y;
                const double w = Wendland(std::sqrt(dx * dx + dy * dy)) * mass / kernel[q].d;
                vol += w;
                pw += w * kernel[q].p;
            });
            const bool fluid = vol >= 0.5;
            grid->SetSolid(i, j, fluid ? FLUID_CELL : EMPTY_CELL);
            grid->SetP(i, j, fluid ? pw / vol : 0.0);
        }
}

double SPH2D::GetMaxVelocity(const ParticleSet* particles)
{
    double m = 0.0;
    const int count = particles->count;
    const Particle *ps = particles->data.data();
    #pragma omp parallel for schedule(static) reduction(max:m)
    for (int p = 0; p < count; p++)
        m = std::max(m, std::sqrt(ps[p].u * ps[p].u +
                                  ps[p].v * ps[p].v));
    return m;
}

double SPH2D::GetPressureAt(double x, double y)
{
    if (kernel.empty() || cellsX <= 0) return 0.0;
    double num = 0.0, den = 0.0;
    SPH_FOR_NEIGHBOURS(x, y, {
        const double dx = x - kernel[q].x;
        const double dy = y - kernel[q].y;
        const double w  = Wendland(std::sqrt(dx * dx + dy * dy));
        num += kernel[q].p * w;
        den += w;
    });
    return (den > 1e-12) ? (num / den) : 0.0;
}

void SPH2D::Dispose()
{
    std::vector<KernelParticle>().swap(kernel);
    std::vector<int>().swap(ghostSource);
    std::vector<int>().swap(cellStart); std::vector<int>().swap(cellEnd);
    std::vector<int>().swap(particleIndex); std::vector<int>().swap(cellOf);
    ghostCount = 0;
}
