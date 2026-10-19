// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "SPH3D.h"
#include "SPHSettings.h"
#include <fstream>
#include <iostream>

std::vector<SPH3D::KernelParticle> SPH3D::kernel;

double SPH3D::restDensity = 1000.0;
double SPH3D::h           = 0.05;
double SPH3D::mass        = 0.0;
double SPH3D::spacing     = 0.0;
int    SPH3D::lastSubsteps = 1;
int    SPH3D::ghostCount   = 0;
std::vector<int> SPH3D::ghostSource;

int SPH3D::cellsX = 0, SPH3D::cellsY = 0, SPH3D::cellsZ = 0;
std::vector<int> SPH3D::cellStart, SPH3D::cellEnd, SPH3D::particleIndex, SPH3D::cellOf;

// Wendland C2 with support h: W = 21/(2 pi h^3) (1 - s)^4 (1 + 4s), s = r/h.
double SPH3D::Wendland(double r)
{
    if (r < 0.0 || r >= h) return 0.0;
    const double s = r / h, a = 1.0 - s;
    return (21.0 / (2.0 * M_PI * h * h * h)) * a * a * a * a * (1.0 + 4.0 * s);
}

double SPH3D::WendlandGradMag(double r)
{
    if (r <= 0.0 || r >= h) return 0.0;
    const double s = r / h, a = 1.0 - s;
    return -(21.0 / (2.0 * M_PI * h * h * h)) * 20.0 * s * a * a * a / h;
}

void SPH3D::BuildCellList(int n)
{
    const double x0 = SIMULATION.domain.x0, y0 = SIMULATION.domain.y0, z0 = SIMULATION.domain.z0;
    const int nCells = cellsX * cellsY * cellsZ;

    std::fill(cellStart.begin(), cellStart.end(), 0);
    for (int p = 0; p < n; p++)
    {
        int j = (int)((kernel[p].x - x0) / h);
        int i = (int)((kernel[p].y - y0) / h);
        int k = (int)((kernel[p].z - z0) / h);
        j = std::min(std::max(j, 0), cellsX - 1);
        i = std::min(std::max(i, 0), cellsY - 1);
        k = std::min(std::max(k, 0), cellsZ - 1);
        const int c = (k * cellsY + i) * cellsX + j;
        cellOf[p] = c;
        cellStart[c]++;
    }
    int run = 0;
    for (int c = 0; c < nCells; c++)
    {
        const int m = cellStart[c];
        cellStart[c] = run; cellEnd[c] = run; run += m;
    }
    for (int p = 0; p < n; p++)
        particleIndex[cellEnd[cellOf[p]]++] = p;
}

// The 3x3x3 bucket walk every loop below shares.
#define SPH3_FOR_NEIGHBOURS(px, py, pz, body)                                             \
    do {                                                                                  \
        int jc = (int)(((px) - SIMULATION.domain.x0) / h);                                \
        int ic = (int)(((py) - SIMULATION.domain.y0) / h);                                \
        int kc = (int)(((pz) - SIMULATION.domain.z0) / h);                                \
        jc = std::min(std::max(jc, 0), cellsX - 1);                                       \
        ic = std::min(std::max(ic, 0), cellsY - 1);                                       \
        kc = std::min(std::max(kc, 0), cellsZ - 1);                                       \
        for (int dk = -1; dk <= 1; dk++)                                                  \
        for (int di = -1; di <= 1; di++)                                                  \
        for (int dj = -1; dj <= 1; dj++) {                                                \
            const int ii = ic + di, jj = jc + dj, kk2 = kc + dk;                           \
            if (ii < 0 || ii >= cellsY || jj < 0 || jj >= cellsX ||                        \
                kk2 < 0 || kk2 >= cellsZ) continue;                                        \
            const int cc = (kk2 * cellsY + ii) * cellsX + jj;                              \
            for (int kk = cellStart[cc]; kk < cellEnd[cc]; kk++) {                        \
                const int q = particleIndex[kk];                                          \
                body                                                                      \
            }                                                                             \
        }                                                                                 \
    } while (0)

// Mirror ghosts. In 3D a particle can be near up to three walls at once, so the mirrors are
// enumerated over the 26 non-zero sign triples: faces (one axis), edges (two) and corners (three).
void SPH3D::EmitGhost(int count, double gx, double gy, double gz, double gu, double gv, double gw, int src)
{
    const int idx = count + ghostCount;
    if (idx >= (int)kernel.size()) return;
    KernelParticle &g = kernel[idx];
    g.x = gx; g.y = gy; g.z = gz;
    g.u = gu; g.v = gv; g.w = gw;
    ghostSource[ghostCount] = src;
    ghostCount++;
}

void SPH3D::BuildGhosts(const ParticleSet* particles)
{
    ghostCount = 0;
    const int maxParticles = (int)kernel.size();
    if ((int)ghostSource.size() < maxParticles) ghostSource.assign(maxParticles, 0);
    const int particleCount = particles->count;
    const Particle *ps = particles->data.data();
    for (int p = 0; p < particleCount; p++)
    {
        kernel[p].x = ps[p].x; kernel[p].y = ps[p].y; kernel[p].z = ps[p].z;
        kernel[p].u = ps[p].u; kernel[p].v = ps[p].v; kernel[p].w = ps[p].w;
    }

    const double x0 = SIMULATION.domain.x0, xf = SIMULATION.domain.xf;
    const double y0 = SIMULATION.domain.y0, yf = SIMULATION.domain.yf;
    const double z0 = SIMULATION.domain.z0, zf = SIMULATION.domain.zf;
    const bool noSlip = (SIMULATION.particleOptions.contact == 1);

    for (int p = 0; p < particleCount; p++)
    {
        const double px = ps[p].x, py = ps[p].y, pz = ps[p].z;
        const double pu = ps[p].u, pv = ps[p].v, pw = ps[p].w;

        // -1 mirrors across the low face, +1 across the high face, 0 leaves the axis alone
        const int mx = ((px - x0) < h) ? -1 : ((xf - px) < h ? 1 : 0);
        const int my = ((py - y0) < h) ? -1 : ((yf - py) < h ? 1 : 0);
        const int mz = ((pz - z0) < h) ? -1 : ((zf - pz) < h ? 1 : 0);

        for (int sx = -1; sx <= 1; sx++)
        for (int sy = -1; sy <= 1; sy++)
        for (int sz = -1; sz <= 1; sz++)
        {
            if (sx == 0 && sy == 0 && sz == 0) continue;
            // only mirror across a face the particle is actually near, and on the side it is on
            if (sx != 0 && sx != mx) continue;
            if (sy != 0 && sy != my) continue;
            if (sz != 0 && sz != mz) continue;

            const double gx = (sx == 0) ? px : (sx < 0 ? 2.0 * x0 - px : 2.0 * xf - px);
            const double gy = (sy == 0) ? py : (sy < 0 ? 2.0 * y0 - py : 2.0 * yf - py);
            const double gz = (sz == 0) ? pz : (sz < 0 ? 2.0 * z0 - pz : 2.0 * zf - pz);
            // negate on every mirrored axis; on the untouched axes negate only for no-slip
            const double gu = (sx != 0) ? -pu : (noSlip ? -pu : pu);
            const double gv = (sy != 0) ? -pv : (noSlip ? -pv : pv);
            const double gw = (sz != 0) ? -pw : (noSlip ? -pw : pw);
            EmitGhost(particleCount, gx, gy, gz, gu, gv, gw, p);
        }

        // Interior solids, mirrored across the ANALYTIC surface -- no voxelisation.
        if (SCENE.loaded)
        {
            double phi = 0.0, sn[3];
            if (SCENE.NearestSolidSurface(px, py, pz, phi, sn) && phi > 0.0 && phi < h)
            {
                const double vn = pu * sn[0] + pv * sn[1] + pw * sn[2];
                const double gu = noSlip ? -pu : pu - 2.0 * vn * sn[0];
                const double gv = noSlip ? -pv : pv - 2.0 * vn * sn[1];
                const double gw = noSlip ? -pw : pw - 2.0 * vn * sn[2];
                EmitGhost(particleCount, px - 2.0 * phi * sn[0], py - 2.0 * phi * sn[1], pz - 2.0 * phi * sn[2],
                     gu, gv, gw, p);
            }
        }
    }
}

void SPH3D::SyncGhostFields(int count)
{
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < ghostCount; g++)
    {
        KernelParticle &gp = kernel[count + g];
        const KernelParticle &src = kernel[ghostSource[g]];
        // The mirror's hydrostatic density, as in SPH2D.
        const double c2 = SPH_SETTINGS.soundSpeed * SPH_SETTINGS.soundSpeed;
        gp.d = src.d + restDensity * (SIMULATION.f.u * (gp.x - src.x) + SIMULATION.f.v * (gp.y - src.y) +
                                      SIMULATION.f.w * (gp.z - src.z)) / c2;
        gp.p = c2 * (gp.d - restDensity);
    }
}

// The 3D twin of SPH2D; see there for the equations.
void SPH3D::LoadDensityPressure(const ParticleSet* particles)
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

void SPH3D::ComputeDensityRate(int count)
{
    const double c0 = SPH_SETTINGS.soundSpeed, c2 = c0 * c0;
    const double hs = 0.5 * h, D = SPH_SETTINGS.delta * hs * c0;
    const double gx = SIMULATION.f.u, gy = SIMULATION.f.v, gz = SIMULATION.f.w;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int p = 0; p < count; p++)
    {
        const double px = kernel[p].x, py = kernel[p].y, pz = kernel[p].z;
        const double pu = kernel[p].u, pv = kernel[p].v, pw = kernel[p].w, dp = kernel[p].d;
        double rate = 0.0;

        SPH3_FOR_NEIGHBOURS(px, py, pz, {
            if (q == p) continue;
            const double dx = px - kernel[q].x;
            const double dy = py - kernel[q].y;
            const double dz = pz - kernel[q].z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 0.0 || r2 >= h * h) continue;
            const double r = std::sqrt(r2);
            const double dWr = WendlandGradMag(r) / r;
            rate += mass * ((pu - kernel[q].u) * dx + (pv - kernel[q].v) * dy + (pw - kernel[q].w) * dz) * dWr;
            const double jump = (kernel[q].d - dp) + restDensity * (gx * dx + gy * dy + gz * dz) / c2;
            rate += D * (mass / kernel[q].d) * (-2.0 * jump) * dWr;
        });
        kernel[p].drho = rate;
    }
}

void SPH3D::ComputeForces(int count)
{
    const double c0 = SPH_SETTINGS.soundSpeed, alpha = SPH_SETTINGS.alpha;
    const double hs = 0.5 * h, eta2 = 0.01 * hs * hs;
    const double gx = SIMULATION.f.u, gy = SIMULATION.f.v, gz = SIMULATION.f.w;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int p = 0; p < count; p++)
    {
        const double px = kernel[p].x, py = kernel[p].y, pz = kernel[p].z;
        const double pu = kernel[p].u, pv = kernel[p].v, pw = kernel[p].w;
        const double dp = kernel[p].d, Pp = kernel[p].p / (dp * dp);
        double ax = 0.0, ay = 0.0, az = 0.0;

        SPH3_FOR_NEIGHBOURS(px, py, pz, {
            if (q == p) continue;
            const double dx = px - kernel[q].x;
            const double dy = py - kernel[q].y;
            const double dz = pz - kernel[q].z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 0.0 || r2 >= h * h) continue;
            const double r = std::sqrt(r2);
            const double dWr = WendlandGradMag(r) / r;
            const double dq = kernel[q].d;
            const double ux = pu - kernel[q].u;
            const double uy = pv - kernel[q].v;
            const double uz = pw - kernel[q].w;
            const double vr = ux * dx + uy * dy + uz * dz;

            double c = -(Pp + kernel[q].p / (dq * dq));
            if (vr < 0.0)
                c += alpha * c0 * (hs * vr / (r2 + eta2)) / (0.5 * (dp + dq));
            ax += mass * c * dWr * dx;
            ay += mass * c * dWr * dy;
            az += mass * c * dWr * dz;

        });
        kernel[p].fx = dp * (ax + gx);
        kernel[p].fy = dp * (ay + gy);
        kernel[p].fz = dp * (az + gz);
    }
}

void SPH3D::DensityPhase(const ParticleSet* particles)
{
    LoadDensityPressure(particles);
    SyncGhostFields(particles->count);
}

void SPH3D::ForcePhase(int count)
{
    ComputeForces(count);    // the density rate is taken in Integrate, after the kick
}

// Hydrostatic start, as in SPH2D, with the columns binned in x and z.
void SPH3D::HydrostaticDensity(ParticleSet* particles)
{
    const int count = particles->count;
    const Particle *ps = particles->data.data();
    std::vector<double> &rho = particles->density;
    rho.assign(particles->data.size(), restDensity);
    const double g = -SIMULATION.f.v, c2 = SPH_SETTINGS.soundSpeed * SPH_SETTINGS.soundSpeed;
    if (g <= 0.0) return;
    const double x0 = SIMULATION.domain.x0, z0 = SIMULATION.domain.z0;
    const int nx = (int)((SIMULATION.domain.xf - x0) / spacing) + 2;
    const int nz = (int)((SIMULATION.domain.zf - z0) / spacing) + 2;
    std::vector<double> top((size_t)nx * nz, -1e300);
    for (int p = 0; p < count; p++)
    {
        const int cx = std::min(std::max((int)((ps[p].x - x0) / spacing), 0), nx - 1);
        const int cz = std::min(std::max((int)((ps[p].z - z0) / spacing), 0), nz - 1);
        top[(size_t)cz * nx + cx] = std::max(top[(size_t)cz * nx + cx], ps[p].y + 0.5 * spacing);
    }
    for (int p = 0; p < count; p++)
    {
        const int cx = std::min(std::max((int)((ps[p].x - x0) / spacing), 0), nx - 1);
        const int cz = std::min(std::max((int)((ps[p].z - z0) / spacing), 0), nz - 1);
        rho[p] = restDensity + restDensity * g * (top[(size_t)cz * nx + cx] - ps[p].y) / c2;
    }
}

void SPH3D::Integrate(ParticleSet* particles, double dt, bool firstKick)
{
    const double kick   = firstKick ? 0.5 * dt : dt;
    const double margin = 0.5 * spacing;
    const int count = particles->count;
    Particle *ps = particles->data.data();

    // Kick, density rate from the kicked velocities, drift -- see SPH2D::Integrate for why this order.
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < count; p++)
    {
        ps[p].u += (kernel[p].fx / kernel[p].d) * kick;
        ps[p].v += (kernel[p].fy / kernel[p].d) * kick;
        ps[p].w += (kernel[p].fz / kernel[p].d) * kick;
    }
    BuildGhosts(particles);
    ComputeDensityRate(count);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < count; p++)
    {
        particles->density[p] += kernel[p].drho * dt;
        ps[p].x += ps[p].u * dt;
        ps[p].y += ps[p].v * dt;
        ps[p].z += ps[p].w * dt;
        Particles::ResolveContactsAnalytic(ps[p], margin);
    }
}

double SPH3D::StableTimestep(const ParticleSet* particles)
{
    const double c = SPH_SETTINGS.soundSpeed;
    const double L = 0.5 * h;   // the smoothing length, half the support
    double vmax = 0.0, amax = 0.0;
    const int count = std::min(particles->count, (int)kernel.size());
    const Particle *ps = particles->data.data();
    #pragma omp parallel for schedule(static) reduction(max:vmax) reduction(max:amax)
    for (int p = 0; p < count; p++)
    {
        const double d = (kernel[p].d > 0.0) ? kernel[p].d : restDensity;
        vmax = std::max(vmax, std::sqrt(ps[p].u * ps[p].u +
                                        ps[p].v * ps[p].v +
                                        ps[p].w * ps[p].w));
        amax = std::max(amax, std::sqrt(kernel[p].fx * kernel[p].fx +
                                        kernel[p].fy * kernel[p].fy +
                                        kernel[p].fz * kernel[p].fz) / d);
    }
    const double big = 1e30;
    const double dtCFL   = 0.25 * L / (c + vmax);
    const double dtForce = (amax > 0.0) ? 0.25 * std::sqrt(L / amax) : big;
    return std::min(dtCFL, dtForce);
}

void SPH3D::AdvanceSPH(MAC* gridAnt, MAC* gridSol, ParticleSet* particles, double time)
{
    restDensity = SIMULATION.RHO;

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


void SPH3D::InitializeSPH3D(ParticleSet* particles, int maxP)
{
    h           = (SPH_SETTINGS.kernelRadius > 0.0) ? SPH_SETTINGS.kernelRadius : 0.05;
    restDensity = SIMULATION.RHO;
    spacing     = SPHSeedSpacing(SPH_SETTINGS);
    mass        = restDensity * spacing * spacing * spacing;   // the density is carried by continuity

    const double x0 = SIMULATION.domain.x0, xf = SIMULATION.domain.xf;
    const double y0 = SIMULATION.domain.y0, yf = SIMULATION.domain.yf;
    const double z0 = SIMULATION.domain.z0, zf = SIMULATION.domain.zf;

    const int maxParticles = maxP;
    particles->data.assign(maxParticles, Particle());
    kernel.assign(maxParticles, KernelParticle());
    Particle *ps = particles->data.data();
    int particleCount = 0;

    const bool useScene = SCENE.loaded && SCENE.HasFluidRegions();

    for (double z = z0 + 0.5 * spacing; z < zf && particleCount < maxParticles; z += spacing)
    for (double y = y0 + 0.5 * spacing; y < yf && particleCount < maxParticles; y += spacing)
    for (double x = x0 + 0.5 * spacing; x < xf && particleCount < maxParticles; x += spacing)
    {
        bool seed = false;
        if (useScene)
        {
            seed = SCENE.IsFluidSeedAt(x, y, z);
            double phi = 0.0, n[3];
            // Not just "outside the solid" -- at least the standoff clear of it.
            if (seed && SCENE.NearestSolidSurface(x, y, z, phi, n) && phi < 0.5 * spacing)
                seed = false;
        }

        if (!seed) continue;
        ps[particleCount].isActive = true;
        ps[particleCount].x = x;
        ps[particleCount].y = y;
        ps[particleCount].z = z;
        particleCount++;
    }

    cellsX = std::max(1, (int)((xf - x0) / h) + 1);
    cellsY = std::max(1, (int)((yf - y0) / h) + 1);
    cellsZ = std::max(1, (int)((zf - z0) / h) + 1);
    const size_t nCells = (size_t)cellsX * cellsY * cellsZ;
    cellStart.assign(nCells, 0);
    cellEnd.assign(nCells, 0);
    particleIndex.assign(maxParticles, 0);
    cellOf.assign(maxParticles, 0);

    particles->count = particleCount;
    HydrostaticDensity(particles);
    BuildGhosts(particles);
    BuildCellList(particles->count + ghostCount);
    DensityPhase(particles);
    ForcePhase(particles->count);

    std::cout << "SPH3D: " << particleCount << " particles, h=" << h
              << " spacing=" << spacing << " mass=" << mass
              << " cells " << cellsX << "x" << cellsY << "x" << cellsZ
              << (useScene ? " (seeded from the scene)" : " -- no `region fluid` in the scene, so nothing was seeded") << std::endl;
    RegisterSimulationStep(SPH3D::AdvanceSPH, "sph", 0);
}

// See SPH2D::DepositToGrid.
void SPH3D::DepositToGrid(MAC* grid)
{
    if (grid == nullptr || grid->Nx <= 0 || kernel.empty() || cellsX <= 0) return;
    const int Nx = grid->Nx, Ny = grid->Ny, Nz = grid->Nz;
    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < Nz; k++)
            {
                if (grid->GetSolid(i, j, k) == SOLID_CELL) continue;
                const double x = grid->CentrePos(MAC::X, j), y = grid->CentrePos(MAC::Y, i), z = grid->CentrePos(MAC::Z, k);
                double vol = 0.0, pw = 0.0;
                SPH3_FOR_NEIGHBOURS(x, y, z, {
                    const double dx = x - kernel[q].x;
                    const double dy = y - kernel[q].y;
                    const double dz = z - kernel[q].z;
                    const double w = Wendland(std::sqrt(dx * dx + dy * dy + dz * dz)) * mass / kernel[q].d;
                    vol += w;
                    pw += w * kernel[q].p;
                });
                const bool fluid = vol >= 0.5;
                grid->SetSolid(i, j, k, fluid ? FLUID_CELL : EMPTY_CELL);
                grid->SetP(i, j, k, fluid ? pw / vol : 0.0);
            }
}

double SPH3D::GetMaxVelocity(const ParticleSet* particles)
{
    double m = 0.0;
    const int count = particles->count;
    const Particle *ps = particles->data.data();
    #pragma omp parallel for schedule(static) reduction(max:m)
    for (int p = 0; p < count; p++)
        m = std::max(m, std::sqrt(ps[p].u * ps[p].u +
                                  ps[p].v * ps[p].v +
                                  ps[p].w * ps[p].w));
    return m;
}

void SPH3D::Dispose()
{
    std::vector<KernelParticle>().swap(kernel);
    std::vector<int>().swap(ghostSource);
    std::vector<int>().swap(cellStart); std::vector<int>().swap(cellEnd);
    std::vector<int>().swap(particleIndex); std::vector<int>().swap(cellOf);
    ghostCount = 0;
}
