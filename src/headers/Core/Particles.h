// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Definitions.h"
#include "MAC.h"
#include "Scene.h"

#include <cmath>

// Particles.h -- what every particle method in this project shares.
namespace Particles
{

// Contact with solid geometry. Advection moves a particle along an interpolated velocity field that
// knows nothing about the geometry, so nothing stops it entering an obstacle.

// How far outside the analytic surface a resolved particle is placed, from PARTICLE_SURFACE_OFFSET
// (in cells). This wants to be a real fraction of a cell, not an epsilon.
inline double SurfaceOffset()
{
    double cells = SIMULATION.particleOptions.surfaceOffset;
    if (cells < 0.0) cells = 0.0;
    if (cells > 0.5) cells = 0.5; // beyond half a cell it reads as a visible gap
    return cells * SIMULATION.dh;
}

// Outward march used to leave a solid-flagged cell.
inline double EscapeStep()  { return 0.25 * SIMULATION.dh; }
inline int    EscapeLimit() { return 8; }

// Substep cap.
inline int MaxSubsteps() { return 8; }

// True when the cell containing a world-space point is marked solid.
inline bool InSolidCell(MAC *grid, double x, double y, double z)
{
    const double dh = SIMULATION.dh;
    if (dh <= 0.0)
        return false;

    // The grid starts one ring cell before the domain (its first face, FacePos(X, 0)).
    int j = (int)std::floor((x - grid->FacePos(MAC::X, 0)) / dh);
    int i = (int)std::floor((y - grid->FacePos(MAC::Y, 0)) / dh);
    j = (j < 0) ? 0 : (j > grid->Nx - 1 ? grid->Nx - 1 : j);
    i = (i < 0) ? 0 : (i > grid->Ny - 1 ? grid->Ny - 1 : i);

    if (DIMENSION == 2)
        return grid->GetSolid(i, j) == SOLID_CELL;

    int k = (int)std::floor((z - grid->FacePos(MAC::Z, 0)) / dh);
    k = (k < 0) ? 0 : (k > grid->Nz - 1 ? grid->Nz - 1 : k);
    return grid->GetSolid(i, j, k) == SOLID_CELL;
}

inline double SolidIndicator(MAC *grid, double x, double y, double z)
{
    return InSolidCell(grid, x, y, z) ? 1.0 : 0.0;
}

// Outward normal read off the voxel mask, for the case where there is no analytic surface to ask.
inline bool MaskNormal(MAC *grid, double x, double y, double z, double n[3])
{
    const double dh = SIMULATION.dh;

    n[0] = SolidIndicator(grid, x - dh, y, z) - SolidIndicator(grid, x + dh, y, z);
    n[1] = SolidIndicator(grid, x, y - dh, z) - SolidIndicator(grid, x, y + dh, z);
    n[2] = (DIMENSION == 3) ? (SolidIndicator(grid, x, y, z - dh) - SolidIndicator(grid, x, y, z + dh)) : 0.0;

    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len < 1e-12)
    {
        n[0] = n[1] = n[2] = 0.0;
        return false;
    }
    n[0] /= len; n[1] /= len; n[2] /= len;
    return true;
}


// The three are one formula.
inline void RespondToContact(double &u, double &v, double &w, const double n[3])
{
    const double vn = u * n[0] + v * n[1] + w * n[2];
    if (vn >= 0.0)
        return; // already moving away from the surface, nothing to answer

    switch (SIMULATION.particleOptions.contact)
    {
    case CONTACT_NOSLIP:
        u = v = w = 0.0;
        break;

    case CONTACT_BOUNCE:
    {
        double e = SIMULATION.particleOptions.restitution;
        e = (e < 0.0) ? 0.0 : (e > 1.0 ? 1.0 : e);
        const double s = (1.0 + e) * vn;
        u -= s * n[0];
        v -= s * n[1];
        w -= s * n[2];
        break;
    }

    default: // CONTACT_SLIP
        u -= vn * n[0];
        v -= vn * n[1];
        w -= vn * n[2];
        break;
    }
}

// Walk outward along n until the containing cell is no longer solid. Returns false if it
// never gets clear, which leaves the caller to fall back on the last known good position.
inline bool EscapeSolidCells(MAC *grid, double &x, double &y, double &z, const double n[3])
{
    if (!InSolidCell(grid, x, y, z))
        return true;
    if (n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0)
        return false;

    const double step = EscapeStep();
    for (int s = 0; s < EscapeLimit(); s++)
    {
        x += n[0] * step;
        y += n[1] * step;
        z += n[2] * step;
        if (!InSolidCell(grid, x, y, z))
            return true;
    }
    return false;
}

// The whole treatment for one particle, applied after it has been moved.
inline void Resolve(MAC *grid,
                    double ox, double oy, double oz,
                    double &x, double &y, double &z,
                    double &u, double &v, double &w)
{
    // The mask decides. A particle that is not in a solid cell is where the solver says
    // fluid may be, whatever the analytic surface thinks.
    if (!InSolidCell(grid, x, y, z))
        return;

    // ---- analytic surface for the direction and depth, when the scene supplies one ----
    if (SCENE.loaded && SCENE.HasSolidRegions())
    {
        double phi = 0.0, n[3] = {0, 0, 0};

        if (SCENE.SolidDistanceAt(x, y, (DIMENSION == 3) ? z : 0.0, phi, n) &&
            (n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0))
        {
            // phi < 0 means the analytic surface agrees the particle is inside, so there is a depth
            // to undo.
            if (phi < 0.0)
            {
                const double push = -phi + SurfaceOffset();
                x += n[0] * push;
                y += n[1] * push;
                if (DIMENSION == 3)
                    z += n[2] * push;
            }

            if (EscapeSolidCells(grid, x, y, z, n))
            {
                RespondToContact(u, v, w, n);
                return;
            }
        }

        // No usable normal (a medial axis), or no way out along it: fall through.
    }

    // ---- voxel mask alone: hardcoded levels, and anything the analytic pass could not
    // resolve ----

    // Walk back along the step to the last position outside the mask. Bisection rather
    // than a straight revert so a particle grazing a wall keeps most of its travel.
    double lo = 0.0, hi = 1.0; // fraction of the step; lo is known good if the old position was
    const double dx = x - ox, dy = y - oy, dz = z - oz;
    if (!InSolidCell(grid, ox, oy, oz))
    {
        for (int it = 0; it < 8; it++)
        {
            const double mid = 0.5 * (lo + hi);
            if (InSolidCell(grid, ox + dx * mid, oy + dy * mid, oz + dz * mid))
                hi = mid;
            else
                lo = mid;
        }
        x = ox + dx * lo;
        y = oy + dy * lo;
        z = oz + dz * lo;
    }
    else
    {
        // Started inside too -- seeded there, or the geometry moved under it. Leave the
        // position alone and just stop it driving deeper.
        x = ox; y = oy; z = oz;
    }

    double n[3];
    if (MaskNormal(grid, x, y, z, n))
    {
        EscapeSolidCells(grid, x, y, z, n);
        RespondToContact(u, v, w, n);
    }
    else
    {
        // Enclosed on every side, or no gradient to read. Nothing to slide along.
        u = v = w = 0.0;
    }
}

// Number of advection substeps for a particle moving at `speed`, so that each substep
// covers about one cell.
inline int SubstepCount(double speed, double dt)
{
    const double dh = SIMULATION.dh;
    if (dh <= 0.0)
        return 1;
    int steps = 1 + (int)(speed * dt / dh);
    const int cap = MaxSubsteps();
    return (steps > cap) ? cap : steps;
}
// The same, in terms of Particle, which is how a solver should call it.

// The walls the liquid is held inside: the domain's faces (for a grid method, the high side is its last
// interior face); SPH collides with the domain box itself.
inline void WallBox(double lo[3], double hi[3])
{
    const Domain &d = SIMULATION.domain;
    const bool grid = MethodUsesBoundaryCells();
    const std::vector<double> *f[3] = {&SIMULATION.spacing.xf, &SIMULATION.spacing.yf, &SIMULATION.spacing.zf};
    const double a0[3] = {d.x0, d.y0, d.z0}, af[3] = {d.xf, d.yf, d.zf};
    for (int a = 0; a < 3; a++)
    {
        lo[a] = a0[a];
        hi[a] = (grid && f[a]->size() >= 3 && (a < 2 || DIMENSION == 3)) ? (*f[a])[f[a]->size() - 2] : af[a];
    }
}

// The contact response against a wall with inward normal (nx, ny, nz).
inline void HitWall(Particle &p, double nx, double ny, double nz)
{
    const double n[3] = {nx, ny, nz};
    RespondToContact(p.u, p.v, p.w, n);
}

inline void ClampToDomain(Particle &p)
{
    double lo[3], hi[3];
    WallBox(lo, hi);

    // Each wall answers with the same response an obstacle would, so the setting means one thing
    // everywhere.

    if (p.x <= lo[0]) { p.x = lo[0]; HitWall(p, +1, 0, 0); }
    if (p.x >= hi[0]) { p.x = hi[0]; HitWall(p, -1, 0, 0); }
    if (p.y <= lo[1]) { p.y = lo[1]; HitWall(p, 0, +1, 0); }
    if (p.y >= hi[1]) { p.y = hi[1]; HitWall(p, 0, -1, 0); }
    if (DIMENSION == 3)
    {
        if (p.z <= lo[2]) { p.z = lo[2]; HitWall(p, 0, 0, +1); }
        if (p.z >= hi[2]) { p.z = hi[2]; HitWall(p, 0, 0, -1); }
    }
}

inline bool InSolidCell(MAC *grid, const Particle &p)
{
    return InSolidCell(grid, p.x, p.y, p.z);
}

// `previous` is where the particle was before this substep -- a position that was
// acceptable, and therefore the fallback when no way out can be found.
inline void Resolve(MAC *grid, const Particle &previous, Particle &p)
{
    Resolve(grid, previous.x, previous.y, previous.z, p.x, p.y, p.z, p.u, p.v, p.w);
}

inline double Speed(const Particle &p)
{
    return (DIMENSION == 3) ? std::sqrt(p.u * p.u + p.v * p.v + p.w * p.w)
                            : std::sqrt(p.u * p.u + p.v * p.v);
}

// Walls, then obstacles, then walls again: an obstacle touching a wall could otherwise
// push a particle straight through it.
inline void ResolveContacts(MAC *grid, const Particle &previous, Particle &p)
{
    ClampToDomain(p);
    Resolve(grid, previous, p);
    ClampToDomain(p);
}

// The MESHLESS variant, for SPH.
inline void ClampToDomainAnalytic(Particle &p, double margin)
{

    if (p.x <= SIMULATION.domain.x0 + margin) { p.x = SIMULATION.domain.x0 + margin; HitWall(p, +1, 0, 0); }
    if (p.x >= SIMULATION.domain.xf - margin) { p.x = SIMULATION.domain.xf - margin; HitWall(p, -1, 0, 0); }
    if (p.y <= SIMULATION.domain.y0 + margin) { p.y = SIMULATION.domain.y0 + margin; HitWall(p, 0, +1, 0); }
    if (p.y >= SIMULATION.domain.yf - margin) { p.y = SIMULATION.domain.yf - margin; HitWall(p, 0, -1, 0); }
    if (DIMENSION == 3)
    {
        if (p.z <= SIMULATION.domain.z0 + margin) { p.z = SIMULATION.domain.z0 + margin; HitWall(p, 0, 0, +1); }
        if (p.z >= SIMULATION.domain.zf - margin) { p.z = SIMULATION.domain.zf - margin; HitWall(p, 0, 0, -1); }
    }
}

inline void ResolveAnalytic(Particle &p, double margin)
{
    if (!SCENE.loaded)
        return;

    double phi = 0.0, n[3] = {0.0, 0.0, 0.0};
    if (!SCENE.SolidDistanceAt(p.x, p.y, p.z, phi, n))
        return;                       // no solid region owns this point
    if (phi >= 0.0)
        return;                       // outside the surface already

    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len <= 0.0)
        return;                       // medial axis: no single way out, leave it alone
    n[0] /= len; n[1] /= len; n[2] /= len;

    const double push = -phi + margin;   // to the surface, plus the standoff
    p.x += n[0] * push;
    p.y += n[1] * push;
    if (DIMENSION == 3)
        p.z += n[2] * push;

    RespondToContact(p.u, p.v, p.w, n);
}

// Walls, then obstacles, then walls again -- the same ordering as the grid path, so an
// obstacle touching a wall cannot push a particle out through it.
inline void ResolveContactsAnalytic(Particle &p, double margin)
{
    ClampToDomainAnalytic(p, margin);
    ResolveAnalytic(p, margin);
    ClampToDomainAnalytic(p, margin);
}

} // namespace Particles
