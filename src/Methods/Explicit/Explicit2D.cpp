// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Explicit2D.h"
#include "ExplicitSettings.h"

#include <algorithm>
#include <cmath>
#include <vector>

static double Laplacian(MAC *g, MAC::Component c, int i, int j)
{
    const double u = g->GetComponent(c, i, j);
    double sum = 0.0;
    for (int s = -1; s <= 1; s += 2)
        sum += g->WallLinkFactor(c, MAC::X, i, j, s) * (g->GetComponent(c, i, j + s) - u)
             + g->WallLinkFactor(c, MAC::Y, i, j, s) * (g->GetComponent(c, i + s, j) - u);
    return sum / (g->dh * g->dh);
}

void Explicit2D::Momentum(MAC *ant, MAC *sol, ParticleSet *, double time)
{
    const double dt = SIMULATION.dt, nu = SIMULATION.EPS, h2 = 2.0 * ant->dh;
    sol->CopyFlowFields(*ant);
    for (int i = 1; i < ant->Ny - 1; i++)
        for (int j = 1; j < ant->Nx; j++)
            if (ant->FaceGate(MAC::U, i, j) == GateState::OPEN)
            {
                const double u = ant->GetU(i, j), v = ant->getVatU(i, j);
                const double dudx = (ant->GetU(i, j + 1) - ant->GetU(i, j - 1)) / h2;
                const double dudy = (ant->GetU(i + 1, j) - ant->GetU(i - 1, j)) / h2;
                sol->SetU(i, j, u + dt * (-u * dudx - v * dudy + nu * Laplacian(ant, MAC::U, i, j)));
            }
    for (int i = 1; i < ant->Ny; i++)
        for (int j = 1; j < ant->Nx - 1; j++)
            if (ant->FaceGate(MAC::V, i, j) == GateState::OPEN)
            {
                const double u = ant->getUatV(i, j), v = ant->GetV(i, j);
                const double dvdx = (ant->GetV(i, j + 1) - ant->GetV(i, j - 1)) / h2;
                const double dvdy = (ant->GetV(i + 1, j) - ant->GetV(i - 1, j)) / h2;
                sol->SetV(i, j, v + dt * (-u * dvdx - v * dvdy + nu * Laplacian(ant, MAC::V, i, j)));
            }
    sol->AddForce(time, dt);
    sol->EnforceVelocityBC(time + dt);
}

void Explicit2D::Pressure(MAC *, MAC *g, ParticleSet *, double time)
{
    static std::vector<double> b;
    const int nx = g->Nx, ny = g->Ny;
    const double h2 = g->dh * g->dh, w = EXPLICIT_SETTINGS.relaxation;
    const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
    b.assign(nx * ny, 0.0);
    for (int i = 1; i < ny - 1; i++)
        for (int j = 1; j < nx - 1; j++)
            if (g->GetSolid(i, j) == FLUID_CELL)
                b[i * nx + j] = h2 * SIMULATION.RHO / SIMULATION.dt * g->GetDivergencyAt(i, j);
    g->EnforcePressureBC(time + SIMULATION.dt);

    const double bmax = std::max(*std::max_element(b.begin(), b.end()), -*std::min_element(b.begin(), b.end()));
    bool singular = true;
    for (int it = 0; it < EXPLICIT_SETTINGS.maxIterations; it++)
    {
        double residual = 0.0;
        for (int i = 1; i < ny - 1; i++)
            for (int j = 1; j < nx - 1; j++)
            {
                if (g->GetSolid(i, j) != FLUID_CELL) continue;
                double sum = 0.0, diag = 0.0;
                for (int n = 0; n < 4; n++)
                {
                    const int ni = i + di[n], nj = j + dj[n];
                    const bool outlet = g->IsPressureOutlet(ni, nj);
                    const double c = g->IsFluid(ni, nj) ? 1.0 : (outlet ? 2.0 : 0.0);
                    sum += c * g->GetP(ni, nj);
                    diag += c;
                    if (outlet) singular = false;
                }
                if (diag == 0.0) continue;
                const double p = g->GetP(i, j);
                const double r = sum - diag * p - b[i * nx + j];
                g->SetP(i, j, p + w * r / diag);
                residual = std::max(residual, std::fabs(r));
            }
        if (residual <= EXPLICIT_SETTINGS.tolerance * bmax) break;
    }

    if (singular)
    {
        double mean = 0.0;
        int count = 0;
        for (int i = 1; i < ny - 1; i++)
            for (int j = 1; j < nx - 1; j++)
                if (g->GetSolid(i, j) == FLUID_CELL) { mean += g->GetP(i, j); count++; }
        for (int i = 1; i < ny - 1; i++)
            for (int j = 1; j < nx - 1; j++)
                if (g->GetSolid(i, j) == FLUID_CELL) g->SetP(i, j, g->GetP(i, j) - mean / std::max(count, 1));
    }
    g->EnforcePressureBC(time + SIMULATION.dt);
}

void Explicit2D::Projection(MAC *, MAC *g, ParticleSet *, double)
{
    const double k = SIMULATION.dt / (SIMULATION.RHO * g->dh);
    for (int i = 1; i < g->Ny - 1; i++)
        for (int j = 1; j < g->Nx; j++)
            if (g->FaceGate(MAC::U, i, j) != GateState::WALL)
            {
                const bool half = g->IsPressureOutlet(i, j) || g->IsPressureOutlet(i, j - 1);
                g->SetU(i, j, g->GetU(i, j) - (half ? 2.0 : 1.0) * k * (g->GetP(i, j) - g->GetP(i, j - 1)));
            }
    for (int i = 1; i < g->Ny; i++)
        for (int j = 1; j < g->Nx - 1; j++)
            if (g->FaceGate(MAC::V, i, j) != GateState::WALL)
            {
                const bool half = g->IsPressureOutlet(i, j) || g->IsPressureOutlet(i - 1, j);
                g->SetV(i, j, g->GetV(i, j) - (half ? 2.0 : 1.0) * k * (g->GetP(i, j) - g->GetP(i - 1, j)));
            }
}

double Explicit2D::StableTimestep()
{
    const double h = SIMULATION.GRID_ANT->dh, nu = SIMULATION.EPS;
    const double u = std::max(SIMULATION.GRID_ANT->GetMaxVelocity(), 1e-6);
    return MAX_CFL * std::min({h / u, h * h / (4.0 * nu), 2.0 * nu / (u * u)});
}
