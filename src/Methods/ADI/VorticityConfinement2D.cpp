// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "VorticityConfinement2D.h"
#include "ADISettings.h"

#include <cmath>

using Eigen::VectorXd;

VectorXd VorticityConfinement2D::corner = VectorXd();
VectorXd VorticityConfinement2D::forceX = VectorXd();
VectorXd VorticityConfinement2D::forceY = VectorXd();
std::vector<char> VorticityConfinement2D::active;
bool VorticityConfinement2D::ready = false;

static bool IsSolidWithin(MAC *grid, int i, int j, int margin)
{
    for (int di = -margin; di <= margin; di++)
        for (int dj = -margin; dj <= margin; dj++)
        {
            const int ii = i + di, jj = j + dj;
            if (ii < 0 || jj < 0 || ii >= grid->Ny || jj >= grid->Nx) continue;
            if (grid->GetSolid(ii, jj) == SOLID_CELL) return true;
        }
    return false;
}

void VorticityConfinement2D::Initialize(MAC *grid)
{
    const int Nx = grid->Nx, Ny = grid->Ny;
    corner = VectorXd((long)(Ny + 1) * (Nx + 1));
    forceX = VectorXd((long)Ny * Nx);
    forceY = VectorXd((long)Ny * Nx);
    corner.setZero(); forceX.setZero(); forceY.setZero();

    // A cell is active when it and its eight neighbours are interior fluid -- so every corner
    // around it has a vorticity and every corner of THOSE has one for the gradient.
    const int margin = (std::max)(0, ADI_SETTINGS.vorticitySolidMargin);
    active.assign((size_t)Ny * Nx, 0);
    for (int i = 2; i < Ny - 2; i++)
        for (int j = 2; j < Nx - 2; j++)
        {
            bool ok = true;
            for (int di = -1; di <= 1 && ok; di++)
                for (int dj = -1; dj <= 1 && ok; dj++)
                    ok = (grid->GetSolid(i + di, j + dj) == FLUID_CELL);
            if (ok && margin > 0 && IsSolidWithin(grid, i, j, margin)) ok = false;
            active[(size_t)i * Nx + j] = ok ? 1 : 0;
        }
    ready = true;
}

void VorticityConfinement2D::Apply(MAC *grid, double dt)
{
    if (!ready || ADI_SETTINGS.vorticityEpsilon == 0.0) return;
    const int Nx = grid->Nx, Ny = grid->Ny;
    const double eps = ADI_SETTINGS.vorticityEpsilon;

    // Corner (i, j) is shared by cells (i-1|i, j-1|j) and sits at (FacePos(X, j), FacePos(Y, i)).
    #pragma omp parallel for
    for (int i = 1; i < Ny; i++)
        for (int j = 1; j < Nx; j++)
        {
            const double dvdx = (grid->GetV(i, j) - grid->GetV(i, j - 1)) / grid->FaceGap(MAC::X, j - 1);
            const double dudy = (grid->GetU(i, j) - grid->GetU(i - 1, j)) / grid->FaceGap(MAC::Y, i - 1);
            corner[(long)i * (Nx + 1) + j] = dvdx - dudy;
        }

    // At each active cell centre: omega is the mean of its four corners, and grad|omega| is the
    // difference of the two corner pairs across the cell.
    #pragma omp parallel for
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
        {
            const long c = (long)i * Nx + j;
            forceX[c] = 0.0;
            forceY[c] = 0.0;
            if (!active[(size_t)c]) continue;

            const double w00 = corner[(long)i * (Nx + 1) + j];
            const double w01 = corner[(long)i * (Nx + 1) + j + 1];
            const double w10 = corner[(long)(i + 1) * (Nx + 1) + j];
            const double w11 = corner[(long)(i + 1) * (Nx + 1) + j + 1];
            const double omega = 0.25 * (w00 + w01 + w10 + w11);

            const double gx = 0.5 * ((std::fabs(w01) + std::fabs(w11)) - (std::fabs(w00) + std::fabs(w10))) / grid->Width(MAC::X, j);
            const double gy = 0.5 * ((std::fabs(w10) + std::fabs(w11)) - (std::fabs(w00) + std::fabs(w01))) / grid->Width(MAC::Y, i);
            const double h = std::sqrt(grid->Width(MAC::X, j) * grid->Width(MAC::Y, i));
            const double norm = std::sqrt(gx * gx + gy * gy);
            // Where |omega| is flat there is no core to point at; the normalisation would only
            // amplify round-off into a direction.
            if (norm <= 1e-12 * (std::fabs(omega) / h + 1e-300)) continue;

            forceX[c] = eps * h * (gy / norm) * omega;
            forceY[c] = -eps * h * (gx / norm) * omega;
        }

    // Onto the faces between two cells, averaging the two centres. U is pinned two faces deep
    // in x and V two deep in y (MAC::EnforceVelocityBC), which the ranges leave alone.
    #pragma omp parallel for
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 2; j < Nx - 1; j++)
        {
            const long l = (long)i * Nx + (j - 1), r = (long)i * Nx + j;
            if (!active[(size_t)l] && !active[(size_t)r]) continue;
            if (grid->GetSolid(i, j - 1) != FLUID_CELL || grid->GetSolid(i, j) != FLUID_CELL) continue;
            const double a = dt * 0.5 * (forceX[l] + forceX[r]);
            grid->SetU(i, j, grid->GetU(i, j) + a);
        }
    #pragma omp parallel for
    for (int i = 2; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const long b = (long)(i - 1) * Nx + j, t = (long)i * Nx + j;
            if (!active[(size_t)b] && !active[(size_t)t]) continue;
            if (grid->GetSolid(i - 1, j) != FLUID_CELL || grid->GetSolid(i, j) != FLUID_CELL) continue;
            const double a = dt * 0.5 * (forceY[b] + forceY[t]);
            grid->SetV(i, j, grid->GetV(i, j) + a);
        }
}

void VorticityConfinement2D::Dispose()
{
    corner = VectorXd(); forceX = VectorXd(); forceY = VectorXd();
    std::vector<char>().swap(active);
    ready = false;
}
