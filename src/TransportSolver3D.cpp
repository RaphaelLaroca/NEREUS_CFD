// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "headers/Solvers/TransportSolver3D.h"
#include "headers/Core/Scene.h"

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::VectorXd;

VectorXd TransportSolver3D::low = VectorXd();
VectorXd TransportSolver3D::mid = VectorXd();
VectorXd TransportSolver3D::up = VectorXd();
VectorXd TransportSolver3D::rhs = VectorXd();
VectorXd TransportSolver3D::sol = VectorXd();
VectorXd TransportSolver3D::intermX = VectorXd();
VectorXd TransportSolver3D::intermY = VectorXd();
VectorXd TransportSolver3D::staged = VectorXd();

// Cell (i, j, k) in the scalar's own stride, so the scratch fields line up with GetS.
static inline long CellIndex3D(const MAC *g, int i, int j, int k)
{
    return (long)i * ((long)g->Nx * g->Nz) + (long)j * g->Nz + k;
}

void TransportSolver3D::InitializeTransportSolver3D(MAC *grid)
{
    const int n = (std::max)((std::max)(grid->Nx, grid->Ny), grid->Nz);
    low = VectorXd(n); mid = VectorXd(n); up = VectorXd(n);
    rhs = VectorXd(n); sol = VectorXd(n);
    const long cells = (long)grid->Nx * grid->Ny * grid->Nz;
    intermX = VectorXd(cells); intermX.setZero();
    intermY = VectorXd(cells); intermY.setZero();
    staged = VectorXd(cells); staged.setZero();
    RegisterSimulationStep(TransportSolver3D::SolveTransportStep, "transport", 8);
}

// What sits on the other side of a face that is not an interior neighbour. Same rule as TransportSolver2D.
void TransportSolver3D::CloseLowFace(double w, double vel, double dBoundary, TransportScheme scheme,
                                     const std::string &name, double x, double y, double z, int t)
{
    double bc = 0.0;
    const bool pinned = SCENE.loaded && SCENE.ScalarDirichletAt(name, x, y, z, SIMULATION.time, bc);
    const double share = (scheme == TransportScheme::Central) ? 0.5 * vel : (std::max)(vel, 0.0);
    low(t) = 0.0;
    if (pinned) rhs(t) += w * share * bc;
    else        mid(t) -= w * share;
    if (pinned && dBoundary > 0.0) { mid(t) += dBoundary; rhs(t) += dBoundary * bc; }
}

void TransportSolver3D::CloseHighFace(double w, double vel, double dBoundary, TransportScheme scheme,
                                      const std::string &name, double x, double y, double z, int t)
{
    double bc = 0.0;
    const bool pinned = SCENE.loaded && SCENE.ScalarDirichletAt(name, x, y, z, SIMULATION.time, bc);
    const double share = (scheme == TransportScheme::Central) ? 0.5 * vel : (std::min)(vel, 0.0);
    up(t) = 0.0;
    if (pinned) rhs(t) -= w * share * bc;
    else        mid(t) += w * share;
    if (pinned && dBoundary > 0.0) { mid(t) += dBoundary; rhs(t) += dBoundary * bc; }
}

// The convective part of one fluid row in flux form, central or upwind, the same expressions as 2D.
void TransportSolver3D::FillRow(int t, double w, double velLow, double velHigh, TransportScheme scheme,
                                bool hasLow, bool hasHigh)
{
    if (scheme == TransportScheme::Central)
    {
        mid(t) += w * 0.5 * (velHigh - velLow);
        if (hasHigh) up(t) += w * 0.5 * velHigh;
        if (hasLow)  low(t) -= w * 0.5 * velLow;
    }
    else
    {
        mid(t) += w * ((std::max)(velHigh, 0.0) - (std::min)(velLow, 0.0));
        if (hasHigh) up(t) += w * (std::min)(velHigh, 0.0);
        if (hasLow)  low(t) -= w * (std::max)(velLow, 0.0);
    }
}

// The reaction factor, once per step; see TransportSolver2D::React. Serial for the tinyexpr source.
void TransportSolver3D::React(MAC *gridAnt, MAC *gridSol, int src, const ScalarTransport &tr, const SceneDesc::Expr *source)
{
    const int Nx = gridSol->Nx, Ny = gridSol->Ny, Nz = gridSol->Nz;
    const double dt = SIMULATION.dt;
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < Nz; k++)
            {
                const double phi = gridAnt->GetS(src, i, j, k);
                double out = phi;
                if (gridSol->GetSolid(i, j, k) == FLUID_CELL)
                {
                    const double s = source ? source->Eval(gridSol->CentrePos(MAC::X, j), gridSol->CentrePos(MAC::Y, i), gridSol->CentrePos(MAC::Z, k),
                                                           SIMULATION.time) : 0.0;
                    if (tr.gamma >= 0.0) out = (phi + dt * s) / (1.0 + dt * tr.gamma);
                    else                 out = (1.0 - dt * tr.gamma) * phi + dt * s;
                }
                staged[CellIndex3D(gridSol, i, j, k)] = out;
            }
}

// Sweep implicit in x. Reads the scalar from gridAnt (or the reaction factor), writes every cell of intermX.
void TransportSolver3D::SweepX(MAC *gridAnt, MAC *gridSol, int handle, const ScalarTransport &tr,
                               const std::string &name, bool fromStaged)
{
    const int Nx = gridSol->Nx, Ny = gridSol->Ny, Nz = gridSol->Nz;
    const double dt = SIMULATION.dt;
    const double de = dt * tr.epsilon;

    for (int i = 0; i < Ny; i++)
        for (int k = 0; k < Nz; k++)
        {
            for (int j = 0; j < Nx; j++)
            {
                const double w = dt / gridSol->Width(MAC::X, j);
                const double uL = gridSol->GetU(i, j, k);
                const double uR = gridSol->GetU(i, j + 1, k);

                low(j) = 0.0; mid(j) = 1.0; up(j) = 0.0;
                rhs(j) = fromStaged ? staged[CellIndex3D(gridSol, i, j, k)] : gridAnt->GetS(handle, i, j, k);

                if (gridSol->GetSolid(i, j, k) != FLUID_CELL) continue;
                FillRow(j, w, uL, uR, tr.scheme, j > 0, j + 1 < Nx);

                const bool lowOpen = (j > 0) && gridSol->GetSolid(i, j - 1, k) == FLUID_CELL;
                const bool upOpen = (j + 1 < Nx) && gridSol->GetSolid(i, j + 1, k) == FLUID_CELL;
                const double dB = de / (gridSol->Width(MAC::X, j) * 0.5 * gridSol->Width(MAC::X, j));
                if (!lowOpen) CloseLowFace(w, uL, dB, tr.scheme, name, gridSol->FacePos(MAC::X, j), gridSol->CentrePos(MAC::Y, i), gridSol->CentrePos(MAC::Z, k), j);
                if (!upOpen)  CloseHighFace(w, uR, dB, tr.scheme, name, gridSol->FacePos(MAC::X, j + 1), gridSol->CentrePos(MAC::Y, i), gridSol->CentrePos(MAC::Z, k), j);
                if (de > 0.0)
                {
                    if (lowOpen) { const double a = de / (gridSol->Width(MAC::X, j) * gridSol->FaceGap(MAC::X, j - 1)); mid(j) += a; low(j) -= a; }
                    if (upOpen)  { const double a = de / (gridSol->Width(MAC::X, j) * gridSol->FaceGap(MAC::X, j));     mid(j) += a; up(j) -= a; }
                }
                if (tr.form == TransportForm::Advective) mid(j) -= w * (uR - uL);
            }

            TDMA_VECTORIZED_N(low, mid, up, rhs, sol, Nx);
            for (int j = 0; j < Nx; j++) intermX[CellIndex3D(gridSol, i, j, k)] = sol(j);
        }
}

// Sweep implicit in y. Reads intermX, writes every cell of intermY.
void TransportSolver3D::SweepY(MAC *gridSol, const ScalarTransport &tr, const std::string &name)
{
    const int Nx = gridSol->Nx, Ny = gridSol->Ny, Nz = gridSol->Nz;
    const double dt = SIMULATION.dt;
    const double de = dt * tr.epsilon;

    for (int j = 0; j < Nx; j++)
        for (int k = 0; k < Nz; k++)
        {
            for (int i = 0; i < Ny; i++)
            {
                const double w = dt / gridSol->Width(MAC::Y, i);
                const double vL = gridSol->GetV(i, j, k);
                const double vR = gridSol->GetV(i + 1, j, k);

                low(i) = 0.0; mid(i) = 1.0; up(i) = 0.0;
                rhs(i) = intermX[CellIndex3D(gridSol, i, j, k)];

                if (gridSol->GetSolid(i, j, k) != FLUID_CELL) continue;
                FillRow(i, w, vL, vR, tr.scheme, i > 0, i + 1 < Ny);

                const bool lowOpen = (i > 0) && gridSol->GetSolid(i - 1, j, k) == FLUID_CELL;
                const bool upOpen = (i + 1 < Ny) && gridSol->GetSolid(i + 1, j, k) == FLUID_CELL;
                const double dB = de / (gridSol->Width(MAC::Y, i) * 0.5 * gridSol->Width(MAC::Y, i));
                if (!lowOpen) CloseLowFace(w, vL, dB, tr.scheme, name, gridSol->CentrePos(MAC::X, j), gridSol->FacePos(MAC::Y, i), gridSol->CentrePos(MAC::Z, k), i);
                if (!upOpen)  CloseHighFace(w, vR, dB, tr.scheme, name, gridSol->CentrePos(MAC::X, j), gridSol->FacePos(MAC::Y, i + 1), gridSol->CentrePos(MAC::Z, k), i);
                if (de > 0.0)
                {
                    if (lowOpen) { const double a = de / (gridSol->Width(MAC::Y, i) * gridSol->FaceGap(MAC::Y, i - 1)); mid(i) += a; low(i) -= a; }
                    if (upOpen)  { const double a = de / (gridSol->Width(MAC::Y, i) * gridSol->FaceGap(MAC::Y, i));     mid(i) += a; up(i) -= a; }
                }
                if (tr.form == TransportForm::Advective) mid(i) -= w * (vR - vL);
            }

            TDMA_VECTORIZED_N(low, mid, up, rhs, sol, Ny);
            for (int i = 0; i < Ny; i++) intermY[CellIndex3D(gridSol, i, j, k)] = sol(i);
        }
}

// Sweep implicit in z. Reads intermY, writes the scalar into gridSol.
void TransportSolver3D::SweepZ(MAC *gridSol, int handle, const ScalarTransport &tr, const std::string &name)
{
    const int Nx = gridSol->Nx, Ny = gridSol->Ny, Nz = gridSol->Nz;
    const double dt = SIMULATION.dt;
    const double de = dt * tr.epsilon;

    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                const double w = dt / gridSol->Width(MAC::Z, k);
                const double wL = gridSol->GetW(i, j, k);
                const double wR = gridSol->GetW(i, j, k + 1);

                low(k) = 0.0; mid(k) = 1.0; up(k) = 0.0;
                rhs(k) = intermY[CellIndex3D(gridSol, i, j, k)];

                if (gridSol->GetSolid(i, j, k) != FLUID_CELL) continue;
                FillRow(k, w, wL, wR, tr.scheme, k > 0, k + 1 < Nz);

                const bool lowOpen = (k > 0) && gridSol->GetSolid(i, j, k - 1) == FLUID_CELL;
                const bool upOpen = (k + 1 < Nz) && gridSol->GetSolid(i, j, k + 1) == FLUID_CELL;
                const double dB = de / (gridSol->Width(MAC::Z, k) * 0.5 * gridSol->Width(MAC::Z, k));
                if (!lowOpen) CloseLowFace(w, wL, dB, tr.scheme, name, gridSol->CentrePos(MAC::X, j), gridSol->CentrePos(MAC::Y, i), gridSol->FacePos(MAC::Z, k), k);
                if (!upOpen)  CloseHighFace(w, wR, dB, tr.scheme, name, gridSol->CentrePos(MAC::X, j), gridSol->CentrePos(MAC::Y, i), gridSol->FacePos(MAC::Z, k + 1), k);
                if (de > 0.0)
                {
                    if (lowOpen) { const double a = de / (gridSol->Width(MAC::Z, k) * gridSol->FaceGap(MAC::Z, k - 1)); mid(k) += a; low(k) -= a; }
                    if (upOpen)  { const double a = de / (gridSol->Width(MAC::Z, k) * gridSol->FaceGap(MAC::Z, k));     mid(k) += a; up(k) -= a; }
                }
                if (tr.form == TransportForm::Advective) mid(k) -= w * (wR - wL);
            }

            TDMA_VECTORIZED_N(low, mid, up, rhs, sol, Nz);
            for (int k = 0; k < Nz; k++) gridSol->SetS(handle, i, j, k, sol(k));
        }
}

void TransportSolver3D::SolveTransportStep(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
{
    for (int h = 0; h < gridSol->ScalarCount(); h++)
    {
        const int src = gridSol->MatchScalar(*gridAnt, h);
        if (src >= 0)
        {
            const ScalarTransport &tr = gridSol->ScalarTransportOf(h);
            const std::string &name = gridSol->ScalarName(h);
            if (tr.solver == ScalarSolver::None)   // the method updates it; only its ring is set
            {
                gridSol->EnforceScalarBC(h, SIMULATION.time);
                continue;
            }
            const SceneDesc::Expr *source = SCENE.loaded ? SCENE.ScalarSource(SCENE.ScalarIndex(name)) : nullptr;
            const bool react = (tr.gamma != 0.0) || source != nullptr;
            if (react) React(gridAnt, gridSol, src, tr, source);
            SweepX(gridAnt, gridSol, src, tr, name, react);
            SweepY(gridSol, tr, name);
            SweepZ(gridSol, h, tr, name);
            gridSol->EnforceScalarBC(h, SIMULATION.time);
        }
    }
}

void TransportSolver3D::Dispose()
{
    low = mid = up = rhs = sol = intermX = intermY = staged = VectorXd();
}
