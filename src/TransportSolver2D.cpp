// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "headers/Solvers/TransportSolver2D.h"
#include "headers/Core/Scene.h"

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::VectorXd;

VectorXd TransportSolver2D::low = VectorXd();
VectorXd TransportSolver2D::mid = VectorXd();
VectorXd TransportSolver2D::up = VectorXd();
VectorXd TransportSolver2D::rhs = VectorXd();
VectorXd TransportSolver2D::sol = VectorXd();
VectorXd TransportSolver2D::interm = VectorXd();
VectorXd TransportSolver2D::staged = VectorXd();

void TransportSolver2D::InitializeTransportSolver2D(MAC *grid)
{
    const int n = (std::max)(grid->Nx, grid->Ny);
    low = VectorXd(n); mid = VectorXd(n); up = VectorXd(n);
    rhs = VectorXd(n); sol = VectorXd(n);
    interm = VectorXd((long)grid->Nx * grid->Ny);
    interm.setZero();
    staged = VectorXd((long)grid->Nx * grid->Ny);
    staged.setZero();
    RegisterSimulationStep(TransportSolver2D::SolveTransportStep, "transport", 8);
}

// Beyond a non-neighbour face: the scene's Dirichlet value (fluxes to the right-hand side) or zero gradient
// (convective flux on the diagonal). `dBoundary` = dt*epsilon/(h * h/2), the half-cell link.
void TransportSolver2D::CloseLowFace(double w, double vel, double dBoundary, TransportScheme scheme,
                                     const std::string &name, double x, double y, int t)
{
    double bc = 0.0;
    const bool pinned = SCENE.loaded &&
                        SCENE.ScalarDirichletAt(name, x, y, SIMULATION.domain.z0,
                                                SIMULATION.time, bc);
    const double share = (scheme == TransportScheme::Central) ? 0.5 * vel
                                                              : (std::max)(vel, 0.0);
    low(t) = 0.0;
    if (pinned) rhs(t) += w * share * bc;
    else        mid(t) -= w * share;
    if (pinned && dBoundary > 0.0) { mid(t) += dBoundary; rhs(t) += dBoundary * bc; }
}

void TransportSolver2D::CloseHighFace(double w, double vel, double dBoundary, TransportScheme scheme,
                                      const std::string &name, double x, double y, int t)
{
    double bc = 0.0;
    const bool pinned = SCENE.loaded &&
                        SCENE.ScalarDirichletAt(name, x, y, SIMULATION.domain.z0,
                                                SIMULATION.time, bc);
    const double share = (scheme == TransportScheme::Central) ? 0.5 * vel
                                                              : (std::min)(vel, 0.0);
    up(t) = 0.0;
    if (pinned) rhs(t) -= w * share * bc;
    else        mid(t) += w * share;
    if (pinned && dBoundary > 0.0) { mid(t) += dBoundary; rhs(t) += dBoundary * bc; }
}

// The convective part of one fluid row in flux form, central or upwind. The advective form is this
// row minus the phi*du/dx term, applied by the sweep after the faces are closed.
void TransportSolver2D::FillRow(int t, double w, double velLow, double velHigh, TransportScheme scheme,
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

// The reaction factor, once per step: (1 + dt gamma) phi* = phi + dt S; a negative gamma goes explicitly.
// Serial: the source is a tinyexpr.
void TransportSolver2D::React(MAC *gridAnt, MAC *gridSol, int src, const ScalarTransport &tr, const SceneDesc::Expr *source)
{
    const int Nx = gridSol->Nx, Ny = gridSol->Ny;
    const double dt = SIMULATION.dt;
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
        {
            const double phi = gridAnt->GetS(src, i, j);
            double out = phi;
            if (gridSol->GetSolid(i, j) == FLUID_CELL)
            {
                const double s = source ? source->Eval(gridSol->CentrePos(MAC::X, j), gridSol->CentrePos(MAC::Y, i), SIMULATION.domain.z0,
                                                       SIMULATION.time) : 0.0;
                if (tr.gamma >= 0.0) out = (phi + dt * s) / (1.0 + dt * tr.gamma);
                else                 out = (1.0 - dt * tr.gamma) * phi + dt * s;
            }
            staged[(long)i * Nx + j] = out;
        }
}

// Sweep implicit in x. Reads the scalar from gridAnt (or from the reaction factor), the velocity from
// gridSol, writes every cell of `interm` -- solved or pinned, so nothing of the previous step survives.
void TransportSolver2D::SweepX(MAC *gridAnt, MAC *gridSol, int handle, const ScalarTransport &tr,
                               const std::string &name, bool fromStaged)
{
    const int Nx = gridSol->Nx, Ny = gridSol->Ny;
    const double dt = SIMULATION.dt;
    const double de = dt * tr.epsilon;

    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            const double w = dt / gridSol->Width(MAC::X, j);
            const double uL = gridSol->GetU(i, j);
            const double uR = gridSol->GetU(i, j + 1);

            low(j) = 0.0; mid(j) = 1.0; up(j) = 0.0;
            rhs(j) = fromStaged ? staged[(long)i * Nx + j] : gridAnt->GetS(handle, i, j);

            if (gridSol->GetSolid(i, j) != FLUID_CELL) continue;
            FillRow(j, w, uL, uR, tr.scheme, j > 0, j + 1 < Nx);

            const bool lowOpen = (j > 0) && gridSol->GetSolid(i, j - 1) == FLUID_CELL;
            const bool upOpen = (j + 1 < Nx) && gridSol->GetSolid(i, j + 1) == FLUID_CELL;
            const double dB = de / (gridSol->Width(MAC::X, j) * 0.5 * gridSol->Width(MAC::X, j));
            if (!lowOpen) CloseLowFace(w, uL, dB, tr.scheme, name, gridSol->FacePos(MAC::X, j), gridSol->CentrePos(MAC::Y, i), j);
            if (!upOpen)  CloseHighFace(w, uR, dB, tr.scheme, name, gridSol->FacePos(MAC::X, j + 1), gridSol->CentrePos(MAC::Y, i), j);
            if (de > 0.0)
            {
                if (lowOpen) { const double a = de / (gridSol->Width(MAC::X, j) * gridSol->FaceGap(MAC::X, j - 1)); mid(j) += a; low(j) -= a; }
                if (upOpen)  { const double a = de / (gridSol->Width(MAC::X, j) * gridSol->FaceGap(MAC::X, j));     mid(j) += a; up(j) -= a; }
            }
            if (tr.form == TransportForm::Advective) mid(j) -= w * (uR - uL);
        }

        TDMA_VECTORIZED_N(low, mid, up, rhs, sol, Nx);

        for (int j = 0; j < Nx; j++)
            interm[(long)i * Nx + j] = sol(j);
    }
}

// Sweep implicit in y. Reads `interm`, writes the scalar into gridSol.
void TransportSolver2D::SweepY(MAC *gridSol, int handle, const ScalarTransport &tr, const std::string &name)
{
    const int Nx = gridSol->Nx, Ny = gridSol->Ny;
    const double dt = SIMULATION.dt;
    const double de = dt * tr.epsilon;

    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            const double w = dt / gridSol->Width(MAC::Y, i);
            const double vL = gridSol->GetV(i, j);
            const double vR = gridSol->GetV(i + 1, j);

            low(i) = 0.0; mid(i) = 1.0; up(i) = 0.0;
            rhs(i) = interm[(long)i * Nx + j];

            if (gridSol->GetSolid(i, j) != FLUID_CELL) continue;
            FillRow(i, w, vL, vR, tr.scheme, i > 0, i + 1 < Ny);

            const bool lowOpen = (i > 0) && gridSol->GetSolid(i - 1, j) == FLUID_CELL;
            const bool upOpen = (i + 1 < Ny) && gridSol->GetSolid(i + 1, j) == FLUID_CELL;
            const double dB = de / (gridSol->Width(MAC::Y, i) * 0.5 * gridSol->Width(MAC::Y, i));
            if (!lowOpen) CloseLowFace(w, vL, dB, tr.scheme, name, gridSol->CentrePos(MAC::X, j), gridSol->FacePos(MAC::Y, i), i);
            if (!upOpen)  CloseHighFace(w, vR, dB, tr.scheme, name, gridSol->CentrePos(MAC::X, j), gridSol->FacePos(MAC::Y, i + 1), i);
            if (de > 0.0)
            {
                if (lowOpen) { const double a = de / (gridSol->Width(MAC::Y, i) * gridSol->FaceGap(MAC::Y, i - 1)); mid(i) += a; low(i) -= a; }
                if (upOpen)  { const double a = de / (gridSol->Width(MAC::Y, i) * gridSol->FaceGap(MAC::Y, i));     mid(i) += a; up(i) -= a; }
            }
            if (tr.form == TransportForm::Advective) mid(i) -= w * (vR - vL);
        }

        TDMA_VECTORIZED_N(low, mid, up, rhs, sol, Ny);

        for (int i = 0; i < Ny; i++)
            gridSol->SetS(handle, i, j, sol(i));
    }
}

void TransportSolver2D::SolveTransportStep(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time)
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
            SweepY(gridSol, h, tr, name);
            gridSol->EnforceScalarBC(h, SIMULATION.time);
        }
    }
}

void TransportSolver2D::Dispose()
{
    low = mid = up = rhs = sol = interm = staged = VectorXd();
}
