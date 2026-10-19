// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ADI3D.h"
#include "ADISettings.h"
#include "../PPESolver/PressureSolver3D.h"
// X

// Y

// Z

MAC ADI3D::X_STEP_SOL = MAC();
MAC ADI3D::Y_STEP_SOL = MAC();
MAC ADI3D::Z_STEP_SOL = MAC();

double ADI3D::dh = 0.0;

Vec3(*ADI3D::VelocityBorderFunction)(double, double, double,double) = nullptr;
double(*ADI3D::PressureFunction)(double, double, double,double) = nullptr;

// Rectilinear coefficients for the 3D sweeps.
// bw/fw are the upwind one-sided convective coefficients, dt_dir/h- and dt_dir/h+.
struct ADICoef { double lo, up, rho, bw, fw; };

static inline ADICoef CoefFrom(double nu_dt, double dt_dir, double hm, double hp, double w)
{ return ADICoef{ nu_dt / (hm * w), nu_dt / (hp * w), dt_dir / (hm + hp), dt_dir / hm, dt_dir / hp }; }

// Along the component's OWN normal.
static inline ADICoef CoefNormX(const MAC *g, int j, double nu_dt, double dt_dir, double sig, double rho)
{ return g->nonUniform ? CoefFrom(nu_dt, dt_dir, g->Width(MAC::X, j-1), g->Width(MAC::X, j), g->FaceGap(MAC::X, j-1)) : ADICoef{sig,sig,rho,2.0*rho,2.0*rho}; }
static inline ADICoef CoefNormY(const MAC *g, int i, double nu_dt, double dt_dir, double sig, double rho)
{ return g->nonUniform ? CoefFrom(nu_dt, dt_dir, g->Width(MAC::Y, i-1), g->Width(MAC::Y, i), g->FaceGap(MAC::Y, i-1)) : ADICoef{sig,sig,rho,2.0*rho,2.0*rho}; }
static inline ADICoef CoefNormZ(const MAC *g, int k, double nu_dt, double dt_dir, double sig, double rho)
{ return g->nonUniform ? CoefFrom(nu_dt, dt_dir, g->Width(MAC::Z, k-1), g->Width(MAC::Z, k), g->FaceGap(MAC::Z, k-1)) : ADICoef{sig,sig,rho,2.0*rho,2.0*rho}; }

// Along a TANGENTIAL axis -- the same formula with the two roles swapped.
static inline ADICoef CoefTangX(const MAC *g, int j, double nu_dt, double dt_dir, double sig, double rho)
{ return g->nonUniform ? CoefFrom(nu_dt, dt_dir, g->FaceGap(MAC::X, j-1), g->FaceGap(MAC::X, j), g->Width(MAC::X, j)) : ADICoef{sig,sig,rho,2.0*rho,2.0*rho}; }
static inline ADICoef CoefTangY(const MAC *g, int i, double nu_dt, double dt_dir, double sig, double rho)
{ return g->nonUniform ? CoefFrom(nu_dt, dt_dir, g->FaceGap(MAC::Y, i-1), g->FaceGap(MAC::Y, i), g->Width(MAC::Y, i)) : ADICoef{sig,sig,rho,2.0*rho,2.0*rho}; }
static inline ADICoef CoefTangZ(const MAC *g, int k, double nu_dt, double dt_dir, double sig, double rho)
{ return g->nonUniform ? CoefFrom(nu_dt, dt_dir, g->FaceGap(MAC::Z, k-1), g->FaceGap(MAC::Z, k), g->Width(MAC::Z, k)) : ADICoef{sig,sig,rho,2.0*rho,2.0*rho}; }

// PARALLEL OPENMP TESTING

void ADI3D::SolveADI_X_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridAnt->Nx;
    int Ny = gridAnt->Ny;
    int Nz = gridAnt->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    // Note: dt/(dh*dh*3) for 3D diffusion
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int u_size = Nx + 1 - 4;

    #pragma omp parallel
    {
        // Allocation: Once per thread
        Eigen::VectorXd low(u_size), mid(u_size), up(u_size);
        Eigen::VectorXd RHS(u_size), SOL(u_size);

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {
            for(int k = 1; k < Nz-1; k++) {

                // FIX GHOSTING: Reset tridiagonal vectors for every (i, k) pencil
                for(int m = 0; m < u_size; m++) {
                    const ADICoef cm = CoefNormX(g, m + 2, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int j = 2; j < Nx+1-2; j++) {
                    const ADICoef cI = CoefNormX(g, j, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefTangY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefTangZ(g, k, nu_dt, dt_dir, sig, rho);

                    // Explicit terms (Y and Z directions)
                    double Y_DIFF = (gridAnt->GetU(i+1,j,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i-1,j,k));
                    const double Y_DIFF_NU = NU ? (cA.up*((gridAnt->GetU(i+1,j,k)) - (gridAnt->GetU(i,j,k))) - cA.lo*((gridAnt->GetU(i,j,k)) - (gridAnt->GetU(i-1,j,k)))) : 0.0;
                    double Z_DIFF = (gridAnt->GetU(i,j,k+1) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j,k-1));
                    const double Z_DIFF_NU = NU ? (cB.up*((gridAnt->GetU(i,j,k+1)) - (gridAnt->GetU(i,j,k))) - cB.lo*((gridAnt->GetU(i,j,k)) - (gridAnt->GetU(i,j,k-1)))) : 0.0;

                    double V_CONV = velocityField->getVatU(i,j,k) * (gridAnt->GetU(i+1,j,k) - gridAnt->GetU(i-1,j,k));
                    double W_CONV = velocityField->getWatU(i,j,k) * (gridAnt->GetU(i,j,k+1) - gridAnt->GetU(i,j,k-1));

                    const double DIFF_EXP = NU ? (Y_DIFF_NU + Z_DIFF_NU) : sig*(Y_DIFF + Z_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::Y, i,j,k), UPWIND, gridAnt->GetU(i-1,j,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i+1,j,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::Z, i,j,k), UPWIND, gridAnt->GetU(i,j,k-1), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j,k+1)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->getVatU(i,j,k), gridAnt->GetU(i-1,j,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i+1,j,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->getWatU(i,j,k), gridAnt->GetU(i,j,k-1), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j,k+1), cB.bw, cB.fw))
                        : NU ? (cA.rho*V_CONV + cB.rho*W_CONV) : rho*(V_CONV + W_CONV);
                    RHS(c) = gridAnt->GetU(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::U, MAC::Y, i, j, k, cA.lo, cA.up, gridAnt->GetU(i-1,j,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i+1,j,k));
                    RHS(c) += WallExplicit(gridSol, MAC::U, MAC::Z, i, j, k, cB.lo, cB.up, gridAnt->GetU(i,j,k-1), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j,k+1));
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::X, i,j,k);   // grad p^n

                    // Implicit Convection (X direction)
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::X, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < u_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetU(i,Nx-1,k);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetU(i,1,k);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->GetU(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < u_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double u_vel = cI.rho * velocityField->GetU(i,j,k);
                        if(c < u_size - 1) up(c)  += u_vel;
                        if(c > 0)          low(c) -= u_vel;
                    }
                    c++;
                }

                // Boundary Conditions
                const ADICoef cLo = CoefNormX(g, 2, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefNormX(g, 2 + u_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo*gridSol->GetU(i,1,k);
                    RHS(u_size-1) += cHi.up*gridSol->GetU(i,Nx-1,k);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->GetU(i,2,k), cLo.bw, cLo.fw).low * gridSol->GetU(i,1,k) + cLo.lo*gridSol->GetU(i,1,k);
                    RHS(u_size-1) += -UpwindCoefficients(velocityField->GetU(i,Nx-2,k), cHi.bw, cHi.fw).up * gridSol->GetU(i,Nx-1,k) + cHi.up*gridSol->GetU(i,Nx-1,k);
                }
                else
                {
                    RHS(0)        +=  cLo.rho*gridSol->GetU(i,1,k)     * velocityField->GetU(i,2,k)      + cLo.lo*gridSol->GetU(i,1,k);
                    RHS(u_size-1) += -cHi.rho*gridSol->GetU(i,Nx-1,k)   * velocityField->GetU(i,Nx-2,k)    + cHi.up*gridSol->GetU(i,Nx-1,k);
                }

                // Solid Masking
                for(int j = 0; j < u_size; j++) {
                    int grid_j = j + 2;
                    if(gridSol->FaceGate(MAC::U, i, grid_j, k) == GateState::WALL) {
                        RHS(j) = 0.0;
                        mid(j) = 1.0;
                        if(j > 0) low(j) = 0.0;
                        if(j < u_size-1) up(j) = 0.0;
                        // Zero out dependencies from neighbors
                        if(j > 0) up(j-1) = 0.0;
                        if(j < u_size-1) low(j+1) = 0.0;
                    }
                }

                // Solve the tridiagonal pencil
                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                // Write solution back
                for(int j = 0; j < u_size; j++) {
                    gridSol->SetU(i, j + 2, k, SOL(j));
                }
            }
        }
    }
}
void ADI3D::SolveADI_Y_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int u_size = Ny - 2;

    #pragma omp parallel
    {
        // Thread-local allocation (O(N) vs your previous O(N^2))
        Eigen::VectorXd low(u_size), mid(u_size), up(u_size);
        Eigen::VectorXd RHS(u_size), SOL(u_size);

        #pragma omp for
        for(int j = 2; j < Nx+1-2; j++) {
            for(int k = 1; k < Nz-1; k++) {

                // RESET VECTORS: Clear ghost values from the previous pencil
                for(int m = 0; m < u_size; m++) {
                    const ADICoef cm = CoefTangY(g, m + 1, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    const ADICoef cI = CoefTangY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefNormX(g, j, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefTangZ(g, k, nu_dt, dt_dir, sig, rho);

                    // Explicit X and Z terms
                    double X_DIFF = (gridAnt->GetU(i,j+1,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j-1,k));
                    const double X_DIFF_NU = NU ? (cA.up*((gridAnt->GetU(i,j+1,k)) - (gridAnt->GetU(i,j,k))) - cA.lo*((gridAnt->GetU(i,j,k)) - (gridAnt->GetU(i,j-1,k)))) : 0.0;
                    double Z_DIFF = (gridAnt->GetU(i,j,k+1) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j,k-1));
                    const double Z_DIFF_NU = NU ? (cB.up*((gridAnt->GetU(i,j,k+1)) - (gridAnt->GetU(i,j,k))) - cB.lo*((gridAnt->GetU(i,j,k)) - (gridAnt->GetU(i,j,k-1)))) : 0.0;

                    double U_CONV = velocityField->GetU(i,j,k) * (gridAnt->GetU(i,j+1,k) - gridAnt->GetU(i,j-1,k));
                    double W_CONV = velocityField->getWatU(i,j,k) * (gridAnt->GetU(i,j,k+1) - gridAnt->GetU(i,j,k-1));

                    const double DIFF_EXP = NU ? (X_DIFF_NU + Z_DIFF_NU) : sig*(X_DIFF + Z_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::X, i,j,k), UPWIND, gridAnt->GetU(i,j-1,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j+1,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::Z, i,j,k), UPWIND, gridAnt->GetU(i,j,k-1), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j,k+1)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->GetU(i,j,k), gridAnt->GetU(i,j-1,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j+1,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->getWatU(i,j,k), gridAnt->GetU(i,j,k-1), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j,k+1), cB.bw, cB.fw))
                        : NU ? (cA.rho*U_CONV + cB.rho*W_CONV) : rho*(U_CONV + W_CONV);
                    RHS(c) = gridAnt->GetU(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::U, MAC::Z, i, j, k, cB.lo, cB.up, gridAnt->GetU(i,j,k-1), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j,k+1));
                    mid(c) += WallDiagonal(gridSol, MAC::U, MAC::Y, i, j, k, cI.lo, cI.up);
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::X, i,j,k);   // grad p^n

                    // Implicit Y Convection
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::Y, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < u_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetU(Ny-1,j,k);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetU(0,j,k);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->getVatU(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < u_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double v_vel = cI.rho * velocityField->getVatU(i,j,k);
                        if(c < u_size - 1) up(c)  += v_vel;
                        if(c > 0)          low(c) -= v_vel;
                    }
                    c++;
                }

                // Boundary Conditions (i=0 and i=Ny-1)
                const ADICoef cLo = CoefTangY(g, 1, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefTangY(g, 1 + u_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo * gridSol->GetU(0,j,k);
                    RHS(u_size-1) += cHi.up * gridSol->GetU(Ny-1,j,k);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->getVatU(1,j,k), cLo.bw, cLo.fw).low * gridSol->GetU(0,j,k) + cLo.lo * gridSol->GetU(0,j,k);
                    RHS(u_size-1) += -UpwindCoefficients(velocityField->getVatU(Ny-2,j,k), cHi.bw, cHi.fw).up * gridSol->GetU(Ny-1,j,k) + cHi.up * gridSol->GetU(Ny-1,j,k);
                }
                else
                {
                    RHS(0)        +=  cLo.rho * velocityField->getVatU(1,j,k)    * gridSol->GetU(0,j,k)    + cLo.lo * gridSol->GetU(0,j,k);
                    RHS(u_size-1) += -cHi.rho * velocityField->getVatU(Ny-2,j,k) * gridSol->GetU(Ny-1,j,k) + cHi.up * gridSol->GetU(Ny-1,j,k);
                }
                RHS(0)        += WallRing(gridSol, MAC::U, MAC::Y, 1,j,k, -1, cLo.lo, gridSol->GetU(0,j,k));
                RHS(u_size-1) += WallRing(gridSol, MAC::U, MAC::Y, Ny-2,j,k, +1, cHi.up, gridSol->GetU(Ny-1,j,k));

                // Solid Masking
                for(int i = 0; i < u_size; i++) {
                    int grid_i = i + 1;
                    if(gridSol->FaceGate(MAC::U, grid_i, j, k) == GateState::WALL) {
                        RHS(i) = 0.0;
                        mid(i) = 1.0;
                        if(i > 0) low(i) = 0.0;
                        if(i < u_size-1) up(i) = 0.0;
                        if(i > 0) up(i-1) = 0.0;
                        if(i < u_size-1) low(i+1) = 0.0;
                    }
                }

                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                for(int i = 0; i < u_size; i++) {
                    gridSol->SetU(i + 1, j, k, SOL(i));
                }
            }
        }
    }
}
void ADI3D::SolveADI_Z_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int u_size = Nz - 2;

    #pragma omp parallel
    {
        Eigen::VectorXd low(u_size), mid(u_size), up(u_size);
        Eigen::VectorXd RHS(u_size), SOL(u_size);

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {
            for(int j = 2; j < Nx+1-2; j++) {

                // RESET VECTORS for every (i, j) pencil
                for(int m = 0; m < u_size; m++) {
                    const ADICoef cm = CoefTangZ(g, m + 1, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int k = 1; k < Nz-1; k++) {
                    const ADICoef cI = CoefTangZ(g, k, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefTangY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefNormX(g, j, nu_dt, dt_dir, sig, rho);

                    // Explicit Y and X terms
                    double Y_DIFF = (gridAnt->GetU(i+1,j,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i-1,j,k));
                    const double Y_DIFF_NU = NU ? (cA.up*((gridAnt->GetU(i+1,j,k)) - (gridAnt->GetU(i,j,k))) - cA.lo*((gridAnt->GetU(i,j,k)) - (gridAnt->GetU(i-1,j,k)))) : 0.0;
                    double X_DIFF = (gridAnt->GetU(i,j+1,k) - 2*gridAnt->GetU(i,j,k) + gridAnt->GetU(i,j-1,k));
                    const double X_DIFF_NU = NU ? (cB.up*((gridAnt->GetU(i,j+1,k)) - (gridAnt->GetU(i,j,k))) - cB.lo*((gridAnt->GetU(i,j,k)) - (gridAnt->GetU(i,j-1,k)))) : 0.0;

                    double U_CONV = velocityField->GetU(i,j,k) * (gridAnt->GetU(i,j+1,k) - gridAnt->GetU(i,j-1,k));
                    double V_CONV = velocityField->getVatU(i,j,k) * (gridAnt->GetU(i+1,j,k) - gridAnt->GetU(i-1,j,k));

                    const double DIFF_EXP = NU ? (Y_DIFF_NU + X_DIFF_NU) : sig*(Y_DIFF + X_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::X, i,j,k), UPWIND, gridAnt->GetU(i,j-1,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j+1,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::Y, i,j,k), UPWIND, gridAnt->GetU(i-1,j,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i+1,j,k)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->getVatU(i,j,k), gridAnt->GetU(i-1,j,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i+1,j,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->GetU(i,j,k), gridAnt->GetU(i,j-1,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i,j+1,k), cB.bw, cB.fw))
                        : NU ? (cA.rho*V_CONV + cB.rho*U_CONV) : rho*(V_CONV + U_CONV);
                    RHS(c) = gridAnt->GetU(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::U, MAC::Y, i, j, k, cA.lo, cA.up, gridAnt->GetU(i-1,j,k), gridAnt->GetU(i,j,k), gridAnt->GetU(i+1,j,k));
                    mid(c) += WallDiagonal(gridSol, MAC::U, MAC::Z, i, j, k, cI.lo, cI.up);
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::X, i,j,k);   // grad p^n

                    // Implicit Z Convection
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::U, MAC::Z, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < u_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetU(i,j,Nz-1);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetU(i,j,0);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->getWatU(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < u_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double w_vel = cI.rho * velocityField->getWatU(i,j,k);
                        if(c < u_size - 1) up(c)  += w_vel;
                        if(c > 0)          low(c) -= w_vel;
                    }
                    c++;
                }

                // Boundary Conditions (k=0 and k=Nz-1)
                const ADICoef cLo = CoefTangZ(g, 1, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefTangZ(g, 1 + u_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo * gridSol->GetU(i,j,0);
                    RHS(u_size-1) += cHi.up * gridSol->GetU(i,j,Nz-1);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->getWatU(i,j,1), cLo.bw, cLo.fw).low * gridSol->GetU(i,j,0) + cLo.lo * gridSol->GetU(i,j,0);
                    RHS(u_size-1) += -UpwindCoefficients(velocityField->getWatU(i,j,Nz-2), cHi.bw, cHi.fw).up * gridSol->GetU(i,j,Nz-1) + cHi.up * gridSol->GetU(i,j,Nz-1);
                }
                else
                {
                    RHS(0)        +=  cLo.rho * velocityField->getWatU(i,j,1)    * gridSol->GetU(i,j,0)    + cLo.lo * gridSol->GetU(i,j,0);
                    RHS(u_size-1) += -cHi.rho * velocityField->getWatU(i,j,Nz-2) * gridSol->GetU(i,j,Nz-1) + cHi.up * gridSol->GetU(i,j,Nz-1);
                }
                RHS(0)        += WallRing(gridSol, MAC::U, MAC::Z, i,j,1, -1, cLo.lo, gridSol->GetU(i,j,0));
                RHS(u_size-1) += WallRing(gridSol, MAC::U, MAC::Z, i,j,Nz-2, +1, cHi.up, gridSol->GetU(i,j,Nz-1));

                // Solid Masking
                for(int k = 0; k < u_size; k++) {
                    int grid_k = k + 1;
                    if(gridSol->FaceGate(MAC::U, i, j, grid_k) == GateState::WALL) {
                        RHS(k) = 0.0;
                        mid(k) = 1.0;
                        if(k > 0) low(k) = 0.0;
                        if(k < u_size-1) up(k) = 0.0;
                        if(k > 0) up(k-1) = 0.0;
                        if(k < u_size-1) low(k+1) = 0.0;
                    }
                }

                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                for(int k = 0; k < u_size; k++) {
                    gridSol->SetU(i, j, k + 1, SOL(k));
                }
            }
        }
    }
}

void ADI3D::SolveADI_X_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int v_size = Nx - 2;

    #pragma omp parallel
    {
        // Allocation: Once per thread
        Eigen::VectorXd low(v_size), mid(v_size), up(v_size);
        Eigen::VectorXd RHS(v_size), SOL(v_size);

        #pragma omp for
        for(int i = 2; i < Ny-1; i++) { // Note: V is staggered in Y, check boundaries
            for(int k = 1; k < Nz-1; k++) {

                // RESET VECTORS for every (i, k) pencil
                for(int m = 0; m < v_size; m++) {
                    const ADICoef cm = CoefTangX(g, m + 1, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int j = 1; j < Nx-1; j++) {
                    const ADICoef cI = CoefTangX(g, j, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefNormY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefTangZ(g, k, nu_dt, dt_dir, sig, rho);

                    // Explicit terms (Y and Z directions)
                    double Y_DIFF = (gridAnt->GetV(i+1,j,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i-1,j,k));
                    const double Y_DIFF_NU = NU ? (cA.up*((gridAnt->GetV(i+1,j,k)) - (gridAnt->GetV(i,j,k))) - cA.lo*((gridAnt->GetV(i,j,k)) - (gridAnt->GetV(i-1,j,k)))) : 0.0;
                    double Z_DIFF = (gridAnt->GetV(i,j,k+1) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j,k-1));
                    const double Z_DIFF_NU = NU ? (cB.up*((gridAnt->GetV(i,j,k+1)) - (gridAnt->GetV(i,j,k))) - cB.lo*((gridAnt->GetV(i,j,k)) - (gridAnt->GetV(i,j,k-1)))) : 0.0;

                    double V_CONV = (velocityField->GetV(i,j,k)) * ((gridAnt->GetV(i+1,j,k) - gridAnt->GetV(i-1,j,k)));
                    double W_CONV = (velocityField->getWatV(i,j,k)) * ((gridAnt->GetV(i,j,k+1) - gridAnt->GetV(i,j,k-1)));

                    const double DIFF_EXP = NU ? (Y_DIFF_NU + Z_DIFF_NU) : sig*(Y_DIFF + Z_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::Y, i,j,k), UPWIND, gridAnt->GetV(i-1,j,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i+1,j,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::Z, i,j,k), UPWIND, gridAnt->GetV(i,j,k-1), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j,k+1)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->GetV(i,j,k), gridAnt->GetV(i-1,j,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i+1,j,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->getWatV(i,j,k), gridAnt->GetV(i,j,k-1), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j,k+1), cB.bw, cB.fw))
                        : NU ? (cA.rho*V_CONV + cB.rho*W_CONV) : rho*(V_CONV + W_CONV);
                    RHS(c) = gridAnt->GetV(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::V, MAC::Z, i, j, k, cB.lo, cB.up, gridAnt->GetV(i,j,k-1), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j,k+1));
                    mid(c) += WallDiagonal(gridSol, MAC::V, MAC::X, i, j, k, cI.lo, cI.up);
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::Y, i,j,k);   // grad p^n

                    // Implicit Convection (X direction)
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::X, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < v_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetV(i,Nx-1,k);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetV(i,0,k);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->getUatV(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < v_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double u_vel = cI.rho * velocityField->getUatV(i,j,k);
                        if(c < v_size - 1) up(c)  += u_vel;
                        if(c > 0)          low(c) -= u_vel;
                    }
                    c++;
                }

                // Boundary Conditions
                const ADICoef cLo = CoefTangX(g, 1, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefTangX(g, 1 + v_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo * gridSol->GetV(i,0,k);
                    RHS(v_size-1) += cHi.up * gridSol->GetV(i,Nx-1,k);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->getUatV(i,1,k), cLo.bw, cLo.fw).low * gridSol->GetV(i,0,k) + cLo.lo * gridSol->GetV(i,0,k);
                    RHS(v_size-1) += -UpwindCoefficients(velocityField->getUatV(i,Nx-2,k), cHi.bw, cHi.fw).up * gridSol->GetV(i,Nx-1,k) + cHi.up * gridSol->GetV(i,Nx-1,k);
                }
                else
                {
                    RHS(0)        +=  cLo.rho * gridSol->GetV(i,0,k)    * velocityField->getUatV(i,1,k)    + cLo.lo * gridSol->GetV(i,0,k);
                    RHS(v_size-1) += -cHi.rho * gridSol->GetV(i,Nx-1,k) * velocityField->getUatV(i,Nx-2,k) + cHi.up * gridSol->GetV(i,Nx-1,k);
                }
                RHS(0)        += WallRing(gridSol, MAC::V, MAC::X, i,1,k, -1, cLo.lo, gridSol->GetV(i,0,k));
                RHS(v_size-1) += WallRing(gridSol, MAC::V, MAC::X, i,Nx-2,k, +1, cHi.up, gridSol->GetV(i,Nx-1,k));

                // Solid Masking
                for(int j = 0; j < v_size; j++) {
                    int grid_j = j + 1;
                    if(gridSol->FaceGate(MAC::V, i, grid_j, k) == GateState::WALL) {
                        RHS(j) = 0.0; mid(j) = 1.0;
                        if(j > 0) { low(j) = 0.0; up(j-1) = 0.0; }
                        if(j < v_size-1) { up(j) = 0.0; low(j+1) = 0.0; }
                    }
                }

                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                for(int j = 0; j < v_size; j++) {
                    gridSol->SetV(i, j + 1, k, SOL(j));
                }
            }
        }
    }
}
void ADI3D::SolveADI_Y_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int v_size = Ny + 1 - 4;

    #pragma omp parallel
    {
        Eigen::VectorXd low(v_size), mid(v_size), up(v_size);
        Eigen::VectorXd RHS(v_size), SOL(v_size);

        #pragma omp for
        for(int j = 1; j < Nx-1; j++) {
            for(int k = 1; k < Nz-1; k++) {

                // RESET VECTORS for every (j, k) pencil
                for(int m = 0; m < v_size; m++) {
                    const ADICoef cm = CoefNormY(g, m + 2, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int i = 2; i < Ny+1-2; i++) {
                    const ADICoef cI = CoefNormY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefTangX(g, j, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefTangZ(g, k, nu_dt, dt_dir, sig, rho);

                    // Explicit X and Z terms
                    double X_DIFF = (gridAnt->GetV(i,j+1,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j-1,k));
                    const double X_DIFF_NU = NU ? (cA.up*((gridAnt->GetV(i,j+1,k)) - (gridAnt->GetV(i,j,k))) - cA.lo*((gridAnt->GetV(i,j,k)) - (gridAnt->GetV(i,j-1,k)))) : 0.0;
                    double Z_DIFF = (gridAnt->GetV(i,j,k+1) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j,k-1));
                    const double Z_DIFF_NU = NU ? (cB.up*((gridAnt->GetV(i,j,k+1)) - (gridAnt->GetV(i,j,k))) - cB.lo*((gridAnt->GetV(i,j,k)) - (gridAnt->GetV(i,j,k-1)))) : 0.0;

                    double U_CONV = velocityField->getUatV(i,j,k) * (gridAnt->GetV(i,j+1,k) - gridAnt->GetV(i,j-1,k));
                    double W_CONV = velocityField->getWatV(i,j,k) * (gridAnt->GetV(i,j,k+1) - gridAnt->GetV(i,j,k-1));

                    const double DIFF_EXP = NU ? (X_DIFF_NU + Z_DIFF_NU) : sig*(X_DIFF + Z_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::X, i,j,k), UPWIND, gridAnt->GetV(i,j-1,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j+1,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::Z, i,j,k), UPWIND, gridAnt->GetV(i,j,k-1), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j,k+1)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->getUatV(i,j,k), gridAnt->GetV(i,j-1,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j+1,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->getWatV(i,j,k), gridAnt->GetV(i,j,k-1), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j,k+1), cB.bw, cB.fw))
                        : NU ? (cA.rho*U_CONV + cB.rho*W_CONV) : rho*(U_CONV + W_CONV);
                    RHS(c) = gridAnt->GetV(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::V, MAC::X, i, j, k, cA.lo, cA.up, gridAnt->GetV(i,j-1,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j+1,k));
                    RHS(c) += WallExplicit(gridSol, MAC::V, MAC::Z, i, j, k, cB.lo, cB.up, gridAnt->GetV(i,j,k-1), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j,k+1));
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::Y, i,j,k);   // grad p^n

                    // Implicit Y Convection (for V velocity)
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::Y, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < v_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetV(Ny-1,j,k);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetV(1,j,k);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->GetV(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < v_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double v_vel = cI.rho * velocityField->GetV(i,j,k);
                        if(c < v_size - 1) up(c)  += v_vel;
                        if(c > 0)          low(c) -= v_vel;
                    }
                    c++;
                }

                // Boundary Conditions
                const ADICoef cLo = CoefNormY(g, 2, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefNormY(g, 2 + v_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo * gridSol->GetV(1,j,k);
                    RHS(v_size-1) += cHi.up * gridSol->GetV(Ny-1,j,k);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->GetV(2,j,k), cLo.bw, cLo.fw).low * gridSol->GetV(1,j,k) + cLo.lo * gridSol->GetV(1,j,k);
                    RHS(v_size-1) += -UpwindCoefficients(velocityField->GetV(Ny-2,j,k), cHi.bw, cHi.fw).up * gridSol->GetV(Ny-1,j,k) + cHi.up * gridSol->GetV(Ny-1,j,k);
                }
                else
                {
                    RHS(0)        +=  cLo.rho * gridSol->GetV(1,j,k)      * velocityField->GetV(2,j,k)      + cLo.lo * gridSol->GetV(1,j,k);
                    RHS(v_size-1) += -cHi.rho * gridSol->GetV(Ny-1,j,k)   * velocityField->GetV(Ny-2,j,k)   + cHi.up * gridSol->GetV(Ny-1,j,k);
                }

                // Solid Masking
                for(int i = 0; i < v_size; i++) {
                    int grid_i = i + 2;
                    if(gridSol->FaceGate(MAC::V, grid_i, j, k) == GateState::WALL) {
                        RHS(i) = 0.0; mid(i) = 1.0;
                        if(i > 0) { low(i) = 0.0; up(i-1) = 0.0; }
                        if(i < v_size-1) { up(i) = 0.0; low(i+1) = 0.0; }
                    }
                }

                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                for(int i = 0; i < v_size; i++) {
                    gridSol->SetV(i + 2, j, k, SOL(i));
                }
            }
        }
    }
}
void ADI3D::SolveADI_Z_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*3))* (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int v_size = Nz - 2;

    #pragma omp parallel
    {
        Eigen::VectorXd low(v_size), mid(v_size), up(v_size);
        Eigen::VectorXd RHS(v_size), SOL(v_size);

        #pragma omp for
        for(int i = 2; i < Ny-1; i++) {
            for(int j = 1; j < Nx-1; j++) {

                // RESET VECTORS for every (i, j) pencil
                for(int m = 0; m < v_size; m++) {
                    const ADICoef cm = CoefTangZ(g, m + 1, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int k = 1; k < Nz-1; k++) {
                    const ADICoef cI = CoefTangZ(g, k, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefNormY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefTangX(g, j, nu_dt, dt_dir, sig, rho);

                    // Explicit X and Y terms
                    double Y_DIFF = (gridAnt->GetV(i+1,j,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i-1,j,k));
                    const double Y_DIFF_NU = NU ? (cA.up*((gridAnt->GetV(i+1,j,k)) - (gridAnt->GetV(i,j,k))) - cA.lo*((gridAnt->GetV(i,j,k)) - (gridAnt->GetV(i-1,j,k)))) : 0.0;
                    double X_DIFF = (gridAnt->GetV(i,j+1,k) - 2*gridAnt->GetV(i,j,k) + gridAnt->GetV(i,j-1,k));
                    const double X_DIFF_NU = NU ? (cB.up*((gridAnt->GetV(i,j+1,k)) - (gridAnt->GetV(i,j,k))) - cB.lo*((gridAnt->GetV(i,j,k)) - (gridAnt->GetV(i,j-1,k)))) : 0.0;

                    double U_CONV = velocityField->getUatV(i,j,k) * (gridAnt->GetV(i,j+1,k) - gridAnt->GetV(i,j-1,k));
                    double V_CONV = velocityField->GetV(i,j,k) * (gridAnt->GetV(i+1,j,k) - gridAnt->GetV(i-1,j,k));

                    const double DIFF_EXP = NU ? (Y_DIFF_NU + X_DIFF_NU) : sig*(Y_DIFF + X_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::X, i,j,k), UPWIND, gridAnt->GetV(i,j-1,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j+1,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::Y, i,j,k), UPWIND, gridAnt->GetV(i-1,j,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i+1,j,k)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->GetV(i,j,k), gridAnt->GetV(i-1,j,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i+1,j,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->getUatV(i,j,k), gridAnt->GetV(i,j-1,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j+1,k), cB.bw, cB.fw))
                        : NU ? (cA.rho*V_CONV + cB.rho*U_CONV) : rho*(V_CONV + U_CONV);
                    RHS(c) = gridAnt->GetV(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::V, MAC::X, i, j, k, cB.lo, cB.up, gridAnt->GetV(i,j-1,k), gridAnt->GetV(i,j,k), gridAnt->GetV(i,j+1,k));
                    mid(c) += WallDiagonal(gridSol, MAC::V, MAC::Z, i, j, k, cI.lo, cI.up);
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::Y, i,j,k);   // grad p^n

                    // Implicit Convection (Z direction)
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::V, MAC::Z, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < v_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetV(i,j,Nz-1);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetV(i,j,0);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->getWatV(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < v_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double w_vel = cI.rho * velocityField->getWatV(i,j,k);
                        if(c < v_size - 1) up(c)  += w_vel;
                        if(c > 0)          low(c) -= w_vel;
                    }
                    c++;
                }

                // Boundary Conditions
                const ADICoef cLo = CoefTangZ(g, 1, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefTangZ(g, 1 + v_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo * gridSol->GetV(i,j,0);
                    RHS(v_size-1) += cHi.up * gridSol->GetV(i,j,Nz-1);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->getWatV(i,j,1), cLo.bw, cLo.fw).low * gridSol->GetV(i,j,0) + cLo.lo * gridSol->GetV(i,j,0);
                    RHS(v_size-1) += -UpwindCoefficients(velocityField->getWatV(i,j,Nz-2), cHi.bw, cHi.fw).up * gridSol->GetV(i,j,Nz-1) + cHi.up * gridSol->GetV(i,j,Nz-1);
                }
                else
                {
                    RHS(0)        +=  cLo.rho * gridSol->GetV(i,j,0)     * velocityField->getWatV(i,j,1)    + cLo.lo * gridSol->GetV(i,j,0);
                    RHS(v_size-1) += -cHi.rho * gridSol->GetV(i,j,Nz-1)  * velocityField->getWatV(i,j,Nz-2) + cHi.up * gridSol->GetV(i,j,Nz-1);
                }
                RHS(0)        += WallRing(gridSol, MAC::V, MAC::Z, i,j,1, -1, cLo.lo, gridSol->GetV(i,j,0));
                RHS(v_size-1) += WallRing(gridSol, MAC::V, MAC::Z, i,j,Nz-2, +1, cHi.up, gridSol->GetV(i,j,Nz-1));

                // Solid Masking
                for(int k = 0; k < v_size; k++) {
                    int grid_k = k + 1;
                    if(gridSol->FaceGate(MAC::V, i, j, grid_k) == GateState::WALL) {
                        RHS(k) = 0.0; mid(k) = 1.0;
                        if(k > 0) { low(k) = 0.0; up(k-1) = 0.0; }
                        if(k < v_size-1) { up(k) = 0.0; low(k+1) = 0.0; }
                    }
                }

                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                for(int k = 0; k < v_size; k++) {
                    gridSol->SetV(i, j, k + 1, SOL(k));
                }
            }
        }
    }
}

void ADI3D::SolveADI_X_W_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridAnt->Nx;
    int Ny = gridAnt->Ny;
    int Nz = gridAnt->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int w_size = Nx - 2;

    #pragma omp parallel
    {
        Eigen::VectorXd low(w_size), mid(w_size), up(w_size);
        Eigen::VectorXd RHS(w_size), SOL(w_size);

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {
            for(int k = 2; k < Nz-1; k++) {

                // RESET tridiagonal system for each pencil
                for(int m = 0; m < w_size; m++) {
                    const ADICoef cm = CoefTangX(g, m + 1, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int j = 1; j < Nx-1; j++) {
                    const ADICoef cI = CoefTangX(g, j, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefTangY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefNormZ(g, k, nu_dt, dt_dir, sig, rho);

                    // Explicit Y and Z terms
                    double Y_DIFF = (gridAnt->GetW(i+1,j,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i-1,j,k));
                    const double Y_DIFF_NU = NU ? (cA.up*((gridAnt->GetW(i+1,j,k)) - (gridAnt->GetW(i,j,k))) - cA.lo*((gridAnt->GetW(i,j,k)) - (gridAnt->GetW(i-1,j,k)))) : 0.0;
                    double Z_DIFF = (gridAnt->GetW(i,j,k+1) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j,k-1));
                    const double Z_DIFF_NU = NU ? (cB.up*((gridAnt->GetW(i,j,k+1)) - (gridAnt->GetW(i,j,k))) - cB.lo*((gridAnt->GetW(i,j,k)) - (gridAnt->GetW(i,j,k-1)))) : 0.0;
                    double V_CONV = (velocityField->getVatW(i,j,k)) * ((gridAnt->GetW(i+1,j,k) - gridAnt->GetW(i-1,j,k)));
                    double W_CONV = (velocityField->GetW(i,j,k)) * ((gridAnt->GetW(i,j,k+1) - gridAnt->GetW(i,j,k-1)));

                    const double DIFF_EXP = NU ? (Y_DIFF_NU + Z_DIFF_NU) : sig*(Y_DIFF + Z_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::Y, i,j,k), UPWIND, gridAnt->GetW(i-1,j,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i+1,j,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::Z, i,j,k), UPWIND, gridAnt->GetW(i,j,k-1), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j,k+1)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->getVatW(i,j,k), gridAnt->GetW(i-1,j,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i+1,j,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->GetW(i,j,k), gridAnt->GetW(i,j,k-1), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j,k+1), cB.bw, cB.fw))
                        : NU ? (cA.rho*V_CONV + cB.rho*W_CONV) : rho*(V_CONV + W_CONV);
                    RHS(c) = gridAnt->GetW(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::W, MAC::Y, i, j, k, cA.lo, cA.up, gridAnt->GetW(i-1,j,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i+1,j,k));
                    mid(c) += WallDiagonal(gridSol, MAC::W, MAC::X, i, j, k, cI.lo, cI.up);
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::Z, i,j,k);   // grad p^n

                    // Implicit X Convection
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::X, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < w_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetW(i,Nx-1,k);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetW(i,0,k);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->getUatW(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < w_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double u_vel = cI.rho * velocityField->getUatW(i,j,k);
                        if(c < w_size - 1) up(c)  += u_vel;
                        if(c > 0)          low(c) -= u_vel;
                    }
                    c++;
                }

                // Boundary Conditions
                const ADICoef cLo = CoefTangX(g, 1, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefTangX(g, 1 + w_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo * gridSol->GetW(i,0,k);
                    RHS(w_size-1) += cHi.up * gridSol->GetW(i,Nx-1,k);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->getUatW(i,1,k), cLo.bw, cLo.fw).low * gridSol->GetW(i,0,k) + cLo.lo * gridSol->GetW(i,0,k);
                    RHS(w_size-1) += -UpwindCoefficients(velocityField->getUatW(i,Nx-2,k), cHi.bw, cHi.fw).up * gridSol->GetW(i,Nx-1,k) + cHi.up * gridSol->GetW(i,Nx-1,k);
                }
                else
                {
                    RHS(0)        +=  cLo.rho * velocityField->getUatW(i,1,k)    * gridSol->GetW(i,0,k)    + cLo.lo * gridSol->GetW(i,0,k);
                    RHS(w_size-1) += -cHi.rho * velocityField->getUatW(i,Nx-2,k) * gridSol->GetW(i,Nx-1,k) + cHi.up * gridSol->GetW(i,Nx-1,k);
                }
                RHS(0)        += WallRing(gridSol, MAC::W, MAC::X, i,1,k, -1, cLo.lo, gridSol->GetW(i,0,k));
                RHS(w_size-1) += WallRing(gridSol, MAC::W, MAC::X, i,Nx-2,k, +1, cHi.up, gridSol->GetW(i,Nx-1,k));

                // Solid Masking
                for(int j = 0; j < w_size; j++) {
                    int grid_j = j + 1;
                    if(gridSol->FaceGate(MAC::W, i, grid_j, k) == GateState::WALL) {
                        RHS(j) = 0.0; mid(j) = 1.0;
                        if(j > 0) { low(j) = 0.0; up(j-1) = 0.0; }
                        if(j < w_size-1) { up(j) = 0.0; low(j+1) = 0.0; }
                    }
                }

                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                for(int j = 0; j < w_size; j++) {
                    gridSol->SetW(i, j + 1, k, SOL(j));
                }
            }
        }
    }
}
void ADI3D::SolveADI_Y_W_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int w_size = Ny - 2;

    #pragma omp parallel
    {
        Eigen::VectorXd low(w_size), mid(w_size), up(w_size);
        Eigen::VectorXd RHS(w_size), SOL(w_size);

        #pragma omp for
        for(int j = 1; j < Nx-1; j++) {
            for(int k = 2; k < Nz-1; k++) {

                for(int m = 0; m < w_size; m++) {
                    const ADICoef cm = CoefTangY(g, m + 1, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int i = 1; i < Ny-1; i++) {
                    const ADICoef cI = CoefTangY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefTangX(g, j, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefNormZ(g, k, nu_dt, dt_dir, sig, rho);

                    // Explicit X and Z terms
                    double X_DIFF = (gridAnt->GetW(i,j+1,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j-1,k));
                    const double X_DIFF_NU = NU ? (cA.up*((gridAnt->GetW(i,j+1,k)) - (gridAnt->GetW(i,j,k))) - cA.lo*((gridAnt->GetW(i,j,k)) - (gridAnt->GetW(i,j-1,k)))) : 0.0;
                    double Z_DIFF = (gridAnt->GetW(i,j,k+1) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j,k-1));
                    const double Z_DIFF_NU = NU ? (cB.up*((gridAnt->GetW(i,j,k+1)) - (gridAnt->GetW(i,j,k))) - cB.lo*((gridAnt->GetW(i,j,k)) - (gridAnt->GetW(i,j,k-1)))) : 0.0;
                    double U_CONV = velocityField->getUatW(i,j,k) * (gridAnt->GetW(i,j+1,k) - gridAnt->GetW(i,j-1,k));
                    double W_CONV = velocityField->GetW(i,j,k) * (gridAnt->GetW(i,j,k+1) - gridAnt->GetW(i,j,k-1));

                    const double DIFF_EXP = NU ? (X_DIFF_NU + Z_DIFF_NU) : sig*(X_DIFF + Z_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::X, i,j,k), UPWIND, gridAnt->GetW(i,j-1,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j+1,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::Z, i,j,k), UPWIND, gridAnt->GetW(i,j,k-1), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j,k+1)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->getUatW(i,j,k), gridAnt->GetW(i,j-1,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j+1,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->GetW(i,j,k), gridAnt->GetW(i,j,k-1), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j,k+1), cB.bw, cB.fw))
                        : NU ? (cA.rho*U_CONV + cB.rho*W_CONV) : rho*(U_CONV + W_CONV);
                    RHS(c) = gridAnt->GetW(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::W, MAC::X, i, j, k, cA.lo, cA.up, gridAnt->GetW(i,j-1,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j+1,k));
                    mid(c) += WallDiagonal(gridSol, MAC::W, MAC::Y, i, j, k, cI.lo, cI.up);
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::Z, i,j,k);   // grad p^n

                    // Implicit Y Convection
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::Y, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < w_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetW(Ny-1,j,k);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetW(0,j,k);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->getVatW(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < w_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double v_vel = cI.rho * velocityField->getVatW(i,j,k);
                        if(c < w_size - 1) up(c)  += v_vel;
                        if(c > 0)          low(c) -= v_vel;
                    }
                    c++;
                }

                // Boundary Conditions
                const ADICoef cLo = CoefTangY(g, 1, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefTangY(g, 1 + w_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo * gridSol->GetW(0,j,k);
                    RHS(w_size-1) += cHi.up * gridSol->GetW(Ny-1,j,k);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->getVatW(1,j,k), cLo.bw, cLo.fw).low * gridSol->GetW(0,j,k) + cLo.lo * gridSol->GetW(0,j,k);
                    RHS(w_size-1) += -UpwindCoefficients(velocityField->getVatW(Ny-2,j,k), cHi.bw, cHi.fw).up * gridSol->GetW(Ny-1,j,k) + cHi.up * gridSol->GetW(Ny-1,j,k);
                }
                else
                {
                    RHS(0)        +=  cLo.rho * velocityField->getVatW(1,j,k)    * gridSol->GetW(0,j,k)    + cLo.lo * gridSol->GetW(0,j,k);
                    RHS(w_size-1) += -cHi.rho * velocityField->getVatW(Ny-2,j,k) * gridSol->GetW(Ny-1,j,k) + cHi.up * gridSol->GetW(Ny-1,j,k);
                }
                RHS(0)        += WallRing(gridSol, MAC::W, MAC::Y, 1,j,k, -1, cLo.lo, gridSol->GetW(0,j,k));
                RHS(w_size-1) += WallRing(gridSol, MAC::W, MAC::Y, Ny-2,j,k, +1, cHi.up, gridSol->GetW(Ny-1,j,k));

                // Solid Masking
                for(int i = 0; i < w_size; i++) {
                    int grid_i = i + 1;
                    if(gridSol->FaceGate(MAC::W, grid_i, j, k) == GateState::WALL) {
                        RHS(i) = 0.0; mid(i) = 1.0;
                        if(i > 0) { low(i) = 0.0; up(i-1) = 0.0; }
                        if(i < w_size-1) { up(i) = 0.0; low(i+1) = 0.0; }
                    }
                }

                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                for(int i = 0; i < w_size; i++) {
                    gridSol->SetW(i + 1, j, k, SOL(i));
                }
            }
        }
    }
}
void ADI3D::SolveADI_Z_W_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    int Nz = gridSol->Nz;
    double dh = ADI3D::dh;
    double dt = SIMULATION.dt;
    double sig = (dt/(dh*dh*3)) * (SIMULATION.EPS);
    double rho = (dt/(6*dh));
    // Rectilinear: each direction carries dt/3. See the coefficient helpers at the top of
    // this file; on a uniform grid every one of them returns sig/rho unchanged.
    const MAC *g = gridAnt;
    const bool NU = g->nonUniform;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;
    const double dt_dir = dt / 3.0;
    const double nu_dt  = dt_dir * SIMULATION.EPS;

    int w_size = Nz + 1 - 4;

    #pragma omp parallel
    {
        Eigen::VectorXd low(w_size), mid(w_size), up(w_size);
        Eigen::VectorXd RHS(w_size), SOL(w_size);

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {
            for(int j = 1; j < Nx-1; j++) {

                for(int m = 0; m < w_size; m++) {
                    const ADICoef cm = CoefNormZ(g, m + 2, nu_dt, dt_dir, sig, rho);
                    // 1 + (lo + up); parenthesised so the uniform case is exactly 1 + 2*sig.
                    mid(m) = 1.0 + (cm.lo + cm.up);
                    low(m) = -cm.lo;
                    up(m)  = -cm.up;
                }

                int c = 0;
                for(int k = 2; k < Nz+1-2; k++) {
                    const ADICoef cI = CoefNormZ(g, k, nu_dt, dt_dir, sig, rho);
                    const ADICoef cA = CoefTangY(g, i, nu_dt, dt_dir, sig, rho);
                    const ADICoef cB = CoefTangX(g, j, nu_dt, dt_dir, sig, rho);

                    // Explicit X and Y terms
                    double Y_DIFF = (gridAnt->GetW(i+1,j,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i-1,j,k));
                    const double Y_DIFF_NU = NU ? (cA.up*((gridAnt->GetW(i+1,j,k)) - (gridAnt->GetW(i,j,k))) - cA.lo*((gridAnt->GetW(i,j,k)) - (gridAnt->GetW(i-1,j,k)))) : 0.0;
                    double X_DIFF = (gridAnt->GetW(i,j+1,k) - 2*gridAnt->GetW(i,j,k) + gridAnt->GetW(i,j-1,k));
                    const double X_DIFF_NU = NU ? (cB.up*((gridAnt->GetW(i,j+1,k)) - (gridAnt->GetW(i,j,k))) - cB.lo*((gridAnt->GetW(i,j,k)) - (gridAnt->GetW(i,j-1,k)))) : 0.0;
                    double U_CONV = (velocityField->getUatW(i,j,k)) * ((gridAnt->GetW(i,j+1,k) - gridAnt->GetW(i,j-1,k)));
                    double V_CONV = (velocityField->getVatW(i,j,k)) * ((gridAnt->GetW(i+1,j,k) - gridAnt->GetW(i-1,j,k)));

                    const double DIFF_EXP = NU ? (Y_DIFF_NU + X_DIFF_NU) : sig*(Y_DIFF + X_DIFF);
                    const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::X, i,j,k), UPWIND, gridAnt->GetW(i,j-1,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j+1,k)) +
                        FluxDifference(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::Y, i,j,k), UPWIND, gridAnt->GetW(i-1,j,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i+1,j,k)))
                        : UPWIND
                        ? (UpwindDifference(velocityField->getVatW(i,j,k), gridAnt->GetW(i-1,j,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i+1,j,k), cA.bw, cA.fw) +
                           UpwindDifference(velocityField->getUatW(i,j,k), gridAnt->GetW(i,j-1,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j+1,k), cB.bw, cB.fw))
                        : NU ? (cA.rho*V_CONV + cB.rho*U_CONV) : rho*(V_CONV + U_CONV);
                    RHS(c) = gridAnt->GetW(i,j,k) + DIFF_EXP - CONV_EXP;
                    RHS(c) += WallExplicit(gridSol, MAC::W, MAC::Y, i, j, k, cA.lo, cA.up, gridAnt->GetW(i-1,j,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i+1,j,k));
                    RHS(c) += WallExplicit(gridSol, MAC::W, MAC::X, i, j, k, cB.lo, cB.up, gridAnt->GetW(i,j-1,k), gridAnt->GetW(i,j,k), gridAnt->GetW(i,j+1,k));
                    if (INCREMENTAL) RHS(c) -= (dt/3.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::Z, i,j,k);   // grad p^n

                    // Implicit Z Convection (for W velocity)
                    if (FLUX)
                    {
                        const UpwindRow fr = FluxCoefficients(dt/3.0, FluxGeomAt(velocityField, MAC::W, MAC::Z, i,j,k), UPWIND);
                        mid(c) += fr.mid;
                        if(c < w_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetW(i,j,Nz-1);
                        if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetW(i,j,1);
                    }
                    else if (UPWIND)
                    {
                        const UpwindRow uw = UpwindCoefficients(velocityField->GetW(i,j,k), cI.bw, cI.fw);
                        mid(c) += uw.mid;
                        if(c < w_size - 1) up(c)  += uw.up;
                        if(c > 0)          low(c) += uw.low;
                    }
                    else
                    {
                        double w_vel = cI.rho * velocityField->GetW(i,j,k);
                        if(c < w_size - 1) up(c)  += w_vel;
                        if(c > 0)          low(c) -= w_vel;
                    }
                    c++;
                }

                // Boundary Conditions
                const ADICoef cLo = CoefNormZ(g, 2, nu_dt, dt_dir, sig, rho);
                const ADICoef cHi = CoefNormZ(g, 2 + w_size - 1, nu_dt, dt_dir, sig, rho);
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += cLo.lo * gridSol->GetW(i,j,1);
                    RHS(w_size-1) += cHi.up * gridSol->GetW(i,j,Nz-1);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->GetW(i,j,2), cLo.bw, cLo.fw).low * gridSol->GetW(i,j,1) + cLo.lo * gridSol->GetW(i,j,1);
                    RHS(w_size-1) += -UpwindCoefficients(velocityField->GetW(i,j,Nz-2), cHi.bw, cHi.fw).up * gridSol->GetW(i,j,Nz-1) + cHi.up * gridSol->GetW(i,j,Nz-1);
                }
                else
                {
                    RHS(0)        +=  cLo.rho * gridSol->GetW(i,j,1)      * velocityField->GetW(i,j,2)      + cLo.lo * gridSol->GetW(i,j,1);
                    RHS(w_size-1) += -cHi.rho * gridSol->GetW(i,j,Nz-1)   * velocityField->GetW(i,j,Nz-2)   + cHi.up * gridSol->GetW(i,j,Nz-1);
                }

                // Solid Masking
                for(int k = 0; k < w_size; k++) {
                    int grid_k = k + 2;
                    if(gridSol->FaceGate(MAC::W, i, j, grid_k) == GateState::WALL) {
                        RHS(k) = 0.0; mid(k) = 1.0;
                        if(k > 0) { low(k) = 0.0; up(k-1) = 0.0; }
                        if(k < w_size-1) { up(k) = 0.0; low(k+1) = 0.0; }
                    }
                }

                TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

                for(int k = 0; k < w_size; k++) {
                    gridSol->SetW(i, j, k + 2, SOL(k));
                }
            }
        }
    }
}

void ADI3D::InitializeADI(MAC* grid,double dt,Vec3(*VelocityBorderFunction)(double, double, double,double),double(*PressureFunction)(double, double, double,double)){

    ADI3D::VelocityBorderFunction = VelocityBorderFunction;
    ADI3D::PressureFunction = PressureFunction;

    ADI3D::dh = grid->dh;

    grid->SetBoundaryFunctions(VelocityBorderFunction, PressureFunction);
    ADI3D::X_STEP_SOL.InitializeFromGrid(*grid, SIMULATION.time);
    ADI3D::Y_STEP_SOL.InitializeFromGrid(*grid, SIMULATION.time);
    ADI3D::Z_STEP_SOL.InitializeFromGrid(*grid, SIMULATION.time);

    RegisterSimulationStep(ADI3D::SolveADIStep, "momentum", 0);
}

void ADI3D::SolveADIStep(MAC* gridAnt,MAC* gridSol,ParticleSet *particles, double time){
    // The step's start time, before the sweeps advance `time`: the body force is evaluated here.
    const double stepStart = time;

    // call each solve ADI in parallel using OpenMP sections
    time += SIMULATION.dt/3.0;
    ADI3D::X_STEP_SOL.CopyFlowFields(*gridAnt);
    ADI3D::X_STEP_SOL.EnforceVelocityBC(time);

            SolveADI_X_U_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);

            SolveADI_X_V_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);

            SolveADI_X_W_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);

    time += SIMULATION.dt/3.0;
    ADI3D::Y_STEP_SOL.CopyFlowFields(ADI3D::X_STEP_SOL);
    ADI3D::Y_STEP_SOL.EnforceVelocityBC(time);

            SolveADI_Y_U_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);

            SolveADI_Y_V_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);

            SolveADI_Y_W_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);

    time += SIMULATION.dt/3.0;
    ADI3D::Z_STEP_SOL.CopyFlowFields(ADI3D::Y_STEP_SOL);
    ADI3D::Z_STEP_SOL.EnforceVelocityBC(time);

            SolveADI_Z_U_Step_OPENMP(&Y_STEP_SOL,&Z_STEP_SOL,gridAnt,time);

            SolveADI_Z_V_Step_OPENMP(&Y_STEP_SOL,&Z_STEP_SOL,gridAnt,time);

            SolveADI_Z_W_Step_OPENMP(&Y_STEP_SOL,&Z_STEP_SOL,gridAnt,time);

    // dont know if i need this!

    gridSol->CopyFlowFields(Z_STEP_SOL);   // as ADI2D: the sweep grids carry no scalars
    // The body force goes in AFTER the sweeps and BEFORE the pressure solve, so the projection
    // removes the part of it that is not divergence free.
    gridSol->AddForce(stepStart, SIMULATION.dt);
    gridSol->SetBoundaryFunctions(VelocityBorderFunction, PressureFunction);
    gridSol->EnforceVelocityBC(time);
}

// Frees the sweep grids. The next Initialize builds them again.
void ADI3D::Dispose()
{
    X_STEP_SOL = MAC();
    Y_STEP_SOL = MAC();
    Z_STEP_SOL = MAC();
}
