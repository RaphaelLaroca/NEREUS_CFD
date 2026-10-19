// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ADI2D.h"
#include "ADISettings.h"
#include "../PPESolver/PressureSolver2D.h"
#include "VorticityConfinement2D.h"

// X

// Y

MAC ADI2D::X_STEP_SOL = MAC();
MAC ADI2D::Y_STEP_SOL = MAC();

double ADI2D::dh = 0.0;

Vec2(*ADI2D::VelocityBorderFunction)(double, double,double) = nullptr;
double(*ADI2D::PressureFunction)(double, double,double) = nullptr;

// PARALLEL OPENMP TESTING

// Rectilinear metric factors for the ADI sweeps.
static inline double SigOf(const MAC *g, double dt, double dh, double nu,
                           double hSide, double hControl)
{
    if (!g->nonUniform)
        return (dt / (dh * dh * 2)) * nu;
    return (dt * 0.5) * nu / (hSide * hControl);
}

// Convection carries dt/2 * 1/(h- + h+); uniform collapses to dt/(4*dh).
static inline double RhoOf(const MAC *g, double dt, double dh, double hMinus, double hPlus)
{
    if (!g->nonUniform)
        return (dt / (4 * dh));
    return dt / (2.0 * (hMinus + hPlus));
}

// Upwind one-sided coefficients from the central rho: dt/2 / h- (backward) and dt/2 / h+ (forward).
static inline double UpwindBack(double rho, double hMinus, double hPlus) { return rho * (hMinus + hPlus) / hMinus; }
static inline double UpwindFore(double rho, double hMinus, double hPlus) { return rho * (hMinus + hPlus) / hPlus; }

void ADI2D::SolveADI_X_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridAnt->Nx;
    int Ny = gridAnt->Ny;
    double dh = ADI2D::dh;
    double dt = SIMULATION.dt;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;

    int u_size = Nx + 1 - 4;

    #pragma omp parallel
    {
        // Allocate vectors once per thread
        Eigen::VectorXd low(u_size), mid(u_size), up(u_size);
        Eigen::VectorXd RHS(u_size), SOL(u_size);

        #pragma omp for
        for(int i = 1; i < Ny-1; i++) {

            int gj = 2;
            for(int j = 0; j < u_size; j++) {
                // u face at column gj: cells Width(X, gj-1) behind and Width(X, gj) ahead, own width FaceGap(X, gj-1)
                const double hCu = gridAnt->FaceGap(MAC::X, gj - 1);
                if (!gridAnt->nonUniform)
                {
                    mid(j) = 1.0 + 2.0 * ((dt/(dh*dh*2)) * (SIMULATION.EPS)); 
                    low(j) = -((dt/(dh*dh*2)) * (SIMULATION.EPS)); 
                    up(j)  = -((dt/(dh*dh*2)) * (SIMULATION.EPS)); 
                }
                else
                {
                    const double nuC = SIMULATION.EPS;
                    mid(j) = 1.0 + SigOf(gridAnt, dt, dh, nuC, gridAnt->Width(MAC::X, gj - 1), hCu)
                                 + SigOf(gridAnt, dt, dh, nuC, gridAnt->Width(MAC::X, gj), hCu);
                    low(j) = -SigOf(gridAnt, dt, dh,
                                    SIMULATION.EPS,
                                    gridAnt->Width(MAC::X, gj - 1), hCu);
                    up(j)  = -SigOf(gridAnt, dt, dh,
                                    SIMULATION.EPS,
                                    gridAnt->Width(MAC::X, gj), hCu);
                }
                gj++;
            }

            int c = 0;
            for(int j = 2; j < Nx+1-2; j++) {
                // along y a u face sits at a cell centre: DyFace outside, Width(Y, i) as the width
                const double hCuy = gridAnt->Width(MAC::Y, i);
                double sig_center =   SigOf(gridAnt, dt, dh, SIMULATION.EPS,   gridAnt->FaceGap(MAC::Y, i-1), hCuy);
                double sig_top =      SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::Y, i),   hCuy);
                double sig_bot =      SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::Y, i-1), hCuy);
                double sig_centerUp = gridAnt->nonUniform
                                    ? SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::Y, i), hCuy)
                                    : sig_center;
                double Y_DIFF = gridAnt->nonUniform
                    ? (sig_top*gridAnt->GetU(i+1,j) - (sig_center + sig_centerUp)*gridAnt->GetU(i,j) + sig_bot*gridAnt->GetU(i-1,j))
                    : (sig_top*gridAnt->GetU(i+1,j) - 2*sig_center*gridAnt->GetU(i,j) + sig_bot*gridAnt->GetU(i-1,j));
                double V_CONV = velocityField->getVatU(i,j) * (gridAnt->GetU(i+1,j) - gridAnt->GetU(i-1,j));

                // Convection carries dt/2 * 1/(h- + h+), and the two directions in this sweep do
                // NOT share a spacing once the grid is stretched: the explicit y term spans two.
                const double rhoY = RhoOf(gridAnt, dt, dh, gridAnt->FaceGap(MAC::Y, i-1), gridAnt->FaceGap(MAC::Y, i));
                const double rhoX = RhoOf(gridAnt, dt, dh, gridAnt->Width(MAC::X, j-1), gridAnt->Width(MAC::X, j));

                const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/2.0, FluxGeomAt(velocityField, MAC::U, MAC::Y, i,j), UPWIND, gridAnt->GetU(i-1,j), gridAnt->GetU(i,j), gridAnt->GetU(i+1,j)))
                        : UPWIND
                    ? UpwindDifference(velocityField->getVatU(i,j), gridAnt->GetU(i-1,j), gridAnt->GetU(i,j), gridAnt->GetU(i+1,j), UpwindBack(rhoY, gridAnt->FaceGap(MAC::Y, i-1), gridAnt->FaceGap(MAC::Y, i)), UpwindFore(rhoY, gridAnt->FaceGap(MAC::Y, i-1), gridAnt->FaceGap(MAC::Y, i)))
                    : rhoY*(V_CONV);
                RHS(c) = gridAnt->GetU(i,j) + (Y_DIFF) - CONV_EXP; //should the sig here be the main cell or each cell
                RHS(c) += WallExplicit(gridSol, MAC::U, MAC::Y, i, j, 0, sig_bot, sig_top, gridAnt->GetU(i-1,j), gridAnt->GetU(i,j), gridAnt->GetU(i+1,j));
                if (INCREMENTAL) RHS(c) -= (dt/2.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::X, i,j);   // grad p^n

                // Add Convective terms to the diagonals
                if (FLUX)
                {
                    const UpwindRow fr = FluxCoefficients(dt/2.0, FluxGeomAt(velocityField, MAC::U, MAC::X, i,j), UPWIND);
                    mid(c) += fr.mid;
                    if(c < u_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetU(i,Nx-1);
                    if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetU(i,1);
                }
                else if (UPWIND)
                {
                    const UpwindRow uw = UpwindCoefficients(velocityField->GetU(i,j), UpwindBack(rhoX, gridAnt->Width(MAC::X, j-1), gridAnt->Width(MAC::X, j)), UpwindFore(rhoX, gridAnt->Width(MAC::X, j-1), gridAnt->Width(MAC::X, j)));
                    mid(c) += uw.mid;
                    if(c < u_size - 1) up(c)  += uw.up;
                    if(c > 0)          low(c) += uw.low;
                }
                else
                {
                    double u_vel = rhoX * velocityField->GetU(i,j);
                    if(c < u_size - 1) up(c)  += u_vel;
                    if(c > 0)          low(c) -= u_vel;
                }
                c++;
            }

            // Boundary Conditions Boundary rows: the first and last unknowns couple to a KNOWN
            // face, so that coupling moves to the right-hand side.
            {
                const double rhoLo = RhoOf(gridAnt, dt, dh, gridAnt->Width(MAC::X, 1), gridAnt->Width(MAC::X, 2));
                const double rhoHi = RhoOf(gridAnt, dt, dh, gridAnt->Width(MAC::X, Nx-3), gridAnt->Width(MAC::X, Nx-2));
                const double aLo = SigOf(gridAnt, dt, dh,
                                         SIMULATION.EPS,
                                         gridAnt->Width(MAC::X, 1), gridAnt->FaceGap(MAC::X, 1));
                const double aHi = SigOf(gridAnt, dt, dh,
                                         SIMULATION.EPS,
                                         gridAnt->Width(MAC::X, Nx-2), gridAnt->FaceGap(MAC::X, Nx-3));
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += aLo*gridSol->GetU(i,1);
                    RHS(u_size-1) += aHi*gridSol->GetU(i,Nx-1);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->GetU(i,2), UpwindBack(rhoLo, gridAnt->Width(MAC::X, 1), gridAnt->Width(MAC::X, 2)), UpwindFore(rhoLo, gridAnt->Width(MAC::X, 1), gridAnt->Width(MAC::X, 2))).low * gridSol->GetU(i,1) + aLo*gridSol->GetU(i,1);
                    RHS(u_size-1) += -UpwindCoefficients(velocityField->GetU(i,Nx-2), UpwindBack(rhoHi, gridAnt->Width(MAC::X, Nx-3), gridAnt->Width(MAC::X, Nx-2)), UpwindFore(rhoHi, gridAnt->Width(MAC::X, Nx-3), gridAnt->Width(MAC::X, Nx-2))).up * gridSol->GetU(i,Nx-1) + aHi*gridSol->GetU(i,Nx-1);
                }
                else
                {
                    RHS(0)        +=  rhoLo*gridSol->GetU(i,1)     *velocityField->GetU(i,2)    + aLo*gridSol->GetU(i,1);
                    RHS(u_size-1) += -rhoHi*gridSol->GetU(i,Nx-1)  *velocityField->GetU(i,Nx-2) + aHi*gridSol->GetU(i,Nx-1);
                }
            }

            // Solid Masking Logic
            for(int j = 0; j < u_size; j++) {
                int grid_j = j + 2;
                if(gridSol->FaceGate(MAC::U, i, grid_j) == GateState::WALL) {
                    RHS(j) = 0.0;
                    mid(j) = 1.0;
                    if(j > 0) low(j) = 0.0;
                    if(j < u_size-1) up(j) = 0.0;

                    // Crucial: Clear the "neighboring" dependency to keep the matrix symmetric/correct
                    if(j > 0) up(j-1) = 0.0;
                    if(j < u_size-1) low(j+1) = 0.0;
                }
            }

            TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

            for(int j = 0; j < u_size; j++) {
                gridSol->SetU(i, j + 2, SOL(j));
            }
        }
    }
}
void ADI2D::SolveADI_Y_U_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    double dh = ADI2D::dh;
    double dt = SIMULATION.dt;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;

    int u_size = Ny - 2;

    #pragma omp parallel
    {
        // Allocate vectors once per thread to avoid repeated heap allocation
        Eigen::VectorXd low(u_size), mid(u_size), up(u_size);
        Eigen::VectorXd RHS(u_size), SOL(u_size);

        #pragma omp for
        for(int j = 2; j < Nx+1-2; j++) {

            int gi = 1; //separate counter for the indexing
            for(int i = 0; i < u_size; i++) {
                // Diagonal values from the implicit diffusion operator
                // u face along y sits at cell centres: DyFace outside, Width(Y, gi) as the width
                const double hCu = gridAnt->Width(MAC::Y, gi);
                if (!gridAnt->nonUniform)
                {
                    mid(i) = 1.0 + 2.0 * ((dt/(dh*dh*2)) * (SIMULATION.EPS)); 
                    low(i) = -((dt/(dh*dh*2)) * (SIMULATION.EPS)); 
                    up(i)  = -((dt/(dh*dh*2)) * (SIMULATION.EPS)); 
                }
                else
                {
                    const double nuC = SIMULATION.EPS;
                    mid(i) = 1.0 + SigOf(gridAnt, dt, dh, nuC, gridAnt->FaceGap(MAC::Y, gi - 1), hCu)
                                 + SigOf(gridAnt, dt, dh, nuC, gridAnt->FaceGap(MAC::Y, gi), hCu);
                    low(i) = -SigOf(gridAnt, dt, dh,
                                    SIMULATION.EPS,
                                    gridAnt->FaceGap(MAC::Y, gi - 1), hCu);
                    up(i)  = -SigOf(gridAnt, dt, dh,
                                    SIMULATION.EPS,
                                    gridAnt->FaceGap(MAC::Y, gi), hCu);
                }
                gi++;
            }

            int c = 0;
            for(int i = 1; i < Ny-1; i++) {

                const double hCux = gridAnt->FaceGap(MAC::X, j - 1);
                double sig_center =   SigOf(gridAnt, dt, dh, SIMULATION.EPS,   gridAnt->Width(MAC::X, j-1), hCux);
                double sig_right =    SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->Width(MAC::X, j),   hCux);
                double sig_left =     SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->Width(MAC::X, j-1), hCux);
                double sig_centerR =  gridAnt->nonUniform
                                    ? SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->Width(MAC::X, j), hCux)
                                    : sig_center;

                // Explicit Diffusion and Convection in X (from gridAnt)
                double X_DIFF = gridAnt->nonUniform
                    ? (sig_right*gridAnt->GetU(i,j+1) - (sig_center + sig_centerR)*gridAnt->GetU(i,j) + sig_left*gridAnt->GetU(i,j-1))
                    : (sig_right*gridAnt->GetU(i,j+1) - 2*sig_center*gridAnt->GetU(i,j) + sig_left*gridAnt->GetU(i,j-1));
                double U_CONV = velocityField->GetU(i,j) * (gridAnt->GetU(i,j+1) - gridAnt->GetU(i,j-1));

                const double rhoX = RhoOf(gridAnt, dt, dh, gridAnt->Width(MAC::X, j-1), gridAnt->Width(MAC::X, j));
                const double rhoY = RhoOf(gridAnt, dt, dh, gridAnt->FaceGap(MAC::Y, i-1), gridAnt->FaceGap(MAC::Y, i));

                const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/2.0, FluxGeomAt(velocityField, MAC::U, MAC::X, i,j), UPWIND, gridAnt->GetU(i,j-1), gridAnt->GetU(i,j), gridAnt->GetU(i,j+1)))
                        : UPWIND
                    ? UpwindDifference(velocityField->GetU(i,j), gridAnt->GetU(i,j-1), gridAnt->GetU(i,j), gridAnt->GetU(i,j+1), UpwindBack(rhoX, gridAnt->Width(MAC::X, j-1), gridAnt->Width(MAC::X, j)), UpwindFore(rhoX, gridAnt->Width(MAC::X, j-1), gridAnt->Width(MAC::X, j)))
                    : rhoX*(U_CONV);
                RHS(c) = gridAnt->GetU(i,j) + (X_DIFF) - CONV_EXP;
                mid(c) += WallDiagonal(gridSol, MAC::U, MAC::Y, i, j, 0,
                                       SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::Y, i-1), gridAnt->Width(MAC::Y, i)),
                                       SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::Y, i),   gridAnt->Width(MAC::Y, i)));
                if (INCREMENTAL) RHS(c) -= (dt/2.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::X, i,j);   // grad p^n

                // Implicit Convective terms in Y (added to the diagonals)
                if (FLUX)
                {
                    const UpwindRow fr = FluxCoefficients(dt/2.0, FluxGeomAt(velocityField, MAC::U, MAC::Y, i,j), UPWIND);
                    mid(c) += fr.mid;
                    if(c < u_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetU(Ny-1,j);
                    if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetU(0,j);
                }
                else if (UPWIND)
                {
                    const UpwindRow uw = UpwindCoefficients(velocityField->getVatU(i,j), UpwindBack(rhoY, gridAnt->FaceGap(MAC::Y, i-1), gridAnt->FaceGap(MAC::Y, i)), UpwindFore(rhoY, gridAnt->FaceGap(MAC::Y, i-1), gridAnt->FaceGap(MAC::Y, i)));
                    mid(c) += uw.mid;
                    if(c < u_size - 1) up(c)  += uw.up;
                    if(c > 0)          low(c) += uw.low;
                }
                else
                {
                    double v_vel = rhoY * velocityField->getVatU(i,j);
                    if(c < u_size - 1) up(c)  += v_vel;
                    if(c > 0)          low(c) -= v_vel;
                }

                c++;
            }

            // Boundary Terms (Top and Bottom)
            {
                const double rhoLo = RhoOf(gridAnt, dt, dh, gridAnt->FaceGap(MAC::Y, 0), gridAnt->FaceGap(MAC::Y, 1));
                const double rhoHi = RhoOf(gridAnt, dt, dh, gridAnt->FaceGap(MAC::Y, Ny-3), gridAnt->FaceGap(MAC::Y, Ny-2));
                const double aLo = SigOf(gridAnt, dt, dh,
                                         SIMULATION.EPS,
                                         gridAnt->FaceGap(MAC::Y, 0), gridAnt->Width(MAC::Y, 1));
                const double aHi = SigOf(gridAnt, dt, dh,
                                         SIMULATION.EPS,
                                         gridAnt->FaceGap(MAC::Y, Ny-2), gridAnt->Width(MAC::Y, Ny-2));
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += aLo*gridSol->GetU(0,j);
                    RHS(u_size-1) += aHi*gridSol->GetU(Ny-1,j);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->getVatU(1,j), UpwindBack(rhoLo, gridAnt->FaceGap(MAC::Y, 0), gridAnt->FaceGap(MAC::Y, 1)), UpwindFore(rhoLo, gridAnt->FaceGap(MAC::Y, 0), gridAnt->FaceGap(MAC::Y, 1))).low * gridSol->GetU(0,j) + aLo*gridSol->GetU(0,j);
                    RHS(u_size-1) += -UpwindCoefficients(velocityField->getVatU(Ny-2,j), UpwindBack(rhoHi, gridAnt->FaceGap(MAC::Y, Ny-3), gridAnt->FaceGap(MAC::Y, Ny-2)), UpwindFore(rhoHi, gridAnt->FaceGap(MAC::Y, Ny-3), gridAnt->FaceGap(MAC::Y, Ny-2))).up * gridSol->GetU(Ny-1,j) + aHi*gridSol->GetU(Ny-1,j);
                }
                else
                {
                    RHS(0)        +=  rhoLo*velocityField->getVatU(1,j)*gridSol->GetU(0,j)       + aLo*gridSol->GetU(0,j);
                    RHS(u_size-1) += -rhoHi*velocityField->getVatU(Ny-2,j)*gridSol->GetU(Ny-1,j) + aHi*gridSol->GetU(Ny-1,j);
                }
                RHS(0)        += WallRing(gridSol, MAC::U, MAC::Y, 1,    j, 0, -1, aLo, gridSol->GetU(0,j));
                RHS(u_size-1) += WallRing(gridSol, MAC::U, MAC::Y, Ny-2, j, 0, +1, aHi, gridSol->GetU(Ny-1,j));
            }

            // Solid Mask Logic
            for(int i = 0; i < u_size; i++) {
                int grid_i = i + 1; // mapping back to grid index
                if(gridSol->FaceGate(MAC::U, grid_i, j) == GateState::WALL) {
                    RHS(i) = 0.0;
                    mid(i) = 1.0;
                    if(i > 0) low(i) = 0.0;
                    if(i < u_size-1) up(i) = 0.0;

                    // Zero out the connections from neighbors to this row
                    if(i > 0)        up(i-1)  = 0.0;
                    if(i < u_size-1) low(i+1) = 0.0;
                }
            }

            // Solve the tridiagonal system for this column
            TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

            // Store results back to gridSol
            for(int i = 0; i < u_size; i++) {
                gridSol->SetU(i + 1, j, SOL(i));
            }
        }
    }
}

void ADI2D::SolveADI_X_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    double dh = ADI2D::dh;
    double dt = SIMULATION.dt;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;

    int v_size = Nx - 2;

    #pragma omp parallel
    {
        Eigen::VectorXd low(v_size), mid(v_size), up(v_size);
        Eigen::VectorXd RHS(v_size), SOL(v_size);

        #pragma omp for
        for(int i = 2; i < Ny+1-2; i++) {
            int gj = 1;
            for(int k = 0; k < v_size; k++) {
                // v face along x sits at cell centres: DxFace outside, Width(X, gj) as the width
                if (gridAnt->nonUniform)
                {
                    const double hCv = gridAnt->Width(MAC::X, gj);
                    const double nuC = SIMULATION.EPS;
                    mid(k) = 1.0 + SigOf(gridAnt, dt, dh, nuC, gridAnt->FaceGap(MAC::X, gj - 1), hCv)
                                 + SigOf(gridAnt, dt, dh, nuC, gridAnt->FaceGap(MAC::X, gj), hCv);
                    low(k) = -SigOf(gridAnt, dt, dh,
                                    SIMULATION.EPS,
                                    gridAnt->FaceGap(MAC::X, gj - 1), hCv);
                    up(k)  = -SigOf(gridAnt, dt, dh,
                                    SIMULATION.EPS,
                                    gridAnt->FaceGap(MAC::X, gj), hCv);
                }
                else
                {
                mid(k) = 1.0 + 2.0 * (dt/(dh*dh*2)) * (SIMULATION.EPS);
                low(k) = -(dt/(dh*dh*2)) * (SIMULATION.EPS);
                up(k)  = -(dt/(dh*dh*2)) * (SIMULATION.EPS);
                }
                gj++;
            }

            int c = 0;
            for(int j = 1; j < Nx-1; j++) {
                // v face along y: cells Width(Y, i-1) below and Width(Y, i) above, own width FaceGap(Y, i-1)
                const double hCvy = gridAnt->FaceGap(MAC::Y, i - 1);
                double sig_center =   SigOf(gridAnt, dt, dh, SIMULATION.EPS,   gridAnt->Width(MAC::Y, i-1), hCvy);
                double sig_top =      SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->Width(MAC::Y, i),   hCvy);
                double sig_bot =      SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->Width(MAC::Y, i-1), hCvy);
                double sig_centerT =  gridAnt->nonUniform
                                    ? SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->Width(MAC::Y, i), hCvy)
                                    : sig_center;


                double Y_DIFF = gridAnt->nonUniform
                    ? (sig_top*gridAnt->GetV(i+1,j) - (sig_center + sig_centerT)*gridAnt->GetV(i,j) + sig_bot*gridAnt->GetV(i-1,j))
                    : (sig_top*gridAnt->GetV(i+1,j) - 2*sig_center*gridAnt->GetV(i,j) + sig_bot*gridAnt->GetV(i-1,j));
                double V_CONV = (velocityField->GetV(i,j)) * ((gridAnt->GetV(i+1,j) - gridAnt->GetV(i-1,j)));

                // v faces are whole cells apart in y (Dy) but half-cells apart in x (DxFace)
                const double rhoY = RhoOf(gridAnt, dt, dh, gridAnt->Width(MAC::Y, i-1), gridAnt->Width(MAC::Y, i));
                const double rhoX = RhoOf(gridAnt, dt, dh, gridAnt->FaceGap(MAC::X, j-1), gridAnt->FaceGap(MAC::X, j));

                const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/2.0, FluxGeomAt(velocityField, MAC::V, MAC::Y, i,j), UPWIND, gridAnt->GetV(i-1,j), gridAnt->GetV(i,j), gridAnt->GetV(i+1,j)))
                        : UPWIND
                    ? UpwindDifference(velocityField->GetV(i,j), gridAnt->GetV(i-1,j), gridAnt->GetV(i,j), gridAnt->GetV(i+1,j), UpwindBack(rhoY, gridAnt->Width(MAC::Y, i-1), gridAnt->Width(MAC::Y, i)), UpwindFore(rhoY, gridAnt->Width(MAC::Y, i-1), gridAnt->Width(MAC::Y, i)))
                    : rhoY*(V_CONV);
                RHS(c) = gridAnt->GetV(i,j) + (Y_DIFF) - CONV_EXP;
                mid(c) += WallDiagonal(gridSol, MAC::V, MAC::X, i, j, 0,
                                       SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::X, j-1), gridAnt->Width(MAC::X, j)),
                                       SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::X, j),   gridAnt->Width(MAC::X, j)));
                if (INCREMENTAL) RHS(c) -= (dt/2.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::Y, i,j);   // grad p^n

                // Implicit Convection
                if (FLUX)
                {
                    const UpwindRow fr = FluxCoefficients(dt/2.0, FluxGeomAt(velocityField, MAC::V, MAC::X, i,j), UPWIND);
                    mid(c) += fr.mid;
                    if(c < v_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetV(i,Nx-1);
                    if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetV(i,0);
                }
                else if (UPWIND)
                {
                    const UpwindRow uw = UpwindCoefficients(velocityField->getUatV(i,j), UpwindBack(rhoX, gridAnt->FaceGap(MAC::X, j-1), gridAnt->FaceGap(MAC::X, j)), UpwindFore(rhoX, gridAnt->FaceGap(MAC::X, j-1), gridAnt->FaceGap(MAC::X, j)));
                    mid(c) += uw.mid;
                    if(c < v_size - 1) up(c)  += uw.up;
                    if(c > 0)          low(c) += uw.low;
                }
                else
                {
                    double u_vel = rhoX * velocityField->getUatV(i,j);
                    if(c < v_size - 1) up(c)  += u_vel;
                    if(c > 0)          low(c) -= u_vel;
                }
                c++;
            }

            // Boundary Conditions (Left/Right)
            {
                const double rhoLo = RhoOf(gridAnt, dt, dh, gridAnt->FaceGap(MAC::X, 0), gridAnt->FaceGap(MAC::X, 1));
                const double rhoHi = RhoOf(gridAnt, dt, dh, gridAnt->FaceGap(MAC::X, Nx-3), gridAnt->FaceGap(MAC::X, Nx-2));
                const double aLo = SigOf(gridAnt, dt, dh,
                                         SIMULATION.EPS,
                                         gridAnt->FaceGap(MAC::X, 0), gridAnt->Width(MAC::X, 1));
                const double aHi = SigOf(gridAnt, dt, dh,
                                         SIMULATION.EPS,
                                         gridAnt->FaceGap(MAC::X, Nx-2), gridAnt->Width(MAC::X, Nx-2));
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += aLo * gridSol->GetV(i,0);
                    RHS(v_size-1) += aHi * gridSol->GetV(i,Nx-1);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->getUatV(i,1), UpwindBack(rhoLo, gridAnt->FaceGap(MAC::X, 0), gridAnt->FaceGap(MAC::X, 1)), UpwindFore(rhoLo, gridAnt->FaceGap(MAC::X, 0), gridAnt->FaceGap(MAC::X, 1))).low * gridSol->GetV(i,0) + aLo * gridSol->GetV(i,0);
                    RHS(v_size-1) += -UpwindCoefficients(velocityField->getUatV(i,Nx-2), UpwindBack(rhoHi, gridAnt->FaceGap(MAC::X, Nx-3), gridAnt->FaceGap(MAC::X, Nx-2)), UpwindFore(rhoHi, gridAnt->FaceGap(MAC::X, Nx-3), gridAnt->FaceGap(MAC::X, Nx-2))).up * gridSol->GetV(i,Nx-1) + aHi * gridSol->GetV(i,Nx-1);
                }
                else
                {
                    RHS(0)        +=  rhoLo * velocityField->getUatV(i,1)    * gridSol->GetV(i,0)    + aLo * gridSol->GetV(i,0);
                    RHS(v_size-1) += -rhoHi * velocityField->getUatV(i,Nx-2) * gridSol->GetV(i,Nx-1) + aHi * gridSol->GetV(i,Nx-1);
                }
                RHS(0)        += WallRing(gridSol, MAC::V, MAC::X, i, 1,    0, -1, aLo, gridSol->GetV(i,0));
                RHS(v_size-1) += WallRing(gridSol, MAC::V, MAC::X, i, Nx-2, 0, +1, aHi, gridSol->GetV(i,Nx-1));
            }

            // Solid Masking
            for(int j = 0; j < v_size; j++) {
                int grid_j = j + 1;
                if(gridSol->FaceGate(MAC::V, i, grid_j) == GateState::WALL) {
                    RHS(j) = 0.0;
                    mid(j) = 1.0;
                    if(j > 0) low(j) = 0.0;
                    if(j < v_size-1) up(j) = 0.0;
                    if(j > 0) up(j-1) = 0.0;
                    if(j < v_size-1) low(j+1) = 0.0;
                }
            }

            TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

            for(int j = 0; j < v_size; j++) {
                gridSol->SetV(i, j + 1, SOL(j));
            }
        }
    }
}
void ADI2D::SolveADI_Y_V_Step_OPENMP(MAC* gridAnt, MAC* gridSol, MAC* velocityField, double time) {
    int Nx = gridSol->Nx;
    int Ny = gridSol->Ny;
    double dh = ADI2D::dh;
    double dt = SIMULATION.dt;
    const bool UPWIND = ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND;
    const bool FLUX = ADI_SETTINGS.convectionForm == CONVECTION_FLUX;
    const bool INCREMENTAL = ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL;

    int v_size = Ny + 1 - 4;

    #pragma omp parallel
    {
        Eigen::VectorXd low(v_size), mid(v_size), up(v_size);
        Eigen::VectorXd RHS(v_size), SOL(v_size);

        #pragma omp for
        for(int j = 1; j < Nx-1; j++) {
            // FIX GHOSTING: Reset vectors for every column
            int gi = 2;
            for(int k = 0; k < v_size; k++) {
                if (gridAnt->nonUniform)
                {
                    const double hCv = gridAnt->FaceGap(MAC::Y, gi - 1);
                    const double nuC = SIMULATION.EPS;
                    mid(k) = 1.0 + SigOf(gridAnt, dt, dh, nuC, gridAnt->Width(MAC::Y, gi - 1), hCv)
                                 + SigOf(gridAnt, dt, dh, nuC, gridAnt->Width(MAC::Y, gi), hCv);
                    low(k) = -SigOf(gridAnt, dt, dh,
                                    SIMULATION.EPS,
                                    gridAnt->Width(MAC::Y, gi - 1), hCv);
                    up(k)  = -SigOf(gridAnt, dt, dh,
                                    SIMULATION.EPS,
                                    gridAnt->Width(MAC::Y, gi), hCv);
                }
                else
                {
                mid(k) = 1.0 + 2.0 * (dt/(dh*dh*2)) * (SIMULATION.EPS);
                low(k) = -(dt/(dh*dh*2)) * (SIMULATION.EPS);
                up(k)  = -(dt/(dh*dh*2)) * (SIMULATION.EPS);
                }
                gi++;
            }

            int c = 0;
            for(int i = 2; i < Ny+1-2; i++) {

                const double hCvx = gridAnt->Width(MAC::X, j);
                double sig_center =   SigOf(gridAnt, dt, dh, SIMULATION.EPS,   gridAnt->FaceGap(MAC::X, j-1), hCvx);
                double sig_right =    SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::X, j),   hCvx);
                double sig_left =     SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::X, j-1), hCvx);
                double sig_centerR =  gridAnt->nonUniform
                                    ? SigOf(gridAnt, dt, dh, SIMULATION.EPS, gridAnt->FaceGap(MAC::X, j), hCvx)
                                    : sig_center;


                double X_DIFF = gridAnt->nonUniform
                    ? (sig_right*gridAnt->GetV(i,j+1) - (sig_center + sig_centerR)*gridAnt->GetV(i,j) + sig_left*gridAnt->GetV(i,j-1))
                    : (sig_right*gridAnt->GetV(i,j+1) - 2*sig_center*gridAnt->GetV(i,j) + sig_left*gridAnt->GetV(i,j-1));
                double U_CONV = velocityField->getUatV(i,j) * ((gridAnt->GetV(i,j+1) - gridAnt->GetV(i,j-1)));

                const double rhoX = RhoOf(gridAnt, dt, dh, gridAnt->FaceGap(MAC::X, j-1), gridAnt->FaceGap(MAC::X, j));
                const double rhoY = RhoOf(gridAnt, dt, dh, gridAnt->Width(MAC::Y, i-1), gridAnt->Width(MAC::Y, i));

                const double CONV_EXP = FLUX
                        ? (FluxDifference(dt/2.0, FluxGeomAt(velocityField, MAC::V, MAC::X, i,j), UPWIND, gridAnt->GetV(i,j-1), gridAnt->GetV(i,j), gridAnt->GetV(i,j+1)))
                        : UPWIND
                    ? UpwindDifference(velocityField->getUatV(i,j), gridAnt->GetV(i,j-1), gridAnt->GetV(i,j), gridAnt->GetV(i,j+1), UpwindBack(rhoX, gridAnt->FaceGap(MAC::X, j-1), gridAnt->FaceGap(MAC::X, j)), UpwindFore(rhoX, gridAnt->FaceGap(MAC::X, j-1), gridAnt->FaceGap(MAC::X, j)))
                    : rhoX*(U_CONV);
                RHS(c) = gridAnt->GetV(i,j) + (X_DIFF) - CONV_EXP;
                RHS(c) += WallExplicit(gridSol, MAC::V, MAC::X, i, j, 0, sig_left, sig_right, gridAnt->GetV(i,j-1), gridAnt->GetV(i,j), gridAnt->GetV(i,j+1));
                if (INCREMENTAL) RHS(c) -= (dt/2.0)/SIMULATION.RHO * gridAnt->GetGradP(MAC::Y, i,j);   // grad p^n

                // Implicit Convection
                if (FLUX)
                {
                    const UpwindRow fr = FluxCoefficients(dt/2.0, FluxGeomAt(velocityField, MAC::V, MAC::Y, i,j), UPWIND);
                    mid(c) += fr.mid;
                    if(c < v_size - 1) up(c)  += fr.up;  else RHS(c) -= fr.up * gridSol->GetV(Ny-1,j);
                    if(c > 0)          low(c) += fr.low; else RHS(c) -= fr.low * gridSol->GetV(1,j);
                }
                else if (UPWIND)
                {
                    const UpwindRow uw = UpwindCoefficients(velocityField->GetV(i,j), UpwindBack(rhoY, gridAnt->Width(MAC::Y, i-1), gridAnt->Width(MAC::Y, i)), UpwindFore(rhoY, gridAnt->Width(MAC::Y, i-1), gridAnt->Width(MAC::Y, i)));
                    mid(c) += uw.mid;
                    if(c < v_size - 1) up(c)  += uw.up;
                    if(c > 0)          low(c) += uw.low;
                }
                else
                {
                    double v_vel = rhoY * velocityField->GetV(i,j);
                    if(c < v_size - 1) up(c)  += v_vel;
                    if(c > 0)          low(c) -= v_vel;
                }
                c++;
            }

            // Boundary Conditions (Top/Bottom)
            {
                const double rhoLo = RhoOf(gridAnt, dt, dh, gridAnt->Width(MAC::Y, 1), gridAnt->Width(MAC::Y, 2));
                const double rhoHi = RhoOf(gridAnt, dt, dh, gridAnt->Width(MAC::Y, Ny-3), gridAnt->Width(MAC::Y, Ny-2));
                const double aLo = SigOf(gridAnt, dt, dh,
                                         SIMULATION.EPS,
                                         gridAnt->Width(MAC::Y, 1), gridAnt->FaceGap(MAC::Y, 1));
                const double aHi = SigOf(gridAnt, dt, dh,
                                         SIMULATION.EPS,
                                         gridAnt->Width(MAC::Y, Ny-2), gridAnt->FaceGap(MAC::Y, Ny-3));
                if (FLUX)
                {
                    // the convective share of the known faces is moved in the row loop
                    RHS(0)        += aLo * gridSol->GetV(1,j);
                    RHS(v_size-1) += aHi * gridSol->GetV(Ny-1,j);
                }
                else if (UPWIND)
                {
                    RHS(0)        += -UpwindCoefficients(velocityField->GetV(2,j), UpwindBack(rhoLo, gridAnt->Width(MAC::Y, 1), gridAnt->Width(MAC::Y, 2)), UpwindFore(rhoLo, gridAnt->Width(MAC::Y, 1), gridAnt->Width(MAC::Y, 2))).low * gridSol->GetV(1,j) + aLo * gridSol->GetV(1,j);
                    RHS(v_size-1) += -UpwindCoefficients(velocityField->GetV(Ny-2,j), UpwindBack(rhoHi, gridAnt->Width(MAC::Y, Ny-3), gridAnt->Width(MAC::Y, Ny-2)), UpwindFore(rhoHi, gridAnt->Width(MAC::Y, Ny-3), gridAnt->Width(MAC::Y, Ny-2))).up * gridSol->GetV(Ny-1,j) + aHi * gridSol->GetV(Ny-1,j);
                }
                else
                {
                    RHS(0)        +=  rhoLo * velocityField->GetV(2,j)      * gridSol->GetV(1,j)      + aLo * gridSol->GetV(1,j);
                    RHS(v_size-1) += -rhoHi * velocityField->GetV(Ny-2,j)   * gridSol->GetV(Ny-1,j)   + aHi * gridSol->GetV(Ny-1,j);
                }
            }

            // Solid Masking
            for(int i = 0; i < v_size; i++) {
                int grid_i = i + 2;
                if(gridSol->FaceGate(MAC::V, grid_i, j) == GateState::WALL) {
                    RHS(i) = 0.0;
                    mid(i) = 1.0;
                    if(i > 0) low(i) = 0.0;
                    if(i < v_size-1) up(i) = 0.0;
                    if(i > 0) up(i-1) = 0.0;
                    if(i < v_size-1) low(i+1) = 0.0;
                }
            }

            TDMA_VECTORIZED_N(low, mid, up, RHS, SOL, (int)RHS.size());

            for(int i = 0; i < v_size; i++) {
                gridSol->SetV(i + 2, j, SOL(i));
            }
        }
    }
}

void ADI2D::InitializeADI2D(MAC* grid,double dt,Vec2(*VelocityBorderFunction)(double, double, double),double(*PressureFunction)(double, double, double)){

    ADI2D::VelocityBorderFunction = VelocityBorderFunction;
    ADI2D::PressureFunction = PressureFunction;

    ADI2D::dh = grid->dh;

    // The sweep grids copy the grid's layout and boundary functions; they carry no scalars.
    grid->SetBoundaryFunctions(VelocityBorderFunction, PressureFunction);
    ADI2D::X_STEP_SOL.InitializeFromGrid(*grid, SIMULATION.time);
    ADI2D::Y_STEP_SOL.InitializeFromGrid(*grid, SIMULATION.time);

    // if turbulenc model, init the model here

    if (ADI_SETTINGS.vorticityConfinement)
        VorticityConfinement2D::Initialize(grid);

    RegisterSimulationStep(ADI2D::SolveADIStep, "momentum", 0);
}

// IM BREAKING the component on the exit
void ADI2D::SolveADIStep(MAC* gridAnt,MAC* gridSol,ParticleSet *particles, double time){
    // The step's start time, before the sweeps advance `time`: the body force is evaluated here.
    const double stepStart = time;

    time += SIMULATION.dt/2.0;
    ADI2D::X_STEP_SOL.CopyFlowFields(*gridAnt);
    ADI2D::X_STEP_SOL.EnforceVelocityBC(time);

    #pragma omp single
    {
        #pragma omp task
        {
            SolveADI_X_U_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);
        }
        #pragma omp task
        {
            SolveADI_X_V_Step_OPENMP(gridAnt,&X_STEP_SOL,gridAnt,time);
        }
    }

    time += SIMULATION.dt/2.0;
    ADI2D::Y_STEP_SOL.CopyFlowFields(ADI2D::X_STEP_SOL);
    ADI2D::Y_STEP_SOL.EnforceVelocityBC(time);

    #pragma omp single
    {
        #pragma omp task
        {
            SolveADI_Y_U_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);
        }
        #pragma omp task
        {
            SolveADI_Y_V_Step_OPENMP(&X_STEP_SOL,&Y_STEP_SOL,gridAnt,time);
        }
    }

    // Only what ADI computes. A whole-grid copy also asked Y_STEP_SOL for the scalars, which it
    // does not carry, and tripped MatchScalar's warning on every step with a scalar declared.
    gridSol->CopyFlowFields(Y_STEP_SOL);
    // The body force goes in AFTER the sweeps and BEFORE the pressure solve, so the projection
    // removes the part of it that is not divergence free.
    gridSol->AddForce(stepStart, SIMULATION.dt);
    // Same place and same reason as the body force: before the projection.
    if (ADI_SETTINGS.vorticityConfinement)
        VorticityConfinement2D::Apply(gridSol, SIMULATION.dt);
    gridSol->SetBoundaryFunctions(VelocityBorderFunction, PressureFunction);
    gridSol->EnforceVelocityBC(time);
}

// Frees the sweep grids. The next Initialize builds them again.
void ADI2D::Dispose()
{
    X_STEP_SOL = MAC();
    Y_STEP_SOL = MAC();
}
