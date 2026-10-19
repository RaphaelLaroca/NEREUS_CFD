// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include "DiffusionSolver2D.h"
#include "FLIPSettings.h"

// Static member initialization for DiffusionSolver2D

// Grid dimensions
int DiffusionSolver2D::Nx = 0;
int DiffusionSolver2D::Ny = 0;

// Non-zero count
int DiffusionSolver2D::NON_ZERO = 0;

// Time step
double DiffusionSolver2D::dt = 0.0;

// COO format arrays for U
int *DiffusionSolver2D::u_collums = nullptr;
int *DiffusionSolver2D::u_rows = nullptr;
double *DiffusionSolver2D::u_values = nullptr;

// COO format arrays for V
int *DiffusionSolver2D::v_collums = nullptr;
int *DiffusionSolver2D::v_rows = nullptr;
double *DiffusionSolver2D::v_values = nullptr;

// Sparse matrices
SparseMatrix DiffusionSolver2D::U_DIFFUSION_MATRIX_EIGEN;
SparseMatrix DiffusionSolver2D::V_DIFFUSION_MATRIX_EIGEN;

// CSR matrices

// Index mapping vectors
VectorXd DiffusionSolver2D::U_IDP;
VectorXd DiffusionSolver2D::V_IDP;

// Mask vectors
std::vector<Stencil2D> DiffusionSolver2D::U_MASK;
std::vector<Stencil2D> DiffusionSolver2D::V_MASK;

// Eigen solver
Eigen::BiCGSTAB<SparseMatrix> *DiffusionSolver2D::solver = nullptr;

// ADI Solver
MAC DiffusionSolver2D::gridMid = MAC();

void DiffusionSolver2D::InitializeDiffusionSolver(MAC *grid)
{

    DiffusionSolver2D::Nx = SIMULATION.Nx;
    DiffusionSolver2D::Ny = SIMULATION.Ny;
    // One mask per face, not per fluid cell - the builders push over the whole face array.
    DiffusionSolver2D::U_MASK.reserve((SIMULATION.Nx + 1) * SIMULATION.Ny);
    DiffusionSolver2D::V_MASK.reserve(SIMULATION.Nx * (SIMULATION.Ny + 1));
    DiffusionSolver2D::dt = SIMULATION.dt;
    DiffusionSolver2D::U_IDP = VectorXd((Nx + 1) * Ny);
    DiffusionSolver2D::U_IDP.setConstant(-1);
    DiffusionSolver2D::V_IDP = VectorXd((Nx) * (Ny + 1));
    DiffusionSolver2D::V_IDP.setConstant(-1);

    gridMid.InitializeFromGrid(*grid, SIMULATION.time);
}

// Geometry of the staggered faces this solver works on. U(i,j) sits at x = j*dh, y = i*dh + dh/2,
// between cells (i,j-1) and (i,j).

static inline bool IsU_Boundary(int i, int j)
{
    return (i <= 0 || i >= SIMULATION.Ny - 1 || j <= 1 || j >= SIMULATION.Nx - 1);
}

static inline bool IsV_Boundary(int i, int j)
{
    return (i <= 1 || i >= SIMULATION.Ny - 1 || j <= 0 || j >= SIMULATION.Nx - 1);
}

// A face is an unknown only where the projection will also touch it.
static inline bool IsU_Solvable(MAC *grid, int i, int j)
{
    if (IsU_Boundary(i, j))
        return false;
    return grid->IsFluidFace(MAC::U, i, j);
}

static inline bool IsV_Solvable(MAC *grid, int i, int j)
{
    if (IsV_Boundary(i, j))
        return false;
    return grid->IsFluidFace(MAC::V, i, j);
}

// Coupling coefficients. Each of these is the viscosity on the interface shared by two neighbouring
// face control volumes.

// Interface between U(i,j) and U(i,j+1): the centre of cell (i,j).
static inline double NuU_X(int i, int j)
{
    return SIMULATION.EPS;
}

// Interface between U(i,j) and U(i+1,j): the top edges of cells (i,j-1) and (i,j).
static inline double NuU_Y(int i, int j)
{
    return SIMULATION.EPS;
}

// Interface between V(i,j) and V(i+1,j): the centre of cell (i,j).
static inline double NuV_Y(int i, int j)
{
    return SIMULATION.EPS;
}

// Interface between V(i,j) and V(i,j+1): the right edges of cells (i-1,j) and (i,j).
static inline double NuV_X(int i, int j)
{
    return SIMULATION.EPS;
}

// Neighbour gating. A face reaches its neighbour through the cell (or pair of cells) that separates
// the two control volumes, NOT through the cell one step further out.

// For the transverse directions the interface spans two cells, so it is open only
// when both are fluid and it is a wall as soon as either one is solid.

void DiffusionSolver2D::Update_V_DiffusionMatrix(MAC *grid)
{
    double dh = SIMULATION.dh;
    double factor = dt / (dh * dh);

    DiffusionSolver2D::V_IDP = VectorXd((Nx) * (Ny + 1));
    DiffusionSolver2D::V_IDP.setConstant(-1);
    V_MASK.clear();

    // ========== V COMPONENT ==========

    // First pass: number the unknowns and build the stencil weights.
    int v_dof_count = 0;
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            if (!IsV_Solvable(grid, i, j))
            {
                Set_V_IDP(i, j, -1);
                continue;
            }

            // Only unknowns get a stencil.
            Stencil2D mask;
            mask.setZero();

            // +X : V(i,j+1), through the right edges of cells (i-1,j) and (i,j)
            switch (grid->FaceGate(MAC::V, i, j + 1))
            {
            case GateState::OPEN: mask(1, 2) = 1.0; mask(1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

            // -X : V(i,j-1)
            switch (grid->FaceGate(MAC::V, i, j - 1))
            {
            case GateState::OPEN: mask(1, 0) = 1.0; mask(1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

            // +Y : V(i+1,j) is reached through cell (i,j)
            switch (grid->CellGate(i, j))
            {
            case GateState::OPEN: mask(2, 1) = 1.0; mask(1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

            // -Y : V(i-1,j) is reached through cell (i-1,j)
            switch (grid->CellGate(i - 1, j))
            {
            case GateState::OPEN: mask(0, 1) = 1.0; mask(1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

            Set_V_IDP(i, j, v_dof_count);
            v_dof_count++;
            DiffusionSolver2D::V_MASK.push_back(mask);
        }
    }

    // Second pass: count non-zeros. This has to sweep exactly the same index range
    // that the first pass numbered, or unknowns end up with no equation at all.
    int v_nnz = 0;
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            int MAT_LINE = Get_V_IDP(i, j);
            if (MAT_LINE == -1)
                continue;

            const Stencil2D &mask = GetVMask(i, j);

            if (mask(1, 1) != 0.0)
                v_nnz++;
            if (mask(2, 1) != 0.0 && Get_V_IDP(i + 1, j) != -1)
                v_nnz++;
            if (mask(0, 1) != 0.0 && Get_V_IDP(i - 1, j) != -1)
                v_nnz++;
            if (mask(1, 2) != 0.0 && Get_V_IDP(i, j + 1) != -1)
                v_nnz++;
            if (mask(1, 0) != 0.0 && Get_V_IDP(i, j - 1) != -1)
                v_nnz++;
        }
    }

    // Allocate arrays
    v_collums = (int *)malloc(sizeof(int) * v_nnz);
    v_rows = (int *)malloc(sizeof(int) * v_nnz);
    v_values = (double *)calloc(v_nnz, sizeof(double));

    // Third pass: assemble. The diagonal is rebuilt from the same per-direction
    // coefficients used off the diagonal, so every interior row sums to zero.
    int c = 0;
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            const Stencil2D &mask = GetVMask(i, j);
            int MAT_LINE = Get_V_IDP(i, j);

            if (MAT_LINE == -1)
                continue;

            double nu_up = NuV_Y(i, j);
            double nu_down = NuV_Y(i - 1, j);
            double nu_right = NuV_X(i, j);
            double nu_left = NuV_X(i, j - 1);

            double diagonal = 0.0;
            if (mask(2, 1) != 0.0 || grid->CellGate(i, j) == GateState::WALL)
                diagonal -= factor * nu_up;
            if (mask(0, 1) != 0.0 || grid->CellGate(i - 1, j) == GateState::WALL)
                diagonal -= factor * nu_down;
            if (mask(1, 2) != 0.0 || grid->FaceGate(MAC::V, i, j + 1) == GateState::WALL)
                diagonal -= factor * nu_right;
            if (mask(1, 0) != 0.0 || grid->FaceGate(MAC::V, i, j - 1) == GateState::WALL)
                diagonal -= factor * nu_left;

            if (diagonal != 0.0)
            {
                v_rows[c] = MAT_LINE;
                v_collums[c] = MAT_LINE;
                v_values[c] = diagonal;
                c++;
            }

            if (mask(2, 1) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i + 1, j);
                if (MAT_COLLUM != -1)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_COLLUM;
                    v_values[c] = factor * nu_up;
                    c++;
                }
            }

            if (mask(0, 1) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i - 1, j);
                if (MAT_COLLUM != -1)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_COLLUM;
                    v_values[c] = factor * nu_down;
                    c++;
                }
            }

            if (mask(1, 2) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i, j + 1);
                if (MAT_COLLUM != -1)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_COLLUM;
                    v_values[c] = factor * nu_right;
                    c++;
                }
            }

            if (mask(1, 0) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i, j - 1);
                if (MAT_COLLUM != -1)
                {
                    v_rows[c] = MAT_LINE;
                    v_collums[c] = MAT_COLLUM;
                    v_values[c] = factor * nu_left;
                    c++;
                }
            }
        }
    }

    // Error checking
    if (c != v_nnz)
    {
        std::cerr << "ERROR: V component NNZ mismatch! Expected " << v_nnz << " but got " << c << std::endl;
    }

    for (int i = 0; i < v_nnz; ++i)
    {
        if (v_rows[i] < 0 || v_rows[i] >= v_dof_count ||
            v_collums[i] < 0 || v_collums[i] >= v_dof_count)
        {
            std::cerr << "ERROR V: Out of bounds at entry " << i
                      << " - row=" << v_rows[i]
                      << " col=" << v_collums[i]
                      << " MatSize=" << v_dof_count << std::endl;
        }
    }

    // Build sparse matrix
    std::vector<Eigen::Triplet<double>> triplets_v;
    triplets_v.reserve(v_nnz);
    for (int i = 0; i < v_nnz; ++i)
    {
        triplets_v.push_back(Triplet(v_rows[i], v_collums[i], v_values[i]));
    }

    SparseMatrix mat_v(v_dof_count, v_dof_count);
    mat_v.setFromTriplets(triplets_v.begin(), triplets_v.end());

    SparseMatrix identity_v(v_dof_count, v_dof_count);
    identity_v.setIdentity();
    DiffusionSolver2D::V_DIFFUSION_MATRIX_EIGEN = identity_v - mat_v;

    free(v_collums);
    free(v_rows);
    free(v_values);
}

void DiffusionSolver2D::SolveDiffusion_V_Eigen(MAC *gridAnt, MAC *gridSol)
{
    DiffusionSolver2D::Update_V_DiffusionMatrix(gridAnt);
    int v_matrix_size = V_DIFFUSION_MATRIX_EIGEN.rows();

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-9);

    // ========== Solve V component ==========

    // Build RHS from previous timestep V velocities
    VectorXd RHS_V = VectorXd(v_matrix_size);
    RHS_V.setZero();

    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            int id = Get_V_IDP(i, j);
            if (id == -1)
                continue;
            RHS_V(id) = gridAnt->GetV(i, j);
        }
    }

    solver.compute(V_DIFFUSION_MATRIX_EIGEN);

    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: V matrix decomposition failed!" << std::endl;
        return;
    }

    Eigen::VectorXd SOL_V = solver.solve(RHS_V);

    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: V solve failed!" << std::endl;
        std::cerr << "Iterations: " << solver.iterations() << std::endl;
        std::cerr << "Error: " << solver.error() << std::endl;
        return;
    }

    // Copy V solution back to grid
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            int id = Get_V_IDP(i, j);
            if (id == -1)
                continue;
            gridSol->SetV(i, j, SOL_V(id));
        }
    }
}

void DiffusionSolver2D::Update_U_DiffusionMatrix(MAC *grid)
{
    double dh = SIMULATION.dh;
    double factor = dt / (dh * dh);

    DiffusionSolver2D::U_IDP = VectorXd((Nx + 1) * Ny);
    DiffusionSolver2D::U_IDP.setConstant(-1);
    U_MASK.clear();

    // ========== U COMPONENT ==========

    // First pass: number the unknowns and build the stencil weights.
    int u_dof_count = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            if (!IsU_Solvable(grid, i, j))
            {
                Set_U_IDP(i, j, -1);
                continue;
            }

            // Only unknowns get a stencil.
            Stencil2D mask;
            mask.setZero();

            // +Y : U(i+1,j), through the top edges of cells (i,j-1) and (i,j)
            switch (grid->FaceGate(MAC::U, i + 1, j))
            {
            case GateState::OPEN: mask(2, 1) = 1.0; mask(1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

            // -Y : U(i-1,j)
            switch (grid->FaceGate(MAC::U, i - 1, j))
            {
            case GateState::OPEN: mask(0, 1) = 1.0; mask(1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

            // +X : U(i,j+1) is reached through cell (i,j)
            switch (grid->CellGate(i, j))
            {
            case GateState::OPEN: mask(1, 2) = 1.0; mask(1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

            // -X : U(i,j-1) is reached through cell (i,j-1)
            switch (grid->CellGate(i, j - 1))
            {
            case GateState::OPEN: mask(1, 0) = 1.0; mask(1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

            Set_U_IDP(i, j, u_dof_count);
            u_dof_count++;
            DiffusionSolver2D::U_MASK.push_back(mask);
        }
    }

    // Second pass: count non-zeros over the same range the first pass numbered.
    int u_nnz = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            int MAT_LINE = Get_U_IDP(i, j);
            if (MAT_LINE == -1)
                continue;

            const Stencil2D &mask = GetUMask(i, j);

            if (mask(1, 1) != 0.0)
                u_nnz++;
            if (mask(2, 1) != 0.0 && Get_U_IDP(i + 1, j) != -1)
                u_nnz++;
            if (mask(0, 1) != 0.0 && Get_U_IDP(i - 1, j) != -1)
                u_nnz++;
            if (mask(1, 2) != 0.0 && Get_U_IDP(i, j + 1) != -1)
                u_nnz++;
            if (mask(1, 0) != 0.0 && Get_U_IDP(i, j - 1) != -1)
                u_nnz++;
        }
    }

    // Allocate arrays
    u_collums = (int *)malloc(sizeof(int) * u_nnz);
    u_rows = (int *)malloc(sizeof(int) * u_nnz);
    u_values = (double *)calloc(u_nnz, sizeof(double));

    // Third pass: assemble, diagonal and off-diagonal sharing one coefficient per direction.
    int c = 0;
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            const Stencil2D &mask = GetUMask(i, j);
            int MAT_LINE = Get_U_IDP(i, j);

            if (MAT_LINE == -1)
                continue;

            double nu_up = NuU_Y(i, j);
            double nu_down = NuU_Y(i - 1, j);
            double nu_right = NuU_X(i, j);
            double nu_left = NuU_X(i, j - 1);

            double diagonal = 0.0;
            if (mask(2, 1) != 0.0 || grid->FaceGate(MAC::U, i + 1, j) == GateState::WALL)
                diagonal -= factor * nu_up;
            if (mask(0, 1) != 0.0 || grid->FaceGate(MAC::U, i - 1, j) == GateState::WALL)
                diagonal -= factor * nu_down;
            if (mask(1, 2) != 0.0 || grid->CellGate(i, j) == GateState::WALL)
                diagonal -= factor * nu_right;
            if (mask(1, 0) != 0.0 || grid->CellGate(i, j - 1) == GateState::WALL)
                diagonal -= factor * nu_left;

            if (diagonal != 0.0)
            {
                u_rows[c] = MAT_LINE;
                u_collums[c] = MAT_LINE;
                u_values[c] = diagonal;
                c++;
            }

            if (mask(2, 1) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i + 1, j);
                if (MAT_COLLUM != -1)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_COLLUM;
                    u_values[c] = factor * nu_up;
                    c++;
                }
            }

            if (mask(0, 1) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i - 1, j);
                if (MAT_COLLUM != -1)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_COLLUM;
                    u_values[c] = factor * nu_down;
                    c++;
                }
            }

            if (mask(1, 2) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i, j + 1);
                if (MAT_COLLUM != -1)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_COLLUM;
                    u_values[c] = factor * nu_right;
                    c++;
                }
            }

            if (mask(1, 0) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i, j - 1);
                if (MAT_COLLUM != -1)
                {
                    u_rows[c] = MAT_LINE;
                    u_collums[c] = MAT_COLLUM;
                    u_values[c] = factor * nu_left;
                    c++;
                }
            }
        }
    }

    if (c != u_nnz)
    {
        std::cerr << "ERROR: U component NNZ mismatch! Expected " << u_nnz << " but got " << c << std::endl;
    }

    for (int i = 0; i < u_nnz; ++i)
    {
        if (u_rows[i] < 0 || u_rows[i] >= u_dof_count ||
            u_collums[i] < 0 || u_collums[i] >= u_dof_count)
        {
            std::cerr << "ERROR U: Out of bounds at entry " << i
                      << " - row=" << u_rows[i]
                      << " col=" << u_collums[i]
                      << " MatSize=" << u_dof_count << std::endl;
        }
    }

    std::vector<Eigen::Triplet<double>> triplets_u;
    triplets_u.reserve(u_nnz);
    for (int i = 0; i < u_nnz; ++i)
    {
        triplets_u.push_back(Triplet(u_rows[i], u_collums[i], u_values[i]));
    }

    SparseMatrix mat_u(u_dof_count, u_dof_count);
    mat_u.setFromTriplets(triplets_u.begin(), triplets_u.end());

    SparseMatrix identity_u(u_dof_count, u_dof_count);
    identity_u.setIdentity();
    DiffusionSolver2D::U_DIFFUSION_MATRIX_EIGEN = identity_u - mat_u;

    free(u_collums);
    free(u_rows);
    free(u_values);
}

void DiffusionSolver2D::SolveDiffusion_U_Eigen(MAC *gridAnt, MAC *gridSol)
{
    Update_U_DiffusionMatrix(gridAnt);
    int u_matrix_size = U_DIFFUSION_MATRIX_EIGEN.rows();

    // Build RHS from previous timestep U velocities
    VectorXd RHS_U = VectorXd(u_matrix_size);
    RHS_U.setZero();

    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            int id = Get_U_IDP(i, j);
            if (id == -1)
                continue;
            RHS_U(id) = gridAnt->GetU(i, j);
        }
    }

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(5000);
    solver.setTolerance(1e-12);
    solver.compute(U_DIFFUSION_MATRIX_EIGEN);

    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: U matrix decomposition failed!" << std::endl;
        return;
    }

    Eigen::VectorXd SOL_U = solver.solve(RHS_U);

    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: U solve failed!" << std::endl;
        std::cerr << "Iterations: " << solver.iterations() << std::endl;
        std::cerr << "Error: " << solver.error() << std::endl;
        return;
    }

    // Copy U solution back to grid
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            int id = Get_U_IDP(i, j);
            if (id == -1)
                continue;
            gridSol->SetU(i, j, SOL_U(id));
        }
    }
}

// The body force is deliberately NOT part of these right-hand sides.
void DiffusionSolver2D::SolveDiffusion_Eigen(MAC *gridAnt, MAC *gridSol)
{

    DiffusionSolver2D::dt = SIMULATION.dt;
    Eigen::setNbThreads(THREAD_COUNT);

            SolveDiffusion_U_Eigen(gridAnt, gridSol);

            SolveDiffusion_V_Eigen(gridAnt, gridSol);

}

// for the computaiton of the Eps valyues
// Harmonic mean of two values

void DiffusionSolver2D::SolveDiffusion_ADI(MAC *gridAnt, MAC *gridSol)
{  

    DiffusionSolver2D::dt = SIMULATION.dt;

    gridMid.CopyGrid(*gridAnt);

    SolveDiffusion_U_ADI_NEW(gridAnt, gridSol);
    SolveDiffusion_V_ADI_NEW(gridAnt, gridSol);

}

// The body force is deliberately NOT part of these right-hand sides.

void DiffusionSolver2D::SolveDiffusion_U_ADI_NEW(MAC *grid, MAC *gridSol)
{
    // One scratch per component, reused by all sweeps of this task.
    static PencilLine L;
    ResizePencilLine(L, std::max(Nx + 1, Ny + 1));

    double dh = SIMULATION.dh;
    double factor = dt / (dh * dh);

    // ---- x sweep: implicit in x, over gridMid ----
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            PencilRow &r = L.rows[j];
            r.solvable = IsU_Solvable(grid, i, j);
            if (!r.solvable) continue;
            r.lower = 0.0;
            r.upper = 0.0;
            r.diag = 1.0;
            r.rhs = grid->GetU(i, j);

            // +X : U(i,j+1) through cell (i,j)
            GateState gate_right = grid->CellGate(i, j);
            if (gate_right != GateState::FREE)
            {
                double nu = NuU_X(i, j);
                r.diag += factor * nu;
                if (gate_right == GateState::OPEN && IsU_Solvable(grid, i, j + 1))
                    r.upper = -factor * nu;
            }

            // -X : U(i,j-1) through cell (i,j-1)
            GateState gate_left = grid->CellGate(i, j - 1);
            if (gate_left != GateState::FREE)
            {
                double nu = NuU_X(i, j - 1);
                r.diag += factor * nu;
                if (gate_left == GateState::OPEN && IsU_Solvable(grid, i, j - 1))
                    r.lower = -factor * nu;
            }
        }
        SolvePencilLine(L, Nx + 1);
        for (int j = 0; j < Nx + 1; j++)
            if (L.rows[j].solvable) gridMid.SetU(i, j, L.rows[j].x);
    }

    // ---- y sweep: implicit in y, from gridMid into gridSol ----
    for (int j = 0; j < Nx + 1; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            PencilRow &r = L.rows[i];
            r.solvable = IsU_Solvable(grid, i, j);
            if (!r.solvable) continue;
            r.lower = 0.0;
            r.upper = 0.0;
            r.diag = 1.0;
            r.rhs = gridMid.GetU(i, j);

            // +Y : U(i+1,j) through the top edges of cells (i,j-1) and (i,j)
            GateState gate_up = grid->FaceGate(MAC::U, i + 1, j);
            if (gate_up != GateState::FREE)
            {
                double nu = NuU_Y(i, j);
                r.diag += factor * nu;
                if (gate_up == GateState::OPEN && IsU_Solvable(grid, i + 1, j))
                    r.upper = -factor * nu;
            }

            // -Y : U(i-1,j)
            GateState gate_down = grid->FaceGate(MAC::U, i - 1, j);
            if (gate_down != GateState::FREE)
            {
                double nu = NuU_Y(i - 1, j);
                r.diag += factor * nu;
                if (gate_down == GateState::OPEN && IsU_Solvable(grid, i - 1, j))
                    r.lower = -factor * nu;
            }
        }
        SolvePencilLine(L, Ny);
        for (int i = 0; i < Ny; i++)
            if (L.rows[i].solvable) gridSol->SetU(i, j, L.rows[i].x);
    }
}

void DiffusionSolver2D::SolveDiffusion_V_ADI_NEW(MAC *grid, MAC *gridSol)
{
    // One scratch per component, reused by all sweeps of this task.
    static PencilLine L;
    ResizePencilLine(L, std::max(Nx + 1, Ny + 1));

    double dh = SIMULATION.dh;
    double factor = dt / (dh * dh);

    // ---- x sweep: implicit in x, over gridMid ----
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            PencilRow &r = L.rows[j];
            r.solvable = IsV_Solvable(grid, i, j);
            if (!r.solvable) continue;
            r.lower = 0.0;
            r.upper = 0.0;
            r.diag = 1.0;
            r.rhs = grid->GetV(i, j);

            // +X : V(i,j+1) through the right edges of cells (i-1,j) and (i,j)
            GateState gate_right = grid->FaceGate(MAC::V, i, j + 1);
            if (gate_right != GateState::FREE)
            {
                double nu = NuV_X(i, j);
                r.diag += factor * nu;
                if (gate_right == GateState::OPEN && IsV_Solvable(grid, i, j + 1))
                    r.upper = -factor * nu;
            }

            // -X : V(i,j-1)
            GateState gate_left = grid->FaceGate(MAC::V, i, j - 1);
            if (gate_left != GateState::FREE)
            {
                double nu = NuV_X(i, j - 1);
                r.diag += factor * nu;
                if (gate_left == GateState::OPEN && IsV_Solvable(grid, i, j - 1))
                    r.lower = -factor * nu;
            }
        }
        SolvePencilLine(L, Nx);
        for (int j = 0; j < Nx; j++)
            if (L.rows[j].solvable) gridMid.SetV(i, j, L.rows[j].x);
    }

    // ---- y sweep: implicit in y, from gridMid into gridSol ----
    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny + 1; i++)
        {
            PencilRow &r = L.rows[i];
            r.solvable = IsV_Solvable(grid, i, j);
            if (!r.solvable) continue;
            r.lower = 0.0;
            r.upper = 0.0;
            r.diag = 1.0;
            r.rhs = gridMid.GetV(i, j);

            // +Y : V(i+1,j) through cell (i,j)
            GateState gate_up = grid->CellGate(i, j);
            if (gate_up != GateState::FREE)
            {
                double nu = NuV_Y(i, j);
                r.diag += factor * nu;
                if (gate_up == GateState::OPEN && IsV_Solvable(grid, i + 1, j))
                    r.upper = -factor * nu;
            }

            // -Y : V(i-1,j) through cell (i-1,j)
            GateState gate_down = grid->CellGate(i - 1, j);
            if (gate_down != GateState::FREE)
            {
                double nu = NuV_Y(i - 1, j);
                r.diag += factor * nu;
                if (gate_down == GateState::OPEN && IsV_Solvable(grid, i - 1, j))
                    r.lower = -factor * nu;
            }
        }
        SolvePencilLine(L, Ny + 1);
        for (int i = 0; i < Ny + 1; i++)
            if (L.rows[i].solvable) gridSol->SetV(i, j, L.rows[i].x);
    }
}

void DiffusionSolver2D::Dispose()
{
    U_DIFFUSION_MATRIX_EIGEN = SparseMatrix(); V_DIFFUSION_MATRIX_EIGEN = SparseMatrix();
    U_IDP = VectorXd(); V_IDP = VectorXd();
    gridMid = MAC();
}
