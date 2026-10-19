// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include "DiffusionSolver3D.h"
#include "FLIPSettings.h"

// Static member initialization for DiffusionSolver3D

// Grid dimensions
int DiffusionSolver3D::Nx = 0;
int DiffusionSolver3D::Ny = 0;
int DiffusionSolver3D::Nz = 0;

// Time step
double DiffusionSolver3D::dt = 0.0;

// COO format arrays for U
int *DiffusionSolver3D::u_collums = nullptr;
int *DiffusionSolver3D::u_rows = nullptr;
double *DiffusionSolver3D::u_values = nullptr;

// COO format arrays for V
int *DiffusionSolver3D::v_collums = nullptr;
int *DiffusionSolver3D::v_rows = nullptr;
double *DiffusionSolver3D::v_values = nullptr;

// COO format arrays for W
int *DiffusionSolver3D::w_collums = nullptr;
int *DiffusionSolver3D::w_rows = nullptr;
double *DiffusionSolver3D::w_values = nullptr;

// Sparse matrices
SparseMatrix DiffusionSolver3D::U_DIFFUSION_MATRIX_EIGEN;
SparseMatrix DiffusionSolver3D::V_DIFFUSION_MATRIX_EIGEN;
SparseMatrix DiffusionSolver3D::W_DIFFUSION_MATRIX_EIGEN;

// Index mapping vectors
VectorXd DiffusionSolver3D::U_IDP;
VectorXd DiffusionSolver3D::V_IDP;
VectorXd DiffusionSolver3D::W_IDP;

// Mask vectors (3x3x3 stencils for 3D)
std::vector<Stencil3D> DiffusionSolver3D::U_MASK;
std::vector<Stencil3D> DiffusionSolver3D::V_MASK;
std::vector<Stencil3D> DiffusionSolver3D::W_MASK;

MAC DiffusionSolver3D::gridMid = MAC();
MAC DiffusionSolver3D::gridMid2 = MAC();

void DiffusionSolver3D::InitializeDiffusionSolver(MAC *grid)
{
    DiffusionSolver3D::Nx = SIMULATION.Nx;
    DiffusionSolver3D::Ny = SIMULATION.Ny;
    DiffusionSolver3D::Nz = SIMULATION.Nz;
    // One mask per face, not per fluid cell - the builders push over the whole face array.
    DiffusionSolver3D::U_MASK.reserve((SIMULATION.Nx + 1) * SIMULATION.Ny * SIMULATION.Nz);
    DiffusionSolver3D::V_MASK.reserve(SIMULATION.Nx * (SIMULATION.Ny + 1) * SIMULATION.Nz);
    DiffusionSolver3D::W_MASK.reserve(SIMULATION.Nx * SIMULATION.Ny * (SIMULATION.Nz + 1));
    DiffusionSolver3D::dt = SIMULATION.dt;

    DiffusionSolver3D::U_IDP = VectorXd((Nx + 1) * Ny * Nz);
    DiffusionSolver3D::U_IDP.setConstant(-1);
    DiffusionSolver3D::V_IDP = VectorXd(Nx * (Ny + 1) * Nz);
    DiffusionSolver3D::V_IDP.setConstant(-1);
    DiffusionSolver3D::W_IDP = VectorXd(Nx * Ny * (Nz + 1));
    DiffusionSolver3D::W_IDP.setConstant(-1);

    // The three matrices are rebuilt from scratch every step by Update_*_DiffusionMatrix,
    // so there is nothing to prebuild here.
    gridMid.InitializeFromGrid(*grid, SIMULATION.time);
    gridMid2.InitializeFromGrid(*grid, SIMULATION.time);
}

// Geometry of the staggered faces this solver works on.

static inline bool IsU_Boundary(int i, int j, int k)
{
    return (i <= 0 || i >= SIMULATION.Ny - 1 ||
            j <= 1 || j >= SIMULATION.Nx - 1 ||
            k <= 0 || k >= SIMULATION.Nz - 1);
}

static inline bool IsV_Boundary(int i, int j, int k)
{
    return (i <= 1 || i >= SIMULATION.Ny - 1 ||
            j <= 0 || j >= SIMULATION.Nx - 1 ||
            k <= 0 || k >= SIMULATION.Nz - 1);
}

static inline bool IsW_Boundary(int i, int j, int k)
{
    return (i <= 0 || i >= SIMULATION.Ny - 1 ||
            j <= 0 || j >= SIMULATION.Nx - 1 ||
            k <= 1 || k >= SIMULATION.Nz - 1);
}

// A face is an unknown only where the projection will also touch it.
static inline bool IsU_Solvable(MAC *g, int i, int j, int k)
{
    if (IsU_Boundary(i, j, k)) return false;
    return g->IsFluidFace(MAC::U, i, j, k);
}

static inline bool IsV_Solvable(MAC *g, int i, int j, int k)
{
    if (IsV_Boundary(i, j, k)) return false;
    return g->IsFluidFace(MAC::V, i, j, k);
}

static inline bool IsW_Solvable(MAC *g, int i, int j, int k)
{
    if (IsW_Boundary(i, j, k)) return false;
    return g->IsFluidFace(MAC::W, i, j, k);
}

// Neighbour gating. A face reaches its neighbour through the cell (along its own normal) or the
// pair of cells (along the other two axes) that separate the two control volumes.

void DiffusionSolver3D::Update_U_DiffusionMatrix(MAC *grid)
{
    double dh = SIMULATION.dh;

    DiffusionSolver3D::U_IDP = VectorXd((Nx + 1) * (Ny) * (Nz));
    DiffusionSolver3D::U_IDP.setConstant(-1);
    U_MASK.clear();

    // First pass: number the unknowns and build the stencil weights.
    int u_dof_count = 0;
    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx + 1; j++)
    for (int k = 0; k < Nz; k++)
    {
        if (!IsU_Solvable(grid, i, j, k))
        {
            Set_U_IDP(i, j, k, -1);
            continue;
        }

        // Only unknowns get a stencil.
        Stencil3D mask;
        mask.setZero();

            switch (grid->CellGate(i, j, k))
            {
            case GateState::OPEN: mask(1, 2, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->CellGate(i, j - 1, k))
            {
            case GateState::OPEN: mask(1, 0, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::U, i + 1, j, k))
            {
            case GateState::OPEN: mask(2, 1, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::U, i - 1, j, k))
            {
            case GateState::OPEN: mask(0, 1, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::U, i, j, k + 1))
            {
            case GateState::OPEN: mask(1, 1, 2) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::U, i, j, k - 1))
            {
            case GateState::OPEN: mask(1, 1, 0) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

        Set_U_IDP(i, j, k, u_dof_count);
        u_dof_count++;
        DiffusionSolver3D::U_MASK.push_back(mask);
    }

    // Second pass: count non-zeros over exactly the range the first pass numbered, or
    // unknowns end up with no equation and the matrix picks up empty rows.
    int u_nnz = 0;
    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx + 1; j++)
    for (int k = 0; k < Nz; k++)
    {
        if (Get_U_IDP(i, j, k) == -1)
            continue;
        const Stencil3D &mask = GetUMask(i, j, k);
        // The diagonal's test guards only the diagonal; each off-diagonal tests itself, as in the assembly pass,
        // which keeps u_nnz and the entries written in step.
        if (mask(1, 1, 1) != 0.0)
            u_nnz++;
        if (mask(1, 2, 1) != 0.0 && Get_U_IDP(i, j + 1, k) != -1)
            u_nnz++;
        if (mask(1, 0, 1) != 0.0 && Get_U_IDP(i, j - 1, k) != -1)
            u_nnz++;
        if (mask(2, 1, 1) != 0.0 && Get_U_IDP(i + 1, j, k) != -1)
            u_nnz++;
        if (mask(0, 1, 1) != 0.0 && Get_U_IDP(i - 1, j, k) != -1)
            u_nnz++;
        if (mask(1, 1, 2) != 0.0 && Get_U_IDP(i, j, k + 1) != -1)
            u_nnz++;
        if (mask(1, 1, 0) != 0.0 && Get_U_IDP(i, j, k - 1) != -1)
            u_nnz++;
    }

    u_collums = (int *)malloc(sizeof(int) * u_nnz);
    u_rows = (int *)malloc(sizeof(int) * u_nnz);
    u_values = (double *)calloc(u_nnz, sizeof(double));

    // Third pass: assemble.
    int cc = 0;
    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx + 1; j++)
    for (int k = 0; k < Nz; k++)
    {
        int MAT_LINE = Get_U_IDP(i, j, k);
        if (MAT_LINE == -1)
            continue;
        const Stencil3D &mask = GetUMask(i, j, k);

        if (mask(1, 1, 1) != 0.0)
        {
            u_rows[cc] = MAT_LINE;
            u_collums[cc] = MAT_LINE;
            u_values[cc] = mask(1, 1, 1);
            cc++;
        }

            if (mask(1, 2, 1) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i, j + 1, k);
                if (MAT_COLLUM != -1)
                {
                    u_rows[cc] = MAT_LINE;
                    u_collums[cc] = MAT_COLLUM;
                    u_values[cc] = mask(1, 2, 1);
                    cc++;
                }
            }
            if (mask(1, 0, 1) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i, j - 1, k);
                if (MAT_COLLUM != -1)
                {
                    u_rows[cc] = MAT_LINE;
                    u_collums[cc] = MAT_COLLUM;
                    u_values[cc] = mask(1, 0, 1);
                    cc++;
                }
            }
            if (mask(2, 1, 1) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i + 1, j, k);
                if (MAT_COLLUM != -1)
                {
                    u_rows[cc] = MAT_LINE;
                    u_collums[cc] = MAT_COLLUM;
                    u_values[cc] = mask(2, 1, 1);
                    cc++;
                }
            }
            if (mask(0, 1, 1) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i - 1, j, k);
                if (MAT_COLLUM != -1)
                {
                    u_rows[cc] = MAT_LINE;
                    u_collums[cc] = MAT_COLLUM;
                    u_values[cc] = mask(0, 1, 1);
                    cc++;
                }
            }
            if (mask(1, 1, 2) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i, j, k + 1);
                if (MAT_COLLUM != -1)
                {
                    u_rows[cc] = MAT_LINE;
                    u_collums[cc] = MAT_COLLUM;
                    u_values[cc] = mask(1, 1, 2);
                    cc++;
                }
            }
            if (mask(1, 1, 0) != 0.0)
            {
                int MAT_COLLUM = Get_U_IDP(i, j, k - 1);
                if (MAT_COLLUM != -1)
                {
                    u_rows[cc] = MAT_LINE;
                    u_collums[cc] = MAT_COLLUM;
                    u_values[cc] = mask(1, 1, 0);
                    cc++;
                }
            }
    }

    if (cc != u_nnz)
        std::cerr << "ERROR: U component NNZ mismatch! Expected " << u_nnz
                  << " but got " << cc << std::endl;

    std::vector<Eigen::Triplet<double>> triplets_u;
    triplets_u.reserve(cc);
    for (int t = 0; t < cc; ++t)
        triplets_u.push_back(Triplet(u_rows[t], u_collums[t], u_values[t]));

    SparseMatrix mat_u(u_dof_count, u_dof_count);
    mat_u.setFromTriplets(triplets_u.begin(), triplets_u.end());

    SparseMatrix identity_u(u_dof_count, u_dof_count);
    identity_u.setIdentity();
    DiffusionSolver3D::U_DIFFUSION_MATRIX_EIGEN =
        identity_u - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_u;

    free(u_collums);
    free(u_rows);
    free(u_values);
}

void DiffusionSolver3D::Update_V_DiffusionMatrix(MAC *grid)
{
    double dh = SIMULATION.dh;

    DiffusionSolver3D::V_IDP = VectorXd((Nx) * (Ny + 1) * (Nz));
    DiffusionSolver3D::V_IDP.setConstant(-1);
    V_MASK.clear();

    // First pass: number the unknowns and build the stencil weights.
    int v_dof_count = 0;
    for (int i = 0; i < Ny + 1; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz; k++)
    {
        if (!IsV_Solvable(grid, i, j, k))
        {
            Set_V_IDP(i, j, k, -1);
            continue;
        }

        // Only unknowns get a stencil.
        Stencil3D mask;
        mask.setZero();

            switch (grid->FaceGate(MAC::V, i, j + 1, k))
            {
            case GateState::OPEN: mask(1, 2, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::V, i, j - 1, k))
            {
            case GateState::OPEN: mask(1, 0, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->CellGate(i, j, k))
            {
            case GateState::OPEN: mask(2, 1, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->CellGate(i - 1, j, k))
            {
            case GateState::OPEN: mask(0, 1, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::V, i, j, k + 1))
            {
            case GateState::OPEN: mask(1, 1, 2) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::V, i, j, k - 1))
            {
            case GateState::OPEN: mask(1, 1, 0) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

        Set_V_IDP(i, j, k, v_dof_count);
        v_dof_count++;
        DiffusionSolver3D::V_MASK.push_back(mask);
    }

    // Second pass: count non-zeros over exactly the range the first pass numbered, or
    // unknowns end up with no equation and the matrix picks up empty rows.
    int v_nnz = 0;
    for (int i = 0; i < Ny + 1; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz; k++)
    {
        if (Get_V_IDP(i, j, k) == -1)
            continue;
        const Stencil3D &mask = GetVMask(i, j, k);
        // The diagonal's test guards only the diagonal; each off-diagonal tests itself, as in the assembly pass,
        // which keeps v_nnz and the entries written in step.
        if (mask(1, 1, 1) != 0.0)
            v_nnz++;
        if (mask(1, 2, 1) != 0.0 && Get_V_IDP(i, j + 1, k) != -1)
            v_nnz++;
        if (mask(1, 0, 1) != 0.0 && Get_V_IDP(i, j - 1, k) != -1)
            v_nnz++;
        if (mask(2, 1, 1) != 0.0 && Get_V_IDP(i + 1, j, k) != -1)
            v_nnz++;
        if (mask(0, 1, 1) != 0.0 && Get_V_IDP(i - 1, j, k) != -1)
            v_nnz++;
        if (mask(1, 1, 2) != 0.0 && Get_V_IDP(i, j, k + 1) != -1)
            v_nnz++;
        if (mask(1, 1, 0) != 0.0 && Get_V_IDP(i, j, k - 1) != -1)
            v_nnz++;
    }

    v_collums = (int *)malloc(sizeof(int) * v_nnz);
    v_rows = (int *)malloc(sizeof(int) * v_nnz);
    v_values = (double *)calloc(v_nnz, sizeof(double));

    // Third pass: assemble.
    int cc = 0;
    for (int i = 0; i < Ny + 1; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz; k++)
    {
        int MAT_LINE = Get_V_IDP(i, j, k);
        if (MAT_LINE == -1)
            continue;
        const Stencil3D &mask = GetVMask(i, j, k);

        if (mask(1, 1, 1) != 0.0)
        {
            v_rows[cc] = MAT_LINE;
            v_collums[cc] = MAT_LINE;
            v_values[cc] = mask(1, 1, 1);
            cc++;
        }

            if (mask(1, 2, 1) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i, j + 1, k);
                if (MAT_COLLUM != -1)
                {
                    v_rows[cc] = MAT_LINE;
                    v_collums[cc] = MAT_COLLUM;
                    v_values[cc] = mask(1, 2, 1);
                    cc++;
                }
            }
            if (mask(1, 0, 1) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i, j - 1, k);
                if (MAT_COLLUM != -1)
                {
                    v_rows[cc] = MAT_LINE;
                    v_collums[cc] = MAT_COLLUM;
                    v_values[cc] = mask(1, 0, 1);
                    cc++;
                }
            }
            if (mask(2, 1, 1) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i + 1, j, k);
                if (MAT_COLLUM != -1)
                {
                    v_rows[cc] = MAT_LINE;
                    v_collums[cc] = MAT_COLLUM;
                    v_values[cc] = mask(2, 1, 1);
                    cc++;
                }
            }
            if (mask(0, 1, 1) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i - 1, j, k);
                if (MAT_COLLUM != -1)
                {
                    v_rows[cc] = MAT_LINE;
                    v_collums[cc] = MAT_COLLUM;
                    v_values[cc] = mask(0, 1, 1);
                    cc++;
                }
            }
            if (mask(1, 1, 2) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i, j, k + 1);
                if (MAT_COLLUM != -1)
                {
                    v_rows[cc] = MAT_LINE;
                    v_collums[cc] = MAT_COLLUM;
                    v_values[cc] = mask(1, 1, 2);
                    cc++;
                }
            }
            if (mask(1, 1, 0) != 0.0)
            {
                int MAT_COLLUM = Get_V_IDP(i, j, k - 1);
                if (MAT_COLLUM != -1)
                {
                    v_rows[cc] = MAT_LINE;
                    v_collums[cc] = MAT_COLLUM;
                    v_values[cc] = mask(1, 1, 0);
                    cc++;
                }
            }
    }

    if (cc != v_nnz)
        std::cerr << "ERROR: V component NNZ mismatch! Expected " << v_nnz
                  << " but got " << cc << std::endl;

    std::vector<Eigen::Triplet<double>> triplets_v;
    triplets_v.reserve(cc);
    for (int t = 0; t < cc; ++t)
        triplets_v.push_back(Triplet(v_rows[t], v_collums[t], v_values[t]));

    SparseMatrix mat_v(v_dof_count, v_dof_count);
    mat_v.setFromTriplets(triplets_v.begin(), triplets_v.end());

    SparseMatrix identity_v(v_dof_count, v_dof_count);
    identity_v.setIdentity();
    DiffusionSolver3D::V_DIFFUSION_MATRIX_EIGEN =
        identity_v - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_v;

    free(v_collums);
    free(v_rows);
    free(v_values);
}

void DiffusionSolver3D::Update_W_DiffusionMatrix(MAC *grid)
{
    double dh = SIMULATION.dh;

    DiffusionSolver3D::W_IDP = VectorXd((Nx) * (Ny) * (Nz + 1));
    DiffusionSolver3D::W_IDP.setConstant(-1);
    W_MASK.clear();

    // First pass: number the unknowns and build the stencil weights.
    int w_dof_count = 0;
    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz + 1; k++)
    {
        if (!IsW_Solvable(grid, i, j, k))
        {
            Set_W_IDP(i, j, k, -1);
            continue;
        }

        // Only unknowns get a stencil.
        Stencil3D mask;
        mask.setZero();

            switch (grid->FaceGate(MAC::W, i, j + 1, k))
            {
            case GateState::OPEN: mask(1, 2, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::W, i, j - 1, k))
            {
            case GateState::OPEN: mask(1, 0, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::W, i + 1, j, k))
            {
            case GateState::OPEN: mask(2, 1, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->FaceGate(MAC::W, i - 1, j, k))
            {
            case GateState::OPEN: mask(0, 1, 1) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->CellGate(i, j, k))
            {
            case GateState::OPEN: mask(1, 1, 2) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }
            switch (grid->CellGate(i, j, k - 1))
            {
            case GateState::OPEN: mask(1, 1, 0) = 1.0; mask(1, 1, 1) -= 1.0; break;
            case GateState::WALL: mask(1, 1, 1) -= 1.0; break;
            case GateState::FREE: break;
            }

        Set_W_IDP(i, j, k, w_dof_count);
        w_dof_count++;
        DiffusionSolver3D::W_MASK.push_back(mask);
    }

    // Second pass: count non-zeros over exactly the range the first pass numbered, or
    // unknowns end up with no equation and the matrix picks up empty rows.
    int w_nnz = 0;
    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz + 1; k++)
    {
        if (Get_W_IDP(i, j, k) == -1)
            continue;
        const Stencil3D &mask = GetWMask(i, j, k);
        // The diagonal's test guards only the diagonal; each off-diagonal tests itself, as in the assembly pass,
        // which keeps w_nnz and the entries written in step.
        if (mask(1, 1, 1) != 0.0)
            w_nnz++;
        if (mask(1, 2, 1) != 0.0 && Get_W_IDP(i, j + 1, k) != -1)
            w_nnz++;
        if (mask(1, 0, 1) != 0.0 && Get_W_IDP(i, j - 1, k) != -1)
            w_nnz++;
        if (mask(2, 1, 1) != 0.0 && Get_W_IDP(i + 1, j, k) != -1)
            w_nnz++;
        if (mask(0, 1, 1) != 0.0 && Get_W_IDP(i - 1, j, k) != -1)
            w_nnz++;
        if (mask(1, 1, 2) != 0.0 && Get_W_IDP(i, j, k + 1) != -1)
            w_nnz++;
        if (mask(1, 1, 0) != 0.0 && Get_W_IDP(i, j, k - 1) != -1)
            w_nnz++;
    }

    w_collums = (int *)malloc(sizeof(int) * w_nnz);
    w_rows = (int *)malloc(sizeof(int) * w_nnz);
    w_values = (double *)calloc(w_nnz, sizeof(double));

    // Third pass: assemble.
    int cc = 0;
    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz + 1; k++)
    {
        int MAT_LINE = Get_W_IDP(i, j, k);
        if (MAT_LINE == -1)
            continue;
        const Stencil3D &mask = GetWMask(i, j, k);

        if (mask(1, 1, 1) != 0.0)
        {
            w_rows[cc] = MAT_LINE;
            w_collums[cc] = MAT_LINE;
            w_values[cc] = mask(1, 1, 1);
            cc++;
        }

            if (mask(1, 2, 1) != 0.0)
            {
                int MAT_COLLUM = Get_W_IDP(i, j + 1, k);
                if (MAT_COLLUM != -1)
                {
                    w_rows[cc] = MAT_LINE;
                    w_collums[cc] = MAT_COLLUM;
                    w_values[cc] = mask(1, 2, 1);
                    cc++;
                }
            }
            if (mask(1, 0, 1) != 0.0)
            {
                int MAT_COLLUM = Get_W_IDP(i, j - 1, k);
                if (MAT_COLLUM != -1)
                {
                    w_rows[cc] = MAT_LINE;
                    w_collums[cc] = MAT_COLLUM;
                    w_values[cc] = mask(1, 0, 1);
                    cc++;
                }
            }
            if (mask(2, 1, 1) != 0.0)
            {
                int MAT_COLLUM = Get_W_IDP(i + 1, j, k);
                if (MAT_COLLUM != -1)
                {
                    w_rows[cc] = MAT_LINE;
                    w_collums[cc] = MAT_COLLUM;
                    w_values[cc] = mask(2, 1, 1);
                    cc++;
                }
            }
            if (mask(0, 1, 1) != 0.0)
            {
                int MAT_COLLUM = Get_W_IDP(i - 1, j, k);
                if (MAT_COLLUM != -1)
                {
                    w_rows[cc] = MAT_LINE;
                    w_collums[cc] = MAT_COLLUM;
                    w_values[cc] = mask(0, 1, 1);
                    cc++;
                }
            }
            if (mask(1, 1, 2) != 0.0)
            {
                int MAT_COLLUM = Get_W_IDP(i, j, k + 1);
                if (MAT_COLLUM != -1)
                {
                    w_rows[cc] = MAT_LINE;
                    w_collums[cc] = MAT_COLLUM;
                    w_values[cc] = mask(1, 1, 2);
                    cc++;
                }
            }
            if (mask(1, 1, 0) != 0.0)
            {
                int MAT_COLLUM = Get_W_IDP(i, j, k - 1);
                if (MAT_COLLUM != -1)
                {
                    w_rows[cc] = MAT_LINE;
                    w_collums[cc] = MAT_COLLUM;
                    w_values[cc] = mask(1, 1, 0);
                    cc++;
                }
            }
    }

    if (cc != w_nnz)
        std::cerr << "ERROR: W component NNZ mismatch! Expected " << w_nnz
                  << " but got " << cc << std::endl;

    std::vector<Eigen::Triplet<double>> triplets_w;
    triplets_w.reserve(cc);
    for (int t = 0; t < cc; ++t)
        triplets_w.push_back(Triplet(w_rows[t], w_collums[t], w_values[t]));

    SparseMatrix mat_w(w_dof_count, w_dof_count);
    mat_w.setFromTriplets(triplets_w.begin(), triplets_w.end());

    SparseMatrix identity_w(w_dof_count, w_dof_count);
    identity_w.setIdentity();
    DiffusionSolver3D::W_DIFFUSION_MATRIX_EIGEN =
        identity_w - ((dt / (dh * dh)) * (SIMULATION.EPS)) * mat_w;

    free(w_collums);
    free(w_rows);
    free(w_values);
}

void DiffusionSolver3D::SolveDiffusion_U_Eigen(MAC *gridAnt, MAC *gridSol)
{
    DiffusionSolver3D::Update_U_DiffusionMatrix(gridAnt);

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-6);

    int matrix_size = U_DIFFUSION_MATRIX_EIGEN.rows();
    VectorXd RHS = VectorXd(matrix_size);
    RHS.setZero();

    // The body force is deliberately absent here - see the note above SolveDiffusion_ADI.
    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx + 1; j++)
    for (int k = 0; k < Nz; k++)
    {
        int id = Get_U_IDP(i, j, k);
        if (id == -1)
            continue;
        RHS(id) = gridAnt->GetU(i, j, k);
    }

    solver.compute(U_DIFFUSION_MATRIX_EIGEN);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: U matrix decomposition failed!" << std::endl;
        return;
    }

    Eigen::VectorXd SOL = solver.solve(RHS);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: U solve failed! iterations=" << solver.iterations()
                  << " error=" << solver.error() << std::endl;
        return;
    }

    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx + 1; j++)
    for (int k = 0; k < Nz; k++)
    {
        int id = Get_U_IDP(i, j, k);
        if (id == -1)
            continue;
        gridSol->SetU(i, j, k, SOL(id));
    }
}

void DiffusionSolver3D::SolveDiffusion_V_Eigen(MAC *gridAnt, MAC *gridSol)
{
    DiffusionSolver3D::Update_V_DiffusionMatrix(gridAnt);

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-6);

    int matrix_size = V_DIFFUSION_MATRIX_EIGEN.rows();
    VectorXd RHS = VectorXd(matrix_size);
    RHS.setZero();

    // The body force is deliberately absent here - see the note above SolveDiffusion_ADI.
    for (int i = 0; i < Ny + 1; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz; k++)
    {
        int id = Get_V_IDP(i, j, k);
        if (id == -1)
            continue;
        RHS(id) = gridAnt->GetV(i, j, k);
    }

    solver.compute(V_DIFFUSION_MATRIX_EIGEN);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: V matrix decomposition failed!" << std::endl;
        return;
    }

    Eigen::VectorXd SOL = solver.solve(RHS);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: V solve failed! iterations=" << solver.iterations()
                  << " error=" << solver.error() << std::endl;
        return;
    }

    for (int i = 0; i < Ny + 1; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz; k++)
    {
        int id = Get_V_IDP(i, j, k);
        if (id == -1)
            continue;
        gridSol->SetV(i, j, k, SOL(id));
    }
}

void DiffusionSolver3D::SolveDiffusion_W_Eigen(MAC *gridAnt, MAC *gridSol)
{
    DiffusionSolver3D::Update_W_DiffusionMatrix(gridAnt);

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
    solver.setMaxIterations(2000);
    solver.setTolerance(1e-6);

    int matrix_size = W_DIFFUSION_MATRIX_EIGEN.rows();
    VectorXd RHS = VectorXd(matrix_size);
    RHS.setZero();

    // The body force is deliberately absent here - see the note above SolveDiffusion_ADI.
    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz + 1; k++)
    {
        int id = Get_W_IDP(i, j, k);
        if (id == -1)
            continue;
        RHS(id) = gridAnt->GetW(i, j, k);
    }

    solver.compute(W_DIFFUSION_MATRIX_EIGEN);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: W matrix decomposition failed!" << std::endl;
        return;
    }

    Eigen::VectorXd SOL = solver.solve(RHS);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: W solve failed! iterations=" << solver.iterations()
                  << " error=" << solver.error() << std::endl;
        return;
    }

    for (int i = 0; i < Ny; i++)
    for (int j = 0; j < Nx; j++)
    for (int k = 0; k < Nz + 1; k++)
    {
        int id = Get_W_IDP(i, j, k);
        if (id == -1)
            continue;
        gridSol->SetW(i, j, k, SOL(id));
    }
}

void DiffusionSolver3D::SolveDiffusion_Eigen(MAC *gridAnt, MAC *gridSol)
{
    DiffusionSolver3D::dt = SIMULATION.dt;
    Eigen::setNbThreads(THREAD_COUNT);

#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp task
            SolveDiffusion_U_Eigen(gridAnt, gridSol);
#pragma omp task
            SolveDiffusion_V_Eigen(gridAnt, gridSol);
#pragma omp task
            SolveDiffusion_W_Eigen(gridAnt, gridSol);
        }
    }

}

// The body force is deliberately NOT part of any right-hand side here.
void DiffusionSolver3D::SolveDiffusion_ADI(MAC *gridAnt, MAC *gridSol)
{
    DiffusionSolver3D::dt = SIMULATION.dt;

    gridMid.CopyGrid(*gridAnt);
    gridMid2.CopyGrid(*gridAnt);

    #pragma omp parallel
    {
        #pragma omp single
        {

            #pragma omp task
            SolveDiffusion_U_ADI(gridAnt, gridSol);
            #pragma omp task
            SolveDiffusion_V_ADI(gridAnt, gridSol);
            #pragma omp task
            SolveDiffusion_W_ADI(gridAnt, gridSol);

        }
    }

}

void DiffusionSolver3D::SolveDiffusion_U_ADI(MAC *grid, MAC *gridSol)
{
    double dh = SIMULATION.dh;
    double factor = (dt / (dh * dh)) * SIMULATION.EPS;

    // One scratch per component, reused by all three sweeps of this task.
    static PencilLine L;
    ResizePencilLine(L, std::max(Nx + 1, std::max(Ny, Nz)));

    // ---- sweep implicit in x ----
    for (int i = 0; i < Ny; i++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int j = 0; j < Nx + 1; j++)
            {
                PencilRow &r = L.rows[j];
                r.solvable = IsU_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = grid->GetU(i, j, k);

                GateState gplus = grid->CellGate(i, j, k);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsU_Solvable(grid, i, j + 1, k))
                        r.upper = -factor;
                }

                GateState gminus = grid->CellGate(i, j - 1, k);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsU_Solvable(grid, i, j - 1, k))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Nx + 1);
            for (int j = 0; j < Nx + 1; j++)
                if (L.rows[j].solvable) gridMid.SetU(i, j, k, L.rows[j].x);
        }
    }

    // ---- sweep implicit in y ----
    for (int j = 0; j < Nx + 1; j++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int i = 0; i < Ny; i++)
            {
                PencilRow &r = L.rows[i];
                r.solvable = IsU_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = gridMid.GetU(i, j, k);

                GateState gplus = grid->FaceGate(MAC::U, i + 1, j, k);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsU_Solvable(grid, i + 1, j, k))
                        r.upper = -factor;
                }

                GateState gminus = grid->FaceGate(MAC::U, i - 1, j, k);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsU_Solvable(grid, i - 1, j, k))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Ny);
            for (int i = 0; i < Ny; i++)
                if (L.rows[i].solvable) gridMid2.SetU(i, j, k, L.rows[i].x);
        }
    }

    // ---- sweep implicit in z ----
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx + 1; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                PencilRow &r = L.rows[k];
                r.solvable = IsU_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = gridMid2.GetU(i, j, k);

                GateState gplus = grid->FaceGate(MAC::U, i, j, k + 1);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsU_Solvable(grid, i, j, k + 1))
                        r.upper = -factor;
                }

                GateState gminus = grid->FaceGate(MAC::U, i, j, k - 1);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsU_Solvable(grid, i, j, k - 1))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Nz);
            for (int k = 0; k < Nz; k++)
                if (L.rows[k].solvable) gridSol->SetU(i, j, k, L.rows[k].x);
        }
    }
}

void DiffusionSolver3D::SolveDiffusion_V_ADI(MAC *grid, MAC *gridSol)
{
    // One scratch per component, reused by all sweeps of this task.
    static PencilLine L;
    ResizePencilLine(L, std::max({Nx + 1, Ny + 1, Nz + 1}));

    double dh = SIMULATION.dh;
    double factor = (dt / (dh * dh)) * SIMULATION.EPS;

    // ---- sweep implicit in x ----
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int j = 0; j < Nx; j++)
            {
                PencilRow &r = L.rows[j];
                r.solvable = IsV_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = grid->GetV(i, j, k);

                GateState gplus = grid->FaceGate(MAC::V, i, j + 1, k);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsV_Solvable(grid, i, j + 1, k))
                        r.upper = -factor;
                }

                GateState gminus = grid->FaceGate(MAC::V, i, j - 1, k);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsV_Solvable(grid, i, j - 1, k))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Nx);
            for (int j = 0; j < Nx; j++)
                if (L.rows[j].solvable) gridMid.SetV(i, j, k, L.rows[j].x);
        }
    }

    // ---- sweep implicit in y ----
    for (int j = 0; j < Nx; j++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int i = 0; i < Ny + 1; i++)
            {
                PencilRow &r = L.rows[i];
                r.solvable = IsV_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = gridMid.GetV(i, j, k);

                GateState gplus = grid->CellGate(i, j, k);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsV_Solvable(grid, i + 1, j, k))
                        r.upper = -factor;
                }

                GateState gminus = grid->CellGate(i - 1, j, k);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsV_Solvable(grid, i - 1, j, k))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Ny + 1);
            for (int i = 0; i < Ny + 1; i++)
                if (L.rows[i].solvable) gridMid2.SetV(i, j, k, L.rows[i].x);
        }
    }

    // ---- sweep implicit in z ----
    for (int i = 0; i < Ny + 1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {
                PencilRow &r = L.rows[k];
                r.solvable = IsV_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = gridMid2.GetV(i, j, k);

                GateState gplus = grid->FaceGate(MAC::V, i, j, k + 1);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsV_Solvable(grid, i, j, k + 1))
                        r.upper = -factor;
                }

                GateState gminus = grid->FaceGate(MAC::V, i, j, k - 1);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsV_Solvable(grid, i, j, k - 1))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Nz);
            for (int k = 0; k < Nz; k++)
                if (L.rows[k].solvable) gridSol->SetV(i, j, k, L.rows[k].x);
        }
    }
}

void DiffusionSolver3D::SolveDiffusion_W_ADI(MAC *grid, MAC *gridSol)
{
    // One scratch per component, reused by all sweeps of this task.
    static PencilLine L;
    ResizePencilLine(L, std::max({Nx + 1, Ny + 1, Nz + 1}));

    double dh = SIMULATION.dh;
    double factor = (dt / (dh * dh)) * SIMULATION.EPS;

    // ---- sweep implicit in x ----
    for (int i = 0; i < Ny; i++)
    {
        for (int k = 0; k < Nz + 1; k++)
        {
            for (int j = 0; j < Nx; j++)
            {
                PencilRow &r = L.rows[j];
                r.solvable = IsW_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = grid->GetW(i, j, k);

                GateState gplus = grid->FaceGate(MAC::W, i, j + 1, k);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsW_Solvable(grid, i, j + 1, k))
                        r.upper = -factor;
                }

                GateState gminus = grid->FaceGate(MAC::W, i, j - 1, k);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsW_Solvable(grid, i, j - 1, k))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Nx);
            for (int j = 0; j < Nx; j++)
                if (L.rows[j].solvable) gridMid.SetW(i, j, k, L.rows[j].x);
        }
    }

    // ---- sweep implicit in y ----
    for (int j = 0; j < Nx; j++)
    {
        for (int k = 0; k < Nz + 1; k++)
        {
            for (int i = 0; i < Ny; i++)
            {
                PencilRow &r = L.rows[i];
                r.solvable = IsW_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = gridMid.GetW(i, j, k);

                GateState gplus = grid->FaceGate(MAC::W, i + 1, j, k);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsW_Solvable(grid, i + 1, j, k))
                        r.upper = -factor;
                }

                GateState gminus = grid->FaceGate(MAC::W, i - 1, j, k);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsW_Solvable(grid, i - 1, j, k))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Ny);
            for (int i = 0; i < Ny; i++)
                if (L.rows[i].solvable) gridMid2.SetW(i, j, k, L.rows[i].x);
        }
    }

    // ---- sweep implicit in z ----
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz + 1; k++)
            {
                PencilRow &r = L.rows[k];
                r.solvable = IsW_Solvable(grid, i, j, k);
                if (!r.solvable) continue;
                r.lower = 0.0;
                r.upper = 0.0;
                r.diag = 1.0;
                r.rhs = gridMid2.GetW(i, j, k);

                GateState gplus = grid->CellGate(i, j, k);
                if (gplus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gplus == GateState::OPEN && IsW_Solvable(grid, i, j, k + 1))
                        r.upper = -factor;
                }

                GateState gminus = grid->CellGate(i, j, k - 1);
                if (gminus != GateState::FREE)
                {
                    r.diag += factor;
                    if (gminus == GateState::OPEN && IsW_Solvable(grid, i, j, k - 1))
                        r.lower = -factor;
                }
            }
            SolvePencilLine(L, Nz + 1);
            for (int k = 0; k < Nz + 1; k++)
                if (L.rows[k].solvable) gridSol->SetW(i, j, k, L.rows[k].x);
        }
    }
}

void DiffusionSolver3D::Dispose()
{
    U_DIFFUSION_MATRIX_EIGEN = SparseMatrix(); V_DIFFUSION_MATRIX_EIGEN = SparseMatrix();
    W_DIFFUSION_MATRIX_EIGEN = SparseMatrix();
    U_IDP = VectorXd(); V_IDP = VectorXd(); W_IDP = VectorXd();
    gridMid = MAC(); gridMid2 = MAC();
}
