// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// this will solve the diffusion for u, v and w components it works very similarly to the pressure
// solver.
#include "../../headers/Core/MAC.h"
#include "../../headers/Core/Definitions.h"
#include "../../headers/Utils.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef DIFFUSION_SOLVER3D_H
#define DIFFUSION_SOLVER3D_H

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Sparse;
using Eigen::ConjugateGradient;
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::BiCGSTAB<SparseMatrix> BiCGSTABSolver;

// A face's 3x3x3 Laplacian weights, (i, j, k offset) + 1.
struct Stencil3D {
    double w[27];
    void setZero() { for (double &x : w) x = 0.0; }
    double &operator()(int a, int b, int c) { return w[9 * a + 3 * b + c]; }
    double operator()(int a, int b, int c) const { return w[9 * a + 3 * b + c]; }
};

class DiffusionSolver3D{
private:
    static int Nx, Ny, Nz;
    static int NON_ZERO;
    static double dt;

    static int* u_collums;
    static int* u_rows;
    static double* u_values;

    static int* v_collums;
    static int* v_rows;
    static double* v_values;

    static int* w_collums;
    static int* w_rows;
    static double* w_values;

    // and the CSR representation
    static SparseMatrix U_DIFFUSION_MATRIX_EIGEN;
    static SparseMatrix V_DIFFUSION_MATRIX_EIGEN;
    static SparseMatrix W_DIFFUSION_MATRIX_EIGEN;



    // indexing grid we use that mirrors the mac, for ease of calculation
    static VectorXd U_IDP;
    static VectorXd V_IDP;
    static VectorXd W_IDP;

    static std::vector<Stencil3D> U_MASK; //(Nx+1) * Ny * Nz
    static std::vector<Stencil3D> V_MASK; //Nx * (Ny+1) * Nz
    static std::vector<Stencil3D> W_MASK; //Nx * Ny * (Nz+1)

    static Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>* solver;

public:
    static void InitializeDiffusionSolver(MAC* grid); 
    static void Dispose();   // frees what Initialize allocated
                                                  // we have dirichlet = 0 at solid walls and grad =
                                                  // 0 at empty boundaries we do need to be extra.
    static void SolveDiffusion_Eigen(MAC* gridAnt, MAC* gridSol);
        static void SolveDiffusion_ADI(MAC *gridAnt, MAC *gridSol);

private:
    static void Update_U_DiffusionMatrix(MAC* grid);           
    static void Update_V_DiffusionMatrix(MAC* grid);      
    static void Update_W_DiffusionMatrix(MAC* grid);  

    static void SolveDiffusion_U_Eigen(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_V_Eigen(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_W_Eigen(MAC* gridAnt,MAC* gridSol);

    static MAC gridMid;
    static MAC gridMid2;

    static void SolveDiffusion_U_ADI(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_V_ADI(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_W_ADI(MAC* gridAnt,MAC* gridSol);

    // Indexed by degree of freedom: the mask vector only holds unknowns, so this
    // must not be called for a face whose IDP is -1. Both assembly passes skip those.
    static inline const Stencil3D &GetUMask(int i ,int j, int k){
        return U_MASK[Get_U_IDP(i, j, k)];
    }

    static inline int Get_U_IDP(int i, int j, int k){ 
        return U_IDP(i * ((Nx+1)*Nz) + (j*Nz) + k);
    }

    static inline void Set_U_IDP(int i, int j, int k, int value){
        U_IDP(i * ((Nx+1)*Nz) + (j*Nz) + k) = value;
    }

    // V component accessors: Nx * (Ny+1) * Nz Linearization: i * ((Nx)*Nz) + (j*Nz) + k Indexed by
    // degree of freedom: the mask vector only holds unknowns.
    static inline const Stencil3D &GetVMask(int i ,int j, int k){
        return V_MASK[Get_V_IDP(i, j, k)];
    }

    static inline int Get_V_IDP(int i, int j, int k){ 
        return V_IDP(i * ((Nx)*Nz)  + (j*Nz)  + k );
    }

    static inline void Set_V_IDP(int i, int j, int k, int value){
        V_IDP(i * ((Nx)*Nz)  + (j*Nz)  + k ) = value;
    }

    // W component accessors: Nx * Ny * (Nz+1) Linearization: i * ((Nx)*(Nz+1)) + (j*(Nz+1)) + k
    // Indexed by degree of freedom: the mask vector only holds unknowns.
    static inline const Stencil3D &GetWMask(int i ,int j, int k){
        return W_MASK[Get_W_IDP(i, j, k)];
    }

    static inline int Get_W_IDP(int i, int j, int k){ 
        return W_IDP(i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k );
    }

    static inline void Set_W_IDP(int i, int j, int k, int value){
        W_IDP(i * ((Nx)*(Nz+1))  + (j*(Nz+1))  + k ) = value;
    }
};

#endif