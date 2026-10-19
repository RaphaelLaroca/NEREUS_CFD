// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// this will solve the diffusion for u and v components it works very similarly to the pressure
// solver.

// this implements a default solver to test the diffusion
// after this, there will be a proper ADI implementation, but the update mask should be reusable

#include "../../headers/Core/MAC.h"
#include "../../headers/Core/Definitions.h"
#include "../../headers/Utils.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>

#ifndef DIFFUSION_SOLVER2D_H
#define DIFFUSION_SOLVER2D_H
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Sparse;
using Eigen::ConjugateGradient;

typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::BiCGSTAB<SparseMatrix> BiCGSTABSolver;

// A face's 3x3 Laplacian weights, (row offset, column offset) + 1.
struct Stencil2D {
    double w[9];
    void setZero() { for (double &x : w) x = 0.0; }
    double &operator()(int a, int b) { return w[3 * a + b]; }
    double operator()(int a, int b) const { return w[3 * a + b]; }
};

class DiffusionSolver2D{

    private:
    static int Nx,Ny;

    static int NON_ZERO;

    static double dt;

    static int* u_collums;
    static int* u_rows;
    static double* u_values;

    static int* v_collums;
    static int* v_rows;
    static double* v_values;

    // and the CSR representation
    static SparseMatrix U_DIFFUSION_MATRIX_EIGEN;
    static SparseMatrix V_DIFFUSION_MATRIX_EIGEN;



    // indexing grid we use that mirrors the mac, for ease of calculation
    static VectorXd U_IDP;
    static VectorXd V_IDP;

    static std::vector<Stencil2D> U_MASK;//Nx+1 Ny
    static std::vector<Stencil2D> V_MASK; //Nx Ny+1

    static Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>* solver;

public:
    static void InitializeDiffusionSolver(MAC* grid); 
    static void Dispose();   // frees what Initialize allocated

    static void SolveDiffusion_Eigen(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_ADI(MAC* gridAnt,MAC* gridSol);


private:

    static void Update_U_DiffusionMatrix(MAC* grid);           
    static void Update_V_DiffusionMatrix(MAC* grid);      
    static void SolveDiffusion_U_Eigen(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_V_Eigen(MAC* gridAnt,MAC* gridSol);

    static MAC gridMid;

    static void SolveDiffusion_U_ADI_NEW(MAC* gridAnt,MAC* gridSol);
    static void SolveDiffusion_V_ADI_NEW(MAC* gridAnt,MAC* gridSol);

    // Indexed by degree of freedom: the mask vector only holds unknowns, so this
    // must not be called for a face whose IDP is -1. Both assembly passes skip those.
    static inline const Stencil2D &GetUMask(int i, int j){
        return U_MASK[Get_U_IDP(i, j)];
    }
    static inline int Get_U_IDP(int i,int j){ return U_IDP(i * ((Nx+1))  + (j)   );};
    static inline void Set_U_IDP(int i,int j,int value){U_IDP(i * ((Nx+1))  + (j)) = value;};

    // Indexed by degree of freedom: the mask vector only holds unknowns, so this
    // must not be called for a face whose IDP is -1. Both assembly passes skip those.
    static inline const Stencil2D &GetVMask(int i, int j){
        return V_MASK[Get_V_IDP(i, j)];
    }
    static inline int Get_V_IDP(int i,int j){ return V_IDP(i * ((Nx))  + (j)   );};
    static inline void Set_V_IDP(int i,int j,int value){V_IDP(i * ((Nx))  + (j)) = value;};
};

#endif
