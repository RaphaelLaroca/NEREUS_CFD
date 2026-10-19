// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <string>
#include <ctime>
#include <utility> // for std::forward
#include <time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>

#include "Core/Definitions.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

double GetWallTime();

// The Thomas algorithm, O(n): generic, used by the ADI sweeps, the transport solver and the diffusion pencils.
// TDMA restricted to the first n entries, so one scratch sized to the longest axis serves every line.
void TDMA_VECTORIZED_N(const Eigen::VectorXd& a, Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       Eigen::VectorXd& d, Eigen::VectorXd& x, int n);

class MAC;

// Surface force on the interior solid, by integrating the stress tensor over the solid/fluid faces.
void AeroForces2D(const MAC* grid, int Nx, int Ny, double nu, double& Fd, double& Fl);

// Control-volume force (2D): the momentum balance over a box around the body, per unit density. The staircase
// never enters, and the (grad u^T).n half integrates to zero over the closed box.
struct AeroCVParts2D
{
    double pressure[2] = {0, 0}, momentum[2] = {0, 0};
    double viscGrad[2] = {0, 0}, viscTrans[2] = {0, 0}, unsteady[2] = {0, 0};
    int jL = 0, jR = 0, iB = 0, iT = 0;   // box: x faces FacePos(X, jL), FacePos(X, jR); y faces FacePos(Y, iB), (Y, iT)
    bool valid = false;
    double Deformation(int a) const { return pressure[a] + momentum[a] + viscGrad[a] + viscTrans[a] - unsteady[a]; }
};
// The box around the interior solid's bounding box, `margin` times its larger side beyond it on every side (in metres,
// so the same box at every resolution), kept two cells clear of the ring. False without an interior solid.
bool AeroCVBox2D(const MAC* grid, double margin, int& jL, int& jR, int& iB, int& iT);
// The integral over a box. `prev` is the previous step's field, for the unsteady term; nullptr for a steady evaluation.
void AeroForcesCV2D(const MAC* grid, const MAC* prev, double dt, double rho, double nu,
                    int jL, int jR, int iB, int iT, AeroCVParts2D& out);

// The 3D twin: parts per axis (x drag, y lift, z side), per unit density. A box whose z faces reach the ring (a body
// spanning the channel wall to wall in z) closes on the walls, where only the wall shear on the fluid cells counts.
struct AeroCVParts3D {
    double pressure[3] = {0, 0, 0}, momentum[3] = {0, 0, 0};
    double viscGrad[3] = {0, 0, 0}, viscTrans[3] = {0, 0, 0}, unsteady[3] = {0, 0, 0};
    int jL = 0, jR = 0, iB = 0, iT = 0, kB = 0, kT = 0;   // box faces FacePos(X, jL..jR), (Y, iB..iT), (Z, kB..kT)
    bool valid = false;
    double Deformation(int a) const { return pressure[a] + momentum[a] + viscGrad[a] + viscTrans[a] - unsteady[a]; }
};
// As AeroCVBox2D in x and y (margin times the larger of the solid's x and y sizes); in z the same, or the two walls
// when the solid reaches both of them.
bool AeroCVBox3D(const MAC* grid, double margin, int& jL, int& jR, int& iB, int& iT, int& kB, int& kT);
void AeroForcesCV3D(const MAC* grid, const MAC* prev, double dt, double rho, double nu,
                    int jL, int jR, int iB, int iT, int kB, int kT, AeroCVParts3D& out);

FlowIntegrals ComputeFlowIntegrals2D(MAC* grid);
FlowIntegrals ComputeFlowIntegrals3D(MAC* grid);

// Maximum cell Reynolds number, each staggered component against the spacing along its own axis.
CellReynolds ComputeMaxCellReynolds2D(MAC* grid, double nu);
CellReynolds ComputeMaxCellReynolds3D(MAC* grid, double nu);

// Inside the grid and not in a SOLID cell.
bool IsOpenAt(MAC* grid, double x, double y);
bool IsOpenAt(MAC* grid, double x, double y, double z);
// The velocity at a point; false (and zero) outside the grid, in a solid cell, or where it is not finite.
bool VelocityAt(MAC* grid, double x, double y, double &u, double &v);
bool VelocityAt(MAC* grid, double x, double y, double z, double &u, double &v, double &w);

// Evenly spaced streamlines (Jobard & Lefer 1997): lattice seeds, RK2 along the unit direction, a line stops
// within separation/2 of another, at a solid, at the domain edge or at a stagnation point.
void ComputeStreamlines2D(MAC* grid, const StreamlineSettings &settings, std::vector<Streamline> &out);

struct ParticleSet;
// A scene probe's value at its point, from the grid (and the particles, for `level`).
double EvaluateProbe(MAC* grid, const ParticleSet* particles, const Probe &probe);
// Seed positions (x y z triples) for the settings' source.
void SeedStreamlines3D(const MAC* grid, const Streamline3DSettings &settings, std::vector<double> &xyz);
// One line per usable seed, traced both ways with RK2 on the unit direction until a solid, the domain edge,
// a stagnation point, a reversal inside one step, or maxPoints.
void ComputeStreamlines3D(MAC* grid, const Streamline3DSettings &settings, std::vector<Streamline3D> &out);


// The 3D surface force, split by origin so the pieces can be weighed separately.
struct AeroForces3DParts
{
    double pressure[3] = {0, 0, 0};
    double normalVisc[3] = {0, 0, 0};
    double shearGrad[3] = {0, 0, 0};
    double shearTrans[3] = {0, 0, 0};
    double Gradient(int a) const { return pressure[a] + normalVisc[a] + shearGrad[a]; }
    double Deformation(int a) const
    { return pressure[a] + 2.0 * normalVisc[a] + shearGrad[a] + shearTrans[a]; }
    double Hybrid(int a) const { return pressure[a] + normalVisc[a] + shearGrad[a] + shearTrans[a]; }
    // The form the telemetry reports; see AERO_STRESS_FORM_3D.
    double Total(int a) const;
};
void AeroForces3D(const MAC* grid, int Nx, int Ny, int Nz, double nu, AeroForces3DParts& out);

inline double AeroForces3DParts::Total(int a) const
{
    switch (AERO_STRESS_FORM_3D)
    {
    case AeroStressForm::Gradient:    return Gradient(a);
    case AeroStressForm::Deformation: return Deformation(a);
    default:                          return Hybrid(a);
    }
}



// How many pixels an axis needs so the SMALLEST cell still gets at least `perCell` of them.
int ResampleResolutionFor(const std::vector<double>& faces, int perCell, int cap);

// Resample a cell-indexed image onto one that is uniform in PHYSICAL space.
void ResampleFieldToPhysical(const std::vector<double>& src, int srcRows, int srcCols,
                             const std::vector<double>& xf, const std::vector<double>& yf,
                             std::vector<double>& dst, int dstRows, int dstCols);

// ----- text helpers ----- Leading and trailing whitespace off a copy. Config and scene parsing
// both want this and both had their own lambda for it.
inline std::string TrimWhitespace(const std::string &s)
{
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// The file name without its folder or extension, by string operations only: on Windows a std::filesystem::path
// built from a narrow string throws when a byte is outside the code page (an accented folder name).
inline std::string PathStem(const std::string &p)
{
    const size_t slash = p.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? p : p.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    if (dot != std::string::npos && dot > 0) name.erase(dot);
    return name;
}

// The running executable's path; on Windows from the wide API, since argv[0] is in the ANSI code page.
std::filesystem::path ExecutablePath(const char *argv0);



#endif
