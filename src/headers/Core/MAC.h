// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Definitions.h"
#include "../Utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <filesystem>

#include <vector>
#include <string>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#ifndef MAC_H
#define MAC_H

// All (sensible) functions are overloaded to 2D and 3D, those who are not (no parameters) do the
// check themselves

// How a face reaches a neighbour across a cell interface. OPEN is fluid-fluid, WALL fluid-solid (or the contrary, solid-fluid)
// FREE is empty-fluid, meaning pressure is zero and velocity is neumman homogeneous.
enum class GateState { OPEN, WALL, FREE };
struct GridRefinement;   // Definitions.h, which includes this file


class MAC
{
public:
    // One axis's interior cells from its refinement intervals. Public so the setup screen builds the SAME axis.
    static double FactorAt(const std::vector<GridRefinement> &refs, double x);
    static void BuildAxis(double a0, double af, int nBase, double dh, const std::vector<GridRefinement> &refs, bool nonUniform, std::vector<double> &size, std::vector<double> &face, std::vector<double> &centre);
    // The staggered fields and the axes. Plain enums, so they index the per-component and per-axis tables directly;
    // a velocity component and its own normal axis share a number (U along X, V along Y, W along Z).
    enum Component { U = 0, V = 1, W = 2, P = 3 };
    enum Axis { X = 0, Y = 1, Z = 2 };
    static inline Axis AxisOf(Component c) { return (Axis)c; }             // a velocity component's own axis
    static inline Component ComponentAlong(Axis a) { return (Component)a; } // the velocity component along an axis

    int Nx;
    int Ny;
    int Nz;

    Domain omega;
    double dh;
    bool is2D;

    // Rectilinear (non-uniform) spacing Tensor product: dx varies with j only, dy with i only, dz
    // with k only.
    bool nonUniform = false;

    const std::vector<double> &Faces(Axis axis) const { return axis == X ? xfv : (axis == Y ? yfv : zfv); }
    const std::vector<double> &Centres(Axis axis) const { return axis == X ? xcv : (axis == Y ? ycv : zcv); }


    // The spacing tables, per axis a (X with index j, Y with i, Z with k). One 2D cell (i, j), x across, y up:
    //
    //   FacePos(Y, i+1) - - +-----------------------+
    //                       |                       |
    //                       |           o <--------------- (CentrePos(X, j), CentrePos(Y, i)), the cell centre
    //                       |                       |      Width(Y, i) = FacePos(Y, i+1) - FacePos(Y, i)
    //   FacePos(Y, i)   - - +-----------------------+
    //                       |<---- Width(X, j) ---->|
    //                   FacePos(X, j)        FacePos(X, j+1)
    //
    //   ... --o-------------|-----------o-----------|----------o-- ...   along x
    //    CentrePos(X, j-1)  FacePos(X, j)  CentrePos(X, j)
    //         |<-- FaceGap(X, j-1) --->|
    //
    // FaceGap(a, n) is the centre-to-centre distance across the face between cells n and n+1, (Width(n) + Width(n+1))/2:
    // the d_f of the pressure coefficient A_f/d_f. U(i, j) sits at (FacePos(X, j), CentrePos(Y, i)), V(i, j) at
    // (CentrePos(X, j), FacePos(Y, i)), P at the centre. On a uniform grid every Width and FaceGap is exactly dh.
    inline double Width(Axis a, int n) const { return widthTable[a][n + 1]; }
    inline double FaceGap(Axis a, int n) const { return gapTable[a][n + 1]; }
    inline double FacePos(Axis a, int n) const { return faceTable[a][n + 1]; }
    inline double CentrePos(Axis a, int n) const { return centreTable[a][n + 1]; }





    inline double CellVolume(int i, int j, int k) const { return cellVolume[i * (Nx * Nz) + j * Nz + k]; }
    inline double CellVolume(int i, int j) const { return cellVolume[i * Nx + j]; }

    // Area of cell (i, j, k)'s faces normal to axis a; in 2D a length.
    inline double FaceArea(Axis a, int i, int j, int k) const
    { return a == X ? Width(Y, i) * Width(Z, k) : (a == Y ? Width(X, j) * Width(Z, k) : Width(X, j) * Width(Y, i)); }
    inline double FaceArea(Axis a, int i, int j) const { return a == X ? Width(Y, i) : Width(X, j); }

    // Builds the tables. Called by InitializeGrid once Nx/Ny/Nz and dh are known.
    void BuildSpacing();

    MAC();
    // Whole cells of size dh in the domain's length L; a length that is a whole number of cells up to round-off counts
    // as one. The grid is these plus the border ring, one cell outside each face.
    static int InteriorCells(double L, double dh) { return (int)(L / dh + 1e-9); }

    void InitializeGrid(Domain omega,bool is2d = false);

    void SetSolidMask(int(*SolidMaskFunction)(int,int,int));

    // Zeroes u, v, p (and w in 3D).
    void ClearFlowFields()
    {
        u.setZero();
        v.setZero();
        p.setZero();
        if (!is2D) w.setZero();
    }

    // ---- boundary conditions: forcing the ring's values (prescribed, or copied from inside), by each face's
    // BoundaryKind; the writes are recorded once and replayed. Velocity and pressure are enforced separately. ----
    void SetBoundaryFunctions(Vec3 (*velocity)(double, double, double, double), double (*pressure)(double, double, double, double))
    {
        boundaryVelocity3D = velocity;
        boundaryPressure3D = pressure;
    }
    void SetBoundaryFunctions(Vec2 (*velocity)(double, double, double), double (*pressure)(double, double, double))
    {
        boundaryVelocity2D = velocity;
        boundaryPressure2D = pressure;
    }
    void EnforceVelocityBC(double t);
    void EnforcePressureBC(double t);
    void EnforceBoundaryConditions(double t)
    {
        EnforceVelocityBC(t);
        EnforcePressureBC(t);
    }
    // The factor on a tangential diffusion link toward side s: FaceGap / (Width/2) (2 when uniform) when the neighbour
    // face lies inside a no-slip or moving wall (the reflection 2 u_wall - u), 1 otherwise. Call after EnforceVelocityBC.
    inline double WallLinkFactor(Component c, Axis a, int i, int j, int k, int s) const
    {
        const int ni = i + (a == Y ? s : 0), nj = j + (a == X ? s : 0), nk = k + (a == Z ? s : 0);
        if (!wallGhost[c][FieldIndex(c, ni, nj, nk)]) return 1.0;
        const int n = a == X ? j : (a == Y ? i : k);
        return FaceGap(a, s < 0 ? n - 1 : n) / (0.5 * Width(a, n));
    }
    inline double WallLinkFactor(Component c, Axis a, int i, int j, int s) const { return WallLinkFactor(c, a, i, j, 0, s); }

    double GetDivergencyAt(int i,int j,int k);
    // The pressure gradient on the axis-a face of cell (i, j, k), between the two centres either side, over their gap:
    // the same d_f the pressure matrix uses, so G = -D^T. Not on a boundary face.
    inline double GetGradP(Axis a, int i, int j, int k)
    {
        if (a == X) return (GetP(i, j, k) - GetP(i, j - 1, k)) / FaceGap(X, j - 1);
        if (a == Y) return (GetP(i, j, k) - GetP(i - 1, j, k)) / FaceGap(Y, i - 1);
        return (GetP(i, j, k) - GetP(i, j, k - 1)) / FaceGap(Z, k - 1);
    }

    double MaxAbsoluteDifference(MAC& grid);

    double GetMaxVelocity();
    // Is every stored value a number? GetMaxVelocity cannot tell: a NaN loses every std::max comparison.
    bool AllFinite() const
    {
        return u.allFinite() && v.allFinite() && w.allFinite() && p.allFinite();
    }
    int GetFluidCellCount();
    // The cell types, one byte per cell in GetSolid's order: what the pressure solve compares to decide whether its
    // matrix still stands.
    const std::vector<unsigned char> &CellTypes() const { return cellTypes; }

    double GetDivSum();

    // Everything a grid carries: CopyFlowFields + CopyScalarFields.
    void CopyGrid(MAC& grid)
    {
        CopyFlowFields(grid);
        CopyScalarFields(grid);
    }
    void CopyFlowFields(MAC& grid);    // u, v, (w), p, solid mask
    void CopyScalarFields(MAC& grid);  // registered scalars, matched by handle and name



    // Body force f(x, y, z, t) -- the scene's `force` expression PLUS the config's scalar Fu/Fv/Fw,
    // added as `f*dt` to every fluid face.
    void AddForce(double time, double dt);

    // Seed the velocity field from the scene's `initial` expression, at t = 0. A no-op with no
    // `initial` line, so every existing case is untouched.
    void ApplyInitialField();
    // Seeds this grid from the scene: the `initial` field, the scene's boundary functions, the boundary applied at t.
    void InitializeFromScene(double t = 0.0);
    // A method's internal grid: model's size, cell types, flow fields and boundary functions, the boundary applied at t.
    void InitializeFromGrid(MAC &model, double t = 0.0);
    // Any grid method's start: InitializeFromScene on both of the run's grids. Call it from the method's initialiser.
    static void InitializeSimulationGrids();
    void ResetFluidCells();


    // Component c interpolated to the face (i, j[, k]) of component `target`, distance weighted on a stretched
    // grid. getXatY are its one-line forms: getVatU is V at a U face.
    double ComponentAtFace(Component c, Component target, int i, int j, int k);
    double ComponentAtFace(Component c, Component target, int i, int j)
    {
        return ComponentAtFace(c, target, i, j, 0);
    }
    inline double getVatU(int i, int j, int k) { return ComponentAtFace(V, U, i, j, k); }
    inline double getWatU(int i, int j, int k) { return ComponentAtFace(W, U, i, j, k); }
    inline double getUatV(int i, int j, int k) { return ComponentAtFace(U, V, i, j, k); }
    inline double getWatV(int i, int j, int k) { return ComponentAtFace(W, V, i, j, k); }
    inline double getUatW(int i, int j, int k) { return ComponentAtFace(U, W, i, j, k); }
    inline double getVatW(int i, int j, int k) { return ComponentAtFace(V, W, i, j, k); }
    inline double getVatU(int i, int j) { return ComponentAtFace(V, U, i, j); }
    inline double getUatV(int i, int j) { return ComponentAtFace(U, V, i, j); }




    // ---- point sampling: bilinear (2D) / trilinear (3D) at a physical point, constant beyond the first and
    // last centre. Cell types are not looked at (the probes in Utils do that). ----
    double ComponentAt(Component c, double x, double y);
    double ComponentAt(Component c, double x, double y, double z);
    // A registered scalar, by handle or by name, sampled as P is. 0 for a scalar the grid does not carry.
    double ScalarAt(int handle, double x, double y);
    double ScalarAt(int handle, double x, double y, double z);
    double ScalarAt(const std::string &name, double x, double y) { return ScalarAt(ScalarHandle(name), x, y); }
    double ScalarAt(const std::string &name, double x, double y, double z) { return ScalarAt(ScalarHandle(name), x, y, z); }
    // The cell holding x along an axis, clamped to the grid.
    int CellAt(Axis axis, double x) const { return Locate(faceTable[axis], x, AxisCells(axis)); }

    // ---- fields: U, V, W on their faces, P at the centres, linearised (i, j, k) = (y, x, z). The named accessors
    // index directly and GetComponent dispatches to them (the other way round, the 3D projection ran 1.9x slower). ----
    inline VectorXd &Field(Component c) { return c == U ? u : (c == V ? v : (c == W ? w : p)); }
    inline const VectorXd &Field(Component c) const { return c == U ? u : (c == V ? v : (c == W ? w : p)); }

    //lineariezed acess to the arrays, do not ask how many nights i've spent debugging this!
    inline double GetU(int i, int j, int k) { return u[(i * (Nx + 1) + j) * Nz + k]; }
    inline double GetV(int i, int j, int k) { return v[(i * Nx + j) * Nz + k]; }
    inline double GetW(int i, int j, int k) { return w[(i * Nx + j) * (Nz + 1) + k]; }
    inline double GetP(int i, int j, int k) { return p[(i * Nx + j) * Nz + k]; }
    inline void SetU(int i, int j, int k, double value) { u[(i * (Nx + 1) + j) * Nz + k] = value; }
    inline void SetV(int i, int j, int k, double value) { v[(i * Nx + j) * Nz + k] = value; }
    inline void SetW(int i, int j, int k, double value) { w[(i * Nx + j) * (Nz + 1) + k] = value; }
    inline void SetP(int i, int j, int k, double value) { p[(i * Nx + j) * Nz + k] = value; }

    inline double GetComponent(Component c, int i, int j, int k)
    { return c == U ? GetU(i, j, k) : (c == V ? GetV(i, j, k) : (c == W ? GetW(i, j, k) : GetP(i, j, k))); }
    inline void SetComponent(Component c, int i, int j, int k, double value)
    {
        if (c == U) SetU(i, j, k, value);
        else if (c == V) SetV(i, j, k, value);
        else if (c == W) SetW(i, j, k, value);
        else SetP(i, j, k, value);
    }

    inline int GetSolid(int i, int j, int k) const { return cellTypes[i * (Nx * Nz) + j * Nz + k]; }
    // A change that can move the boundary lists -- a cell becoming or ceasing to be solid, or any change on the ring --
    // marks them stale; FLIP's interior FLUID/EMPTY flips do not.
    inline void SetSolid(int i, int j, int k, int value)
    {
        unsigned char &t = cellTypes[i * (Nx * Nz) + j * Nz + k];
        if (t != value && (t == SOLID_CELL || value == SOLID_CELL ||
                           i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1 || k == 0 || k == Nz - 1))
            boundaryListsStale = true;
        t = (unsigned char)value;
    }

    // ---- cells and faces ----
    // The gate through one cell: a wall if solid, open if fluid, free (air, inflow, pressure) otherwise.
    inline GateState CellGate(int i, int j, int k) const { return GateFromTypes(GetSolid(i, j, k), FLUID_CELL); }
    // The gate through face (i, j, k) of component c, i.e. through the two cells either side of it: a wall if either is
    // solid, open if both are fluid, free otherwise.
    inline GateState FaceGate(Component c, int i, int j, int k) const
    {
        return GateFromTypes(GetSolid(i, j, k), c == U ? GetSolid(i, j - 1, k) : (c == V ? GetSolid(i - 1, j, k) : GetSolid(i, j, k - 1)));
    }
    // Face (i, j, k) of component c touches no solid and at least one fluid cell: a face the flow is solved on.
    inline bool IsFluidFace(Component c, int i, int j, int k) const
    {
        const int a = GetSolid(i, j, k);
        const int b = c == U ? GetSolid(i, j - 1, k) : (c == V ? GetSolid(i - 1, j, k) : GetSolid(i, j, k - 1));
        return a != SOLID_CELL && b != SOLID_CELL && (a == FLUID_CELL || b == FLUID_CELL);
    }
    // A fluid cell off the border ring.
    inline bool IsInteriorFluid(int i, int j, int k) const
    {
        return i >= 1 && i <= Ny - 2 && j >= 1 && j <= Nx - 2 && k >= 1 && k <= Nz - 2 && GetSolid(i, j, k) == FLUID_CELL;
    }
    // Cell (i, j, k) has cell type `type` (FLUID_CELL, SOLID_CELL, EMPTY_CELL, INFLOW_CELL, PRESSURE_CELL,
    // OUTFLOW_CELL). The shorthands below name the common questions.
    inline bool IsCellType(int i, int j, int k, int type) const { return GetSolid(i, j, k) == type; }
    // The boundary kind of a cell (a BoundaryKind): its face's kind on the ring, BC_NOSLIP for an interior solid,
    // -1 for any other interior cell.
    int BoundaryKindAt(int i, int j, int k) const;
    inline bool IsFluid(int i, int j, int k) const { return IsCellType(i, j, k, FLUID_CELL); }
    // Any wall: a solid ring cell or an obstacle, whatever its kind.
    inline bool IsSolidWall(int i, int j, int k) const { return IsCellType(i, j, k, SOLID_CELL); }
    inline bool IsNoSlipWall(int i, int j, int k) const { return IsSolidWall(i, j, k) && BoundaryKindAt(i, j, k) == BC_NOSLIP; }
    inline bool IsSlipWall(int i, int j, int k) const { return IsSolidWall(i, j, k) && BoundaryKindAt(i, j, k) == BC_SLIP; }
    inline bool IsMovingWall(int i, int j, int k) const { return IsSolidWall(i, j, k) && BoundaryKindAt(i, j, k) == BC_MOVING; }
    inline bool IsInlet(int i, int j, int k) const { return IsCellType(i, j, k, INFLOW_CELL); }
    inline bool IsOutflow(int i, int j, int k) const { return IsCellType(i, j, k, OUTFLOW_CELL); }
    // A cell whose pressure is known (Dirichlet): a pressure-outlet ring cell, p = 0 or prescribed. Under FLIP an air
    // cell (EMPTY_CELL) answers true as well, since the free surface is p = 0.
    inline bool IsPressureOutlet(int i, int j, int k) const
    {
        return IsCellType(i, j, k, EMPTY_CELL) || IsCellType(i, j, k, PRESSURE_CELL);
    }

    // ---- the scalar registry: idempotent by name, so both grids can be registered from one place ----
    int RegisterScalar(const std::string &name, TransportScheme scheme = TransportScheme::Upwind)
    {
        ScalarTransport t;
        t.scheme = scheme;
        return RegisterScalar(name, t);
    }
    int RegisterScalar(const std::string &name, const ScalarTransport &transport);
    const ScalarTransport &ScalarTransportOf(int h) const { return scalarTransport[(size_t)h]; }
    // Scalar h's ring: the scene's value on a face that pins it, a copy of the cell inside elsewhere, at time t.
    void EnforceScalarBC(int h, double t);
    // Scalar h's scene source S(x, y, z, t), 0 when it has none. Serial only: the expression is a tinyexpr.
    double ScalarSourceAt(int h, double x, double y, double z, double t) const;
    // -1 when the grid has no such field
    int ScalarHandle(const std::string &name) const
    {
        for (size_t h = 0; h < scalarNames.size(); h++)
            if (scalarNames[h] == name) return (int)h;
        return -1;
    }

    // Which handle on `src` holds this grid's field `h`.
    int MatchScalar(MAC &src, int h);
    static long scalarOrderMismatches;    // handle/name disagreements, counted for the harness
    static long scalarSchemeConflicts;    // one name registered with two different transports
    int ScalarCount() const { return (int)scalars.size(); }
    const std::string &ScalarName(int h) const { return scalarNames[(size_t)h]; }
    // Scalar h's whole array, cell centred and linearised as p: for bulk readers and writers (Checkpoint).
    VectorXd &ScalarField(int h) { return scalars[(size_t)h]; }
    // Cell centred, so exactly the size of `p`, and zeroed.
    // (re)size every registered field
    void AllocateScalars()
    {
        const long n = is2D ? ((long)Nx * Ny) : ((long)Nx * Ny * Nz);
        if (n <= 0) return;
        for (VectorXd &f : scalars)
            if (f.size() != n) { f = VectorXd(n); f.setZero(); }
    }
    void ClearScalars() { scalars.clear(); scalarNames.clear(); scalarTransport.clear(); }

    inline double GetS(int h,int i,int j,int k) const { return this->scalars[(size_t)h][i * ((this->Nx)*(this->Nz))  + (j*this->Nz)  + k ]; }
    inline void SetS(int h,int i,int j,int k,double value){ this->scalars[(size_t)h][i * ((this->Nx)*(this->Nz))  + (j*this->Nz)  + k ] = value; }
    inline double GetS(int h,int i,int j) const { return this->scalars[(size_t)h][i * (this->Nx) + j]; }
    inline void SetS(int h,int i,int j,double value){ this->scalars[(size_t)h][i * (this->Nx) + j] = value; }

    // ================== 2D OVERLOADS ===============================

    void InitializeGrid(Domain2D omega);

    void SetSolidMask(int(*SolidMaskFunction)(int,int));


    double GetDivergencyAt(int i,int j);

    //we will change this in the future versions for a generic getGradComponent function, with shortcuts such as this!
    inline double GetGradP(Axis a, int i, int j)
    {
        if (a == X) return (GetP(i, j) - GetP(i, j - 1)) / FaceGap(X, j - 1);
        return (GetP(i, j) - GetP(i - 1, j)) / FaceGap(Y, i - 1);
    }

    inline double GetU(int i, int j) { return u[i * (Nx + 1) + j]; }
    inline double GetV(int i, int j) { return v[i * Nx + j]; }
    inline double GetP(int i, int j) { return p[i * Nx + j]; }
    inline void SetU(int i, int j, double value) { u[i * (Nx + 1) + j] = value; }
    inline void SetV(int i, int j, double value) { v[i * Nx + j] = value; }
    inline void SetP(int i, int j, double value) { p[i * Nx + j] = value; }

    inline double GetComponent(Component c, int i, int j) { return c == U ? GetU(i, j) : (c == V ? GetV(i, j) : GetP(i, j)); }
    inline void SetComponent(Component c, int i, int j, double value)
    {
        if (c == U) SetU(i, j, value);
        else if (c == V) SetV(i, j, value);
        else SetP(i, j, value);
    }

    inline int GetSolid(int i, int j) const { return cellTypes[i * Nx + j]; }
    inline void SetSolid(int i, int j, int value)
    {
        unsigned char &t = cellTypes[i * Nx + j];
        if (t != value && (t == SOLID_CELL || value == SOLID_CELL || i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1))
            boundaryListsStale = true;
        t = (unsigned char)value;
    }

    inline GateState CellGate(int i, int j) const { return GateFromTypes(GetSolid(i, j), FLUID_CELL); }
    inline GateState FaceGate(Component c, int i, int j) const
    {
        return GateFromTypes(GetSolid(i, j), c == U ? GetSolid(i, j - 1) : GetSolid(i - 1, j));
    }
    inline bool IsFluidFace(Component c, int i, int j) const
    {
        const int a = GetSolid(i, j);
        const int b = c == U ? GetSolid(i, j - 1) : GetSolid(i - 1, j);
        return a != SOLID_CELL && b != SOLID_CELL && (a == FLUID_CELL || b == FLUID_CELL);
    }
    inline bool IsInteriorFluid(int i, int j) const
    {
        return i >= 1 && i <= Ny - 2 && j >= 1 && j <= Nx - 2 && GetSolid(i, j) == FLUID_CELL;
    }
    inline bool IsCellType(int i, int j, int type) const { return GetSolid(i, j) == type; }
    int BoundaryKindAt(int i, int j) const { return BoundaryKindAt(i, j, 0); }
    inline bool IsFluid(int i, int j) const { return IsCellType(i, j, FLUID_CELL); }
    inline bool IsSolidWall(int i, int j) const { return IsCellType(i, j, SOLID_CELL); }
    inline bool IsNoSlipWall(int i, int j) const { return IsSolidWall(i, j) && BoundaryKindAt(i, j) == BC_NOSLIP; }
    inline bool IsSlipWall(int i, int j) const { return IsSolidWall(i, j) && BoundaryKindAt(i, j) == BC_SLIP; }
    inline bool IsMovingWall(int i, int j) const { return IsSolidWall(i, j) && BoundaryKindAt(i, j) == BC_MOVING; }
    inline bool IsInlet(int i, int j) const { return IsCellType(i, j, INFLOW_CELL); }
    inline bool IsOutflow(int i, int j) const { return IsCellType(i, j, OUTFLOW_CELL); }
    inline bool IsPressureOutlet(int i, int j) const
    {
        return IsCellType(i, j, EMPTY_CELL) || IsCellType(i, j, PRESSURE_CELL);
    }

private:
    // ---- the fields ----
    VectorXd u;
    VectorXd v;
    VectorXd w;
    VectorXd p; 

    // ---- registered cell-centred scalars ------------------------------------ Smoke density,
    // temperature, k, omega, a tracer: one equation serves all of them.
    std::vector<VectorXd> scalars;
    std::vector<std::string> scalarNames;
    std::vector<ScalarTransport> scalarTransport;   // scheme, form, epsilon, gamma per field

    std::vector<unsigned char> cellTypes;   // FLUID_CELL, SOLID_CELL, EMPTY_CELL, INFLOW_CELL, PRESSURE_CELL, OUTFLOW_CELL


    // ---- the spacing tables ----
    std::vector<double> dxs, dys, dzs; // cell sizes, indexed by j / i / k
    std::vector<double> xfv, yfv, zfv; // face coordinates, N+1 of them
    std::vector<double> xcv, ycv, zcv; // cell centres, N of them
    // Lookup tables per axis, padded by one ghost entry at each end (entry n at [n + 1]), so the accessors
    // index them with no clamping branch.
    std::vector<double> widthTable[3];    // cell sizes,          n = -1 .. N
    std::vector<double> faceTable[3];     // face coordinates,    n = -1 .. N + 1
    std::vector<double> centreTable[3];   // centre coordinates,  n = -1 .. N
    std::vector<double> gapTable[3];      // centre-to-centre across the face between cells n and n+1, n = -1 .. N - 1
    std::vector<double> cellVolume;       // dx*dy(*dz) per cell, linearised as p

    struct PrescribedEntry { Component c; int index; double x, y, z; };   // c: a Component
    struct CopyEntry { Component c; int dst, src; };
    struct RingEntry { int dst, kind, src, i, j, k; };          // kind: 0 zero, 1 prescribed, 2 copy
    struct FluxFace { Component c; int index, outer; double area, sign; };   // sign: +1 on a high face
    std::vector<PrescribedEntry> prescribedList;
    std::vector<CopyEntry> copyList;
    std::vector<int> solidFaceList[3];
    std::vector<RingEntry> ringList;
    std::vector<FluxFace> inletFaces, outflowFaces;   // the outflow's flux balances the inlets'
    std::vector<signed char> ringKind, ringOwner;     // per cell, -1 off the ring; the owner is a face 0..5
    std::vector<unsigned char> wallGhost[3];          // per face of U, V, W: inside a no-slip or moving wall
    bool boundaryListsStale = true;
    Vec3 (*boundaryVelocity3D)(double, double, double, double) = nullptr;
    double (*boundaryPressure3D)(double, double, double, double) = nullptr;
    Vec2 (*boundaryVelocity2D)(double, double, double) = nullptr;
    double (*boundaryPressure2D)(double, double, double) = nullptr;
    int FieldIndex(Component c, int i, int j, int k) const
    {
        return j * strideTable[c][X] + i * strideTable[c][Y] + k * strideTable[c][Z];
    }
    void RecordPrescribed(Component c, int i, int j, int k, double x, double y, double z)
    {
        prescribedList.push_back({c, FieldIndex(c, i, j, k), x, y, z});
    }
    void RecordCopy(Component c, int i, int j, int k, int si, int sj, int sk)
    {
        copyList.push_back({c, FieldIndex(c, i, j, k), FieldIndex(c, si, sj, sk)});
    }
    void RecordSolidFace(Component c, int i, int j, int k)
    {
        solidFaceList[c].push_back(FieldIndex(c, i, j, k));
    }
    void RecordRing(int i, int j, int k, int si, int sj, int sk)
    {
        const int t = is2D ? GetSolid(i, j) : GetSolid(i, j, k);
        const int kind = (t == EMPTY_CELL) ? 0 : (t == PRESSURE_CELL ? 1 : 2);
        ringList.push_back({FieldIndex(P, i, j, k), kind, FieldIndex(P, si, sj, sk), i, j, k});
    }
    void BuildBoundaryLists();
    void BuildRingKinds();
    int RingKind(int i, int j, int k, int &face) const;
    void BuildPrescribedList3D();
    void BuildPrescribedList2D();
    void BuildCopyList();
    void RecordCellCopies(int i, int j, int k);
    void BuildFluxFaces();
    void ApplyOutflowFlux();
    double PrescribedPressure(int i, int j, int k, double t);
    void BuildSolidFaceList3D();
    void BuildSolidFaceList2D();
    void BuildRingList();
    void BuildWallGhosts();
    static inline GateState GateFromTypes(int a, int b)
    {
        if (a == SOLID_CELL || b == SOLID_CELL) return GateState::WALL;
        if (a == FLUID_CELL && b == FLUID_CELL) return GateState::OPEN;
        return GateState::FREE;
    }
    int strideTable[4][3] = {{0}};   // field c's stride along axis x, y, z (0 along z in 2D)
    // The interval m of a padded axis table holding x: the last m in [0, n-1] with table entry m <= x (0 below).
    int Locate(const std::vector<double> &padded, double x, int n) const;
    int AxisCells(Axis axis) const { return axis == X ? Nx : (axis == Y ? Ny : Nz); }
    // The interpolation behind ComponentAt and ScalarAt: field F with strides st, over faces along axis `own` and cell
    // centres along the others (own = -1: centres on every axis, as for P and the scalars).
    double Sample(const double *F, const int *st, int own, double x, double y);
    double Sample(const double *F, const int *st, int own, double x, double y, double z);
    // ---- helpers of the spacing tables and the point sampler ----
    static void AddRing(std::vector<double> &size, std::vector<double> &face, std::vector<double> &centre);
    // A table padded by one ghost entry at each end, the edge value repeated: padded[n + 1] is entry n.
    static void PadTable(const std::vector<double> &src, std::vector<double> &padded)
    {
        const int n = (int)src.size();
        padded.resize(n + 2);
        for (int m = -1; m <= n; m++)
            padded[m + 1] = src[m < 0 ? 0 : (m >= n ? n - 1 : m)];
    }
    static void GapTable(const std::vector<double> &w, std::vector<double> &gap);
    void BuildStrides();
};

#endif
