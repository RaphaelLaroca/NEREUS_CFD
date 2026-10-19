// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include "headers/Core/MAC.h"
#include <cctype>
#include <cstdio>
// Scene.h deliberately does not include MAC.h (the graph would close on itself), so pulling
// it in from this .cpp is safe and is what AddForce needs for the `force` expression.
#include "headers/Core/Scene.h"

MAC::MAC()
{
}

// Rectilinear spacing tables. The uniform branch fills the tables with the identical expressions
// the solvers used to write inline -- x0 + j*dh for a face, x0 + j*dh + dh/2 for a centre.
double MAC::FactorAt(const std::vector<GridRefinement> &refs, double x)
{
    double f = 1.0;
    for (const GridRefinement &r : refs)
    {
        if (!r.active || r.factor <= 1) continue;
        if (x < r.start || x > r.end) continue;
        double local = (double)r.factor;
        const double span = r.end - r.start;
        if (span > 0.0 && r.mode != REFINE_HARD)
        {
            const double t = (x - r.start) / span;
            // RAMP: 0 -> 1 across the interval, RAMP_BACK the mirror.
            // SYMMETRIC: 0 at both edges, 1 at the centre.
            const double sN = (r.mode == REFINE_RAMP)      ? t
                            : (r.mode == REFINE_RAMP_BACK) ? (1.0 - t)
                                                           : (1.0 - std::fabs(2.0 * t - 1.0));
            local = 1.0 + (r.factor - 1.0) * sN;
        }
        f = std::max(f, local);
    }
    return f;
}

void MAC::BuildAxis(double a0, double af, int nBase, double dh,
                      const std::vector<GridRefinement> &refs, bool nonUniform,
                      std::vector<double> &size, std::vector<double> &face,
                      std::vector<double> &centre)
{
    size.clear();
    face.clear();
    centre.clear();

    bool anyActive = false, anyGraded = false;
    for (const GridRefinement &r : refs)
        if (r.active && r.factor > 1)
        {
            anyActive = true;
            if (r.mode != REFINE_HARD) anyGraded = true;
        }

    if (!nonUniform || !anyActive || nBase <= 0)
    {
        // Uniform: the identical expressions the solvers used to write inline, so a lookup
        // returns the same double and uniform runs stay bit-identical.
        size.assign(nBase, dh);
        face.resize(nBase + 1);
        centre.resize(nBase);
        for (int j = 0; j <= nBase; j++)
            face[j] = a0 + j * dh;
        for (int j = 0; j < nBase; j++)
            centre[j] = a0 + j * dh + dh / 2.0;
        return;
    }

    // The grid does NOT necessarily reach af.
    const double aEnd = a0 + nBase * dh;

    if (!anyGraded)
    {
        // All-hard: split whole base cells, which keeps the refined region snapped to the
        // base grid and is exactly what the single-interval version did.
        for (int j = 0; j < nBase; j++)
        {
            const double c = a0 + j * dh + dh / 2.0;
            const int f = (std::max)(1, (int)(FactorAt(refs, c) + 0.5));
            for (int t = 0; t < f; t++)
                size.push_back(dh / (double)f);
        }
    }
    else
    {
        // Graded. Every HARD interval keeps its own edges and a whole number of equal cells; the spans between edges
        // are marched against the target size and each rescaled to fit itself, so a band around a body stays put.
        std::vector<double> anchors = {a0, aEnd};
        for (const GridRefinement &r : refs)
            if (r.active && r.factor > 1 && r.mode == REFINE_HARD)
                for (double e : {r.start, r.end})
                    if (e > a0 + 1e-12 && e < aEnd - 1e-12) anchors.push_back(e);
        std::sort(anchors.begin(), anchors.end());
        std::vector<double> edges;
        for (double e : anchors)
            if (edges.empty() || e - edges.back() > 1e-12) edges.push_back(e);
        anchors.swap(edges);
        const int guard = 4000000;
        for (size_t s = 0; s + 1 < anchors.size(); s++)
        {
            const double p = anchors[s], q = anchors[s + 1], L = q - p;
            // Constant only when no graded interval overlaps the span: n equal cells. (Sampling the factor at a few
            // points missed a ramp lying strictly inside a span.)
            bool graded = false;
            for (const GridRefinement &r : refs)
                if (r.active && r.factor > 1 && r.mode != REFINE_HARD && r.start < q - 1e-12 && r.end > p + 1e-12)
                    graded = true;
            const double fm = FactorAt(refs, 0.5 * (p + q));
            if (!graded)
            {
                const int n = (std::max)(1, (int)std::lround(L * fm / dh));
                for (int t = 0; t < n; t++) size.push_back(L / n);
                continue;
            }
            // Deliberately NOT clipped to the far edge: overshoot and rescale, so no short last cell.
            const size_t first = size.size();
            double x = p;
            while (x < q - 1e-12 && (int)size.size() < guard)
            {
                const double h = dh / (std::max)(1.0, FactorAt(refs, x));
                if (h <= 0.0 || !std::isfinite(h)) break;
                size.push_back(h);
                x += h;
            }
            if (size.size() == first) size.push_back(L);
            double sum = 0.0;
            for (size_t t = first; t < size.size(); t++) sum += size[t];
            for (size_t t = first; t < size.size(); t++) size[t] *= L / sum;
        }
    }

    const int n = (int)size.size();
    face.resize(n + 1);
    centre.resize(n);
    face[0] = a0;
    for (int j = 0; j < n; j++)
        face[j + 1] = face[j] + size[j];
    face[n] = aEnd; // pin the far face; the running sum drifts by an ulp or two
    for (int j = 0; j < n; j++)
        centre[j] = 0.5 * (face[j] + face[j + 1]);
}

// The border ring, one cell at each end OUTSIDE the axis BuildAxis built: each ring cell mirrors the width of the
// interior cell beside it, so the wall faces stay where the domain puts them and the geometry about a wall is symmetric.
void MAC::AddRing(std::vector<double> &size, std::vector<double> &face, std::vector<double> &centre)
{
    if (size.empty()) return;
    const double w0 = size.front(), w1 = size.back();
    const double a0 = face.front(), a1 = face.back();
    size.insert(size.begin(), w0);
    size.push_back(w1);
    face.insert(face.begin(), a0 - w0);
    face.push_back(a1 + w1);
    centre.insert(centre.begin(), a0 - 0.5 * w0);
    centre.push_back(a1 + 0.5 * w1);
}

// The gap across face m (between cells m and m+1), padded the same way, from the edge-repeated widths --
// the expression DxFace always used, so every lookup returns the same double.
void MAC::GapTable(const std::vector<double> &w, std::vector<double> &gap)
{
    const int n = (int)w.size();
    gap.resize(n + 1);
    for (int m = -1; m <= n - 1; m++)
    {
        const double a = w[m < 0 ? 0 : (m >= n ? n - 1 : m)];
        const double b = w[m + 1 >= n ? n - 1 : m + 1];
        gap[m + 1] = 0.5 * (a + b);
    }
}

void MAC::BuildSpacing()
{
    // EULERIAN ONLY, both dimensions.
    const bool nu = SIMULATION.grid.nonUniform && MethodSupportsRectilinear();
    this->nonUniform = nu;
    // Refinement asked for and refused: say so once, rather than run the uniform grid in silence.
    static bool warnedUniform = false;
    if (SIMULATION.grid.nonUniform && !nu && !warnedUniform)
    {
        bool anyRefine = false;
        for (const std::vector<GridRefinement> *refs : {&SIMULATION.grid.refineX, &SIMULATION.grid.refineY, &SIMULATION.grid.refineZ})
            for (const GridRefinement &r : *refs)
                if (r.active && r.factor > 1) anyRefine = true;
        if (anyRefine)
        {
            std::cerr << "Warning: NON_UNIFORM_GRID = 1, but the method " << ActiveMethod().key
                      << " does not support stretched grids; the grid is built uniform." << std::endl;
            warnedUniform = true;
        }
    }

    // The domain is the fluid box: its cells are built on [a0, af], and the border ring is added one cell OUTSIDE it,
    // so the boundary faces lie on the domain's faces. Nx/Ny/Nz arrive as the interior counts plus the ring.
    BuildAxis(omega.x0, omega.xf, Nx - 2, dh, SIMULATION.grid.refineX, nu, dxs, xfv, xcv);
    BuildAxis(omega.y0, omega.yf, Ny - 2, dh, SIMULATION.grid.refineY, nu, dys, yfv, ycv);
    AddRing(dxs, xfv, xcv);
    AddRing(dys, yfv, ycv);
    if (is2D)
    {
        // Nothing in 2D reads dz, but a zero would silently poison any volume that did.
        dzs.assign(1, dh);
        zfv.assign(2, 0.0);
        zfv[1] = dh;
        zcv.assign(1, dh / 2.0);
    }
    else
    {
        BuildAxis(omega.z0, omega.zf, Nz - 2, dh, SIMULATION.grid.refineZ, nu, dzs, zfv, zcv);
        AddRing(dzs, zfv, zcv);
    }

    // Refinement changes the cell count, so the grid dimensions follow the tables.
    Nx = (int)dxs.size();
    Ny = (int)dys.size();
    if (!is2D)
        Nz = (int)dzs.size();

    PadTable(dxs, widthTable[X]); PadTable(dys, widthTable[Y]); PadTable(dzs, widthTable[Z]);
    PadTable(xfv, faceTable[X]);  PadTable(yfv, faceTable[Y]);  PadTable(zfv, faceTable[Z]);
    PadTable(xcv, centreTable[X]); PadTable(ycv, centreTable[Y]); PadTable(zcv, centreTable[Z]);
    GapTable(dxs, gapTable[X]);   GapTable(dys, gapTable[Y]);   GapTable(dzs, gapTable[Z]);

    // The same products the pressure assembly formed per cell, in the same order.
    if (is2D)
    {
        cellVolume.resize((size_t)Nx * Ny);
        for (int i = 0; i < Ny; i++)
            for (int j = 0; j < Nx; j++)
                cellVolume[(size_t)i * Nx + j] = Width(X, j) * Width(Y, i);
    }
    else
    {
        cellVolume.resize((size_t)Nx * Ny * Nz);
        for (int i = 0; i < Ny; i++)
            for (int j = 0; j < Nx; j++)
                for (int k = 0; k < Nz; k++)
                    cellVolume[(size_t)i * (Nx * Nz) + j * Nz + k] = Width(X, j) * Width(Y, i) * Width(Z, k);
    }

    // Mirror the tables onto SIMULATION so Scene.h can map an index to a coordinate
    // without including MAC.h.
    SIMULATION.spacing.nonUniform = nu;
    SIMULATION.spacing.dx = dxs; SIMULATION.spacing.dy = dys; SIMULATION.spacing.dz = dzs;
    SIMULATION.spacing.xf = xfv; SIMULATION.spacing.yf = yfv; SIMULATION.spacing.zf = zfv;
    SIMULATION.spacing.xc = xcv; SIMULATION.spacing.yc = ycv; SIMULATION.spacing.zc = zcv;

    if (nu)
    {
        // Anything still reading the scalar dh -- the adaptive timestep, the CFL readout -- has to
        // see the tightest spacing.
        double hmin = dh;
        for (double d : dxs) hmin = std::min(hmin, d);
        for (double d : dys) hmin = std::min(hmin, d);
        if (!is2D)
            for (double d : dzs) hmin = std::min(hmin, d);
        this->dh = hmin;

        static bool reported = false;   // both of the run's grids are built alike: say it once
        if (!reported)
        {
            std::cout << "MAC: stretched grid " << Nx << "x" << Ny;
            if (!is2D) std::cout << "x" << Nz;
            std::cout << " cells (ring included), smallest cell " << this->dh << std::endl;
            reported = true;
        }
    }
}

void MAC::InitializeGrid(Domain omega, bool is2D_ )
{
    if(is2D_ == false){

    is2D = false;
    this->omega = omega;
    double min = omega.xf - omega.x0;
    this->Nx = SIMULATION.GRID_SIZE;
    this->Ny = -1;
    this->Nz = -1;
    if ((omega.yf - omega.y0) < min)
    {
        min = omega.yf - omega.y0;
        this->Nx = -1;
        this->Ny = SIMULATION.GRID_SIZE;
        this->Nz = -1;
    }

    // GRID_SIZE interior cells across the shortest side; the ring adds one cell outside each face.
    this->dh = (min / (double)SIMULATION.GRID_SIZE);

    this->Nx = InteriorCells(omega.xf - omega.x0, dh) + 2;
    this->Ny = InteriorCells(omega.yf - omega.y0, dh) + 2;
    this->Nz = InteriorCells(omega.zf - omega.z0, dh) + 2;

    BuildSpacing(); // sizes the tables and, under refinement, the grid itself

    // allocating memory

    this->u = VectorXd(((Nx + 1) * Ny * Nz));        //(double*)calloc(((Nx+1) * Ny * Nz),sizeof(double));
    this->v = VectorXd((Nx * (Ny + 1) * Nz));        //(double*)calloc((Nx * (Ny+1) * Nz),sizeof(double));
    this->w = VectorXd((Nx * (Ny) * (Nz + 1)));      //(double*)calloc((Nx * (Ny) * (Nz+1)),sizeof(double));
    this->p = VectorXd((Nx * (Ny) * (Nz)));          //(double*)calloc((Nx * (Ny) * (Nz)),sizeof(double));
    this->cellTypes.assign((size_t)Nx * Ny * Nz, (unsigned char)FLUID_CELL);
    boundaryListsStale = true;

    this->u.setZero();
    this->v.setZero();
    this->w.setZero();
    this->p.setZero();
    AllocateScalars();   // a field registered before a rebuild comes back at the NEW size
    BuildStrides();
    }
    else{
        this->is2D = true;
        Domain2D omega2d = Domain2D();
        omega2d.x0 = omega.x0;
        omega2d.xf = omega.xf;
        omega2d.y0 = omega.y0;
        omega2d.yf = omega.yf;

        InitializeGrid(omega2d);
    }
}

//  this will probably die
void MAC::SetSolidMask(int (*SolidMaskFunction)(int, int, int))
{
    if(is2D){
        std::cout << "ERROR - MISMATCHED FUNCTION CALLED ON SET LEVEL GEOMETRY" << std::endl;
    }
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            for (int k = 0; k < Nz; k++)
            {

                this->SetSolid(i, j, k, SolidMaskFunction(i, j, k));
            }
        }
    }
}

double MAC::GetDivergencyAt(int i, int j, int k)
{
    if(this->GetSolid(i,j,k) != FLUID_CELL) return 0;

    if (nonUniform)
    {
        // Net face flux over the cell volume, exactly as the 2D path does.
        const double dx = Width(X, j), dy = Width(Y, i), dz = Width(Z, k);
        const double fluxU = (this->GetU(i, j + 1, k) - this->GetU(i, j, k)) * dy * dz;
        const double fluxV = (this->GetV(i + 1, j, k) - this->GetV(i, j, k)) * dx * dz;
        const double fluxW = (this->GetW(i, j, k + 1) - this->GetW(i, j, k)) * dx * dy;
        return (fluxU + fluxV + fluxW) / (dx * dy * dz);
    }

    double divU = (this->GetU(i, j + 1, k) - this->GetU(i, j, k)) / this->dh;
    double divV = (this->GetV(i + 1, j, k) - this->GetV(i, j, k)) / this->dh;
    double divW = (this->GetW(i, j, k + 1) - this->GetW(i, j, k)) / this->dh;
    return (divU + divV + divW);
}

;
;
;

// ======================= 2D OVERLOADS==================

void MAC::InitializeGrid(Domain2D omega)
{
    is2D = true;

    this->omega.x0 = omega.x0;
    this->omega.xf = omega.xf;

    this->omega.y0 = omega.y0;
    this->omega.yf = omega.yf;

    double min = omega.xf - omega.x0;
    this->Nx = SIMULATION.GRID_SIZE;
    this->Ny = -1;

    if ((omega.yf - omega.y0) < min)
    {
        min = omega.yf - omega.y0;
        this->Nx = -1;
        this->Ny = SIMULATION.GRID_SIZE;

    }

    // GRID_SIZE interior cells across the shortest side; the ring adds one cell outside each face.
    this->dh = (min / (double)SIMULATION.GRID_SIZE);

    this->Nx = InteriorCells(omega.xf - omega.x0, dh) + 2;
    this->Ny = InteriorCells(omega.yf - omega.y0, dh) + 2;

    BuildSpacing(); // sizes the tables and, under refinement, the grid itself

    // allocating memory

    this->u = VectorXd(((Nx + 1) * Ny));        //(double*)calloc(((Nx+1) * Ny * Nz),sizeof(double));
    this->v = VectorXd((Nx * (Ny + 1)));        //(double*)calloc((Nx * (Ny+1) * Nz),sizeof(double));

    this->p = VectorXd((Nx * (Ny)));          //(double*)calloc((Nx * (Ny) * (Nz)),sizeof(double));
    this->cellTypes.assign((size_t)Nx * Ny, (unsigned char)FLUID_CELL);
    boundaryListsStale = true;

    this->u.setZero();
    this->v.setZero();

    this->p.setZero();
    AllocateScalars();   // a field registered before a rebuild comes back at the NEW size

    this->Nz = 0;
    BuildStrides();

}

// this will go
void MAC::SetSolidMask(int (*SolidMaskFunction)(int, int))
{
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {

                this->SetSolid(i, j, SolidMaskFunction(i, j));

        }
    }
}

// sets a homogeneous neumman boundary condition at the exit of the x domain, for the canal experiments

// Sets the pressure on the step and everything

double MAC::GetDivergencyAt(int i, int j)
{    if(this->GetSolid(i,j) != FLUID_CELL) return 0;
    if (nonUniform)
    {
        // Net face flux over the cell volume.
        const double dx = Width(X, j), dy = Width(Y, i);
        const double fluxU = (this->GetU(i, j + 1) - this->GetU(i, j)) * dy;
        const double fluxV = (this->GetV(i + 1, j) - this->GetV(i, j)) * dx;
        return (fluxU + fluxV) / (dx * dy);
    }
    double divU = (this->GetU(i, j + 1) - this->GetU(i, j)) / this->dh;
    double divV = (this->GetV(i + 1, j) - this->GetV(i, j)) / this->dh;
    return (divU + divV);
}

// Body force, both the scene expression and the scalar Fu/Fv/Fw. See MAC.h.
void MAC::AddForce(double time, double dt)
{
    const bool sceneForce = SCENE.loaded && SCENE.hasForce;
    const double cu = SIMULATION.f.u, cv = SIMULATION.f.v, cw = SIMULATION.f.w;

    // Nothing to do -- and this is what keeps every existing case bit-identical.
    if (!sceneForce && cu == 0.0 && cv == 0.0 && cw == 0.0)
        return;

    // SERIAL on purpose. The face rows the boundary pins (two deep along each normal) are the boundary's, not the
    // force's, so each normal loop starts at 2.
    double fx = 0.0, fy = 0.0, fz = 0.0;

    if (is2D)
    {
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 2; j < Nx - 1; j++)
            {
                if (!IsFluidFace(MAC::U, i, j)) continue;
                if (sceneForce) SCENE.ForceAt(FacePos(X, j), CentrePos(Y, i), 0.0, time, fx, fy, fz);
                SetU(i, j, GetU(i, j) + (cu + fx) * dt);
            }

        for (int i = 2; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
            {
                if (!IsFluidFace(MAC::V, i, j)) continue;
                if (sceneForce) SCENE.ForceAt(CentrePos(X, j), FacePos(Y, i), 0.0, time, fx, fy, fz);
                SetV(i, j, GetV(i, j) + (cv + fy) * dt);
            }
        return;
    }

    for (int i = 1; i < Ny - 1; i++)
        for (int j = 2; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                if (!IsFluidFace(MAC::U, i, j, k)) continue;
                if (sceneForce) SCENE.ForceAt(FacePos(X, j), CentrePos(Y, i), CentrePos(Z, k), time, fx, fy, fz);
                SetU(i, j, k, GetU(i, j, k) + (cu + fx) * dt);
            }

    for (int i = 2; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                if (!IsFluidFace(MAC::V, i, j, k)) continue;
                if (sceneForce) SCENE.ForceAt(CentrePos(X, j), FacePos(Y, i), CentrePos(Z, k), time, fx, fy, fz);
                SetV(i, j, k, GetV(i, j, k) + (cv + fy) * dt);
            }

    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 2; k < Nz - 1; k++)
            {
                if (!IsFluidFace(MAC::W, i, j, k)) continue;
                if (sceneForce) SCENE.ForceAt(CentrePos(X, j), CentrePos(Y, i), FacePos(Z, k), time, fx, fy, fz);
                SetW(i, j, k, GetW(i, j, k) + (cw + fz) * dt);
            }
}

// Initial velocity field from the scene. S
void MAC::ApplyInitialField()
{
    if (!SCENE.loaded || !SCENE.hasInitial)
        return;

    double u = 0.0, v = 0.0, w = 0.0;   // SERIAL: Expr binds variables to fixed addresses

    if (is2D)
    {
        for (int i = 0; i < Ny; i++)
            for (int j = 0; j < Nx + 1; j++)
            { SCENE.InitialVelocityAt(FacePos(X, j), CentrePos(Y, i), 0.0, u, v, w); SetU(i, j, u); }

        for (int i = 0; i < Ny + 1; i++)
            for (int j = 0; j < Nx; j++)
            { SCENE.InitialVelocityAt(CentrePos(X, j), FacePos(Y, i), 0.0, u, v, w); SetV(i, j, v); }
        return;
    }

    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx + 1; j++)
            for (int k = 0; k < Nz; k++)
            { SCENE.InitialVelocityAt(FacePos(X, j), CentrePos(Y, i), CentrePos(Z, k), u, v, w); SetU(i, j, k, u); }

    for (int i = 0; i < Ny + 1; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < Nz; k++)
            { SCENE.InitialVelocityAt(CentrePos(X, j), FacePos(Y, i), CentrePos(Z, k), u, v, w); SetV(i, j, k, v); }

    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < Nz + 1; k++)
            { SCENE.InitialVelocityAt(CentrePos(X, j), CentrePos(Y, i), FacePos(Z, k), u, v, w); SetW(i, j, k, w); }
}

void MAC::ResetFluidCells(){

    if(is2D){
    for (int i = 0; i < Ny; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            // SOLID, INFLOW and PRESSURE are boundary classifications, not liquid: they come from
            // the geometry and must survive the rebuild.
            const int t = this->GetSolid(i,j);
            if(t != SOLID_CELL && !IsVelocityBoundaryCell(t) && t != PRESSURE_CELL){
                this->SetSolid(i,j,EMPTY_CELL); //resets all fluid cells to empty, keeps the solid mask intact and it is more efficient
            }

        }
    }
    }
    else{
        for (int i = 0; i < Ny; i++)
        {
            for (int j = 0; j < Nx; j++)
            {
                for(int k = 0;k<Nz;k++){
                const int t = this->GetSolid(i,j,k);
                if(t != SOLID_CELL && !IsVelocityBoundaryCell(t) && t != PRESSURE_CELL){
                    this->SetSolid(i,j,k,EMPTY_CELL); //resets all fluid cells to empty, keeps the solid mask intact and it is more efficient
                }
                }

            }
        }
    }

}

// ---- boundary conditions: forcing the ring's values (prescribed, or copied from inside) ----
// Every write is recorded once by BuildBoundaryLists and replayed; the order below is load-bearing.

// A ring cell's prescribed pressure is the value on the boundary face it borders: the face's coordinate along
// each axis on which the cell is on the ring, the centre's elsewhere.
static inline double RingCoordinate(const MAC &g, MAC::Axis a, int n, int N)
{
    if (n == 0) return g.FacePos(a, 1);
    if (n == N - 1) return g.FacePos(a, N - 1);
    return g.CentrePos(a, n);
}

double MAC::PrescribedPressure(int i, int j, int k, double t)
{
    const double x = RingCoordinate(*this, X, j, Nx);
    const double y = RingCoordinate(*this, Y, i, Ny);
    if (is2D)
        return SIMULATION.PressureBoundaryFunction2D ? SIMULATION.PressureBoundaryFunction2D(x, y, t) : 0.0;
    const double z = RingCoordinate(*this, Z, k, Nz);
    return SIMULATION.PressureBoundaryFunction ? SIMULATION.PressureBoundaryFunction(x, y, z, t) : 0.0;
}

void MAC::BuildBoundaryLists()
{
    boundaryListsStale = false;
    prescribedList.clear();
    copyList.clear();
    for (int c = 0; c < 3; c++) solidFaceList[c].clear();
    ringList.clear();
    BuildRingKinds();
    if (is2D) { BuildPrescribedList2D(); BuildSolidFaceList2D(); }
    else      { BuildPrescribedList3D(); BuildSolidFaceList3D(); }
    BuildCopyList();
    BuildFluxFaces();
    BuildRingList();
    BuildWallGhosts();
}

// The faces inside a no-slip or moving wall: both cells either side SOLID, and neither a slip ring cell (a slip wall's
// tangential value is a copy of the interior, zero gradient, not a wall velocity).
void MAC::BuildWallGhosts()
{
    const int nz = is2D ? 1 : Nz;
    const int comps = is2D ? 2 : 3;
    for (int c = 0; c < 3; c++) wallGhost[c].clear();
    for (int c = 0; c < comps; c++)
    {
        wallGhost[c].assign(Field((Component)c).size(), 0);
        const int ext[3] = {Nx + (c == U), Ny + (c == V), nz + (c == W)};   // x, y, z face counts
        for (int i = 0; i < ext[1]; i++)
            for (int j = 0; j < ext[0]; j++)
                for (int k = 0; k < ext[2]; k++)
                {
                    const int bi = i - (c == V), bj = j - (c == U), bk = k - (c == W);   // the cell behind
                    if (bi < 0 || bj < 0 || bk < 0 || i >= Ny || j >= Nx || k >= nz) continue;
                    const int ta = is2D ? GetSolid(i, j) : GetSolid(i, j, k);
                    const int tb = is2D ? GetSolid(bi, bj) : GetSolid(bi, bj, bk);
                    if (ta != SOLID_CELL || tb != SOLID_CELL) continue;
                    if (ringKind[(i * Nx + j) * nz + k] == BC_SLIP || ringKind[(bi * Nx + bj) * nz + bk] == BC_SLIP) continue;
                    wallGhost[c][FieldIndex((Component)c, i, j, k)] = 1;
                }
    }
}

// The cell type each kind puts on the ring; a kind whose cell type disagrees with the grid's (a harness that set the
// mask itself) is not trusted.
static bool KindMatchesCell(int kind, int type)
{
    switch (kind)
    {
    case BC_NOSLIP: case BC_SLIP: case BC_MOVING: return type == SOLID_CELL;
    case BC_INLET:                                return type == INFLOW_CELL;
    case BC_OUTFLOW:                              return type == OUTFLOW_CELL;
    case BC_PRESSURE_OUTLET:                      return type == EMPTY_CELL || type == PRESSURE_CELL;
    default:                                      return false;
    }
}

// A ring cell's kind and owning face. From the scene when there is one; otherwise from the cell type, with a
// corner (on two or three faces) owned by none, so nothing is copied there. -1 off the ring.
int MAC::RingKind(int i, int j, int k, int &face) const
{
    face = -1;
    const bool onX = (j == 0 || j == Nx - 1), onY = (i == 0 || i == Ny - 1);
    const bool onZ = !is2D && (k == 0 || k == Nz - 1);
    const int codim = (int)onX + (int)onY + (int)onZ;
    if (codim == 0) return -1;
    const int type = is2D ? GetSolid(i, j) : GetSolid(i, j, k);

    int kind = SIMULATION.BoundaryKindFunction ? SIMULATION.BoundaryKindFunction(i, j, k, face) : -1;
    if (kind < 0 || face < 0 || !KindMatchesCell(kind, type))
    {
        kind = (type == SOLID_CELL) ? BC_NOSLIP
             : (type == INFLOW_CELL) ? BC_INLET
             : (type == OUTFLOW_CELL) ? BC_OUTFLOW
             : (type == EMPTY_CELL || type == PRESSURE_CELL) ? BC_PRESSURE_OUTLET : -1;
        face = -1;
        if (codim == 1)
            face = onX ? (j == 0 ? 0 : 1) : (onY ? (i == 0 ? 2 : 3) : (k == 0 ? 4 : 5));
    }
    return kind;
}

void MAC::BuildRingKinds()
{
    const int nz = is2D ? 1 : Nz;   // Nz is 0 in 2D
    const int n = Nx * Ny * nz;
    ringKind.assign(n, -1);
    ringOwner.assign(n, -1);
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < nz; k++)
            {
                int face = -1;
                const int kind = RingKind(i, j, k, face);   // -1 and no owner off the ring
                const int cell = (i * Nx + j) * nz + k;
                ringKind[cell] = (signed char)kind;
                ringOwner[cell] = (signed char)face;
            }
}

int MAC::BoundaryKindAt(int i, int j, int k) const
{
    int face = -1;
    const int kind = RingKind(i, j, k, face);
    if (kind >= 0) return kind;
    const int type = is2D ? GetSolid(i, j) : GetSolid(i, j, k);
    return type == SOLID_CELL ? BC_NOSLIP : -1;   // an interior solid is a no-slip wall
}

// The zero-gradient copies of ring cell (i, j, k), by its kind, from the neighbour one cell inward: the normal
// velocity two faces deep (outlets), a tangential component on the faces the cell owns (slip, outlets).
void MAC::RecordCellCopies(int i, int j, int k)
{
    const int nz = is2D ? 1 : Nz;
    const int cell = (i * Nx + j) * nz + k;
    const int kind = ringKind[cell], owner = ringOwner[cell];
    if (owner < 0) return;
    const bool normal = (kind == BC_PRESSURE_OUTLET || kind == BC_OUTFLOW);
    const bool tangential = normal || kind == BC_SLIP;
    if (!tangential) return;

    const int a = owner / 2;
    const bool high = (owner % 2) == 1;
    const int count[3] = {Nx, Ny, nz};
    const int n[3] = {j, i, k};
    const int d = high ? -1 : 1;

    if (normal)
    {
        const Component c = ComponentAlong((Axis)a);
        int f[3] = {n[0], n[1], n[2]}, s[3] = {n[0], n[1], n[2]};
        const int N = count[a];
        const int first = high ? N - 1 : 1, outer = high ? N : 0;
        f[a] = first; s[a] = first + d;
        RecordCopy(c, f[1], f[0], f[2], s[1], s[0], s[2]);
        f[a] = outer; s[a] = first;
        RecordCopy(c, f[1], f[0], f[2], s[1], s[0], s[2]);
    }
    const int axes = is2D ? 2 : 3;
    for (int b = 0; b < axes; b++)
    {
        if (b == a) continue;
        const Component c = ComponentAlong((Axis)b);
        for (int face = n[b]; face <= n[b] + (n[b] == count[b] - 1 ? 1 : 0); face++)
        {
            int f[3] = {n[0], n[1], n[2]}, s[3] = {n[0], n[1], n[2]};
            f[b] = s[b] = face;
            s[a] = n[a] + d;
            RecordCopy(c, f[1], f[0], f[2], s[1], s[0], s[2]);
        }
    }
}

// Cells on one face first, then edges, then corners: a corner copies from an edge, an edge from a face.
void MAC::BuildCopyList()
{
    const int nz = is2D ? 1 : Nz;
    for (int codim = 1; codim <= 3; codim++)
        for (int i = 0; i < Ny; i++)
            for (int j = 0; j < Nx; j++)
                for (int k = 0; k < nz; k++)
                {
                    const int c = (int)(j == 0 || j == Nx - 1) + (int)(i == 0 || i == Ny - 1) +
                                  (int)(!is2D && (k == 0 || k == Nz - 1));
                    if (c == codim) RecordCellCopies(i, j, k);
                }
}

// The boundary faces of the inlets and outflows: the normal face between each ring cell on one face and the
// interior, with its area and outward sign.
void MAC::BuildFluxFaces()
{
    inletFaces.clear();
    outflowFaces.clear();
    const int nz = is2D ? 1 : Nz;
    const int count[3] = {Nx, Ny, nz};
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < nz; k++)
            {
                const int cell = (i * Nx + j) * nz + k;
                const int kind = ringKind[cell], owner = ringOwner[cell];
                if (owner < 0 || (kind != BC_INLET && kind != BC_OUTFLOW)) continue;
                const int codim = (int)(j == 0 || j == Nx - 1) + (int)(i == 0 || i == Ny - 1) +
                                  (int)(!is2D && (k == 0 || k == Nz - 1));
                if (codim != 1) continue;
                const int a = owner / 2;
                const bool high = (owner % 2) == 1;
                const Component c = ComponentAlong((Axis)a);
                int f[3] = {j, i, k};
                f[a] = high ? count[a] - 1 : 1;
                const int index = FieldIndex(c, f[1], f[0], f[2]);
                f[a] = high ? count[a] : 0;
                const int outer = FieldIndex(c, f[1], f[0], f[2]);
                double area = 1.0;
                const int axes = is2D ? 2 : 3;
                for (int b = 0; b < axes; b++)
                    if (b != a) area *= Width((Axis)b, b == X ? j : (b == Y ? i : k));
                const FluxFace ff{c, index, outer, area, high ? 1.0 : -1.0};
                (kind == BC_INLET ? inletFaces : outflowFaces).push_back(ff);
            }
}

// The outflow's zero-gradient velocity carries whatever the interior brings to it; add one uniform correction so
// it takes out exactly what the inlets bring in (the same volume, the fluid being incompressible).
void MAC::ApplyOutflowFlux()
{
    if (outflowFaces.empty()) return;
    double qIn = 0.0, qOut = 0.0, aOut = 0.0;
    for (const FluxFace &f : inletFaces) qIn -= f.sign * Field(f.c)[f.index] * f.area;
    for (const FluxFace &f : outflowFaces)
    {
        qOut += f.sign * Field(f.c)[f.index] * f.area;
        aOut += f.area;
    }
    if (aOut <= 0.0) return;
    const double delta = (qIn - qOut) / aOut;
    for (const FluxFace &f : outflowFaces)
    {
        VectorXd &field = Field(f.c);
        field[f.index] += f.sign * delta;
        field[f.outer] = field[f.index];
    }
}

void MAC::EnforceVelocityBC(double t)
{
    if (boundaryListsStale) BuildBoundaryLists();
    // No flux through a solid face -- the prescribed values below restore a moving wall's ring.
    for (int c = U; c <= W; c++)
    {
        VectorXd &f = Field((Component)c);
        for (int index : solidFaceList[c]) f[index] = 0.0;
    }
    if (is2D && boundaryVelocity2D)
        for (const PrescribedEntry &e : prescribedList)
        {
            const Vec2 vel = boundaryVelocity2D(e.x, e.y, t);
            Field(e.c)[e.index] = (e.c == U) ? vel.u : vel.v;
        }
    else if (!is2D && boundaryVelocity3D)
        for (const PrescribedEntry &e : prescribedList)
        {
            const Vec3 vel = boundaryVelocity3D(e.x, e.y, e.z, t);
            Field(e.c)[e.index] = (e.c == U) ? vel.u : (e.c == V ? vel.v : vel.w);
        }
    for (const CopyEntry &e : copyList)
    {
        VectorXd &f = Field(e.c);
        f[e.dst] = f[e.src];
    }
    ApplyOutflowFlux();
}

void MAC::EnforcePressureBC(double t)
{
    if (boundaryListsStale) BuildBoundaryLists();
    for (const RingEntry &e : ringList)
        p[e.dst] = (e.kind == 0) ? 0.0 : (e.kind == 1 ? PrescribedPressure(e.i, e.j, e.k, t) : p[e.src]);
}

void MAC::BuildPrescribedList3D()
{    
    // u z face
    for (int j = 0; j < Nx + 1; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            RecordPrescribed(U, i, j, 0, FacePos(X, j), CentrePos(Y, i), CentrePos(Z, 0));
            RecordPrescribed(U, i, j, Nz - 1, FacePos(X, j), CentrePos(Y, i), CentrePos(Z, Nz - 1));
        }
    }

    // u y face
    for (int j = 0; j < Nx + 1; j++)
    {
        for (int k = 0; k < Nz; k++)
        {
            RecordPrescribed(U, 0, j, k, FacePos(X, j), CentrePos(Y, 0), CentrePos(Z, k));
            RecordPrescribed(U, Ny - 1, j, k, FacePos(X, j), CentrePos(Y, Ny - 1), CentrePos(Z, k));
        }
    }
    // u x face
    for (int i = 0; i < Ny; i++)
    {
        for (int k = 0; k < Nz; k++)
        {
            RecordPrescribed(U, i, 0, k, FacePos(X, 0), CentrePos(Y, i), CentrePos(Z, k));
            RecordPrescribed(U, i, 1, k, FacePos(X, 1), CentrePos(Y, i), CentrePos(Z, k));

            RecordPrescribed(U, i, Nx + 1 - 1, k, FacePos(X, Nx + 1 - 1), CentrePos(Y, i), CentrePos(Z, k));
            RecordPrescribed(U, i, Nx + 1 - 2, k, FacePos(X, Nx + 1 - 2), CentrePos(Y, i), CentrePos(Z, k));
        }
    }

    // v faces
    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny + 1; i++)
        {
            RecordPrescribed(V, i, j, 0, CentrePos(X, j), FacePos(Y, i), CentrePos(Z, 0));
            RecordPrescribed(V, i, j, Nz - 1, CentrePos(X, j), FacePos(Y, i), CentrePos(Z, Nz - 1));
        }
    }

    for (int k = 0; k < Nz; k++)
    {
        for (int i = 0; i < Ny + 1; i++)
        {
            RecordPrescribed(V, i, 0, k, CentrePos(X, 0), FacePos(Y, i), CentrePos(Z, k));
            RecordPrescribed(V, i, Nx - 1, k, CentrePos(X, Nx - 1), FacePos(Y, i), CentrePos(Z, k));
        }
    }

    for (int k = 0; k < Nz; k++)
    {
        for (int j = 0; j < Nx; j++)
        {
            RecordPrescribed(V, 0, j, k, CentrePos(X, j), FacePos(Y, 0), CentrePos(Z, k));
            RecordPrescribed(V, Ny + 1 - 1, j, k, CentrePos(X, j), FacePos(Y, Ny + 1 - 1), CentrePos(Z, k));

            RecordPrescribed(V, 1, j, k, CentrePos(X, j), FacePos(Y, 1), CentrePos(Z, k));
            RecordPrescribed(V, Ny + 1 - 2, j, k, CentrePos(X, j), FacePos(Y, Ny + 1 - 2), CentrePos(Z, k));
        }
    }

    // w face

    for (int j = 0; j < Nx; j++)
    {
        for (int i = 0; i < Ny; i++)
        {
            RecordPrescribed(W, i, j, 0, CentrePos(X, j), CentrePos(Y, i), FacePos(Z, 0));
            RecordPrescribed(W, i, j, Nz + 1 - 1, CentrePos(X, j), CentrePos(Y, i), FacePos(Z, Nz + 1 - 1));

            RecordPrescribed(W, i, j, 1, CentrePos(X, j), CentrePos(Y, i), FacePos(Z, 1));
            RecordPrescribed(W, i, j, Nz + 1 - 2, CentrePos(X, j), CentrePos(Y, i), FacePos(Z, Nz + 1 - 2));
        }
    }

    for (int j = 0; j < Nx; j++)
    {
        for (int k = 0; k < Nz + 1; k++)
        {
            RecordPrescribed(W, 0, j, k, CentrePos(X, j), CentrePos(Y, 0), FacePos(Z, k));
            RecordPrescribed(W, Ny - 1, j, k, CentrePos(X, j), CentrePos(Y, Ny - 1), FacePos(Z, k));
        }
    }

    for (int i = 0; i < Ny; i++)
    {
        for (int k = 0; k < Nz + 1; k++)
        {
            RecordPrescribed(W, i, 0, k, CentrePos(X, 0), CentrePos(Y, i), FacePos(Z, k));
            RecordPrescribed(W, i, Nx - 1, k, CentrePos(X, Nx - 1), CentrePos(Y, i), FacePos(Z, k));
        }
    }

}

void MAC::BuildPrescribedList2D()
{

    // u y face
    for (int j = 0; j < Nx + 1; j++)
    {

        RecordPrescribed(U, 0, j, 0, FacePos(X, j), CentrePos(Y, 0), 0.0);
        RecordPrescribed(U, Ny - 1, j, 0, FacePos(X, j), CentrePos(Y, (Ny - 1)), 0.0);

    }
    // u x face
    for (int i = 0; i < Ny; i++)
    {

        RecordPrescribed(U, i, 0, 0, FacePos(X, 0), CentrePos(Y, i), 0.0);
        RecordPrescribed(U, i, 1, 0, FacePos(X, 1), CentrePos(Y, i), 0.0);
        RecordPrescribed(U, i, Nx + 1 - 1, 0, FacePos(X, (Nx + 1 - 1)), CentrePos(Y, i), 0.0);
        RecordPrescribed(U, i, Nx + 1 - 2, 0, FacePos(X, (Nx + 1 - 2)), CentrePos(Y, i), 0.0);

    }

    for (int i = 0; i < Ny + 1; i++)
    {
        RecordPrescribed(V, i, 0, 0, CentrePos(X, 0), FacePos(Y, i), 0.0);
        RecordPrescribed(V, i, Nx - 1, 0, CentrePos(X, (Nx - 1)), FacePos(Y, i), 0.0);
    }

    for (int j = 0; j < Nx; j++)
    {
        RecordPrescribed(V, 0, j, 0, CentrePos(X, j), FacePos(Y, 0), 0.0);
        RecordPrescribed(V, Ny + 1 - 1, j, 0, CentrePos(X, j), FacePos(Y, (Ny + 1 - 1)), 0.0);
        RecordPrescribed(V, 1, j, 0, CentrePos(X, j), FacePos(Y, 1), 0.0);
        RecordPrescribed(V, Ny + 1 - 2, j, 0, CentrePos(X, j), FacePos(Y, (Ny + 1 - 2)), 0.0);
    }

}

void MAC::BuildSolidFaceList3D()
{
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx + 1; j++)
            for (int k = 0; k < Nz; k++)
            {
                const int right = (j < Nx) ? GetSolid(i, j, k) : SOLID_CELL;
                const int left = (j > 0) ? GetSolid(i, j - 1, k) : SOLID_CELL;
                if (right == SOLID_CELL || left == SOLID_CELL)
                    RecordSolidFace(U, i, j, k);
            }

    for (int i = 0; i < Ny + 1; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < Nz; k++)
            {
                const int above = (i < Ny) ? GetSolid(i, j, k) : SOLID_CELL;
                const int below = (i > 0) ? GetSolid(i - 1, j, k) : SOLID_CELL;
                if (above == SOLID_CELL || below == SOLID_CELL)
                    RecordSolidFace(V, i, j, k);
            }

    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < Nz + 1; k++)
            {
                const int front = (k < Nz) ? GetSolid(i, j, k) : SOLID_CELL;
                const int back = (k > 0) ? GetSolid(i, j, k - 1) : SOLID_CELL;
                if (front == SOLID_CELL || back == SOLID_CELL)
                    RecordSolidFace(W, i, j, k);
            }
}

void MAC::BuildSolidFaceList2D()
{
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx + 1; j++)
        {
            const int right = (j < Nx) ? GetSolid(i, j) : SOLID_CELL;
            const int left = (j > 0) ? GetSolid(i, j - 1) : SOLID_CELL;
            if (right == SOLID_CELL || left == SOLID_CELL)
                RecordSolidFace(U, i, j, 0);
        }

    for (int i = 0; i < Ny + 1; i++)
        for (int j = 0; j < Nx; j++)
        {
            const int above = (i < Ny) ? GetSolid(i, j) : SOLID_CELL;
            const int below = (i > 0) ? GetSolid(i - 1, j) : SOLID_CELL;
            if (above == SOLID_CELL || below == SOLID_CELL)
                RecordSolidFace(V, i, j, 0);
        }
}

void MAC::BuildRingList()
{
    if(is2D) {
        for (int j = 0; j < Nx; j++) {
            RecordRing(0, j, 0, 1, j, 0);
            RecordRing(Ny - 1, j, 0, Ny - 2, j, 0);
        }
        for (int i = 0; i < Ny; i++) {
            RecordRing(i, 0, 0, i, 1, 0);
            RecordRing(i, Nx - 1, 0, i, Nx - 2, 0);
        }
        return;
    }

    for (int j = 0; j < Nx; j++) {
        for (int k = 0; k < Nz; k++) {
            RecordRing(0, j, k, 1, j, k);
            RecordRing(Ny - 1, j, k, Ny - 2, j, k);
        }
    }
    for (int i = 0; i < Ny; i++) {
        for (int k = 0; k < Nz; k++) {
            RecordRing(i, 0, k, i, 1, k);
            RecordRing(i, Nx - 1, k, i, Nx - 2, k);
        }
    }
    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            RecordRing(i, j, 0, i, j, 1);
            RecordRing(i, j, Nz - 1, i, j, Nz - 2);
        }
    }
}


double MAC::MaxAbsoluteDifference(MAC &grid) {
    double max = 0.0;

    if(is2D) {
        // 2D case - no k index
        max = abs(this->GetU(0, 0) - grid.GetU(0, 0));

        // Check U component
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx + 1; j++) {
                if (max < abs(this->GetU(i, j) - grid.GetU(i, j))) {
                    max = abs(this->GetU(i, j) - grid.GetU(i, j));
                }
            }
        }

        // Check V component
        for (int i = 0; i < this->Ny + 1; i++) {
            for (int j = 0; j < this->Nx; j++) {
                if (max < abs(this->GetV(i, j) - grid.GetV(i, j))) {
                    max = abs(this->GetV(i, j) - grid.GetV(i, j));
                }
            }
        }
    } else {
        // 3D case
        max = abs(this->GetU(0, 0, 0) - grid.GetU(0, 0, 0));

        // Check U component
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx + 1; j++) {
                    if (max < abs(this->GetU(i, j, k) - grid.GetU(i, j, k))) {
                        max = abs(this->GetU(i, j, k) - grid.GetU(i, j, k));
                    }
                }
            }
        }

        // Check V component
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny + 1; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    if (max < abs(this->GetV(i, j, k) - grid.GetV(i, j, k))) {
                        max = abs(this->GetV(i, j, k) - grid.GetV(i, j, k));
                    }
                }
            }
        }

        // Check W component
        for (int k = 0; k < this->Nz + 1; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    if (max < abs(this->GetW(i, j, k) - grid.GetW(i, j, k))) {
                        max = abs(this->GetW(i, j, k) - grid.GetW(i, j, k));
                    }
                }
            }
        }
    }

    return max;
}

double MAC::GetMaxVelocity() {
    double mag = 0.0;

    if(is2D) {
        // 2D case - no k index
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx; j++) {
                float u_val = (j < this->Nx - 1) ? 
                    (GetU(i, j) + GetU(i, j + 1)) / 2.0f : 0.0f;
                float v_val = (i < this->Ny - 1) ? 
                    (GetV(i, j) + GetV(i + 1, j)) / 2.0f : 0.0f;

                double current_mag = sqrt(u_val * u_val + v_val * v_val);

                if (current_mag > mag) {
                    mag = current_mag;
                }
            }
        }
    } else {
        // 3D case
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    float u_val = (j < this->Nx - 1) ? 
                        (GetU(i, j, k) + GetU(i, j + 1, k)) / 2.0f : 0.0f;
                    float v_val = (i < this->Ny - 1) ? 
                        (GetV(i, j, k) + GetV(i + 1, j, k)) / 2.0f : 0.0f;
                    float w_val = (k < this->Nz - 1) ? 
                        (GetW(i, j, k) + GetW(i, j, k + 1)) / 2.0f : 0.0f;

                    double current_mag = sqrt(u_val * u_val + v_val * v_val + w_val * w_val);

                    if (current_mag > mag) {
                        mag = current_mag;
                    }
                }
            }
        }
    }

    return mag;
}

// ---- point sampling -----------------------------------------------------------------------------------

int MAC::Locate(const std::vector<double> &padded, double x, int n) const
{
    if (!nonUniform)
    {
        // Uniform: the arithmetic guess, then corrected against the table, so the answer is the bisection's exactly.
        int m = (int)std::floor((x - padded[1]) / dh);
        m = m < 0 ? 0 : (m > n - 1 ? n - 1 : m);
        while (m < n - 1 && padded[m + 2] <= x) m++;
        while (m > 0 && padded[m + 1] > x) m--;
        return m;
    }
    int lo = 0, hi = n - 1;
    while (lo < hi)
    {
        const int mid = (lo + hi + 1) / 2;
        if (padded[mid + 1] <= x) lo = mid; else hi = mid - 1;
    }
    return lo;
}

double MAC::ScalarAt(int handle, double x, double y)
{
    if (handle < 0 || handle >= ScalarCount()) return 0.0;
    return Sample(scalars[(size_t)handle].data(), strideTable[P], -1, x, y);
}

double MAC::ScalarAt(int handle, double x, double y, double z)
{
    if (handle < 0 || handle >= ScalarCount()) return 0.0;
    return Sample(scalars[(size_t)handle].data(), strideTable[P], -1, x, y, z);
}

double MAC::ComponentAt(Component c, double x, double y)
{
    return Sample(Field(c).data(), strideTable[c], c == P ? -1 : (int)AxisOf(c), x, y);
}

double MAC::ComponentAt(Component c, double x, double y, double z)
{
    return Sample(Field(c).data(), strideTable[c], c == P ? -1 : (int)AxisOf(c), x, y, z);
}

// Product weights, in the order the 2D sampler always summed them. Along the component's own axis the point lies
// between its faces m, m+1; along the other between the centres m, m+1, clamped to the first and last.
double MAC::Sample(const double *F, const int *st, int own, double x, double y)
{
    const double pos[2] = {x, y};
    const int n[2] = {Nx, Ny};
    int m[2];
    double f[2];
    for (int d = 0; d < 2; d++)
    {
        const Axis a = (Axis)d;
        if ((int)a == own)
        {
            m[a] = Locate(faceTable[a], pos[a], n[a]);
            f[a] = (pos[a] - FacePos(a, m[a])) / (FacePos(a, m[a] + 1) - FacePos(a, m[a]));
        }
        else
        {
            m[a] = Locate(centreTable[a], pos[a], n[a] - 1);
            f[a] = std::min(1.0, std::max(0.0, (pos[a] - CentrePos(a, m[a])) / (CentrePos(a, m[a] + 1) - CentrePos(a, m[a]))));
        }
    }
    const int b = m[0] * st[0] + m[1] * st[1];
    return (1.0 - f[0]) * (1.0 - f[1]) * F[b] + f[0] * (1.0 - f[1]) * F[b + st[0]]
         + (1.0 - f[0]) * f[1] * F[b + st[1]] + f[0] * f[1] * F[b + st[0] + st[1]];
}

// Successive linear blends, x then y then z, as the 3D sampler always did; every fraction clamped to [0, 1].
double MAC::Sample(const double *F, const int *st, int own, double x, double y, double z)
{
    const double pos[3] = {x, y, z};
    const int n[3] = {Nx, Ny, Nz};
    int m[3];
    double f[3];
    for (int d = 0; d < 3; d++)
    {
        const Axis a = (Axis)d;
        if ((int)a == own)
        {
            m[a] = Locate(faceTable[a], pos[a], n[a]);
            f[a] = std::min(1.0, std::max(0.0, (pos[a] - FacePos(a, m[a])) / (FacePos(a, m[a] + 1) - FacePos(a, m[a]))));
        }
        else
        {
            m[a] = Locate(centreTable[a], pos[a], n[a] - 1);
            f[a] = std::min(1.0, std::max(0.0, (pos[a] - CentrePos(a, m[a])) / (CentrePos(a, m[a] + 1) - CentrePos(a, m[a]))));
        }
    }
    const int b = m[0] * st[0] + m[1] * st[1] + m[2] * st[2];
    const double s0 = F[b], s1 = F[b + st[0]], s2 = F[b + st[1]], s3 = F[b + st[0] + st[1]];
    const double s4 = F[b + st[2]], s5 = F[b + st[0] + st[2]], s6 = F[b + st[1] + st[2]], s7 = F[b + st[0] + st[1] + st[2]];
    const double x00 = s0 + f[0] * (s1 - s0), x10 = s2 + f[0] * (s3 - s2);
    const double x01 = s4 + f[0] * (s5 - s4), x11 = s6 + f[0] * (s7 - s6);
    const double y0 = x00 + f[1] * (x10 - x00), y1 = x01 + f[1] * (x11 - x01);
    return y0 + f[2] * (y1 - y0);
}


double MAC::GetDivSum() {
    double value = 0.0;

    if(is2D) {
        // 2D case - no k index
        for (int i = 1; i < Ny - 1; i++) {
            for (int j = 1; j < Nx - 1; j++) {
                value += abs(GetDivergencyAt(i, j));
            }
        }
    } else {
        // 3D case
        for (int k = 1; k < Nz - 1; k++) {
            for (int i = 1; i < Ny - 1; i++) {
                for (int j = 1; j < Nx - 1; j++) {
                    value += abs(GetDivergencyAt(i, j, k));
                }
            }
        }
    }

    return value;
}


int MAC::GetFluidCellCount() {
    int c = 0;

    if(is2D) {
        // 2D case - no k index
        for (int i = 0; i < Ny; i++) {
            for (int j = 0; j < Nx; j++) {
                if(this->GetSolid(i, j) == FLUID_CELL) {
                    c++;
                }
            }
        }
    } else {
        // 3D case
        for (int k = 0; k < Nz; k++) {
            for (int i = 0; i < Ny; i++) {
                for (int j = 0; j < Nx; j++) {
                    if(this->GetSolid(i, j, k) == FLUID_CELL) {
                        c++;
                    }
                }
            }
        }
    }

    return c;
}

int MAC::RegisterScalar(const std::string &name, const ScalarTransport &transport)
{
    for (size_t h = 0; h < scalarNames.size(); h++)
        if (scalarNames[h] == name)
        {
            // Registration is idempotent by name, so the FIRST one wins.
            const ScalarTransport &had = scalarTransport[h];
            if (!SameTransport(had, transport))
            {
                scalarSchemeConflicts++;
                std::cout << "WARNING: MAC::RegisterScalar -- '" << name << "' is already registered as "
                          << TransportSchemeName(had.scheme) << "/" << TransportFormName(had.form)
                          << " epsilon " << had.epsilon << " gamma " << had.gamma
                          << " and was asked for again as " << TransportSchemeName(transport.scheme) << "/"
                          << TransportFormName(transport.form) << " epsilon " << transport.epsilon
                          << " gamma " << transport.gamma << " solver " << ScalarSolverName(transport.solver)
                          << ". Keeping the first; register each field once." << std::endl;
            }
            return (int)h;
        }
    scalarNames.push_back(name);
    scalarTransport.push_back(transport);
    scalars.push_back(VectorXd());
    AllocateScalars();
    return (int)scalarNames.size() - 1;
}

long MAC::scalarOrderMismatches = 0;
long MAC::scalarSchemeConflicts = 0;

// The handles are the fast path and the names are the check. GRID_SOL and GRID_ANT are registered
// together, so handle `h` means the same field on both and this is one string comparison.
int MAC::MatchScalar(MAC &src, int h)
{
    const std::string &want = this->ScalarName(h);
    if (h < src.ScalarCount() && src.ScalarName(h) == want) return h;   // the usual case

    scalarOrderMismatches++;
    static bool said = false;
    if (!said)
    {
        said = true;
        std::cout << "WARNING: MAC::CopyGrid -- scalar handle " << h << " is '" << want
                  << "' on the destination but "
                  << (h < src.ScalarCount() ? ("'" + src.ScalarName(h) + "'") : "absent")
                  << " on the source. The two grids registered their fields in a different "
                     "order, so some part of the code is building MACs differently. Copying "
                     "by name instead; register the same fields in the same order on every "
                     "grid." << std::endl;
    }
    return src.ScalarHandle(want);
}

// Velocity, pressure and the solid mask. What a momentum solver computes, and what its internal
// sweep grids -- which carry no scalars -- write back into the solution.
void MAC::CopyFlowFields(MAC &grid) {
    if(is2D) {
        if(!grid.is2D){
            std::cout << "ERROR - MISMATCHE GRID COPY OPERATION (3D COPIED IN 2D)";
            return;
        }
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx; j++) {
                this->SetP(i, j, grid.GetP(i, j));
                this->SetSolid(i, j, grid.GetSolid(i, j));
            }
        }
        for (int i = 0; i < this->Ny; i++) {
            for (int j = 0; j < this->Nx + 1; j++) {
                this->SetU(i, j, grid.GetU(i, j));
            }
        }
        for (int i = 0; i < this->Ny + 1; i++) {
            for (int j = 0; j < this->Nx; j++) {
                this->SetV(i, j, grid.GetV(i, j));
            }
        }
    } else {
        if(grid.is2D){
            std::cout << "ERROR - MISMATCHE GRID COPY OPERATION (2D COPIED IN 3D)";
        }
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    this->SetP(i, j, k, grid.GetP(i, j, k));
                    this->SetSolid(i, j, k, grid.GetSolid(i, j, k));
                }
            }
        }
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx + 1; j++) {
                    this->SetU(i, j, k, grid.GetU(i, j, k));
                }
            }
        }
        for (int k = 0; k < this->Nz; k++) {
            for (int i = 0; i < this->Ny + 1; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    this->SetV(i, j, k, grid.GetV(i, j, k));
                }
            }
        }
        for (int k = 0; k < this->Nz + 1; k++) {
            for (int i = 0; i < this->Ny; i++) {
                for (int j = 0; j < this->Nx; j++) {
                    this->SetW(i, j, k, grid.GetW(i, j, k));
                }
            }
        }
    }
}

// The registered scalars.
void MAC::CopyScalarFields(MAC &grid) {
    if (is2D != grid.is2D) {
        std::cout << "ERROR - MISMATCHED SCALAR COPY (2D and 3D grids)";
        return;
    }
    for (int h = 0; h < this->ScalarCount(); h++) {
        const int src = this->MatchScalar(grid, h);
        if (src < 0) continue;          // the source simply does not carry this field
        if (is2D) {
            for (int i = 0; i < this->Ny; i++)
                for (int j = 0; j < this->Nx; j++)
                    this->SetS(h, i, j, grid.GetS(src, i, j));
        } else {
            for (int k = 0; k < this->Nz; k++)
                for (int i = 0; i < this->Ny; i++)
                    for (int j = 0; j < this->Nx; j++)
                        this->SetS(h, i, j, k, grid.GetS(src, i, j, k));
        }
    }
}

// Strides of each field along each axis (x, y, z), so an entry of field c at indices q = {j, i, k} is
// q[0]*s[0] + q[1]*s[1] + q[2]*s[2] -- the same linear index GetU/GetV/GetW/GetP compute.
void MAC::BuildStrides()
{
    for (int c = U; c <= P; c++)
    {
        const int nx = (c == U) ? Nx + 1 : Nx;
        if (is2D)
        {
            strideTable[c][X] = 1;
            strideTable[c][Y] = nx;
            strideTable[c][Z] = 0;
        }
        else
        {
            const int nz = (c == W) ? Nz + 1 : Nz;
            strideTable[c][X] = nz;
            strideTable[c][Y] = nx * nz;
            strideTable[c][Z] = 1;
        }
    }
}

// Component c at a face of `target`: its two faces along its own axis and the two cells along target's axis,
// averaged along target's axis first (the order the getXatY always used, so bit-identical).
double MAC::ComponentAtFace(Component c, Component target, int i, int j, int k)
{
    const Axis own = AxisOf(c), across = AxisOf(target);
    const int *st = strideTable[c];
    int q[3] = {j, i, k};
    const int n = q[across];
    q[across] = n - 1;
    const int base = q[X] * st[X] + q[Y] * st[Y] + q[Z] * st[Z];
    const double *f = Field(c).data();
    const double lowBehind = f[base], lowAhead = f[base + st[across]];
    const double highBehind = f[base + st[own]], highAhead = f[base + st[own] + st[across]];
    if (nonUniform)
    {
        const double gap = FaceGap(across, n - 1);
        const double wBehind = (Width(across, n) * 0.5) / gap;
        const double wAhead = (Width(across, n - 1) * 0.5) / gap;
        const double low = wBehind * lowBehind + wAhead * lowAhead;
        const double high = wBehind * highBehind + wAhead * highAhead;
        return (low + high) / 2.0;
    }
    const double low = (lowBehind + lowAhead) / 2.0;
    const double high = (highBehind + highAhead) / 2.0;
    return (low + high) / 2.0;
}

void MAC::InitializeFromScene(double t)
{
    ApplyInitialField();
    if (is2D)
        SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction2D, SIMULATION.PressureBoundaryFunction2D);
    else
        SetBoundaryFunctions(SIMULATION.VelocityBoundaryFunction, SIMULATION.PressureBoundaryFunction);
    EnforceBoundaryConditions(t);
}

void MAC::InitializeFromGrid(MAC &model, double t)
{
    InitializeGrid(model.omega, model.is2D);
    CopyFlowFields(model);
    boundaryVelocity2D = model.boundaryVelocity2D;
    boundaryPressure2D = model.boundaryPressure2D;
    boundaryVelocity3D = model.boundaryVelocity3D;
    boundaryPressure3D = model.boundaryPressure3D;
    EnforceBoundaryConditions(t);
}

void MAC::EnforceScalarBC(int h, double t)
{
    const std::string &name = scalarNames[(size_t)h];
    const int nz = is2D ? 1 : Nz;
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < nz; k++)
            {
                const bool inside = i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1 && (is2D || (k > 0 && k < Nz - 1));
                if (inside) continue;
                if ((is2D ? GetSolid(i, j) : GetSolid(i, j, k)) == FLUID_CELL) continue;   // a harness grid, no ring
                const int ii = (std::max)(1, (std::min)(i, Ny - 2)), jj = (std::max)(1, (std::min)(j, Nx - 2));
                const int kk = is2D ? 0 : (std::max)(1, (std::min)(k, Nz - 2));
                double value = is2D ? GetS(h, ii, jj) : GetS(h, ii, jj, kk);
                if ((ii != i) + (jj != j) + (kk != k) == 1)   // a face cell, not an edge or corner
                {
                    const double x = (jj != j) ? FacePos(X, (std::max)(j, jj)) : CentrePos(X, j);
                    const double y = (ii != i) ? FacePos(Y, (std::max)(i, ii)) : CentrePos(Y, i);
                    const double z = is2D ? SIMULATION.domain.z0
                                          : ((kk != k) ? FacePos(Z, (std::max)(k, kk)) : CentrePos(Z, k));
                    double bc = 0.0;
                    if (SCENE.loaded && SCENE.ScalarDirichletAt(name, x, y, z, t, bc)) value = bc;
                }
                if (is2D) SetS(h, i, j, value); else SetS(h, i, j, k, value);
            }
}

double MAC::ScalarSourceAt(int h, double x, double y, double z, double t) const
{
    if (!SCENE.loaded) return 0.0;
    const SceneDesc::Expr *s = SCENE.ScalarSource(SCENE.ScalarIndex(scalarNames[(size_t)h]));
    return s ? s->Eval(x, y, z, t) : 0.0;
}

void MAC::InitializeSimulationGrids()
{
    SIMULATION.GRID_ANT->InitializeFromScene(0.0);
    SIMULATION.GRID_SOL->InitializeFromScene(0.0);
}
