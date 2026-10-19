// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SCENE_H
#define SCENE_H

#include "Definitions.h"
#include "MeshHandler.h"

extern "C" {
#include "tinyexpr.h"
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Data-driven geometry and boundary conditions.
//The code is solid but a refactor on some parts might be good, especially in the SDF

namespace SceneDesc
{

enum Face { FACE_XMIN = 0, FACE_XMAX, FACE_YMIN, FACE_YMAX, FACE_ZMIN, FACE_ZMAX, FACE_COUNT };

// A face's type is a BoundaryKind (Definitions.h), the same one MAC applies.
using BCType = BoundaryKind;

enum PrimitiveKind { PRIM_BOX = 0, PRIM_SPHERE, PRIM_CYLINDER, PRIM_PLANE, PRIM_MESH, PRIM_AIRFOIL };
enum RegionKind { REGION_SOLID = 0, REGION_FLUID, REGION_EMPTY };

// A compiled expression plus the storage tinyexpr reads its variables from.
struct Expr
{
    std::string source;
    bool isConstant = true;
    double constant = 0.0;
    te_expr *compiled = nullptr;

    // tinyexpr resolves variables to ADDRESSES at compile time, so this storage has to outlive the
    // tree and never move.
    mutable double vx = 0.0, vy = 0.0, vz = 0.0, vt = 0.0;
    mutable double vdh = 0.0, vLx = 0.0, vLy = 0.0, vLz = 0.0;

    Expr() = default;
    Expr(const Expr &) = delete;
    Expr &operator=(const Expr &) = delete;
    ~Expr() { if (compiled) te_free(compiled); }

    bool Compile(const std::string &text, std::string &err)
    {
        if (compiled) { te_free(compiled); compiled = nullptr; }
        isConstant = true;
        constant = 0.0;
        source = text;
        // A bare number stays a constant: no tree walk on the hot path.
        try
        {
            size_t used = 0;
            double v = std::stod(text, &used);
            while (used < text.size() && std::isspace((unsigned char)text[used])) used++;
            if (used == text.size())
            {
                isConstant = true;
                constant = v;
                return true;
            }
        }
        catch (...)
        {
        }

        const te_variable vars[] = {
            {"x", &vx, 0, nullptr},   {"y", &vy, 0, nullptr},   {"z", &vz, 0, nullptr},
            {"t", &vt, 0, nullptr},   {"dh", &vdh, 0, nullptr}, {"Lx", &vLx, 0, nullptr},
            {"Ly", &vLy, 0, nullptr}, {"Lz", &vLz, 0, nullptr},
        };
        int errPos = 0;
        compiled = te_compile(text.c_str(), vars, 8, &errPos);
        if (!compiled)
        {
            err = "expression \"" + text + "\" failed to parse near column " + std::to_string(errPos);
            return false;
        }
        isConstant = false;
        return true;
    }

    double Eval(double x, double y, double z, double t) const
    {
        if (isConstant)
            return constant;
        vx = x; vy = y; vz = z; vt = t;
        vdh = SIMULATION.dh;
        vLx = SIMULATION.domain.xf - SIMULATION.domain.x0;
        vLy = SIMULATION.domain.yf - SIMULATION.domain.y0;
        vLz = SIMULATION.domain.zf - SIMULATION.domain.z0;
        return te_eval(compiled);
    }
};

struct BoundarySpec
{
    BCType type = BC_NOSLIP;
    Expr u, v, w;      // prescribed velocity, for MOVING (tangential) and INLET
    Expr pressure;     // PRESSURE_OUTLET: the Dirichlet value; INLET: the pressure level
    bool hasPressure = false;
    bool clampNegativeU = false; // the wind tunnel inlet clamps its parabola at zero

    // Dirichlet values for declared scalars, by name.
    std::vector<std::string> scalarNames;
    std::vector<std::unique_ptr<Expr>> scalarValues;
    // Scalars whose prescribed value is clamped at zero on this face, as clampNegativeU is for
    // the velocity: `d_clamp_negative=1`. A parabola evaluated past its roots stays at zero.
    std::vector<std::string> scalarClampNames;
    bool specified = false;      // an unspecified face keeps the default no-slip
    int order = -1;              // position in the file; the highest one wins at a corner

    BoundarySpec() = default;
    BoundarySpec(const BoundarySpec &) = delete;
    BoundarySpec &operator=(const BoundarySpec &) = delete;

    void Clear()
    {
        std::string err;
        type = BC_NOSLIP;
        u.Compile("0", err);
        v.Compile("0", err);
        w.Compile("0", err);
        pressure.Compile("0", err);
        hasPressure = false;
        clampNegativeU = false;
        // A reparse must not inherit the previous scene's scalar values: the lookup takes the
        // FIRST match by name, so a stale entry would shadow the new one.
        scalarNames.clear();
        scalarValues.clear();
        scalarClampNames.clear();
        specified = false;
        order = -1;
    }
};

// A symmetric NACA 00XX section, leading edge at X = 0: y_t = 5 t c (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2
// + 0.2843 x^3 - a4 x^4), a4 = 0.1015 open / 0.1036 closed (Abbott & von Doenhoff 1959). Owned by the Scene.
//this part and the next is ugly because it supports rotation and some other shenenigans, like the rotation pivot, as different papers
//rotate the airfoil in different places, how I wish for standardized behaviour...
struct AirfoilSection
{
    double chord = 1.0;
    double thickness = 0.12;
    bool closedTE = false;
    std::vector<double> x, y;   // upper surface, X increasing from 0 to c

    double HalfThickness(double X) const
    {
        const double u = X / chord;
        const double a4 = closedTE ? 0.1036 : 0.1015;
        return 5.0 * thickness * chord *
               (0.2969 * std::sqrt(u) - 0.1260 * u - 0.3516 * u * u + 0.2843 * u * u * u - a4 * u * u * u * u);
    }

    void Build(double c, double t, bool closed, int n = 2001)
    {
        chord = c; thickness = t; closedTE = closed;
        x.resize(n); y.resize(n);
        for (int m = 0; m < n; m++)
        {
            const double theta = 3.14159265358979323846 * m / (n - 1);
            x[m] = (m == n - 1) ? c : 0.5 * c * (1.0 - std::cos(theta));
            y[m] = HalfThickness(x[m]);
        }
        y[0] = 0.0;
    }

    double MaxHalfThickness() const { return *std::max_element(y.begin(), y.end()); }

    bool Inside(double X, double Y) const
    {
        return X >= 0.0 && X <= chord && std::fabs(Y) < HalfThickness(X);
    }

    // Distance from (X, |Y|) to the section's outline. Only the segments within the distance to the surface point
    // above X can be nearer, and the upper surface is sorted in X, so those are found by bisection.
    double UnsignedDistance(double X, double Y) const
    {
        Y = std::fabs(Y);
        const double Xc = (std::min)((std::max)(X, 0.0), chord);
        double best = std::hypot(X - Xc, Y - HalfThickness(Xc));
        const double reach = best * 1.0001 + 1e-9 * chord;
        const int n = (int)x.size();
        int lo = (int)(std::lower_bound(x.begin(), x.end(), X - reach) - x.begin()) - 1;
        int hi = (int)(std::upper_bound(x.begin(), x.end(), X + reach) - x.begin());
        lo = (std::max)(lo, 0);
        hi = (std::min)(hi, n - 1);
        for (int m = lo; m < hi; m++)
        {
            const double ex = x[m + 1] - x[m], ey = y[m + 1] - y[m];
            const double len2 = ex * ex + ey * ey;
            double f = (len2 > 0.0) ? ((X - x[m]) * ex + (Y - y[m]) * ey) / len2 : 0.0;
            f = (std::min)(1.0, (std::max)(0.0, f));
            best = (std::min)(best, std::hypot(X - x[m] - f * ex, Y - y[m] - f * ey));
        }
        const double yte = y[n - 1];   // the open form's blunt trailing edge, X = c, 0 <= Y <= yte
        if (yte > 0.0)
            best = (std::min)(best, (Y <= yte) ? std::fabs(X - chord) : std::hypot(X - chord, Y - yte));
        return best;
    }
};

// A signed-distance primitive. Negative inside, positive outside; the sign is all the
// voxel solvers need, but keeping it a true distance leaves the door open for cut cells.
struct Primitive
{
    PrimitiveKind kind = PRIM_BOX;
    RegionKind region = REGION_SOLID;
    bool invert = false;

    double p0[3] = {0, 0, 0}; // box min / centre / plane point
    double p1[3] = {0, 0, 0}; // box max / plane normal
    double radius = 0.0;
    int axis = 2; // cylinder axis: 0 = x, 1 = y, 2 = z

    // PRIM_MESH: the placed model, owned by Scene::meshes.
    const MeshSDF *mesh = nullptr;

    // PRIM_AIRFOIL: the section (owned by Scene::airfoils). p0 is the pivot, `pivot` chords from the leading
    // edge; span along local z (0 = unbounded); rotate in degrees, R = Rz Ry Rx.
    const AirfoilSection *airfoil = nullptr;
    double rotate[3] = {0, 0, 0};
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double span = 0.0;
    double pivot = 0.5;

    // A scene point in the airfoil's frame: X from the leading edge along the chord, Y across it, Z along the span.
    void AirfoilLocal(double x, double y, double z, double &X, double &Y, double &Z) const
    {
        const double r0 = x - p0[0], r1 = y - p0[1], r2 = (DIMENSION == 2) ? 0.0 : z - p0[2];
        X = R[0] * r0 + R[3] * r1 + R[6] * r2 + pivot * airfoil->chord;
        Y = R[1] * r0 + R[4] * r1 + R[7] * r2;
        Z = R[2] * r0 + R[5] * r1 + R[8] * r2;
    }

    // Distance from a local point to the section's box: a lower bound on the distance to the airfoil.
    double AirfoilBoxDistance(double X, double Y, double Z) const
    {
        const double dx = (std::max)((std::max)(-X, X - airfoil->chord), 0.0);
        const double dy = (std::max)(std::fabs(Y) - airfoil->MaxHalfThickness(), 0.0);
        const double dz = (span > 0.0 && DIMENSION == 3) ? (std::max)(std::fabs(Z) - 0.5 * span, 0.0) : 0.0;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    double Distance(double x, double y, double z) const
    {
        // In 2D the z axis is not part of the geometry.
        const int axes = (DIMENSION == 2) ? 2 : 3;

        double d = 0.0;
        switch (kind)
        {
        case PRIM_BOX:
        {
            const double c[3] = {x, y, z};
            double outside = 0.0;
            double inside = -1e30;
            for (int a = 0; a < axes; a++)
            {
                const double dlo = p0[a] - c[a];
                const double dhi = c[a] - p1[a];
                const double dm = (std::max)(dlo, dhi);
                if (dm > 0.0) outside += dm * dm;
                inside = (std::max)(inside, dm);
            }
            d = (outside > 0.0) ? std::sqrt(outside) : inside;
            break;
        }
        case PRIM_SPHERE:
        {
            const double dx = x - p0[0], dy = y - p0[1];
            const double dz = (axes == 3) ? (z - p0[2]) : 0.0;
            d = std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
            break;
        }
        case PRIM_CYLINDER:
        {
            // Infinite along its axis; the domain walls cap it.
            double a1, a2;
            if (axis == 0)      { a1 = y - p0[1]; a2 = z - p0[2]; }
            else if (axis == 1) { a1 = x - p0[0]; a2 = z - p0[2]; }
            else                { a1 = x - p0[0]; a2 = y - p0[1]; }
            d = std::sqrt(a1 * a1 + a2 * a2) - radius;
            break;
        }
        case PRIM_PLANE:
        {
            // Half-space: negative on the side the normal points away from.
            const double nx = p1[0], ny = p1[1];
            const double nz = (axes == 3) ? p1[2] : 0.0;
            const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
            const double inv = (len > 0.0) ? 1.0 / len : 0.0;
            const double dzc = (axes == 3) ? (z - p0[2]) : 0.0;
            d = ((x - p0[0]) * nx + (y - p0[1]) * ny + dzc * nz) * inv;
            break;
        }
        case PRIM_MESH:
            // 2D cuts the model at its mid-plane; see MeshSDF::SectionZ.
            d = mesh ? mesh->Distance(x, y, (axes == 3) ? z : mesh->SectionZ()) : 1e30;
            break;
        case PRIM_AIRFOIL:
        {
            // The section's signed distance, extruded along the span; a finite span caps it with flat ends.
            if (!airfoil) { d = 1e30; break; }
            double X, Y, Z;
            AirfoilLocal(x, y, z, X, Y, Z);
            const double ds = airfoil->Inside(X, Y) ? -airfoil->UnsignedDistance(X, Y) : airfoil->UnsignedDistance(X, Y);
            if (span > 0.0 && axes == 3)
            {
                const double dz = std::fabs(Z) - 0.5 * span;
                d = (std::min)((std::max)(ds, dz), 0.0) + std::hypot((std::max)(ds, 0.0), (std::max)(dz, 0.0));
            }
            else
                d = ds;
            break;
        }
        }
        return invert ? -d : d;
    }

    // Distance, exact where it is below `cap` and otherwise some value >= cap. Only a mesh has
    // anything to skip; an inverted mesh gets no bound, since its distance is the negated one.
    double DistanceBelow(double x, double y, double z, double cap) const
    {
        if (kind == PRIM_MESH && mesh && !invert && DIMENSION == 3)
            return mesh->DistanceBelow(x, y, z, cap);
        if (kind == PRIM_AIRFOIL && airfoil && !invert)
        {
            double X, Y, Z;
            AirfoilLocal(x, y, z, X, Y, Z);
            const double bound = AirfoilBoxDistance(X, Y, Z);
            if (bound >= cap) return bound;
        }
        return Distance(x, y, z);
    }

    bool Contains(double x, double y, double z) const
    {
        // The inside test only needs the sign, and outside the model's bounding box that
        // is known without walking the BVH -- which is every cell of the domain but a few.
        if (kind == PRIM_MESH && mesh && mesh->OutsideBox(x, y, z, DIMENSION == 3))
            return invert;
        if (kind == PRIM_AIRFOIL && airfoil)
        {
            // Only the sign: the thickness formula, no distance search.
            double X, Y, Z;
            AirfoilLocal(x, y, z, X, Y, Z);
            const bool in = airfoil->Inside(X, Y) && (span <= 0.0 || DIMENSION == 2 || std::fabs(Z) < 0.5 * span);
            return invert ? !in : in;
        }
        return Distance(x, y, z) < 0.0;
    }

    // Signed distance and outward unit normal for a mesh, from one query. 2D drops the
    // normal's z and renormalises. False (n zeroed) where the normal is undefined.
    bool MeshDistanceNormal(double x, double y, double z, double &phi, double n[3]) const
    {
        const bool is3D = (DIMENSION == 3);
        bool ok = mesh->DistanceNormal(x, y, is3D ? z : mesh->SectionZ(), phi, n);
        if (!is3D)
        {
            n[2] = 0.0;
            const double len = std::sqrt(n[0] * n[0] + n[1] * n[1]);
            if (len > 1e-30) { n[0] /= len; n[1] /= len; }
            else { n[0] = n[1] = 0.0; ok = false; }
        }
        if (invert) { phi = -phi; n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
        return ok;
    }
};

// The scene itself
class Scene
{
public:
    bool loaded = false;
    std::string path;
    std::string error;

    BoundarySpec faces[FACE_COUNT];

    // Scalars the scene declares, in order. A `scalar` line has to come before any boundary
    // that names it -- which is what lets the boundary parser tell a scalar key from a typo.
    struct ScalarSpec
    {
        std::string name;
        ScalarTransport transport;
        std::unique_ptr<Expr> source;   // S(x, y, z, t); null when the line gives none
    };
    std::vector<ScalarSpec> scalars;

    int ScalarIndex(const std::string &name) const
    {
        for (size_t i = 0; i < scalars.size(); i++)
            if (scalars[i].name == name) return (int)i;
        return -1;
    }
    std::vector<Primitive> regions;
    // The placed models behind every `region ... mesh`. unique_ptr so the addresses the
    // regions hold survive the vector growing.
    std::vector<std::unique_ptr<MeshSDF>> meshes;
    std::vector<std::unique_ptr<AirfoilSection>> airfoils;

    // Point probes, in declaration order: `probe <name> <quantity> at=(x,y,z)`. Written to the telemetry
    // CSV as <name>_<quantity>. A scalar probe needs the scalar declared first, as a boundary does.
    std::vector<Probe> probes;

    // ---- body force, f(x, y, z, t) ---- `force u="..." v="..." w="..."`.
    Expr forceU, forceV, forceW;
    bool hasForce = false;

    // ---- initial velocity field, u(x, y, z, 0) ---- `initial u="..." v="..." w="..."`.
    Expr initU, initV, initW;
    bool hasInitial = false;

    // Where a cell is sampled for the region test.
    bool sampleAtCenter = false;

    int nextOrder = 0;

    void Reset()
    {
        for (int f = 0; f < FACE_COUNT; f++)
            faces[f].Clear();
        scalars.clear();
        regions.clear();
        meshes.clear();
        airfoils.clear();
        probes.clear();
        sampleAtCenter = false;
        hasForce = false;
        hasInitial = false;
        { std::string ignore; forceU.Compile("0", ignore);
                              forceV.Compile("0", ignore);
                              forceW.Compile("0", ignore);
                              initU.Compile("0", ignore);
                              initV.Compile("0", ignore);
                              initW.Compile("0", ignore); }
        nextOrder = 0;
        loaded = false;
        error.clear();
    }

    // ----- parsing ----- A scene reaches this class two ways: as its own file, and as a block
    // embedded in simulation_config.txt.
    bool Load(const std::string &file)
    {
        std::ifstream in(file);
        if (!in.is_open())
        {
            Reset();
            path = file;
            error = "could not open scene file '" + file + "'";
            return false;
        }
        return LoadStream(in, file);
    }

    bool LoadText(const std::string &text, const std::string &label = "<config>")
    {
        std::istringstream in(text);
        return LoadStream(in, label);
    }

    bool LoadStream(std::istream &in, const std::string &label)
    {
        Reset();
        path = label;

        std::string raw;
        int lineNo = 0;
        while (std::getline(in, raw))
        {
            lineNo++;
            const std::string line = StripComment(raw);
            if (line.find_first_not_of(" \t\r") == std::string::npos)
                continue;

            std::vector<std::string> tok = Tokenize(line);
            if (tok.empty())
                continue;

            if (tok[0] == "boundary")
            {
                if (!ParseBoundary(tok, lineNo))
                    return false;
            }
            else if (tok[0] == "region")
            {
                if (!ParseRegion(tok, lineNo))
                    return false;
            }
            else if (tok[0] == "scalar")
            {
                if (!ParseScalar(tok, lineNo))
                    return false;
            }
            else if (tok[0] == "force")
            {
                if (!ParseForce(tok, lineNo))
                    return false;
            }
            else if (tok[0] == "initial")
            {
                if (!ParseInitial(tok, lineNo))
                    return false;
            }
            else if (tok[0] == "probe")
            {
                if (!ParseProbe(tok, lineNo))
                    return false;
            }
            else if (tok[0] == "sampling")
            {
                if (tok.size() < 2) { error = Where(lineNo) + "sampling needs corner or center"; return false; }
                sampleAtCenter = (tok[1] == "center" || tok[1] == "centre");
            }
            else
            {
                error = Where(lineNo) + "unknown directive '" + tok[0] + "'";
                return false;
            }
        }

        // An outflow takes out what the inlets bring in; beside a pressure outlet that split is not defined.
        bool anyOutflow = false, anyPressureOutlet = false;
        for (int f = 0; f < FACE_COUNT; f++)
        {
            anyOutflow = anyOutflow || (faces[f].specified && faces[f].type == BC_OUTFLOW);
            anyPressureOutlet = anyPressureOutlet || (faces[f].specified && faces[f].type == BC_PRESSURE_OUTLET);
        }
        if (anyOutflow && anyPressureOutlet)
        {
            error = "an 'outflow' face takes out what the inlets bring in, which is not defined beside a "
                    "'pressure_outlet' (the pressure decides that face's flux); use one or the other";
            return false;
        }

        loaded = true;
        return true;
    }

    // True when face f is one of the six the current dimension has.
    static bool FaceActive(int f)
    {
        if (DIMENSION == 3) return true;
        return (f != FACE_ZMIN && f != FACE_ZMAX);
    }

    // Of all the faces a point lies on, the one declared last in the file. -1 if none.
    int WinningFace(const bool on[FACE_COUNT]) const
    {
        int best = -1, bestOrder = -2;
        for (int f = 0; f < FACE_COUNT; f++)
        {
            if (!on[f] || !FaceActive(f)) continue;
            if (faces[f].order > bestOrder) { bestOrder = faces[f].order; best = f; }
        }
        return best;
    }

    // The first or last cell size along an axis; dh on a uniform grid.
    static double EdgeCellSize(const std::vector<double> &sz, bool last)
    {
        if (!SIMULATION.spacing.nonUniform || sz.empty())
            return SIMULATION.dh;
        return last ? sz.back() : sz.front();
    }

    // Where the fluid box ends on an axis: the last interior face (the ring lies outside it). A grid can stop a sliver
    // short of the domain's end when the length is not a whole number of cells; before a grid is built, the domain's.
    static double InteriorEnd(const std::vector<double> &faces, double fallback)
    {
        return faces.size() >= 3 ? faces[faces.size() - 2] : fallback;
    }

    // Which faces a point lies on: on a face or beyond it (the ring lies outside the domain), to a
    // thousandth of the edge cell.
    void FacesAtPoint(double x, double y, double z, bool on[FACE_COUNT]) const
    {
        const Domain &d = SIMULATION.domain;
        const double tx0 = 1e-3 * EdgeCellSize(SIMULATION.spacing.dx, false);
        const double tx1 = 1e-3 * EdgeCellSize(SIMULATION.spacing.dx, true);
        const double ty0 = 1e-3 * EdgeCellSize(SIMULATION.spacing.dy, false);
        const double ty1 = 1e-3 * EdgeCellSize(SIMULATION.spacing.dy, true);
        const double tz0 = 1e-3 * EdgeCellSize(SIMULATION.spacing.dz, false);
        const double tz1 = 1e-3 * EdgeCellSize(SIMULATION.spacing.dz, true);
        on[FACE_XMIN] = (x <= d.x0 + tx0);
        on[FACE_XMAX] = (x >= InteriorEnd(SIMULATION.spacing.xf, d.xf) - tx1);
        on[FACE_YMIN] = (y <= d.y0 + ty0);
        on[FACE_YMAX] = (y >= InteriorEnd(SIMULATION.spacing.yf, d.yf) - ty1);
        // 2D has no z ring: z stays at the domain's z0 and both z faces read as they always did.
        on[FACE_ZMIN] = (z <= d.z0 + (DIMENSION == 3 ? tz0 : EdgeCellSize(SIMULATION.spacing.dz, false) * 1.0001));
        on[FACE_ZMAX] = (z >= (DIMENSION == 3 ? InteriorEnd(SIMULATION.spacing.zf, d.zf) - tz1
                                              : d.zf - EdgeCellSize(SIMULATION.spacing.dz, true) * 1.0001));
    }

    void FacesAtCell(int i, int j, int k, bool on[FACE_COUNT]) const
    {
        on[FACE_XMIN] = (j == 0);
        on[FACE_XMAX] = (j == SIMULATION.Nx - 1);
        on[FACE_YMIN] = (i == 0);
        on[FACE_YMAX] = (i == SIMULATION.Ny - 1);
        on[FACE_ZMIN] = (k == 0);
        on[FACE_ZMAX] = (k == SIMULATION.Nz - 1);
    }

    // Index to physical coordinate.
    static double AxisCoord(const std::vector<double> &face,
                            const std::vector<double> &centre,
                            double origin, double dh, int idx, bool atCentre)
    {
        // The tables: with the ring outside the domain, cell 0 starts one cell before the domain's origin. Without
        // tables (no grid built yet) the same layout from the origin.
        const std::vector<double> &t = atCentre ? centre : face;
        if (t.empty())
            return origin - dh + (idx + (atCentre ? 0.5 : 0.0)) * dh;
        const int n = (int)t.size();
        return t[idx < 0 ? 0 : (idx >= n ? n - 1 : idx)];
    }

    // ----- evaluation: cell types -----
    // i is the row (y), j the column (x), k the slab (z), matching the MAC accessors.
    int CellTypeAt(int i, int j, int k) const
    {
        const double dh = SIMULATION.dh;
        const double x = AxisCoord(SIMULATION.spacing.xf, SIMULATION.spacing.xc,
                                   SIMULATION.domain.x0, dh, j, sampleAtCenter);
        const double y = AxisCoord(SIMULATION.spacing.yf, SIMULATION.spacing.yc,
                                   SIMULATION.domain.y0, dh, i, sampleAtCenter);
        const double z = AxisCoord(SIMULATION.spacing.zf, SIMULATION.spacing.zc,
                                   SIMULATION.domain.z0, dh, k, sampleAtCenter);

        // Interior regions first; a boundary face overrides them, so a solid ring always
        // wins over a fluid region that spills into it.
        int type = FLUID_CELL;
        for (const Primitive &p : regions)
        {
            if (!p.Contains(x, y, z))
                continue;
            switch (p.region)
            {
            case REGION_SOLID: type = SOLID_CELL; break;
            case REGION_FLUID: type = FLUID_CELL; break;
            case REGION_EMPTY: type = EMPTY_CELL; break;
            }
        }

        // The outer ring is one cell thick, exactly as every mask in Functions.h builds it.
        bool on[FACE_COUNT];
        FacesAtCell(i, j, k, on);
        const int win = WinningFace(on);
        if (win >= 0)
        {
            type = CellTypeOf(faces[win]);
        }
        return type;
    }

    // The ring cell type a boundary kind puts on the ring. A pressure outlet with no p= is the p = 0 Dirichlet cell
    // FLIP's air also uses; an outflow's velocity is known before the solve, so the pressure treats it as an inlet.
    static int CellTypeOf(const BoundarySpec &b)
    {
        switch (b.type)
        {
        case BC_INLET:            return INFLOW_CELL;
        case BC_OUTFLOW:          return OUTFLOW_CELL;
        case BC_PRESSURE_OUTLET:  return b.hasPressure ? PRESSURE_CELL : EMPTY_CELL;
        default:                  return SOLID_CELL;
        }
    }

    // The kind of ring cell (i, j, k) and the face that owns it -- the one declared last among the faces it lies
    // on. -1 off the ring.
    int BoundaryKindAtCell(int i, int j, int k, int &face) const
    {
        bool on[FACE_COUNT];
        FacesAtCell(i, j, k, on);
        face = WinningFace(on);
        return face < 0 ? -1 : (int)faces[face].type;
    }

    // ----- evaluation: prescribed velocity ----- Coordinates are domain-relative metres.
    void VelocityAt(double x, double y, double z, double t, double &u, double &v, double &w) const
    {
        u = v = w = 0.0;

        bool on[FACE_COUNT];
        FacesAtPoint(x, y, z, on);
        const int win = WinningFace(on);
        if (win < 0)
            return;

        const BoundarySpec &b = faces[win];
        if (b.type == BC_MOVING || b.type == BC_INLET)
        {
            u = b.u.Eval(x, y, z, t);
            v = b.v.Eval(x, y, z, t);
            w = b.w.Eval(x, y, z, t);
            if (b.clampNegativeU && u < 0.0)
                u = 0.0;
        }
        // Every other kind prescribes zero here; what is zero-gradient on it (slip tangential, outlet and outflow
        // velocity) MAC copies from the interior after these values are written.
    }

    // The pressure LEVEL carried by an inflow face's p=: velocity pinned, so the pressure there is Neumann
    // and the expression only fixes the constant of a singular system. False on any other face.
    bool PressureLevelAt(double x, double y, double z, double t, double &p) const
    {
        bool on[FACE_COUNT];
        FacesAtPoint(x, y, z, on);
        const int win = WinningFace(on);
        if (win < 0 || faces[win].type != BC_INLET || !faces[win].hasPressure)
            return false;
        p = faces[win].pressure.Eval(x, y, z, t);
        return true;
    }

    double PressureAt(double x, double y, double z, double t) const
    {
        bool on[FACE_COUNT];
        FacesAtPoint(x, y, z, on);
        const int win = WinningFace(on);
        if (win >= 0 && faces[win].hasPressure)
            return faces[win].pressure.Eval(x, y, z, t);
        return 0.0;
    }

    // Does the face at this point pin `name`? False means zero gradient, which is the zero-flux
    // wall and the outflow both.
    bool ScalarDirichletAt(const std::string &name, double x, double y, double z, double t,
                           double &value) const
    {
        bool on[FACE_COUNT];
        FacesAtPoint(x, y, z, on);
        const int win = WinningFace(on);
        if (win < 0) return false;
        const BoundarySpec &b = faces[win];
        for (size_t i = 0; i < b.scalarNames.size(); i++)
            if (b.scalarNames[i] == name)
            {
                value = b.scalarValues[i]->Eval(x, y, z, t);
                if (value < 0.0 && ScalarClampedOn(b, name))
                    value = 0.0;
                return true;
            }
        return false;
    }

    // The `source` expression of declared scalar `index` (ScalarIndex), null when it has none.
    // Evaluated serially by the transport solvers: Expr binds its variables to fixed addresses.
    const Expr *ScalarSource(int index) const
    {
        if (index < 0 || index >= (int)scalars.size()) return nullptr;
        return scalars[(size_t)index].source.get();
    }

    static bool ScalarClampedOn(const BoundarySpec &b, const std::string &name)
    {
        for (const std::string &c : b.scalarClampNames)
            if (c == name) return true;
        return false;
    }

    // ----- evaluation: where FLIP should seed particles ----- Independent of CellTypeAt: for FLIP
    // the initial FLUID/EMPTY split does not survive the first UpdateFluidCells (which resets.
    bool HasFluidRegions() const
    {
        for (const Primitive &p : regions)
            if (p.region == REGION_FLUID)
                return true;
        return false;
    }

    bool IsFluidSeedAt(double x, double y, double z) const
    {
        bool matched = false;
        RegionKind last = REGION_EMPTY;
        for (const Primitive &p : regions)
        {
            if (!p.Contains(x, y, z))
                continue;
            matched = true;
            last = p.region;
        }
        return matched && last == REGION_FLUID;
    }

    // ----- evaluation: interior solid geometry, for particle collision ----- The mask tells a
    // solver which cells are solid.
    bool HasSolidRegions() const
    {
        for (const Primitive &p : regions)
            if (p.region == REGION_SOLID)
                return true;
        return false;
    }

    // Index of the last region containing the point, -1 if none.
    int WinningRegion(double x, double y, double z) const
    {
        int win = -1;
        for (size_t r = 0; r < regions.size(); r++)
            if (regions[r].Contains(x, y, z))
                win = (int)r;
        return win;
    }

    // phi < 0 inside the solid, n is the outward unit normal.
    bool SolidDistanceAt(double x, double y, double z, double &phi, double n[3]) const
    {
        const int w = WinningRegion(x, y, z);
        if (w < 0 || regions[w].region != REGION_SOLID)
            return false;

        const Primitive &prim = regions[w];
        if (prim.kind == PRIM_MESH && prim.mesh)
        {
            prim.MeshDistanceNormal(x, y, z, phi, n);
            return true;
        }
        phi = prim.Distance(x, y, z);

        // Central differences on an exact signed distance.
        const double e = (SIMULATION.dh > 0.0) ? 1e-4 * SIMULATION.dh : 1e-8;
        n[0] = prim.Distance(x + e, y, z) - prim.Distance(x - e, y, z);
        n[1] = prim.Distance(x, y + e, z) - prim.Distance(x, y - e, z);
        n[2] = (DIMENSION == 3) ? (prim.Distance(x, y, z + e) - prim.Distance(x, y, z - e)) : 0.0;

        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 1e-30)
        {
            n[0] /= len; n[1] /= len; n[2] /= len;
        }
        else
        {
            n[0] = n[1] = n[2] = 0.0;
        }
        return true;
    }

    // Nearest solid surface, working from OUTSIDE as well as inside. SolidDistanceAt above answers
    // only for a point a solid region already Contains, because it goes through WinningRegion.
    bool NearestSolidSurface(double x, double y, double z, double &phi, double n[3]) const
    {
        int best = -1;
        double bestPhi = 0.0;
        for (size_t r = 0; r < regions.size(); r++)
        {
            if (regions[r].region != REGION_SOLID) continue;
            const double d = regions[r].Distance(x, y, z);
            if (best < 0 || d < bestPhi) { best = (int)r; bestPhi = d; }
        }
        if (best < 0)
            return false;

        // A later `region fluid` containing this point has carved the solid away here.
        for (size_t r = (size_t)best + 1; r < regions.size(); r++)
            if (regions[r].region == REGION_FLUID && regions[r].Contains(x, y, z))
                return false;

        const Primitive &prim = regions[best];
        phi = bestPhi;
        if (prim.kind == PRIM_MESH && prim.mesh)
            return prim.MeshDistanceNormal(x, y, z, phi, n);

        const double e = (SIMULATION.dh > 0.0) ? 1e-4 * SIMULATION.dh : 1e-8;
        n[0] = prim.Distance(x + e, y, z) - prim.Distance(x - e, y, z);
        n[1] = prim.Distance(x, y + e, z) - prim.Distance(x, y - e, z);
        n[2] = (DIMENSION == 3) ? (prim.Distance(x, y, z + e) - prim.Distance(x, y, z - e)) : 0.0;

        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len <= 1e-30)
        {
            n[0] = n[1] = n[2] = 0.0;
            return false;
        }
        n[0] /= len; n[1] /= len; n[2] /= len;
        return true;
    }

    // The solid geometry as a plain signed distance: negative inside a solid, positive outside,
    // +1e30 where there is no solid at all.
    double SolidFieldAt(double x, double y, double z) const
    {
        int best = -1;
        double bestPhi = 0.0;
        for (size_t r = 0; r < regions.size(); r++)
        {
            if (regions[r].region != REGION_SOLID) continue;
            const double d = regions[r].Distance(x, y, z);
            if (best < 0 || d < bestPhi) { best = (int)r; bestPhi = d; }
        }
        if (best < 0) return 1e30;
        for (size_t r = (size_t)best + 1; r < regions.size(); r++)
            if (regions[r].region == REGION_FLUID && regions[r].Contains(x, y, z))
                return 1e30;
        return bestPhi;
    }

    // SolidFieldAt, exact wherever the answer is below `cap` and otherwise some value >= cap; same winner,
    // same carve.
    double SolidFieldBelow(double x, double y, double z, double cap) const
    {
        int best = -1;
        double bestPhi = 0.0;
        for (size_t r = 0; r < regions.size(); r++)
        {
            if (regions[r].region != REGION_SOLID) continue;
            const double bound = (best < 0) ? cap : (std::min)(cap, bestPhi);
            const double d = regions[r].DistanceBelow(x, y, z, bound);
            if (best < 0 || d < bestPhi) { best = (int)r; bestPhi = d; }
        }
        if (best < 0) return 1e30;
        if (bestPhi >= cap) return bestPhi;
        for (size_t r = (size_t)best + 1; r < regions.size(); r++)
            if (regions[r].region == REGION_FLUID && regions[r].Contains(x, y, z))
                return 1e30;
        return bestPhi;
    }

    static const char *BoundaryName(int kind)
    {
        static const char *n[] = {"noslip", "slip", "moving", "inlet", "pressure_outlet", "outflow"};
        return (kind >= 0 && kind < BC_KIND_COUNT) ? n[kind] : "?";
    }

    std::string Summary() const
    {
        static const char *faceNames[] = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        std::ostringstream os;
        os << "scene '" << path << "': ";
        for (int f = 0; f < FACE_COUNT; f++)
            os << faceNames[f] << "=" << BoundaryName(faces[f].type) << (f + 1 < FACE_COUNT ? " " : "");
        os << " | " << regions.size() << " region(s)"
           << " | sampling=" << (sampleAtCenter ? "center" : "corner");
        return os.str();
    }

private:
    static std::string Where(int line) { return "scene line " + std::to_string(line) + ": "; }

    static std::string StripComment(const std::string &s)
    {
        // '#' starts a comment, but not inside a quoted expression.
        bool inQuote = false;
        for (size_t i = 0; i < s.size(); i++)
        {
            if (s[i] == '"') inQuote = !inQuote;
            else if (s[i] == '#' && !inQuote) return s.substr(0, i);
        }
        return s;
    }

    // Splits on whitespace but keeps "quoted expressions" and (parenthesised,tuples) whole.
    static std::vector<std::string> Tokenize(const std::string &s)
    {
        std::vector<std::string> out;
        std::string cur;
        bool inQuote = false;
        int depth = 0;
        for (char c : s)
        {
            if (c == '"') { inQuote = !inQuote; continue; }
            if (!inQuote)
            {
                if (c == '(') depth++;
                if (c == ')') depth--;
                if ((c == ' ' || c == '\t' || c == '\r') && depth == 0)
                {
                    if (!cur.empty()) { out.push_back(cur); cur.clear(); }
                    continue;
                }
            }
            cur.push_back(c);
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    }

    static bool SplitParam(const std::string &tok, std::string &key, std::string &value)
    {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) return false;
        key = tok.substr(0, eq);
        value = tok.substr(eq + 1);
        return true;
    }

    static bool ParseTriple(const std::string &v, double out[3])
    {
        std::string s = v;
        if (!s.empty() && s.front() == '(') s.erase(s.begin());
        if (!s.empty() && s.back() == ')') s.pop_back();
        std::replace(s.begin(), s.end(), ',', ' ');
        std::istringstream is(s);
        int n = 0;
        double vals[3] = {0, 0, 0};
        while (n < 3 && (is >> vals[n])) n++;
        if (n < 2) return false;
        out[0] = vals[0]; out[1] = vals[1]; out[2] = (n >= 3) ? vals[2] : 0.0;
        return true;
    }

    // One number is a uniform scale; three are per axis; two leave z at 1 (a 2D scene).
    static bool ParseScale(const std::string &v, double out[3])
    {
        std::string s = v;
        if (!s.empty() && s.front() == '(') s.erase(s.begin());
        if (!s.empty() && s.back() == ')') s.pop_back();
        std::replace(s.begin(), s.end(), ',', ' ');
        std::istringstream is(s);
        double vals[3];
        int n = 0;
        while (n < 3 && (is >> vals[n])) n++;
        if (n == 1) { out[0] = out[1] = out[2] = vals[0]; return true; }
        if (n == 2) { out[0] = vals[0]; out[1] = vals[1]; out[2] = 1.0; return true; }
        if (n == 3) { out[0] = vals[0]; out[1] = vals[1]; out[2] = vals[2]; return true; }
        return false;
    }

    // Shared by `force` and `initial`: `<kw> u="<expr>" v="<expr>" w="<expr>"`, any subset.
    bool ParseVectorExpr(const std::vector<std::string> &tok, int lineNo, const char *kw,
                         Expr &eu, Expr &ev, Expr &ew, bool &flag)
    {
        for (size_t i = 1; i < tok.size(); i++)
        {
            const size_t eq = tok[i].find('=');
            if (eq == std::string::npos)
            { error = Where(lineNo) + kw + " takes u=, v= or w="; return false; }
            const std::string k = tok[i].substr(0, eq);
            std::string v = tok[i].substr(eq + 1);
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"')
                v = v.substr(1, v.size() - 2);

            std::string err;
            Expr *target = (k == "u") ? &eu : (k == "v") ? &ev : (k == "w") ? &ew : nullptr;
            if (!target)
            { error = Where(lineNo) + kw + ": unknown component '" + k + "'"; return false; }
            if (!target->Compile(v, err))
            { error = Where(lineNo) + kw + " " + k + ": " + err; return false; }
        }
        flag = true;
        return true;
    }

    bool ParseProbe(const std::vector<std::string> &tok, int line)
    {
        if (tok.size() < 4)
        {
            error = Where(line) + "probe needs a name, a quantity and at=(x,y,z)";
            return false;
        }
        Probe pr;
        pr.name = tok[1];
        pr.quantity = tok[2];
        if (pr.name.find_first_of(",\"") != std::string::npos)
        {
            error = Where(line) + "probe name '" + pr.name + "' cannot hold a comma or a quote";
            return false;
        }
        for (const Probe &other : probes)
            if (other.name == pr.name)
            {
                error = Where(line) + "probe '" + pr.name + "' is declared twice";
                return false;
            }

        if (pr.quantity == "p" || pr.quantity == "pressure") { pr.quantity = "p"; pr.kind = ProbeQuantity::Pressure; }
        else if (pr.quantity == "u") pr.kind = ProbeQuantity::U;
        else if (pr.quantity == "v") pr.kind = ProbeQuantity::V;
        else if (pr.quantity == "w") pr.kind = ProbeQuantity::W;
        else if (pr.quantity == "speed") pr.kind = ProbeQuantity::Speed;
        else if (pr.quantity == "level") pr.kind = ProbeQuantity::Level;
        else if (pr.quantity == "top") pr.kind = ProbeQuantity::Top;
        else if (ScalarIndex(pr.quantity) >= 0) pr.kind = ProbeQuantity::Scalar;
        else
        {
            error = Where(line) + "probe quantity '" + pr.quantity +
                    "' is not p, u, v, w, speed, level, top or a scalar declared above";
            return false;
        }

        bool placed = false;
        for (size_t a = 3; a < tok.size(); a++)
        {
            std::string k, v;
            if (!SplitParam(tok[a], k, v))
            {
                error = Where(line) + "expected key=value, got '" + tok[a] + "'";
                return false;
            }
            if (k == "at")
            {
                if (!ParseTriple(v, pr.at)) { error = Where(line) + "bad tuple '" + v + "'"; return false; }
                placed = true;
            }
            else if (k == "radius")
            {
                pr.radius = atof(v.c_str());
                if (!(pr.radius >= 0.0))
                {
                    error = Where(line) + "probe radius must be zero or positive, got '" + v + "'";
                    return false;
                }
            }
            else
            {
                error = Where(line) + "unknown probe key '" + k + "'";
                return false;
            }
        }
        if (!placed)
        {
            error = Where(line) + "probe '" + pr.name + "' needs at=(x,y,z)";
            return false;
        }
        probes.push_back(pr);
        return true;
    }

    bool ParseInitial(const std::vector<std::string> &tok, int lineNo)
    { return ParseVectorExpr(tok, lineNo, "initial", initU, initV, initW, hasInitial); }

    // `force u="<expr>" v="<expr>" w="<expr>"` -- any subset; the rest stay zero.
    bool ParseForce(const std::vector<std::string> &tok, int lineNo)
    { return ParseVectorExpr(tok, lineNo, "force", forceU, forceV, forceW, hasForce); }

    // The body force at a point, in m/s^2. Zero everywhere when no `force` line was given, so a
    // caller can add it unconditionally.
public:
    // Initial velocity. Same thread-safety rule as ForceAt: evaluate SERIALLY.
    void InitialVelocityAt(double x, double y, double z,
                           double &u, double &v, double &w) const
    {
        if (!hasInitial) { u = v = w = 0.0; return; }
        u = initU.Eval(x, y, z, 0.0);
        v = initV.Eval(x, y, z, 0.0);
        w = initW.Eval(x, y, z, 0.0);
    }

    void ForceAt(double x, double y, double z, double t,
                 double &fx, double &fy, double &fz) const
    {
        if (!hasForce) { fx = fy = fz = 0.0; return; }
        fx = forceU.Eval(x, y, z, t);
        fy = forceV.Eval(x, y, z, t);
        fz = forceW.Eval(x, y, z, t);
    }

private:
    static int FaceFromName(const std::string &n)
    {
        if (n == "xmin") return FACE_XMIN;
        if (n == "xmax") return FACE_XMAX;
        if (n == "ymin") return FACE_YMIN;
        if (n == "ymax") return FACE_YMAX;
        if (n == "zmin") return FACE_ZMIN;
        if (n == "zmax") return FACE_ZMAX;
        return -1;
    }

    // The names a scene writes; older spellings still read (freeslip, inflow, pressure, dirichlet). `outflow` is
    // the continuity outflow; a face that pins p is `pressure_outlet`.
    static int BCFromName(const std::string &n)
    {
        if (n == "noslip" || n == "wall") return BC_NOSLIP;
        if (n == "slip" || n == "freeslip") return BC_SLIP;
        if (n == "moving") return BC_MOVING;
        if (n == "inlet" || n == "inflow") return BC_INLET;
        if (n == "pressure_outlet" || n == "pressure" || n == "dirichlet") return BC_PRESSURE_OUTLET;
        if (n == "outflow") return BC_OUTFLOW;
        return -1;
    }

    bool ParseBoundary(const std::vector<std::string> &tok, int line)
    {
        if (tok.size() < 3)
        {
            error = Where(line) + "boundary needs a face and a type";
            return false;
        }
        const int f = FaceFromName(tok[1]);
        if (f < 0) { error = Where(line) + "unknown face '" + tok[1] + "'"; return false; }
        const int bc = BCFromName(tok[2]);
        if (bc < 0) { error = Where(line) + "unknown boundary type '" + tok[2] + "'"; return false; }

        // Built in place: an Expr cannot be copied without invalidating its variable
        // bindings, so there is no local to assign from.
        BoundarySpec &spec = faces[f];
        spec.Clear();
        spec.type = (BCType)bc;
        spec.specified = true;
        spec.order = nextOrder++;
        std::string err;
        bool normalGiven = false, velocityGiven = false;
        const char *normalKey = (f == FACE_XMIN || f == FACE_XMAX) ? "u" : ((f == FACE_YMIN || f == FACE_YMAX) ? "v" : "w");

        for (size_t a = 3; a < tok.size(); a++)
        {
            std::string k, v;
            if (!SplitParam(tok[a], k, v))
            {
                error = Where(line) + "expected key=value, got '" + tok[a] + "'";
                return false;
            }
            if (k == "u" || k == "v" || k == "w")
            {
                if (k == normalKey) normalGiven = true;
                velocityGiven = true;
                Expr &target = (k == "u") ? spec.u : (k == "v" ? spec.v : spec.w);
                if (!target.Compile(v, err)) { error = Where(line) + err; return false; }
            }
            else if (ScalarIndex(k) >= 0)
            {
                auto e = std::make_unique<Expr>();
                if (!e->Compile(v, err)) { error = Where(line) + err; return false; }
                spec.scalarNames.push_back(k);
                spec.scalarValues.push_back(std::move(e));
            }
            else if (k == "p" || k == "pressure")
            {
                if (!spec.pressure.Compile(v, err)) { error = Where(line) + err; return false; }
                spec.hasPressure = true;
            }
            else if (k == "clamp_negative" || k == "clamp")
            {
                spec.clampNegativeU = (v != "0" && v != "false");
            }
            else if (k.size() > SCALAR_CLAMP_SUFFIX.size() &&
                     k.compare(k.size() - SCALAR_CLAMP_SUFFIX.size(), SCALAR_CLAMP_SUFFIX.size(),
                               SCALAR_CLAMP_SUFFIX) == 0 &&
                     ScalarIndex(k.substr(0, k.size() - SCALAR_CLAMP_SUFFIX.size())) >= 0)
            {
                if (v != "0" && v != "false")
                    spec.scalarClampNames.push_back(k.substr(0, k.size() - SCALAR_CLAMP_SUFFIX.size()));
            }
            else
            {
                error = Where(line) + "unknown boundary parameter '" + k + "'";
                return false;
            }
        }

        // Each kind takes only what it prescribes. A face prescribes the velocity or the pressure, never both: with
        // the normal velocity pinned the projection cannot honour a Dirichlet pressure there, and p loses an order.
        const std::string name = std::string("'") + BoundaryName(spec.type) + "'";
        if (velocityGiven && spec.type != BC_INLET && spec.type != BC_MOVING)
        {
            error = Where(line) + "a " + name + " face takes no velocity: " +
                    (spec.type == BC_PRESSURE_OUTLET
                         ? "its velocity is zero-gradient and the pressure drives it; to prescribe both, use "
                           "'inlet' with p=, which pins the velocity and uses p only for the pressure level"
                         : (spec.type == BC_OUTFLOW ? "its velocity is zero-gradient, its flux set by the inlets"
                                                    : "a wall's velocity is zero (use 'moving' for a sliding wall)"));
            return false;
        }
        if (spec.type == BC_MOVING && normalGiven)
        {
            const Expr &n = (normalKey[0] == 'u') ? spec.u : (normalKey[0] == 'v' ? spec.v : spec.w);
            if (!n.isConstant || n.constant != 0.0)
            {
                error = Where(line) + "a 'moving' wall slides along itself: its normal velocity (" + normalKey +
                        ") must be 0; use 'inlet' to push fluid through a face";
                return false;
            }
        }
        if (spec.hasPressure && spec.type != BC_INLET && spec.type != BC_PRESSURE_OUTLET)
        {
            error = Where(line) + "a " + name + " face takes no p: only a 'pressure_outlet' pins the pressure (and "
                    "an 'inlet' may set its level)";
            return false;
        }

        return true;
    }

    static bool ParseWholeNumber(const std::string &v, double &out)
    {
        try
        {
            size_t used = 0;
            out = std::stod(v, &used);
            return used == v.size() && std::isfinite(out);
        }
        catch (...) { return false; }
    }

    // scalar <name> [solver=adi|none] [scheme=] [form=] [epsilon=] [gamma=] [source="<expr>"], declared before any boundary
    // that names it, so the boundary parser can tell a scalar key from a misspelt one.
    bool ParseScalar(const std::vector<std::string> &tok, int line)
    {
        if (tok.size() < 2)
        {
            error = Where(line) + "scalar needs a name";
            return false;
        }
        const std::string name = tok[1];
        if (name == "u" || name == "v" || name == "w" || name == "p" || name == "pressure")
        {
            error = Where(line) + "'" + name + "' is a reserved boundary key";
            return false;
        }
        if (ScalarIndex(name) >= 0)
        {
            error = Where(line) + "scalar '" + name + "' is declared twice";
            return false;
        }

        ScalarSpec spec;
        spec.name = name;
        for (size_t a = 2; a < tok.size(); a++)
        {
            std::string k, v;
            if (!SplitParam(tok[a], k, v))
            {
                error = Where(line) + "expected key=value, got '" + tok[a] + "'";
                return false;
            }
            if (k == "solver")
            {
                if (v == "adi") spec.transport.solver = ScalarSolver::Adi;
                else if (v == "none") spec.transport.solver = ScalarSolver::None;
                else
                {
                    error = Where(line) + "solver must be 'adi' or 'none', got '" + v + "'";
                    return false;
                }
            }
            else if (k == "scheme")
            {
                if (v == "upwind") spec.transport.scheme = TransportScheme::Upwind;
                else if (v == "central") spec.transport.scheme = TransportScheme::Central;
                else
                {
                    error = Where(line) + "scheme must be 'upwind' or 'central', got '" + v + "'";
                    return false;
                }
            }
            else if (k == "form")
            {
                if (v == "flux") spec.transport.form = TransportForm::Flux;
                else if (v == "advective") spec.transport.form = TransportForm::Advective;
                else
                {
                    error = Where(line) + "form must be 'flux' or 'advective', got '" + v + "'";
                    return false;
                }
            }
            else if (k == "epsilon")
            {
                if (!ParseWholeNumber(v, spec.transport.epsilon) || spec.transport.epsilon < 0.0)
                {
                    error = Where(line) + "epsilon (diffusivity) must be a number >= 0, got '" + v + "'";
                    return false;
                }
            }
            else if (k == "gamma")
            {
                if (!ParseWholeNumber(v, spec.transport.gamma))
                {
                    error = Where(line) + "gamma (reactivity) must be a number, got '" + v + "'";
                    return false;
                }
            }
            else if (k == "source")
            {
                if (v.find_first_not_of(" \t") == std::string::npos) { spec.source.reset(); continue; }
                spec.source = std::make_unique<Expr>();
                std::string err;
                if (!spec.source->Compile(v, err)) { error = Where(line) + "source: " + err; return false; }
            }
            else
            {
                error = Where(line) + "unknown scalar parameter '" + k + "'";
                return false;
            }
        }
        scalars.push_back(std::move(spec));
        return true;
    }

    bool ParseRegion(const std::vector<std::string> &tok, int line)
    {
        if (tok.size() < 3)
        {
            error = Where(line) + "region needs a kind and a primitive";
            return false;
        }

        Primitive p;
        std::string meshFile;
        MeshTransform xf;
        double airfoilThickness = 0.0, airfoilChord = 1.0;
        bool airfoilClosed = false;
        if (tok[1] == "solid") p.region = REGION_SOLID;
        else if (tok[1] == "fluid") p.region = REGION_FLUID;
        else if (tok[1] == "empty" || tok[1] == "air") p.region = REGION_EMPTY;
        else { error = Where(line) + "unknown region kind '" + tok[1] + "'"; return false; }

        if (tok[2] == "box") p.kind = PRIM_BOX;
        else if (tok[2] == "sphere") p.kind = PRIM_SPHERE;
        else if (tok[2] == "cylinder") p.kind = PRIM_CYLINDER;
        else if (tok[2] == "plane" || tok[2] == "halfspace") p.kind = PRIM_PLANE;
        else if (tok[2] == "mesh" || tok[2] == "stl") p.kind = PRIM_MESH;
        else if (tok[2].size() == 8 && tok[2].compare(0, 4, "naca") == 0 &&
                 std::isdigit((unsigned char)tok[2][4]) && std::isdigit((unsigned char)tok[2][5]) &&
                 std::isdigit((unsigned char)tok[2][6]) && std::isdigit((unsigned char)tok[2][7]))
        {
            // naca00XX: symmetric, thickness XX percent of the chord. A cambered section needs its mean line.
            if (tok[2][4] != '0' || tok[2][5] != '0')
            {
                error = Where(line) + "'" + tok[2] + "' is cambered; only symmetric sections, naca00XX, are supported";
                return false;
            }
            p.kind = PRIM_AIRFOIL;
            airfoilThickness = 0.01 * std::stoi(tok[2].substr(6, 2));
            if (airfoilThickness <= 0.0) { error = Where(line) + "'" + tok[2] + "' has no thickness"; return false; }
        }
        else { error = Where(line) + "unknown primitive '" + tok[2] + "'"; return false; }

        for (size_t a = 3; a < tok.size(); a++)
        {
            std::string k, v;
            if (!SplitParam(tok[a], k, v))
            {
                error = Where(line) + "expected key=value, got '" + tok[a] + "'";
                return false;
            }
            if (k == "min" || k == "center" || k == "centre" || k == "point")
            {
                if (!ParseTriple(v, p.p0)) { error = Where(line) + "bad tuple '" + v + "'"; return false; }
            }
            else if (k == "max" || k == "normal")
            {
                if (!ParseTriple(v, p.p1)) { error = Where(line) + "bad tuple '" + v + "'"; return false; }
            }
            else if (k == "radius" || k == "r")
            {
                try { p.radius = std::stod(v); }
                catch (...) { error = Where(line) + "bad radius '" + v + "'"; return false; }
            }
            else if (k == "axis")
            {
                p.axis = (v == "x") ? 0 : (v == "y") ? 1 : 2;
            }
            else if (k == "invert")
            {
                p.invert = (v != "0" && v != "false");
            }
            else if (k == "file" && p.kind == PRIM_MESH)
            {
                meshFile = v;
            }
            else if ((k == "rotate" || k == "rotation") && p.kind == PRIM_MESH)
            {
                if (!ParseTriple(v, xf.rotate)) { error = Where(line) + "bad tuple '" + v + "'"; return false; }
            }
            else if ((k == "rotate" || k == "rotation") && p.kind == PRIM_AIRFOIL)
            {
                if (!ParseTriple(v, p.rotate)) { error = Where(line) + "bad tuple '" + v + "'"; return false; }
            }
            else if (k == "chord" && p.kind == PRIM_AIRFOIL)
            {
                try { airfoilChord = std::stod(v); }
                catch (...) { error = Where(line) + "bad chord '" + v + "'"; return false; }
                if (!(airfoilChord > 0.0)) { error = Where(line) + "chord must be positive"; return false; }
            }
            else if (k == "pivot" && p.kind == PRIM_AIRFOIL)
            {
                try { p.pivot = std::stod(v); }
                catch (...) { error = Where(line) + "bad pivot '" + v + "'"; return false; }
            }
            else if (k == "span" && p.kind == PRIM_AIRFOIL)
            {
                try { p.span = std::stod(v); }
                catch (...) { error = Where(line) + "bad span '" + v + "'"; return false; }
            }
            else if ((k == "te" || k == "trailing_edge") && p.kind == PRIM_AIRFOIL)
            {
                if (v == "open") airfoilClosed = false;
                else if (v == "closed") airfoilClosed = true;
                else { error = Where(line) + "te must be 'open' or 'closed', got '" + v + "'"; return false; }
            }
            else if (k == "scale" && p.kind == PRIM_MESH)
            {
                if (!ParseScale(v, xf.scale)) { error = Where(line) + "bad scale '" + v + "'"; return false; }
            }
            else if (k == "components" && p.kind == PRIM_MESH)
            {
                // `split` signs each query against the nearest CONNECTED COMPONENT and unions them;
                // `whole` signs against the whole file at once.
                if (v == "split") xf.splitComponents = true;
                else if (v == "whole") xf.splitComponents = false;
                else { error = Where(line) + "components must be 'split' or 'whole', got '" + v + "'"; return false; }
            }
            else
            {
                error = Where(line) + "unknown region parameter '" + k + "'";
                return false;
            }
        }

        if (p.kind == PRIM_MESH)
        {
            if (meshFile.empty()) { error = Where(line) + "mesh needs file=<path to .stl>"; return false; }
            for (int a = 0; a < 3; a++) xf.center[a] = p.p0[a];
            auto m = std::make_unique<MeshSDF>();
            std::string err;
            if (!m->Build(meshFile, xf, err)) { error = Where(line) + err; return false; }
            p.mesh = m.get();
            meshes.push_back(std::move(m));
        }
        if (p.kind == PRIM_AIRFOIL)
        {
            auto a = std::make_unique<AirfoilSection>();
            a->Build(airfoilChord, airfoilThickness, airfoilClosed);
            MeshRotation(p.rotate, p.R);
            p.airfoil = a.get();
            airfoils.push_back(std::move(a));
        }

        regions.push_back(p);
        return true;
    }
};

} // namespace SceneDesc

// The one scene the run uses. Empty unless a SCENE_FILE was given.
inline SceneDesc::Scene SCENE;

// The generic triple. These have exactly the signatures Functions.h uses, so ConfigReader
// wires them into SIMULATION like any hardcoded case and no solver notices the difference.

inline int SCENE_SOLID_MASK(int i, int j, int k) { return SCENE.CellTypeAt(i, j, k); }
inline int SCENE_SOLID_MASK_2D(int i, int j) { return SCENE.CellTypeAt(i, j, 0); }

inline Vec3 SCENE_VELOCITY(double x, double y, double z, double t)
{
    Vec3 r;
    SCENE.VelocityAt(x, y, z, t, r.u, r.v, r.w);
    return r;
}

inline Vec2 SCENE_VELOCITY_2D(double x, double y, double t)
{
    double u, v, w;
    SCENE.VelocityAt(x, y, 0.0, t, u, v, w);
    Vec2 r;
    r.u = u;
    r.v = v;
    return r;
}

inline double SCENE_PRESSURE(double x, double y, double z, double t) { return SCENE.PressureAt(x, y, z, t); }
inline double SCENE_PRESSURE_2D(double x, double y, double t) { return SCENE.PressureAt(x, y, 0.0, t); }
inline int SCENE_BOUNDARY_KIND(int i, int j, int k, int &face) { return SCENE.BoundaryKindAtCell(i, j, k, face); }

#endif
