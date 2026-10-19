// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef MESH_HANDLER_H
#define MESH_HANDLER_H

//i will be honest, this is a copy paste of our internal experimental opengl render engine which
// couldnt make the cut for nereus, but the mesh loader was needed so... 
// Everything mesh, both directions.

// IMPORT: an STL file turned into a signed distance.

#include "Definitions.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "../../../include/stl_reader/stl_reader.h"
#include "../../../include/tmd/TriangleMeshDistance.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <functional>
#include <unordered_map>

#include <filesystem>

namespace SceneDesc
{

// The file as read: welded vertices and triangle indices, in the model's own units
// (an STL carries none -- CAD exports are usually millimetres).
struct RawMesh
{
    std::string file;
    std::vector<double> v;       // x y z per vertex
    std::vector<unsigned> tri;   // three indices per triangle
    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    double volume = 0.0;         // signed; flipped to positive on load
    bool flipped = false;        // the file's normals pointed inward and were reversed
    long long mtime = 0;

    size_t VertexCount() const { return v.size() / 3; }
    size_t TriangleCount() const { return tri.size() / 3; }
    double Centre(int a) const { return 0.5 * (lo[a] + hi[a]); }
};

// Read an STL, cached by path and modification time: the setup screen reparses the scene
// on every Validate, and there is no reason to re-read an unchanged file each time.
inline std::shared_ptr<const RawMesh> LoadRawMesh(const std::string &file, std::string &err)
{
    static std::map<std::string, std::shared_ptr<const RawMesh>> cache;

    std::error_code ec;
    std::filesystem::file_time_type stamp{};
    try { stamp = std::filesystem::last_write_time(file, ec); }
    catch (const std::exception &) { ec = std::make_error_code(std::errc::invalid_argument); }
    if (ec)
    {
        err = "model file '" + file + "' not found";
        return nullptr;
    }
    const long long mtime = (long long)stamp.time_since_epoch().count();
    auto hit = cache.find(file);
    if (hit != cache.end() && hit->second->mtime == mtime)
        return hit->second;

    auto m = std::make_shared<RawMesh>();
    m->file = file;
    m->mtime = mtime;
    std::vector<double> normals;
    std::vector<unsigned> solids;
    try
    {
        if (!stl_reader::ReadStlFile(file.c_str(), m->v, normals, m->tri, solids))
        {
            err = "could not read STL '" + file + "'";
            return nullptr;
        }
    }
    catch (const std::exception &e)
    {
        err = "could not read STL '" + file + "': " + e.what();
        return nullptr;
    }
    if (m->tri.empty())
    {
        err = "STL '" + file + "' has no triangles";
        return nullptr;
    }

    for (int a = 0; a < 3; a++) { m->lo[a] = 1e300; m->hi[a] = -1e300; }
    for (size_t i = 0; i < m->VertexCount(); i++)
        for (int a = 0; a < 3; a++)
        {
            m->lo[a] = (std::min)(m->lo[a], m->v[3 * i + a]);
            m->hi[a] = (std::max)(m->hi[a], m->v[3 * i + a]);
        }

    // Orientation. The pseudo-normal sign convention is "normals point OUT of the enclosed volume";
    // an inside-out file would make the whole domain solid and the model itself fluid.
    double vol = 0.0;
    for (size_t t = 0; t < m->TriangleCount(); t++)
    {
        const double *a = &m->v[3 * m->tri[3 * t + 0]];
        const double *b = &m->v[3 * m->tri[3 * t + 1]];
        const double *c = &m->v[3 * m->tri[3 * t + 2]];
        vol += a[0] * (b[1] * c[2] - b[2] * c[1])
             - a[1] * (b[0] * c[2] - b[2] * c[0])
             + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    vol /= 6.0;
    if (vol < 0.0)
    {
        for (size_t t = 0; t < m->TriangleCount(); t++)
            std::swap(m->tri[3 * t + 1], m->tri[3 * t + 2]);
        m->flipped = true;
        vol = -vol;
    }
    m->volume = vol;

    cache[file] = m;
    return m;
}

// Placement of a model in the scene. The PIVOT is the centre of the model's own bounding box: scale
// and rotation act about it, and `center` is where it lands.
struct MeshTransform
{
    double center[3] = {0, 0, 0};
    double rotate[3] = {0, 0, 0};
    double scale[3] = {1, 1, 1};

    // Sign each query against the connected component nearest to it and take the union, or against
    // the whole file at once.
    bool splitComponents = true;
};

inline void MeshRotation(const double deg[3], double R[9])
{
    const double k = 3.14159265358979323846 / 180.0;
    const double cx = std::cos(deg[0] * k), sx = std::sin(deg[0] * k);
    const double cy = std::cos(deg[1] * k), sy = std::sin(deg[1] * k);
    const double cz = std::cos(deg[2] * k), sz = std::sin(deg[2] * k);
    // Rz * Ry * Rx, row major.
    R[0] = cz * cy; R[1] = cz * sy * sx - sz * cx; R[2] = cz * sy * cx + sz * sx;
    R[3] = sz * cy; R[4] = sz * sy * sx + cz * cx; R[5] = sz * sy * cx - cz * sx;
    R[6] = -sy;     R[7] = cy * sx;                R[8] = cy * cx;
}

// Model coordinates -> scene coordinates.
inline void MeshTransformPoint(const RawMesh &m, const MeshTransform &xf, const double R[9],
                               const double in[3], double out[3])
{
    double s[3];
    for (int a = 0; a < 3; a++)
        s[a] = (in[a] - m.Centre(a)) * xf.scale[a];
    for (int r = 0; r < 3; r++)
        out[r] = R[3 * r + 0] * s[0] + R[3 * r + 1] * s[1] + R[3 * r + 2] * s[2] + xf.center[r];
}

// The whole model placed in the scene, as flat arrays -- what the distance structure is
// built from and what the setup screen draws.
inline void MeshWorldGeometry(const RawMesh &m, const MeshTransform &xf,
                              std::vector<double> &wv, std::vector<unsigned> &wt,
                              double lo[3], double hi[3])
{
    double R[9];
    MeshRotation(xf.rotate, R);
    wv.resize(m.v.size());
    for (int a = 0; a < 3; a++) { lo[a] = 1e300; hi[a] = -1e300; }
    for (size_t i = 0; i < m.VertexCount(); i++)
    {
        MeshTransformPoint(m, xf, R, &m.v[3 * i], &wv[3 * i]);
        for (int a = 0; a < 3; a++)
        {
            lo[a] = (std::min)(lo[a], wv[3 * i + a]);
            hi[a] = (std::max)(hi[a], wv[3 * i + a]);
        }
    }
    wt = m.tri;
    // A negative determinant (an odd number of negative scale factors) is a mirror, and a
    // mirror reverses the winding -- which would turn the model inside out.
    if (xf.scale[0] * xf.scale[1] * xf.scale[2] < 0.0)
        for (size_t t = 0; t < wt.size() / 3; t++)
            std::swap(wt[3 * t + 1], wt[3 * t + 2]);
}

// A placed model and its signed distance. Owned by the Scene; a Primitive only points at
// one, which keeps Primitive trivially copyable.
struct MeshSDF
{
    std::string file;              // as written in the scene, for the editor round trip
    MeshTransform xf;
    std::shared_ptr<const RawMesh> raw;
    std::vector<double> wv;        // vertices in scene coordinates
    std::vector<unsigned> wt;
    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};

    // ---- one distance structure per CONNECTED COMPONENT ----------------------
    // TriangleMeshDistance signs a query by the pseudo-normal of the NEAREST triangle.
    struct Part
    {
        std::vector<double> v;
        std::vector<unsigned> t;
        tmd::TriangleMeshDistance bvh;
        double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0}; // this component's own bounding box
    };
    std::vector<Part> parts;
    int componentCount = 0;   // components FOUND, whatever the mode; the editor reports it

    bool Build(const std::string &path, const MeshTransform &t, std::string &err)
    {
        file = path;
        xf = t;
        for (int a = 0; a < 3; a++)
            if (!(std::fabs(xf.scale[a]) > 0.0) || !std::isfinite(xf.scale[a]))
            {
                err = "model scale must be non-zero on every axis";
                return false;
            }
        raw = LoadRawMesh(path, err);
        if (!raw)
            return false;
        MeshWorldGeometry(*raw, xf, wv, wt, lo, hi);

        const size_t nv = wv.size() / 3, nt = wt.size() / 3;

        // Connected components over shared vertices. The placement is affine so it cannot
        // change connectivity; this is the file's own topology, measured once.
        std::vector<int> comp(nv, -1);
        componentCount = 0;
        {
            std::vector<std::vector<unsigned>> adj(nv);
            for (size_t k = 0; k < nt; k++)
            {
                const unsigned a = wt[3 * k], b = wt[3 * k + 1], c = wt[3 * k + 2];
                adj[a].push_back(b); adj[b].push_back(c); adj[c].push_back(a);
                adj[b].push_back(a); adj[c].push_back(b); adj[a].push_back(c);
            }
            std::vector<unsigned> stack;
            for (size_t v = 0; v < nv; v++)
            {
                if (comp[v] != -1) continue;
                comp[v] = componentCount;
                stack.push_back((unsigned)v);
                while (!stack.empty())
                {
                    const unsigned u = stack.back();
                    stack.pop_back();
                    for (unsigned w : adj[u])
                        if (comp[w] == -1) { comp[w] = componentCount; stack.push_back(w); }
                }
                componentCount++;
            }
        }

        const int nparts = (xf.splitComponents && componentCount > 1) ? componentCount : 1;
        parts.clear();
        parts.resize((size_t)nparts);
        if (nparts == 1)
        {
            parts[0].v = wv;
            parts[0].t = wt;
        }
        else
        {
            std::vector<std::vector<int>> remap((size_t)nparts, std::vector<int>(nv, -1));
            for (size_t k = 0; k < nt; k++)
            {
                const int c = comp[wt[3 * k]];
                unsigned idx[3];
                for (int e = 0; e < 3; e++)
                {
                    const unsigned g = wt[3 * k + e];
                    if (remap[(size_t)c][g] < 0)
                    {
                        remap[(size_t)c][g] = (int)(parts[(size_t)c].v.size() / 3);
                        parts[(size_t)c].v.push_back(wv[3 * g + 0]);
                        parts[(size_t)c].v.push_back(wv[3 * g + 1]);
                        parts[(size_t)c].v.push_back(wv[3 * g + 2]);
                    }
                    idx[e] = (unsigned)remap[(size_t)c][g];
                }
                parts[(size_t)c].t.push_back(idx[0]);
                parts[(size_t)c].t.push_back(idx[1]);
                parts[(size_t)c].t.push_back(idx[2]);
            }
        }

        for (Part &pt : parts)
        {
            for (int a = 0; a < 3; a++) { pt.lo[a] = 1e300; pt.hi[a] = -1e300; }
            for (size_t v = 0; v < pt.v.size(); v++)
            {
                const int a = (int)(v % 3);
                pt.lo[a] = (std::min)(pt.lo[a], pt.v[v]);
                pt.hi[a] = (std::max)(pt.hi[a], pt.v[v]);
            }
            pt.bvh.construct(pt.v.data(), pt.v.size() / 3, pt.t.data(), pt.t.size() / 3);
            // Manifold is a PER-COMPONENT property, which is exactly why the whole-file check
            // never caught the hollow car: every edge there really is shared by two triangles.
            if (!pt.bvh.is_mesh_manifold())
            {
                err = "model '" + path + "' is not watertight (an edge is not shared by exactly "
                      "two triangles), so inside and outside are undefined";
                return false;
            }
        }
        return true;
    }

    // 2D runs cut the model at the mid-plane of its placed bounding box. An extruded profile -- an
    // airfoil section -- is exactly what that is for.
    double SectionZ() const { return 0.5 * (lo[2] + hi[2]); }

    // Outside the placed bounding box, inflated by a hair. A cheap reject for the
    // inside test, which only needs the sign; the distance itself is always exact.
    bool OutsideBox(double x, double y, double z, bool use3D) const
    {
        const double eps = 1e-12 * (1.0 + std::fabs(hi[0] - lo[0]) + std::fabs(hi[1] - lo[1]));
        if (x < lo[0] - eps || x > hi[0] + eps) return true;
        if (y < lo[1] - eps || y > hi[1] + eps) return true;
        if (use3D && (z < lo[2] - eps || z > hi[2] + eps)) return true;
        return false;
    }

    // The union of the parts: min over their signed distances.
    double Distance(double x, double y, double z) const
    {
        double best = 1e300;
        for (const Part &pt : parts)
        {
            const double d = pt.bvh.signed_distance({x, y, z}).distance;
            if (d < best) best = d;
        }
        return best;
    }

    // A lower bound on the distance of a closed surface inside the box [lo, hi]: the distance to the box from
    // outside, -inf inside; shaved by a hair so the exact query never lands below it.
    static double BoxLowerBound(const double lo[3], const double hi[3], double x, double y, double z)
    {
        const double q[3] = {x, y, z};
        double s = 0.0;
        for (int a = 0; a < 3; a++)
        {
            const double d = (std::max)(lo[a] - q[a], q[a] - hi[a]);
            if (d > 0.0) s += d * d;
        }
        if (s <= 0.0) return -1e300;
        const double d = std::sqrt(s);
        return d - 1e-12 * (1.0 + d);
    }

    // The union distance, exact wherever it is below `cap`, otherwise some value >= cap: a BVH a bounding box
    // rules out is never walked.
    double DistanceBelow(double x, double y, double z, double cap) const
    {
        const double whole = BoxLowerBound(lo, hi, x, y, z);
        if (whole >= cap) return whole;
        double best = 1e300, pruned = 1e300;
        for (const Part &pt : parts)
        {
            const double lb = BoxLowerBound(pt.lo, pt.hi, x, y, z);
            if (lb >= (std::min)(cap, best)) { pruned = (std::min)(pruned, lb); continue; }
            const double d = pt.bvh.signed_distance({x, y, z}).distance;
            if (d < best) best = d;
        }
        return (std::min)(best, pruned);
    }

    // Distance and outward unit normal from ONE query.
    bool DistanceNormal(double x, double y, double z, double &phi, double n[3]) const
    {
        tmd::Result r{};
        const Part *winner = nullptr;
        double best = 1e300;
        for (const Part &pt : parts)
        {
            const tmd::Result c = pt.bvh.signed_distance({x, y, z});
            if (c.distance < best) { best = c.distance; r = c; winner = &pt; }
        }
        if (winner == nullptr) { phi = 0.0; n[0] = n[1] = n[2] = 0.0; return false; }
        phi = r.distance;
        double d[3] = {x - r.nearest_point[0], y - r.nearest_point[1], z - r.nearest_point[2]};
        double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const double scaleRef = 1e-9 * (1.0 + (hi[0] - lo[0]) + (hi[1] - lo[1]) + (hi[2] - lo[2]));
        if (len > scaleRef)
        {
            // d points from the surface to the query; outward is +d outside, -d inside.
            const double s = (phi >= 0.0 ? 1.0 : -1.0) / len;
            n[0] = d[0] * s; n[1] = d[1] * s; n[2] = d[2] * s;
            return true;
        }
        // On the surface: the face normal of the nearest triangle.
        const unsigned *t = &winner->t[3 * (size_t)r.triangle_id];
        const double *a = &winner->v[3 * t[0]], *b = &winner->v[3 * t[1]], *c = &winner->v[3 * t[2]];
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        n[0] = e1[1] * e2[2] - e1[2] * e2[1];
        n[1] = e1[2] * e2[0] - e1[0] * e2[2];
        n[2] = e1[0] * e2[1] - e1[1] * e2[0];
        len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len <= 0.0) { n[0] = n[1] = n[2] = 0.0; return false; }
        n[0] /= len; n[1] /= len; n[2] /= len;
        return true;
    }
};

} // namespace SceneDesc

// EXPORT: a particle cloud turned into a triangle mesh on disk. The field is a SIGNED DISTANCE, not
// a density.
namespace MeshExport
{

struct TriMesh
{
    std::vector<double> vtx;   // x y z per vertex
    std::vector<double> nrm;   // one normal per vertex, from the field gradient
    std::vector<unsigned> tri; // three indices per triangle

    size_t VertexCount() const { return vtx.size() / 3; }
    size_t TriangleCount() const { return tri.size() / 3; }
    void Clear() { vtx.clear(); nrm.clear(); tri.clear(); }
};

struct SurfaceParams
{
    // All three follow the particle SPACING, which is what the field is really scaled by.
    double voxel = 0.0;          // lattice pitch; 0 = 0.75 * spacing
    double influence = 0.0;      // R in the formula above; 0 = 2 * spacing
    double particleRadius = 0.0; // r_i; 0 = 1.0 * spacing (>= 0.5 * spacing, per the paper)

    double spacing = 0.0;        // particle spacing; required
    int smoothIterations = 2;    // Laplacian passes over phi
    double shrinkOffset = 0.0;   // phi -= this; 0 = 0.25 * voxel, undoing smoothing's shrink
    double pad = 0.0;            // lattice margin around the cloud; 0 = influence + 2 voxels

    // Trim the liquid out of solids, phi = max(phi, -s). solidSDF(x, y, z, cap) must be exact when s < cap and
    // may return anything >= cap otherwise. Called from OpenMP threads.
    std::function<double(double, double, double, double)> solidSDF = nullptr;

};

// The scalar field on its lattice, kept around because the tests compare the mesh against it.
struct Field
{
    double origin[3] = {0, 0, 0};
    double voxel = 0.0;
    int n[3] = {0, 0, 0}; // SAMPLE counts per axis, so there are (n-1) cells
    double far = 0.0;      // what a node with no particle within R carries; see FarValue
    std::vector<double> phi;

    size_t Index(int i, int j, int k) const
    { return ((size_t)k * (size_t)n[1] + (size_t)j) * (size_t)n[0] + (size_t)i; }
    double At(int i, int j, int k) const { return phi[Index(i, j, k)]; }
    double X(int i) const { return origin[0] + i * voxel; }
    double Y(int j) const { return origin[1] + j * voxel; }
    double Z(int k) const { return origin[2] + k * voxel; }

    // Trilinear sample, for checking the mesh against the field it came from.
    double Sample(double x, double y, double z) const
    {
        double g[3] = {(x - origin[0]) / voxel, (y - origin[1]) / voxel, (z - origin[2]) / voxel};
        int c[3];
        double t[3];
        for (int a = 0; a < 3; a++)
        {
            double f = std::floor(g[a]);
            c[a] = (int)f;
            if (c[a] < 0) { c[a] = 0; f = 0.0; }
            if (c[a] > n[a] - 2) { c[a] = n[a] - 2; f = (double)c[a]; }
            t[a] = g[a] - f;
            if (t[a] < 0.0) t[a] = 0.0;
            if (t[a] > 1.0) t[a] = 1.0;
        }
        double out = 0.0;
        for (int dk = 0; dk < 2; dk++)
            for (int dj = 0; dj < 2; dj++)
                for (int di = 0; di < 2; di++)
                {
                    const double wx = di ? t[0] : 1.0 - t[0];
                    const double wy = dj ? t[1] : 1.0 - t[1];
                    const double wz = dk ? t[2] : 1.0 - t[2];
                    out += wx * wy * wz * At(c[0] + di, c[1] + dj, c[2] + dk);
                }
        return out;
    }
};

// A node with no particle within R gets a CAPPED positive distance, not a sentinel.
inline double FarValue(double influence, double particleRadius)
{
    const double v = influence - particleRadius;
    return (v > 0.0) ? v : 0.5 * influence;
}

// ---- the field ---------------------------------------------------------- Particles come in as a
// flat array of x y z triples so this file needs to know nothing about FLIP.
inline SurfaceParams Resolved(const SurfaceParams &pIn)
{
    SurfaceParams p = pIn;
    if (!(p.voxel > 0.0)) p.voxel = 0.75 * p.spacing;
    if (!(p.influence > 0.0)) p.influence = 2.0 * p.spacing;
    if (!(p.particleRadius > 0.0)) p.particleRadius = 1.0 * p.spacing;
    if (!(p.pad > 0.0)) p.pad = p.influence + 2.0 * p.voxel;
    if (p.shrinkOffset == 0.0) p.shrinkOffset = 0.25 * p.voxel;
    return p;
}

// The cell of side R holding point q in a list anchored at lo, clamped to the grid.
inline void ParticleCell(const double *q, const double lo[3], double R, const int cn[3], int *c)
{
    for (int a = 0; a < 3; a++)
    {
        c[a] = (int)std::floor((q[a] - lo[a]) / R);
        if (c[a] < 0) c[a] = 0;
        if (c[a] > cn[a] - 1) c[a] = cn[a] - 1;
    }
}

inline bool BuildField(const double *xyz, size_t count, const SurfaceParams &pIn, Field &out,
                       std::string &err)
{
    if (xyz == nullptr || count == 0) { err = "no particles"; return false; }
    if (!(pIn.spacing > 0.0)) { err = "particle spacing must be given"; return false; }
    const SurfaceParams p = Resolved(pIn);

    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < count; i++)
        for (int a = 0; a < 3; a++)
        {
            lo[a] = (std::min)(lo[a], xyz[3 * i + a]);
            hi[a] = (std::max)(hi[a], xyz[3 * i + a]);
        }

    out.voxel = p.voxel;
    for (int a = 0; a < 3; a++)
    {
        // The lattice is anchored to whole multiples of the voxel size in WORLD space, not to the
        // cloud: a lattice that shifts with the fluid makes every vertex jitter from frame to.
        const double start = std::floor((lo[a] - p.pad) / p.voxel) * p.voxel;
        const double end = std::ceil((hi[a] + p.pad) / p.voxel) * p.voxel;
        out.origin[a] = start;
        out.n[a] = (int)std::llround((end - start) / p.voxel) + 1;
        if (out.n[a] < 2) out.n[a] = 2;
    }
    out.far = FarValue(p.influence, p.particleRadius);
    out.phi.assign((size_t)out.n[0] * out.n[1] * out.n[2], out.far);

    // ---- a uniform cell list over the particles, cells of side R ---- Counting sort, the same
    // shape SPH already uses: this is what makes the node loop below cost a fixed number of.
    const double R = p.influence, R2 = R * R;
    int cn[3];
    for (int a = 0; a < 3; a++)
    {
        cn[a] = (int)std::floor((hi[a] - lo[a]) / R) + 1;
        if (cn[a] < 1) cn[a] = 1;
    }
    const size_t cells = (size_t)cn[0] * cn[1] * cn[2];
    std::vector<int> start(cells + 1, 0), order(count);
    for (size_t i = 0; i < count; i++)
    {
        int c[3];
        ParticleCell(&xyz[3 * i], lo, R, cn, c);
        start[((size_t)c[2] * cn[1] + c[1]) * cn[0] + c[0] + 1]++;
    }
    for (size_t c = 0; c < cells; c++) start[c + 1] += start[c];
    { std::vector<int> fill(start.begin(), start.end() - 1);
      for (size_t i = 0; i < count; i++)
      {
          int c[3];
          ParticleCell(&xyz[3 * i], lo, R, cn, c);
          order[fill[((size_t)c[2] * cn[1] + c[1]) * cn[0] + c[0]]++] = (int)i;
      } }
    // The positions themselves in cell order, so the node loop reads them contiguously.
    std::vector<double> sorted(3 * count);
    for (size_t e = 0; e < count; e++)
        for (int a = 0; a < 3; a++) sorted[3 * e + a] = xyz[3 * (size_t)order[e] + a];

    const double rbarConst = p.particleRadius; // one radius for every particle, for now

    // Dynamic: a node beside the liquid sums hundreds of particles, one in the air sums none.
    #pragma omp parallel for collapse(2) schedule(dynamic, 64)
    for (int k = 0; k < out.n[2]; k++)
        for (int j = 0; j < out.n[1]; j++)
            for (int i = 0; i < out.n[0]; i++)
            {
                const double x[3] = {out.X(i), out.Y(j), out.Z(k)};
                int c[3];
                ParticleCell(x, lo, R, cn, c);
                double wsum = 0.0, xbar[3] = {0, 0, 0};
                for (int dz = -1; dz <= 1; dz++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            const int cx = c[0] + dx, cy = c[1] + dy, cz = c[2] + dz;
                            if (cx < 0 || cy < 0 || cz < 0 || cx >= cn[0] || cy >= cn[1] || cz >= cn[2])
                                continue;
                            const size_t cid = ((size_t)cz * cn[1] + cy) * cn[0] + cx;
                            for (int e = start[cid]; e < start[cid + 1]; e++)
                            {
                                const double *q = &sorted[3 * (size_t)e];
                                const double d[3] = {x[0] - q[0], x[1] - q[1], x[2] - q[2]};
                                const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                                if (r2 >= R2) continue;
                                // k(s) = (1 - s^2)^3, so no square root is needed here.
                                const double s2 = r2 / R2;
                                const double t = 1.0 - s2;
                                const double w = t * t * t;
                                wsum += w;
                                for (int a = 0; a < 3; a++) xbar[a] += w * q[a];
                            }
                        }
                if (wsum <= 0.0) continue; // keeps the capped far value
                for (int a = 0; a < 3; a++) xbar[a] /= wsum;
                const double d[3] = {x[0] - xbar[0], x[1] - xbar[1], x[2] - xbar[2]};
                out.phi[out.Index(i, j, k)] =
                    std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - rbarConst;
            }

    // ---- smooth, then put back the volume smoothing takes away ----
    if (p.smoothIterations > 0)
    {
        std::vector<double> tmp(out.phi);
        for (int pass = 0; pass < p.smoothIterations; pass++)
        {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = 1; k < out.n[2] - 1; k++)
                for (int j = 1; j < out.n[1] - 1; j++)
                    for (int i = 1; i < out.n[0] - 1; i++)
                    {
                        const double c0 = out.At(i, j, k);
                        double sum = 0.0;
                        const int off[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
                        for (const auto &o : off) sum += out.At(i + o[0], j + o[1], k + o[2]);
                        tmp[out.Index(i, j, k)] = 0.5 * c0 + 0.5 * sum / 6.0;
                    }
            out.phi.swap(tmp);
        }
    }
    if (p.shrinkOffset != 0.0)
    {
        const long nodes = (long)out.phi.size();
        #pragma omp parallel for schedule(static)
        for (long q = 0; q < nodes; q++) out.phi[(size_t)q] -= p.shrinkOffset;
    }

    // ---- trim the liquid out of the solids: max(v, -s) moves v only where s < -v, so -v is the cap; dynamic
    // schedule, the cost per node ranging from a box test to a BVH walk ----
    if (p.solidSDF)
    {
        #pragma omp parallel for collapse(2) schedule(dynamic, 64)
        for (int k = 0; k < out.n[2]; k++)
            for (int j = 0; j < out.n[1]; j++)
                for (int i = 0; i < out.n[0]; i++)
                {
                    double &v = out.phi[out.Index(i, j, k)];
                    const double s = p.solidSDF(out.X(i), out.Y(j), out.Z(k), -v);
                    v = (std::max)(v, -s);
                }
    }
    return true;
}

// ---- marching tetrahedra ------------------------------------------------
namespace detail
{

// The six tetrahedra of one cell, as corner indices in the (i,j,k) binary numbering: corner c sits
// at (i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)).
constexpr int TETS[6][4] = {
    {0, 1, 3, 7}, {0, 1, 5, 7}, {0, 2, 3, 7},
    {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 4, 6, 7},
};

// Every tetrahedron edge joins a LOWER corner to one whose offset from it is 0 or 1 on each axis,
// so a lattice edge is named by its lower node and type = dx + 2 dy + 4 dz - 1, in 0..6.
inline uint32_t PlaneEdgeKey(const Field &f, int i, int j, int type)
{
    return ((uint32_t)j * (uint32_t)f.n[0] + (uint32_t)i) * 7u + (uint32_t)type;
}

inline double ClampedAt(const Field &f, int a, int b, int c)
{
    a = (std::max)(0, (std::min)(f.n[0] - 1, a));
    b = (std::max)(0, (std::min)(f.n[1] - 1, b));
    c = (std::max)(0, (std::min)(f.n[2] - 1, c));
    return f.At(a, b, c);
}

// The field gradient, by central differences, is the normal: cheaper than averaging face normals
// and smoother, because it comes from the surface's own definition.
inline void FieldGradient(const Field &f, int i, int j, int k, double g[3])
{
    g[0] = ClampedAt(f, i + 1, j, k) - ClampedAt(f, i - 1, j, k);
    g[1] = ClampedAt(f, i, j + 1, k) - ClampedAt(f, i, j - 1, k);
    g[2] = ClampedAt(f, i, j, k + 1) - ClampedAt(f, i, j, k - 1);
}

// Position and normal of the vertex on a cut edge, always interpolated from the lower node to the
// upper one, so every cell and every thread that reaches the edge agrees on the same doubles.
inline void EdgeVertex(const Field &f, int i, int j, int k, int type, double *p, double *nrm)
{
    const int bits = type + 1;
    const int i2 = i + (bits & 1), j2 = j + ((bits >> 1) & 1), k2 = k + ((bits >> 2) & 1);
    const double a = f.At(i, j, k), b = f.At(i2, j2, k2);
    double t = 0.5;
    if (a != b) t = (0.0 - a) / (b - a);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    const double pa[3] = {f.X(i), f.Y(j), f.Z(k)};
    const double pb[3] = {f.X(i2), f.Y(j2), f.Z(k2)};
    double ga[3], gb[3];
    FieldGradient(f, i, j, k, ga);
    FieldGradient(f, i2, j2, k2, gb);
    for (int c = 0; c < 3; c++)
    {
        p[c] = pa[c] + t * (pb[c] - pa[c]);
        nrm[c] = ga[c] + t * (gb[c] - ga[c]);
    }
    const double len = std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
    for (int c = 0; c < 3; c++) nrm[c] = (len > 1e-30) ? nrm[c] / len : 0.0;
}

// The global id of the vertex on the edge between corners ca and cb of cell (i, j, k): the cut
// edges of each node plane are listed in key order, and their ids follow the planes' prefix sums.
inline unsigned SlabVertex(const Field &f, const std::vector<std::vector<uint32_t>> &cut,
                           const std::vector<size_t> &base, int i, int j, int k, int ca, int cb)
{
    const int lo = ((ca & cb) == ca) ? ca : cb;
    const int hi = (lo == ca) ? cb : ca;
    const int kk = k + ((lo >> 2) & 1);
    const uint32_t key = PlaneEdgeKey(f, i + (lo & 1), j + ((lo >> 1) & 1), (hi ^ lo) - 1);
    const std::vector<uint32_t> &keys = cut[(size_t)kk];
    const size_t e = (size_t)(std::lower_bound(keys.begin(), keys.end(), key) - keys.begin());
    return (unsigned)(base[(size_t)kk] + e);
}

// Weld key: three rounded coordinates.
// its a hash, ugly, but works
struct WeldKeyHash
{
    size_t operator()(const std::array<long long, 3> &q) const
    {
        uint64_t h = (uint64_t)q[0] * 0x9E3779B97F4A7C15ull;
        h ^= (uint64_t)q[1] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        h ^= (uint64_t)q[2] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

} // namespace detail

// ---- weld coincident vertices, then drop what collapsed -------------------- A clip plane that
// lands exactly on a lattice NODE plane puts `phi == 0` on a whole plane of nodes.
inline void WeldCoincident(TriMesh &mesh, double voxel)
{
    if (mesh.tri.empty() || !(voxel > 0.0)) return;

    // A thousandth of a voxel: far below any feature the field can represent, and wide enough to
    // catch the near-coincident case that smoothing leaves behind when phi is only ALMOST exactly.
    const double inv = 1.0 / (1e-3 * voxel);
    const long nv = (long)mesh.VertexCount();
    std::vector<std::array<long long, 3>> keys((size_t)nv);
    #pragma omp parallel for schedule(static)
    for (long v = 0; v < nv; v++)
        for (int a = 0; a < 3; a++)
            keys[(size_t)v][a] = (long long)std::llround(mesh.vtx[3 * (size_t)v + a] * inv);

    // Each vertex maps to the FIRST one carrying its key, so the result does not depend on how the
    // table orders them.
    std::unordered_map<std::array<long long, 3>, unsigned, detail::WeldKeyHash> seen;
    seen.reserve((size_t)nv);
    std::vector<unsigned> remap((size_t)nv, 0);
    std::vector<double> vtx, nrm;
    vtx.reserve(mesh.vtx.size());
    nrm.reserve(mesh.nrm.size());
    for (long v = 0; v < nv; v++)
    {
        const unsigned fresh = (unsigned)(vtx.size() / 3);
        const auto ins = seen.emplace(keys[(size_t)v], fresh);
        if (!ins.second) { remap[(size_t)v] = ins.first->second; continue; }
        remap[(size_t)v] = fresh;
        for (int a = 0; a < 3; a++) vtx.push_back(mesh.vtx[3 * (size_t)v + a]);
        for (int a = 0; a < 3; a++) nrm.push_back(mesh.nrm[3 * (size_t)v + a]);
    }
    if ((long)(vtx.size() / 3) == nv) return; // nothing coincided: every triangle keeps its corners

    std::vector<unsigned> tri;
    tri.reserve(mesh.tri.size());
    for (size_t t = 0; t < mesh.TriangleCount(); t++)
    {
        const unsigned a = remap[mesh.tri[3 * t + 0]],
                       b = remap[mesh.tri[3 * t + 1]],
                       c = remap[mesh.tri[3 * t + 2]];
        // A triangle that lost a distinct corner had no area to begin with; dropping it
        // cannot open a hole, because the two edges that merged were the same edge.
        if (a == b || b == c || a == c) continue;
        tri.push_back(a); tri.push_back(b); tri.push_back(c);
    }

    mesh.vtx.swap(vtx);
    mesh.nrm.swap(nrm);
    mesh.tri.swap(tri);
}

// Appends a triangle wound against the field gradient, so its face normal points outward.
inline void EmitOutward(const TriMesh &mesh, std::vector<unsigned> &out, unsigned a, unsigned b, unsigned c)
{
    const double *p0 = &mesh.vtx[3 * a], *p1 = &mesh.vtx[3 * b], *p2 = &mesh.vtx[3 * c];
    const double e1[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
    const double e2[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
    const double fn[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
    const double *vn = &mesh.nrm[3 * a];
    if (fn[0]*vn[0] + fn[1]*vn[1] + fn[2]*vn[2] >= 0.0)
    { out.push_back(a); out.push_back(b); out.push_back(c); }
    else
    { out.push_back(a); out.push_back(c); out.push_back(b); }
}

// phi < 0 inside, triangles wound outward. Three OpenMP passes over z slabs: cut edges per node plane,
// vertex ids from prefix sums, then each slab's triangles, concatenated in z order.
inline void Polygonise(const Field &f, TriMesh &mesh)
{
    mesh.Clear();
    const int n0 = f.n[0], n1 = f.n[1], n2 = f.n[2];

    std::vector<std::vector<uint32_t>> cut((size_t)n2);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < n2; k++)
    {
        std::vector<uint32_t> &keys = cut[(size_t)k];
        for (int j = 0; j < n1; j++)
            for (int i = 0; i < n0; i++)
            {
                const bool in = f.At(i, j, k) < 0.0;
                for (int type = 0; type < 7; type++)
                {
                    const int bits = type + 1;
                    const int i2 = i + (bits & 1), j2 = j + ((bits >> 1) & 1), k2 = k + ((bits >> 2) & 1);
                    if (i2 >= n0 || j2 >= n1 || k2 >= n2) continue;
                    if ((f.At(i2, j2, k2) < 0.0) != in) keys.push_back(detail::PlaneEdgeKey(f, i, j, type));
                }
            }
    }

    std::vector<size_t> base((size_t)n2 + 1, 0);
    for (int k = 0; k < n2; k++) base[(size_t)k + 1] = base[(size_t)k] + cut[(size_t)k].size();
    mesh.vtx.resize(3 * base[(size_t)n2]);
    mesh.nrm.resize(3 * base[(size_t)n2]);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < n2; k++)
    {
        const std::vector<uint32_t> &keys = cut[(size_t)k];
        for (size_t e = 0; e < keys.size(); e++)
        {
            const uint32_t node = keys[e] / 7u;
            const int type = (int)(keys[e] % 7u);
            const int i = (int)(node % (uint32_t)n0), j = (int)(node / (uint32_t)n0);
            const size_t id = base[(size_t)k] + e;
            detail::EdgeVertex(f, i, j, k, type, &mesh.vtx[3 * id], &mesh.nrm[3 * id]);
        }
    }

    std::vector<std::vector<unsigned>> slabs((size_t)(n2 > 1 ? n2 - 1 : 0));
    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < n2 - 1; k++)
    {
        std::vector<unsigned> &out = slabs[(size_t)k];
        for (int j = 0; j < n1 - 1; j++)
            for (int i = 0; i < n0 - 1; i++)
            {
                double v[8];
                int signs = 0;
                for (int c = 0; c < 8; c++)
                {
                    v[c] = f.At(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                    if (v[c] < 0.0) signs |= (1 << c);
                }
                if (signs == 0 || signs == 255) continue;

                for (const auto &tet : detail::TETS)
                {
                    // Sign pattern of the four corners: 16 cases, and only the sign matters.
                    int code = 0;
                    for (int c = 0; c < 4; c++)
                        if (v[tet[c]] < 0.0) code |= (1 << c);
                    if (code == 0 || code == 15) continue;

                    // Each case is "which corners are inside"; the surface cuts the edges between
                    // an inside and an outside corner.
                    int in[4], outv[4], ni = 0, no = 0;
                    for (int c = 0; c < 4; c++)
                        (v[tet[c]] < 0.0 ? in[ni++] : outv[no++]) = tet[c];

                    if (ni == 1 || ni == 3)
                    {
                        const int apex = (ni == 1) ? in[0] : outv[0];
                        const int *rest = (ni == 1) ? outv : in;
                        EmitOutward(mesh, out, detail::SlabVertex(f, cut, base, i, j, k, apex, rest[0]),
                                    detail::SlabVertex(f, cut, base, i, j, k, apex, rest[1]),
                                    detail::SlabVertex(f, cut, base, i, j, k, apex, rest[2]));
                    }
                    else // two in, two out: a quad
                    {
                        const unsigned a = detail::SlabVertex(f, cut, base, i, j, k, in[0], outv[0]),
                                       b = detail::SlabVertex(f, cut, base, i, j, k, in[0], outv[1]),
                                       c = detail::SlabVertex(f, cut, base, i, j, k, in[1], outv[1]),
                                       d = detail::SlabVertex(f, cut, base, i, j, k, in[1], outv[0]);
                        EmitOutward(mesh, out, a, b, c);
                        EmitOutward(mesh, out, a, c, d);
                    }
                }
            }
    }

    size_t total = 0;
    for (const std::vector<unsigned> &sl : slabs) total += sl.size();
    mesh.tri.reserve(total);
    for (const std::vector<unsigned> &sl : slabs) mesh.tri.insert(mesh.tri.end(), sl.begin(), sl.end());

    WeldCoincident(mesh, f.voxel);
}

// One call: particles in, mesh out. ---- the last surface produced, kept for the viewer
// ---------------------- The exporter writes a PLY and forgets it.
struct LastSurface
{
    TriMesh mesh;
    int frame = -1;
    double time = 0.0;
    double voxel = 0.0;
    double spacing = 0.0;
    unsigned generation = 0;
    bool Empty() const { return mesh.tri.empty(); }
};

inline LastSurface LAST_SURFACE;
inline bool CAPTURE_LAST_SURFACE = false;

// How many triangles to skip so a preview stays under its budget. All the vertices are kept
// either way, so the shape still reads; this only thins the index buffer.
inline int PreviewStride(size_t triangles, int budget)
{
    if (budget <= 0 || triangles == 0) return 1;
    if (triangles <= (size_t)budget) return 1;
    return (int)((triangles + (size_t)budget - 1) / (size_t)budget);
}

inline void CaptureLastSurface(const TriMesh &m, const SurfaceParams &resolved)
{
    if (!CAPTURE_LAST_SURFACE) return;
    LAST_SURFACE.mesh = m;
    LAST_SURFACE.voxel = resolved.voxel;
    LAST_SURFACE.spacing = resolved.spacing;
    LAST_SURFACE.generation++;
}

inline bool SurfaceFromParticles(const double *xyz, size_t count, const SurfaceParams &p,
                                 TriMesh &mesh, std::string &err, Field *keepField = nullptr)
{
    Field f;
    if (!BuildField(xyz, count, p, f, err)) return false;
    Polygonise(f, mesh);
    CaptureLastSurface(mesh, Resolved(p));
    if (keepField) *keepField = std::move(f);
    return true;
}

// ---- a surface from any signed distance ---------------------------------- The polygoniser does
// not care where phi came from.
inline bool FieldFromSDF(const std::function<double(double, double, double, double)> &sdf,
                         const double lo[3], const double hi[3], double voxel, Field &out,
                         std::string &err)
{
    if (!sdf) { err = "no signed distance given"; return false; }
    if (!(voxel > 0.0)) { err = "voxel size must be positive"; return false; }
    out.voxel = voxel;
    for (int a = 0; a < 3; a++)
    {
        const double start = std::floor((lo[a] - voxel) / voxel) * voxel;
        const double end = std::ceil((hi[a] + voxel) / voxel) * voxel;
        out.origin[a] = start;
        out.n[a] = (int)std::llround((end - start) / voxel) + 1;
        if (out.n[a] < 2) out.n[a] = 2;
    }
    out.far = 3.0 * voxel;
    out.phi.assign((size_t)out.n[0] * out.n[1] * out.n[2], out.far);

    // Values are capped at +far below, so the sdf is asked for nothing above it (see SurfaceParams).
    bool anyInside = false;
    #pragma omp parallel for collapse(2) schedule(dynamic, 64) reduction(||:anyInside)
    for (int k = 0; k < out.n[2]; k++)
        for (int j = 0; j < out.n[1]; j++)
            for (int i = 0; i < out.n[0]; i++)
            {
                double v = sdf(out.X(i), out.Y(j), out.Z(k), out.far);
                if (v > out.far) v = out.far;      // far field, capped
                if (v < -out.far) v = -out.far;
                out.phi[out.Index(i, j, k)] = v;
                if (v < 0.0) anyInside = true;
            }
    if (!anyInside) { err = "the signed distance is positive everywhere in these bounds"; return false; }
    return true;
}

inline bool SurfaceFromSDF(const std::function<double(double, double, double, double)> &sdf,
                           const double lo[3], const double hi[3], double voxel, TriMesh &mesh,
                           std::string &err)
{
    Field f;
    if (!FieldFromSDF(sdf, lo, hi, voxel, f, err)) return false;
    Polygonise(f, mesh);
    return true;
}

// ---- PLY ---------------------------------------------------------------- Binary little-endian,
// positions + normals + triangles: what Blender imports, a third the size of ASCII.
inline bool WritePLY(const std::string &path, const TriMesh &mesh, std::string &err)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) { err = "could not open " + path; return false; }
    f << "ply\nformat binary_little_endian 1.0\n";
    f << "comment NEREUS particle surface\n";
    f << "element vertex " << mesh.VertexCount() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property float nx\nproperty float ny\nproperty float nz\n";
    f << "element face " << mesh.TriangleCount() << "\n";
    f << "property list uchar int vertex_indices\n";
    f << "end_header\n";
    for (size_t i = 0; i < mesh.VertexCount(); i++)
    {
        float rec[6] = {(float)mesh.vtx[3 * i], (float)mesh.vtx[3 * i + 1], (float)mesh.vtx[3 * i + 2],
                        (float)mesh.nrm[3 * i], (float)mesh.nrm[3 * i + 1], (float)mesh.nrm[3 * i + 2]};
        f.write((const char *)rec, sizeof(rec));
    }
    for (size_t t = 0; t < mesh.TriangleCount(); t++)
    {
        const unsigned char three = 3;
        const int idx[3] = {(int)mesh.tri[3 * t], (int)mesh.tri[3 * t + 1], (int)mesh.tri[3 * t + 2]};
        f.write((const char *)&three, 1);
        f.write((const char *)idx, sizeof(idx));
    }
    f.close();
    if (!f) { err = "write failed on " + path; return false; }
    return true;
}

} // namespace MeshExport

#endif
