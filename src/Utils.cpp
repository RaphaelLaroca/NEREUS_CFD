// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Utils.h"
#include <chrono>
#include "Core/MAC.h"
#include "Core/Scene.h"

// The cell whose low face is the last one at or below c, by bisection on the face table.
static int CellContaining(const std::vector<double> &f, double c, int n)
{
    int lo = 0, hi = n - 1;
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (f[mid] <= c) lo = mid; else hi = mid - 1; }
    return lo < 0 ? 0 : (lo > n - 1 ? n - 1 : lo);
}

bool IsOpenAt(MAC* g, double x, double y)
{
    if (!(x >= g->FacePos(MAC::X, 0) && x <= g->FacePos(MAC::X, g->Nx) && y >= g->FacePos(MAC::Y, 0) && y <= g->FacePos(MAC::Y, g->Ny))) return false;
    return g->GetSolid(g->CellAt(MAC::Y, y), g->CellAt(MAC::X, x)) != SOLID_CELL;
}

bool IsOpenAt(MAC* g, double x, double y, double z)
{
    if (!(x >= g->FacePos(MAC::X, 0) && x <= g->FacePos(MAC::X, g->Nx) && y >= g->FacePos(MAC::Y, 0) && y <= g->FacePos(MAC::Y, g->Ny) && z >= g->FacePos(MAC::Z, 0) && z <= g->FacePos(MAC::Z, g->Nz)))
        return false;
    return g->GetSolid(g->CellAt(MAC::Y, y), g->CellAt(MAC::X, x), g->CellAt(MAC::Z, z)) != SOLID_CELL;
}

bool VelocityAt(MAC* g, double x, double y, double &u, double &v)
{
    u = v = 0.0;
    if (g->Nx < 2 || g->Ny < 2 || !IsOpenAt(g, x, y)) return false;
    u = g->ComponentAt(MAC::U, x, y);
    v = g->ComponentAt(MAC::V, x, y);
    return std::isfinite(u) && std::isfinite(v);
}

bool VelocityAt(MAC* g, double x, double y, double z, double &u, double &v, double &w)
{
    u = v = w = 0.0;
    if (g->Nx < 2 || g->Ny < 2 || g->Nz < 2 || !IsOpenAt(g, x, y, z)) return false;
    u = g->ComponentAt(MAC::U, x, y, z);
    v = g->ComponentAt(MAC::V, x, y, z);
    w = g->ComponentAt(MAC::W, x, y, z);
    return std::isfinite(u) && std::isfinite(v) && std::isfinite(w);
}

double GetWallTime() {
    const auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

// O(n) Thomas Algorithm
void TDMA_VECTORIZED_N(const Eigen::VectorXd& a, Eigen::VectorXd& b, const Eigen::VectorXd& c,
                      Eigen::VectorXd& d, Eigen::VectorXd& x, int n) {
    if (n <= 0) return;

    // Forward elimination
    for (int i = 1; i < n; i++) {
        double m = a(i) / b(i-1);
        b(i) = b(i) - m * c(i-1);
        d(i) = d(i) - m * d(i-1);
    }

    // Back substitution
    x(n-1) = d(n-1) / b(n-1);
    for (int i = n-2; i >= 0; i--) {
        x(i) = (d(i) - c(i) * x(i+1)) / b(i);
    }
}

// Flow integrals

FlowIntegrals ComputeFlowIntegrals2D(MAC* g)
{
    FlowIntegrals out;
    if (g == nullptr || g->Nx < 3 || g->Ny < 3) return out;
    const int Nx = g->Nx, Ny = g->Ny;

    double volume = 0.0, energy = 0.0, enstrophy = 0.0;
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            if (g->IsInteriorFluid(i, j)) volume += g->Width(MAC::X, j) * g->Width(MAC::Y, i);

    // U(i,j) sits on the x face between cells j-1 and j; its control volume spans the two
    // half cells in x and the whole cell row in y.
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 2; j < Nx - 1; j++)
            if (g->IsInteriorFluid(i, j - 1) && g->IsInteriorFluid(i, j))
            {
                const double u = g->GetU(i, j);
                energy += 0.5 * u * u * g->FaceGap(MAC::X, j - 1) * g->Width(MAC::Y, i);
            }
    for (int i = 2; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            if (g->IsInteriorFluid(i - 1, j) && g->IsInteriorFluid(i, j))
            {
                const double v = g->GetV(i, j);
                energy += 0.5 * v * v * g->Width(MAC::X, j) * g->FaceGap(MAC::Y, i - 1);
            }

    // Vorticity on the corner shared by cells (i-1|i, j-1|j): dv/dx - du/dy from the two V and
    // the two U faces that meet there.
    for (int i = 2; i < Ny - 1; i++)
        for (int j = 2; j < Nx - 1; j++)
        {
            if (!g->IsInteriorFluid(i - 1, j - 1) || !g->IsInteriorFluid(i - 1, j) ||
                !g->IsInteriorFluid(i, j - 1) || !g->IsInteriorFluid(i, j))
                continue;
            const double dvdx = (g->GetV(i, j) - g->GetV(i, j - 1)) / g->FaceGap(MAC::X, j - 1);
            const double dudy = (g->GetU(i, j) - g->GetU(i - 1, j)) / g->FaceGap(MAC::Y, i - 1);
            const double w = dvdx - dudy;
            enstrophy += 0.5 * w * w * g->FaceGap(MAC::X, j - 1) * g->FaceGap(MAC::Y, i - 1);
        }

    out.fluidVolume = volume;
    if (volume > 0.0)
    {
        out.kineticEnergy = energy / volume;
        out.enstrophy = enstrophy / volume;
    }
    return out;
}

// A face counts when neither cell beside it is solid; wall faces are pinned and carry no flow.
static void KeepLarger(CellReynolds &best, double re, int component, double x, double y, double z)
{
    if (re > best.value) { best.value = re; best.component = component; best.x = x; best.y = y; best.z = z; }
}

CellReynolds ComputeMaxCellReynolds2D(MAC* g, double nu)
{
    CellReynolds best;
    if (g == nullptr || nu <= 0.0 || g->Nx < 3 || g->Ny < 3) return best;
    best.valid = true;
    const int Nx = g->Nx, Ny = g->Ny;
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx; j++)
            if (g->FaceGate(MAC::U, i, j) != GateState::WALL)
                KeepLarger(best, std::fabs(g->GetU(i, j)) * g->FaceGap(MAC::X, j - 1) / nu, 0, g->FacePos(MAC::X, j), g->CentrePos(MAC::Y, i), 0.0);
    for (int i = 1; i < Ny; i++)
        for (int j = 1; j < Nx - 1; j++)
            if (g->FaceGate(MAC::V, i, j) != GateState::WALL)
                KeepLarger(best, std::fabs(g->GetV(i, j)) * g->FaceGap(MAC::Y, i - 1) / nu, 1, g->CentrePos(MAC::X, j), g->FacePos(MAC::Y, i), 0.0);
    return best;
}

CellReynolds ComputeMaxCellReynolds3D(MAC* g, double nu)
{
    CellReynolds best;
    if (g == nullptr || nu <= 0.0 || g->Nx < 3 || g->Ny < 3 || g->Nz < 3) return best;
    best.valid = true;
    const int Nx = g->Nx, Ny = g->Ny, Nz = g->Nz;
    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx; j++)
                if (g->FaceGate(MAC::U, i, j, k) != GateState::WALL)
                    KeepLarger(best, std::fabs(g->GetU(i, j, k)) * g->FaceGap(MAC::X, j - 1) / nu, 0, g->FacePos(MAC::X, j), g->CentrePos(MAC::Y, i), g->CentrePos(MAC::Z, k));
    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny; i++)
            for (int j = 1; j < Nx - 1; j++)
                if (g->FaceGate(MAC::V, i, j, k) != GateState::WALL)
                    KeepLarger(best, std::fabs(g->GetV(i, j, k)) * g->FaceGap(MAC::Y, i - 1) / nu, 1, g->CentrePos(MAC::X, j), g->FacePos(MAC::Y, i), g->CentrePos(MAC::Z, k));
    for (int k = 1; k < Nz; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
                if (g->FaceGate(MAC::W, i, j, k) != GateState::WALL)
                    KeepLarger(best, std::fabs(g->GetW(i, j, k)) * g->FaceGap(MAC::Z, k - 1) / nu, 2, g->CentrePos(MAC::X, j), g->CentrePos(MAC::Y, i), g->FacePos(MAC::Z, k));
    return best;
}

// Points already laid down, binned by position, so the distance tests look at a few neighbours only.
struct StreamOccupancy
{
    struct Point { double x, y; int line, half, index; };
    double x0 = 0.0, y0 = 0.0, cell = 1.0;
    int nx = 1, ny = 1;
    std::vector<std::vector<Point>> bins;
};

static void OccupancyAdd(StreamOccupancy &occ, double x, double y, int line, int half, int index)
{
    const int bx = std::min(occ.nx - 1, std::max(0, (int)((x - occ.x0) / occ.cell)));
    const int by = std::min(occ.ny - 1, std::max(0, (int)((y - occ.y0) / occ.cell)));
    occ.bins[(size_t)by * occ.nx + bx].push_back(StreamOccupancy::Point{x, y, line, half, index});
}

// A point of another line within radius, or of this line's own half from well behind (a loop closing).
// ownGap points either side of the seed and of the current point are not the line meeting itself.
static bool OccupancyNear(const StreamOccupancy &occ, double x, double y, double radius,
                          int line, int half, int index, int ownGap)
{
    const int r = (int)std::ceil(radius / occ.cell);
    const int bx = (int)((x - occ.x0) / occ.cell), by = (int)((y - occ.y0) / occ.cell);
    const double r2 = radius * radius;
    for (int b = std::max(0, by - r); b <= std::min(occ.ny - 1, by + r); b++)
        for (int a = std::max(0, bx - r); a <= std::min(occ.nx - 1, bx + r); a++)
            for (const StreamOccupancy::Point &p : occ.bins[(size_t)b * occ.nx + a])
            {
                const double dx = p.x - x, dy = p.y - y;
                if (dx * dx + dy * dy >= r2) continue;
                if (p.line != line) return true;
                const bool nearSeed = p.index < ownGap && index < ownGap;
                const bool recent = p.half == half && index - p.index < ownGap;
                if (!nearSeed && !recent) return true;
            }
    return false;
}

// One direction from the seed. The seed itself is recorded by the forward half only.
static void TraceStreamHalf(MAC* g, const StreamlineSettings &s, StreamOccupancy &occ, double sx, double sy,
                            double sign, int line, int half, Streamline &out)
{
    const double h = s.step, dTest = 0.5 * s.separation;
    const int ownGap = (int)std::ceil(2.0 * dTest / h) + 2;
    // The window the occupancy bins cover: a line stops where it leaves it.
    const double wx1 = occ.x0 + occ.nx * occ.cell, wy1 = occ.y0 + occ.ny * occ.cell;
    double x = sx, y = sy;
    for (int n = 0; n < s.maxPoints / 2; n++)
    {
        if (x < occ.x0 || x > wx1 || y < occ.y0 || y > wy1) break;
        double u, v;
        if (!VelocityAt(g, x, y, u, v)) break;
        const double speed = std::sqrt(u * u + v * v);
        if (speed < s.minSpeed) break;
        if (half == 0 || n > 0)
        {
            out.x.push_back(x); out.y.push_back(y); out.speed.push_back(speed);
            OccupancyAdd(occ, x, y, line, half, n);
        }
        const double ex = sign * u / speed, ey = sign * v / speed;
        double um, vm;
        if (!VelocityAt(g, x + 0.5 * h * ex, y + 0.5 * h * ey, um, vm)) break;
        const double sm = std::sqrt(um * um + vm * vm);
        if (sm < s.minSpeed) break;
        const double mx = sign * um / sm, my = sign * vm / sm;
        if (mx * ex + my * ey < 0.0) break;   // the direction reverses inside one step: a critical point
        const double nx = x + h * mx, ny = y + h * my;
        if (OccupancyNear(occ, nx, ny, dTest, line, half, n + 1, ownGap)) break;
        x = nx; y = ny;
    }
}

void ComputeStreamlines2D(MAC* g, const StreamlineSettings &s, std::vector<Streamline> &out)
{
    out.clear();
    if (g == nullptr || g->Nx < 3 || g->Ny < 3 || !(s.separation > 0.0) || !(s.step > 0.0)) return;
    // Seeds cover the domain, i.e. the interior faces; the ring lies outside it.
    const std::vector<double> &fx = g->Faces(MAC::X), &fy = g->Faces(MAC::Y);
    double xa = fx[1], xb = fx[fx.size() - 2], ya = fy[1], yb = fy[fy.size() - 2];
    if (s.x1 > s.x0 && s.y1 > s.y0)
    {
        xa = std::max(xa, s.x0); xb = std::min(xb, s.x1);
        ya = std::max(ya, s.y0); yb = std::min(yb, s.y1);
        if (!(xb > xa) || !(yb > ya)) return;
    }

    StreamOccupancy occ;
    occ.x0 = xa; occ.y0 = ya; occ.cell = 0.5 * s.separation;
    occ.nx = std::max(1, (int)std::ceil((xb - xa) / occ.cell));
    occ.ny = std::max(1, (int)std::ceil((yb - ya) / occ.cell));
    occ.bins.assign((size_t)occ.nx * occ.ny, std::vector<StreamOccupancy::Point>());

    int line = 0;
    for (double sy = ya + 0.5 * s.separation; sy < yb; sy += s.separation)
        for (double sx = xa + 0.5 * s.separation; sx < xb; sx += s.separation)
        {
            double u, v;
            if (!VelocityAt(g, sx, sy, u, v) || std::sqrt(u * u + v * v) < s.minSpeed) continue;
            if (OccupancyNear(occ, sx, sy, s.separation, -1, 0, 0, 0)) continue;

            Streamline fwd, bwd;
            TraceStreamHalf(g, s, occ, sx, sy, +1.0, line, 0, fwd);
            TraceStreamHalf(g, s, occ, sx, sy, -1.0, line, 1, bwd);
            line++;

            Streamline joined;
            const size_t n = bwd.x.size() + fwd.x.size();
            if (n < 3) continue;
            joined.x.reserve(n); joined.y.reserve(n); joined.speed.reserve(n);
            for (size_t k = bwd.x.size(); k-- > 0;)
            {
                joined.x.push_back(bwd.x[k]); joined.y.push_back(bwd.y[k]); joined.speed.push_back(bwd.speed[k]);
            }
            joined.x.insert(joined.x.end(), fwd.x.begin(), fwd.x.end());
            joined.y.insert(joined.y.end(), fwd.y.begin(), fwd.y.end());
            joined.speed.insert(joined.speed.end(), fwd.speed.begin(), fwd.speed.end());
            out.push_back(std::move(joined));
        }
}

// ---- probes ----

// `level`: the wave-probe level, the top of the liquid column CONNECTED TO THE BED (solids stepped over, stops at
// the first EMPTY cell), area-averaged over the radius. `top`: the highest particle near the line, spray included.
static double LevelProbe(MAC* g, const Probe &pr)
{
    const bool is3D = (DIMENSION == 3);
    const int jc = g->CellAt(MAC::X, pr.at[0]);
    const int kc = is3D ? g->CellAt(MAC::Z, pr.at[2]) : 0;

    double sum = 0.0, weight = 0.0;
    for (int j = 0; j < g->Nx; j++)
    {
        if (pr.radius > 0.0) { if (std::abs(g->CentrePos(MAC::X, j) - pr.at[0]) > pr.radius) continue; }
        else if (j != jc) continue;
        for (int k = 0; k < (is3D ? g->Nz : 1); k++)
        {
            if (is3D)
            {
                if (pr.radius > 0.0) { if (std::abs(g->CentrePos(MAC::Z, k) - pr.at[2]) > pr.radius) continue; }
                else if (k != kc) continue;
            }
            double top = 0.0;
            bool wet = false;
            for (int i = 0; i < g->Ny; i++)
            {
                const int type = is3D ? (int)g->GetSolid(i, j, k) : (int)g->GetSolid(i, j);
                if (type == SOLID_CELL) { if (wet) break; else continue; }   // step over the bed and any obstacle
                if (type != FLUID_CELL) break;                                // the first air cell ends the column
                top = g->FacePos(MAC::Y, i + 1);
                wet = true;
            }
            const double area = is3D ? (g->Width(MAC::X, j) * g->Width(MAC::Z, k)) : g->Width(MAC::X, j);
            sum += area * (wet ? top : 0.0);
            weight += area;
        }
    }
    return (weight > 0.0) ? sum / weight : 0.0;
}

// The highest particle within the radius (half a cell when 0) of the probe's vertical line; without particles,
// the top face of the highest fluid cell in the column. Spray and sheets count, which is the point of it.
static double TopProbe(MAC* g, const ParticleSet* ps, const Probe &pr)
{
    const bool is3D = (DIMENSION == 3);
    const int j = g->CellAt(MAC::X, pr.at[0]);
    const int k = is3D ? g->CellAt(MAC::Z, pr.at[2]) : 0;
    if (ps != nullptr && ps->count > 0)
    {
        const double hx = (pr.radius > 0.0) ? pr.radius : 0.5 * g->Width(MAC::X, j);
        const double hz = (pr.radius > 0.0) ? pr.radius : (is3D ? 0.5 * g->Width(MAC::Z, k) : 0.0);
        double top = 0.0;
        for (int n = 0; n < ps->count; n++)
        {
            const Particle &p = ps->data[n];
            if (std::abs(p.x - pr.at[0]) > hx || (is3D && std::abs(p.z - pr.at[2]) > hz)) continue;
            top = std::max(top, p.y);
        }
        return top;
    }
    for (int i = g->Ny - 1; i >= 0; i--)
        if ((is3D ? (int)g->GetSolid(i, j, k) : (int)g->GetSolid(i, j)) == FLUID_CELL)
            return g->FacePos(MAC::Y, i + 1);
    return 0.0;
}

// A finite sensor: the cell-volume-weighted mean over the fluid cells whose centre lies within `radius` (a
// top hat, as a diaphragm reads). False when no centre is inside, so the caller interpolates at the point.
static bool AveragedProbe(MAC* g, const Probe &pr, int handle, int velocity, double &out)
{
    const bool is3D = (DIMENSION == 3);
    const int Nx = g->Nx, Ny = g->Ny, Nz = is3D ? g->Nz : 1;
    const double R = pr.radius, R2 = R * R;
    double sum = 0.0, weight = 0.0;
    for (int i = 0; i < Ny; i++)
    {
        const double dy = g->CentrePos(MAC::Y, i) - pr.at[1];
        if (dy * dy > R2) continue;
        for (int j = 0; j < Nx; j++)
        {
            const double dx = g->CentrePos(MAC::X, j) - pr.at[0];
            if (dx * dx + dy * dy > R2) continue;
            for (int k = 0; k < Nz; k++)
            {
                if (is3D)
                {
                    const double dz = g->CentrePos(MAC::Z, k) - pr.at[2];
                    if (dx * dx + dy * dy + dz * dz > R2) continue;
                }
                const int type = is3D ? (int)g->GetSolid(i, j, k) : (int)g->GetSolid(i, j);
                if (type == SOLID_CELL) continue;
                const double vol = is3D ? (g->Width(MAC::X, j) * g->Width(MAC::Y, i) * g->Width(MAC::Z, k)) : (g->Width(MAC::X, j) * g->Width(MAC::Y, i));
                double value;
                if (velocity >= 0)
                {
                    double u = 0.0, v = 0.0, w = 0.0;
                    if (is3D) VelocityAt(g, g->CentrePos(MAC::X, j), g->CentrePos(MAC::Y, i), g->CentrePos(MAC::Z, k), u, v, w);
                    else      VelocityAt(g, g->CentrePos(MAC::X, j), g->CentrePos(MAC::Y, i), u, v);
                    value = (velocity == 0) ? u : (velocity == 1) ? v : (velocity == 2) ? w
                                                 : std::sqrt(u * u + v * v + w * w);
                }
                else if (handle < 0) value = is3D ? g->GetP(i, j, k) : g->GetP(i, j);
                else                 value = is3D ? g->GetS(handle, i, j, k) : g->GetS(handle, i, j);
                sum += vol * value;
                weight += vol;
            }
        }
    }
    if (weight <= 0.0) return false;
    out = sum / weight;
    return true;
}

// The lower of the two cell centres around x along an axis, clamped to [0, n - 2].
static int LowerCentre(MAC* g, MAC::Axis a, double x, int n)
{
    int m = g->CellAt(a, x);
    if (x < g->CentrePos(a, m)) m--;
    return m < 0 ? 0 : (m > n - 2 ? n - 2 : m);
}

// A cell-centred field (-1 the pressure, else scalar `handle`) at a point, linear between the cell centres around it
// with SOLID cells left out and the weights renormalised, so a sensor on a wall reads the fluid beside it.
static double FluidCentredProbe(MAC* g, int handle, double x, double y, double z)
{
    const bool is3D = (DIMENSION == 3);
    const int jb = LowerCentre(g, MAC::X, x, g->Nx), ib = LowerCentre(g, MAC::Y, y, g->Ny);
    const int kb = is3D ? LowerCentre(g, MAC::Z, z, g->Nz) : 0;
    const double cx = std::min(1.0, std::max(0.0, (x - g->CentrePos(MAC::X, jb)) / (g->CentrePos(MAC::X, jb + 1) - g->CentrePos(MAC::X, jb))));
    const double cy = std::min(1.0, std::max(0.0, (y - g->CentrePos(MAC::Y, ib)) / (g->CentrePos(MAC::Y, ib + 1) - g->CentrePos(MAC::Y, ib))));
    const double cz = is3D ? std::min(1.0, std::max(0.0, (z - g->CentrePos(MAC::Z, kb)) / (g->CentrePos(MAC::Z, kb + 1) - g->CentrePos(MAC::Z, kb)))) : 0.0;

    double sum = 0.0, weight = 0.0;
    for (int n = 0; n < (is3D ? 8 : 4); n++)
    {
        const int dj = n & 1, di = (n >> 1) & 1, dk = (n >> 2) & 1;
        const int j = jb + dj, i = ib + di, k = kb + dk;
        const double w = (dj ? cx : 1.0 - cx) * (di ? cy : 1.0 - cy) * (is3D ? (dk ? cz : 1.0 - cz) : 1.0);
        if (w <= 0.0) continue;
        const int type = is3D ? (int)g->GetSolid(i, j, k) : (int)g->GetSolid(i, j);
        if (type == SOLID_CELL) continue;
        double value;
        if (handle < 0) value = is3D ? g->GetP(i, j, k) : g->GetP(i, j);
        else            value = is3D ? g->GetS(handle, i, j, k) : g->GetS(handle, i, j);
        sum += w * value;
        weight += w;
    }
    return (weight > 0.0) ? sum / weight : 0.0;
}

double EvaluateProbe(MAC* g, const ParticleSet* particles, const Probe &pr)
{
    if (g == nullptr || g->Nx < 2 || g->Ny < 2 || (DIMENSION == 3 && g->Nz < 2)) return 0.0;
    const int handle = (pr.kind == ProbeQuantity::Scalar) ? g->ScalarHandle(pr.quantity) : -1;
    if (pr.kind == ProbeQuantity::Scalar && handle < 0) return 0.0;

    // A sensor with a radius averages over its face; one without samples the point.
    if (pr.radius > 0.0 && pr.kind != ProbeQuantity::Level && pr.kind != ProbeQuantity::Top)
    {
        const int velocity = (pr.kind == ProbeQuantity::U) ? 0 : (pr.kind == ProbeQuantity::V) ? 1
                           : (pr.kind == ProbeQuantity::W) ? 2 : (pr.kind == ProbeQuantity::Speed) ? 3 : -1;
        double avg = 0.0;
        if (AveragedProbe(g, pr, handle, velocity, avg)) return avg;
        // Nothing inside the radius: it is smaller than a cell, so fall through to the point value.
    }

    if (pr.kind == ProbeQuantity::Pressure || pr.kind == ProbeQuantity::Scalar)
        return FluidCentredProbe(g, handle, pr.at[0], pr.at[1], DIMENSION == 3 ? pr.at[2] : 0.0);
    if (pr.kind == ProbeQuantity::Level) return LevelProbe(g, pr);
    if (pr.kind == ProbeQuantity::Top) return TopProbe(g, particles, pr);

    // Velocity: the staggered samplers, which read 0 in a solid cell or outside the grid.
    double u = 0.0, v = 0.0, w = 0.0;
    if (DIMENSION == 3) VelocityAt(g, pr.at[0], pr.at[1], pr.at[2], u, v, w);
    else                VelocityAt(g, pr.at[0], pr.at[1], u, v);
    if (pr.kind == ProbeQuantity::U) return u;
    if (pr.kind == ProbeQuantity::V) return v;
    if (pr.kind == ProbeQuantity::W) return w;
    return std::sqrt(u * u + v * v + w * w);
}

void SeedStreamlines3D(const MAC* g, const Streamline3DSettings &s, std::vector<double> &xyz)
{
    xyz.clear();
    if (g == nullptr || s.count <= 0) return;
    // The domain, i.e. the interior faces; the ring lies outside it.
    const std::vector<double> &fx = g->Faces(MAC::X), &fy = g->Faces(MAC::Y), &fz = g->Faces(MAC::Z);
    const double lo[3] = {fx[1], fy[1], fz[1]};
    const double hi[3] = {fx[fx.size() - 2], fy[fy.size() - 2], fz[fz.size() - 2]};
    if (s.source == STREAM_SEED_PLANE)
    {
        const int a = (s.planeAxis + 1) % 3, b = (s.planeAxis + 2) % 3;
        for (int p = 0; p < s.count; p++)
            for (int q = 0; q < s.count; q++)
            {
                double pt[3];
                pt[s.planeAxis] = s.planePosition;
                pt[a] = lo[a] + (p + 0.5) * (hi[a] - lo[a]) / s.count;
                pt[b] = lo[b] + (q + 0.5) * (hi[b] - lo[b]) / s.count;
                xyz.insert(xyz.end(), pt, pt + 3);
            }
    }
    else if (s.source == STREAM_SEED_LINE)
    {
        for (int n = 0; n < s.count; n++)
        {
            const double t = (s.count == 1) ? 0.5 : (double)n / (s.count - 1);
            for (int c = 0; c < 3; c++) xyz.push_back(s.p0[c] + t * (s.p1[c] - s.p0[c]));
        }
    }
    else
    {
        // Fibonacci sphere: near-uniform points without a lattice's pole clustering.
        const double golden = M_PI * (3.0 - std::sqrt(5.0));
        for (int n = 0; n < s.count; n++)
        {
            const double yy = 1.0 - 2.0 * (n + 0.5) / s.count;
            const double r = std::sqrt(std::max(0.0, 1.0 - yy * yy));
            const double th = golden * n;
            xyz.push_back(s.centre[0] + s.radius * r * std::cos(th));
            xyz.push_back(s.centre[1] + s.radius * yy);
            xyz.push_back(s.centre[2] + s.radius * r * std::sin(th));
        }
    }
}

static void TraceStreamHalf3D(MAC* g, const Streamline3DSettings &s, double sx, double sy, double sz, double sign,
                              bool keepSeed, Streamline3D &out)
{
    const double h = s.step;
    double x = sx, y = sy, z = sz;
    for (int n = 0; n < s.maxPoints / 2; n++)
    {
        double u, v, w;
        if (!VelocityAt(g, x, y, z, u, v, w)) break;
        const double speed = std::sqrt(u * u + v * v + w * w);
        if (speed < s.minSpeed) break;
        if (keepSeed || n > 0)
        {
            out.x.push_back(x); out.y.push_back(y); out.z.push_back(z); out.speed.push_back(speed);
        }
        const double ex = sign * u / speed, ey = sign * v / speed, ez = sign * w / speed;
        double um, vm, wm;
        if (!VelocityAt(g, x + 0.5 * h * ex, y + 0.5 * h * ey, z + 0.5 * h * ez, um, vm, wm)) break;
        const double sm = std::sqrt(um * um + vm * vm + wm * wm);
        if (sm < s.minSpeed) break;
        const double mx = sign * um / sm, my = sign * vm / sm, mz = sign * wm / sm;
        if (mx * ex + my * ey + mz * ez < 0.0) break;
        x += h * mx; y += h * my; z += h * mz;
    }
}

void ComputeStreamlines3D(MAC* g, const Streamline3DSettings &s, std::vector<Streamline3D> &out)
{
    out.clear();
    if (g == nullptr || g->Nx < 3 || g->Ny < 3 || g->Nz < 3 || !(s.step > 0.0)) return;
    std::vector<double> seeds;
    SeedStreamlines3D(g, s, seeds);
    for (size_t n = 0; n + 2 < seeds.size(); n += 3)
    {
        Streamline3D fwd, bwd;
        TraceStreamHalf3D(g, s, seeds[n], seeds[n + 1], seeds[n + 2], +1.0, true, fwd);
        if (fwd.x.empty()) continue;
        TraceStreamHalf3D(g, s, seeds[n], seeds[n + 1], seeds[n + 2], -1.0, false, bwd);
        Streamline3D line;
        const size_t m = bwd.x.size() + fwd.x.size();
        if (m < 2) continue;
        line.x.reserve(m); line.y.reserve(m); line.z.reserve(m); line.speed.reserve(m);
        for (size_t k = bwd.x.size(); k-- > 0;)
        {
            line.x.push_back(bwd.x[k]); line.y.push_back(bwd.y[k]); line.z.push_back(bwd.z[k]); line.speed.push_back(bwd.speed[k]);
        }
        line.x.insert(line.x.end(), fwd.x.begin(), fwd.x.end());
        line.y.insert(line.y.end(), fwd.y.begin(), fwd.y.end());
        line.z.insert(line.z.end(), fwd.z.begin(), fwd.z.end());
        line.speed.insert(line.speed.end(), fwd.speed.begin(), fwd.speed.end());
        out.push_back(std::move(line));
    }
}

FlowIntegrals ComputeFlowIntegrals3D(MAC* g)
{
    FlowIntegrals out;
    if (g == nullptr || g->Nx < 3 || g->Ny < 3 || g->Nz < 3) return out;
    const int Nx = g->Nx, Ny = g->Ny, Nz = g->Nz;

    double volume = 0.0, energy = 0.0, enstrophy = 0.0;
    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
                if (g->IsInteriorFluid(i, j, k)) volume += g->Width(MAC::X, j) * g->Width(MAC::Y, i) * g->Width(MAC::Z, k);

    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
            {
                if (!g->IsInteriorFluid(i, j, k)) continue;
                if (j >= 2 && g->IsInteriorFluid(i, j - 1, k))
                {
                    const double u = g->GetU(i, j, k);
                    energy += 0.5 * u * u * g->FaceGap(MAC::X, j - 1) * g->Width(MAC::Y, i) * g->Width(MAC::Z, k);
                }
                if (i >= 2 && g->IsInteriorFluid(i - 1, j, k))
                {
                    const double v = g->GetV(i, j, k);
                    energy += 0.5 * v * v * g->Width(MAC::X, j) * g->FaceGap(MAC::Y, i - 1) * g->Width(MAC::Z, k);
                }
                if (k >= 2 && g->IsInteriorFluid(i, j, k - 1))
                {
                    const double w = g->GetW(i, j, k);
                    energy += 0.5 * w * w * g->Width(MAC::X, j) * g->Width(MAC::Y, i) * g->FaceGap(MAC::Z, k - 1);
                }
            }

    // Each vorticity component lives on the cell edges parallel to its axis, where the two
    // velocity components it differences meet.
    for (int k = 2; k < Nz - 1; k++)
        for (int i = 2; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
            {
                // x edge at (xc j, yf i, zf k): cells (i-1|i, j, k-1|k)
                if (g->IsInteriorFluid(i - 1, j, k - 1) && g->IsInteriorFluid(i, j, k - 1) &&
                    g->IsInteriorFluid(i - 1, j, k) && g->IsInteriorFluid(i, j, k))
                {
                    const double dwdy = (g->GetW(i, j, k) - g->GetW(i - 1, j, k)) / g->FaceGap(MAC::Y, i - 1);
                    const double dvdz = (g->GetV(i, j, k) - g->GetV(i, j, k - 1)) / g->FaceGap(MAC::Z, k - 1);
                    const double wx = dwdy - dvdz;
                    enstrophy += 0.5 * wx * wx * g->Width(MAC::X, j) * g->FaceGap(MAC::Y, i - 1) * g->FaceGap(MAC::Z, k - 1);
                }
            }
    for (int k = 2; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 2; j < Nx - 1; j++)
            {
                // y edge at (xf j, yc i, zf k): cells (i, j-1|j, k-1|k)
                if (g->IsInteriorFluid(i, j - 1, k - 1) && g->IsInteriorFluid(i, j, k - 1) &&
                    g->IsInteriorFluid(i, j - 1, k) && g->IsInteriorFluid(i, j, k))
                {
                    const double dudz = (g->GetU(i, j, k) - g->GetU(i, j, k - 1)) / g->FaceGap(MAC::Z, k - 1);
                    const double dwdx = (g->GetW(i, j, k) - g->GetW(i, j - 1, k)) / g->FaceGap(MAC::X, j - 1);
                    const double wy = dudz - dwdx;
                    enstrophy += 0.5 * wy * wy * g->FaceGap(MAC::X, j - 1) * g->Width(MAC::Y, i) * g->FaceGap(MAC::Z, k - 1);
                }
            }
    for (int k = 1; k < Nz - 1; k++)
        for (int i = 2; i < Ny - 1; i++)
            for (int j = 2; j < Nx - 1; j++)
            {
                // z edge at (xf j, yf i, zc k): cells (i-1|i, j-1|j, k)
                if (g->IsInteriorFluid(i - 1, j - 1, k) && g->IsInteriorFluid(i - 1, j, k) &&
                    g->IsInteriorFluid(i, j - 1, k) && g->IsInteriorFluid(i, j, k))
                {
                    const double dvdx = (g->GetV(i, j, k) - g->GetV(i, j - 1, k)) / g->FaceGap(MAC::X, j - 1);
                    const double dudy = (g->GetU(i, j, k) - g->GetU(i - 1, j, k)) / g->FaceGap(MAC::Y, i - 1);
                    const double wz = dvdx - dudy;
                    enstrophy += 0.5 * wz * wz * g->FaceGap(MAC::X, j - 1) * g->FaceGap(MAC::Y, i - 1) * g->Width(MAC::Z, k);
                }
            }

    out.fluidVolume = volume;
    if (volume > 0.0)
    {
        out.kineticEnergy = energy / volume;
        out.enstrophy = enstrophy / volume;
    }
    return out;
}

void AeroForces2D(const MAC* gridIn, int Nx, int Ny, double nu, double& Fd, double& Fl)
{
    Fd = 0.0;
    Fl = 0.0;
    if (gridIn == nullptr || Nx <= 2 || Ny <= 2) return;

    MAC* grid = const_cast<MAC*>(gridIn); // the MAC accessors are not const-qualified
    const double mi = nu;
    // GetP is the actual pressure and nu is kinematic, so the pressure is divided by rho to keep the
    // whole traction per unit density; the caller multiplies by rho once.
    const double invRho = 1.0 / ((SIMULATION.RHO > 0.0) ? SIMULATION.RHO : 1.0);

    // The normal viscous stress carries a coefficient of ONE, not two. This is the GRADIENT form of
    // the traction, and it is deliberate -- do not "correct" it to 2*mu.
    double dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;

    // For all non-boundary cell centers
    for(int i = 1; i < Ny-1; i++) {
        for(int j = 1; j < Nx-1; j++) {
            // Skip cells that are not solid
            if(grid->GetSolid(i,j) != SOLID_CELL) {
                continue;
            }

            // Every spacing below is the LOCAL one.
            const double dxj = grid->Width(MAC::X, j), dyi = grid->Width(MAC::Y, i);

            // Process solid cells that have ANY interface with fluid
            // Right face - normal vector (1,0)
            if(grid->GetSolid(i,j+1) == FLUID_CELL) {
                // u(i,j+2) sits one CELL (j+1) beyond the wall face at FacePos(X, j+1).
                dudx = (grid->GetU(i,j+2) - 0) / grid->Width(MAC::X, j+1);
                dvdx = (grid->getVatU(i,j+2) - 0) / grid->Width(MAC::X, j+1);
                // getUatV(i,.) sits at FacePos(Y, i), so i -> i+1 spans one cell in y.
                dudy = (grid->getUatV(i+1,j+1)- (grid->getUatV(i,j+1))) / dyi;

                // Stress components for right face
                double tau_xx = 1.0 * mi * dudx - grid->GetP(i,j+1) * invRho;
                double tau_yx = mi * (dudy + dvdx);

                // For right face (normal: 1,0): the face length is the cell's y extent
                Fd += tau_xx * dyi;
                Fl += tau_yx * dyi;
            }

            // Left face - normal vector (-1,0)
            if(grid->GetSolid(i,j-1) == FLUID_CELL) {
                dudx = (0 - grid->GetU(i,j-1)) / grid->Width(MAC::X, j-1);
                dvdx = (0 - grid->getVatU(i,j-1)) / grid->Width(MAC::X, j-1);
                dudy = ((grid->getUatV(i+1,j-1))- (grid->getUatV(i,j-1))) / dyi;

                // Stress components for left face
                double tau_xx = 1.0 * mi * dudx - grid->GetP(i,j-1) * invRho;
                double tau_yx = mi * (dudy + dvdx);

                // For left face (normal: -1,0):
                Fd += -tau_xx * dyi;
                Fl += -tau_yx * dyi;
            }

            // Top face - normal vector (0,1)
            if(grid->GetSolid(i+1,j) == FLUID_CELL) {
                dvdy = (grid->GetV(i+2,j) - 0) / grid->Width(MAC::Y, i+1);
                // getVatU(.,j) sits at FacePos(X, j), so j -> j+1 spans one cell in x.
                dvdx = ((grid->getVatU(i+1,j+1))- (grid->getVatU(i+1,j))) / dxj;
                dudy = (grid->getUatV(i+2,j) - 0) / grid->Width(MAC::Y, i+1);

                // Stress components for top face
                double tau_xy = mi * (dudy + dvdx);
                double tau_yy = 1.0 * mi * dvdy - grid->GetP(i+1,j) * invRho;

                // For top face (normal: 0,1): the face length is the cell's x extent
                Fd += tau_xy * dxj;
                Fl += tau_yy * dxj;
            }

            // Bottom face - normal vector (0,-1)
            if(grid->GetSolid(i-1,j) == FLUID_CELL) {
                dvdy = (0 - grid->GetV(i-1,j)) / grid->Width(MAC::Y, i-1);
                dvdx = (grid->getVatU(i-1,j+1)- (grid->getVatU(i-1,j))) / dxj;
                dudy = (0 - grid->getUatV(i-1,j)) / grid->Width(MAC::Y, i-1);

                // Stress components for bottom face
                double tau_xy = mi * (dudy + dvdx);
                double tau_yy = 1.0 * mi * dvdy - grid->GetP(i-1,j) * invRho;

                // For bottom face (normal: 0,-1):
                Fd += -tau_xy * dxj;
                Fl += -tau_yy * dxj;
            }
        }
    }
}

// A cell-centred value linearly at face n of axis a, from the centres either side.
static double CentresToFace(MAC* g, MAC::Axis a, int n, double lo, double hi)
{
    const double c0 = g->CentrePos(a, n - 1), c1 = g->CentrePos(a, n);
    const double w = (g->FacePos(a, n) - c0) / (c1 - c0);
    return lo + w * (hi - lo);
}

bool AeroCVBox2D(const MAC* gridIn, double margin, int& jL, int& jR, int& iB, int& iT)
{
    if (gridIn == nullptr) return false;
    MAC* g = const_cast<MAC*>(gridIn);
    const int Nx = g->Nx, Ny = g->Ny;
    int iMin = Ny, iMax = -1, jMin = Nx, jMax = -1;
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            if (g->GetSolid(i, j) == SOLID_CELL)
            {
                iMin = (std::min)(iMin, i); iMax = (std::max)(iMax, i);
                jMin = (std::min)(jMin, j); jMax = (std::max)(jMax, j);
            }
    if (iMax < iMin || jMax < jMin) return false;
    const double x0 = g->FacePos(MAC::X, jMin), x1 = g->FacePos(MAC::X, jMax + 1);
    const double y0 = g->FacePos(MAC::Y, iMin), y1 = g->FacePos(MAC::Y, iMax + 1);
    const double m = margin * (std::max)(x1 - x0, y1 - y0);
    // The faces nearest the requested box; the stencils reach one cell past a box face and must stay off the ring.
    jL = jMin; while (jL > 2 && g->FacePos(MAC::X, jL) > x0 - m) jL--;
    jR = jMax + 1; while (jR < Nx - 2 && g->FacePos(MAC::X, jR) < x1 + m) jR++;
    iB = iMin; while (iB > 2 && g->FacePos(MAC::Y, iB) > y0 - m) iB--;
    iT = iMax + 1; while (iT < Ny - 2 && g->FacePos(MAC::Y, iT) < y1 + m) iT++;
    return jL < jMin && jR > jMax + 1 && iB < iMin && iT > iMax + 1;
}

void AeroForcesCV2D(const MAC* gridIn, const MAC* prevIn, double dt, double rho, double nu,
                    int jL, int jR, int iB, int iT, AeroCVParts2D& out)
{
    out = AeroCVParts2D();
    out.jL = jL; out.jR = jR; out.iB = iB; out.iT = iT;
    if (gridIn == nullptr || jR <= jL || iT <= iB) return;
    MAC* g = const_cast<MAC*>(gridIn);
    const double irho = (rho > 0.0) ? 1.0 / rho : 1.0;
    const MAC::Axis X = MAC::X, Y = MAC::Y;

    // Vertical faces x = FacePos(X, j), j = jL (n = -x) and jR (n = +x): u native there, p and du/dx from the centres
    // either side, du/dy and dv/dx by central differences.
    for (int side = 0; side < 2; side++)
    {
        const int j = side ? jR : jL;
        const double sgn = side ? 1.0 : -1.0;
        for (int i = iB; i < iT; i++)
        {
            const double A = g->Width(Y, i);
            const double u = g->GetU(i, j), v = g->getVatU(i, j);
            const double p = CentresToFace(g, X, j, g->GetP(i, j - 1), g->GetP(i, j));
            const double dudx = CentresToFace(g, X, j, (g->GetU(i, j) - g->GetU(i, j - 1)) / g->Width(X, j - 1),
                                              (g->GetU(i, j + 1) - g->GetU(i, j)) / g->Width(X, j));
            const double dudy = (g->GetU(i + 1, j) - g->GetU(i - 1, j)) / (g->CentrePos(Y, i + 1) - g->CentrePos(Y, i - 1));
            const double hx = g->CentrePos(X, j) - g->CentrePos(X, j - 1);
            const double dvdx = 0.5 * ((g->GetV(i, j) - g->GetV(i, j - 1)) + (g->GetV(i + 1, j) - g->GetV(i + 1, j - 1))) / hx;
            const double un = sgn * u;
            out.pressure[0]  += -p * irho * sgn * A;
            out.momentum[0]  += -u * un * A;
            out.momentum[1]  += -v * un * A;
            out.viscGrad[0]  += nu * dudx * sgn * A;
            out.viscGrad[1]  += nu * dvdx * sgn * A;
            out.viscTrans[0] += nu * dudx * sgn * A;
            out.viscTrans[1] += nu * dudy * sgn * A;
        }
    }
    // Horizontal faces y = FacePos(Y, i), i = iB (n = -y) and iT (n = +y).
    for (int side = 0; side < 2; side++)
    {
        const int i = side ? iT : iB;
        const double sgn = side ? 1.0 : -1.0;
        for (int j = jL; j < jR; j++)
        {
            const double A = g->Width(X, j);
            const double v = g->GetV(i, j), u = g->getUatV(i, j);
            const double p = CentresToFace(g, Y, i, g->GetP(i - 1, j), g->GetP(i, j));
            const double dvdy = CentresToFace(g, Y, i, (g->GetV(i, j) - g->GetV(i - 1, j)) / g->Width(Y, i - 1),
                                              (g->GetV(i + 1, j) - g->GetV(i, j)) / g->Width(Y, i));
            const double dvdx = (g->GetV(i, j + 1) - g->GetV(i, j - 1)) / (g->CentrePos(X, j + 1) - g->CentrePos(X, j - 1));
            const double hy = g->CentrePos(Y, i) - g->CentrePos(Y, i - 1);
            const double dudy = 0.5 * ((g->GetU(i, j) - g->GetU(i - 1, j)) + (g->GetU(i, j + 1) - g->GetU(i - 1, j + 1))) / hy;
            const double vn = sgn * v;
            out.pressure[1]  += -p * irho * sgn * A;
            out.momentum[0]  += -u * vn * A;
            out.momentum[1]  += -v * vn * A;
            out.viscGrad[0]  += nu * dudy * sgn * A;
            out.viscGrad[1]  += nu * dvdy * sgn * A;
            out.viscTrans[0] += nu * dvdx * sgn * A;
            out.viscTrans[1] += nu * dvdy * sgn * A;
        }
    }
    // d/dt of the momentum inside: each staggered velocity carries the volume around its face; a face ON the box keeps
    // only its inside half.
    if (prevIn != nullptr && dt > 0.0)
    {
        MAC* q = const_cast<MAC*>(prevIn);
        double mu = 0.0, mv = 0.0;
        for (int i = iB; i < iT; i++)
            for (int j = jL; j <= jR; j++)
            {
                const double w = (j == jL) ? 0.5 * g->Width(X, jL) : (j == jR) ? 0.5 * g->Width(X, jR - 1)
                                                                              : 0.5 * (g->Width(X, j - 1) + g->Width(X, j));
                mu += (g->GetU(i, j) - q->GetU(i, j)) * w * g->Width(Y, i);
            }
        for (int i = iB; i <= iT; i++)
            for (int j = jL; j < jR; j++)
            {
                const double h = (i == iB) ? 0.5 * g->Width(Y, iB) : (i == iT) ? 0.5 * g->Width(Y, iT - 1)
                                                                              : 0.5 * (g->Width(Y, i - 1) + g->Width(Y, i));
                mv += (g->GetV(i, j) - q->GetV(i, j)) * h * g->Width(X, j);
            }
        out.unsteady[0] = mu / dt;
        out.unsteady[1] = mv / dt;
    }
    out.valid = true;
}

// The derivative at x1 of the parabola through three samples: the central difference on a uniform grid, second order
// on a stretched one, and with a wall sample it is the one-sided wall derivative the no-slip condition implies.
static double Derivative3(double x0, double f0, double x1, double f1, double x2, double f2)
{
    return f0 * (x1 - x2) / ((x0 - x1) * (x0 - x2)) + f1 * (2.0 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2)) +
           f2 * (x1 - x0) / ((x2 - x0) * (x2 - x1));
}

// d/d(axis a) of a velocity sampled at centre positions n-1, n, n+1 along a; a neighbour in the SOLID ring is replaced
// by the wall (value 0) on the face between them.
static double TangentialDerivative(MAC* g, MAC::Axis a, int n, bool loWall, bool hiWall, double f0, double f1, double f2)
{
    const double x0 = loWall ? g->FacePos(a, n) : g->CentrePos(a, n - 1);
    const double x2 = hiWall ? g->FacePos(a, n + 1) : g->CentrePos(a, n + 1);
    return Derivative3(x0, loWall ? 0.0 : f0, g->CentrePos(a, n), f1, x2, hiWall ? 0.0 : f2);
}

static inline bool RingWall(MAC* g, int i, int j, int k)
{
    return g->GetSolid(i, j, k) == SOLID_CELL;
}

bool AeroCVBox3D(const MAC* gridIn, double margin, int& jL, int& jR, int& iB, int& iT, int& kB, int& kT)
{
    if (gridIn == nullptr) return false;
    MAC* g = const_cast<MAC*>(gridIn);
    const int Nx = g->Nx, Ny = g->Ny, Nz = g->Nz;
    int iMin = Ny, iMax = -1, jMin = Nx, jMax = -1, kMin = Nz, kMax = -1;
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
                if (g->GetSolid(i, j, k) == SOLID_CELL)
                {
                    iMin = (std::min)(iMin, i); iMax = (std::max)(iMax, i);
                    jMin = (std::min)(jMin, j); jMax = (std::max)(jMax, j);
                    kMin = (std::min)(kMin, k); kMax = (std::max)(kMax, k);
                }
    if (iMax < iMin) return false;
    const double x0 = g->FacePos(MAC::X, jMin), x1 = g->FacePos(MAC::X, jMax + 1);
    const double y0 = g->FacePos(MAC::Y, iMin), y1 = g->FacePos(MAC::Y, iMax + 1);
    const double z0 = g->FacePos(MAC::Z, kMin), z1 = g->FacePos(MAC::Z, kMax + 1);
    const double m = margin * (std::max)(x1 - x0, y1 - y0);
    jL = jMin; while (jL > 2 && g->FacePos(MAC::X, jL) > x0 - m) jL--;
    jR = jMax + 1; while (jR < Nx - 2 && g->FacePos(MAC::X, jR) < x1 + m) jR++;
    iB = iMin; while (iB > 2 && g->FacePos(MAC::Y, iB) > y0 - m) iB--;
    iT = iMax + 1; while (iT < Ny - 2 && g->FacePos(MAC::Y, iT) < y1 + m) iT++;
    const bool spans = (kMin == 1 && kMax == Nz - 2);
    if (spans) { kB = 1; kT = Nz - 1; }
    else
    {
        kB = kMin; while (kB > 2 && g->FacePos(MAC::Z, kB) > z0 - m) kB--;
        kT = kMax + 1; while (kT < Nz - 2 && g->FacePos(MAC::Z, kT) < z1 + m) kT++;
    }
    return jL < jMin && jR > jMax + 1 && iB < iMin && iT > iMax + 1 && (spans || (kB < kMin && kT > kMax + 1));
}

void AeroForcesCV3D(const MAC* gridIn, const MAC* prevIn, double dt, double rho, double nu,
                    int jL, int jR, int iB, int iT, int kB, int kT, AeroCVParts3D& out)
{
    out = AeroCVParts3D();
    out.jL = jL; out.jR = jR; out.iB = iB; out.iT = iT; out.kB = kB; out.kT = kT;
    if (gridIn == nullptr || jR <= jL || iT <= iB || kT <= kB) return;
    MAC* g = const_cast<MAC*>(gridIn);
    const double irho = (rho > 0.0) ? 1.0 / rho : 1.0;
    const MAC::Axis X = MAC::X, Y = MAC::Y, Z = MAC::Z;
    const int Nz = g->Nz;

    // x faces at FacePos(X, j), j = jL (n = -x) and jR (n = +x): u native, p and du/dx from the centres either side,
    // v and w from their four faces around, the u tangential derivatives by Derivative3.
    for (int side = 0; side < 2; side++)
    {
        const int j = side ? jR : jL;
        const double sgn = side ? 1.0 : -1.0;
        const double hx = g->CentrePos(X, j) - g->CentrePos(X, j - 1);
        for (int i = iB; i < iT; i++)
            for (int k = kB; k < kT; k++)
            {
                const double A = g->Width(Y, i) * g->Width(Z, k);
                const bool zLo = RingWall(g, i, j, k - 1), zHi = RingWall(g, i, j, k + 1);
                const double u = g->GetU(i, j, k), v = g->getVatU(i, j, k), w = g->getWatU(i, j, k);
                const double p = CentresToFace(g, X, j, g->GetP(i, j - 1, k), g->GetP(i, j, k));
                const double dudx = CentresToFace(g, X, j, (g->GetU(i, j, k) - g->GetU(i, j - 1, k)) / g->Width(X, j - 1),
                                                  (g->GetU(i, j + 1, k) - g->GetU(i, j, k)) / g->Width(X, j));
                const double dudy = TangentialDerivative(g, Y, i, false, false, g->GetU(i - 1, j, k), u, g->GetU(i + 1, j, k));
                const double dudz = TangentialDerivative(g, Z, k, zLo, zHi, g->GetU(i, j, k - 1), u, g->GetU(i, j, k + 1));
                const double dvdx = 0.5 * ((g->GetV(i, j, k) - g->GetV(i, j - 1, k)) +
                                           (g->GetV(i + 1, j, k) - g->GetV(i + 1, j - 1, k))) / hx;
                const double dwdx = 0.5 * ((g->GetW(i, j, k) - g->GetW(i, j - 1, k)) +
                                           (g->GetW(i, j, k + 1) - g->GetW(i, j - 1, k + 1))) / hx;
                const double un = sgn * u;
                out.pressure[0]  += -p * irho * sgn * A;
                out.momentum[0]  += -u * un * A;  out.momentum[1] += -v * un * A;  out.momentum[2] += -w * un * A;
                out.viscGrad[0]  += nu * dudx * sgn * A;
                out.viscGrad[1]  += nu * dvdx * sgn * A;
                out.viscGrad[2]  += nu * dwdx * sgn * A;
                out.viscTrans[0] += nu * dudx * sgn * A;
                out.viscTrans[1] += nu * dudy * sgn * A;
                out.viscTrans[2] += nu * dudz * sgn * A;
            }
    }
    // y faces at FacePos(Y, i), i = iB (n = -y) and iT (n = +y).
    for (int side = 0; side < 2; side++)
    {
        const int i = side ? iT : iB;
        const double sgn = side ? 1.0 : -1.0;
        const double hy = g->CentrePos(Y, i) - g->CentrePos(Y, i - 1);
        for (int j = jL; j < jR; j++)
            for (int k = kB; k < kT; k++)
            {
                const double A = g->Width(X, j) * g->Width(Z, k);
                const bool zLo = RingWall(g, i, j, k - 1), zHi = RingWall(g, i, j, k + 1);
                const double v = g->GetV(i, j, k), u = g->getUatV(i, j, k), w = g->getWatV(i, j, k);
                const double p = CentresToFace(g, Y, i, g->GetP(i - 1, j, k), g->GetP(i, j, k));
                const double dvdy = CentresToFace(g, Y, i, (g->GetV(i, j, k) - g->GetV(i - 1, j, k)) / g->Width(Y, i - 1),
                                                  (g->GetV(i + 1, j, k) - g->GetV(i, j, k)) / g->Width(Y, i));
                const double dvdx = TangentialDerivative(g, X, j, false, false, g->GetV(i, j - 1, k), v, g->GetV(i, j + 1, k));
                const double dvdz = TangentialDerivative(g, Z, k, zLo, zHi, g->GetV(i, j, k - 1), v, g->GetV(i, j, k + 1));
                const double dudy = 0.5 * ((g->GetU(i, j, k) - g->GetU(i - 1, j, k)) +
                                           (g->GetU(i, j + 1, k) - g->GetU(i - 1, j + 1, k))) / hy;
                const double dwdy = 0.5 * ((g->GetW(i, j, k) - g->GetW(i - 1, j, k)) +
                                           (g->GetW(i, j, k + 1) - g->GetW(i - 1, j, k + 1))) / hy;
                const double vn = sgn * v;
                out.pressure[1]  += -p * irho * sgn * A;
                out.momentum[0]  += -u * vn * A;  out.momentum[1] += -v * vn * A;  out.momentum[2] += -w * vn * A;
                out.viscGrad[0]  += nu * dudy * sgn * A;
                out.viscGrad[1]  += nu * dvdy * sgn * A;
                out.viscGrad[2]  += nu * dwdy * sgn * A;
                out.viscTrans[0] += nu * dvdx * sgn * A;
                out.viscTrans[1] += nu * dvdy * sgn * A;
                out.viscTrans[2] += nu * dvdz * sgn * A;
            }
    }
    // z faces at FacePos(Z, k), k = kB (n = -z) and kT (n = +z). On a wall (the ring beyond it SOLID) the velocity is 0:
    // no momentum flux, and the traction is the wall shear nu du_t/dz over the half cell, on the fluid cells only.
    for (int side = 0; side < 2; side++)
    {
        const int k = side ? kT : kB;
        const int kIn = side ? kT - 1 : kB, kOut = side ? kT : kB - 1;
        const double sgn = side ? 1.0 : -1.0;
        for (int i = iB; i < iT; i++)
            for (int j = jL; j < jR; j++)
            {
                const double A = g->Width(X, j) * g->Width(Y, i);
                if (RingWall(g, i, j, kOut) && (kOut == 0 || kOut == Nz - 1))
                {
                    if (g->GetSolid(i, j, kIn) == SOLID_CELL) continue;
                    const double d = std::abs(g->CentrePos(Z, kIn) - g->FacePos(Z, k));
                    const double uc = 0.5 * (g->GetU(i, j, kIn) + g->GetU(i, j + 1, kIn));
                    const double vc = 0.5 * (g->GetV(i, j, kIn) + g->GetV(i + 1, j, kIn));
                    // sgn * du/dz = -u_c / d on either wall
                    out.pressure[2] += -g->GetP(i, j, kIn) * irho * sgn * A;
                    out.viscGrad[0] += -nu * uc / d * A;
                    out.viscGrad[1] += -nu * vc / d * A;
                    continue;
                }
                const double hz = g->CentrePos(Z, k) - g->CentrePos(Z, k - 1);
                const double w = g->GetW(i, j, k), u = g->getUatW(i, j, k), v = g->getVatW(i, j, k);
                const double p = CentresToFace(g, Z, k, g->GetP(i, j, k - 1), g->GetP(i, j, k));
                const double dwdz = CentresToFace(g, Z, k, (g->GetW(i, j, k) - g->GetW(i, j, k - 1)) / g->Width(Z, k - 1),
                                                  (g->GetW(i, j, k + 1) - g->GetW(i, j, k)) / g->Width(Z, k));
                const double dwdx = TangentialDerivative(g, X, j, false, false, g->GetW(i, j - 1, k), w, g->GetW(i, j + 1, k));
                const double dwdy = TangentialDerivative(g, Y, i, false, false, g->GetW(i - 1, j, k), w, g->GetW(i + 1, j, k));
                const double dudz = 0.5 * ((g->GetU(i, j, k) - g->GetU(i, j, k - 1)) +
                                           (g->GetU(i, j + 1, k) - g->GetU(i, j + 1, k - 1))) / hz;
                const double dvdz = 0.5 * ((g->GetV(i, j, k) - g->GetV(i, j, k - 1)) +
                                           (g->GetV(i + 1, j, k) - g->GetV(i + 1, j, k - 1))) / hz;
                const double wn = sgn * w;
                out.pressure[2]  += -p * irho * sgn * A;
                out.momentum[0]  += -u * wn * A;  out.momentum[1] += -v * wn * A;  out.momentum[2] += -w * wn * A;
                out.viscGrad[0]  += nu * dudz * sgn * A;
                out.viscGrad[1]  += nu * dvdz * sgn * A;
                out.viscGrad[2]  += nu * dwdz * sgn * A;
                out.viscTrans[0] += nu * dwdx * sgn * A;
                out.viscTrans[1] += nu * dwdy * sgn * A;
                out.viscTrans[2] += nu * dwdz * sgn * A;
            }
    }
    // d/dt of the momentum inside: each staggered velocity carries the volume around its face, a face on the box only
    // its inside half.
    if (prevIn != nullptr && dt > 0.0)
    {
        MAC* q = const_cast<MAC*>(prevIn);
        double mu = 0.0, mv = 0.0, mw = 0.0;
        for (int i = iB; i < iT; i++)
            for (int j = jL; j <= jR; j++)
                for (int k = kB; k < kT; k++)
                {
                    const double wx = (j == jL) ? 0.5 * g->Width(X, jL) : (j == jR) ? 0.5 * g->Width(X, jR - 1)
                                                                            : 0.5 * (g->Width(X, j - 1) + g->Width(X, j));
                    mu += (g->GetU(i, j, k) - q->GetU(i, j, k)) * wx * g->Width(Y, i) * g->Width(Z, k);
                }
        for (int i = iB; i <= iT; i++)
            for (int j = jL; j < jR; j++)
                for (int k = kB; k < kT; k++)
                {
                    const double wy = (i == iB) ? 0.5 * g->Width(Y, iB) : (i == iT) ? 0.5 * g->Width(Y, iT - 1)
                                                                            : 0.5 * (g->Width(Y, i - 1) + g->Width(Y, i));
                    mv += (g->GetV(i, j, k) - q->GetV(i, j, k)) * wy * g->Width(X, j) * g->Width(Z, k);
                }
        for (int i = iB; i < iT; i++)
            for (int j = jL; j < jR; j++)
                for (int k = kB; k <= kT; k++)
                {
                    const double wz = (k == kB) ? 0.5 * g->Width(Z, kB) : (k == kT) ? 0.5 * g->Width(Z, kT - 1)
                                                                            : 0.5 * (g->Width(Z, k - 1) + g->Width(Z, k));
                    mw += (g->GetW(i, j, k) - q->GetW(i, j, k)) * wz * g->Width(X, j) * g->Width(Y, i);
                }
        out.unsteady[0] = mu / dt;
        out.unsteady[1] = mv / dt;
        out.unsteady[2] = mw / dt;
    }
    out.valid = true;
}

void AeroForces3D(const MAC* gridIn, int Nx, int Ny, int Nz, double nu, AeroForces3DParts& out)
{
    out = AeroForces3DParts();
    if (gridIn == nullptr || Nx <= 2 || Ny <= 2 || Nz <= 2) return;
    MAC* grid = const_cast<MAC*>(gridIn); // the MAC accessors are not const-qualified
    const double mi = nu;
    const double invRho = 1.0 / ((SIMULATION.RHO > 0.0) ? SIMULATION.RHO : 1.0);   // per unit density, as in 2D

    // The 3D port of AeroForces2D, face for face.
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                if (grid->GetSolid(i, j, k) != SOLID_CELL) continue;
                const double dxj = grid->Width(MAC::X, j), dyi = grid->Width(MAC::Y, i), dzk = grid->Width(MAC::Z, k);
                const double Ax = dyi * dzk, Ay = dxj * dzk, Az = dxj * dyi;

                // ===== x+ face, n = (1,0,0) =====
                if (grid->GetSolid(i, j + 1, k) == FLUID_CELL)
                {
                    const double h = grid->Width(MAC::X, j + 1);
                    const double dudx = grid->GetU(i, j + 2, k) / h;
                    const double dvdx = grid->getVatU(i, j + 2, k) / h;
                    const double dwdx = grid->getWatU(i, j + 2, k) / h;
                    const double dudy = (grid->getUatV(i + 1, j + 1, k) - grid->getUatV(i, j + 1, k)) / dyi;
                    const double dudz = (grid->getUatW(i, j + 1, k + 1) - grid->getUatW(i, j + 1, k)) / dzk;
                    out.pressure[0]   += -grid->GetP(i, j + 1, k) * invRho * Ax;
                    out.normalVisc[0] += mi * dudx * Ax;
                    out.shearGrad[1]  += mi * dvdx * Ax;  out.shearTrans[1] += mi * dudy * Ax;
                    out.shearGrad[2]  += mi * dwdx * Ax;  out.shearTrans[2] += mi * dudz * Ax;
                }
                // ===== x- face, n = (-1,0,0) =====
                if (grid->GetSolid(i, j - 1, k) == FLUID_CELL)
                {
                    const double h = grid->Width(MAC::X, j - 1);
                    const double dudx = (0.0 - grid->GetU(i, j - 1, k)) / h;
                    const double dvdx = (0.0 - grid->getVatU(i, j - 1, k)) / h;
                    const double dwdx = (0.0 - grid->getWatU(i, j - 1, k)) / h;
                    const double dudy = (grid->getUatV(i + 1, j - 1, k) - grid->getUatV(i, j - 1, k)) / dyi;
                    const double dudz = (grid->getUatW(i, j - 1, k + 1) - grid->getUatW(i, j - 1, k)) / dzk;
                    out.pressure[0]   -= -grid->GetP(i, j - 1, k) * invRho * Ax;
                    out.normalVisc[0] -= mi * dudx * Ax;
                    out.shearGrad[1]  -= mi * dvdx * Ax;  out.shearTrans[1] -= mi * dudy * Ax;
                    out.shearGrad[2]  -= mi * dwdx * Ax;  out.shearTrans[2] -= mi * dudz * Ax;
                }
                // ===== y+ face, n = (0,1,0) =====
                if (grid->GetSolid(i + 1, j, k) == FLUID_CELL)
                {
                    const double h = grid->Width(MAC::Y, i + 1);
                    const double dvdy = grid->GetV(i + 2, j, k) / h;
                    const double dudy = grid->getUatV(i + 2, j, k) / h;
                    const double dwdy = grid->getWatV(i + 2, j, k) / h;
                    const double dvdx = (grid->getVatU(i + 1, j + 1, k) - grid->getVatU(i + 1, j, k)) / dxj;
                    const double dvdz = (grid->getVatW(i + 1, j, k + 1) - grid->getVatW(i + 1, j, k)) / dzk;
                    out.pressure[1]   += -grid->GetP(i + 1, j, k) * invRho * Ay;
                    out.normalVisc[1] += mi * dvdy * Ay;
                    out.shearGrad[0]  += mi * dudy * Ay;  out.shearTrans[0] += mi * dvdx * Ay;
                    out.shearGrad[2]  += mi * dwdy * Ay;  out.shearTrans[2] += mi * dvdz * Ay;
                }
                // ===== y- face, n = (0,-1,0) =====
                if (grid->GetSolid(i - 1, j, k) == FLUID_CELL)
                {
                    const double h = grid->Width(MAC::Y, i - 1);
                    const double dvdy = (0.0 - grid->GetV(i - 1, j, k)) / h;
                    const double dudy = (0.0 - grid->getUatV(i - 1, j, k)) / h;
                    const double dwdy = (0.0 - grid->getWatV(i - 1, j, k)) / h;
                    const double dvdx = (grid->getVatU(i - 1, j + 1, k) - grid->getVatU(i - 1, j, k)) / dxj;
                    const double dvdz = (grid->getVatW(i - 1, j, k + 1) - grid->getVatW(i - 1, j, k)) / dzk;
                    out.pressure[1]   -= -grid->GetP(i - 1, j, k) * invRho * Ay;
                    out.normalVisc[1] -= mi * dvdy * Ay;
                    out.shearGrad[0]  -= mi * dudy * Ay;  out.shearTrans[0] -= mi * dvdx * Ay;
                    out.shearGrad[2]  -= mi * dwdy * Ay;  out.shearTrans[2] -= mi * dvdz * Ay;
                }
                // ===== z+ face, n = (0,0,1) =====
                if (grid->GetSolid(i, j, k + 1) == FLUID_CELL)
                {
                    const double h = grid->Width(MAC::Z, k + 1);
                    const double dwdz = grid->GetW(i, j, k + 2) / h;
                    const double dudz = grid->getUatW(i, j, k + 2) / h;
                    const double dvdz = grid->getVatW(i, j, k + 2) / h;
                    const double dwdx = (grid->getWatU(i, j + 1, k + 1) - grid->getWatU(i, j, k + 1)) / dxj;
                    const double dwdy = (grid->getWatV(i + 1, j, k + 1) - grid->getWatV(i, j, k + 1)) / dyi;
                    out.pressure[2]   += -grid->GetP(i, j, k + 1) * invRho * Az;
                    out.normalVisc[2] += mi * dwdz * Az;
                    out.shearGrad[0]  += mi * dudz * Az;  out.shearTrans[0] += mi * dwdx * Az;
                    out.shearGrad[1]  += mi * dvdz * Az;  out.shearTrans[1] += mi * dwdy * Az;
                }
                // ===== z- face, n = (0,0,-1) =====
                if (grid->GetSolid(i, j, k - 1) == FLUID_CELL)
                {
                    const double h = grid->Width(MAC::Z, k - 1);
                    const double dwdz = (0.0 - grid->GetW(i, j, k - 1)) / h;
                    const double dudz = (0.0 - grid->getUatW(i, j, k - 1)) / h;
                    const double dvdz = (0.0 - grid->getVatW(i, j, k - 1)) / h;
                    const double dwdx = (grid->getWatU(i, j + 1, k - 1) - grid->getWatU(i, j, k - 1)) / dxj;
                    const double dwdy = (grid->getWatV(i + 1, j, k - 1) - grid->getWatV(i, j, k - 1)) / dyi;
                    out.pressure[2]   -= -grid->GetP(i, j, k - 1) * invRho * Az;
                    out.normalVisc[2] -= mi * dwdz * Az;
                    out.shearGrad[0]  -= mi * dudz * Az;  out.shearTrans[0] -= mi * dwdx * Az;
                    out.shearGrad[1]  -= mi * dvdz * Az;  out.shearTrans[1] -= mi * dwdy * Az;
                }
            }
}

int ResampleResolutionFor(const std::vector<double>& faces, int perCell, int cap)
{
    if (faces.size() < 2 || perCell < 1 || cap < 1) return cap > 0 ? cap : 1;
    const double span = faces.back() - faces.front();
    if (!(span > 0.0)) return cap;

    double smallest = span;
    for (size_t j = 1; j < faces.size(); j++)
    {
        const double h = faces[j] - faces[j - 1];
        if (h > 0.0 && h < smallest) smallest = h;
    }
    if (!(smallest > 0.0)) return cap;

    // Floor first so a coarse grid still gets a usable image, then cap -- in that order,
    // or a cap below the floor is silently ignored.
    double want = std::ceil(span / smallest) * (double)perCell;
    if (want < 64.0) want = 64.0;
    if (want > (double)cap) want = (double)cap;
    return (int)want;
}

void ResampleFieldToPhysical(const std::vector<double>& src, int srcRows, int srcCols,
                             const std::vector<double>& xf, const std::vector<double>& yf,
                             std::vector<double>& dst, int dstRows, int dstCols)
{
    dst.clear();
    if (srcRows <= 0 || srcCols <= 0 || dstRows <= 0 || dstCols <= 0) return;
    if (xf.size() < 2 || yf.size() < 2) return;
    if ((int)src.size() < srcRows * srcCols) return;

    const int nx = (int)xf.size() - 1, ny = (int)yf.size() - 1;
    const double xa = xf.front(), xb = xf.back();
    const double ya = yf.front(), yb = yf.back();
    if (!(xb > xa) || !(yb > ya)) return;

    dst.assign((size_t)dstRows * dstCols, 0.0);
    for (int v = 0; v < dstRows; v++)
    {
        // Row 0 is the TOP of the image, so it maps to the HIGHEST y.
        const double py = yb - (v + 0.5) * (yb - ya) / (double)dstRows;
        const int ci = CellContaining(yf, py, ny);
        int srcRow = (srcRows - 1) - ci;
        if (srcRow < 0) srcRow = 0;
        if (srcRow > srcRows - 1) srcRow = srcRows - 1;
        for (int h = 0; h < dstCols; h++)
        {
            const double px = xa + (h + 0.5) * (xb - xa) / (double)dstCols;
            int srcCol = CellContaining(xf, px, nx);
            if (srcCol > srcCols - 1) srcCol = srcCols - 1;
            dst[(size_t)v * dstCols + h] = src[(size_t)srcRow * srcCols + srcCol];
        }
    }
}
