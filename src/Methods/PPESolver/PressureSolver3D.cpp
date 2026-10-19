// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "PressureSolver3D.h"
#include "../../headers/Core/Scene.h"

int PressureSolver3D::Nx = 0;
int PressureSolver3D::Ny = 0;
int PressureSolver3D::Nz = 0;
std::vector<int> PressureSolver3D::IDP;
PressureSystem PressureSolver3D::SYSTEM;

// An inflow face's p= fixes the level of a singular (pure-Neumann) system; see PressureSolver2D.cpp.
static void AccumulateLevel3D(MAC *g, const VectorXd &p, int ri, int rj, int rk, int i1, int j1, int k1,
                              int i2, int j2, int k2, double sR, double s1, double s2, double &sum, int &n)
{
    if (g->GetSolid(ri, rj, rk) != INFLOW_CELL || g->GetSolid(i1, j1, k1) != FLUID_CELL ||
        g->GetSolid(i2, j2, k2) != FLUID_CELL)
        return;
    const int id1 = PressureSolver3D::GetIDP(i1, j1, k1), id2 = PressureSolver3D::GetIDP(i2, j2, k2);
    if (id1 < 0 || id2 < 0)
        return;
    double level = 0.0;
    if (!SCENE.PressureLevelAt(g->CentrePos(MAC::X, rj), g->CentrePos(MAC::Y, ri), g->CentrePos(MAC::Z, rk), SIMULATION.time, level))
        return;
    const double p1 = p(id1), p2 = p(id2);
    sum += level - (p1 + (p1 - p2) * (sR - s1) / (s1 - s2));
    n++;
}

// The one scaling, as in 2D: uniform, the stencil's +-1 times 1/dh^2; stretched, face coefficients A_f/d_f and a
// volume-weighted right-hand side. The matrix is L on a uniform grid and V L on a stretched one.
static inline double MatrixScale3D(const MAC *g)
{
    return g->nonUniform ? 1.0 : (1.0 / (g->dh * g->dh));
}

static inline double CellVolume3D(const MAC *g, int i, int j, int k)
{
    // Multiplying by exactly 1.0 leaves the uniform right-hand side bit-identical.
    return g->nonUniform ? g->CellVolume(i, j, k) : 1.0;
}

// Face coefficients A_f / d_f toward (i+1), (i-1), (j+1), (j-1), (k+1), (k-1). Uniform returns 1.0 (the
// true value is dh, the area being dh^2); MatrixScale3D supplies the 1/dh^2.
static inline void FaceCoefs3D(const MAC *g, int i, int j, int k, double c[6])
{
    if (!g->nonUniform)
    {
        for (int q = 0; q < 6; q++) c[q] = 1.0;
        return;
    }
    const double ax = g->FaceArea(MAC::X, i, j, k);
    const double ay = g->FaceArea(MAC::Y, i, j, k);
    const double az = g->FaceArea(MAC::Z, i, j, k);
    c[0] = ay / g->FaceGap(MAC::Y, i);
    c[1] = ay / g->FaceGap(MAC::Y, i - 1);
    c[2] = ax / g->FaceGap(MAC::X, j);
    c[3] = ax / g->FaceGap(MAC::X, j - 1);
    c[4] = az / g->FaceGap(MAC::Z, k);
    c[5] = az / g->FaceGap(MAC::Z, k - 1);
}

//masks
static const int DI[6] = {1, -1, 0, 0, 0, 0};
static const int DJ[6] = {0, 0, 1, -1, 0, 0};
static const int DK[6] = {0, 0, 0, 0, 1, -1};

//checls if we need to ensure the compatibility condition based on the user settings AND the own system check
static inline bool NeedsCompatibility(const PressureSystem &s)
{
    return PRESSURE_SETTINGS.needsCompatibilityCondition && s.singular;
}

static inline bool IsDirichlet(int t) { return t == EMPTY_CELL || t == PRESSURE_CELL; }

// Neighbour q of fluid cell (i, j, k), in (i+1, i-1, j+1, j-1, k+1, k-1) order: the factor on its face coefficient
// when it is a Dirichlet ring cell (DirichletRingFactor), 1 otherwise.
static inline double RingFactor3D(const MAC *g, int i, int j, int k, int q)
{
    const int ni = i + (q == 0) - (q == 1), nj = j + (q == 2) - (q == 3), nk = k + (q == 4) - (q == 5);
    if (!IsDirichlet(g->GetSolid(ni, nj, nk))) return 1.0;
    if (ni != 0 && ni != g->Ny - 1 && nj != 0 && nj != g->Nx - 1 && nk != 0 && nk != g->Nz - 1) return 1.0;
    const int s = (q % 2 == 0) ? 1 : -1;
    return q < 2 ? DirichletRingFactor(g, MAC::Y, i, s) : (q < 4 ? DirichletRingFactor(g, MAC::X, j, s) : DirichletRingFactor(g, MAC::Z, k, s));
}

void PressureSolver3D::InitializePressureSolver(MAC *grid, PressureScheme scheme)
{
    DestroyPressureBackend(SYSTEM, false);
    SYSTEM = PressureSystem();
    SYSTEM.scheme = scheme;
    SYSTEM.eigen = ReadEigenSettings(PressureConfigText(), 3);
#ifdef NVIDIA_GPU
    SYSTEM.backend = GPU_ACCELERATION ? PRESSURE_BACKEND_AMGX : PRESSURE_BACKEND_CPU;
#endif
    CreatePressureBackend(SYSTEM);
    RegisterSimulationStep(scheme == PRESSURE_INCREMENTAL ? SolveIncremental : SolveChorin, "pressure", 4, LastIterations);
    RegisterSimulationStep(Project, "projection", 6);
    AssembleSystem(grid);
}

void PressureSolver3D::DestroyPressureSolver()
{
    DestroyPressureBackend(SYSTEM, true);
    ReleasePressureSolver();
}

void PressureSolver3D::ReleasePressureSolver()
{
    DestroyPressureBackend(SYSTEM, false);
    SYSTEM = PressureSystem();
    std::vector<int>().swap(IDP);
    AMG_CACHE = AmgPressureCache();
}

//checks if the system needs reasembly, this saves us time in cases where it doesnt, such as a flip steady state
//eulerian ensures but it never rebuilds, as it never changes (unless we add moving solids in the future!)
void PressureSolver3D::EnsureSystem(MAC *grid)
{
    if (PressureLayoutChanged(SYSTEM, grid))
        AssembleSystem(grid);
}

// As in 2D: interior FLUID cells are the unknowns, numbered in (i, j, k) order; the diagonal sums the fluid faces,
// then the Dirichlet faces, in (i+1, i-1, j+1, j-1, k+1, k-1) order.
void PressureSolver3D::AssembleSystem(MAC *grid)
{
    Nx = grid->Nx;
    Ny = grid->Ny;
    Nz = grid->Nz;
    IDP.assign((size_t)Nx * Ny * Nz, -1);
    int n = 0;
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
                if (grid->GetSolid(i, j, k) == FLUID_CELL)
                    IDP[i * (Nx * Nz) + j * Nz + k] = n++;

    const double scale = MatrixScale3D(grid);
    std::vector<Eigen::Triplet<double>> t;
    t.reserve((size_t)7 * n);
    bool dirichlet = false;
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                const int row = GetIDP(i, j, k);
                if (row < 0) continue;
                double c[6];
                FaceCoefs3D(grid, i, j, k, c);
                double diag = 0.0;
                for (int q = 0; q < 6; q++)
                    if (grid->GetSolid(i + DI[q], j + DJ[q], k + DK[q]) == FLUID_CELL)
                    {
                        diag -= c[q];
                        const int col = GetIDP(i + DI[q], j + DJ[q], k + DK[q]);
                        if (col >= 0) t.push_back(Eigen::Triplet<double>(row, col, scale * c[q]));
                        else dirichlet = true;
                    }
                for (int q = 0; q < 6; q++)
                    if (IsDirichlet(grid->GetSolid(i + DI[q], j + DJ[q], k + DK[q])))
                    {
                        diag -= c[q] * RingFactor3D(grid, i, j, k, q);
                        dirichlet = true;
                    }
                if (diag != 0.0)
                    t.push_back(Eigen::Triplet<double>(row, row, scale * diag));
            }

    SYSTEM.singular = !dirichlet;
    CommitPressureMatrix(SYSTEM, grid, t, n);
    if (SYSTEM.singular && !PRESSURE_SETTINGS.needsCompatibilityCondition && MethodFixedFluidRegion())
        std::cerr << "Warning: the pressure system is singular (no outflow or prescribed-pressure face, so p is "
                     "fixed only up to a constant) and NEEDS_COMPATIBILITY_CONDITION is off. The right-hand side "
                     "is then inconsistent: the solve is undefined and can diverge. Turn it on." << std::endl;
}

// The constant level of p, in ONE place; see PressureSolver2D::LevelPressure.
void PressureSolver3D::LevelPressure(MAC *g, VectorXd &p)
{
    if (!NeedsCompatibility(SYSTEM) || p.size() == 0)
        return;
    p.array() -= p.mean();
    if (!SCENE.loaded)
        return;
    const int Nx = g->Nx, Ny = g->Ny, Nz = g->Nz;
    double sum = 0.0;
    int n = 0;
    for (int i = 1; i < Ny - 1; i++)
        for (int k = 1; k < Nz - 1; k++)
        {
            AccumulateLevel3D(g, p, i, 0, k, i, 1, k, i, 2, k, g->CentrePos(MAC::X, 0), g->CentrePos(MAC::X, 1), g->CentrePos(MAC::X, 2), sum, n);
            AccumulateLevel3D(g, p, i, Nx - 1, k, i, Nx - 2, k, i, Nx - 3, k, g->CentrePos(MAC::X, Nx - 1), g->CentrePos(MAC::X, Nx - 2), g->CentrePos(MAC::X, Nx - 3), sum, n);
        }
    for (int j = 1; j < Nx - 1; j++)
        for (int k = 1; k < Nz - 1; k++)
        {
            AccumulateLevel3D(g, p, 0, j, k, 1, j, k, 2, j, k, g->CentrePos(MAC::Y, 0), g->CentrePos(MAC::Y, 1), g->CentrePos(MAC::Y, 2), sum, n);
            AccumulateLevel3D(g, p, Ny - 1, j, k, Ny - 2, j, k, Ny - 3, j, k, g->CentrePos(MAC::Y, Ny - 1), g->CentrePos(MAC::Y, Ny - 2), g->CentrePos(MAC::Y, Ny - 3), sum, n);
        }
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            AccumulateLevel3D(g, p, i, j, 0, i, j, 1, i, j, 2, g->CentrePos(MAC::Z, 0), g->CentrePos(MAC::Z, 1), g->CentrePos(MAC::Z, 2), sum, n);
            AccumulateLevel3D(g, p, i, j, Nz - 1, i, j, Nz - 2, i, j, Nz - 3, g->CentrePos(MAC::Z, Nz - 1), g->CentrePos(MAC::Z, Nz - 2), g->CentrePos(MAC::Z, Nz - 3), sum, n);
        }
    if (n > 0)
        p.array() += sum / n;
}

// Warm start: see PressureSolver2D, but it is basically using the last pressure solution as the initial guess,
//CAN save time or make it worse, be sensible using it.
bool PressureSolver3D::WarmStartEnabled(const MAC *grid)
{
    if (grid != nullptr && grid->nonUniform)
        return false;
    return PRESSURE_SETTINGS.warmStart && MethodFixedFluidRegion();
}

bool PressureSolver3D::BuildWarmStartGuess(MAC *gridAnt, VectorXd &x0)
{
    if (gridAnt == nullptr || !WarmStartEnabled(gridAnt))
        return false;
    const int n = (int)x0.size();
    if (n <= 0)
        return false;

    x0.setZero();
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
            for (int k = 0; k < Nz; k++)
            {
                const int id = GetIDP(i, j, k);
                if (id >= 0 && id < n)
                    x0(id) = gridAnt->GetP(i, j, k);
            }
    if (x0.squaredNorm() == 0.0)
        return false;
    if (NeedsCompatibility(SYSTEM))
        x0.array() -= x0.mean();
    return true;
}

// The known pressures on a cell's Dirichlet neighbours, weighted as in the matrix. Under Incremental the
// known value is p_b(t^{n+1}) - p_b(t^n), `old` holding the latter (null under Chorin).
double PressureSolver3D::DirichletSum(MAC *grid, MAC *old, int i, int j, int k)
{
    double c[6];
    FaceCoefs3D(grid, i, j, k, c);
    double s = 0.0;
    for (int q = 0; q < 6; q++)
        if (grid->GetSolid(i + DI[q], j + DJ[q], k + DK[q]) == PRESSURE_CELL)
        {
            const int a = i + DI[q], b = j + DJ[q], d = k + DK[q];
            s += c[q] * RingFactor3D(grid, i, j, k, q) * (old ? grid->GetP(a, b, d) - old->GetP(a, b, d) : grid->GetP(a, b, d));
        }
    return s;
}

double PressureSolver3D::IncrementReferenceNorm(MAC *gridAnt, const VectorXd &b)
{
    const int n = (int)b.size();
    if (gridAnt == nullptr || SYSTEM.A.rows() != n)
        return 0.0;
    VectorXd pn = VectorXd::Zero(n);
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                const int id = GetIDP(i, j, k);
                if (id >= 0) pn(id) = gridAnt->GetP(i, j, k);
            }
    return (b + SYSTEM.A * pn).norm();
}

// The skeleton both schemes share; see PressureSolver2D::SolveScheme.
void PressureSolver3D::SolveScheme(MAC *gridAnt, MAC *grid, bool incremental)
{
    grid->EnforcePressureBC(SIMULATION.time);
    EnsureSystem(grid);
    const int n = SystemSize();
    const double scale = MatrixScale3D(grid);
    const double rhoDt = SIMULATION.RHO / SIMULATION.dt;

    MAC *old = incremental ? gridAnt : nullptr;
    VectorXd b = VectorXd::Zero(n);
    #pragma omp parallel for
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                const int id = GetIDP(i, j, k);
                if (id >= 0)
                    b(id) = rhoDt * grid->GetDivergencyAt(i, j, k) * CellVolume3D(grid, i, j, k)
                            - scale * DirichletSum(grid, old, i, j, k);
            }

    if (NeedsCompatibility(SYSTEM) && n > 0)
    {
        double mean = 0.0;
        for (int q = 0; q < n; q++) mean += b(q);
        mean /= n;
        for (int q = 0; q < n; q++) b(q) -= mean;
        SubtractMean(b.data(), n);   // the second pass takes the leftover from round-off to ~0
    }

    // On AMGX a closed box never warm starts: learning its residual floor costs more than the guess saves.
    // 2D measured no stall and has no such gate.
    VectorXd x, x0(n);
    const bool warm = !incremental &&
                      !(SYSTEM.backend == PRESSURE_BACKEND_AMGX && PRESSURE_SETTINGS.needsCompatibilityCondition) &&
                      BuildWarmStartGuess(gridAnt, x0);
    const double ref = incremental ? IncrementReferenceNorm(gridAnt, b) : 0.0;
    PRESSURE_TELEMETRY.lastIterations = SolvePressureSystem(SYSTEM, b, x, warm ? &x0 : nullptr, ref);
    // The increment's constant is irrelevant (only its gradient is used); the level is set on p^n + phi instead.
    if (!incremental)
        LevelPressure(grid, x);

    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                const int id = GetIDP(i, j, k);
                if (id >= 0) grid->SetP(i, j, k, x(id));
            }
    grid->EnforcePressureBC(SIMULATION.time);
    // The projection differences phi, so a prescribed-pressure ring cell carries phi's value there until the
    // accumulation's EnforcePressureBC puts p_b back.
    if (incremental && gridAnt != nullptr)
        for (int i = 0; i < Ny; i++)
            for (int j = 0; j < Nx; j++)
                for (int k = 0; k < Nz; k++)
                    if (grid->GetSolid(i, j, k) == PRESSURE_CELL)
                        grid->SetP(i, j, k, grid->GetP(i, j, k) - gridAnt->GetP(i, j, k));
}

void PressureSolver3D::SolveChorin(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    SolveScheme(gridAnt, grid, false);
}

void PressureSolver3D::SolveIncremental(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    SolveScheme(gridAnt, grid, true);
}

void PressureSolver3D::SolvePressure(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    SolveScheme(gridAnt, grid, SYSTEM.scheme == PRESSURE_INCREMENTAL);
}

// A = L on a uniform grid and V L on a stretched one, so L x = f is A x = V f.
bool PressureSolver3D::SolveLaplacian(MAC *grid, const VectorXd &f, VectorXd &x)
{
    EnsureSystem(grid);
    const int n = SystemSize();
    if (n == 0 || f.size() != n)
        return false;
    VectorXd b(n);
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                const int id = GetIDP(i, j, k);
                if (id >= 0) b(id) = f(id) * CellVolume3D(grid, i, j, k);
            }
    SolvePressureSystem(SYSTEM, b, x, nullptr, 0.0);
    return true;
}

// Incremental: the solve returned phi = p^{n+1} - p^n; p = p^n + phi, levelled like any pressure.
void PressureSolver3D::AccumulatePressureIncrement(MAC *gridAnt, MAC *grid)
{
    VectorXd p(SystemSize());
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                const int id = GetIDP(i, j, k);
                if (id >= 0) p(id) = gridAnt->GetP(i, j, k) + grid->GetP(i, j, k);
            }
    LevelPressure(grid, p);
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            for (int k = 1; k < Nz - 1; k++)
            {
                const int id = GetIDP(i, j, k);
                if (id >= 0) grid->SetP(i, j, k, p(id));
            }
    grid->EnforcePressureBC(SIMULATION.time);
}

// The projection step both methods run; see PressureSolver2D::Project.
void PressureSolver3D::Project(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    ProjectPressure(grid);
    if (SYSTEM.scheme == PRESSURE_INCREMENTAL)
        AccumulatePressureIncrement(gridAnt, grid);
}

// The gradient across the axis-a face of cell (i, j, k): a Dirichlet ring cell on either side puts its value on the
// face, half a cell from the fluid centre -- the same gap the matrix used.
static inline double ProjectionGradient3D(MAC *g, MAC::Axis a, int i, int j, int k)
{
    const double grad = g->GetGradP(a, i, j, k);
    const int n = a == MAC::X ? j : (a == MAC::Y ? i : k), N = a == MAC::X ? g->Nx : (a == MAC::Y ? g->Ny : g->Nz);
    if (n != 1 && n != N - 1) return grad;
    const int r = n == 1 ? 0 : N - 1;
    const int t = a == MAC::X ? g->GetSolid(i, r, k) : (a == MAC::Y ? g->GetSolid(r, j, k) : g->GetSolid(i, j, r));
    if (!IsDirichlet(t)) return grad;
    return n == 1 ? grad * DirichletRingFactor(g, a, 1, -1) : grad * DirichletRingFactor(g, a, N - 2, +1);
}

// u^{n+1} = u* - (dt/rho) grad p on every face that is not a wall.
void PressureSolver3D::ProjectPressure(MAC *grid)
{
    const int nx = grid->Nx, ny = grid->Ny, nz = grid->Nz;
    const double dt = SIMULATION.dt;
    for (int i = 1; i < ny - 1; i++)
        for (int j = 1; j < nx; j++)
            for (int k = 1; k < nz - 1; k++)
                if (grid->FaceGate(MAC::U, i, j, k) != GateState::WALL)
                    grid->SetU(i, j, k, grid->GetU(i, j, k) - (dt / SIMULATION.RHO) * ProjectionGradient3D(grid, MAC::X, i, j, k));

    for (int i = 1; i < ny; i++)
        for (int j = 1; j < nx - 1; j++)
            for (int k = 1; k < nz - 1; k++)
                if (grid->FaceGate(MAC::V, i, j, k) != GateState::WALL)
                    grid->SetV(i, j, k, grid->GetV(i, j, k) - (dt / SIMULATION.RHO) * ProjectionGradient3D(grid, MAC::Y, i, j, k));

    for (int i = 1; i < ny - 1; i++)
        for (int j = 1; j < nx - 1; j++)
            for (int k = 1; k < nz; k++)
                if (grid->FaceGate(MAC::W, i, j, k) != GateState::WALL)
                    grid->SetW(i, j, k, grid->GetW(i, j, k) - (dt / SIMULATION.RHO) * ProjectionGradient3D(grid, MAC::Z, i, j, k));
}
