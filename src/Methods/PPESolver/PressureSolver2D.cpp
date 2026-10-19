// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "PressureSolver2D.h"
#include "../../headers/Core/Scene.h"

int PressureSolver2D::Nx = 0;
int PressureSolver2D::Ny = 0;
std::vector<int> PressureSolver2D::IDP;
PressureSystem PressureSolver2D::SYSTEM;

// An inflow face's p= fixes the level of a singular system: the shift that makes the interior pressure,
// extrapolated to the ring centres, match it on average. Serial (scene expressions).
static void AccumulateLevel2D(MAC *g, const VectorXd &p, int ri, int rj, int i1, int j1, int i2, int j2,
                              double sR, double s1, double s2, double &sum, int &n)
{
    if (g->GetSolid(ri, rj) != INFLOW_CELL || g->GetSolid(i1, j1) != FLUID_CELL || g->GetSolid(i2, j2) != FLUID_CELL)
        return;
    const int id1 = PressureSolver2D::GetIDP(i1, j1), id2 = PressureSolver2D::GetIDP(i2, j2);
    if (id1 < 0 || id2 < 0)
        return;
    double level = 0.0;
    if (!SCENE.PressureLevelAt(g->CentrePos(MAC::X, rj), g->CentrePos(MAC::Y, ri), 0.0, SIMULATION.time, level))
        return;
    const double p1 = p(id1), p2 = p(id2);
    sum += level - (p1 + (p1 - p2) * (sR - s1) / (s1 - s2));
    n++;
}

// The one scaling: uniform, the stencil times 1/dh^2 (the matrix is L); stretched, face coefficients
// A_f/d_f unscaled and the right-hand side times the volume (V L, symmetric).
static inline double MatrixScale(const MAC *g)
{
    return g->nonUniform ? 1.0 : (1.0 / (g->dh * g->dh));
}

static inline double CellVolume2D(const MAC *g, int i, int j)
{
    // Multiplying by exactly 1.0 leaves the uniform right-hand side bit-identical.
    return g->nonUniform ? g->CellVolume(i, j) : 1.0;
}

// A pure-Neumann system fixes p only up to a constant (A 1 = 0): its right-hand side needs the mean removed, and
// the solver is free to drift along the constant.
static inline bool NeedsCompatibility(const PressureSystem &s)
{
    return PRESSURE_SETTINGS.needsCompatibilityCondition && s.singular;
}

static inline bool IsDirichlet(int t) { return t == EMPTY_CELL || t == PRESSURE_CELL; }

// Neighbour q of fluid cell (i, j), in N, S, E, W order: the factor on its face coefficient when it is a Dirichlet
// ring cell (DirichletRingFactor), 1 otherwise.
static inline double RingFactor2D(const MAC *g, int i, int j, int q)
{
    const int ni = i + (q == 0) - (q == 1), nj = j + (q == 2) - (q == 3);
    if (!IsDirichlet(g->GetSolid(ni, nj)) || (ni != 0 && ni != g->Ny - 1 && nj != 0 && nj != g->Nx - 1)) return 1.0;
    return q < 2 ? DirichletRingFactor(g, MAC::Y, i, q == 0 ? 1 : -1) : DirichletRingFactor(g, MAC::X, j, q == 2 ? 1 : -1);
}

void PressureSolver2D::InitializePressureSolver(MAC *grid, PressureScheme scheme)
{
    DestroyPressureBackend(SYSTEM, false);
    SYSTEM = PressureSystem();
    SYSTEM.scheme = scheme;
    SYSTEM.eigen = ReadEigenSettings(PressureConfigText(), 2);
#ifdef NVIDIA_GPU
    SYSTEM.backend = GPU_ACCELERATION ? PRESSURE_BACKEND_AMGX : PRESSURE_BACKEND_CPU;
#endif
    CreatePressureBackend(SYSTEM);
    RegisterSimulationStep(scheme == PRESSURE_INCREMENTAL ? SolveIncremental : SolveChorin, "pressure", 4, LastIterations);
    RegisterSimulationStep(Project, "projection", 6);
    AssembleSystem(grid);
    std::cout << "Pressure Solver was initialized!\n";
}

void PressureSolver2D::DestroyPressureSolver()
{
    DestroyPressureBackend(SYSTEM, true);
    ReleasePressureSolver();
}

void PressureSolver2D::ReleasePressureSolver()
{
    DestroyPressureBackend(SYSTEM, false);
    SYSTEM = PressureSystem();
    std::vector<int>().swap(IDP);
    AMG_CACHE = AmgPressureCache();
}

void PressureSolver2D::EnsureSystem(MAC *grid)
{
    if (PressureLayoutChanged(SYSTEM, grid))
        AssembleSystem(grid);
}

// Interior FLUID cells are the unknowns. A fluid neighbour couples; EMPTY/PRESSURE are Dirichlet (value to
// the right-hand side); solid/inflow are Neumann (dropped). Diagonal: fluid faces, then Dirichlet, N S E W.
void PressureSolver2D::AssembleSystem(MAC *grid)
{
    Nx = grid->Nx;
    Ny = grid->Ny;
    IDP.assign((size_t)Nx * Ny, -1);
    int n = 0;
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
            if (grid->GetSolid(i, j) == FLUID_CELL)
                IDP[i * Nx + j] = n++;

    const double scale = MatrixScale(grid);
    std::vector<Eigen::Triplet<double>> t;
    t.reserve((size_t)5 * n);
    bool dirichlet = false;
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const int row = GetIDP(i, j);
            if (row < 0) continue;
            // Finite-volume face coefficients, A_f / d_f: exactly 1.0 on a uniform grid.
            const int ni[4] = {i + 1, i - 1, i, i};
            const int nj[4] = {j, j, j + 1, j - 1};
            const double c[4] = {grid->Width(MAC::X, j) / grid->FaceGap(MAC::Y, i),
                                 grid->Width(MAC::X, j) / grid->FaceGap(MAC::Y, i - 1),
                                 grid->Width(MAC::Y, i) / grid->FaceGap(MAC::X, j),
                                 grid->Width(MAC::Y, i) / grid->FaceGap(MAC::X, j - 1)};
            double diag = 0.0;
            for (int q = 0; q < 4; q++)
                if (grid->GetSolid(ni[q], nj[q]) == FLUID_CELL)
                {
                    diag -= c[q];
                    const int col = GetIDP(ni[q], nj[q]);
                    if (col >= 0) t.push_back(Eigen::Triplet<double>(row, col, scale * c[q]));
                    else dirichlet = true;
                }
            for (int q = 0; q < 4; q++)
                if (IsDirichlet(grid->GetSolid(ni[q], nj[q])))
                {
                    diag -= c[q] * RingFactor2D(grid, i, j, q);
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

// The constant level of p, in ONE place. Only a singular system has a free constant: its mean is pinned to zero,
// then, if an inflow face carries p=, shifted to that level. A system with a Dirichlet cell is left alone.
void PressureSolver2D::LevelPressure(MAC *grid, VectorXd &p)
{
    if (!NeedsCompatibility(SYSTEM) || p.size() == 0)
        return;
    p.array() -= p.mean();
    if (!SCENE.loaded)
        return;
    const int nx = grid->Nx, ny = grid->Ny;
    double sum = 0.0;
    int n = 0;
    for (int i = 1; i < ny - 1; i++)
    {
        AccumulateLevel2D(grid, p, i, 0, i, 1, i, 2, grid->CentrePos(MAC::X, 0), grid->CentrePos(MAC::X, 1), grid->CentrePos(MAC::X, 2), sum, n);
        AccumulateLevel2D(grid, p, i, nx - 1, i, nx - 2, i, nx - 3, grid->CentrePos(MAC::X, nx - 1), grid->CentrePos(MAC::X, nx - 2), grid->CentrePos(MAC::X, nx - 3), sum, n);
    }
    for (int j = 1; j < nx - 1; j++)
    {
        AccumulateLevel2D(grid, p, 0, j, 1, j, 2, j, grid->CentrePos(MAC::Y, 0), grid->CentrePos(MAC::Y, 1), grid->CentrePos(MAC::Y, 2), sum, n);
        AccumulateLevel2D(grid, p, ny - 1, j, ny - 2, j, ny - 3, j, grid->CentrePos(MAC::Y, ny - 1), grid->CentrePos(MAC::Y, ny - 2), grid->CentrePos(MAC::Y, ny - 3), sum, n);
    }
    if (n > 0)
        p.array() += sum / n;
}

// Warm start: the pressure between consecutive steps is highly coherent, and in an Eulerian run the fluid region
// does not move at all.
bool PressureSolver2D::WarmStartEnabled(const MAC *grid)
{
    // FLIP is excluded: its fluid region moves, so newly wetted cells have no previous value. A rectilinear grid is
    // excluded too, a workaround rather than a limitation.
    if (grid != nullptr && grid->nonUniform)
        return false;
    return PRESSURE_SETTINGS.warmStart && MethodFixedFluidRegion();
}

bool PressureSolver2D::BuildWarmStartGuess(MAC *gridAnt, VectorXd &x0)
{
    if (gridAnt == nullptr || !WarmStartEnabled(gridAnt))
        return false;
    const int n = (int)x0.size();
    if (n <= 0)
        return false;

    x0.setZero();
    for (int i = 0; i < Ny; i++)
        for (int j = 0; j < Nx; j++)
        {
            const int id = GetIDP(i, j);
            if (id >= 0 && id < n)
                x0(id) = gridAnt->GetP(i, j);
        }
    // A field still at zero (the first step) is no guess at all: solve cold.
    if (x0.squaredNorm() == 0.0)
        return false;
    // Only where the system is singular.
    if (NeedsCompatibility(SYSTEM))
        x0.array() -= x0.mean();
    return true;
}

// The known pressures on a cell's Dirichlet neighbours, weighted as in the matrix. Under Incremental the
// known value is p_b(t^{n+1}) - p_b(t^n), `old` holding the latter (null under Chorin).
static inline double KnownPressure2D(MAC *grid, MAC *old, int i, int j)
{
    return old ? grid->GetP(i, j) - old->GetP(i, j) : grid->GetP(i, j);
}

double PressureSolver2D::DirichletSum(MAC *grid, MAC *old, int i, int j)
{
    double s = 0.0;
    if (grid->GetSolid(i + 1, j) == PRESSURE_CELL)
        s += (grid->Width(MAC::X, j) / grid->FaceGap(MAC::Y, i)) * RingFactor2D(grid, i, j, 0) * KnownPressure2D(grid, old, i + 1, j);
    if (grid->GetSolid(i - 1, j) == PRESSURE_CELL)
        s += (grid->Width(MAC::X, j) / grid->FaceGap(MAC::Y, i - 1)) * RingFactor2D(grid, i, j, 1) * KnownPressure2D(grid, old, i - 1, j);
    if (grid->GetSolid(i, j + 1) == PRESSURE_CELL)
        s += (grid->Width(MAC::Y, i) / grid->FaceGap(MAC::X, j)) * RingFactor2D(grid, i, j, 2) * KnownPressure2D(grid, old, i, j + 1);
    if (grid->GetSolid(i, j - 1) == PRESSURE_CELL)
        s += (grid->Width(MAC::Y, i) / grid->FaceGap(MAC::X, j - 1)) * RingFactor2D(grid, i, j, 3) * KnownPressure2D(grid, old, i, j - 1);
    return s;
}

double PressureSolver2D::IncrementReferenceNorm(MAC *gridAnt, const VectorXd &b)
{
    const int n = (int)b.size();
    if (gridAnt == nullptr || SYSTEM.A.rows() != n)
        return 0.0;
    VectorXd pn = VectorXd::Zero(n);
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const int id = GetIDP(i, j);
            if (id >= 0) pn(id) = gridAnt->GetP(i, j);
        }
    return (b + SYSTEM.A * pn).norm();
}

// The skeleton both schemes share: prescribed pressure, matrix check, right-hand side, compatibility, solve,
// constant level, scatter, border.
void PressureSolver2D::SolveScheme(MAC *gridAnt, MAC *grid, bool incremental)
{
    grid->EnforcePressureBC(SIMULATION.time);
    EnsureSystem(grid);
    const int n = SystemSize();
    const double scale = MatrixScale(grid);
    const double rhoDt = SIMULATION.RHO / SIMULATION.dt;

    MAC *old = incremental ? gridAnt : nullptr;
    VectorXd b = VectorXd::Zero(n);
    #pragma omp parallel for
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const int id = GetIDP(i, j);
            if (id >= 0)
                b(id) = rhoDt * grid->GetDivergencyAt(i, j) * CellVolume2D(grid, i, j) - scale * DirichletSum(grid, old, i, j);
        }

    if (NeedsCompatibility(SYSTEM) && n > 0)
    {
        double mean = 0.0;
        for (int k = 0; k < n; k++) mean += b(k);
        mean /= n;
        for (int k = 0; k < n; k++) b(k) -= mean;
        SubtractMean(b.data(), n);   // the second pass takes the leftover from round-off to ~0
    }

    VectorXd x, x0(n);
    const bool warm = !incremental && BuildWarmStartGuess(gridAnt, x0);
    const double ref = incremental ? IncrementReferenceNorm(gridAnt, b) : 0.0;
    PRESSURE_TELEMETRY.lastIterations = SolvePressureSystem(SYSTEM, b, x, warm ? &x0 : nullptr, ref);
    // The increment's constant is irrelevant (only its gradient is used); the level is set on p^n + phi instead.
    if (!incremental)
        LevelPressure(grid, x);

    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const int id = GetIDP(i, j);
            if (id >= 0) grid->SetP(i, j, x(id));
        }
    grid->EnforcePressureBC(SIMULATION.time);
    // The projection differences phi, so a prescribed-pressure ring cell carries phi's value there until the
    // accumulation's EnforcePressureBC puts p_b back.
    if (incremental && gridAnt != nullptr)
        for (int i = 0; i < Ny; i++)
            for (int j = 0; j < Nx; j++)
                if (grid->GetSolid(i, j) == PRESSURE_CELL)
                    grid->SetP(i, j, grid->GetP(i, j) - gridAnt->GetP(i, j));
}

void PressureSolver2D::SolveChorin(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    SolveScheme(gridAnt, grid, false);
}

void PressureSolver2D::SolveIncremental(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    SolveScheme(gridAnt, grid, true);
}

void PressureSolver2D::SolvePressure(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    SolveScheme(gridAnt, grid, SYSTEM.scheme == PRESSURE_INCREMENTAL);
}

// A = L on a uniform grid and V L on a stretched one (see MatrixScale), so L x = f is A x = V f.
bool PressureSolver2D::SolveLaplacian(MAC *grid, const VectorXd &f, VectorXd &x)
{
    EnsureSystem(grid);
    const int n = SystemSize();
    if (n == 0 || f.size() != n)
        return false;
    VectorXd b(n);
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const int id = GetIDP(i, j);
            if (id >= 0) b(id) = f(id) * CellVolume2D(grid, i, j);
        }
    SolvePressureSystem(SYSTEM, b, x, nullptr, 0.0);
    return true;
}

// Incremental: the solve returned phi = p^{n+1} - p^n; p = p^n + phi, levelled like any pressure.
void PressureSolver2D::AccumulatePressureIncrement(MAC *gridAnt, MAC *grid)
{
    VectorXd p(SystemSize());
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const int id = GetIDP(i, j);
            if (id >= 0) p(id) = gridAnt->GetP(i, j) + grid->GetP(i, j);
        }
    LevelPressure(grid, p);
    for (int i = 1; i < Ny - 1; i++)
        for (int j = 1; j < Nx - 1; j++)
        {
            const int id = GetIDP(i, j);
            if (id >= 0) grid->SetP(i, j, p(id));
        }
    grid->EnforcePressureBC(SIMULATION.time);
}

// The projection step both methods run: the velocity corrected by the pressure the solve left on the grid, and
// under Incremental that pressure then made p^{n+1}.
void PressureSolver2D::Project(MAC *gridAnt, MAC *grid, ParticleSet *particles, double time)
{
    ProjectPressure(grid);
    if (SYSTEM.scheme == PRESSURE_INCREMENTAL)
        AccumulatePressureIncrement(gridAnt, grid);
}

// The gradient across the face at index n of axis a between cells n-1 and n: a Dirichlet ring cell on either side
// puts its value on the face, half a cell from the fluid centre -- the same gap the matrix used.
static inline double ProjectionGradient2D(MAC *g, MAC::Axis a, int i, int j)
{
    const double grad = g->GetGradP(a, i, j);
    const int n = a == MAC::X ? j : i, N = a == MAC::X ? g->Nx : g->Ny;
    if (n == 1 && IsDirichlet(a == MAC::X ? g->GetSolid(i, 0) : g->GetSolid(0, j)))
        return grad * DirichletRingFactor(g, a, 1, -1);
    if (n == N - 1 && IsDirichlet(a == MAC::X ? g->GetSolid(i, N - 1) : g->GetSolid(N - 1, j)))
        return grad * DirichletRingFactor(g, a, N - 2, +1);
    return grad;
}

// u^{n+1} = u* - (dt/rho) grad p on every face that is not a wall.
void PressureSolver2D::ProjectPressure(MAC *grid)
{
    const int nx = grid->Nx, ny = grid->Ny;
    const double dt = SIMULATION.dt;
    for (int i = 1; i < ny - 1; i++)
        for (int j = 1; j < nx; j++)
            if (grid->FaceGate(MAC::U, i, j) != GateState::WALL)
                grid->SetU(i, j, grid->GetU(i, j) - (dt / SIMULATION.RHO) * ProjectionGradient2D(grid, MAC::X, i, j));

    for (int i = 1; i < ny; i++)
        for (int j = 1; j < nx - 1; j++)
            if (grid->FaceGate(MAC::V, i, j) != GateState::WALL)
                grid->SetV(i, j, grid->GetV(i, j) - (dt / SIMULATION.RHO) * ProjectionGradient2D(grid, MAC::Y, i, j));
}
