// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef PRESSURE_SETTINGS_H
#define PRESSURE_SETTINGS_H

// What PressureSolver2D/3D share: PRESSURE_SETTINGS and the PRESSURE block (AMGX reads every key but
// eigen_*, the CPU path the eigen_* keys), the assembled system and the backends.

#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <Eigen/Sparse>
#ifdef NVIDIA_GPU
#include <amgx_c.h>
#endif
#include "../../headers/Utils.h"
#include <algorithm>
#include <memory>
#include <tuple>
#include <Eigen/IterativeLinearSolvers>

// AMGCL (include/amgcl, header only) for the CPU pressure solve. its Boost-free parameter structs.
// In case someone reads this, AMGCL is a very good library and I really like it!
#define AMGCL_NO_BOOST
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>


// The file a config with no PRESSURE block falls back to.
//we WILL remove this in the future, this is legacy behaviour!
inline const char *LEGACY_PRESSURE_CONFIG = "solver_pressure_config.txt";

enum EigenSolverKind { EIGEN_BICGSTAB = 0, EIGEN_CG = 1, EIGEN_AMG = 2 };   // AMG: AMGCL, include/amgcl
enum EigenPreconditionerKind { EIGEN_PRECON_DIAGONAL = 0, EIGEN_PRECON_IDENTITY = 1, EIGEN_PRECON_ILUT = 2, EIGEN_PRECON_IC = 3 };

struct EigenSolverSettings
{
    int solver = EIGEN_BICGSTAB;
    int preconditioner = EIGEN_PRECON_DIAGONAL;
    double tolerance = 1e-9;
    int maxIters = 2000;
};

// What the last solve reported, kept apart from the settings (as SPH_TIMINGS is from SPH_SETTINGS).
struct PressureTelemetry
{
    // Iterations the last solve took -- the number that shows whether a warm start is doing anything.
    int lastIterations = 0;
};
inline PressureTelemetry PRESSURE_TELEMETRY;

struct PressureSettings
{
    // Start each solve from the previous step's field instead of from zero.
    bool warmStart = false;
    // The projection subtracts the mean from the right-hand side, which is only legitimate for a singular
    // pure-Neumann system.
    bool needsCompatibilityCondition = false;
    // The config's PRESSURE block: the AMGX settings and the eigen_* keys. Empty is filled from
    // solver_pressure_config.txt on first use, by PressureConfigText(). Not a table field: the reader collects it.
    std::string text;

    void ReadFrom(const ConfigValues &v);
    void WriteTo(std::ostream &f) const;
};
inline PressureSettings PRESSURE_SETTINGS;

// The two plain settings as a table, so the setup screen draws them like a method's. They live in the PRESSURE
// block beside the AMGX and eigen_* keys (a config older than that carries them at the top level, and is still read).
inline const SettingField PRESSURE_FIELDS[] = {
    BoolSetting("NEEDS_COMPATIBILITY_CONDITION", "Needs compatibility condition",
                "Removes the mean from the pressure source. Needed when no face fixes the pressure",
                offsetof(PressureSettings, needsCompatibilityCondition), false),
    BoolSetting("PRESSURE_WARM_START", "Warm Start of pressure solve",
                "Start each solve from the previous pressure",
                offsetof(PressureSettings, warmStart), false),
};
inline constexpr int PRESSURE_FIELD_COUNT = (int)(sizeof(PRESSURE_FIELDS) / sizeof(PRESSURE_FIELDS[0]));

// Resets the table fields (a missing key is the default) and the last solve's report; `text` is left to the reader.
inline void PressureSettings::ReadFrom(const ConfigValues &v)
{
    ReadSettings(PRESSURE_FIELDS, PRESSURE_FIELD_COUNT, this, v);
    PRESSURE_TELEMETRY = PressureTelemetry();
}

inline void PressureSettings::WriteTo(std::ostream &f) const
{
    WriteSettings(PRESSURE_FIELDS, PRESSURE_FIELD_COUNT, this, f);
}

inline const char *DEFAULT_PRESSURE_CONFIG =
    "config_version=2\nsolver=PCG\nmax_iters=10000\nnorm=L2\nconvergence=ABSOLUTE\nmonitor_residual=1\n"
    "relaxation_factor=1.0\ntolerance=1e-12\npreconditioner(amg_solver)=AMG\namg_solver:algorithm=CLASSICAL\n"
    "amg_solver:selector=PMIS\namg_solver:max_iters=1\namg_solver:presweeps=1\namg_solver:postsweeps=1\n"
    "amg_solver:cycle=V\nprint_solve_stats=0\nprint_grid_stats=0\nobtain_timings=0\n";

// The block text, falling back (with a message) to solver_pressure_config.txt, then to the built-in default.
inline const std::string &PressureConfigText()
{
    if (!PRESSURE_SETTINGS.text.empty())
        return PRESSURE_SETTINGS.text;
    std::ifstream in(LEGACY_PRESSURE_CONFIG);
    if (in.is_open())
    {
        PRESSURE_SETTINGS.text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        std::cout << "No PRESSURE block in the config: pressure solver settings read from "
                  << LEGACY_PRESSURE_CONFIG << "." << std::endl;
    }
    if (PRESSURE_SETTINGS.text.empty())
    {
        PRESSURE_SETTINGS.text = DEFAULT_PRESSURE_CONFIG;
        std::cout << "No PRESSURE block in the config and no " << LEGACY_PRESSURE_CONFIG
                  << ": using the default pressure solver settings." << std::endl;
    }
    return PRESSURE_SETTINGS.text;
}

// Splits the block into trimmed key=value tokens on newlines and commas, dropping blanks and # comments.
inline std::vector<std::string> PressureTokens(const std::string &text)
{
    std::vector<std::string> out;
    std::string token;
    for (size_t n = 0; n <= text.size(); n++)
    {
        const char c = (n < text.size()) ? text[n] : '\n';
        if (c != ',' && c != '\n') { token.push_back(c); continue; }
        const size_t hash = token.find('#');
        if (hash != std::string::npos) token.erase(hash);
        token = TrimWhitespace(token);
        if (!token.empty()) out.push_back(token);
        token.clear();
    }
    return out;
}

// A key of the block that belongs to PRESSURE_SETTINGS' table rather than to AMGX or the Eigen path.
inline bool IsPressureSettingKey(const std::string &token)
{
    const size_t eq = token.find('=');
    return eq != std::string::npos &&
           FindSettingField(PRESSURE_FIELDS, PRESSURE_FIELD_COUNT, TrimWhitespace(token.substr(0, eq))) != nullptr;
}

// The AMGX part of the block as one AMGX_config_create string. `absolute` forces convergence=ABSOLUTE,
// for the Incremental solve, which normalises b by max(||b||, ||b + A p^n||).
inline std::string AmgxConfigString(const std::string &text, bool absolute = false)
{
    std::string out;
    for (std::string t : PressureTokens(text))
    {
        if (t.rfind("eigen_", 0) == 0 || IsPressureSettingKey(t)) continue;
        const size_t eq = t.find('=');
        if (absolute && eq != std::string::npos && TrimWhitespace(t.substr(0, eq)) == "convergence")
            continue;
        if (!out.empty()) out += ", ";
        out += t;
    }
    if (absolute)
        out += std::string(out.empty() ? "" : ", ") + "convergence=ABSOLUTE";
    return out;
}

// For a singular system: no direct solve on the coarsest level (its dense LU divides by a round-off pivot
// and the solve stalls). Empty when the block names a coarse solver itself.
inline std::string AmgxSingularCoarseFix(const std::string &text)
{
    std::string fix;
    if (text.find("coarse_solver") != std::string::npos) return fix;
    for (const std::string &t : PressureTokens(text))
    {
        const size_t eq = t.find('=');
        if (eq == std::string::npos || TrimWhitespace(t.substr(eq + 1)) != "AMG") continue;
        const std::string key = TrimWhitespace(t.substr(0, eq));
        const size_t open = key.find('('), close = key.find(')');
        std::string item;
        if (open != std::string::npos && close != std::string::npos && close > open + 1)
            item = key.substr(open + 1, close - open - 1) + ":coarse_solver=NOSOLVER";
        else if (key == "solver")
            item = "coarse_solver=NOSOLVER";
        if (!item.empty()) fix += (fix.empty() ? "" : ", ") + item;
    }
    return fix;
}

// Reads PRESSURE_SETTINGS from the block. A block carrying none of the table's keys -- a config written before they
// moved in -- falls back to the top level, as a method with no block of its own does.
inline void ReadPressureSettings(const std::string &blockText, const ConfigValues &topLevel)
{
    ConfigValues block;
    for (const std::string &t : PressureTokens(blockText))
        if (IsPressureSettingKey(t))
        {
            const size_t eq = t.find('=');
            block.values[TrimWhitespace(t.substr(0, eq))] = TrimWhitespace(t.substr(eq + 1));
        }
    PRESSURE_SETTINGS.ReadFrom(block.values.empty() ? topLevel : block);
}

// The eigen_* keys; missing keys keep the defaults (tolerance 1e-9 in 2D, 1e-12 in 3D, 2000 iterations).
inline EigenSolverSettings ReadEigenSettings(const std::string &text, int dimension)
{
    EigenSolverSettings s;
    s.tolerance = (dimension == 3) ? 1e-12 : 1e-9;
    for (const std::string &t : PressureTokens(text))
    {
        const size_t eq = t.find('=');
        if (eq == std::string::npos || t.rfind("eigen_", 0) != 0) continue;
        const std::string key = TrimWhitespace(t.substr(0, eq));
        std::string val = TrimWhitespace(t.substr(eq + 1));
        for (char &c : val) c = (char)std::toupper((unsigned char)c);
        try
        {
            if (key == "eigen_tolerance")      s.tolerance = std::stod(val);
            else if (key == "eigen_max_iters") s.maxIters = std::stoi(val);
            else if (key == "eigen_solver")
            {
                if (val == "BICGSTAB") s.solver = EIGEN_BICGSTAB;
                else if (val == "CG")  s.solver = EIGEN_CG;
                else if (val == "AMG") s.solver = EIGEN_AMG;
                else std::cerr << "Warning: unknown eigen_solver '" << val << "', using BICGSTAB." << std::endl;
            }
            else if (key == "eigen_preconditioner")
            {
                if (val == "DIAGONAL")      s.preconditioner = EIGEN_PRECON_DIAGONAL;
                else if (val == "IDENTITY") s.preconditioner = EIGEN_PRECON_IDENTITY;
                else if (val == "ILUT")     s.preconditioner = EIGEN_PRECON_ILUT;
                else if (val == "IC")       s.preconditioner = EIGEN_PRECON_IC;
                else std::cerr << "Warning: unknown eigen_preconditioner '" << val << "', using DIAGONAL." << std::endl;
            }
            else std::cerr << "Warning: unknown pressure key '" << key << "'." << std::endl;
        }
        catch (...)
        {
            std::cerr << "Warning: could not read " << key << " = " << val << "." << std::endl;
        }
    }
    if (s.solver == EIGEN_CG && s.preconditioner == EIGEN_PRECON_ILUT)
        std::cerr << "Warning: eigen_solver=CG with eigen_preconditioner=ILUT does not converge: "
                     "ILUT is not symmetric." << std::endl;
    if (!(s.tolerance > 0.0)) s.tolerance = (dimension == 3) ? 1e-12 : 1e-9;
    if (s.maxIters <= 0) s.maxIters = 2000;
    return s;
}

// ---- shared by the 2D and 3D solvers ----

// What the unknown is: Chorin solves for p, Incremental for phi = p^{n+1} - p^n. The method says which, at init.
enum PressureScheme { PRESSURE_CHORIN = 0, PRESSURE_INCREMENTAL = 1 };
// Who solves, chosen at init: AMGX on the GPU, or the CPU path (Eigen's Krylov solvers or AMGCL, by eigen_solver).
enum PressureBackend { PRESSURE_BACKEND_CPU = 0, PRESSURE_BACKEND_AMGX = 1 };

#ifdef NVIDIA_GPU
struct AMGXSolver {
    AMGX_matrix_handle AmgxA;
    AMGX_vector_handle Amgxb, Amgxx;
    AMGX_solver_handle solver;
    AMGX_resources_handle rsrc;
    AMGX_config_handle config;
};
// AMGX does not survive initialize -> finalize -> initialize in one process, so the library comes up once
// and only DestroyPressureSolver finalizes it, when no solver holds a handle.
inline bool AMGX_LIBRARY_UP = false;
inline int AMGX_USERS = 0;
#else
struct AMGXSolver;
#endif

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMatrix;

// The assembled system, stored CSR: AMGX uploads its arrays as they are and Eigen threads the product.
// `layout` is the cell-type array A was built from; while it stands, A and the hierarchy are reused.
struct PressureSystem
{
    RowMatrix A;
    std::vector<unsigned char> layout;
    int Nx = 0, Ny = 0, Nz = 0;
    bool built = false;
    bool singular = false;   // no Dirichlet neighbour anywhere: p is fixed only up to a constant
    long version = 0;        // which assembly this is; the AMGCL hierarchy is keyed on it
    PressureScheme scheme = PRESSURE_CHORIN;
    PressureBackend backend = PRESSURE_BACKEND_CPU;
    EigenSolverSettings eigen;
    AMGXSolver *amgx = nullptr;
    bool amgxNoCoarse = false;   // the AMGX config carries AmgxSingularCoarseFix (set for a singular system)
};
inline long PRESSURE_ASSEMBLIES = 0;   // unique across the 2D and 3D solvers, so one AMGCL cache serves both

inline bool PressureLayoutChanged(const PressureSystem &s, const MAC *g)
{
    return !s.built || s.Nx != g->Nx || s.Ny != g->Ny || s.Nz != g->Nz || s.layout != g->CellTypes();
}

// A pressure-outlet ring cell's value belongs on the boundary face, half the fluid cell from its centre, so
// its coefficient takes the half cell: FaceGap / (Width/2). An interior Dirichlet cell keeps the full gap.
inline double DirichletRingFactor(const MAC *g, MAC::Axis a, int n, int s)
{
    return g->FaceGap(a, s < 0 ? n - 1 : n) / (0.5 * g->Width(a, n));
}

// Subtracts the mean of v. Called twice after the compatibility step: one pass leaves ~n*eps of the mean,
// which BiCGSTAB amplified until it diverged.
inline void SubtractMean(double *v, int n)
{
    if (v == nullptr || n <= 0) return;
    double m = 0.0;
    for (int i = 0; i < n; i++) m += v[i];
    m /= (double)n;
    for (int i = 0; i < n; i++) v[i] -= m;
}

// ---- AMGCL ----

typedef amgcl::backend::builtin<double> AmgBackend;
typedef amgcl::make_solver<amgcl::amg<AmgBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
                           amgcl::solver::cg<AmgBackend>> AmgSolver;

// The AMG hierarchy, built once per assembly: an Eulerian run keeps one matrix for the whole run, FLIP rebuilds
// it when its fluid region moves. The version (PressureSystem::version) says which.
struct AmgPressureCache
{
    std::unique_ptr<AmgSolver> solver;
    std::unique_ptr<AmgBackend::matrix> system;   // the singular system CG iterates on; null when not singular
    long version = -1;
    std::vector<double> rhs, sol;
};
inline AmgPressureCache AMG_CACHE;

// CG preconditioned by smoothed-aggregation AMG on (-A) x = -b. A singular system builds its hierarchy on a
// copy with one diagonal doubled, so the coarsest level can be factorised; CG iterates on the true A.
inline int SolveAmgclPressure(const RowMatrix &A, long version, const Eigen::VectorXd &b, Eigen::VectorXd &x,
                              const Eigen::VectorXd *guess, const EigenSolverSettings &s, double referenceNorm,
                              bool singular, double &error)
{
    AmgPressureCache &c = AMG_CACHE;
    const long n = (long)A.rows();
    if (!c.solver || c.version != version)
    {
        RowMatrix R = -A;
        R.makeCompressed();
        std::vector<ptrdiff_t> ptr(R.outerIndexPtr(), R.outerIndexPtr() + n + 1);
        std::vector<ptrdiff_t> col(R.innerIndexPtr(), R.innerIndexPtr() + R.nonZeros());
        std::vector<double> val(R.valuePtr(), R.valuePtr() + R.nonZeros());
        AmgSolver::params prm;
        prm.precond.direct_coarse = true;
        c.system.reset();
        if (singular && n > 0)
        {
            c.system.reset(new AmgBackend::matrix(std::tie(n, ptr, col, val)));
            for (ptrdiff_t q = ptr[0]; q < ptr[1]; q++)
                if (col[q] == 0) val[q] *= 2.0;   // row 0's diagonal: the copy the hierarchy is built on
        }
        c.solver.reset(new AmgSolver(std::tie(n, ptr, col, val), prm));
        c.version = version;
        c.rhs.resize(n); c.sol.resize(n);
    }
    // The stopping rule is set per call, so a changing floor does not rebuild the hierarchy.
    c.solver->solver().prm.tol = s.tolerance;
    c.solver->solver().prm.abstol = s.tolerance * referenceNorm;
    c.solver->solver().prm.maxiter = s.maxIters;
    for (long i = 0; i < n; i++) { c.rhs[i] = -b(i); c.sol[i] = guess ? (*guess)(i) : 0.0; }
    size_t iters = 0;
    if (c.system) std::tie(iters, error) = (*c.solver)(*c.system, c.rhs, c.sol);
    else          std::tie(iters, error) = (*c.solver)(c.rhs, c.sol);
    x.resize(n);
    for (long i = 0; i < n; i++) x(i) = c.sol[i];
    return (int)iters;
}

// ---- Eigen's Krylov solvers ----

// One solve with one Eigen solver type. Every solver/preconditioner pair is a different C++ type, so this is a
// template usually i avoid templates, but in this case, there is no escaping This was a 5 if else chain before
template <class Solver>
inline int RunKrylov(const RowMatrix &A, const Eigen::VectorXd &b, Eigen::VectorXd &x, const Eigen::VectorXd *guess,
                     const EigenSolverSettings &s, Eigen::ComputationInfo &info)
{
    Solver solver;
    solver.setMaxIterations(s.maxIters);
    solver.setTolerance(s.tolerance);
    solver.compute(A);
    if (guess) x = solver.solveWithGuess(b, *guess); else x = solver.solve(b);
    info = solver.info();
    return (int)solver.iterations();
}

inline constexpr int UPLO = Eigen::Lower | Eigen::Upper;
typedef Eigen::IncompleteCholesky<double, Eigen::Lower | Eigen::Upper> IC;
typedef Eigen::ConjugateGradient<RowMatrix, UPLO, Eigen::DiagonalPreconditioner<double>> CGDiagonal;
typedef Eigen::ConjugateGradient<RowMatrix, UPLO, Eigen::IdentityPreconditioner> CGIdentity;
typedef Eigen::ConjugateGradient<RowMatrix, UPLO, Eigen::IncompleteLUT<double>> CGIlut;
typedef Eigen::ConjugateGradient<RowMatrix, UPLO, IC> CGIc;
typedef Eigen::BiCGSTAB<RowMatrix, Eigen::DiagonalPreconditioner<double>> BiCGSTABDiagonal;
typedef Eigen::BiCGSTAB<RowMatrix, Eigen::IdentityPreconditioner> BiCGSTABIdentity;
typedef Eigen::BiCGSTAB<RowMatrix, Eigen::IncompleteLUT<double>> BiCGSTABIlut;
typedef Eigen::BiCGSTAB<RowMatrix, IC> BiCGSTABIc;

// Solves A x = b with the chosen Eigen solver and preconditioner (or AMGCL), from `guess` when not null.
// Stops at ||r|| < tol * max(||b||, referenceNorm): a non-zero referenceNorm is an absolute floor. Returns the iterations.
inline int SolveEigenPressure(const RowMatrix &A, long version, const Eigen::VectorXd &b, Eigen::VectorXd &x,
                       const Eigen::VectorXd *guess, const EigenSolverSettings &sIn, double referenceNorm = 0.0,
                       bool singular = true)
{
    // An absolute floor tol * referenceNorm, expressed as the relative tolerance Eigen understands. When
    // b is already below it there is nothing to solve: the answer is the guess (the increment is zero).
    EigenSolverSettings s = sIn;
    if (referenceNorm > 0.0 && s.solver != EIGEN_AMG)
    {
        const double bn = b.norm();
        if (bn <= s.tolerance * referenceNorm)
        {
            if (guess) x = *guess; else x = Eigen::VectorXd::Zero(b.size());
            return 0;
        }
        s.tolerance = std::min(0.5, s.tolerance * referenceNorm / bn);
    }
    int iters = 0;
    Eigen::ComputationInfo info = Eigen::Success;

    if (s.solver == EIGEN_AMG)
    {
        double error = 0.0;
        iters = SolveAmgclPressure(A, version, b, x, guess, s, referenceNorm, singular, error);
        if (iters >= s.maxIters) info = Eigen::NoConvergence;
    }
    else
    {
        const RowMatrix &R = A;
        const bool cg = (s.solver == EIGEN_CG);
        switch (s.preconditioner)
        {
        case EIGEN_PRECON_IDENTITY:
            iters = cg ? RunKrylov<CGIdentity>(R, b, x, guess, s, info) : RunKrylov<BiCGSTABIdentity>(R, b, x, guess, s, info);
            break;
        case EIGEN_PRECON_ILUT:
            iters = cg ? RunKrylov<CGIlut>(R, b, x, guess, s, info) : RunKrylov<BiCGSTABIlut>(R, b, x, guess, s, info);
            break;
        case EIGEN_PRECON_IC:
            iters = cg ? RunKrylov<CGIc>(R, b, x, guess, s, info) : RunKrylov<BiCGSTABIc>(R, b, x, guess, s, info);
            break;
        default:
            iters = cg ? RunKrylov<CGDiagonal>(R, b, x, guess, s, info) : RunKrylov<BiCGSTABDiagonal>(R, b, x, guess, s, info);
            break;
        }

        // BiCGSTAB can break down on this singular symmetric system and return garbage. The matrix is
        // symmetric, so the same solve is redone with CG; say so once.
        if (info != Eigen::Success && s.solver == EIGEN_BICGSTAB)
        {
            static bool fellBack = false;
            if (!fellBack)
            {
                fellBack = true;
                std::cerr << "Warning: BiCGSTAB did not converge on a pressure solve; retrying that solve with CG "
                             "(reported once)." << std::endl;
            }
            iters += RunKrylov<CGDiagonal>(R, b, x, guess, s, info);
        }
    }

    static bool reported = false;
    if (info != Eigen::Success && !reported)
    {
        reported = true;
        std::cerr << "Warning: the Eigen pressure solve did not converge ("
                  << (info == Eigen::NoConvergence ? "iteration limit reached" : "preconditioner or breakdown failure")
                  << " after " << iters << " iterations). Reported once." << std::endl;
    }
    return iters;
}

// ---- the backend: one set of calls whatever solves ----

// Stands up the backend the system names. AMGX gets its config once; under Incremental its convergence is ABSOLUTE
// (see AmgxConfigString). The CPU path needs nothing beyond the eigen_* settings already read.
inline void CreatePressureBackend(PressureSystem &s)
{
#ifdef NVIDIA_GPU
    if (s.backend != PRESSURE_BACKEND_AMGX)
        return;
    if (!AMGX_LIBRARY_UP)
    {
        AMGX_initialize();
        AMGX_LIBRARY_UP = true;
    }
    AMGX_USERS++;
    s.amgx = new AMGXSolver();
    std::string cfg = AmgxConfigString(PressureConfigText(), s.scheme == PRESSURE_INCREMENTAL);
    if (s.amgxNoCoarse)
    {
        const std::string fix = AmgxSingularCoarseFix(PressureConfigText());
        if (!fix.empty()) cfg += (cfg.empty() ? "" : ", ") + fix;
    }
    AMGX_config_create(&s.amgx->config, cfg.c_str());
    AMGX_resources_create_simple(&s.amgx->rsrc, s.amgx->config);
    AMGX_matrix_create(&s.amgx->AmgxA, s.amgx->rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&s.amgx->Amgxb, s.amgx->rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&s.amgx->Amgxx, s.amgx->rsrc, AMGX_mode_dDDI);
    AMGX_solver_create(&s.amgx->solver, s.amgx->rsrc, AMGX_mode_dDDI, s.amgx->config);
#else
    s.backend = PRESSURE_BACKEND_CPU;
#endif
}

inline void DestroyPressureBackend(PressureSystem &s, bool finalize)
{
#ifdef NVIDIA_GPU
    if (s.amgx != nullptr)
    {
        AMGX_solver_destroy(s.amgx->solver);
        AMGX_vector_destroy(s.amgx->Amgxx);
        AMGX_vector_destroy(s.amgx->Amgxb);
        AMGX_matrix_destroy(s.amgx->AmgxA);
        AMGX_resources_destroy(s.amgx->rsrc);
        AMGX_config_destroy(s.amgx->config);
        delete s.amgx;
        s.amgx = nullptr;
        AMGX_USERS--;
    }
    if (finalize && AMGX_LIBRARY_UP && AMGX_USERS == 0)
    {
        AMGX_finalize();
        AMGX_LIBRARY_UP = false;
    }
#endif
    s.built = false;
}

// Makes the triplets the system's matrix, records the layout it belongs to, and hands it to the backend: AMGX
// uploads it and builds its hierarchy here, AMGCL on its first solve (keyed on the new version).
inline void CommitPressureMatrix(PressureSystem &s, const MAC *g, std::vector<Eigen::Triplet<double>> &t, int n)
{
#ifdef NVIDIA_GPU
    // A singular system must not be solved directly on the coarsest level: rebuild the AMGX handles with the coarse
    // solver off when singularity changes (once for a closed box; FLIP's free surface makes its system regular).
    const bool noCoarse = s.singular && !AmgxSingularCoarseFix(PressureConfigText()).empty();
    if (s.amgx != nullptr && noCoarse != s.amgxNoCoarse)
    {
        DestroyPressureBackend(s, false);
        s.amgxNoCoarse = noCoarse;
        CreatePressureBackend(s);
        if (noCoarse)
            std::cout << "AMGX: singular pressure system, coarsest level smoothed only (coarse_solver=NOSOLVER)\n";
    }
#endif
    s.A.resize(n, n);
    s.A.setFromTriplets(t.begin(), t.end());
    s.A.makeCompressed();
    s.layout = g->CellTypes();
    s.Nx = g->Nx; s.Ny = g->Ny; s.Nz = g->Nz;
    s.version = ++PRESSURE_ASSEMBLIES;
    s.built = true;
#ifdef NVIDIA_GPU
    if (s.amgx != nullptr)
    {
        AMGX_SAFE_CALL(AMGX_matrix_upload_all(s.amgx->AmgxA, n, (int)s.A.nonZeros(), 1, 1, s.A.outerIndexPtr(),
                                              s.A.innerIndexPtr(), s.A.valuePtr(), nullptr));
        AMGX_SAFE_CALL(AMGX_vector_set_zero(s.amgx->Amgxx, n, 1));
        AMGX_SAFE_CALL(AMGX_vector_set_zero(s.amgx->Amgxb, n, 1));
        AMGX_SAFE_CALL(AMGX_solver_setup(s.amgx->solver, s.amgx->AmgxA));
    }
#endif
}

// Solves A x = b on the backend, from `guess` when not null, to ||r|| < tol max(||b||, referenceNorm). AMGX
// with a guess or under Incremental solves A y = b/beta, beta = max(||b||, referenceNorm). Returns iterations.
inline int SolvePressureSystem(PressureSystem &s, const Eigen::VectorXd &b, Eigen::VectorXd &x,
                               const Eigen::VectorXd *guess, double referenceNorm)
{
    const int n = (int)s.A.rows();
    x.resize(n);
#ifdef NVIDIA_GPU
    if (s.amgx != nullptr)
    {
        Eigen::VectorXd bs = b;
        double beta = 0.0;
        const bool normalise = (guess != nullptr) || s.scheme == PRESSURE_INCREMENTAL;
        if (normalise)
        {
            for (int i = 0; i < n; i++)
                beta += b(i) * b(i);
            beta = std::max(std::sqrt(beta), referenceNorm);
            const double inv = (beta > 0.0) ? (1.0 / beta) : 1.0;
            for (int i = 0; i < n; i++)
            {
                bs(i) = b(i) * inv;
                if (guess) x(i) = (*guess)(i) * inv;
            }
        }
        AMGX_SAFE_CALL(AMGX_vector_upload(s.amgx->Amgxb, n, 1, bs.data()));
        if (guess)
        {
            AMGX_SAFE_CALL(AMGX_vector_upload(s.amgx->Amgxx, n, 1, x.data()));
            AMGX_SAFE_CALL(AMGX_solver_solve(s.amgx->solver, s.amgx->Amgxb, s.amgx->Amgxx));
        }
        else
        {
            // with_0_initial_guess ASSUMES x is zero and does not clear it: skip this and a second solve returns
            // the previous answer plus the new one.
            AMGX_SAFE_CALL(AMGX_vector_set_zero(s.amgx->Amgxx, n, 1));
            AMGX_SAFE_CALL(AMGX_solver_solve_with_0_initial_guess(s.amgx->solver, s.amgx->Amgxb, s.amgx->Amgxx));
        }
        AMGX_SAFE_CALL(AMGX_vector_download(s.amgx->Amgxx, x.data()));
        if (normalise && beta > 0.0)
            for (int i = 0; i < n; i++)
                x(i) *= beta;
        int its = 0;
        AMGX_solver_get_iterations_number(s.amgx->solver, &its);
        return its;
    }
#endif
    return SolveEigenPressure(s.A, s.version, b, x, guess, s.eigen, referenceNorm, s.singular);
}

#endif
