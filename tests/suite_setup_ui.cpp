// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The setup_ui suite: setup_ui_check. Each case runs in a child process of its own, as it did
// when each was a separate program, so no state leaks between them. Built and run by tests/run_tests.sh.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include <fstream>
#include <sstream>
#include "headers/Core/MAC.h"
#include "headers/Core/Definitions.h"
#include "headers/Core/ConfigReader.h"
#include "headers/Core/SetupScreen.h"
#include "headers/Core/SimulationManager.h"
#include "TestMethods.h"
#ifdef NVIDIA_GPU
#include <amgx_c.h>   // the AMGX checks need the GPU build; a CPU-only build skips them
#endif
#include "SuiteMain.h"

// ------------------------------------------------------------------------------------------------------------------
// setup_ui_check
// ------------------------------------------------------------------------------------------------------------------
namespace setup_ui_check
{
// Headless check for the setup screen's non-UI logic: that the compatibility rules match
// what the backend actually dispatches on, that a written config round-trips, that
// ConfigReader parses it into the state the panel intended, and that AMGX accepts the
// solver string it generates. Render() is never called, so no GL context is needed.



// The editor reads scenes only as text (from a config's SCENE_BEGIN block); fixtures here
// are still written as files, so they are read into a string first.
static std::string ReadText(const std::string &path)
{
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static int failures = 0;
static void check(bool ok, const std::string &name, const std::string &detail = "")
{
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) failures++;
}

// The harness runs in a scratch directory so it cannot clobber the project's own config
// files. The shipped presets live in the project root, so remember it before moving.
static std::string PROJECT_ROOT = ".";

int run(int argc, char **argv)
{
    RegisterBuiltinMethods();   // the panel and ConfigReader list and look up the registered methods
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    {
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd))) PROJECT_ROOT = cwd;
        if (const char *root = std::getenv("NEREUS_ROOT")) PROJECT_ROOT = root;   // read only: the shipped presets
    }
    if (argc > 1 && chdir(argv[1]) != 0)
    {
        std::printf("could not chdir to %s\n", argv[1]);
        return 1;
    }

    // ---- 1. the method table matches the backend dispatch ----
    //
    // There is no geometry table any more: the scene carries the geometry and the boundary
    // conditions, and a config with no scene is a closed box, so every method runs every case.
    // What is left to police is the DIMENSION each method offers, which each method's registered
    // row holds and SimulationManager::InitializeSimulation dispatches on.
    std::printf("\n== offered combinations ==\n");
    {
        struct Expect { int dim; const char *method; bool ok; };
        const Expect expects[] = {
            {2, "ADI",  true},
            {3, "ADI",  true},
            {2, "FLIP", true},
            {3, "FLIP", true},
            {2, "SPH",  true},
            {3, "SPH",  true},   // SPH3D is wired into the manager
        };
        bool tableOk = true;
        for (const Expect &e : expects)
            if (MethodFromKey(e.method) < 0 ||
                SimulationSetupUI::MethodSupported(e.dim, MethodFromKey(e.method)) != e.ok)
            {
                tableOk = false;
                std::printf("     mismatch: %dD %s\n", e.dim, e.method);
            }
        check(tableOk, "method list matches the manager's wired branches");

        // Every method in the table is reachable from the panel in at least one dimension --
        // a registered method appears in the radio list by existing.
        bool reachable = METHOD_COUNT > 0;
        for (int i = 0; i < METHOD_COUNT; i++)
            if (!MethodOf(i).supports2D && !MethodOf(i).supports3D)
                reachable = false;
        check(reachable, "every registered method is offered in some dimension");
    }

    // ---- 2. an illegal selection is snapped back into legal territory ----
    {
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("SPH");
        ui.dimension = 3;           // user flips to 3D with SPH selected
        ui.ReconcileSelections();
        check(SimulationSetupUI::MethodSupported(ui.dimension, ui.method),
              "switching dimension leaves a legal method selected",
              std::string("-> ") + ui.MethodKey());
    }

    // ---- 3. round trip through the written file ----
    std::printf("\n== round trip ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 3;
        ui.method = MethodFromKey("FLIP");
        ui.ReconcileSelections();
        ui.x0 = 0.0; ui.xf = 0.6; ui.y0 = 0.0; ui.yf = 0.3; ui.z0 = 0.0; ui.zf = 0.3;
        ui.gridSize = 30;
        ui.viscosity = 2.5e-3;
        ui.density = 998.0;
        ui.forceU = "0"; ui.forceV = "-9.81"; ui.forceW = "0";
        ui.adaptiveTimestep = false;
        ui.timeStep = 1.5e-3;
        FLIP_SETTINGS.particlesPerCell = 8;
        FLIP_SETTINGS.alpha = 0.93;
        FLIP_SETTINGS.physicalDiffusion = true;
        FLIP_SETTINGS.diffusionSolver = DIFFUSION_SOLVERS::ADI;
        ui.useGPU = true;
        ui.threadCount = 5;
        ui.characteristicLength2 = 0.0205;
        ui.exports.dambreak = true;
        ui.exports.aerodynamics = true;
        std::snprintf(ui.exportPath, sizeof(ui.exportPath), "%s", "ExportsTest");

        std::string err;
        check(ui.WriteFiles(err), "WriteFiles succeeds", err);

        // The panel edits the method's one global instance, so scramble it: only the file can put
        // the values back.
        FLIP_SETTINGS.particlesPerCell = 1; FLIP_SETTINGS.alpha = 0.0; FLIP_SETTINGS.physicalDiffusion = false;
        FLIP_SETTINGS.diffusionSolver = DIFFUSION_SOLVERS::IMPLICIT;
        SimulationSetupUI back;
        back.LoadFrom("simulation_config.txt");
        bool same = back.dimension == 3 &&
                    back.method == MethodFromKey("FLIP") &&
                    back.exports.dambreak && back.exports.aerodynamics &&
                    back.gridSize == 30 &&
                    std::abs(back.viscosity - 2.5e-3) < 1e-12 &&
                    std::abs(back.density - 998.0) < 1e-9 &&
                    back.forceV == "-9.81" &&
                    std::abs(back.timeStep - 1.5e-3) < 1e-12 &&
                    FLIP_SETTINGS.particlesPerCell == 8 &&
                    std::abs(FLIP_SETTINGS.alpha - 0.93) < 1e-9 &&
                    FLIP_SETTINGS.physicalDiffusion && FLIP_SETTINGS.diffusionSolver == DIFFUSION_SOLVERS::ADI &&
                    back.useGPU && back.threadCount == 5 &&
                    std::string(back.exportPath) == "ExportsTest" &&
                    std::abs(back.characteristicLength2 - 0.0205) < 1e-12;
        check(same, "every field survives write then read");

        // ---- 4. ConfigReader agrees with what the panel meant ----
        SIMULATION_CONFIG probe;
        ConfigReader::loadConfig(probe, "simulation_config.txt");
        check(DIMENSION == 3, "ConfigReader: DIMENSION", std::to_string(DIMENSION));
        check(SIM_TYPE == MethodFromKey("FLIP"), "ConfigReader: TYPE = FLIP");
        check(SIMULATION.exports.dambreak && SIMULATION.exports.aerodynamics,
              "ConfigReader: EXPORT_DAMBREAK / EXPORT_AERODYNAMICS");
#ifdef NVIDIA_GPU
        check(GPU_ACCELERATION == 1, "ConfigReader: GPU_ACCELERATION");
#else
        check(GPU_ACCELERATION == 0, "ConfigReader: GPU_ACCELERATION forced off in a CPU-only build");
#endif
        check(THREAD_COUNT == 5, "ConfigReader: THREAD_COUNT");
        check(std::abs(SIMULATION.EPS - 2.5e-3) < 1e-12, "ConfigReader: VISCOSITY");
        check(std::abs(SIMULATION.RHO - 998.0) < 1e-9, "ConfigReader: DENSITY");
        check(std::abs(SIMULATION.f.v + 9.81) < 1e-9, "ConfigReader: Fv");
        check(std::abs(SIMULATION.CHARACTERISTIC_LENGTH_2 - 0.0205) < 1e-12, "ConfigReader: CHARACTERISTIC_LENGTH_2");
        check(FLIP_SETTINGS.physicalDiffusion == 1, "ConfigReader: PHYSICAL_DIFFUSION");
        check(FLIP_SETTINGS.diffusionSolver == DIFFUSION_SOLVERS::ADI, "ConfigReader: DIFFUSION_SOLVER");
        check(SIMULATION.SolidMaskFunction != nullptr, "ConfigReader: a 3D solid mask got wired up");

        double dh; int Nx, Ny, Nz;
        ui.DerivedGrid(dh, Nx, Ny, Nz);
        check(Nx == SIMULATION.Nx && Ny == SIMULATION.Ny && Nz == SIMULATION.Nz,
              "the panel's grid preview matches what MAC actually builds",
              "preview " + std::to_string(Nx) + "x" + std::to_string(Ny) + "x" + std::to_string(Nz) +
                  " vs actual " + std::to_string(SIMULATION.Nx) + "x" +
                  std::to_string(SIMULATION.Ny) + "x" + std::to_string(SIMULATION.Nz));
    }

    // ---- 4b. hand-written files use looser formatting than the writer emits ----
    std::printf("\n== hand-written formatting ==\n");
    {
        {
            std::ofstream f("messy_config.txt");
            f << "# a comment line\n";
            f << "DIMENSION = 2\n";
            f << "TYPE = FLIP\n";
            f << "GPU_ACCELERATION = 1 #trailing comment after a space\n";
            f << "ALPHA = 0.98#comment with no space\n";
            f << "Fu = 0.0;\n";                       // stray semicolon
            f << "Fv = -9.8\n";
            f << "VISCOSITY =  1e-6\n";               // double space
            f << "DENSITY = 1000.0 \n";               // trailing space
            f << "GRID_SIZE = 64\n";
            f << "REYNOLDS_NUMBER = 0     #naming only\n";
        }
        SimulationSetupUI m;
        m.LoadFrom("messy_config.txt");
        check(m.dimension == 2, "messy: DIMENSION");
        check(m.method == MethodFromKey("FLIP"), "messy: TYPE");
        check(m.useGPU, "messy: value with a spaced trailing comment");
        check(std::abs(FLIP_SETTINGS.alpha - 0.98) < 1e-9, "messy: comment with no space before it (a top-level method key, from before the blocks)",
              std::to_string(FLIP_SETTINGS.alpha));
        check(m.forceU == "0", "messy: stray semicolon", m.forceU);
        check(m.forceV == "-9.8", "messy: Fv", m.forceV);
        check(std::abs(m.viscosity - 1e-6) < 1e-18, "messy: double space", std::to_string(m.viscosity));
        check(std::abs(m.density - 1000.0) < 1e-9, "messy: trailing space");
        check(m.gridSize == 64, "messy: GRID_SIZE");
        check(std::abs(m.reynoldsForNaming) < 1e-12, "messy: REYNOLDS_NUMBER = 0");
    }

    // ---- 4c. the PRESSURE block round-trips, AMGX and Eigen settings both, and the backend reads it ----
    std::printf("\n== pressure block round trip ==\n");
    {
        {
            std::ofstream f("pressure_rt.cfg");
            f << "DIMENSION = 2\nTYPE = ADI\nGRID_SIZE = 16\n\nPRESSURE_BEGIN\n"
              << "config_version=2\nsolver=PBICGSTAB\nmax_iters=1234\nnorm=LMAX\n"
              << "convergence=ABSOLUTE\nmonitor_residual=1\nrelaxation_factor=1.0\n"
              << "tolerance=1e-15\npreconditioner(amg_solver)=AMG\n"
              << "amg_solver:algorithm=AGGREGATION\namg_solver:max_iters=2\n"
              << "amg_solver:presweeps=3\namg_solver:postsweeps=4\namg_solver:cycle=W\n"
              << "print_solve_stats=1\nprint_grid_stats=0\nobtain_timings=1\n"
              << "eigen_solver=CG\neigen_preconditioner=ILUT\neigen_tolerance=3e-11\neigen_max_iters=777\n"
              << "PRESSURE_END\n";
        }
        SimulationSetupUI a;
        a.LoadFrom("pressure_rt.cfg");
        check(a.amgxSolver == 1, "AMGX: solver", std::to_string(a.amgxSolver));
        check(a.amgxMaxIters == 1234, "AMGX: max_iters");
        check(std::abs(a.amgxTolerance - 1e-15) < 1e-25, "AMGX: tolerance preserved",
              std::to_string(a.amgxTolerance));
        check(a.amgxNorm == 2 && a.amgxConvergence == 1, "AMGX: norm and convergence",
              "norm idx " + std::to_string(a.amgxNorm));
        check(a.amgxUsePrecon && a.amgxPreconAlg == 1 && a.amgxCycle == 1,
              "AMGX: preconditioner, coarsening and cycle");
        check(a.amgxPresweeps == 3 && a.amgxPostsweeps == 4 && a.amgxPreIters == 2,
              "AMGX: sweep counts");
        check(a.amgxSolveStats && !a.amgxGridStats && a.amgxTimings, "AMGX: stats flags");
        check(a.eigenSolver == EIGEN_CG && a.eigenPreconditioner == EIGEN_PRECON_ILUT &&
                  std::abs(a.eigenTolerance - 3e-11) < 1e-22 && a.eigenMaxIters == 777,
              "Eigen: solver, preconditioner, tolerance, iterations");

        std::string err;
        a.useGPU = true;
        check(a.WriteFiles(err, "pressure_rt.cfg"), "the block is written back", err);
        SimulationSetupUI b;
        b.LoadFrom("pressure_rt.cfg");
        check(b.amgxSolver == 1 && b.amgxMaxIters == 1234 && b.amgxNorm == 2 &&
                  b.amgxConvergence == 1 && b.amgxPreconAlg == 1 && b.amgxCycle == 1 &&
                  b.amgxPresweeps == 3 && b.amgxPostsweeps == 4,
              "AMGX settings survive write then read");
        check(b.eigenSolver == EIGEN_CG && b.eigenPreconditioner == EIGEN_PRECON_ILUT &&
                  std::abs(b.eigenTolerance - 3e-11) < 1e-22 && b.eigenMaxIters == 777,
              "Eigen settings survive write then read");
        check(a.PressureBlockText() == b.PressureBlockText(), "a second trip is byte for byte identical");

        SIMULATION_CONFIG c;
        ConfigReader::loadConfig(c, "pressure_rt.cfg");
        const EigenSolverSettings es = ReadEigenSettings(PressureConfigText(), 2);
        check(es.solver == EIGEN_CG && es.preconditioner == EIGEN_PRECON_ILUT && es.maxIters == 777,
              "the backend reads the Eigen settings from the block");
        const std::string amgx = AmgxConfigString(PressureConfigText());
        check(amgx.find("solver=PBICGSTAB") != std::string::npos && amgx.find("eigen_") == std::string::npos &&
                  amgx.find('\n') == std::string::npos,
              "and hands AMGX one string without the eigen_* keys", amgx);
#ifdef NVIDIA_GPU
        AMGX_initialize();
        AMGX_config_handle cfg = nullptr;
        check(AMGX_config_create(&cfg, amgx.c_str()) == AMGX_RC_OK, "which AMGX parses", amgx);
        if (cfg) AMGX_config_destroy(cfg);
        AMGX_finalize();
#endif

        {
            std::ofstream f("legacy_probe.cfg");
            f << "DIMENSION = 2\nTYPE = ADI\nGRID_SIZE = 16\nDOMAIN_X0 = 0\nDOMAIN_XF = 1\nDOMAIN_Y0 = 0\n"
              << "DOMAIN_YF = 1\nDOMAIN_Z0 = 0\nDOMAIN_ZF = 1\n";
            std::ofstream g("legacy_probe.pcfg");
            g << "config_version=2\nsolver=FGMRES\ntolerance=1e-7\n";
        }
        ConfigReader::loadConfig(c, "legacy_probe.cfg");
        const EigenSolverSettings ls = ReadEigenSettings(PressureConfigText(), 3);
        check(PressureConfigText().find("solver=FGMRES") != std::string::npos,
              "a config without the block reads the legacy .pcfg beside it");
        check(ls.solver == EIGEN_BICGSTAB && ls.preconditioner == EIGEN_PRECON_DIAGONAL &&
                  ls.tolerance == 1e-12 && ls.maxIters == 2000,
              "and missing eigen_* keys keep the old defaults (1e-12 in 3D)");
        std::remove("legacy_probe.pcfg");
        ConfigReader::loadConfig(c, "legacy_probe.cfg");
        check(PressureConfigText().find("config_version=2") != std::string::npos,
              "with neither, the pressure settings still resolve");
        std::remove("legacy_probe.cfg");
        std::remove("pressure_rt.cfg");
    }

    // ---- 4d. the scene editor round-trips through the real parser ----
    std::printf("\n== scene editor ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        ui.x0 = 0; ui.xf = 1.1; ui.y0 = 0; ui.yf = 0.41; ui.gridSize = 40;

        // walls last so they beat the inlet at the corners
        ui.faceOrder[0] = 0; ui.faceOrder[1] = 1; ui.faceOrder[2] = 4;
        ui.faceOrder[3] = 5; ui.faceOrder[4] = 2; ui.faceOrder[5] = 3;
        ui.uiFaces[0].type = BC_INLET;
        ui.uiFaces[0].u = "1.5 - (1.5/(0.205*0.205))*(y-0.205)^2";
        ui.uiFaces[0].clampNegative = true;
        ui.uiFaces[1].type = BC_PRESSURE_OUTLET;
        ui.uiFaces[2].type = 0; ui.uiFaces[3].type = 0;

        SimulationSetupUI::UiRegion cyl;
        cyl.kind = 0; cyl.prim = 2; cyl.a[0] = 0.2; cyl.a[1] = 0.2; cyl.radius = 0.05; cyl.axis = 2;
        ui.uiRegions.push_back(cyl);

        check(ui.ApplyScene(), "editor state parses as a scene", ui.sceneStatus);
        check(SCENE.regions.size() == 1 && SCENE.regions[0].radius == 0.05,
              "the region survives the round trip");
        check(SCENE.faces[0].type == BC_INLET &&
                  SCENE.faces[0].u.source.find("0.205") != std::string::npos,
              "the inlet expression survives the round trip");
        check(SCENE.faces[3].order > SCENE.faces[0].order,
              "the declared order is preserved (walls after the inlet)");

        // reading it back into a fresh editor must reproduce the same state
        SimulationSetupUI back;
        back.LoadSceneTextIntoEditor(ui.SceneText());
        check(back.uiRegions.size() == 1 && back.uiRegions[0].prim == 2 &&
                  std::abs(back.uiRegions[0].radius - 0.05) < 1e-12,
              "loading the scene back into the editor recovers the region");
        check(std::string(back.uiFaces[0].u) == std::string(ui.uiFaces[0].u),
              "loading recovers the expression text verbatim",
              std::string(back.uiFaces[0].u));
        // In 2D the z faces are not written at all, so absolute positions cannot round
        // trip. What has to survive is the relative order of the faces that were written,
        // because that is what decides the corners.
        auto posOf = [&](int face)
        {
            for (int n = 0; n < 6; n++) if (back.faceOrder[n] == face) return n;
            return -1;
        };
        check(posOf(2) > posOf(0) && posOf(2) > posOf(1) &&
                  posOf(3) > posOf(0) && posOf(3) > posOf(1),
              "loading recovers the relative order (walls still after the inlet)",
              "xmin@" + std::to_string(posOf(0)) + " xmax@" + std::to_string(posOf(1)) +
                  " ymin@" + std::to_string(posOf(2)) + " ymax@" + std::to_string(posOf(3)));

        // a bad expression is caught in the editor, not at startup
        ui.uiFaces[0].u = "1.5 - ((((";
        check(!ui.ApplyScene(), "a broken expression is rejected by Validate",
              ui.sceneStatus);
    }

    // ---- 4d'. a mesh region round-trips through the editor ----
    //          (file, centre, rotation and a NON-uniform scale; a fitted scale carries
    //           more digits than the stream's default six, which would round it)
    std::printf("\n== mesh region round trip ==\n");
    {
        // A closed unit cube, written here so the check does not depend on Models/;
        // `drop` leaves one triangle out, which opens it.
        auto writeCube = [](const char *file, int drop)
        {
            std::ofstream f(file);
            const int T[12][3] = {{0,4,6},{0,6,2},{1,3,7},{1,7,5},{0,1,5},{0,5,4},
                                  {2,6,7},{2,7,3},{0,2,3},{0,3,1},{4,5,7},{4,7,6}};
            f << "solid c\n";
            for (int t = 0; t < 12; t++)
            {
                if (t == drop) continue;
                f << "facet normal 0 0 0\nouter loop\n";
                for (int v : T[t])
                    f << "vertex " << ((v & 1) ? 1 : -1) << " " << ((v & 2) ? 1 : -1) << " "
                      << ((v & 4) ? 1 : -1) << "\n";
                f << "endloop\nendfacet\n";
            }
            f << "endsolid c\n";
        };
        writeCube("mesh_cube.stl", -1);
        writeCube("mesh_open.stl", 5);
        SimulationSetupUI ui;
        ui.dimension = 3;
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        ui.x0 = 0; ui.xf = 1; ui.y0 = 0; ui.yf = 1; ui.z0 = 0; ui.zf = 1; ui.gridSize = 24;
        SimulationSetupUI::UiRegion m;
        m.kind = 0; m.prim = 4; m.meshFile = "mesh_cube.stl";
        m.a[0] = 0.5; m.a[1] = 0.45; m.a[2] = 0.55;
        m.rot[0] = 10; m.rot[1] = 20; m.rot[2] = -5;
        m.scl[0] = 0.123456789012; m.scl[1] = 0.1; m.scl[2] = 0.2; m.uniformScale = false;
        ui.uiRegions.push_back(m);

        check(ui.ApplyScene(), "a mesh region parses as a scene", ui.sceneStatus);
        check(SCENE.regions.size() == 1 && SCENE.regions[0].kind == SceneDesc::PRIM_MESH &&
                  SCENE.regions[0].mesh != nullptr,
              "it becomes a mesh primitive with a built distance field");

        SimulationSetupUI back;
        back.dimension = 3;
        back.LoadSceneTextIntoEditor(ui.SceneText());
        bool same = back.uiRegions.size() == 1 && back.uiRegions[0].prim == 4 &&
                    back.uiRegions[0].meshFile == "mesh_cube.stl" && !back.uiRegions[0].uniformScale;
        double worst = 0.0;
        if (same)
            for (int a = 0; a < 3; a++)
            {
                worst = std::max(worst, std::abs(back.uiRegions[0].a[a] - m.a[a]));
                worst = std::max(worst, std::abs(back.uiRegions[0].rot[a] - m.rot[a]));
                worst = std::max(worst, std::abs(back.uiRegions[0].scl[a] - m.scl[a]) / std::abs(m.scl[a]));
            }
        check(same && worst < 1e-11, "file, centre, rotation and per-axis scale come back",
              "worst relative difference " + std::to_string(worst));
        back.x0 = 0; back.xf = 1; back.y0 = 0; back.yf = 1; back.z0 = 0; back.zf = 1;
        back.gridSize = 24;
        check(back.SceneText() == ui.SceneText(), "re-saving writes the same scene text");

        // The 3D preview's voxels are the solver's mask; every one must lie inside the
        // placed model's bounding box (give or take a cell), and there must be some.
        ui.previewDirty = true;
        ui.BuildPreview();
        const SceneDesc::MeshSDF &ms = *SCENE.regions[0].mesh;
        const double dh = 1.0 / 24;
        bool inside = !ui.voxX.empty();
        for (size_t n = 0; n < ui.voxX.size(); n++)
            inside = inside && ui.voxX[n] > ms.lo[0] - dh && ui.voxX[n] < ms.hi[0] + dh &&
                     ui.voxY[n] > ms.lo[1] - dh && ui.voxY[n] < ms.hi[1] + dh &&
                     ui.voxZ[n] > ms.lo[2] - dh && ui.voxZ[n] < ms.hi[2] + dh;
        check(inside, "3D preview voxelises the model where it was placed",
              std::to_string(ui.voxX.size()) + " surface cells");

        // A fluid region shows in the 3D preview too: its surface cells, inside the box it
        // declares, and none of them where the solid is.
        check(ui.fluX.empty(), "no fluid region, no fluid voxels",
              std::to_string(ui.fluX.size()));
        SimulationSetupUI::UiRegion water;
        water.kind = 1; water.prim = 0;
        water.a[0] = 0.05; water.a[1] = 0.05; water.a[2] = 0.05;
        water.b[0] = 0.35; water.b[1] = 0.40; water.b[2] = 0.95;
        ui.uiRegions.push_back(water);
        check(ui.ApplyScene(), "a fluid box beside the model parses", ui.sceneStatus);
        ui.previewDirty = true;
        ui.BuildPreview();
        bool inBox = !ui.fluX.empty();
        for (size_t n = 0; n < ui.fluX.size(); n++)
            inBox = inBox && ui.fluX[n] > 0.05 - dh && ui.fluX[n] < 0.35 + dh &&
                    ui.fluY[n] > 0.05 - dh && ui.fluY[n] < 0.40 + dh &&
                    ui.fluZ[n] > 0.05 - dh && ui.fluZ[n] < 0.95 + dh;
        check(inBox, "3D preview voxelises the fluid region where it was declared",
              std::to_string(ui.fluX.size()) + " surface fluid cells");
        check(!ui.voxX.empty(), "and still shows the solid", std::to_string(ui.voxX.size()));
        ui.uiRegions.pop_back();

        // An open mesh is reported by Validate, not discovered at startup.
        ui.uiRegions[0].meshFile = "mesh_open.stl";
        check(!ui.ApplyScene() && ui.sceneStatus.find("watertight") != std::string::npos,
              "Validate reports an open mesh", ui.sceneStatus);
    }

    // ---- 4d''. the 3D scene preview on a stretched grid ----
    //          (the slice slider was sized off the BASE Nz and the stretched branch always
    //           cut XY at k = 0, so every slice showed the same picture)
    std::printf("\n== 3D preview slices on a stretched grid ==\n");
    {
        {
            std::ofstream f("sphere3d.scene");
            f << "sampling center\n"
                 "region solid sphere center=(0.5,0.5,0.75) radius=0.12\n";
        }
        SimulationSetupUI ui;
        ui.dimension = 3;
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        ui.x0 = 0; ui.xf = 1; ui.y0 = 0; ui.yf = 1; ui.z0 = 0; ui.zf = 1; ui.gridSize = 20;
        ui.nonUniformGrid = true;
        SimulationSetupUI::RefineSpec rz;
        rz.start = 0.6; rz.end = 0.9; rz.factor = 3;
        ui.refineZ.push_back(rz);
        ui.LoadSceneTextIntoEditor(ReadText("sphere3d.scene"));
        check(SCENE.loaded, "sphere scene loads", SCENE.error);

        std::vector<double> hf, vf, ff;
        ui.previewPlane = 0;
        ui.PreviewAxes(hf, vf, ff);
        const int nz = (int)ff.size() - 1;
        check(ui.PreviewSliceCount() == nz && nz > 20,
              "the slice count is the REFINED Nz", std::to_string(ui.PreviewSliceCount()) +
                  " slices, refined Nz " + std::to_string(nz));

        // The pixel at the domain centre is inside the sphere exactly when that slice's
        // physical z is within the radius of 0.75.
        int wrong = 0, tested = 0;
        std::vector<double> first;
        bool differ = false;
        for (int k = 0; k < nz; k++)
        {
            ui.previewSlice = k;
            ui.previewDirty = true;
            ui.BuildPreview();
            if (ui.previewData.empty()) { wrong++; continue; }
            const double zc = 0.5 * (ff[k] + ff[k + 1]);
            // The first and last slices ARE the domain ring, which is a solid wall (an
            // unspecified face defaults to no-slip), so they say nothing about the sphere.
            if (k == 0 || k == nz - 1) { if (k == 0) first = ui.previewData; continue; }
            const double code = ui.previewData[(size_t)(ui.previewRows / 2) * ui.previewCols + ui.previewCols / 2];
            const double gap = std::abs(zc - 0.75);
            if (gap < 0.10 || gap > 0.14)
            {
                tested++;
                if ((code == 1.0) != (gap < 0.12)) wrong++;
            }
            if (k == 0) first = ui.previewData;
            else if (ui.previewData != first) differ = true;
        }
        check(wrong == 0 && tested > 10, "the solid appears at the right PHYSICAL z",
              std::to_string(wrong) + " wrong of " + std::to_string(tested) + " slices");
        check(differ, "different slices give different pictures");

        // Plane XZ through y = 0.5: the solid's z range in the image is the sphere's.
        ui.previewPlane = 1;
        ui.PreviewAxes(hf, vf, ff);
        int yMid = 0;
        for (int i = 0; i + 1 < (int)ff.size(); i++)
            if (ff[i] <= 0.5 && ff[i + 1] > 0.5) yMid = i;
        ui.previewSlice = yMid;
        ui.previewDirty = true;
        ui.BuildPreview();
        double zlo = 1e9, zhi = -1e9;
        const int col = ui.previewCols / 2;
        // The ring is a solid wall and is pinned to the EDGE of the image, several pixels
        // thick once the axis is refined, so the sphere is measured away from it: a physical
        // margin of one base cell, not a pixel count.
        const double margin = 1.0 / 20;
        for (int r = 1; r < ui.previewRows - 1; r++)
            if (ui.previewData[(size_t)r * ui.previewCols + col] == 1.0)
            {
                // row 0 is the top of the image, i.e. the high end of the vertical axis
                const double z = vf.front() + (ui.previewRows - 1 - r + 0.5) * (vf.back() - vf.front()) / ui.previewRows;
                if (z < vf.front() + margin || z > vf.back() - margin) continue;
                zlo = std::min(zlo, z); zhi = std::max(zhi, z);
            }
        check(std::abs(zlo - 0.63) < 0.05 && std::abs(zhi - 0.87) < 0.05,
              "XZ plane: the sphere spans z in [0.63, 0.87]",
              "drawn [" + std::to_string(zlo) + ", " + std::to_string(zhi) + "]");
    }

    // ---- 4e. a written config must carry a usable scene ----
    //          (checking the box and starting used to reference a file that was never
    //           written, so ConfigReader fell back to the built-in geometry; the scene now
    //           travels in the config)
    std::printf("\n== Start writes the scene ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("FLIP");
        ui.ReconcileSelections();
        SimulationSetupUI::UiRegion water;
        water.kind = 1; water.prim = 0;
        water.a[0] = 0; water.a[1] = 0; water.a[2] = 0;
        water.b[0] = 0.3; water.b[1] = 0.2; water.b[2] = 1.0;
        ui.uiRegions.push_back(water);

        // This is what the Start button now does before writing the config.
        const bool sceneValid = ui.ApplyScene();
        std::string err;
        const bool cfgWritten = ui.WriteFiles(err);
        check(sceneValid && cfgWritten, "Start validates the scene and writes the config",
              ui.sceneStatus + " / " + err);

        // the config carries the scene, so ConfigReader picks it up
        SIMULATION_CONFIG c;
        ConfigReader::loadConfig(c, "simulation_config.txt");
        check(SIMULATION.SolidMaskFunction2D == SCENE_SOLID_MASK_2D,
              "ConfigReader uses the scene rather than falling back");
        check(SCENE.HasFluidRegions(), "the fluid region reached the solver");
    }

    // ---- 4e1. a scene survives a round trip through the editor ----
    //           The writer used to emit u/v/w only for `moving` and `inflow`, so saving a
    //           prescribed-pressure scene from the editor silently deleted its velocity
    //           boundary conditions -- turning a manufactured-solution case into a box of
    //           stationary walls that still ran, and still looked plausible.
    std::printf("\n== scene round trip through the editor ==\n");
    {
        const char *src = "roundtrip_in.scene";
        {
            std::ofstream f(src);
            f << "sampling center\n"
              << "boundary xmin inflow u=\"-cos(x)*sin(y)\" v=\"sin(x)*cos(y)\" p=\"0.5*x\"\n"
              << "boundary xmax inflow u=\"-cos(x)*sin(y)\" v=\"sin(x)*cos(y)\" p=\"0.5*x\"\n"
              << "boundary ymin inflow u=\"-cos(x)*sin(y)\" v=\"sin(x)*cos(y)\" p=\"0.5*x\"\n"
              << "boundary ymax inflow u=\"-cos(x)*sin(y)\" v=\"sin(x)*cos(y)\" p=\"0.5*x\"\n";
        }

        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        ui.LoadSceneTextIntoEditor(ReadText(src));
        check(!ui.sceneStatusIsError, "the scene loads into the editor", ui.sceneStatus);

        check(ui.ApplyScene(), "and validates", ui.sceneStatus);

        // Read what was written, through the real parser. (Scene is deliberately
        // non-copyable -- tinyexpr binds variables to addresses -- so Load resets in place.)
        const bool ok = SCENE.LoadText(ui.SceneText());
        check(ok, "the emitted scene parses", ok ? "" : SCENE.error);
        if (ok)
        {
            int lostVelocity = 0, lostPressure = 0;
            for (int f = 0; f < 4; f++)
            {
                if (SCENE.faces[f].type != BC_INLET) { lostPressure++; continue; }
                if (SCENE.faces[f].u.source.empty() || SCENE.faces[f].u.source == "0") lostVelocity++;
                if (SCENE.faces[f].v.source.empty() || SCENE.faces[f].v.source == "0") lostVelocity++;
                if (!SCENE.faces[f].hasPressure) lostPressure++;
            }
            check(lostVelocity == 0, "an inflow face with a pressure level keeps its velocity",
                  std::to_string(lostVelocity) + " component(s) came back as 0");
            check(lostPressure == 0, "and keeps its pressure level",
                  std::to_string(lostPressure) + " face(s) lost it");
        }
        std::remove(src);

        // A face prescribes velocity or pressure, never both: a pressure outlet takes no velocity at all (it is
        // zero-gradient there), and each kind refuses what it does not prescribe.
        check(!SCENE.LoadText("sampling center\nboundary xmax pressure_outlet u=\"1\" p=\"2\"\n"),
              "a pressure outlet that sets a normal velocity is refused", SCENE.error);
        check(!SCENE.LoadText("sampling center\nboundary xmax pressure_outlet v=\"0.25\" p=\"2\"\n"),
              "and one that sets a tangential velocity", SCENE.error);
        check(!SCENE.LoadText("sampling center\nboundary ymax slip u=\"1\"\n"), "a slip wall with a velocity is refused",
              SCENE.error);
        check(!SCENE.LoadText("sampling center\nboundary ymax noslip p=\"0\"\n"), "a wall with a p is refused",
              SCENE.error);
        check(!SCENE.LoadText("sampling center\nboundary xmax moving u=\"1\"\n"),
              "a moving wall pushed along its own normal is refused", SCENE.error);
        check(SCENE.LoadText("sampling center\nboundary ymax moving u=\"1\" v=\"0\"\n"),
              "a moving wall with its normal written as 0 loads", SCENE.error);
        check(!SCENE.LoadText("sampling center\nboundary xmin inlet u=\"1\"\nboundary xmax outflow\n"
                              "boundary ymax pressure_outlet\n"),
              "an outflow beside a pressure outlet is refused", SCENE.error);
        check(SCENE.LoadText("sampling center\nboundary xmin inflow u=\"1\"\nboundary xmax pressure p=\"1\"\n"
                             "boundary ymax freeslip\n") &&
                  SCENE.faces[SceneDesc::FACE_XMIN].type == BC_INLET &&
                  SCENE.faces[SceneDesc::FACE_XMAX].type == BC_PRESSURE_OUTLET &&
                  SCENE.faces[SceneDesc::FACE_YMAX].type == BC_SLIP,
              "the old spellings still read (inflow, pressure, freeslip)", SCENE.error);

        SimulationSetupUI up;
        up.dimension = 2;
        up.method = MethodFromKey("ADI");
        up.ReconcileSelections();
        up.LoadSceneTextIntoEditor("sampling center\nboundary xmin inlet u=\"1\"\nboundary xmax pressure_outlet p=\"2\"\n"
                                   "boundary ymin pressure_outlet\nboundary ymax slip\n");
        check(!up.sceneStatusIsError, "outlets with and without p load into the editor", up.sceneStatus);
        const std::string text = up.SceneText();
        const bool okP = SCENE.LoadText(text);
        check(okP && SCENE.faces[SceneDesc::FACE_XMAX].type == BC_PRESSURE_OUTLET &&
                  SCENE.faces[SceneDesc::FACE_XMAX].hasPressure && SCENE.faces[SceneDesc::FACE_XMAX].pressure.source == "2" &&
                  SCENE.faces[SceneDesc::FACE_YMIN].type == BC_PRESSURE_OUTLET && !SCENE.faces[SceneDesc::FACE_YMIN].hasPressure &&
                  SCENE.faces[SceneDesc::FACE_YMAX].type == BC_SLIP &&
                  text.find("boundary ymax slip\n") != std::string::npos,
              "and round-trip: p kept where given, none written where absent, the new names written",
              okP ? "" : SCENE.error);
    }

    // ---- 4e1b. rectilinear grid ----
    std::printf("\n== non-uniform grid ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 2; ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        ui.x0 = 0.0; ui.xf = 1.0; ui.y0 = 0.0; ui.yf = 1.0; ui.gridSize = 20;
        ui.nonUniformGrid = true;
        SimulationSetupUI::RefineSpec r; r.start = 0.4; r.end = 0.6; r.factor = 2;
        ui.refineX.push_back(r);

        std::vector<double> fx, fy;
        ui.DerivedAxes(fx, fy);
        check((int)fx.size() - 1 == 26 && (int)fy.size() - 1 == 22,   // the interior plus the ring
              "the panel's cell count matches what MAC builds",
              std::to_string((int)fx.size() - 1) + "x" + std::to_string((int)fy.size() - 1));
        check(std::abs(fx[1] - 0.0) < 1e-15 && std::abs(fx[fx.size() - 2] - 1.0) < 1e-15,
              "the previewed interior faces span the domain exactly");

        std::vector<std::string> problems, notes;
        ui.Validate(problems, notes);
        check(problems.empty(), "a sane refinement is accepted",
              problems.empty() ? "" : problems[0]);

        // ----- several intervals on one axis -----
        SimulationSetupUI::RefineSpec r2; r2.start = 0.05; r2.end = 0.15; r2.factor = 2;
        ui.refineX.push_back(r2);
        ui.DerivedAxes(fx, fy);
        check((int)fx.size() - 1 == 28, "a second interval adds its own cells",
              std::to_string((int)fx.size() - 1));
        bool inc = true;
        for (size_t j = 1; j < fx.size(); j++) if (!(fx[j] > fx[j - 1])) inc = false;
        check(inc, "two intervals still give a strictly increasing axis");

        // Overlap is legal, and must be reported rather than silently resolved.
        ui.refineX[1].start = 0.5; ui.refineX[1].end = 0.7;
        problems.clear(); notes.clear();
        ui.Validate(problems, notes);
        bool sawOverlap = false;
        for (const std::string &n : notes) if (n.find("overlap") != std::string::npos) sawOverlap = true;
        check(problems.empty() && sawOverlap, "overlapping intervals are allowed but flagged");
        ui.refineX.pop_back();

        // ----- ramp direction -----
        // The two ramps are mirrors: whichever end is finest, the other must be coarsest.
        {
            SimulationSetupUI d;
            d.dimension = 2; d.method = MethodFromKey("ADI"); d.ReconcileSelections();
            d.x0 = 0.0; d.xf = 1.0; d.y0 = 0.0; d.yf = 1.0; d.gridSize = 32;
            d.nonUniformGrid = true;
            SimulationSetupUI::RefineSpec rr;
            rr.start = 0.2; rr.end = 0.8; rr.factor = 2; rr.mode = REFINE_RAMP;
            d.refineX.push_back(rr);
            std::vector<double> gx, gy;
            d.DerivedAxes(gx, gy);
            auto sizeAt = [](const std::vector<double> &f, double x)
            {
                for (size_t j = 1; j < f.size(); j++)
                    if (f[j] > x) return f[j] - f[j - 1];
                return f.back() - f[f.size() - 2];
            };
            const double fwdNear = sizeAt(gx, 0.25), fwdFar = sizeAt(gx, 0.75);
            d.refineX[0].mode = REFINE_RAMP_BACK;
            d.DerivedAxes(gx, gy);
            const double bwdNear = sizeAt(gx, 0.25), bwdFar = sizeAt(gx, 0.75);
            char rb[200];
            std::snprintf(rb, sizeof(rb), "ramp> %.5f->%.5f, ramp< %.5f->%.5f",
                          fwdNear, fwdFar, bwdNear, bwdFar);
            check(fwdNear > fwdFar && bwdNear < bwdFar,
                  "the two ramps refine toward opposite ends", rb);
        }

        // ----- graded modes -----
        ui.refineX[0].mode = REFINE_SYMMETRIC;
        ui.DerivedAxes(fx, fy);
        double hmin = 1e30, hmax = 0.0;
        for (size_t j = 2; j + 1 < fx.size(); j++)   // interior cells
        { const double h = fx[j] - fx[j - 1]; hmin = std::min(hmin, h); hmax = std::max(hmax, h); }
        check(std::abs(fx[fx.size() - 2] - 1.0) < 1e-12, "a graded axis still spans the domain exactly");
        check(hmax / hmin > 1.5 && hmax / hmin < 2.2,
              "graded reaches about the requested factor at its finest",
              "coarse/fine = " + std::to_string(hmax / hmin));

        // The point of grading: neighbouring cells step gently instead of jumping by the
        // full factor the way a hard interval does.
        auto worstJump = [](const std::vector<double> &f)
        {
            double w = 1.0;
            for (size_t j = 2; j < f.size(); j++)
            {
                const double a = f[j - 1] - f[j - 2], b = f[j] - f[j - 1];
                if (a > 0 && b > 0) w = std::max(w, std::max(a / b, b / a));
            }
            return w;
        };
        const double gradedJump = worstJump(fx);
        ui.refineX[0].mode = REFINE_HARD;
        ui.DerivedAxes(fx, fy);
        const double hardJump = worstJump(fx);
        check(gradedJump < hardJump - 1e-9,
              "graded has a gentler worst neighbouring jump than hard",
              "graded " + std::to_string(gradedJump) + " vs hard " + std::to_string(hardJump));

        // An interval outside the domain refines nothing and has to be caught.
        ui.refineX[0].start = 5.0; ui.refineX[0].end = 6.0;
        problems.clear(); notes.clear();
        ui.Validate(problems, notes);
        check(!problems.empty(), "an interval outside the domain is refused");

        ui.method = MethodFromKey("FLIP");
        ui.ReconcileSelections();
        check(!ui.nonUniformGrid, "switching to FLIP clears the non-uniform flag");

        // ----- every mode round trips -----
        // Covering only two of them is how ramp_back shipped reloading as hard: LoadFrom
        // silently falls back when it does not recognise a name.
        {
            bool allModes = true; std::string bad;
            for (int m = 0; m < 4; m++)
            {
                SimulationSetupUI wm;
                wm.dimension = 2; wm.method = MethodFromKey("ADI");
                wm.ReconcileSelections();
                wm.nonUniformGrid = true;
                SimulationSetupUI::RefineSpec rm;
                rm.start = 0.2; rm.end = 0.8; rm.factor = 2; rm.mode = m;
                wm.refineX.push_back(rm);
                std::string e;
                wm.WriteFiles(e, "mode_cfg.txt");
                SimulationSetupUI rb;
                rb.LoadFrom("mode_cfg.txt");
                if (rb.refineX.size() != 1 || rb.refineX[0].mode != m)
                {
                    allModes = false;
                    bad = std::string("wrote ") + SimulationSetupUI::ModeName(m) + ", read back " +
                          (rb.refineX.empty() ? "nothing"
                                              : SimulationSetupUI::ModeName(rb.refineX[0].mode));
                }
                std::remove("mode_cfg.txt");
            }
            check(allModes, "every refinement mode survives a save/load round trip", bad);
        }

        // ----- round trip, two intervals with different modes -----
        SimulationSetupUI w;
        w.dimension = 2; w.method = MethodFromKey("ADI"); w.ReconcileSelections();
        w.nonUniformGrid = true;
        SimulationSetupUI::RefineSpec a1; a1.start = 0.1; a1.end = 0.3; a1.factor = 2; a1.mode = REFINE_SYMMETRIC;
        SimulationSetupUI::RefineSpec a2; a2.start = 0.6; a2.end = 0.9; a2.factor = 3; a2.mode = REFINE_HARD;
        w.refineX.push_back(a1); w.refineX.push_back(a2);
        std::string werr;
        check(w.WriteFiles(werr, "nu_cfg.txt"), "a rectilinear config is written", werr);
        SimulationSetupUI back;
        back.LoadFrom("nu_cfg.txt");
        char nb[260];
        std::snprintf(nb, sizeof(nb), "%zu interval(s); #1 [%.3f,%.3f] x%d %s, #2 [%.3f,%.3f] x%d %s",
                      back.refineX.size(),
                      back.refineX.size() > 0 ? back.refineX[0].start : 0.0,
                      back.refineX.size() > 0 ? back.refineX[0].end : 0.0,
                      back.refineX.size() > 0 ? back.refineX[0].factor : 0,
                      back.refineX.size() > 0 ? SimulationSetupUI::ModeName(back.refineX[0].mode) : "-",
                      back.refineX.size() > 1 ? back.refineX[1].start : 0.0,
                      back.refineX.size() > 1 ? back.refineX[1].end : 0.0,
                      back.refineX.size() > 1 ? back.refineX[1].factor : 0,
                      back.refineX.size() > 1 ? SimulationSetupUI::ModeName(back.refineX[1].mode) : "-");
        check(back.nonUniformGrid && back.refineX.size() == 2 &&
                  std::abs(back.refineX[0].start - 0.1) < 1e-9 &&
                  back.refineX[0].factor == 2 && back.refineX[0].mode == REFINE_SYMMETRIC &&
                  std::abs(back.refineX[1].end - 0.9) < 1e-9 &&
                  back.refineX[1].factor == 3 && back.refineX[1].mode == REFINE_HARD,
              "both intervals and their modes survive the round trip", nb);
        std::remove("nu_cfg.txt");
    }

    // ---- 4e1c. the preview on a NON-SQUARE domain ----
    // MAC takes nBase = floor(L/dh), so a domain whose length is not a whole multiple of dh
    // is covered only up to a0 + nBase*dh. The obstacle case is 1.1 x 0.41: the x grid stops
    // at 1.0987. A refined axis has to cover that same span -- pinning it to xf stretched
    // the mesh out from under the scene, which is what put the cylinder and the inlet in the
    // wrong place.
    std::printf("\n== preview on a non-square domain ==\n");
    {
        // Self-contained: the harness runs in an empty scratch directory.
        {
            std::ofstream sf("nonsquare.scene");
            sf << "sampling center\n"
                  "boundary xmin inflow u=\"1.0\" v=\"0\"\n"
                  "boundary xmax pressure_outlet\n"
                  "boundary ymin noslip\n"
                  "boundary ymax noslip\n"
                  "region solid box min=(0.2,0.15,-1) max=(0.41,0.30,1)\n";
        }
        auto setup = [](SimulationSetupUI &ui, bool nu, int mode)
        {
            ui.dimension = 2; ui.method = MethodFromKey("ADI");
            ui.ReconcileSelections();
            ui.x0 = 0.0; ui.xf = 1.1; ui.y0 = 0.0; ui.yf = 0.41;
            ui.gridSize = 128;
            ui.nonUniformGrid = nu;
            ui.refineX.clear(); ui.refineY.clear();
            if (nu)
            {
                SimulationSetupUI::RefineSpec r;
                r.start = 0.15; r.end = 0.30; r.factor = 2; r.mode = mode;
                ui.refineX.push_back(r);
            }
            // BuildPreview asks the live evaluator, so the scene has to be loaded.
            SCENE.Load("nonsquare.scene");
        };

        // Where the solid sits in the preview image, as a fraction of the drawn width.
        auto solidBox = [](SimulationSetupUI &ui, double &fx0, double &fx1, double &fy0, double &fy1)
        {
            ui.BuildPreview();
            fx0 = fy0 = 1e30; fx1 = fy1 = -1e30;
            for (int v = 0; v < ui.previewRows; v++)
                for (int h = 0; h < ui.previewCols; h++)
                {
                    if (ui.previewData[(size_t)v * ui.previewCols + h] != 1.0) continue;
                    // The domain wall ring is solid too, and it is now several PIXELS thick
                    // (the image is sized off the finest cell, not the cell count), so a
                    // one-pixel exclusion is not enough. Skip a physical margin instead --
                    // wide enough for any ring, far narrower than the solid's clearance.
                    const double fu = (h + 0.5) / (double)ui.previewCols;
                    const double fv = (v + 0.5) / (double)ui.previewRows;
                    if (fu < 0.04 || fu > 0.96 || fv < 0.04 || fv > 0.96) continue;
                    const double u = (h + 0.5) / (double)ui.previewCols;
                    const double w = (v + 0.5) / (double)ui.previewRows;
                    fx0 = std::min(fx0, u); fx1 = std::max(fx1, u);
                    fy0 = std::min(fy0, w); fy1 = std::max(fy1, w);
                }
        };

        SimulationSetupUI uu; setup(uu, false, 0);
        double ux0, ux1, uy0, uy1; solidBox(uu, ux0, ux1, uy0, uy1);
        check(ux1 > ux0 && uy1 > uy0, "the uniform preview finds the interior solid",
              "x [" + std::to_string(ux0) + ", " + std::to_string(ux1) + "]");

        const char *names[] = {"hard", "ramp", "graded", "ramp_back"};
        bool allSame = true; std::string worst;
        for (int m = 0; m < 4; m++)
        {
            SimulationSetupUI un; setup(un, true, m);
            double rx0, rx1, ry0, ry1; solidBox(un, rx0, rx1, ry0, ry1);
            // The bound is PHYSICAL, not pixel-based: a finer grid voxelises the cylinder
            // slightly differently, so an edge may move by up to about one base cell. The
            // image resolution is decoupled from the cell count now, so a pixel-based
            // tolerance would tighten as the picture got sharper, which is meaningless.
            const double dhBase = std::min(1.1 - 0.0, 0.41 - 0.0) / 128.0;
            const double tolX = 1.5 * dhBase / (1.1 - 0.0);
            const double tolY = 1.5 * dhBase / (0.41 - 0.0);
            if (std::abs(rx0 - ux0) > tolX || std::abs(rx1 - ux1) > tolX ||
                std::abs(ry0 - uy0) > tolY || std::abs(ry1 - uy1) > tolY)
            {
                allSame = false;
                worst = std::string(names[m]) + " x [" + std::to_string(rx0) + ", " +
                        std::to_string(rx1) + "] vs uniform [" + std::to_string(ux0) + ", " +
                        std::to_string(ux1) + "]";
            }
        }
        check(allSame, "the solid is drawn in the same place at every refinement mode", worst);

        // ----- the boundary ring survives heavy refinement -----
        // The ring is ONE cell thick. Refine hard enough and that cell is thinner than a
        // preview pixel, so a pure point sample loses it -- and with it the only indication
        // of what each face does. Every edge of the image must still carry a ring cell.
        {
            SimulationSetupUI ui;
            setup(ui, true, REFINE_HARD);
            ui.refineX.clear();
            SimulationSetupUI::RefineSpec deep;
            deep.start = 0.0; deep.end = 0.02; deep.factor = 8; deep.mode = REFINE_HARD;
            ui.refineX.push_back(deep);           // hammer the cells right at the inlet
            ui.BuildPreview();

            // xmin is an inflow (code 3), xmax an outflow (2), ymin/ymax noslip walls (1).
            auto px = [&](int v, int h) { return ui.previewData[(size_t)v * ui.previewCols + h]; };
            int leftRing = 0, rightRing = 0, topRing = 0, botRing = 0;
            for (int v = 0; v < ui.previewRows; v++)
            {
                if (px(v, 0) != 0.0) leftRing++;
                if (px(v, ui.previewCols - 1) != 0.0) rightRing++;
            }
            for (int h = 0; h < ui.previewCols; h++)
            {
                if (px(0, h) != 0.0) topRing++;
                if (px(ui.previewRows - 1, h) != 0.0) botRing++;
            }
            char rb[220];
            std::snprintf(rb, sizeof(rb), "left %d/%d, right %d/%d, top %d/%d, bottom %d/%d",
                          leftRing, ui.previewRows, rightRing, ui.previewRows,
                          topRing, ui.previewCols, botRing, ui.previewCols);
            check(leftRing == ui.previewRows && rightRing == ui.previewRows &&
                      topRing == ui.previewCols && botRing == ui.previewCols,
                  "every edge of the preview still shows the boundary ring under 8x refinement", rb);

            // And it must be the INFLOW on xmin specifically, not just "something non-fluid".
            bool sawInflow = false;
            for (int v = 1; v < ui.previewRows - 1; v++)
                if (px(v, 0) == 3.0) sawInflow = true;
            check(sawInflow, "the inflow face keeps its own colour on the refined edge");
        }

        // ----- the ring is drawn the right way up -----
        // A lid on ymax must appear on the ymax edge of the image and NOWHERE on ymin.
        // Getting this backwards draws a moving wall over the bottom solid wall, and a
        // preview that is merely upside down still looks entirely plausible.
        {
            {
                std::ofstream lf("lid.scene");
                lf << "sampling center\n"
                      "boundary xmin noslip\n"
                      "boundary xmax noslip\n"
                      "boundary ymin noslip\n"
                      "boundary ymax moving u=\"1.0\" v=\"0\"\n";
            }
            SimulationSetupUI ui;
            ui.dimension = 2; ui.method = MethodFromKey("ADI");
            ui.ReconcileSelections();
            ui.x0 = 0.0; ui.xf = 1.0; ui.y0 = 0.0; ui.yf = 1.0; ui.gridSize = 64;
            ui.nonUniformGrid = true;
            ui.refineX.clear(); ui.refineY.clear();
            SimulationSetupUI::RefineSpec rr;
            rr.start = 0.0; rr.end = 0.1; rr.factor = 4; rr.mode = REFINE_HARD;
            ui.refineY.push_back(rr);          // refine hard against the BOTTOM wall
            SCENE.Load("lid.scene");
            ui.BuildPreview();

            auto px = [&](int v, int h) { return ui.previewData[(size_t)v * ui.previewCols + h]; };
            // BuildPreview WRITES flipped (previewRows - 1 - v), because ImPlot draws row 0
            // at the top -- so stored row 0 is the TOP of the domain and the last row is the
            // bottom. 5.0 is the moving-wall code, 1.0 a static wall.
            const int topRow = 0, bottomRow = ui.previewRows - 1;
            int movingOnBottom = 0, movingOnTop = 0, wallOnBottom = 0;
            for (int h = 1; h < ui.previewCols - 1; h++)
            {
                if (px(bottomRow, h) == 5.0) movingOnBottom++;
                if (px(bottomRow, h) == 1.0) wallOnBottom++;
                if (px(topRow, h) == 5.0) movingOnTop++;
            }
            char ob[200];
            std::snprintf(ob, sizeof(ob), "bottom: %d moving / %d wall, top: %d moving (of %d)",
                          movingOnBottom, wallOnBottom, movingOnTop, ui.previewCols - 2);
            check(movingOnBottom == 0, "no moving wall is drawn on the ymin edge", ob);
            check(wallOnBottom > 0, "the ymin edge shows a static wall", ob);
            check(movingOnTop > 0, "the moving lid is drawn on the ymax edge", ob);
            std::remove("lid.scene");
        }

        // BuildPreview stands the globals up to ask the evaluator; it must put them back.
        SimulationSetupUI un2; setup(un2, true, REFINE_SYMMETRIC);
        const bool wasNU = SIMULATION.spacing.nonUniform;
        const int wNx = SIMULATION.Nx, wNy = SIMULATION.Ny;
        un2.BuildPreview();
        check(SIMULATION.spacing.nonUniform == wasNU && SIMULATION.Nx == wNx &&
                  SIMULATION.Ny == wNy,
              "BuildPreview restores the globals it borrowed");
        std::remove("nonsquare.scene");
    }

    // ---- 4e2. saved configurations ----
    std::printf("\n== configuration presets ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 3;
        ui.method = MethodFromKey("FLIP");
        ui.ReconcileSelections();
        ui.gridSize = 37;                 // a value nothing else would produce
        ui.viscosity = 4.2e-4;
        ui.particleOptions.contact = 2;
        ui.particleOptions.restitution = 0.55;
        ui.amgxMaxIters = 4321;

        std::string err;
        const bool saved = ui.SavePreset("__harness_tmp", err);
        check(saved, "a preset is written", err);

        ui.RefreshPresets();
        bool listed = false;
        for (const std::string &n : ui.presets)
            if (n == "__harness_tmp") listed = true;
        check(listed, "and it is listed back",
              std::to_string(ui.presets.size()) + " preset(s) found");

        // A fresh panel, loaded from the preset, has to come back with the same values --
        // including the AMGX side, which lives in the PRESSURE block.
        SimulationSetupUI back;
        back.LoadPreset("__harness_tmp");
        char b[256];
        std::snprintf(b, sizeof(b), "dim %d, grid %d, nu %.3e, contact %d, e %.2f, amgx iters %d",
                      back.dimension, back.gridSize, back.viscosity, back.particleOptions.contact,
                      back.particleOptions.restitution, back.amgxMaxIters);
        check(back.dimension == 3 && back.gridSize == 37 &&
                  std::abs(back.viscosity - 4.2e-4) < 1e-12 &&
                  back.particleOptions.contact == 2 &&
                  std::abs(back.particleOptions.restitution - 0.55) < 1e-9 &&
                  back.amgxMaxIters == 4321,
              "loading it restores every field, pressure config included", b);

        // and loading a preset must not disturb the live config the solver reads
        std::remove("configs/__harness_tmp.cfg");
    }

    // ---- 4e3. the shipped presets are all loadable and internally consistent ----
    //          Run from the project root: the presets and the scenes they name live there.
    std::printf("\n== shipped presets ==\n");
    {
        char scratch[4096];
        const bool haveScratch = (getcwd(scratch, sizeof(scratch)) != nullptr);
        if (chdir(PROJECT_ROOT.c_str()) != 0)
            check(false, "can reach the project root", PROJECT_ROOT);
        else
        {
            SimulationSetupUI ui;
            ui.RefreshPresets();
            check(!ui.presets.empty(), "configs/ has presets",
                  std::to_string(ui.presets.size()) + " found");
            for (const std::string &n : ui.presets)
            {
                SimulationSetupUI p;
                p.LoadPreset(n);
                std::vector<std::string> probs, notes;
                p.Validate(probs, notes);
                std::string joined;
                for (const std::string &q : probs) joined += q + " | ";
                check(probs.empty(), "preset '" + n + "' is valid as saved", joined);
            }
            if (haveScratch && chdir(scratch) != 0) std::cerr << "could not return to " << scratch << "\n";
        }
    }

    // ---- 4f. reopening the panel brings the scene back with the config ----
    //          (simulation_config.txt above carries the scene in its SCENE_BEGIN block, so
    //           the scene comes back from the config itself rather than from a path in it)
    std::printf("\n== reopening loads the scene too ==\n");
    {
        SimulationSetupUI ui; // the constructor is what runs when the screen opens
        check(ui.uiRegions.size() == 1 && ui.uiRegions[0].kind == 1,
              "the scene's regions came back with it",
              "regions=" + std::to_string(ui.uiRegions.size()));
        check(!ui.sceneStatusIsError, "no scene error on open", ui.sceneStatus);
    }

    // ---- 4g. the panel refuses scenes that cannot be solved ----
    //          (these blow up on step 0: a pure-Neumann pressure system only has a
    //           solution when the net boundary flux is zero)
    std::printf("\n== unsolvable scenes are blocked ==\n");
    {
        auto problemsFor = [](SimulationSetupUI &u)
        {
            std::vector<std::string> probs, notes;
            u.Validate(probs, notes);
            return probs;
        };
        auto mentions = [](const std::vector<std::string> &v, const char *needle)
        {
            for (const std::string &s : v)
                if (s.find(needle) != std::string::npos) return true;
            return false;
        };

        SimulationSetupUI base;
        base.dimension = 2;
        base.method = MethodFromKey("ADI");
        base.ReconcileSelections();
        base.uiRegions.push_back(SimulationSetupUI::UiRegion());

        // a lid: moving wall, tangential component -> fine
        SimulationSetupUI lid = base;
        lid.uiFaces[3].type = BC_MOVING;                          // ymax moving
        lid.uiFaces[3].u = "1.0";        // u is tangential to ymax
        check(!mentions(problemsFor(lid), "normal to that wall"),
              "a lid (tangential component) is accepted");

        // the same speed on the wall's own normal is never written: a moving wall slides along itself
        SimulationSetupUI piston = base;
        piston.uiFaces[1].type = BC_MOVING;                       // xmax moving
        piston.uiFaces[1].u = "1.0";     // u is NORMAL to xmax
        const std::string pt = piston.SceneText();
        const size_t at = pt.find("boundary xmax moving");
        const std::string line = (at == std::string::npos) ? "" : pt.substr(at, pt.find('\n', at) - at);
        check(!line.empty() && line.find(" u=") == std::string::npos,
              "a moving wall's normal velocity is not written", line);

        // inflow with nowhere to go -> blocked
        SimulationSetupUI noOut = base;
        noOut.uiFaces[0].type = BC_INLET;                         // xmin inlet
        noOut.uiFaces[0].u = "1.0";
        check(mentions(problemsFor(noOut), "no way out"), "an inlet with no way out is blocked");

        // a prescribed-pressure face is also a way out: it pins the pressure, so the system
        // is no longer pure Neumann and net flux through the domain is fine.
        SimulationSetupUI pressureOut = noOut;
        pressureOut.uiFaces[1].type = BC_PRESSURE_OUTLET;         // xmax, p = 0.5
        pressureOut.uiFaces[1].p = "0.5";
        pressureOut.uiFaces[1].hasPressure = true;
        check(!mentions(problemsFor(pressureOut), "no way out"), "an inlet with a pressure outlet is accepted");

        // a continuity outflow is a way out too, but pins no pressure: it needs the compatibility condition
        SimulationSetupUI outflow = noOut;
        outflow.uiFaces[1].type = BC_OUTFLOW;
        PRESSURE_SETTINGS.needsCompatibilityCondition = false;
        const std::vector<std::string> pOut = problemsFor(outflow);
        check(!mentions(pOut, "no way out") && mentions(pOut, "pure Neumann"),
              "an inlet with an outflow is a way out, and asks for the compatibility condition");
        PRESSURE_SETTINGS.needsCompatibilityCondition = true;
        check(!mentions(problemsFor(outflow), "pure Neumann"), "which it accepts once that is on");
        PRESSURE_SETTINGS.needsCompatibilityCondition = false;

        // an outflow beside a pressure outlet is refused
        SimulationSetupUI mixed = outflow;
        mixed.uiFaces[3].type = BC_PRESSURE_OUTLET;
        check(mentions(problemsFor(mixed), "Use one or the other"), "an outflow beside a pressure outlet is refused");
    }

    // ---- 5. AMGX accepts every solver string the panel can emit (GPU build only) ----
#ifdef NVIDIA_GPU
    std::printf("\n== AMGX config strings ==\n");
    {
        AMGX_initialize();
        int rejected = 0, total = 0;
        for (int solver = 0; solver < 5; solver++)
            for (int precon = 0; precon < 2; precon++)
                for (int alg = 0; alg < 2; alg++)
                    for (int cycle = 0; cycle < 3; cycle++)
                        for (int norm = 0; norm < 3; norm++)
                            for (int conv = 0; conv < 3; conv++)
                            {
                                // Explicit state: the constructor reads whatever config is in the
                                // working directory, so an earlier sub-test would leak in here.
                                SimulationSetupUI ui;
                                ui.useGPU = true;
                                ui.amgxSolver = solver;
                                ui.amgxUsePrecon = (precon == 1);
                                ui.amgxPreconAlg = alg;
                                ui.amgxCycle = cycle;
                                ui.amgxNorm = norm;
                                ui.amgxConvergence = conv;
                                ui.amgxMaxIters = 1000;
                                ui.amgxTolerance = 1e-12;
                                ui.amgxPresweeps = 1;
                                ui.amgxPostsweeps = 1;
                                ui.amgxPreIters = 1;
                                total++;

                                AMGX_config_handle cfg = nullptr;
                                if (AMGX_config_create(&cfg, AmgxConfigString(ui.PressureBlockText()).c_str()) != AMGX_RC_OK)
                                {
                                    if (rejected < 6)
                                        std::printf("     rejected: solver=%d precon=%d alg=%d cycle=%d norm=%d conv=%d\n",
                                                    solver, precon, alg, cycle, norm, conv);
                                    rejected++;
                                }
                                if (cfg) AMGX_config_destroy(cfg);
                            }
        check(rejected == 0,
              "AMGX parses every option combination the panel offers (" + std::to_string(total) + ")",
              rejected ? std::to_string(rejected) + " rejected" : "");

        // Parsing is not enough. A config can parse and still blow up when the hierarchy is
        // built: `algorithm=AGGREGATION` with no `selector` inherits the default PMIS, which
        // is registered only for the classical path, and AMGX_solver_setup throws
        // "SelectorFactory 'PMIS' has not been registered" -- reported later, and
        // confusingly, as "Incorrect C API mode" from the next call. So actually stand a
        // solver up on a small SPD matrix for the combinations that decide the coarsening.
        int setupFailed = 0, setupTotal = 0;
        {
            const int n = 64;                       // 1D Laplacian: SPD, valid for every solver
            std::vector<int> row(n + 1, 0), col;
            std::vector<double> val;
            for (int i = 0; i < n; i++)
            {
                if (i > 0)     { col.push_back(i - 1); val.push_back(-1.0); }
                col.push_back(i); val.push_back(2.0);
                if (i < n - 1) { col.push_back(i + 1); val.push_back(-1.0); }
                row[i + 1] = (int)col.size();
            }

            for (int solver = 0; solver < 5; solver++)
                for (int precon = 0; precon < 2; precon++)
                    for (int alg = 0; alg < 2; alg++)
                    {
                        SimulationSetupUI ui;
                        ui.useGPU = true;
                        ui.amgxSolver = solver;
                        ui.amgxUsePrecon = (precon == 1);
                        ui.amgxPreconAlg = alg;
                        ui.amgxMaxIters = 10;
                        ui.amgxTolerance = 1e-8;
                        ui.amgxPresweeps = 1; ui.amgxPostsweeps = 1; ui.amgxPreIters = 1;
                        setupTotal++;

                        AMGX_config_handle cfg = nullptr;
                        AMGX_resources_handle rsrc = nullptr;
                        AMGX_matrix_handle A = nullptr;
                        AMGX_solver_handle slv = nullptr;
                        bool ok = (AMGX_config_create(&cfg, AmgxConfigString(ui.PressureBlockText()).c_str()) == AMGX_RC_OK);
                        if (ok) ok = (AMGX_resources_create_simple(&rsrc, cfg) == AMGX_RC_OK);
                        if (ok) ok = (AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI) == AMGX_RC_OK);
                        if (ok) ok = (AMGX_solver_create(&slv, rsrc, AMGX_mode_dDDI, cfg) == AMGX_RC_OK);
                        if (ok) ok = (AMGX_matrix_upload_all(A, n, (int)col.size(), 1, 1,
                                                             row.data(), col.data(), val.data(), nullptr) == AMGX_RC_OK);
                        if (ok) ok = (AMGX_solver_setup(slv, A) == AMGX_RC_OK);
                        if (!ok)
                        {
                            if (setupFailed < 6)
                                std::printf("     setup failed: solver=%d precon=%d alg=%d\n",
                                            solver, precon, alg);
                            setupFailed++;
                        }
                        if (slv) AMGX_solver_destroy(slv);
                        if (A) AMGX_matrix_destroy(A);
                        if (rsrc) AMGX_resources_destroy(rsrc);
                        if (cfg) AMGX_config_destroy(cfg);
                    }
        }
        check(setupFailed == 0,
              "and builds a working hierarchy for each (" + std::to_string(setupTotal) + ")",
              setupFailed ? std::to_string(setupFailed) + " failed at solver_setup" : "");
        AMGX_finalize();
    }
#else
    std::printf("\n== AMGX config strings: skipped, built without NVIDIA_GPU ==\n");
#endif

    // ---- long expressions must survive the editor ----
    //
    // UiBoundary held its expressions in char[128]. The Ethier-Steinman scene has a
    // 500-character pressure expression, so loading it into the editor truncated every field
    // at 127 characters and Save wrote the truncation back -- the file on disk was destroyed
    // by opening it. This pins that a long expression round-trips intact, and it is a real
    // guard rather than a style check: the failure was silent and permanent.
    {
        std::string longExpr = "0.0";
        for (int k = 0; k < 40; k++)
            longExpr += "+0.001*sin(" + std::to_string(k + 1) + ".0*x)*cos(" +
                        std::to_string(k + 1) + ".0*y)";
        check(longExpr.size() > 400, "the probe expression is longer than the old buffer",
              std::to_string(longExpr.size()) + " chars");

        SimulationSetupUI ui;
        ui.dimension = 3;
        ui.sceneHasInitial = true;
        ui.initU = longExpr; ui.initV = "0"; ui.initW = "0";
        ui.forceU = "0"; ui.forceV = longExpr; ui.forceW = "0";
        ui.uiFaces[0].type = BC_PRESSURE_OUTLET;
        ui.uiFaces[0].hasPressure = true;
        ui.uiFaces[0].p = longExpr;

        const std::string text = ui.SceneText();
        check(text.find(longExpr) != std::string::npos,
              "SceneText writes a long boundary/initial/force expression in full");
        check(text.find("initial ") != std::string::npos,
              "SceneText emits the `initial` line at all (it used to be dropped)");
        check(text.find("force ") != std::string::npos,
              "SceneText emits the `force` line at all (it used to be dropped)");

        // and it comes back through the parser unchanged
        const std::string tmp = "ui_longexpr.scene";
        { std::ofstream f(tmp); f << text; }
        SimulationSetupUI ui2;
        ui2.dimension = 3;
        ui2.LoadSceneTextIntoEditor(ReadText(tmp));
        check(ui2.sceneHasInitial && ui2.initU == longExpr,
              "and the editor loads it back without truncating",
              "got " + std::to_string(ui2.initU.size()) + " of " +
              std::to_string(longExpr.size()) + " chars");
        check(ui2.forceV == longExpr,
              "force survives the round trip too",
              "got " + std::to_string(ui2.forceV.size()) + " chars");
        std::remove(tmp.c_str());
    }

    std::printf("\n== transported scalars round trip through the editor ==\n");
    {
        // A NACA region must survive the editor: thickness (in its name), chord, pivot, rotation, span, trailing edge.
        const std::string in = "sampling center\n"
                               "region solid naca0015 center=(1,0.5,0.25) chord=0.8 rotate=(0,0,-10) pivot=0.25 span=0.4 te=closed\n"
                               "region solid naca0012 center=(2,0.5,0) chord=1 rotate=(0,0,0)\n";
        SimulationSetupUI ui;
        ui.dimension = 3;
        ui.LoadSceneTextIntoEditor(in);
        const bool fields = ui.uiRegions.size() == 2 && ui.uiRegions[0].prim == 5 && ui.uiRegions[0].thicknessPercent == 15 &&
                            ui.uiRegions[0].chord == 0.8 && ui.uiRegions[0].pivot == 0.25 && ui.uiRegions[0].span == 0.4 &&
                            ui.uiRegions[0].closedTE && ui.uiRegions[0].rot[2] == -10 && ui.uiRegions[0].a[2] == 0.25 &&
                            ui.uiRegions[1].thicknessPercent == 12 && !ui.uiRegions[1].closedTE && ui.uiRegions[1].pivot == 0.5;
        check(fields, "a naca00XX region loads into the editor with every parameter", "");
        const std::string text = ui.SceneText();
        check(text.find("naca0015") != std::string::npos && text.find("te=closed") != std::string::npos &&
                  text.find("pivot=0.25") != std::string::npos && text.find("span=0.4") != std::string::npos,
              "and is written back as naca0015 with its keys", text);
        SimulationSetupUI back;
        back.dimension = 3;
        back.LoadSceneTextIntoEditor(text);
        check(back.SceneText() == text, "a second round trip writes the same text", "");
    }
    {
        // A scalar declaration and a face that pins it must survive Save -> Load -> Save.
        // This is the same failure mode as the prescribed-pressure velocity and the `force`
        // line: the editor drops what it does not know about, and Save writes the loss back.
        {
            std::ofstream f("ui_scalar.scene");
            f << "sampling center\n"
                 "scalar smoke scheme=central\n"
                 "scalar T scheme=upwind\n"
                 "scalar k scheme=upwind form=advective epsilon=1.5e-05 gamma=0.09 source=\"0.5*exp(0-((x-0.3)/0.1)^2)\"\n"
                 "boundary xmin inflow u=\"1\" smoke=\"1.5\" smoke_clamp_negative=1 T=\"300\"\n"
                 "boundary xmax pressure_outlet\nboundary ymin noslip\nboundary ymax noslip\n";
        }
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.LoadSceneTextIntoEditor(ReadText("ui_scalar.scene"));
        check(ui.uiScalars.size() == 3, "all three scalars load",
              std::to_string(ui.uiScalars.size()));
        check(ui.uiScalars.size() == 3 && ui.uiScalars[0].name == "smoke" &&
              ui.uiScalars[0].scheme == 0 && ui.uiScalars[1].scheme == 1,
              "with their names and schemes");
        check(ui.uiScalars.size() == 3 && ui.uiScalars[2].form == 1 && ui.uiScalars[2].epsilon == 1.5e-05 &&
              ui.uiScalars[2].gamma == 0.09 && ui.uiScalars[2].source == "0.5*exp(0-((x-0.3)/0.1)^2)" &&
              ui.uiScalars[0].form == 0 && ui.uiScalars[0].epsilon == 0.0 && ui.uiScalars[0].source.empty(),
              "with form, epsilon, gamma and source, and the defaults where none were given");

        const std::string text = ui.SceneText();
        check(text.find("scalar smoke scheme=central\n") != std::string::npos &&
              text.find("scalar T scheme=upwind\n") != std::string::npos,
              "and are written back as declarations, a plain one with nothing added");
        check(text.find("scalar k scheme=upwind form=advective epsilon=1.5e-05 gamma=0.09 "
                        "source=\"0.5*exp(0-((x-0.3)/0.1)^2)\"\n") != std::string::npos,
              "the transport parameters are written back");
        check(text.find("smoke=\"1.5\"") != std::string::npos &&
              text.find("T=\"300\"") != std::string::npos,
              "with the face values that named them");
        check(text.find("smoke_clamp_negative=1") != std::string::npos &&
              text.find("T_clamp_negative") == std::string::npos,
              "and the clamp on the one field that had it");

        // And the round trip has to be stable, not merely non-empty.
        {
            std::ofstream f("ui_scalar2.scene");
            f << text;
        }
        SimulationSetupUI ui2;
        ui2.dimension = 2;
        ui2.LoadSceneTextIntoEditor(ReadText("ui_scalar2.scene"));
        check(ui2.uiScalars.size() == 3 && ui2.SceneText() == text,
              "a second round trip is byte for byte the same");
        std::remove("ui_scalar.scene");
        std::remove("ui_scalar2.scene");

        // solver=none: written as a bare declaration, read back as none, and its face value kept.
        SimulationSetupUI ui3;
        ui3.dimension = 2;
        ui3.LoadSceneTextIntoEditor("sampling center\nscalar k solver=none epsilon=0.002 source=\"x*y\"\n"
                                    "boundary xmin inflow u=\"1\" k=\"2\"\nboundary xmax pressure_outlet\n"
                                    "boundary ymin noslip\nboundary ymax noslip\n");
        const std::string text3 = ui3.SceneText();
        check(ui3.uiScalars.size() == 1 && ui3.uiScalars[0].solver == 1 && ui3.uiScalars[0].epsilon == 0.002 &&
              ui3.uiScalars[0].source == "x*y", "solver=none loads into the editor with its parameters");
        check(text3.find("scalar k solver=none epsilon=0.002 source=\"x*y\"\n") != std::string::npos && text3.find("k=\"2\"") != std::string::npos,
              "and is written back with its parameters and face value", text3);
    }

    std::printf("\n== the scene travels INSIDE the config ==\n");
    {
        // The point of the block: a saved configuration is ONE self-contained file. Before
        // this, a .cfg carried SCENE_FILE and several presets pointed at the same
        // scenes/*.scene, so saving any one of them rewrote what the others read.
        {
            std::ofstream f("emb_in.scene");
            f << "sampling center\n"
                 "scalar smoke scheme=upwind form=advective epsilon=0.002 gamma=0.25 source=\"x*y\"\n"
                 "boundary xmin inflow u=\"1.5 - (1.5/(0.205*0.205))*(y-0.205)^2\" smoke=\"1\"\n"
                 "boundary xmax pressure_outlet\nboundary ymin noslip\nboundary ymax noslip\n"
                 "region solid cylinder axis=z center=(0.2,0.2) radius=0.05\n";
        }
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.LoadSceneTextIntoEditor(ReadText("emb_in.scene"));
        const std::string wanted = ui.SceneText();

        std::string err;
        check(ui.WriteFiles(err, "emb_cfg.txt"),
              "a config with a scene is written", err);

        std::string cfg;
        {
            std::ifstream in("emb_cfg.txt");
            std::stringstream ss; ss << in.rdbuf(); cfg = ss.str();
        }
        check(cfg.find("SCENE_BEGIN") != std::string::npos &&
              cfg.find("SCENE_END") != std::string::npos,
              "and carries the scene between the markers");
        check(cfg.find("SCENE_FILE") == std::string::npos,
              "and no longer points at a separate file");
        check(cfg.find("scalar smoke scheme=upwind") != std::string::npos,
              "the scalar declarations are in the config");

        // Back into a fresh editor. The block has to win the same way ConfigReader makes it
        // win, and the scene has to come back byte for byte -- the expression carries an
        // '=' and parentheses, which the KEY = value loop would have split and truncated.
        SimulationSetupUI back;
        back.LoadFrom("emb_cfg.txt");
        check(back.uiScalars.size() == 1 && back.uiScalars[0].name == "smoke" &&
              back.uiScalars[0].scheme == 1, "with the scalar and its scheme");
        check(back.uiRegions.size() == 1, "and the region",
              std::to_string(back.uiRegions.size()));
        check(back.SceneText() == wanted, "and the scene text is unchanged");

        // What the SOLVER sees, which is the claim that matters: the real ConfigReader path.
        SCENE.Reset();
        check(ConfigReader::loadFromFile("emb_cfg.txt"), "ConfigReader reads it");
        check(SCENE.LoadText(ConfigReader::sceneBlock), "and the block it collected parses",
              SCENE.error);
        check(SCENE.ScalarIndex("smoke") == 0,
              "with the scalar registered from the config alone");

        // And the fields exist on the grids, which is what the viewer's dropdown lists. The scene
        // used to parse the declaration while nothing registered the field.
        {
            SIMULATION_CONFIG c;
            ConfigReader::loadConfig(c, "emb_cfg.txt");
            const int hs = SIMULATION.GRID_SOL->ScalarHandle("smoke");
            const int ha = SIMULATION.GRID_ANT->ScalarHandle("smoke");
            check(hs == 0 && ha == 0 && SIMULATION.GRID_SOL->ScalarCount() == 1,
                  "loadConfig registers the scene's scalar on GRID_SOL and GRID_ANT",
                  std::to_string(hs) + " / " + std::to_string(ha));
            check(hs == 0 && SIMULATION.GRID_SOL->ScalarTransportOf(0).scheme == TransportScheme::Upwind,
                  "with its declared scheme");
            if (hs == 0 && ha == 0)
            {
                const ScalarTransport &ts = SIMULATION.GRID_SOL->ScalarTransportOf(0);
                check(ts.form == TransportForm::Advective && ts.epsilon == 0.002 && ts.gamma == 0.25 &&
                      SameTransport(ts, SIMULATION.GRID_ANT->ScalarTransportOf(0)),
                      "and its form, epsilon and gamma, the same on both grids");
                const SceneDesc::Expr *src = SCENE.ScalarSource(SCENE.ScalarIndex("smoke"));
                check(src != nullptr && src->Eval(0.5, 0.4, 0.0, 0.0) == 0.5 * 0.4,
                      "and the scene carries its source expression");
            }
        }

        // SCENE_FILE still has to work: 25 harnesses and every hand-written config use it.
        {
            std::ofstream f("emb_ptr.txt");
            f << "DIMENSION = 2\nTYPE = ADI\nSCENE_FILE = emb_in.scene\n";
        }
        SCENE.Reset();
        check(ConfigReader::loadFromFile("emb_ptr.txt"), "a config with SCENE_FILE reads");
        check(ConfigReader::sceneBlock.empty(), "and collects no block");
        check(ConfigReader::getString("SCENE_FILE", "") == "emb_in.scene",
              "with the path intact");

        // A config with NEITHER is a closed box, and must not inherit the last scene.
        {
            std::ofstream f("emb_none.txt");
            f << "DIMENSION = 2\nTYPE = ADI\n";
        }
        back.LoadFrom("emb_none.txt");
        {
            bool walls = back.uiRegions.empty() && back.uiScalars.empty();
            for (int f = 0; f < 6; f++) walls = walls && back.uiFaces[f].type == BC_NOSLIP;
            check(walls, "a config with no scene at all opens as a closed box, not the last scene");
        }
        {
            // A complete config from the panel, just without a scene.
            SimulationSetupUI plain;
            plain.dimension = 2;
            plain.method = MethodFromKey("ADI");
            plain.ReconcileSelections();
            std::string perr;
            check(plain.WriteFiles(perr, "emb_plain_cfg.txt"),
                  "a config with no scene is written", perr);
            SIMULATION_CONFIG c;
            ConfigReader::loadConfig(c, "emb_plain_cfg.txt");
            check(SIMULATION.GRID_SOL->ScalarCount() == 0 && SIMULATION.GRID_ANT->ScalarCount() == 0,
                  "and a config with no scene does not inherit the last one's fields",
                  std::to_string(SIMULATION.GRID_SOL->ScalarCount()));
            std::remove("emb_plain_cfg.txt");
        }

        std::remove("emb_in.scene");
        std::remove("emb_cfg.txt");
        std::remove("emb_ptr.txt");
        std::remove("emb_none.txt");
    }

    std::printf("\n== the current file: Start runs what was last loaded, and exports under its name ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        check(ui.configPath == "simulation_config.txt",
              "a fresh panel works on the config in the working directory", ui.configPath);
        check(!ui.HasPreset(), "and holds Start until the setup has a name");

        std::string err;
        check(ui.SavePreset("current_probe", err), "saving a preset", err);
        check(ui.configPath == "configs/current_probe.cfg", "makes it the current file", ui.configPath);
        std::ifstream noPcfg("configs/current_probe.pcfg");
        check(!noPcfg.is_open(), "and writes no separate pressure file");

        SimulationSetupUI other;
        other.LoadPreset("current_probe");
        check(other.configPath == "configs/current_probe.cfg",
              "and loading a preset makes it current too", other.configPath);

        // What Start writes and what the solver then reads are the same file.
        other.gridSize = 37;
        check(other.WriteFiles(err, other.configPath), "Start writes the current file", err);
        SimulationSetupUI back;
        back.LoadFrom("configs/current_probe.cfg");
        check(back.gridSize == 37, "so the edit lands in the preset, not in simulation_config.txt",
              std::to_string(back.gridSize));

        SIMULATION_CONFIG c;
        ConfigReader::loadConfig(c, other.configPath);
        check(SIMULATION.CONFIG_NAME == "current_probe", "the solver names the run after the file",
              SIMULATION.CONFIG_NAME);
        check(PRESSURE_SETTINGS.text.find("eigen_solver=") != std::string::npos,
              "and reads the pressure settings from its block", PRESSURE_SETTINGS.text);
        const std::string dir = ExportDirectory("VTK");
        check(dir.find("/current_probe/") != std::string::npos && dir.find("CAVITY") == std::string::npos,
              "exports go under the config's name, not the level's", dir);

        ConfigReader::loadConfig(c, "simulation_config.txt");
        check(SIMULATION.CONFIG_NAME == "simulation_config" && !PRESSURE_SETTINGS.text.empty(),
              "the working-directory config carries its own block", SIMULATION.CONFIG_NAME);

        // Remembered across launches -- but only from the panel's own buttons.
        std::remove(SimulationSetupUI::LastPresetFile().c_str());
        SimulationSetupUI loader;
        loader.LoadPreset("current_probe");
        std::ifstream noFile(SimulationSetupUI::LastPresetFile());
        check(!noFile.is_open(), "loading a preset from code does not overwrite the remembered one");

        ui.RememberPreset("current_probe");
        SimulationSetupUI relaunched;   // the constructor is what a launch runs
        check(relaunched.HasPreset() && relaunched.configPath == "configs/current_probe.cfg",
              "a relaunch opens the last preset picked", relaunched.configPath);
        check(relaunched.presetIdx >= 0 && relaunched.presets[(size_t)relaunched.presetIdx] == "current_probe",
              "with it selected in the combo");

        std::remove("configs/current_probe.cfg");
        SimulationSetupUI gone;
        check(!gone.HasPreset(), "a remembered preset that no longer exists falls back, and Start is held",
              gone.configPath);
        ui.ForgetPreset();
    }

    // ---- the body force is ONE field, shown and saved for every method ----
    //      (it used to be two: Fu/Fv/Fw, hidden for ADI yet written and applied, and the scene's
    //       `force` line; and a config without the keys kept the previous config's gravity)
    std::printf("\n== one body force ==\n");
    {
        auto mentions = [](const std::vector<std::string> &v, const char *needle)
        {
            for (const std::string &s : v)
                if (s.find(needle) != std::string::npos) return true;
            return false;
        };
        // FLIP with gravity, switched to ADI: the force is still there, and the file says so.
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("FLIP");
        ui.ReconcileSelections();
        ui.forceV = "-9.8";
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        std::string err;
        check(ui.WriteFiles(err, "force_const_cfg.txt"), "an ADI config with gravity is written", err);
        const std::string constText = ReadText("force_const_cfg.txt");
        check(constText.find("Fv = -9.8") != std::string::npos, "a constant force is written as Fv");
        check(constText.find("force u=") == std::string::npos, "and no scene `force` line");
        SIMULATION_CONFIG c;
        ConfigReader::loadConfig(c, "force_const_cfg.txt");
        check(std::abs(SIMULATION.f.v + 9.8) < 1e-12 && !(SCENE.loaded && SCENE.hasForce),
              "ConfigReader applies exactly that force, once");

        // A config with no Fu/Fv/Fw after one with gravity reads as NO force.
        {
            std::ofstream f("force_none_cfg.txt");
            f << "DIMENSION = 2\nTYPE = ADI\nGRID_SIZE = 16\n";
        }
        SimulationSetupUI back;
        back.LoadFrom("force_const_cfg.txt");
        check(back.forceV == "-9.8", "the panel reloads Fv = -9.8", back.forceV);
        back.LoadFrom("force_none_cfg.txt");
        check(back.BodyForceZero(), "a config without Fu/Fv/Fw resets the force to zero",
              back.forceU + " " + back.forceV + " " + back.forceW);

        // An expression goes to the scene; Fu/Fv/Fw are written as 0 so it is not applied twice.
        SimulationSetupUI ex;
        ex.dimension = 2;
        ex.method = MethodFromKey("ADI");
        ex.ReconcileSelections();
        ex.uiFaces[3].type = BC_MOVING; ex.uiFaces[3].u = "1.0";
        ex.forceU = "sin(6.2831853*y)";
        ex.forceV = "0";
        check(ex.ApplyScene(), "a scene with an expression force validates", ex.sceneStatus);
        check(ex.forceU == "sin(6.2831853*y)", "Validate leaves the field as typed", ex.forceU);
        check(ex.WriteFiles(err, "force_expr_cfg.txt"), "the expression config is written", err);
        const std::string exText = ReadText("force_expr_cfg.txt");
        check(exText.find("Fu = 0") != std::string::npos && exText.find("force u=\"sin(6.2831853*y)\"") != std::string::npos,
              "an expression is written as the scene's force, with Fu/Fv/Fw at 0");
        ConfigReader::loadConfig(c, "force_expr_cfg.txt");
        check(SIMULATION.f.u == 0.0 && SIMULATION.f.v == 0.0 && SCENE.hasForce &&
                  SCENE.forceU.source == "sin(6.2831853*y)",
              "ConfigReader: the scene carries the expression, the scalars are zero");
        SimulationSetupUI exBack;
        exBack.LoadFrom("force_expr_cfg.txt");
        check(exBack.forceU == "sin(6.2831853*y)" && exBack.forceV == "0",
              "the panel reloads the expression into the same field", exBack.forceU + " / " + exBack.forceV);

        // An older config carrying both shows their sum, which is what the solver applies.
        {
            std::ofstream f("force_both_cfg.txt");
            f << "DIMENSION = 2\nTYPE = ADI\nGRID_SIZE = 16\nFv = -9.8\n";
            f << SCENE_BLOCK_BEGIN << "\nboundary ymax moving u=\"1\"\nforce u=\"0\" v=\"sin(x)\"\n" << SCENE_BLOCK_END << "\n";
        }
        SimulationSetupUI both;
        both.LoadFrom("force_both_cfg.txt");
        check(both.forceV == "(sin(x)) + -9.8" && both.forceU == "0",
              "Fv plus a scene force load as their sum", both.forceU + " / " + both.forceV);

        // Refusal: SPH takes constants only.
        std::vector<std::string> probs, notes;
        SimulationSetupUI sph = ex;
        sph.method = MethodFromKey("SPH");
        sph.ReconcileSelections();
        sph.forceU = "sin(y)";
        sph.Validate(probs, notes);
        check(mentions(probs, "takes a constant body force"), "SPH refuses an expression force");
        for (const char *t : {"force_const_cfg.txt", "force_none_cfg.txt", "force_expr_cfg.txt", "force_both_cfg.txt"})
            std::remove(t);
    }

    // ---- convection scheme: written, reloaded, read by the backend, reset by a config without it ----
    {
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        ADI_SETTINGS.convectionScheme = CONVECTION_UPWIND;
        std::string err;
        check(ui.WriteFiles(err, "conv_cfg.txt"), "an upwind config is written", err);
        ADI_SETTINGS.convectionScheme = CONVECTION_CENTRAL;
        SimulationSetupUI back;
        back.LoadFrom("conv_cfg.txt");
        check(ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND, "the panel reloads CONVECTION_SCHEME = UPWIND");
        SIMULATION_CONFIG c;
        ADI_SETTINGS.convectionScheme = CONVECTION_CENTRAL;
        ConfigReader::loadConfig(c, "conv_cfg.txt");
        check(ADI_SETTINGS.convectionScheme == CONVECTION_UPWIND, "ConfigReader reads UPWIND");

        ADI_SETTINGS.convectionScheme = CONVECTION_CENTRAL;
        check(ui.WriteFiles(err, "conv_c_cfg.txt"), "a central config is written", err);
        ADI_SETTINGS.convectionScheme = CONVECTION_UPWIND;
        back.LoadFrom("conv_c_cfg.txt");
        check(ADI_SETTINGS.convectionScheme == CONVECTION_CENTRAL, "loading a central config after an upwind one resets it");
        ADI_SETTINGS.convectionScheme = CONVECTION_UPWIND;
        ConfigReader::loadConfig(c, "conv_c_cfg.txt");
        check(ADI_SETTINGS.convectionScheme == CONVECTION_CENTRAL, "ConfigReader reads CENTRAL");
    }

    // ---- method settings blocks: one per registered method, read through the table ----
    std::printf("\n== method settings blocks ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("SPH");
        ui.ReconcileSelections();
        SPH_SETTINGS.soundSpeed = 123.5;
        SPH_SETTINGS.delta = 0.0;
        FLIP_SETTINGS.alpha = 0.61;
        ADI_SETTINGS.projectionScheme = PROJECTION_INCREMENTAL;
        std::string err;
        check(ui.WriteFiles(err, "blocks_cfg.txt"), "a config with every block is written", err);

        std::ifstream in("blocks_cfg.txt");
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        bool everyBlock = true;
        for (int m = 0; m < METHOD_COUNT; m++)
            if (METHODS[m].settingCount > 0)
                everyBlock = everyBlock && text.find("\n" + MethodBlockBegin(METHODS[m]) + "\n") != std::string::npos &&
                             text.find("\n" + MethodBlockEnd(METHODS[m]) + "\n") != std::string::npos;
        check(everyBlock, "every registered method's block is written, not only the active one's");
        check(text.find("\nALPHA = ") == text.find("\nALPHA = ", text.find("FLIP_BEGIN")) &&
              text.find("\nALPHA = ") != std::string::npos,
              "a method's keys are written inside its block, not at top level");

        SPH_SETTINGS = SPHSettings(); FLIP_SETTINGS = FLIPSettings(); ADI_SETTINGS = ADISettings();
        SimulationSetupUI back;
        back.LoadFrom("blocks_cfg.txt");
        check(SPH_SETTINGS.soundSpeed == 123.5 && SPH_SETTINGS.delta == 0.0, "the panel reads the SPH block");
        check(FLIP_SETTINGS.alpha == 0.61, "and the FLIP block of an SPH config");
        check(ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL, "and the ADI block");

        SPH_SETTINGS = SPHSettings(); FLIP_SETTINGS = FLIPSettings(); ADI_SETTINGS = ADISettings();
        SIMULATION_CONFIG c;
        ConfigReader::loadConfig(c, "blocks_cfg.txt");
        check(SPH_SETTINGS.soundSpeed == 123.5 && SPH_SETTINGS.delta == 0.0 && FLIP_SETTINGS.alpha == 0.61 &&
              ADI_SETTINGS.projectionScheme == PROJECTION_INCREMENTAL, "ConfigReader reads every block");
        check(ActiveMethodSetting("SPH_SOUND_SPEED", -1.0) == 123.5 &&
              ActiveMethodSetting("PROJECTION_SCHEME", -1.0) == -1.0,
              "ActiveMethodSetting answers for the active method only");

        // A block wins over a stale top-level copy of the same key, and a key the block does not
        // carry is the default, not the top-level value or the last run's.
        {
            // A full panel-written config, with its FLIP block cut down to one key and stale
            // top-level copies of two keys added.
            std::string mixed = text;
            const size_t fb = mixed.find("FLIP_BEGIN\n"), fe = mixed.find("FLIP_END\n");
            mixed.replace(fb, fe - fb, "FLIP_BEGIN\nALPHA = 0.7\n");
            std::ofstream f("mixed_cfg.txt");
            f << "ALPHA = 0.5\nPARTICLES_PER_CELL = 9\n" << mixed;
        }
        FLIP_SETTINGS.particlesPerCell = 27;
        back.LoadFrom("mixed_cfg.txt");
        check(FLIP_SETTINGS.alpha == 0.7, "the block wins over a top-level key");
        check(FLIP_SETTINGS.particlesPerCell == 4, "a key the block lacks is the default");
        {
            // the run-time loader refuses the top-level ALPHA and PARTICLES_PER_CELL nobody reads (exit 1) ...
            const pid_t pid = fork();
            if (pid == 0)
            {
                std::freopen("/dev/null", "w", stderr);
                ConfigReader::loadConfig(c, "mixed_cfg.txt");
                std::_Exit(0);
            }
            int status = 0;
            waitpid(pid, &status, 0);
            check(WIFEXITED(status) && WEXITSTATUS(status) == 1, "ConfigReader refuses the shadowed top-level keys");
            // ... and agrees with the setup screen once they are gone
            std::ifstream in("mixed_cfg.txt");
            std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            all.erase(0, std::string("ALPHA = 0.5\nPARTICLES_PER_CELL = 9\n").size());
            std::ofstream("mixed_clean_cfg.txt") << all;
        }
        ConfigReader::loadConfig(c, "mixed_clean_cfg.txt");
        check(FLIP_SETTINGS.alpha == 0.7 && FLIP_SETTINGS.particlesPerCell == 4, "ConfigReader agrees");
        {
            std::string bad = text;
            const size_t k = bad.find("CONVECTION_SCHEME = ");
            bad.replace(k, bad.find('\n', k) - k, "CONVECTION_SCHEME = SIDEWAYS");
            std::ofstream f("badenum_cfg.txt");
            f << bad;
        }
        ADI_SETTINGS.convectionScheme = CONVECTION_UPWIND;
        back.LoadFrom("badenum_cfg.txt");
        check(ADI_SETTINGS.convectionScheme == CONVECTION_CENTRAL, "an unknown enum value is the default");
        SPH_SETTINGS = SPHSettings(); FLIP_SETTINGS = FLIPSettings(); ADI_SETTINGS = ADISettings();
    }

    // ---- stopping rule: written, reloaded, read into SIMULATION.stop, reset by a config without it ----
    std::printf("\n== stopping rule in the config ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 2;
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        ui.stop.condition = STOP_RESIDUAL;
        ui.stop.residualTol = 3.5e-8;
        ui.stop.tFinal = 12.5;
        ui.stop.exportOnStop = false;
        std::string err;
        check(ui.WriteFiles(err, "stop_cfg.txt"), "a config with a stopping rule is written", err);
        SimulationSetupUI back;
        back.LoadFrom("stop_cfg.txt");
        check(back.stop.condition == STOP_RESIDUAL && std::abs(back.stop.residualTol - 3.5e-8) < 1e-20 &&
                  std::abs(back.stop.tFinal - 12.5) < 1e-12 && !back.stop.exportOnStop,
              "the panel reloads STOP_CONDITION, STOP_RESIDUAL, STOP_TIME and STOP_EXPORT");
        SIMULATION.stop.triggered = true;
        SIMULATION.stop.exported = true;
        SIMULATION_CONFIG c;
        ConfigReader::loadConfig(c, "stop_cfg.txt");
        check(SIMULATION.stop.condition == STOP_RESIDUAL && std::abs(SIMULATION.stop.residualTol - 3.5e-8) < 1e-20 &&
                  std::abs(SIMULATION.stop.tFinal - 12.5) < 1e-12 && !SIMULATION.stop.exportOnStop,
              "ConfigReader puts the rule on SIMULATION.stop");
        check(!SIMULATION.stop.triggered && !SIMULATION.stop.exported, "and a fresh load starts the rule armed");

        {
            std::ofstream f("stop_none_cfg.txt");
            f << "DIMENSION = 2\nTYPE = ADI\nGRID_SIZE = 16\nDOMAIN_X0 = 0\nDOMAIN_XF = 1\nDOMAIN_Y0 = 0\n"
              << "DOMAIN_YF = 1\nDOMAIN_Z0 = 0\nDOMAIN_ZF = 1\n";
        }
        back.LoadFrom("stop_none_cfg.txt");
        check(back.stop.condition == STOP_NEVER && back.stop.exportOnStop, "a config without the keys resets the panel to NEVER");
        ConfigReader::loadConfig(c, "stop_none_cfg.txt");
        check(SIMULATION.stop.condition == STOP_NEVER, "and the backend to NEVER");

        ui.stop.condition = STOP_TIME;
        ui.stop.tFinal = 0.0;
        std::vector<std::string> probs, notes;
        ui.Validate(probs, notes);
        bool refused = false;
        for (const std::string &q : probs) if (q.find("final time") != std::string::npos) refused = true;
        check(refused, "a non-positive final time is refused");
        std::remove("stop_cfg.txt");
        std::remove("stop_none_cfg.txt");
    }

    // ---- pressure settings: inside the PRESSURE block, never at the top level, never handed to AMGX ----
    {
        std::printf("\n== pressure settings in the PRESSURE block ==\n");
        SimulationSetupUI ui;
        ui.method = MethodFromKey("ADI");
        ui.ReconcileSelections();
        PRESSURE_SETTINGS.warmStart = true;
        PRESSURE_SETTINGS.needsCompatibilityCondition = true;
        std::string err;
        const std::string path = "pressure_block_probe.cfg";
        check(ui.WriteFiles(err, path), "writing a config", err);
        std::string text;
        {
            std::ifstream in(path);
            text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        const size_t b = text.find("PRESSURE_BEGIN"), e = text.find("PRESSURE_END");
        const std::string top = text.substr(0, b), block = (b == std::string::npos) ? "" : text.substr(b, e - b);
        check(top.find("PRESSURE_WARM_START") == std::string::npos &&
              top.find("NEEDS_COMPATIBILITY_CONDITION") == std::string::npos,
              "neither key is written at the top level");
        check(block.find("PRESSURE_WARM_START = 1") != std::string::npos &&
              block.find("NEEDS_COMPATIBILITY_CONDITION = 1") != std::string::npos,
              "both are written inside the block", block);

        PRESSURE_SETTINGS = PressureSettings();
        SimulationSetupUI back;
        back.LoadFrom(path);
        check(PRESSURE_SETTINGS.warmStart && PRESSURE_SETTINGS.needsCompatibilityCondition,
              "the panel reads them back from the block");

        PRESSURE_SETTINGS = PressureSettings();
        SIMULATION_CONFIG c;
        ConfigReader::loadConfig(c, path);
        check(PRESSURE_SETTINGS.warmStart && PRESSURE_SETTINGS.needsCompatibilityCondition,
              "and so does the backend");
        const std::string amgx = AmgxConfigString(PRESSURE_SETTINGS.text);
        check(amgx.find("WARM_START") == std::string::npos && amgx.find("COMPATIBILITY") == std::string::npos &&
              amgx.find("config_version=2") != std::string::npos,
              "AMGX's string does not carry them", amgx);

        // A config written before the keys moved in: both at the top level, none in the block.
        std::string legacy = text;
        const std::string k1 = "NEEDS_COMPATIBILITY_CONDITION = 1\n", k2 = "PRESSURE_WARM_START = 1\n";
        legacy.erase(legacy.find(k1), k1.size());
        legacy.erase(legacy.find(k2), k2.size());
        legacy = k1 + k2 + legacy;
        {
            std::ofstream out("pressure_legacy_probe.cfg");
            out << legacy;
        }
        PRESSURE_SETTINGS = PressureSettings();
        ConfigReader::loadConfig(c, "pressure_legacy_probe.cfg");
        check(PRESSURE_SETTINGS.warmStart && PRESSURE_SETTINGS.needsCompatibilityCondition,
              "a config with them at the top level is still read");
        std::remove(path.c_str());
        std::remove("pressure_legacy_probe.cfg");
        PRESSURE_SETTINGS = PressureSettings();
    }

    // ---- export and checkpoint: the panel must not silently drop what it does not draw ----
    // Everything a headless run needs is in the file, so the panel has to round-trip ALL of it --
    // the telemetry columns included, which it has no widgets for. Pressing Start would otherwise
    // delete them, which is exactly how the prescribed-pressure velocity, `force` and `initial`
    // were each lost.
    std::printf("\n== export and checkpoint in the config ==\n");
    {
        SimulationSetupUI ui;
        ui.dimension = 3;
        ui.method = MethodFromKey("FLIP");
        ui.ReconcileSelections();
        ui.exports.enabled = true;
        ui.exports.telemetryInterval = 7;
        ui.exports.gridEveryTime = true;
        ui.exports.gridTimeInterval = 0.025;
        ui.exports.gridInterval = 33;
        ui.exports.exportGridData = true;
        ui.exports.exportParticleData = true;
        ui.exports.exportSurfaceMesh = true;
        ui.exports.surfaceInfluenceScale = 4.0;
        ui.exports.surfaceRadiusScale = 0.95;
        ui.exports.exportCFL = false;            // no widget draws this one
        ui.exports.exportFlowIntegrals = false;  // nor this
        ui.checkpoint.autoSave = true;
        ui.checkpoint.interval = 40;
        std::snprintf(ui.checkpoint.path, sizeof(ui.checkpoint.path), "runs/night.ckpt");

        std::string err;
        check(ui.WriteFiles(err, "exp_cfg.txt"), "a config with export and checkpoint settings is written", err);

        SimulationSetupUI back;
        back.LoadFrom("exp_cfg.txt");
        check(back.exports.telemetryInterval == 7 && back.exports.gridInterval == 33 &&
                  back.exports.gridEveryTime && std::abs(back.exports.gridTimeInterval - 0.025) < 1e-12,
              "the panel reloads the intervals and the frame mode");
        check(back.exports.exportGridData && back.exports.exportParticleData && back.exports.exportSurfaceMesh,
              "and which artifacts to write");
        check(std::abs(back.exports.surfaceInfluenceScale - 4.0) < 1e-12 &&
                  std::abs(back.exports.surfaceRadiusScale - 0.95) < 1e-12,
              "and the surface settings");
        check(!back.exports.exportCFL && !back.exports.exportFlowIntegrals,
              "and the telemetry columns it has no widgets for");
        check(back.checkpoint.autoSave && back.checkpoint.interval == 40 &&
                  std::string(back.checkpoint.path) == "runs/night.ckpt",
              "and the checkpoint interval and path");

        // A second trip has to be byte for byte the first, or a save that changes nothing still
        // shows up as a diff.
        std::string err2;
        check(back.WriteFiles(err2, "exp_cfg2.txt"), "the reloaded panel writes again", err2);
        check(ReadText("exp_cfg.txt") == ReadText("exp_cfg2.txt"), "a second round trip is byte identical");

        SIMULATION_CONFIG c;
        ConfigReader::loadConfig(c, "exp_cfg.txt");
        check(SIMULATION.exports.telemetryInterval == 7 && SIMULATION.exports.gridEveryTime &&
                  SIMULATION.exports.exportParticleData && !SIMULATION.exports.exportCFL,
              "ConfigReader puts them on SIMULATION.exports");
        check(SIMULATION.checkpoint.autoSave && SIMULATION.checkpoint.interval == 40 &&
                  std::string(SIMULATION.checkpoint.path) == "runs/night.ckpt",
              "and the checkpoint on SIMULATION.checkpoint");

        {
            std::ofstream f("exp_none_cfg.txt");
            f << "DIMENSION = 2\nTYPE = ADI\nGRID_SIZE = 16\nDOMAIN_X0 = 0\nDOMAIN_XF = 1\nDOMAIN_Y0 = 0\n"
              << "DOMAIN_YF = 1\nDOMAIN_Z0 = 0\nDOMAIN_ZF = 1\n";
        }
        back.LoadFrom("exp_none_cfg.txt");
        check(back.exports.telemetryInterval == 1 && !back.exports.exportGridData && back.exports.exportCFL,
              "a config without the keys resets the panel to the defaults");
        ConfigReader::loadConfig(c, "exp_none_cfg.txt");
        check(!SIMULATION.checkpoint.autoSave && SIMULATION.checkpoint.interval == 250 &&
                  std::string(SIMULATION.checkpoint.path) == "checkpoint.ckpt",
              "and the backend to the checkpoint defaults");

        std::remove("exp_cfg.txt");
        std::remove("exp_cfg2.txt");
        std::remove("exp_none_cfg.txt");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
} // namespace setup_ui_check

static const SuiteCase CASES[] = {
    {"setup screen", setup_ui_check::run, {"@WORKDIR"}, false},
};

int main(int argc, char **argv) { return RunSuite("setup_ui", CASES, sizeof(CASES) / sizeof(CASES[0]), argc, argv); }
