// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/*Made by Raphael Laroca*/
#include <iostream>
#include <iomanip>
#include <time.h>
#include <sstream>
#include <csignal>
#include <cmath>
#include <filesystem>

// The methods' settings panels draw with ImGui only in the GUI program; solvers and harnesses compile
// the same headers without it and get panels that draw nothing.
#ifndef HEADLESS
#define NEREUS_GUI
#endif

#include "headers/Core/SimulationManager.h"
// The methods this program is built with. Each settings header carries its method's Register function.
#include "Methods/ADI/ADISettings.h"
#include "Methods/FLIP/FLIPSettings.h"
#include "Methods/SPH/SPHSettings.h"
#include "Methods/Explicit/ExplicitSettings.h"

#ifndef HEADLESS
// The UI: InitializeUI, UpdateUI, RestartUI and DisposeUI. Everything ImGui/GLFW stays behind these four.
#include "headers/Core/SimulationManagerUI.h"
#endif

#include "headers/Utils.h"


/*To do for the CPC paper
-encapsulate all UI inside a simple Create, Update Exit() loop
-Write the EULERIAN FVM Explicit method, 100% self contained, both momentum and pressure solver (gauss seidal, cpu only)
-Rework the GridVisualizar, remove any mention of methods, make a taskbar at the top, dock the windows
-Rework the pressure solver, so that it behaves more like a method. ADI register its momentum at 0, Pressure itself register its step and Projection - DONE!
-Projection can be generalized for both flip and adi. - DONE!
-Document every core file, try to make most things private when possible, we need to hide as much as we can for the user. At the end, I want main to look like. Cut as much repetition we can
 from functions, just like we did with MAC

int main(){
RegisterMethod()
InitSimulation(configfile)
if(UI){
    initUI()
}
while(simulation.isrunning()){
if(UI)
updateUI()
simulation.step()
}

destroySimulation()
destroyUI()

return 0
}
*/

// ---------------- the program: read the arguments, register the methods, run headless or in the window ----------------

enum HeadlessExit { HL_OK = 0, HL_USAGE = 1, HL_DIVERGED = 2, HL_INTERRUPTED = 3 };

struct HeadlessOptions
{
    std::string config = "simulation_config.txt";
    long long progress = 100; // steps between progress lines
    bool quiet = false;
    bool buildInfo = false;
    bool headless = false;  // GUI build only: run without opening the window
    bool help = false;
    bool configGiven = false;  // a config path came from the command line
};

static bool ParseArguments(int argc, char *argv[], HeadlessOptions &o, std::string &err);
static void PrintUsage();
static void PrintBuildInfo();
static void MoveToProjectRoot(const char *argv0, HeadlessOptions &o);
static int RunHeadless(const HeadlessOptions &o);
#ifndef HEADLESS
static int RunWindow();
#endif

int main(int argc, char *argv[])
{
    HeadlessOptions options;
    std::string err;
    if (!ParseArguments(argc, argv, options, err)) { std::cerr << err << "\n\n"; PrintUsage(); return HL_USAGE; }
    if (options.help) { PrintUsage(); return HL_OK; }
    if (options.buildInfo) { PrintBuildInfo(); return HL_OK; }
    MoveToProjectRoot(argv[0], options);

    // The methods this program can run, registered before anything reads a config or lists them.
    RegisterADI();
    RegisterFLIP();
    RegisterSPH();
    RegisterExplicit();

#ifdef HEADLESS
    return RunHeadless(options);  // this build has no window to open, so --headless is implied
#else
    return options.headless ? RunHeadless(options) : RunWindow();
#endif
}

// ---------------- headless: the config alone drives the run; nothing on this path touches ImGui or GLFW ----------------

// True (with a message on stderr) when the step just taken left a non-finite value in the field. One scan of u, v,
// w and p per step: a blown-up field costs that to catch here and hours to discover in the exports.
static bool FieldWentNonFinite()
{
    if (SIMULATION.GRID_SOL == nullptr || SIMULATION.GRID_SOL->AllFinite()) return false;
    std::cerr << "The field went non-finite at step " << SIMULATION.telemetry.iteration
              << " (t = " << SIMULATION.telemetry.clock << ")." << std::endl;
    return true;
}

// Set by the signal handler: the current step finishes, the checkpoint is written, and the process leaves
// through the normal shutdown (which runs the AMGX finalize).
static volatile std::sig_atomic_t INTERRUPTED = 0;
static void OnInterrupt(int) { INTERRUPTED = 1; }

static void PrintUsage()
{
    std::cout << "NEREUS " << NEREUS_VERSION << "\n\n"
        "Usage: program [--headless <config.cfg>] [options]\n"
        "\n"
        "  With no arguments the setup screen opens and the run is driven from there.\n"
        "\n"
        "Headless options:\n"
        "  --headless [<config>]  run <config> with no window (default simulation_config.txt)\n"
        "  --progress N           a progress line every N steps (default 100, 0 for none)\n"
        "  --quiet                the summary only\n"
        "  --build-info           print the version and what this binary was built with (gpu 0|1), and exit\n"
        "  --help                 this text\n"
        "\n"
        "Exit codes: 0 finished, 1 usage or configuration, 2 the field went non-finite,\n"
        "3 interrupted (a checkpoint was written if the config names one).\n";
}

// For scripts: whether this binary can run a GPU_ACCELERATION = 1 config on the GPU.
static void PrintBuildInfo()
{
    std::cout << "version " << NEREUS_VERSION << std::endl;
#ifdef NVIDIA_GPU
    std::cout << "gpu 1" << std::endl;
#else
    std::cout << "gpu 0" << std::endl;
#endif
}

// Last, at exit: finalizes AMGX (the pressure solver keeps the library up across runs). Safe when it was never used.
static void DestroyPressureSolver()
{
    PressureSolver2D::DestroyPressureSolver();
    PressureSolver3D::DestroyPressureSolver();
}

// How far through the run we are, as a fraction, for the ETA. Only a rule with a target can answer:
// a residual rule has no idea when the flow will settle.
static double HeadlessProgressFraction(const HeadlessOptions &o)
{
    if (SIMULATION.stop.condition == STOP_TIME && SIMULATION.stop.tFinal > 0.0)
        return SIMULATION.telemetry.clock / SIMULATION.stop.tFinal;
    return -1.0;
}

static void PrintProgress(const HeadlessOptions &o, double wallStart)
{
    const double elapsed = GetWallTime() - wallStart;
    std::cout << "step " << std::setw(8) << SIMULATION.telemetry.iteration
              << "   t " << std::fixed << std::setprecision(5) << SIMULATION.telemetry.clock
              << "   dt " << std::scientific << std::setprecision(3) << SIMULATION.dt
              << "   res " << SIMULATION.telemetry.lastResidual
              << "   " << std::fixed << std::setprecision(2)
              << SIMULATION.telemetry.lastStepWallTime * 1000.0 << " ms/step";

    const double done = HeadlessProgressFraction(o);
    if (done > 1e-6 && done < 0.999)
        std::cout << "   " << std::setprecision(1) << 100.0 * done << "%"
                  << "   eta " << std::setprecision(0) << elapsed * (1.0 / done - 1.0) << " s";
    std::cout << std::defaultfloat << std::endl;
}

// --quiet: everything the run prints to std::cout is dropped until the summary; errors still reach std::cerr.
struct NullBuffer : std::streambuf
{
    int overflow(int c) override { return c; }
};

struct QuietOutput
{
    NullBuffer sink;
    std::streambuf *saved = nullptr;
    explicit QuietOutput(bool quiet) { if (quiet) saved = std::cout.rdbuf(&sink); }
    void Restore() { if (saved) { std::cout.rdbuf(saved); saved = nullptr; } }
    ~QuietOutput() { Restore(); }
};

static int RunHeadless(const HeadlessOptions &o)
{
    QuietOutput quiet(o.quiet);
    SimulationManager::InitializeSimulation(o.config);
    // A scene that did not parse leaves a closed box behind; in a batch run that is a whole run in the wrong geometry.
    if (!SIMULATION.sceneError.empty())
    {
        std::cerr << "Refusing to run: the scene in " << o.config << " did not parse (" << SIMULATION.sceneError
                  << "). A headless run does not fall back to a closed box." << std::endl;
        SimulationManager::DisposeSimulation();
        DestroyPressureSolver();
        return HL_USAGE;
    }
    // Only when the config asks: with export off a headless run leaves no directory and no CSV (the GUI opens
    // the file anyway, since its box can be ticked mid-run).
    if (SIMULATION.exports.enabled) Exports::InitializeTelemetry();

    // A run with no way to end is not a batch job: the end of a run is in its file (STOP_TIME or STOP_RESIDUAL).
    if (SIMULATION.stop.condition == STOP_NEVER)
    {
        std::cerr << "Nothing would stop this run: " << o.config
                  << " has STOP_CONDITION = NEVER. Set TIME or RESIDUAL in the file." << std::endl;
        SimulationManager::DisposeSimulation();
        DestroyPressureSolver();
        return HL_USAGE;
    }

    std::signal(SIGINT, OnInterrupt);
    std::signal(SIGTERM, OnInterrupt);

    if (!o.quiet)
    {
        std::cout << "\n--- NEREUS " << NEREUS_VERSION << ", headless run ---\n";
        std::cout << "config   " << o.config << "\n";
        std::cout << "solver   " << ActiveMethod().key << ", " << DIMENSION << "D, "
                  << (GPU_ACCELERATION ? "GPU (AMGX)" : "CPU")
                  << ", " << THREAD_COUNT << " threads\n";
        std::cout << "grid     " << SIMULATION.Nx << " x " << SIMULATION.Ny;
        if (DIMENSION == 3) std::cout << " x " << SIMULATION.Nz;
        std::cout << (SIMULATION.spacing.nonUniform ? "  (refined)" : "") << "\n";
        std::cout << "stop     " << StopConditionKey(SIMULATION.stop.condition);
        if (SIMULATION.stop.condition == STOP_TIME) std::cout << " at t = " << SIMULATION.stop.tFinal;
        if (SIMULATION.stop.condition == STOP_RESIDUAL) std::cout << " below " << SIMULATION.stop.residualTol;
        std::cout << "\n";
        std::cout << "export   " << (SIMULATION.exports.enabled ? ExportDirectory("") : std::string("off"))
                  << "\n\n";
    }

    int code = HL_OK;
    std::string ending = "the stopping rule fired";
    const double wallStart = GetWallTime();

    SIMULATION.running = true;
    while (SIMULATION.running)
    {
        SimulationManager::StepSimulation();

        const bool last = !SIMULATION.running || INTERRUPTED;
        const bool tick = (o.progress > 0 && (long long)SIMULATION.telemetry.iteration % o.progress == 0);

        // Every step, before the exit tests, so the last step is checked too.
        if (FieldWentNonFinite())
        {
            ending = "the field went non-finite";
            code = HL_DIVERGED;
            break;
        }
        if ((tick || last) && !o.quiet) PrintProgress(o, wallStart);

        if (INTERRUPTED)
        {
            ending = "interrupted";
            code = HL_INTERRUPTED;
            break;
        }
    }

    const double wall = GetWallTime() - wallStart;
    const unsigned long long steps = SIMULATION.telemetry.iteration;
    quiet.Restore();

    if (code == HL_INTERRUPTED && SIMULATION.checkpoint.path[0] != '\0')
    {
        std::string err;
        if (Checkpoint::Save(SIMULATION.checkpoint.path, SIMULATION.telemetry.clock, steps, err))
            std::cout << "Checkpoint written to " << SIMULATION.checkpoint.path << std::endl;
        else
            std::cerr << "Checkpoint failed: " << err << std::endl;
    }

    std::cout << "\n--- done: " << ending << " ---\n";
    if (SIMULATION.stop.triggered && !SIMULATION.stop.reason.empty())
        std::cout << "rule     " << SIMULATION.stop.reason << "\n";
    std::cout << "steps    " << steps << "\n";
    std::cout << "time     " << std::fixed << std::setprecision(6) << SIMULATION.telemetry.clock
              << " s simulated\n";
    std::cout << "wall     " << std::fixed << std::setprecision(2) << wall << " s";
    if (steps > 0)
        std::cout << ", " << std::setprecision(2) << (wall / (double)steps) * 1000.0 << " ms/step, "
                  << std::setprecision(1) << (double)steps / (wall > 0.0 ? wall : 1.0) << " steps/s";
    std::cout << std::defaultfloat << std::endl;
    if (SIMULATION.exports.enabled)
        std::cout << "exports  " << ExportDirectory("") << std::endl;

    SimulationManager::DisposeSimulation();
    DestroyPressureSolver();
    return code;
}

static bool ParseArguments(int argc, char *argv[], HeadlessOptions &o, std::string &err)
{
    bool configGiven = false;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        const bool hasNext = (i + 1 < argc) && argv[i + 1][0] != '-';

        if (a == "--help" || a == "-h") { o.help = true; }
        else if (a == "--headless")
        {
            o.headless = true;
            if (hasNext) { o.config = argv[++i]; configGiven = true; }
            o.configGiven = configGiven;
        }
        else if (a == "--progress" && i + 1 < argc) { o.progress = std::atoll(argv[++i]); }
        else if (a == "--quiet") { o.quiet = true; }
        else if (a == "--build-info") { o.buildInfo = true; }
        else if (a[0] != '-' && !configGiven) { o.config = a; configGiven = o.configGiven = true; }
        else { err = "Unknown option: " + a; return false; }
    }
    return true;
}

// With no config named and no configs/ here (bin/nereus run from bin/, a double-click), work from the folder
// beside or above the executable that holds configs/. A run given a config stays where it was started.
static void MoveToProjectRoot(const char *argv0, HeadlessOptions &o)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    if (o.configGiven || fs::is_directory("configs", ec)) return;
    // The path stays a fs::path and is printed as UTF-8: a narrow conversion throws on Windows for a folder name
    // outside the code page, which closed the program at start.
    try
    {
        const fs::path exe = fs::weakly_canonical(ExecutablePath(argv0), ec);
        if (ec || !exe.has_parent_path()) return;
        for (const fs::path &root : {exe.parent_path(), exe.parent_path().parent_path()})
        {
            if (!fs::is_directory(root / "configs", ec)) continue;
            fs::current_path(root, ec);
            if (!ec) std::cout << "Working from " << root.u8string() << "\n";
            return;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Could not locate the project folder (" << e.what() << "); working from the current folder.\n";
    }
}

#ifndef HEADLESS
// ---------------- the window ----------------
static int RunWindow()
{
    // The window and the setup screen, which picks the config to run.
    std::string configFile;
    if (!InitializeUI(configFile))
    {
        DisposeUI();
        return 0;
    }

    // A step that leaves the field non-finite pauses the run and says so on the status line; the program then exits
    // with 2, as a headless run does. Restart ends the run and opens the setup screen again.
    int code = HL_OK;
    do
    {
        code = HL_OK;
        SimulationManager::InitializeSimulation(configFile);
        Exports::InitializeTelemetry();
        do
        {
            const unsigned long long before = SIMULATION.telemetry.iteration;
            SimulationManager::StepSimulation();
            if (SIMULATION.telemetry.iteration != before && FieldWentNonFinite())
            {
                code = HL_DIVERGED;
                SIMULATION.running = false;
                SIMULATION.stop.triggered = true;
                SIMULATION.stop.reason = "STOPPED: the field went non-finite at step " +
                                         std::to_string(SIMULATION.telemetry.iteration) + ".";
            }
        }
        while (UpdateUI());
        SimulationManager::DisposeSimulation();
    }
    while (RestartUI(configFile));

    DisposeUI();
    DestroyPressureSolver();
    return code;
}
#endif
