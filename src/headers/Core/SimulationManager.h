// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SIMULATIONMANAGER_H
#define SIMULATIONMANAGER_H

#include <iostream>
#include "MAC.h"
#include "ConfigReader.h"
#include "Checkpoint.h"
#include "Exports.h"

// The manager knows no method. A method is a row in the registry (Definitions.h), put there by the program before
// InitializeSimulation (RegisterADI() etc., in each method's settings header); the config's TYPE picks the row, and the row's
// initialize builds the method's solvers and registers its steps.

class SimulationManager{

public:
    static void InitializeSimulation(const std::string& configFile = "");
    static void StepSimulation();
    // Ends the run: the method frees its solvers, the grids and the registered steps go, and SIMULATION is back to its
    // defaults, so InitializeSimulation can start another run in the same program.
    static void DisposeSimulation();

private:

    // Adaptive dt for the active method (no-op when ADAPTATIVE_TIMESTEP is off), and the two CFL numbers from a
    // velocity and length scale.
    static void AdaptTimestep();
    static void ComputeCFL(double maxVel, double h);
    static void SetCheckpointStatus(bool ok, const std::string &done, const std::string &err);

    // telemetry update functions
    static void UpdateTelemetry();
    static void RecordFlowIntegrals(double t);
    static void UpdateAerodynamicsTelemetry();
    static void updateDambreakTelemetry();

    static void CheckStoppingRule();

};


void SimulationManager::ComputeCFL(double maxVel, double h)
{
    SIMULATION.telemetry.lastAdvectionCFL = (maxVel / h) * SIMULATION.dt;
    SIMULATION.telemetry.lastDiffusionCFL = (SIMULATION.EPS * SIMULATION.dt) / (h * h);
}

void SimulationManager::AdaptTimestep()
{
    if (!ADAPTATIVE_TIMESTEP) return;
    if (ActiveMethod().stableTimestep)
    {
        // The method's own stability criteria -- for SPH, Monaghan's, the same ones AdvanceSPH
        // substeps against.
        SIMULATION.dt = std::min(ActiveMethod().stableTimestep(), SIMULATION.dtConfigured);
    }
    else
    {
        // TIME_STEP is the ceiling and adaptivity only ever shrinks it.
        SIMULATION.dt = ComputeAdaptiveTimestep(SIMULATION.GRID_ANT->GetMaxVelocity(), SIMULATION.dh, MAX_CFL,
                                                SIMULATION.dtConfigured);
    }
}

void SimulationManager::SetCheckpointStatus(bool ok, const std::string &done, const std::string &err)
{
    SIMULATION.checkpoint.status = ok ? done : err;
    SIMULATION.checkpoint.statusIsError = !ok;
}

void SimulationManager::UpdateTelemetry(){
    // A method with no grid says what to measure the CFL against; everything else uses the grid.
    if (ActiveMethod().maxVelocity && ActiveMethod().cflLength)
        ComputeCFL(ActiveMethod().maxVelocity(), ActiveMethod().cflLength());
    else
        ComputeCFL(SIMULATION.GRID_SOL->GetMaxVelocity(), SIMULATION.dh);

    CellReynolds cellRe;
    if (MethodUsesGrid())
        cellRe = (DIMENSION == 2) ? ComputeMaxCellReynolds2D(SIMULATION.GRID_SOL, SIMULATION.EPS)
                                  : ComputeMaxCellReynolds3D(SIMULATION.GRID_SOL, SIMULATION.EPS);

    // Outside the `if` on purpose: a method with no grid still pushes a sample, with cellRe left
    // at its default. The indentation used to say otherwise.
    // Labelled with the time the new field is AT, like the iteration count after the step.
    // The divergence the step left behind, measured here rather than reported by a step.
    const double divergence = MethodUsesGrid() ? SIMULATION.GRID_SOL->GetDivSum() : 0.0;
    SIMULATION.telemetry.Push(
        SIMULATION.telemetry.clock + SIMULATION.dt,
        divergence,
        SIMULATION.telemetry.lastAdvectionCFL,
        SIMULATION.telemetry.lastDiffusionCFL,
        SIMULATION.telemetry.lastResidual,
        SIMULATION.telemetry.lastStepTimings,
        cellRe
    );

    RecordFlowIntegrals(SIMULATION.telemetry.clock + SIMULATION.dt);
}

// Kinetic energy and enstrophy. Only while the
// export panel asks for them -- one pass over the grid per step that is wasted otherwise.
void SimulationManager::RecordFlowIntegrals(double t)
{
    // Both flags: the checkbox is only visible while export is enabled, so the flag alone would
    // keep computing behind a hidden, default-on box.
    if (!SIMULATION.exports.enabled || !SIMULATION.exports.exportFlowIntegrals || !MethodUsesGrid())
    {
        if (SIMULATION.flowIntegrals.available) SIMULATION.flowIntegrals.Clear();
        return;
    }
    const FlowIntegrals fi = (DIMENSION == 2) ? ComputeFlowIntegrals2D(SIMULATION.GRID_SOL)
                                              : ComputeFlowIntegrals3D(SIMULATION.GRID_SOL);
    SIMULATION.flowIntegrals.Push(t, fi.kineticEnergy, fi.enstrophy, fi.fluidVolume, SIMULATION.RHO);
}

void SimulationManager::UpdateAerodynamicsTelemetry() {
    if (!SIMULATION.exports.aerodynamics) return;

    if(DIMENSION == 2) {

    SIMULATION.aero.time.push_back(SIMULATION.telemetry.clock + SIMULATION.dt);

    // The integral itself lives in Utils, free of ImGui, so it can be tested directly --
    // see run_aero_check. This panel only formats what it returns.
    double Fd = 0.0, Fl = 0.0;
    AeroForces2D(SIMULATION.GRID_SOL, SIMULATION.Nx, SIMULATION.Ny, SIMULATION.EPS, Fd, Fl);

    // Cd = 2F / (rho * Umean^2 * L). The density was missing, which is invisible at DENSITY = 1 --
    // every shipped config -- and wrong by a factor of rho otherwise.
    const double rho = (SIMULATION.RHO > 0.0) ? SIMULATION.RHO : 1.0;
    double coeff = 2.0 / (rho * pow(SIMULATION.MEAN_VELOCITY,2) * SIMULATION.CHARACTERISTIC_LENGTH);
    double Cd = coeff * rho * Fd;
    double Cl = coeff * rho * Fl;

    // Store in telemetry for visualization
    SIMULATION.aero.Cl.push_back(Cl);
    SIMULATION.aero.Cd.push_back(Cd);

    // The same force by the momentum balance over a box around the body (half its size beyond it on every side), which
    // never touches the staircase. GRID_ANT still holds the previous step here, for the unsteady term.
    int jL, jR, iB, iT;
    AeroCVParts2D cv;
    if (AeroCVBox2D(SIMULATION.GRID_SOL, 0.5, jL, jR, iB, iT))
        AeroForcesCV2D(SIMULATION.GRID_SOL, SIMULATION.GRID_ANT, SIMULATION.dt, rho, SIMULATION.EPS, jL, jR, iB, iT, cv);
    SIMULATION.aero.CdCV.push_back(cv.valid ? coeff * rho * cv.Deformation(0) : 0.0);
    SIMULATION.aero.ClCV.push_back(cv.valid ? coeff * rho * cv.Deformation(1) : 0.0);
    // Cd passes through zero on a symmetric start-up transient; without the guard the
    // ratio goes infinite and takes the ImPlot autoscale with it.
    SIMULATION.aero.LtoDRatio.push_back((std::abs(Cd) > 1e-12) ? (Cl / Cd) : 0.0);
    }
    else if(DIMENSION == 3) {
        // The 3D twin of the branch above, rebuilt the same way the 2D one was: the integral lives
        // in Utils (AeroForces3D, testable without ImGui.
        SIMULATION.aero.time.push_back(SIMULATION.telemetry.clock + SIMULATION.dt);

        MAC *grid0 = SIMULATION.GRID_SOL;
        AeroForces3DParts parts;
        AeroForces3D(grid0, SIMULATION.Nx, SIMULATION.Ny, SIMULATION.Nz, SIMULATION.EPS, parts);

        // Cd = 2F / (rho U^2 S), S = L1 * L2 from the config (for a wall-to-wall cylinder, D * H).
        const double S = SIMULATION.CHARACTERISTIC_LENGTH * SIMULATION.CHARACTERISTIC_LENGTH_2;
        const double rho = (SIMULATION.RHO > 0.0) ? SIMULATION.RHO : 1.0;
        const double coeff = 2.0 / (rho * pow(SIMULATION.MEAN_VELOCITY, 2) * S);
        // Total() is the stress form AERO_STRESS_FORM_3D selects: see AeroForces3DParts,
        // and the Schaefer 3D-1Z benchmark runs that chose it.
        const double Cd = coeff * rho * parts.Total(0);
        const double Cl = coeff * rho * parts.Total(1);
        const double Cs = coeff * rho * parts.Total(2);

        SIMULATION.aero.Cl.push_back(Cl);
        SIMULATION.aero.Cd.push_back(Cd);
        SIMULATION.aero.Cs.push_back(Cs);

        // The momentum balance over a box around the body, as in 2D; a body spanning the channel closes it on the walls.
        int jL, jR, iB, iT, kB, kT;
        AeroCVParts3D cv;
        if (AeroCVBox3D(grid0, 0.5, jL, jR, iB, iT, kB, kT))
            AeroForcesCV3D(grid0, SIMULATION.GRID_ANT, SIMULATION.dt, rho, SIMULATION.EPS, jL, jR, iB, iT, kB, kT, cv);
        SIMULATION.aero.CdCV.push_back(cv.valid ? coeff * rho * cv.Deformation(0) : 0.0);
        SIMULATION.aero.ClCV.push_back(cv.valid ? coeff * rho * cv.Deformation(1) : 0.0);
        SIMULATION.aero.LtoDRatio.push_back((std::abs(Cd) > 1e-12) ? (Cl / Cd) : 0.0);
    }
}
void SimulationManager::updateDambreakTelemetry(){
    if (!SIMULATION.exports.dambreak) return;

    // The top face of the highest FLUID cell and the right face of the rightmost one, read off the face tables, so
    // they are physical coordinates whatever the origin, the spacing, or the ring outside the domain. 3D scans the
    // z centre plane.
    MAC *g = SIMULATION.GRID_SOL;
    const int k = (DIMENSION == 3) ? SIMULATION.Nz / 2 : 0;
    double fluidLevel = 0.0, fluidFrontPos = 0.0;
    bool levelFound = false, frontFound = false;
    for (int i = SIMULATION.Ny - 1; i >= 0 && !levelFound; i--)
        for (int j = 0; j < SIMULATION.Nx; j++)
            if (((DIMENSION == 3) ? g->GetSolid(i, j, k) : g->GetSolid(i, j)) == FLUID_CELL)
            {
                fluidLevel = g->FacePos(MAC::Y, i + 1);
                levelFound = true;
                break;
            }
    for (int j = SIMULATION.Nx - 1; j >= 0 && !frontFound; j--)
        for (int i = 0; i < SIMULATION.Ny; i++)
            if (((DIMENSION == 3) ? g->GetSolid(i, j, k) : g->GetSolid(i, j)) == FLUID_CELL)
            {
                fluidFrontPos = g->FacePos(MAC::X, j + 1);
                frontFound = true;
                break;
            }
    SIMULATION.dambreak.fluidLevel.push_back(fluidLevel);
    SIMULATION.dambreak.fluidFrontPosition.push_back(fluidFrontPos);
}

// Stop the run on its own terms. Checked after a step has completed, so the clock and
// `residual` are the values that step produced.
void SimulationManager::CheckStoppingRule()
{
    if (SIMULATION.stop.condition == STOP_NEVER || SIMULATION.stop.triggered) return;

    std::string why;
    if (!StoppingRuleFires(SIMULATION.stop, SIMULATION.telemetry.clock, SIMULATION.telemetry.lastResidual, SIMULATION.telemetry.iteration, why)) return;

    SIMULATION.stop.triggered = true;
    SIMULATION.stop.reason = why;
    SIMULATION.running = false;
    SIMULATION.stepOnce = false;
    std::cout << "Simulation stopped: " << why << std::endl;

    if (SIMULATION.stop.exportOnStop && !SIMULATION.stop.exported)
    {
        SIMULATION.stop.exported = true;
        Exports::WriteFinal();
        std::cout << "  final field and telemetry exported." << std::endl;
    }
}

void SimulationManager::InitializeSimulation(const std::string& configFile){
    // The registry is the program's to fill, before this: the config's TYPE is looked up in it, and building the grid
    // asks the method whether a stretched grid is allowed.
    if (METHOD_COUNT == 0)
    {
        std::cerr << "ERROR: no simulation method is registered. Call each method's Register function "
                     "(RegisterADI(), ... in its settings header) before InitializeSimulation."
                  << std::endl;
        std::exit(1);
    }

    std::string file = configFile;
    if (file.empty())
    {
        file = "simulation_config.txt";
        std::cout << "No configuration file was specified, so it defaulted to " << file << "." << std::endl;
    }
    ConfigReader::loadConfig(SIMULATION, file);
    omp_set_num_threads(THREAD_COUNT);
    // Solvers register their steps from their Initialize; a re-initialisation starts from none.
    ClearSimulationSteps();
    SIMULATION.particles = ParticleSet();
    SIMULATION.telemetry.Push(0.0, 0.0, 0.0, 0.0, 0.0, std::vector<double>());

    std::cout << "MAC Grid initialized - Parameters\n"
              << "-dt = " << std::to_string(SIMULATION.dt) << "\n"
              << "-dh = " << std::to_string(SIMULATION.dh) 
              << "\n-Nx = " << std::to_string(SIMULATION.Nx) 
              << "\n-Ny = " << std::to_string(SIMULATION.Ny)
              << "\n-Nz = " << std::to_string(SIMULATION.Nz) << std::endl;
    std::cout << "Total node count: " 
              << ((SIMULATION.Nx+1) * (SIMULATION.Ny+1) * (SIMULATION.Nz+1)) * 3 
              << " - Re = " << SIMULATION.RE 
              << " - Grid size: " << SIMULATION.GRID_SIZE << std::endl;

    if (ActiveMethod().initialize)
        ActiveMethod().initialize();

    std::cout << "Simulation steps:";
    for (const SimulationStep &step : SIMULATION_STEPS)
        std::cout << "  " << step.order << " " << step.name;
    std::cout << std::endl;
    SIMULATION.telemetry.SetSteps(SIMULATION_STEPS);

    // The seeded field at t = 0: E(0) is the reference, and the first step has a difference to take.
    SIMULATION.flowIntegrals.Clear();
    RecordFlowIntegrals(0.0);
}

void SimulationManager::DisposeSimulation()
{
    if (ActiveMethod().dispose)
        ActiveMethod().dispose();
    ClearSimulationSteps();
    if (SIMULATION.exports.telemetryFile.is_open())
        SIMULATION.exports.telemetryFile.close();
    delete SIMULATION.GRID_SOL;
    delete SIMULATION.GRID_ANT;
    SIMULATION = SIMULATION_CONFIG();
    SIM_TYPE = -1;
}

void SimulationManager::StepSimulation(){



    // ---- checkpoints ---- Between the frame and the step is the only safe point: the solvers are
    // not mid-sweep, and GRID_SOL/GRID_ANT hold a complete state. The panel only ever sets a flag.
    SIMULATION.checkpoint.it = SIMULATION.telemetry.iteration;
    SIMULATION.checkpoint.time = SIMULATION.telemetry.clock;
    if (SIMULATION.checkpoint.loadRequested)
    {
        SIMULATION.checkpoint.loadRequested = false;
        std::string err;
        double t = SIMULATION.telemetry.clock;
        uint64_t it = SIMULATION.telemetry.iteration;
        const bool ok = Checkpoint::Load(SIMULATION.checkpoint.path, t, it, err);
        if (ok) { SIMULATION.telemetry.clock = t; SIMULATION.telemetry.iteration = (unsigned int)it; }
        SetCheckpointStatus(ok, "loaded " + std::string(SIMULATION.checkpoint.path), err);
    }
    if (SIMULATION.checkpoint.saveRequested)
    {
        SIMULATION.checkpoint.saveRequested = false;
        std::string err;
        const bool ok = Checkpoint::Save(SIMULATION.checkpoint.path, SIMULATION.telemetry.clock, SIMULATION.telemetry.iteration, err);
        SetCheckpointStatus(ok, "saved at step " + std::to_string(SIMULATION.telemetry.iteration), err);
    }

    AdaptTimestep();

    if (SIMULATION.running || SIMULATION.stepOnce){
        double start = GetWallTime();
        // The last step of a time rule is shortened to land on STOP_TIME; the configured dt comes back after it.
        const double dtBeforeClip = SIMULATION.dt;
        SIMULATION.dt = ClipStepToStopTime(SIMULATION.stop, SIMULATION.telemetry.clock, SIMULATION.dt);
        // The time this step is solving FOR, not the one it starts from: a prescribed pressure is a
        // boundary condition on the new field, so it is evaluated at t + dt.
        SIMULATION.time = SIMULATION.telemetry.clock + SIMULATION.dt;

        SIMULATION.GRID_SOL->ClearFlowFields();

        SIMULATION.telemetry.lastStepTimings.assign(SIMULATION_STEPS.size(), 0.0);
        SIMULATION.telemetry.lastStepCounters.assign(SIMULATION_STEPS.size(), -1.0);
        for (size_t s = 0; s < SIMULATION_STEPS.size(); s++)
        {
            const double stepStart = GetWallTime();
            SIMULATION_STEPS[s].function(SIMULATION.GRID_ANT, SIMULATION.GRID_SOL, &SIMULATION.particles, SIMULATION.telemetry.clock);
            SIMULATION.telemetry.lastStepTimings[s] = GetWallTime() - stepStart;
            // The manager asks the step for its work count; the step itself writes nothing here.
            if (SIMULATION_STEPS[s].counter)
                SIMULATION.telemetry.lastStepCounters[s] = SIMULATION_STEPS[s].counter();
        }

        SIMULATION.telemetry.lastResidual = SIMULATION.GRID_ANT->MaxAbsoluteDifference(*SIMULATION.GRID_SOL);

        UpdateTelemetry();
        UpdateAerodynamicsTelemetry();
        updateDambreakTelemetry();

        SIMULATION.GRID_ANT->CopyGrid(*SIMULATION.GRID_SOL);

        SIMULATION.telemetry.clock += SIMULATION.dt;
        SIMULATION.telemetry.iteration++;
        SIMULATION.dt = dtBeforeClip;
        SIMULATION.stepOnce = false;

        Exports::Update();
        CheckStoppingRule();

        // Auto-save: ONE file, overwritten.
        if (ShouldAutoSaveCheckpoint(SIMULATION.telemetry.iteration, SIMULATION.checkpoint))
        {
            std::string err;
            const bool ok = Checkpoint::Save(SIMULATION.checkpoint.path, SIMULATION.telemetry.clock, SIMULATION.telemetry.iteration, err);
            SetCheckpointStatus(ok, "auto-saved at step " + std::to_string(SIMULATION.telemetry.iteration), "auto-save off: " + err);
            if (!ok) SIMULATION.checkpoint.autoSave = false;
        }

        double end  = GetWallTime();
        SIMULATION.telemetry.lastStepWallTime = end-start;
    }

}

#endif