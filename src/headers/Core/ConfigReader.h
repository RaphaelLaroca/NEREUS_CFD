// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <fstream>
#include <sstream>
#include <cctype>
#include <string>
#include <map>
#include <iostream>
#include <cstdlib>

#include "Definitions.h"
#include "Scene.h"
#include "Utils.h"
#include "../../Methods/PPESolver/PressureSettings.h"

namespace ConfigReader {
    static std::map<std::string, std::string> parameters;

    // The scene directives embedded in the config, verbatim, between SCENE_BEGIN and SCENE_END.
    // Empty when the file carries no block.
    static std::string sceneBlock;
    // The pressure solver settings between PRESSURE_BEGIN and PRESSURE_END, verbatim.
    static std::string pressureBlock;
    // Each method's settings, between <KEY>_BEGIN and <KEY>_END, by method key.
    static std::map<std::string, ConfigValues> methodBlocks;
    // Every top-level key some reader asked for, and every problem found; loadConfig refuses the file on any.
    static std::set<std::string> usedKeys;
    static std::vector<std::string> configErrors;

    static bool loadFromFile(const std::string& filename) {
        std::ifstream configFile(filename);
        if (!configFile.is_open()) {
            std::cerr << "ERROR: configuration file '" << filename << "' not found." << std::endl;
            return false;
        }

        parameters.clear();
        usedKeys.clear();
        configErrors.clear();
        sceneBlock.clear();
        pressureBlock.clear();
        methodBlocks.clear();
        bool inScene = false, inPressure = false;
        // Inside a method block the KEY = value lines go to that method's map, not the top level.
        int inMethod = -1;
        std::string unknownBlockEnd;   // set while skipping the block of a method this program did not register
        std::string line;
        while (std::getline(configFile, line)) {
            const std::string trimmed = TrimWhitespace(line);
            if (!unknownBlockEnd.empty()) { if (trimmed == unknownBlockEnd) unknownBlockEnd.clear(); continue; }
            if (inMethod >= 0 && trimmed == MethodBlockEnd(METHODS[inMethod])) { inMethod = -1; continue; }
            if (inMethod < 0 && !inScene && !inPressure)
            {
                const int m = MethodBlockOpenedBy(trimmed);
                if (m >= 0) { inMethod = m; methodBlocks[METHODS[m].key]; continue; }
                const std::string open = "_BEGIN";
                if (trimmed.size() > open.size() && trimmed.find('=') == std::string::npos &&
                    trimmed.compare(trimmed.size() - open.size(), open.size(), open) == 0 &&
                    trimmed != PRESSURE_BLOCK_BEGIN && trimmed != SCENE_BLOCK_BEGIN)
                {
                    const std::string key = trimmed.substr(0, trimmed.size() - open.size());
                    configErrors.push_back("block " + trimmed + " belongs to a method this program does not "
                                           "register (" + key + ")");
                    unknownBlockEnd = key + "_END";
                    continue;
                }
            }
            if (!inScene && !inPressure && trimmed == PRESSURE_BLOCK_BEGIN) { inPressure = true; continue; }
            if (inPressure)
            {
                if (trimmed == PRESSURE_BLOCK_END) { inPressure = false; continue; }
                pressureBlock += trimmed;
                pressureBlock += '\n';
                continue;
            }
            if (!inScene && trimmed == SCENE_BLOCK_BEGIN) { inScene = true; continue; }
            if (inScene)
            {
                if (trimmed == SCENE_BLOCK_END) { inScene = false; continue; }
                sceneBlock += line;
                sceneBlock += '\n';
                continue;
            }

            if (line.empty() || line[0] == '#' || line[0] == '/') {
                continue;
            }

            std::istringstream lineStream(line);
            std::string key, value;

            // Format expected: key = value
            if (std::getline(lineStream, key, '=') && std::getline(lineStream, value)) {
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);

                // Trailing comments have to come off the VALUE too, not just off whole lines.
                const size_t hash = value.find('#');
                if (hash != std::string::npos)
                    value.erase(hash);
                value.erase(0, value.find_first_not_of(" \t"));
                const size_t last = value.find_last_not_of(" \t\r");
                value.erase(last == std::string::npos ? 0 : last + 1);

                if (inMethod >= 0) methodBlocks[METHODS[inMethod].key].values[key] = value;
                else parameters[key] = value;
            }
        }

        configFile.close();
        return true;
    }

    static std::string getString(const std::string& key, const std::string& defaultValue = "") {
        usedKeys.insert(key);
        auto it = parameters.find(key);
        if (it != parameters.end()) {
            return it->second;
        }
        return defaultValue;
    }

    static double getDouble(const std::string& key, double defaultValue = 0.0) {
        usedKeys.insert(key);
        auto it = parameters.find(key);
        if (it == parameters.end()) return defaultValue;
        double x;
        if (ConfigValues::ParseNumber(it->second, x)) return x;
        configErrors.push_back(key + " = '" + it->second + "' is not a number");
        return defaultValue;
    }

    static int getInt(const std::string& key, int defaultValue = 0) {
        usedKeys.insert(key);
        auto it = parameters.find(key);
        if (it == parameters.end()) return defaultValue;
        int x;
        if (ConfigValues::ParseInteger(it->second, x)) return x;
        configErrors.push_back(key + " = '" + it->second + "' is not an integer");
        return defaultValue;
    }

    // Every key of the file must have been asked for by some reader, and every value must have parsed; otherwise
    // the run is refused with the whole list, so a typo is found before the run, not after it.
    static void RefuseInvalidConfig(const std::string &filename, const ConfigValues &shared)
    {
        std::vector<std::string> problems = configErrors;
        problems.insert(problems.end(), shared.errors.begin(), shared.errors.end());
        for (const auto &kv : parameters)
            if (!usedKeys.count(kv.first) && !shared.used.count(kv.first))
                problems.push_back("unknown key " + kv.first);
        for (const auto &block : methodBlocks)
        {
            problems.insert(problems.end(), block.second.errors.begin(), block.second.errors.end());
            for (const auto &kv : block.second.values)
                if (!block.second.used.count(kv.first))
                    problems.push_back("unknown key " + kv.first + " in the " + block.first + " block");
        }
        if (problems.empty()) return;
        std::cerr << "ERROR: " << filename << " is not a valid configuration:" << std::endl;
        for (const std::string &p : problems) std::cerr << "  " << p << std::endl;
        std::exit(1);
    }

    // A config without a PRESSURE block reads the legacy configs/<name>.pcfg beside it when there is one,
    // otherwise PressureConfigText() falls back to solver_pressure_config.txt.
    static std::string LegacyPressureConfig(const std::string& filename) {
        if (filename.size() < 4 || filename.compare(filename.size() - 4, 4, ".cfg") != 0) return "";
        const std::string q = filename.substr(0, filename.size() - 4) + ".pcfg";
        std::ifstream in(q);
        if (!in.is_open()) return "";
        std::cout << "No PRESSURE block in " << filename << ": pressure solver settings read from "
                  << q << "." << std::endl;
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }

    // REFINE_* is 'start end factor [mode]' intervals separated by ';'; the mode defaults to hard.
    static void ReadRefinement(const std::string &key, std::vector<GridRefinement> &out)
    {
        out.clear();
        const std::string spec = getString(key, "");
        if (spec.empty()) return;
        std::stringstream all(spec);
        std::string part;
        while (std::getline(all, part, ';'))
        {
            std::istringstream ss(part);
            double a, b; int f;
            if (TrimWhitespace(part).empty()) continue;
            if (!(ss >> a >> b >> f))
            {
                configErrors.push_back(key + " interval '" + part + "' does not parse (expected: start end factor [mode])");
                continue;
            }
            if (f < 1 || b <= a)
            {
                configErrors.push_back(key + " interval '" + part + "' is empty or reversed");
                continue;
            }
            GridRefinement r;
            r.active = true; r.start = a; r.end = b; r.factor = f;
            std::string mode;
            if (ss >> mode)
            {
                for (char &c : mode) c = (char)std::tolower((unsigned char)c);
                r.mode = (mode == "ramp") ? REFINE_RAMP
                       : (mode == "ramp_back" || mode == "rampback") ? REFINE_RAMP_BACK
                       : ((mode == "graded" || mode == "symmetric") ? REFINE_SYMMETRIC
                                                                   : REFINE_HARD);
            }
            out.push_back(r);
        }
    }

    static void loadConfig(SIMULATION_CONFIG& config, const std::string& filename) {
        SIMULATION.CONFIG_NAME = PathStem(filename);
        PRESSURE_SETTINGS.text.clear();
        if (!loadFromFile(filename)) std::exit(1);
        PRESSURE_SETTINGS.text = pressureBlock.empty() ? LegacyPressureConfig(filename) : pressureBlock;
        DIMENSION = getInt("DIMENSION", 3);
        GPU_ACCELERATION = getInt("GPU_ACCELERATION",0);
#ifndef NVIDIA_GPU
        if (GPU_ACCELERATION)
            std::cerr << "GPU_ACCELERATION ignored: built without NVIDIA_GPU=1\n";
        GPU_ACCELERATION = false;
#endif
        THREAD_COUNT = getInt("THREAD_COUNT",4);
        {
            // An unknown TYPE ends the process: a run with no solver would step nothing and still exit 0.
            // The methods must be registered first; with no TYPE key the first one runs.
            const std::string typeKey = getString("TYPE", MethodOf(0).key);
            SIM_TYPE = MethodFromKey(typeKey);
            if (SIM_TYPE < 0)
            {
                std::cerr << "ERROR: invalid key TYPE = '" << typeKey << "' in " << filename
                          << ". Registered methods:";
                for (int i = 0; i < METHOD_COUNT; i++) std::cerr << ' ' << METHODS[i].key;
                std::cerr << std::endl;
                std::exit(1);
            }
        }

        ADAPTATIVE_TIMESTEP = getInt("ADAPTATIVE_TIMESTEP",0);
        MAX_CFL =  getDouble("MAX_CFL",1.0);

        // A second load in the same process (the setup screen's harness, a restart) replaces the grids
        // rather than leaking the previous pair.
        delete SIMULATION.GRID_SOL;
        delete SIMULATION.GRID_ANT;
        SIMULATION.GRID_SOL = new MAC();
        SIMULATION.GRID_ANT = new MAC();
        SIMULATION.domain = Domain();

        SIMULATION.domain.x0 = getDouble("DOMAIN_X0", config.domain.x0);
        SIMULATION.domain.xf = getDouble("DOMAIN_XF", config.domain.xf);
        SIMULATION.domain.y0 = getDouble("DOMAIN_Y0", config.domain.y0);
        SIMULATION.domain.yf = getDouble("DOMAIN_YF", config.domain.yf);
        SIMULATION.domain.z0 = getDouble("DOMAIN_Z0", config.domain.z0);
        SIMULATION.domain.zf = getDouble("DOMAIN_ZF", config.domain.zf);

        SIMULATION.dt = getDouble("TIME_STEP", config.dt);
        SIMULATION.dtConfigured = SIMULATION.dt;
        SIMULATION.RE = getDouble("REYNOLDS_NUMBER", config.RE);
        SIMULATION.EPS = getDouble("VISCOSITY", 0.01);
        SIMULATION.RHO = getDouble("DENSITY",1.0);
        SIMULATION.GRID_SIZE = getInt("GRID_SIZE", config.GRID_SIZE);
        SIMULATION.ExportPath = getString("EXPORT_BASE_PATH", "Exports");
        SIMULATION.CHARACTERISTIC_LENGTH = getDouble("CHARACTERISTIC_LENGTH", config.CHARACTERISTIC_LENGTH);
        // The second length of the 3D reference area; absent means L1 again (a square area).
        SIMULATION.CHARACTERISTIC_LENGTH_2 = getDouble("CHARACTERISTIC_LENGTH_2", SIMULATION.CHARACTERISTIC_LENGTH);
        SIMULATION.MEAN_VELOCITY = getDouble("MEAN_VELOCITY", config.MEAN_VELOCITY);

        // Every settings struct reads its OWN keys. A method's come from its own block, through the
        // table it registered, so this file names none of them; the rest are shared.
        ConfigValues sharedValues;
        {
            ConfigValues &values = sharedValues;
            values.values = parameters;
            ReadMethodSettings(methodBlocks, values);
            SIMULATION.particleOptions.ReadFrom(values);
            ReadPressureSettings(PRESSURE_SETTINGS.text, values);
            SIMULATION.stop.ReadFrom(values);
            SIMULATION.exports.ReadFrom(values);
            SIMULATION.checkpoint.ReadFrom(values);
        }

        // MUST be read BEFORE the grids are built: MAC::BuildSpacing reads these to size the
        // spacing tables, and refinement changes the cell count.
        SIMULATION.grid.nonUniform = getInt("NON_UNIFORM_GRID", 0) != 0;
        {
            // Several intervals per axis, separated by ';': REFINE_X = 0.1 0.3 2 graded ; 0.6 0.9 2
            // Modes: hard | ramp | ramp_back | graded.
            ReadRefinement("REFINE_X", SIMULATION.grid.refineX);
            ReadRefinement("REFINE_Y", SIMULATION.grid.refineY);
            ReadRefinement("REFINE_Z", SIMULATION.grid.refineZ);
        }
        SIMULATION.f.u = getDouble("Fu",0.0);
        SIMULATION.f.v = getDouble("Fv",0.0);
        SIMULATION.f.w = getDouble("Fw",0.0);
        const std::string sceneFile = getString("SCENE_FILE", "");
        RefuseInvalidConfig(filename, sharedValues);

        SIMULATION.GRID_SOL->InitializeGrid(SIMULATION.domain,DIMENSION == 3? false:true);
        SIMULATION.GRID_ANT->InitializeGrid(SIMULATION.domain,DIMENSION == 3? false:true);

        SIMULATION.dh = (SIMULATION.GRID_ANT->dh);
        SIMULATION.Nx = SIMULATION.GRID_ANT->Nx;
        SIMULATION.Ny = SIMULATION.GRID_ANT->Ny;
        SIMULATION.Nz = SIMULATION.GRID_ANT->Nz;

        // The scene IS the geometry and the boundary conditions. The solvers only ever see the
        // six function pointers, so binding the scene behind them is the whole switch-over.
        bool haveScene = false;
        SIMULATION.sceneError.clear();
        if (!sceneBlock.empty())
        {
            if (!sceneFile.empty())
                std::cerr << "WARNING: config has both an embedded scene block and SCENE_FILE = "
                          << sceneFile << "; the embedded block is used." << std::endl;
            haveScene = SCENE.LoadText(sceneBlock, filename);
        }
        else if (!sceneFile.empty())
        {
            haveScene = SCENE.Load(sceneFile);
        }

        if (!haveScene)
        {
            if (!sceneBlock.empty() || !sceneFile.empty())
            {
                std::cerr << "ERROR: " << SCENE.error << std::endl;
                SIMULATION.sceneError = SCENE.error;
            }

            // No scene, or one that did not parse: an EMPTY scene, i.e. a closed box (an unmentioned face is a
            // no-slip wall).
            std::cerr << (sceneBlock.empty() && sceneFile.empty()
                              ? "Note: config declares no scene; running a closed box."
                              : "       falling back to a closed box.")
                      << std::endl;
            haveScene = SCENE.LoadText("", filename);
        }

        // The registry survives InitializeGrid, so a config loaded after another one must not
        // inherit its fields.
        SIMULATION.GRID_SOL->ClearScalars();
        SIMULATION.GRID_ANT->ClearScalars();

        {
            std::cout << SCENE.Summary() << std::endl;

            // A `scalar` line in the scene is a field on both grids, registered in declaration
            // order so a handle means the same field on each.
            for (const SceneDesc::Scene::ScalarSpec &spec : SCENE.scalars)
            {
                SIMULATION.GRID_SOL->RegisterScalar(spec.name, spec.transport);
                SIMULATION.GRID_ANT->RegisterScalar(spec.name, spec.transport);
            }

            if (DIMENSION == 3)
            {
                SIMULATION.SolidMaskFunction = SCENE_SOLID_MASK;
                SIMULATION.VelocityBoundaryFunction = SCENE_VELOCITY;
                SIMULATION.PressureBoundaryFunction = SCENE_PRESSURE;
                SIMULATION.BoundaryKindFunction = SCENE_BOUNDARY_KIND;
                SIMULATION.GRID_SOL->SetSolidMask(SIMULATION.SolidMaskFunction);
                SIMULATION.GRID_ANT->SetSolidMask(SIMULATION.SolidMaskFunction);
                SIMULATION.GRID_SOL->ClearFlowFields();
                SIMULATION.GRID_ANT->ClearFlowFields();
            }
            else
            {
                SIMULATION.SolidMaskFunction2D = SCENE_SOLID_MASK_2D;
                SIMULATION.VelocityBoundaryFunction2D = SCENE_VELOCITY_2D;
                SIMULATION.PressureBoundaryFunction2D = SCENE_PRESSURE_2D;
                SIMULATION.BoundaryKindFunction = SCENE_BOUNDARY_KIND;
                SIMULATION.GRID_SOL->SetSolidMask(SIMULATION.SolidMaskFunction2D);
                SIMULATION.GRID_ANT->SetSolidMask(SIMULATION.SolidMaskFunction2D);
                SIMULATION.GRID_SOL->ClearFlowFields();
                SIMULATION.GRID_ANT->ClearFlowFields();
            }

            std::cout << "Simulation config finished!" << std::endl;
        }
    }
};
#endif
