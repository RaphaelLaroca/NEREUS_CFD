// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.


#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <ostream>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <limits>

// A build without OpenMP (OPENMP=0, or a Mac with no libomp) runs serially.
#ifdef _OPENMP
#include <omp.h>
#else
inline void omp_set_num_threads(int) {}
#endif

// The release, printed by --help, --build-info, the headless banner and the window title.
inline constexpr const char *NEREUS_VERSION = "1.0";

struct Vec3 { double u; double v; double w; };
struct Vec2 { double u; double v; };

struct Domain { double x0; double xf; double y0; double yf; double z0; double zf; };
struct Domain2D { double x0; double xf; double y0; double yf; };

// How a particle answers a surface it has just been pushed out of.
enum ContactResponse
{
    CONTACT_SLIP = 0,  // remove the inward normal component, keep the tangential motion
    CONTACT_NOSLIP,    // stop dead
    CONTACT_BOUNCE     // reflect the normal component, scaled by the restitution
};

// Which form of the viscous traction the 3D force telemetry reports; Hybrid matches 2D (Schaefer 2D-1).
enum class AeroStressForm { Gradient, Deformation, Hybrid };
constexpr AeroStressForm AERO_STRESS_FORM_3D = AeroStressForm::Hybrid;

// Volume averages over the fluid, per unit mass: kinetic energy and enstrophy.
struct FlowIntegrals
{
    double kineticEnergy = 0.0;
    double enstrophy = 0.0;
    double fluidVolume = 0.0;
};

// A point probe from the scene (`probe <name> <quantity> at=(x,y,z)`), one telemetry column each.
// `quantity` is the text as written: p, u, v, w, speed, level, or a declared scalar's name.
enum class ProbeQuantity { Pressure, U, V, W, Speed, Level, Top, Scalar };
struct Probe
{
    std::string name;
    std::string quantity;
    ProbeQuantity kind = ProbeQuantity::Pressure;
    double at[3] = {0.0, 0.0, 0.0};
    double radius = 0.0;   // a real sensor's sensing radius; 0 samples the point
};

// One streamline in physical coordinates, with the speed at each point.
struct Streamline
{
    std::vector<double> x, y, speed;
};

// Lengths in metres: separation between neighbouring lines, integration step along a line.
struct StreamlineSettings
{
    double separation = 0.0;
    double step = 0.0;
    int maxPoints = 4000;
    double minSpeed = 1e-9;
    // Seeds and lines are kept inside [x0, x1] x [y0, y1] (clipped to the domain): the part of the field on screen.
    // Left empty (x1 <= x0) it is the whole domain.
    double x0 = 0.0, x1 = 0.0, y0 = 0.0, y1 = 0.0;
};

// 3D streamlines are seeded from a source, not spread through the volume (a filled volume hides itself).
enum StreamSeedSource { STREAM_SEED_PLANE = 0, STREAM_SEED_LINE = 1, STREAM_SEED_SPHERE = 2 };

struct Streamline3D
{
    std::vector<double> x, y, z, speed;
};

// Plane: count x count seeds on the plane normal to planeAxis (0 x, 1 y, 2 z) at planePosition, over the grid's
// extent. Line: count seeds from p0 to p1. Sphere: count seeds on the sphere surface. Lengths in metres.
struct Streamline3DSettings
{
    int source = STREAM_SEED_PLANE;
    int planeAxis = 0;
    double planePosition = 0.0;
    int count = 12;
    double p0[3] = {0.0, 0.0, 0.0};
    double p1[3] = {0.0, 0.0, 0.0};
    double centre[3] = {0.0, 0.0, 0.0};
    double radius = 0.1;
    double step = 0.0;
    int maxPoints = 4000;
    double minSpeed = 1e-9;
};

// Largest |u_d| h_d / nu over the faces between two non-solid cells, with the location it sits at.
// component 0/1/2 = u/v/w; valid is false when nu <= 0 or there is no grid.
struct CellReynolds
{
    double value = 0.0;
    int component = -1;
    double x = 0.0, y = 0.0, z = 0.0;
    bool valid = false;
};



// How the transport solver differences a registered scalar's convective term. Two, and deliberately
// only two, ADI with central differencing, or ADI with first-order upwinding.
enum class TransportScheme { Central = 0, Upwind = 1 };




//just a name for the scheme name, used in the UI
inline const char *TransportSchemeName(TransportScheme s)
{ return (s == TransportScheme::Central) ? "central" : "upwind"; }

// Which form a scalar's sweeps discretise: div(u phi) (Flux, conserves the total) or u.grad(phi)
// (Advective, keeps a uniform field uniform under the directional split).

enum class TransportForm { Flux = 0, Advective = 1 };

inline const char *TransportFormName(TransportForm f)
{ return (f == TransportForm::Flux) ? "flux" : "advective"; }

// Who advances a scalar. Adi: the shared transport solver. None: the grid stores it and sets its ring every step, and
// the method (a turbulence model, say) updates the values itself.
enum class ScalarSolver { Adi = 0, None = 1 };

inline const char *ScalarSolverName(ScalarSolver s)
{ return (s == ScalarSolver::Adi) ? "adi" : "none"; }

// Per-field transport: d(phi)/dt + conv = div(epsilon grad phi) - gamma phi + S, with S the scene's
// `source` expression. Defaults are pure advection.

struct ScalarTransport
{
    TransportScheme scheme = TransportScheme::Upwind;
    TransportForm form = TransportForm::Flux;
    double epsilon = 0.0;   // diffusivity, m^2/s, >= 0
    double gamma = 0.0;     // reactivity, 1/s: decay when positive, growth when negative
    ScalarSolver solver = ScalarSolver::Adi;
};

//comparison
inline bool SameTransport(const ScalarTransport &a, const ScalarTransport &b)
{
    return a.scheme == b.scheme && a.form == b.form && a.epsilon == b.epsilon && a.gamma == b.gamma &&
           a.solver == b.solver;
}



// ----- the scene block inside a config ----- The scene is embedded between these two markers.
// A separate file (SCENE_FILE) is still read, as a fallback for the test fixtures.
inline const char *SCENE_BLOCK_BEGIN = "SCENE_BEGIN";
inline const char *SCENE_BLOCK_END   = "SCENE_END";
//same thing here, old fallback
// ----- the pressure solver block inside the config ----- AMGX key=value lines plus the eigen_* keys.
inline const char *PRESSURE_BLOCK_BEGIN = "PRESSURE_BEGIN";
inline const char *PRESSURE_BLOCK_END   = "PRESSURE_END";

// `<scalar>_clamp_negative=1` on a boundary line clamps that scalar's prescribed value at zero.
inline const std::string SCALAR_CLAMP_SUFFIX = "_clamp_negative";

// The setup screen remembers the last preset picked or saved in configs/<this>, across launches.
inline const char *LAST_PRESET_FILE = ".last_preset";

// Cell types: what the solvers do with a cell. EMPTY is air or a p = 0 outlet, PRESSURE a known non-zero Dirichlet
// pressure, INFLOW a prescribed inlet, OUTFLOW a continuity outflow.
#define FLUID_CELL  0
#define SOLID_CELL  1
#define EMPTY_CELL  2
#define INFLOW_CELL 3
#define PRESSURE_CELL 4
#define OUTFLOW_CELL 5

// A ring cell whose velocity is known before the pressure solve (an inlet or a continuity outflow): Neumann to the
// pressure, no flux shift through it.
inline bool IsVelocityBoundaryCell(int t) { return t == INFLOW_CELL || t == OUTFLOW_CELL; }

// What a domain face does -- the scene's boundary types, and what MAC::EnforceBoundaryConditions applies on the ring
// cells each one owns. The cell type is what the solvers see (the pressure matrix, the projection, the sweeps); the
// kind says how the ring's values are set:
//   BC_NOSLIP          wall, velocity 0, Neumann p                                   -> SOLID_CELL
//   BC_SLIP            wall, normal velocity 0, tangential zero-gradient, Neumann p  -> SOLID_CELL
//   BC_MOVING          wall sliding along itself: tangential prescribed, normal 0    -> SOLID_CELL
//   BC_INLET           velocity prescribed, Neumann p (p= only sets the level)       -> INFLOW_CELL
//   BC_PRESSURE_OUTLET p prescribed (0 unless p= is given), velocity zero-gradient   -> EMPTY_CELL / PRESSURE_CELL
//   BC_OUTFLOW         velocity zero-gradient, its flux corrected to what the inlets bring in; Neumann p
//                      (the velocity is known, so the pressure solve treats it as an inlet) -> OUTFLOW_CELL
enum BoundaryKind { BC_NOSLIP = 0, BC_SLIP, BC_MOVING, BC_INLET, BC_PRESSURE_OUTLET, BC_OUTFLOW, BC_KIND_COUNT };

#include "MAC.h"

// A particle's kinematic state, shared by every particle method; method-specific fields live in the method.
struct Particle
{
    bool isActive = false;
    double x = 0.0, y = 0.0, z = 0.0;
    double u = 0.0, v = 0.0, w = 0.0;
};

// The run's particles. `data is sized once by the method's Initialize (its size is the capacity), and
// the first `count` entries are live. B
struct ParticleSet
{
    std::vector<Particle> data;
    int count = 0;
    // Per-particle density, for a method that carries one between steps (WCSPH); empty otherwise. Sized like `data`
    // and written to the checkpoint when it is not empty.
    std::vector<double> density;
};


// One refined interval on one axis. Deliberately minimal, a base cell whose centre falls inside
// [start, end] is split into `factor` equal sub-cells, everything else keeps dh
enum RefineMode
{
    REFINE_HARD = 0,
    REFINE_RAMP = 1,
    REFINE_SYMMETRIC = 2,
    REFINE_RAMP_BACK = 3
};

struct GridRefinement
{
    bool active = false;
    double start = 0.0;
    double end = 0.0;
    int factor = 1;
    int mode = REFINE_HARD;
};

// A config's KEY = value pairs, so a settings struct can read itself without knowing the file format.
// Structs are kept plain and public; only MAC hides its storage.
struct ConfigValues
{
    std::map<std::string, std::string> values;
    // Every key some reader asked for, and every value that did not parse. The run-time loader turns a key nobody
    // asked for, or any error, into a refusal; the setup screen keeps the fallback and shows its own checks.
    mutable std::set<std::string> used;
    mutable std::vector<std::string> errors;

    bool Has(const std::string &key) const { used.insert(key); return values.find(key) != values.end(); }

    std::string Text(const std::string &key, const std::string &fallback) const
    {
        used.insert(key);
        std::map<std::string, std::string>::const_iterator it = values.find(key);
        return (it == values.end()) ? fallback : it->second;
    }

    double Number(const std::string &key, double fallback) const
    {
        used.insert(key);
        std::map<std::string, std::string>::const_iterator it = values.find(key);
        if (it == values.end()) return fallback;
        double x;
        if (ParseNumber(it->second, x)) return x;
        errors.push_back(key + " = '" + it->second + "' is not a number");
        return fallback;
    }

    int Integer(const std::string &key, int fallback) const
    {
        used.insert(key);
        std::map<std::string, std::string>::const_iterator it = values.find(key);
        if (it == values.end()) return fallback;
        int x;
        if (ParseInteger(it->second, x)) return x;
        errors.push_back(key + " = '" + it->second + "' is not an integer");
        return fallback;
    }

    bool Flag(const std::string &key, bool fallback) const { return Integer(key, fallback ? 1 : 0) != 0; }

    // The whole text must be the number (surrounding blanks allowed): stod alone would read "1abc" as 1.
    static bool ParseNumber(const std::string &text, double &x)
    {
        try
        {
            size_t used = 0;
            x = std::stod(text, &used);
            return text.find_first_not_of(" \t\r", used) == std::string::npos && std::isfinite(x);
        }
        catch (...) { return false; }
    }
    static bool ParseInteger(const std::string &text, int &x)
    {
        try
        {
            size_t used = 0;
            x = std::stoi(text, &used);
            return text.find_first_not_of(" \t\r", used) == std::string::npos;
        }
        catch (...) { return false; }
    }
};

// ---- a settings struct described as data: SettingField rows give each field's offset, key and widget,
// so reading, writing and the default UI walk one table. ENUM fields are stored as int.
enum SettingKind { SETTING_INT, SETTING_DOUBLE, SETTING_BOOL, SETTING_ENUM };

inline constexpr double SETTING_NO_MIN = -1e300;
inline constexpr double SETTING_NO_MAX = 1e300;

struct SettingField
{
    const char *key;                 // the config key, "ALPHA"
    const char *label;               // what the setup screen calls it
    const char *help;                // tooltip; "" for none
    SettingKind kind;
    size_t offset;                   // offsetof(the struct, the field)
    double defaultValue;             // an ENUM's default is its index
    double minValue, maxValue;       // INT: slider range when both finite; DOUBLE: clamp
    const char *const *enumKeys;     // ENUM: the file spelling of each value, in enum order
    const char *const *enumLabels;   // ENUM: the UI spelling
    int enumCount;
    const char *unit = nullptr;      // shown after the value on the setup screen; nullptr for none
};

constexpr SettingField IntSetting(const char *key, const char *label, const char *help, size_t offset,
                                  int def, double lo = SETTING_NO_MIN, double hi = SETTING_NO_MAX)
{
    return {key, label, help, SETTING_INT, offset, (double)def, lo, hi, nullptr, nullptr, 0};
}
constexpr SettingField DoubleSetting(const char *key, const char *label, const char *help, size_t offset,
                                     double def, double lo = SETTING_NO_MIN, double hi = SETTING_NO_MAX)
{
    return {key, label, help, SETTING_DOUBLE, offset, def, lo, hi, nullptr, nullptr, 0};
}
constexpr SettingField BoolSetting(const char *key, const char *label, const char *help, size_t offset, bool def)
{
    return {key, label, help, SETTING_BOOL, offset, def ? 1.0 : 0.0, 0.0, 1.0, nullptr, nullptr, 0};
}
constexpr SettingField EnumSetting(const char *key, const char *label, const char *help, size_t offset, int def,
                                   const char *const *keys, const char *const *labels, int count)
{
    return {key, label, help, SETTING_ENUM, offset, (double)def, 0.0, (double)(count - 1), keys, labels, count};
}
// The same field with the unit the setup screen shows after its value: WithUnit(DoubleSetting(...), "m").
constexpr SettingField WithUnit(SettingField f, const char *unit)
{
    f.unit = unit;
    return f;
}

// Any field as a number. INT and ENUM go through memcpy, so an enum member with an int underlying
// type is read without an aliasing cast.
inline double GetSettingValue(const void *base, const SettingField &f)
{
    const char *p = (const char *)base + f.offset;
    if (f.kind == SETTING_DOUBLE) { double v; std::memcpy(&v, p, sizeof v); return v; }
    if (f.kind == SETTING_BOOL)   { bool v;   std::memcpy(&v, p, sizeof v); return v ? 1.0 : 0.0; }
    int v; std::memcpy(&v, p, sizeof v); return (double)v;
}

inline void SetSettingValue(void *base, const SettingField &f, double value)
{
    char *p = (char *)base + f.offset;
    if (f.kind == SETTING_DOUBLE) { std::memcpy(p, &value, sizeof value); return; }
    if (f.kind == SETTING_BOOL)   { const bool v = value != 0.0; std::memcpy(p, &v, sizeof v); return; }
    const int v = (int)std::lround(value);
    std::memcpy(p, &v, sizeof v);
}

inline const SettingField *FindSettingField(const SettingField *fields, int n, const std::string &key)
{
    for (int i = 0; i < n; i++)
        if (key == fields[i].key) return &fields[i];
    return nullptr;
}

// Every field back to its default. Members the table does not list are left alone.
inline void ResetSettings(const SettingField *fields, int n, void *base)
{
    for (int i = 0; i < n; i++)
        SetSettingValue(base, fields[i], fields[i].defaultValue);
}

// A key the config does not carry means the default, not whatever the last config left there.
inline void ReadSettings(const SettingField *fields, int n, void *base, const ConfigValues &v)
{
    ResetSettings(fields, n, base);
    for (int i = 0; i < n; i++)
    {
        const SettingField &f = fields[i];
        if (!v.Has(f.key)) continue;
        switch (f.kind)
        {
            case SETTING_INT:    SetSettingValue(base, f, v.Integer(f.key, (int)f.defaultValue)); break;
            case SETTING_DOUBLE: SetSettingValue(base, f, v.Number(f.key, f.defaultValue)); break;
            case SETTING_BOOL:   SetSettingValue(base, f, v.Flag(f.key, f.defaultValue != 0.0) ? 1.0 : 0.0); break;
            case SETTING_ENUM:
            {
                const std::string text = v.Text(f.key, "");
                int found = -1;
                for (int e = 0; e < f.enumCount; e++)
                    if (text == f.enumKeys[e]) found = e;
                if (found < 0)
                {
                    std::string allowed;
                    for (int e = 0; e < f.enumCount; e++) allowed += std::string(e ? ", " : "") + f.enumKeys[e];
                    v.errors.push_back(std::string(f.key) + " = '" + text + "' is not one of " + allowed);
                }
                else
                    SetSettingValue(base, f, found);
                break;
            }
        }
    }
}

inline void WriteSettings(const SettingField *fields, int n, const void *base, std::ostream &out)
{
    for (int i = 0; i < n; i++)
    {
        const SettingField &f = fields[i];
        const double x = GetSettingValue(base, f);
        out << f.key << " = ";
        if (f.kind == SETTING_DOUBLE) out << x;
        else if (f.kind == SETTING_ENUM)
        {
            const int e = (int)x;
            out << ((e >= 0 && e < f.enumCount) ? f.enumKeys[e] : f.enumKeys[(int)f.defaultValue]);
        }
        else out << (int)x;
        out << "\n";
    }
}

// What a method's settings panel or check may read of the shared sections (g, H, dh, dt, ...) without
// owning them.
struct SettingsContext
{
    int dimension = 2;
    double dh = 0.0;                          // base spacing; 0 for a meshless method
    double density = 1.0, viscosity = 0.0;
    double timeStep = 0.0;
    bool adaptiveTimestep = false;
    double x0 = 0.0, xf = 1.0, y0 = 0.0, yf = 1.0, z0 = 0.0, zf = 1.0;
    double fu = 0.0, fv = 0.0, fw = 0.0;       // body force
    double meanVelocity = 1.0, characteristicLength = 0.1;
};

// When to stop a run on its own, instead of watching it and hitting Pause. STOP_TIME stop once the
// simulation clock reaches tFinal.
enum StopCondition
{
    STOP_NEVER = 0,
    STOP_TIME = 1,
    STOP_RESIDUAL = 2
};

// STOP_CONDITION = NEVER | TIME | RESIDUAL in the config; an unknown key reads as NEVER.
inline int StopConditionFromKey(const std::string &key)
{
    if (key == "TIME") return STOP_TIME;
    if (key == "RESIDUAL") return STOP_RESIDUAL;
    return STOP_NEVER;
}

inline const char *StopConditionKey(int condition)
{
    if (condition == STOP_TIME) return "TIME";
    if (condition == STOP_RESIDUAL) return "RESIDUAL";
    return "NEVER";
}

//the structure to the halt condition, has functions to read from and write to, as all configs,
//but it has a rearm function, that basiclaly just resets it
struct StoppingRule
{
    int condition = STOP_NEVER;
    double tFinal = 1.0;
    double residualTol = 1e-6;
    bool exportOnStop = true;

    // Set by the manager when the rule fires, so the panel can say why it stopped and the
    // export runs exactly once.
    bool triggered = false;
    bool exported = false;
    std::string reason;

    void Rearm()
    {
        triggered = false;
        exported = false;
        reason.clear();
    }

    void ReadFrom(const ConfigValues &v)
    {
        *this = StoppingRule();
        const std::string key = v.Text("STOP_CONDITION", "NEVER");
        condition = StopConditionFromKey(key);
        if (key != StopConditionKey(condition))
            std::cerr << "Warning: STOP_CONDITION '" << key << "' is not NEVER, TIME or RESIDUAL. Using NEVER."
                      << std::endl;
        tFinal = v.Number("STOP_TIME", tFinal);
        residualTol = v.Number("STOP_RESIDUAL", residualTol);
        exportOnStop = v.Flag("STOP_EXPORT", exportOnStop);
    }

    void WriteTo(std::ostream &f) const
    {
        f << "STOP_CONDITION = " << StopConditionKey(condition) << "\n";
        f << "STOP_TIME = " << tFinal << "\n";
        f << "STOP_RESIDUAL = " << residualTol << "\n";
        f << "STOP_EXPORT = " << (exportOnStop ? 1 : 0) << "\n";
    }
};

// n steps of tFinal/n land a hair either side of tFinal in floating point: within this counts as there,
// so a run never takes one step past STOP_TIME.
inline double StopTimeEpsilon(double tFinal)
{
    return 1e-9 * std::max(1.0, std::abs(tFinal));
}

// The step to take from `clock`: dt, or what is left to tFinal when dt would pass it, so a time rule ends ON tFinal.
// A step that lands within StopTimeEpsilon is left alone, so a dt that divides tFinal is never touched.
inline double ClipStepToStopTime(const StoppingRule &r, double clock, double dt)
{
    if (r.condition != STOP_TIME || r.triggered) return dt;
    const double remaining = r.tFinal - clock;
    const double eps = StopTimeEpsilon(r.tFinal);
    if (remaining > eps && dt > remaining + eps) return remaining;
    return dt;
}

// Does the rule fire, given the state a completed step produced? Pure, so the manager and its test call it.
inline bool StoppingRuleFires(const StoppingRule &r, double time, double residual,  int iteration, std::string &why)
{
    if (r.condition == STOP_NEVER || r.triggered) return false;

    if (r.condition == STOP_TIME && time >= r.tFinal - StopTimeEpsilon(r.tFinal))
    {
        why = "reached t = " + std::to_string(time) +
              " (target " + std::to_string(r.tFinal) + ")";
        return true;
    }
    // iteration > 1 because the first step compares against the initial field, which is an
    // arbitrarily large or small difference depending on how the case starts.
    if (r.condition == STOP_RESIDUAL && iteration > 1 && residual <= r.residualTol)
    {
        why = "residual " + std::to_string(residual) +
              " fell below " + std::to_string(r.residualTol);
        return true;
    }
    return false;
}

// Set by the Quit button. main's loop turns it into a window close so shutdown follows the
// normal path rather than a hard exit.
inline bool QUIT_REQUESTED = false;

// ---- the step pipeline: each solver registers its steps from its Initialize; the manager runs them in
// ascending order, ties in registration order ----
class MAC; //foward def

using SimulationStepFunction = void (*)(MAC *gridAnt, MAC *gridSol, ParticleSet *particles, double time);

// A step's own work count for the step it just ran (a linear solver's iterations). It exists so the manager can be
// told the count without asking any method for it: a step reports through it, and keeps nothing on SIMULATION.
using SimulationStepCounter = int (*)();

struct SimulationStep
{
    SimulationStepFunction function;
    std::string name;
    int order;
    SimulationStepCounter counter = nullptr;   // nullptr: the step has no count, reported as -1
};

//the step registry
inline std::vector<SimulationStep> SIMULATION_STEPS;


inline bool SimulationStepBefore(const SimulationStep &a, const SimulationStep &b)
{
    return a.order < b.order;
}


//register function that methods should call to register themselves
inline void RegisterSimulationStep(SimulationStepFunction function, const std::string &name, int order,
                                   SimulationStepCounter counter = nullptr)
{
    SIMULATION_STEPS.push_back({function, name, order, counter});
    std::stable_sort(SIMULATION_STEPS.begin(), SIMULATION_STEPS.end(), SimulationStepBefore);
}

//self explanatory
inline void ClearSimulationSteps()
{
    SIMULATION_STEPS.clear();
}


//now this is a big struct, but it contains all of the simulation telemetry, this feeds the UI plots.
struct SimulationTelemetry {
    std::vector<double> time;
    std::vector<double> div_sum;
    std::vector<double> advcfl;
    std::vector<double> diffcfl;
    std::vector<double> residual;
    std::vector<double> cell_reynolds;
    CellReynolds lastCellReynolds;

    // Wall seconds of each registered step, in pipeline order; stepNames[s] + "_timing" names the series.
    std::vector<std::string> stepNames;
    std::vector<std::vector<double>> stepTimings;
    // Each step's counter after it ran (-1 for a step with none); stepNames[s] + "_counter" names the series.
    std::vector<std::vector<double>> stepCounters;

    size_t max_samples = 50000;

    // The run's current state, what the viewer and a headless loop read between steps.
    unsigned int iteration = 0;
    double clock = 0.0;              // simulated time at the end of the last completed step
    double lastResidual = 0.0;
    double lastStepWallTime = 0.0;   // wall seconds of the whole last step
    double lastAdvectionCFL = 0.0;
    double lastDiffusionCFL = 0.0;
    std::vector<double> lastStepTimings;
    std::vector<double> lastStepCounters;

    // Starts one series per step, zero-filled to the samples already taken, so every series stays aligned.
    void SetSteps(const std::vector<SimulationStep> &steps)
    {
        stepNames.clear();
        for (const SimulationStep &step : steps)
            stepNames.push_back(step.name);
        stepTimings.assign(stepNames.size(), std::vector<double>(time.size(), 0.0));
        stepCounters.assign(stepNames.size(), std::vector<double>(time.size(), -1.0));
    }

    void Push(double t,
              double div,
              double advcfl_,
              double diffcfl_,
              double res,
              const std::vector<double> &timings,
              const CellReynolds &cellRe = CellReynolds())
    {
        Append(time, t);
        Append(div_sum, div);
        Append(advcfl, advcfl_);
        Append(diffcfl, diffcfl_);
        Append(residual, res);
        for (size_t s = 0; s < stepTimings.size(); s++)
            Append(stepTimings[s], s < timings.size() ? timings[s] : 0.0);
        for (size_t s = 0; s < stepCounters.size(); s++)
            Append(stepCounters[s], s < lastStepCounters.size() ? lastStepCounters[s] : -1.0);
        Append(cell_reynolds, cellRe.value);
        lastCellReynolds = cellRe;
    }

    void Append(std::vector<double> &v, double x)
    {
        v.push_back(x);
        if (v.size() > max_samples)
            v.erase(v.begin());
    }
};

//this is the aerodynamics telemetry, made for schafer.
struct AerodynamicInformation{
    std::vector<double> time;
    std::vector<double> Cd;
    std::vector<double> Cl;
    std::vector<double> LtoDRatio;
    std::vector<double> Cs;              // side force (z), 3D only -- empty in 2D
    std::vector<double> CdCV, ClCV;      // control-volume force (AeroForcesCV2D / AeroForcesCV3D)
};

//currently kinda obsolete with the probes!
struct DambreakInformation{
    std::vector<double> time;
    std::vector<double> fluidLevel;
    std::vector<double> fluidFrontPosition;

};
// The open telemetry file. Copying it gives a stream with no file open, so ExportSettings (and the setup
// screen holding one) stays copyable: the stream belongs to the run, not to the settings.
struct TelemetryStream : public std::ofstream
{
    TelemetryStream() = default;
    // Delegates rather than naming a base: std::basic_ios is a VIRTUAL base of ofstream, so a
    // copy ctor that lists std::ofstream() leaves the most-derived class not initialising it.
    TelemetryStream(const TelemetryStream &) : TelemetryStream() {}
    TelemetryStream &operator=(const TelemetryStream &) { return *this; }
};

//this huge struct has all the eport settings
struct ExportSettings {
    bool enabled = true;
    int telemetryInterval = 1;  // Export telemetry every N frames
    int gridInterval = 100;      // Export grid every N ITERATIONS (see gridEveryTime)

    // ---- frames at a constant RATE, not every N iterations ---- With ADAPTATIVE_TIMESTEP the step
    // size moves with the flow.
    bool gridEveryTime = false;
    double gridTimeInterval = 0.04;   // seconds of simulated time between frames (25 fps)
    double nextGridExportTime = 0.0;  // the deadline, always a multiple of the interval
    int lastExportedTelemetryFrame = 0;
    int lastExportedGridFrame = 0;
    int gridExportCounter = 1;  // Counter for grid file naming

    // Telemetry export options
    bool exportTime = true;
    bool exportDivSum = true;
    bool exportCFL = true;
    bool exportResidual = true;
    bool exportStepTimings = true;     // one <name>_timing column per registered step
    bool exportFlowIntegrals = true;   // kinetic energy and enstrophy

    // The two telemetry groups that cost a pass over the grid: the flag gates the computation as well as
    // the column and the panel. Off by default.
    bool aerodynamics = false;   // drag, lift, side force, Delta-p
    bool dambreak = false;       // the free surface's height and front position

    bool exportFluidHeight = true;
    bool exportFluidFront = true;

    // Aerodynamics export options
    bool exportCl = true;
    bool exportCd = true;
    bool exportLtoDRatio = true;

    // Grid export
    bool exportGridData = false;
    // Particle export
    bool exportParticleData = false;

    // Surface mesh export (MeshExport in MeshHandler.h): the liquid as a watertight PLY per frame,
    // for rendering. 3D particle methods only.
    bool exportSurfaceMesh = false;
    bool exportSurfaceSolids = true;  // the scene's obstacles, once, as their own PLY
    double surfaceVoxelScale = 0.75;  // lattice pitch, in particle spacings
    double surfaceRadiusScale = 1.0;  // per-particle radius, in particle spacings
    // R in the Zhu-Bridson weights, in particle spacings.
    double surfaceInfluenceScale = 2.0;
    int surfaceSmoothing = 2;         // Laplacian passes over the field
    bool surfaceSolidsWritten = false;
    std::string surfaceStatus;

    // The registered scalars as NumPy arrays plus a JSON sidecar per frame, in .../Volumetric/, for a
    // renderer's import script (Exports::WriteVolumetric). The .vtr still carries them too.
    bool exportVolumetric = false;
    std::string volumetricStatus;

    TelemetryStream telemetryFile;
    bool fileInitialized = false;
    // The CSV opens at its first row, not at startup, so a checkpoint loaded before it keeps the rows up to
    // the restored iteration. -1 once handled; set by Checkpoint::Load.
    std::string telemetryPath, telemetryHeader;
    long long telemetryResumeIteration = -1;

    // ---- keys: everything the panel can set, so a headless run is configured by the file alone. The runtime
    // members below (deadlines, counters, the open file) are state and are not written.
    void ReadFrom(const ConfigValues &v)
    {
        *this = ExportSettings();   // a key the config does not carry means the default
        enabled = v.Flag("EXPORT_ENABLED", enabled);
        telemetryInterval = v.Integer("EXPORT_TELEMETRY_INTERVAL", telemetryInterval);
        gridInterval = v.Integer("EXPORT_GRID_INTERVAL", gridInterval);
        gridEveryTime = (v.Text("EXPORT_FRAME_MODE", "ITERATIONS") == "TIME");
        gridTimeInterval = v.Number("EXPORT_FRAME_INTERVAL", gridTimeInterval);

        exportGridData = v.Flag("EXPORT_GRID", exportGridData);
        exportParticleData = v.Flag("EXPORT_PARTICLES", exportParticleData);
        exportSurfaceMesh = v.Flag("EXPORT_SURFACE", exportSurfaceMesh);
        exportSurfaceSolids = v.Flag("EXPORT_SURFACE_SOLIDS", exportSurfaceSolids);
        surfaceVoxelScale = v.Number("EXPORT_SURFACE_VOXEL_SCALE", surfaceVoxelScale);
        surfaceRadiusScale = v.Number("EXPORT_SURFACE_RADIUS_SCALE", surfaceRadiusScale);
        surfaceInfluenceScale = v.Number("EXPORT_SURFACE_INFLUENCE_SCALE", surfaceInfluenceScale);
        surfaceSmoothing = v.Integer("EXPORT_SURFACE_SMOOTHING", surfaceSmoothing);
        exportVolumetric = v.Flag("EXPORT_VOLUMETRIC", exportVolumetric);

        exportTime = v.Flag("EXPORT_COLUMN_TIME", exportTime);
        exportDivSum = v.Flag("EXPORT_COLUMN_DIVERGENCE", exportDivSum);
        exportCFL = v.Flag("EXPORT_COLUMN_CFL", exportCFL);
        exportResidual = v.Flag("EXPORT_COLUMN_RESIDUAL", exportResidual);
        exportStepTimings = v.Flag("EXPORT_COLUMN_STEP_TIMINGS", exportStepTimings);
        exportFlowIntegrals = v.Flag("EXPORT_COLUMN_FLOW_INTEGRALS", exportFlowIntegrals);
        aerodynamics = v.Flag("EXPORT_AERODYNAMICS", aerodynamics);
        dambreak = v.Flag("EXPORT_DAMBREAK", dambreak);
        exportFluidHeight = v.Flag("EXPORT_COLUMN_FLUID_HEIGHT", exportFluidHeight);
        exportFluidFront = v.Flag("EXPORT_COLUMN_FLUID_FRONT", exportFluidFront);
        exportCl = v.Flag("EXPORT_COLUMN_CL", exportCl);
        exportCd = v.Flag("EXPORT_COLUMN_CD", exportCd);
        exportLtoDRatio = v.Flag("EXPORT_COLUMN_LD_RATIO", exportLtoDRatio);
    }

    void WriteTo(std::ostream &f) const
    {
        f << "EXPORT_ENABLED = " << (enabled ? 1 : 0) << "\n";
        f << "EXPORT_TELEMETRY_INTERVAL = " << telemetryInterval << "\n";
        f << "EXPORT_GRID_INTERVAL = " << gridInterval << "\n";
        f << "EXPORT_FRAME_MODE = " << (gridEveryTime ? "TIME" : "ITERATIONS") << "\n";
        f << "EXPORT_FRAME_INTERVAL = " << gridTimeInterval << "\n";
        f << "EXPORT_GRID = " << (exportGridData ? 1 : 0) << "\n";
        f << "EXPORT_PARTICLES = " << (exportParticleData ? 1 : 0) << "\n";
        f << "EXPORT_SURFACE = " << (exportSurfaceMesh ? 1 : 0) << "\n";
        f << "EXPORT_SURFACE_SOLIDS = " << (exportSurfaceSolids ? 1 : 0) << "\n";
        f << "EXPORT_SURFACE_VOXEL_SCALE = " << surfaceVoxelScale << "\n";
        f << "EXPORT_SURFACE_RADIUS_SCALE = " << surfaceRadiusScale << "\n";
        f << "EXPORT_SURFACE_INFLUENCE_SCALE = " << surfaceInfluenceScale << "\n";
        f << "EXPORT_SURFACE_SMOOTHING = " << surfaceSmoothing << "\n";
        f << "EXPORT_VOLUMETRIC = " << (exportVolumetric ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_TIME = " << (exportTime ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_DIVERGENCE = " << (exportDivSum ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_CFL = " << (exportCFL ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_RESIDUAL = " << (exportResidual ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_STEP_TIMINGS = " << (exportStepTimings ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_FLOW_INTEGRALS = " << (exportFlowIntegrals ? 1 : 0) << "\n";
        f << "EXPORT_AERODYNAMICS = " << (aerodynamics ? 1 : 0) << "\n";
        f << "EXPORT_DAMBREAK = " << (dambreak ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_FLUID_HEIGHT = " << (exportFluidHeight ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_FLUID_FRONT = " << (exportFluidFront ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_CL = " << (exportCl ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_CD = " << (exportCd ? 1 : 0) << "\n";
        f << "EXPORT_COLUMN_LD_RATIO = " << (exportLtoDRatio ? 1 : 0) << "\n";
    }
};
// Integral diagnostics of the flow, one sample per step: the kinetic energy and the enstrophy, volume averaged
// per unit mass. Their budget (-dE/dt against 2 nu Z) is left to the post-processing.
struct FlowIntegralTelemetry
{
    std::vector<double> time;
    std::vector<double> kineticEnergy;
    std::vector<double> enstrophy;
    // dE/dt between consecutive samples. NaN for the first two: sample 0 is the initial field and sample 1 the first
    // step, whose change is the start-up transient rather than the flow's own rate.
    std::vector<double> energyRate;

    double fluidVolume = 0.0;
    double density = 1.0;
    bool available = false;
    size_t max_samples = 50000;
    size_t pushed = 0;   // samples since the last Clear, so trimming the front cannot shift the NaN prefix

    void Clear()
    {
        time.clear(); kineticEnergy.clear(); enstrophy.clear(); energyRate.clear();
        available = false;
        pushed = 0;
    }

    void Push(double t, double energy, double zeta, double volume, double rho)
    {
        double rate = std::numeric_limits<double>::quiet_NaN();
        if (pushed >= 2 && !time.empty() && t > time.back())
            rate = (energy - kineticEnergy.back()) / (t - time.back());
        pushed++;
        Append(time, t);
        Append(kineticEnergy, energy);
        Append(enstrophy, zeta);
        Append(energyRate, rate);
        fluidVolume = volume;
        density = rho;
        available = true;
    }

    void Append(std::vector<double> &v, double x)
    {
        v.push_back(x);
        if (v.size() > max_samples)
            v.erase(v.begin());
    }
};
// Checkpointing: the run's state in ONE file, overwritten each time; deliberately not an export.
struct CheckpointSettings {
    bool autoSave = false;          // off by default: it writes a file into the working directory
    int  interval = 250;            // iterations between auto-saves
    char path[256] = "checkpoint.ckpt";

    bool saveRequested = false;     // set by the UI, consumed by StepSimulation
    bool loadRequested = false;

    unsigned long long it = 0;      // mirror of the manager's clock, for a save while paused
    double time = 0.0;

    std::string status;             // what the last save or load did, for the panel
    bool statusIsError = false;

    void ReadFrom(const ConfigValues &v)
    {
        autoSave = v.Flag("CHECKPOINT_AUTOSAVE", false);
        interval = v.Integer("CHECKPOINT_INTERVAL", 250);
        const std::string p = v.Text("CHECKPOINT_PATH", "checkpoint.ckpt");
        std::snprintf(path, sizeof(path), "%s", p.c_str());
        saveRequested = false;
        loadRequested = false;
        status.clear();
        statusIsError = false;
    }

    void WriteTo(std::ostream &f) const
    {
        f << "CHECKPOINT_AUTOSAVE = " << (autoSave ? 1 : 0) << "\n";
        f << "CHECKPOINT_INTERVAL = " << interval << "\n";
        f << "CHECKPOINT_PATH = " << path << "\n";
    }
};


// ---- settings no single method owns, held by SIMULATION_CONFIG. A method's own settings live in its
// settings header and reach the registry through its row ----

// Shared by every particle method, because Particles.h reads them for all of them.
struct ParticleOptions
{
    // How a particle responds at a wall or an obstacle: 0 = free slip, 1 = no slip, 2 = bounce
    // with `restitution`.
    int contact = 0;
    double restitution = 0.3;
    // How far outside a solid surface a pushed-out particle is parked, in CELLS.
    double surfaceOffset = 0.01;
    // Seeding displaces each particle inside its own sub-cell by this fraction of the sub-cell
    // pitch. Measured to make a rendered surface WORSE, which is why it defaults to 0.
    double seedJitter = 0.0;

    static const char *ContactKey(int c) { return (c == CONTACT_NOSLIP) ? "NOSLIP" : (c == CONTACT_BOUNCE) ? "BOUNCE" : "SLIP"; }

    void ReadFrom(const ConfigValues &v)
    {
        *this = ParticleOptions();
        const std::string c = v.Text("PARTICLE_CONTACT", "SLIP");
        contact = (c == "NOSLIP") ? CONTACT_NOSLIP : (c == "BOUNCE") ? CONTACT_BOUNCE : CONTACT_SLIP;
        restitution = v.Number("PARTICLE_RESTITUTION", restitution);
        surfaceOffset = v.Number("PARTICLE_SURFACE_OFFSET", surfaceOffset);
        seedJitter = v.Number("PARTICLE_SEED_JITTER", seedJitter);
    }

    void WriteTo(std::ostream &f) const
    {
        f << "PARTICLE_CONTACT = " << ContactKey(contact) << "\n";
        if (contact == CONTACT_BOUNCE) f << "PARTICLE_RESTITUTION = " << restitution << "\n";
        f << "PARTICLE_SURFACE_OFFSET = " << surfaceOffset << "\n";
        f << "PARTICLE_SEED_JITTER = " << seedJitter << "\n";
    }
};


// The rectilinear grid as the config asks for it. Eulerian only: FLIP's particle-to-cell lookups
// are x/dh always which completely breaks it. Also std flip doesnt gel well with non-uniform grids.
struct GridConfig
{
    bool nonUniform = false;
    // Several intervals per axis; where they overlap the finest one wins.
    std::vector<GridRefinement> refineX;
    std::vector<GridRefinement> refineY;
    std::vector<GridRefinement> refineZ;
};

// The spacing tables MAC::BuildSpacing publishes. NOT settings: Scene.h has to map an index to a
// coordinate and must not include MAC.h, so the tables are mirrored here instead, a good control to have
struct GridSpacing
{
    bool nonUniform = false;
    std::vector<double> xf, yf, zf;   // face coordinates
    std::vector<double> xc, yc, zc;   // cell centres
    std::vector<double> dx, dy, dz;   // cell sizes
};


//the heart of the simulation, here we hold every information we need for a run.
struct SIMULATION_CONFIG{

    double dh;
    double dt;

    double RE;
    double EPS;
    double RHO = 1.0;

    int Nx;
    int Ny;
    int Nz;

    int GRID_SIZE = 16;

    // The TIME_STEP from the config: SIMULATION.dt is what the adaptive controller overwrites, so
    // the configured value has to be kept separately to serve as the cap.
    double dtConfigured = 1e-3;
    // The time the current step is solving FOR, which a prescribed pressure needs and MAC cannot
    // otherwise reach.
    double time = 0.0;

    Vec3 f;

    Domain domain;

    Vec3(*VelocityBoundaryFunction)(double, double, double,double);
    double(*PressureBoundaryFunction)(double, double, double,double);
    int(*SolidMaskFunction)(int,int,int);

    Vec2(*VelocityBoundaryFunction2D)(double, double, double);
    double(*PressureBoundaryFunction2D)(double, double, double);
    // The boundary kind of ring cell (i, j, k) and the face that owns it (0..5 = xmin xmax ymin ymax zmin zmax), from
    // the scene. Null without one: MAC then reads the kind off the cell type.
    int(*BoundaryKindFunction)(int, int, int, int &) = nullptr;
    int(*SolidMaskFunction2D)(int,int);

    MAC* GRID_SOL = nullptr;
    MAC* GRID_ANT = nullptr;
    ParticleSet particles;

    std::string ExportPath;
    // The config file this run was started from; its stem names the export folder.
    std::string CONFIG_NAME;
    // Non-empty when the config's scene failed to parse and the run fell back to a closed box. The GUI shows it and
    // carries on; a headless run refuses to start, since a batch job in the wrong geometry wastes the whole run.
    std::string sceneError;
    double CHARACTERISTIC_LENGTH = 0.1;
    double MEAN_VELOCITY = 1.0;
    // 3D force coefficients divide by CHARACTERISTIC_LENGTH * CHARACTERISTIC_LENGTH_2 (D * H for a wall-to-wall
    // cylinder, chord * span for a wing); 2D divides by CHARACTERISTIC_LENGTH alone.
    double CHARACTERISTIC_LENGTH_2 = 0.1;

    // Run control, set by the control panel or by a headless loop, read by the manager.
    bool running = false;
    bool stepOnce = false;

    // ---- the grouped settings ----
    ExportSettings exports;
    CheckpointSettings checkpoint;
    StoppingRule stop;
    AerodynamicInformation aero;
    DambreakInformation dambreak;
    FlowIntegralTelemetry flowIntegrals;
    GridConfig grid;
    GridSpacing spacing;
    ParticleOptions particleOptions;

    SimulationTelemetry telemetry;
};




// ---- the methods: ONE row each, registered by the program before ConfigReader::loadConfig (TYPE is looked
// up in it). The family below is what kind of method it is; every capability follows from it ----
enum class MethodFamily
{
    EULERIAN,    // the solution lives on the grid (ADI, and an explicit projection method)
    HYBRID,      // particles carry it, a grid does the projection (FLIP)
    LAGRANGIAN,  // particles are the whole discretisation, no grid at all (SPH)
    NONE         // no method selected: UNKNOWN_METHOD, what an unknown TYPE gets.
};

// The families, for grouping in the setup screen; a new family is one row here plus one case in
// FamilyCapabilities. NONE is absent: it is an error, not a method.
struct MethodFamilyInfo
{
    MethodFamily family;
    const char *label;
    const char *help;
};

inline constexpr MethodFamilyInfo FAMILY_TABLE[] = {
    {MethodFamily::EULERIAN, "Eulerian (grid)",
     "The solution lives on the MAC grid"},
    {MethodFamily::HYBRID, "Hybrid (particles + grid)",
     "Particles carry the flow, a grid solves the pressure"},
    {MethodFamily::LAGRANGIAN, "Lagrangian (meshless)",
     "Particles only, no grid"},
};

// What a method can do, one bit each, ellegant but hard to read, its just a bit flag
// 00000110 is a particle method, with a fixed region for example (doesnt ex)
enum MethodCapability : unsigned
{
    CAP_GRID           = 1u << 0,  // a MAC grid carries the solution: field views, .vtr export, grid diagnostics
    CAP_PARTICLES      = 1u << 1,  // there is a particle set to advect, draw and export
    CAP_RECTILINEAR    = 1u << 2,  // may run on a stretched grid
    CAP_BOUNDARY_CELLS = 1u << 3,  // the domain border is a one-cell ring of boundary conditions
    CAP_FIXED_REGION   = 1u << 4,  // the fluid region does not move (no free surface), so the matrix stands
    CAP_SCALARS        = 1u << 5,  // registered scalars are advected by a transport solver
    CAP_PRESSURE_SOLVER = 1u << 6, // solves its pressure with the shared PPE solver (Methods/PPESolver): the
                                   // setup screen draws the Pressure section only for a method that says so
};

// What a family can offer its members. A family added to the enum with no case here gets NOTHING.
constexpr unsigned FamilyCapabilities(MethodFamily f)
{
    switch (f)
    {
        case MethodFamily::EULERIAN:
            return CAP_GRID | CAP_RECTILINEAR | CAP_BOUNDARY_CELLS | CAP_FIXED_REGION | CAP_SCALARS | CAP_PRESSURE_SOLVER;
        case MethodFamily::HYBRID:
            return CAP_GRID | CAP_PARTICLES | CAP_BOUNDARY_CELLS | CAP_PRESSURE_SOLVER;
        case MethodFamily::LAGRANGIAN:
            return CAP_PARTICLES | CAP_FIXED_REGION;
        case MethodFamily::NONE:
            return 0;
    }
    return 0;
}

struct SimulationMethod
{
    const char *key;        // the config's TYPE value, and the checkpoint's solver tag
    const char *label;      // what the setup screen calls it
    const char *help;
    MethodFamily family;
    bool supports2D;
    bool supports3D;

    // What THIS method does; the family is a CEILING (effective set = caps & FamilyCapabilities(family)),
    // so e.g. a uniform-only Eulerian scheme leaves CAP_RECTILINEAR off its row.
    unsigned caps;

    void (*initialize)();                               // builds the solvers and registers the steps
    void (*dispose)();                                  // frees what initialize allocated; null when nothing to free

    // The method's settings struct (one global instance) and its table, written as the <KEY>_BEGIN ... _END
    // block. Null / 0 when the method has none.
    void *settings;
    const SettingField *settingFields;
    int settingCount;
    // Optional: drawSettings replaces the default widgets for the table; checkSettings adds the method's
    // refusals and notes to Validate.
    void (*drawSettings)(const SettingsContext &ctx);
    void (*checkSettings)(const SettingsContext &ctx, std::vector<std::string> &problems,
                          std::vector<std::string> &notes);

    // Questions to the method, if null, the simulation goes to default grid computations
    double (*stableTimestep)();                         // the method's own dt limit
    double (*maxVelocity)();                            // for the CFL
    double (*cflLength)();                              // the length scale that goes with it
    double (*particleSpacing)();                        // the seeding pitch, for the surface export
};

// What the method does, limited to what its kind can offer.
constexpr unsigned MethodCapabilities(const SimulationMethod &m)
{
    return m.caps & FamilyCapabilities(m.family);
}
constexpr bool MethodSupports(const SimulationMethod &m, unsigned cap)
{
    return (MethodCapabilities(m) & cap) != 0;
}

// The registry. Here we tell everything the method needs to know; indexed by the id registration
// hands back, so a lookup has no search in it.
inline constexpr int MAX_METHODS = 16;
inline SimulationMethod METHODS[MAX_METHODS] = {};
inline int METHOD_COUNT = 0;

// What a lookup that names nothing gets: no capability, null pointers, so the run visibly does nothing.
inline const SimulationMethod UNKNOWN_METHOD = {"UNKNOWN", "(no method)",
    "The config named a TYPE no method implements.", MethodFamily::NONE, false, false, 0,
    nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

// Adds a row and returns its id. A duplicate key, or caps the family cannot offer, is a programming
// error and stops the program: to give a KIND of method a new capability, widen FamilyCapabilities.
// The longest method key, terminator included: the checkpoint stores the key in a field this long.
constexpr int MAX_METHOD_KEY = 64;

inline int RegisterSimulationMethod(const SimulationMethod &m)
{
    if (m.key == nullptr || m.key[0] == '\0' || std::strlen(m.key) >= MAX_METHOD_KEY)
    {
        std::cerr << "ERROR: a method key must be 1 to " << MAX_METHOD_KEY - 1 << " characters long ('"
                  << (m.key ? m.key : "") << "')." << std::endl;
        std::exit(1);
    }
    for (int i = 0; i < METHOD_COUNT; i++)
        if (std::string(METHODS[i].key) == m.key)
        {
            std::cerr << "ERROR: method '" << m.key << "' is registered twice." << std::endl;
            std::exit(1);
        }
    if ((m.caps & ~FamilyCapabilities(m.family)) != 0)
    {
        std::cerr << "ERROR: method '" << m.key << "' claims capabilities its family does not offer "
                     "-- widen FamilyCapabilities first." << std::endl;
        std::exit(1);
    }
    if (METHOD_COUNT >= MAX_METHODS)
    {
        std::cerr << "ERROR: more than " << MAX_METHODS << " methods registered." << std::endl;
        std::exit(1);
    }
    METHODS[METHOD_COUNT] = m;
    return METHOD_COUNT++;
}

//just a simple getter; total, so an unknown id is the UNKNOWN row
inline const SimulationMethod &MethodOf(int id)
{
    return (id >= 0 && id < METHOD_COUNT) ? METHODS[id] : UNKNOWN_METHOD;
}

// The active method: an id into METHODS, -1 when none.
inline int SIM_TYPE = -1;

// The active method, and the questions the rest of the code asks about it. One indexed load each.
inline const SimulationMethod &ActiveMethod() { return MethodOf(SIM_TYPE); }
inline bool MethodSupports(int id, unsigned cap) { return MethodSupports(MethodOf(id), cap); }
inline bool MethodUsesGrid() { return MethodSupports(SIM_TYPE, CAP_GRID); }
inline bool MethodUsesParticles() { return MethodSupports(SIM_TYPE, CAP_PARTICLES); }
inline bool MethodSupportsRectilinear() { return MethodSupports(SIM_TYPE, CAP_RECTILINEAR); }
inline bool MethodUsesBoundaryCells() { return MethodSupports(SIM_TYPE, CAP_BOUNDARY_CELLS); }
inline bool MethodFixedFluidRegion() { return MethodSupports(SIM_TYPE, CAP_FIXED_REGION); }
inline bool MethodTransportsScalars() { return MethodSupports(SIM_TYPE, CAP_SCALARS); }

// The config's TYPE value -> the method's id. -1 when the key names nothing.
inline int MethodFromKey(const std::string &key)
{
    for (int i = 0; i < METHOD_COUNT; i++)
        if (key == METHODS[i].key) return i;
    return -1;
}

// The config block a method's settings live in: FLIP_BEGIN ... FLIP_END.
inline std::string MethodBlockBegin(const SimulationMethod &m) { return std::string(m.key) + "_BEGIN"; }
inline std::string MethodBlockEnd(const SimulationMethod &m) { return std::string(m.key) + "_END"; }

// The method whose block `line` opens, or -1. Only a registered method's key opens a block.
inline int MethodBlockOpenedBy(const std::string &line)
{
    for (int i = 0; i < METHOD_COUNT; i++)
        if (METHODS[i].settingCount > 0 && line == MethodBlockBegin(METHODS[i])) return i;
    return -1;
}

// Each method reads its settings from its own block; a config written before the blocks existed
// carries the same keys at top level, and those are read instead.
inline void ReadMethodSettings(const std::map<std::string, ConfigValues> &blocks, const ConfigValues &topLevel)
{
    for (int i = 0; i < METHOD_COUNT; i++)
    {
        const SimulationMethod &m = METHODS[i];
        if (!m.settings || m.settingCount <= 0) continue;
        std::map<std::string, ConfigValues>::const_iterator it = blocks.find(m.key);
        ReadSettings(m.settingFields, m.settingCount, m.settings, it != blocks.end() ? it->second : topLevel);
    }
}

// Every registered method's block, not only the active one's, so switching method in the setup
// screen and back does not lose settings.
inline void WriteMethodSettings(std::ostream &out)
{
    for (int i = 0; i < METHOD_COUNT; i++)
    {
        const SimulationMethod &m = METHODS[i];
        if (!m.settings || m.settingCount <= 0) continue;
        out << "\n# " << m.label << "\n" << MethodBlockBegin(m) << "\n";
        WriteSettings(m.settingFields, m.settingCount, m.settings, out);
        out << MethodBlockEnd(m) << "\n";
    }
}

// A setting of the ACTIVE method by key, for shared code that must not name the method; `fallback` when
// the method has no such field.
inline double ActiveMethodSetting(const char *key, double fallback)
{
    const SimulationMethod &m = ActiveMethod();
    const SettingField *f = m.settings ? FindSettingField(m.settingFields, m.settingCount, key) : nullptr;
    return f ? GetSettingValue(m.settings, *f) : fallback;
}

inline bool MethodSupportsDimension(int id, int dim)
{
    return (dim == 2) ? MethodOf(id).supports2D : MethodOf(id).supports3D;
}

//we should move these to the sim config struct later!
// Performance parameters
inline int DIMENSION = 3;
#ifdef NVIDIA_GPU
inline bool GPU_ACCELERATION = 1;
#else
inline bool GPU_ACCELERATION = 0;
#endif
inline int THREAD_COUNT = 4;
inline bool ADAPTATIVE_TIMESTEP = 0;
inline double MAX_CFL = 1.0;

// The adaptive timestep, from the fastest velocity on the grid.
inline double ComputeAdaptiveTimestep(double maxVelocity, double dh, double maxCFL,
                                      double dtConfigured)
{
    const double v = (maxVelocity > 1e-6) ? maxVelocity : 1e-6;
    const double dtAdvection = (maxCFL * dh) / v;
    return (dtAdvection < dtConfigured) ? dtAdvection : dtConfigured;
}

// Simulation parameters global declaration
inline SIMULATION_CONFIG SIMULATION;





//i had no idea where to put these so here they are

// Whether iteration `it` is an auto-save point.
inline double FrameExportEpsilon(double interval) { return 1e-9 * interval; }

// Is a export needed currenly?
inline bool FrameExportDue(unsigned long long it, double time, const ExportSettings &e)
{
    if (e.gridEveryTime)
    {
        if (!(e.gridTimeInterval > 0.0)) return false;
        // A clock that moved BACKWARDS -- a checkpoint load -- leaves a deadline far in the future,
        // and a run that then wrote nothing until it arrived would look like a broken export.
        if (e.nextGridExportTime > time + e.gridTimeInterval) return true;
        return time >= e.nextGridExportTime - FrameExportEpsilon(e.gridTimeInterval);
    }
    return e.gridInterval > 0 && (it % (unsigned long long)e.gridInterval) == 0;
}

// The next deadline is a MULTIPLE of the interval rather than "now + interval", so the frame times
// never drift: accumulating the interval from wherever each step happened to land would add the.
inline void AdvanceFrameExportClock(double time, ExportSettings &e)
{
    if (!(e.gridTimeInterval > 0.0)) return;
    // The SAME tolerance the predicate accepts.
    const double t = time + FrameExportEpsilon(e.gridTimeInterval);
    e.nextGridExportTime = (std::floor(t / e.gridTimeInterval) + 1.0) * e.gridTimeInterval;
}

//should we checkpoint the sim
inline bool ShouldAutoSaveCheckpoint(unsigned long long it, const CheckpointSettings &c)
{
    if (!c.autoSave || c.interval <= 0) return false;
    if (it == 0) return false; // step 0 is the initial condition, not a step's worth of work
    return (it % (unsigned long long)c.interval) == 0;
}

#endif
