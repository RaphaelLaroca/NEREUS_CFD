// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SETUP_SCREEN_H
#define SETUP_SCREEN_H

#include "Definitions.h"
#include "../../Methods/PPESolver/PressureSettings.h"
#include "Scene.h"

#include "MAC.h"
#include "../Utils.h"
#include "Exports.h"

#include <algorithm>
#include <cstdarg>
#include <ctime>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <filesystem>

// Pre-run configuration screen. The simulation is still driven entirely by the config file it writes.

class SimulationSetupUI
{
public:
    // ----- solver -----
    int dimension = 3;
    int method = 0;   // an id into METHODS (Definitions.h); 0 is the first registered

    // ----- domain and resolution -----
    double x0 = 0.0, xf = 1.0;
    double y0 = 0.0, yf = 1.0;
    double z0 = 0.0, zf = 1.0;
    int gridSize = 32;

    // ----- rectilinear (non-uniform) grid ----- Mirrors SIMULATION.grid.nonUniform / REFINE_*. A
    // base cell whose centre falls inside [start, end] is split into `factor` equal sub-cells.
    struct RefineSpec
    {
        bool active = true;
        double start = 0.0;
        double end = 1.0;
        int factor = 2;
        int mode = REFINE_HARD; // hard / ramp / graded
    };
    bool nonUniformGrid = false;
    std::vector<RefineSpec> refineX, refineY, refineZ;

    static const char *ModeName(int m);

    // A config value: trimmed, cut at a trailing '#', and at a stray ';' unless the key uses ';' itself.
    static void TrimConfigValue(std::string &s, bool keepSemicolons);

    static void ParseDouble(const std::string &v, double &dst) { try { dst = std::stod(v); } catch (...) {} }

    // True when `v` is a plain number and nothing else, as the Fu/Fv/Fw keys hold.
    static bool NumberOnly(const std::string &v, double &dst);
    static std::string NumberText(double v) { std::ostringstream o; o << v; return o.str(); }
    static void ParseInt(const std::string &v, int &dst) { try { dst = std::stoi(v); } catch (...) {} }
    static void ParseBool(const std::string &v, bool &dst) { try { dst = std::stoi(v) != 0; } catch (...) {} }

    static int IndexOfName(const std::string &v, const char *const *names, int n, int fallback);

    // The cell whose low face is the last one at or below c.
    static int FaceCellOf(const std::vector<double> &f, double c);

    static float CanvasX(double x, double x0, double L, float left, float width)
    {
        return left + (float)((x - x0) / L) * width;
    }

    // y is drawn upward, so the domain reads the way the simulation does.
    static float CanvasY(double y, double y0, double L, float top, float height)
    {
        return top + height - (float)((y - y0) / L) * height;
    }

    // A segment in the 3D preview's plot order (x, z, y).
    static void PushSegment(std::vector<double> &sx, std::vector<double> &sy, std::vector<double> &sz,
                            double ax, double ay, double az, double bx, double by, double bz);

    // Draw at most ~60 grid lines per axis.
    static size_t GridLineStride(size_t n) { return (std::max)((size_t)1, (n + 59) / 60); }

    void WriteRefineKey(std::ostream &f, const char *key, const std::vector<RefineSpec> &v) const;

    void DrawRefineAxis(const char *label, std::vector<RefineSpec> &v, double lo, double hi);

    // ----- fluid -----
    double viscosity = 1e-3;
    double density = 1.0;

    // ----- time stepping -----
    bool adaptiveTimestep = false;
    double timeStep = 1e-3;

    double maxCFL = 1.0;

    // ----- the shared settings: the panel edits the backend structs themselves through ReadFrom/WriteTo.
    // A method's settings are its own global instance, edited through its table. -----
    ParticleOptions particleOptions;
    StoppingRule stop;
    ExportSettings exports;
    CheckpointSettings checkpoint;

    // ----- pressure -----
#ifdef NVIDIA_GPU
    bool useGPU = true;
#else
    bool useGPU = false;
#endif
    int threadCount = 6;

    // ----- AMGX, the PRESSURE block -----
    int amgxSolver = 0;      // PCG, PBICGSTAB, FGMRES, GMRES, AMG
    int amgxMaxIters = 10000;
    double amgxTolerance = 1e-15;
    int amgxNorm = 1;        // L1, L2, LMAX  (AMGX rejects LINF)
    int amgxConvergence = 0; // RELATIVE_INI_CORE, ABSOLUTE, RELATIVE_MAX_CORE
    bool amgxUsePrecon = true;
    int amgxPreconAlg = 0;   // CLASSICAL, AGGREGATION
    int amgxCycle = 0;       // V, W, F
    int amgxPreIters = 1;
    int amgxPresweeps = 1;
    int amgxPostsweeps = 1;
    bool amgxSolveStats = false;
    bool amgxGridStats = false;
    bool amgxTimings = false;

    // ----- Eigen, the eigen_* keys of the PRESSURE block -----
    int eigenSolver = EIGEN_BICGSTAB;
    int eigenPreconditioner = EIGEN_PRECON_DIAGONAL;
    double eigenTolerance = 1e-9;
    int eigenMaxIters = 2000;

    // ----- export / naming -----
    char exportPath[256] = "Exports";
    double reynoldsForNaming = 1000.0;
    double characteristicLength = 0.1;
    double characteristicLength2 = 0.1; // 3D coefficients divide by L1 * L2 (a config without the key: L1)
    double meanVelocity = 1.0;

    // ----- scene editor ----- The editor keeps its own plain-data copy of the scene and
    // round-trips through the real parser: Apply writes the file and calls SCENE.Load on it.
    struct UiBoundary
    {
        int type = BC_NOSLIP; // a BoundaryKind
        // std::string, NOT char[128].
        std::string u = "0";
        std::string v = "0";
        std::string w = "0";
        std::string p = "0";
        bool hasPressure = false;
        bool clampNegative = false;
        // Dirichlet values for declared scalars, one slot per declared field and in the same order.
        std::vector<std::string> scalarValue;
        std::vector<char> scalarClamp;   // clamp that prescribed value at zero, same slots
    };
    struct UiRegion
    {
        int kind = 0; // solid, fluid, empty
        int prim = 0; // box, sphere, cylinder, plane, mesh, naca00XX
        double a[3] = {0, 0, 0};
        double b[3] = {1, 1, 1};
        double radius = 0.1;
        int axis = 2;
        bool invert = false;
        // mesh: the model file and its placement. `a` is where the model's bounding-box
        // centre lands; rotation (degrees, about x then y then z) and scale act about it.
        std::string meshFile;
        double rot[3] = {0, 0, 0};
        double scl[3] = {1, 1, 1};
        bool uniformScale = true; // editor-only: one field drives all three
        // Sign each query against the nearest connected COMPONENT and union them, or against the
        // whole file at once.
        bool splitComponents = true;
        // naca00XX: `a` is the pivot, `pivot` chords from the leading edge; `rot` as a mesh's; span 0 is unbounded.
        double chord = 1.0;
        int thicknessPercent = 12;
        double pivot = 0.5;
        double span = 0.0;
        bool closedTE = false;
    };

    // Scalars the scene declares. A transported field needs a name and a scheme before any boundary
    // can name it, so this is editor state of its own rather than a property of a boundary.
    struct UiScalar
    {
        std::string name = "d";
        int solver = 0;   // 0 the ADI transport solver, 1 none -- matches ScalarSolver
        int scheme = 1;   // 0 central, 1 upwind -- matches TransportScheme
        int form = 0;     // 0 flux, 1 advective -- matches TransportForm
        double epsilon = 0.0;   // diffusivity, m^2/s
        double gamma = 0.0;     // reactivity, 1/s
        std::string source;     // S(x, y, z, t); empty is none
    };
    std::vector<UiScalar> uiScalars;

    // The ONE body force, in m/s^2: a number (saved as Fu/Fv/Fw) or an expression in x y z t
    // (saved as the scene's `force` line). The scene's `initial` directive beside it.
    std::string forceU = "0", forceV = "0", forceW = "0";
    bool sceneHasInitial = false;
    std::string initU = "0", initV = "0", initW = "0";
    // The scene's `probe` lines, edited as they are parsed.
    std::vector<Probe> uiProbes;

    // Faces in the order they will be written; order is what resolves the corners.
    int faceOrder[6] = {0, 1, 4, 5, 2, 3}; // xmin xmax zmin zmax ymin ymax
    UiBoundary uiFaces[6];
    std::vector<UiRegion> uiRegions;
    std::string sceneStatus;
    bool sceneStatusIsError = false;

    // preview
    std::vector<double> previewData;
    int previewRows = 0, previewCols = 0;
    unsigned previewVersion = 0;   // bumped by every BuildPreview, so the drawing re-uploads its texture only then
    int previewPlane = 0; // 0 = XY, 1 = XZ, 2 = ZY
    int previewSlice = 0;
    bool previewDirty = true;
    // The slice preview's framing: re-set to the whole domain when the extent, plane or plot size
    // changes (or on Fit), kept otherwise so the wheel zoom and the drag pan survive.
    bool previewFit = true;
    double previewViewKey[7] = {0, 0, 0, 0, 0, 0, 0};

    // ----- named configurations ----- A preset is one saved config, configs/<name>.cfg, carrying its
    // scene and pressure solver blocks.
    static constexpr const char *PRESET_DIR = "configs";
    std::vector<std::string> presets;
    int presetIdx = -1;
    char presetName[96] = "";

    void RefreshPresets();

    // ----- models for `region ... mesh` ----- Every .stl under Models/, listed as the path the
    // scene file will carry.
    static constexpr const char *MODEL_DIR = "Models";
    std::vector<std::string> models;
    bool modelsScanned = false;

    void RefreshModels();

    std::string PresetPath(const std::string &name, const char *ext) const
    {
        return std::string(PRESET_DIR) + "/" + name + ext;
    }

    // The file this panel reads from and writes to, and the one Start runs. It follows the last
    // preset loaded or saved; until then it is simulation_config.txt in the working directory.
    std::string configPath = "simulation_config.txt";

    void UseFile(const std::string &cfg)
    {
        configPath = cfg;
    }

    // True until a preset is loaded or saved. The run would then export under the working
    // directory file's own name, so Start is held until the setup has a name.
    bool HasPreset() const { return configPath != "simulation_config.txt"; }

    // The last preset picked or saved from the panel, remembered across launches.
    static std::string LastPresetFile() { return std::string(PRESET_DIR) + "/" + LAST_PRESET_FILE; }

    void RememberPreset(const std::string &name);

    void ForgetPreset()
    {
        std::remove(LastPresetFile().c_str());
    }

    // Opens the remembered preset if it still exists. False leaves the working-directory pair.
    bool RestoreLastPreset();

    bool SavePreset(const std::string &name, std::string &err);

    void LoadPreset(const std::string &name);

    std::string statusMessage;
    bool statusIsError = false;

    SimulationSetupUI();

    // Compatibility, taken from the backend dispatch

    // Which dimensions a method offers, from its registered row (Definitions.h) -- the same one
    // ConfigReader reads, so the panel cannot carry a second copy of the rule.
    static bool MethodSupported(int dim, int m) { return MethodSupportsDimension(m, dim); }

    const char *MethodKey() const { return MethodOf(method).key; }

    // What KIND of method is selected. Every general rule below asks this rather than naming a
    // method, so an Eulerian scheme added later inherits the whole panel.
    MethodFamily Family() const { return MethodOf(method).family; }

    // Snap the current selections back into legal territory after a change upstream.
    void ReconcileSelections();

    // The backend gate, and it is a FAMILY property: only the Eulerian paths index cells through
    // the spacing tables. MAC::BuildSpacing asks the same question of the same table.
    bool RectilinearSupported() const { return MethodSupports(method, CAP_RECTILINEAR); }

    // The axis MAC::BuildAxis builds, plus the ring, so the cell count and the mesh preview are what the solver
    // will build. run_setup_ui_check asserts the two.
    static void BuildAxisPreview(double a0, double af, int nBase, double dh,
                                 const std::vector<RefineSpec> &refs, bool nu,
                                 std::vector<double> &face);

    // Face coordinates for both axes, refinement included.
    void DerivedAxes(std::vector<double> &fx, std::vector<double> &fy) const;

    // The face tables of the preview's three axes -- ACROSS, UP, and the one the slice FIXES -- for
    // the current dimension and plane.
    void PreviewAxes(std::vector<double> &hf, std::vector<double> &vf, std::vector<double> &ff) const;

    // How many slices the fixed axis has -- the REFINED count on a stretched grid, which is
    // not the base Nz/Ny/Nx the slider used to be sized off.
    int PreviewSliceCount() const;

    // Derived quantities, mirroring MAC::InitializeGrid so the preview is honest
    void DerivedGrid(double &dh, int &Nx, int &Ny, int &Nz) const;
    // Whether the grid the panel describes can be previewed: every extent positive, at least one interior cell along
    // each axis, and no more cells than the preview budget. `why` says what is wrong when it cannot.
    bool GridPreviewable(std::string &why) const;
    static constexpr double PREVIEW_CELL_BUDGET = 2.0e7;

    // Reading an existing config so the panel opens on what is already there
    void LoadFrom(const std::string &path);

    // The PRESSURE block's settings, so the panel opens on what the config already holds.
    void LoadPressureText(const std::string &text);

    // Scene editor: serialise, apply through the real parser, and preview
    static const char *FaceName(int f);
    static const char *BcName(int t)
    {
        return SceneDesc::Scene::BoundaryName(t);
    }
    static const char *RegionKindName(int k);
    static const char *PrimName(int k);

    // naca00XX: thickness, chord, pivot and placement. In 2D only the rotation about z applies.
    void DrawAirfoilRegionEditor(UiRegion &g, int nc);

    std::string SceneText() const;

    // Round-trips the editor state through the real parser, in memory. Anything the solver
    // would reject is reported here instead.
    bool ApplyScene();

    // Pulls a scene into the editor from the block embedded in a config. Expr keeps its
    // source text, so the expressions come back exactly as written.
    void LoadSceneTextIntoEditor(const std::string &text);

    // Everything after a successful parse.
    void PullSceneIntoEditor();

    // Rasterises one slice of the scene using the same evaluator the solver uses, on the
    // grid the current domain and grid size would produce.
    void BuildPreview();
    // The PRESSURE block: the AMGX settings, then the eigen_* keys for the CPU path. Both are always
    // written, so a preset keeps its GPU settings when saved on a machine that runs it on the CPU.
    std::string PressureBlockText() const;

    // Writes the config the backend reads, scene and pressure blocks included.
    bool WriteFiles(std::string &err, const std::string &simPath = "simulation_config.txt") const;

    // The velocity component along a face's own normal: u for xmin/xmax, v for ymin/ymax,
    // w for zmin/zmax.
    static int NormalAxisOf(int face) { return face / 2; }
    // What each section is allowed to draw
    bool ShowParticleOptions() const { return MethodSupports(method, CAP_PARTICLES); }
    // SPH is meshless: GRID_SIZE, the refinement intervals and the mesh preview are all
    // answers to questions it does not ask. Its resolution is the kernel radius.
    bool UsesGrid() const { return MethodSupports(method, CAP_GRID); }   // a meshless family has no grid
    // The body force as constants, when every component is a plain number (w is 0 in 2D).
    bool BodyForceConstant(double f[3]) const;
    bool BodyForceZero() const;

    // What a method's panel and check are told about the shared sections.
    SettingsContext Context() const;

    // The settings every particle method shares (ParticleOptions). The surface offset and the seed
    // jitter are about a particle and a grid cell, so a meshless method does not get them.
    void DrawParticleOptions();

    // The extents the geometry is written against. A scene names its regions in metres, so the
    // extents belong next to it: change them and the same file describes a different obstacle.
    void DrawDomainExtents();

    void DrawResolution();

    // The numbers behind the mesh drawing: counts, spacing per axis, and what the finest cell means for the run
    // (cell Reynolds number, CFL-limited step, viscous number).
    void DrawGridStats(const std::vector<double> &fx, const std::vector<double> &fy, const std::vector<double> &fz,
                       int Nx, int Ny, int Nz, double worstRatio) const;

    void RenderMeshPreview();

    // `region ... mesh`: model choice, placement, and what the grid will make of it Section 3.
    // Point probes: one telemetry column each, <name>_<quantity>.
    void DrawProbes();

    void DrawScalarFields();

    void DrawMeshRegionEditor(UiRegion &g);

    // 3D preview: domain box, the grid on three walls, the placed models, and the
    // voxelised solid the solver will actually see
    bool show3DModels = true;
    bool show3DVoxels = true;
    bool show3DFluid = true;
    bool show3DGrid = true;
    bool preview3DFit = true;
    double preview3DKey[6] = {0, 0, 0, 0, 0, 0};
    std::vector<double> voxX, voxY, voxZ; // surface solid cells, cell centres
    std::vector<double> fluX, fluY, fluZ; // surface cells of the `region fluid` liquid
    std::string voxKey;                   // what they were built from
    int sceneGeneration = 0;              // bumped whenever SCENE is reparsed

    // Solid cells with at least one non-solid face neighbour, excluding the domain ring. A filled
    // obstacle hides its own interior, so the surface layer is all worth drawing.
    void BuildVoxelPreview(bool nu);

    static size_t VoxelIndex(int i, int j, int k, int Nx, int Ny)
    {
        return ((size_t)k * Ny + i) * Nx + j;
    }

    // The liquid a `region fluid` seeds: interior cells that are not solid and whose centre the
    // scene calls fluid, reduced to the ones with a non-fluid face neighbour.
    void BuildFluidVoxelPreview(bool nu);

    void Render3DPreview();

    // The probes on the slice preview, in the coordinates of the plane on screen.
    void DrawProbeMarkers2D();

    void RenderScenePreview();

    void RenderSlicePreview();

    // The panel. Returns true once the user has committed and the files are on disk.
    bool Render();

private:
public:
    // Public so the headless checks can assert on what the panel refuses.
    void Validate(std::vector<std::string> &problems, std::vector<std::string> &notes) const;
};

#endif
