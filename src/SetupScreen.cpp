// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The setup screen's state and logic: loading, writing and validating a config, and the previews' data.
// Nothing here draws; the drawing half is src/UI/SimulationManagerUI.cpp.

#include "headers/Core/SetupScreen.h"

const char *SimulationSetupUI::ModeName(int m)
{
    return (m == REFINE_RAMP) ? "ramp"
         : (m == REFINE_RAMP_BACK) ? "ramp_back"
         : (m == REFINE_SYMMETRIC ? "graded" : "hard");
}

void SimulationSetupUI::TrimConfigValue(std::string &s, bool keepSemicolons)
{
    size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) { s.clear(); return; }
    size_t b = s.find_last_not_of(" \t");
    s = s.substr(a, b - a + 1);
    size_t h = s.find('#');
    if (h != std::string::npos) s = s.substr(0, h);
    if (!keepSemicolons)
    {
        size_t sc = s.find(';');
        if (sc != std::string::npos) s = s.substr(0, sc);
    }
    size_t b2 = s.find_last_not_of(" \t");
    if (b2 != std::string::npos) s = s.substr(0, b2 + 1);
}

bool SimulationSetupUI::NumberOnly(const std::string &v, double &dst)
{
    const std::string t = TrimWhitespace(v);
    if (t.empty()) return false;
    char *end = nullptr;
    const double d = std::strtod(t.c_str(), &end);
    if (end != t.c_str() + t.size()) return false;
    dst = d;
    return true;
}

int SimulationSetupUI::IndexOfName(const std::string &v, const char *const *names, int n, int fallback)
{
    for (int i = 0; i < n; i++)
        if (v == names[i]) return i;
    return fallback;
}

int SimulationSetupUI::FaceCellOf(const std::vector<double> &f, double c)
{
    if (f.size() < 2) return 0;
    int lo = 0, hi = (int)f.size() - 2;
    while (lo < hi)
    {
        const int mid = (lo + hi + 1) / 2;
        if (f[mid] <= c) lo = mid; else hi = mid - 1;
    }
    return lo;
}

void SimulationSetupUI::PushSegment(std::vector<double> &sx, std::vector<double> &sy, std::vector<double> &sz,
                        double ax, double ay, double az, double bx, double by, double bz)
{
    sx.push_back(ax); sy.push_back(az); sz.push_back(ay);
    sx.push_back(bx); sy.push_back(bz); sz.push_back(by);
}

void SimulationSetupUI::WriteRefineKey(std::ostream &f, const char *key, const std::vector<RefineSpec> &v) const
{
    if (!nonUniformGrid) return;
    bool first = true;
    for (const RefineSpec &r : v)
    {
        if (!r.active || r.factor <= 1 || r.end <= r.start) continue;
        if (first) { f << key << " = "; first = false; }
        else f << " ; ";
        f << r.start << " " << r.end << " " << r.factor << " " << ModeName(r.mode);
    }
    if (!first) f << "\n";
}

void SimulationSetupUI::RefreshPresets()
{
    presets.clear();
    namespace fs = std::filesystem;
    std::error_code ec;
    // Recursive: presets are grouped in subfolders (Ghia/, Schafer/, ...); a name carries its folder,
    // "Ghia/2d_ghia_re100", which is what PresetPath expects.
    for (fs::recursive_directory_iterator it(PRESET_DIR, ec), end; !ec && it != end; it.increment(ec))
    {
        try
        {
            std::error_code fe;
            if (!it->is_regular_file(fe)) continue;
            const std::string n = fs::relative(it->path(), PRESET_DIR).generic_string();
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".cfg") == 0)
                presets.push_back(n.substr(0, n.size() - 4));
        }
        catch (const std::exception &) {}   // a name outside the code page (Windows) cannot be listed
    }
    std::sort(presets.begin(), presets.end());
}

void SimulationSetupUI::RefreshModels()
{
    models.clear();
    modelsScanned = true;
    namespace fs = std::filesystem;
    std::error_code ec;
    for (fs::directory_iterator it(MODEL_DIR, ec), end; !ec && it != end; it.increment(ec))
    {
        try
        {
            std::string n = it->path().filename().string();
            std::string lower = n;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.size() > 4 && lower.compare(lower.size() - 4, 4, ".stl") == 0)
                models.push_back(std::string(MODEL_DIR) + "/" + n);
        }
        catch (const std::exception &) {}   // a name outside the code page (Windows) cannot be listed
    }
    std::sort(models.begin(), models.end());
}

void SimulationSetupUI::RememberPreset(const std::string &name)
{
    std::ofstream out(LastPresetFile());
    if (out.is_open()) out << name << "\n";
}

bool SimulationSetupUI::RestoreLastPreset()
{
    std::ifstream in(LastPresetFile());
    std::string name;
    if (!in.is_open() || !std::getline(in, name)) return false;
    name = TrimWhitespace(name);
    if (name.empty()) return false;
    std::ifstream probe(PresetPath(name, ".cfg"));
    if (!probe.is_open()) return false;
    LoadPreset(name);
    for (int i = 0; i < (int)presets.size(); i++)
        if (presets[i] == name) presetIdx = i;
    return true;
}

bool SimulationSetupUI::SavePreset(const std::string &name, std::string &err)
{
    if (name.empty()) { err = "Give the preset a name."; return false; }
    // The interface types UTF-8 and Windows opens files in its code page, so a name is kept to printable ASCII.
    for (unsigned char c : name)
        if (c < 0x20 || c > 0x7E)
        {
            err = "Use only ASCII letters, digits, spaces and - _ . / in a preset name.";
            return false;
        }
    std::error_code ec;
    // The preset's own folder (a name may carry one, "Ghia/..."); fine if it already exists.
    std::filesystem::create_directories(std::filesystem::path(PresetPath(name, ".cfg")).parent_path(), ec);
    // Validate only -- the scene goes INTO the .cfg, so a preset no longer depends on a
    // scenes/*.scene that another preset may be sharing.
    if (!ApplyScene()) { err = "scene: " + sceneStatus; return false; }
    if (!WriteFiles(err, PresetPath(name, ".cfg")))
        return false;
    UseFile(PresetPath(name, ".cfg"));
    return true;
}

void SimulationSetupUI::LoadPreset(const std::string &name)
{
    UseFile(PresetPath(name, ".cfg"));
    LoadFrom(configPath);
    std::snprintf(presetName, sizeof(presetName), "%s", name.c_str());
    statusMessage = "Loaded preset '" + name + "'.";
    statusIsError = false;
}

SimulationSetupUI::SimulationSetupUI()
{
    LoadFrom(configPath);
    RefreshPresets();
    RestoreLastPreset();
}

void SimulationSetupUI::ReconcileSelections()
{
    if (!MethodSupported(dimension, method))
        method = 0;

    // Same for the rectilinear grid: MAC::BuildSpacing refuses it outside 2D Eulerian,
    // so carrying the flag would write a config the solver silently ignores.
    if (nonUniformGrid && !RectilinearSupported())
        nonUniformGrid = false;
}

void SimulationSetupUI::BuildAxisPreview(double a0, double af, int nBase, double dh,
                             const std::vector<RefineSpec> &refs, bool nu,
                             std::vector<double> &face)
{
    face.clear();
    nBase -= 2;   // the interior; the ring is added outside it at the end, as MAC::AddRing does
    if (nBase <= 0 || dh <= 0.0)
        return;
    // MAC::BuildAxis itself, so the preview and the cell count are the solver's axis, not a copy of it.
    std::vector<GridRefinement> g;
    for (const RefineSpec &r : refs)
    {
        GridRefinement q;
        q.active = r.active; q.start = r.start; q.end = r.end; q.factor = r.factor; q.mode = r.mode;
        g.push_back(q);
    }
    std::vector<double> size, centre;
    MAC::BuildAxis(a0, af, nBase, dh, g, nu, size, face, centre);
    // The ring, one cell outside each end, as wide as the interior cell beside it.
    const double w0 = face[1] - face[0], w1 = face[face.size() - 1] - face[face.size() - 2];
    face.insert(face.begin(), face.front() - w0);
    face.push_back(face.back() + w1);
}

void SimulationSetupUI::DerivedAxes(std::vector<double> &fx, std::vector<double> &fy) const
{
    double dh; int Nx, Ny, Nz;
    DerivedGrid(dh, Nx, Ny, Nz);
    const bool nu = nonUniformGrid && RectilinearSupported();
    BuildAxisPreview(x0, xf, Nx, dh, refineX, nu, fx);
    BuildAxisPreview(y0, yf, Ny, dh, refineY, nu, fy);
}

void SimulationSetupUI::PreviewAxes(std::vector<double> &hf, std::vector<double> &vf, std::vector<double> &ff) const
{
    std::vector<double> fx, fy, fz;
    DerivedAxes(fx, fy);
    if (dimension == 3)
    {
        double dh; int Nx, Ny, Nz;
        DerivedGrid(dh, Nx, Ny, Nz);
        BuildAxisPreview(z0, zf, Nz, dh, refineZ, nonUniformGrid && RectilinearSupported(), fz);
    }
    if (dimension != 3 || previewPlane == 0) { hf = fx; vf = fy; ff = fz; }
    else if (previewPlane == 1)              { hf = fx; vf = fz; ff = fy; }
    else                                     { hf = fz; vf = fy; ff = fx; }
}

int SimulationSetupUI::PreviewSliceCount() const
{
    if (dimension != 3) return 1;
    std::vector<double> hf, vf, ff;
    PreviewAxes(hf, vf, ff);
    return (ff.size() > 1) ? (int)ff.size() - 1 : 1;
}

bool SimulationSetupUI::GridPreviewable(std::string &why) const
{
    why.clear();
    const double L[3] = {xf - x0, yf - y0, zf - z0};
    const char *axis[3] = {"X", "Y", "Z"};
    const int axes = (dimension == 3) ? 3 : 2;
    for (int a = 0; a < axes; a++)
        if (!(L[a] > 0.0)) { why = std::string(axis[a]) + " max must be larger than " + axis[a] + " min."; return false; }
    const double dh = (gridSize > 0) ? (std::min)(L[0], L[1]) / (double)gridSize : 0.0;
    if (!(dh > 0.0)) { why = "The grid size must be positive."; return false; }
    double cells = 1.0;
    for (int a = 0; a < axes; a++)
    {
        const double n = L[a] / dh;
        if (n < 1.0 - 1e-9)
        {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "%s is shorter than one cell (%.4g m against dh = %.4g m).", axis[a], L[a], dh);
            why = buf;
            return false;
        }
        cells *= n;
    }
    if (cells > PREVIEW_CELL_BUDGET)
    {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%.3g cells: too many to preview (the budget is %.3g).", cells, PREVIEW_CELL_BUDGET);
        why = buf;
        return false;
    }
    return true;
}

void SimulationSetupUI::DerivedGrid(double &dh, int &Nx, int &Ny, int &Nz) const
{
    const double Lx = xf - x0, Ly = yf - y0, Lz = zf - z0;
    // MAC sizes dh off the smaller of the x and y extents; z is deliberately left out
    // of that minimum, so a thin domain in z produces more cells than GRID_SIZE.
    const double minExtent = (std::min)(Lx, Ly);
    // GRID_SIZE interior cells across the shorter side; the ring adds one cell OUTSIDE each face.
    dh = (gridSize > 0) ? minExtent / (double)gridSize : 0.0;
    Nx = (dh > 0.0) ? MAC::InteriorCells(Lx, dh) + 2 : 0;
    Ny = (dh > 0.0) ? MAC::InteriorCells(Ly, dh) + 2 : 0;
    Nz = (dimension == 3 && dh > 0.0) ? MAC::InteriorCells(Lz, dh) + 2 : 1;
}

void SimulationSetupUI::LoadFrom(const std::string &path)
{
    std::ifstream in(path);
    if (!in.is_open())
        return;

    // Refinement is cleared up front rather than left to the parser.
    nonUniformGrid = false;
    characteristicLength2 = -1.0; // optional key: absent means L1 again, not "keep the last"
    // Absent Fu/Fv/Fw mean no force, as ConfigReader reads them -- not the last config's.
    double loadedForce[3] = {0.0, 0.0, 0.0};
    refineX.clear();
    refineY.clear();
    refineZ.clear();

    ConfigValues values;   // every KEY = value pair, for the settings structs to read themselves
    std::map<std::string, ConfigValues> methodBlocks;   // each method's block, by method key
    std::string embeddedScene, pressureBlock;
    bool inScene = false, inPressure = false, hasPressure = false;
    int inMethod = -1;

    std::string line;
    while (std::getline(in, line))
    {
        // The scene block is collected verbatim: its lines are not KEY = value, and the
        // value rules below (trailing '#', stray ';') would corrupt an expression.
        const std::string trimmed = TrimWhitespace(line);
        if (inMethod >= 0 && trimmed == MethodBlockEnd(METHODS[inMethod])) { inMethod = -1; continue; }
        if (inMethod < 0 && !inScene && !inPressure)
        {
            const int m = MethodBlockOpenedBy(trimmed);
            if (m >= 0) { inMethod = m; methodBlocks[METHODS[m].key]; continue; }
        }
        if (!inScene && !inPressure && trimmed == PRESSURE_BLOCK_BEGIN) { inPressure = hasPressure = true; continue; }
        if (inPressure)
        {
            if (trimmed == PRESSURE_BLOCK_END) inPressure = false;
            else pressureBlock += trimmed + "\n";
            continue;
        }
        if (!inScene && trimmed == SCENE_BLOCK_BEGIN) { inScene = true; continue; }
        if (inScene)
        {
            if (trimmed == SCENE_BLOCK_END) { inScene = false; continue; }
            embeddedScene += line;
            embeddedScene += '\n';
            continue;
        }

        if (line.empty() || line[0] == '#' || line[0] == '/')
            continue;
        std::istringstream ls(line);
        std::string key, value;
        if (!(std::getline(ls, key, '=') && std::getline(ls, value)))
            continue;

        TrimConfigValue(key, false);
        TrimConfigValue(value, key == "REFINE_X" || key == "REFINE_Y" || key == "REFINE_Z");
        if (key.empty() || value.empty())
            continue;

        // A method's block holds only that method's keys.
        if (inMethod >= 0)
        {
            methodBlocks[METHODS[inMethod].key].values[key] = value;
            continue;
        }
        values.values[key] = value;

        if (key == "DIMENSION") ParseInt(value, dimension);
        else if (key == "TYPE")
            {
                const int t = MethodFromKey(value);
                // The editor has to show something runnable, so an unrecognised TYPE falls
                // back to the first registered method rather than naming one here.
                method = (t < 0) ? 0 : t;
            }
        else if (key == "GPU_ACCELERATION") ParseBool(value, useGPU);
        else if (key == "THREAD_COUNT") ParseInt(value, threadCount);
        else if (key == "ADAPTATIVE_TIMESTEP") ParseBool(value, adaptiveTimestep);
        else if (key == "MAX_CFL") ParseDouble(value, maxCFL);
        else if (key == "DOMAIN_X0") ParseDouble(value, x0);
        else if (key == "DOMAIN_XF") ParseDouble(value, xf);
        else if (key == "DOMAIN_Y0") ParseDouble(value, y0);
        else if (key == "DOMAIN_YF") ParseDouble(value, yf);
        else if (key == "DOMAIN_Z0") ParseDouble(value, z0);
        else if (key == "DOMAIN_ZF") ParseDouble(value, zf);
        else if (key == "TIME_STEP") ParseDouble(value, timeStep);
        else if (key == "VISCOSITY") ParseDouble(value, viscosity);
        else if (key == "DENSITY") ParseDouble(value, density);
        else if (key == "GRID_SIZE") ParseInt(value, gridSize);
        else if (key == "NON_UNIFORM_GRID") ParseBool(value, nonUniformGrid);
        else if (key == "REFINE_X" || key == "REFINE_Y" || key == "REFINE_Z")
        {
            std::vector<RefineSpec> &out = (key == "REFINE_X") ? refineX
                                         : (key == "REFINE_Y") ? refineY : refineZ;
            out.clear();
            std::stringstream all(value);
            std::string part;
            while (std::getline(all, part, ';'))
            {
                std::istringstream rs(part);
                double a, b; int f;
                if (!(rs >> a >> b >> f) || f < 1 || b <= a) continue;
                RefineSpec r;
                r.active = true; r.start = a; r.end = b; r.factor = f;
                std::string m;
                if (rs >> m)
                    // Must accept every name ModeName can emit, or a saved interval
                    // silently reloads as `hard`.
                    r.mode = (m == "ramp") ? REFINE_RAMP
                           : (m == "ramp_back" || m == "rampback") ? REFINE_RAMP_BACK
                           : ((m == "graded" || m == "symmetric") ? REFINE_SYMMETRIC
                                                                 : REFINE_HARD);
                out.push_back(r);
            }
        }
        else if (key == "Fu") ParseDouble(value, loadedForce[0]);
        else if (key == "Fv") ParseDouble(value, loadedForce[1]);
        else if (key == "Fw") ParseDouble(value, loadedForce[2]);
        else if (key == "REYNOLDS_NUMBER") ParseDouble(value, reynoldsForNaming);
        else if (key == "CHARACTERISTIC_LENGTH") ParseDouble(value, characteristicLength);
        else if (key == "CHARACTERISTIC_LENGTH_2") ParseDouble(value, characteristicLength2);
        else if (key == "MEAN_VELOCITY") ParseDouble(value, meanVelocity);
        else if (key == "EXPORT_BASE_PATH")
        {
            std::snprintf(exportPath, sizeof(exportPath), "%s", value.c_str());
        }
    }
    // Each settings struct reads its own keys, and resets itself when the config carries none.
    // Every method reads its block (or, in a config older than the blocks, the top level).
    ReadMethodSettings(methodBlocks, values);
    particleOptions.ReadFrom(values);
    stop.ReadFrom(values);
    exports.ReadFrom(values);
    checkpoint.ReadFrom(values);

    ReconcileSelections();

    // A config that carries a scene has to open with that scene in the editor.
    if (characteristicLength2 < 0.0) characteristicLength2 = characteristicLength;
    forceU = NumberText(loadedForce[0]);
    forceV = NumberText(loadedForce[1]);
    forceW = NumberText(loadedForce[2]);
    // A config with no scene block is an empty scene -- a closed box, as the backend runs it --
    // never the last config's scene.
    LoadSceneTextIntoEditor(embeddedScene);
    // A scene `force` ADDS to Fu/Fv/Fw in the solver; the one field shows their sum.
    if (SCENE.loaded && SCENE.hasForce)
    {
        std::string *comp[3] = {&forceU, &forceV, &forceW};
        for (int a = 0; a < 3; a++)
            if (loadedForce[a] != 0.0)
                *comp[a] = "(" + *comp[a] + ") + " + NumberText(loadedForce[a]);
    }

    // A config without a PRESSURE block opens on the legacy file the backend would read for it.
    if (!hasPressure)
    {
        std::string legacy = LEGACY_PRESSURE_CONFIG;
        if (path.size() > 4 && path.compare(path.size() - 4, 4, ".cfg") == 0)
            legacy = path.substr(0, path.size() - 4) + ".pcfg";
        std::ifstream pin(legacy);
        if (pin.is_open())
            pressureBlock.assign(std::istreambuf_iterator<char>(pin), std::istreambuf_iterator<char>());
    }
    ReadPressureSettings(pressureBlock, values);
    LoadPressureText(pressureBlock);
}

void SimulationSetupUI::LoadPressureText(const std::string &text)
{
    std::istringstream in(text);
    const EigenSolverSettings es = ReadEigenSettings(text, dimension);
    eigenSolver = es.solver;
    eigenPreconditioner = es.preconditioner;
    eigenTolerance = es.tolerance;
    eigenMaxIters = es.maxIters;

    static const char *solvers[] = {"PCG", "PBICGSTAB", "FGMRES", "CG", "AMG"};
    static const char *norms[] = {"L1", "L2", "LMAX"};
    static const char *convs[] = {"RELATIVE_INI_CORE", "ABSOLUTE", "RELATIVE_MAX_CORE"};
    static const char *algs[] = {"CLASSICAL", "AGGREGATION"};
    static const char *cycles[] = {"V", "W", "F"};

    bool sawPrecon = false;
    std::string line;
    while (std::getline(in, line))
    {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        k = TrimWhitespace(k);
        v = TrimWhitespace(v);
        if (k.empty() || v.empty()) continue;

        if (k == "solver") amgxSolver = IndexOfName(v, solvers, 5, amgxSolver);
        else if (k == "max_iters") ParseInt(v, amgxMaxIters);
        else if (k == "tolerance") { try { amgxTolerance = std::stod(v); } catch (...) {} }
        else if (k == "norm") amgxNorm = IndexOfName(v, norms, 3, amgxNorm);
        else if (k == "convergence") amgxConvergence = IndexOfName(v, convs, 3, amgxConvergence);
        else if (k.rfind("preconditioner", 0) == 0) { sawPrecon = (v != "NOSOLVER"); }
        else if (k == "amg_solver:algorithm" || k == "algorithm")
            amgxPreconAlg = IndexOfName(v, algs, 2, amgxPreconAlg);
        else if (k == "amg_solver:cycle" || k == "cycle")
            amgxCycle = IndexOfName(v, cycles, 3, amgxCycle);
        else if (k == "amg_solver:presweeps" || k == "presweeps") ParseInt(v, amgxPresweeps);
        else if (k == "amg_solver:postsweeps" || k == "postsweeps") ParseInt(v, amgxPostsweeps);
        else if (k == "amg_solver:max_iters") ParseInt(v, amgxPreIters);
        else if (k == "print_solve_stats") ParseBool(v, amgxSolveStats);
        else if (k == "print_grid_stats") ParseBool(v, amgxGridStats);
        else if (k == "obtain_timings") ParseBool(v, amgxTimings);
    }
    amgxUsePrecon = sawPrecon;
}

const char *SimulationSetupUI::FaceName(int f)
{
    static const char *n[] = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    return n[f];
}

const char *SimulationSetupUI::RegionKindName(int k)
{
    static const char *n[] = {"solid", "fluid", "empty"};
    return n[k];
}

const char *SimulationSetupUI::PrimName(int k)
{
    static const char *n[] = {"box", "sphere", "cylinder", "plane", "mesh", "naca"};
    return n[k];
}

std::string SimulationSetupUI::SceneText() const
{
    std::ostringstream f;
    f << "# Written by the NEREUS setup screen.\n";
    f << "#\n";
    f << "# Boundary and region lines are applied in order and the LAST match wins, which\n";
    f << "# is what resolves the corners. Put a moving lid last to have it beat the side\n";
    f << "# walls; put no-slip walls last to have them beat an inlet.\n\n";
    f << "sampling center\n\n";   // the panel always samples cell centres; `corner` is for the old fixtures

    // Declared before the boundaries, because a boundary may only name a scalar that
    // already exists -- that is what lets the parser tell a scalar key from a typo.
    for (const UiScalar &sc : uiScalars)
    {
        if (sc.name.empty()) continue;
        f << "scalar " << sc.name;
        if (sc.solver == 1)   // no solver: scheme and form are the ADI solver's, so they are not written
            f << " solver=none";
        else
        {
            f << " scheme=" << (sc.scheme == 0 ? "central" : "upwind");
            // Only what differs from pure advection is written, so a plain declaration reads as before.
            if (sc.form == 1) f << " form=advective";
        }
        const std::streamsize old = f.precision(12);
        if (sc.epsilon != 0.0) f << " epsilon=" << sc.epsilon;
        if (sc.gamma != 0.0) f << " gamma=" << sc.gamma;
        f.precision(old);
        if (sc.source.find_first_not_of(" \t") != std::string::npos)
            f << " source=\"" << sc.source << "\"";
        f << "\n";
    }
    if (!uiScalars.empty()) f << "\n";

    for (int n = 0; n < 6; n++)
    {
        const int fi = faceOrder[n];
        if (dimension == 2 && (fi == 4 || fi == 5))
            continue; // zmin/zmax are meaningless in 2D
        const UiBoundary &b = uiFaces[fi];
        f << "boundary " << FaceName(fi) << " " << BcName(b.type);
        // An inlet prescribes the whole velocity, a moving wall its tangential part; nothing else takes one.
        if (b.type == BC_INLET || b.type == BC_MOVING)
        {
            const int nax = (b.type == BC_MOVING) ? NormalAxisOf(fi) : -1;
            if (nax != 0) f << " u=\"" << b.u << "\"";
            if (nax != 1) f << " v=\"" << b.v << "\"";
            if (dimension == 3 && nax != 2)
                f << " w=\"" << b.w << "\"";
            if (b.clampNegative && b.type == BC_INLET)
                f << " clamp_negative=1";
        }
        // A pressure outlet's p (0 when absent), or an inlet's pressure level.
        if (b.hasPressure && !b.p.empty() && (b.type == BC_INLET || b.type == BC_PRESSURE_OUTLET))
            f << " p=\"" << b.p << "\"";
        for (size_t sc = 0; sc < uiScalars.size() && sc < b.scalarValue.size(); sc++)
            if (!uiScalars[sc].name.empty() && !b.scalarValue[sc].empty())
            {
                f << " " << uiScalars[sc].name << "=\"" << b.scalarValue[sc] << "\"";
                if (sc < b.scalarClamp.size() && b.scalarClamp[sc])
                    f << " " << uiScalars[sc].name << SCALAR_CLAMP_SUFFIX << "=1";
            }
        f << "\n";
    }

    if (!uiRegions.empty())
        f << "\n";
    for (const UiRegion &r : uiRegions)
    {
        f << "region " << RegionKindName(r.kind) << " ";
        if (r.prim == 5)
        {
            char name[16];
            std::snprintf(name, sizeof(name), "naca00%02d", r.thicknessPercent);
            f << name;
        }
        else
            f << PrimName(r.prim);
        switch (r.prim)
        {
        case 0: // box
            f << " min=(" << r.a[0] << "," << r.a[1] << "," << r.a[2] << ")"
              << " max=(" << r.b[0] << "," << r.b[1] << "," << r.b[2] << ")";
            break;
        case 1: // sphere
            f << " center=(" << r.a[0] << "," << r.a[1] << "," << r.a[2] << ")"
              << " radius=" << r.radius;
            break;
        case 2: // cylinder
            f << " axis=" << (r.axis == 0 ? "x" : r.axis == 1 ? "y" : "z")
              << " center=(" << r.a[0] << "," << r.a[1] << "," << r.a[2] << ")"
              << " radius=" << r.radius;
            break;
        case 3: // plane
            f << " point=(" << r.a[0] << "," << r.a[1] << "," << r.a[2] << ")"
              << " normal=(" << r.b[0] << "," << r.b[1] << "," << r.b[2] << ")";
            break;
        case 4: // mesh -- always all three components: a model is a 3D object even
                // when a 2D run only cuts it, and a scale of 0 on z would be degenerate.
        {
            // A CAD model in millimetres wants a scale like 0.0025; the stream's default
            // six significant digits would round a fitted value, so widen it here.
            const std::streamsize old = f.precision(12);
            f << " file=\"" << r.meshFile << "\""
              << " center=(" << r.a[0] << "," << r.a[1] << "," << r.a[2] << ")"
              << " rotate=(" << r.rot[0] << "," << r.rot[1] << "," << r.rot[2] << ")"
              << " scale=(" << r.scl[0] << "," << r.scl[1] << "," << r.scl[2] << ")";
            if (!r.splitComponents) f << " components=whole";
            f.precision(old);
            break;
        }
        case 5: // naca00XX -- the thickness is in the name
        {
            const std::streamsize old = f.precision(12);
            f << " center=(" << r.a[0] << "," << r.a[1] << "," << r.a[2] << ")"
              << " chord=" << r.chord
              << " rotate=(" << r.rot[0] << "," << r.rot[1] << "," << r.rot[2] << ")";
            if (r.pivot != 0.5) f << " pivot=" << r.pivot;
            if (r.span > 0.0) f << " span=" << r.span;
            if (r.closedTE) f << " te=closed";
            f.precision(old);
            break;
        }
        }
        if (r.invert)
            f << " invert=1";
        f << "\n";
    }

    // These two go LAST and are emitted whenever they are set. Leaving them out is what
    // made saving an Ethier-Steinman or 3D Taylor-Green scene delete its initial field.
    if (sceneHasInitial)
    {
        f << "\ninitial u=\"" << initU << "\" v=\"" << initV << "\"";
        if (dimension == 3) f << " w=\"" << initW << "\"";
        f << "\n";
    }
    double constantForce[3];
    if (!BodyForceConstant(constantForce))
    {
        f << "\nforce u=\"" << forceU << "\" v=\"" << forceV << "\"";
        if (dimension == 3) f << " w=\"" << forceW << "\"";
        f << "\n";
    }
    // After the scalars, which a probe may name.
    if (!uiProbes.empty()) f << "\n";
    for (const Probe &pr : uiProbes)
    {
        if (pr.name.empty()) continue;
        const std::streamsize old = f.precision(12);
        f << "probe " << pr.name << " " << pr.quantity << " at=(" << pr.at[0] << "," << pr.at[1] << ","
          << pr.at[2] << ")";
        if (pr.radius > 0.0) f << " radius=" << pr.radius;
        f << "\n";
        f.precision(old);
    }
    return f.str();
}

bool SimulationSetupUI::ApplyScene()
{
    const std::string text = SceneText();
    if (!SCENE.LoadText(text))
    {
        sceneStatus = SCENE.error;
        sceneStatusIsError = true;
        return false;
    }
    sceneGeneration++;
    sceneStatus = "scene is valid";
    sceneStatusIsError = false;
    previewDirty = true;
    return true;
}

void SimulationSetupUI::LoadSceneTextIntoEditor(const std::string &text)
{
    if (!SCENE.LoadText(text))
    {
        sceneStatus = SCENE.error;
        sceneStatusIsError = true;
        return;
    }
    PullSceneIntoEditor();
    sceneStatus = "loaded from the config";
}

void SimulationSetupUI::PullSceneIntoEditor()
{
    sceneGeneration++;

    // Only a scene that carries a `force` line sets it: a constant force lives in Fu/Fv/Fw,
    // so the scene the panel writes has none and must not zero the field on Validate.
    if (SCENE.hasForce)
    {
        forceU = SCENE.forceU.source; forceV = SCENE.forceV.source; forceW = SCENE.forceW.source;
    }
    sceneHasInitial = SCENE.hasInitial;
    initU = SCENE.initU.source; initV = SCENE.initV.source; initW = SCENE.initW.source;
    uiProbes = SCENE.probes;

    // Scalars first: a face's values are stored by position in this list.
    uiScalars.clear();
    for (const SceneDesc::Scene::ScalarSpec &sc : SCENE.scalars)
    {
        UiScalar u;
        u.name = sc.name;
        u.solver = (sc.transport.solver == ScalarSolver::None) ? 1 : 0;
        u.scheme = (sc.transport.scheme == TransportScheme::Central) ? 0 : 1;
        u.form = (sc.transport.form == TransportForm::Advective) ? 1 : 0;
        u.epsilon = sc.transport.epsilon;
        u.gamma = sc.transport.gamma;
        u.source = sc.source ? sc.source->source : std::string();
        uiScalars.push_back(u);
    }

    for (int f = 0; f < 6; f++)
    {
        UiBoundary &b = uiFaces[f];
        b = UiBoundary();
        b.scalarValue.assign(uiScalars.size(), std::string());
        b.scalarClamp.assign(uiScalars.size(), 0);
        for (size_t i = 0; i < SCENE.faces[f].scalarNames.size(); i++)
        {
            const int at = SCENE.ScalarIndex(SCENE.faces[f].scalarNames[i]);
            if (at >= 0 && at < (int)b.scalarValue.size())
                b.scalarValue[(size_t)at] = SCENE.faces[f].scalarValues[i]->source;
        }
        for (const std::string &name : SCENE.faces[f].scalarClampNames)
        {
            const int at = SCENE.ScalarIndex(name);
            if (at >= 0 && at < (int)b.scalarClamp.size())
                b.scalarClamp[(size_t)at] = 1;
        }
        b.type = (int)SCENE.faces[f].type;
        b.u = SCENE.faces[f].u.source;
        b.v = SCENE.faces[f].v.source;
        b.w = SCENE.faces[f].w.source;
        // Empty when the face carries no p: a pressure outlet without one is p = 0 on an EMPTY_CELL.
        b.p = SCENE.faces[f].hasPressure ? SCENE.faces[f].pressure.source : "";

        b.hasPressure = SCENE.faces[f].hasPressure;
        b.clampNegative = SCENE.faces[f].clampNegativeU;
    }
    // Recover the declaration order; unspecified faces trail behind.
    std::vector<std::pair<int, int>> ord;
    for (int f = 0; f < 6; f++)
        ord.push_back({SCENE.faces[f].order < 0 ? 1000 + f : SCENE.faces[f].order, f});
    std::sort(ord.begin(), ord.end());
    for (int n = 0; n < 6; n++)
        faceOrder[n] = ord[n].second;

    uiRegions.clear();
    for (const SceneDesc::Primitive &p : SCENE.regions)
    {
        UiRegion r;
        r.kind = (int)p.region;
        r.prim = (int)p.kind;
        for (int a = 0; a < 3; a++) { r.a[a] = p.p0[a]; r.b[a] = p.p1[a]; }
        r.radius = p.radius;
        r.axis = p.axis;
        r.invert = p.invert;
        if (p.kind == SceneDesc::PRIM_MESH && p.mesh)
        {
            r.meshFile = p.mesh->file;
            for (int a = 0; a < 3; a++)
            { r.a[a] = p.mesh->xf.center[a]; r.rot[a] = p.mesh->xf.rotate[a]; r.scl[a] = p.mesh->xf.scale[a]; }
            r.uniformScale = (r.scl[0] == r.scl[1] && r.scl[1] == r.scl[2]);
            r.splitComponents = p.mesh->xf.splitComponents;
        }
        if (p.kind == SceneDesc::PRIM_AIRFOIL && p.airfoil)
        {
            for (int a = 0; a < 3; a++) r.rot[a] = p.rotate[a];
            r.chord = p.airfoil->chord;
            r.thicknessPercent = (int)std::lround(100.0 * p.airfoil->thickness);
            r.pivot = p.pivot;
            r.span = p.span;
            r.closedTE = p.airfoil->closedTE;
        }
        uiRegions.push_back(r);
    }

    sceneStatusIsError = false;
    previewDirty = true;
}

void SimulationSetupUI::BuildPreview()
{
    previewData.clear();
    previewRows = previewCols = 0;
    previewVersion++;
    if (!SCENE.loaded)
        return;
    // A half-typed extent can describe a grid with no cell along an axis (whose face table is then empty, and the
    // samples below would read past it) or billions of cells. Neither is drawn.
    std::string why;
    if (!GridPreviewable(why))
    {
        previewDirty = false;
        return;
    }

    // The evaluator reads the live SIMULATION geometry, so stand it up from the panel values
    // and put it back afterwards - nothing else is running during setup.
    const int sNx = SIMULATION.Nx, sNy = SIMULATION.Ny, sNz = SIMULATION.Nz;
    const double sdh = SIMULATION.dh;
    const Domain sdom = SIMULATION.domain;
    const int sdim = DIMENSION;

    double dh; int Nx, Ny, Nz;
    DerivedGrid(dh, Nx, Ny, Nz);
    if (Nx <= 0 || Ny <= 0 || dh <= 0.0)
        return;

    // Scene.h maps an index to a coordinate through the tables MAC publishes on SIMULATION.
    const bool sNU = SIMULATION.spacing.nonUniform;
    const std::vector<double> sXF = SIMULATION.spacing.xf, sYF = SIMULATION.spacing.yf;
    const std::vector<double> sXC = SIMULATION.spacing.xc, sYC = SIMULATION.spacing.yc;
    const std::vector<double> sDX = SIMULATION.spacing.dx, sDY = SIMULATION.spacing.dy;
    const std::vector<double> sZF = SIMULATION.spacing.zf, sZC = SIMULATION.spacing.zc,
                              sDZ = SIMULATION.spacing.dz;

    std::vector<double> pfx, pfy, pfz;
    DerivedAxes(pfx, pfy);
    if (dimension == 3)
        BuildAxisPreview(z0, zf, Nz, dh, refineZ, nonUniformGrid && RectilinearSupported(), pfz);
    const bool previewNU = nonUniformGrid && RectilinearSupported() &&
                           pfx.size() > 1 && pfy.size() > 1 &&
                           (dimension != 3 || pfz.size() > 1);
    // Published for a uniform grid too: Scene::CellTypeAt reads the tables whenever they exist, and with the ring
    // outside the domain they are the only honest index-to-coordinate map.
    if (pfx.size() > 1 && pfy.size() > 1 && (dimension != 3 || pfz.size() > 1))
    {
        SIMULATION.spacing.nonUniform = previewNU;
        SIMULATION.spacing.xf = pfx; SIMULATION.spacing.yf = pfy;
        const int rNx = (int)pfx.size() - 1, rNy = (int)pfy.size() - 1;
        SIMULATION.spacing.xc.resize(rNx); SIMULATION.spacing.dx.resize(rNx);
        for (int j = 0; j < rNx; j++)
        { SIMULATION.spacing.xc[j] = 0.5 * (pfx[j] + pfx[j + 1]); SIMULATION.spacing.dx[j] = pfx[j + 1] - pfx[j]; }
        SIMULATION.spacing.yc.resize(rNy); SIMULATION.spacing.dy.resize(rNy);
        for (int i = 0; i < rNy; i++)
        { SIMULATION.spacing.yc[i] = 0.5 * (pfy[i] + pfy[i + 1]); SIMULATION.spacing.dy[i] = pfy[i + 1] - pfy[i]; }
        SIMULATION.Nx = rNx; SIMULATION.Ny = rNy;
        if (dimension == 3)
        {
            // Scene::CellTypeAt maps k through GRID_ZF/ZC exactly as it maps i and j, so a 3D
            // preview that published only x and y asked the evaluator about a z axis belonging.
            SIMULATION.spacing.zf = pfz;
            const int rNz = (int)pfz.size() - 1;
            SIMULATION.spacing.zc.resize(rNz); SIMULATION.spacing.dz.resize(rNz);
            for (int k = 0; k < rNz; k++)
            { SIMULATION.spacing.zc[k] = 0.5 * (pfz[k] + pfz[k + 1]); SIMULATION.spacing.dz[k] = pfz[k + 1] - pfz[k]; }
        }
    }

    SIMULATION.domain = Domain{x0, xf, y0, yf, z0, zf};
    SIMULATION.dh = dh;
    SIMULATION.Nz = (previewNU && dimension == 3) ? (int)pfz.size() - 1 : Nz;
    if (!previewNU) { SIMULATION.Nx = Nx; SIMULATION.Ny = Ny; }
    DIMENSION = dimension;

    const int fixedMax = PreviewSliceCount();
    previewSlice = (std::max)(0, (std::min)(previewSlice, fixedMax - 1));

    // The across / up faces of whichever plane is on screen; on a stretched 3D grid these
    // are what the pixels are sampled against.
    std::vector<double> phf, pvf, pff;
    PreviewAxes(phf, pvf, pff);

    int hCells, vCells;
    if (dimension == 2)      { hCells = Nx; vCells = Ny; }
    else if (previewPlane == 0) { hCells = Nx; vCells = Ny; }
    else if (previewPlane == 1) { hCells = Nx; vCells = Nz; }
    else                        { hCells = Nz; vCells = Ny; }

    // Sized off the SMALLEST cell, not the cell count -- the same rule the running viewer uses.
    if (previewNU)
    {
        // One pixel per finest cell, matching the running viewer.
        hCells = ResampleResolutionFor(phf, 1, 1024);
        vCells = ResampleResolutionFor(pvf, 1, 1024);
    }

    previewRows = vCells;
    previewCols = hCells;
    previewData.assign((size_t)previewRows * previewCols, 0.0);

    for (int v = 0; v < vCells; v++)
        for (int h = 0; h < hCells; h++)
        {
            int i, j, k;
            // On a stretched grid the pixel is sampled at its own physical coordinate and
            // mapped to the cell that contains it.
            if (previewNU)
            {
                // Sample against the faces of the plane actually on screen.
                const double ph = phf.front() + (h + 0.5) * (phf.back() - phf.front()) / hCells;
                const double pv = pvf.front() + (v + 0.5) * (pvf.back() - pvf.front()) / vCells;
                int a = FaceCellOf(phf, ph);
                int b = FaceCellOf(pvf, pv);

                // The boundary ring is ONE cell thick, so refinement can make it thinner than a
                // pixel and it disappears.
                const int rH = (int)phf.size() - 1, rV = (int)pvf.size() - 1;
                // This preview indexes rows as b = v -- row 0 is LOW on the vertical axis --
                // which is the opposite of the running viewer's buffer.
                if (h == 0)           a = 0;
                if (h == hCells - 1)  a = rH - 1;
                if (v == 0)           b = 0;
                if (v == vCells - 1)  b = rV - 1;

                // The same (across, up, slice) -> (i, j, k) mapping as the uniform branch.
                if (dimension == 2 || previewPlane == 0) { j = a; i = b; k = (dimension == 2) ? 0 : previewSlice; }
                else if (previewPlane == 1)             { j = a; k = b; i = previewSlice; }
                else                                    { k = a; i = b; j = previewSlice; }
            }
            else if (dimension == 2)     { j = h; i = v; k = 0; }
            else if (previewPlane == 0)  { j = h; i = v; k = previewSlice; }
            else if (previewPlane == 1)  { j = h; k = v; i = previewSlice; }
            else                         { k = h; i = v; j = previewSlice; }

            int t = SCENE.CellTypeAt(i, j, k);
            double code;
            switch (t)
            {
            case SOLID_CELL:   code = 1.0; break;
            case EMPTY_CELL:   code = 2.0; break;
            case INFLOW_CELL:  code = 3.0; break;
            case OUTFLOW_CELL: code = 3.0; break;   // refined below by the owning face's kind
            case PRESSURE_CELL: code = 6.0; break;
            default:           code = 0.0; break;
            }
            // The cell type alone does not name the boundary: moving and slip walls are SOLID_CELLs like a
            // no-slip one. Ask which face owns it.
            if (t == SOLID_CELL || IsVelocityBoundaryCell(t))
            {
                bool on[SceneDesc::FACE_COUNT];
                SCENE.FacesAtCell(i, j, k, on);
                const int win = SCENE.WinningFace(on);
                if (win >= 0)
                {
                    const int kind = (int)SCENE.faces[win].type;
                    if (kind == BC_MOVING)       code = 5.0;
                    else if (kind == BC_SLIP)    code = 7.0;
                    else if (kind == BC_OUTFLOW) code = 8.0;
                }
            }
            // Liquid the FLIP seeder would place, drawn over the fluid/empty background.
            const double sx = SIMULATION.spacing.xc[j];
            const double sy = SIMULATION.spacing.yc[i];
            // z was (k + 0.5)*dh -- no origin, and no table on a stretched grid.
            const double sz = (dimension != 3) ? (k + 0.5) * dh
                            : (dimension == 3 ? SIMULATION.spacing.zc[k] : z0 + (k + 0.5) * dh);
            if (t != SOLID_CELL && SCENE.IsFluidSeedAt(sx, sy, sz))
                code = 4.0;

            // ImPlot draws row 0 at the top; flip so the origin is bottom left.
            previewData[(size_t)(previewRows - 1 - v) * previewCols + h] = code;
        }

    // The 3D tab's voxels, while the globals describe the configured grid. Cached on
    // the scene generation and the grid, so a slice change does not redo N^3 queries.
    BuildVoxelPreview(previewNU);

    SIMULATION.Nx = sNx; SIMULATION.Ny = sNy; SIMULATION.Nz = sNz;
    SIMULATION.dh = sdh; SIMULATION.domain = sdom;
    DIMENSION = sdim;
    SIMULATION.spacing.nonUniform = sNU;
    SIMULATION.spacing.xf = sXF; SIMULATION.spacing.yf = sYF;
    SIMULATION.spacing.xc = sXC; SIMULATION.spacing.yc = sYC;
    SIMULATION.spacing.dx = sDX; SIMULATION.spacing.dy = sDY;
    SIMULATION.spacing.zf = sZF; SIMULATION.spacing.zc = sZC; SIMULATION.spacing.dz = sDZ;
    previewDirty = false;
    previewVersion++;
}

std::string SimulationSetupUI::PressureBlockText() const
{
    std::ostringstream f;
    PRESSURE_SETTINGS.WriteTo(f);   // the solve's own keys first, then AMGX's, then the Eigen path's
    {
        static const char *solvers[] = {"PCG", "PBICGSTAB", "FGMRES", "CG", "AMG"};
        static const char *norms[] = {"L1", "L2", "LMAX"};
        static const char *convs[] = {"RELATIVE_INI_CORE", "ABSOLUTE", "RELATIVE_MAX_CORE"};
        static const char *algs[] = {"CLASSICAL", "AGGREGATION"};
        // The coarsening selector has to match the algorithm.
        static const char *selectors[] = {"PMIS", "SIZE_2"};
        static const char *cycles[] = {"V", "W", "F"};

        f << "config_version=2\n";
        f << "solver=" << solvers[amgxSolver] << "\n";
        f << "max_iters=" << amgxMaxIters << "\n";
        f << "norm=" << norms[amgxNorm] << "\n";
        f << "convergence=" << convs[amgxConvergence] << "\n";
        f << "monitor_residual=1\n";
        f << "relaxation_factor=1.0\n";
        f << "tolerance=" << amgxTolerance << "\n";
        if (amgxUsePrecon && amgxSolver != 4)
        {
            f << "preconditioner(amg_solver)=AMG\n";
            f << "amg_solver:algorithm=" << algs[amgxPreconAlg] << "\n";
            f << "amg_solver:selector=" << selectors[amgxPreconAlg] << "\n";
            f << "amg_solver:max_iters=" << amgxPreIters << "\n";
            f << "amg_solver:presweeps=" << amgxPresweeps << "\n";
            f << "amg_solver:postsweeps=" << amgxPostsweeps << "\n";
            f << "amg_solver:cycle=" << cycles[amgxCycle] << "\n";
        }
        else if (amgxSolver != 4)
        {
            // A Krylov solver with no preconditioner has to say so. Leaving the key out
            // does not mean "none" -- AMGX_solver_setup fails on the missing block.
            f << "preconditioner=NOSOLVER\n";
        }
        if (amgxSolver == 4)
        {
            f << "algorithm=" << algs[amgxPreconAlg] << "\n";
            f << "selector=" << selectors[amgxPreconAlg] << "\n";
            f << "presweeps=" << amgxPresweeps << "\n";
            f << "postsweeps=" << amgxPostsweeps << "\n";
            f << "cycle=" << cycles[amgxCycle] << "\n";
        }
        f << "print_solve_stats=" << (amgxSolveStats ? 1 : 0) << "\n";
        f << "print_grid_stats=" << (amgxGridStats ? 1 : 0) << "\n";
        f << "obtain_timings=" << (amgxTimings ? 1 : 0) << "\n";
    }
    static const char *eigenSolvers[] = {"BICGSTAB", "CG", "AMG"};
    static const char *eigenPrecons[] = {"DIAGONAL", "IDENTITY", "ILUT", "IC"};
    f << "eigen_solver=" << eigenSolvers[eigenSolver] << "\n";
    f << "eigen_preconditioner=" << eigenPrecons[eigenPreconditioner] << "\n";
    f << "eigen_tolerance=" << eigenTolerance << "\n";
    f << "eigen_max_iters=" << eigenMaxIters << "\n";
    return f.str();
}

bool SimulationSetupUI::WriteFiles(std::string &err, const std::string &simPath) const
{
    {
        std::ofstream f(simPath);
        if (!f.is_open()) { err = "Could not write " + simPath; return false; }

        f << "# Written by the NEREUS setup screen.\n";
        f << "\n";

        f << "# Performance Parameters\n";
        f << "DIMENSION = " << dimension << "\n";
        f << "TYPE = " << MethodKey() << "\n";
        f << "GPU_ACCELERATION = " << (useGPU ? 1 : 0) << "\n";
        f << "THREAD_COUNT = " << threadCount << "\n";
        f << "ADAPTATIVE_TIMESTEP = " << (adaptiveTimestep ? 1 : 0) << "\n";
        f << "MAX_CFL = " << maxCFL << "\n\n";

        f << "# Domain dimensions\n";
        f << "DOMAIN_X0 = " << x0 << "\nDOMAIN_XF = " << xf << "\n";
        f << "DOMAIN_Y0 = " << y0 << "\nDOMAIN_YF = " << yf << "\n";
        f << "DOMAIN_Z0 = " << z0 << "\nDOMAIN_ZF = " << zf << "\n\n";

        f << "# Simulation parameters\n";
        f << "NON_UNIFORM_GRID = " << (nonUniformGrid ? 1 : 0) << "\n";
        WriteRefineKey(f, "REFINE_X", refineX);
        WriteRefineKey(f, "REFINE_Y", refineY);
        WriteRefineKey(f, "REFINE_Z", refineZ);
        // The scene itself is written at the END of this file, as a SCENE_BEGIN block, so a
        // config is self-contained.
        f << "# The geometry, boundary conditions and scalar fields are in the\n";
        f << "# SCENE_BEGIN block at the end of this file.\n";
        f << "TIME_STEP = " << timeStep << "\n";
        f << "VISCOSITY = " << viscosity << "\n";
        f << "DENSITY = " << density << "\n";
        f << "GRID_SIZE = " << gridSize << "\n";
        f << "\n";

        if (ShowParticleOptions())
        {
            f << "# Particle options, shared by every particle method\n";
            particleOptions.WriteTo(f);
            f << "\n";
        }

        f << "# Stopping rule: NEVER, TIME (at STOP_TIME) or RESIDUAL (step-to-step change <= STOP_RESIDUAL)\n";
        stop.WriteTo(f);
        f << "\n";

        // One body force: constants here, an expression as the scene's `force` line instead.
        double bf[3];
        if (!BodyForceConstant(bf)) bf[0] = bf[1] = bf[2] = 0.0;
        f << "# Body force (m/s2); an expression is the scene's `force` line instead\n";
        f << "Fu = " << bf[0] << "\n";
        f << "Fv = " << bf[1] << "\n";
        f << "Fw = " << bf[2] << "\n\n";

        f << "# Exporting settings\n";
        f << "EXPORT_BASE_PATH = " << exportPath << "\n";
        exports.WriteTo(f);
        checkpoint.WriteTo(f);
        f << "REYNOLDS_NUMBER = " << (long)reynoldsForNaming
          << "   # export directory naming only\n";
        f << "CHARACTERISTIC_LENGTH = " << characteristicLength << "\n";
        f << "MEAN_VELOCITY = " << meanVelocity << "\n";
        if (dimension == 3)
            f << "CHARACTERISTIC_LENGTH_2 = " << characteristicLength2 << "   # 3D coefficients divide by L1 * L2\n";

        // One block per registered method, each written from its own table.
        WriteMethodSettings(f);

        f << "\n" << PRESSURE_BLOCK_BEGIN << "\n";
        f << PressureBlockText();
        f << PRESSURE_BLOCK_END << "\n";

        // Last, because it is the long part and it is a different language: everything
        // between the markers is scene syntax, read verbatim and handed to Scene.h.
        f << "\n" << SCENE_BLOCK_BEGIN << "\n";
        f << SceneText();
        f << SCENE_BLOCK_END << "\n";
    }
    return true;
}

bool SimulationSetupUI::BodyForceConstant(double f[3]) const
{
    f[0] = f[1] = f[2] = 0.0;
    return NumberOnly(forceU, f[0]) && NumberOnly(forceV, f[1]) &&
           (dimension != 3 || NumberOnly(forceW, f[2]));
}

bool SimulationSetupUI::BodyForceZero() const
{
    double f[3];
    return BodyForceConstant(f) && f[0] == 0.0 && f[1] == 0.0 && f[2] == 0.0;
}

SettingsContext SimulationSetupUI::Context() const
{
    SettingsContext c;
    double dh; int Nx, Ny, Nz;
    DerivedGrid(dh, Nx, Ny, Nz);
    c.dimension = dimension;
    c.dh = UsesGrid() ? dh : 0.0;
    c.density = density;
    c.viscosity = viscosity;
    c.timeStep = timeStep;
    c.adaptiveTimestep = adaptiveTimestep;
    c.x0 = x0; c.xf = xf; c.y0 = y0; c.yf = yf; c.z0 = z0; c.zf = zf;
    double f[3];
    BodyForceConstant(f);   // an expression reads as 0 here; the panels only want gravity
    c.fu = f[0]; c.fv = f[1]; c.fw = f[2];
    c.meanVelocity = meanVelocity;
    c.characteristicLength = characteristicLength;
    return c;
}

void SimulationSetupUI::BuildVoxelPreview(bool nu)
{
    std::ostringstream key;
    key.precision(17);
    key << sceneGeneration << ' ' << SIMULATION.Nx << ' ' << SIMULATION.Ny << ' ' << SIMULATION.Nz
        << ' ' << SIMULATION.dh << ' ' << x0 << ' ' << xf << ' ' << y0 << ' ' << yf << ' ' << z0
        << ' ' << zf << ' ' << nu << ' ' << SCENE.sampleAtCenter;
    if (key.str() == voxKey) return;
    voxKey = key.str();
    voxX.clear(); voxY.clear(); voxZ.clear();
    fluX.clear(); fluY.clear(); fluZ.clear();
    if (dimension != 3) return;
    BuildFluidVoxelPreview(nu);
    if (!SCENE.HasSolidRegions()) return;

    const int Nx = SIMULATION.Nx, Ny = SIMULATION.Ny, Nz = SIMULATION.Nz;
    const double dh = SIMULATION.dh;
    std::vector<char> solid((size_t)Nx * Ny * Nz, 0);
    // Interior only: the ring is the boundary condition, not geometry.
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
                solid[VoxelIndex(i, j, k, Nx, Ny)] = (SCENE.CellTypeAt(i, j, k) == SOLID_CELL);

    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
            {
                if (!solid[VoxelIndex(i, j, k, Nx, Ny)]) continue;
                const bool surface = !solid[VoxelIndex(i, j - 1, k, Nx, Ny)] || !solid[VoxelIndex(i, j + 1, k, Nx, Ny)] ||
                                     !solid[VoxelIndex(i - 1, j, k, Nx, Ny)] || !solid[VoxelIndex(i + 1, j, k, Nx, Ny)] ||
                                     !solid[VoxelIndex(i, j, k - 1, Nx, Ny)] || !solid[VoxelIndex(i, j, k + 1, Nx, Ny)];
                if (!surface) continue;
                voxX.push_back(SIMULATION.spacing.xc[j]);
                voxY.push_back(SIMULATION.spacing.yc[i]);
                voxZ.push_back(dimension == 3 ? SIMULATION.spacing.zc[k] : z0 + (k + 0.5) * dh);
            }
}

void SimulationSetupUI::BuildFluidVoxelPreview(bool nu)
{
    if (!SCENE.HasFluidRegions()) return;
    const int Nx = SIMULATION.Nx, Ny = SIMULATION.Ny, Nz = SIMULATION.Nz;
    const double dh = SIMULATION.dh;
    std::vector<char> fluid((size_t)Nx * Ny * Nz, 0);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
            {
                if (SCENE.CellTypeAt(i, j, k) == SOLID_CELL) continue;
                const double cx = SIMULATION.spacing.xc[j];
                const double cy = SIMULATION.spacing.yc[i];
                const double cz = (dimension == 3 ? SIMULATION.spacing.zc[k] : z0 + (k + 0.5) * dh);
                fluid[VoxelIndex(i, j, k, Nx, Ny)] = SCENE.IsFluidSeedAt(cx, cy, cz);
            }

    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
            {
                if (!fluid[VoxelIndex(i, j, k, Nx, Ny)]) continue;
                const bool surface = !fluid[VoxelIndex(i, j - 1, k, Nx, Ny)] || !fluid[VoxelIndex(i, j + 1, k, Nx, Ny)] ||
                                     !fluid[VoxelIndex(i - 1, j, k, Nx, Ny)] || !fluid[VoxelIndex(i + 1, j, k, Nx, Ny)] ||
                                     !fluid[VoxelIndex(i, j, k - 1, Nx, Ny)] || !fluid[VoxelIndex(i, j, k + 1, Nx, Ny)];
                if (!surface) continue;
                fluX.push_back(SIMULATION.spacing.xc[j]);
                fluY.push_back(SIMULATION.spacing.yc[i]);
                fluZ.push_back(dimension == 3 ? SIMULATION.spacing.zc[k] : z0 + (k + 0.5) * dh);
            }
}

void SimulationSetupUI::Validate(std::vector<std::string> &problems, std::vector<std::string> &notes) const
{
    if (stop.condition == STOP_TIME && !(stop.tFinal > 0.0))
        problems.push_back("Stopping: the final time must be positive.");
    if (stop.condition == STOP_RESIDUAL && !(stop.residualTol > 0.0))
        problems.push_back("Stopping: the residual tolerance must be positive.");
    if (eigenSolver == EIGEN_CG && eigenPreconditioner == EIGEN_PRECON_ILUT)
        problems.push_back("Eigen: conjugate gradient needs a symmetric preconditioner, and ILUT is not one "
                           "(it does not converge). Use BiCGSTAB with ILUT, or another preconditioner.");
    // ----- rectilinear grid -----
    if (nonUniformGrid)
    {
        if (!RectilinearSupported())
            problems.push_back("Non-uniform grid is Eulerian only -- FLIP indexes cells "
                               "as x/dh with no spacing table, and SPH is meshless.");
        else
        {
            struct AxisRef { const char *name; const std::vector<RefineSpec> *v; double lo, hi; };
            const AxisRef axes[] = {{"X", &refineX, x0, xf}, {"Y", &refineY, y0, yf},
                                    {"Z", &refineZ, z0, zf}};
            int active = 0;
            for (const AxisRef &a : axes)
            {
                if (a.v == &refineZ && dimension != 3) continue;
                for (size_t n = 0; n < a.v->size(); n++)
                {
                    const RefineSpec &r = (*a.v)[n];
                    if (!r.active) continue;
                    active++;
                    const std::string tag = std::string("Refine ") + a.name + " #" +
                                            std::to_string(n + 1);
                    if (r.end <= r.start)
                        problems.push_back(tag + ": the interval end must be greater than "
                                                 "its start.");
                    else if (r.end <= a.lo || r.start >= a.hi)
                        problems.push_back(tag + ": the interval lies outside the domain, "
                                                 "so it refines nothing.");
                    if (r.factor < 2)
                        notes.push_back(tag + ": a factor of 1 leaves the spacing unchanged.");
                    if (r.factor > 4 && r.mode == REFINE_HARD)
                        notes.push_back(tag + ": a hard factor of " +
                                        std::to_string(r.factor) + " makes a sharp size jump "
                                        "at the interval edges. \"graded\" removes it.");
                }
                // Overlapping intervals are legal -- the finest wins -- but two HARD intervals
                // that merely touch produce a jump where they meet.
                for (size_t m = 0; m < a.v->size(); m++)
                    for (size_t n = m + 1; n < a.v->size(); n++)
                    {
                        const RefineSpec &p1 = (*a.v)[m], &p2 = (*a.v)[n];
                        if (!p1.active || !p2.active) continue;
                        if (p1.start < p2.end && p2.start < p1.end)
                            notes.push_back(std::string("Refine ") + a.name + ": intervals #" +
                                            std::to_string(m + 1) + " and #" +
                                            std::to_string(n + 1) +
                                            " overlap; the finer factor wins there.");
                    }
            }
            if (active == 0)
                notes.push_back("Non-uniform grid is on but no interval is set -- the grid "
                                "will be built uniform.");
        }
    }

    if (xf <= x0 || yf <= y0 || (dimension == 3 && zf <= z0))
        problems.push_back("Domain: each maximum must be greater than its minimum.");
    else
    {
        // dh comes from the shorter of x and y, so z (3D) can be shorter than one cell: no interior cell along it.
        double dh; int nx, ny, nz;
        DerivedGrid(dh, nx, ny, nz);
        if (dimension == 3 && dh > 0.0 && zf - z0 < dh * (1.0 - 1e-9))
            problems.push_back("Domain: Z is shorter than one cell (dh is sized off the shorter of X and Y).");
    }
    if (gridSize < 4)
        problems.push_back("Grid size is too small to resolve anything.");
    if (timeStep <= 0.0)
        problems.push_back("Timestep must be positive.");
    if (density <= 0.0)
        problems.push_back("Density must be positive.");
    if (viscosity < 0.0)
        problems.push_back("Viscosity cannot be negative.");
    if (!MethodSupported(dimension, method))
        problems.push_back("That solver has no implementation in this dimension.");


    {
        bool anyFluid = false;
        for (const UiRegion &r : uiRegions)
            if (r.kind == 1) anyFluid = true;
        if (!anyFluid && MethodSupports(method, CAP_PARTICLES))
            problems.push_back("A particle method needs a `region fluid` in the scene to say where the liquid "
                               "starts; without one nothing is seeded.");
        if (sceneStatusIsError)
            problems.push_back("The scene does not parse: " + sceneStatus);

        // What leaves the domain, and whether anything fixes the pressure (only a pressure outlet does).
        const int nFaces = (dimension == 3) ? 6 : 4;
        bool anyInlet = false, anyOutflow = false, anyPressureOutlet = false;
        bool allInletsCarryLevel = true;
        for (int f = 0; f < nFaces; f++)
        {
            const int t = uiFaces[f].type;
            anyInlet = anyInlet || t == BC_INLET;
            anyOutflow = anyOutflow || t == BC_OUTFLOW;
            anyPressureOutlet = anyPressureOutlet || t == BC_PRESSURE_OUTLET;
            if (t == BC_INLET && (!uiFaces[f].hasPressure || uiFaces[f].p.empty()))
                allInletsCarryLevel = false;
        }
        allInletsCarryLevel = allInletsCarryLevel && anyInlet;

        if (anyOutflow && anyPressureOutlet)
            problems.push_back("An outflow takes out what the inlets bring in, which is not defined beside a "
                               "pressure outlet (the pressure decides that face's flux). Use one or the other.");
        // Velocity on every face with p= on each inlet is a manufactured solution: its boundary flux balances
        // by construction and p= only sets the level.
        if (anyInlet && !anyOutflow && !anyPressureOutlet && !allInletsCarryLevel)
            problems.push_back("There is an inlet but no way out -- no outflow and no pressure outlet. The "
                               "fluid is incompressible and the domain volume is fixed, so the injected mass "
                               "has nowhere to go and the pressure solve has no solution.");
        // Without a pressure outlet every face gives the velocity, so p is fixed only up to a constant.
        if (!anyPressureOutlet && !PRESSURE_SETTINGS.needsCompatibilityCondition &&
            Family() == MethodFamily::EULERIAN && MethodSupports(method, CAP_PRESSURE_SOLVER))
        {
            if (anyOutflow || allInletsCarryLevel)
                problems.push_back("No face pins the pressure (an outflow and an inlet both give the velocity), "
                                   "so the pressure is pure Neumann and singular. Turn "
                                   "NEEDS_COMPATIBILITY_CONDITION on.");
            else if (!anyInlet)
                notes.push_back("Every face is a wall, so the pressure is fixed only up to a constant (the "
                                "system is singular) and NEEDS_COMPATIBILITY_CONDITION is off. The right-hand "
                                "side is then inconsistent: the solve is undefined and can diverge. Turn it on.");
        }
        // The compatibility projection subtracts the mean from the right-hand side, which is only legitimate
        // when the system is singular.
        if (PRESSURE_SETTINGS.needsCompatibilityCondition && anyPressureOutlet &&
            MethodSupports(method, CAP_PRESSURE_SOLVER))
            problems.push_back("NEEDS_COMPATIBILITY_CONDITION is on, but a pressure outlet pins the pressure. "
                               "The system is not singular, so subtracting the mean changes the solution "
                               "rather than making it solvable. Turn it off.");
    }
    double dh; int Nx, Ny, Nz;
    DerivedGrid(dh, Nx, Ny, Nz);

    if (dimension == 3)
    {
        double cells = (double)Nx * Ny * Nz;
        if (nonUniformGrid && RectilinearSupported())
        {
            std::vector<double> fx, fy, fz;
            DerivedAxes(fx, fy);
            BuildAxisPreview(z0, zf, Nz, dh, refineZ, true, fz);
            if (fx.size() > 1 && fy.size() > 1 && fz.size() > 1)
                cells = (double)(fx.size() - 1) * (fy.size() - 1) * (fz.size() - 1);
        }
        if (cells > 3.0e6)
            notes.push_back("Over 3 M cells. A 3D run of this size will be slow and the "
                            "pressure solve will dominate.");
        if (Nz > gridSize)
            notes.push_back("The grid is sized off the shorter of the X and Y extents, so Z "
                            "ends up with more cells than the grid size you set.");
    }

    // The selected method's own refusals and notes.
    if (MethodOf(method).checkSettings)
        MethodOf(method).checkSettings(Context(), problems, notes);

    double bf[3];
    if (!BodyForceConstant(bf))
    {
        if (!MethodSupports(method, CAP_GRID))
            problems.push_back(std::string(MethodOf(method).label) +
                               " takes a constant body force: give Fx, Fy, Fz as numbers.");
    }
    if (MethodSupports(method, CAP_PARTICLES) && BodyForceZero())
        notes.push_back("No body force set, so nothing will fall.");

    // Particle methods keep their particles inside fixed walls: no injection at an inlet, no removal at an outlet, and
    // the contact treats every wall as stationary. The scene accepts these kinds, the particles ignore them.
    if (MethodSupports(method, CAP_PARTICLES))
    {
        static const char *faceNames[6] = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        const int nFaces = (dimension == 3) ? 6 : 4;
        std::string faces;
        for (int f = 0; f < nFaces; f++)
            if (uiFaces[f].type == BC_MOVING || uiFaces[f].type == BC_INLET || uiFaces[f].type == BC_OUTFLOW ||
                uiFaces[f].type == BC_PRESSURE_OUTLET)
                faces += (faces.empty() ? "" : ", ") + std::string(faceNames[f]);
        if (!faces.empty())
            notes.push_back("Not supported for particle methods (" + faces + "): particles are not injected at an "
                            "inlet or removed at an outlet, and every wall acts as a fixed wall.");
    }

    // Something has to drive the flow: a moving wall, an inlet, or a body force.
    {
        bool driven = !BodyForceZero();
        for (int f = 0; f < 6; f++)
            if (uiFaces[f].type == BC_MOVING || uiFaces[f].type == BC_INLET)
                driven = true;
        if (!driven)
            notes.push_back("Nothing drives this scene: every face is a static wall and there "
                            "is no body force. Set a face to 'moving' (a lid) or 'inlet', or "
                            "give the fluid a body force.");
    }

    if (!adaptiveTimestep && dh > 0.0 && meanVelocity > 0.0)
    {
        const double cfl = meanVelocity * timeStep / dh;
        if (cfl > 1.0)
            notes.push_back("At the mean velocity given, the advection CFL is above 1. "
                            "Particle advection is explicit and will smear at that step.");
    }
}
