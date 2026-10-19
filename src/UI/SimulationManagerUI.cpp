// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The window, the setup screen's drawing and the viewer: everything ImGui/ImPlot/GLFW. Compiled into the GUI
// program only; a program reaches it through the three calls in headers/Core/SimulationManagerUI.h.

#include "../headers/Core/SetupScreen.h"
#include "../headers/Core/SimulationManagerUI.h"
#include "../headers/Core/SettingsPanel.h"
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <misc/cpp/imgui_stdlib.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>

// One placed model, ready for ImPlot3D::PlotMesh in the preview's axis order.
struct MeshDraw { std::vector<ImPlot3DPoint> vtx; std::vector<unsigned> idx; };
static std::map<std::string, MeshDraw> meshDrawCache;

static const MeshDraw *PlacedMeshForDraw(const SimulationSetupUI::UiRegion &g)
{
    std::string err;
    auto raw = SceneDesc::LoadRawMesh(g.meshFile, err);
    if (!raw) return nullptr;
    std::ostringstream key;
    key.precision(17);
    key << g.meshFile << ' ' << raw->mtime;
    for (int a = 0; a < 3; a++) key << ' ' << g.a[a] << ' ' << g.rot[a] << ' ' << g.scl[a];
    auto it = meshDrawCache.find(key.str());
    if (it != meshDrawCache.end()) return &it->second;
    if (meshDrawCache.size() > 32) meshDrawCache.clear(); // editing leaves a trail of keys

    SceneDesc::MeshTransform xfm;
    for (int a = 0; a < 3; a++) { xfm.center[a] = g.a[a]; xfm.rotate[a] = g.rot[a]; xfm.scale[a] = g.scl[a]; }
    std::vector<double> wv; std::vector<unsigned> wt; double lo[3], hi[3];
    SceneDesc::MeshWorldGeometry(*raw, xfm, wv, wt, lo, hi);
    MeshDraw &d = meshDrawCache[key.str()];
    d.vtx.resize(wv.size() / 3);
    // ImPlot3D draws its third axis vertically and the simulation's vertical is y, so
    // (x, y, z) goes in as (x, z, y) -- the same order the running particle view uses.
    for (size_t i = 0; i < d.vtx.size(); i++)
        d.vtx[i] = ImPlot3DPoint(wv[3 * i + 0], wv[3 * i + 2], wv[3 * i + 1]);
    d.idx = wt;
    return &d;
}

// Set by the Restart buttons: UpdateUI then ends the run's loop without closing the window, and RestartUI reopens
// the setup screen.
static bool RESTART_REQUESTED = false;

// Cycles the 16 built-in colormaps only: they are numbered alike in ImPlot and ImPlot3D, while the theme's
// series maps exist in ImPlot alone (an invalid index in ImPlot3D).
static bool FieldColormapButton(int &map, float width)
{
    if (map < 0 || map >= ImPlotColormap_Greys + 1) map = ImPlotColormap_Viridis;
    if (!ImPlot::ColormapButton(ImPlot::GetColormapName(map), ImVec2(width, 0), map)) return false;
    map = (map + 1) % (ImPlotColormap_Greys + 1);
    return true;
}

static void StatRow(const char *label, const char *value, UiTone tone = UiTone::Good, bool coloured = false)
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextDisabled("%s", label);
    ImGui::TableNextColumn();
    if (coloured) ImGui::TextColored(ToneColour(tone), "%s", value);
    else ImGui::TextUnformatted(value);
}

void SimulationSetupUI::DrawRefineAxis(const char *label, std::vector<RefineSpec> &v, double lo, double hi)
{
    ImGui::PushID(label);
    ImGui::TextUnformatted(label);
    ImGui::SameLine();
    if (ImGui::SmallButton("+ interval"))
    {
        RefineSpec r;
        r.start = lo + 0.25 * (hi - lo);
        r.end = lo + 0.75 * (hi - lo);
        v.push_back(r);
        previewDirty = true;
    }

    int erase = -1;
    for (size_t n = 0; n < v.size(); n++)
    {
        RefineSpec &r = v[n];
        ImGui::PushID((int)n);
        bool ch = ImGui::Checkbox("##on", &r.active);
        // Widths from the font, not pixels: the text size is a UI setting.
        const float pad = 2.0f * ImGui::GetStyle().FramePadding.x;
        const float numW = ImGui::CalcTextSize("-00.0000").x + pad;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(numW);
        ch |= ImGui::InputDouble("##from", &r.start, 0.0, 0.0, "%.4g");
        ImGui::SameLine();
        ImGui::TextUnformatted("to");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(numW);
        ch |= ImGui::InputDouble("##to", &r.end, 0.0, 0.0, "%.4g");
        ImGui::SameLine();
        ImGui::TextDisabled("m");
        ImGui::SameLine();
        ImGui::TextUnformatted("by");
        ImGui::SameLine();
        // The step buttons took most of the box and cut the factor off; type it instead.
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("000").x + pad);
        ch |= ImGui::InputInt("##fac", &r.factor, 0, 0);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Refinement factor, 1-64");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("graded").x + ImGui::GetFrameHeight() + pad);
        // Order matches the RefineMode enum: hard, ramp, graded, ramp_back.
        static const char *modeNames[] = {"hard", "ramp >", "graded", "ramp <"};
        ch |= ImGui::Combo("##mode", &r.mode, modeNames, 4);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "hard   -- the whole interval is refined; the size jumps by the\n"
                "          factor at both edges, and it snaps to the base cells.\n"
                "ramp > -- the refinement factor rises linearly from 1 at the\n"
                "          start to `factor` at the end, so the size dh/factor\n"
                "          falls fastest at the coarse end. Use it as a transition\n"
                "          into a hard interval; for a large factor, stack several\n"
                "          nested ramps (the finest wins) to keep the jump small.\n"
                "ramp < -- the mirror, finest at the start. The pair lets a\n"
                "          transition be laid on either side of a refined band.\n"
                "graded -- dh at both edges, dh/factor at the centre. The shape\n"
                "          that refines around a feature without leaving a sharp\n"
                "          jump anywhere.\n\n"
                "A jump is anti-diffusive where the flow leaves the band, which is\n"
                "what the graded modes are for.");
        ImGui::SameLine();
        if (ImGui::SmallButton("x")) erase = (int)n;
        r.factor = (std::max)(1, (std::min)(64, r.factor));   // the grid builder takes any; 64 only stops a typo
        r.start = (std::max)(lo, (std::min)(hi, r.start));
        r.end = (std::max)(lo, (std::min)(hi, r.end));
        if (ch) previewDirty = true;
        ImGui::PopID();
    }
    if (erase >= 0)
    {
        v.erase(v.begin() + erase);
        previewDirty = true;
    }
    ImGui::PopID();
}

void SimulationSetupUI::DrawAirfoilRegionEditor(UiRegion &g, int nc)
{
    if (FormSliderInt("Thickness", &g.thicknessPercent, 1, 40, "% of the chord")) previewDirty = true;
    ImGui::SameLine();
    ImGui::TextDisabled("(NACA 00%02d)", g.thicknessPercent);
    if (FormDouble("Chord", &g.chord, "m", "%.5g"))
    {
        if (g.chord <= 0.0) g.chord = 1e-3;
        previewDirty = true;
    }
    if (FormVec("Pivot position", g.a, nc, "m", "%.5g", "Where the pivot lands in the domain")) previewDirty = true;
    if (FormDouble("Pivot fraction", &g.pivot, "chords from the leading edge", "%.4g",
                   "0.5 is mid-chord, 0.25 the quarter chord"))
        previewDirty = true;
    if (dimension == 2)
    {
        if (FormDouble("Rotation about z", &g.rot[2], "deg", "%.4g",
                       "Counter-clockwise positive, so an angle of attack alpha in a +x stream is -alpha"))
            previewDirty = true;
    }
    else
    {
        if (FormVec("Rotation", g.rot, 3, "deg", "%.4g", "Degrees about x, then y, then z")) previewDirty = true;
        if (FormDouble("Span", &g.span, "m", "%.5g", "0 runs through the whole domain")) previewDirty = true;
    }
    if (ImGui::Checkbox("Closed trailing edge (-0.1036 x^4)", &g.closedTE)) previewDirty = true;
}

void SimulationSetupUI::DrawParticleOptions()
{
    static const char *contactNames[] = {"Free slip", "No slip", "Bounce"};
    FormCombo("On contact", &particleOptions.contact, contactNames, IM_ARRAYSIZE(contactNames),
              "What a particle does at a wall or an obstacle. Bounce is scaled by the restitution");
    if (particleOptions.contact == 2)
    {
        ImGui::Indent();
        FormDouble("Restitution", &particleOptions.restitution, nullptr, "%.2f");
        particleOptions.restitution = (std::max)(0.0, (std::min)(1.0, particleOptions.restitution));
        if (particleOptions.restitution > 0.7)
            ImGui::TextColored(ToneColour(UiTone::Warn),
                               "A high restitution adds energy at a free surface.");
        ImGui::Unindent();
    }
    if (!UsesGrid())
    {
        return;
    }

    FormDouble("Surface offset", &particleOptions.surfaceOffset, "cells", "%.3f",
               "Distance a pushed-out particle is parked from a solid");
    particleOptions.surfaceOffset = (std::max)(0.0, (std::min)(0.5, particleOptions.surfaceOffset));

    FormDouble("Seed jitter", &particleOptions.seedJitter, "sub-cells", "%.2f",
               "Random offset of each particle within its sub-cell at seeding");
    particleOptions.seedJitter = (std::max)(0.0, (std::min)(1.0, particleOptions.seedJitter));
}

// A number box that commits only when the edit is finished (Enter, Tab, focus lost): a half-typed domain
// extent would rebuild a grid nobody asked for.
static bool InputDoubleOnCommit(const char *label, double *v, const char *fmt)
{
    static std::map<ImGuiID, double> pending;
    const ImGuiID id = ImGui::GetID(label);
    std::map<ImGuiID, double>::iterator it = pending.find(id);
    double edit = (it != pending.end()) ? it->second : *v;
    ImGui::InputDouble(label, &edit, 0.0, 0.0, fmt);
    if (ImGui::IsItemActive())
    {
        pending[id] = edit;
        return false;
    }
    if (it == pending.end()) return false;
    pending.erase(it);
    if (edit == *v) return false;
    *v = edit;
    return true;
}

void SimulationSetupUI::DrawDomainExtents()
{
    static const char *rows[3] = {"x", "y", "z"};
    static const char *ids[3][2] = {{"##x0", "##xf"}, {"##y0", "##yf"}, {"##z0", "##zf"}};
    double *lo[3] = {&x0, &y0, &z0}, *hi[3] = {&xf, &yf, &zf};
    bool ch = false;
    for (int a = 0; a < ((dimension == 3) ? 3 : 2); a++)
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(rows[a]);
        ImGui::SameLine(ImGui::GetStyle().IndentSpacing + FormEm() * 3.0f);
        ImGui::TextDisabled("from");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(FormNumberWidth());
        ch |= InputDoubleOnCommit(ids[a][0], lo[a], "%.4f");
        ImGui::SameLine();
        ImGui::TextDisabled("to");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(FormNumberWidth());
        ch |= InputDoubleOnCommit(ids[a][1], hi[a], "%.4f");
        FormUnit("m");
    }
    if (ch) previewDirty = true;
}

void SimulationSetupUI::DrawResolution()
{
    bool ch = false;
    FormLabel("Grid size", "Cells across the shorter of the x and y extents");
    ImGui::SetNextItemWidth(FormNumberWidth());
    ch |= ImGui::SliderInt("##gridSizeSlider", &gridSize, 8, 512);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(FormEm() * 8.0f + 2.0f * (ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x));
    ch |= ImGui::InputInt("##gridSize", &gridSize, 1, 16);
    gridSize = (std::max)(8, (std::min)(2048, gridSize));
    FormUnit("cells across the shorter of x and y");
    if (ch) previewDirty = true;

    double dh; int Nx, Ny, Nz;
    DerivedGrid(dh, Nx, Ny, Nz);
    std::string gridProblem;
    const bool gridOk = GridPreviewable(gridProblem);

    // ----- rectilinear spacing -----
    ImGui::Spacing();
    const bool canStretch = RectilinearSupported();
    if (!canStretch)
        ImGui::BeginDisabled();
    if (ImGui::Checkbox("Non-uniform grid", &nonUniformGrid))
        previewDirty = true;
    if (!canStretch)
        ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(canStretch ? "Refines chosen intervals along each axis. dt follows the smallest cell"
                                     : "Stretched grids need a method that supports them");

    if (nonUniformGrid && canStretch)
    {
        ImGui::Indent();
        DrawRefineAxis("Refine X", refineX, x0, xf);
        DrawRefineAxis("Refine Y", refineY, y0, yf);
        if (dimension == 3)
            DrawRefineAxis("Refine Z", refineZ, z0, zf);
        ImGui::Unindent();
    }

    ImGui::Spacing();
    if (!gridOk)
        ImGui::TextColored(ToneColour(UiTone::Warn), "%s", gridProblem.c_str());
    else if (dimension == 3 && nonUniformGrid && canStretch)
    {
        // The refined counts, from the same axis construction MAC uses -- the base Nx/Ny/Nz are before refinement.
        std::vector<double> fx, fy, fz;
        DerivedAxes(fx, fy);
        BuildAxisPreview(z0, zf, Nz, dh, refineZ, true, fz);
        const int rNx = fx.empty() ? 0 : (int)fx.size() - 1;
        const int rNy = fy.empty() ? 0 : (int)fy.size() - 1;
        const int rNz = fz.empty() ? 0 : (int)fz.size() - 1;
        double hmin = dh;
        for (size_t j = 1; j < fx.size(); j++) hmin = (std::min)(hmin, fx[j] - fx[j - 1]);
        for (size_t i = 1; i < fy.size(); i++) hmin = (std::min)(hmin, fy[i] - fy[i - 1]);
        for (size_t k = 1; k < fz.size(); k++) hmin = (std::min)(hmin, fz[k] - fz[k - 1]);
        ImGui::Text("Grid: %d x %d x %d   dh %.5f -> %.5f m   %.2f M cells",
                    rNx, rNy, rNz, dh, hmin, (double)rNx * rNy * rNz / 1e6);
        ImGui::TextDisabled("Base %d x %d x %d before refinement. dt is sized off %.5f m.",
                            Nx, Ny, Nz, hmin);
    }
    else if (dimension == 3)
        ImGui::Text("Grid: %d x %d x %d   dh = %.5f m   %.2f M cells",
                    Nx, Ny, Nz, dh, (double)Nx * Ny * Nz / 1e6);
    else if (nonUniformGrid && canStretch)
    {
        std::vector<double> fx, fy;
        DerivedAxes(fx, fy);
        const int rNx = fx.empty() ? 0 : (int)fx.size() - 1;
        const int rNy = fy.empty() ? 0 : (int)fy.size() - 1;
        double hmin = dh;
        for (size_t j = 1; j < fx.size(); j++) hmin = (std::min)(hmin, fx[j] - fx[j - 1]);
        for (size_t i = 1; i < fy.size(); i++) hmin = (std::min)(hmin, fy[i] - fy[i - 1]);
        ImGui::Text("Grid: %d x %d   dh %.5f -> %.5f m   %d cells",
                    rNx, rNy, dh, hmin, rNx * rNy);
        ImGui::TextDisabled("Base %d x %d before refinement. dt is sized off %.5f m.",
                            Nx, Ny, hmin);
    }
    else
        ImGui::Text("Grid: %d x %d   dh = %.5f m   %d cells", Nx, Ny, dh, Nx * Ny);
}

void SimulationSetupUI::DrawGridStats(const std::vector<double> &fx, const std::vector<double> &fy, const std::vector<double> &fz,
                   int Nx, int Ny, int Nz, double worstRatio) const
{
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {0.0, 0.0, 0.0};
    const std::vector<double> *t[3] = {&fx, &fy, &fz};
    for (int a = 0; a < 3; a++)
        for (size_t i = 0; i + 1 < t[a]->size(); i++)
        {
            const double d = (*t[a])[i + 1] - (*t[a])[i];
            lo[a] = (std::min)(lo[a], d);
            hi[a] = (std::max)(hi[a], d);
        }
    const int axes = (dimension == 3) ? 3 : 2;
    double hmin = 1e30, hmax = 0.0;
    for (int a = 0; a < axes; a++) { hmin = (std::min)(hmin, lo[a]); hmax = (std::max)(hmax, hi[a]); }

    if (!ImGui::BeginTable("##gridstats", 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg))
        return;
    char buf[128];
    const char *axisName[3] = {"dx", "dy", "dz"};
    if (dimension == 3)
        std::snprintf(buf, sizeof(buf), "%d x %d x %d = %.3g", Nx, Ny, Nz, (double)Nx * Ny * Nz);
    else
        std::snprintf(buf, sizeof(buf), "%d x %d = %d", Nx, Ny, Nx * Ny);
    StatRow("Cells", buf);
    for (int a = 0; a < axes; a++)
    {
        if (hi[a] - lo[a] <= 1e-12 * hi[a]) std::snprintf(buf, sizeof(buf), "%.5g (uniform)", lo[a]);
        else std::snprintf(buf, sizeof(buf), "%.5g to %.5g", lo[a], hi[a]);
        StatRow(axisName[a], buf);
    }
    std::snprintf(buf, sizeof(buf), "%.5g m", hmin);
    StatRow("Finest cell", buf);
    std::snprintf(buf, sizeof(buf), "%.3gx", worstRatio);
    StatRow("Largest size jump", buf, worstRatio > 1.25 ? UiTone::Note : UiTone::Good, worstRatio > 1.001);
    if (viscosity > 0.0 && meanVelocity > 0.0)
    {
        std::snprintf(buf, sizeof(buf), "%.3g to %.3g", meanVelocity * hmin / viscosity, meanVelocity * hmax / viscosity);
        StatRow("Cell Re at U", buf, UiTone::Warn, meanVelocity * hmax / viscosity > 2.0);
    }
    if (meanVelocity > 0.0)
    {
        std::snprintf(buf, sizeof(buf), "%.4g s", maxCFL * hmin / meanVelocity);
        StatRow("dt at max CFL", buf);
    }
    if (viscosity > 0.0 && timeStep > 0.0)
    {
        std::snprintf(buf, sizeof(buf), "%.3g", viscosity * timeStep / (hmin * hmin));
        StatRow("nu dt / h^2", buf);
    }
    ImGui::EndTable();
}

void SimulationSetupUI::RenderMeshPreview()
{
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    const float px = vp->WorkPos.x + vp->WorkSize.x * 0.55f;
    const float pw = vp->WorkSize.x * 0.45f;
    const float top = vp->WorkSize.y * 0.62f;
    ImGui::SetNextWindowPos(ImVec2(px, vp->WorkPos.y + top), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(pw, vp->WorkSize.y - top), ImGuiCond_Always);
    ImGui::Begin("Mesh preview", nullptr, ImGuiWindowFlags_NoScrollbar);

    std::string why;
    if (!GridPreviewable(why))
    {
        ImGui::TextColored(ToneColour(UiTone::Warn), "%s", why.c_str());
        ImGui::End();
        return;
    }
    std::vector<double> fx, fy;
    DerivedAxes(fx, fy);
    if (fx.size() < 2 || fy.size() < 2)
    {
        ImGui::TextDisabled("No grid yet.");
        ImGui::End();
        return;
    }

    const int Nx = (int)fx.size() - 1, Ny = (int)fy.size() - 1;
    std::vector<double> fz;
    int Nz = 1;
    if (dimension == 3)
    {
        double dhz; int nx, ny;
        DerivedGrid(dhz, nx, ny, Nz);
        BuildAxisPreview(z0, zf, Nz, dhz, refineZ, nonUniformGrid && RectilinearSupported(), fz);
        Nz = (int)fz.size() - 1;
    }

    // The stretching ratio decides whether the grid is stable: a size jump along the flow is anti-diffusive.
    double worstRatio = 1.0;
    for (const std::vector<double> *t : {&fx, &fy, &fz})
        for (size_t i = 1; i + 1 < t->size(); i++)
        {
            const double a = (*t)[i] - (*t)[i - 1], b = (*t)[i + 1] - (*t)[i];
            worstRatio = (std::max)(worstRatio, (std::max)(a / b, b / a));
        }

    // ----- canvas -----
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 40.0f || avail.y < 40.0f) { ImGui::End(); return; }
    const double Lx = fx.back() - fx.front(), Ly = fy.back() - fy.front();
    if (Lx <= 0.0 || Ly <= 0.0) { ImGui::End(); return; }

    // Keep the aspect ratio: a stretched picture of a stretched grid is unreadable. The statistics go under the
    // drawing when the domain is wide enough to leave room there, beside it otherwise.
    const float pad = 8.0f;
    const float statsW = 340.0f, statsH = 9.0f * ImGui::GetTextLineHeightWithSpacing() + 2 * pad;
    const float want = (float)(Ly / Lx);
    float w = avail.x - 2 * pad, h = w * want;
    const bool stacked = avail.y - (h + 2 * pad) >= statsH;
    if (!stacked)
    {
        w = (std::max)(60.0f, avail.x - statsW - 3 * pad);
        h = avail.y - 2 * pad;
        if (h / w > want) h = w * want; else w = h / want;
    }
    else if (h > avail.y - statsH - 2 * pad)
    {
        h = avail.y - statsH - 2 * pad;
        w = h / want;
    }

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 tl(origin.x + pad, origin.y + pad);
    ImDrawList *dl = ImGui::GetWindowDrawList();

    // y is drawn upward, so the domain reads the way the simulation does.

    dl->AddRectFilled(tl, ImVec2(tl.x + w, tl.y + h), IM_COL32(18, 22, 26, 255));

    // Shade the refined intervals so the band is visible even when the lines are dense.
    if (nonUniformGrid && RectilinearSupported())
    {
        const ImU32 shade = IM_COL32(70, 120, 130, 45);
        for (const RefineSpec &r : refineX)
            if (r.active && r.factor > 1)
                dl->AddRectFilled(ImVec2(CanvasX(r.start, fx.front(), Lx, tl.x, w), tl.y),
                                  ImVec2(CanvasX(r.end, fx.front(), Lx, tl.x, w), tl.y + h), shade);
        for (const RefineSpec &r : refineY)
            if (r.active && r.factor > 1)
                dl->AddRectFilled(ImVec2(tl.x, CanvasY(r.end, fy.front(), Ly, tl.y, h)),
                                  ImVec2(tl.x + w, CanvasY(r.start, fy.front(), Ly, tl.y, h)), shade);
    }

    // Thin the lines out rather than drawing a solid block of pixels.
    const int strideX = (std::max)(1, (int)std::ceil(Nx / (double)(std::max)(1.0f, w / 3.0f)));
    const int strideY = (std::max)(1, (int)std::ceil(Ny / (double)(std::max)(1.0f, h / 3.0f)));
    const ImU32 line = IM_COL32(120, 140, 150, 150);
    for (int j = 0; j <= Nx; j += strideX)
        dl->AddLine(ImVec2(CanvasX(fx[j], fx.front(), Lx, tl.x, w), tl.y), ImVec2(CanvasX(fx[j], fx.front(), Lx, tl.x, w), tl.y + h), line);
    for (int i = 0; i <= Ny; i += strideY)
        dl->AddLine(ImVec2(tl.x, CanvasY(fy[i], fy.front(), Ly, tl.y, h)), ImVec2(tl.x + w, CanvasY(fy[i], fy.front(), Ly, tl.y, h)), line);

    dl->AddRect(tl, ImVec2(tl.x + w, tl.y + h), IM_COL32(200, 210, 215, 220));
    if (stacked)
        ImGui::Dummy(ImVec2(avail.x, h + 2 * pad));
    else
    {
        ImGui::Dummy(ImVec2(w + 2 * pad, h + 2 * pad));
        ImGui::SameLine();
    }
    ImGui::BeginGroup();
    DrawGridStats(fx, fy, fz, Nx, Ny, Nz, worstRatio);
    if (strideX > 1 || strideY > 1)
        ImGui::TextDisabled("Drawing every %d x line and every %d y line", strideX, strideY);
    ImGui::EndGroup();
    ImGui::End();
}

void SimulationSetupUI::DrawProbes()
{
    ImGui::SeparatorText("Probes");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("One telemetry column per probe. A radius averages over a sensor of that size");

    std::vector<std::string> quantities = {"p", "u", "v", "w", "speed", "level", "top"};
    for (const UiScalar &sc : uiScalars)
        if (!sc.name.empty()) quantities.push_back(sc.name);

    int remove = -1;
    const int columns = (dimension == 3) ? 7 : 6;
    if (ImGui::BeginTable("##probes", columns, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV))
    {
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, 90);
        ImGui::TableSetupColumn("quantity", ImGuiTableColumnFlags_WidthFixed, 90);
        ImGui::TableSetupColumn("x (m)", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("y (m)", ImGuiTableColumnFlags_WidthFixed, 80);
        if (dimension == 3) ImGui::TableSetupColumn("z (m)", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("radius (m)", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 24);
        ImGui::TableHeadersRow();

        for (size_t n = 0; n < uiProbes.size(); n++)
        {
            Probe &pr = uiProbes[n];
            ImGui::PushID((int)n + 7000);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##name", &pr.name);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::BeginCombo("##quantity", pr.quantity.c_str()))
            {
                for (const std::string &q : quantities)
                    if (ImGui::Selectable(q.c_str(), q == pr.quantity))
                    {
                        pr.quantity = q;
                        pr.kind = (q == "p") ? ProbeQuantity::Pressure : (q == "u") ? ProbeQuantity::U
                                : (q == "v") ? ProbeQuantity::V : (q == "w") ? ProbeQuantity::W
                                : (q == "speed") ? ProbeQuantity::Speed : (q == "level") ? ProbeQuantity::Level
                                : (q == "top") ? ProbeQuantity::Top : ProbeQuantity::Scalar;
                    }
                ImGui::EndCombo();
            }
            for (int a = 0; a < ((dimension == 3) ? 3 : 2); a++)
            {
                ImGui::TableNextColumn();
                ImGui::PushID(a);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputDouble("##c", &pr.at[a], 0.0, 0.0, "%.5g");
                ImGui::PopID();
            }
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputDouble("##r", &pr.radius, 0.0, 0.0, "%.5g");
            if (pr.radius < 0.0) pr.radius = 0.0;
            ImGui::TableNextColumn();
            if (ImGui::SmallButton("x")) remove = (int)n;
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (remove >= 0) uiProbes.erase(uiProbes.begin() + remove);

    if (ImGui::SmallButton("+ probe"))
    {
        Probe pr;
        pr.name = "P" + std::to_string(uiProbes.size() + 1);
        pr.quantity = "p";
        uiProbes.push_back(pr);
        previewDirty = true;
    }
    if (!uiProbes.empty())
    {
        double dh; int Nx, Ny, Nz;
        DerivedGrid(dh, Nx, Ny, Nz);
        ImGui::SameLine();
        ImGui::TextDisabled("cell size %.4g m -- a radius below that samples the point", dh);
    }
}

void SimulationSetupUI::DrawScalarFields()
{

    int removeAt = -1;
    for (size_t sc = 0; sc < uiScalars.size(); sc++)
    {
        ImGui::PushID((int)(4000 + sc));
        ImGui::Separator();

        FormLabel("Name");
        ImGui::SetNextItemWidth(FormNumberWidth());
        if (ImGui::InputText("##name", &uiScalars[sc].name)) previewDirty = true;
        ImGui::SameLine();
        if (ImGui::Button("Remove")) removeAt = (int)sc;
        if (FormCombo("Solver", &uiScalars[sc].solver, "ADI transport\0none\0",
                      "None stores the field and sets its boundary, and a method of yours updates it", FormEm() * 18.0f))
            previewDirty = true;
        const bool adiSolver = uiScalars[sc].solver == 0;
        if (adiSolver)
        {
            if (FormCombo("Scheme", &uiScalars[sc].scheme, "central\0upwind\0",
                          "Upwind is bounded but smears fronts. Central is sharp but overshoots"))
                previewDirty = true;
            if (FormCombo("Form", &uiScalars[sc].form, "flux\0advective\0",
                          "Flux conserves the total. Advective keeps a uniform field uniform"))
                previewDirty = true;
        }

        if (uiScalars[sc].name.empty())
        {
            ImGui::TextColored(ToneColour(UiTone::Error), "a scalar needs a name");
            ImGui::PopID();
            continue;
        }

        // Diffusivity, reactivity and source are the field's physics, kept with or without a solver; with none,
        // the method that updates the field reads them.
        ImGui::Indent();
        if (FormDouble("Diffusivity epsilon", &uiScalars[sc].epsilon, "m\u00b2/s", "%.6g",
                       adiSolver ? "Physical diffusivity. Upwind already adds |u| h / 2" : "Physical diffusivity"))
        {
            if (uiScalars[sc].epsilon < 0.0) uiScalars[sc].epsilon = 0.0;
            previewDirty = true;
        }
        if (FormDouble("Reactivity gamma", &uiScalars[sc].gamma, "1/s", "%.6g",
                       "Decay rate. Positive decays, negative grows"))
            previewDirty = true;
        FormLabel("Source S", "Source per second, an expression in x y z t");
        ImGui::SetNextItemWidth(FormWideWidth());
        if (ImGui::InputTextWithHint("##source", "none -- expression in x y z t dh Lx Ly Lz", &uiScalars[sc].source))
            previewDirty = true;
        FormUnit("field units per s");
        if (adiSolver && uiScalars[sc].epsilon > 0.0 && meanVelocity > 0.0)
        {
            double dh = 0.0; int gNx = 0, gNy = 0, gNz = 0;
            DerivedGrid(dh, gNx, gNy, gNz);
            ImGui::TextDisabled("cell Peclet U dh / epsilon = %.3g (at MEAN_VELOCITY; above 2 central oscillates)",
                                meanVelocity * dh / uiScalars[sc].epsilon);
        }
        if (uiScalars[sc].gamma > 0.0)
            ImGui::TextDisabled("decay time 1/gamma = %.3g s", 1.0 / uiScalars[sc].gamma);
        ImGui::TextDisabled("boundary conditions");
        const int faceCount = (dimension == 3) ? 6 : 4;
        for (int fi = 0; fi < faceCount; fi++)
        {
            if (uiFaces[fi].scalarValue.size() < uiScalars.size())
                uiFaces[fi].scalarValue.resize(uiScalars.size());
            if (uiFaces[fi].scalarClamp.size() < uiScalars.size())
                uiFaces[fi].scalarClamp.resize(uiScalars.size(), 0);

            ImGui::PushID(fi);
            bool pinned = !uiFaces[fi].scalarValue[sc].empty();
            FormLabel(FaceName(fi));
            ImGui::SetNextItemWidth(FormEm() * 18.0f);
            int mode = pinned ? 1 : 0;
            if (ImGui::Combo("##mode", &mode, "zero gradient\0prescribed\0"))
            {
                // An empty string IS the zero-gradient condition, so switching modes is
                // just clearing or seeding the value the scene will carry.
                uiFaces[fi].scalarValue[sc] = (mode == 1) ? std::string("0") : std::string();
                previewDirty = true;
            }
            if (mode == 1)
            {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(FormEm() * 22.0f);
                if (ImGui::InputText("##val", &uiFaces[fi].scalarValue[sc]))
                    previewDirty = true;
                ImGui::SameLine();
                bool clamp = uiFaces[fi].scalarClamp[sc] != 0;
                if (ImGui::Checkbox("Clamp negative", &clamp))
                {
                    uiFaces[fi].scalarClamp[sc] = clamp ? 1 : 0;
                    previewDirty = true;
                }
            }
            ImGui::PopID();
        }
        ImGui::TextDisabled(
            "zero gradient is the zero-flux wall and the outflow both, so normally only an\n"
            "inflow needs a value. Expressions in x y z t dh Lx Ly Lz.");
        ImGui::Unindent();
        ImGui::PopID();
    }

    if (removeAt >= 0)
    {
        uiScalars.erase(uiScalars.begin() + removeAt);
        for (int f2 = 0; f2 < 6; f2++)
        {
            if ((size_t)removeAt < uiFaces[f2].scalarValue.size())
                uiFaces[f2].scalarValue.erase(uiFaces[f2].scalarValue.begin() + removeAt);
            if ((size_t)removeAt < uiFaces[f2].scalarClamp.size())
                uiFaces[f2].scalarClamp.erase(uiFaces[f2].scalarClamp.begin() + removeAt);
        }
        previewDirty = true;
    }

    ImGui::Separator();
    if (ImGui::Button("+ scalar field"))
    {
        uiScalars.push_back(UiScalar());
        for (int f2 = 0; f2 < 6; f2++)
        {
            uiFaces[f2].scalarValue.push_back(std::string());
            uiFaces[f2].scalarClamp.push_back(0);
        }
        previewDirty = true;
    }
}

void SimulationSetupUI::DrawMeshRegionEditor(UiRegion &g)
{
    if (!modelsScanned) RefreshModels();

    FormLabel("Model", "Closed STL surfaces in the Models folder");
    ImGui::SetNextItemWidth(FormEm() * 30.0f);
    if (ImGui::BeginCombo("##model", g.meshFile.empty() ? "(choose a model)" : g.meshFile.c_str()))
    {
        if (models.empty())
            ImGui::TextDisabled("no .stl files in %s/", MODEL_DIR);
        for (const std::string &m : models)
            if (ImGui::Selectable(m.c_str(), m == g.meshFile))
            { g.meshFile = m; previewDirty = true; }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Rescan")) RefreshModels();

    std::string err;
    std::shared_ptr<const SceneDesc::RawMesh> raw;
    if (!g.meshFile.empty())
        raw = SceneDesc::LoadRawMesh(g.meshFile, err);

    if (FormVec("Centre", g.a, 3, "m", "%.5g", "Where the model's bounding-box centre lands")) previewDirty = true;
    ImGui::SameLine();
    if (ImGui::SmallButton("centre in domain"))
    {
        g.a[0] = 0.5 * (x0 + xf); g.a[1] = 0.5 * (y0 + yf); g.a[2] = 0.5 * (z0 + zf);
        previewDirty = true;
    }

    if (FormVec("Rotation", g.rot, 3, "deg", "%.4g",
                "Degrees about x, then y, then z. Angle of attack is a negative z rotation"))
        previewDirty = true;

    if (g.uniformScale)
    {
        if (FormDouble("Scale", &g.scl[0], "m per model unit", "%.6g",
                       "Model units to metres. Millimetre CAD files need 0.001"))
        { g.scl[1] = g.scl[2] = g.scl[0]; previewDirty = true; }
    }
    else
    {
        if (FormVec("Scale", g.scl, 3, "m per model unit", "%.6g",
                    "Model units to metres. Millimetre CAD files need 0.001"))
            previewDirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("uniform", &g.uniformScale) && g.uniformScale)
    { g.scl[1] = g.scl[2] = g.scl[0]; previewDirty = true; }
    ImGui::SameLine();
    if (raw && ImGui::SmallButton("fit"))
    {
        // Largest uniform scale that puts the ROTATED model inside half the domain on
        // every axis the run has. A starting point, not a recommendation.
        SceneDesc::MeshTransform unit;
        for (int a = 0; a < 3; a++) { unit.rotate[a] = g.rot[a]; unit.center[a] = 0.0; }
        std::vector<double> wv; std::vector<unsigned> wt; double lo[3], hi[3];
        SceneDesc::MeshWorldGeometry(*raw, unit, wv, wt, lo, hi);
        const double L[3] = {xf - x0, yf - y0, zf - z0};
        double s = 1e300;
        for (int a = 0; a < (dimension == 3 ? 3 : 2); a++)
            if (hi[a] - lo[a] > 0.0) s = (std::min)(s, 0.5 * L[a] / (hi[a] - lo[a]));
        if (s < 1e300)
        { g.scl[0] = g.scl[1] = g.scl[2] = s; g.uniformScale = true; previewDirty = true; }
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Uniform scale so the model spans half the domain");

    // ---- what this placement means for the grid ----
    if (g.meshFile.empty()) return;
    if (!raw)
    {
        ImGui::TextColored(ToneColour(UiTone::Error), "%s", err.c_str());
        return;
    }
    SceneDesc::MeshTransform xfm;
    for (int a = 0; a < 3; a++) { xfm.center[a] = g.a[a]; xfm.rotate[a] = g.rot[a]; xfm.scale[a] = g.scl[a]; }
    xfm.splitComponents = g.splitComponents;
    std::vector<double> wv; std::vector<unsigned> wt; double lo[3], hi[3];
    SceneDesc::MeshWorldGeometry(*raw, xfm, wv, wt, lo, hi);

    // ---- how many closed shells is this file, and which signing mode ---- Counted here from
    // the file's own topology so the panel can say it before Validate.
    int components = 0;
    {
        const size_t nv = raw->VertexCount(), nt = raw->TriangleCount();
        std::vector<int> comp(nv, -1);
        std::vector<std::vector<unsigned>> adj(nv);
        for (size_t k = 0; k < nt; k++)
        {
            const unsigned a = raw->tri[3 * k], b = raw->tri[3 * k + 1], c = raw->tri[3 * k + 2];
            adj[a].push_back(b); adj[b].push_back(c); adj[c].push_back(a);
            adj[b].push_back(a); adj[c].push_back(b); adj[a].push_back(c);
        }
        std::vector<unsigned> stack;
        for (size_t v = 0; v < nv; v++)
        {
            if (comp[v] != -1) continue;
            comp[v] = components; stack.push_back((unsigned)v);
            while (!stack.empty())
            {
                const unsigned u = stack.back(); stack.pop_back();
                for (unsigned w : adj[u]) if (comp[w] == -1) { comp[w] = components; stack.push_back(w); }
            }
            components++;
        }
    }

    ImGui::Checkbox("Union the components", &g.splitComponents);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Treat each closed part separately, as assembled models need");
    ImGui::SameLine();
    if (components > 1 && !g.splitComponents)
        ImGui::TextColored(ToneColour(UiTone::Error),
                           "%d closed shells -- OFF will hollow the interior", components);
    else if (components > 1)
        ImGui::TextDisabled("%d closed shells, unioned", components);
    else
        ImGui::TextDisabled("one closed shell (the setting makes no difference)");

    ImGui::TextDisabled("%zu triangles, %zu vertices; file extent %.4g x %.4g x %.4g%s",
                        raw->TriangleCount(), raw->VertexCount(),
                        raw->hi[0] - raw->lo[0], raw->hi[1] - raw->lo[1], raw->hi[2] - raw->lo[2],
                        raw->flipped ? "  (inward normals, reversed)" : "");
    ImGui::TextDisabled("placed: x [%.4f, %.4f]  y [%.4f, %.4f]  z [%.4f, %.4f] m",
                        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);

    double dh; int Nx, Ny, Nz;
    DerivedGrid(dh, Nx, Ny, Nz);
    const int axes = (dimension == 3) ? 3 : 2;
    const double d0[3] = {x0, y0, z0}, d1[3] = {xf, yf, zf};
    static const char *an[] = {"x", "y", "z"};
    for (int a = 0; a < axes; a++)
    {
        if (lo[a] < d0[a] || hi[a] > d1[a])
            ImGui::TextColored(ToneColour(UiTone::Warn),
                               "Extends past the domain in %s; the walls cut it off there.", an[a]);
        // Anything thinner than two cells voxelises to one layer or to nothing.
        if (dh > 0.0 && (hi[a] - lo[a]) < 2.0 * dh)
            ImGui::TextColored(ToneColour(UiTone::Warn),
                               "Only %.1f cells thick in %s (dh = %.4g m): it will voxelise to "
                               "one layer or vanish.", (hi[a] - lo[a]) / dh, an[a], dh);
    }
    if (dimension == 2)
        ImGui::TextDisabled("2D: the model is cut at its mid-plane, z = %.4f m.", 0.5 * (lo[2] + hi[2]));
}

void SimulationSetupUI::Render3DPreview()
{
    {
        std::string why;
        if (!GridPreviewable(why))
        {
            ImGui::TextColored(ToneColour(UiTone::Warn), "%s", why.c_str());
            return;
        }
    }
    ImGui::Checkbox("Models", &show3DModels);
    ImGui::SameLine();
    ImGui::BeginDisabled(dimension != 3);
    ImGui::Checkbox("Voxelised solid", &show3DVoxels);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Solid cells the solver will see, from the last Validate");
    ImGui::SameLine();
    ImGui::BeginDisabled(dimension != 3);
    ImGui::Checkbox("Fluid regions", &show3DFluid);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Liquid seeded by fluid regions, from the last Validate");
    ImGui::SameLine();
    ImGui::Checkbox("Grid", &show3DGrid);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Cell faces on three domain walls");
    ImGui::SameLine();
    if (ImGui::Button("Fit view##3d")) preview3DFit = true;

    // In 2D the domain has no z extent of its own; show the span of the models so the
    // cut plane has something to sit in.
    double zlo = z0, zhi = zf;
    if (dimension == 2)
    {
        zlo = 1e300; zhi = -1e300;
        for (const UiRegion &g : uiRegions)
        {
            if (g.prim != 4 || g.meshFile.empty()) continue;
            if (const MeshDraw *d = PlacedMeshForDraw(g))
                for (const ImPlot3DPoint &p : d->vtx) { zlo = (std::min)(zlo, p.y); zhi = (std::max)(zhi, p.y); }
        }
        if (!(zhi > zlo)) { zlo = -0.5 * (std::min)(xf - x0, yf - y0); zhi = -zlo; }
    }

    const double Lx = xf - x0, Ly = yf - y0, Lz = zhi - zlo;
    const double Lmax = (std::max)(Lx, (std::max)(Ly, Lz));
    if (!(Lmax > 0.0)) { ImGui::TextDisabled("Set a domain first."); return; }

    // The plot fills what is left of the window; the wheel zooms it. The limits are re-set only
    // when the box changes or on Fit, so the zoom survives the next frame.
    const ImVec2 avail3 = ImGui::GetContentRegionAvail();
    const float side = (std::max)(200.0f, (std::min)(avail3.x, avail3.y - ImGui::GetFrameHeightWithSpacing()));
    if (!ImPlot3D::BeginPlot("##scene3d", ImVec2(side, side)))
        return;
    ImPlot3D::SetupAxes("X (m)", "Z (m)", "Y (m)");
    {
        const double key[6] = {x0, xf, zlo, zhi, y0, yf};
        bool changed = preview3DFit;
        for (int n = 0; n < 6; n++) changed = changed || key[n] != preview3DKey[n];
        if (changed)
        {
            for (int n = 0; n < 6; n++) preview3DKey[n] = key[n];
            preview3DFit = false;
            ImPlot3D::SetupAxesLimits(x0, xf, zlo, zhi, y0, yf, ImPlot3DCond_Always);
        }
    }
    ImPlot3D::SetupBoxScale(Lx / Lmax, Lz / Lmax, Ly / Lmax);

    // Segment buffers, in plot order (x, z, y).
    std::vector<double> sx, sy, sz;

    if (show3DGrid)
    {
        std::vector<double> fx, fy, fz;
        DerivedAxes(fx, fy);
        if (dimension == 3)
        {
            double dh; int Nx, Ny, Nz;
            DerivedGrid(dh, Nx, Ny, Nz);
            BuildAxisPreview(z0, zf, Nz, dh, refineZ, nonUniformGrid && RectilinearSupported(), fz);
        }
        const size_t stx = GridLineStride(fx.size()), sty = GridLineStride(fy.size()), stz = GridLineStride(fz.size());
        if (dimension == 3 && fz.size() > 1)
        {
            // floor (y = y0), back wall (z = z0), side wall (x = x0)
            for (size_t j = 0; j < fx.size(); j += stx) { PushSegment(sx, sy, sz, fx[j], y0, z0, fx[j], y0, zf); PushSegment(sx, sy, sz, fx[j], y0, z0, fx[j], yf, z0); }
            for (size_t k = 0; k < fz.size(); k += stz) { PushSegment(sx, sy, sz, x0, y0, fz[k], xf, y0, fz[k]); PushSegment(sx, sy, sz, x0, y0, fz[k], x0, yf, fz[k]); }
            for (size_t i = 0; i < fy.size(); i += sty) { PushSegment(sx, sy, sz, x0, fy[i], z0, xf, fy[i], z0); PushSegment(sx, sy, sz, x0, fy[i], z0, x0, fy[i], zf); }
        }
        else
        {
            // 2D: the grid lies in the plane the models are cut at.
            const double zc = 0.5 * (zlo + zhi);
            for (size_t j = 0; j < fx.size(); j += stx) PushSegment(sx, sy, sz, fx[j], y0, zc, fx[j], yf, zc);
            for (size_t i = 0; i < fy.size(); i += sty) PushSegment(sx, sy, sz, x0, fy[i], zc, xf, fy[i], zc);
        }
        if (!sx.empty())
        {
            ImPlot3D::SetNextLineStyle(ImVec4(0.55f, 0.62f, 0.68f, 0.35f), 1.0f);
            ImPlot3D::PlotLine("grid", sx.data(), sy.data(), sz.data(), (int)sx.size(),
                               ImPlot3DLineFlags_Segments);
        }
    }

    // domain box
    sx.clear(); sy.clear(); sz.clear();
    const double bx[2] = {x0, xf}, by[2] = {y0, yf}, bz[2] = {zlo, zhi};
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
        {
            PushSegment(sx, sy, sz, x0, by[a], bz[b], xf, by[a], bz[b]);
            PushSegment(sx, sy, sz, bx[a], y0, bz[b], bx[a], yf, bz[b]);
            PushSegment(sx, sy, sz, bx[a], by[b], zlo, bx[a], by[b], zhi);
        }
    ImPlot3D::SetNextLineStyle(ImGui::GetStyleColorVec4(ImGuiCol_Text), 1.5f);
    ImPlot3D::PlotLine("domain", sx.data(), sy.data(), sz.data(), (int)sx.size(),
                       ImPlot3DLineFlags_Segments);

    if (show3DVoxels && dimension == 3 && SCENE.loaded && !voxX.empty())
    {
        const ImVec4 c(0.95f, 0.55f, 0.25f, 0.55f);
        ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Square, 2.5f, c, 0.0f, c);
        ImPlot3D::PlotScatter("solid cells", voxX.data(), voxZ.data(), voxY.data(), (int)voxX.size());
    }

    if (show3DFluid && dimension == 3 && SCENE.loaded && !fluX.empty())
    {
        const ImVec4 c(0.25f, 0.55f, 0.95f, 0.45f);
        ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Square, 2.5f, c, 0.0f, c);
        ImPlot3D::PlotScatter("fluid cells", fluX.data(), fluZ.data(), fluY.data(), (int)fluX.size());
    }

    if (show3DModels)
    {
        int n = 0;
        for (const UiRegion &g : uiRegions)
        {
            if (g.prim != 4 || g.meshFile.empty()) continue;
            const MeshDraw *d = PlacedMeshForDraw(g);
            if (!d || d->idx.empty()) continue;
            const ImVec4 fill = (g.kind == 0) ? ImVec4(0.70f, 0.72f, 0.76f, 1.0f)
                              : (g.kind == 1) ? ImVec4(0.20f, 0.55f, 0.95f, 1.0f)
                                              : ImVec4(0.30f, 0.30f, 0.32f, 1.0f);
            ImPlot3D::SetNextFillStyle(fill, 0.55f);
            ImPlot3D::SetNextLineStyle(ImVec4(0.15f, 0.17f, 0.2f, 0.8f), 1.0f);
            char label[64];
            n++;
            std::snprintf(label, sizeof(label), "model %d##m%d", n, n);
            ImPlot3D::PlotMesh(label, d->vtx.data(), d->idx.data(), (int)d->vtx.size(),
                               (int)d->idx.size(), ImPlot3DMeshFlags_NoMarkers);

            // 2D: the section the solver actually uses.
            if (dimension == 2)
            {
                double zmin = 1e300, zmax = -1e300;
                for (const ImPlot3DPoint &p : d->vtx) { zmin = (std::min)(zmin, p.y); zmax = (std::max)(zmax, p.y); }
                const double zc = 0.5 * (zmin + zmax);
                const double qx[4] = {x0, xf, xf, x0}, qy[4] = {zc, zc, zc, zc}, qz[4] = {y0, y0, yf, yf};
                ImPlot3D::SetNextFillStyle(ImVec4(0.3f, 0.8f, 0.5f, 1.0f), 0.18f);
                ImPlot3D::PlotQuad("2D section", qx, qy, qz, 4);
            }
        }
    }
    // The probes, in plot order (x, z, y) like everything else in this view.
    if (!uiProbes.empty())
    {
        std::vector<double> px, py, pz;
        for (const Probe &pr : uiProbes)
        { px.push_back(pr.at[0]); pz.push_back(pr.at[2]); py.push_back(pr.at[1]); }
        ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Cross, 5.0f, ImVec4(1, 1, 1, 1), 1.5f, ImVec4(1, 1, 1, 1));
        ImPlot3D::PlotScatter("probes", px.data(), pz.data(), py.data(), (int)px.size());
    }

    ImPlot3D::EndPlot();

    if ((show3DVoxels || show3DFluid) && dimension == 3)
    {
        if (!SCENE.loaded)
            ImGui::TextDisabled("Validate the scene to voxelise it.");
        else
            ImGui::TextDisabled("%zu surface solid cells, %zu surface fluid cells, from the last Validate.",
                                voxX.size(), fluX.size());
    }
    ImGui::TextDisabled("The voxels change only on Validate.");
}

void SimulationSetupUI::DrawProbeMarkers2D()
{
    if (uiProbes.empty()) return;
    int ha = 0, va = 1;                       // XY
    if (dimension == 3 && previewPlane == 1) { ha = 0; va = 2; }   // XZ
    else if (dimension == 3 && previewPlane == 2) { ha = 2; va = 1; }   // ZY

    ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 5.0f);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 5.0f, ImVec4(1, 1, 1, 1), 1.5f, ImVec4(1, 1, 1, 1));
    for (const Probe &pr : uiProbes)
    {
        const double px = pr.at[ha], py = pr.at[va];
        ImPlot::PlotScatter("##probe", &px, &py, 1);
        if (pr.radius > 0.0)
        {
            double rx[33], ry[33];
            for (int k = 0; k <= 32; k++)
            {
                const double a = 2.0 * M_PI * k / 32.0;
                rx[k] = px + pr.radius * std::cos(a);
                ry[k] = py + pr.radius * std::sin(a);
            }
            ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 0.8f), 1.0f);
            ImPlot::PlotLine("##probeR", rx, ry, 33);
        }
        ImPlot::PlotText(pr.name.c_str(), px, py, ImVec2(10, -10));
    }
    ImPlot::PopStyleVar();
}

// The colours of the slice preview, one per cell type plus the liquid FLIP would seed (the legend below matches).
static const unsigned char PREVIEW_COLOURS[9][3] = {
    {41, 51, 71},     // 0 fluid domain (background)
    {140, 140, 148},  // 1 no-slip wall / solid region
    {26, 28, 33},     // 2 pressure outlet, p = 0
    {230, 115, 51},   // 3 inlet
    {51, 140, 242},   // 4 liquid seeded by region fluid
    {242, 204, 64},   // 5 moving wall
    {166, 89, 217},   // 6 pressure outlet, p prescribed
    {102, 184, 168},  // 7 slip wall
    {217, 77, 115},   // 8 continuity outflow
};

// The slice preview as one OpenGL texture, re-uploaded only when BuildPreview has produced a new image. Nearest
// filtering keeps every cell a sharp block at any zoom.
static unsigned PreviewTexture(const std::vector<double> &data, int rows, int cols, unsigned version)
{
    static GLuint tex = 0;
    static unsigned built = ~0u;
    if (tex == 0) glGenTextures(1, &tex);
    if (version == built || rows <= 0 || cols <= 0 || data.size() < (size_t)rows * cols) return tex;
    std::vector<unsigned char> rgba((size_t)rows * cols * 4);
    for (size_t n = 0; n < (size_t)rows * cols; n++)
    {
        const int c = (std::max)(0, (std::min)(8, (int)std::lround(data[n])));
        rgba[4 * n + 0] = PREVIEW_COLOURS[c][0];
        rgba[4 * n + 1] = PREVIEW_COLOURS[c][1];
        rgba[4 * n + 2] = PREVIEW_COLOURS[c][2];
        rgba[4 * n + 3] = 255;
    }
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    built = version;
    return tex;
}

// One run of ring pixels of the same kind, drawn as a strip in plot coordinates.
static void RingStrip(ImDrawList *dl, int code, double xa, double ya, double xb, double yb)
{
    if (code == 0 || code == 4) return;   // fluid, liquid: not a boundary
    const unsigned char *c = PREVIEW_COLOURS[(std::max)(0, (std::min)(8, code))];
    const ImVec2 a = ImPlot::PlotToPixels(xa, ya), b = ImPlot::PlotToPixels(xb, yb);
    dl->AddRectFilled(ImVec2((std::min)(a.x, b.x), (std::min)(a.y, b.y)), ImVec2((std::max)(a.x, b.x), (std::max)(a.y, b.y)),
                      IM_COL32(c[0], c[1], c[2], 255));
}

// The boundary ring over the slice image, read off its pinned edge pixels and drawn at least `minPx`
// thick, so it survives minification on fine grids. `hf`, `vf`: face tables across and up.
static void DrawPreviewRing(const std::vector<double> &data, int rows, int cols, double ax0, double av0, double w,
                            double h, const std::vector<double> &hf, const std::vector<double> &vf)
{
    if (rows < 3 || cols < 3 || hf.size() < 4 || vf.size() < 4 || data.size() < (size_t)rows * cols) return;
    const float minPx = 3.0f;
    const ImPlotPoint p0 = ImPlot::PixelsToPlot(ImVec2(0.0f, 0.0f)), p1 = ImPlot::PixelsToPlot(ImVec2(1.0f, 1.0f));
    const double pxX = std::fabs(p1.x - p0.x), pxY = std::fabs(p1.y - p0.y);
    const double xl = hf[1], xr = hf[hf.size() - 2], yb = vf[1], yt = vf[vf.size() - 2];
    const double tl = (std::max)(xl - hf.front(), minPx * pxX), tr = (std::max)(hf.back() - xr, minPx * pxX);
    const double tb = (std::max)(yb - vf.front(), minPx * pxY), tt = (std::max)(vf.back() - yt, minPx * pxY);
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    ImPlot::PushPlotClipRect();
    // Bottom (stored row rows - 1) and top (row 0): columns across, the corners stretched to the side strips.
    for (int side = 0; side < 2; side++)
    {
        const int row = side == 0 ? rows - 1 : 0;
        const double ya = side == 0 ? yb - tb : yt, yb2 = side == 0 ? yb : yt + tt;
        int c = 0;
        while (c < cols)
        {
            const int code = (int)std::lround(data[(size_t)row * cols + c]);
            int e = c + 1;
            while (e < cols && (int)std::lround(data[(size_t)row * cols + e]) == code) e++;
            const double xa = (c == 0) ? xl - tl : ax0 + c * w / cols;
            const double xb = (e == cols) ? xr + tr : ax0 + e * w / cols;
            RingStrip(dl, code, xa, ya, xb, yb2);
            c = e;
        }
    }
    // Left (column 0) and right (column cols - 1): rows up, the corners left to the bottom and top strips.
    for (int side = 0; side < 2; side++)
    {
        const int col = side == 0 ? 0 : cols - 1;
        const double xa = side == 0 ? xl - tl : xr, xb = side == 0 ? xl : xr + tr;
        int r = 1;
        while (r < rows - 1)
        {
            const int code = (int)std::lround(data[(size_t)r * cols + col]);
            int e = r + 1;
            while (e < rows - 1 && (int)std::lround(data[(size_t)e * cols + col]) == code) e++;
            // Stored row r is the top of the image: it spans y from av0 + h - (r + 1) h / rows up to av0 + h - r h / rows.
            const double ytop = (r == 1) ? yt : av0 + h - r * h / rows;
            const double ybot = (e == rows - 1) ? yb : av0 + h - e * h / rows;
            RingStrip(dl, code, xa, ybot, xb, ytop);
            r = e;
        }
    }
    ImPlot::PopPlotClipRect();
}

void SimulationSetupUI::RenderScenePreview()
{

    const ImGuiViewport *vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.55f, vp->WorkPos.y),
                            ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x * 0.45f, vp->WorkSize.y * 0.62f),
                             ImGuiCond_Always);
    ImGui::Begin("Scene preview", nullptr, ImGuiWindowFlags_NoScrollbar);
    if (ImGui::BeginTabBar("##previewTabs"))
    {
        if (ImGui::BeginTabItem("Slice"))
        {
            RenderSlicePreview();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("3D"))
        {
            if (previewDirty && SCENE.loaded)
                BuildPreview();
            Render3DPreview();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void SimulationSetupUI::RenderSlicePreview()
{
    if (!SCENE.loaded)
    {
        ImGui::TextDisabled("Press \"Validate / preview\" to rasterise the scene.");
        return;
    }
    {
        std::string why;
        if (!GridPreviewable(why))
        {
            ImGui::TextColored(ToneColour(UiTone::Warn), "%s", why.c_str());
            return;
        }
    }

    if (dimension == 3)
    {
        static const char *planeNames[] = {"XY (fix Z)", "XZ (fix Y)", "ZY (fix X)"};
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Plane");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(FormEm() * 12.0f);
        if (ImGui::Combo("##plane", &previewPlane, planeNames, 3)) previewDirty = true;

        const int maxSlice = PreviewSliceCount() - 1;
        ImGui::SameLine();
        ImGui::TextUnformatted("Slice");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(FormEm() * 18.0f);
        if (ImGui::SliderInt("##slice", &previewSlice, 0, (std::max)(0, maxSlice))) previewDirty = true;
    }
    if (dimension == 3) ImGui::SameLine();
    if (ImGui::Button("Fit view##slice")) previewFit = true;

    if (previewDirty)
        BuildPreview();

    if (previewData.empty())
    {
        ImGui::TextDisabled("Nothing to draw.");
        return;
    }

    // The plot takes what the window has left above the legend (five rows of two columns and a
    // status line); the domain is framed inside it with equal axes, so no aspect sum is needed.
    const float legendH = 7.0f * ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    // As tall as the domain needs at the window's width (axes included), at most what is left.
    float domainAspect = 1.0f;
    {
        std::vector<double> ahf, avf, aff;
        PreviewAxes(ahf, avf, aff);
        if (ahf.size() > 1 && avf.size() > 1 && avf.back() > avf.front())
            domainAspect = (float)((ahf.back() - ahf.front()) / (avf.back() - avf.front()));
    }
    const float plotW = (std::max)(150.0f, avail.x);
    const float plotH = (std::max)(150.0f, (std::min)(avail.y - legendH, (plotW - 70.0f) / domainAspect + 60.0f));
    const ImVec2 plotSize(plotW, plotH);


    const ImTextureID previewTex = (ImTextureID)PreviewTexture(previewData, previewRows, previewCols, previewVersion);
    if (ImPlot::BeginPlot("##scenePreview", plotSize,
                          ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText | ImPlotFlags_Equal))
    {
        double dh; int Nx, Ny, Nz;
        DerivedGrid(dh, Nx, Ny, Nz);
        // The span the GRID actually covers, and where it starts: from the face tables, ring included -- the ring
        // lies one cell outside the domain, and on a stretched grid the image size is not the cell count.
        double ax0 = (dimension == 3 && previewPlane == 2) ? z0 : x0;
        double av0 = (dimension == 3 && previewPlane == 1) ? z0 : y0;
        double w = previewCols * dh, h = previewRows * dh;
        {
            std::vector<double> ehf, evf, eff;
            PreviewAxes(ehf, evf, eff);
            if (ehf.size() > 1 && evf.size() > 1)
            {
                ax0 = ehf.front(); av0 = evf.front();
                w = ehf.back() - ehf.front();
                h = evf.back() - evf.front();
            }
        }
        const char *hl = "X (m)", *vl = "Y (m)";
        if (dimension == 3)
        {
            if (previewPlane == 1) { hl = "X (m)"; vl = "Z (m)"; }
            else if (previewPlane == 2) { hl = "Z (m)"; vl = "Y (m)"; }
        }
        ImPlot::SetupAxes(hl, vl, 0, 0);
        {
            const double key[7] = {ax0, av0, w, h, std::floor(plotSize.x), std::floor(plotSize.y),
                                   (double)(dimension * 10 + previewPlane)};
            bool changed = previewFit;
            for (int n = 0; n < 7; n++) changed = changed || key[n] != previewViewKey[n];
            if (changed)
            {
                for (int n = 0; n < 7; n++) previewViewKey[n] = key[n];
                previewFit = false;
                // The whole domain in view: equal axes then widen whichever axis has room to spare.
                const double mx = 0.025 * w, my = 0.025 * h;   // 2.5% each side, as the field view
                ImPlot::SetupAxisLimits(ImAxis_X1, ax0 - mx, ax0 + w + mx, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, av0 - my, av0 + h + my, ImGuiCond_Always);
            }
        }
        // One textured quad, not a heatmap: PlotHeatmap emits a rectangle per cell every frame, and a fine grid is
        // millions of them. Row 0 of the image is the top, which is texture coordinate v = 0.
        ImPlot::PlotImage("##cells", previewTex, ImPlotPoint(ax0, av0), ImPlotPoint(ax0 + w, av0 + h));
        {
            std::vector<double> rhf, rvf, rff;
            PreviewAxes(rhf, rvf, rff);
            DrawPreviewRing(previewData, previewRows, previewCols, ax0, av0, w, h, rhf, rvf);
        }
        DrawProbeMarkers2D();
        ImPlot::EndPlot();
    }

    ImGui::Spacing();
    struct Legend { ImVec4 c; const char *label; };
    const Legend legend[] = {
        {ImVec4(0.16f, 0.20f, 0.28f, 1), "Fluid domain (solved)"},
        {ImVec4(0.55f, 0.55f, 0.58f, 1), "No-slip wall / solid region"},
        {ImVec4(0.40f, 0.72f, 0.66f, 1), "Slip wall"},
        {ImVec4(0.95f, 0.80f, 0.25f, 1), "Moving wall"},
        {ImVec4(0.90f, 0.45f, 0.20f, 1), "Inlet (u given, p Neumann)"},
        {ImVec4(0.10f, 0.11f, 0.13f, 1), "Pressure outlet, p = 0"},
        {ImVec4(0.65f, 0.35f, 0.85f, 1), "Pressure outlet, p given"},
        {ImVec4(0.85f, 0.30f, 0.45f, 1), "Outflow (flux = inlets)"},
        {ImVec4(0.20f, 0.55f, 0.95f, 1), "Liquid (region fluid)"},
    };
    if (ImGui::BeginTable("##legend", 2, ImGuiTableFlags_SizingStretchSame))
    {
        for (const Legend &l : legend)
        {
            ImGui::TableNextColumn();
            ImGui::ColorButton(l.label, l.c, ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14));
            ImGui::SameLine();
            ImGui::TextUnformatted(l.label);
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("%d x %d cells", previewCols, previewRows);
    if (MethodSupports(method, CAP_PARTICLES) && !SCENE.HasFluidRegions())
        ImGui::TextColored(ToneColour(UiTone::Warn), "No `region fluid`: a particle method would seed nothing.");
}

bool SimulationSetupUI::Render()
{
    bool start = false;

    const ImGuiViewport *vp = ImGui::GetMainViewport();
    // Leave room on the right for the scene preview.
    const float mainW = vp->WorkSize.x * 0.55f;
    ImGui::SetNextWindowPos(vp->WorkPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(mainW, vp->WorkSize.y), ImGuiCond_Always);
    ImGui::Begin("NEREUS - Simulation Setup", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::TextUnformatted("NEREUS");
    ImGui::SameLine();
    ImGui::Separator();

    // ---------------- saved configurations ----------------
    {
        ImGui::TextUnformatted("Configuration");
        ImGui::SameLine();

        std::string combo = presets.empty() ? "(none saved)"
                                            : (presetIdx >= 0 && presetIdx < (int)presets.size()
                                                   ? presets[presetIdx]
                                                   : "select...");
        ImGui::SetNextItemWidth(220);
        if (ImGui::BeginCombo("##preset", combo.c_str()))
        {
            for (int i = 0; i < (int)presets.size(); i++)
            {
                const bool sel = (i == presetIdx);
                if (ImGui::Selectable(presets[i].c_str(), sel))
                {
                    presetIdx = i;
                    LoadPreset(presets[i]);
                    RememberPreset(presets[i]);
                }
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Saved setups in %s/. Save and Start write back into the selected one", PRESET_DIR);

        ImGui::SameLine();
        ImGui::SetNextItemWidth(160);
        ImGui::InputTextWithHint("##presetName", "name to save as", presetName,
                                 IM_ARRAYSIZE(presetName));
        ImGui::SameLine();
        if (ImGui::Button("Save as"))
        {
            std::string err;
            if (SavePreset(presetName, err))
            {
                RememberPreset(presetName);
                RefreshPresets();
                for (int i = 0; i < (int)presets.size(); i++)
                    if (presets[i] == presetName) presetIdx = i;
                statusMessage = std::string("Saved preset '") + presetName + "'.";
                statusIsError = false;
            }
            else { statusMessage = err; statusIsError = true; }
        }
        ImGui::SameLine();
        if (ImGui::Button("Refresh")) RefreshPresets();
        ImGui::SameLine();
        const bool canDelete = (presetIdx >= 0 && presetIdx < (int)presets.size());
        if (!canDelete) ImGui::BeginDisabled();
        if (ImGui::Button("Delete"))
        {
            // Deleting the current file would leave Start writing it straight back.
            if (configPath == PresetPath(presets[presetIdx], ".cfg"))
            {
                UseFile("simulation_config.txt");
                ForgetPreset();
            }
            std::remove(PresetPath(presets[presetIdx], ".cfg").c_str());
            statusMessage = "Deleted preset '" + presets[presetIdx] + "'.";
            statusIsError = false;
            presetIdx = -1;
            RefreshPresets();
        }
        if (!canDelete) ImGui::EndDisabled();
    }
    ImGui::Separator();

    ImGui::BeginChild("##setupScroll", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 3.2f));
    ImGui::PushItemWidth(260);

    // ---------------- 1. Solver ----------------
    if (ImGui::CollapsingHeader("1. Solver", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent();
        int prevDim = dimension;
        int prevMethod = method;

        ImGui::TextUnformatted("Dimension");
        ImGui::RadioButton("2D", &dimension, 2);
        ImGui::SameLine();
        ImGui::RadioButton("3D", &dimension, 3);

        ImGui::Spacing();
        ImGui::TextUnformatted("Method");
        // Straight off FAMILY_TABLE and the registered METHODS, so a method appears by existing, under its
        // family's heading.
        for (const MethodFamilyInfo &fam : FAMILY_TABLE)
        {
            int inFamily = 0;
            for (int m = 0; m < METHOD_COUNT; m++)
                if (METHODS[m].family == fam.family) inFamily++;
            if (inFamily == 0) continue;   // a family nothing implements yet draws no heading

            ImGui::SeparatorText(fam.label);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", fam.help);

            ImGui::Indent();
            for (int m = 0; m < METHOD_COUNT; m++)
            {
                const SimulationMethod &mi = METHODS[m];
                if (mi.family != fam.family) continue;
                const bool ok = MethodSupported(dimension, m);
                if (!ok) ImGui::BeginDisabled();
                if (ImGui::RadioButton(mi.label, method == m)) method = m;
                if (!ok) ImGui::EndDisabled();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", mi.help);
            }
            ImGui::Unindent();
        }
        ImGui::Spacing();

        // Every method runs its loops on OpenMP, so this is the solver's, not the pressure solve's.
        FormSliderInt("OpenMP threads", &threadCount, 1, 32);

        if (dimension != prevDim || method != prevMethod)
            ReconcileSelections();
        ImGui::Unindent();
    }

    // ---------------- 2. Geometry ----------------
    if (ImGui::CollapsingHeader("2. Geometry", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent();

        {
            // The extents live with the scene: the regions below are written in metres
            // against them, so the two are only meaningful read together.
            ImGui::SeparatorText("Domain");
            DrawDomainExtents();

            // ----- boundaries, in the order they will be written -----
            ImGui::SeparatorText("Boundaries (last match wins at a corner)");
            static const char *bcNames[] = {"noslip", "slip", "moving", "inlet", "pressure_outlet", "outflow"};
            for (int n = 0; n < 6; n++)
            {
                const int fi = faceOrder[n];
                if (dimension == 2 && (fi == 4 || fi == 5))
                    continue;
                ImGui::PushID(1000 + n);

                ImGui::Text("%d.", n + 1);
                ImGui::SameLine();
                if (ImGui::ArrowButton("##up", ImGuiDir_Up) && n > 0)
                { std::swap(faceOrder[n], faceOrder[n - 1]); previewDirty = true; }
                ImGui::SameLine();
                if (ImGui::ArrowButton("##down", ImGuiDir_Down) && n < 5)
                { std::swap(faceOrder[n], faceOrder[n + 1]); previewDirty = true; }

                ImGui::SameLine();
                ImGui::Text("%-5s", FaceName(fi));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(FormEm() * 19.0f);
                if (ImGui::Combo("##bc", &uiFaces[fi].type, bcNames, IM_ARRAYSIZE(bcNames)))
                    previewDirty = true;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("noslip           zero velocity\n"
                                  "slip             zero normal velocity, free tangential\n"
                                  "moving           wall sliding at a set speed\n"
                                  "inlet            set velocity\n"
                                  "pressure_outlet  set pressure, 0 if empty\n"
                                  "outflow          zero gradient, flux matches the inlets");

                // A pressure outlet carries the Dirichlet value for the pressure solve; empty is 0.
                if (uiFaces[fi].type == BC_PRESSURE_OUTLET)
                {
                    ImGui::Indent();
                    FormLabel("Pressure", "Outlet pressure, an expression in x y z t. Empty is 0");
                    ImGui::SetNextItemWidth(FormWideWidth());
                    if (ImGui::InputTextWithHint("##p", "0", &uiFaces[fi].p))
                        previewDirty = true;
                    FormUnit("Pa");
                    uiFaces[fi].hasPressure = !uiFaces[fi].p.empty();
                    ImGui::Unindent();
                }

                // An inflow can carry the pressure LEVEL of a case whose every face prescribes velocity:
                // there the pressure is Neumann and p= only fixes its constant.
                if (uiFaces[fi].type == BC_INLET)
                {
                    ImGui::Indent();
                    if (ImGui::Checkbox("Pressure level", &uiFaces[fi].hasPressure))
                        previewDirty = true;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Sets only the pressure level of a manufactured case");
                    if (uiFaces[fi].hasPressure)
                    {
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(FormWideWidth());
                        ImGui::InputText("##level", &uiFaces[fi].p);
                        FormUnit("Pa");
                    }
                    ImGui::Unindent();
                }

                if (uiFaces[fi].type == BC_MOVING || uiFaces[fi].type == BC_INLET)
                {
                    ImGui::Indent();
                    const int nax = NormalAxisOf(fi);
                    const int ncomp = (dimension == 3) ? 3 : 2;
                    std::string *fields[3] = {&uiFaces[fi].u, &uiFaces[fi].v, &uiFaces[fi].w};
                    const char *labels[3] = {"u", "v", "w"};
                    const char *normalLabels[3] = {"u (normal)", "v (normal)", "w (normal)"};
                    for (int a = 0; a < ncomp; a++)
                    {
                        if (a == nax && uiFaces[fi].type == BC_MOVING)
                        {
                            FormLabel(normalLabels[a]);
                            ImGui::TextDisabled("0, the wall slides along itself");
                            continue;
                        }
                        FormLabel(a == nax ? normalLabels[a] : labels[a],
                                  "An expression in x y z t dh Lx Ly Lz, or a number");
                        ImGui::PushID(a);
                        ImGui::SetNextItemWidth(FormWideWidth());
                        ImGui::InputText("##vel", fields[a]);
                        ImGui::PopID();
                        FormUnit("m/s");
                    }
                    if (uiFaces[fi].type == BC_INLET)
                        ImGui::Checkbox("clamp negative u", &uiFaces[fi].clampNegative);
                    ImGui::Unindent();
                }
                ImGui::PopID();
            }

            // ----- regions -----
            ImGui::SeparatorText("Regions (later entries override earlier ones)");
            static const char *kindNames[] = {"solid", "fluid", "empty"};
            static const char *primNames[] = {"box", "sphere", "cylinder", "plane", "mesh", "naca 00XX"};
            const bool flat = (dimension == 2);
            int removeAt = -1;
            for (size_t r = 0; r < uiRegions.size(); r++)
            {
                UiRegion &g = uiRegions[r];
                ImGui::PushID(2000 + (int)r);
                ImGui::AlignTextToFramePadding();
                ImGui::Text("Region %d", (int)r + 1);
                ImGui::SameLine();
                if (ImGui::ArrowButton("##rup", ImGuiDir_Up) && r > 0)
                { std::swap(uiRegions[r], uiRegions[r - 1]); previewDirty = true; }
                ImGui::SameLine();
                if (ImGui::ArrowButton("##rdn", ImGuiDir_Down) && r + 1 < uiRegions.size())
                { std::swap(uiRegions[r], uiRegions[r + 1]); previewDirty = true; }
                ImGui::SameLine();
                if (ImGui::Button("Remove")) removeAt = (int)r;

                ImGui::Indent();
                if (FormCombo("Kind", &g.kind, kindNames, IM_ARRAYSIZE(kindNames),
                              "Solid is an obstacle. Fluid seeds liquid at t = 0 for particle methods. Empty is air"))
                    previewDirty = true;
                if (FormCombo("Shape", &g.prim, primNames, IM_ARRAYSIZE(primNames)))
                {
                    previewDirty = true;
                    // A fresh mesh region starts on the first model, centred in the
                    // domain -- the file's own origin is arbitrary, the domain is not.
                    if (g.prim == 4)
                    {
                        RefreshModels();
                        if (g.meshFile.empty() && !models.empty()) g.meshFile = models[0];
                        g.a[0] = 0.5 * (x0 + xf); g.a[1] = 0.5 * (y0 + yf); g.a[2] = 0.5 * (z0 + zf);
                    }
                    // A fresh airfoil: centred, a quarter of the domain's length for a chord.
                    if (g.prim == 5)
                    {
                        g.a[0] = 0.5 * (x0 + xf); g.a[1] = 0.5 * (y0 + yf); g.a[2] = 0.5 * (z0 + zf);
                        g.chord = 0.25 * (xf - x0);
                    }
                }
                static const char *shapeNotes2D[] = {"a rectangle between two corners", "a circle",
                                                     "a circle (its axis is z)",
                                                     "the side of a line the normal points away from",
                                                     "the section of a closed STL model at mid-depth",
                                                     "a symmetric 4-digit airfoil"};
                static const char *shapeNotes3D[] = {"an axis-aligned box between two corners", "a sphere",
                                                     "a cylinder along an axis, capped by the walls",
                                                     "the side of a plane the normal points away from",
                                                     "a closed STL model", "a symmetric 4-digit airfoil, extruded"};
                if (g.prim >= 0 && g.prim < IM_ARRAYSIZE(primNames))
                {
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", flat ? shapeNotes2D[g.prim] : shapeNotes3D[g.prim]);
                }

                const int nc = flat ? 2 : 3;
                if (g.prim == 0)
                {
                    if (FormVec("Min corner", g.a, nc, "m")) previewDirty = true;
                    if (FormVec("Max corner", g.b, nc, "m")) previewDirty = true;
                    if (flat)
                        ImGui::TextDisabled("size %.4g x %.4g m", g.b[0] - g.a[0], g.b[1] - g.a[1]);
                    else
                        ImGui::TextDisabled("size %.4g x %.4g x %.4g m", g.b[0] - g.a[0], g.b[1] - g.a[1],
                                            g.b[2] - g.a[2]);
                }
                else if (g.prim == 4)
                    DrawMeshRegionEditor(g);
                else if (g.prim == 5)
                    DrawAirfoilRegionEditor(g, nc);
                else if (g.prim == 3)
                {
                    if (FormVec("Point on the plane", g.a, nc, "m")) previewDirty = true;
                    if (FormVec("Normal", g.b, nc, nullptr, "%.4g", "Points out of the region; its length is ignored"))
                        previewDirty = true;
                }
                else
                {
                    if (FormVec("Centre", g.a, nc, "m")) previewDirty = true;
                    if (FormDouble("Radius", &g.radius, "m", "%.5g")) previewDirty = true;
                    if (g.prim == 2 && !flat)
                    {
                        static const char *axisNames[] = {"x", "y", "z"};
                        if (FormCombo("Axis", &g.axis, axisNames, 3)) previewDirty = true;
                    }
                }
                if (ImGui::Checkbox("Invert (keep the outside)", &g.invert)) previewDirty = true;
                ImGui::Unindent();
                ImGui::Spacing();
                ImGui::PopID();
            }
            if (removeAt >= 0)
            { uiRegions.erase(uiRegions.begin() + removeAt); previewDirty = true; }

            if (ImGui::Button("Add region"))
            { uiRegions.push_back(UiRegion()); previewDirty = true; }

            // ---- initial field and body force ----
            ImGui::Spacing();
            ImGui::Separator();

            ImGui::Checkbox("Initial velocity field", &sceneHasInitial);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Velocity at t = 0. Without it the run starts from rest");
            if (sceneHasInitial)
            {
                ImGui::Indent();
                std::string *init[3] = {&initU, &initV, &initW};
                static const char *initIds[3] = {"##initu", "##initv", "##initw"};
                static const char *comp[3] = {"u", "v", "w"};
                for (int a = 0; a < ((dimension == 3) ? 3 : 2); a++)
                {
                    FormLabel(comp[a]);
                    ImGui::SetNextItemWidth(FormWideWidth());
                    ImGui::InputText(initIds[a], init[a]);
                    FormUnit("m/s");
                }
                ImGui::TextDisabled("expressions in x y z t dh Lx Ly Lz; evaluated once, at t = 0");
                ImGui::Unindent();
            }

            ImGui::TextDisabled("The body force is in \"5. Fluid\": a number or an expression in x y z t.");
            DrawProbes();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("Validate / preview"))
                ApplyScene();

            if (!sceneStatus.empty())
                ImGui::TextColored(sceneStatusIsError ? ToneColour(UiTone::Error)
                                                      : ToneColour(UiTone::Good),
                                   "%s", sceneStatus.c_str());
        }
        ImGui::Unindent();
    }

    // ---------------- 3.
    if (ImGui::CollapsingHeader("3. Scalar fields", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent();
        DrawScalarFields();
        ImGui::Unindent();
    }

    // ---------------- 4.
    if (ImGui::CollapsingHeader("4. Resolution", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent();
        // SPH has no grid at all: its resolution is the kernel radius and the particle
        // spacing, both of which live in the SPH section rather than here.
        if (UsesGrid())
            DrawResolution();
        else
            ImGui::TextDisabled("SPH is meshless -- resolution is the kernel radius,\n"
                                "set in the SPH parameters section below.");
        ImGui::Unindent();
    }

    // ---------------- 5. Fluid ----------------
    if (ImGui::CollapsingHeader("5. Fluid", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent();
        FormDouble("Kinematic viscosity", &viscosity, "m²/s", "%.6g");
        FormDouble("Density", &density, "kg/m³", "%.4g");

        // ONE body force for every method: a number per component, or an expression in
        // x y z t dh Lx Ly Lz (then written as the scene's `force` line; Eulerian and FLIP).
        ImGui::Spacing();
        ImGui::TextUnformatted("Body force");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Acceleration on every fluid face, a number or an expression in x y z t");
        std::string *force[3] = {&forceU, &forceV, &forceW};
        static const char *forceNames[3] = {"Fx", "Fy", "Fz"};
        for (int a = 0; a < ((dimension == 3) ? 3 : 2); a++)
        {
            FormLabel(forceNames[a], "A number or an expression in x y z t dh Lx Ly Lz");
            ImGui::PushID(a);
            ImGui::SetNextItemWidth(FormWideWidth());
            ImGui::InputText("##force", force[a]);
            ImGui::PopID();
            FormUnit("m/s²");
        }
        ImGui::Unindent();
    }

    // ---------------- 6. Time stepping ----------------
    if (ImGui::CollapsingHeader("6. Time stepping", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent();
        ImGui::Checkbox("Adaptive timestep", &adaptiveTimestep);
        if (adaptiveTimestep)
        {
            FormDouble("Max CFL", &maxCFL, nullptr, "%.3f", "Caps the advection CFL each step");
            FormDouble("Max timestep", &timeStep, "s", "%.6g");
        }
        else
        {
            FormDouble("Timestep", &timeStep, "s", "%.6g");
            double dh; int Nx, Ny, Nz;
            DerivedGrid(dh, Nx, Ny, Nz);
            if (dh > 0.0)
            {
                const double diffNumber = viscosity * timeStep / (dh * dh);
                ImGui::Text("nu*dt/dh^2 = %.3f", diffNumber);
                ImGui::TextDisabled("The advection CFL depends on the flow speed and is not known yet.");
            }
        }

        ImGui::Spacing();
        static const char *stopNames[] = {"never (run until paused)", "at a final time",
                                          "when the residual settles"};
        FormCombo("Stop automatically", &stop.condition, stopNames, IM_ARRAYSIZE(stopNames),
                  "The residual only settles in flows with a steady state", FormEm() * 30.0f);
        if (stop.condition == STOP_TIME)
            FormDouble("Final time", &stop.tFinal, "s", "%.6g");
        else if (stop.condition == STOP_RESIDUAL)
            FormDouble("Residual tolerance", &stop.residualTol, nullptr, "%.3e");
        if (stop.condition != STOP_NEVER)
            ImGui::Checkbox("Export the final field and telemetry", &stop.exportOnStop);
        ImGui::Unindent();
    }

    // ---------------- 7. Particle options (shared by every particle method) ----------------
    int section = 7;
    if (ShowParticleOptions())
    {
        char head[64];
        std::snprintf(head, sizeof(head), "%d. Particle options###particleOptions", section++);
        if (ImGui::CollapsingHeader(head, ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Indent();
            DrawParticleOptions();
            ImGui::Unindent();
        }
    }

    // ---------------- the method's own settings ----------------
    // Its own panel or the default widgets for its table, editing the method's global instance.
    {
        const SimulationMethod &mi = MethodOf(method);
        if (mi.settings && mi.settingCount > 0)
        {
            char head[128];
            std::snprintf(head, sizeof(head), "%d. %s settings###methodSettings", section++, mi.label);
            if (ImGui::CollapsingHeader(head, ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Indent();
                ImGui::PushID(mi.key);
                if (mi.drawSettings) mi.drawSettings(Context());
                else DrawSettingFields(mi.settingFields, mi.settingCount, mi.settings);
                ImGui::PopID();
                ImGui::Unindent();
            }
        }
    }

    // ---------------- pressure solver ---------------- (CAP_PRESSURE_SOLVER only; the block is
    // still written for the others, so switching back keeps it)
    char pressureHead[64];
    std::snprintf(pressureHead, sizeof(pressureHead), "%d. Pressure solver###pressure", section);
    const bool sharedPressure = MethodSupports(method, CAP_PRESSURE_SOLVER);
    if (sharedPressure) section++;
    if (sharedPressure &&
        ImGui::CollapsingHeader(pressureHead, ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent();
#ifdef NVIDIA_GPU
        const bool amgxBuild = true;
        int backend = useGPU ? 1 : 0;
        ImGui::RadioButton("CPU (Eigen)", &backend, 0);
        ImGui::SameLine();
        ImGui::RadioButton("GPU (NVIDIA AMGX)", &backend, 1);
        useGPU = (backend == 1);
#else
        const bool amgxBuild = false;
        int backend = 0;
        ImGui::RadioButton("CPU (Eigen)", &backend, 0);
        ImGui::SameLine();
        ImGui::BeginDisabled();
        ImGui::RadioButton("GPU (NVIDIA AMGX)", &backend, 1);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("This build has no NVIDIA GPU support (make NVIDIA_GPU=1).");
#endif


        ImGui::Spacing();
        {
            // Warm start and the compatibility checks only mean something where the fluid region, and so the matrix,
            // stands still (not FLIP).
            const bool eulerian = MethodSupports(method, CAP_GRID) && MethodSupports(method, CAP_FIXED_REGION);
            if (!eulerian) ImGui::BeginDisabled();
            DrawSetting(PRESSURE_FIELDS, PRESSURE_FIELD_COUNT, &PRESSURE_SETTINGS, "PRESSURE_WARM_START");
            DrawSetting(PRESSURE_FIELDS, PRESSURE_FIELD_COUNT, &PRESSURE_SETTINGS, "NEEDS_COMPATIBILITY_CONDITION");
            if (!eulerian)
            {
                ImGui::EndDisabled();
                ImGui::TextDisabled("Fixed-matrix (Eulerian) solvers only: FLIP moves its fluid\n"
                                    "region every step, so the previous field no longer lines up.");
            }
            if (PRESSURE_SETTINGS.warmStart && eulerian && amgxBuild)
                ImGui::TextDisabled("AMGX convergence is forced to ABSOLUTE; see below.");
        }

        if (useGPU && amgxBuild)
        {
            ImGui::Spacing();
            ImGui::SeparatorText("AMGX");

            static const char *solverNames[] = {"PCG", "PBICGSTAB", "FGMRES", "CG", "AMG"};
            FormCombo("Krylov method", &amgxSolver, solverNames, IM_ARRAYSIZE(solverNames),
                      "PCG suits the symmetric pressure system. AMG runs multigrid on its own");
            FormInt("Max iterations", &amgxMaxIters);
            FormDouble("Tolerance", &amgxTolerance, nullptr, "%.3e",
                       "Sets how small the divergence after projection gets");

            static const char *normNames[] = {"L1", "L2", "LMAX"};
            FormCombo("Norm", &amgxNorm, normNames, IM_ARRAYSIZE(normNames));
            static const char *convNames[] = {"RELATIVE_INI_CORE", "ABSOLUTE", "RELATIVE_MAX_CORE"};
            // RELATIVE_INI_CORE stops at tol * ||r_initial||.
            const bool convForced = PRESSURE_SETTINGS.warmStart && Family() == MethodFamily::EULERIAN;
            if (convForced)
            {
                amgxConvergence = 1; // ABSOLUTE
                ImGui::BeginDisabled();
            }
            FormCombo("Convergence", &amgxConvergence, convNames, IM_ARRAYSIZE(convNames), nullptr, FormEm() * 22.0f);
            if (convForced)
            {
                ImGui::EndDisabled();
                ImGui::TextDisabled("Forced by warm start: a criterion measured against the\n"
                                    "initial residual cancels the benefit of a better guess.");
            }

            ImGui::Spacing();
            if (amgxSolver == 4)
            {
                ImGui::TextDisabled("AMG is the solver, so it has no separate preconditioner.");
            }
            else
            {
                ImGui::Checkbox("AMG preconditioner", &amgxUsePrecon);
            }

            if (amgxUsePrecon || amgxSolver == 4)
            {
                ImGui::Indent();
                static const char *algNames[] = {"CLASSICAL", "AGGREGATION"};
                FormCombo("Coarsening", &amgxPreconAlg, algNames, IM_ARRAYSIZE(algNames));
                static const char *cycleNames[] = {"V", "W", "F"};
                FormCombo("Cycle", &amgxCycle, cycleNames, IM_ARRAYSIZE(cycleNames));
                FormSliderInt("Pre-sweeps", &amgxPresweeps, 0, 4);
                FormSliderInt("Post-sweeps", &amgxPostsweeps, 0, 4);
                if (amgxSolver != 4)
                    FormSliderInt("AMG iterations", &amgxPreIters, 1, 4);
                ImGui::Unindent();
            }

            ImGui::Spacing();
            ImGui::Checkbox("Print solve stats", &amgxSolveStats);
            ImGui::SameLine();
            ImGui::Checkbox("Print grid stats", &amgxGridStats);
            ImGui::SameLine();
            ImGui::Checkbox("Timings", &amgxTimings);
        }
        else
        {
            ImGui::Spacing();
            ImGui::SeparatorText("Eigen");
            static const char *eigenSolverNames[] = {"BiCGSTAB", "Conjugate gradient", "CG + AMG (AMGCL)"};
            FormCombo("Krylov method", &eigenSolver, eigenSolverNames, IM_ARRAYSIZE(eigenSolverNames), nullptr,
                      FormEm() * 22.0f);
            static const char *eigenPreconNames[] = {"Diagonal (Jacobi)", "None", "Incomplete LU (ILUT)",
                                                     "Incomplete Cholesky"};
            if (eigenSolver == EIGEN_AMG)
                ImGui::TextDisabled("Preconditioner: smoothed-aggregation AMG, built once per matrix.");
            else
            {
                FormCombo("Preconditioner", &eigenPreconditioner, eigenPreconNames, IM_ARRAYSIZE(eigenPreconNames),
                          "Incomplete Cholesky needs a positive definite matrix", FormEm() * 22.0f);
            }
            FormInt("Max iterations", &eigenMaxIters);
            FormDouble("Tolerance", &eigenTolerance, nullptr, "%.3e", "Relative residual ||r|| / ||b||");
        }
        ImGui::Unindent();
    }

    // ---------------- export ----------------
    char exportHead[64];
    std::snprintf(exportHead, sizeof(exportHead), "%d. Export###export", section++);
    if (ImGui::CollapsingHeader(exportHead))
    {
        ImGui::Indent();
        FormLabel("Base path");
        ImGui::SetNextItemWidth(FormWideWidth());
        ImGui::InputText("##basePath", exportPath, IM_ARRAYSIZE(exportPath));
        FormDouble("Reynolds number", &reynoldsForNaming, nullptr, "%.0f",
                   "Directory naming only. The effective Re comes from the viscosity, length and velocity below");
        FormDouble("Characteristic length", &characteristicLength, "m", "%.4f");
        FormDouble("Mean velocity", &meanVelocity, "m/s", "%.4f");
        if (viscosity > 0.0)
            ImGui::Text("Effective Re = %.4g", meanVelocity * characteristicLength / viscosity);
        if (dimension == 3)
        {
            FormDouble("Second length", &characteristicLength2, "m", "%.4f",
                       "3D coefficients divide by L1 * L2, e.g. D * H or chord * span");
            ImGui::Text("S = %.6g m^2", characteristicLength * characteristicLength2);
        }

        ImGui::Spacing();
        ImGui::SeparatorText("What to write");
        ImGui::Checkbox("Enable export", &exports.enabled);
        if (!exports.enabled) ImGui::BeginDisabled();
        FormInt("Telemetry every", &exports.telemetryInterval, "steps");
        if (exports.telemetryInterval < 1) exports.telemetryInterval = 1;

        int frameMode = exports.gridEveryTime ? 1 : 0;
        FormCombo("Frames", &frameMode, "every N iterations\0every X seconds of simulated time\0", nullptr,
                  FormEm() * 34.0f);
        exports.gridEveryTime = (frameMode == 1);
        if (exports.gridEveryTime)
        {
            FormDouble("Frame interval", &exports.gridTimeInterval, "s of simulated time", "%.4f");
            if (exports.gridTimeInterval > 0.0)
                ImGui::TextDisabled("%.1f frames per second of simulated time.",
                                    1.0 / exports.gridTimeInterval);
        }
        else
        {
            FormInt("Frame interval", &exports.gridInterval, "iterations");
            if (exports.gridInterval < 1) exports.gridInterval = 1;
        }

        ImGui::Checkbox("Grid fields (VTK .vtr)", &exports.exportGridData);
        const bool particleRun = MethodSupports(method, CAP_PARTICLES);
        if (!particleRun) ImGui::BeginDisabled();
        ImGui::Checkbox("Particles (CSV)", &exports.exportParticleData);
        if (!particleRun) ImGui::EndDisabled();

        const bool canSurface = particleRun && dimension == 3;
        if (!canSurface) ImGui::BeginDisabled();
        ImGui::Checkbox("Liquid surface (PLY)", &exports.exportSurfaceMesh);
        if (!canSurface) ImGui::EndDisabled();
        if (!canSurface && ImGui::IsItemHovered())
            ImGui::SetTooltip("3D particle methods only");
        if (canSurface && exports.exportSurfaceMesh)
        {
            ImGui::Indent();
            ImGui::Checkbox("Obstacles as their own PLY", &exports.exportSurfaceSolids);
            FormDouble("Voxel pitch", &exports.surfaceVoxelScale, "particle spacings", "%.3f");
            FormDouble("Particle radius", &exports.surfaceRadiusScale, "particle spacings", "%.3f");
            FormDouble("Influence radius R", &exports.surfaceInfluenceScale, "particle spacings", "%.3f",
                       "Larger R gives a smoother surface. Raise the particle radius with it");
            FormInt("Field smoothing", &exports.surfaceSmoothing, "passes");
            if (exports.surfaceSmoothing < 0) exports.surfaceSmoothing = 0;
            ImGui::Unindent();
        }

        const bool canVolumes = MethodSupports(method, CAP_SCALARS) && !uiScalars.empty();
        if (!canVolumes) ImGui::BeginDisabled();
        ImGui::Checkbox("Scalar volumes (NumPy .npy)", &exports.exportVolumetric);
        if (!canVolumes) ImGui::EndDisabled();
        if (!canVolumes && ImGui::IsItemHovered())
            ImGui::SetTooltip("Writes the scalar fields as .npy for a renderer");
        ImGui::Spacing();
        ImGui::SeparatorText("Derived telemetry");
        ImGui::TextDisabled("Each costs a pass over the grid every step.");
        ImGui::Checkbox("Aerodynamics (Cd, Cl, Delta-p)", &exports.aerodynamics);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Drag and lift on the solid region");
        ImGui::Checkbox("Free surface (height, front)", &exports.dambreak);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Liquid height and front position over time");
        if (!exports.enabled) ImGui::EndDisabled();

        ImGui::Spacing();
        ImGui::SeparatorText("Checkpoint");
        ImGui::TextDisabled("One file, overwritten in place, so a long run can be resumed.");
        ImGui::Checkbox("Auto-save", &checkpoint.autoSave);
        if (!checkpoint.autoSave) ImGui::BeginDisabled();
        FormInt("Save every", &checkpoint.interval, "iterations");
        if (checkpoint.interval < 1) checkpoint.interval = 1;
        if (!checkpoint.autoSave) ImGui::EndDisabled();
        FormLabel("File");
        ImGui::SetNextItemWidth(FormWideWidth());
        ImGui::InputText("##checkpointFile", checkpoint.path, IM_ARRAYSIZE(checkpoint.path));

        ImGui::Unindent();
    }

    ImGui::PopItemWidth();
    ImGui::EndChild();

    // ---------------- problems and the commit row ----------------
    ImGui::Separator();
    std::vector<std::string> problems, notes;
    Validate(problems, notes);

    for (const std::string &p : problems)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, ToneColour(UiTone::Error));
        ImGui::TextWrapped("%s", p.c_str());
        ImGui::PopStyleColor();
    }
    for (const std::string &n : notes)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, ToneColour(UiTone::Warn));
        ImGui::TextWrapped("%s", n.c_str());
        ImGui::PopStyleColor();
    }
    if (problems.empty() && notes.empty())
        ImGui::TextDisabled("No problems found.");

    ImGui::Spacing();
    ImGui::TextDisabled("Current file: %s", configPath.c_str());
    if (!HasPreset())
        ImGui::TextColored(ToneColour(UiTone::Note),
                           "No preset selected. Name this setup in \"Save as\" at the top (or pick a preset)\n"
                           "before starting -- the run and its exports take that name.");
    const bool blocked = !problems.empty() || !HasPreset();
    if (blocked) ImGui::BeginDisabled();
    if (ImGui::Button("Start simulation", ImVec2(180, 0)))
    {
        // The scene is written inside the config, so it only has to PARSE here.
        std::string err;
        if (!ApplyScene())
        {
            statusMessage = "scene: " + sceneStatus;
            statusIsError = true;
        }
        else if (WriteFiles(err, configPath))
        {
            start = true;
        }
        else
        {
            statusMessage = err;
            statusIsError = true;
        }
    }
    if (blocked) ImGui::EndDisabled();

    ImGui::SameLine();
    if (ImGui::Button("Save without starting", ImVec2(180, 0)))
    {
        std::string err;
        bool ok = ApplyScene() && WriteFiles(err, configPath);
        statusIsError = !ok;
        statusMessage = ok ? ("Written to " + configPath + ".")
                           : (sceneStatusIsError ? "scene: " + sceneStatus : err);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reload from file", ImVec2(150, 0)))
    {
        LoadFrom(configPath);
        statusMessage = "Reloaded " + configPath + ".";
        statusIsError = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Quit", ImVec2(80, 0)))
        glfwSetWindowShouldClose(glfwGetCurrentContext(), GLFW_TRUE);

    if (!statusMessage.empty())
    {
        ImGui::SameLine();
        ImGui::TextColored(statusIsError ? ToneColour(UiTone::Error) : ToneColour(UiTone::Good),
                           "%s", statusMessage.c_str());
    }

    ImGui::End();

    RenderScenePreview();
    // Meshless: there is no mesh to preview. The scene preview stays -- geometry is
    // very much SPH's business, it is only the grid that is not.
    if (UsesGrid())
        RenderMeshPreview();
    return start;
}


// ======== The viewer: field, particles, telemetry. It names no method: what it shows follows the
// active method's capabilities. ========

// 3D slice planes: 0 XY (fixed k), 1 XZ (fixed i, top-down), 2 ZY (fixed j). Y, the gravity axis, stays
// vertical wherever it appears; everything is sampled at cell centres and plotted in metres.
struct SlicePlaneInfo
{
    const char *hLabel;
    const char *vLabel;
    const char *fixedLabel;
    int hCells, vCells, fixedCells;
};

SlicePlaneInfo GetSlicePlaneInfo(int plane, int Nx, int Ny, int Nz)
{
    switch (plane)
    {
    case 1:  return {"X (m)", "Z (m)", "Y", Nx, Nz, Ny};
    case 2:  return {"Z (m)", "Y (m)", "X", Nz, Ny, Nx};
    default: return {"X (m)", "Y (m)", "Z", Nx, Ny, Nz};
    }
}

// Maps a (horizontal, vertical) cell in the slice back to the grid triple (i=y, j=x, k=z).
void SliceToGrid(int plane, int h, int v, int fixedIdx, int &i, int &j, int &k)
{
    switch (plane)
    {
    case 1: j = h; k = v; i = fixedIdx; break; // XZ, fixed y
    case 2: k = h; i = v; j = fixedIdx; break; // ZY, fixed x
    default: j = h; i = v; k = fixedIdx; break; // XY, fixed z
    }
}

// Honestly, this code is basically a monolith, its mainly ImGUI and ImPlots code to plot everything of the grid and simulation telemetry.

// Telemetry plots are read, not explored: they auto-fit every frame, and with inputs on the mouse wheel zoomed
// them instead of scrolling the pane.
constexpr ImPlotFlags TELEMETRY_PLOT_FLAGS = ImPlotFlags_NoInputs;

// A colour bar whose tick labels show enough digits to tell the ticks apart: fixed decimals for ordinary
// magnitudes, scientific outside 1e-3..1e5. Sized to its widest label so nothing is clipped.
struct ScaleFormat { char fmt[16]; float width; };
// The width is STICKY per bar label: it only grows. A width that followed the labels resized the plot
// beside it whenever the range gained a digit, which re-framed the view mid-run.
ScaleFormat MakeScaleFormat(double lo, double hi, const char *label)
{
    const bool labelled = label[0] != '#';
    ScaleFormat f;
    const double mag = (std::max)(std::fabs(lo), std::fabs(hi));
    const double span = std::fabs(hi - lo);
    if (mag > 0.0 && (mag < 1e-3 || mag >= 1e5))
        std::snprintf(f.fmt, sizeof(f.fmt), "%%.3e");
    else
    {
        // A tick every ~span/8: enough decimals for that step, plus one.
        int dec = (span > 0.0) ? (int)std::ceil(-std::log10(span / 8.0)) + 1 : 3;
        dec = (std::max)(0, (std::min)(dec, 8));
        std::snprintf(f.fmt, sizeof(f.fmt), "%%.%df", dec);
    }
    char a[64], b[64];
    std::snprintf(a, sizeof(a), f.fmt, lo);
    std::snprintf(b, sizeof(b), f.fmt, hi);
    f.width = (std::max)(ImGui::CalcTextSize(a).x, ImGui::CalcTextSize(b).x) + 40.0f
            + (labelled ? ImGui::GetTextLineHeightWithSpacing() : 0.0f);
    static std::map<std::string, float> widest;
    float &w = widest[label];
    w = (std::max)(w, f.width);
    f.width = w;
    return f;
}

void ColourBar(const char *label, double lo, double hi, float height)
{
    const ScaleFormat f = MakeScaleFormat(lo, hi, label);
    ImPlot::ColormapScale(label, lo, hi, ImVec2(f.width, height), f.fmt);
}

class SimulationViewer
{
private:
    MAC *grid3D;
    MAC *grid2D;

    // UI state
    int selectedComponent = 7; // 0=velocity, 1=u, 2=v, 3=w (3D only), 4=p, 5=solid, 6=divergence, 7=magnitude
    int slicePlane = 0;        // 0=XY, 1=XZ (3D only), 2=YZ (3D only)
    int sliceIndex = 0;
    bool autoScale = true;
    float minValue = 0.0f;
    float maxValue = 1.0f;

    float dh = 0.0f;


    // The field view's framing. It is set when the extent, the plane or the plot size changes, or on Fit view;
    // between those the user's wheel zoom and drag pan are kept.
    bool fieldViewReset = true;
    double fieldViewKey[7] = {0, 0, 0, 0, 0, 0, 0};
    // The plot's DATA area from the previous frame (the frame less its tick labels and padding). The fit is sized
    // on it: sized on the whole frame, the equal-axes plot shrank the margin away on the tight side.
    float plotAreaW = 0.0f, plotAreaH = 0.0f;
    float fitAreaW = -1.0f, fitAreaH = -1.0f;   // the data area the current fit was sized on
    int fitSettleFrames = 0;                    // frames after a re-fit during which a new data area refines it
    void RecordPlotArea();
    void SetupFieldView(double ox, double oy, double w, double h, float plotW, float plotH, ImVec2 pane);

    std::vector<float> quiverX;
    std::vector<float> quiverY;
    std::vector<float> quiverU;
    std::vector<float> quiverV;

    // --- particle view --- a random subset, stored sorted by colour bucket so each bucket is one
    // contiguous PlotScatter; rebuilt only when a step, the count or a control changes.
    static constexpr int PARTICLE_BUCKETS = 32;
    std::vector<float> p3x, p3y, p3z;
    int bucketStart[PARTICLE_BUCKETS + 1] = {};  // bucket b is [bucketStart[b], bucketStart[b + 1])
    std::vector<int> sampleIdx, sampleBucket;   // scratch for a resample
    std::vector<float> sampleVal;
    int sampledIteration = -1, sampledTotal = -1, sampledBudget = -1;
    unsigned int sampledSeed = 0;
    bool sampledFromGrid = false, sampledBySpeed = false, sampledAuto = false;
    float sampledMin = 0.0f, sampledMax = 0.0f;
    int particleBudget = 20000;    // how many particles to draw at most
    unsigned int particleSeed = 1; // re-roll to get a different subset
    bool particleColorBySpeed = true;
    // Colour from the grid velocity or the particle's own. FLIP particles carry what P2G maps to zero, which
    // the grid does not have: the grid shows the fluid, the particle shows what FLIP carries.
    bool particleColorFromGrid = true;   // the fluid velocity by default; SPH has no grid and ignores it
    float particleSize = 2.5f;          // 3D marker radius in pixels
    float particlePointScale = 1.5f;    // 2D marker diameter in particle spacings; above 1 so the lattice does not show through
    float particleMin = 0.0f, particleMax = 1.0f;
    bool particleAutoScale = true;

    // --- solid geometry, drawn under the particles: surface cells only, the domain ring off by default
    // (a shell around the domain hides the scene).
    std::vector<float> solidX, solidY, solidZ;
    const MAC *solidCellsGrid = nullptr;   // the grid and wall setting solidX/Y/Z were collected for
    int solidCellsWalls = -1;
    bool showSolids = true;
    bool solidsIncludeWalls = false;
    float solidSize = 3.0f;
    void CollectSolidCells();

    // Without a grid (SPH) the solids come from the scene's analytic surface, sampled once on a lattice
    // at the particle spacing -- what SPH actually collides with.
    void CollectAnalyticSolids();
    bool analyticSolidsBuilt = false;
    double analyticPitch = 0.01;   // the lattice CollectAnalyticSolids sampled on


    float baseSize = 12.0f;
    ImPlotQuiverFlags quiverFlags = ImPlotQuiverFlags_Colored | ImPlotQuiverFlags_Normalize;
    ImPlotColormap colormap = ImPlotColormap_Jet;
    ImPlot3DColormap colormap3D = ImPlotColormap_Jet;

    std::vector<double> DataK;

    std::vector<double> heatmapData;
    // The heatmap drawn as one texture: coloured on the CPU through the active colormap and uploaded only when the
    // data, the range or the colormap change (PlotHeatmap emitted a rectangle per cell per frame).
    unsigned heatmapVersion = 0;
    unsigned fieldTex = 0;
    unsigned fieldTexVersion = ~0u;
    float fieldTexMin = 0.0f, fieldTexMax = 0.0f;
    int fieldTexColormap = -1;
    std::vector<unsigned char> fieldRGBA;
    ImTextureID FieldTexture();
    // Draw the cell faces over the field, so a stretched grid shows where its cells are.
    bool showMeshOverlay = false;
    // The plot's size in screen pixels, from the previous frame: it bounds the resample resolution.
    float lastPlotPx = 600.0f;
    // What the cached slice was built from; it is rebuilt only when one of these changes.
    int sliceIT = -1, sliceComponent = -1, slicePlaneCached = -1, sliceIndexCached = -1;
    int sliceNx = -1, sliceNy = -1;
    int frameIT = 0;
    int meshOverlayAlpha = 70;
    int heatmapRows = 0;
    int heatmapCols = 0;

    const char *componentNames[8] = {"Velocity Field", "U Velocity", "V Velocity", "W Velocity", "Pressure", "Solid Mask", "Divergence", "Magnitude"};
    // The label of the component on screen: one of the fixed eight, or a registered scalar's name. Indexing
    // componentNames with a scalar's id (SCALAR_COMPONENT_BASE + handle) read past the array and crashed.
    const char *ComponentLabel() const
    {
        if (selectedComponent >= 0 && selectedComponent < SCALAR_COMPONENT_BASE) return componentNames[selectedComponent];
        const MAC *g = (DIMENSION == 2) ? grid2D : grid3D;
        const int h = selectedComponent - SCALAR_COMPONENT_BASE;
        if (g != nullptr && h >= 0 && h < g->ScalarCount()) return g->ScalarName(h).c_str();
        return "Scalar";
    }

    void ExtractSliceData();
    void ResampleHeatmapToPhysical();

    // The grid on view and the face tables of the two axes on screen (x/y in 2D, the slice plane's pair in 3D).
    // The resample, the extent, the mesh overlay and the quiver all ask this.
    const MAC *ViewGrid() const;
    bool ViewFaceTables(const std::vector<double> *&hFaces, const std::vector<double> *&vFaces) const;
    void ViewExtent(double &ox, double &oy, double &w, double &h) const;
    void DrawMeshOverlay();
    // Streamlines over the 2D field view, recomputed only when the field or the settings change.
    bool showStreamlines = false;
    bool streamlinesOnly = false;       // draw them without the field underneath
    bool streamArrows = true;
    int streamColour = 0;               // 0 speed, 1 white, 2 black
    float streamSpacingPx = 24.0f;      // between lines, in screen pixels: the count follows the plot, not the grid
    float streamWidth = 1.5f;
    std::vector<Streamline> streamlines;
    std::vector<ImVec2> streamPixels;
    int streamIT = -1;
    double streamKey[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // view window, plot size, spacing, grid size: re-trace when it moves
    double streamTracedAt = -1.0;       // wall time of the last trace; a new iteration alone re-traces at most every 0.25 s
    double streamMaxSpeed = 0.0;
    void UpdateStreamlines();
    void DrawStreamlines();

    // Line profile: a field sampled between two points of the plane on screen, plotted against the distance from
    // the first. The points are typed in or dragged on the field view.
    bool showProfile = false;
    bool profilePlaced = false;
    double profileA[3] = {0.0, 0.0, 0.0}, profileB[3] = {0.0, 0.0, 0.0};   // physical x, y, z
    int profileQuantity = 0;            // index into the names ProfileQuantities builds
    int profileSamples = 200;
    std::vector<double> profileS, profileV;
    std::vector<double> profileC;      // the physical coordinate along the line's main axis, what the plot is drawn against
    bool profileVertical = false;      // the line is closer to vertical on screen: the coordinate goes on the plot's y axis
    int profileAxis = 0;               // that axis, 0 x, 1 y, 2 z
    std::string profileStatus;
    void ProfileQuantities(std::vector<std::string> &names) const;
    bool ProfilePoint(double a, double b, double &x, double &y, double &z) const;
    void PlaneAxes(int &h, int &v, int &f) const;
    void DomainBounds(int a, double &lo, double &hi) const;
    void PlaceProfile(bool horizontal);
    double ProfileSample(int q, double x, double y, double z) const;
    void UpdateProfile();
    void ProfileControls();
    void DrawProfileLine();
    void SaveProfile();

    // Probe history for the Probes tab, kept by the viewer: one sample per completed step it sees.
    std::vector<double> probeTime;
    std::vector<std::vector<double>> probeValues;
    unsigned int probeIT = 0;
    int probeQuantity = 0;
    void RecordProbes();
    void RenderProbesTab();
    void DrawAnalysisPanel();

    // The point under the mouse on the field plot, in the plane's coordinates, from the previous frame.
    bool hoverValid = false;
    double hoverA = 0.0, hoverB = 0.0;
    void TrackHover();
    void HoverReadout();

    // 3D streamlines tab: seeded from a plane, a line or a sphere, drawn as depth-sorted segments grouped by speed.
    Streamline3DSettings stream3D;
    bool stream3DInitialised = false;
    bool stream3DDirty = true;
    bool stream3DShowSolids = true;
    float stream3DStepCells = 0.5f;
    float stream3DWidth = 1.5f;
    float stream3DSolidSize = 3.0f;
    int stream3DBudget = 40000;
    int stream3DIT = -1;
    double stream3DMaxSpeed = 0.0;
    size_t stream3DPoints = 0, stream3DSegments = 0;
    std::vector<Streamline3D> streamlines3D;
    std::vector<std::vector<float>> s3SegX, s3SegY, s3SegZ;   // per colour bin, endpoint pairs
    std::vector<float> s3SolidX, s3SolidY, s3SolidZ;
    int s3SolidNx = -1, s3SolidNy = -1, s3SolidNz = -1;
    void RenderStreamlines3DTab();
    void StreamlineControls3D();   // in the left column while the Streamlines 3D tab is open
    // A 3D plot fills what is left of the pane beside a colour bar of width barWidth (0 for none).
    ImVec2 Plot3DSize(float barWidth) const;
    // The 3D views share one zoom: the mouse wheel over a plot scales the box (ImPlot3D's view scale, since the
    // axis limits are pinned to the domain), and Fit view restores the scale and the default rotation.
    float view3DZoom = 1.0f, view3DSavedScale = 1.0f;
    bool view3DFit = false;
    ImVec2 view3DPos, view3DSize;
    bool Begin3DView(const char *id, ImVec2 size);
    void End3DView();
    void Fit3DViewButton();
    void DefaultStreamlineSeeds3D();
    void UpdateStreamlines3D();
    void CollectSolidSurface3D();
    void ExtractQuiverData();
    void SampleParticles();
    float ParticleColourValue(const Particle &q) const;
    bool ColourFromGrid() const;
    void ParticleControls(bool column);
    void RenderParticles2D();
    void RenderParticles3D();
    // 2D solids as filled rectangles (x0, y0, x1, y1), rebuilt when the wall setting changes.
    struct SolidRect { float x0, y0, x1, y1; };
    std::vector<SolidRect> solidRects;
    int solidRectKey = -1;
    void CollectSolidRects2D();
    void ComputeMinMax();

    int GetNx() const { return (DIMENSION == 2) ? grid2D->Nx : grid3D->Nx; }
    int GetNy() const { return (DIMENSION == 2) ? grid2D->Ny : grid3D->Ny; }
    int GetNz() const { return (DIMENSION == 2) ? 1 : grid3D->Nz; }

    // Component ids 0..7 are the fixed fields; a registered scalar is this plus its handle.
    static const int SCALAR_COMPONENT_BASE = 8;

    void FieldControls();
    void FieldPlot();
    void RenderSurfaceTab();

    // ---- the surface preview: draw buffers for MeshExport::LAST_SURFACE, rebuilt only when its
    // `generation` or the budget changes ----
    unsigned surfaceDrawnGeneration = 0;  // 0 = nothing built yet
    int surfaceDrawnBudget = -1;
    std::vector<ImPlot3DPoint> surfaceVtx;   // in PLOT order (x, z, y), see the note below
    std::vector<unsigned> surfaceIdx;
    int surfaceBudget = 40000;               // triangles actually drawn
    bool surfaceWireframe = false;
    bool surfaceShowDomain = true;

    void RenderControlUI();
    void RenderExportUI();
    void RenderSolverTelemetryTab();
    void RenderFlowIntegralsTab();
    void RenderAerodynamicsUI();
    void RenderParticleView(bool controlsInline);
    // The view tab drawn last frame: the left column is drawn before the tab bar, so it follows the previous
    // frame's choice (the particle controls when Particles is open, the field controls otherwise).
    enum ViewTab { VIEW_FIELD, VIEW_STREAMLINES3D, VIEW_PARTICLES, VIEW_SURFACE };
    int viewTab = VIEW_FIELD;
    void RenderDambreakUI();

public:
    SimulationViewer(MAC *gridPtr);
    ~SimulationViewer();

    // The three panes of the main layout; each draws into the window the layout opened for it.
    void DrawViewPane();
    void DrawControlPane();
    void DrawTelemetryPane();
    void UpdateGrid(MAC *newGrid);
    double drawTime = 0.0;   // seconds the last frame's panes took, shown in the control pane
    // The Field tab's left column, as a fraction of the view pane's width, and the line / probe panel at its
    // foot, as a fraction of its height. Set by the layout from the UI settings every frame.
    float controlsWidth = 0.30f;
    float analysisHeight = 0.45f;
};

SimulationViewer::SimulationViewer(MAC *gridPtr) : grid3D(gridPtr), grid2D(nullptr)
{
    if (gridPtr->is2D)
    {

        sliceIndex = 0; // No slicing needed for 2D
        dh = gridPtr->dh;
        particleBudget = 200000;   // a 2D cloud is cheap to draw; 3D keeps the smaller default
    }
    else
    {

        sliceIndex = grid3D->Nz / 2; // the middle of z: the default plane is XY, which fixes z

        dh = grid3D->dh;
    }
}

SimulationViewer::~SimulationViewer() {}

void SimulationViewer::UpdateGrid(MAC *newGrid)
{

    if (newGrid->is2D)
    {
        grid2D = newGrid;
        grid3D = nullptr;
    }
    else
    {

        grid3D = newGrid;
        grid2D = nullptr;
    }
}

void SimulationViewer::ComputeMinMax()
{

    // --- Velocity Field (Quiver only) ---
    if (selectedComponent == 0)
    {
        minValue = 0.0f;
        maxValue = 0.0f;

        // Both 2D and a 3D slice fill quiverU/quiverV with the two in-plane components.
        for (size_t i = 0; i < quiverU.size(); i++)
        {
            float mag = std::sqrt(quiverU[i] * quiverU[i] + quiverV[i] * quiverV[i]);
            maxValue = (std::max)(maxValue, mag);
        }

        if (maxValue < 1e-12f)
            maxValue = 1e-12f;
        return;
    }

    // --- Solid mask ---
    if (selectedComponent == 5)
    {
        minValue = 0.0f;
        maxValue = 3.0f;
        return;
    }

    // --- Heatmap-based components (Divergence, Magnitude, Pressure, etc.) ---
    if (heatmapData.empty())
        return;

    minValue = (float)heatmapData[0];
    maxValue = (float)heatmapData[0];

    for (double v : heatmapData)
    {
        minValue = (std::min)(minValue, (float)v);
        maxValue = (std::max)(maxValue, (float)v);
    }

    // A non-finite value would reach the colormap index and abort inside ImPlot: clamp to a usable range,
    // so a diverging run shows as saturated colour.
    if (!std::isfinite(minValue) || !std::isfinite(maxValue))
    {
        minValue = 0.0f;
        maxValue = 1.0f;
        return;
    }

    // A uniform field (a free stream) has no range, and 1e-12 added to a value near 1 is lost in a float. Open the range
    // by 1% of the value on each side, so the field draws as the middle of the colour map.
    const float mid = 0.5f * (minValue + maxValue);
    if (maxValue - minValue <= 1e-6f * (std::max)(std::fabs(mid), 1e-30f))
    {
        const float eps = (std::fabs(mid) > 0.0f) ? 0.01f * std::fabs(mid) : 1e-6f;
        minValue = mid - eps;
        maxValue = mid + eps;
    }
}

void SimulationViewer::ExtractQuiverData()
{
    quiverX.clear();
    quiverY.clear();
    quiverU.clear();
    quiverV.clear();

    if (DIMENSION == 2)
    {
        if (!grid2D)
            return;

        // 2D: Only XY plane, show u and v velocities
        try
        {
            for (int j = 0; j < grid2D->Ny; j++)
            {
                for (int i = 0; i < grid2D->Nx; i++)
                {
                    quiverX.push_back(i + 0.5f);
                    quiverY.push_back(j + 0.5f);

                    // Interpolate u velocity to cell center
                    float u = 0.5f * (grid2D->GetU(j, i) + grid2D->GetU(j, i + 1));
                    // Interpolate v velocity to cell center
                    float v = 0.5f * (grid2D->GetV(j, i) + grid2D->GetV(j + 1, i));

                    quiverU.push_back(u);
                    quiverV.push_back(v);
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cout << "[EXCEPTION] Error in ExtractQuiverData (2D): " << e.what() << std::endl;
            return;
        }
    }
    else
    {
        if (!grid3D)
            return;

        const int Nx = grid3D->Nx, Ny = grid3D->Ny, Nz = grid3D->Nz;
        SlicePlaneInfo pi = GetSlicePlaneInfo(slicePlane, Nx, Ny, Nz);
        sliceIndex = (std::max)(0, (std::min)(sliceIndex, pi.fixedCells - 1));

        try
        {
            for (int v = 0; v < pi.vCells; v++)
            {
                for (int h = 0; h < pi.hCells; h++)
                {
                    int i, j, k;
                    SliceToGrid(slicePlane, h, v, sliceIndex, i, j, k);

                    double u = 0.5 * (grid3D->GetU(i, j, k) + grid3D->GetU(i, j + 1, k));
                    double vv = 0.5 * (grid3D->GetV(i, j, k) + grid3D->GetV(i + 1, j, k));
                    double w = 0.5 * (grid3D->GetW(i, j, k) + grid3D->GetW(i, j, k + 1));

                    // Whichever two components lie in the plane, mapped onto the plot's
                    // horizontal and vertical - same convention as the heatmap.
                    double hu, hv;
                    switch (slicePlane)
                    {
                    case 1:  hu = u;  hv = w;  break; // XZ
                    case 2:  hu = w;  hv = vv; break; // ZY
                    default: hu = u;  hv = vv; break; // XY
                    }

                    // Physical cell centres off the face tables, matching the heatmap extents.
                    const std::vector<double> *hf = nullptr, *vf = nullptr;
                    if (ViewFaceTables(hf, vf))
                    {
                        quiverX.push_back((float)(0.5 * ((*hf)[h] + (*hf)[h + 1])));
                        quiverY.push_back((float)(0.5 * ((*vf)[v] + (*vf)[v + 1])));
                    }
                    else
                    {
                        quiverX.push_back((h + 0.5f) * dh);
                        quiverY.push_back((v + 0.5f) * dh);
                    }
                    quiverU.push_back((float)hu);
                    quiverV.push_back((float)hv);
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cout << "[EXCEPTION] Error in ExtractQuiverData (3D): " << e.what() << std::endl;
            return;
        }
    }

    if (autoScale)
    {
        ComputeMinMax();
    }
}

const MAC *SimulationViewer::ViewGrid() const
{
    return (DIMENSION == 2) ? grid2D : grid3D;
}

bool SimulationViewer::ViewFaceTables(const std::vector<double> *&hFaces,
                                    const std::vector<double> *&vFaces) const
{
    const MAC *g = ViewGrid();
    hFaces = vFaces = nullptr;
    if (g == nullptr) return false;
    if (DIMENSION == 2) { hFaces = &g->Faces(MAC::X); vFaces = &g->Faces(MAC::Y); }
    else
    {
        // Must agree with GetSlicePlaneInfo / SliceToGrid on which axis is across and up.
        switch (slicePlane)
        {
        case 1:  hFaces = &g->Faces(MAC::X); vFaces = &g->Faces(MAC::Z); break; // XZ, fixed y
        case 2:  hFaces = &g->Faces(MAC::Z); vFaces = &g->Faces(MAC::Y); break; // ZY, fixed x
        default: hFaces = &g->Faces(MAC::X); vFaces = &g->Faces(MAC::Y); break; // XY, fixed z
        }
    }
    return hFaces->size() >= 2 && vFaces->size() >= 2;
}

// The span the view covers, from the face tables -- never `cells * dh`, which is wrong twice
// on a stretched grid (dh is the SMALLEST cell there) and ignores the domain origin besides.
void SimulationViewer::ViewExtent(double &ox, double &oy, double &w, double &h) const
{
    const std::vector<double> *hf = nullptr, *vf = nullptr;
    if (!ViewFaceTables(hf, vf))
    {
        ox = SIMULATION.domain.x0; oy = SIMULATION.domain.y0;
        w = SIMULATION.domain.xf - ox; h = SIMULATION.domain.yf - oy;
        return;
    }
    ox = hf->front(); oy = vf->front();
    w = hf->back() - ox;  h = vf->back() - oy;
}

void SimulationViewer::DefaultStreamlineSeeds3D()
{
    const Domain &d = SIMULATION.domain;
    const double lx = d.xf - d.x0, ly = d.yf - d.y0, lz = d.zf - d.z0;
    stream3D.source = STREAM_SEED_PLANE;
    stream3D.planeAxis = 0;
    stream3D.planePosition = d.x0 + 0.05 * lx;
    stream3D.count = 12;
    stream3D.p0[0] = d.x0 + 0.1 * lx; stream3D.p0[1] = d.y0 + 0.05 * ly; stream3D.p0[2] = d.z0 + 0.5 * lz;
    stream3D.p1[0] = d.x0 + 0.1 * lx; stream3D.p1[1] = d.yf - 0.05 * ly; stream3D.p1[2] = d.z0 + 0.5 * lz;
    stream3D.centre[0] = d.x0 + 0.5 * lx; stream3D.centre[1] = d.y0 + 0.5 * ly; stream3D.centre[2] = d.z0 + 0.5 * lz;
    stream3D.radius = 0.1 * (std::min)(lx, (std::min)(ly, lz));
    stream3D.maxPoints = 4000;
    stream3DInitialised = true;
    stream3DDirty = true;
}

// Traced when the field or a setting changes; the segments are regrouped into colour bins then, not per frame.
void SimulationViewer::UpdateStreamlines3D()
{
    if (grid3D == nullptr) return;
    if (!stream3DDirty && frameIT == stream3DIT) return;
    stream3D.step = (double)stream3DStepCells * grid3D->dh;
    ComputeStreamlines3D(grid3D, stream3D, streamlines3D);

    stream3DMaxSpeed = 0.0;
    stream3DPoints = 0;
    for (const Streamline3D &l : streamlines3D)
    {
        stream3DPoints += l.x.size();
        for (double v : l.speed) stream3DMaxSpeed = (std::max)(stream3DMaxSpeed, v);
    }
    // ImPlot3D depth-sorts every segment every frame, so the count is capped by thinning each line evenly.
    const size_t total = stream3DPoints > streamlines3D.size() ? stream3DPoints - streamlines3D.size() : 0;
    const size_t stride = (stream3DBudget > 0 && total > (size_t)stream3DBudget)
                              ? (total + stream3DBudget - 1) / stream3DBudget : 1;

    const int bins = 24;
    s3SegX.assign(bins, std::vector<float>());
    s3SegY.assign(bins, std::vector<float>());
    s3SegZ.assign(bins, std::vector<float>());
    stream3DSegments = 0;
    const double inv = stream3DMaxSpeed > 0.0 ? 1.0 / stream3DMaxSpeed : 0.0;
    for (const Streamline3D &l : streamlines3D)
    {
        const size_t n = l.x.size();
        size_t a = 0;
        while (a + 1 < n)
        {
            const size_t b = (std::min)(n - 1, a + stride);
            const int bin = (int)(std::min)((double)(bins - 1), 0.5 * (l.speed[a] + l.speed[b]) * inv * bins);
            s3SegX[bin].push_back((float)l.x[a]); s3SegX[bin].push_back((float)l.x[b]);
            s3SegY[bin].push_back((float)l.y[a]); s3SegY[bin].push_back((float)l.y[b]);
            s3SegZ[bin].push_back((float)l.z[a]); s3SegZ[bin].push_back((float)l.z[b]);
            stream3DSegments++;
            a = b;
        }
    }
    stream3DIT = frameIT;
    stream3DDirty = false;
}

// Surface solid cells at their table centres, walls left out; rebuilt when the grid size changes.
void SimulationViewer::CollectSolidSurface3D()
{
    if (grid3D == nullptr) return;
    if (grid3D->Nx == s3SolidNx && grid3D->Ny == s3SolidNy && grid3D->Nz == s3SolidNz) return;
    s3SolidX.clear(); s3SolidY.clear(); s3SolidZ.clear();
    const int Nx = grid3D->Nx, Ny = grid3D->Ny, Nz = grid3D->Nz;
    for (int k = 1; k < Nz - 1; k++)
        for (int i = 1; i < Ny - 1; i++)
            for (int j = 1; j < Nx - 1; j++)
            {
                if (grid3D->GetSolid(i, j, k) != SOLID_CELL) continue;
                const bool exposed = grid3D->GetSolid(i + 1, j, k) != SOLID_CELL || grid3D->GetSolid(i - 1, j, k) != SOLID_CELL ||
                                     grid3D->GetSolid(i, j + 1, k) != SOLID_CELL || grid3D->GetSolid(i, j - 1, k) != SOLID_CELL ||
                                     grid3D->GetSolid(i, j, k + 1) != SOLID_CELL || grid3D->GetSolid(i, j, k - 1) != SOLID_CELL;
                if (!exposed) continue;
                s3SolidX.push_back((float)grid3D->CentrePos(MAC::X, j));
                s3SolidY.push_back((float)grid3D->CentrePos(MAC::Y, i));
                s3SolidZ.push_back((float)grid3D->CentrePos(MAC::Z, k));
            }
    s3SolidNx = Nx; s3SolidNy = Ny; s3SolidNz = Nz;
}

ImVec2 SimulationViewer::Plot3DSize(float barWidth) const
{
    const ImVec2 area = ImGui::GetContentRegionAvail();
    const float gap = barWidth > 0.0f ? barWidth + 2.0f * ImGui::GetStyle().ItemSpacing.x : 0.0f;
    return ImVec2((std::max)(300.0f, area.x - gap), (std::max)(300.0f, area.y - 4.0f));
}

bool SimulationViewer::Begin3DView(const char *id, ImVec2 size)
{
    view3DPos = ImGui::GetCursorScreenPos();
    view3DSize = size;
    float &scale = ImPlot3D::GetStyle().ViewScaleFactor;
    view3DSavedScale = scale;
    scale = view3DZoom;
    if (ImPlot3D::BeginPlot(id, size, ImPlot3DFlags_NoZoom))
    {
        // A setup call: it must come before any item locks the setup.
        if (view3DFit)
            ImPlot3D::SetupBoxRotation(ImPlot3DQuat(-0.513269, -0.212596, -0.318184, 0.76819), true, ImPlot3DCond_Always);
        view3DFit = false;
        return true;
    }
    scale = view3DSavedScale;
    return false;
}

// Call after the plot's items; ends the plot.
void SimulationViewer::End3DView()
{
    ImPlot3D::EndPlot();
    ImPlot3D::GetStyle().ViewScaleFactor = view3DSavedScale;
    ImGuiIO &io = ImGui::GetIO();
    const ImVec2 hi(view3DPos.x + view3DSize.x, view3DPos.y + view3DSize.y);
    if (io.MouseWheel != 0.0f && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
        ImGui::IsMouseHoveringRect(view3DPos, hi))
    {
        view3DZoom = (std::min)(12.0f, (std::max)(0.3f, view3DZoom * std::pow(1.15f, io.MouseWheel)));
    }
}

void SimulationViewer::Fit3DViewButton()
{
    if (ImGui::SmallButton("Fit view")) { view3DZoom = 1.0f; view3DFit = true; }
    ImGui::SameLine();
    ImGui::TextDisabled("wheel zooms (%.0f%%)", 100.0 * view3DZoom);
}

// The 3D streamline settings, as labelled rows in the left column.
void SimulationViewer::StreamlineControls3D()
{
    if (grid3D == nullptr) return;
    if (!stream3DInitialised) DefaultStreamlineSeeds3D();
    const Domain &d = SIMULATION.domain;
    const float full = (std::max)(110.0f, ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("Points/line ").x);
    bool changed = false;

    ImGui::SeparatorText("Seeds");
    static const char *sources[] = {"Plane", "Line", "Sphere"};
    ImGui::SetNextItemWidth(full);
    changed |= ImGui::Combo("Seed from", &stream3D.source, sources, 3);
    if (stream3D.source == STREAM_SEED_PLANE)
    {
        static const char *normals[] = {"X (normal)", "Y (normal)", "Z (normal)"};
        ImGui::SetNextItemWidth(full);
        if (ImGui::Combo("Plane", &stream3D.planeAxis, normals, 3))
        {
            const double lo[3] = {d.x0, d.y0, d.z0}, hi[3] = {d.xf, d.yf, d.zf};
            const int a = stream3D.planeAxis;
            stream3D.planePosition = lo[a] + 0.05 * (hi[a] - lo[a]);
            changed = true;
        }
        const double lo[3] = {d.x0, d.y0, d.z0}, hi[3] = {d.xf, d.yf, d.zf};
        const int a = stream3D.planeAxis;
        ImGui::SetNextItemWidth(full);
        changed |= ImGui::SliderScalar("Position", ImGuiDataType_Double, &stream3D.planePosition, &lo[a], &hi[a], "%.3f m");
        ImGui::SetNextItemWidth(full);
        changed |= ImGui::SliderInt("Per side", &stream3D.count, 2, 40);
    }
    else if (stream3D.source == STREAM_SEED_LINE)
    {
        ImGui::SetNextItemWidth(full);
        changed |= ImGui::InputScalarN("From (m)", ImGuiDataType_Double, stream3D.p0, 3, nullptr, nullptr, "%.3f");
        ImGui::SetNextItemWidth(full);
        changed |= ImGui::InputScalarN("To (m)", ImGuiDataType_Double, stream3D.p1, 3, nullptr, nullptr, "%.3f");
        ImGui::SetNextItemWidth(full);
        changed |= ImGui::SliderInt("Seeds", &stream3D.count, 2, 400);
    }
    else
    {
        ImGui::SetNextItemWidth(full);
        changed |= ImGui::InputScalarN("Centre (m)", ImGuiDataType_Double, stream3D.centre, 3, nullptr, nullptr, "%.3f");
        ImGui::SetNextItemWidth(full);
        changed |= ImGui::InputDouble("Radius (m)", &stream3D.radius, 0.0, 0.0, "%.4f");
        ImGui::SetNextItemWidth(full);
        changed |= ImGui::SliderInt("Seeds", &stream3D.count, 2, 800);
    }
    if (ImGui::SmallButton("Reset seeds")) DefaultStreamlineSeeds3D();

    ImGui::SeparatorText("Tracing");
    ImGui::SetNextItemWidth(full);
    changed |= ImGui::SliderFloat("Step", &stream3DStepCells, 0.1f, 2.0f, "%.2f cells");
    ImGui::SetNextItemWidth(full);
    changed |= ImGui::SliderInt("Points/line", &stream3D.maxPoints, 200, 20000, "%d", ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("The most points a line is traced to");

    ImGui::SeparatorText("Drawing");
    ImGui::SetNextItemWidth(full);
    ImGui::SliderInt("Segments", &stream3DBudget, 2000, 200000, "%d", ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("ImPlot3D depth-sorts every segment every frame. Past the budget each line is thinned\n"
                          "evenly; the tracing itself keeps its full resolution.");
    ImGui::SetNextItemWidth(full);
    ImGui::SliderFloat("Line width", &stream3DWidth, 0.5f, 5.0f, "%.1f");
    FieldColormapButton(colormap3D, full);
    ImGui::Checkbox("Show solids", &stream3DShowSolids);
    if (stream3DShowSolids)
    {
        ImGui::SetNextItemWidth(full);
        ImGui::SliderFloat("Solid size", &stream3DSolidSize, 0.5f, 12.0f, "%.1f");
    }

    if (changed) stream3DDirty = true;
    ImGui::Spacing();
    ImGui::TextDisabled("%zu lines, %zu points", streamlines3D.size(), stream3DPoints);
    ImGui::TextDisabled("%zu segments drawn", stream3DSegments);
    ImGui::TextDisabled("max speed %.4g m/s", stream3DMaxSpeed);
    Fit3DViewButton();
}

void SimulationViewer::RenderStreamlines3DTab()
{
    if (grid3D == nullptr) return;
    if (!stream3DInitialised) DefaultStreamlineSeeds3D();
    const Domain &d = SIMULATION.domain;
    UpdateStreamlines3D();
    const double maxSpeed = stream3DMaxSpeed > 0.0 ? stream3DMaxSpeed : 1.0;
    const ImVec2 size = Plot3DSize(MakeScaleFormat(0.0, maxSpeed, "Speed (m/s)").width);

    const double Lx = d.xf - d.x0, Ly = d.yf - d.y0, Lz = d.zf - d.z0;
    const double Lmax = (std::max)(Lx, (std::max)(Ly, Lz));
    ImPlot3D::PushColormap(colormap3D);
    ImGui::BeginGroup();
    if (Begin3DView("##Streamlines3D", size))
    {
        // Plot order is (X, Z, Y): ImPlot3D's third axis is vertical and the simulation's vertical is y.
        ImPlot3D::SetupAxes("X (m)", "Z (m)", "Y (m)");
        ImPlot3D::SetupAxesLimits(d.x0, d.xf, d.z0, d.zf, d.y0, d.yf, ImPlot3DCond_Always);
        if (Lmax > 0.0) ImPlot3D::SetupBoxScale(Lx / Lmax, Lz / Lmax, Ly / Lmax);

        if (stream3DShowSolids)
        {
            CollectSolidSurface3D();
            if (!s3SolidX.empty())
            {
                const ImVec4 solidCol(0.55f, 0.55f, 0.58f, 0.85f);
                ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Square, stream3DSolidSize, solidCol, 0.0f, solidCol);
                ImPlot3D::PlotScatter("##s3solids", s3SolidX.data(), s3SolidZ.data(), s3SolidY.data(), (int)s3SolidX.size());
            }
        }

        const int bins = (int)s3SegX.size();
        for (int b = 0; b < bins; b++)
        {
            if (s3SegX[b].empty()) continue;
            const ImVec4 col = ImPlot3D::SampleColormap((b + 0.5f) / bins, colormap3D);
            ImPlot3D::SetNextLineStyle(col, stream3DWidth);
            ImPlot3D::PlotLine("##s3bin", s3SegX[b].data(), s3SegZ[b].data(), s3SegY[b].data(), (int)s3SegX[b].size(),
                               ImPlot3DLineFlags_Segments | ImPlot3DLineFlags_NoLegend);
        }
        End3DView();
    }
    ImGui::EndGroup();
    ImGui::SameLine();
    ImPlot::PushColormap(colormap3D);
    ColourBar("Speed (m/s)", 0.0, maxSpeed, size.y);
    ImPlot::PopColormap();
    ImPlot3D::PopColormap();
}

void SimulationViewer::UpdateStreamlines()
{
    if (grid2D == nullptr) return;
    // Called inside the field plot: the window on screen and its scale set the spacing and the region traced.
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    const ImVec2 size = ImPlot::GetPlotSize();
    if (size.x < 2.0f || size.y < 2.0f) return;
    const double key[8] = {lim.X.Min, lim.X.Max, lim.Y.Min, lim.Y.Max, std::floor(size.x), std::floor(size.y),
                           (double)streamSpacingPx, (double)grid2D->Nx * 100003.0 + grid2D->Ny};
    bool moved = false;
    for (int n = 0; n < 8; n++) moved = moved || key[n] != streamKey[n];
    const double now = GetWallTime();
    const bool newStep = frameIT != streamIT && now - streamTracedAt >= 0.25;
    if (!moved && !newStep) return;
    for (int n = 0; n < 8; n++) streamKey[n] = key[n];

    // Spacing in pixels at this zoom, never below one cell; the step a fraction of it, between a quarter cell and
    // one cell, so a zoomed-out fine grid traces a few thousand points rather than every cell.
    const double perPx = (std::max)((lim.X.Max - lim.X.Min) / size.x, (lim.Y.Max - lim.Y.Min) / size.y);
    const double dh = grid2D->dh;
    StreamlineSettings st;
    st.separation = (std::max)(streamSpacingPx * perPx, dh);
    st.step = (std::max)(0.25 * dh, (std::min)(dh, st.separation / 8.0));
    st.maxPoints = 8000;
    st.minSpeed = 1e-9;
    st.x0 = lim.X.Min; st.x1 = lim.X.Max; st.y0 = lim.Y.Min; st.y1 = lim.Y.Max;
    ComputeStreamlines2D(grid2D, st, streamlines);
    streamMaxSpeed = 0.0;
    for (const Streamline &l : streamlines)
        for (double v : l.speed) streamMaxSpeed = (std::max)(streamMaxSpeed, v);
    streamIT = frameIT;
    streamTracedAt = now;
}

// Each line is split into runs of one colour bin and drawn as polylines, so joins stay clean and the
// draw list gets one call per run rather than one per segment.
void SimulationViewer::DrawStreamlines()
{
    if (!showStreamlines || DIMENSION != 2) return;
    UpdateStreamlines();
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    if (dl == nullptr) return;

    const int bins = 32;
    const ImU32 flat = (streamColour == 2) ? IM_COL32(15, 15, 15, 230) : IM_COL32(245, 245, 245, 230);
    const double inv = streamMaxSpeed > 0.0 ? 1.0 / streamMaxSpeed : 0.0;
    ImPlot::PushPlotClipRect();
    for (const Streamline &l : streamlines)
    {
        const size_t n = l.x.size();
        if (n < 2) continue;
        streamPixels.resize(n);
        for (size_t k = 0; k < n; k++) streamPixels[k] = ImPlot::PlotToPixels(l.x[k], l.y[k]);

        size_t start = 0;
        int bin = (int)std::min((double)(bins - 1), l.speed[0] * inv * bins);
        for (size_t k = 1; k <= n; k++)
        {
            const int b = (k < n) ? (int)std::min((double)(bins - 1), l.speed[k] * inv * bins) : -1;
            if (streamColour != 0 && k < n) continue;
            if (b == bin && k < n) continue;
            const size_t end = (k < n) ? k : n - 1;
            ImU32 col = flat;
            if (streamColour == 0) col = ImGui::GetColorU32(ImPlot::SampleColormap((bin + 0.5f) / bins));
            if (end > start)
                dl->AddPolyline(&streamPixels[start], (int)(end - start + 1), col, ImDrawFlags_None, streamWidth);
            start = end;
            bin = b;
        }

        if (streamArrows && n >= 5)
        {
            const size_t m = n / 2;
            ImVec2 d(streamPixels[m + 1].x - streamPixels[m - 1].x, streamPixels[m + 1].y - streamPixels[m - 1].y);
            const float len = std::sqrt(d.x * d.x + d.y * d.y);
            if (len > 0.0f)
            {
                d.x /= len; d.y /= len;
                const float a = 3.0f + 2.5f * streamWidth;
                const ImVec2 tip(streamPixels[m].x + d.x * a, streamPixels[m].y + d.y * a);
                const ImVec2 left(streamPixels[m].x - d.x * a + d.y * a * 0.7f, streamPixels[m].y - d.y * a - d.x * a * 0.7f);
                const ImVec2 right(streamPixels[m].x - d.x * a - d.y * a * 0.7f, streamPixels[m].y - d.y * a + d.x * a * 0.7f);
                ImU32 col = flat;
                if (streamColour == 0)
                    col = ImGui::GetColorU32(ImPlot::SampleColormap((float)std::min(1.0, l.speed[m] * inv)));
                dl->AddTriangleFilled(tip, left, right, col);
            }
        }
    }
    ImPlot::PopPlotClipRect();
}

void SimulationViewer::DrawMeshOverlay()
{
    if (!showMeshOverlay) return;
    const std::vector<double> *hf = nullptr, *vf = nullptr;
    if (!ViewFaceTables(hf, vf)) return;

    ImDrawList *dl = ImPlot::GetPlotDrawList();
    if (dl == nullptr) return;

    const int nh = (int)hf->size() - 1, nv = (int)vf->size() - 1;
    const double xa = hf->front(), xb = hf->back();
    const double ya = vf->front(), yb = vf->back();

    // How many pixels the plot spans right now, so the stride follows the zoom: zoom in and
    // more of the mesh appears rather than the whole thing staying a grey wash.
    const ImVec2 pa = ImPlot::PlotToPixels(xa, ya);
    const ImVec2 pb = ImPlot::PlotToPixels(xb, yb);
    const float wpx = std::fabs(pb.x - pa.x), hpx = std::fabs(pb.y - pa.y);
    const int strideX = (std::max)(1, (int)std::ceil(nh / (double)(std::max)(1.0f, wpx / 3.0f)));
    const int strideY = (std::max)(1, (int)std::ceil(nv / (double)(std::max)(1.0f, hpx / 3.0f)));

    const ImU32 col = IM_COL32(235, 240, 245, meshOverlayAlpha);
    ImPlot::PushPlotClipRect();
    for (int a = 0; a <= nh; a += strideX)
        dl->AddLine(ImPlot::PlotToPixels((*hf)[a], ya), ImPlot::PlotToPixels((*hf)[a], yb), col);
    for (int b = 0; b <= nv; b += strideY)
        dl->AddLine(ImPlot::PlotToPixels(xa, (*vf)[b]), ImPlot::PlotToPixels(xb, (*vf)[b]), col);
    ImPlot::PopPlotClipRect();
}

// The field resampled onto an image uniform in PHYSICAL space, so the heatmap's equal cells cannot distort
// a stretched grid (Utils::ResampleFieldToPhysical).
void SimulationViewer::ResampleHeatmapToPhysical()
{
    const MAC *g = ViewGrid();
    if (g == nullptr || !g->nonUniform) return;
    if (heatmapData.empty() || heatmapRows <= 0 || heatmapCols <= 0) return;
    const std::vector<double> *hf = nullptr, *vf = nullptr;
    if (!ViewFaceTables(hf, vf)) return;

    // Sized off the SMALLEST cell, one pixel each (a count-based size aliases the fine cells away), and
    // capped by the plot's screen size: PlotHeatmap's cost is the image area.
    const int screenCap = (std::max)(64, (int)(2.0f * lastPlotPx));
    const int cap = (std::min)(2048, screenCap);
    const int outCols = ResampleResolutionFor(*hf, 1, cap);
    const int outRows = ResampleResolutionFor(*vf, 1, cap);

    std::vector<double> out;
    ResampleFieldToPhysical(heatmapData, heatmapRows, heatmapCols, *hf, *vf,
                            out, outRows, outCols);
    if (out.empty()) return;

    heatmapData.swap(out);
    heatmapRows = outRows;
    heatmapCols = outCols;
}

void SimulationViewer::ExtractSliceData()
{
    heatmapData.clear();

    if (DIMENSION == 2)
    {
        if (!grid2D)
        {
            std::cout << "[ERROR] Grid2D is null!" << std::endl;
            return;
        }

        try
        {
            if (selectedComponent == 1)
            { // U
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx + 1;
                heatmapData.resize(heatmapRows * heatmapCols);

                for (int y = 0; y < grid2D->Ny; y++)
                {
                    for (int x = 0; x <= grid2D->Nx; x++)
                    {
                        int flipped_y = (grid2D->Ny - 1) - y;
                        int idx = flipped_y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetU(y, x);
                    }
                }
            }
            else if (selectedComponent == 2)
            { // V
                heatmapRows = grid2D->Ny + 1;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);

                for (int y = 0; y <= grid2D->Ny; y++)
                {
                    for (int x = 0; x < grid2D->Nx; x++)
                    {
                        int flipped_y = grid2D->Ny - y;
                        int idx = flipped_y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetV(y, x);
                    }
                }
            }
            else if (selectedComponent == 4)
            { // Pressure
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);

                for (int y = 0; y < grid2D->Ny; y++)
                {
                    for (int x = 0; x < grid2D->Nx; x++)
                    {
                        int flipped_y = (grid2D->Ny - 1) - y;
                        int idx = flipped_y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetP(y, x);
                    }
                }
            }
            else if (selectedComponent >= SCALAR_COMPONENT_BASE)
            { // a registered scalar -- the pressure path with GetS in place of GetP
                const int h = selectedComponent - SCALAR_COMPONENT_BASE;
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);

                for (int y = 0; y < grid2D->Ny; y++)
                {
                    for (int x = 0; x < grid2D->Nx; x++)
                    {
                        int flipped_y = (grid2D->Ny - 1) - y;
                        int idx = flipped_y * heatmapCols + x;
                        heatmapData[idx] = (h < grid2D->ScalarCount()) ? grid2D->GetS(h, y, x) : 0.0;
                    }
                }
            }
            else if (selectedComponent == 5)
            { // Solid Mask
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);

                for (int y = 0; y < grid2D->Ny; y++)
                {
                    for (int x = 0; x < grid2D->Nx; x++)
                    {
                        int flipped_y = (grid2D->Ny - 1) - y;
                        int idx = flipped_y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetSolid(y, x);
                    }
                }
            }
            else if (selectedComponent == 6)
            { // Divergence
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);

                for (int y = 0; y < grid2D->Ny; y++)
                {
                    for (int x = 0; x < grid2D->Nx; x++)
                    {
                        int flipped_y = (grid2D->Ny - 1) - y;
                        int idx = flipped_y * heatmapCols + x;
                        heatmapData[idx] = grid2D->GetDivergencyAt(y, x);
                    }
                }
            }
            else if (selectedComponent == 7)
            { // Magnitude
                heatmapRows = grid2D->Ny;
                heatmapCols = grid2D->Nx;
                heatmapData.resize(heatmapRows * heatmapCols);

                for (int y = 0; y < grid2D->Ny; y++)
                {
                    for (int x = 0; x < grid2D->Nx; x++)
                    {
                        // Interpolate velocities to cell center
                        float u = 0.5f * (grid2D->GetU(y, x) + grid2D->GetU(y, x + 1));
                        float v = 0.5f * (grid2D->GetV(y, x) + grid2D->GetV(y + 1, x));
                        float mag = std::sqrt(u * u + v * v);

                        int flipped_y = (grid2D->Ny - 1) - y;
                        int idx = flipped_y * heatmapCols + x;
                        heatmapData[idx] = mag;
                    }
                }
            }

            // Uniform grids fall straight through; a stretched one is resampled so the
            // heatmap's equal-width cells cannot distort the picture.
            ResampleHeatmapToPhysical();
        }
        catch (const std::exception &e)
        {
            std::cout << "[EXCEPTION] Error in ExtractSliceData (2D): " << e.what() << std::endl;
            return;
        }
    }
    else
    {
        if (!grid3D)
            return;

        const int Nx = grid3D->Nx, Ny = grid3D->Ny, Nz = grid3D->Nz;
        SlicePlaneInfo pi = GetSlicePlaneInfo(slicePlane, Nx, Ny, Nz);
        sliceIndex = (std::max)(0, (std::min)(sliceIndex, pi.fixedCells - 1));

        heatmapRows = pi.vCells;
        heatmapCols = pi.hCells;
        heatmapData.assign((size_t)heatmapRows * heatmapCols, 0.0);

        // Every component is sampled at the cell centre, so all of them share one grid,
        // one aspect ratio and one orientation.
        auto sample = [&](int i, int j, int k) -> double
        {
            switch (selectedComponent)
            {
            case 1: // U
                return 0.5 * (grid3D->GetU(i, j, k) + grid3D->GetU(i, j + 1, k));
            case 2: // V
                return 0.5 * (grid3D->GetV(i, j, k) + grid3D->GetV(i + 1, j, k));
            case 3: // W
                return 0.5 * (grid3D->GetW(i, j, k) + grid3D->GetW(i, j, k + 1));
            case 4:
                return grid3D->GetP(i, j, k);
            case 5:
                return grid3D->GetSolid(i, j, k);
            case 6:
                return grid3D->GetDivergencyAt(i, j, k);
            case 7:
            {
                double u = 0.5 * (grid3D->GetU(i, j, k) + grid3D->GetU(i, j + 1, k));
                double v = 0.5 * (grid3D->GetV(i, j, k) + grid3D->GetV(i + 1, j, k));
                double w = 0.5 * (grid3D->GetW(i, j, k) + grid3D->GetW(i, j, k + 1));
                return std::sqrt(u * u + v * v + w * w);
            }
            default:
                // Registered scalars sit past the fixed eight -- the pressure case with
                // GetS in place of GetP.
                if (selectedComponent >= SCALAR_COMPONENT_BASE)
                {
                    const int h = selectedComponent - SCALAR_COMPONENT_BASE;
                    if (h < grid3D->ScalarCount()) return grid3D->GetS(h, i, j, k);
                }
                return 0.0;
            }
        };

        try
        {
            for (int v = 0; v < pi.vCells; v++)
            {
                for (int h = 0; h < pi.hCells; h++)
                {
                    int i, j, k;
                    SliceToGrid(slicePlane, h, v, sliceIndex, i, j, k);
                    // ImPlot draws heatmap row 0 at the top, so flip the vertical axis to
                    // put the origin at the bottom left where the physical axes expect it.
                    int row = (pi.vCells - 1) - v;
                    heatmapData[(size_t)row * heatmapCols + h] = sample(i, j, k);
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cout << "[EXCEPTION] Error in ExtractSliceData (3D): " << e.what() << std::endl;
            heatmapData.clear();
            return;
        }

        // Same as 2D: a stretched grid is resampled onto an image uniform in physical space,
        // using the face tables of whichever two axes this plane puts on screen.
        ResampleHeatmapToPhysical();
    }

    heatmapVersion++;
    if (autoScale)
    {
        ComputeMinMax();
    }
}

// The view pane follows the method's capabilities, never its name: a grid method shows the field, a particle
// method its particles, a hybrid both.
void SimulationViewer::DrawViewPane()
{
    frameIT = (int)SIMULATION.telemetry.iteration;
    if ((DIMENSION == 2 && !grid2D) || (DIMENSION == 3 && !grid3D))
        return;
    RecordProbes();
    // With a grid the left column (field controls, line profile, probes) stays put and the tabs switch only the
    // plot beside it.
    if (MethodUsesGrid())
    {
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float colW = (std::max)(260.0f, (std::min)(avail.x - 200.0f, avail.x * controlsWidth));
        const float gapY = ImGui::GetStyle().ItemSpacing.y;
        const float analysisH = (std::max)(160.0f, (std::min)(avail.y - 160.0f, avail.y * analysisHeight));
        ImGui::BeginGroup();
        ImGui::BeginChild("##fieldControls", ImVec2(colW, avail.y - analysisH - gapY), ImGuiChildFlags_Borders);
        if (viewTab == VIEW_PARTICLES && MethodUsesParticles())
            ParticleControls(true);
        else if (viewTab == VIEW_STREAMLINES3D && DIMENSION == 3 && grid3D != nullptr)
            StreamlineControls3D();
        else
            FieldControls();
        ImGui::EndChild();
        ImGui::BeginChild("##analysis", ImVec2(colW, analysisH), ImGuiChildFlags_Borders);
        DrawAnalysisPanel();
        ImGui::EndChild();
        ImGui::EndGroup();
        ImGui::SameLine();
    }
    ImGui::BeginChild("##viewPlots", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (ImGui::BeginTabBar("ViewTabs"))
    {
        if (MethodUsesGrid() && ImGui::BeginTabItem("Field"))
        {
            viewTab = VIEW_FIELD;
            ImGui::BeginChild("##fieldPlot", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            FieldPlot();
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (MethodUsesGrid() && DIMENSION == 3 && grid3D != nullptr && ImGui::BeginTabItem("Streamlines 3D"))
        {
            viewTab = VIEW_STREAMLINES3D;
            RenderStreamlines3DTab();
            ImGui::EndTabItem();
        }
        if (MethodUsesParticles() && ImGui::BeginTabItem("Particles"))
        {
            viewTab = VIEW_PARTICLES;
            RenderParticleView(!MethodUsesGrid());   // without a grid there is no left column
            ImGui::EndTabItem();
        }
        if (MethodUsesParticles() && DIMENSION == 3 && ImGui::BeginTabItem("Surface"))
        {
            viewTab = VIEW_SURFACE;
            RenderSurfaceTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();
}

void SimulationViewer::DrawControlPane()
{
    if (!ImGui::BeginTabBar("ControlTabs"))
        return;
    if (ImGui::BeginTabItem("Run"))
    {
        RenderControlUI();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Export"))
    {
        RenderExportUI();
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

// The telemetry the run records, one tab per series; the derived ones only when their export is on, since
// that is what switches their computation on.
void SimulationViewer::DrawTelemetryPane()
{
    if (!ImGui::BeginTabBar("TelemetryTabs"))
        return;
    if (ImGui::BeginTabItem("Solver"))
    {
        RenderSolverTelemetryTab();
        ImGui::EndTabItem();
    }
    if (MethodUsesGrid() && ImGui::BeginTabItem("Flow integrals"))
    {
        RenderFlowIntegralsTab();
        ImGui::EndTabItem();
    }
    if (MethodUsesGrid() && SIMULATION.exports.aerodynamics && ImGui::BeginTabItem("Aerodynamics"))
    {
        RenderAerodynamicsUI();
        ImGui::EndTabItem();
    }
    if (SIMULATION.exports.dambreak && ImGui::BeginTabItem("Free surface"))
    {
        RenderDambreakUI();
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

void SimulationViewer::RenderDambreakUI(){
    if (!SIMULATION.exports.dambreak)
{
    return;
}

const double t = SIMULATION.telemetry.time.empty() ? 0.0 : SIMULATION.telemetry.time.back();
const double window = 5.0;

float avail = ImGui::GetContentRegionAvail().x;
float plot_w = avail * 0.5f - ImGui::GetStyle().ItemSpacing.x * 0.5f;
ImVec2 plot_size(plot_w, 300);

// ---- Row 1: Fluid Level and Front Position ----
if (ImPlot::BeginPlot("Fluid Level (Height)", plot_size, TELEMETRY_PLOT_FLAGS))
{
    ImPlot::SetupAxes("Time", "Height (m)",
                      ImPlotAxisFlags_None,
                      ImPlotAxisFlags_AutoFit);

    if (!SIMULATION.dambreak.fluidLevel.empty())
    {
        ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(0));
        
        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            0.0,
            t + window * 0.5,
            ImGuiCond_Always);
        ImPlot::SetupAxisLimits(
            ImAxis_Y1,
            SIMULATION.domain.y0,
            SIMULATION.domain.yf,
            ImGuiCond_Always);
        
        ImPlot::PlotLine("Height",
                         SIMULATION.telemetry.time.data(),
                         SIMULATION.dambreak.fluidLevel.data(),
                         SIMULATION.dambreak.fluidLevel.size());
        ImPlot::PopStyleColor();

        // Show current value
        double current_height = SIMULATION.dambreak.fluidLevel.back();
        ImPlot::PlotText(("H = " + std::to_string(current_height)).c_str(),
                         SIMULATION.telemetry.time.back(), current_height);
    }
    ImPlot::EndPlot();
}

ImGui::SameLine();

if (ImPlot::BeginPlot("Fluid Front Position", plot_size, TELEMETRY_PLOT_FLAGS))
{
    ImPlot::SetupAxes("Time", "Position (m)",
                      ImPlotAxisFlags_None,
                      ImPlotAxisFlags_AutoFit);

    if (!SIMULATION.dambreak.fluidFrontPosition.empty())
    {
        ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(1));
        
        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            0.0,
            t + window * 0.5,
            ImGuiCond_Always);
        ImPlot::SetupAxisLimits(
            ImAxis_Y1,
            SIMULATION.domain.x0,
            SIMULATION.domain.xf,
            ImGuiCond_Always);


        ImPlot::PlotLine("Front",
                         SIMULATION.telemetry.time.data(),
                         SIMULATION.dambreak.fluidFrontPosition.data(),
                         SIMULATION.dambreak.fluidFrontPosition.size());
        ImPlot::PopStyleColor();
        
        // Show current value
        double current_pos = SIMULATION.dambreak.fluidFrontPosition.back();
        ImPlot::PlotText(("X = " + std::to_string(current_pos)).c_str(),
                         SIMULATION.telemetry.time.back(), current_pos);
    }
    ImPlot::EndPlot();
}

}

// The next widget fills the rest of the line after its own label, so a narrow pane never cuts it off.
static void FillLineWidth(const char *label)
{
    const float w = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(label, nullptr, true).x
                    - ImGui::GetStyle().ItemInnerSpacing.x;
    ImGui::SetNextItemWidth((std::max)(w, 40.0f));
}

// The span the grid covers and where it starts, off the face tables: Nx*dh is wrong on a stretched grid
// (dh is the smallest cell) and when the length is not a whole number of cells.
static inline void GridExtent2D(double &ox, double &oy, double &w, double &h)
{
    const MAC *g = SIMULATION.GRID_SOL;
    if (g == nullptr || g->Nx <= 0 || g->Ny <= 0)
    {
        ox = SIMULATION.domain.x0; oy = SIMULATION.domain.y0;
        w = SIMULATION.domain.xf - ox; h = SIMULATION.domain.yf - oy;
        return;
    }
    ox = g->FacePos(MAC::X, 0); oy = g->FacePos(MAC::Y, 0);
    w = g->FacePos(MAC::X, g->Nx) - ox;
    h = g->FacePos(MAC::Y, g->Ny) - oy;
}

// The surface preview: the mesh that went to disk (MeshExport::LAST_SURFACE), swizzled once per new surface
// into plot order (x, z, y), its triangles strided to the budget.
void SimulationViewer::RenderSurfaceTab()
{
    // Capturing is off until something asks to see it, so the headless harnesses -- which
    // call SurfaceFromParticles thousands of times in a sweep -- never pay for the copy.
    MeshExport::CAPTURE_LAST_SURFACE = true;

    const MeshExport::LastSurface &ls = MeshExport::LAST_SURFACE;

    if (ls.Empty())
    {
        ImGui::TextWrapped(
            "No surface has been generated yet.\n\n"
            "Turn on \"Export Surface Mesh (PLY)\" in the Export panel and run: every frame "
            "it writes is cached here as it is produced. 3D particle methods only -- there is no "
            "2D polygoniser, and a grid-only method has no particle cloud to surface.");
        return;
    }

    const size_t tris = ls.mesh.TriangleCount();
    ImGui::Text("%zu triangles, %zu vertices", tris, ls.mesh.VertexCount());
    ImGui::SameLine();
    ImGui::TextDisabled("| voxel %.5f m, spacing %.5f m", ls.voxel, ls.spacing);

    ImGui::SetNextItemWidth(220);
    ImGui::SliderInt("Triangle budget", &surfaceBudget, 2000, 400000, "%d",
                     ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("PlotMesh emits and depth-sorts every triangle every frame, so the\n"
                          "whole mesh is not what you want on screen while a run is going.\n"
                          "The budget strides the index buffer; all the vertices are kept, so\n"
                          "the shape still reads at a fraction of the triangles.");
    ImGui::Checkbox("Wireframe", &surfaceWireframe);
    ImGui::SameLine();
    ImGui::Checkbox("Domain box", &surfaceShowDomain);

    // ---- rebuild the draw buffers only when they are actually stale ----
    if (ls.generation != surfaceDrawnGeneration || surfaceBudget != surfaceDrawnBudget)
    {
        surfaceDrawnGeneration = ls.generation;
        surfaceDrawnBudget = surfaceBudget;

        surfaceVtx.resize(ls.mesh.VertexCount());
        for (size_t v = 0; v < ls.mesh.VertexCount(); v++)
            surfaceVtx[v] = ImPlot3DPoint(ls.mesh.vtx[3 * v + 0],   // x
                                          ls.mesh.vtx[3 * v + 2],   // z  -- the plot's second
                                          ls.mesh.vtx[3 * v + 1]);  // y  -- the plot's vertical

        const int stride = MeshExport::PreviewStride(tris, surfaceBudget);
        surfaceIdx.clear();
        surfaceIdx.reserve(3 * (tris / (size_t)stride + 1));
        for (size_t t = 0; t < tris; t += (size_t)stride)
        {
            surfaceIdx.push_back(ls.mesh.tri[3 * t + 0]);
            surfaceIdx.push_back(ls.mesh.tri[3 * t + 1]);
            surfaceIdx.push_back(ls.mesh.tri[3 * t + 2]);
        }
    }

    const size_t drawn = surfaceIdx.size() / 3;
    if (drawn < tris)
        ImGui::TextDisabled("drawing %zu of %zu triangles (1 in %d)", drawn, tris,
                            MeshExport::PreviewStride(tris, surfaceBudget));
    else
        ImGui::TextDisabled("drawing the whole mesh");

    const double Lx = SIMULATION.domain.xf - SIMULATION.domain.x0;
    const double Ly = SIMULATION.domain.yf - SIMULATION.domain.y0;
    const double Lz = SIMULATION.domain.zf - SIMULATION.domain.z0;
    const double Lmax = (std::max)(Lx, (std::max)(Ly, Lz));

    // An explicit size from the space left (a -1 size made an auto-resizing window shrink by a pixel per frame).
    Fit3DViewButton();
    if (Begin3DView("##SurfacePreview", Plot3DSize(0.0f)))
    {
        // The same (x, z, y) order every other 3D view here uses, so the simulation's y is up
        // and the preview matches the particle view beside it.
        ImPlot3D::SetupAxes("X (m)", "Z (m)", "Y (m)");
        ImPlot3D::SetupAxesLimits(SIMULATION.domain.x0, SIMULATION.domain.xf,
                                  SIMULATION.domain.z0, SIMULATION.domain.zf,
                                  SIMULATION.domain.y0, SIMULATION.domain.yf,
                                  ImPlot3DCond_Always);
        if (Lmax > 0.0)
            ImPlot3D::SetupBoxScale(Lx / Lmax, Lz / Lmax, Ly / Lmax);

        ImPlot3DMeshFlags flags = surfaceWireframe ? ImPlot3DMeshFlags_NoFill
                                                   : ImPlot3DMeshFlags_NoLines;
        if (!surfaceVtx.empty() && !surfaceIdx.empty())
            ImPlot3D::PlotMesh("liquid", surfaceVtx.data(), surfaceIdx.data(),
                               (int)surfaceVtx.size(), (int)surfaceIdx.size(), flags);

        if (surfaceShowDomain)
        {
            // The liquid reaches the domain box itself: the border ring lies outside it, so there is
            // no inner wall box to draw any more (the second pass below never runs).
            const double w = 0.0;
            const double bx[2] = {SIMULATION.domain.x0, SIMULATION.domain.xf};
            const double by[2] = {SIMULATION.domain.y0, SIMULATION.domain.yf};
            const double bz[2] = {SIMULATION.domain.z0, SIMULATION.domain.zf};
            for (int pass = 0; pass < 2; pass++)
            {
                const double in = (pass == 0) ? 0.0 : w;
                if (pass == 1 && !(w > 0.0)) break;
                const double X[2] = {bx[0] + in, bx[1] - in};
                const double Y[2] = {by[0] + in, by[1] - in};
                const double Z[2] = {bz[0] + in, bz[1] - in};
                for (int a = 0; a < 2; a++)
                    for (int b = 0; b < 2; b++)
                    {
                        double ex[2] = {X[0], X[1]}, ey[2] = {Y[a], Y[a]}, ez[2] = {Z[b], Z[b]};
                        ImPlot3D::PlotLine(pass ? "##wallx" : "##boxx", ex, ez, ey, 2,
                                           ImPlot3DLineFlags_NoLegend);
                        double fx[2] = {X[a], X[a]}, fy[2] = {Y[0], Y[1]}, fz[2] = {Z[b], Z[b]};
                        ImPlot3D::PlotLine(pass ? "##wally" : "##boxy", fx, fz, fy, 2,
                                           ImPlot3DLineFlags_NoLegend);
                        double gx[2] = {X[a], X[a]}, gy[2] = {Y[b], Y[b]}, gz[2] = {Z[0], Z[1]};
                        ImPlot3D::PlotLine(pass ? "##wallz" : "##boxz", gx, gz, gy, 2,
                                           ImPlot3DLineFlags_NoLegend);
                    }
            }
        }

        End3DView();
    }
}

void SimulationViewer::FieldControls()
{
    {
        {

            // Component selection dropdown
            ImGui::Text("Component:");
            ImGui::SetNextItemWidth(200);
            const char *componentLabels[8] = {
                "Velocity Field",
                "U Velocity",
                "V Velocity",
                "W Velocity",
                "Pressure",
                "Solid Mask",
                "Divergence",
                "Magnitude"};

            // Registered scalars follow the fixed eight (SCALAR_COMPONENT_BASE + handle); rebuilt every frame, since
            // a scalar can be registered after the viewer exists.
            MAC *cg = (DIMENSION == 2) ? grid2D : grid3D;
            std::vector<const char *> labels;
            std::vector<int> componentOf;
            for (int c = 0; c < 8; c++)
            {
                if (DIMENSION == 2 && c == 3) continue;   // no W in 2D
                labels.push_back(componentLabels[c]);
                componentOf.push_back(c);
            }
            for (int h = 0; cg != nullptr && h < cg->ScalarCount(); h++)
            {
                labels.push_back(cg->ScalarName(h).c_str());
                componentOf.push_back(SCALAR_COMPONENT_BASE + h);
            }

            int displayIndex = 0;
            for (size_t c = 0; c < componentOf.size(); c++)
                if (componentOf[c] == selectedComponent) displayIndex = (int)c;
            if (ImGui::Combo("##Component", &displayIndex, labels.data(), (int)labels.size()))
                selectedComponent = componentOf[(size_t)displayIndex];

            ImGui::Separator();

            // Plane selection - only for 3D
            if (DIMENSION == 3)
            {
                SlicePlaneInfo pi = GetSlicePlaneInfo(slicePlane, GetNx(), GetNy(), GetNz());

                ImGui::Text("Slice Plane:");
                ImGui::RadioButton("XY (fix Z)", &slicePlane, 0);
                ImGui::SameLine();
                ImGui::RadioButton("XZ (fix Y)", &slicePlane, 1);
                ImGui::SameLine();
                ImGui::RadioButton("ZY (fix X)", &slicePlane, 2);

                pi = GetSlicePlaneInfo(slicePlane, GetNx(), GetNy(), GetNz());
                sliceIndex = (std::max)(0, (std::min)(sliceIndex, pi.fixedCells - 1));
                ImGui::SliderInt("Slice Index", &sliceIndex, 0, pi.fixedCells - 1);
                {
                    // The fixed axis's cell centre from the tables -- index*dh is wrong on
                    // a stretched grid and ignores the domain origin.
                    const MAC *sg = grid3D;
                    double at = sliceIndex * dh;
                    if (sg != nullptr)
                        at = (slicePlane == 1) ? sg->CentrePos(MAC::Y, sliceIndex)
                           : (slicePlane == 2) ? sg->CentrePos(MAC::X, sliceIndex) : sg->CentrePos(MAC::Z, sliceIndex);
                    ImGui::Text("%s = %.4f m   (%s across, %s up)",
                                pi.fixedLabel, at, pi.hLabel, pi.vLabel);
                }

                ImGui::Separator();
            }
            else
            {
                // Force XY plane for 2D
                slicePlane = 0;
                sliceIndex = 0;
            }

            // Controls based on component type
            if (selectedComponent == 0)
            {
                // Quiver plot controls
                ImGui::Text("Quiver Plot Settings:");

                FieldColormapButton(colormap, 150);
                ImGui::SameLine();
                ImGui::Checkbox("Auto Scale", &autoScale);


                if (!autoScale)
                {
                    ImGui::SetNextItemWidth(225);
                    ImGui::DragFloatRange2("Min / Max", &minValue, &maxValue, 0.01f, -20, 20,
                                           nullptr, nullptr, ImGuiSliderFlags_AlwaysClamp);
                    if (maxValue <= minValue + 0.01f)
                    {
                        maxValue = minValue + 0.01f;
                    }
                }

                ImGui::SetNextItemWidth(225);
                ImGui::DragFloat("Arrow Size", &baseSize, 0.1f, 0, 100);

                ImGui::CheckboxFlags("Normalize", (unsigned int *)&quiverFlags, ImPlotQuiverFlags_Normalize);
                ImGui::CheckboxFlags("Color Coded", (unsigned int *)&quiverFlags, ImPlotQuiverFlags_Colored);
            }
            else
            {
                // Heatmap controls
                if (selectedComponent == 5)
                {
                    // For solid mask, disable auto scale and fix range
                    ImGui::Text("Cell Types: 0=Fluid, 1=Solid, 2=Empty, 3=Inflow, 4=Pressure, 5=Outflow");
                    autoScale = false;
                    minValue = 0.0f;
                    maxValue = 5.0f;
                }
                else
                {
                    ImGui::Checkbox("Auto Scale", &autoScale);
                    if (!autoScale)
                    {
                        ImGui::SliderFloat("Min Value", &minValue, -10.0f, 10.0f);
                        ImGui::SliderFloat("Max Value", &maxValue, -10.0f, 10.0f);
                    }
                }
            }

            {
                ImGui::Checkbox("Show mesh", &showMeshOverlay);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Draws the cell faces over the field, in physical coordinates, so it\n"
                        "follows pan and zoom.\n\n"
                        "On a stretched grid the field itself is resampled onto a uniform\n"
                        "image before drawing -- ImPlot::PlotHeatmap divides one rectangle\n"
                        "into equal cells, so without that a narrow column would be painted\n"
                        "as wide as a fat one and geometry inside a refined band would come\n"
                        "out distorted. The overlay is how you see where the cells actually\n"
                        "are, since the resampled image no longer shows them.");
                if (showMeshOverlay)
                {
                    FillLineWidth("Mesh opacity");
                    ImGui::SliderInt("Mesh opacity", &meshOverlayAlpha, 10, 255);
                }
                if (ViewGrid() != nullptr && ViewGrid()->nonUniform)
                    ImGui::TextDisabled("Stretched grid, field resampled");
            }

            if (DIMENSION == 2)
            {
                const bool quiver = (selectedComponent == 0);
                if (quiver) ImGui::BeginDisabled();
                ImGui::Checkbox("Streamlines", &showStreamlines);
                if (quiver) ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip(quiver ? "Drawn over the scalar views; pick one (Magnitude works well).\n"
                                               "The vector view still plots in cell indices, not metres."
                                             : "Evenly spaced streamlines of the in-plane velocity, traced through\n"
                                               "the staggered field. They stop at solids, at the domain edge, at\n"
                                               "stagnation points and within half a spacing of another line.");
                if (showStreamlines && !quiver)
                {
                    ImGui::Checkbox("Hide field", &streamlinesOnly);
                    ImGui::SameLine();
                    ImGui::Checkbox("Arrows", &streamArrows);
                    FillLineWidth("Spacing (px)");
                    ImGui::SliderFloat("Spacing (px)", &streamSpacingPx, 8.0f, 80.0f, "%.0f");
                    FillLineWidth("Line width");
                    ImGui::SliderFloat("Line width", &streamWidth, 0.5f, 4.0f, "%.1f");
                    static const char *colours[] = {"Speed", "White", "Black"};
                    FillLineWidth("Line colour");
                    ImGui::Combo("Line colour", &streamColour, colours, 3);
                    ImGui::TextDisabled("%zu lines, max speed %.4g", streamlines.size(), streamMaxSpeed);
                }
            }

            ImGui::Separator();

            if (ImGui::Button("Fit view")) fieldViewReset = true;
            ImGui::TextDisabled("Mouse wheel zooms, drag pans.");

            // Extract data
            if (selectedComponent == 0)
            {
                ExtractQuiverData();
            }
            else
            {
                // Only when something it depends on actually changed.
                const int gnx = GetNx(), gny = GetNy();
                if (frameIT != sliceIT || selectedComponent != sliceComponent ||
                    slicePlane != slicePlaneCached || sliceIndex != sliceIndexCached ||
                    gnx != sliceNx || gny != sliceNy || heatmapData.empty())
                {
                    ExtractSliceData();
                    sliceIT = frameIT; sliceComponent = selectedComponent;
                    slicePlaneCached = slicePlane; sliceIndexCached = sliceIndex;
                    sliceNx = gnx; sliceNy = gny;
                }
            }

            // Display info
            ImGui::Separator();
            if (DIMENSION == 2)
            {
                const MAC *vg = ViewGrid();
                ImGui::Text("Grid: %dx%d (2D)", GetNx(), GetNy());
                ImGui::Text("Domain: %.4g x %.4g", vg->FacePos(MAC::X, GetNx() - 1) - vg->FacePos(MAC::X, 1),
                            vg->FacePos(MAC::Y, GetNy() - 1) - vg->FacePos(MAC::Y, 1));
            }
            else
            {
                const MAC *vg = ViewGrid();
                ImGui::Text("Grid: %dx%dx%d (3D)", GetNx(), GetNy(), GetNz());
                ImGui::Text("Domain: %.4g x %.4g x %.4g", vg->FacePos(MAC::X, GetNx() - 1) - vg->FacePos(MAC::X, 1),
                            vg->FacePos(MAC::Y, GetNy() - 1) - vg->FacePos(MAC::Y, 1),
                            vg->FacePos(MAC::Z, GetNz() - 1) - vg->FacePos(MAC::Z, 1));
            }

            if (selectedComponent == 0)
            {
                ImGui::Text("Vectors: %zu", quiverX.size());
            }
            else
            {
                ImGui::Text("Heatmap: %dx%d", heatmapCols, heatmapRows);
            }
            ImGui::Text("Range: [%.6f, %.6f]", minValue, maxValue);

            ImGui::Separator();
            HoverReadout();
        }
    }
}

// Inside a field plot: remember where the mouse is, in the plane's physical coordinates.
void SimulationViewer::TrackHover()
{
    hoverValid = ImPlot::IsPlotHovered();
    if (!hoverValid) return;
    const ImPlotPoint m = ImPlot::GetPlotMousePos();
    hoverA = m.x;
    hoverB = m.y;
}

// The values at the point under the mouse, sampled as the line profile samples them.
void SimulationViewer::HoverReadout()
{
    ImGui::TextUnformatted("Under the pointer");
    if (!hoverValid)
    {
        ImGui::TextDisabled(DIMENSION == 2 && selectedComponent == 0 ? "Hover a scalar view of the field."
                                                                      : "Hover the field plot.");
        return;
    }
    double x, y, z;
    ProfilePoint(hoverA, hoverB, x, y, z);
    MAC *g = (DIMENSION == 2) ? grid2D : grid3D;
    const bool inside = (DIMENSION == 2) ? (x >= g->FacePos(MAC::X, 0) && x <= g->FacePos(MAC::X, g->Nx) &&
                                            y >= g->FacePos(MAC::Y, 0) && y <= g->FacePos(MAC::Y, g->Ny))
                                         : (x >= g->FacePos(MAC::X, 0) && x <= g->FacePos(MAC::X, g->Nx) &&
                                            y >= g->FacePos(MAC::Y, 0) && y <= g->FacePos(MAC::Y, g->Ny) &&
                                            z >= g->FacePos(MAC::Z, 0) && z <= g->FacePos(MAC::Z, g->Nz));
    if (DIMENSION == 3) ImGui::Text("x, y, z  %.4g, %.4g, %.4g", x, y, z);
    else ImGui::Text("x, y  %.4g, %.4g", x, y);
    if (!inside)
    {
        ImGui::TextDisabled("Outside the grid.");
        return;
    }
    const int i = g->CellAt(MAC::Y, y), j = g->CellAt(MAC::X, x);
    const int k = (DIMENSION == 3) ? g->CellAt(MAC::Z, z) : 0;
    static const char *types[6] = {"fluid", "solid", "empty", "inlet", "pressure", "outflow"};
    const int t = (DIMENSION == 3) ? g->GetSolid(i, j, k) : g->GetSolid(i, j);
    const char *typeName = (t >= 0 && t < 6) ? types[t] : "?";
    if (DIMENSION == 3) ImGui::TextDisabled("cell (i %d, j %d, k %d), %s", i, j, k, typeName);
    else ImGui::TextDisabled("cell (i %d, j %d), %s", i, j, typeName);

    std::vector<std::string> names;
    ProfileQuantities(names);
    for (int q = 0; q < (int)names.size(); q++)
    {
        const double val = ProfileSample(q, x, y, z);
        if (std::isfinite(val)) ImGui::Text("%-6s %12.6g", names[(size_t)q].c_str(), val);
        else ImGui::TextDisabled("%-6s %12s", names[(size_t)q].c_str(), "-");
    }
}

// The panel under the field controls: a line profile through the field, and the scene's probes against time.
void SimulationViewer::DrawAnalysisPanel()
{
    if (!ImGui::BeginTabBar("AnalysisTabs"))
        return;
    if (ImGui::BeginTabItem("Line profile"))
    {
        ProfileControls();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Probes"))
    {
        RenderProbesTab();
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

// ---- line profile ---------------------------------------------------------------------------------------------

// u, v, (w), p, speed, then the registered scalars by name.
void SimulationViewer::ProfileQuantities(std::vector<std::string> &names) const
{
    names.clear();
    names.push_back("u");
    names.push_back("v");
    if (DIMENSION == 3) names.push_back("w");
    names.push_back("p");
    names.push_back("speed");
    const MAC *g = ViewGrid();
    for (int h = 0; h < g->ScalarCount(); h++) names.push_back(g->ScalarName(h));
}

// A point of the plane on screen (a across, b up) as physical (x, y, z). In 3D the fixed coordinate is the centre
// of the slice the view shows.
bool SimulationViewer::ProfilePoint(double a, double b, double &x, double &y, double &z) const
{
    z = 0.0;
    if (DIMENSION == 2) { x = a; y = b; return true; }
    const MAC *g = ViewGrid();
    switch (slicePlane)
    {
    case 1:  x = a; z = b; y = g->CentrePos(MAC::Y, sliceIndex); break;
    case 2:  z = a; y = b; x = g->CentrePos(MAC::X, sliceIndex); break;
    default: x = a; y = b; z = g->CentrePos(MAC::Z, sliceIndex); break;
    }
    return true;
}

// The quantity at a point, NaN in a solid or outside the grid so the plot shows a gap there.
double SimulationViewer::ProfileSample(int q, double x, double y, double z) const
{
    MAC *g = (DIMENSION == 2) ? grid2D : grid3D;
    const bool open = (DIMENSION == 2) ? IsOpenAt(g, x, y) : IsOpenAt(g, x, y, z);
    if (!open) return std::numeric_limits<double>::quiet_NaN();
    const int nVel = (DIMENSION == 3) ? 3 : 2;
    if (q < nVel)
    {
        const MAC::Component c = q == 0 ? MAC::U : (q == 1 ? MAC::V : MAC::W);
        return (DIMENSION == 2) ? g->ComponentAt(c, x, y) : g->ComponentAt(c, x, y, z);
    }
    // Cell-centred fields go through the probe's sampler, which leaves SOLID cells out of the interpolation: a point
    // beside a wall reads the fluid, not a blend with the meaningless value stored inside the solid.
    Probe pr;
    pr.at[0] = x; pr.at[1] = y; pr.at[2] = z;
    if (q == nVel) { pr.kind = ProbeQuantity::Pressure; return EvaluateProbe(g, nullptr, pr); }
    if (q == nVel + 1)
    {
        double u = 0.0, v = 0.0, w = 0.0;
        if (DIMENSION == 2) VelocityAt(g, x, y, u, v);
        else VelocityAt(g, x, y, z, u, v, w);
        return std::sqrt(u * u + v * v + w * w);
    }
    const int h = q - nVel - 2;
    if (h < 0 || h >= g->ScalarCount()) return std::numeric_limits<double>::quiet_NaN();
    pr.kind = ProbeQuantity::Scalar;
    pr.quantity = g->ScalarName(h);
    return EvaluateProbe(g, nullptr, pr);
}

// Which axes (0 x, 1 y, 2 z) the view puts across, up, and fixes; the same mapping as SliceToGrid.
void SimulationViewer::PlaneAxes(int &h, int &v, int &f) const
{
    if (DIMENSION == 2) { h = 0; v = 1; f = 2; return; }
    switch (slicePlane)
    {
    case 1:  h = 0; v = 2; f = 1; break;
    case 2:  h = 2; v = 1; f = 0; break;
    default: h = 0; v = 1; f = 2; break;
    }
}

// The domain along one axis: the first and last interior faces (the ring lies outside it).
void SimulationViewer::DomainBounds(int a, double &lo, double &hi) const
{
    const MAC *g = ViewGrid();
    const MAC::Axis ax = (MAC::Axis)a;
    const int n = (a == 0) ? g->Nx : (a == 1) ? g->Ny : g->Nz;
    lo = g->FacePos(ax, 1);
    hi = g->FacePos(ax, n - 1);
}

// A line across the view through its middle, in the plane of the current slice.
void SimulationViewer::PlaceProfile(bool horizontal)
{
    int h, v, f;
    PlaneAxes(h, v, f);
    const int along = horizontal ? h : v, across = horizontal ? v : h;
    double lo, hi, clo, chi;
    DomainBounds(along, lo, hi);
    DomainBounds(across, clo, chi);
    profileA[along] = lo;
    profileB[along] = hi;
    profileA[across] = profileB[across] = 0.5 * (clo + chi);
    profileA[f] = profileB[f] = (DIMENSION == 3) ? ViewGrid()->CentrePos((MAC::Axis)f, sliceIndex) : 0.0;
}

// Samples the line evenly, ends included. Run every frame the profile is shown: a few hundred interpolations.
void SimulationViewer::UpdateProfile()
{
    const int n = (std::max)(2, profileSamples);
    double d[3], len2 = 0.0;
    for (int a = 0; a < 3; a++) { d[a] = profileB[a] - profileA[a]; len2 += d[a] * d[a]; }
    if (DIMENSION == 2) { d[2] = 0.0; len2 = d[0] * d[0] + d[1] * d[1]; }
    const double len = std::sqrt(len2);
    // Plotted against the coordinate along the axis the line mostly follows on screen, not the distance from its
    // start: the plot then reads the same way as the field whichever end is "0", and a vertical line plots vertically.
    int h, v, f0;
    PlaneAxes(h, v, f0);
    profileVertical = std::fabs(d[v]) > std::fabs(d[h]);
    profileAxis = profileVertical ? v : h;
    profileS.resize((size_t)n);
    profileV.resize((size_t)n);
    profileC.resize((size_t)n);
    for (int k = 0; k < n; k++)
    {
        const double f = (double)k / (double)(n - 1);
        profileS[(size_t)k] = f * len;
        profileC[(size_t)k] = profileA[profileAxis] + f * d[profileAxis];
        profileV[(size_t)k] = ProfileSample(profileQuantity, profileA[0] + f * d[0], profileA[1] + f * d[1],
                                            profileA[2] + f * d[2]);
    }
}

// One labelled coordinate box in the point table: "x0 [  ]".
static void PointBox(const char *label, double *value)
{
    ImGui::TextUnformatted(label);
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::PushID(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputDouble("##c", value, 0.0, 0.0, "%.4g");
    ImGui::PopID();
}

void SimulationViewer::ProfileControls()
{
    if (!profilePlaced)
    {
        PlaceProfile(true);   // across the middle of the view: a cavity's centreline, a channel's section
        profilePlaced = true;
    }
    ImGui::Checkbox("Show on the field", &showProfile);
    ImGui::SameLine();
    if (ImGui::SmallButton("Horizontal")) PlaceProfile(true);
    ImGui::SameLine();
    if (ImGui::SmallButton("Vertical")) PlaceProfile(false);

    const int dims = (DIMENSION == 3) ? 3 : 2;
    static const char *startLabels[3] = {"x0", "y0", "z0"}, *endLabels[3] = {"xf", "yf", "zf"};
    if (ImGui::BeginTable("##profilePoints", dims, ImGuiTableFlags_SizingStretchSame))
    {
        ImGui::TableNextRow();
        for (int a = 0; a < dims; a++) { ImGui::TableNextColumn(); PointBox(startLabels[a], &profileA[a]); }
        ImGui::TableNextRow();
        for (int a = 0; a < dims; a++) { ImGui::TableNextColumn(); PointBox(endLabels[a], &profileB[a]); }
        ImGui::EndTable();
    }
    if (DIMENSION == 3) ImGui::TextDisabled("Metres. Dragging the ends moves them in the slice's plane.");
    else ImGui::TextDisabled("Metres. Drag the ends on the field to move them.");

    std::vector<std::string> names;
    ProfileQuantities(names);
    profileQuantity = (std::max)(0, (std::min)(profileQuantity, (int)names.size() - 1));
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::BeginCombo("Quantity", names[(size_t)profileQuantity].c_str()))
    {
        for (int q = 0; q < (int)names.size(); q++)
            if (ImGui::Selectable(names[(size_t)q].c_str(), q == profileQuantity)) profileQuantity = q;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70.0f);
    ImGui::InputInt("Samples", &profileSamples, 0, 0);
    profileSamples = (std::max)(10, (std::min)(profileSamples, 5000));

    UpdateProfile();

    const float plotH = (std::max)(120.0f, ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing());
    if (ImPlot::BeginPlot("##profile", ImVec2(-1, plotH), ImPlotFlags_NoLegend | ImPlotFlags_NoMenus))
    {
        static const char *coordLabel[3] = {"x (m)", "y (m)", "z (m)"};
        const char *q = names[(size_t)profileQuantity].c_str();
        const int count = (int)profileC.size();
        // A NaN (a solid, or outside the grid) breaks the line: it is not drawn across.
        if (profileVertical)
        {
            ImPlot::SetupAxes(q, coordLabel[profileAxis], ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                              ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine(q, profileV.data(), profileC.data(), count);
        }
        else
        {
            ImPlot::SetupAxes(coordLabel[profileAxis], q, ImPlotAxisFlags_AutoFit,
                              ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
            ImPlot::PlotLine(q, profileC.data(), profileV.data(), count);
        }
        ImPlot::EndPlot();
    }
    if (ImGui::Button("Save CSV")) SaveProfile();
    if (!profileStatus.empty())
    {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", profileStatus.c_str());
    }
}

// The line and its two ends on the field view, projected onto the plane on screen; the ends can be dragged there.
void SimulationViewer::DrawProfileLine()
{
    if (!showProfile || !profilePlaced) return;
    int h, v, f;
    PlaneAxes(h, v, f);
    const ImVec4 ink = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    double xs[2] = {profileA[h], profileB[h]}, ys[2] = {profileA[v], profileB[v]};
    ImPlot::SetNextLineStyle(ink, 2.0f);
    ImPlot::PlotLine("##profileLine", xs, ys, 2);
    ImPlot::DragPoint(1, &profileA[h], &profileA[v], ink, 5.0f);
    ImPlot::DragPoint(2, &profileB[h], &profileB[v], ink, 5.0f);
    ImPlot::PlotText("0", profileA[h], profileA[v], ImVec2(0, -12));
    ImPlot::PlotText("f", profileB[h], profileB[v], ImVec2(0, -12));
}

// distance, x, y, (z), value: the line as it is now, under the run's export folder.
void SimulationViewer::SaveProfile()
{
    std::vector<std::string> names;
    ProfileQuantities(names);
    const std::string q = names[(size_t)profileQuantity];
    char file[96];
    std::snprintf(file, sizeof(file), "profile_%s_%06u.csv", q.c_str(), SIMULATION.telemetry.iteration);
    const std::string path = ExportDirectory("Profiles") + file;
    std::ofstream out(path);
    if (!out)
    {
        profileStatus = "could not write " + path;
        return;
    }
    out << std::setprecision(12);
    out << "distance,x,y" << (DIMENSION == 3 ? ",z," : ",") << q << "\n";
    const int n = (int)profileS.size();
    for (int k = 0; k < n; k++)
    {
        const double f = n > 1 ? (double)k / (double)(n - 1) : 0.0;
        out << profileS[(size_t)k];
        for (int a = 0; a < ((DIMENSION == 3) ? 3 : 2); a++) out << "," << profileA[a] + f * (profileB[a] - profileA[a]);
        out << "," << profileV[(size_t)k] << "\n";
    }
    profileStatus = "saved " + path;
}

// ---- probes ---------------------------------------------------------------------------------------------------

// One sample per completed step, read from the field the step produced. A restart or a new run (the iteration
// going back, or another probe count) clears the history.
void SimulationViewer::RecordProbes()
{
    const size_t n = SCENE.probes.size();
    const unsigned int it = SIMULATION.telemetry.iteration;
    if (n == 0) return;
    if (probeValues.size() != n || it < probeIT)
    {
        probeTime.clear();
        probeValues.assign(n, std::vector<double>());
        probeIT = 0;
    }
    if (it == 0 || it == probeIT) return;
    probeIT = it;
    if (probeTime.size() >= SIMULATION.telemetry.max_samples)
    {
        probeTime.erase(probeTime.begin());
        for (std::vector<double> &s : probeValues) s.erase(s.begin());
    }
    probeTime.push_back(SIMULATION.telemetry.clock);
    for (size_t p = 0; p < n; p++)
        probeValues[p].push_back(EvaluateProbe(SIMULATION.GRID_SOL, &SIMULATION.particles, SCENE.probes[p]));
}

// The scene's probes now, and the history of those that read one quantity (one axis, one unit).
void SimulationViewer::RenderProbesTab()
{
    const size_t n = SCENE.probes.size();
    if (n == 0)
    {
        ImGui::TextDisabled("The scene declares no probes.");
        ImGui::TextDisabled("Add one with  probe <name> <quantity> at=(x,y,z)");
        return;
    }
    if (ImGui::BeginTable("##probes", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Probe");
        ImGui::TableSetupColumn("Quantity");
        ImGui::TableSetupColumn("Position (m)");
        ImGui::TableSetupColumn("Value");
        ImGui::TableHeadersRow();
        for (size_t p = 0; p < n; p++)
        {
            const Probe &pr = SCENE.probes[p];
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(pr.name.c_str());
            ImGui::TableNextColumn(); ImGui::TextUnformatted(pr.quantity.c_str());
            ImGui::TableNextColumn();
            if (DIMENSION == 3) ImGui::Text("%.4g, %.4g, %.4g", pr.at[0], pr.at[1], pr.at[2]);
            else ImGui::Text("%.4g, %.4g", pr.at[0], pr.at[1]);
            if (pr.radius > 0.0) { ImGui::SameLine(); ImGui::TextDisabled("r %.3g", pr.radius); }
            ImGui::TableNextColumn();
            if (p < probeValues.size() && !probeValues[p].empty()) ImGui::Text("%.6g", probeValues[p].back());
            else ImGui::TextDisabled("-");
        }
        ImGui::EndTable();
    }

    std::vector<std::string> kinds;
    for (const Probe &pr : SCENE.probes)
        if (std::find(kinds.begin(), kinds.end(), pr.quantity) == kinds.end()) kinds.push_back(pr.quantity);
    probeQuantity = (std::max)(0, (std::min)(probeQuantity, (int)kinds.size() - 1));
    ImGui::Spacing();
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::BeginCombo("History of", kinds[(size_t)probeQuantity].c_str()))
    {
        for (int k = 0; k < (int)kinds.size(); k++)
            if (ImGui::Selectable(kinds[(size_t)k].c_str(), k == probeQuantity)) probeQuantity = k;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%zu samples, since the viewer opened", probeTime.size());

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##probeHistory", ImVec2(-1, (std::max)(160.0f, avail.y - 4.0f)), TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes("Time (s)", kinds[(size_t)probeQuantity].c_str(),
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
        for (size_t p = 0; p < n && p < probeValues.size(); p++)
            if (SCENE.probes[p].quantity == kinds[(size_t)probeQuantity] && !probeTime.empty())
                ImPlot::PlotLine(SCENE.probes[p].name.c_str(), probeTime.data(), probeValues[p].data(), (int)probeTime.size());
        ImPlot::EndPlot();
    }
}

// Inside a field or particle plot, after its first item: the data area this frame, which the next fit
// (the whole domain, centred) is sized on.
void SimulationViewer::RecordPlotArea()
{
    const ImVec2 a = ImPlot::GetPlotSize();
    plotAreaW = a.x;
    plotAreaH = a.y;
}

void SimulationViewer::SetupFieldView(double ox, double oy, double w, double h, float plotW, float plotH, ImVec2 pane)
{
    // Keyed on the PANE's size: the data area and the plot width move with tick labels and the colour bar,
    // and keying on them re-framed the view in the middle of a zoom.
    const double key[7] = {ox, oy, w, h, std::floor(pane.x), std::floor(pane.y), (double)(DIMENSION * 10 + slicePlane)};
    bool changed = fieldViewReset;
    for (int n = 0; n < 7; n++) changed = changed || key[n] != fieldViewKey[n];
    if (changed)
    {
        for (int n = 0; n < 7; n++) fieldViewKey[n] = key[n];
        fieldViewReset = false;
        fitSettleFrames = 3;   // the labels of the new framing settle over a frame or two: refine the fit until then
        fitAreaW = fitAreaH = -1.0f;
    }
    else if (fitSettleFrames <= 0 || (plotAreaW == fitAreaW && plotAreaH == fitAreaH))
    {
        if (fitSettleFrames > 0) fitSettleFrames--;
        return;
    }
    else fitSettleFrames--;
    fitAreaW = plotAreaW;
    fitAreaH = plotAreaH;
    // Sized on the data area when it is known (the frame less the tick labels), the frame otherwise.
    if (plotAreaW > 0.0f && plotAreaH > 0.0f) { plotW = plotAreaW; plotH = plotAreaH; }
    // The domain plus 2.5% on each side, so its border is not drawn on the plot's edge.
    const double pad = 1.05;
    const double scale = (std::min)(plotW / (std::max)(pad * w, 1e-12), plotH / (std::max)(pad * h, 1e-12));
    const double vw = plotW / scale, vh = plotH / scale;
    const double x0 = ox - 0.5 * (vw - w), y0 = oy - 0.5 * (vh - h);
    ImPlot::SetupAxisLimits(ImAxis_X1, x0, x0 + vw, ImGuiCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, y0, y0 + vh, ImGuiCond_Always);
}

// The heatmap as an RGBA texture, t = (v - min)/(max - min) clamped, through the plot's colormap. Row 0 is
// the top (v = 0); a non-finite value is transparent.
ImTextureID SimulationViewer::FieldTexture()
{
    if (fieldTex == 0)
    {
        GLuint t = 0;
        glGenTextures(1, &t);
        fieldTex = t;
    }
    const int cmap = ImPlot::GetStyle().Colormap;
    const size_t n = (size_t)heatmapRows * heatmapCols;
    if (fieldTexVersion == heatmapVersion && fieldTexMin == minValue && fieldTexMax == maxValue &&
        fieldTexColormap == cmap)
        return (ImTextureID)fieldTex;
    if (heatmapRows <= 0 || heatmapCols <= 0 || heatmapData.size() < n)
        return (ImTextureID)fieldTex;

    unsigned char lut[256][4];
    for (int k = 0; k < 256; k++)
    {
        const ImU32 c = ImGui::ColorConvertFloat4ToU32(ImPlot::SampleColormap(k / 255.0f, cmap));
        lut[k][0] = (unsigned char)(c >> IM_COL32_R_SHIFT);
        lut[k][1] = (unsigned char)(c >> IM_COL32_G_SHIFT);
        lut[k][2] = (unsigned char)(c >> IM_COL32_B_SHIFT);
        lut[k][3] = 255;
    }
    const double lo = minValue, span = (maxValue > minValue) ? (double)maxValue - minValue : 1.0;
    fieldRGBA.resize(4 * n);
    for (size_t q = 0; q < n; q++)
    {
        const double v = heatmapData[q];
        unsigned char *px = &fieldRGBA[4 * q];
        if (!std::isfinite(v)) { px[0] = px[1] = px[2] = px[3] = 0; continue; }
        const double t = (std::min)(1.0, (std::max)(0.0, (v - lo) / span));
        const int k = (int)(t * 255.0 + 0.5);
        px[0] = lut[k][0]; px[1] = lut[k][1]; px[2] = lut[k][2]; px[3] = 255;
    }
    glBindTexture(GL_TEXTURE_2D, (GLuint)fieldTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, heatmapCols, heatmapRows, 0, GL_RGBA, GL_UNSIGNED_BYTE, fieldRGBA.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    fieldTexVersion = heatmapVersion;
    fieldTexMin = minValue;
    fieldTexMax = maxValue;
    fieldTexColormap = cmap;
    return (ImTextureID)fieldTex;
}

void SimulationViewer::FieldPlot()
{
    hoverValid = false;   // set again below while the mouse is over the plot
    {

        // Use a discrete colormap for solid mask with exactly 4 colors
        if (selectedComponent == 5)
        {
            // Create custom discrete colormap for cell types
            static bool colormapCreated = false;
            static ImPlotColormap solidColormap;
            if (!colormapCreated)
            {
                ImVec4 colors[6] = {
                    ImVec4(0.2f, 0.4f, 0.8f, 1.0f), // 0 = FLUID_CELL    (blue)
                    ImVec4(0.3f, 0.3f, 0.3f, 1.0f), // 1 = SOLID_CELL    (gray)
                    ImVec4(0.9f, 0.9f, 0.9f, 1.0f), // 2 = EMPTY_CELL    (light gray)
                    ImVec4(0.8f, 0.2f, 0.2f, 1.0f), // 3 = INFLOW_CELL   (red)
                    ImVec4(0.6f, 0.4f, 0.8f, 1.0f), // 4 = PRESSURE_CELL (purple)
                    ImVec4(0.9f, 0.6f, 0.2f, 1.0f)  // 5 = OUTFLOW_CELL  (orange)
                };
                solidColormap = ImPlot::AddColormap("SolidMask", colors, 6, false);
                colormapCreated = true;
            }
            ImPlot::PushColormap(solidColormap);
        }
        else
        {
            ImPlot::PushColormap(colormap);
        }

        // Calculate physical dimensions
        float width = 1.0f, height = 1.0f;

        if (DIMENSION == 2)
        {
            // 2D mode: always XY plane
            double ox, oy, ww, hh;
            GridExtent2D(ox, oy, ww, hh);
            width = (float)ww;
            height = (float)hh;
        }
        else
        {
            // 3D mode: the physical span of the two axes the plane puts on screen.
            double ox, oy, ww, hh;
            ViewExtent(ox, oy, ww, hh);
            width = (float)ww;
            height = (float)hh;
        }

        // The plot fills the pane, less the colour bar; the domain is framed inside it by SetupFieldView.
        const ImVec2 fitArea = ImGui::GetContentRegionAvail();
        const bool speedBar = DIMENSION == 2 && showStreamlines && streamlinesOnly && streamColour == 0;
        const ScaleFormat bar = speedBar ? MakeScaleFormat(0.0, streamMaxSpeed > 0.0 ? streamMaxSpeed : 1.0, "Speed (m/s)")
                                         : MakeScaleFormat(minValue, maxValue, "##HeatmapScale");
        const float plotWidth = (std::max)(200.0f, fitArea.x - bar.width - 2.0f * ImGui::GetStyle().ItemSpacing.x);
        const float plotHeight = (std::max)(200.0f, fitArea.y - 4.0f);
        // Pixels across the domain's long side at the fitted framing, which sizes the resampled image.
        const float fill = (std::min)(plotWidth / (std::max)(width, 1e-12f), plotHeight / (std::max)(height, 1e-12f));
        lastPlotPx = (std::max)(width, height) * fill;
        const ImPlotFlags fieldFlags = ImPlotFlags_Equal;

        // --- Quiver Plot ---
        if (selectedComponent == 0 && !quiverX.empty())
        {
            if (ImPlot::BeginPlot("##QuiverPlot", ImVec2(plotWidth, plotHeight), fieldFlags))
            {
                if (DIMENSION == 2)
                {
                    ImPlot::SetupAxes("X", "Y", 0, 0);
                    SetupFieldView(0.0, 0.0, grid2D->Nx, grid2D->Ny, plotWidth, plotHeight, fitArea);
                }
                else
                {
                    // Same extents and labels as the heatmap, so switching between the
                    // vector field and a scalar keeps the framing identical.
                    SlicePlaneInfo pi = GetSlicePlaneInfo(slicePlane, GetNx(), GetNy(), GetNz());
                    ImPlot::SetupAxes(pi.hLabel, pi.vLabel, 0, 0);
                    double qox, qoy, qww, qhh;
                    ViewExtent(qox, qoy, qww, qhh);
                    SetupFieldView(qox, qoy, qww, qhh, plotWidth, plotHeight, fitArea);
                }

                ImPlot::SetNextQuiverStyle(baseSize, ImPlot::GetColormapColor(1));
                ImPlot::PlotQuiver("Velocity",
                                   quiverX.data(), quiverY.data(),
                                   quiverU.data(), quiverV.data(),
                                   quiverX.size(),
                                   minValue, maxValue,
                                   quiverFlags);
                DrawMeshOverlay();
                if (DIMENSION == 3) TrackHover();   // the 2D vector view plots in cell indices

                RecordPlotArea();
                ImPlot::EndPlot();
            }
        }

        // --- Heatmap Plot ---
        else if (selectedComponent > 0 && !heatmapData.empty())
        {
            if (ImPlot::BeginPlot("##Heatmap", ImVec2(plotWidth, plotHeight), fieldFlags))
            {
                if (DIMENSION == 3)
                {
                    SlicePlaneInfo pi = GetSlicePlaneInfo(slicePlane, GetNx(), GetNy(), GetNz());
                    ImPlot::SetupAxes(pi.hLabel, pi.vLabel, 0, 0);
                }
                else
                {
                    ImPlot::SetupAxes(nullptr, nullptr, 0, 0);
                }
                // Limits must match the bounds the heatmap is drawn over, or the mesh
                // overlay -- which is in physical coordinates -- sits offset from it.
                double hox, hoy, hww, hhh;
                ViewExtent(hox, hoy, hww, hhh);
                SetupFieldView(hox, hoy, hww, hhh, plotWidth, plotHeight, fitArea);
                const ImPlotPoint bmin(hox, hoy);
                const ImPlotPoint bmax(hox + hww, hoy + hhh);
                const bool streamOnly = DIMENSION == 2 && showStreamlines && streamlinesOnly;
                if (!streamOnly)
                    ImPlot::PlotImage(ComponentLabel(), FieldTexture(), bmin, bmax);

                DrawMeshOverlay();
                DrawStreamlines();
                DrawProfileLine();
                TrackHover();

                RecordPlotArea();
                ImPlot::EndPlot();
            }

            ImGui::SameLine();

            if (DIMENSION == 2 && showStreamlines && streamlinesOnly && streamColour == 0)
                ColourBar("Speed (m/s)", 0.0, streamMaxSpeed > 0.0 ? streamMaxSpeed : 1.0, plotHeight);
            else
                ColourBar("##HeatmapScale", minValue, maxValue, plotHeight);
        }

        ImPlot::PopColormap();
    }
}

void SimulationViewer::RenderParticleView(bool controlsInline)
{
    if (SIMULATION.particles.count <= 0)
    {
        ImGui::TextDisabled("No particles.");
        return;
    }
    if (controlsInline) ParticleControls(false);
    SampleParticles();
    if (DIMENSION == 2) RenderParticles2D();
    else RenderParticles3D();
}

// Grid colouring needs a grid; SPH has none, so it always colours from the particle.
bool SimulationViewer::ColourFromGrid() const
{
    return particleColorFromGrid && MethodUsesGrid() && SIMULATION.GRID_SOL != nullptr;
}

// The particle view's controls: a column of labelled rows in the left pane (column), or three compact rows above
// the plot for a method without a grid, which has no left pane.
void SimulationViewer::ParticleControls(bool column)
{
    const float w = ImGui::CalcTextSize("0000000").x + 2.0f * ImGui::GetStyle().FramePadding.x;
    const float full = column ? (std::max)(120.0f, ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("velocity ").x)
                              : 0.0f;
    if (column) ImGui::SeparatorText("Particles");

    static const char *sources[] = {"fluid (grid)", "particle"};
    int source = ColourFromGrid() ? 0 : 1;
    ImGui::BeginDisabled(!MethodUsesGrid());
    ImGui::SetNextItemWidth(column ? full : ImGui::CalcTextSize("fluid (grid)").x + ImGui::GetFrameHeight() + 12.0f);
    if (ImGui::Combo("velocity", &source, sources, 2)) particleColorFromGrid = (source == 0);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Grid interpolates the solved field at each particle; particle shows the FLIP velocity it carries");
    if (!column) ImGui::SameLine();
    static const char *quantities[] = {"speed", "vertical v"};
    int quantity = particleColorBySpeed ? 0 : 1;
    ImGui::SetNextItemWidth(column ? full : ImGui::CalcTextSize("vertical v").x + ImGui::GetFrameHeight() + 12.0f);
    if (ImGui::Combo("colour", &quantity, quantities, 2)) particleColorBySpeed = (quantity == 0);
    if (!column) ImGui::SameLine();
    ImPlotColormap &map = (DIMENSION == 2) ? colormap : colormap3D;
    FieldColormapButton(map, column ? full : ImGui::CalcTextSize("Viridis").x + 24.0f);

    if (column) ImGui::SeparatorText("Sample");
    ImGui::SetNextItemWidth(column ? full : w * 1.6f);
    ImGui::SliderInt("drawn", &particleBudget, 500, 1000000, "%d", ImGuiSliderFlags_Logarithmic);
    if (!column) ImGui::SameLine();
    if (ImGui::SmallButton("resample")) particleSeed++;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Draw another random subset of the particles");
    if (!column) ImGui::SameLine();
    ImGui::SetNextItemWidth(column ? full : w);
    if (DIMENSION == 2)
        ImGui::SliderFloat("size", &particlePointScale, 0.2f, 3.0f, "%.1f");
    else
        ImGui::SliderFloat("size", &particleSize, 0.5f, 12.0f, "%.1f px");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(DIMENSION == 2 ? "Marker diameter in particle spacings, so it follows the zoom" : "Marker radius in pixels");
    if (!column) ImGui::SameLine();
    ImGui::Checkbox("auto scale", &particleAutoScale);
    if (!particleAutoScale)
    {
        if (!column) ImGui::SameLine();
        ImGui::SetNextItemWidth(column ? full : w * 2.0f);
        ImGui::DragFloatRange2("range", &particleMin, &particleMax, 0.01f, -50.0f, 50.0f, "%.3g", nullptr,
                               ImGuiSliderFlags_AlwaysClamp);
        if (particleMax <= particleMin + 1e-6f) particleMax = particleMin + 1e-6f;
    }

    if (column) ImGui::SeparatorText("Geometry");
    ImGui::Checkbox("solids", &showSolids);
    if (showSolids)
    {
        ImGui::SameLine();
        ImGui::Checkbox("domain walls", &solidsIncludeWalls);
    }
    if (!column) ImGui::SameLine();
    if (DIMENSION == 2)
    {
        if (ImGui::SmallButton("Fit view")) fieldViewReset = true;
        if (!column) ImGui::SameLine();
    }
    else
    {
        Fit3DViewButton();
        if (!column) ImGui::SameLine();
    }
    ImGui::TextDisabled("%zu of %d drawn", p3x.size(), SIMULATION.particles.count);
}

// The quantity the colour is taken from, for one particle: speed or vertical velocity, of the particle itself
// or of the solved field interpolated at its position (see particleColorFromGrid).
float SimulationViewer::ParticleColourValue(const Particle &q) const
{
    double u = q.u, v = q.v, w = (DIMENSION == 3) ? q.w : 0.0;
    if (ColourFromGrid())
    {
        MAC *g = SIMULATION.GRID_SOL;
        if (DIMENSION == 2)
        {
            u = g->ComponentAt(MAC::U, q.x, q.y);
            v = g->ComponentAt(MAC::V, q.x, q.y);
        }
        else
        {
            u = g->ComponentAt(MAC::U, q.x, q.y, q.z);
            v = g->ComponentAt(MAC::V, q.x, q.y, q.z);
            w = g->ComponentAt(MAC::W, q.x, q.y, q.z);
        }
    }
    return particleColorBySpeed ? (float)std::sqrt(u * u + v * v + w * w) : (float)v;
}

// 2D solids as filled rectangles in physical coordinates (row runs merged, through the face tables), or
// the analytic lattice without a grid. Rebuilt when the grid or the wall setting changes.
void SimulationViewer::CollectSolidRects2D()
{
    const int key = (int)solidsIncludeWalls + 2 * (MethodUsesGrid() ? 1 : 0);
    if (solidRectKey == key && !solidRects.empty()) return;
    solidRectKey = key;
    solidRects.clear();
    if (!MethodUsesGrid() || grid2D == nullptr)
    {
        if (!analyticSolidsBuilt) CollectAnalyticSolids();
        const float h = 0.5f * (float)analyticPitch;
        for (size_t n = 0; n < solidX.size(); n++)
            solidRects.push_back({solidX[n] - h, solidY[n] - h, solidX[n] + h, solidY[n] + h});
        return;
    }
    const MAC *g = grid2D;
    for (int i = 0; i < g->Ny; i++)
    {
        int j = 0;
        while (j < g->Nx)
        {
            const bool wallRow = (i == 0 || i == g->Ny - 1);
            if (g->GetSolid(i, j) != SOLID_CELL || ((wallRow || j == 0 || j == g->Nx - 1) && !solidsIncludeWalls))
            {
                j++;
                continue;
            }
            int e = j + 1;
            while (e < g->Nx && g->GetSolid(i, e) == SOLID_CELL && (solidsIncludeWalls || (!wallRow && e != g->Nx - 1)))
                e++;
            solidRects.push_back({(float)g->FacePos(MAC::X, j), (float)g->FacePos(MAC::Y, i),
                                  (float)g->FacePos(MAC::X, e), (float)g->FacePos(MAC::Y, i + 1)});
            j = e;
        }
    }
}

void SimulationViewer::RenderParticles2D()
{
    const ImPlotColormap map = colormap;
    ImPlot::PushColormap(map);

    const double x0 = SIMULATION.domain.x0, y0 = SIMULATION.domain.y0;
    const double w = SIMULATION.domain.xf - x0, h = SIMULATION.domain.yf - y0;
    const char *barLabel = particleColorBySpeed ? "Speed (m/s)" : "v (m/s)";
    const ScaleFormat bar = MakeScaleFormat(particleMin, particleMax, barLabel);
    const ImVec2 area = ImGui::GetContentRegionAvail();
    const float plotW = (std::max)(200.0f, area.x - bar.width - 2.0f * ImGui::GetStyle().ItemSpacing.x);
    const float plotH = (std::max)(200.0f, area.y - 4.0f);

    if (ImPlot::BeginPlot("##Particles2D", ImVec2(plotW, plotH), ImPlotFlags_Equal | ImPlotFlags_NoLegend))
    {
        ImPlot::SetupAxes("x (m)", "y (m)", 0, 0);
        SetupFieldView(x0, y0, w, h, plotW, plotH, area);

        // Pixels per metre at the current zoom: markers are sized in particle spacings so they tile the liquid
        // at any zoom instead of a fixed pixel size that is a dust at one zoom and a smear at another.
        const double ppm = ImPlot::PlotToPixels(1.0, 0.0).x - ImPlot::PlotToPixels(0.0, 0.0).x;
        double spacing = ActiveMethod().particleSpacing ? ActiveMethod().particleSpacing() : SIMULATION.dh;
        if (!(spacing > 0.0)) spacing = SIMULATION.dh;
        const float radius = (float)(std::max)(0.75, (std::min)(14.0, 0.5 * particlePointScale * spacing * ppm));

        ImDrawList *dl = ImPlot::GetPlotDrawList();
        ImPlot::PushPlotClipRect();
        if (showSolids)
        {
            CollectSolidRects2D();
            const ImU32 solidCol = ImGui::GetColorU32(ImGuiCol_TextDisabled, 0.55f);
            for (const SolidRect &r : solidRects)
            {
                const ImVec2 a = ImPlot::PlotToPixels(r.x0, r.y1), b = ImPlot::PlotToPixels(r.x1, r.y0);
                dl->AddRectFilled(a, b, solidCol);
            }
        }
        // The domain box, so the liquid is seen against the walls it is held by.
        dl->AddRect(ImPlot::PlotToPixels(x0, y0 + h), ImPlot::PlotToPixels(x0 + w, y0),
                    ImGui::GetColorU32(ImGuiCol_Text, 0.6f), 0.0f, 0, 1.0f);
        ImPlot::PopPlotClipRect();

        for (int b = 0; b < PARTICLE_BUCKETS; b++)
        {
            const int n0 = bucketStart[b], n = bucketStart[b + 1] - n0;
            if (n <= 0) continue;
            const ImVec4 col = ImPlot::SampleColormap((float)b / (float)(PARTICLE_BUCKETS - 1));
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, radius, col, 0.0f, col);
            ImPlot::PlotScatter("##bucket", p3x.data() + n0, p3y.data() + n0, n);
        }
        RecordPlotArea();
        ImPlot::EndPlot();
    }
    ImGui::SameLine();
    ColourBar(barLabel, particleMin, particleMax, plotH);
    ImPlot::PopColormap();
}

// A stable integer hash: keeps a particle in the sample when hash(index ^ seed) falls in the accepted fraction.
static unsigned int ParticleSampleHash(unsigned int x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// A random subset of the particles (a hash of index and seed, so it stays put between frames), coloured once
// per particle in parallel and sorted into colour buckets. Rebuilt only when an input changes.
void SimulationViewer::SampleParticles()
{
    const int total = SIMULATION.particles.count;
    const int budget = (std::max)(1, particleBudget);
    const bool fromGrid = ColourFromGrid();
    if (total == sampledTotal && frameIT == sampledIteration && budget == sampledBudget && particleSeed == sampledSeed &&
        fromGrid == sampledFromGrid && particleColorBySpeed == sampledBySpeed && particleAutoScale == sampledAuto &&
        (particleAutoScale || (particleMin == sampledMin && particleMax == sampledMax)))
        return;
    sampledTotal = total;
    sampledIteration = frameIT;
    sampledBudget = budget;
    sampledSeed = particleSeed;
    sampledFromGrid = fromGrid;
    sampledBySpeed = particleColorBySpeed;
    sampledAuto = particleAutoScale;

    p3x.clear();
    p3y.clear();
    p3z.clear();
    for (int b = 0; b <= PARTICLE_BUCKETS; b++) bucketStart[b] = 0;
    if (total <= 0)
        return;

    unsigned int threshold = 0xFFFFFFFFu;
    if (budget < total)
        threshold = (unsigned int)((double)0xFFFFFFFFu * ((double)budget / (double)total));
    const unsigned int salt = particleSeed * 0x9E3779B9u;

    sampleIdx.clear();
    for (int p = 0; p < total; p++)
    {
        if (ParticleSampleHash((unsigned int)p ^ salt) > threshold) continue;
        const Particle &q = SIMULATION.particles.data[p];
        if (std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z)) sampleIdx.push_back(p);
    }
    const int n = (int)sampleIdx.size();
    sampleVal.resize(n);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < n; m++)
        sampleVal[m] = ParticleColourValue(SIMULATION.particles.data[sampleIdx[m]]);

    if (particleAutoScale)
    {
        float lo = FLT_MAX, hi = -FLT_MAX;
        for (int m = 0; m < n; m++)
            if (std::isfinite(sampleVal[m])) { lo = (std::min)(lo, sampleVal[m]); hi = (std::max)(hi, sampleVal[m]); }
        if (lo > hi) { lo = 0.0f; hi = 1.0f; }
        // Speed is a magnitude, anchored at zero; signed vertical velocity gets a range symmetric about zero, so a
        // particle at rest always lands where rest belongs on the colour map.
        if (particleColorBySpeed) lo = 0.0f;
        else { const float mx = (std::max)(std::fabs(lo), std::fabs(hi)); lo = -mx; hi = mx; }
        if (hi - lo < 1e-9f) hi = lo + 1e-9f;
        particleMin = lo;
        particleMax = hi;
    }
    sampledMin = particleMin;
    sampledMax = particleMax;

    // Counting sort into the buckets, so each bucket is one contiguous run.
    const float span = (particleMax > particleMin) ? (particleMax - particleMin) : 1.0f;
    sampleBucket.resize(n);
    int count[PARTICLE_BUCKETS] = {};
    for (int m = 0; m < n; m++)
    {
        const float val = std::isfinite(sampleVal[m]) ? sampleVal[m] : particleMin;
        int b = (int)(((val - particleMin) / span) * (PARTICLE_BUCKETS - 1));
        b = (std::max)(0, (std::min)(PARTICLE_BUCKETS - 1, b));
        sampleBucket[m] = b;
        count[b]++;
    }
    for (int b = 0; b < PARTICLE_BUCKETS; b++) bucketStart[b + 1] = bucketStart[b] + count[b];
    int next[PARTICLE_BUCKETS];
    for (int b = 0; b < PARTICLE_BUCKETS; b++) next[b] = bucketStart[b];
    p3x.resize(n);
    p3y.resize(n);
    p3z.resize(n);
    for (int m = 0; m < n; m++)
    {
        const Particle &q = SIMULATION.particles.data[sampleIdx[m]];
        const int at = next[sampleBucket[m]]++;
        p3x[at] = (float)q.x;
        p3y[at] = (float)q.y;
        p3z[at] = (float)q.z;
    }
}

// The solids of a method without a grid, sampled from the scene's analytic surface on a lattice, once (the
// geometry is static). In 3D only the surface shell is kept: a filled obstacle hides its own far side.
void SimulationViewer::CollectAnalyticSolids()
{
    solidX.clear();
    solidY.clear();
    solidZ.clear();
    analyticSolidsBuilt = true;
    if (!SCENE.loaded)
        return;

    const double x0 = SIMULATION.domain.x0, xf = SIMULATION.domain.xf;
    const double y0 = SIMULATION.domain.y0, yf = SIMULATION.domain.yf;
    double pitch = ActiveMethod().particleSpacing ? ActiveMethod().particleSpacing() : 0.0;
    if (pitch <= 0.0) pitch = 0.01;
    // Bound the work: the sample count is display detail, not physics.
    const int maxSide = 320;
    if ((xf - x0) / pitch > maxSide) pitch = (xf - x0) / maxSide;
    if ((yf - y0) / pitch > maxSide) pitch = (yf - y0) / maxSide;

    const double z0 = SIMULATION.domain.z0, zf = SIMULATION.domain.zf;
    if (DIMENSION == 3 && (zf - z0) / pitch > maxSide) pitch = (zf - z0) / maxSide;
    analyticPitch = pitch;
    const double zStart = (DIMENSION == 3) ? z0 + 0.5 * pitch : 0.0;
    const double zEnd   = (DIMENSION == 3) ? zf : 0.5 * pitch;

    for (double z = zStart; z < zEnd; z += pitch)
    for (double y = y0 + 0.5 * pitch; y < yf; y += pitch)
    for (double x = x0 + 0.5 * pitch; x < xf; x += pitch)
    {
        double phi = 0.0, n[3];
        if (SCENE.NearestSolidSurface(x, y, z, phi, n) && phi < 0.0)
        {
            // In 3D only the SURFACE shell is kept: a filled solid hides its own far side
            // and everything behind it, exactly as CollectSolidCells does for FLIP.
            if (DIMENSION == 3 && phi < -pitch) continue;
            solidX.push_back((float)x);
            solidY.push_back((float)y);
            solidZ.push_back((float)z);
        }
    }
}

// Whether (i, j, k) is solid; outside the grid counts as solid, so the outward face of a wall cell does not read
// as exposed and the whole domain shell does not qualify as surface.
static bool SolidOrOutside(const MAC *g, int i, int j, int k)
{
    if (i < 0 || i >= g->Ny || j < 0 || j >= g->Nx) return true;
    if (DIMENSION == 2) return k != 0 || g->GetSolid(i, j) == SOLID_CELL;
    if (k < 0 || k >= g->Nz) return true;
    return g->GetSolid(i, j, k) == SOLID_CELL;
}

// The exposed solid cells, at their centres from the face tables. Collected once per grid and wall setting: the
// solid mask is static (FLIP re-derives only the fluid/empty split).
void SimulationViewer::CollectSolidCells()
{
    // A method without a grid collides with the analytic surface, so that is what is drawn; it is static.
    if (!MethodUsesGrid())
    {
        if (!analyticSolidsBuilt) CollectAnalyticSolids();
        return;
    }
    const MAC *g = (DIMENSION == 2) ? grid2D : grid3D;
    if (g == nullptr) return;
    if (solidCellsGrid == g && solidCellsWalls == (int)solidsIncludeWalls) return;
    solidCellsGrid = g;
    solidCellsWalls = (int)solidsIncludeWalls;
    solidX.clear();
    solidY.clear();
    solidZ.clear();

    const int Nx = g->Nx, Ny = g->Ny, Nz = (DIMENSION == 3) ? g->Nz : 1;
    for (int k = 0; k < Nz; k++)
        for (int i = 0; i < Ny; i++)
            for (int j = 0; j < Nx; j++)
            {
                if (!SolidOrOutside(g, i, j, k)) continue;
                const bool onWall = (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1 ||
                                     (DIMENSION == 3 && (k == 0 || k == Nz - 1)));
                if (onWall && !solidsIncludeWalls) continue;
                const bool exposed = !SolidOrOutside(g, i + 1, j, k) || !SolidOrOutside(g, i - 1, j, k) ||
                                     !SolidOrOutside(g, i, j + 1, k) || !SolidOrOutside(g, i, j - 1, k) ||
                                     (DIMENSION == 3 && (!SolidOrOutside(g, i, j, k + 1) || !SolidOrOutside(g, i, j, k - 1)));
                if (!exposed) continue;
                solidX.push_back((float)g->CentrePos(MAC::X, j));
                solidY.push_back((float)g->CentrePos(MAC::Y, i));
                solidZ.push_back((float)(DIMENSION == 3 ? g->CentrePos(MAC::Z, k) : 0.0));
            }
}

void SimulationViewer::RenderParticles3D()
{
    // The plot fills the pane, less the colour bar.
    const char *barLabel = particleColorBySpeed ? "Speed (m/s)" : "V (m/s)";
    const ScaleFormat bar = MakeScaleFormat(particleMin, particleMax, barLabel);
    const ImVec2 size = Plot3DSize(bar.width);

    const double Lx = SIMULATION.domain.xf - SIMULATION.domain.x0;
    const double Ly = SIMULATION.domain.yf - SIMULATION.domain.y0;
    const double Lz = SIMULATION.domain.zf - SIMULATION.domain.z0;
    const double Lmax = (std::max)(Lx, (std::max)(Ly, Lz));

    ImPlot3D::PushColormap(colormap3D);
    ImGui::BeginGroup();

    if (Begin3DView("##Particles3D", size))
    {
        // ImPlot3D draws its third axis vertically, so the simulation's Y (gravity) goes third and Z second;
        // data, labels, limits and box scale all follow that order.
        ImPlot3D::SetupAxes("X (m)", "Z (m)", "Y (m)");
        ImPlot3D::SetupAxesLimits(SIMULATION.domain.x0, SIMULATION.domain.xf,
                                  SIMULATION.domain.z0, SIMULATION.domain.zf,
                                  SIMULATION.domain.y0, SIMULATION.domain.yf,
                                  ImPlot3DCond_Always);
        // Keep the box proportional to the real domain instead of forcing a cube.
        if (Lmax > 0.0)
            ImPlot3D::SetupBoxScale(Lx / Lmax, Lz / Lmax, Ly / Lmax);

        if (showSolids)
        {
            CollectSolidCells();
            if (!solidX.empty())
            {
                const ImVec4 solidCol(0.55f, 0.55f, 0.58f, 0.85f);
                ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Square, solidSize, solidCol,
                                             0.0f, solidCol);
                // Drawn first so particles land on top of the geometry rather than behind
                // it. Same (X, Z, Y) ordering as the particles: Y is the vertical axis.
                ImPlot3D::PlotScatter("##solids", solidX.data(), solidZ.data(),
                                      solidY.data(), (int)solidX.size());
            }
        }

        for (int b = 0; b < PARTICLE_BUCKETS; b++)
        {
            const int n0 = bucketStart[b], n = bucketStart[b + 1] - n0;
            if (n <= 0) continue;
            ImVec4 col = ImPlot3D::SampleColormap((float)b / (float)(PARTICLE_BUCKETS - 1), colormap3D);
            ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Circle, particleSize, col, 0.0f, col);
            // One call per bucket, on its run of the sorted sample. z before y: the plot wants (X, Z, Y) so that Y
            // stays vertical.
            ImPlot3D::PlotScatter("##bucket", p3x.data() + n0, p3z.data() + n0, p3y.data() + n0, n);
        }

        End3DView();
    }

    ImGui::EndGroup();
    ImGui::SameLine();

    ImPlot::PushColormap(colormap3D);
    ColourBar(barLabel, particleMin, particleMax, size.y);
    ImPlot::PopColormap();
    ImPlot3D::PopColormap();

}

void SimulationViewer::RenderControlUI()
{
    const int IT = (int)SIMULATION.telemetry.iteration;
    const double time = SIMULATION.telemetry.clock;
    const double residual = SIMULATION.telemetry.lastResidual;
    const double frameTime = SIMULATION.telemetry.lastStepWallTime;
    bool &simulationRunning = SIMULATION.running;
    bool &stepOnce = SIMULATION.stepOnce;

    ImGui::Text("Iteration: %d", IT);
    ImGui::Text("Time: %.4f", time);
    ImGui::Text("Residual: %.8f", residual);
    ImGui::Text("Timestep: %.8f", SIMULATION.dt);

    ImGui::Text("Divergence: %.6e", SIMULATION.GRID_SOL->GetDivSum());

    // Upwind's numerical viscosity is |u| h / 2, largest where the cell Reynolds number is.
    if (SIMULATION.telemetry.lastCellReynolds.valid)
    {
        ImGui::Text("Max cell Re: %.3g", SIMULATION.telemetry.lastCellReynolds.value);
    }

    ImGui::Text("Frame Time: %.8f ms", frameTime * 1000.0);
    ImGui::Text("Draw Time: %.8f ms", drawTime * 1000.0);

    if (simulationRunning)
    {
        ImGui::Text("Status: Running");
    }
    else
    {
        ImGui::Text("Status: Paused");
    }

    ImGui::Separator();

    if (ImGui::Button(simulationRunning ? "Pause" : "Run"))
    {
        simulationRunning = !simulationRunning;
        // Running again after a stop means the user wants to carry on past it, so the rule
        // is re-armed rather than firing again on the very next step.
        if (simulationRunning && SIMULATION.stop.triggered)
            SIMULATION.stop.Rearm();
    }
    ImGui::SameLine();
    if (ImGui::Button("Step"))
    {
        stepOnce = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Quit"))
    {
        QUIT_REQUESTED = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Restart"))
        RESTART_REQUESTED = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("End this run and go back to the setup screen");

    // ---- checkpoint: ONE file, overwritten. The buttons only raise a flag; the manager saves or loads
    // between frames, where no solver is mid-sweep.
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Checkpoint", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Auto-save every", &SIMULATION.checkpoint.autoSave);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90);
        if (ImGui::InputInt("iterations", &SIMULATION.checkpoint.interval, 0, 0) && SIMULATION.checkpoint.interval < 1)
            SIMULATION.checkpoint.interval = 1;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The whole state -- both grids, the masks, the solvers' own step\n"
                              "state and the clock -- written to one file and overwritten each\n"
                              "time. It is not an export: only the newest is of any use, so it\n"
                              "does not accumulate and does not go to the export directory.\n\n"
                              "Written through a .tmp and renamed, so a crash mid-write cannot\n"
                              "destroy the previous checkpoint.");

        ImGui::SetNextItemWidth(260);
        ImGui::InputText("File", SIMULATION.checkpoint.path, IM_ARRAYSIZE(SIMULATION.checkpoint.path));

        if (ImGui::Button("Save now"))
            SIMULATION.checkpoint.saveRequested = true;
        ImGui::SameLine();
        if (ImGui::Button("Load"))
            SIMULATION.checkpoint.loadRequested = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Restores into the grids this run already built, so the config and\n"
                              "the scene must be the ones the checkpoint was written with. A file\n"
                              "from another grid size, dimension or solver is refused.");

        if (SIMULATION.checkpoint.autoSave && SIMULATION.checkpoint.interval > 0)
            ImGui::TextDisabled("next at step %llu",
                                (unsigned long long)((SIMULATION.checkpoint.it / SIMULATION.checkpoint.interval + 1) *
                                                     SIMULATION.checkpoint.interval));
        if (!SIMULATION.checkpoint.status.empty())
            ImGui::TextColored(SIMULATION.checkpoint.statusIsError ? ToneColour(UiTone::Error)
                                                        : ToneColour(UiTone::Good),
                               "%s", SIMULATION.checkpoint.status.c_str());
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Closes the window through the normal shutdown path.");

    ImGui::Separator();
    ImGui::TextUnformatted("Stop automatically");

    static const char *stopNames[] = {"never (run until paused)",
                                      "at a final time",
                                      "when the residual settles"};
    ImGui::SetNextItemWidth(240);
    ImGui::Combo("##stopcond", &SIMULATION.stop.condition, stopNames, 3);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Final time stops the clock at a chosen t.\n\n"
            "Residual stops once the step-to-step change falls below a tolerance, i.e. the\n"
            "flow has stopped evolving. That only means anything for a case that HAS a steady\n"
            "state -- a shedding wake never settles, so it would run forever.");

    if (SIMULATION.stop.condition == STOP_TIME)
    {
        ImGui::SetNextItemWidth(160);
        ImGui::InputDouble("Final time t", &SIMULATION.stop.tFinal, 0.0, 0.0, "%.4f");
        if (SIMULATION.stop.tFinal < 0.0) SIMULATION.stop.tFinal = 0.0;
        ImGui::SameLine();
        ImGui::TextDisabled("now t = %.4f", time);
    }
    else if (SIMULATION.stop.condition == STOP_RESIDUAL)
    {
        ImGui::SetNextItemWidth(160);
        ImGui::InputDouble("Residual <=", &SIMULATION.stop.residualTol, 0.0, 0.0, "%.3e");
        if (SIMULATION.stop.residualTol < 0.0) SIMULATION.stop.residualTol = 0.0;
        ImGui::SameLine();
        ImGui::TextDisabled("now %.3e", residual);
    }

    if (SIMULATION.stop.condition != STOP_NEVER)
    {
        ImGui::Checkbox("Export the final field and telemetry", &SIMULATION.stop.exportOnStop);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Writes the grid, the telemetry and (with particles) the particles once,\n"
                              "when the run stops -- regardless of the export intervals.");
    }

    if (SIMULATION.stop.triggered)
    {
        ImGui::TextColored(ToneColour(UiTone::Good), "Stopped: %s",
                           SIMULATION.stop.reason.c_str());
        if (SIMULATION.stop.exported)
            ImGui::TextDisabled("Final field and telemetry written.");
    }

}

// Divergence, CFL, residual and the performance bars.
void SimulationViewer::RenderSolverTelemetryTab()
{
    const double t = SIMULATION.telemetry.time.back();
    const double window = 5.0;
    float avail = ImGui::GetContentRegionAvail().x;
    float plot_w = avail * 0.5f - ImGui::GetStyle().ItemSpacing.x * 0.5f;
    ImVec2 plot_size(plot_w, 300);

    // ---- Row 1 ----
    if (ImPlot::BeginPlot("Divergence Evolution", plot_size, TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes("Time", "Divergence Sum",
                          ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t - window * 0.5,
            t + window * 0.5,
            ImGuiCond_Always);
        ImPlot::PlotLine("Divergence",
                         SIMULATION.telemetry.time.data(), SIMULATION.telemetry.div_sum.data(), SIMULATION.telemetry.time.size());
        ImPlot::EndPlot();
    }

    ImGui::SameLine();

    if (ImPlot::BeginPlot("CFL and cell Reynolds", plot_size, TELEMETRY_PLOT_FLAGS))
    {
        // Cell Re = |u| h / nu on the right axis: central convection stops damping above ~2.
        ImPlot::SetupAxes("Time", "CFL",
                          ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxis(ImAxis_Y2, "Cell Re", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t - window * 0.5,
            t + window * 0.5,
            ImGuiCond_Always);

        ImPlot::PlotLine("Adv. CFL",
                         SIMULATION.telemetry.time.data(), SIMULATION.telemetry.advcfl.data(), SIMULATION.telemetry.time.size());
        ImPlot::PlotLine("Diff. CFL",
                         SIMULATION.telemetry.time.data(), SIMULATION.telemetry.diffcfl.data(), SIMULATION.telemetry.time.size());
        double cflLimit = 1.0;
        ImPlot::SetNextLineStyle(ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), 1.0f);
        ImPlot::PlotInfLines("CFL = 1", &cflLimit, 1, ImPlotInfLinesFlags_Horizontal);

        const CellReynolds &re = SIMULATION.telemetry.lastCellReynolds;
        const int reCount = (int)std::min(SIMULATION.telemetry.time.size(), SIMULATION.telemetry.cell_reynolds.size());
        if (re.valid && reCount > 0)
        {
            static const char *reComponent[3] = {"u", "v", "w"};
            char reLabel[160];
            if (DIMENSION == 3)
                std::snprintf(reLabel, sizeof(reLabel), "Cell Re max %.3g (%s at %.3g, %.3g, %.3g)###cellre",
                              re.value, re.component >= 0 ? reComponent[re.component] : "-", re.x, re.y, re.z);
            else
                std::snprintf(reLabel, sizeof(reLabel), "Cell Re max %.3g (%s at %.3g, %.3g)###cellre",
                              re.value, re.component >= 0 ? reComponent[re.component] : "-", re.x, re.y);
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine(reLabel, SIMULATION.telemetry.time.data() + (SIMULATION.telemetry.time.size() - reCount),
                             SIMULATION.telemetry.cell_reynolds.data() + (SIMULATION.telemetry.cell_reynolds.size() - reCount), reCount);
            double reLimit = 2.0;
            ImPlot::SetNextLineStyle(ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), 1.0f);
            ImPlot::PlotInfLines("Cell Re = 2", &reLimit, 1, ImPlotInfLinesFlags_Horizontal);
        }
        ImPlot::EndPlot();
    }

    // ---- Row 2 ----
    if (ImPlot::BeginPlot("Grid Residual", plot_size, TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes(
            "Time", "Residual",
            ImPlotAxisFlags_None,
            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);

        ImPlot::SetupAxisLimits(
            ImAxis_X1,
            t - window * 0.5,
            t + window * 0.5,
            ImGuiCond_Always);

        ImPlot::PlotLine(
            "Iteration Residual",
            SIMULATION.telemetry.time.data(),
            SIMULATION.telemetry.residual.data(),
            SIMULATION.telemetry.time.size());

        ImPlot::EndPlot();
    }

    ImGui::SameLine();

    // Step timings: one horizontal bar per registered step, so the step names read along the axis however many
    // a method registers (FLIP has six). Milliseconds of the last step, with the share of the whole step.
    if (ImPlot::BeginPlot("Step timings", plot_size, TELEMETRY_PLOT_FLAGS | ImPlotFlags_NoLegend))
    {
        const int n = (int)SIMULATION.telemetry.stepNames.size();
        static std::vector<double> ys, values;
        static std::vector<const char *> labels;
        ys.resize(n);
        values.resize(n);
        labels.resize(n);
        double total = 0.0, vmax = 0.0;
        for (int k = 0; k < n; k++)
        {
            ys[k] = (double)(n - 1 - k);   // registration order, top to bottom
            labels[k] = SIMULATION.telemetry.stepNames[k].c_str();
            values[k] = SIMULATION.telemetry.stepTimings[k].empty() ? 0.0
                                                                    : 1e3 * SIMULATION.telemetry.stepTimings[k].back();
            total += values[k];
            vmax = (std::max)(vmax, values[k]);
        }
        // The value written past each bar, and the headroom it needs: the widest label in pixels, as a share of the
        // plotting area (the plot less the step names on the left and the padding).
        static std::vector<std::string> texts;
        texts.resize(n);
        float textW = 0.0f, nameW = 0.0f;
        for (int k = 0; k < n; k++)
        {
            char buffer[48];
            if (total > 0.0)
                std::snprintf(buffer, sizeof(buffer), "%.3g ms  %.0f%%", values[k], 100.0 * values[k] / total);
            else
                std::snprintf(buffer, sizeof(buffer), "%.3g ms", values[k]);
            texts[k] = buffer;
            textW = (std::max)(textW, ImGui::CalcTextSize(buffer).x);
            nameW = (std::max)(nameW, ImGui::CalcTextSize(labels[k]).x);
        }
        const float area = plot_size.x - nameW - 4.0f * ImPlot::GetStyle().PlotPadding.x;
        const double share = area > 1.0f ? (textW + 14.0f) / area : 0.5;
        const double xMax = (vmax > 0.0 ? vmax : 1.0) / (std::max)(0.25, 1.0 - share);
        ImPlot::SetupAxes("ms", nullptr, ImPlotAxisFlags_None, ImPlotAxisFlags_NoGridLines);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, xMax, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -0.6, (std::max)(0, n - 1) + 0.6, ImGuiCond_Always);
        if (n > 0)
        {
            static std::vector<double> ticks;
            static std::vector<const char *> tickLabels;
            ticks.assign(ys.rbegin(), ys.rend());
            tickLabels.assign(labels.rbegin(), labels.rend());
            ImPlot::SetupAxisTicks(ImAxis_Y1, ticks.data(), n, tickLabels.data());
        }
        const double barHeight = n > 8 ? 0.8 : 0.6;
        for (int k = 0; k < n; k++)
        {
            ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(k));
            ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(k));
            ImPlot::PlotBars(labels[k], &values[k], &ys[k], 1, barHeight, ImPlotBarsFlags_Horizontal);
            // PlotText centres on the point; shift by half the label so it starts just past the bar.
            const char *buffer = texts[k].c_str();
            ImPlot::PlotText(buffer, values[k], ys[k], ImVec2(0.5f * ImGui::CalcTextSize(buffer).x + 6.0f, 0.0f));
        }
        ImPlot::EndPlot();
    }
}

// Kinetic energy and enstrophy, volume averaged per unit mass.
void SimulationViewer::RenderFlowIntegralsTab()
{
    const FlowIntegralTelemetry &fi = SIMULATION.flowIntegrals;
    if (!fi.available || fi.time.empty())
    {
        if (!SIMULATION.exports.enabled || !SIMULATION.exports.exportFlowIntegrals)
            ImGui::TextDisabled("Not recorded: tick \"Enable Export\" and \"Flow integrals\" in the export\n"
                                "panel. They are skipped otherwise, to save a pass over the grid per step.\n"
                                "Recording starts from the step they are enabled.");
        else
            ImGui::TextDisabled("No samples yet -- the flow integrals are recorded once per step on the\n"
                                "methods with a grid.");
        return;
    }

    const size_t last = fi.time.size() - 1;
    const double E = fi.kineticEnergy[last];
    ImGui::Text("Kinetic energy  E = %.6e m^2/s^2   (total %.6e %s)", E, E * fi.density * fi.fluidVolume,
                DIMENSION == 2 ? "J/m" : "J");
    ImGui::Text("Enstrophy       Z = %.6e 1/s^2", fi.enstrophy[last]);
    if (std::isfinite(fi.energyRate[last])) ImGui::Text("dE/dt           = %.6e m^2/s^3", fi.energyRate[last]);
    if (fi.kineticEnergy.front() > 0.0)
        ImGui::Text("E / E(0) = %.6f", E / fi.kineticEnergy.front());
    ImGui::Separator();

    const int n = (int)fi.time.size();
    // The same framing as the residual and divergence plots: the last 5 s, the y axis fitted to what is in view.
    const double t = fi.time.back(), window = 5.0;
    const float avail = ImGui::GetContentRegionAvail().x;
    const float plotW = avail * 0.5f - ImGui::GetStyle().ItemSpacing.x * 0.5f;
    const ImVec2 plotSize(plotW, 300);

    if (ImPlot::BeginPlot("Kinetic energy", plotSize, TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes("Time", "E (m^2/s^2)", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, t - window * 0.5, t + window * 0.5, ImGuiCond_Always);
        ImPlot::PlotLine("E", fi.time.data(), fi.kineticEnergy.data(), n);
        ImPlot::EndPlot();
    }
    ImGui::SameLine();
    if (ImPlot::BeginPlot("Enstrophy", plotSize, TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes("Time", "Z (1/s^2)", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, t - window * 0.5, t + window * 0.5, ImGuiCond_Always);
        ImPlot::PlotLine("Z", fi.time.data(), fi.enstrophy.data(), n);
        ImPlot::EndPlot();
    }
    // The first two samples carry no rate (NaN): the line starts at the second step.
    if (ImPlot::BeginPlot("dE/dt", ImVec2(avail, 300), TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes("Time", "dE/dt (m^2/s^3)", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, t - window * 0.5, t + window * 0.5, ImGuiCond_Always);
        ImPlot::PlotLine("dE/dt", fi.time.data(), fi.energyRate.data(), n);
        ImPlot::EndPlot();
    }
}

// The control-volume coefficient beside the surface one, in the second series colour. It is pushed with every
// aerodynamics sample, so it shares the time axis; an empty series (no box fits) draws nothing.
static void PlotControlVolume(const std::vector<double> &cv)
{
    const std::vector<double> &t = SIMULATION.aero.time;
    if (cv.empty() || cv.size() != t.size()) return;
    ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(1));
    ImPlot::PlotLine("control volume", t.data(), cv.data(), (int)cv.size());
    ImPlot::PopStyleColor();
}

// One statistics row: current value, mean and standard deviation over the last 100 samples.
static void StatsRow(const char *name, const std::vector<double> &data)
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(name);
    if (data.empty())
    {
        for (int c = 0; c < 3; c++) { ImGui::TableNextColumn(); ImGui::TextUnformatted("-"); }
        return;
    }
    const int start = (std::max)(0, (int)data.size() - 100);
    double mean = 0.0, sq = 0.0;
    for (int i = start; i < (int)data.size(); i++) mean += data[i];
    mean /= (double)(data.size() - start);
    for (int i = start; i < (int)data.size(); i++) sq += (data[i] - mean) * (data[i] - mean);
    ImGui::TableNextColumn(); ImGui::Text("%.6f", data.back());
    ImGui::TableNextColumn(); ImGui::Text("%.6f", mean);
    ImGui::TableNextColumn(); ImGui::Text("%.6f", std::sqrt(sq / (double)(data.size() - start)));
}

void SimulationViewer::RenderAerodynamicsUI()
{
    if (!SIMULATION.exports.aerodynamics)
    {
        return;
    }

    // Guarded: this panel renders every frame, including before the first step has pushed
    // anything, and .back() on an empty vector is undefined behaviour.
    const double t = SIMULATION.telemetry.time.empty() ? 0.0 : SIMULATION.telemetry.time.back();
    const double window = 5.0;

    if (DIMENSION == 3)
        ImGui::TextDisabled("Coefficients divide by S = L1 x L2 = %.6g m^2",
                            SIMULATION.CHARACTERISTIC_LENGTH * SIMULATION.CHARACTERISTIC_LENGTH_2);

    float avail = ImGui::GetContentRegionAvail().x;
    float plot_w = avail * 0.5f - ImGui::GetStyle().ItemSpacing.x * 0.5f;
    ImVec2 plot_size(plot_w, 300);

    // ---- Row 1: Lift and Drag Coefficients ----
    if (ImPlot::BeginPlot("Lift Coefficient (Cl)", plot_size, TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes("Time", "Cl",
                          ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit);

        if (!SIMULATION.aero.Cl.empty())
        {
            ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(0));

            ImPlot::SetupAxisLimits(
                ImAxis_X1,
                t - window * 0.5,
                t + window * 0.5,
                ImGuiCond_Always);
            ImPlot::PlotLine("surface",
                             SIMULATION.aero.time.data(),
                             SIMULATION.aero.Cl.data(),
                             SIMULATION.aero.Cl.size());
            ImPlot::PopStyleColor();
            PlotControlVolume(SIMULATION.aero.ClCV);
        }
        ImPlot::EndPlot();
    }

    ImGui::SameLine();

    if (ImPlot::BeginPlot("Drag Coefficient (Cd)", plot_size, TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes("Time", "Cd",
                          ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit);

        if (!SIMULATION.aero.Cd.empty())
        {
            ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(0));
            ImPlot::SetupAxisLimits(
                ImAxis_X1,
                t - window * 0.5,
                t + window * 0.5,
                ImGuiCond_Always);

            ImPlot::PlotLine("surface",
                             SIMULATION.aero.time.data(),
                             SIMULATION.aero.Cd.data(),
                             SIMULATION.aero.Cd.size());
            ImPlot::PopStyleColor();
            PlotControlVolume(SIMULATION.aero.CdCV);
        }
        ImPlot::EndPlot();
    }

    // ---- Row 2: Lift/Drag Ratio ----
    if (ImPlot::BeginPlot("Lift-to-Drag Ratio", plot_size, TELEMETRY_PLOT_FLAGS))
    {
        ImPlot::SetupAxes("Time", "Cl/Cd",
                          ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit);

        if (!SIMULATION.aero.Cl.empty() && !SIMULATION.aero.Cd.empty())
        {

            ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(2));
            ImPlot::SetupAxisLimits(
                ImAxis_X1,
                t - window * 0.5,
                t + window * 0.5,
                ImGuiCond_Always);
            ImPlot::PlotLine("L/D",
                             SIMULATION.aero.time.data(),
                             SIMULATION.aero.LtoDRatio.data(),
                             SIMULATION.aero.LtoDRatio.size());
            ImPlot::PopStyleColor();
        }
        ImPlot::EndPlot();
    }

    // ---- Row 3: Statistics Table ----
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Current Aerodynamic Data:");
    ImGui::Spacing();
    if (ImGui::BeginTable("AeroStatsTable", 4,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Parameter");
        ImGui::TableSetupColumn("Current");
        ImGui::TableSetupColumn("Mean (last 100)");
        ImGui::TableSetupColumn("Std Dev (last 100)");
        ImGui::TableHeadersRow();

        auto compute_stats = [](const std::vector<double> &data, int n_samples)
        {
            if (data.empty())
                return std::make_pair(0.0, 0.0);

            int start = (std::max)(0, (int)data.size() - n_samples);
            int count = data.size() - start;

            double sum = 0.0;
            for (int i = start; i < (int)data.size(); i++)
            {
                sum += data[i];
            }
            double mean = sum / count;

            double sq_sum = 0.0;
            for (int i = start; i < (int)data.size(); i++)
            {
                double diff = data[i] - mean;
                sq_sum += diff * diff;
            }
            double std_dev = std::sqrt(sq_sum / count);

            return std::make_pair(mean, std_dev);
        };

        // Lift coefficient row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Cl");
        ImGui::TableNextColumn();
        if (!SIMULATION.aero.Cl.empty())
        {
            ImGui::Text("%.6f", SIMULATION.aero.Cl.back());
            ImGui::TableNextColumn();
            auto stats = compute_stats(SIMULATION.aero.Cl, 100);
            ImGui::Text("%.6f", stats.first);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", stats.second);
        }
        else
        {
            ImGui::TableNextColumn();
            ImGui::Text("-");
            ImGui::TableNextColumn();
            ImGui::Text("-");
            ImGui::TableNextColumn();
            ImGui::Text("-");
        }

        // Drag coefficient row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Cd");
        ImGui::TableNextColumn();
        if (!SIMULATION.aero.Cd.empty())
        {
            ImGui::Text("%.6f", SIMULATION.aero.Cd.back());
            ImGui::TableNextColumn();
            auto stats = compute_stats(SIMULATION.aero.Cd, 100);
            ImGui::Text("%.6f", stats.first);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", stats.second);
        }
        else
        {
            ImGui::TableNextColumn();
            ImGui::Text("-");
            ImGui::TableNextColumn();
            ImGui::Text("-");
            ImGui::TableNextColumn();
            ImGui::Text("-");
        }

        StatsRow("Cl (control volume)", SIMULATION.aero.ClCV);
        StatsRow("Cd (control volume)", SIMULATION.aero.CdCV);

        // Side force row -- 3D only; the 2D path never fills it.
        if (!SIMULATION.aero.Cs.empty())
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Cs (side)");
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", SIMULATION.aero.Cs.back());
            ImGui::TableNextColumn();
            auto stats = compute_stats(SIMULATION.aero.Cs, 100);
            ImGui::Text("%.6f", stats.first);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", stats.second);
        }

        // L/D ratio row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("L/D");
        ImGui::TableNextColumn();
        if (!SIMULATION.aero.Cl.empty() && !SIMULATION.aero.Cd.empty())
        {
            double cd = SIMULATION.aero.Cd.back();
            double ld = (std::abs(cd) > 1e-10) ? SIMULATION.aero.Cl.back() / cd : 0.0;
            ImGui::Text("%.6f", ld);

            // Compute stats for L/D
            std::vector<double> ld_ratio;
            int n_samples = (std::min)(100, (int)SIMULATION.aero.Cl.size());
            int start = SIMULATION.aero.Cl.size() - n_samples;

            for (int i = start; i < (int)SIMULATION.aero.Cl.size(); i++)
            {
                double cd_val = SIMULATION.aero.Cd[i];
                if (std::abs(cd_val) > 1e-10)
                {
                    ld_ratio.push_back(SIMULATION.aero.Cl[i] / cd_val);
                }
            }

            if (!ld_ratio.empty())
            {
                auto stats = compute_stats(ld_ratio, ld_ratio.size());
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", stats.first);
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", stats.second);
            }
            else
            {
                ImGui::TableNextColumn();
                ImGui::Text("-");
                ImGui::TableNextColumn();
                ImGui::Text("-");
            }
        }
        else
        {
            ImGui::TableNextColumn();
            ImGui::Text("-");
            ImGui::TableNextColumn();
            ImGui::Text("-");
            ImGui::TableNextColumn();
            ImGui::Text("-");
        }

        ImGui::EndTable();
    }
}

void SimulationViewer::RenderExportUI()
{

    ImGui::Checkbox("Enable Export", &SIMULATION.exports.enabled);

    if (SIMULATION.exports.enabled)
    {
        ImGui::Separator();
        ImGui::Text("Export Intervals");
        ImGui::SliderInt("Telemetry (Every N Frames)", &SIMULATION.exports.telemetryInterval, 1, 100);

        // The frame exports -- grid, particles, surface mesh -- share one trigger, because
        // they are one frame written three ways.
        int frameMode = SIMULATION.exports.gridEveryTime ? 1 : 0;
        ImGui::SetNextItemWidth(220);
        if (ImGui::Combo("Frames", &frameMode, "Every N iterations\0Every X seconds\0"))
        {
            SIMULATION.exports.gridEveryTime = (frameMode == 1);
            AdvanceFrameExportClock(SIMULATION.time, SIMULATION.exports);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("With ADAPTATIVE_TIMESTEP the step size moves with the flow, so\n"
                              "\"every N iterations\" spaces frames unevenly in SIMULATED TIME\n"
                              "and an animation plays back at a varying speed. \"Every X\n"
                              "seconds\" writes a frame whenever the clock crosses the next\n"
                              "multiple of the interval, which is a constant frame rate.");

        if (SIMULATION.exports.gridEveryTime)
        {
            ImGui::SetNextItemWidth(220);
            if (ImGui::InputDouble("Frame interval (s)", &SIMULATION.exports.gridTimeInterval, 0.0, 0.0, "%.4f"))
            {
                if (SIMULATION.exports.gridTimeInterval < 1e-6) SIMULATION.exports.gridTimeInterval = 1e-6;
                AdvanceFrameExportClock(SIMULATION.time, SIMULATION.exports);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("= %.1f fps", 1.0 / SIMULATION.exports.gridTimeInterval);
            ImGui::TextDisabled("next frame at t = %.4f s", SIMULATION.exports.nextGridExportTime);
        }
        else
        {
            ImGui::SliderInt("Grid (Every N Frames)", &SIMULATION.exports.gridInterval, 1, 500);
        }

        ImGui::Separator();
        ImGui::Text("Grid Export");
        if (MethodUsesGrid())
        {
            ImGui::Checkbox("Export Grid (VTK)", &SIMULATION.exports.exportGridData);

            // The scalars alone, in .../Volumetric/, for a renderer. Only a family that transports
            // them can fill them.
            const bool canVolumes = MethodTransportsScalars() &&
                                    SIMULATION.GRID_SOL && SIMULATION.GRID_SOL->ScalarCount() > 0;
            if (!canVolumes) SIMULATION.exports.exportVolumetric = false;
            ImGui::BeginDisabled(!canVolumes);
            ImGui::Checkbox("Export Volumes (NPY)", &SIMULATION.exports.exportVolumetric);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip(
                    canVolumes
                        ? "Every registered scalar as <name>_NNNNN.npy (float32, one value per cell,\n"
                          "array[row y, column x] in 2D, [z, y, x] in 3D, ring included) plus frame_NNNNN.json with the time,\n"
                          "shape and cell coordinates, into .../Volumetric/. np.load and json.load are\n"
                          "all an import script needs. The .vtr keeps carrying the scalars as well."
                        : "Needs a method that transports scalars, and a scene that declares one.");
            if (SIMULATION.exports.exportVolumetric && !SIMULATION.exports.volumetricStatus.empty())
                ImGui::TextDisabled("%s", SIMULATION.exports.volumetricStatus.c_str());
        }
        if (MethodUsesParticles())
        {
            ImGui::Checkbox("Export Particles (CSV)", &SIMULATION.exports.exportParticleData);

            // The liquid as a watertight mesh, for rendering. Written on the same interval as
            // the grid and particle exports, into .../Surface/ beside them.
            const bool canSurface = (DIMENSION == 3) && MethodUsesParticles();
            ImGui::BeginDisabled(!canSurface);
            ImGui::Checkbox("Export Surface Mesh (PLY)", &SIMULATION.exports.exportSurfaceMesh);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip(
                    canSurface
                        ? "The liquid surface as a closed triangle mesh, one PLY per exported\n"
                          "frame: the particles become a signed distance (Zhu-Bridson) and the\n"
                          "zero level set is meshed. Clipped against the walls and any solids,\n"
                          "so the liquid ends ON them instead of bulging through."
                        : "3D particle methods only: there is no 2D polygoniser, and a grid-only\n"
                          "method has no particle cloud to surface.");
            if (SIMULATION.exports.exportSurfaceMesh && canSurface)
            {
                ImGui::Indent();
                ImGui::Checkbox("Solids as a separate PLY", &SIMULATION.exports.exportSurfaceSolids);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("The scene's obstacles, written ONCE as solids.ply -- they do\n"
                                      "not move. A separate file is a separate object in the\n"
                                      "renderer, which is how it gets its own material.\n\n"
                                      "It is the ANALYTIC surface, so a cylinder is round rather\n"
                                      "than the staircase the solver works with.");
                ImGui::SetNextItemWidth(120);
                ImGui::InputDouble("Voxel (x spacing)", &SIMULATION.exports.surfaceVoxelScale, 0, 0, "%.2f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Lattice pitch in particle spacings. Smaller resolves more and\n"
                                      "costs more; below about 0.5 it starts resolving the particles\n"
                                      "themselves rather than the liquid.");
                ImGui::SetNextItemWidth(120);
                ImGui::InputDouble("Radius (x spacing)", &SIMULATION.exports.surfaceRadiusScale, 0, 0, "%.2f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Per-particle radius in spacings. Larger fills gaps and swells\n"
                                      "the volume; smaller is truer to the liquid volume and can\n"
                                      "open holes in thin sheets.");
                ImGui::SetNextItemWidth(120);
                ImGui::InputDouble("Influence R (x spacing)", &SIMULATION.exports.surfaceInfluenceScale, 0, 0, "%.2f");
                if (SIMULATION.exports.surfaceInfluenceScale < 1.0) SIMULATION.exports.surfaceInfluenceScale = 1.0;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("How far the weighted mean averages, in spacings. THIS is the knob\n"
                                      "for a bumpy liquid -- the radius only offsets the level set, while\n"
                                      "R is what the field averages over. Measured on a jittered sheet,\n"
                                      "roughness 0.066 -> 0.055 -> 0.050 of a spacing at R = 2, 3, 4.\n"
                                      "Costs (R/2)^3 in build time and rounds off thin detail.");
                ImGui::SetNextItemWidth(120);
                ImGui::InputInt("Smoothing passes", &SIMULATION.exports.surfaceSmoothing);
                if (SIMULATION.exports.surfaceSmoothing < 0) SIMULATION.exports.surfaceSmoothing = 0;
                if (!SIMULATION.exports.surfaceStatus.empty())
                    ImGui::TextDisabled("%s", SIMULATION.exports.surfaceStatus.c_str());
                ImGui::Unindent();
            }
        }

        ImGui::Separator();
        ImGui::Text("Simulation Telemetry");
        ImGui::Checkbox("Time", &SIMULATION.exports.exportTime);
        ImGui::Checkbox("Divergence Sum", &SIMULATION.exports.exportDivSum);
        ImGui::Checkbox("CFL Number", &SIMULATION.exports.exportCFL);
        ImGui::Checkbox("Residual", &SIMULATION.exports.exportResidual);
        ImGui::Checkbox("Step timings (<step>_timing)", &SIMULATION.exports.exportStepTimings);
        ImGui::Checkbox("Flow integrals (E, Z)", &SIMULATION.exports.exportFlowIntegrals);
        ImGui::Separator();
        // These two cost a pass over the grid, so the box gates the COMPUTATION as well as the
        // column and the panel -- off by default, since most cases want neither.
        ImGui::Checkbox("Aerodynamics (Cd, Cl, Delta-p)", &SIMULATION.exports.aerodynamics);
        ImGui::Checkbox("Free surface (height, front)", &SIMULATION.exports.dambreak);
        if (SIMULATION.exports.aerodynamics)
        {
            ImGui::Text("Aerodynamics Telemetry");
            ImGui::Checkbox("Lift Coefficient (Cl)", &SIMULATION.exports.exportCl);
            ImGui::Checkbox("Drag Coefficient (Cd)", &SIMULATION.exports.exportCd);
            ImGui::Checkbox("L/D Ratio", &SIMULATION.exports.exportLtoDRatio);
        }

        ImGui::Separator();
        ImGui::TextWrapped("Last Telemetry Export: Frame %d", SIMULATION.exports.lastExportedTelemetryFrame);
        ImGui::TextWrapped("Last Grid Export: Frame %d (File #%d)",
                           SIMULATION.exports.lastExportedGridFrame,
                           SIMULATION.exports.gridExportCounter - 1);
    }

}

// ======== The UI shell: the window, the theme, the top bar and the tiled layout around the viewer's
// three panes. Its public calls are at the bottom of this file. ========

namespace NereusUI
{
enum Theme { THEME_LIGHT = 0, THEME_DARK = 1 };
enum WindowMode { WINDOW_WINDOWED = 0, WINDOW_FULLSCREEN = 1 };

// What the UI settings panel edits, kept per user in configs/.ui_settings.
struct Settings
{
    int theme = THEME_LIGHT;
    float fontScale = 1.0f;
    float sideWidth = 0.34f;      // the right column, as a fraction of the window width
    float controlHeight = 0.45f;  // the control pane, as a fraction of the right column
    float fieldControls = 0.30f;  // the Field tab's left column, as a fraction of the view pane
    float analysisHeight = 0.45f; // the line and probe panel, as a fraction of that column
    bool showSide = true;
    bool vsync = true;
    int windowMode = WINDOW_WINDOWED;     // a decorated window over the work area, or exclusive fullscreen
};

struct State
{
    GLFWwindow *window = nullptr;
    SimulationViewer *viewer = nullptr;
    Settings settings;
    bool settingsOpen = false;
    std::string configPath;
    double lastFrame = 0.0, fps = 0.0;
};

State &UI()
{
    static State state;
    return state;
}

const char *SETTINGS_FILE = "configs/.ui_settings";
const char *FONT_FILE = "include/imgui-1.92.4/misc/fonts/Cousine-Regular.ttf";

void LoadSettings(Settings &s)
{
    std::ifstream f(SETTINGS_FILE);
    std::string key;
    double value;
    while (f >> key >> value)
    {
        if (key == "THEME") s.theme = (int)value;
        else if (key == "FONT_SCALE") s.fontScale = (float)value;
        else if (key == "SIDE_WIDTH") s.sideWidth = (float)value;
        else if (key == "CONTROL_HEIGHT") s.controlHeight = (float)value;
        else if (key == "FIELD_CONTROLS") s.fieldControls = (float)value;
        else if (key == "ANALYSIS_HEIGHT") s.analysisHeight = (float)value;
        else if (key == "SHOW_SIDE") s.showSide = value != 0.0;
        else if (key == "VSYNC") s.vsync = value != 0.0;
        else if (key == "WINDOW_MODE") s.windowMode = (int)value;
    }
}

void SaveSettings(const Settings &s)
{
    std::error_code ec;
    std::filesystem::create_directories("configs", ec);
    std::ofstream f(SETTINGS_FILE);
    f << "THEME " << s.theme << "\nFONT_SCALE " << s.fontScale << "\nSIDE_WIDTH " << s.sideWidth
      << "\nCONTROL_HEIGHT " << s.controlHeight << "\nFIELD_CONTROLS " << s.fieldControls
      << "\nANALYSIS_HEIGHT " << s.analysisHeight << "\nSHOW_SIDE " << (s.showSide ? 1 : 0)
      << "\nVSYNC " << (s.vsync ? 1 : 0) << "\nWINDOW_MODE " << s.windowMode << "\n";
}

// ---- the palette: a light paper theme after a utilitarian terminal, and its dark inverse ----
ImVec4 Hex(unsigned rgb, float a = 1.0f)
{
    return ImVec4(((rgb >> 16) & 0xFF) / 255.0f, ((rgb >> 8) & 0xFF) / 255.0f, (rgb & 0xFF) / 255.0f, a);
}

struct Palette
{
    unsigned page, pane, field, fieldHover, fieldActive, button, buttonHover, buttonActive, header, headerHover,
        headerActive, border, text, muted, strip, stripText, scroll, scrollGrab, rowAlt, select;
};

Palette ThemePalette(int theme)
{
    if (theme == THEME_DARK)
        return {0x121211, 0x222220, 0x2C2B28, 0x37352F, 0x43403A, 0x33312C, 0x403D37, 0x4C4942, 0x34322D, 0x3E3B35,
                0x4A4740, 0x57544C, 0xE6E3D8, 0x8D8A80, 0xE6E3D8, 0x1C1C1A, 0x2A2926, 0x4E4B44, 0x282725, 0x3C4A38};
    return {0x2A2A28, 0xECEAE0, 0xF6F5EF, 0xE6E3D8, 0xD9D5C7, 0xE2DFD3, 0xD2CEBF, 0xBDB8A6, 0xDAD6C8, 0xCFCABA,
            0xBFB9A6, 0x7A776C, 0x1C1C1A, 0x6E6C64, 0x1C1C1A, 0xECEAE0, 0xE2DFD3, 0xA9A598, 0xE4E1D6, 0xC9D7C2};
}

ImVec4 PageColour() { return Hex(ThemePalette(UI().settings.theme).page); }
ImU32 StripColour() { return ImGui::GetColorU32(Hex(ThemePalette(UI().settings.theme).strip)); }
ImU32 StripTextColour() { return ImGui::GetColorU32(Hex(ThemePalette(UI().settings.theme).stripText)); }

void ApplyTheme(const Settings &s)
{
    const Palette p = ThemePalette(s.theme);
    ImGuiStyle &st = ImGui::GetStyle();
    st.WindowPadding = ImVec2(8.0f, 8.0f);
    st.FramePadding = ImVec2(6.0f, 3.0f);
    st.CellPadding = ImVec2(6.0f, 3.0f);
    st.ItemSpacing = ImVec2(8.0f, 5.0f);
    st.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    st.ScrollbarSize = 12.0f;
    st.GrabMinSize = 10.0f;
    st.WindowBorderSize = st.ChildBorderSize = st.PopupBorderSize = st.FrameBorderSize = 1.0f;
    st.TabBorderSize = 1.0f;
    st.WindowRounding = st.ChildRounding = st.FrameRounding = st.PopupRounding = 0.0f;
    st.ScrollbarRounding = st.GrabRounding = st.TabRounding = 0.0f;
    st.FontScaleMain = s.fontScale;

    ImVec4 *c = st.Colors;
    c[ImGuiCol_Text] = Hex(p.text);
    c[ImGuiCol_InputTextCursor] = Hex(p.text);   // the caret follows the ink, or it is white on paper
    c[ImGuiCol_TextDisabled] = Hex(p.muted);
    c[ImGuiCol_WindowBg] = Hex(p.pane);
    c[ImGuiCol_ChildBg] = Hex(p.pane, 0.0f);
    c[ImGuiCol_PopupBg] = Hex(p.field);
    c[ImGuiCol_Border] = Hex(p.border);
    c[ImGuiCol_BorderShadow] = Hex(0, 0.0f);
    c[ImGuiCol_FrameBg] = Hex(p.field);
    c[ImGuiCol_FrameBgHovered] = Hex(p.fieldHover);
    c[ImGuiCol_FrameBgActive] = Hex(p.fieldActive);
    c[ImGuiCol_TitleBg] = c[ImGuiCol_TitleBgActive] = c[ImGuiCol_TitleBgCollapsed] = Hex(p.header);
    c[ImGuiCol_MenuBarBg] = Hex(p.pane);
    c[ImGuiCol_ScrollbarBg] = Hex(p.scroll);
    c[ImGuiCol_ScrollbarGrab] = Hex(p.scrollGrab);
    c[ImGuiCol_ScrollbarGrabHovered] = c[ImGuiCol_ScrollbarGrabActive] = Hex(p.text, 0.6f);
    c[ImGuiCol_CheckMark] = Hex(p.text);
    c[ImGuiCol_SliderGrab] = Hex(p.text, 0.65f);
    c[ImGuiCol_SliderGrabActive] = Hex(p.text);
    c[ImGuiCol_Button] = Hex(p.button);
    c[ImGuiCol_ButtonHovered] = Hex(p.buttonHover);
    c[ImGuiCol_ButtonActive] = Hex(p.buttonActive);
    c[ImGuiCol_Header] = Hex(p.header);
    c[ImGuiCol_HeaderHovered] = Hex(p.headerHover);
    c[ImGuiCol_HeaderActive] = Hex(p.headerActive);
    c[ImGuiCol_Separator] = c[ImGuiCol_SeparatorHovered] = c[ImGuiCol_SeparatorActive] = Hex(p.border, 0.7f);
    c[ImGuiCol_ResizeGrip] = c[ImGuiCol_ResizeGripHovered] = c[ImGuiCol_ResizeGripActive] = Hex(0, 0.0f);
    c[ImGuiCol_Tab] = Hex(p.header);
    c[ImGuiCol_TabHovered] = Hex(p.field);
    c[ImGuiCol_TabSelected] = Hex(p.pane);
    c[ImGuiCol_TabSelectedOverline] = Hex(p.text);
    c[ImGuiCol_TabDimmed] = Hex(p.header);
    c[ImGuiCol_TabDimmedSelected] = Hex(p.pane);
    c[ImGuiCol_TabDimmedSelectedOverline] = Hex(p.muted);
    c[ImGuiCol_TableHeaderBg] = Hex(p.header);
    c[ImGuiCol_TableBorderStrong] = Hex(p.border);
    c[ImGuiCol_TableBorderLight] = Hex(p.border, 0.4f);
    c[ImGuiCol_TableRowBg] = Hex(0, 0.0f);
    c[ImGuiCol_TableRowBgAlt] = Hex(p.rowAlt, 0.6f);
    c[ImGuiCol_TextSelectedBg] = Hex(p.select);
    c[ImGuiCol_PlotLines] = c[ImGuiCol_PlotHistogram] = Hex(p.text);
    c[ImGuiCol_ModalWindowDimBg] = Hex(0, 0.35f);

    ImPlot::StyleColorsAuto();
    // Every auto-fitted axis keeps 2.5% clear at each end, so a value on the edge keeps its tick label.
    ImPlot::GetStyle().FitPadding = ImVec2(0.025f, 0.025f);
    ImPlot::GetStyle().Colors[ImPlotCol_PlotBg] = Hex(p.field);
    ImPlot::GetStyle().Colors[ImPlotCol_FrameBg] = Hex(p.pane, 0.0f);
    ImPlot::GetStyle().Colors[ImPlotCol_AxisGrid] = Hex(p.border, 0.25f);
    // The plot series: muted hues on the light paper, their pastel twins on the dark. The order keeps
    // neighbouring series apart under the three colour-vision deficiencies (OKLab dE >= 5.9).
    static const unsigned seriesLight[8] = {0x3E7CB1, 0xC96A2E, 0x8A6BBE, 0x4E9A55,
                                            0xC7609D, 0x9C7650, 0x2E9A98, 0xC9505C};
    static const unsigned seriesDark[8] = {0x8FB8DE, 0xF2B37E, 0xC3AEE2, 0x9ED39F,
                                           0xF0AACD, 0xCDB08A, 0x8FD0CF, 0xEE9AA0};
    const char *seriesName = s.theme == THEME_DARK ? "SeriesDark" : "SeriesLight";
    ImPlotColormap series = ImPlot::GetColormapIndex(seriesName);
    if (series < 0)
    {
        ImVec4 cols[8];
        for (int k = 0; k < 8; k++) cols[k] = Hex(s.theme == THEME_DARK ? seriesDark[k] : seriesLight[k]);
        series = ImPlot::AddColormap(seriesName, cols, 8, true);
    }
    ImPlot::GetStyle().Colormap = series;
    if (s.theme == THEME_DARK) ImPlot3D::StyleColorsDark(); else ImPlot3D::StyleColorsLight();
}

// ---- a pane: a fixed tile with a header strip, after a tiling window manager ----
bool BeginPane(const char *id, const char *title, const char *subtitle, ImVec2 pos, ImVec2 size)
{
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::Begin(id, nullptr, flags);
    const float strip = ImGui::GetFrameHeight() + 4.0f;
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + strip), StripColour());
    const float ty = pos.y + 0.5f * (strip - ImGui::GetFontSize());
    dl->AddText(ImVec2(pos.x + 10.0f, ty), StripTextColour(), title);
    const float tw = ImGui::CalcTextSize(title).x;
    if (subtitle && subtitle[0])
        dl->AddText(ImVec2(pos.x + 22.0f + tw, ty), ImGui::GetColorU32(ImGuiCol_TextDisabled), subtitle);
    ImGui::SetCursorScreenPos(ImVec2(pos.x + ImGui::GetStyle().WindowPadding.x, pos.y + strip + 6.0f));
    return ImGui::BeginChild("##paneBody", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None);
}

void EndPane()
{
    ImGui::EndChild();
    ImGui::End();
}

// A label set in the strip colours, as the top bar's tags.
void Tag(const char *text)
{
    const ImVec2 p = ImGui::GetCursorScreenPos(), sz = ImGui::CalcTextSize(text);
    const float h = ImGui::GetFrameHeight();
    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p.x, p.y), ImVec2(p.x + sz.x + 12.0f, p.y + h), StripColour());
    ImGui::GetWindowDrawList()->AddText(ImVec2(p.x + 6.0f, p.y + 0.5f * (h - sz.y)), StripTextColour(), text);
    ImGui::Dummy(ImVec2(sz.x + 12.0f, h));
}

// A label and its value. `minChars` reserves room for the value, so a number that changes width as it ticks does not
// shift everything to its right every frame.
void Field(int minChars, const char *label, const char *fmt, ...)
{
    char value[96];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(value, sizeof(value), fmt, args);
    va_end(args);
    ImGui::SameLine(0.0f, 16.0f);
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine(0.0f, 5.0f);
    const float start = ImGui::GetCursorPosX();
    ImGui::TextUnformatted(value);
    const float reserve = minChars * ImGui::CalcTextSize("0").x;
    const float used = ImGui::CalcTextSize(value).x;
    if (used < reserve)
    {
        ImGui::SameLine(0.0f, 0.0f);
        ImGui::SetCursorPosX(start + reserve);
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
    }
}

// ---- the settings panel, behind the button at the top-left corner ----
// Windowed: decorated, over the monitor's work area (the screen less the task bars). Fullscreen: the monitor, exclusive.
void ApplyWindowMode(GLFWwindow *w, int mode)
{
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *vm = glfwGetVideoMode(monitor);
    if (mode == WINDOW_FULLSCREEN)
        glfwSetWindowMonitor(w, monitor, 0, 0, vm->width, vm->height, vm->refreshRate);
    else
    {
        int x = 0, y = 0, ww = vm->width, hh = vm->height;
        glfwGetMonitorWorkarea(monitor, &x, &y, &ww, &hh);
        glfwSetWindowMonitor(w, nullptr, x, y, ww, hh, GLFW_DONT_CARE);
    }
    glfwSwapInterval(UI().settings.vsync ? 1 : 0);   // a mode switch may hand back a new swap chain
}

void DrawSettingsPanel()
{
    State &u = UI();
    Settings &s = u.settings;
    bool changed = false;
    ImGui::SeparatorText("Appearance");
    changed |= ImGui::RadioButton("Light", &s.theme, THEME_LIGHT);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Dark", &s.theme, THEME_DARK);
    ImGui::SetNextItemWidth(200);
    changed |= ImGui::SliderFloat("Text size", &s.fontScale, 0.75f, 1.75f, "%.2fx");
    ImGui::SeparatorText("Layout");
    changed |= ImGui::Checkbox("Side panes", &s.showSide);
    ImGui::BeginDisabled(!s.showSide);
    ImGui::SetNextItemWidth(200);
    changed |= ImGui::SliderFloat("Side width", &s.sideWidth, 0.2f, 0.6f, "%.2f");
    ImGui::SetNextItemWidth(200);
    changed |= ImGui::SliderFloat("Control height", &s.controlHeight, 0.2f, 0.8f, "%.2f");
    ImGui::EndDisabled();
    ImGui::SetNextItemWidth(200);
    changed |= ImGui::SliderFloat("Field controls width", &s.fieldControls, 0.15f, 0.6f, "%.2f");
    ImGui::SetNextItemWidth(200);
    changed |= ImGui::SliderFloat("Line and probe height", &s.analysisHeight, 0.2f, 0.8f, "%.2f");
    ImGui::SeparatorText("Display");
    const int previousMode = s.windowMode;
    ImGui::RadioButton("Windowed", &s.windowMode, WINDOW_WINDOWED);
    ImGui::SameLine();
    ImGui::RadioButton("Fullscreen", &s.windowMode, WINDOW_FULLSCREEN);
    if (s.windowMode != previousMode)
    {
        ApplyWindowMode(u.window, s.windowMode);
        changed = true;
    }
    if (ImGui::Checkbox("Vertical sync", &s.vsync))
    {
        glfwSwapInterval(s.vsync ? 1 : 0);
        changed = true;
    }
    if (changed)
    {
        ApplyTheme(s);
        SaveSettings(s);
    }
    ImGui::TextDisabled("Saved to %s", SETTINGS_FILE);
}

// ---- the top bar: the settings button, run control, and the run's state at a glance ----
void DrawTopBar()
{
    State &u = UI();
    if (!ImGui::BeginMainMenuBar())
        return;
    if (ImGui::Button(" UI "))
        ImGui::OpenPopup("##uisettings");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("UI settings: theme, text size, layout.");
    if (ImGui::BeginPopup("##uisettings"))
    {
        DrawSettingsPanel();
        ImGui::EndPopup();
    }

    if (ImGui::BeginMenu("VIEW"))
    {
        if (ImGui::MenuItem("Side panes", nullptr, &u.settings.showSide))
            SaveSettings(u.settings);
        ImGui::EndMenu();
    }

    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::Button(SIMULATION.running ? "PAUSE" : " RUN "))
    {
        SIMULATION.running = !SIMULATION.running;
        // Running again after a stop carries on past it, so the rule is re-armed rather than firing next step.
        if (SIMULATION.running && SIMULATION.stop.triggered)
            SIMULATION.stop.Rearm();
    }
    ImGui::SameLine();
    if (ImGui::Button("STEP"))
        SIMULATION.stepOnce = true;
    ImGui::SameLine();
    if (ImGui::Button("RESTART"))
        RESTART_REQUESTED = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("End this run and go back to the setup screen");
    ImGui::SameLine();
    if (ImGui::Button("QUIT"))
        QUIT_REQUESTED = true;

    ImGui::SameLine(0.0f, 16.0f);
    Tag(SIMULATION.stop.triggered ? "STOPPED" : (SIMULATION.running ? "RUNNING" : "PAUSED"));
    Field(0, "METHOD", "%s %dD", ActiveMethod().key, DIMENSION);
    if (MethodUsesGrid())
    {
        if (DIMENSION == 3) Field(0, "GRID", "%dx%dx%d", SIMULATION.Nx, SIMULATION.Ny, SIMULATION.Nz);
        else Field(0, "GRID", "%dx%d", SIMULATION.Nx, SIMULATION.Ny);
    }
    if (MethodUsesParticles()) Field(9, "PARTICLES", "%d", SIMULATION.particles.count);
    Field(8, "IT", "%llu", (unsigned long long)SIMULATION.telemetry.iteration);
    Field(10, "T", "%.5g", SIMULATION.telemetry.clock);
    Field(10, "DT", "%.3e", SIMULATION.dt);
    Field(11, "STEP", "%.2f ms", SIMULATION.telemetry.lastStepWallTime * 1000.0);
    Field(4, "FPS", "%.0f", u.fps);

    char clock[40];
    const std::time_t now = std::time(nullptr);
    std::strftime(clock, sizeof(clock), "%Y-%m-%d %a %H:%M:%S", std::localtime(&now));
    const float w = ImGui::CalcTextSize(clock).x;
    ImGui::SameLine(ImGui::GetWindowWidth() - w - 12.0f);
    ImGui::TextUnformatted(clock);
    ImGui::EndMainMenuBar();
}

// ---- the layout: the view on the left, control and telemetry stacked on the right, a status line below ----
void DrawLayout()
{
    State &u = UI();
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    const float gap = 6.0f, footer = ImGui::GetFrameHeight() + 4.0f;
    const ImVec2 o(vp->WorkPos.x + gap, vp->WorkPos.y + gap);
    const float W = vp->WorkSize.x - 2.0f * gap, H = vp->WorkSize.y - 2.0f * gap - footer - gap;

    const float sideW = u.settings.showSide ? (std::max)(320.0f, W * u.settings.sideWidth) : 0.0f;
    const float viewW = u.settings.showSide ? W - sideW - gap : W;

    char sub[96];
    std::snprintf(sub, sizeof(sub), "%s, %dD", SIMULATION.CONFIG_NAME.empty() ? "untitled" : SIMULATION.CONFIG_NAME.c_str(),
                  DIMENSION);
    u.viewer->controlsWidth = u.settings.fieldControls;
    u.viewer->analysisHeight = u.settings.analysisHeight;
    if (BeginPane("##view", "VIEW/1", sub, o, ImVec2(viewW, H)))
        u.viewer->DrawViewPane();
    EndPane();

    if (u.settings.showSide)
    {
        const float x = o.x + viewW + gap;
        const float ctlH = (std::max)(160.0f, H * u.settings.controlHeight);
        if (BeginPane("##control", "CONTROL/1", "run, checkpoint, export", ImVec2(x, o.y), ImVec2(sideW, ctlH)))
            u.viewer->DrawControlPane();
        EndPane();
        if (BeginPane("##telemetry", "TELEMETRY/1", "solver and flow", ImVec2(x, o.y + ctlH + gap),
                      ImVec2(sideW, H - ctlH - gap)))
            u.viewer->DrawTelemetryPane();
        EndPane();
    }

    // The status line: the file being run and where its exports go.
    ImGui::SetNextWindowPos(ImVec2(o.x, o.y + H + gap));
    ImGui::SetNextWindowSize(ImVec2(W, footer));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 2.0f));
    ImGui::Begin("##status", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                          ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImGui::TextUnformatted(SIMULATION.stop.triggered ? SIMULATION.stop.reason.c_str() : "READY.");
    ImGui::SameLine(0.0f, 24.0f);
    ImGui::TextDisabled("CONFIG %s", u.configPath.c_str());
    ImGui::SameLine(0.0f, 24.0f);
    ImGui::TextDisabled("EXPORT %s", SIMULATION.exports.enabled ? ExportDirectory("").c_str() : "off");
    ImGui::End();
    ImGui::PopStyleVar();
}

// One frame of an ImGui program on this window: poll, draw `body`, render and swap.
void BeginFrame()
{
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void EndFrame()
{
    State &u = UI();
    ImGui::Render();
    int w, h;
    glfwGetFramebufferSize(u.window, &w, &h);
    glViewport(0, 0, w, h);
    const ImVec4 page = PageColour();
    glClearColor(page.x, page.y, page.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(u.window);
}

// The setup screen, on the window InitializeUI opened. True when Start was pressed; `configFile` is then the file
// the panel wrote, which is the last preset loaded or saved.
bool RunSimulationSetupScreen(std::string &configFile)
{
    SimulationSetupUI ui;
    while (!glfwWindowShouldClose(UI().window))
    {
        BeginFrame();
        const bool start = ui.Render();
        EndFrame();
        if (start)
        {
            configFile = ui.configPath;
            return true;
        }
    }
    return false;
}
} // namespace NereusUI

// ---------------- The calls a program makes ----------------

// Opens the window and ImGui, applies the theme, and runs the setup screen. True with `configFile` set when Start was
// pressed; false if the window was closed (or could not be opened), after which DisposeUI is still safe to call.
bool InitializeUI(std::string &configFile)
{
    using namespace NereusUI;
    State &u = UI();
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);   // macOS gives a core profile only with this
#endif
    // A decorated window over the work area; fullscreen, when the settings ask for it, is applied below.
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);
    int wx = 0, wy = 0, ww = mode->width, wh = mode->height;
    glfwGetMonitorWorkarea(monitor, &wx, &wy, &ww, &wh);
    u.window = glfwCreateWindow(ww, wh, (std::string("NEREUS ") + NEREUS_VERSION).c_str(), nullptr, nullptr);
    if (u.window == nullptr)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    glfwSetWindowPos(u.window, wx, wy);
    glfwMakeContextCurrent(u.window);

    LoadSettings(u.settings);
    if (u.settings.windowMode != WINDOW_FULLSCREEN) u.settings.windowMode = WINDOW_WINDOWED;   // any other value, windowed
    if (u.settings.windowMode == WINDOW_FULLSCREEN) ApplyWindowMode(u.window, WINDOW_FULLSCREEN);
    glfwSwapInterval(u.settings.vsync ? 1 : 0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImPlot3D::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(u.window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    // A plain monospace face, the vendored Cousine; ImGui's own font when it is not found from here.
    if (std::filesystem::exists(FONT_FILE))
        ImGui::GetIO().Fonts->AddFontFromFileTTF(FONT_FILE, 15.0f);
    ApplyTheme(u.settings);

    if (!RunSimulationSetupScreen(configFile))
        return false;
    u.configPath = configFile;
    return true;
}

// One frame: the panes for the run's current state. Call after each simulation step. False once the window is closing.
bool UpdateUI()
{
    using namespace NereusUI;
    State &u = UI();
    if (u.window == nullptr)
        return false;
    if (QUIT_REQUESTED)
        glfwSetWindowShouldClose(u.window, GLFW_TRUE);
    if (glfwWindowShouldClose(u.window))
        return false;
    // A restart ends the run's loop. The viewer holds pointers into the grids, so it goes before they do.
    if (RESTART_REQUESTED)
    {
        delete u.viewer;
        u.viewer = nullptr;
        return false;
    }

    BeginFrame();
    if (u.viewer == nullptr && SIMULATION.GRID_SOL != nullptr)
        u.viewer = new SimulationViewer(SIMULATION.GRID_SOL);
    if (u.viewer != nullptr)
    {
        const double t0 = GetWallTime();
        u.viewer->UpdateGrid(SIMULATION.GRID_SOL);
        DrawTopBar();
        DrawLayout();
        u.viewer->drawTime = GetWallTime() - t0;
    }
    EndFrame();

    const double now = GetWallTime();
    if (u.lastFrame > 0.0 && now > u.lastFrame)
        u.fps = 0.9 * u.fps + 0.1 / (now - u.lastFrame);
    u.lastFrame = now;
    return !glfwWindowShouldClose(u.window);
}

// After UpdateUI returned false: true when it was a restart and Start was pressed again on the setup screen, with
// `configFile` the file to run; false when the window is closing.
bool RestartUI(std::string &configFile)
{
    using namespace NereusUI;
    State &u = UI();
    if (!RESTART_REQUESTED || u.window == nullptr || glfwWindowShouldClose(u.window))
        return false;
    RESTART_REQUESTED = false;
    if (!RunSimulationSetupScreen(configFile))
        return false;
    u.configPath = configFile;
    return true;
}

// Closes everything InitializeUI opened.
void DisposeUI()
{
    using namespace NereusUI;
    State &u = UI();
    delete u.viewer;
    u.viewer = nullptr;
    if (u.window == nullptr)
        return;
    SaveSettings(u.settings);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImPlot3D::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(u.window);
    glfwTerminate();
    u.window = nullptr;
}

