// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef EXPORTS_H
#define EXPORTS_H

#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "MAC.h"
#include "Particles.h"
#include "Scene.h"
#include "MeshHandler.h"
#include "../Utils.h"
#include <cctype>
#include <filesystem>
#include <iomanip>

// Everything a run writes out: telemetry, frames (.vtr, particle CSV, PLY, .npy) and the final state. The
// manager decides WHEN, this decides WHAT; nothing here is called from a solver.
inline std::string ExportDirectory(const std::string &sub)
{
    const std::string base = SIMULATION.ExportPath.empty() ? std::string("Exports") : SIMULATION.ExportPath;
    // A run that never went through ConfigReader -- a harness -- has no config name, so it is
    // filed under the method instead.
    const std::string name = SIMULATION.CONFIG_NAME.empty() ? std::string(ActiveMethod().key)
                                                            : SIMULATION.CONFIG_NAME;
    std::string dir = base + "/" + name + "/" + std::to_string(SIMULATION.GRID_SIZE) +
                      (DIMENSION == 3 ? "_3D" : "_2D") +
                      "_re" + std::to_string(int(SIMULATION.RE)) + "/";
    if (!sub.empty()) dir += sub + "/";
    std::filesystem::create_directories(dir);
    return dir;
}

namespace Exports
{

// ---- the grid writers ----

inline bool WriteNpyFloat32(const std::string &path, const std::vector<size_t> &shape,
                     const std::vector<float> &data, std::string &err)
{
    size_t count = 1;
    for (size_t n : shape) count *= n;
    if (count != data.size()) { err = "npy: shape does not match the data"; return false; }

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t a = 0; a < shape.size(); a++)
        header += std::to_string(shape[a]) + ((shape.size() == 1 || a + 1 < shape.size()) ? ", " : "");
    header += "), }";
    // Magic (6) + version (2) + length (2) + header, padded with spaces to a multiple of 64 and
    // ended by a newline, as the format asks.
    const size_t unpadded = 10 + header.size() + 1;
    header.append((64 - unpadded % 64) % 64, ' ');
    header += '\n';

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) { err = "could not open " + path; return false; }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    f.write((const char *)magic, 8);
    const unsigned char len[2] = {(unsigned char)(header.size() & 0xff), (unsigned char)((header.size() >> 8) & 0xff)};
    f.write((const char *)len, 2);
    f.write(header.data(), (std::streamsize)header.size());
    f.write((const char *)data.data(), (std::streamsize)(data.size() * sizeof(float)));
    if (!f) { err = "write failed: " + path; return false; }
    return true;
}

inline void WriteGrid(MAC &g, int iteration)
{
    const std::string exportPath = ExportDirectory("VTK");

    // Always RectilinearGrid (.vtr), uniform or stretched, so every run imports into ParaView the same way.
    std::ostringstream oss;
    oss << exportPath << "grid-" << std::setw(4) << std::setfill('0') << iteration << ".vtr";
    std::string filename = oss.str();

    std::ofstream vtiFile(filename);
    if (!vtiFile.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    int xExtent = g.Nx - 1;
    int yExtent = g.Ny - 1;
    int zExtent = g.is2D ? 0 : (g.Nz - 1);

    // VTK Header
    vtiFile << "<?xml version=\"1.0\"?>\n";
    vtiFile << "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtiFile << "  <RectilinearGrid WholeExtent=\"0 " << xExtent << " 0 " << yExtent
            << " 0 " << zExtent << "\">\n";
    vtiFile << "    <Piece Extent=\"0 " << xExtent << " 0 " << yExtent 
            << " 0 " << zExtent << "\">\n";
    // The fields at 12 significant digits in Float64, like the coordinates: the default 6 digits of a float quantised
    // a velocity near 0.5 to ~5e-7, which is the size of the finest errors a convergence study measures.
    const std::streamsize fieldPrec = vtiFile.precision();
    vtiFile << std::setprecision(12);
    vtiFile << "      <PointData>\n";

    // Velocity vector field
    vtiFile << "        <DataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    if(g.is2D) {
        // 2D case - no k loops
        for (int i = 0; i < g.Ny; ++i) {
            for (int j = 0; j < g.Nx; ++j) {
                double u_val = (j < g.Nx - 1) ? (g.GetU(i, j) + g.GetU(i, j + 1)) / 2.0 : 0.0;
                double v_val = (i < g.Ny - 1) ? (g.GetV(i, j) + g.GetV(i + 1, j)) / 2.0 : 0.0;
                double w_val = 0.0;

                u_val = std::isfinite(u_val) ? u_val : 0.0;
                v_val = std::isfinite(v_val) ? v_val : 0.0;

                vtiFile << u_val << " " << v_val << " " << w_val << " ";
            }
            vtiFile << "\n";
        }
    } else {
        // 3D case
        for (int k = 0; k < g.Nz; ++k) {
            for (int i = 0; i < g.Ny; ++i) {
                for (int j = 0; j < g.Nx; ++j) {
                    double u_val = (j < g.Nx - 1) ? (g.GetU(i, j, k) + g.GetU(i, j + 1, k)) / 2.0 : 0.0;
                    double v_val = (i < g.Ny - 1) ? (g.GetV(i, j, k) + g.GetV(i + 1, j, k)) / 2.0 : 0.0;
                    double w_val = (k < g.Nz - 1) ? (g.GetW(i, j, k) + g.GetW(i, j, k + 1)) / 2.0 : 0.0;

                    u_val = std::isfinite(u_val) ? u_val : 0.0;
                    v_val = std::isfinite(v_val) ? v_val : 0.0;
                    w_val = std::isfinite(w_val) ? w_val : 0.0;

                    vtiFile << u_val << " " << v_val << " " << w_val << " ";
                }
                vtiFile << "\n";
            }
        }
    }
    vtiFile << "        </DataArray>\n";

    // Pressure field
    vtiFile << "        <DataArray type=\"Float64\" Name=\"Pressure\" format=\"ascii\">\n";
    if(g.is2D) {
        for (int i = 0; i < g.Ny; ++i) {
            for (int j = 0; j < g.Nx; ++j) {
                vtiFile << g.GetP(i, j) << " ";
            }
            vtiFile << "\n";
        }
    } else {
        for (int k = 0; k < g.Nz; ++k) {
            for (int i = 0; i < g.Ny; ++i) {
                for (int j = 0; j < g.Nx; ++j) {
                    vtiFile << g.GetP(i, j, k) << " ";
                }
                vtiFile << "\n";
            }
        }
    }
    vtiFile << "        </DataArray>\n";

    // Divergence field
    vtiFile << "        <DataArray type=\"Float64\" Name=\"Divergence\" format=\"ascii\">\n";
    if(g.is2D) {
        for (int i = 0; i < g.Ny; ++i) {
            for (int j = 0; j < g.Nx; ++j) {
                vtiFile << g.GetDivergencyAt(i, j) << " ";
            }
            vtiFile << "\n";
        }
    } else {
        for (int k = 0; k < g.Nz; ++k) {
            for (int i = 0; i < g.Ny; ++i) {
                for (int j = 0; j < g.Nx; ++j) {
                    vtiFile << g.GetDivergencyAt(i, j, k) << " ";
                }
                vtiFile << "\n";
            }
        }
    }
    vtiFile << "        </DataArray>\n";

    // Solid field
    vtiFile << "        <DataArray type=\"Int32\" Name=\"Solid\" format=\"ascii\">\n";
    if(g.is2D) {
        for (int i = 0; i < g.Ny; ++i) {
            for (int j = 0; j < g.Nx; ++j) {
                vtiFile << g.GetSolid(i, j) << " ";
            }
            vtiFile << "\n";
        }
    } else {
        for (int k = 0; k < g.Nz; ++k) {
            for (int i = 0; i < g.Ny; ++i) {
                for (int j = 0; j < g.Nx; ++j) {
                    vtiFile << g.GetSolid(i, j, k) << " ";
                }
                vtiFile << "\n";
            }
        }
    }
    vtiFile << "        </DataArray>\n";

    // Registered scalars, each under its own name -- so a density field shows up in ParaView
    // beside Velocity and Pressure with no further plumbing.
    for (int h = 0; h < g.ScalarCount(); h++)
    {
        vtiFile << "        <DataArray type=\"Float64\" Name=\"" << g.ScalarName(h)
                << "\" format=\"ascii\">\n";
        if (g.is2D) {
            for (int i = 0; i < g.Ny; ++i) {
                for (int j = 0; j < g.Nx; ++j) vtiFile << g.GetS(h, i, j) << " ";
                vtiFile << "\n";
            }
        } else {
            for (int k = 0; k < g.Nz; ++k) {
                for (int i = 0; i < g.Ny; ++i) {
                    for (int j = 0; j < g.Nx; ++j) vtiFile << g.GetS(h, i, j, k) << " ";
                    vtiFile << "\n";
                }
            }
        }
        vtiFile << "        </DataArray>\n";
    }

    vtiFile << "      </PointData>\n";
    vtiFile << std::setprecision(fieldPrec);

    // One coordinate per written value, i.e. the cell centres, at 12 digits in Float64: ParaView
    // differences these to get the cell widths.
    {
        const std::streamsize savedPrec = vtiFile.precision();
        vtiFile << std::setprecision(12);
        vtiFile << "      <Coordinates>\n";
        vtiFile << "        <DataArray type=\"Float64\" Name=\"X\" format=\"ascii\">\n";
        for (int j = 0; j < g.Nx; ++j) vtiFile << g.CentrePos(MAC::X, j) << " ";
        vtiFile << "\n        </DataArray>\n";
        vtiFile << "        <DataArray type=\"Float64\" Name=\"Y\" format=\"ascii\">\n";
        for (int i = 0; i < g.Ny; ++i) vtiFile << g.CentrePos(MAC::Y, i) << " ";
        vtiFile << "\n        </DataArray>\n";
        vtiFile << "        <DataArray type=\"Float64\" Name=\"Z\" format=\"ascii\">\n";
        if (g.is2D) vtiFile << 0.0 << " ";
        else for (int k = 0; k < g.Nz; ++k) vtiFile << g.CentrePos(MAC::Z, k) << " ";
        vtiFile << "\n        </DataArray>\n";
        vtiFile << "      </Coordinates>\n";
        vtiFile << std::setprecision(savedPrec);
    }

    vtiFile << "    </Piece>\n";
    vtiFile << "  </RectilinearGrid>\n";
    vtiFile << "</VTKFile>\n";

    vtiFile.close();
}

// A scalar name as a file name: anything but letters, digits, '-' and '_' becomes '_'.
inline std::string VolumetricFileStem(const std::string &name)
{
    std::string out = name;
    for (char &c : out)
        if (!std::isalnum((unsigned char)c) && c != '-' && c != '_') c = '_';
    return out;
}

inline void WriteJsonList(std::ofstream &f, const std::vector<double> &v)
{
    f << "[";
    for (size_t q = 0; q < v.size(); q++) f << (q ? ", " : "") << v[q];
    f << "]";
}

inline bool WriteVolumetric(MAC &g, int frame, double time, std::string &message)
{
    if (g.ScalarCount() == 0) { message = "volumetric: the grid carries no scalars"; return false; }
    const std::string dir = ExportDirectory("Volumetric");
    char tag[16];
    std::snprintf(tag, sizeof(tag), "%05d", frame);

    // Same order as the .vtr: [i, j] in 2D, [k, i, j] in 3D -- so array[..., row y, column x], ring included.
    const int nz = g.is2D ? 1 : g.Nz;
    const std::vector<size_t> shape = g.is2D ? std::vector<size_t>{(size_t)g.Ny, (size_t)g.Nx}
                                           : std::vector<size_t>{(size_t)nz, (size_t)g.Ny, (size_t)g.Nx};
    std::vector<float> data((size_t)nz * (size_t)g.Ny * (size_t)g.Nx);
    std::vector<std::string> files;
    for (int h = 0; h < g.ScalarCount(); h++)
    {
        size_t q = 0;
        for (int k = 0; k < nz; k++)
            for (int i = 0; i < g.Ny; i++)
                for (int j = 0; j < g.Nx; j++)
                    data[q++] = (float)(g.is2D ? g.GetS(h, i, j) : g.GetS(h, i, j, k));
        const std::string file = VolumetricFileStem(g.ScalarName(h)) + "_" + tag + ".npy";
        if (!WriteNpyFloat32(dir + file, shape, data, message)) return false;
        files.push_back(file);
    }

    const std::string jsonPath = dir + "frame_" + tag + ".json";
    std::ofstream f(jsonPath);
    if (!f.is_open()) { message = "could not open " + jsonPath; return false; }
    f << std::setprecision(12);
    f << "{\n  \"format\": \"NEREUS volumetric\",\n  \"version\": 1,\n";
    f << "  \"frame\": " << frame << ",\n  \"time\": " << time << ",\n";
    f << "  \"dimension\": " << (g.is2D ? 2 : 3) << ",\n";
    f << "  \"axes\": " << (g.is2D ? "[\"y\", \"x\"]" : "[\"z\", \"y\", \"x\"]") << ",\n";
    f << "  \"shape\": [";
    for (size_t a = 0; a < shape.size(); a++) f << (a ? ", " : "") << shape[a];
    f << "],\n  \"dtype\": \"float32\",\n  \"up\": \"y\",\n";
    f << "  \"uniform\": " << (g.nonUniform ? "false" : "true") << ",\n  \"dh\": " << g.dh << ",\n";
    f << "  \"faces\": {\"x\": "; WriteJsonList(f, g.Faces(MAC::X)); f << ", \"y\": "; WriteJsonList(f, g.Faces(MAC::Y));
    if (!g.is2D) { f << ", \"z\": "; WriteJsonList(f, g.Faces(MAC::Z)); }
    f << "},\n  \"centres\": {\"x\": "; WriteJsonList(f, g.Centres(MAC::X)); f << ", \"y\": "; WriteJsonList(f, g.Centres(MAC::Y));
    if (!g.is2D) { f << ", \"z\": "; WriteJsonList(f, g.Centres(MAC::Z)); }
    f << "},\n  \"fields\": [";
    for (int h = 0; h < g.ScalarCount(); h++)
        f << (h ? ", " : "") << "{\"name\": \"" << g.ScalarName(h) << "\", \"file\": \"" << files[(size_t)h] << "\"}";
    f << "]\n}\n";
    if (!f) { message = "write failed: " + jsonPath; return false; }

    message = "frame_" + std::string(tag) + ": " + std::to_string(g.ScalarCount()) + " field(s), " +
              std::to_string(g.Nx) + " x " + std::to_string(g.Ny) + (g.is2D ? "" : " x " + std::to_string(nz));
    return true;
}


inline void InitializeTelemetry()
{
    if (SIMULATION.exports.fileInitialized) return;

    const std::string exportPath = ExportDirectory("");

    // Only the path and the header here: the file opens at the first row (OpenTelemetryFile), after any
    // checkpoint load, so a resumed run keeps the rows it already wrote.
    SIMULATION.exports.telemetryPath = exportPath + "telemetry.csv";
    std::ostringstream header;

    // Write CSV header
    header << "iteration";

    if (SIMULATION.exports.exportTime) 
        header << ",time";
    if (SIMULATION.exports.exportDivSum) 
        header << ",div_sum";
    if (SIMULATION.exports.exportCFL) 
        header << ",advcfl,diffcfl,cell_reynolds";
    if (SIMULATION.exports.exportResidual) 
        header << ",residual";
    if (SIMULATION.exports.exportStepTimings) {
        for (const std::string &name : SIMULATION.telemetry.stepNames)
            header << "," << name << "_timing";
        for (const std::string &name : SIMULATION.telemetry.stepNames)
            header << "," << name << "_counter";
    }
    if (SIMULATION.exports.exportFlowIntegrals)
        header << ",kinetic_energy,enstrophy,dE_dt";

    // Aerodynamics headers, when the run asked for them
    if (SIMULATION.exports.aerodynamics) {
        if (SIMULATION.exports.exportCl) 
            header << ",cl";
        if (SIMULATION.exports.exportCd) 
            header << ",cd";
        if (SIMULATION.exports.exportCd)
            header << ",cd_cv";
        if (SIMULATION.exports.exportCl)
            header << ",cl_cv";
        if (SIMULATION.exports.exportLtoDRatio) 
            header << ",ld_ratio";
    }

    if (SIMULATION.exports.dambreak) {
        if (SIMULATION.exports.exportFluidHeight) 
            header << ",FluidHeight";
        if (SIMULATION.exports.exportFluidFront) 
            header << ",FluidFront";
        // The liquid volume the grid sees (FLUID cells x cell volume; an area in 2D): FLIP's volume drift.
        if (MethodUsesParticles() && MethodUsesGrid())
            header << ",FluidVolume";
    }

    // Scene probes: one column each, always written when the scene declares any.
    for (const Probe &pr : SCENE.probes)
        header << "," << pr.name << "_" << pr.quantity;

    header << "\n";
    SIMULATION.exports.telemetryHeader = header.str();
    SIMULATION.exports.fileInitialized = true;
}

// Truncates the CSV to its header plus the rows at or before `keepThrough` (-1 keeps none). A file whose
// header differs is a different set of columns, so none of its rows are kept.
inline bool OpenTelemetryFile(long long keepThrough)
{
    ExportSettings &e = SIMULATION.exports;
    std::vector<std::string> kept;
    if (e.telemetryFile.is_open()) e.telemetryFile.close();
    if (keepThrough >= 0)
    {
        std::ifstream in(e.telemetryPath);
        std::string line;
        if (in && std::getline(in, line) && line + "\n" == e.telemetryHeader)
            while (std::getline(in, line))
                if (!line.empty() && std::atoll(line.c_str()) <= keepThrough) kept.push_back(line);
    }
    e.telemetryFile.open(e.telemetryPath, std::ios::out | std::ios::trunc);
    if (!e.telemetryFile.is_open())
    {
        std::cerr << "Failed to open telemetry export file: " << e.telemetryPath << std::endl;
        e.fileInitialized = false;
        return false;
    }
    e.telemetryFile << e.telemetryHeader;
    for (const std::string &l : kept) e.telemetryFile << l << "\n";
    if (keepThrough >= 0)
        std::cout << "Telemetry: resumed at iteration " << keepThrough << ", kept " << kept.size()
                  << " earlier rows of " << e.telemetryPath << std::endl;
    return true;
}

inline void WriteTelemetryRow()
{
    if (!SIMULATION.exports.enabled || !SIMULATION.exports.fileInitialized) return;
    // First row, or the first after a checkpoint load: (re)open, keeping the rows up to the restored step.
    if (!SIMULATION.exports.telemetryFile.is_open() || SIMULATION.exports.telemetryResumeIteration >= 0)
    {
        const long long keep = SIMULATION.exports.telemetryResumeIteration;
        SIMULATION.exports.telemetryResumeIteration = -1;
        if (!OpenTelemetryFile(keep)) return;
    }

    SIMULATION.exports.telemetryFile << SIMULATION.telemetry.iteration;

    // Export simulation telemetry
    if (SIMULATION.exports.exportTime) {
        if (!SIMULATION.telemetry.time.empty())
            SIMULATION.exports.telemetryFile << "," << SIMULATION.telemetry.time.back();
        else
            SIMULATION.exports.telemetryFile << ",0.0";
    }

    if (SIMULATION.exports.exportDivSum) {
        if (!SIMULATION.telemetry.div_sum.empty())
            SIMULATION.exports.telemetryFile << "," << SIMULATION.telemetry.div_sum.back();
        else
            SIMULATION.exports.telemetryFile << ",0.0";
    }


    if (SIMULATION.exports.exportCFL) {
        if (!SIMULATION.telemetry.advcfl.empty()){
            SIMULATION.exports.telemetryFile << "," << SIMULATION.telemetry.advcfl.back();
            SIMULATION.exports.telemetryFile << "," << SIMULATION.telemetry.diffcfl.back();
            SIMULATION.exports.telemetryFile << "," << SIMULATION.telemetry.cell_reynolds.back();
        }
        else
            SIMULATION.exports.telemetryFile << ",0.0,0.0,0.0";
    }

    if (SIMULATION.exports.exportResidual) {
        if (!SIMULATION.telemetry.residual.empty())
            SIMULATION.exports.telemetryFile << "," << SIMULATION.telemetry.residual.back();
        else
            SIMULATION.exports.telemetryFile << ",0.0";
    }

    if (SIMULATION.exports.exportStepTimings) {
        for (const std::vector<double> &series : SIMULATION.telemetry.stepTimings) {
            if (!series.empty())
                SIMULATION.exports.telemetryFile << "," << series.back();
            else
                SIMULATION.exports.telemetryFile << ",0.0";
        }
        for (const std::vector<double> &series : SIMULATION.telemetry.stepCounters)
            SIMULATION.exports.telemetryFile << "," << (series.empty() ? -1 : (long)series.back());
    }

    if (SIMULATION.exports.exportFlowIntegrals) {
        if (SIMULATION.flowIntegrals.available && !SIMULATION.flowIntegrals.time.empty())
            SIMULATION.exports.telemetryFile << "," << SIMULATION.flowIntegrals.kineticEnergy.back()
                                          << "," << SIMULATION.flowIntegrals.enstrophy.back()
                                          << "," << SIMULATION.flowIntegrals.energyRate.back();
        else
            SIMULATION.exports.telemetryFile << ",0.0,0.0,nan";
    }

    // Aerodynamics telemetry, when the run asked for it
    if (SIMULATION.exports.aerodynamics) {
        if (SIMULATION.exports.exportCl) {
            if (!SIMULATION.aero.Cl.empty())
                SIMULATION.exports.telemetryFile << "," << SIMULATION.aero.Cl.back();
            else
                SIMULATION.exports.telemetryFile << ",0.0";
        }

        if (SIMULATION.exports.exportCd) {
            if (!SIMULATION.aero.Cd.empty())
                SIMULATION.exports.telemetryFile << "," << SIMULATION.aero.Cd.back();
            else
                SIMULATION.exports.telemetryFile << ",0.0";
        }
        if (SIMULATION.exports.exportCd)
            SIMULATION.exports.telemetryFile << "," << (SIMULATION.aero.CdCV.empty() ? 0.0 : SIMULATION.aero.CdCV.back());
        if (SIMULATION.exports.exportCl)
            SIMULATION.exports.telemetryFile << "," << (SIMULATION.aero.ClCV.empty() ? 0.0 : SIMULATION.aero.ClCV.back());

        if (SIMULATION.exports.exportLtoDRatio) {
            if (!SIMULATION.aero.LtoDRatio.empty())
                SIMULATION.exports.telemetryFile << "," << SIMULATION.aero.LtoDRatio.back();
            else
                SIMULATION.exports.telemetryFile << ",0.0";
        }
    }
    if (SIMULATION.exports.dambreak) {
        if(SIMULATION.exports.exportFluidHeight){
            if(!SIMULATION.dambreak.fluidLevel.empty()){
                SIMULATION.exports.telemetryFile << "," << SIMULATION.dambreak.fluidLevel.back();
            }
        }
        if(SIMULATION.exports.exportFluidFront){
            if(!SIMULATION.dambreak.fluidFrontPosition.empty()){
                SIMULATION.exports.telemetryFile << "," << SIMULATION.dambreak.fluidFrontPosition.back();
            }
        }
        if (MethodUsesParticles() && MethodUsesGrid())
            SIMULATION.exports.telemetryFile << "," << SIMULATION.GRID_SOL->GetFluidCellCount() *
                                                          std::pow(SIMULATION.GRID_SOL->dh, DIMENSION);

    }

    // Scene probes, read from the field this step produced; the raw value, filtered afterwards if needed.
    for (const Probe &pr : SCENE.probes)
        SIMULATION.exports.telemetryFile << "," << EvaluateProbe(SIMULATION.GRID_SOL, &SIMULATION.particles, pr);

    SIMULATION.exports.telemetryFile << "\n";
    SIMULATION.exports.telemetryFile.flush(); // Ensure data is written
}

// The liquid is clipped against the walls (Particles::WallBox) and against any scene solids.
// `cap`: the clip needs the distance only where it is below cap (MeshExport::SurfaceParams::solidSDF).
inline double LiquidClipSDF(double x, double y, double z, double cap)
{
    double lo[3], hi[3];
    Particles::WallBox(lo, hi);
    const double box = (std::min)((std::min)(x - lo[0], hi[0] - x),
                       (std::min)((std::min)(y - lo[1], hi[1] - y),
                                  (std::min)(z - lo[2], hi[2] - z)));
    if (!(SCENE.loaded && SCENE.HasSolidRegions())) return box;
    return (std::min)(box, SCENE.SolidFieldBelow(x, y, z, (std::min)(cap, box)));
}

inline double SceneSolidField(double x, double y, double z, double cap)
{
    return SCENE.SolidFieldBelow(x, y, z, cap);
}

// The liquid surface as a watertight PLY, and (once) the scene's obstacles as their own file.
inline void WriteSurface(int frame)
{
    const bool is3D = (DIMENSION == 3);
    const bool particleSolver = MethodUsesParticles();
    if (!is3D || !particleSolver)
    {
        SIMULATION.exports.surfaceStatus = "surface export needs a 3D particle method";
        return;
    }

    // Gather positions and the seeding pitch the field is scaled by.
    std::vector<double> xyz;
    double spacing = 0.0;
    const int n = SIMULATION.particles.count;
    xyz.reserve((size_t)n * 3);
    for (int i = 0; i < n; i++)
    {
        const Particle &q = SIMULATION.particles.data[i];
        xyz.push_back(q.x); xyz.push_back(q.y); xyz.push_back(q.z);
    }
    // The seeding pitch is the method's answer; a method that does not give one is taken to seed
    // one particle per cell.
    spacing = ActiveMethod().particleSpacing ? ActiveMethod().particleSpacing() : SIMULATION.dh;
    if (xyz.empty() || !(spacing > 0.0))
    {
        SIMULATION.exports.surfaceStatus = "no particles to surface";
        return;
    }

    const std::string dir = ExportDirectory("Surface");

    MeshExport::SurfaceParams p;
    p.spacing = spacing;
    p.voxel = SIMULATION.exports.surfaceVoxelScale * spacing;
    p.particleRadius = SIMULATION.exports.surfaceRadiusScale * spacing;
    p.influence = SIMULATION.exports.surfaceInfluenceScale * spacing;
    p.smoothIterations = SIMULATION.exports.surfaceSmoothing;
    // Trim the liquid at the walls and out of the obstacles, so it ends ON them rather than bulging
    // through.
    p.solidSDF = LiquidClipSDF;

    MeshExport::TriMesh mesh;
    std::string err;
    if (!MeshExport::SurfaceFromParticles(xyz.data(), xyz.size() / 3, p, mesh, err))
    { SIMULATION.exports.surfaceStatus = err; return; }
    char name[512];
    std::snprintf(name, sizeof(name), "%sliquid_%05d.ply", dir.c_str(), frame);
    if (!MeshExport::WritePLY(name, mesh, err))
    { SIMULATION.exports.surfaceStatus = err; return; }

    // The obstacles are static, so they are written ONCE, as their own file: a separate
    // object in the renderer, which is what gives it its own material.
    const bool sceneSolids = SCENE.loaded && SCENE.HasSolidRegions();
    if (SIMULATION.exports.exportSurfaceSolids && !SIMULATION.exports.surfaceSolidsWritten && sceneSolids)
    {
        const Domain &d = SIMULATION.domain;
        const double lo[3] = {d.x0, d.y0, d.z0}, hi[3] = {d.xf, d.yf, d.zf};
        MeshExport::TriMesh solids;
        std::string serr;
        if (MeshExport::SurfaceFromSDF(SceneSolidField, lo, hi, p.voxel, solids, serr))
        {
            const std::string sname = dir + "solids.ply";
            if (MeshExport::WritePLY(sname, solids, serr))
                SIMULATION.exports.surfaceSolidsWritten = true;
        }
        if (!SIMULATION.exports.surfaceSolidsWritten)
            SIMULATION.exports.surfaceStatus = "solids: " + serr;
    }

    char msg[256];
    std::snprintf(msg, sizeof(msg), "liquid_%05d.ply: %zu triangles%s", frame,
                  mesh.TriangleCount(),
                  SIMULATION.exports.surfaceSolidsWritten ? " (+ solids.ply)" : "");
    SIMULATION.exports.surfaceStatus = msg;
}

// The run's particle set as CSV, position and velocity of every live particle; nothing when there are none.
inline void WriteParticles(int frame)
{
    const ParticleSet &set = SIMULATION.particles;
    if (set.count <= 0) return;
    const std::string path = ExportDirectory(ActiveMethod().key) + "particles_" + std::to_string(frame) + ".csv";
    std::ofstream out(path);
    if (!out.is_open()) { std::cerr << "Particle export failed: " << path << std::endl; return; }
    out << (DIMENSION == 2 ? "x,y,u,v" : "x,y,z,u,v,w") << "\n";
    for (int i = 0; i < set.count; i++)
    {
        const Particle &p = set.data[(size_t)i];
        if (!p.isActive) continue;
        if (DIMENSION == 2) out << p.x << "," << p.y << "," << p.u << "," << p.v << "\n";
        else out << p.x << "," << p.y << "," << p.z << "," << p.u << "," << p.v << "," << p.w << "\n";
    }
}

inline void Update()
{
    if (!SIMULATION.exports.enabled) return;

    // Export telemetry at telemetry interval
    if ((SIMULATION.telemetry.iteration) % SIMULATION.exports.telemetryInterval == 0) {
        WriteTelemetryRow();
        SIMULATION.exports.lastExportedTelemetryFrame = SIMULATION.telemetry.iteration;
    }

    // Grid, particles and the surface mesh are ONE frame written three ways, so they share a single
    // decision.
    const bool frameDue = FrameExportDue(SIMULATION.telemetry.iteration, SIMULATION.telemetry.clock, SIMULATION.exports);
    const bool wantFrame = SIMULATION.exports.exportGridData || SIMULATION.exports.exportParticleData ||
                           SIMULATION.exports.exportSurfaceMesh || SIMULATION.exports.exportVolumetric;

    if (frameDue)
    {
        if (SIMULATION.exports.exportGridData)
            WriteGrid(*SIMULATION.GRID_SOL, SIMULATION.exports.gridExportCounter);

        if (SIMULATION.exports.exportParticleData)
            WriteParticles(SIMULATION.exports.gridExportCounter);

        if (SIMULATION.exports.exportSurfaceMesh)
            WriteSurface(SIMULATION.exports.gridExportCounter);

        if (SIMULATION.exports.exportVolumetric)
            WriteVolumetric(*SIMULATION.GRID_SOL, SIMULATION.exports.gridExportCounter, SIMULATION.telemetry.clock,
                                                  SIMULATION.exports.volumetricStatus);

        // The counter is the FRAME NUMBER, so it advances once per frame written and not once per
        // step: a renderer wants 1, 2, 3 rather than 1, 101, 201.
        if (wantFrame) {
            SIMULATION.exports.lastExportedGridFrame = SIMULATION.telemetry.iteration;
            SIMULATION.exports.gridExportCounter++;
        }
    }

    // Advanced whatever was enabled, so turning the exports on mid-run picks up the next
    // deadline rather than firing immediately for every one it slept through.
    if (frameDue) AdvanceFrameExportClock(SIMULATION.telemetry.clock, SIMULATION.exports);
}

// The final state, written once when a stopping rule fires, straight to the exporters (Update() writes only
// on its interval).
inline void WriteFinal()
{
    // The step that fired the rule may already have written its row on the interval.
    if ((long long)SIMULATION.exports.lastExportedTelemetryFrame != (long long)SIMULATION.telemetry.iteration)
        WriteTelemetryRow();
    WriteGrid(*SIMULATION.GRID_SOL, SIMULATION.exports.gridExportCounter);
    WriteParticles(SIMULATION.exports.gridExportCounter);
}

}  // namespace Exports

#endif
