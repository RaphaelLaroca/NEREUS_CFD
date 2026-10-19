// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CHECKPOINT_H
#define CHECKPOINT_H

// Saving and restoring a run: only what the next step reads that the config cannot rebuild -- the clock,
// GRID_ANT's velocity and scalars, and the particle set. No solver holds state across steps.

#include "Definitions.h"
#include "MAC.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace Checkpoint
{

constexpr char MAGIC[8] = {'F', 'L', 'U', 'I', 'D', 'C', 'K', 'P'};
constexpr uint32_t VERSION = 8;

struct Header
{
    char magic[8];
    uint32_t version;
    int32_t dimension;
    char method[MAX_METHOD_KEY];   // the method's key, not its registration id, which is only an order
    int32_t Nx, Ny, Nz;
    double faceSum[3]; // sum of each axis's face table: refuses a run on a different grid
    double time;
    uint64_t iteration;
};

inline double FaceSum(const std::vector<double> &faces)
{
    double s = 0.0;
    for (double x : faces) s += x;
    return s;
}

inline void WriteString(std::ostream &f, const std::string &s)
{
    const uint32_t n = (uint32_t)s.size();
    f.write((const char *)&n, sizeof(n));
    f.write(s.data(), (std::streamsize)n);
}

inline std::string ReadString(std::istream &f)
{
    uint32_t n = 0;
    f.read((char *)&n, sizeof(n));
    std::string s(n, '\0');
    if (n) f.read(&s[0], (std::streamsize)n);
    return s;
}

inline void WriteVec(std::ostream &f, const Eigen::VectorXd &v)
{
    const uint64_t n = (uint64_t)v.size();
    f.write((const char *)&n, sizeof(n));
    if (n) f.write((const char *)v.data(), (std::streamsize)(n * sizeof(double)));
}

inline bool ReadVec(std::istream &f, Eigen::VectorXd &v, const std::string &what, std::string &err)
{
    uint64_t n = 0;
    f.read((char *)&n, sizeof(n));
    if ((uint64_t)v.size() != n)
    {
        err = what + ": checkpoint holds " + std::to_string(n) + " values, this grid wants " +
              std::to_string(v.size());
        return false;
    }
    if (n) f.read((char *)v.data(), (std::streamsize)(n * sizeof(double)));
    return (bool)f;
}

// One record per live particle: isActive, then x y z u v w; then a flag and, when set, one density per particle.
inline void WriteParticles(std::ostream &f, const ParticleSet &ps)
{
    const uint64_t n = (ps.count > 0) ? (uint64_t)ps.count : 0;
    f.write((const char *)&n, sizeof(n));
    for (uint64_t i = 0; i < n; i++)
    {
        const Particle &p = ps.data[(size_t)i];
        const uint8_t a = p.isActive ? 1 : 0;
        const double xyzuvw[6] = {p.x, p.y, p.z, p.u, p.v, p.w};
        f.write((const char *)&a, sizeof(a));
        f.write((const char *)xyzuvw, sizeof(xyzuvw));
    }
    const uint8_t hasDensity = (n > 0 && ps.density.size() >= n) ? 1 : 0;
    f.write((const char *)&hasDensity, sizeof(hasDensity));
    if (hasDensity) f.write((const char *)ps.density.data(), (std::streamsize)(n * sizeof(double)));
}

inline bool ReadParticles(std::istream &f, ParticleSet &ps, std::string &err)
{
    uint64_t n = 0;
    f.read((char *)&n, sizeof(n));
    if (!f) { err = "truncated before the particle count"; return false; }
    if (n > (uint64_t)ps.data.size())
    {
        err = "checkpoint holds " + std::to_string(n) + " particles, this run allocated room for " +
              std::to_string(ps.data.size());
        return false;
    }
    for (uint64_t i = 0; i < n; i++)
    {
        Particle &p = ps.data[(size_t)i];
        uint8_t a = 0;
        double xyzuvw[6] = {0, 0, 0, 0, 0, 0};
        f.read((char *)&a, sizeof(a));
        f.read((char *)xyzuvw, sizeof(xyzuvw));
        p.isActive = (a != 0);
        p.x = xyzuvw[0]; p.y = xyzuvw[1]; p.z = xyzuvw[2];
        p.u = xyzuvw[3]; p.v = xyzuvw[4]; p.w = xyzuvw[5];
    }
    uint8_t hasDensity = 0;
    f.read((char *)&hasDensity, sizeof(hasDensity));
    if (!f) { err = "truncated inside the particle array"; return false; }
    if (hasDensity)
    {
        if (ps.density.size() < n)
        { err = "checkpoint carries particle densities this run's method does not"; return false; }
        f.read((char *)ps.density.data(), (std::streamsize)(n * sizeof(double)));
        if (!f) { err = "truncated inside the particle densities"; return false; }
    }
    ps.count = (int)n;
    return true;
}

// ---- save / load ----
inline bool Save(const std::string &path, double time, uint64_t iteration, std::string &err)
{
    MAC *ant = SIMULATION.GRID_ANT;
    if (ant == nullptr) { err = "no grid to save"; return false; }

    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f.is_open()) { err = "could not open " + tmp; return false; }

    Header h{};
    std::memcpy(h.magic, MAGIC, sizeof(MAGIC));
    h.version = VERSION;
    h.dimension = DIMENSION;
    std::strncpy(h.method, ActiveMethod().key, sizeof(h.method) - 1);
    h.Nx = ant->Nx; h.Ny = ant->Ny; h.Nz = ant->Nz;
    h.faceSum[0] = FaceSum(ant->Faces(MAC::X)); h.faceSum[1] = FaceSum(ant->Faces(MAC::Y)); h.faceSum[2] = FaceSum(ant->Faces(MAC::Z));
    h.time = time;
    h.iteration = iteration;
    f.write((const char *)&h, sizeof(h));

    WriteVec(f, ant->Field(MAC::U)); WriteVec(f, ant->Field(MAC::V)); WriteVec(f, ant->Field(MAC::W));
    WriteVec(f, ant->Field(MAC::P));   // the incremental projection reads p^n
    const uint32_t ns = (uint32_t)ant->ScalarCount();
    f.write((const char *)&ns, sizeof(ns));
    for (int s = 0; s < (int)ns; s++)
    {
        WriteString(f, ant->ScalarName(s));
        WriteVec(f, ant->ScalarField(s));
    }
    WriteParticles(f, SIMULATION.particles);

    f.close();
    if (!f) { err = "write failed on " + tmp; std::remove(tmp.c_str()); return false; }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
    { err = "could not rename " + tmp + " to " + path; return false; }
    return true;
}

inline bool Load(const std::string &path, double &time, uint64_t &iteration, std::string &err)
{
    MAC *sol = SIMULATION.GRID_SOL, *ant = SIMULATION.GRID_ANT;
    if (sol == nullptr || ant == nullptr) { err = "no grids to load into"; return false; }

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { err = "could not open " + path; return false; }

    Header h{};
    f.read((char *)&h, sizeof(h));
    if (!f) { err = path + ": too short to hold a header"; return false; }
    if (std::memcmp(h.magic, MAGIC, sizeof(MAGIC)) != 0) { err = path + ": not a checkpoint"; return false; }
    if (h.version != VERSION)
    { err = path + ": checkpoint version " + std::to_string(h.version) + ", this build reads " +
            std::to_string(VERSION); return false; }
    if (h.dimension != DIMENSION || std::string(h.method, strnlen(h.method, sizeof(h.method))) != ActiveMethod().key)
    { err = path + ": saved for a different dimension or solver"; return false; }
    if (h.Nx != ant->Nx || h.Ny != ant->Ny || h.Nz != ant->Nz)
    { err = path + ": saved on " + std::to_string(h.Nx) + "x" + std::to_string(h.Ny) + "x" +
            std::to_string(h.Nz) + ", this run is " + std::to_string(ant->Nx) + "x" +
            std::to_string(ant->Ny) + "x" + std::to_string(ant->Nz); return false; }
    const double live[3] = {FaceSum(ant->Faces(MAC::X)), FaceSum(ant->Faces(MAC::Y)), FaceSum(ant->Faces(MAC::Z))};
    for (int a = 0; a < 3; a++)
        if (std::fabs(h.faceSum[a] - live[a]) > 1e-9 * (1.0 + std::fabs(live[a])))
        { err = path + ": saved on a grid with different spacing"; return false; }

    if (!(ReadVec(f, ant->Field(MAC::U), "u", err) && ReadVec(f, ant->Field(MAC::V), "v", err) && ReadVec(f, ant->Field(MAC::W), "w", err) &&
          ReadVec(f, ant->Field(MAC::P), "p", err)))
        return false;
    uint32_t ns = 0;
    f.read((char *)&ns, sizeof(ns));
    if (!f) { err = path + ": truncated before the scalar count"; return false; }
    if ((int)ns != ant->ScalarCount())
    { err = path + ": checkpoint carries " + std::to_string(ns) + " scalar field(s), this run registered " +
            std::to_string(ant->ScalarCount()); return false; }
    for (uint32_t s = 0; s < ns; s++)
    {
        const std::string name = ReadString(f);
        const int dest = ant->ScalarHandle(name);
        if (dest < 0)
        { err = path + ": checkpoint holds a scalar '" + name + "' this run did not register"; return false; }
        if (!ReadVec(f, ant->ScalarField(dest), name, err)) return false;
    }
    sol->CopyGrid(*ant);
    if (!ReadParticles(f, SIMULATION.particles, err)) { err = path + ": " + err; return false; }

    time = h.time;
    iteration = h.iteration;
    SIMULATION.time = h.time;
    SIMULATION.exports.telemetryResumeIteration = (long long)h.iteration;   // the CSV keeps its rows up to here
    return true;
}

inline bool Describe(const std::string &path, std::string &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    Header h{};
    f.read((char *)&h, sizeof(h));
    if (!f || std::memcmp(h.magic, MAGIC, sizeof(MAGIC)) != 0) return false;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%dD %s  %dx%dx%d  t = %.6f  step %llu",
                  h.dimension, std::string(h.method, strnlen(h.method, sizeof(h.method))).c_str(),
                  h.Nx, h.Ny, h.Nz, h.time, (unsigned long long)h.iteration);
    out = buf;
    return true;
}

} // namespace Checkpoint

#endif
