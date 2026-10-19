#!/usr/bin/env python3
"""Convert the latest time of a 2D OpenFOAM cavity case to NEREUS's grid export layout.

usage: foam_to_vtr.py CASE_DIR N OUT.vtr

NEREUS (Exports::WriteGrid) writes an (N+2) x (N+2) VTK RectilinearGrid: the N x N fluid cells on [0,1]^2 plus the
one-cell solid ring OUTSIDE the domain (since 2026-09-28). PointData Velocity (3 components), Pressure, Solid, with the
X/Y coordinates at the cell centres (j - 0.5)/N, j = 0..N+1, and Z = 0, rows ordered y then x. The OpenFOAM mesh is
the N x N fluid region, so each OpenFOAM cell is placed at the NEREUS index its centre falls on, and the ring is
filled with boundary values:
velocity (1, 0) on the lid row and 0 elsewhere, pressure copied from the nearest interior cell, Solid = 1.

Pressure is shifted to a zero mean over the interior, since NEREUS pins the mean of its closed-box pressure while
OpenFOAM fixes p = 0 at a reference cell. icoFoam's p is kinematic (p/rho); NEREUS's cavity preset has DENSITY = 1,
so the two are directly comparable. NEREUS also writes a Divergence array; it is not written here.
"""
import os
import re
import sys

import numpy as np


def latest_time_dir(case):
    best, best_t = None, -1.0
    for name in os.listdir(case):
        path = os.path.join(case, name)
        if not os.path.isdir(path):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        if t > best_t and os.path.exists(os.path.join(path, "U")):
            best, best_t = path, t
    if best is None:
        sys.exit("no time directory with U in " + case)
    return best


def read_internal_field(path, components):
    text = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(", text)
    if m is None:
        sys.exit("expected a nonuniform internalField in " + path)
    count = int(m.group(1))
    body = text[m.end():]
    if components == 1:
        values = np.array(body.split(None, count)[:count], dtype=float)
    else:
        tuples = re.findall(r"\(([^()]*)\)", body)[:count]
        values = np.array([t.split() for t in tuples], dtype=float)
    if len(values) != count:
        sys.exit("read %d of %d values from %s" % (len(values), count, path))
    return values


def write_rows(f, fmt, rows):
    for row in rows:
        f.write(" ".join(fmt % v for v in row) + " \n")


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    case, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tdir = latest_time_dir(case)
    U = read_internal_field(os.path.join(tdir, "U"), 3)
    p = read_internal_field(os.path.join(tdir, "p"), 1)
    C = read_internal_field(os.path.join(tdir, "C"), 3)
    h = 1.0 / n
    m = n + 2                                   # fluid cells plus the ring on each side

    u = np.zeros((m, m))
    v = np.zeros((m, m))
    pr = np.zeros((m, m))
    solid = np.ones((m, m), dtype=int)
    j = np.rint(C[:, 0] / h + 0.5).astype(int)
    i = np.rint(C[:, 1] / h + 0.5).astype(int)
    if j.min() < 1 or j.max() > n or i.min() < 1 or i.max() > n:
        sys.exit("cell centres do not map onto NEREUS's interior cells")
    u[i, j] = U[:, 0]
    v[i, j] = U[:, 1]
    pr[i, j] = p
    solid[i, j] = 0
    if (solid[1:n + 1, 1:n + 1] != 0).any():
        sys.exit("not every interior cell was filled")

    pr[1:n + 1, 1:n + 1] -= pr[1:n + 1, 1:n + 1].mean()
    u[m - 1, :] = 1.0
    idx = np.clip(np.arange(m), 1, n)
    pr = pr[np.ix_(idx, idx)]

    with open(out, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n')
        f.write('  <RectilinearGrid WholeExtent="0 %d 0 %d 0 0">\n' % (m - 1, m - 1))
        f.write('    <Piece Extent="0 %d 0 %d 0 0">\n' % (m - 1, m - 1))
        f.write('      <PointData>\n')
        f.write('        <DataArray type="Float64" Name="Velocity" NumberOfComponents="3" format="ascii">\n')
        for a in range(m):
            f.write(" ".join("%.12g %.12g 0" % (u[a, b], v[a, b]) for b in range(m)) + " \n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Float64" Name="Pressure" format="ascii">\n')
        write_rows(f, "%.12g", pr)
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="Solid" format="ascii">\n')
        write_rows(f, "%d", solid)
        f.write('        </DataArray>\n')
        f.write('      </PointData>\n')
        f.write('      <Coordinates>\n')
        centres = " ".join("%.12g" % ((k - 0.5) * h) for k in range(m))
        f.write('        <DataArray type="Float64" Name="X" format="ascii">\n' + centres + " \n        </DataArray>\n")
        f.write('        <DataArray type="Float64" Name="Y" format="ascii">\n' + centres + " \n        </DataArray>\n")
        f.write('        <DataArray type="Float64" Name="Z" format="ascii">\n0 \n        </DataArray>\n')
        f.write('      </Coordinates>\n')
        f.write('    </Piece>\n')
        f.write('  </RectilinearGrid>\n')
        f.write('</VTKFile>\n')


if __name__ == "__main__":
    main()
