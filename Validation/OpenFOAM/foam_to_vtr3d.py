#!/usr/bin/env python3
"""Convert the latest time of a 3D OpenFOAM cavity case to NEREUS's grid export layout.

usage: foam_to_vtr3d.py CASE_DIR N OUT.vtr

The 3D twin of foam_to_vtr.py. NEREUS (Exports::WriteGrid) writes an (N+2)^3 VTK RectilinearGrid:
the N^3 fluid cells on [0,1]^3 plus the one-cell solid ring outside the domain. PointData Velocity
(3 components), Pressure, Solid, coordinates at the cell centres (k - 0.5)/N, k = 0..N+1, ordered z
then y then x. The OpenFOAM mesh is the fluid region, so each OpenFOAM cell is placed at
the NEREUS index its centre falls on and the ring is filled with boundary values: velocity
(1, 0, 0) on the lid plane and 0 elsewhere, pressure copied from the nearest interior cell,
Solid = 1.

Pressure is shifted to a zero mean over the interior, since NEREUS pins the mean of its
closed-box pressure while OpenFOAM fixes p = 0 at a reference cell. icoFoam's p is kinematic
(p/rho) and the NEREUS case runs DENSITY = 1, so the two are directly comparable.

At N = 128 this reads about 2.1 M cells and writes an ASCII file of a few hundred MB; the field
parsing is done with one numpy pass rather than the 2D script's per-tuple regex, which would take
minutes at this size.
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
    """The internalField as (count, components). One numpy parse over the whole block."""
    text = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(", text)
    if m is None:
        sys.exit("expected a nonuniform internalField in " + path)
    count = int(m.group(1))
    body = text[m.end():]
    # The list closes with ")\n;" -- take everything up to it, then drop the per-tuple parentheses
    # so the remainder is plain whitespace-separated numbers.
    end = body.find(")\n;")
    if end == -1:
        end = len(body)
    body = body[:end]
    if components > 1:
        body = body.replace("(", " ").replace(")", " ")
    values = np.fromstring(body, dtype=float, sep=" ")
    if values.size < count * components:
        sys.exit("read %d of %d values from %s" % (values.size, count * components, path))
    values = values[:count * components]
    return values.reshape(count, components) if components > 1 else values


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

    # NEREUS's 3D arrays are ordered [k, i, j] -- slab z, row y, column x.
    shape = (m, m, m)
    u = np.zeros(shape)
    v = np.zeros(shape)
    w = np.zeros(shape)
    pr = np.zeros(shape)
    solid = np.ones(shape, dtype=np.int32)

    j = np.rint(C[:, 0] / h + 0.5).astype(int)
    i = np.rint(C[:, 1] / h + 0.5).astype(int)
    k = np.rint(C[:, 2] / h + 0.5).astype(int)
    lo = min(j.min(), i.min(), k.min())
    hi = max(j.max(), i.max(), k.max())
    if lo < 1 or hi > n:
        sys.exit("cell centres do not map onto NEREUS's interior cells")

    u[k, i, j] = U[:, 0]
    v[k, i, j] = U[:, 1]
    w[k, i, j] = U[:, 2]
    pr[k, i, j] = p
    solid[k, i, j] = 0
    if (solid[1:n + 1, 1:n + 1, 1:n + 1] != 0).any():
        sys.exit("not every interior cell was filled")

    interior = pr[1:n + 1, 1:n + 1, 1:n + 1]
    interior -= interior.mean()

    # The ring: the lid drives +x on the y = max plane, every other wall is at rest, and the
    # pressure is carried out from the nearest interior cell so the ring is not a hole.
    u[:, m - 1, :] = 1.0
    idx = np.clip(np.arange(m), 1, n)
    pr = pr[np.ix_(idx, idx, idx)] * (solid != 0) + pr * (solid == 0)

    def flat(a):
        return a.reshape(-1)

    with open(out, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n')
        f.write('  <RectilinearGrid WholeExtent="0 %d 0 %d 0 %d">\n' % (m - 1, m - 1, m - 1))
        f.write('    <Piece Extent="0 %d 0 %d 0 %d">\n' % (m - 1, m - 1, m - 1))
        f.write('      <PointData>\n')
        f.write('        <DataArray type="Float64" Name="Velocity" NumberOfComponents="3" format="ascii">\n')
        uu, vv, ww = flat(u), flat(v), flat(w)
        step = m * m
        for s in range(0, uu.size, step):
            f.write(" ".join("%.12g %.12g %.12g" % t for t in zip(uu[s:s + step], vv[s:s + step], ww[s:s + step])) + " \n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Float64" Name="Pressure" format="ascii">\n')
        pp = flat(pr)
        for s in range(0, pp.size, step):
            f.write(" ".join("%.12g" % x for x in pp[s:s + step]) + " \n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="Solid" format="ascii">\n')
        ss = flat(solid)
        for s in range(0, ss.size, step):
            f.write(" ".join("%d" % x for x in ss[s:s + step]) + " \n")
        f.write('        </DataArray>\n')
        f.write('      </PointData>\n')
        f.write('      <Coordinates>\n')
        centres = " ".join("%.12g" % ((c - 0.5) * h) for c in range(m))
        for axis in ("X", "Y", "Z"):
            f.write('        <DataArray type="Float64" Name="%s" format="ascii">\n' % axis)
            f.write(centres + " \n        </DataArray>\n")
        f.write('      </Coordinates>\n')
        f.write('    </Piece>\n')
        f.write('  </RectilinearGrid>\n')
        f.write('</VTKFile>\n')


if __name__ == "__main__":
    main()
