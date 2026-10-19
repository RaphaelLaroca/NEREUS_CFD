"""Draws the NEREUS logo: "NERE" in a bold serif, a trident for the U, a scaled sea serpent for the S, and a wave
underneath that the trident's shaft dips into.

    python3 docs/assets/make_logo.py

writes nereus_logo.svg / .png (dark ink, for light backgrounds) and nereus_logo_dark.svg / .png (light ink,
for dark backgrounds) beside this script. Only matplotlib and numpy are needed; the font is DejaVu Serif Bold,
which ships with matplotlib, so the result is the same on every machine.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Polygon, Ellipse, Circle
from matplotlib.path import Path
from matplotlib.textpath import TextPath

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = FontProperties(family="DejaVu Serif", weight="bold")

# The letters are laid out in units of the font's cap height H (the height of "E").
CAP = TextPath((0, 0), "E", size=1.0, prop=FONT).get_extents().height
SCALE = 1.0 / CAP
STEM = 0.18      # stroke width of the trident and the snake, in H (about the font's stem)
GAP = 0.10       # space between letters, in H


def text_path(s, x0):
    """The glyph outlines of `s` scaled to a cap height of 1, starting at x0. Returns the path and its right edge."""
    p = TextPath((0, 0), s, size=SCALE, prop=FONT)
    ext = p.get_extents()
    v = p.vertices.copy()
    v[:, 0] += x0 - ext.x0
    return Path(v, p.codes), x0 + ext.width


def spearhead(cx, tip, w):
    """A barbed spearhead centred on cx whose point is at height `tip`, for a shaft of width w."""
    h = 0.30
    return [(cx - w / 2, tip - h), (cx - w * 1.15, tip - h - 0.10), (cx, tip),
            (cx + w * 1.15, tip - h - 0.10), (cx + w / 2, tip - h)]


def trident_u(x0):
    """The U as a trident: the two outer prongs are the U's arms, the bowl is the crossbar, and the shaft runs
    below the baseline. Returns the polygons and the right edge."""
    w = STEM
    x0 += w * 0.65                   # the barbs overhang the arms: keep them clear of the previous letter
    width = 0.86                     # outer width of the U
    xl, xr = x0 + w / 2, x0 + width - w / 2   # centre lines of the two arms
    r = (xr - xl) / 2                # bowl radius (centre line)
    cy = 0.38                        # bowl centre height
    cx = (xl + xr) / 2
    polys = []
    # The bowl and arms as one thick stroke along a U-shaped centre line.
    t = np.linspace(np.pi, 2 * np.pi, 40)
    centre = [(xl, 0.86)] + [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t] + [(xr, 0.86)]
    polys.append(thick_stroke(np.array(centre), lambda s: w))
    # Barbed tips on the arms, and the shorter middle prong rising from the bowl.
    polys.append(spearhead(xl, 1.16, w))
    polys.append(spearhead(xr, 1.16, w))
    mid_w = w * 0.85
    polys.append([(cx - mid_w / 2, cy - r + 0.02), (cx + mid_w / 2, cy - r + 0.02),
                  (cx + mid_w / 2, 0.66), (cx - mid_w / 2, 0.66)])
    polys.append(spearhead(cx, 0.96, mid_w))
    # The shaft, below the bowl, reaching down into the wave.
    polys.append([(cx - w / 2, cy - r), (cx + w / 2, cy - r), (cx + w / 2, -0.40), (cx - w / 2, -0.40)])
    return polys, x0 + width + w * 0.65


def thick_stroke(c, width_at):
    """A polygon around the centre line c (N x 2) whose width at arc-length fraction s is width_at(s)."""
    d = np.gradient(c, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    s = np.linspace(0.0, 1.0, len(c))
    half = np.array([width_at(si) for si in s])[:, None] / 2
    left, right = c + n * half, c - n * half
    return np.vstack([left, right[::-1]])


def snake_s(x0):
    """The S as a snake: the head at the upper end, the body tapering to the tail at the lower end."""
    r = 0.25                          # radius of each loop's centre line
    cx = x0 + r + STEM / 2 + 0.02
    up = np.linspace(np.radians(25), np.radians(270), 60)     # upper loop, counter-clockwise
    lo = np.linspace(np.radians(90), np.radians(-155), 60)    # lower loop, clockwise
    c = np.vstack([np.stack([cx + r * np.cos(up), 0.75 + r * np.sin(up)], 1),
                   np.stack([cx + r * np.cos(lo), 0.25 + r * np.sin(lo)], 1)[1:]])
    def width(s):
        return STEM * (1.0 if s < 0.70 else max(0.08, 1.0 - (s - 0.70) / 0.30))   # the tail tapers
    body = thick_stroke(c, width)
    head_at = c[0]
    tangent = c[0] - c[3]
    angle = np.degrees(np.arctan2(tangent[1], tangent[0]))
    return body, head_at, angle, cx + r + STEM / 2, c, cx


def fins(ax, cx, ink):
    """Swept-back dorsal fins along the outside of the upper loop."""
    r0 = 0.25
    for a in np.radians([95, 125, 155, 185]):
        n = np.array([np.cos(a), np.sin(a)])
        t = np.array([-np.sin(a), np.cos(a)])
        b = np.array([cx, 0.75]) + n * (r0 + STEM / 2 - 0.01)
        ax.add_patch(Polygon([b - t * 0.06, b + t * 0.03, b + n * 0.075 - t * 0.075], closed=True, fc=ink, ec="none"))


def scales(ax, c, eye, step=0.055, rows=(-0.38, 0.0, 0.38), size=0.030):
    """Crescent scales cut into the body in the background colour, in staggered rows, thinning along the tail."""
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    total = arc[-1]
    th = np.radians(np.linspace(-90, 90, 20))
    for k, sa in enumerate(np.arange(0.16 * total, 0.80 * total, step)):
        i = min(max(np.searchsorted(arc, sa), 1), len(c) - 2)
        t = c[i + 1] - c[i - 1]
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        frac = sa / total
        width = 1.0 if frac < 0.70 else max(0.0, 1 - (frac - 0.70) / 0.30)
        for j, off in enumerate(rows):
            if (k + j) % 2 or abs(off) > 0.5 * width + 1e-9:
                continue
            q = c[i] + n * off * STEM
            pts = q + size * (np.cos(th)[:, None] * t + np.sin(th)[:, None] * n)
            ax.plot(pts[:, 0], pts[:, 1], color=eye, lw=1.1, solid_capstyle="round", zorder=3)


def fwd0(angle_deg):
    a = np.radians(angle_deg)
    return np.array([np.cos(a), np.sin(a)])


def draw(ink, eye, path_png, path_svg):
    fig = plt.figure(figsize=(6.4, 2.2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")

    nere, x = text_path("NERE", 0.0)
    ax.add_patch(PathPatch(nere, fc=ink, ec="none"))
    polys, x = trident_u(x + GAP)
    for p in polys:
        ax.add_patch(Polygon(p, closed=True, fc=ink, ec="none"))
    body, head, ang, x, centre, cx = snake_s(x + GAP)
    ax.add_patch(Polygon(body, closed=True, fc=ink, ec="none"))
    fins(ax, cx, ink)
    scales(ax, centre, eye)
    # Head: an elongated ellipse along the body's direction, a pointed snout, a swept-back horn and an eye.
    a = np.radians(ang)
    fwd, side = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    ax.add_patch(Ellipse(head + fwd * 0.06, 0.42, 0.24, angle=ang, fc=ink, ec="none"))
    ax.add_patch(Polygon([head + fwd * 0.22 + side * 0.05, head + fwd * 0.34, head + fwd * 0.22 - side * 0.06],
                         closed=True, fc=ink, ec="none"))
    ax.add_patch(Polygon([head - fwd * 0.05 + side * 0.09, head - fwd * 0.22 + side * 0.20, head - fwd * 0.10 + side * 0.02],
                         closed=True, fc=ink, ec="none"))
    ax.add_patch(Circle(head + fwd * 0.12 + side * 0.045, 0.03, fc=eye, ec="none"))
    x += 0.10

    # The underscore: a wave under the whole word, thickest in the middle and tapering at both ends.
    xe = x + 0.12
    xs = np.linspace(-0.02, xe, 400)
    ys = -0.30 + 0.04 * np.sin((xs + 0.1) * 2 * np.pi / 0.62)
    wave = thick_stroke(np.stack([xs, ys], 1), lambda s: 0.025 + 0.065 * np.sin(np.pi * s))
    ax.add_patch(Polygon(wave, closed=True, fc=ink, ec="none"))

    ax.set_xlim(-0.10, xe + 0.12)
    ax.set_ylim(-0.50, 1.30)
    for p in (path_png, path_svg):
        fig.savefig(p, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    draw("#0b3d5c", "#ffffff", os.path.join(HERE, "nereus_logo.png"), os.path.join(HERE, "nereus_logo.svg"))
    draw("#e6f1f8", "#0d1117", os.path.join(HERE, "nereus_logo_dark.png"),
         os.path.join(HERE, "nereus_logo_dark.svg"))
