"""Part: tower interior. Rest platforms under the flange joints and the top
platform (openings for the service lift, the ladder and, at the top, the cable
loop), ladder with fall-arrest rail per section, power cables, wall lamps, the
service lift (rails and car) and the ground controller cabinet.

Meshes are in their section frames (bottom of the section); the lift car in its
own frame at the car floor centre.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

s = c.site_dims
P = "tower_interior"
c.reset_scene()
ZS = [0.0] + list(s.TOWER_FLANGES) + [s.TOWER_HEIGHT]
R_IN = lambda z: s.tower_radius(z) - s.tower_wall(z)   # noqa: E731
LX, LY = s.LIFT_CENTER
CX, CY, CZ = s.LIFT_CAR


def polar(r, a, z):
    return (r * math.cos(a), r * math.sin(a), z)


def export(name, objs):
    c.export_part(P, [c.join(name, objs) if len(objs) > 1 else objs[0]], obj_name=name)


def section_of(z):
    return max(k for k in range(1, 5) if ZS[k - 1] <= z)


def ladder_point(z):
    return polar(R_IN(z) - s.LADDER_WALL_GAP, s.LADDER_AZIMUTH, z)


def lift_opening(z, z0):
    m = s.LIFT_SHAFT_MARGIN
    return c.box("lift_hole", (LX - CX / 2 - m, LX + CX / 2 + m), (LY - CY / 2 - m, LY + CY / 2 + m), (z - z0 - 1, z - z0 + 1))


# %% platforms (in the frame of the section they sit in)
for i, z in enumerate(s.platform_heights(), start=1):
    k = section_of(z)
    z0 = ZS[k - 1]
    local = z - z0
    r = R_IN(z) - 0.01
    deck = c.cylinder(f"platform{i}", r, s.PLATFORM_T, (0, 0, local - s.PLATFORM_T / 2), vertices=64, mat="grating")
    lx, ly, _ = ladder_point(z)
    holes = [lift_opening(z, z0), c.box("ladder_hole", (lx - 0.35, lx + 0.35), (ly - 0.35, ly + 0.35), (local - 1, local + 1))]
    if i == len(s.TOWER_FLANGES) + 1:
        holes.append(c.cylinder("cable_hole", s.CABLE_HOLE_R, 2.0, (0, 0, local)))
    c.cut(deck, *holes)
    parts = [deck]
    # railing around the lift shaft (open towards the car door, +X) and the ladder opening
    m = s.LIFT_SHAFT_MARGIN + 0.05
    x0, x1 = LX - CX / 2 - m, LX + CX / 2 + m
    y0, y1 = LY - CY / 2 - m, LY + CY / 2 + m
    for a, b in (((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))):
        parts.append(c.rod("rail", (a[0], a[1], local + 1.0), (b[0], b[1], local + 1.0), 0.02, mat="hatch_yellow"))
        parts.append(c.rod("rail_mid", (a[0], a[1], local + 0.5), (b[0], b[1], local + 0.5), 0.015, mat="hatch_yellow"))
    for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        parts.append(c.rod("post", (px, py, local), (px, py, local + 1.0), 0.02, mat="hatch_yellow"))
    for px, py in ((lx - 0.35, ly - 0.35), (lx + 0.35, ly - 0.35), (lx - 0.35, ly + 0.35)):
        parts.append(c.rod("ladder_post", (px, py, local), (px, py, local + 1.0), 0.02, mat="hatch_yellow"))
    # the lamp above the platform
    parts.append(c.box("lamp", (-0.15, 0.15), (R_IN(z + 2.0) - 0.08, R_IN(z + 2.0) - 0.02), (local + 2.0, local + 2.1), mat="lamp"))
    export(f"platform_{i}", parts)

# %% ladder with fall-arrest rail, cables and a lamp per section
for k in range(1, 5):
    z0, z1 = ZS[k - 1], ZS[k]
    lo, hi = ladder_point(z0 + 0.1), ladder_point(z1 - 0.05)
    tangent = (-math.sin(s.LADDER_AZIMUTH), math.cos(s.LADDER_AZIMUTH))
    w = s.LADDER_WIDTH / 2
    parts = []
    for side in (-1, 1):
        a = (lo[0] + side * w * tangent[0], lo[1] + side * w * tangent[1], lo[2] - z0)
        b = (hi[0] + side * w * tangent[0], hi[1] + side * w * tangent[1], hi[2] - z0)
        parts.append(c.rod("stile", a, b, 0.02, mat="pipe_steel", vertices=6))
    parts.append(c.rod("fall_arrest", (lo[0], lo[1], lo[2] - z0), (hi[0], hi[1], hi[2] - z0), 0.012, mat="hatch_yellow", vertices=6))
    n = int((z1 - z0 - 0.2) / s.RUNG_PITCH)
    for j in range(n):
        z = z0 + 0.2 + j * s.RUNG_PITCH
        p = ladder_point(z)
        parts.append(c.rod("rung", (p[0] - w * tangent[0], p[1] - w * tangent[1], z - z0),
                           (p[0] + w * tangent[0], p[1] + w * tangent[1], z - z0), 0.012, mat="pipe_steel", vertices=6))
    for dj, dy in enumerate((-0.08, 0.0, 0.08)):
        a = s.CABLE_AZIMUTH + dy / R_IN(z0)
        parts.append(c.rod("cable", polar(R_IN(z0) - 0.12, a, 0.0), polar(R_IN(z1) - 0.12, a, z1 - z0), 0.035, mat="cable_black", vertices=8))
    zm = (z0 + z1) / 2
    parts.append(c.box("lamp", (-0.15, 0.15), (R_IN(zm) - 0.08, R_IN(zm) - 0.02), (zm - z0, zm - z0 + 0.1), mat="lamp"))
    export(f"ladder_{k}", parts)

# cable loop from the yaw bearing down through the top platform to the wall (section 4 frame)
top = ZS[4] - ZS[3]
LOOP_END = polar(R_IN(ZS[4] - 10) - 0.12, s.CABLE_AZIMUTH, top - 10.0)
loop = [c.curve_pipe("loop", (0.0, 0.0, top), (0, 0, -3.0), LOOP_END, (0.6, 0.6, -0.2), 0.09, mat="cable_black"),
        c.cylinder("saddle", 0.2, 0.08, (0, 0, top - 2.5), mat="steel_grey")]
export("cable_loop", loop)


def loop_point(z):
    """Point on the cable loop's centre line at height z (section 4 frame)."""
    p0, p3 = (0.0, 0.0, top), LOOP_END
    p1 = (0.0, 0.0, top - 3.0)
    p2 = tuple(a - b for a, b in zip(p3, (0.6, 0.6, -0.2)))

    def at(t):
        return tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * cc + t ** 3 * e
                     for a, b, cc, e in zip(p0, p1, p2, p3))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if at(mid)[2] > z else (lo, mid)
    return at(lo)


# fault: insulation chafed through where the loop rubs on the saddle edge (copper showing)
cx_, cy_, cz_ = loop_point(top - s.TOP_PLATFORM_BELOW_TOP + 0.3)
chafe = [c.cylinder("bare_copper", 0.093, 0.14, (cx_, cy_, cz_), vertices=24, mat="copper")]
rng = random.Random(5)
for i in range(7):
    a = rng.uniform(0, 2 * math.pi)
    chafe.append(c.box(f"shred{i}", (-0.012, 0.012), (-0.004, 0.004), (-0.05, 0.05), mat="cable_black"))
    c.placed(chafe[-1], c.homogeneous_z(a, (cx_ + 0.1 * math.cos(a), cy_ + 0.1 * math.sin(a), cz_ + rng.uniform(-0.09, 0.09))))
# rubbed-off insulation on the deck around the cable opening
chafe.append(c.puddle("flakes", 0.16, (cx_ + 0.3, cy_ - 0.3, top - s.TOP_PLATFORM_BELOW_TOP + 0.002), seed=3, mat="bolt_black"))
export("fault_cable_chafe", chafe)
print("cable chafe at", round(cx_, 3), round(cy_, 3), round(cz_, 3))

# %% ground controller cabinet opposite the door (section 1 frame, ground at -BASE)
g = -s.TOWER_BASE_Z
yw = R_IN(0.5) - 0.05
export("ground_controller", [
    c.box("cabinet", (-0.45, 0.45), (yw - 0.55, yw), (g, g + 2.0), mat="cabinet_grey"),
    c.box("door_seam", (-0.004, 0.004), (yw - 0.556, yw - 0.55), (g + 0.1, g + 1.95), mat="bolt_black"),
    c.box("display", (0.15, 0.35), (yw - 0.56, yw - 0.55), (g + 1.4, g + 1.6), mat="bolt_black"),
    c.cylinder("estop", 0.035, 0.04, (-0.25, yw - 0.57, g + 1.3), rotation=(math.pi / 2, 0, 0), mat="alarm_red"),
])

# %% service lift: guide wires (section 1 frame, full height) and the car (car frame)
wires = []
for dy in (-CY / 2 - 0.06, CY / 2 + 0.06):
    wires.append(c.rod("guide_wire", (LX, LY + dy, g), (LX, LY + dy, s.TOWER_HEIGHT - 0.3), 0.008, mat="pipe_steel", vertices=6))
wires.append(c.rod("hoist_wire", (LX, LY, g + CZ), (LX, LY, s.TOWER_HEIGHT - 0.3), 0.006, mat="pipe_steel", vertices=6))
wires.append(c.box("top_anchor", (LX - 0.3, LX + 0.3), (LY - 0.55, LY + 0.55), (s.TOWER_HEIGHT - 0.35, s.TOWER_HEIGHT - 0.25), mat="steel_grey"))
export("lift_rails", wires)

car = [c.box("floor", (-CX / 2, CX / 2), (-CY / 2, CY / 2), (-0.04, 0.0), mat="grating"),
       c.box("roof", (-CX / 2, CX / 2), (-CY / 2, CY / 2), (CZ, CZ + 0.04), mat="hatch_yellow"),
       c.box("hoist", (-0.2, 0.2), (-0.2, 0.2), (CZ + 0.04, CZ + 0.3), mat="generator_paint")]
for px, py in ((-CX / 2, -CY / 2), (-CX / 2, CY / 2), (CX / 2, -CY / 2), (CX / 2, CY / 2)):
    car.append(c.rod("corner", (px, py, 0.0), (px, py, CZ), 0.02, mat="hatch_yellow"))
for z in (0.5, 1.0):   # waist and knee bars on the closed sides (the door side +X stays open)
    for a, b in (((-CX / 2, -CY / 2), (CX / 2, -CY / 2)), ((-CX / 2, CY / 2), (CX / 2, CY / 2)), ((-CX / 2, -CY / 2), (-CX / 2, CY / 2))):
        car.append(c.rod("bar", (a[0], a[1], z), (b[0], b[1], z), 0.015, mat="hatch_yellow"))
car.append(c.box("mesh_back", (-CX / 2 - 0.005, -CX / 2), (-CY / 2, CY / 2), (0.05, 1.1), mat="steel_grey"))
car.append(c.box("control_panel", (-CX / 2 + 0.01, -CX / 2 + 0.06), (0.1, 0.3), (1.1, 1.35), mat="bolt_black"))
export("lift_car", car)

c.save_blend("tower_interior")
print("built tower_interior")
