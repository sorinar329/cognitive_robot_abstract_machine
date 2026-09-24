"""Outside, ground level up to the tower top: ground, foundation plinth with grout
and earthing, 4 tower sections (base flange with anchor nuts, door, internal
flanges), entrance door leaf, outside stairs, yaw bearing ring. Plus variants and
fault overlays for the foundation and the tower coating.

Frames: ``foundation`` = world (ground level on the tower axis); ``tower_section_k``
at the bottom of section k; ``tower_door`` at its hinge; ``yaw_bearing`` at the
tower top. Dimensions: turbine.dims.site (tower from IEA-3.4-130-RWT).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

s = c.site_dims
c.reset_scene()
P = "site"
R = s.tower_radius
BASE = s.TOWER_BASE_Z


def export(name, objs):
    c.export_part(P, [c.join(name, objs) if len(objs) > 1 else objs[0]], obj_name=name)


def annulus(name, z0, z1, r_in, r_out, mat, segments=96):
    return c.revolve(name, [(z0, r_in), (z0, r_out), (z1, r_out), (z1, r_in), (z0, r_in)],
                     segments=segments, mat=mat, close_ends=(False, False), smooth=False)


def polar(r, a, z):
    return (r * math.cos(a), r * math.sin(a), z)


# %% ground (world frame)
grass = c.cylinder("grass", s.GROUND_RADIUS, 0.02, (0, 0, -0.02), vertices=96, mat="grass")
apron = annulus("apron", -0.012, 0.0, s.PLINTH_R - 0.01, 9.0, "gravel", segments=96)
pad = c.box("crane_pad", s.CRANE_PAD_X, s.CRANE_PAD_Y, (-0.012, 0.0), mat="gravel")
road = c.box("road", (s.CRANE_PAD_X[1], s.GROUND_RADIUS - 2), (14.0, 14.0 + s.ROAD_WIDTH), (-0.012, -0.001), mat="gravel")
export("ground", [grass, apron, pad, road])

# %% foundation (world frame)
top = s.PLINTH_Z[1]
plinth = c.revolve("plinth", [(s.PLINTH_Z[0], s.PLINTH_R), (top - 0.05, s.PLINTH_R), (top, s.PLINTH_R - 0.05)],
                   segments=96, mat="concrete", smooth=False)
export("foundation", [plinth])


def grout(broken):
    ring = annulus("grout", s.GROUT_Z[0], s.GROUT_Z[1], s.GROUT_R[0], s.GROUT_R[1], "grout")
    if not broken:
        return [ring]
    a = math.radians(-30)
    chunk = c.box("chunk", (-0.2, 0.2), (-0.18, 0.18), (s.GROUT_Z[0] - 0.01, s.GROUT_Z[1] + 0.01))
    c.placed(chunk, c.homogeneous_z(a, polar(s.GROUT_R[1], a, 0)))
    c.cut(ring, chunk)
    rng = random.Random(3)
    debris = []
    for i in range(5):
        d = c.box(f"debris{i}", (-0.03, 0.03), (-0.02, 0.025), (0, 0.025), mat="grout")
        aa = a + rng.uniform(-0.06, 0.06)
        rr = s.GROUT_R[1] + rng.uniform(0.08, 0.4)
        c.placed(d, c.homogeneous_z(rng.uniform(0, 3), polar(rr, aa, top + 0.001)))
        debris.append(d)
    return [ring] + debris


def earthing(loose):
    a = s.EARTHING_AZIMUTH
    flange_side = s.BASE_FLANGE_R_OUT + 0.012
    pts = [polar(flange_side, a, BASE + 0.06), polar(flange_side + 0.02, a, top + 0.012),
           polar(s.PLINTH_R - 0.06, a, top + 0.012), polar(s.PLINTH_R + 0.012, a, top - 0.06),
           polar(s.PLINTH_R + 0.012, a, 0.12)]
    objs = [c.box("terminal", (-0.05, 0.05), (-0.04, 0.04), (0.02, 0.16), mat="steel_grey")]
    c.placed(objs[0], c.homogeneous_z(a, polar(s.PLINTH_R + 0.04, a, 0.0)))
    if loose:   # torn off the flange: the strap ends on the plinth top, curled
        pts = pts[2:][::-1] + [polar(s.PLINTH_R - 0.25, a + 0.03, top + 0.012),
                               polar(s.PLINTH_R - 0.4, a - 0.02, top + 0.03)]
        pts = pts[::-1]
        objs.append(c.cylinder("empty_hole", 0.012, 0.004, (0, 0, 0), mat="crack"))
        c.placed(objs[-1], c.homogeneous_x_to(a, polar(s.BASE_FLANGE_R_OUT + 0.002, a, BASE + 0.06)))
    else:
        objs.append(c.cylinder("lug_bolt", 0.018, 0.02, (0, 0, 0), vertices=6, mat="bolt_black"))
        c.placed(objs[-1], c.homogeneous_x_to(a, polar(s.BASE_FLANGE_R_OUT + 0.02, a, BASE + 0.06)))
    for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
        objs.append(c.rod(f"strap{i}", p0, p1, 0.011, mat="copper"))
    return objs


for name, objs in (("grout_ok", grout(False)), ("grout_broken", grout(True)),
                   ("earthing_ok", earthing(False)), ("earthing_loose", earthing(True))):
    export(name, objs)

# plinth crack: a jagged line down the plinth side and a short way across its top
rng = random.Random(11)
a0, pts = math.radians(-45), []
z = top - 0.01
while z > 0.0:
    pts.append((s.PLINTH_R + 0.004, a0 + rng.uniform(-0.012, 0.012), z))
    z -= 0.06
crack = []
for i, ((r0, a1, z1), (r1, a2, z2)) in enumerate(zip(pts, pts[1:])):
    crack.append(c.rod(f"crack{i}", polar(r0, a1, z1), polar(r1, a2, z2), 0.011, mat="crack", vertices=4))
r_top = s.PLINTH_R - 0.05
for i in range(6):
    r1, r2 = r_top - i * 0.07, r_top - (i + 1) * 0.07
    crack.append(c.rod(f"crack_top{i}", polar(r1, a0 + rng.uniform(-0.01, 0.01), top + 0.002),
                       polar(r2, a0 + rng.uniform(-0.01, 0.01), top + 0.002), 0.009, mat="crack", vertices=4))
export("fault_plinth_crack", crack)

# %% tower sections (each in its own frame at the section bottom)
zs_all = [0.0] + list(s.TOWER_FLANGES) + [s.TOWER_HEIGHT]


def shell(name, z0, z1, mat="tower_paint"):
    zs = [z0] + [z for z in s.TOWER_Z if z0 < z < z1] + [z1]
    n = s.TOWER_SEGMENTS
    verts, faces = [], []
    for z in zs:
        for k in range(n):
            verts.append(polar(R(z), 2 * math.pi * k / n, z - z0))
        for k in range(n):
            verts.append(polar(R(z) - s.tower_wall(z), 2 * math.pi * k / n, z - z0))
    for i in range(len(zs) - 1):
        o, o2 = i * 2 * n, (i + 1) * 2 * n
        for k in range(n):
            k2 = (k + 1) % n
            faces.append((o + k, o + k2, o2 + k2, o2 + k))                       # outer
            faces.append((o + n + k2, o + n + k, o2 + n + k, o2 + n + k2))       # inner
    last = (len(zs) - 1) * 2 * n
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((k2, k, n + k, n + k2))                                     # bottom annulus
        faces.append((last + k, last + k2, last + n + k2, last + n + k))         # top annulus
    return c.mesh_object(name, verts, faces, mat)


for idx, (z0, z1) in enumerate(zip(zs_all, zs_all[1:]), start=1):
    wall = shell(f"section{idx}", z0, z1)
    objs = []
    h = z1 - z0
    if idx == 1:   # entrance door through the wall, with a reinforcing collar
        rd = R(s.DOOR_SILL_Z)
        hole = c.box("door_hole", (-s.DOOR_WIDTH / 2, s.DOOR_WIDTH / 2), (-rd - 0.5, -rd + 0.5),
                     (s.DOOR_SILL_Z, s.DOOR_SILL_Z + s.DOOR_HEIGHT))
        c.cut(wall, hole)
        collar = annulus("collar", s.DOOR_SILL_Z - 0.2, s.DOOR_SILL_Z + s.DOOR_HEIGHT + 0.2, rd - 0.09, rd + 0.06, "tower_paint")
        c.intersect(collar, c.box("collar_zone", (-s.DOOR_WIDTH / 2 - 0.2, s.DOOR_WIDTH / 2 + 0.2), (-rd - 1, -rd + 1), (-1, 10)))
        c.cut(collar, c.box("door_hole2", (-s.DOOR_WIDTH / 2, s.DOOR_WIDTH / 2), (-rd - 1, -rd + 1),
                            (s.DOOR_SILL_Z, s.DOOR_SILL_Z + s.DOOR_HEIGHT)))
        objs.append(collar)
        # T-flange on the grout with the anchor bolt nuts outside the wall
        objs.append(annulus("base_flange", 0.0, s.BASE_FLANGE_H, R(0) - s.tower_wall(0) - 0.25, s.BASE_FLANGE_R_OUT, "steel_grey"))
        for k in range(s.ANCHOR_NUTS):
            a = 2 * math.pi * k / s.ANCHOR_NUTS
            x, y, _ = polar(s.ANCHOR_NUT_R, a, 0)
            objs.append(c.cylinder(f"nut{k}", 0.034, 0.045, (x, y, s.BASE_FLANGE_H + 0.0225), vertices=6, mat="bolt_black"))
            objs.append(c.cylinder(f"stud{k}", 0.018, 0.06, (x, y, s.BASE_FLANGE_H + 0.075), vertices=8, mat="pipe_steel"))
    if idx > 1:    # internal L-flange at the bottom
        ri = R(z0) - s.tower_wall(z0)
        objs.append(annulus("flange_bottom", 0.0, s.FLANGE_H, ri - s.FLANGE_WIDTH, ri + 0.001, "steel_grey"))
    ri = R(z1) - s.tower_wall(z1)
    objs.append(annulus("flange_top", h - s.FLANGE_H, h, ri - s.FLANGE_WIDTH, ri + 0.001, "steel_grey"))
    if idx < 4:    # painted seam where the next section sits on
        objs.append(annulus("seam", h - 0.015, h + 0.015, R(z1) - 0.01, R(z1) + 0.004, "steel_grey"))
    export(f"tower_section_{idx}", [wall] + objs)

# door leaf, in its hinge frame (vertical axis at the +X edge of the opening)
rd = R(s.DOOR_SILL_Z)
leaf = annulus("leaf", s.DOOR_SILL_Z + 0.01, s.DOOR_SILL_Z + s.DOOR_HEIGHT - 0.01, rd - 0.03, rd + 0.035, "tower_paint")
c.intersect(leaf, c.box("leaf_zone", (-s.DOOR_WIDTH / 2 + 0.01, s.DOOR_WIDTH / 2 - 0.01), (-rd - 1, -rd + 1), (-1, 10)))
handle = c.box("handle", (-s.DOOR_WIDTH / 2 + 0.08, -s.DOOR_WIDTH / 2 + 0.2), (-rd - 0.09, -rd - 0.035),
               (s.DOOR_SILL_Z + 1.02, s.DOOR_SILL_Z + 1.06), mat="bolt_black")
door = c.join("tower_door", [leaf, handle])
hinge_y = -math.sqrt(rd ** 2 - (s.DOOR_WIDTH / 2) ** 2)
c.placed(door, c.translation((-s.DOOR_WIDTH / 2, -hinge_y, -s.DOOR_SILL_Z)))
export("tower_door", [door])

# outside stairs and landing (section 1 frame: ground is at -BASE)
ground = -BASE
y_land = -rd + 0.05
y_edge = y_land - s.LANDING_DEPTH
hw = s.STAIR_WIDTH / 2
n_steps = round((s.DOOR_SILL_Z - ground) / s.STEP_RISE)
rise = (s.DOOR_SILL_Z - ground) / n_steps
parts = [c.box("landing", (-hw - 0.1, hw + 0.1), (y_edge, y_land), (s.DOOR_SILL_Z - 0.08, s.DOOR_SILL_Z), mat="steel_grey")]
for x in (-hw - 0.05, hw + 0.05):
    for y in (y_edge + 0.05, y_land - 0.1):
        parts.append(c.rod("post", (x, y, ground), (x, y, s.DOOR_SILL_Z - 0.08), 0.04, mat="steel_grey"))
for k in range(1, n_steps):
    y1 = y_edge - (n_steps - 1 - k) * s.STEP_RUN
    parts.append(c.box(f"step{k}", (-hw, hw), (y1 - s.STEP_RUN, y1), (ground + k * rise - 0.04, ground + k * rise), mat="steel_grey"))
y_bottom = y_edge - (n_steps - 1) * s.STEP_RUN
for x in (-hw - 0.02, hw + 0.02):
    parts.append(c.rod("stringer", (x, y_bottom, ground + 0.05), (x, y_edge, s.DOOR_SILL_Z - 0.05), 0.03, mat="steel_grey"))
    rail_h = 1.0
    parts.append(c.rod("rail_stairs", (x, y_bottom, ground + rail_h), (x, y_edge, s.DOOR_SILL_Z + rail_h), 0.022, mat="pipe_steel"))
    parts.append(c.rod("rail_landing", (x, y_edge, s.DOOR_SILL_Z + rail_h), (x, y_land - 0.05, s.DOOR_SILL_Z + rail_h), 0.022, mat="pipe_steel"))
    for y, z0 in ((y_bottom, ground), (y_edge, s.DOOR_SILL_Z), (y_land - 0.05, s.DOOR_SILL_Z)):
        parts.append(c.rod("rail_post", (x, y, z0), (x, y, z0 + rail_h), 0.02, mat="pipe_steel"))
export("entrance_stairs", parts)

# tower coating damage near the base (section 1 frame): rust patches and a run-off streak
rust = [c.cylinder_blob(f"rust{i}", R, math.radians(a), z, sz, seed, "rust")
        for i, (a, z, sz, seed) in enumerate(((-58, 0.9, 0.16, 1), (-52, 1.5, 0.1, 2), (-63, 2.1, 0.07, 3), (-49, 0.55, 0.06, 4)))]
rust.append(c.cylinder_blob("streak", R, math.radians(-58), 0.45, 0.05, 5, "rust", stretch=(0.5, 8.0)))
export("fault_tower_coating", rust)

# anchor nut corrosion (section 1 frame): rusted nuts in one sector plus a streak on the plinth
corroded = []
for k in range(54, 60):
    a = 2 * math.pi * k / s.ANCHOR_NUTS
    x, y, _ = polar(s.ANCHOR_NUT_R, a, 0)
    corroded.append(c.cylinder(f"rust_nut{k}", 0.037, 0.05, (x, y, s.BASE_FLANGE_H + 0.024), vertices=7, mat="rust"))
a = 2 * math.pi * 57 / s.ANCHOR_NUTS
corroded.append(c.rod("rust_run", polar(s.BASE_FLANGE_R_OUT + 0.004, a, 0.1), polar(s.BASE_FLANGE_R_OUT + 0.004, a, 0.0), 0.02, mat="rust"))
corroded.append(c.rod("rust_run2", polar(s.BASE_FLANGE_R_OUT + 0.02, a, -0.055), polar(s.PLINTH_R - 0.1, a, -0.055), 0.025, mat="rust"))
export("fault_anchor_corrosion", corroded)

# %% yaw bearing ring (frame at the tower top)
export("yaw_bearing", [annulus("yaw_ring", 0.0, s.YAW_BEARING_H, s.YAW_BEARING_R[0], s.YAW_BEARING_R[1], "gear_paint")])

c.save_blend("site")
print("built site")
