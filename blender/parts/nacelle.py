"""Part 1: nacelle cover (GRP shell), floor grating and the tower access hatch.

Link frame ``nacelle``: origin on the yaw axis at floor level, X upwind
(towards the hub), Z up. Sized after a ~2 MW geared turbine (Vestas V90:
10.4 m x 3.5 m nacelle).
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

d = c.dims
c.reset_scene()

# --- cover: outer box minus inner box, open at the bottom -------------------
X, Y, Z = d.NACELLE_X, d.NACELLE_Y, d.NACELLE_Z
shell = c.box("nacelle_cover", X, Y, Z, mat="grp_white")
inner = c.box("inner", d.INNER_X, d.INNER_Y, (Z[0] - 0.1, Z[1] - d.WALL))
bevel = shell.modifiers.new("round_roof", "BEVEL")
bevel.width, bevel.segments = 0.25, 4
c.bpy.context.view_layer.objects.active = shell
c.bpy.ops.object.modifier_apply(modifier=bevel.name)
shaft_hole = c.cylinder("shaft_hole", d.SHAFT_HOLE_R, 0.5, (X[1], 0, d.SHAFT_HOLE_Z), rotation=(0, 1.5708, 0))
roof_hatch = c.box("roof_hatch", d.ROOF_HATCH_X, d.ROOF_HATCH_Y, (Z[1] - 0.5, Z[1] + 0.5))
c.cut(shell, inner, shaft_hole, roof_hatch)

# --- floor grating with the tower access opening ----------------------------
floor = c.box("nacelle_floor", d.INNER_X, d.INNER_Y, (-d.FLOOR, 0.0), mat="steel_grey")
opening = c.box("floor_opening", d.FLOOR_HATCH_X, d.FLOOR_HATCH_Y, (-1, 1))
c.cut(floor, opening)

c.export_part("nacelle", [shell], obj_name="cover")
c.export_part("nacelle", [floor], obj_name="floor")

# --- hatch lid, built in its hinge frame (hinge along X at the opening's +Y edge)
hx, hy = d.FLOOR_HATCH_X, d.FLOOR_HATCH_Y
half_len = (hx[1] - hx[0]) / 2
lid = c.box("floor_hatch_lid", (-half_len, half_len), (hy[0] - hy[1], 0.0), (0.0, d.HATCH_THICKNESS), mat="hatch_yellow")
c.export_part("nacelle", [lid], obj_name="floor_hatch")

# --- weather mast on the roof: anemometer, wind vane, aviation light (variants)
roof = Z[1]
mx, top = d.MAST_X, Z[1] + d.MAST_HEIGHT
mast = [c.rod("mast", (mx, 0, roof), (mx, 0, top), 0.045, mat="steel_grey"),
        c.rod("crossbar", (mx, -d.MAST_ARM, top), (mx, d.MAST_ARM, top), 0.03, mat="steel_grey"),
        c.box("mast_foot", (mx - 0.15, mx + 0.15), (-0.15, 0.15), (roof - 0.02, roof + 0.03), mat="steel_grey")]
c.export_part("nacelle", [c.join("weather_mast", mast)], obj_name="weather_mast")


def anemometer(damaged):
    """Cup anemometer on the +Y end of the crossbar. Damaged: one cup torn off,
    one arm bent down."""
    cx, cy, cz = mx, d.MAST_ARM, top
    objs = [c.rod("spindle", (cx, cy, cz), (cx, cy, cz + 0.18), 0.015, mat="bolt_black")]
    for k in range(3):
        a = 2 * math.pi * k / 3
        droop = -0.07 if damaged and k == 1 else 0.0
        end = (cx + 0.16 * math.cos(a), cy + 0.16 * math.sin(a), cz + 0.18 + droop)
        objs.append(c.rod(f"arm{k}", (cx, cy, cz + 0.18), end, 0.006, mat="bolt_black"))
        if damaged and k == 2:
            continue
        c.bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=end)
        cup = c.bpy.context.active_object
        cup.data.materials.append(c.material("grp_white"))
        objs.append(cup)
    return objs


def wind_vane():
    cx, cy, cz = mx, -d.MAST_ARM, top
    return [c.rod("vane_spindle", (cx, cy, cz), (cx, cy, cz + 0.12), 0.012, mat="bolt_black"),
            c.rod("vane_tail", (cx + 0.12, cy, cz + 0.12), (cx - 0.25, cy, cz + 0.12), 0.008, mat="bolt_black"),
            c.box("vane_fin", (cx - 0.3, cx - 0.14), (cy - 0.004, cy + 0.004), (cz + 0.06, cz + 0.2), mat="grp_white")]


def aviation_light(broken):
    """Red obstruction light on top of the mast. Broken: dark dome, a piece missing."""
    base = c.cylinder("light_base", 0.07, 0.06, (mx, 0, top + 0.06), mat="steel_grey")
    c.bpy.ops.mesh.primitive_uv_sphere_add(radius=0.075, location=(mx, 0, top + 0.09))
    dome = c.bpy.context.active_object
    dome.data.materials.append(c.material("dead_glass" if broken else "light_red"))   # before cutting
    c.cut(dome, c.box("below", (mx - 1, mx + 1), (-1, 1), (top - 1, top + 0.09)))
    if broken:
        c.cut(dome, c.box("shard", (mx - 0.02, mx + 0.1), (-0.1, -0.02), (top + 0.1, top + 0.2)))
    return [base, dome]


for name, objs in (("anemometer_ok", anemometer(False) + wind_vane()),
                   ("anemometer_damaged", anemometer(True) + wind_vane()),
                   ("aviation_light_ok", aviation_light(False)),
                   ("aviation_light_broken", aviation_light(True))):
    c.export_part("nacelle", [c.join(name, objs)], obj_name=name)

# --- crack in the GRP cover on the -Y side wall (overlay)
import random  # noqa: E402
rng = random.Random(7)
cx, cz = d.COVER_CRACK_XZ
pts = [(cx, cz)]
for _ in range(9):
    pts.append((pts[-1][0] + rng.uniform(0.04, 0.16), pts[-1][1] + rng.uniform(-0.12, 0.08)))
crack = [c.rod(f"crack{i}", (x0, Y[0] - 0.004, z0), (x1, Y[0] - 0.004, z1), 0.009, mat="crack", vertices=4)
         for i, ((x0, z0), (x1, z1)) in enumerate(zip(pts, pts[1:]))]
crack.append(c.rod("branch", (pts[4][0], Y[0] - 0.004, pts[4][1]), (pts[4][0] + 0.1, Y[0] - 0.004, pts[4][1] + 0.18), 0.006, mat="crack", vertices=4))
c.export_part("nacelle", [c.join("fault_cover_crack", crack)], obj_name="fault_cover_crack")

c.save_blend("nacelle")
print("built nacelle")
