"""Part: 3-stage gearbox (1 planetary + 2 helical) with torque arms, oil cooler,
offline oil filter, sight glass and a bolted inspection cover.

Link frame ``gearbox``: on the input axis at the front face of the input stub,
X upwind, Z up. Outputs:

  gearbox.obj                          housing, arms, bushings, cooler, filter, pipe
  sight_glass_{ok,low}.obj             oil level variants
  filter_indicator_{ok,clogged}.obj    clogging indicator variants
  cover_bolts_{ok,missing}.obj         inspection cover bolt variants
  bushing_left_{ok,cracked}.obj        -Y torque arm bushing variants
  fault_oil_streak.obj                 overlay: oil running down from the input seal
  fault_oil_puddle.obj                 overlay: puddle, in its own frame (on the floor)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

d = c.dims
AX = (0, math.pi / 2, 0)   # rotate a Z cylinder onto the X axis
AY = (math.pi / 2, 0, 0)   # ... onto the Y axis


def xcyl(name, r, x, y=0.0, z=0.0, mat="gear_paint", vertices=64):
    return c.cylinder(name, r, x[1] - x[0], ((x[0] + x[1]) / 2, y, z), rotation=AX, vertices=vertices, mat=mat)


c.reset_scene()

# --------------------------------------------------------------- housing
parts = [
    xcyl("input_stub", d.GB_INPUT_R, d.GB_INPUT_X, mat="pipe_steel"),
    xcyl("front_cover", d.GB_FRONT_COVER_R, d.GB_FRONT_COVER_X),
    xcyl("planet_housing", d.GB_PLANET_R, d.GB_PLANET_X),
]
for i, x in enumerate(d.GB_RIB_X):  # stiffening ribs
    parts.append(xcyl(f"rib{i}", d.GB_PLANET_R + 0.03, (x - 0.03, x + 0.03)))

helical = c.box("helical_stage", d.GB_HELICAL_X, d.GB_HELICAL_Y, d.GB_HELICAL_Z, mat="gear_paint")
bev = helical.modifiers.new("round", "BEVEL")
bev.width, bev.segments = 0.12, 3
c.bpy.context.view_layer.objects.active = helical
c.bpy.ops.object.modifier_apply(modifier=bev.name)
parts.append(helical)
parts.append(xcyl("output_stub", d.GB_OUTPUT_R, d.GB_OUTPUT_X, z=d.GB_OUTPUT_Z, mat="pipe_steel"))

# torque arms with elastomer bushings and steel pins (one per side)
for side in (-1, 1):
    arm = c.box(f"arm{side}", d.GB_ARM_X, sorted((side * d.GB_ARM_Y_IN, side * d.GB_ARM_Y_OUT)), d.GB_ARM_Z, mat="gear_paint")
    eye = xcyl("eye", d.GB_BUSHING_R, (-2, 2), y=side * d.GB_BUSHING_Y, z=d.GB_BUSHING_Z)
    c.cut(arm, eye)
    parts.append(arm)
    if side > 0:  # the -Y bushing is a variant (ok / cracked)
        parts.append(xcyl("bushing_right", d.GB_BUSHING_R, d.GB_BUSHING_X, y=d.GB_BUSHING_Y, z=d.GB_BUSHING_Z, mat="rubber"))
    bx = d.GB_BUSHING_X
    parts.append(xcyl(f"pin{side}", d.GB_PIN_R, (bx[0] - d.GB_PIN_EXT, bx[1] + d.GB_PIN_EXT), y=side * d.GB_BUSHING_Y, z=d.GB_BUSHING_Z, mat="pipe_steel"))

# oil cooler with fan on top of the helical stage
parts.append(c.box("cooler", d.GB_COOLER_X, d.GB_COOLER_Y, d.GB_COOLER_Z, mat="steel_grey"))
cx = sum(d.GB_COOLER_X) / 2
parts.append(c.cylinder("fan_guard", 0.19, 0.03, (cx, 0, d.GB_COOLER_Z[1] + 0.015), mat="bolt_black"))

# inspection cover plate (bolts are a variant)
parts.append(c.box("insp_cover", d.GB_COVER_X, d.GB_COVER_Y, (d.GB_HELICAL_Z[1], d.GB_HELICAL_Z[1] + d.GB_COVER_T), mat="steel_grey"))

# sight glass frame on the -Y face
gx, gz = d.GB_GLASS_XZ
yface = d.GB_HELICAL_Y[0]
frame = c.cylinder("glass_frame", d.GB_GLASS_R + 0.025, 0.02, (gx, yface - 0.01, gz), rotation=AY, mat="pipe_steel")
c.cut(frame, c.cylinder("glass_hole", d.GB_GLASS_R, 0.1, (gx, yface, gz), rotation=AY))
parts.append(frame)

# offline filter + pipe up to the cooler
fx, fy = d.GB_FILTER_XY
parts.append(c.cylinder("filter", d.GB_FILTER_R, d.GB_FILTER_Z[1] - d.GB_FILTER_Z[0], (fx, fy, sum(d.GB_FILTER_Z) / 2), mat="steel_grey"))
parts.append(c.box("filter_bracket", (fx - 0.05, fx + 0.05), (fy, yface), (-0.1, 0.05), mat="steel_grey"))
c.bpy.ops.curve.primitive_bezier_curve_add()
pipe = c.bpy.context.active_object
pipe.name = "oil_pipe"
pts = pipe.data.splines[0].bezier_points
pts[0].co, pts[1].co = (fx + 0.05, fy, d.GB_FILTER_Z[1]), (d.GB_COOLER_X[1] - 0.05, d.GB_COOLER_Y[0], sum(d.GB_COOLER_Z) / 2)
for p, h in ((pts[0], (0, 0, 0.4)), (pts[1], (0, -0.3, 0))):
    p.handle_left_type = p.handle_right_type = "FREE"
    p.handle_right = (p.co[0] + h[0], p.co[1] + h[1], p.co[2] + h[2])
    p.handle_left = (p.co[0] - h[0], p.co[1] - h[1], p.co[2] - h[2])
pts[1].handle_left, pts[1].handle_right = pts[1].handle_right, pts[1].handle_left
pipe.data.bevel_depth = 0.02
pipe.data.dimensions = "3D"
c.bpy.ops.object.convert(target="MESH")
pipe.data.materials.append(c.material("pipe_steel"))
parts.append(pipe)

housing = c.join("gearbox", parts)
c.export_part("gearbox", [housing])
c.bpy.data.objects.remove(housing, do_unlink=True)  # keep the .blend tidy: variants only


# --------------------------------------------------------------- variants
def sight_glass(level):
    glass = c.cylinder("glass", d.GB_GLASS_R, 0.008, (gx, yface - 0.01, gz), rotation=AY, mat="glass")
    oil = c.cylinder("oil", d.GB_GLASS_R, 0.004, (gx, yface - 0.016, gz), rotation=AY, mat="oil")
    top = gz - d.GB_GLASS_R + level * 2 * d.GB_GLASS_R
    c.cut(oil, c.box("above", (gx - 1, gx + 1), (yface - 1, yface + 1), (top, top + 1)))
    return [glass, oil]


def filter_indicator(clogged):
    h = 0.06 if clogged else 0.015
    return [c.cylinder("indicator", 0.02, h, (fx, fy, d.GB_FILTER_Z[1] + h / 2), mat="alarm_red" if clogged else "ok_green")]


def cover_bolts(missing):
    x0, x1 = d.GB_COVER_X
    y0, y1 = d.GB_COVER_Y
    z = d.GB_HELICAL_Z[1] + d.GB_COVER_T + 0.007
    m = 0.04
    spots = [(x, y) for x in (x0 + m, (x0 + x1) / 2, x1 - m) for y in (y0 + m, y1 - m)]
    spots += [(x, (y0 + y1) / 2) for x in (x0 + m, x1 - m)]
    objs = []
    for i, (x, y) in enumerate(spots):
        if missing and i == 1:   # front -Y bolt gone: leave the empty thread hole
            objs.append(c.cylinder("hole", 0.011, 0.002, (x, y, z - 0.006), mat="crack"))
        else:
            objs.append(c.hex_bolt(f"bolt{i}", (x, y, z)))
    return objs


def bushing_left(cracked):
    """-Y torque arm bushing. Cracked: aged rubber, two radial tears through the
    front end face and the rubber bulging out of the arm eye."""
    bx = d.GB_BUSHING_X
    if not cracked:
        return [xcyl("bushing", d.GB_BUSHING_R, bx, y=-d.GB_BUSHING_Y, z=d.GB_BUSHING_Z, mat="rubber")]
    rub = xcyl("bushing", d.GB_BUSHING_R, (bx[0], bx[1] + 0.03), y=-d.GB_BUSHING_Y, z=d.GB_BUSHING_Z, mat="rubber")
    tears = []
    for i, ang in enumerate((-0.7, 1.4, 3.0)):
        t = c.box(f"tear{i}", (bx[1] - 0.12, bx[1] + 0.1), (-0.007, 0.007), (d.GB_PIN_R + 0.012, d.GB_BUSHING_R + 0.03))
        c.bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)  # mesh in place, origin at 0
        t.rotation_euler = (ang, 0, 0)  # rotate about the bushing axis ...
        t.location = (0, -d.GB_BUSHING_Y, d.GB_BUSHING_Z)  # ... then move onto it
        tears.append(t)
    c.cut(rub, *tears)
    # rubber squeezed out of the eye: a ring around the rim only, so the tears stay visible
    bulge = xcyl("bulge", d.GB_BUSHING_R + 0.012, (bx[1] - 0.01, bx[1] + 0.02), y=-d.GB_BUSHING_Y, z=d.GB_BUSHING_Z, mat="rubber")
    c.cut(bulge, xcyl("bulge_in", d.GB_BUSHING_R - 0.01, (bx[1] - 0.1, bx[1] + 0.1), y=-d.GB_BUSHING_Y, z=d.GB_BUSHING_Z))
    return [rub, bulge]


for name, objs in (
    ("bushing_left_ok", bushing_left(False)),
    ("bushing_left_cracked", bushing_left(True)),
    ("sight_glass_ok", sight_glass(d.GB_OIL_LEVEL_OK)),
    ("sight_glass_low", sight_glass(d.GB_OIL_LEVEL_LOW)),
    ("filter_indicator_ok", filter_indicator(False)),
    ("filter_indicator_clogged", filter_indicator(True)),
    ("cover_bolts_ok", cover_bolts(False)),
    ("cover_bolts_missing", cover_bolts(True)),
):
    c.export_part("gearbox", [c.join(name, objs)], obj_name=name)

# --------------------------------------------------------------- fault overlays
# oil running down the front cover from the input shaft seal, dripping off the housing
streak = [
    c.box("streak", (0.0, 0.004), (-0.035, 0.03), (-d.GB_FRONT_COVER_R + 0.02, -d.GB_INPUT_R), mat="oil"),
    c.box("streak_under", (-0.6, 0.0), (-0.03, 0.03), (-d.GB_PLANET_R - 0.004, -d.GB_PLANET_R + 0.02), mat="oil"),
]
drop = c.bpy.ops.mesh.primitive_uv_sphere_add(radius=0.012, location=(-0.3, 0, -d.GB_PLANET_R - 0.02))
c.bpy.context.active_object.data.materials.append(c.material("oil"))
streak.append(c.bpy.context.active_object)
c.export_part("gearbox", [c.join("fault_oil_streak", streak)], obj_name="fault_oil_streak")

c.export_part("gearbox", [c.puddle("fault_oil_puddle", 0.35, (0, 0, 0.0015), seed=4)], obj_name="fault_oil_puddle")


c.save_blend("gearbox")
print("built gearbox")
