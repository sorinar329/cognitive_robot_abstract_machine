"""Part: main bearing (pillow block with grease collector), main shaft with
hub flange, rotor lock disc and shrink disc, and the rotor lock.

All meshes are in the ``main_bearing`` frame (bearing centre, X upwind), which
``main_shaft`` and ``rotor_lock`` share; the lock pin is in its own frame at
the housing front face. Outputs:

  main_bearing.obj                         housing, foot, seals, grease pipe
  grease_collector_{ok,full}.obj           collector bottle variants
  main_shaft.obj                           shaft, flange, lock disc, shrink disc
  shrink_disc_bolts_{ok,loose}.obj         bolts with torque marker stripes
  rotor_lock.obj, rotor_lock_pin.obj       housing + handwheel, sliding pin
  fault_grease_leak.obj                    overlay: grease pushed out of the front seal
  fault_fretting.obj                       overlay: red-brown fretting powder at the shrink disc
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

d = c.dims
AX = (0, math.pi / 2, 0)


def xcyl(name, r, x, y=0.0, z=0.0, mat="gear_paint", vertices=64):
    return c.cylinder(name, r, x[1] - x[0], ((x[0] + x[1]) / 2, y, z), rotation=AX, vertices=vertices, mat=mat)


c.reset_scene()
W = d.MB_HALF_W
foot_top = d.MB_FOOT_Z[1]

# --------------------------------------------------------------- main bearing
ring = xcyl("housing", d.MB_R, (-W, W))
web = c.box("web", (-W + 0.05, W - 0.05), (-0.55, 0.55), (d.MB_FOOT_Z[0], 0.0), mat="gear_paint")
foot = c.box("foot", (-W, W), (-d.MB_FOOT_Y, d.MB_FOOT_Y), d.MB_FOOT_Z, mat="gear_paint")
c.cut(ring, xcyl("bore", d.MB_BORE_R, (-1, 1)))
c.cut(web, xcyl("bore2", d.MB_R - 0.02, (-1, 1)))
parts = [ring, web, foot]
for sx in (-1, 1):
    seal = xcyl(f"seal{sx}", d.MB_SEAL_R, (sx * W - 0.015, sx * W + 0.015), mat="rubber")
    c.cut(seal, xcyl("seal_in", d.MB_BORE_R, (-1, 1)))
    parts.append(seal)
    for y in (-0.72, -0.5, 0.5, 0.72):   # foot bolts
        parts.append(c.hex_bolt(f"fb{sx}{y}", (sx * 0.15, y, foot_top + 0.007)))
parts.append(c.cylinder("grease_nipple", 0.012, 0.05, (0, 0, d.MB_R + 0.025), mat="pipe_steel"))
cxy = d.MB_COLLECTOR_XY
parts.append(c.curve_pipe("drain", (W, -0.3, -0.35), (0.1, 0, 0), (cxy[0], cxy[1], foot_top + d.MB_COLLECTOR_H + 0.03),
                          (0, 0, -0.1), 0.012))
c.export_part("main_shaft", [c.join("main_bearing", parts)], obj_name="main_bearing")
for o in list(c.bpy.data.objects):
    c.bpy.data.objects.remove(o, do_unlink=True)


def collector(level):
    h, r = d.MB_COLLECTOR_H, d.MB_COLLECTOR_R
    bottle = c.cylinder("bottle", r, h, (cxy[0], cxy[1], foot_top + h / 2), mat="bottle_white")
    fill = c.cylinder("fill", r + 0.002, level * h, (cxy[0], cxy[1], foot_top + level * h / 2 + 0.002), mat="grease")
    cap = c.cylinder("cap", r * 0.6, 0.03, (cxy[0], cxy[1], foot_top + h + 0.015), mat="bolt_black")
    return [bottle, fill, cap]


for name, level in (("grease_collector_ok", d.MB_GREASE_OK), ("grease_collector_full", d.MB_GREASE_FULL)):
    c.export_part("main_shaft", [c.join(name, collector(level))], obj_name=name)

# --------------------------------------------------------------- main shaft
parts = [xcyl(f"seg{i}", r, x, mat="pipe_steel") for i, (r, x) in enumerate(d.MS_SEGMENTS)]
flange = xcyl("flange", d.MS_FLANGE_R, d.MS_FLANGE_X, mat="pipe_steel")
parts.append(flange)
for k in range(24):
    y, z = c.polar(d.MS_FLANGE_R - 0.07, 2 * math.pi * k / 24)
    parts.append(c.hex_bolt(f"flange_bolt{k}", (d.MS_FLANGE_X[0] - 0.007, y, z), axis="X"))
disc = xcyl("lock_disc", d.MS_LOCK_DISC_R, d.MS_LOCK_DISC_X, mat="gear_paint")
holes = []
for k in range(d.MS_LOCK_HOLES):
    y, z = c.polar(d.MS_LOCK_HOLE_PCD, 2 * math.pi * k / d.MS_LOCK_HOLES)
    holes.append(xcyl(f"lock_hole{k}", d.MS_LOCK_HOLE_R, (-2, 2), y=y, z=z))
c.cut(disc, *holes)
parts.append(disc)
shrink = xcyl("shrink_disc", d.MS_SHRINK_R, d.MS_SHRINK_X, mat="pipe_steel")
groove = xcyl("groove", d.MS_SHRINK_R + 0.01, (sum(d.MS_SHRINK_X) / 2 - 0.01, sum(d.MS_SHRINK_X) / 2 + 0.01))
c.cut(groove, xcyl("groove_in", d.MS_SHRINK_R - 0.015, (-3, 3)))
c.cut(shrink, groove)  # the two clamping rings
parts.append(shrink)
c.export_part("main_shaft", [c.join("main_shaft", parts)], obj_name="main_shaft")


def shrink_bolts(loose):
    """Axial bolts on the upwind face of the shrink disc, each with a yellow
    torque marker stripe running from the head onto the disc. A bolt that has
    backed out sits proud of the face and its stripe no longer lines up."""
    face = d.MS_SHRINK_X[1]
    objs = []
    for k in range(d.MS_SHRINK_BOLTS):
        a = 2 * math.pi * (k + 0.5) / d.MS_SHRINK_BOLTS
        y, z = c.polar(d.MS_SHRINK_BOLT_PCD, a)
        out = 0.022 if loose and k == d.MS_LOOSE_BOLT else 0.0
        objs.append(c.hex_bolt(f"sb{k}", (face + 0.007 + out, y, z), axis="X"))
        if out:
            objs.append(xcyl(f"thread{k}", 0.011, (face, face + out), y=y, z=z, mat="pipe_steel", vertices=12))
        twist = 0.7 if out else 0.0
        objs.append(c.radial_box(f"mark_head{k}", (face + 0.014 + out, face + 0.016 + out), (-0.013, 0.013), 0.006,
                                 a + twist, center=(y, z), mat="hatch_yellow"))
        objs.append(c.radial_box(f"mark_disc{k}", (face, face + 0.002), (d.MS_SHRINK_BOLT_PCD + 0.02, d.MS_SHRINK_BOLT_PCD + 0.06),
                                 0.006, a, mat="hatch_yellow"))
    return objs


for name, loose in (("shrink_disc_bolts_ok", False), ("shrink_disc_bolts_loose", True)):
    c.export_part("main_shaft", [c.join(name, shrink_bolts(loose))], obj_name=name)

# --------------------------------------------------------------- rotor lock
hx = d.RL_HOUSING_X
housing = c.box("rl_housing", hx, (d.RL_Y - d.RL_HALF, d.RL_Y + d.RL_HALF), (-d.RL_HALF, d.RL_HALF), mat="hatch_yellow")
bracket = c.box("rl_bracket", (0.2, hx[0] + 0.12), (d.RL_Y - 0.1, d.RL_Y + 0.1), (foot_top, -d.RL_HALF), mat="gear_paint")
c.bpy.ops.mesh.primitive_torus_add(major_radius=0.1, minor_radius=0.012, location=(hx[0] - 0.08, d.RL_Y, 0), rotation=AX)
wheel = c.bpy.context.active_object
wheel.data.materials.append(c.material("alarm_red"))
spindle = xcyl("rl_spindle", 0.02, (hx[0] - 0.08, hx[0]), y=d.RL_Y, mat="pipe_steel")
spokes = [c.radial_box(f"spoke{i}", (hx[0] - 0.085, hx[0] - 0.075), (0.0, 0.1), 0.012, i * 2 * math.pi / 3,
                       center=(d.RL_Y, 0), mat="alarm_red") for i in range(3)]
c.export_part("main_shaft", [c.join("rotor_lock", [housing, bracket, wheel, spindle] + spokes)], obj_name="rotor_lock")
# pin frame at the housing front face; red warning band just behind the tip
pin = xcyl("pin", d.RL_PIN_R, (-d.RL_PIN_LEN, -0.06), mat="pipe_steel")
band = xcyl("band", d.RL_PIN_R, (-0.06, -0.01), mat="alarm_red")
tip = xcyl("tip", d.RL_PIN_R, (-0.01, 0.0), mat="pipe_steel")
c.export_part("main_shaft", [c.join("rotor_lock_pin", [pin, band, tip])], obj_name="rotor_lock_pin")

# --------------------------------------------------------------- fault overlays
blobs = []
for i, a in enumerate([2.2, 2.55, 2.9, 3.2, 3.5, 3.85, 4.1]):
    y, z = c.polar(d.MB_SEAL_R - 0.01, a)
    c.bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035 + 0.01 * (i % 3), location=(W + 0.02, y, z), scale=(0.6, 1, 1))
    blobs.append(c.bpy.context.active_object)
    blobs[-1].data.materials.append(c.material("grease"))
blobs.append(c.box("run", (W + 0.012, W + 0.02), (-0.05, 0.04), (foot_top, -d.MB_SEAL_R + 0.02), mat="grease"))
blobs.append(c.puddle("on_foot", 0.08, (W - 0.08, 0.02, foot_top + 0.002), seed=2, mat="grease"))
blobs.append(c.puddle("on_bedplate", 0.12, (W + 0.12, -0.15, d.MB_FOOT_Z[0] + 0.002), seed=5, mat="grease"))
c.export_part("main_shaft", [c.join("fault_grease_leak", blobs)], obj_name="fault_grease_leak")

# fretting: rust-coloured powder ring where the shaft enters the shrink disc, plus a run-off streak
x0 = d.MS_SHRINK_X[1]
fret = xcyl("fret_ring", d.MS_SEGMENTS[3][0] + 0.012, (x0, x0 + 0.05), mat="rust")
c.cut(fret, xcyl("fret_in", d.MS_SEGMENTS[3][0] - 0.01, (-3, 3)))
streak = c.radial_box("fret_streak", (x0, x0 + 0.003), (d.MS_SEGMENTS[3][0], d.MS_SHRINK_R - 0.01), 0.04, math.pi * 0.95,
                      mat="rust")
c.export_part("main_shaft", [c.join("fault_fretting", [fret, streak])], obj_name="fault_fretting")

c.save_blend("main_shaft")
print("built main_shaft")
