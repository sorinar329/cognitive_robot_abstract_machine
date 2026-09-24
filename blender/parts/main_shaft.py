"""Part: main shaft with its two main bearings (IEA: CARB front, SRB rear), the
rotor lock and the grease collector. Everything is in the ``drivetrain`` frame
(hub centre, X along the tilted shaft axis upwind); the lock pin has its own
frame at the housing's front face.

Outputs:
  main_bearing.obj                         both bearing housings, pedestals, seals
  grease_collector_{ok,full}.obj           collector bottle on the front bearing
  main_shaft.obj                           shaft, flange, lock disc, shrink disc
  shrink_disc_bolts_{ok,loose}.obj         bolts with torque marker stripes
  rotor_lock.obj, rotor_lock_pin.obj       housing + handwheel, sliding pin
  fault_grease_leak.obj                    grease pushed out of the front bearing's upwind seal
  fault_fretting.obj                       red-brown fretting powder at the shrink disc
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

d = c.dims
AX = (0, math.pi / 2, 0)
P = "main_shaft"


def xcyl(name, r, x, y=0.0, z=0.0, mat="gear_paint", vertices=64):
    return c.cylinder(name, r, x[1] - x[0], ((x[0] + x[1]) / 2, y, z), rotation=AX, vertices=vertices, mat=mat)


def foot_z(x):
    """Drivetrain-frame height of the bedplate top below drivetrain x."""
    return d.nacelle_height_in_drivetrain(x, d.BP_TOP_Z)


c.reset_scene()
W = d.MB_HALF_W

# --------------------------------------------------------------- main bearings
parts = []
for i, xb in enumerate(d.MB_XS):
    xr = (xb - W, xb + W)
    ring = xcyl(f"housing{i}", d.MB_R, xr)
    c.cut(ring, xcyl("bore", d.MB_BORE_R, (-20, 20)))
    fz = foot_z(xb)
    web = c.box(f"web{i}", (xr[0] + 0.05, xr[1] - 0.05), (-0.55, 0.55), (fz, 0.0), mat="gear_paint")
    c.cut(web, xcyl("bore2", d.MB_R - 0.02, (-20, 20)))
    foot = c.box(f"foot{i}", xr, (-d.MB_FOOT_Y, d.MB_FOOT_Y), (fz - 0.05, fz + 0.3), mat="gear_paint")
    parts += [ring, web, foot]
    for sx in (-1, 1):
        seal = xcyl(f"seal{i}{sx}", d.MB_SEAL_R, (xb + sx * W - 0.015, xb + sx * W + 0.015), mat="rubber")
        c.cut(seal, xcyl("seal_in", d.MB_BORE_R, (-20, 20)))
        parts.append(seal)
        for y in (-0.95, -0.7, 0.7, 0.95):
            parts.append(c.hex_bolt(f"fb{i}{sx}{y}", (xb + sx * 0.14, y, fz + 0.3 + 0.007)))
    parts.append(c.cylinder(f"nipple{i}", 0.012, 0.05, (xb, 0, d.MB_R + 0.025), mat="pipe_steel"))
# drain pipe from the front bearing's upwind seal to the collector on its -Y foot
xb0 = d.MB_XS[0]
col_x, col_base = xb0 + 0.1, foot_z(xb0) + 0.3
parts.append(c.curve_pipe("drain", (xb0 + W, -0.3, -0.35), (0.1, 0, 0), (col_x, d.MB_COLLECTOR_Y, col_base + d.MB_COLLECTOR_H + 0.03),
                          (0, 0, -0.1), 0.012))
c.export_part(P, [c.join("main_bearing", parts)], obj_name="main_bearing")


def collector(level):
    h, r = d.MB_COLLECTOR_H, d.MB_COLLECTOR_R
    bottle = c.cylinder("bottle", r, h, (col_x, d.MB_COLLECTOR_Y, col_base + h / 2), mat="bottle_white")
    fill = c.cylinder("fill", r + 0.002, level * h, (col_x, d.MB_COLLECTOR_Y, col_base + level * h / 2 + 0.002), mat="grease")
    cap = c.cylinder("cap", r * 0.6, 0.03, (col_x, d.MB_COLLECTOR_Y, col_base + h + 0.015), mat="bolt_black")
    return [bottle, fill, cap]


for name, level in (("grease_collector_ok", d.MB_GREASE_OK), ("grease_collector_full", d.MB_GREASE_FULL)):
    c.export_part(P, [c.join(name, collector(level))], obj_name=name)

# --------------------------------------------------------------- main shaft
parts = [xcyl(f"seg{i}", r, x, mat="pipe_steel") for i, (r, x) in enumerate(d.MS_SEGMENTS)]
parts.append(xcyl("flange", d.MS_FLANGE_R, d.MS_FLANGE_X, mat="pipe_steel"))
for k in range(28):
    y, z = c.polar(d.MS_FLANGE_R - 0.08, 2 * math.pi * k / 28)
    parts.append(c.hex_bolt(f"flange_bolt{k}", (d.MS_FLANGE_X[0] - 0.007, y, z), axis="X"))
disc = xcyl("lock_disc", d.MS_LOCK_DISC_R, d.MS_LOCK_DISC_X, mat="gear_paint")
holes = []
for k in range(d.MS_LOCK_HOLES):
    y, z = c.polar(d.MS_LOCK_HOLE_PCD, 2 * math.pi * k / d.MS_LOCK_HOLES)
    holes.append(xcyl(f"lock_hole{k}", d.MS_LOCK_HOLE_R, (-20, 20), y=y, z=z))
c.cut(disc, *holes)
parts.append(disc)
shrink = xcyl("shrink_disc", d.MS_SHRINK_R, d.MS_SHRINK_X, mat="pipe_steel")
groove = xcyl("groove", d.MS_SHRINK_R + 0.01, (sum(d.MS_SHRINK_X) / 2 - 0.01, sum(d.MS_SHRINK_X) / 2 + 0.01))
c.cut(groove, xcyl("groove_in", d.MS_SHRINK_R - 0.015, (-20, 20)))
c.cut(shrink, groove)
parts.append(shrink)
c.export_part(P, [c.join("main_shaft", parts)], obj_name="main_shaft")


def shrink_bolts(loose):
    """Axial bolts on the upwind face of the shrink disc, each with a yellow torque
    marker stripe from the head onto the disc. A bolt that backed out sits proud
    and its stripe no longer lines up."""
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
    c.export_part(P, [c.join(name, shrink_bolts(loose))], obj_name=name)

# --------------------------------------------------------------- rotor lock (on the front bearing, -Y)
hx = d.RL_HOUSING_X
housing = c.box("rl_housing", hx, (d.RL_Y - d.RL_HALF, d.RL_Y + d.RL_HALF), (-d.RL_HALF, d.RL_HALF), mat="hatch_yellow")
clamp = c.box("rl_clamp", (hx[0] + 0.05, hx[1] - 0.05), (d.RL_Y + d.RL_HALF - 0.02, -d.MB_R + 0.08), (-0.06, 0.06), mat="gear_paint")
c.bpy.ops.mesh.primitive_torus_add(major_radius=0.1, minor_radius=0.012, location=(hx[0] - 0.08, d.RL_Y, 0), rotation=AX)
wheel = c.bpy.context.active_object
wheel.data.materials.append(c.material("alarm_red"))
spindle = xcyl("rl_spindle", 0.02, (hx[0] - 0.08, hx[0]), y=d.RL_Y, mat="pipe_steel")
spokes = [c.radial_box(f"spoke{i}", (hx[0] - 0.085, hx[0] - 0.075), (0.0, 0.1), 0.012, i * 2 * math.pi / 3,
                       center=(d.RL_Y, 0), mat="alarm_red") for i in range(3)]
c.export_part(P, [c.join("rotor_lock", [housing, clamp, wheel, spindle] + spokes)], obj_name="rotor_lock")
pin = xcyl("pin", d.RL_PIN_R, (-d.RL_PIN_LEN, -0.06), mat="pipe_steel")
band = xcyl("band", d.RL_PIN_R, (-0.06, -0.01), mat="alarm_red")
tip = xcyl("tip", d.RL_PIN_R, (-0.01, 0.0), mat="pipe_steel")
c.export_part(P, [c.join("rotor_lock_pin", [pin, band, tip])], obj_name="rotor_lock_pin")

# --------------------------------------------------------------- fault overlays
xs = xb0 - W                     # downwind seal of the front bearing (open gap between the bearings)
fz0 = foot_z(xb0) + 0.3
blobs = []
for i, a in enumerate([1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7]):    # bottom and +Y side of the seal
    y, z = c.polar(d.MB_SEAL_R - 0.01, -a)
    c.bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035 + 0.01 * (i % 3), location=(xs - 0.02, y, z), scale=(0.6, 1, 1))
    blobs.append(c.bpy.context.active_object)
    blobs[-1].data.materials.append(c.material("grease"))
blobs.append(c.box("run", (xs - 0.02, xs - 0.012), (0.2, 0.29), (fz0, -0.3), mat="grease"))
blobs.append(c.puddle("on_foot", 0.1, (xs - 0.08, 0.3, fz0 + 0.002), seed=2, mat="grease"))
c.export_part(P, [c.join("fault_grease_leak", blobs)], obj_name="fault_grease_leak")

x0 = d.MS_SHRINK_X[1]
r_in = d.MS_SEGMENTS[-1][0]
fret = xcyl("fret_ring", r_in + 0.03, (x0, x0 + 0.05), mat="rust")
c.cut(fret, xcyl("fret_in", r_in - 0.01, (-20, 20)))
streak = c.radial_box("fret_streak", (x0, x0 + 0.003), (r_in, d.MS_SHRINK_R - 0.01), 0.04, math.pi * 0.95, mat="rust")
c.export_part(P, [c.join("fault_fretting", [fret, streak])], obj_name="fault_fretting")

c.save_blend("main_shaft")
print("built main_shaft")
