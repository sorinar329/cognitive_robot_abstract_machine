"""Part: nacelle systems (nacelle frame): dry-type transformer in its cage, the
converter (+Y) and controller (-Y) cabinets with doors, status lights and an
emergency stop, hydraulic unit with the hose to the brake, cooling unit, service
crane (rail, trolley, hook) and safety equipment.

Door meshes are built in their hinge frames, the crane trolley and hook in theirs.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

d = c.dims
P = "systems"
c.reset_scene()


def export(name, objs):
    c.export_part(P, [c.join(name, objs) if len(objs) > 1 else objs[0]], obj_name=name)


# %% transformer: three cast resin coils on a core frame, inside a mesh cage
tx, ty, tz = d.TRAFO_X, d.TRAFO_Y, d.TRAFO_Z
cx = sum(tx) / 2
parts = [c.box("base", (tx[0] + 0.15, tx[1] - 0.15), (ty[0] + 0.15, ty[1] - 0.15), (0.0, 0.15), mat="steel_grey"),
         c.box("yoke_bottom", (cx - 0.25, cx + 0.25), (ty[0] + 0.2, ty[1] - 0.2), (0.15, 0.4), mat="bolt_black"),
         c.box("yoke_top", (cx - 0.25, cx + 0.25), (ty[0] + 0.2, ty[1] - 0.2), (1.75, 2.0), mat="bolt_black")]
for i, y in enumerate((-0.5, 0.0, 0.5)):
    parts.append(c.cylinder(f"coil{i}", 0.24, 1.35, (cx, y, 1.075), mat="resin"))
    parts.append(c.cylinder(f"bushing{i}", 0.03, 0.2, (cx + 0.2, y, 2.1), mat="bottle_white"))
bars = []
for x0, x1, y0, y1 in ((tx[0], tx[1], ty[0], ty[0]), (tx[0], tx[1], ty[1], ty[1]), (tx[0], tx[0], ty[0], ty[1]), (tx[1], tx[1], ty[0], ty[1])):
    n = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) / 0.2))
    for k in range(n + 1):
        x, y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
        bars.append(c.rod("bar", (x, y, 0.0), (x, y, tz[1]), 0.008, mat="steel_grey", vertices=6))
    for z in (0.05, tz[1] / 2, tz[1]):
        bars.append(c.rod("rail", (x0, y0, z), (x1, y1, z), 0.015, mat="steel_grey", vertices=6))
bars.append(c.box("warning_sign", (tx[1] + 0.005, tx[1] + 0.01), (-0.15, 0.15), (1.3, 1.6), mat="hatch_yellow"))
export("transformer", parts + bars)

# %% cabinets (+Y converter, -Y controller): body, and doors in their hinge frames
for side, name in ((1, "converter"), (-1, "controller")):
    ys = sorted((side * d.CABINET_Y[0], side * d.CABINET_Y[1]))
    body = c.box("body", d.CABINET_X, ys, d.CABINET_Z, mat="cabinet_grey")
    plinth = c.box("plinth", (d.CABINET_X[0] + 0.02, d.CABINET_X[1] - 0.02), (ys[0] + 0.02, ys[1] - 0.02), (0.0, 0.1), mat="bolt_black")
    export(f"cabinet_{name}", [body, plinth])
    w = d.CABINET_Y[1] - d.CABINET_Y[0] - 0.02
    y_range = (-w, 0.0) if side > 0 else (0.0, w)       # hinge on the wall side
    door = [c.box("door", (0.0, 0.025), y_range, (d.CABINET_Z[0] + 0.12, d.CABINET_Z[1] - 0.02), mat="cabinet_grey")]
    free_edge = -side * (w - 0.08)
    door.append(c.box("handle", (0.025, 0.06), (free_edge - 0.015, free_edge + 0.015), (1.0, 1.2), mat="bolt_black"))
    door.append(c.box("label", (0.025, 0.028), (-side * 0.25 - 0.15, -side * 0.25 + 0.15), (1.55, 1.7), mat="bottle_white"))
    if name == "controller":   # emergency stop on the controller door
        door.append(c.box("estop_plate", (0.025, 0.03), (side * -0.55 - 0.06, side * -0.55 + 0.06), (1.3, 1.42), mat="hatch_yellow"))
        door.append(c.cylinder("estop", 0.035, 0.04, (0.05, side * -0.55, 1.36), rotation=(0, math.pi / 2, 0), mat="alarm_red"))
    export(f"cabinet_{name}_door", door)


def status_light(ok, side):
    """Status light on the door, in the door's hinge frame."""
    y = -side * 0.25
    return [c.cylinder("led", 0.025, 0.02, (0.035, y, 1.85), rotation=(0, math.pi / 2, 0), mat="led_green" if ok else "led_red")]


export("status_light_ok_p", status_light(True, 1))
export("status_light_error_p", status_light(False, 1))

# %% hydraulic unit (+Y, beside the generator) and its hose to the brake caliper
hx, hy, hz = d.HYDRAULIC_X, d.HYDRAULIC_Y, d.HYDRAULIC_Z
hyd = [c.box("tank", hx, hy, (hz[0], hz[0] + 0.4), mat="steel_grey"),
       c.cylinder("motor", 0.11, 0.35, (hx[0] + 0.25, sum(hy) / 2, hz[0] + 0.58), mat="generator_paint"),
       c.box("valve_block", (hx[1] - 0.35, hx[1] - 0.1), (hy[0] + 0.05, hy[1] - 0.05), (hz[0] + 0.4, hz[0] + 0.6), mat="pipe_steel"),
       c.box("stand", (hx[0], hx[1]), (hy[0] + 0.1, hy[1] - 0.1), (0.0, hz[0]), mat="gear_paint")]
for i, x in enumerate((hx[1] - 0.25, hx[1] - 0.12)):
    hyd.append(c.cylinder(f"accumulator{i}", 0.07, 0.45, (x, hy[1] - 0.12, hz[0] + 0.83), mat="alarm_red"))
caliper = d.drivetrain_to_nacelle(d.BRAKE_DISC_X[1] + 0.1, d.GB_OUTPUT_Z + d.CALIPER_Z[1] + 0.2)
fit = (hx[1] - 0.2, hy[1] - 0.05, hz[0] + 0.55)     # hose fitting on the walkway (+Y) side
hyd.append(c.curve_pipe("hose", fit, (0.3, -0.3, 0.5), caliper, (0.2, -0.1, -0.3), 0.014, mat="rubber"))
export("hydraulic_unit", hyd)

# leak: oil running down the tank's walkway side from the hose fitting, puddle spreading onto the walkway
leak = [c.box("wet", (fit[0] - 0.05, fit[0] + 0.04), (hy[1], hy[1] + 0.004), (0.05, hz[0] + 0.5), mat="hydraulic_oil"),
        c.puddle("puddle", 0.25, (fit[0] - 0.1, hy[1] + 0.12, 0.002), seed=13, mat="hydraulic_oil")]
export("fault_hydraulic_leak", leak)

# %% cooling unit (-Y, beside the generator): pump, expansion tank with a level tube (variant)
kx, ky, kz = d.COOLING_X, d.COOLING_Y, d.COOLING_Z
tank_y = ky[0] + 0.2
cool = [c.box("frame", kx, ky, (0.0, kz[0]), mat="gear_paint"),
        c.cylinder("pump", 0.12, 0.3, (kx[1] - 0.2, sum(ky) / 2, kz[0] + 0.15), rotation=(0, math.pi / 2, 0), mat="generator_paint"),
        c.cylinder("tank", 0.18, 0.55, (sum(kx) / 2, tank_y, kz[1] - 0.25), rotation=(0, math.pi / 2, 0), mat="steel_grey"),
        c.rod("pipe_out", (kx[1] - 0.05, sum(ky) / 2, kz[0] + 0.15), (kx[1] + 0.3, -0.5, 1.0), 0.03, mat="pipe_steel"),
        c.rod("tube_frame", (sum(kx) / 2, ky[0] - 0.005, kz[1] - 0.45), (sum(kx) / 2, ky[0] - 0.005, kz[1] - 0.05), 0.022, mat="pipe_steel")]
export("cooling_unit", cool)


def coolant_level(fraction):
    x, y = sum(kx) / 2, ky[0] - 0.03
    z0, z1 = kz[1] - 0.42, kz[1] - 0.08
    return [c.rod("tube", (x, y, z0), (x, y, z1), 0.012, mat="glass"),
            c.rod("level", (x, y - 0.002, z0), (x, y - 0.002, z0 + (z1 - z0) * fraction), 0.013, mat="coolant"),
            c.box("min_mark", (x - 0.03, x + 0.03), (y - 0.02, y - 0.015), (z0 + 0.1 * (z1 - z0) + 0.1, z0 + 0.1 * (z1 - z0) + 0.105), mat="alarm_red")]


export("coolant_level_ok", coolant_level(0.75))
export("coolant_level_low", coolant_level(0.08))

# %% service crane: rail under the roof; trolley (frame at the rail) and hook (frame at the hook) move
rx, rz = d.CRANE_X, d.CRANE_Z
export("crane_rail", [c.box("web", rx, (-0.01, 0.01), (rz, rz + 0.25), mat="hatch_yellow"),
                      c.box("flange_bottom", rx, (-0.09, 0.09), (rz - 0.015, rz), mat="hatch_yellow"),
                      c.box("flange_top", rx, (-0.09, 0.09), (rz + 0.25, rz + 0.265), mat="hatch_yellow")]
       + [c.box(f"hanger{i}", (x - 0.04, x + 0.04), (-0.04, 0.04), (rz + 0.265, d.NACELLE_Z[1] - d.WALL), mat="steel_grey")
          for i, x in enumerate((rx[0] + 0.3, (rx[0] + rx[1]) / 2, rx[1] - 0.3))])
trolley = [c.box("trolley", (-0.25, 0.25), (-0.14, 0.14), (-0.33, -0.03), mat="hatch_yellow"),
           c.cylinder("hoist", 0.1, 0.35, (0.0, 0.0, -0.22), rotation=(0, math.pi / 2, 0), mat="generator_paint")]
for x in (-0.18, 0.18):
    for y in (-0.11, 0.11):
        trolley.append(c.cylinder("wheel", 0.035, 0.02, (x, y, -0.035), rotation=(math.pi / 2, 0, 0), mat="bolt_black"))
export("crane_trolley", trolley)
hook = [c.rod("chain", (0, 0, 0.1), (0, 0, 0.6), 0.01, mat="bolt_black", vertices=6),
        c.box("hook_block", (-0.07, 0.07), (-0.04, 0.04), (0.0, 0.14), mat="hatch_yellow")]
c.bpy.ops.mesh.primitive_torus_add(major_radius=0.05, minor_radius=0.013, location=(0, 0, -0.04), rotation=(math.pi / 2, 0, 0))
hook.append(c.bpy.context.active_object)
hook[-1].data.materials.append(c.material("pipe_steel"))
export("crane_hook", hook)

# %% safety: fire extinguisher (variant), first-aid box, emergency stop by the tower hatch
ex, ey = d.EXTINGUISHER_XY
wall = d.INNER_Y[0]
bracket = [c.box("bracket", (ex - 0.1, ex + 0.1), (wall, wall + 0.03), (0.35, 0.95), mat="steel_grey"),
           c.box("strap", (ex - 0.1, ex + 0.1), (wall + 0.03, wall + 0.2), (0.75, 0.78), mat="bolt_black")]
extinguisher = [c.cylinder("bottle", 0.08, 0.5, (ex, wall + 0.12, 0.62), mat="alarm_red"),
                c.cylinder("valve", 0.03, 0.08, (ex, wall + 0.12, 0.91), mat="bolt_black"),
                c.rod("hose", (ex + 0.03, wall + 0.12, 0.93), (ex + 0.07, wall + 0.2, 0.5), 0.01, mat="rubber")]
export("extinguisher_present", bracket + extinguisher)
export("extinguisher_missing", [c.box("bracket", (ex - 0.1, ex + 0.1), (wall, wall + 0.03), (0.35, 0.95), mat="steel_grey"),
                                c.box("strap", (ex - 0.1, ex + 0.1), (wall + 0.03, wall + 0.2), (0.75, 0.78), mat="bolt_black")])
fx = -1.0
safety = [c.box("first_aid", (fx - 0.2, fx + 0.2), (d.INNER_Y[1] - 0.12, d.INNER_Y[1]), (1.2, 1.5), mat="led_green"),
          c.box("cross_h", (fx - 0.07, fx + 0.07), (d.INNER_Y[1] - 0.125, d.INNER_Y[1] - 0.12), (1.33, 1.37), mat="bottle_white"),
          c.box("cross_v", (fx - 0.02, fx + 0.02), (d.INNER_Y[1] - 0.125, d.INNER_Y[1] - 0.12), (1.28, 1.42), mat="bottle_white")]
sx, sz = d.ESTOP_XZ
safety += [c.box("estop_box", (sx - 0.07, sx + 0.07), (wall, wall + 0.08), (sz - 0.07, sz + 0.07), mat="hatch_yellow"),
           c.cylinder("estop", 0.035, 0.04, (sx, wall + 0.1, sz), rotation=(math.pi / 2, 0, 0), mat="alarm_red")]
export("safety", safety)

# %% task equipment: tool rack (rear, -Y wall), tool tray on the -Y girder, technician placeholder
rx, ry, rt = d.TOOL_RACK_X, d.TOOL_RACK_Y, d.TOOL_RACK_TOP
rack = [c.box("rack_shelf", rx, ry, (rt - 0.03, rt), mat="steel_grey"),
        c.box("rack_shelf_low", rx, ry, (0.35, 0.38), mat="steel_grey"),
        c.box("rack_lip", rx, (ry[1] - 0.015, ry[1]), (rt, rt + 0.04), mat="hatch_yellow")]
for x in rx:
    rack.append(c.box("rack_side", (x - 0.015, x + 0.015), ry, (0.0, rt + 0.45), mat="steel_grey"))
rack.append(c.box("rack_back", rx, (ry[0], ry[0] + 0.02), (0.0, rt + 0.45), mat="steel_grey"))
export("tool_rack", rack)
tx, ty, tt = d.TOOL_TRAY_X, d.TOOL_TRAY_Y, d.TOOL_TRAY_TOP
tray = [c.box("tray", tx, ty, (tt - 0.025, tt), mat="hatch_yellow"),
        c.box("tray_rim_x0", (tx[0], tx[0] + 0.015), ty, (tt, tt + 0.03), mat="hatch_yellow"),
        c.box("tray_rim_x1", (tx[1] - 0.015, tx[1]), ty, (tt, tt + 0.03), mat="hatch_yellow"),
        c.box("tray_post", (sum(tx) / 2 - 0.03, sum(tx) / 2 + 0.03), (sum(ty) / 2 - 0.03, sum(ty) / 2 + 0.03),
              (d.BP_TOP_Z, tt - 0.025), mat="steel_grey")]
export("tool_tray", tray)


def technician():
    """Placeholder person (1.78 m) in the link frame at the feet, facing +Y."""
    body = []
    for sx in (-0.1, 0.1):
        body.append(c.rod("leg", (sx, 0, 0.05), (sx, 0, 0.88), 0.07, mat="alarm_red", vertices=16))
        body.append(c.box("boot", (sx - 0.06, sx + 0.06), (-0.06, 0.14), (0.0, 0.08), mat="bolt_black"))
        body.append(c.rod("arm", (sx * 2.2, 0.0, 1.42), (sx * 2.3, 0.12, 0.98), 0.05, mat="alarm_red", vertices=12))
    body.append(c.box("torso", (-0.19, 0.19), (-0.11, 0.11), (0.85, 1.48), mat="alarm_red"))
    body.append(c.box("harness", (-0.2, 0.2), (-0.115, 0.115), (1.05, 1.1), mat="bolt_black"))
    c.bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(0, 0, 1.6))
    body.append(c.bpy.context.active_object)
    body[-1].data.materials.append(c.material("bottle_white"))
    c.bpy.ops.mesh.primitive_uv_sphere_add(radius=0.125, location=(0, 0.01, 1.66))
    hat = c.bpy.context.active_object
    hat.data.materials.append(c.material("hatch_yellow"))
    c.cut(hat, c.box("below", (-1, 1), (-1, 1), (0.0, 1.66)))
    body.append(hat)
    return body


export("technician", technician())

c.save_blend("systems")
print("built systems")
