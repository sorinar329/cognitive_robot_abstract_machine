"""Part: fast shaft with brake disc and flexible coupling (rotating), brake
caliper, and the DFIG generator with terminal box and slip-ring housing.

All in the ``drivetrain`` frame; the rotating parts (``fast_shaft``) and the
generator have their frame on the fast shaft axis (drivetrain z = GB_OUTPUT_Z).

Outputs (models/generator/):
  fast_shaft.obj, brake_disc_{ok,overheated}.obj, coupling_{ok,cracked}.obj
  brake_caliper_{ok,worn}.obj, generator.obj
  fault_carbon_dust.obj, fault_terminal_heat.obj
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

d = c.dims
AX = (0, math.pi / 2, 0)
P = "generator"
c.reset_scene()


def xcyl(name, r, x, y=0.0, z=0.0, mat="pipe_steel", vertices=64):
    return c.cylinder(name, r, x[1] - x[0], ((x[0] + x[1]) / 2, y, z), rotation=AX, vertices=vertices, mat=mat)


def export(name, objs):
    c.export_part(P, [c.join(name, objs) if len(objs) > 1 else objs[0]], obj_name=name)


def floor_z(x):
    """Frame-local z (axis at 0) of the rear frame top at drivetrain x."""
    return d.nacelle_height_in_drivetrain(x, d.BP_REAR_TOP_Z) - d.GB_OUTPUT_Z


# %% fast shaft (rotating): shaft, coupling hubs
export("fast_shaft", [xcyl("hss", d.HSS_R, d.HSS_X)])


def brake_disc(overheated):
    disc = xcyl("disc", d.BRAKE_DISC_R, d.BRAKE_DISC_X, mat="pipe_steel", vertices=96)
    objs = [disc, xcyl("disc_hub", 0.22, (d.BRAKE_DISC_X[0] - 0.05, d.BRAKE_DISC_X[1] + 0.05), mat="gear_paint")]
    if overheated:   # blue/purple temper bands on both friction faces
        for sx, x in ((-1, d.BRAKE_DISC_X[0]), (1, d.BRAKE_DISC_X[1])):
            band = xcyl("band", d.BRAKE_DISC_R - 0.03, (x + sx * 0.001, x + sx * 0.003), mat="heat_blue", vertices=96)
            c.cut(band, xcyl("band_in", d.BRAKE_DISC_R - 0.17, (-20, 20)))
            objs.append(band)
    return objs


def coupling(cracked):
    x0, x1 = d.COUPLING_X
    objs = [xcyl("hub_a", 0.2, (x1 - 0.08, x1)), xcyl("hub_b", 0.2, (x0, x0 + 0.08)),
            xcyl("spacer", 0.12, (x0 + 0.1, x1 - 0.1), mat="gear_paint")]
    for i, (a, b) in enumerate(((x1 - 0.1, x1 - 0.08), (x0 + 0.08, x0 + 0.1))):
        pack = xcyl(f"pack{i}", d.COUPLING_R, (a, b), mat="bolt_black")
        if cracked and i == 0:   # a radial crack through the composite disc pack, one segment displaced
            # crack faces the -Y walkway (angle 1.2 rad ~ towards -Y, slightly up)
            c.cut(pack, c.radial_box("crack", (a - 0.01, b + 0.01), (0.08, d.COUPLING_R + 0.05), 0.03, 1.2))
            chip = c.radial_box("chip", (a - 0.004, b - 0.004), (0.14, d.COUPLING_R - 0.01), 0.07, 1.42, mat="bolt_black")
            c.placed(chip, c.translation((0.035, 0.0, 0.0)))
            objs.append(chip)
        objs.append(pack)
        for k in range(6):
            y, z = c.polar(d.COUPLING_R - 0.05, 2 * math.pi * k / 6)
            objs.append(c.hex_bolt(f"cb{i}{k}", ((a + b) / 2, y, z), axis="X"))
    return objs


for name, objs in (("brake_disc_ok", brake_disc(False)), ("brake_disc_overheated", brake_disc(True)),
                   ("coupling_ok", coupling(False)), ("coupling_cracked", coupling(True))):
    export(name, objs)


# %% brake caliper (fixed): straddles the top of the disc; pads are the variant
def caliper(pad):
    x0, x1 = d.BRAKE_DISC_X
    z0, z1 = d.CALIPER_Z
    hy = d.CALIPER_Y_HALF
    objs = [c.box("bridge", (x0 - 0.12, x1 + 0.12), (-hy, hy), (z1, z1 + 0.08), mat="alarm_red"),
            c.box("jaw_front", (x1 + 0.04, x1 + 0.12), (-hy, hy), (z0, z1 + 0.01), mat="alarm_red"),
            c.box("jaw_rear", (x0 - 0.12, x0 - 0.04), (-hy, hy), (z0, z1 + 0.01), mat="alarm_red"),
            c.box("mount", (x0 - 0.12, x1 + 0.12), (-0.05, 0.05), (z1 + 0.08, z1 + 0.35), mat="gear_paint")]
    for side, x in ((1, x1), (-1, x0)):    # piston pushes the pad backing plate against the disc face
        gap = 0.04 - pad - 0.004
        back = (x + side * (pad + 0.001), x + side * (pad + 0.001 + 0.008))
        objs.append(c.box("backing", tuple(sorted(back)), (-hy + 0.02, hy - 0.02), (z0 + 0.02, z1 - 0.02), mat="pipe_steel"))
        objs.append(c.box("pad", tuple(sorted((x + side * 0.001, x + side * (pad + 0.001)))), (-hy + 0.03, hy - 0.03),
                          (z0 + 0.03, z1 - 0.03), mat="rubber"))
        objs.append(c.box("piston", tuple(sorted((back[1], back[1] + side * gap))), (-0.04, 0.04), (0.36, 0.44), mat="pipe_steel"))
    if pad < d.BRAKE_PAD_OK:   # wear indicator pin sticks out, brake dust below the caliper
        objs.append(c.cylinder("wear_pin", 0.01, 0.06, (x1 + 0.08, hy + 0.01, z1 + 0.03), mat="alarm_red"))
        objs.append(c.puddle("brake_dust", 0.12, (x0 - 0.05, 0.0, floor_z(x0) + 0.002), seed=9, mat="carbon"))
    return objs


export("brake_caliper_ok", caliper(d.BRAKE_PAD_OK))
export("brake_caliper_worn", caliper(d.BRAKE_PAD_WORN))

# %% generator (fixed): finned housing, end shields, feet, terminal box, slip-ring housing
gx, h = d.GEN_X, d.GEN_HALF
body = c.box("housing", gx, (-h, h), (-h, h), mat="generator_paint")
bev = body.modifiers.new("round", "BEVEL")
bev.width, bev.segments = 0.12, 3
c.bpy.context.view_layer.objects.active = body
c.bpy.ops.object.modifier_apply(modifier=bev.name)
parts = [body, xcyl("front_shield", 0.55, (gx[1], gx[1] + 0.12), mat="generator_paint"),
         xcyl("bearing_cap", 0.2, (gx[1] + 0.12, gx[1] + 0.2), mat="gear_paint")]
for k in range(14):                          # cooling fins on top and both sides
    x = gx[0] + 0.12 + k * (gx[1] - gx[0] - 0.24) / 13
    parts.append(c.box(f"fin_top{k}", (x - 0.012, x + 0.012), (-h + 0.1, h - 0.1), (h - 0.02, h + 0.07), mat="generator_paint"))
    for sy in (-1, 1):
        parts.append(c.box(f"fin_side{k}{sy}", (x - 0.012, x + 0.012), tuple(sorted((sy * (h - 0.02), sy * (h + 0.07)))),
                           (-h + 0.1, h - 0.1), mat="generator_paint"))
for x in (gx[0] + 0.3, gx[1] - 0.3):          # feet onto the rear frame via rubber mounts
    for sy in (-1, 1):
        fz = floor_z(x)
        parts.append(c.box("foot", (x - 0.15, x + 0.15), tuple(sorted((sy * 0.5, sy * 0.85))), (fz + 0.08, -h + 0.05), mat="generator_paint"))
        parts.append(c.cylinder("mount", 0.07, 0.08, (x, sy * 0.65, fz + 0.04), mat="rubber"))
tx = d.GEN_TERMINAL_X
top = h + 0.07
parts.append(c.box("terminal_box", tx, (-0.3, 0.3), (top, top + 0.32), mat="generator_paint"))
for i, y in enumerate((-0.18, 0.0, 0.18)):
    parts.append(c.cylinder(f"gland{i}", 0.035, 0.08, ((tx[0] + tx[1]) / 2, y, top + 0.36), mat="bolt_black"))
    parts.append(c.rod(f"cable{i}", ((tx[0] + tx[1]) / 2, y, top + 0.38), ((tx[0] + tx[1]) / 2 - 0.4, y, top + 1.2), 0.03, mat="rubber"))
sx0, sx1 = d.SLIP_X
parts.append(c.box("slip_housing", (sx0, sx1), (-d.SLIP_HALF, d.SLIP_HALF), (-d.SLIP_HALF, d.SLIP_HALF), mat="generator_paint"))
parts.append(c.box("brush_cover", (sx0 + 0.06, sx1 - 0.06), (-d.SLIP_HALF - 0.02, -d.SLIP_HALF), (-0.3, 0.3), mat="steel_grey"))
parts.append(xcyl("encoder", 0.08, (sx0 - 0.1, sx0), mat="bolt_black"))
export("generator", parts)

# %% fault overlays
# carbon dust from worn slip-ring brushes: settled below the housing and smeared on the cover
dust = [c.puddle("dust_floor", 0.3, ((sx0 + sx1) / 2, -0.2, floor_z(sx0) + 0.003), seed=21, mat="carbon")]
rng = random.Random(4)
for i in range(6):
    y = -d.SLIP_HALF - 0.023
    x = rng.uniform(sx0 + 0.1, sx1 - 0.1)
    z = rng.uniform(-0.35, -0.05)
    dust.append(c.box(f"smear{i}", (x - 0.05, x + 0.05), (y - 0.002, y), (z - 0.12, z), mat="carbon"))
export("fault_carbon_dust", dust)
# overheating terminal box: scorched paint on the lid and around the glands
heat = [c.box("scorch_lid", (tx[0] + 0.1, tx[1] - 0.1), (-0.25, 0.25), (top + 0.32, top + 0.325), mat="heat_brown"),
        c.box("scorch_side", (tx[0] + 0.15, tx[1] - 0.15), (-0.305, -0.3), (top + 0.1, top + 0.3), mat="heat_brown")]
export("fault_terminal_heat", heat)

c.save_blend("generator")
print("built generator")
