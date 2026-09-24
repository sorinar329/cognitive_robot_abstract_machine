"""Part: bedplate (nacelle frame, level). Cast front bedplate with two side
girders carrying the main bearings and the gearbox torque arm brackets, a cable
opening on the yaw axis; welded rear frame for generator and transformer.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402

d = c.dims
c.reset_scene()
AY = (math.pi / 2, 0, 0)
parts = []
gy, fx = d.BP_GIRDER_Y, d.BP_FRONT_X
for side in (-1, 1):
    ys = sorted((side * gy[0], side * gy[1]))
    girder = c.box(f"girder{side}", fx, ys, (0.0, d.BP_TOP_Z), mat="gear_paint")
    holes = [c.cylinder(f"hole{side}_{i}", 0.13, 1.0, (x, side * sum(gy) / 2, d.BP_TOP_Z / 2), rotation=AY)
             for i, x in enumerate((-2.6, -1.9, -1.2))]
    c.cut(girder, *holes)
    parts.append(girder)
    # rear frame girders under generator and transformer
    ry = sorted((side * d.BP_REAR_Y[0], side * d.BP_REAR_Y[1]))
    parts.append(c.box(f"rear{side}", d.BP_REAR_X, ry, (0.0, d.BP_REAR_TOP_Z), mat="gear_paint"))

# front block under both main bearings, with the cable opening on the yaw axis
front = c.box("front_block", (-0.5, fx[1]), (-gy[1], gy[1]), (0.0, d.BP_TOP_Z), mat="gear_paint")
c.cut(front, c.cylinder("cable_hole", d.BP_CABLE_HOLE_R, 2.0, (0, 0, 0)))
parts.append(front)
for x in (d.BP_REAR_X[0] + 0.1, -7.6, -5.0, fx[0] + 0.1):          # rear frame cross members
    parts.append(c.box(f"cross{x}", (x - 0.1, x + 0.1), (-d.BP_REAR_Y[1], d.BP_REAR_Y[1]), (0.0, d.BP_REAR_TOP_Z), mat="gear_paint"))
parts.append(c.box("link_beam", (fx[0], fx[0] + 0.25), (-gy[1], gy[1]), (0.0, d.BP_TOP_Z), mat="gear_paint"))

# torque arm pin brackets: two plates per side around each gearbox bushing
for side in (-1, 1):
    for lx in (d.GB_BUSHING_X[0] - d.GB_PIN_EXT + 0.05, d.GB_BUSHING_X[1] + d.GB_PIN_EXT - 0.05):
        x, _, z = d.drivetrain_to_nacelle(d.GEARBOX_X + lx, d.GB_BUSHING_Z)
        yc = side * d.GB_BUSHING_Y
        parts.append(c.box(f"bracket{side}{lx}", (x - 0.05, x + 0.05), (yc - 0.12, yc + 0.12), (d.BP_TOP_Z, z + 0.14), mat="gear_paint"))
        for yy in (yc - 0.08, yc + 0.08):
            parts.append(c.hex_bolt(f"bb{side}{lx}{yy}", (x + (0.08 if lx > -0.5 else -0.08), yy, d.BP_TOP_Z + 0.007)))

c.export_part("bedplate", [c.join("bedplate", parts)])
c.save_blend("bedplate")
print("built bedplate")
