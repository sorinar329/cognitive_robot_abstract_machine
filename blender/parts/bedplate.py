"""Part: bedplate (main frame). Frame ``bedplate`` == ``nacelle`` frame.

Two box girders along X carry the front block (main bearing seat) and the
torque arm brackets; a low deck between them has the cable opening on the yaw
axis. Lightening holes in the girders make it read as a welded/cast frame.
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
gx, gy = d.BP_GIRDER_X, d.BP_GIRDER_Y
for side in (-1, 1):
    ys = sorted((side * gy[0], side * gy[1]))
    girder = c.box(f"girder{side}", gx, ys, (0.0, d.BP_TOP_Z), mat="gear_paint")
    holes = [c.cylinder(f"hole{side}_{i}", 0.17, 1.0, (x, side * sum(gy) / 2, d.BP_TOP_Z / 2), rotation=AY)
             for i, x in enumerate((0.3, 1.1, 1.7))]
    c.cut(girder, *holes)
    parts.append(girder)
    # torque arm pin brackets: two plates per side, pin eye at the bushing axis
    pin_z = d.SHAFT_Z + d.GB_BUSHING_Z
    for i, bx in enumerate(d.BP_BRACKET_XS):
        yc = side * d.GB_BUSHING_Y
        plate = c.box(f"bracket{side}_{i}", bx, (yc - d.BP_BRACKET_HALF_Y, yc + d.BP_BRACKET_HALF_Y),
                      (d.BP_TOP_Z, pin_z + 0.14), mat="gear_paint")
        parts.append(plate)
        for k, zz in enumerate((d.BP_TOP_Z + 0.01,)):
            for yy in (yc - 0.08, yc + 0.08):
                parts.append(c.hex_bolt(f"bb{side}{i}{yy}", (sum(bx) / 2 + (0.06 if i else -0.06), yy, zz + 0.007)))

front = c.box("front_block", d.BP_FRONT_X, (-gy[1], gy[1]), (0.0, d.BP_TOP_Z), mat="gear_paint")
parts.append(front)
deck = c.box("deck", d.BP_DECK_X, (-gy[0], gy[0]), (0.0, d.BP_DECK_Z), mat="gear_paint")
c.cut(deck, c.cylinder("cable_hole", d.BP_CABLE_HOLE_R, 1.0, (0, 0, 0)))
parts.append(deck)

bedplate = c.join("bedplate", parts)
c.export_part("bedplate", [bedplate])
c.save_blend("bedplate")
print("built bedplate")
