"""Rotor: hub casting, pitch bearings and spinner (frame ``hub``: rotor axis X
upwind), and the blade lofted from the IEA-3.4-130-RWT planform (frame
``blade``: root centre, Z to the tip, X upwind, leading edge towards -Y).

Outputs: hub.obj, blade.obj, and blade fault overlays (blade frame):
fault_le_erosion, fault_lightning, fault_te_crack.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as c  # noqa: E402
from turbine import blade_geometry as bg  # noqa: E402
from turbine import frames  # noqa: E402

r = c.rotor_dims
c.reset_scene()
P = "rotor"


def blade_placement(i, radial):
    """Hub-frame 4x4 for a point on blade i's axis at ``radial``, Z along the blade."""
    m = frames.blade_rotation(i)
    t = frames.apply(m, (0, 0, radial))
    return frames.homogeneous(m, t)


# %% hub + spinner
x0, x1 = r.SPINNER_X
profile = []
for k in range(25):
    x = x0 + (x1 - x0) * k / 24
    if x <= 0.3:
        rr = 2.05 + (r.SPINNER_R - 2.05) * (x - x0) / (0.3 - x0)
    else:
        rr = r.SPINNER_R * math.sqrt(max(0.0, 1 - ((x - 0.3) / (x1 - 0.3)) ** 2))
    profile.append((x, max(rr, 0.02)))
outer = c.revolve("spinner", profile, segments=72, axis="X", mat="grp_white", close_ends=(True, True))
inner = c.revolve("spinner_in", [(x - 0.0, max(rr - 0.04, 0.01)) for x, rr in profile[:-1]] + [(x1 + 0.2, 0.01)],
                  segments=72, axis="X", close_ends=(True, True))
c.placed(inner, c.translation((-0.05, 0, 0)))
c.cut(outer, inner)
openings = []
for i in range(r.BLADES):
    hole = c.cylinder(f"opening{i}", r.BLADE_OPENING_R, 2.0, (0, 0, 0), vertices=48)
    c.placed(hole, blade_placement(i, 2.2))
    openings.append(hole)
c.cut(outer, *openings)
parts = [outer]
casting = c.bpy.ops.mesh.primitive_uv_sphere_add(radius=r.CASTING_R, segments=48, ring_count=24, location=(0, 0, 0))
casting = c.bpy.context.active_object
casting.data.materials.append(c.material("gear_paint"))
parts.append(casting)
for i in range(r.BLADES):
    neck = c.cylinder(f"neck{i}", 1.2, r.HUB_R - 1.0, (0, 0, 0), vertices=48, mat="gear_paint")
    c.placed(neck, blade_placement(i, 1.0 + (r.HUB_R - 1.0) / 2))
    ring = c.cylinder(f"pitch_bearing{i}", r.PITCH_BEARING_R[1], 0.18, (0, 0, 0), vertices=64, mat="pipe_steel")
    c.cut(ring, c.cylinder("ring_in", r.PITCH_BEARING_R[0], 1.0, (0, 0, 0), vertices=64))
    c.placed(ring, blade_placement(i, r.HUB_R - 0.09))
    parts += [neck, ring]
# shaft flange at the back of the hub
parts.append(c.cylinder("hub_flange", 1.0, 0.2, (x0 + 0.35, 0, 0), rotation=(0, math.pi / 2, 0), vertices=64, mat="gear_paint"))
c.export_part(P, [c.join("hub", parts)], obj_name="hub")

# %% blade
verts, faces = bg.surface_mesh()
blade = c.mesh_object("blade", verts, faces, "blade_white", smooth=True)
receptors = []
for side in ("suction", "pressure"):
    for t in (0.9, 0.975):
        p = bg.surface_point(t, side, 0.35, offset=0.004)
        c.bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=p)
        receptors.append(c.bpy.context.active_object)
        receptors[-1].data.materials.append(c.material("pipe_steel"))
c.export_part(P, [c.join("blade", [blade] + receptors)], obj_name="blade")

n = bg.POINTS_PER_SIDE
le = n - 1                                  # ring index of the leading edge


def overlay(name, span_range, index_fn, mat, offset=0.004):
    v, f = bg.surface_patch(span_range, index_fn, offset=offset)
    c.export_part(P, [c.mesh_object(name, v, f, mat)], obj_name=name)


def band_around_le(t, u):
    """Across the patch: suction side at chord w -> leading edge -> pressure side at w."""
    w = 0.07 + 0.03 * math.sin(41 * t) + 0.015 * math.sin(97 * t)
    return bg.ring_position("suction", w * (1 - 2 * u)) if u < 0.5 else bg.ring_position("pressure", w * (2 * u - 1))


# leading-edge erosion on the outer blade: a ragged band wrapped around the LE
overlay("fault_le_erosion", (0.8, 0.985), band_around_le, "erosion")
# lightning strike near the tip receptor on the suction side
overlay("fault_lightning", (0.955, 0.985),
        lambda t, u: bg.ring_position("suction", 0.35 + (u - 0.5) * (0.1 + 0.25 * math.sin(math.pi * (t - 0.955) / 0.03))), "burn")
# trailing-edge crack (split bond line) near the maximum chord: dark strip along the TE, suction side
overlay("fault_te_crack", (0.19, 0.29),
        lambda t, u: bg.ring_position("suction", 1.0 - u * (0.012 + 0.006 * math.sin(157 * t))), "crack", offset=0.006)

c.save_blend("rotor")
print("built rotor")
