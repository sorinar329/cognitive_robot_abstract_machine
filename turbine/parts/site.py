"""Outside at ground level up to the tower top: ground, foundation (plinth, grout,
earthing), 4 tower sections, entrance door (at ground level), yaw bearing.

Ground-level inspection points are what the G1 reaches on flat ground (task T0/T1).
"""
import math

from turbine.dims import site as s
from turbine.urdf import disk_boxes, fault, inspection_point, link, ring_boxes, span_box, z_cylinder

M = "site/"
BASE = s.TOWER_BASE_Z
ZS = [0.0] + list(s.TOWER_FLANGES) + [s.TOWER_HEIGHT]
RD = s.tower_radius(s.DOOR_SILL_Z)
HINGE = (s.DOOR_WIDTH / 2, -math.sqrt(RD ** 2 - (s.DOOR_WIDTH / 2) ** 2), s.DOOR_SILL_Z)
DOOR_HALF_ANGLE = math.asin(s.DOOR_WIDTH / 2 / RD) + 0.05


def radial(r, azimuth_deg, z):
    a = math.radians(azimuth_deg)
    return (r * math.cos(a), r * math.sin(a), z)


def outward(azimuth_deg, up=0.3):
    a = math.radians(azimuth_deg)
    return (math.cos(a), math.sin(a), up)


def section(k):
    z0, z1 = ZS[k - 1], ZS[k]

    def skip(a, zm):   # leave the door opening free in section 1
        d = (a - s.DOOR_AZIMUTH + math.pi) % (2 * math.pi) - math.pi
        return k == 1 and abs(d) < DOOR_HALF_ANGLE and s.DOOR_SILL_Z - 0.2 < zm < s.DOOR_SILL_Z + s.DOOR_HEIGHT + 0.2
    walls = ring_boxes(lambda z: s.tower_radius(z + z0), lambda z: s.tower_wall(z + z0), 0.0, z1 - z0,
                       segments=24 if k == 1 else 16, band=2.2 if k == 1 else 10.8, skip=skip)
    parent = "foundation" if k == 1 else f"tower_section_{k - 1}"
    xyz = (0, 0, BASE) if k == 1 else (0, 0, z0 - ZS[k - 2])
    return link(f"tower_section_{k}", parent, mesh=M + f"tower_section_{k}.obj", xyz=xyz, collisions=walls)


LINKS = [
    link("ground", "world", mesh=M + "ground.obj",
         collisions=[span_box((-s.GROUND_RADIUS, s.GROUND_RADIUS), (-s.GROUND_RADIUS, s.GROUND_RADIUS), (-0.05, 0.0))]),
    link("foundation", "world", mesh=M + "foundation.obj",
         collisions=disk_boxes(s.PLINTH_R, (0.0, s.PLINTH_Z[1]))),
    link("foundation_grout", "foundation", variants={"ok": M + "grout_ok.obj", "broken": M + "grout_broken.obj"}),
    link("foundation_earthing", "foundation", variants={"ok": M + "earthing_ok.obj", "loose": M + "earthing_loose.obj"}),
    section(1), section(2), section(3), section(4),
    link("tower_door", "tower_section_1", mesh=M + "tower_door.obj", xyz=HINGE,
         joint="revolute", axis=(0, 0, 1), limits=(0.0, 1.7),
         collisions=[span_box((-s.DOOR_WIDTH + 0.02, 0.0), (-0.1, 0.06), (0.01, s.DOOR_HEIGHT - 0.01))]),
    link("yaw_bearing", "tower_section_4", mesh=M + "yaw_bearing.obj", xyz=(0, 0, ZS[4] - ZS[3]),
         collisions=[z_cylinder(s.YAW_BEARING_R[1], (0.0, s.YAW_BEARING_H))]),
]

INSPECTION_POINTS = [
    # the G1's camera looks down: the door (above its camera) only fits in the image from further away
    inspection_point("tower_inspect_door", "tower_section_1", (0.0, -RD - 0.05, s.DOOR_SILL_Z + 0.5), (0, -1, 0.0),
                     "tower entrance door (closed, undamaged)", distance=6.0, outside=True),
    inspection_point("foundation_inspect_plinth", "foundation", radial(s.PLINTH_R - 0.25, -45, s.PLINTH_Z[1]), outward(-45, 1.2),
                     "foundation plinth concrete: cracks, spalling", distance=2.0, outside=True),
    inspection_point("foundation_inspect_grout", "foundation", radial(s.GROUT_R[1], -30, s.GROUT_Z[1]), outward(-30, 0.6),
                     "grout joint under the tower base flange", distance=1.8, outside=True),
    inspection_point("foundation_inspect_earthing", "foundation", radial(s.PLINTH_R - 0.35, -120, s.PLINTH_Z[1]), outward(-120, 0.9),
                     "earthing strap from the tower flange to the plinth terminal", distance=1.8, outside=True),
    inspection_point("tower_inspect_anchor_nuts", "tower_section_1", radial(s.ANCHOR_NUT_R, -150, s.BASE_FLANGE_H), outward(-150, 0.8),
                     "anchor bolt nuts on the base flange", distance=1.8, outside=True),
    inspection_point("tower_inspect_base_coating", "tower_section_1", radial(s.tower_radius(0.6), -56, 0.6), outward(-56, 0.15),
                     "tower coating near the base: corrosion, paint damage", distance=3.0, outside=True),
]

FAULTS = {
    "foundation.plinth_crack": fault(
        "site", "foundation plinth", "Crack running down the plinth and across its top towards the grout",
        "foundation_inspect_plinth", ["rgb"], severity="medium",
        overlays=[link("foundation_fault_plinth_crack", "foundation", mesh=M + "fault_plinth_crack.obj")]),
    "foundation.grout_breakout": fault(
        "site", "grout joint", "Grout broken out under the base flange, debris on the plinth",
        "foundation_inspect_grout", ["rgb"], severity="high",
        set_variants={"foundation_grout": "broken"}),
    "foundation.earthing_disconnected": fault(
        "site", "earthing strap", "Earthing strap torn off the tower flange, lying on the plinth",
        "foundation_inspect_earthing", ["rgb"], severity="high",
        set_variants={"foundation_earthing": "loose"}, signals={"earthing_resistance_ohm": 999.0}),
    "tower.anchor_nut_corrosion": fault(
        "site", "anchor bolts", "Corroded anchor nuts in one sector, rust running onto the plinth",
        "tower_inspect_anchor_nuts", ["rgb"], severity="medium",
        overlays=[link("tower_fault_anchor_corrosion", "tower_section_1", mesh=M + "fault_anchor_corrosion.obj")]),
    "tower.base_coating_damage": fault(
        "site", "tower coating", "Paint damage with rust patches and a rust streak near the tower base",
        "tower_inspect_base_coating", ["rgb"], severity="low",
        overlays=[link("tower_fault_coating", "tower_section_1", mesh=M + "fault_tower_coating.obj")]),
}

SIGNALS = {"earthing_resistance_ohm": 2.1}
