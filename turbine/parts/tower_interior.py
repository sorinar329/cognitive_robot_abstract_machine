"""Tower interior: rest platforms under the flange joints, the yaw deck (top
platform), ladders, cables, the cable loop, the ground controller and the service
lift. The lift car is a prismatic joint on the tower axis direction: 0 = car floor at
ground level, LIFT_TRAVEL = at the yaw deck.

Platform collisions leave the lift shaft, the ladder and (yaw deck) the cable loop
open, so a robot sees where it can stand.
"""
import math

from turbine import bolts as tb
from turbine.dims import site as s
from turbine.urdf import fault, floor_boxes, inspection_point, link, span_box

M = "tower_interior/"
ZS = [0.0] + list(s.TOWER_FLANGES) + [s.TOWER_HEIGHT]
LX, LY = s.LIFT_CENTER
CX, CY, CZ = s.LIFT_CAR


def r_in(z):
    return s.tower_radius(z) - s.tower_wall(z)


def section_of(z):
    return max(k for k in range(1, 5) if ZS[k - 1] <= z)


def ladder_point(z):
    r = r_in(z) - s.LADDER_WALL_GAP
    return r * math.cos(s.LADDER_AZIMUTH), r * math.sin(s.LADDER_AZIMUTH)


def platform(i, z):
    k = section_of(z)
    local = z - ZS[k - 1]
    m = s.LIFT_SHAFT_MARGIN
    lx, ly = ladder_point(z)
    holes = [((LX - CX / 2 - m, LX + CX / 2 + m), (LY - CY / 2 - m, LY + CY / 2 + m)),
             ((lx - 0.35, lx + 0.35), (ly - 0.35, ly + 0.35))]
    if i == len(PLATFORMS):
        holes.append(((-s.CABLE_HOLE_R, s.CABLE_HOLE_R), (-s.CABLE_HOLE_R, s.CABLE_HOLE_R)))
    return link(f"platform_{i}", f"tower_section_{k}", mesh=M + f"platform_{i}.obj",
                collisions=floor_boxes((local - s.PLATFORM_T, local), radius=r_in(z) - 0.02, holes=holes))


PLATFORMS = s.platform_heights()
GROUND = -s.TOWER_BASE_Z                    # ground level in the section-1 frame
CAR = [span_box((-CX / 2, CX / 2), (-CY / 2, CY / 2), (-0.04, 0.0)),               # floor
       span_box((-CX / 2, CX / 2), (-CY / 2 - 0.02, -CY / 2), (0.0, 1.1)),          # closed sides (bars) and back
       span_box((-CX / 2, CX / 2), (CY / 2, CY / 2 + 0.02), (0.0, 1.1)),
       span_box((-CX / 2 - 0.02, -CX / 2), (-CY / 2, CY / 2), (0.0, 1.1)),
       span_box((-CX / 2, CX / 2), (-CY / 2, CY / 2), (CZ, CZ + 0.3))]             # roof and hoist

LINKS = [platform(i, z) for i, z in enumerate(PLATFORMS, start=1)]
LINKS += [link(f"ladder_{k}", f"tower_section_{k}", mesh=M + f"ladder_{k}.obj") for k in range(1, 5)]
LINKS += [
    link("cable_loop", "tower_section_4", mesh=M + "cable_loop.obj"),
    link("ground_controller", "tower_section_1", mesh=M + "ground_controller.obj",
         collisions=[span_box((-0.45, 0.45), (r_in(0.5) - 0.6, r_in(0.5) - 0.05), (GROUND, GROUND + 2.0))]),
    link("lift_rails", "tower_section_1", mesh=M + "lift_rails.obj"),
    link("service_lift", "tower_section_1", mesh=M + "lift_car.obj", xyz=(LX, LY, s.LIFT_BOTTOM_Z),
         joint="prismatic", axis=(0, 0, 1), limits=(-0.1, s.LIFT_TRAVEL + 0.2), velocity=s.LIFT_SPEED, collisions=CAR),
]

# flange bolts with torque markings (section-k frame, where flange k is the top of section k)
for k in range(1, len(s.TOWER_FLANGES) + 1):
    name, mesh = f"flange_{k}_bolts", M + f"flange_{k}_bolts_"
    LINKS.append(link(name, f"tower_section_{k}", variants={"ok": mesh + "ok.obj", "loose": mesh + "loose.obj"})
                 if k in s.FLANGE_MARKING_FAULTS else link(name, f"tower_section_{k}", mesh=mesh + "ok.obj"))


def flange_point(k, bolt):
    """The torque marking on a nut, seen from the tower axis and a little above."""
    fl = tb.flange(k)
    _, a, x, y = fl["bolts"][bolt]
    ap = fl["spec"]["nut_af"] / 2
    return inspection_point(f"tower_inspect_flange_{k}", f"tower_section_{k}",
                            (x - ap * math.cos(a), y - ap * math.sin(a), tb.nut_height(k) - ZS[k - 1]),
                            (-math.cos(a), -math.sin(a), 0.2),
                            f"flange {k} bolts ({fl['spec']['size']}): torque markings on the nuts under the flange",
                            distance=0.55, zone="tower")


# the cable loop where it passes the saddle on the yaw deck (section 4 frame; see blender/parts/tower_interior.py)
CHAFE = (0.069 + 0.03, 0.069 + 0.09, PLATFORMS[-1] - ZS[3] + 0.3)

INSPECTION_POINTS = [flange_point(k, max(s.FLANGE_MARKING_FAULTS.get(k, {0: 0}).items(), key=lambda kv: kv[1])[0])
                     for k in range(1, len(s.TOWER_FLANGES) + 1)] + [
    inspection_point("tower_inspect_cable_loop", "tower_section_4", CHAFE, (0.3, 0.95, 0.95),
                     "power cable loop at the saddle on the yaw deck: chafed insulation", distance=1.45, zone="tower"),
]

FAULTS = {
    "tower.cable_loop_chafed": fault(
        "tower_interior", "cable loop", "Cable loop rubbed through at the saddle edge, copper showing, insulation flakes on the deck",
        "tower_inspect_cable_loop", ["rgb"], severity="high",
        overlays=[link("tower_fault_cable_chafe", "tower_section_4", mesh=M + "fault_cable_chafe.obj")]),
}

for k, turned in s.FLANGE_MARKING_FAULTS.items():
    b = tb.spec(s.FLANGE_BOLTS[k - 1][0])
    worst = max(turned.values())
    FAULTS[f"tower.flange_{k}_bolts_loose"] = fault(
        "tower_interior", f"flange {k} bolts",
        f"Torque markings turned on flange {k}: " + ", ".join(f"nut {i} by {v:.0f} deg" for i, v in sorted(turned.items()))
        + f" (turning back {b['turn_to_preload_deg']:.0f} deg releases all preload)",
        f"tower_inspect_flange_{k}", ["rgb"], severity="high" if tb.preload_ratio(worst, b) < tb.BOLT_MIN else "medium",
        set_variants={f"flange_{k}_bolts": "loose"},
        signals={f"flange_{k}_marking_offsets_deg": {i: float(v) for i, v in turned.items()}})

SIGNALS = {}
