"""Part: main bearing, main shaft (continuous), rotor lock (prismatic pin).

Faults: grease pushed out of the main bearing seal, grease collector full,
a shrink disc bolt backing out (torque marker misaligned), fretting corrosion
at the shaft/shrink disc interface, rotor lock left engaged.
"""
import math

from turbine import dims as d
from turbine.urdf import fault, inspection_point, link, span_box, x_cylinder

M = "main_shaft/"
W = d.MB_HALF_W
XB0 = d.MB_XS[0]
seg = lambda i: x_cylinder(d.MS_SEGMENTS[i][0], d.MS_SEGMENTS[i][1])  # noqa: E731


def foot_z(x):
    return d.nacelle_height_in_drivetrain(x, d.BP_TOP_Z)


BEARING_COLLISIONS = []
for xb in d.MB_XS:
    BEARING_COLLISIONS += [x_cylinder(d.MB_R, (xb - W, xb + W)),
                           span_box((xb - W, xb + W), (-d.MB_FOOT_Y, d.MB_FOOT_Y), (foot_z(xb) - 0.05, foot_z(xb) + 0.3)),
                           span_box((xb - W + 0.05, xb + W - 0.05), (-0.55, 0.55), (foot_z(xb), -0.3))]
COLLECTOR = (XB0 + 0.1, d.MB_COLLECTOR_Y, foot_z(XB0) + 0.3 + 0.1)

LINKS = [
    # the tilted shaft axis: origin at the hub centre, X upwind along the shaft
    link("drivetrain", "nacelle", xyz=d.HUB_IN_NACELLE, rpy=(0.0, -d.TILT, 0.0)),
    link("main_bearing", "drivetrain", mesh=M + "main_bearing.obj", collisions=BEARING_COLLISIONS),
    link("main_bearing_grease_collector", "main_bearing",
         variants={"ok": M + "grease_collector_ok.obj", "full": M + "grease_collector_full.obj"}),
    link("main_shaft", "drivetrain", mesh=M + "main_shaft.obj", joint="continuous", axis=(1, 0, 0),
         collisions=[x_cylinder(d.MS_FLANGE_R, d.MS_FLANGE_X), x_cylinder(d.MS_LOCK_DISC_R, d.MS_LOCK_DISC_X),
                     x_cylinder(d.MS_SHRINK_R, d.MS_SHRINK_X), seg(0), seg(1), seg(2), seg(3)]),
    link("main_shaft_shrink_disc_bolts", "main_shaft",
         variants={"ok": M + "shrink_disc_bolts_ok.obj", "loose": M + "shrink_disc_bolts_loose.obj"}),
    link("rotor_lock", "main_bearing", mesh=M + "rotor_lock.obj",
         collisions=[span_box(d.RL_HOUSING_X, (d.RL_Y - d.RL_HALF, d.RL_Y + d.RL_HALF), (-d.RL_HALF, d.RL_HALF))]),
    link("rotor_lock_pin", "rotor_lock", mesh=M + "rotor_lock_pin.obj", xyz=(d.RL_HOUSING_X[1], d.RL_Y, 0.0),
         joint="prismatic", axis=(1, 0, 0), limits=(0.0, d.RL_STROKE),
         collisions=[x_cylinder(d.RL_PIN_R, (-0.02, 0.0))]),   # only the part that sticks out
]

# camera positions lie over the walkways (|y| >= 1.35 m in the nacelle) at G1 head height
INSPECTION_POINTS = [
    inspection_point("main_bearing_inspect_front_seal", "drivetrain", (XB0 - W - 0.02, 0.3, -0.3), (-0.2, 1, 0.3),
                     "front main bearing, downwind seal: grease leaking out", distance=1.45),
    inspection_point("main_bearing_inspect_grease_collector", "drivetrain", COLLECTOR, (0.2, -1, 0.5),
                     "used-grease collector fill level", distance=1.1),
    inspection_point("main_shaft_inspect_shrink_disc", "drivetrain", (d.MS_SHRINK_X[1], -0.3, 0.0), (0.5, -1, 0.2),
                     "shrink disc bolts (torque markers) and fretting", distance=1.6),
    inspection_point("rotor_lock_inspect", "drivetrain", (d.RL_HOUSING_X[1] + 0.05, d.RL_Y, 0.0), (0.23, -1, 0.03),
                     "rotor lock pin position (must be retracted in operation)", distance=0.92),
]

FAULTS = {
    "main_bearing.grease_leak_front_seal": fault(
        "main_shaft", "main bearing upwind seal", "Front main bearing downwind seal worn: grease pushed out and running onto the bearing foot",
        "main_bearing_inspect_front_seal", ["rgb"], severity="medium",
        overlays=[link("main_bearing_fault_grease_leak", "main_bearing", mesh=M + "fault_grease_leak.obj")],
        signals={"main_bearing_temperature_c": 58}),
    "main_bearing.grease_collector_full": fault(
        "main_shaft", "grease collector", "Used-grease collector full, must be emptied",
        "main_bearing_inspect_grease_collector", ["rgb"], severity="low",
        set_variants={"main_bearing_grease_collector": "full"}),
    "main_shaft.shrink_disc_bolt_loose": fault(
        "main_shaft", "shrink disc", "One shrink disc bolt backed out, torque marker no longer aligned",
        "main_shaft_inspect_shrink_disc", ["rgb"], severity="high",
        set_variants={"main_shaft_shrink_disc_bolts": "loose"}),
    "main_shaft.fretting_corrosion": fault(
        "main_shaft", "shaft / shrink disc interface", "Fretting corrosion: red-brown powder where the shaft enters the shrink disc",
        "main_shaft_inspect_shrink_disc", ["rgb"], severity="medium",
        overlays=[link("main_shaft_fault_fretting", "main_shaft", mesh=M + "fault_fretting.obj")],
        signals={"drivetrain_vibration_mm_s": 4.8}),
    "rotor_lock.left_engaged": fault(
        "main_shaft", "rotor lock", "Rotor lock pin left engaged after maintenance",
        "rotor_lock_inspect", ["rgb"], severity="high",
        joint_states={"rotor_lock_pin_joint": d.RL_STROKE}),
}

SIGNALS = {"main_bearing_temperature_c": 41, "drivetrain_vibration_mm_s": 1.6}

# sanity: at shaft angle 0 a lock hole must sit in front of the pin, outside the bearing housing
assert d.MS_LOCK_HOLE_PCD - d.RL_HALF > d.MB_R, "rotor lock housing collides with the bearing housing"
_holes = [(-d.MS_LOCK_HOLE_PCD * math.sin(2 * math.pi * k / d.MS_LOCK_HOLES),
           d.MS_LOCK_HOLE_PCD * math.cos(2 * math.pi * k / d.MS_LOCK_HOLES)) for k in range(d.MS_LOCK_HOLES)]
assert any(abs(y - d.RL_Y) < 1e-6 and abs(z) < 1e-6 for y, z in _holes), "rotor lock pin misses the lock disc holes"
