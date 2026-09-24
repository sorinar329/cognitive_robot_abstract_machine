"""Part: gearbox with torque arms, oil cooler, filter, sight glass, inspection cover.

Faults are the most common gearbox findings from nacelle inspections:
seal leaks, low oil, clogged filter, loose/missing bolts, worn torque arm bushings.
"""
from turbine import dims as d
from turbine.urdf import fault, inspection_point, link, span_box, x_cylinder

M = "gearbox/"
gx, gz = d.GB_GLASS_XZ
fx, fy = d.GB_FILTER_XY
top = d.GB_HELICAL_Z[1]

COLLISIONS = [
    x_cylinder(d.GB_FRONT_COVER_R, (d.GB_FRONT_COVER_X[0], d.GB_INPUT_X[1])),
    x_cylinder(d.GB_PLANET_R, d.GB_PLANET_X),
    span_box(d.GB_HELICAL_X, d.GB_HELICAL_Y, d.GB_HELICAL_Z),
    x_cylinder(d.GB_PLANET2_R, d.GB_PLANET2_X),
    span_box(d.GB_ARM_X, (-d.GB_ARM_Y_OUT, d.GB_ARM_Y_OUT), d.GB_ARM_Z),
    span_box(d.GB_COOLER_X, d.GB_COOLER_Y, (top, d.GB_COOLER_Z[1] + 0.03)),
    span_box((fx - d.GB_FILTER_R, fx + d.GB_FILTER_R), (fy - d.GB_FILTER_R, d.GB_HELICAL_Y[0]), d.GB_FILTER_Z),
]


def component(name, variants):
    """A fixed sub-link sharing the gearbox frame whose mesh depends on its state."""
    return link(name, "gearbox", variants={s: M + f for s, f in variants.items()})


LINKS = [
    link("gearbox", "drivetrain", mesh=M + "gearbox.obj", xyz=(d.GEARBOX_X, 0, 0), collisions=COLLISIONS),
    component("gearbox_sight_glass", {"ok": "sight_glass_ok.obj", "low": "sight_glass_low.obj"}),
    component("gearbox_filter_indicator", {"ok": "filter_indicator_ok.obj", "clogged": "filter_indicator_clogged.obj"}),
    component("gearbox_cover_bolts", {"ok": "cover_bolts_ok.obj", "missing": "cover_bolts_missing.obj"}),
    component("gearbox_bushing_left", {"ok": "bushing_left_ok.obj", "cracked": "bushing_left_cracked.obj"}),
]

INSPECTION_POINTS = [
    # cameras over the walkways at G1 head height (checked by scripts/check_inspection_points.py)
    inspection_point("gearbox_inspect_sight_glass", "gearbox", (gx, d.GB_HELICAL_Y[0] - 0.02, gz), (0.14, -1, 0.74),
                     "oil level in the sight glass", distance=1.17),
    inspection_point("gearbox_inspect_filter", "gearbox", (fx, fy, d.GB_FILTER_Z[1]), (0.23, -1, 0.22),
                     "oil filter clogging indicator (green = ok, red pin out = clogged)", distance=0.81),
    inspection_point("gearbox_inspect_input_seal", "gearbox", (0.0, -d.MS_SHRINK_R - 0.07, -0.35), (0.56, -1, 0.3),
                     "input shaft seal and front cover, below it the floor", distance=1.35),
    inspection_point("gearbox_inspect_cover", "gearbox", (sum(d.GB_COVER_X) / 2, d.GB_HELICAL_Y[1] + 0.03, sum(d.GB_COVER_Z) / 2),
                     (0.18, 1, 0.16), "side inspection cover bolts", distance=0.95),
    inspection_point("gearbox_inspect_bushing_left", "gearbox", (d.GB_BUSHING_X[1], -d.GB_BUSHING_Y, d.GB_BUSHING_Z),
                     (0.6, -0.3, 0.75), "torque arm elastomer bushing (-Y side)", distance=0.7),
    inspection_point("gearbox_inspect_bushing_right", "gearbox", (d.GB_BUSHING_X[1], d.GB_BUSHING_Y, d.GB_BUSHING_Z),
                     (0.6, 0.3, 0.75), "torque arm elastomer bushing (+Y side)", distance=0.7),
]

FAULTS = {
    "gearbox.oil_leak_input_seal": fault(
        "gearbox", "input shaft seal", "Worn input shaft seal: oil runs down the front cover and pools on the floor",
        "gearbox_inspect_input_seal", ["rgb"], severity="high",
        overlays=[
            link("gearbox_fault_oil_streak", "gearbox", mesh=M + "fault_oil_streak.obj"),
            # dripped onto the nacelle floor below the gearbox front, between the girders
            link("gearbox_fault_oil_puddle", "nacelle", mesh=M + "fault_oil_puddle.obj",
                 xyz=(d.drivetrain_to_nacelle(d.GEARBOX_X - 0.3)[0], -0.45, 0.001)),
        ],
        signals={"gearbox_oil_level": 0.8}),
    "gearbox.oil_level_low": fault(
        "gearbox", "oil sump", "Oil level below minimum in the sight glass",
        "gearbox_inspect_sight_glass", ["rgb"], severity="high",
        set_variants={"gearbox_sight_glass": "low"},
        signals={"gearbox_oil_level": 0.2, "gearbox_oil_temperature_c": 78}),
    "gearbox.filter_clogged": fault(
        "gearbox", "offline oil filter", "Oil filter clogged, differential pressure indicator popped out",
        "gearbox_inspect_filter", ["rgb"], severity="medium",
        set_variants={"gearbox_filter_indicator": "clogged"},
        signals={"gearbox_filter_dp_bar": 3.2, "gearbox_oil_particle_count": 1800}),
    "gearbox.cover_bolt_missing": fault(
        "gearbox", "inspection cover", "One inspection cover bolt missing (loosened by vibration)",
        "gearbox_inspect_cover", ["rgb"], severity="low",
        set_variants={"gearbox_cover_bolts": "missing"}),
    "gearbox.torque_arm_bushing_cracked": fault(
        "gearbox", "torque arm bushing (-Y)", "Elastomer bushing aged: torn and bulging out of the arm eye, gearbox vibrates more",
        "gearbox_inspect_bushing_left", ["rgb", "vibration"], severity="medium",
        set_variants={"gearbox_bushing_left": "cracked"},
        signals={"gearbox_vibration_mm_s": 7.5}),
}

# healthy baseline of the non-visual signals (read by simulated sensors)
SIGNALS = {"gearbox_oil_level": 1.0, "gearbox_oil_temperature_c": 62, "gearbox_filter_dp_bar": 0.8,
           "gearbox_oil_particle_count": 300, "gearbox_vibration_mm_s": 2.1}
