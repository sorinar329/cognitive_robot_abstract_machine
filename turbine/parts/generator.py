"""Part: fast shaft (continuous) with brake disc and flexible coupling, brake
caliper, and the DFIG generator. Frames sit on the fast shaft axis, GB_OUTPUT_Z
above the main shaft axis in the drivetrain frame.

Faults: worn brake pads, overheated brake disc, cracked coupling disc pack,
carbon dust from the slip-ring brushes, overheating terminal box.
"""
from turbine import dims as d
from turbine.urdf import fault, inspection_point, link, span_box, x_cylinder

M = "generator/"
AXIS = (0.0, 0.0, d.GB_OUTPUT_Z)
h, gx, tx = d.GEN_HALF, d.GEN_X, d.GEN_TERMINAL_X
x0, x1 = d.BRAKE_DISC_X

LINKS = [
    link("fast_shaft", "drivetrain", mesh=M + "fast_shaft.obj", xyz=AXIS, joint="continuous", axis=(1, 0, 0),
         collisions=[x_cylinder(d.HSS_R, d.HSS_X)]),
    link("fast_shaft_brake_disc", "fast_shaft",
         variants={"ok": M + "brake_disc_ok.obj", "overheated": M + "brake_disc_overheated.obj"},
         collisions=[x_cylinder(d.BRAKE_DISC_R, d.BRAKE_DISC_X)]),
    link("fast_shaft_coupling", "fast_shaft",
         variants={"ok": M + "coupling_ok.obj", "cracked": M + "coupling_cracked.obj"},
         collisions=[x_cylinder(d.COUPLING_R, d.COUPLING_X)]),
    link("brake_caliper", "drivetrain", xyz=AXIS,
         variants={"ok": M + "brake_caliper_ok.obj", "worn": M + "brake_caliper_worn.obj"},
         collisions=[span_box((x0 - 0.12, x1 + 0.12), (-d.CALIPER_Y_HALF, d.CALIPER_Y_HALF), (d.CALIPER_Z[0], d.CALIPER_Z[1] + 0.35))]),
    link("generator", "drivetrain", mesh=M + "generator.obj", xyz=AXIS,
         collisions=[span_box(gx, (-h - 0.07, h + 0.07), (-h, h + 0.07)),
                     span_box(tx, (-0.3, 0.3), (h + 0.07, h + 0.4)),
                     span_box(d.SLIP_X, (-d.SLIP_HALF - 0.02, d.SLIP_HALF), (-d.SLIP_HALF, d.SLIP_HALF))]),
]

INSPECTION_POINTS = [
    # from the -Y walkway, slightly upwind: the disc's friction face below the caliper and the caliper jaws
    inspection_point("brake_inspect_caliper", "brake_caliper", (x1 + 0.005, -0.25, 0.1), (0.33, -1, -0.03),
                     "brake pads (thickness, wear pin) and the disc's friction face", distance=1.47),
    inspection_point("coupling_inspect", "brake_caliper", (sum(d.COUPLING_X) / 2, -d.COUPLING_R, 0.0), (0.03, -1, 0.05),
                     "flexible coupling disc packs: cracks, loose bolts", distance=1.3),
    inspection_point("generator_inspect_slip_ring", "generator", (sum(d.SLIP_X) / 2, -d.SLIP_HALF, -0.2), (0.0, -1, 0.4),
                     "slip-ring housing and brush cover: carbon dust", distance=1.3),
    inspection_point("generator_inspect_terminal_box", "generator", (sum(tx) / 2, -0.3, h + 0.2), (0.14, -1, -0.44),
                     "generator terminal box: overheating (thermal camera), scorched paint", distance=1.54),
]

FAULTS = {
    "brake.pads_worn": fault(
        "generator", "brake pads", "Brake pads worn to the backing plate, wear pin out, brake dust below",
        "brake_inspect_caliper", ["rgb"], severity="medium",
        set_variants={"brake_caliper": "worn"}, signals={"brake_pad_wear_percent": 92}),
    "brake.disc_overheated": fault(
        "generator", "brake disc", "Brake disc overheated: blue temper colours on the friction faces",
        "brake_inspect_caliper", ["rgb", "thermal"], severity="high",
        set_variants={"fast_shaft_brake_disc": "overheated"}, signals={"brake_disc_temperature_c": 185}),
    "coupling.disc_pack_cracked": fault(
        "generator", "flexible coupling", "Crack through the gearbox-side coupling disc pack, a segment displaced",
        "coupling_inspect", ["rgb", "vibration"], severity="high",
        set_variants={"fast_shaft_coupling": "cracked"}, signals={"generator_vibration_mm_s": 6.8}),
    "generator.slip_ring_carbon_dust": fault(
        "generator", "slip ring / brushes", "Worn slip-ring brushes: carbon dust below the housing and on the cover",
        "generator_inspect_slip_ring", ["rgb"], severity="medium",
        overlays=[link("generator_fault_carbon_dust", "generator", mesh=M + "fault_carbon_dust.obj")],
        signals={"generator_brush_wear_percent": 85}),
    "generator.terminal_box_overheating": fault(
        "generator", "terminal box", "Loose power connection overheating the terminal box, paint scorched",
        "generator_inspect_terminal_box", ["thermal", "rgb"], severity="high",
        overlays=[link("generator_fault_terminal_heat", "generator", mesh=M + "fault_terminal_heat.obj")],
        signals={"generator_terminal_temperature_c": 104}),
}

SIGNALS = {"brake_pad_wear_percent": 20, "brake_disc_temperature_c": 55, "generator_vibration_mm_s": 1.9,
           "generator_brush_wear_percent": 30, "generator_terminal_temperature_c": 46}
