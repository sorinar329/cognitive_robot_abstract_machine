"""Hub (fixed to the main shaft, which turns about the 5 deg tilted axis) and three pitchable blades lofted
from the IEA-3.4-130-RWT data, with blade faults a drone inspection finds."""
from turbine import blade_geometry as bg
from turbine import frames
from turbine.dims import rotor as r
from turbine.urdf import fault, inspection_point, link, span_box, x_cylinder

M = "rotor/"
BLADE_COLLISIONS = [span_box(*b) for b in bg.collision_boxes()]


def blade(i):
    n = i + 1
    return link(f"blade_{n}", "hub", mesh=M + "blade.obj", xyz=frames.blade_origin(i),
                rpy=frames.to_rpy(frames.blade_rotation(i)), joint="revolute", axis=(0, 0, 1),
                limits=r.PITCH_LIMITS, collisions=BLADE_COLLISIONS)


LINKS = [
    # bolted to the main shaft flange: the rotor turns with main_shaft_joint, and the rotor lock holds it
    link("hub", "main_shaft", mesh=M + "hub.obj", collisions=[x_cylinder(r.SPINNER_R, r.SPINNER_X)]),
] + [blade(i) for i in range(r.BLADES)]

INSPECTION_POINTS, FAULTS = [], {}
for i in range(r.BLADES):
    n = i + 1
    b = f"blade_{n}"
    INSPECTION_POINTS += [
        inspection_point(f"{b}_inspect_leading_edge", b, bg.surface_point(0.9, "suction", 0.0), (0.8, -1, 0.15),
                         f"blade {n} leading edge, outer third: erosion", distance=5.0, outside=True),
        inspection_point(f"{b}_inspect_tip_receptor", b, bg.surface_point(0.97, "suction", 0.45), (-1, 0.1, 0),
                         f"blade {n} tip and lightning receptor, suction side", distance=4.0, outside=True),
        inspection_point(f"{b}_inspect_trailing_edge", b, bg.surface_point(0.24, "suction", 1.0), (-0.7, 1, 0),
                         f"blade {n} trailing edge near the maximum chord: cracks, split bond line", distance=5.0, outside=True),
    ]
    FAULTS.update({
        f"{b}.leading_edge_erosion": fault(
            "rotor", f"blade {n} leading edge", "Leading-edge erosion on the outer blade: gelcoat worn through to the laminate",
            f"{b}_inspect_leading_edge", ["drone_rgb"], severity="medium",
            overlays=[link(f"{b}_fault_le_erosion", b, mesh=M + "fault_le_erosion.obj")],
            signals={"power_curve_deficit_percent": 2.5}),
        f"{b}.lightning_damage": fault(
            "rotor", f"blade {n} tip", "Lightning strike next to the tip receptor: burnt and delaminated shell",
            f"{b}_inspect_tip_receptor", ["drone_rgb"], severity="high",
            overlays=[link(f"{b}_fault_lightning", b, mesh=M + "fault_lightning.obj")],
            signals={f"{b}_lightning_counter": 1}),
        f"{b}.trailing_edge_crack": fault(
            "rotor", f"blade {n} trailing edge", "Trailing-edge bond line split open near the maximum chord",
            f"{b}_inspect_trailing_edge", ["drone_rgb"], severity="high",
            overlays=[link(f"{b}_fault_te_crack", b, mesh=M + "fault_te_crack.obj")]),
    })

SIGNALS = {"power_curve_deficit_percent": 0.0, **{f"blade_{i + 1}_lightning_counter": 0 for i in range(r.BLADES)}}
