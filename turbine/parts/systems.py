"""Part: nacelle systems: transformer, converter and controller cabinets (doors),
hydraulic unit, cooling unit, service crane (trolley, hook), safety equipment.

Faults: converter error light, controller door left open, hydraulic hose leak,
low coolant, missing fire extinguisher.
"""
from turbine import dims as d
from turbine.urdf import fault, inspection_point, link, span_box

M = "systems/"
CX, CY, CZ = d.CABINET_X, d.CABINET_Y, d.CABINET_Z
HINGE_Y = CY[1] - 0.01


def cabinet(name, side):
    ys = tuple(sorted((side * CY[0], side * CY[1])))
    return [
        link(f"cabinet_{name}", "nacelle", mesh=M + f"cabinet_{name}.obj", collisions=[span_box(CX, ys, CZ)]),
        link(f"cabinet_{name}_door", f"cabinet_{name}", mesh=M + f"cabinet_{name}_door.obj",
             xyz=(CX[1], side * HINGE_Y, 0.0), joint="revolute", axis=(0, 0, side), limits=(0.0, 1.9),
             collisions=[span_box((0.0, 0.03), tuple(sorted((0.0, -side * (CY[1] - CY[0] - 0.02)))), (CZ[0] + 0.12, CZ[1] - 0.02))]),
    ]


LINKS = [
    link("transformer", "nacelle", mesh=M + "transformer.obj", collisions=[span_box(d.TRAFO_X, d.TRAFO_Y, d.TRAFO_Z)]),
    *cabinet("converter", 1),
    link("converter_status_light", "cabinet_converter_door",
         variants={"ok": M + "status_light_ok_p.obj", "error": M + "status_light_error_p.obj"}),
    *cabinet("controller", -1),
    link("hydraulic_unit", "nacelle", mesh=M + "hydraulic_unit.obj",
         collisions=[span_box(d.HYDRAULIC_X, d.HYDRAULIC_Y, (0.0, d.HYDRAULIC_Z[1] + 0.2))]),
    link("cooling_unit", "nacelle", mesh=M + "cooling_unit.obj", collisions=[span_box(d.COOLING_X, d.COOLING_Y, (0.0, d.COOLING_Z[1]))]),
    link("cooling_level", "cooling_unit", variants={"ok": M + "coolant_level_ok.obj", "low": M + "coolant_level_low.obj"}),
    link("crane_rail", "nacelle", mesh=M + "crane_rail.obj",
         collisions=[span_box(d.CRANE_X, (-0.09, 0.09), (d.CRANE_Z - 0.015, d.CRANE_Z + 0.265))]),
    link("crane_trolley", "crane_rail", mesh=M + "crane_trolley.obj", xyz=(d.CRANE_TRAVEL[0], 0.0, d.CRANE_Z),
         joint="prismatic", axis=(1, 0, 0), limits=(0.0, d.CRANE_TRAVEL[1] - d.CRANE_TRAVEL[0]),
         collisions=[span_box((-0.25, 0.25), (-0.14, 0.14), (-0.33, -0.03))]),
    link("crane_hook", "crane_trolley", mesh=M + "crane_hook.obj", xyz=(0.0, 0.0, -0.95),
         joint="prismatic", axis=(0, 0, -1), limits=(0.0, 2.8),
         collisions=[span_box((-0.07, 0.07), (-0.04, 0.04), (-0.1, 0.14))]),
    link("fire_extinguisher", "nacelle",
         variants={"present": M + "extinguisher_present.obj", "missing": M + "extinguisher_missing.obj"}),
    link("safety_equipment", "nacelle", mesh=M + "safety.obj"),
]

WALL = d.INNER_Y[0]
INSPECTION_POINTS = [
    inspection_point("converter_inspect_status", "cabinet_converter", (CX[1] + 0.035, HINGE_Y - 0.25, 1.85), (1, 0.0, -0.3),
                     "converter cabinet status light (green = ok)", distance=1.1),
    inspection_point("controller_inspect_door", "cabinet_controller", (CX[1] + 0.03, -(CY[0] + CY[1]) / 2, 1.2), (1, 0.0, 0.2),
                     "controller cabinet: door closed, emergency stop not pressed", distance=1.3),
    inspection_point("hydraulic_inspect", "hydraulic_unit", (d.HYDRAULIC_X[1] - 0.2, d.HYDRAULIC_Y[1], d.HYDRAULIC_Z[0] + 0.55), (0.8, 0.9, 1.0),
                     "hydraulic unit: hose fitting and tank, oil on the walkway", distance=0.78),
    inspection_point("cooling_inspect_level", "cooling_unit", (sum(d.COOLING_X) / 2, d.COOLING_Y[0], d.COOLING_Z[1] - 0.25),
                     (0.6, -1, 0.7), "coolant expansion tank level tube (above the red mark)", distance=0.57),
    inspection_point("safety_inspect_extinguisher", "nacelle", (d.EXTINGUISHER_XY[0], WALL + 0.12, 0.65), (1, 0.35, 0.35),
                     "fire extinguisher in its bracket by the tower hatch", distance=1.2),
]

FAULTS = {
    "converter.fault_light": fault(
        "systems", "converter cabinet", "Converter reports a fault: red status light on the cabinet door",
        "converter_inspect_status", ["rgb"], severity="high",
        set_variants={"converter_status_light": "error"}, signals={"converter_fault_code": 3107}),
    "controller.door_left_open": fault(
        "systems", "controller cabinet", "Controller cabinet door left open after maintenance",
        "controller_inspect_door", ["rgb"], severity="low",
        joint_states={"cabinet_controller_door_joint": 1.3}),
    "hydraulic.hose_leak": fault(
        "systems", "hydraulic unit", "Brake hose fitting leaking: oil down the tank and onto the walkway",
        "hydraulic_inspect", ["rgb"], severity="medium",
        overlays=[link("hydraulic_fault_leak", "hydraulic_unit", mesh=M + "fault_hydraulic_leak.obj")],
        signals={"hydraulic_pressure_bar": 118}),
    "cooling.coolant_low": fault(
        "systems", "cooling unit", "Coolant below the minimum mark in the expansion tank",
        "cooling_inspect_level", ["rgb"], severity="medium",
        set_variants={"cooling_level": "low"}, signals={"generator_winding_temperature_c": 128}),
    "safety.extinguisher_missing": fault(
        "systems", "fire extinguisher", "Fire extinguisher missing from its bracket",
        "safety_inspect_extinguisher", ["rgb"], severity="high",
        set_variants={"fire_extinguisher": "missing"}),
}

SIGNALS = {"converter_fault_code": 0, "hydraulic_pressure_bar": 160, "generator_winding_temperature_c": 95}
