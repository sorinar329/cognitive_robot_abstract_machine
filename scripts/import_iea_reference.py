#!/usr/bin/env python3
"""Extract the blade and tower data we model from the IEA-3.4-130-RWT windIO file.

Writes references/iea34/geometry.json (plain JSON, so Blender's Python can read it).
Source: https://github.com/IEAWindSystems/IEA-3.4-130-RWT (Apache-2.0), NREL/TP-5000-73492.
"""
import json
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "references", "iea34", "IEA-3.4-130-RWT.yaml")

with open(SRC) as f:
    d = yaml.safe_load(f)
blade = d["components"]["blade"]
shape = blade["outer_shape"]
tower = d["components"]["tower"]
out = {
    "source": "IEA-3.4-130-RWT (windIO), NREL/TP-5000-73492",
    "blade": {
        "grid": shape["chord"]["grid"],
        "chord": shape["chord"]["values"],
        "twist_rad": shape["twist"]["values"] if max(abs(v) for v in shape["twist"]["values"]) < 1.0
        else [v * 3.141592653589793 / 180 for v in shape["twist"]["values"]],
        "pitch_axis_from_le": shape["section_offset_y"]["values"],
        "rthick": shape["rthick"]["values"],
        "ref_axis_grid": blade["reference_axis"]["z"]["grid"],
        "ref_axis_x": blade["reference_axis"]["x"]["values"],
        "ref_axis_z": blade["reference_axis"]["z"]["values"],
    },
    "airfoils": [
        {"name": a["name"], "rthick": a["rthick"], "x": a["coordinates"]["x"], "y": a["coordinates"]["y"]}
        for a in d["airfoils"]
    ],
    "tower": {
        "z": tower["reference_axis"]["z"]["values"],
        "outer_diameter": tower["outer_shape"]["outer_diameter"]["values"],
        "wall_thickness": tower["structure"]["layers"][0]["thickness"]["values"],
    },
    "drivetrain": d["components"]["drivetrain"]["outer_shape"],
    "hub_diameter": d["components"]["hub"]["diameter"],
    "cone_angle_deg": d["components"]["hub"]["cone_angle"],
    "hub_height": d["assembly"]["hub_height"],
}
with open(os.path.join(ROOT, "references", "iea34", "geometry.json"), "w") as f:
    json.dump(out, f, indent=1)
print("twist range", min(out["blade"]["twist_rad"]), max(out["blade"]["twist_rad"]))
print("airfoils", [(a["name"], a["rthick"], len(a["x"])) for a in out["airfoils"]])
