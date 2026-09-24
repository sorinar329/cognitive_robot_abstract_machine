#!/usr/bin/env python3
"""Write CRAMERA scene bundles of the nacelle (healthy and fault scenarios).

A bundle has the layout of CRAMERA's ``precision_lab``: ``scene.json``,
``environment.urdf`` with GLB visuals under ``assets/``, ``trajectory.json``
holding the initial joint state, and ``semantics.json`` with the inspection
points, active faults and sensor signals.

  build_cramera_bundle.py                         healthy (+ windturbine_open) and every scenarios/*.yaml
  build_cramera_bundle.py --scenario scenarios/x.yaml
  build_cramera_bundle.py --output DIR            default: ~/.cramera/scenes

A scenario's ``view`` (turbine | tower_base | nacelle, default turbine) sets the
start camera; the ``nacelle`` view leaves the cover visual out to see inside.

Open in CRAMERA: http://localhost:8711/?scene=windturbine
"""
import argparse
import glob
import json
import os
import shutil
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from generate_urdf import build, collect  # noqa: E402
from turbine import urdf  # noqa: E402

PREFIX = "windturbine"
COVER = "nacelle/cover.obj"
from turbine.dims import site  # noqa: E402

_N = site.NACELLE_ORIGIN_Z
VIEWS = {
    # view: (camera, show the nacelle cover)
    "turbine": ({"position": [150.0, -210.0, 90.0], "target": [0.0, 0.0, 85.0]}, True),
    "tower_base": ({"position": [11.0, -17.0, 5.3], "target": [0.0, -2.0, 1.5]}, True),
    "nacelle": ({"position": [2.0, -12.5, _N + 8.5], "target": [-2.5, 0.0, _N + 0.8]}, False),
}
MAX_CAMERA_DISTANCE = 320.0   # read by CRAMERA (rendering.maxCameraDistance) so the whole turbine fits


def glb_asset(mesh):
    return "assets/" + os.path.splitext(mesh)[0] + ".glb"


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def bundle(name, description, fault_ids, out_root, view="turbine"):
    camera, with_cover = VIEWS[view]
    links, points, faults, signals = collect()
    all_links, states, truth = build(fault_ids, links, points, faults, signals)
    out = os.path.join(out_root, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    skip = () if with_cover else (COVER,)
    urdf.write(os.path.join(out, "environment.urdf"), PREFIX, all_links, states, mesh_uri=glb_asset, skip_meshes=skip)
    used = set()
    for spec in all_links:
        meshes = spec["mesh"] if isinstance(spec["mesh"], list) else [spec["mesh"]] if spec["mesh"] else []
        if spec["variants"]:
            meshes = [spec["variants"][states.get(spec["name"], next(iter(spec["variants"])))]]
        used.update(m for m in meshes if m not in skip)
    for mesh in sorted(used):
        target = os.path.join(out, glb_asset(mesh))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, "models", os.path.splitext(mesh)[0] + ".glb"), target)

    movable = [urdf.joint_name(l) for l in all_links if l["joint"] != "fixed"]
    initial = {j: float(truth["initial_joint_states"].get(j, 0.0)) for j in movable}
    title = "Wind turbine" + (f" · {description}" if description else "")
    write_json(os.path.join(out, "scene.json"), {
        "name": name,
        "rendering": {"ambientOcclusion": False, "exposure": 0.6, "maxCameraDistance": MAX_CAMERA_DISTANCE},
        "environmentName": title,
        "task": "Inspection" + (f" · {len(fault_ids)} faults" if fault_ids else " · healthy"),
        "framesPerSecond": 1,
        "trajectory": "trajectory.json",
        "camera": camera,
        "models": [{"name": "nacelle", "urdf": "environment.urdf", "prefix": "", "robot": False,
                    "links": len(all_links) + 1, "movableJoints": movable, "preserveMaterials": True}],
        "objects": [],
        "segments": [], "actions": [], "planTrees": [], "placeTarget": None,
        "validation": {"mode": "manual_kinematic", "robotExecutionVerified": False},
    })
    write_json(os.path.join(out, "trajectory.json"),
               {"framesPerSecond": 1, "frames": [initial], "objects": [{}]})
    write_json(os.path.join(out, "semantics.json"), {
        "units": "metres",
        "upAxis": "Z",
        "coverVisual": with_cover,
        "inspectionPoints": [dict(p, xyz=list(p["xyz"]), view_from=list(p["view_from"])) for p in points],
        "faults": truth["faults"],
        "initialJointStates": truth["initial_joint_states"],
        "signals": truth["signals"],
    })
    print(f"wrote {out} ({len(fault_ids)} faults, {len(used)} meshes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", help="scenario YAML (repeatable); default: all")
    ap.add_argument("--output", default=os.path.expanduser("~/.cramera/scenes"))
    args = ap.parse_args()

    scenarios = args.scenario or sorted(glob.glob(os.path.join(ROOT, "scenarios", "*.yaml")))
    if not args.scenario:
        bundle(PREFIX, "", [], args.output, "turbine")
        bundle(f"{PREFIX}_open", "nacelle opened", [], args.output, "nacelle")
    for path in scenarios:
        with open(path) as f:
            spec = yaml.safe_load(f)
        name = f"{PREFIX}_{os.path.splitext(os.path.basename(path))[0]}"
        bundle(name, spec.get("description", ""), spec.get("faults") or [], args.output, spec.get("view", "turbine"))


if __name__ == "__main__":
    main()
