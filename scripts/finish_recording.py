#!/usr/bin/env python3
"""Tidy a CRAMERA recording of a G1 inspection round (written by cramera-onboard).

  finish_recording.py ~/.cramera/scenes/windturbine_g1_t1 --view tower_base

- the G1 model gets its own prefix and the robot entry's name, so the recorded
  ``offis_unitree_g1/...`` joint positions and base poses drive it (the onboarder
  labels it with the environment's prefix and the URDF's name when the robot comes in
  through a RobotSpecification);
- the extra ``environment`` model the onboarder rebuilds from the world duplicates
  the G1 and is dropped;
- the turbine's OBJ meshes are swapped for the GLB meshes with their materials;
- the start camera and the camera distance match the static turbine scenes.
"""
import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from turbine.views import MAX_CAMERA_DISTANCE, VIEWS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_gait import add_gait  # noqa: E402

ROBOT_PREFIX = "offis_unitree_g1"
MESH_PREFIX = "meshes/windturbine_model/models/"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("--view", choices=sorted(VIEWS), default="tower_base")
    ap.add_argument("--no-gait", action="store_true", help="keep the recorded straight legs")
    args = ap.parse_args()
    scene_path = os.path.join(args.bundle, "scene.json")
    with open(scene_path) as f:
        scene = json.load(f)

    models = []
    for model in scene["models"]:
        if model["name"] == "environment":
            continue                                  # duplicate of the G1
        if model.get("robot"):
            model["prefix"] = ROBOT_PREFIX
            model["name"] = scene["robot"]["name"]      # the viewer pairs robot entry and model by name
        else:
            urdf_path = os.path.join(args.bundle, model["urdf"])
            with open(urdf_path) as f:
                text = f.read()
            for obj in sorted(set(re.findall(re.escape(MESH_PREFIX) + r"([^\"]+)\.obj", text))):
                src = os.path.join(ROOT, "models", obj + ".glb")
                dst = os.path.join(args.bundle, MESH_PREFIX + obj + ".glb")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
                text = text.replace(MESH_PREFIX + obj + ".obj", MESH_PREFIX + obj + ".glb")
            if not VIEWS[args.view][1]:      # views into the nacelle leave the cover visual out
                text = re.sub(r"<visual>(?:(?!</visual>).)*nacelle/cover\.glb(?:(?!</visual>).)*</visual>", "", text, flags=re.S)
            with open(urdf_path, "w") as f:
                f.write(text)
            model["preserveMaterials"] = True
        models.append(model)
    scene["models"] = models

    from turbine import dims
    for obj in scene.get("objects") or []:          # compound objects: their parts, for physics and videos
        if obj["key"] in dims.OBJECT_PARTS:
            parts, mass = dims.OBJECT_PARTS[obj["key"]]
            obj["parts"] = [dict(centre=list(c), size=list(s)) for c, s in parts]
            obj["mass"] = mass

    camera, _ = VIEWS[args.view]
    scene["camera"] = camera
    rendering = scene.setdefault("rendering", {})
    rendering.update({"ambientOcclusion": False, "exposure": 0.6, "maxCameraDistance": MAX_CAMERA_DISTANCE})
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=1)
    print(f"finished {args.bundle}: models {[m['name'] for m in models]}, view {args.view}")
    if not args.no_gait:
        add_gait(args.bundle)          # legs walk instead of sliding (visual only)


if __name__ == "__main__":
    main()
