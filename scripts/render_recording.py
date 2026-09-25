"""Render a CRAMERA recording (a cramera-onboard bundle) to an MP4 video.

  blender -b --factory-startup -P scripts/render_recording.py -- BUNDLE OUT.mp4 [--step N] [--view NAME]

Replays the bundle's trajectory: every model's joints (keyed ``prefix/joint``),
the robot base pose and the tracked objects. The turbine is built from the
repository's OBJ meshes (the bundle's GLBs are Z-up and would be turned by
Blender's glTF importer), the robot from the bundle's STL meshes. The camera
comes from the named view in turbine/views.py (default: the
bundle's scene camera).
"""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_MESHES = "meshes/windturbine_model/models/"

argv = sys.argv[sys.argv.index("--") + 1:]
bundle, out = os.path.expanduser(argv[0]), argv[1]
step = int(argv[argv.index("--step") + 1]) if "--step" in argv else 2
view = argv[argv.index("--view") + 1] if "--view" in argv else None

with open(os.path.join(bundle, "scene.json")) as f:
    scene_spec = json.load(f)
with open(os.path.join(bundle, "trajectory.json")) as f:
    traj = json.load(f)
fps_in = traj.get("framesPerSecond") or scene_spec.get("framesPerSecond") or 25

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def origin_of(el):
    o = el.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return Matrix.Translation(Vector(xyz)) @ Euler(rpy, "XYZ").to_matrix().to_4x4()


def import_mesh(path):
    """Import one mesh file and return a single joined object (world transform identity)."""
    before = set(bpy.data.objects)
    low = path.lower()
    if low.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=path, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        return None
    if len(new) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in new:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        bpy.ops.object.join()
    obj = new[0]
    if not obj.data.materials:          # STL: plain light grey like the G1 in CRAMERA
        mat = bpy.data.materials.get("robot_grey") or bpy.data.materials.new("robot_grey")
        mat.diffuse_color = (0.75, 0.76, 0.78, 1.0)
        obj.data.materials.append(mat)
    return obj


class Model:
    """One URDF of the bundle: links with their visual objects, joints for FK."""

    def __init__(self, spec):
        self.prefix = spec.get("prefix") or ""
        self.robot = bool(spec.get("robot"))
        root = ET.parse(os.path.join(bundle, spec["urdf"])).getroot()
        self.joints = {}
        for j in root.iter("joint"):
            axis = j.find("axis")
            self.joints[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"), origin=origin_of(j),
                axis=Vector([float(v) for v in axis.get("xyz").split()]) if axis is not None else Vector((1, 0, 0)))
        self.visuals = []              # (link, visual origin, object)
        for link in root.iter("link"):
            for vis in link.findall("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is None:
                    continue
                rel = mesh.get("filename")
                if rel.startswith(REPO_MESHES):
                    path = os.path.join(ROOT, "models", os.path.splitext(rel[len(REPO_MESHES):])[0] + ".obj")
                else:
                    path = os.path.join(bundle, rel)
                if not os.path.exists(path):
                    continue
                obj = import_mesh(path)
                if obj is None:
                    continue
                scale = mesh.get("scale")
                s = Matrix.Diagonal([float(v) for v in scale.split()] + [1.0]) if scale else Matrix.Identity(4)
                self.visuals.append((link.get("name"), origin_of(vis) @ s, obj))
        linked = set(self.joints) | {j["parent"] for j in self.joints.values()}
        roots = [l for l in linked if l not in self.joints]
        self.root = next((l for l in roots if any(v[0] == l for v in self.visuals)), roots[0] if roots else None)
        base_body = scene_spec.get("robot", {}).get("baseBody")
        if self.robot and base_body in linked:        # the recorded base pose belongs to this body
            self.root = base_body

    def transforms(self, positions, base):
        cache = {}

        def tf(link):
            if link in cache:
                return cache[link]
            j = self.joints.get(link)
            if j is None or (self.robot and link == self.root):
                m = base if (self.robot and link == self.root) else Matrix.Identity(4)
            else:
                q = positions.get(f"{self.prefix}/{j['name']}", positions.get(j["name"], 0.0))
                motion = Matrix.Identity(4)
                if j["type"] in ("revolute", "continuous"):
                    motion = Matrix.Rotation(q, 4, j["axis"])
                elif j["type"] == "prismatic":
                    motion = Matrix.Translation(j["axis"] * q)
                m = tf(j["parent"]) @ j["origin"] @ motion
            cache[link] = m
            return m
        return {link: tf(link) for link, _, _ in self.visuals}


def pose_matrix(p):
    return Matrix.Translation(Vector(p[:3])) @ Quaternion((p[6], p[3], p[4], p[5])).to_matrix().to_4x4()


models = [Model(m) for m in scene_spec["models"]]
for m in models:
    print(f"\nmodel prefix={m.prefix!r} robot={m.robot} visuals={len(m.visuals)} root={m.root}", flush=True)
objects = {}
for spec in scene_spec.get("objects") or []:
    if "box" in spec:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        o = bpy.context.active_object
        o.data.transform(Matrix.Diagonal(list(spec["box"]) + [1.0]))
        mat = bpy.data.materials.new(spec["key"])
        hexcol = spec.get("color", "#cc2222").lstrip("#")
        mat.diffuse_color = tuple(int(hexcol[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)
        o.data.materials.append(mat)
        objects[spec["key"]] = o

# %% keyframes
frames = list(range(0, len(traj["frames"]), step))
for k, i in enumerate(frames):
    positions = traj["frames"][i]
    base = pose_matrix(traj["base"][i]) if traj.get("base") and traj["base"][i] else Matrix.Identity(4)
    for model in models:
        if k > 0 and not model.robot:
            continue                            # the environment only moves if its joints do; keep it static
        tfs = model.transforms(positions, base)
        for link, vis_origin, obj in model.visuals:
            obj.matrix_world = tfs[link] @ vis_origin
            if model.robot:
                obj.rotation_mode = "QUATERNION"
                obj.keyframe_insert("location", frame=k + 1)
                obj.keyframe_insert("rotation_quaternion", frame=k + 1)
    tracked = traj["objects"][i] if traj.get("objects") and i < len(traj["objects"]) else {}
    for key, obj in objects.items():
        if key in tracked:
            obj.matrix_world = pose_matrix(tracked[key])
            obj.rotation_mode = "QUATERNION"
            obj.keyframe_insert("location", frame=k + 1)
            obj.keyframe_insert("rotation_quaternion", frame=k + 1)

# %% camera, look, output
if view:
    sys.path.insert(0, ROOT)
    from turbine.views import VIEWS  # noqa: E402
    camera = VIEWS[view][0]
else:
    camera = scene_spec.get("camera") or {"position": [10, -10, 8], "target": [0, 0, 1]}
cam_data = bpy.data.cameras.new("cam")
cam_data.lens, cam_data.clip_start, cam_data.clip_end = 22, 0.05, 2000
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
cam.location = Vector(camera["position"])
cam.rotation_euler = (Vector(camera["target"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam

scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.light = "STUDIO"
scene.display.shading.show_cavity = True
scene.display.shading.show_shadows = False
scene.world = bpy.data.worlds.new("w")
scene.world.color = (0.055, 0.075, 0.095)
scene.render.resolution_x, scene.render.resolution_y = 1280, 720
scene.frame_start, scene.frame_end = 1, len(frames)
scene.render.fps = max(1, round(fps_in / step))
settings = scene.render.image_settings
if hasattr(settings, "media_type"):
    settings.media_type = "VIDEO"
settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.filepath = os.path.abspath(out)
print(f"rendering {len(frames)} frames at {scene.render.fps} fps to {out}", flush=True)
bpy.ops.render.render(animation=True)
print("wrote", out)
