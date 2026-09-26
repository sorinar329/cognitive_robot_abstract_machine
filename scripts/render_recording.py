"""Render a CRAMERA recording (a cramera-onboard bundle) to an MP4 video.

  blender -b --factory-startup -P scripts/render_recording.py -- BUNDLE OUT.mp4 [--step N] [--view NAME] [--speed S] [--look flat|eevee] [--stills F,F,..]

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
speed = float(argv[argv.index("--speed") + 1]) if "--speed" in argv else 1.0
look = argv[argv.index("--look") + 1] if "--look" in argv else "flat"
"""--look eevee: sky, sun, PBR materials, G1 colours (scripts/blender_look.py); flat: fast Workbench."""
"""--speed 2: play back twice as fast as recorded."""
stills = [float(v) for v in argv[argv.index("--stills") + 1].split(",")] if "--stills" in argv else None
"""--stills 0.1,0.5,0.9: instead of the video, PNGs at these fractions of the run (OUT is a directory)."""

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

    def transforms(self, positions, base, links=None):
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
        return {link: tf(link) for link in (links or [v[0] for v in self.visuals])}

    def moving_links(self, frames):
        """Links below a joint whose position changes during the recording."""
        def value(frame, j):
            return frame.get(f"{self.prefix}/{j['name']}", frame.get(j["name"], 0.0))
        moving = {child for child, j in self.joints.items() if j["type"] != "fixed"
                  and any(abs(value(f, j) - value(frames[0], j)) > 1e-4 for f in frames)}

        def below(link):
            while link is not None:
                if link in moving:
                    return True
                link = self.joints[link]["parent"] if link in self.joints else None
            return False
        return {link for link, _, _ in self.visuals if below(link)}


def pose_matrix(p):
    return Matrix.Translation(Vector(p[:3])) @ Quaternion((p[6], p[3], p[4], p[5])).to_matrix().to_4x4()


models = [Model(m) for m in scene_spec["models"]]
for m in models:
    print(f"\nmodel prefix={m.prefix!r} robot={m.robot} visuals={len(m.visuals)} root={m.root}", flush=True)
objects = {}
for spec in scene_spec.get("objects") or []:
    parts = [(p["centre"], p["size"]) for p in spec["parts"]] if spec.get("parts") else \
        [((0, 0, 0), spec["box"])] if "box" in spec else []
    if parts:
        pieces = []
        for centre, size in parts:
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            piece = bpy.context.active_object
            piece.data.transform(Matrix.Translation(Vector(centre)) @ Matrix.Diagonal(list(size) + [1.0]))
            pieces.append(piece)
        if len(pieces) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for piece in pieces:
                piece.select_set(True)
            bpy.context.view_layer.objects.active = pieces[0]
            bpy.ops.object.join()
        o = pieces[0]
        mat = bpy.data.materials.new(spec["key"])
        hexcol = spec.get("color", "#cc2222").lstrip("#")
        srgb = [int(hexcol[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        mat.diffuse_color = tuple(srgb) + (1.0,)
        mat.use_nodes = True                     # for EEVEE: a painted, slightly glossy case
        shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        shader.inputs["Base Color"].default_value = tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in srgb) + (1.0,)
        shader.inputs["Roughness"].default_value = 0.4
        o.data.materials.append(mat)
        objects[spec["key"]] = o

# %% view: camera, optional robot-following and tower cutaway
if view:
    sys.path.insert(0, ROOT)
    from turbine.views import VIEWS  # noqa: E402
    camera = VIEWS[view][0]
else:
    camera = scene_spec.get("camera") or {"position": [10, -10, 8], "target": [0, 0, 1]}
if camera.get("cut_tower"):
    import bmesh
    side = Vector(camera["cut_tower"] + [0.0]).normalized()
    for model in models:
        for link, vis_origin, obj in model.visuals:
            if not link.startswith(("tower_section", "flange_")):
                continue
            m = model.transforms(traj["frames"][0], Matrix.Identity(4), [link])[link] @ vis_origin
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            doomed = [f for f in bm.faces if (m @ f.calc_center_median()).xy.dot(side.xy) > 0.2]
            bmesh.ops.delete(bm, geom=doomed, context="FACES")
            bm.to_mesh(obj.data)
            bm.free()

# hoist rope: drawn between the crane trolley and the hook (the URDF cannot stretch a mesh)
rope = None
env = next((m for m in models if not m.robot and "crane_hook" in m.joints), None)
if env:
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.012, depth=1.0)
    rope = bpy.context.active_object
    rope.data.transform(Matrix.Translation((0, 0, 0.5)))
    rope_mat = bpy.data.materials.new("rope")
    rope_mat.diffuse_color = (0.08, 0.08, 0.08, 1.0)
    rope.data.materials.append(rope_mat)

# %% keyframes
frames = list(range(0, len(traj["frames"]), step))
moving = {id(m): (m.moving_links([traj["frames"][i] for i in frames]) if not m.robot else None) for m in models}
robot_model = next((m for m in models if m.robot), None)
cam_data = bpy.data.cameras.new("cam")
cam_data.lens, cam_data.clip_start, cam_data.clip_end = 22, 0.05, 2000
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
for k, i in enumerate(frames):
    positions = traj["frames"][i]
    base = pose_matrix(traj["base"][i]) if traj.get("base") and traj["base"][i] else Matrix.Identity(4)
    for model in models:
        animated = moving[id(model)]
        if k > 0 and not model.robot and not animated:
            continue                            # the environment only moves if its joints do
        tfs = model.transforms(positions, base)
        for link, vis_origin, obj in model.visuals:
            if k > 0 and not model.robot and link not in animated:
                continue
            obj.matrix_world = tfs[link] @ vis_origin
            if model.robot or link in animated:
                obj.rotation_mode = "QUATERNION"
                obj.keyframe_insert("location", frame=k + 1)
                obj.keyframe_insert("rotation_quaternion", frame=k + 1)
    if rope:
        t = env.transforms(positions, base, ["crane_trolley", "crane_hook"])
        top, bottom = t["crane_trolley"] @ Vector((0, 0, -0.33)), t["crane_hook"] @ Vector((0, 0, 0.6))
        length = max((top - bottom).length, 1e-3)
        rope.matrix_world = Matrix.Translation(bottom) @ Matrix.Diagonal((1, 1, length, 1))
        rope.keyframe_insert("location", frame=k + 1)
        rope.keyframe_insert("scale", frame=k + 1)
    if "follow" in camera and robot_model:
        at = base.translation
        cam.location = at + Vector(camera["follow"])
        cam.rotation_euler = (at + Vector(camera["look"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=k + 1)
        cam.keyframe_insert("rotation_euler", frame=k + 1)
    tracked = traj["objects"][i] if traj.get("objects") and i < len(traj["objects"]) else {}
    for key, obj in objects.items():
        if key in tracked:
            obj.matrix_world = pose_matrix(tracked[key])
            obj.rotation_mode = "QUATERNION"
            obj.keyframe_insert("location", frame=k + 1)
            obj.keyframe_insert("rotation_quaternion", frame=k + 1)

# %% camera, look, output
if "follow" not in camera:
    cam.location = Vector(camera["position"])
    cam.rotation_euler = (Vector(camera["target"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam

if look == "eevee":
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import blender_look  # noqa: E402
    blender_look.setup(scene)
    for model in models:
        if model.robot:
            blender_look.color_robot([(link, obj) for link, _, obj in model.visuals])
else:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.show_cavity = True
    scene.display.shading.show_shadows = False
    scene.world = bpy.data.worlds.new("w")
    scene.world.color = (0.055, 0.075, 0.095)
scene.render.resolution_x, scene.render.resolution_y = 1280, 720
scene.frame_start, scene.frame_end = 1, len(frames)
scene.render.fps = max(1, round(fps_in / step * speed))
settings = scene.render.image_settings
if hasattr(settings, "media_type"):
    settings.media_type = "VIDEO"
settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.filepath = os.path.abspath(out)
if stills:
    if hasattr(settings, "media_type"):
        settings.media_type = "IMAGE"
    settings.file_format = "PNG"
    for fraction in stills:
        scene.frame_set(1 + round(fraction * (len(frames) - 1)))
        scene.render.filepath = os.path.join(os.path.abspath(out), f"still_{fraction:.3f}.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", scene.render.filepath)
else:
    print(f"rendering {len(frames)} frames at {scene.render.fps} fps to {out}", flush=True)
    bpy.ops.render.render(animation=True)
    print("wrote", out)
