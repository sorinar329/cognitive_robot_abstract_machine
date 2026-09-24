"""Render a URDF: overviews (whole turbine, tower base, rotor, opened nacelle) and one close-up per
inspection point, looking from the point's ``view_from`` direction.

blender -b --factory-startup -P scripts/render_preview.py -- [urdf] [out_dir] [--no-closeups]

Interactive (no -b): opens the Blender GUI with the assembled scene instead of rendering.
blender --factory-startup -P scripts/render_preview.py -- [urdf] --view [--no-cutaway]
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from turbine import parts  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
closeups = "--no-closeups" not in argv
view = "--view" in argv
cutaway = "--no-cutaway" not in argv
argv = [a for a in argv if not a.startswith("--")]
urdf_path = argv[0] if argv else os.path.join(ROOT, "urdf", "windturbine.urdf")
out_dir = argv[1] if len(argv) > 1 else os.path.join(ROOT, "preview")
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
urdf = ET.parse(urdf_path).getroot()


def origin(el):
    o = el.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return Matrix.Translation(Vector(xyz)) @ Euler(rpy, "XYZ").to_matrix().to_4x4()


# initial joint states from the scenario's ground truth (Blender has no PyYAML:
# read the flat "initial_joint_states:" block by hand)
joint_pos = {}
truth = urdf_path.replace(".urdf", "_ground_truth.yaml")
if os.path.exists(truth):
    block = False
    for line in open(truth):
        if line.startswith("initial_joint_states:"):
            block = "{}" not in line
        elif block and line.startswith("  "):
            k, v = line.strip().split(":")
            joint_pos[k.strip()] = float(v)
        else:
            block = False


def joint_tf(j):
    tf = origin(j)
    pos = joint_pos.get(j.get("name"), 0.0)
    if pos and j.find("axis") is not None:
        axis = Vector([float(v) for v in j.find("axis").get("xyz").split()])
        if j.get("type") == "prismatic":
            tf = tf @ Matrix.Translation(axis * pos)
        else:
            tf = tf @ Matrix.Rotation(pos, 4, axis)
    return tf


parent_of = {j.find("child").get("link"): (j.find("parent").get("link"), joint_tf(j)) for j in urdf.iter("joint")}


def world_tf(link):
    if link not in parent_of:
        return Matrix.Identity(4)
    parent, tf = parent_of[link]
    return world_tf(parent) @ tf


for link in urdf.iter("link"):
    for vis in link.findall("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        path = mesh.get("filename").replace("package://windturbine_model", ROOT)
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
        for obj in bpy.context.selected_objects:
            obj.matrix_world = world_tf(link.get("name")) @ origin(vis) @ obj.matrix_world
            obj.name = link.get("name") + "/" + obj.name

from turbine import dims as nd  # noqa: E402


def apply_cutaway():
    """Open the nacelle cover: drop its -Y wall and roof faces (in nacelle coordinates)."""
    to_nacelle = world_tf("nacelle").inverted()
    for obj in [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("nacelle/nacelle_cover")]:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        doomed = []
        for face in bm.faces:
            pts = [to_nacelle @ (obj.matrix_world @ v.co) for v in face.verts]
            if all(p.y < nd.INNER_Y[0] + 0.05 and p.z > 0.05 for p in pts) or all(p.z > nd.NACELLE_Z[1] - 0.15 for p in pts):
                doomed.append(face)
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.to_mesh(obj.data)
        bm.free()


scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
scene.display.shading.light = "STUDIO"
cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def shoot(name, location, target, lens, res=(1600, 900)):
    cam.location = Vector(location)
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = lens
    cam_data.clip_start = 0.01
    cam_data.clip_end = 2000.0
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.filepath = os.path.join(out_dir, name + ".png")
    bpy.ops.render.render(write_still=True)
    print("rendered", scene.render.filepath)


if view:
    # interactive: material colours in the viewport, look from the camera's overview spot
    cam.location = Vector((1.5, -9.5, 6.5))
    cam.rotation_euler = (Vector((-1.6, 0, 0.8)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.shading.type = "SOLID"
                space.shading.color_type = "MATERIAL"
                space.shading.show_cavity = True
                space.clip_start = 0.01
                space.region_3d.view_perspective = "CAMERA"
else:
    nacelle = world_tf("nacelle")
    tower_base = world_tf("tower_section_1").translation
    shoot("overview", (190, -265, 95), (0, 0, 88), 35)
    shoot("tower_base", tower_base + Vector((11, -17, 5)), tower_base + Vector((0, -2, 1.2)), 30)
    shoot("rotor", nacelle @ Vector((40, -38, 8)), nacelle @ Vector((4, 0, 2)), 28)
if closeups and not view:
    link_names = {l.get("name") for l in urdf.iter("link")}
    points = [p for part in parts.ALL for p in part.INSPECTION_POINTS if p["name"] in link_names]

    def closeup(p):
        target = world_tf(p["name"]).translation
        direction = world_tf(p["parent"]).to_3x3() @ Vector(p["view_from"]).normalized()   # view_from is in the parent frame
        shoot(p["name"], target + p["distance"] * direction, target, 30, res=(800, 600))

    for p in points:
        if p["outside"]:
            closeup(p)
if not view and cutaway:
    apply_cutaway()
    shoot("nacelle", world_tf("nacelle") @ Vector((2.0, -12.5, 8.5)), world_tf("nacelle") @ Vector((-2.5, 0, 0.8)), 24)
    if closeups:
        for p in points:
            if not p["outside"]:
                closeup(p)
