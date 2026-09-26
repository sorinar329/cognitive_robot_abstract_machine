"""Shared helpers for the per-part Blender scripts.

Every part script builds its geometry in the part's *link frame* (Z up, X
pointing upwind towards the hub, metres) and calls ``export_part``, which
writes ``models/<part>/<part>.obj`` (+ .mtl) and a ``.blend`` for manual edits.
"""
import math
import os
import random
import sys

import bpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from turbine import dims  # noqa: E402,F401  (re-exported for the part scripts)
from turbine.dims import rotor as rotor_dims, site as site_dims  # noqa: E402,F401

# Colours are written as display (sRGB) values; ``material`` converts them to
# the linear values Blender and glTF store.
COLORS = {
    "grp_white": (0.85, 0.86, 0.84, 1.0),     # glass fibre nacelle cover
    "steel_grey": (0.35, 0.36, 0.38, 1.0),    # floor grating / bedplate
    "hatch_yellow": (0.9, 0.7, 0.05, 1.0),    # safety marked hatches
    "gear_paint": (0.42, 0.5, 0.52, 1.0),     # painted cast housings
    "rubber": (0.07, 0.07, 0.07, 1.0),
    "bolt_black": (0.08, 0.08, 0.09, 1.0),
    "pipe_steel": (0.6, 0.6, 0.62, 1.0),
    "glass": (0.75, 0.85, 0.9, 1.0),
    "oil": (0.16, 0.09, 0.02, 1.0),           # gear oil, fresh and in leaks
    "ok_green": (0.1, 0.7, 0.15, 1.0),
    "alarm_red": (0.85, 0.05, 0.03, 1.0),
    "crack": (0.01, 0.01, 0.01, 1.0),
    "grease": (0.62, 0.45, 0.12, 1.0),        # yellow-brown lithium grease
    "rust": (0.48, 0.18, 0.05, 1.0),          # fretting corrosion powder
    "bottle_white": (0.88, 0.88, 0.84, 1.0),
    "grass": (0.36, 0.47, 0.24, 1.0),
    "gravel": (0.62, 0.6, 0.56, 1.0),
    "concrete": (0.66, 0.66, 0.63, 1.0),
    "grout": (0.55, 0.55, 0.53, 1.0),
    "tower_paint": (0.8, 0.81, 0.79, 1.0),     # RAL 7035 light grey
    "blade_white": (0.93, 0.93, 0.91, 1.0),
    "erosion": (0.58, 0.54, 0.48, 1.0),        # worn-through gelcoat, exposed laminate
    "burn": (0.06, 0.05, 0.05, 1.0),           # lightning burn / soot
    "copper": (0.72, 0.45, 0.2, 1.0),
    "light_red": (0.95, 0.08, 0.05, 1.0),      # aviation obstruction light, glowing
    "dead_glass": (0.35, 0.3, 0.3, 1.0),       # broken, unlit light dome
    "heat_blue": (0.5, 0.42, 0.95, 1.0),       # temper colours of an overheated brake disc
    "carbon": (0.14, 0.14, 0.15, 1.0),         # brush dust
    "heat_brown": (0.5, 0.33, 0.18, 1.0),      # scorched paint
    "generator_paint": (0.3, 0.42, 0.55, 1.0),
    "cabinet_grey": (0.78, 0.78, 0.76, 1.0),   # RAL 7035 switch cabinets
    "resin": (0.2, 0.22, 0.2, 1.0),            # cast resin transformer coils
    "coolant": (0.2, 0.75, 0.3, 1.0),          # green glycol
    "led_green": (0.1, 0.95, 0.2, 1.0),
    "led_red": (1.0, 0.1, 0.05, 1.0),
    "hydraulic_oil": (0.55, 0.4, 0.1, 1.0),
    "lamp": (1.0, 0.97, 0.88, 1.0),
    "grating": (0.52, 0.54, 0.55, 1.0),
    "cable_black": (0.1, 0.1, 0.11, 1.0),
}

# PBR finish per material: (roughness, metallic, transmission); default (0.5, 0, 0)
FINISHES = {
    "grp_white": (0.45, 0.0, 0.0),
    "steel_grey": (0.55, 0.6, 0.0),
    "hatch_yellow": (0.4, 0.0, 0.0),
    "gear_paint": (0.35, 0.1, 0.0),
    "rubber": (0.8, 0.0, 0.0),
    "bolt_black": (0.4, 0.7, 0.0),
    "pipe_steel": (0.25, 0.9, 0.0),
    "glass": (0.05, 0.0, 0.8),
    "oil": (0.08, 0.0, 0.0),       # wet and glossy
    "grease": (0.3, 0.0, 0.0),
    "rust": (0.9, 0.0, 0.0),
    "bottle_white": (0.3, 0.0, 0.3),
    "grass": (0.95, 0.0, 0.0),
    "gravel": (0.95, 0.0, 0.0),
    "concrete": (0.85, 0.0, 0.0),
    "grout": (0.9, 0.0, 0.0),
    "tower_paint": (0.4, 0.0, 0.0),
    "blade_white": (0.3, 0.0, 0.0),
    "erosion": (0.9, 0.0, 0.0),
    "burn": (0.95, 0.0, 0.0),
    "copper": (0.3, 1.0, 0.0),
    "light_red": (0.2, 0.0, 0.0),
    "dead_glass": (0.3, 0.0, 0.0),
    "heat_blue": (0.35, 0.25, 0.0),
    "carbon": (0.95, 0.0, 0.0),
    "heat_brown": (0.8, 0.0, 0.0),
    "generator_paint": (0.35, 0.1, 0.0),
    "cabinet_grey": (0.45, 0.0, 0.0),
    "resin": (0.35, 0.0, 0.0),
    "coolant": (0.1, 0.0, 0.3),
    "led_green": (0.2, 0.0, 0.0),
    "led_red": (0.2, 0.0, 0.0),
    "hydraulic_oil": (0.08, 0.0, 0.0),
}

# emissive materials: (strength); colour = base colour
EMISSION = {"light_red": 6.0, "led_green": 4.0, "led_red": 4.0, "lamp": 3.0}


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def material(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        colour = tuple(srgb_to_linear(c) for c in COLORS[name][:3]) + (1.0,)
        mat.diffuse_color = colour
        mat.use_nodes = True
        shader = mat.node_tree.nodes["Principled BSDF"]
        shader.inputs["Base Color"].default_value = colour
        roughness, metallic, transmission = FINISHES.get(name, (0.5, 0.0, 0.0))
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
        shader.inputs["Transmission Weight"].default_value = transmission
        if name in EMISSION:
            shader.inputs["Emission Color"].default_value = colour
            shader.inputs["Emission Strength"].default_value = EMISSION[name]
        add_texture(mat, shader, name)
    return mat


TEXTURE_DIR = os.path.join(ROOT, "textures")
TEXTURE_TILE = {   # metres per texture repeat (scripts/prepare_textures.py makes the images)
    "grass": 3.0, "gravel": 1.5, "concrete": 2.0, "grout": 1.0, "grating": 0.6, "steel_grey": 1.0,
    "pipe_steel": 0.5, "gear_paint": 1.5, "generator_paint": 1.5, "tower_paint": 2.5, "cabinet_grey": 1.5,
    "hatch_yellow": 1.0, "grp_white": 2.0, "blade_white": 3.0,
}
NORMAL_STRENGTH = 0.6


def add_texture(mat, shader, name):
    """Colour and normal image for textured materials, repeated every TEXTURE_TILE metres of the
    world-scale UVs that ``export_part`` projects (the GLB keeps the repeat as a texture transform)."""
    colour_path = os.path.join(TEXTURE_DIR, f"{name}_color.jpg")
    if name not in TEXTURE_TILE or not os.path.exists(colour_path):
        return
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    uv = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0 / TEXTURE_TILE[name],) * 3
    links.new(uv.outputs["UV"], mapping.inputs["Vector"])
    colour = nodes.new("ShaderNodeTexImage")
    colour.image = bpy.data.images.load(colour_path, check_existing=True)
    links.new(mapping.outputs["Vector"], colour.inputs["Vector"])
    links.new(colour.outputs["Color"], shader.inputs["Base Color"])
    normal_path = os.path.join(TEXTURE_DIR, f"{name}_normal.jpg")
    if os.path.exists(normal_path):
        normal = nodes.new("ShaderNodeTexImage")
        normal.image = bpy.data.images.load(normal_path, check_existing=True)
        normal.image.colorspace_settings.name = "Non-Color"
        links.new(mapping.outputs["Vector"], normal.inputs["Vector"])
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = NORMAL_STRENGTH
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])


def textured(obj):
    return obj.type == "MESH" and any(s.material and s.material.name.split(".")[0] in TEXTURE_TILE
                                      for s in obj.material_slots)


def box_uv(obj):
    """World-scale box projection (1 UV unit = 1 m) on each face's dominant axis, so a
    texture has the same size on every part."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    layer = bm.loops.layers.uv.verify()
    m = obj.matrix_world
    rot = m.to_3x3()
    for face in bm.faces:
        n = rot @ face.normal
        axis = max(range(3), key=lambda i: abs(n[i]))
        u_axis, v_axis = [(1, 2), (0, 2), (0, 1)][axis]
        for loop in face.loops:
            p = m @ loop.vert.co
            loop[layer].uv = (p[u_axis], p[v_axis])
    bm.to_mesh(obj.data)
    bm.free()


def box(name, x, y, z, mat=None):
    """Axis aligned box spanning the given (min, max) intervals."""
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    obj.location = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(material(mat))
    return obj


def cylinder(name, radius, depth, location, rotation=(0, 0, 0), vertices=48, mat=None):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, location=location, rotation=rotation, vertices=vertices
    )
    obj = bpy.context.active_object
    obj.name = name
    if mat:
        obj.data.materials.append(material(mat))
    return obj


def cut(target, *cutters):
    """Boolean-subtract the cutters from target and delete the cutters."""
    for c in cutters:
        mod = target.modifiers.new("cut_" + c.name, "BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.solver = "EXACT"
        mod.object = c
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(c, do_unlink=True)
    return target


def hex_bolt(name, location, axis="Z", mat="bolt_black"):
    """Hex bolt head (M20-ish), axis along the given local axis."""
    rot = {"X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0), "Z": (0, 0, 0)}[axis]
    return cylinder(name, 0.017, 0.014, location, rotation=rot, vertices=6, mat=mat)


def puddle(name, radius, location, seed=0, mat="oil"):
    """Flat irregular blob, e.g. an oil puddle lying on the floor."""
    rng = random.Random(seed)
    waves = [(n, rng.uniform(0.04, 0.14), rng.uniform(0, 2 * math.pi)) for n in (2, 3, 5)]
    obj = cylinder(name, radius, 0.003, location, vertices=64, mat=mat)
    for v in obj.data.vertices:
        if math.hypot(v.co.x, v.co.y) > 1e-6:
            a = math.atan2(v.co.y, v.co.x)
            k = 1.0 + sum(amp * math.sin(n * a + ph) for n, amp, ph in waves)
            v.co.x *= k * 1.3   # a bit elongated
            v.co.y *= k
    return obj


def radial_box(name, x, r, width, angle, center=(0.0, 0.0), mat=None):
    """Box spanning x, thin in the tangential direction, from radius r[0] to r[1]
    along the direction (-sin(angle), cos(angle)) in the YZ plane around center."""
    obj = box(name, x, (-width / 2, width / 2), r, mat=mat)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    obj.rotation_euler = (angle, 0, 0)
    obj.location = (0, center[0], center[1])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    return obj


def polar(r, angle, center=(0.0, 0.0)):
    """(y, z) of the point at radius r in direction angle (see radial_box)."""
    return center[0] - r * math.sin(angle), center[1] + r * math.cos(angle)


def curve_pipe(name, p0, h0, p1, h1, radius, mat="pipe_steel"):
    """Bezier tube from p0 (leaving along h0) to p1 (arriving along h1)."""
    bpy.ops.curve.primitive_bezier_curve_add()
    obj = bpy.context.active_object
    obj.name = name
    a, b = obj.data.splines[0].bezier_points
    for pt, co, h in ((a, p0, h0), (b, p1, h1)):
        pt.handle_left_type = pt.handle_right_type = "FREE"
        pt.co = co
        pt.handle_left = tuple(c - v for c, v in zip(co, h))
        pt.handle_right = tuple(c + v for c, v in zip(co, h))
    obj.data.bevel_depth = radius
    obj.data.dimensions = "3D"
    bpy.ops.object.convert(target="MESH")
    obj.data.materials.append(material(mat))
    return obj


def mesh_object(name, verts, faces, mat=None, smooth=False):
    """Object from raw vertex / face lists."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if mat:
        obj.data.materials.append(material(mat))
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    bpy.context.view_layer.objects.active = obj
    return obj


def revolve(name, profile, segments=64, axis="Z", mat=None, close_ends=(True, True), smooth=True):
    """Surface of revolution. ``profile`` is [(axial, radius), ...] along the axis;
    a radius of 0 closes the surface to a point."""
    verts, faces = [], []
    for a, r in profile:
        for k in range(segments):
            phi = 2 * math.pi * k / segments
            u, v = r * math.cos(phi), r * math.sin(phi)
            verts.append({"Z": (u, v, a), "X": (a, u, v), "Y": (v, a, u)}[axis])
    n = len(profile)
    for i in range(n - 1):
        for k in range(segments):
            k2 = (k + 1) % segments
            faces.append((i * segments + k, i * segments + k2, (i + 1) * segments + k2, (i + 1) * segments + k))
    for end, idx in ((0, 0), (1, n - 1)):
        if close_ends[end] and profile[idx][1] > 0:
            ring = [idx * segments + k for k in range(segments)]
            faces.append(tuple(ring[::-1] if end == 0 else ring))
    return mesh_object(name, verts, faces, mat, smooth)


def intersect(target, other):
    """Keep only the part of target inside other; deletes other."""
    mod = target.modifiers.new("keep_" + other.name, "BOOLEAN")
    mod.operation, mod.solver, mod.object = "INTERSECT", "EXACT", other
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(other, do_unlink=True)
    return target


def union(target, *others):
    for other in others:
        mod = target.modifiers.new("add_" + other.name, "BOOLEAN")
        mod.operation, mod.solver, mod.object = "UNION", "EXACT", other
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(other, do_unlink=True)
    return target


def translation(t):
    return [[1, 0, 0, t[0]], [0, 1, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]]


def homogeneous_z(angle, t):
    """Rotation about Z by angle, then translation t."""
    ca, sa = math.cos(angle), math.sin(angle)
    return [[ca, -sa, 0, t[0]], [sa, ca, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]]


def homogeneous_x_to(angle, t):
    """Map local Z onto the horizontal radial direction at azimuth ``angle`` (for
    bolts sticking out of a cylinder wall), then translate to t."""
    ca, sa = math.cos(angle), math.sin(angle)
    # columns: local x -> tangential, local y -> world z, local z -> radial
    return [[-sa, 0, ca, t[0]], [ca, 0, sa, t[1]], [0, 1, 0, t[2]], [0, 0, 0, 1]]


def placed(obj, matrix):
    """Bake a 4x4 transform (nested lists or mathutils Matrix) into obj's mesh."""
    from mathutils import Matrix
    obj.data.transform(Matrix(matrix))
    obj.data.update()
    return obj


def rod(name, p0, p1, radius, mat=None, vertices=12):
    """Cylinder from point p0 to point p1."""
    from mathutils import Vector
    a, b = Vector(p0), Vector(p1)
    d = b - a
    obj = cylinder(name, radius, d.length, (0, 0, 0), vertices=vertices, mat=mat)
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    obj.data.transform(rot)
    obj.data.transform(__import__("mathutils").Matrix.Translation((a + b) / 2))
    obj.data.update()
    return obj


def blob_outline(radius, seed, points=40, roughness=0.3):
    """Irregular closed outline as [(u, v)] around the origin."""
    rng = random.Random(seed)
    waves = [(n, rng.uniform(0.3, 1.0) * roughness / n ** 0.5, rng.uniform(0, 2 * math.pi)) for n in (2, 3, 5, 7)]
    out = []
    for k in range(points):
        a = 2 * math.pi * k / points
        rr = radius * (1 + sum(amp * math.sin(n * a + ph) for n, amp, ph in waves))
        out.append((rr * math.cos(a), rr * math.sin(a)))
    return out


def cylinder_blob(name, radius_at, angle, z, size, seed, mat, offset=0.004, stretch=(1.0, 1.0)):
    """Irregular patch lying on the outside of a vertical cylinder (e.g. rust on the
    tower). ``radius_at(z)`` gives the surface radius; ``angle`` the azimuth."""
    verts = []
    for u, v in [(0.0, 0.0)] + blob_outline(size, seed):
        u, v = u * stretch[0], v * stretch[1]
        zz = z + v
        r = radius_at(zz) + offset
        phi = angle + u / r
        verts.append((r * math.cos(phi), r * math.sin(phi), zz))
    n = len(verts) - 1
    faces = [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)]
    return mesh_object(name, verts, faces, mat)


def join(name, objects):
    """Join objects into the first one (keeps per-face materials)."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    objects[0].name = name
    return objects[0]


def export_part(part, objects, obj_name=None):
    """Export the given objects as one OBJ in raw Blender coordinates (Z up)."""
    out_dir = os.path.join(ROOT, "models", part)
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        if textured(o):
            box_uv(o)
        o.select_set(True)
    bpy.ops.wm.obj_export(
        filepath=os.path.join(out_dir, (obj_name or part) + ".obj"),
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_materials=True,
        export_normals=True,
        export_uv=False,
        export_triangulated_mesh=True,   # RViz/assimp choke on the n-gons booleans leave behind
    )
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(out_dir, (obj_name or part) + ".glb"),
        export_format="GLB",
        use_selection=True,
        export_yup=False,          # keep Z up, as in the URDF
        export_apply=True,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_materials="EXPORT",
        export_image_format="JPEG",        # keep the textures small in the GLB
    )


def save_blend(part):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, "blender", "parts", part + ".blend"))
