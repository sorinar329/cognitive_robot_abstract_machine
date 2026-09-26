"""Realistic look for Blender renders of the turbine and the G1 (EEVEE, sky, sun, PBR).

Used by scripts/render_recording.py (``--look eevee``) and scripts/render_preview.py.
The OBJ files keep only base colours, so the materials are rebuilt by name from
blender/common.py (colour, roughness, metallic, transmission, emission).
"""
import math
import os
import sys

import bpy
from mathutils import Euler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "blender"))
import common as c  # noqa: E402

LAMP_BOOST = 6.0          # emissive parts (lamps, LEDs) brighter than in the flat previews
G1_SHELL = ((0.80, 0.81, 0.82), 0.35, 0.0)          # colour, roughness, metallic
G1_DARK = ((0.035, 0.037, 0.04), 0.45, 0.3)
G1_DARK_LINKS = ("hip_roll", "hip_yaw", "ankle", "wrist", "hand", "waist", "logo", "imu", "d435", "mid360")


def principled(mat):
    mat.use_nodes = True
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def fix_materials():
    """Rebuild imported turbine materials from the part definitions (names like 'gear_paint.001')."""
    for mat in bpy.data.materials:
        name = mat.name.split(".")[0]
        if name not in c.COLORS:
            continue
        shader = principled(mat)
        if shader is None:
            continue
        colour = tuple(c.srgb_to_linear(v) for v in c.COLORS[name][:3]) + (1.0,)
        roughness, metallic, transmission = c.FINISHES.get(name, (0.5, 0.0, 0.0))
        shader.inputs["Base Color"].default_value = colour
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
        shader.inputs["Transmission Weight"].default_value = transmission
        shader.inputs["Specular IOR Level"].default_value = 0.5
        if name in c.EMISSION:
            shader.inputs["Emission Color"].default_value = colour
            shader.inputs["Emission Strength"].default_value = c.EMISSION[name] * LAMP_BOOST
        else:
            shader.inputs["Emission Strength"].default_value = 0.0


def make_material(name, colour, roughness, metallic):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    shader = principled(mat)
    shader.inputs["Base Color"].default_value = tuple(colour) + (1.0,)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat


def color_robot(link_objects):
    """Unitree look: light shells, dark motors, feet and hands. ``link_objects``: [(link, object)]."""
    shell = make_material("g1_shell", *G1_SHELL)
    dark = make_material("g1_dark", *G1_DARK)
    for link, obj in link_objects:
        mat = dark if any(k in link for k in G1_DARK_LINKS) else shell
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def set_if(obj, **values):
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def setup(scene, sun_elevation=38.0, sun_azimuth=215.0, sun_strength=2.6, samples=48, exposure=-0.6, sky_strength=0.2):
    """EEVEE with a physical sky and a sun lamp; AgX colour management."""
    scene.render.engine = "BLENDER_EEVEE"
    set_if(scene.eevee, taa_render_samples=samples, use_raytracing=True, use_shadows=True,
           shadow_ray_count=2, shadow_step_count=8, use_fast_gi=True, fast_gi_distance=4.0,
           use_gtao=True, gtao_distance=1.0, use_bloom=True)
    rt = getattr(scene.eevee, "ray_tracing_options", None)
    if rt is not None:
        set_if(rt, resolution_scale="2", use_denoise=True)

    world = scene.world or bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    sky = nodes.new("ShaderNodeTexSky")
    for kind in ("MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
        try:
            sky.sky_type = kind
            break
        except TypeError:
            continue
    set_if(sky, sun_disc=False, sun_elevation=math.radians(sun_elevation), sun_rotation=math.radians(sun_azimuth),
           altitude=300.0, air_density=1.0, dust_density=1.5, ozone_density=1.0)
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = sky_strength
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], out.inputs["Surface"])

    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = sun_strength
    sun_data.angle = math.radians(0.8)
    set_if(sun_data, use_shadow=True)
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = Euler((math.radians(90 - sun_elevation), 0.0, math.radians(sun_azimuth + 90)), "XYZ")
    scene.collection.objects.link(sun)

    vs = scene.view_settings
    for transform in ("AgX", "Filmic", "Standard"):
        try:
            vs.view_transform = transform
            break
        except TypeError:
            continue
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "None"):
        try:
            vs.look = look
            break
        except TypeError:
            continue
    vs.exposure = exposure
    fix_materials()
