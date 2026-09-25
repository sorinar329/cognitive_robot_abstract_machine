"""Link/joint specs, collision helpers and the URDF writer.

A part module (turbine/parts/*.py) exposes:
  LINKS              list of link specs (see ``link``)
  INSPECTION_POINTS  list of ``inspection_point`` specs
  FAULTS             dict fault_id -> ``fault`` spec
"""
import math
from xml.dom import minidom
from xml.etree import ElementTree as ET

PKG = "windturbine_model"
X_AXIS_RPY = (0.0, math.pi / 2, 0.0)   # URDF cylinders are along Z; this puts them along X


def span_box(x, y, z):
    """Collision box given as (min, max) intervals in the link frame."""
    return dict(type="box", size=(x[1] - x[0], y[1] - y[0], z[1] - z[0]),
                xyz=((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))


def z_cylinder(radius, z, x=0.0, y=0.0):
    return dict(type="cylinder", radius=radius, length=z[1] - z[0], xyz=(x, y, (z[0] + z[1]) / 2))


def ring_boxes(radius_at, wall_at, z0, z1, segments=16, band=10.8, skip=None):
    """Hollow cylinder wall as boxes (URDF has no tube). ``skip(angle, z)`` drops boxes,
    e.g. for a door opening. Heights are in the link frame."""
    boxes = []
    n_bands = max(1, math.ceil((z1 - z0) / band))
    for b in range(n_bands):
        za, zb = z0 + (z1 - z0) * b / n_bands, z0 + (z1 - z0) * (b + 1) / n_bands
        zm = (za + zb) / 2
        r, t = radius_at(zm), wall_at(zm)
        chord = 2 * r * math.tan(math.pi / segments) + 0.02
        for k in range(segments):
            a = 2 * math.pi * k / segments
            if skip and skip(a, zm):
                continue
            rm = r - t / 2
            boxes.append(dict(type="box", size=(t, chord, zb - za), xyz=(rm * math.cos(a), rm * math.sin(a), zm),
                              rpy=(0, 0, a)))
    return boxes


def disk_boxes(radius, z, segments=24, rim=None):
    """Solid vertical cylinder as boxes: an inscribed square plus a rim of radial boxes.
    Planners that use axis-aligned bounds of each shape (coraplex navigation) then see a
    near-round footprint instead of the cylinder's bounding square."""
    rim = rim or radius * (1 - math.cos(math.pi / 4)) + 0.02
    half = radius * math.cos(math.pi / 4)
    boxes = [span_box((-half, half), (-half, half), z)]
    r_in = radius - rim
    chord = 2 * radius * math.tan(math.pi / segments) + 0.02
    for k in range(segments):
        a = 2 * math.pi * k / segments
        rm = (r_in + radius) / 2
        boxes.append(dict(type="box", size=(rim, chord, z[1] - z[0]),
                          xyz=(rm * math.cos(a), rm * math.sin(a), (z[0] + z[1]) / 2), rpy=(0, 0, a)))
    return boxes


def floor_boxes(z, x_range=None, y_range=None, radius=None, holes=(), strip=0.25):
    """A flat deck (z range) as boxes, leaving out rectangular ``holes`` [(x range, y range)].
    Rectangular deck: bounded by x/y ranges, split in rows at the holes' y edges.
    Round deck of ``radius``: strips of ``strip`` along X, cut to the circle."""
    if radius is not None:
        n = max(1, round(2 * radius / strip))
        ys = [-radius + 2 * radius * i / n for i in range(n + 1)]
    else:
        ys = sorted({y_range[0], y_range[1], *(min(max(e, y_range[0]), y_range[1]) for _, hy in holes for e in hy)})
    boxes = []
    for y0, y1 in zip(ys, ys[1:]):
        if radius is not None:
            half = math.sqrt(max(radius ** 2 - max(abs(y0), abs(y1)) ** 2, 0.0))
            xs = (-half, half)
        else:
            xs = x_range
        x = xs[0]
        for c0, c1 in sorted(hx for hx, hy in holes if hy[0] < y1 - 1e-6 and hy[1] > y0 + 1e-6) + [(xs[1], xs[1])]:
            if min(c0, xs[1]) > x + 0.02:
                boxes.append(span_box((x, min(c0, xs[1])), (y0, y1), z))
            x = max(x, c1)
    return boxes


def x_cylinder(radius, x, y=0.0, z=0.0):
    return dict(type="cylinder", radius=radius, length=x[1] - x[0],
                xyz=((x[0] + x[1]) / 2, y, z), rpy=X_AXIS_RPY)


def link(name, parent, mesh=None, variants=None, xyz=(0, 0, 0), rpy=(0, 0, 0),
         joint="fixed", axis=None, limits=None, collisions=(), velocity=1.0):
    """``mesh`` is one path or a list; ``variants`` maps state -> mesh (first = healthy)."""
    return dict(name=name, parent=parent, mesh=mesh, variants=variants, xyz=xyz, rpy=rpy,
                joint=joint, axis=axis, limits=limits, collisions=list(collisions), velocity=velocity)


def inspection_point(name, parent, xyz, view_from, what, distance=1.0, outside=False, zone=None):
    """A frame on the thing to look at. A good camera position is ``distance``
    metres from it along ``view_from`` (a direction in the parent frame).
    ``outside``: seen from outside the turbine (ground robot or drone).
    ``zone``: where the robot stands: "outside", "tower" or "nacelle" (default from ``outside``)."""
    return dict(name=name, parent=parent, xyz=tuple(xyz), view_from=tuple(view_from), distance=distance,
                what=what, outside=outside, zone=zone or ("outside" if outside else "nacelle"))


def fault(part, component, description, inspection_point, observable_by,
          set_variants=None, overlays=(), severity="medium", signals=None, joint_states=None):
    """``set_variants``: {link_name: state}; ``overlays``: extra ``link`` specs;
    ``joint_states``: {joint_name: position} the world must be set to (URDF has
    no initial state, so these go to the ground truth for the loader to apply)."""
    return dict(part=part, component=component, description=description,
                inspection_point=inspection_point, observable_by=list(observable_by),
                set_variants=set_variants or {}, overlays=list(overlays),
                severity=severity, signals=signals or {}, joint_states=joint_states or {})


def joint_name(child):
    """Joints are named after the link they move, like ``laboratory_drawer_joint``."""
    return f"{child['name']}_joint"


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _geometry(parent, spec):
    geo = ET.SubElement(parent, "geometry")
    if spec["type"] == "box":
        ET.SubElement(geo, "box", size=_fmt(spec["size"]))
    elif spec["type"] == "cylinder":
        ET.SubElement(geo, "cylinder", radius=f"{spec['radius']:.6g}", length=f"{spec['length']:.6g}")
    else:
        ET.SubElement(geo, "mesh", filename=spec["filename"])
    ET.SubElement(parent, "origin", xyz=_fmt(spec.get("xyz", (0, 0, 0))), rpy=_fmt(spec.get("rpy", (0, 0, 0))))


def package_uri(mesh):
    """ROS package URI of a mesh in ``models/`` (RViz, ROS tooling)."""
    return f"package://{PKG}/models/{mesh}"


def write(path, name, links, states, mesh_uri=package_uri, skip_meshes=()):
    """``states`` maps link name -> chosen variant state; ``mesh_uri`` turns a
    ``models/``-relative mesh path into the URDF filename; visuals whose mesh is
    in ``skip_meshes`` are left out (their collisions stay)."""
    robot = ET.Element("robot", name=name)
    ET.SubElement(robot, "link", name="world")
    for spec in links:
        el = ET.SubElement(robot, "link", name=spec["name"])
        inertial = ET.SubElement(el, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="1", ixy="0", ixz="0", iyy="1", iyz="0", izz="1")
        mesh = spec["mesh"]
        if spec["variants"]:
            mesh = spec["variants"][states.get(spec["name"], next(iter(spec["variants"])))]
        for m in ([mesh] if isinstance(mesh, str) else mesh or []):
            if m not in skip_meshes:
                _geometry(ET.SubElement(el, "visual"), dict(type="mesh", filename=mesh_uri(m)))
        for col in spec["collisions"]:
            _geometry(ET.SubElement(el, "collision"), col)

        joint = ET.SubElement(robot, "joint", name=joint_name(spec), type=spec["joint"])
        ET.SubElement(joint, "parent", link=spec["parent"])
        ET.SubElement(joint, "child", link=spec["name"])
        ET.SubElement(joint, "origin", xyz=_fmt(spec["xyz"]), rpy=_fmt(spec["rpy"]))
        if spec["joint"] != "fixed":
            ET.SubElement(joint, "axis", xyz=_fmt(spec["axis"]))
        if spec["limits"]:
            ET.SubElement(joint, "limit", lower=str(spec["limits"][0]), upper=str(spec["limits"][1]),
                          effort="100", velocity=str(spec.get("velocity", 1.0)))
    with open(path, "w") as f:
        f.write(minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  "))
