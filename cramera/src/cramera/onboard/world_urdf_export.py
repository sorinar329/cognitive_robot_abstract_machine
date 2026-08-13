"""
Export a robot's kinematic tree straight from the parsed :class:`World`, into a URDF
plus its meshes.

For demos that never call :class:`~semantic_digital_twin.adapters.urdf.URDFParser` at
all -- an MJCF-loaded world, for instance -- :class:`~cramera.onboard.demo.Recorder`'s
asset hooks have nothing to capture, so ``scene.json`` ends up with no robot model.
This walks the already-resolved World object instead of any source file, which works
regardless of which parser built it, and writes a URDF that
:func:`cramera.onboard.bundle_urdf.bundle_urdf`'s pipeline can bundle exactly like a
captured one.

Only ``<visual>`` geometry is written -- the viewer never uses ``<collision>``.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from typing_extensions import List, Optional, Set

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.world_entity import Body

from cramera.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_JOINT_EFFORT = 100.0
"""
``<limit effort="...">`` written for every non-fixed joint.

The viewer only ever replays recorded or live joint positions -- it never runs
dynamics -- so this value is never acted on; URDF just requires a ``<limit>`` on
non-continuous revolute/prismatic joints.
"""

DEFAULT_JOINT_VELOCITY = 1.0
"""
``<limit velocity="...">`` written for every non-fixed joint, for the same reason as
:data:`DEFAULT_JOINT_EFFORT`.
"""


def _robot_subtree(world: World, robot: AbstractRobot) -> List[Body]:
    """
    :return: ``robot.root`` and every body reachable from it through the world's
        connections, in topological (parent-before-child) order.
    """
    subtree: Set[Body] = {robot.root}
    ordered: List[Body] = [robot.root]
    for body in world.bodies_topologically_sorted:
        if body in subtree:
            continue
        connection = body.parent_connection
        if connection is not None and connection.parent in subtree:
            subtree.add(body)
            ordered.append(body)
    return ordered


def _xyz_rpy(transform: HomogeneousTransformationMatrix) -> tuple[str, str]:
    """
    :return: ``transform``'s translation and rotation, as the ``xyz``/``rpy`` strings a
        URDF ``<origin>`` element takes.
    """
    x, y, z = (float(value) for value in transform.to_position().to_np().flatten()[:3])
    roll, pitch, yaw = transform.to_rotation_matrix().to_rpy()
    return (
        "%.6f %.6f %.6f" % (x, y, z),
        "%.6f %.6f %.6f" % (float(roll), float(pitch), float(yaw)),
    )


def _build_geometry_element(
    shape: Shape, meshes_directory: Path
) -> Optional[ET.Element]:
    """
    Build the ``<geometry>`` element for ``shape`` (copying its file, for a mesh).

    :param shape: The shape to render.
    :param meshes_directory: Directory mesh files are copied into.
    :return: The ``<geometry>`` element, or ``None`` if ``shape`` cannot be rendered (an
        unsupported shape type, or a mesh whose file no longer exists).
    """
    geometry = ET.Element("geometry")
    if isinstance(shape, Mesh):
        source = Path(shape.filename)
        if not source.is_file():
            logger.debug("mesh file missing, skipping visual: %s", source)
            return None
        meshes_directory.mkdir(parents=True, exist_ok=True)
        destination = meshes_directory / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        mesh_element = ET.SubElement(geometry, "mesh", filename="meshes/" + source.name)
        if (shape.scale.x, shape.scale.y, shape.scale.z) != (1.0, 1.0, 1.0):
            mesh_element.set(
                "scale",
                "%.6f %.6f %.6f" % (shape.scale.x, shape.scale.y, shape.scale.z),
            )
    elif isinstance(shape, Box):
        ET.SubElement(
            geometry,
            "box",
            size="%.6f %.6f %.6f" % (shape.scale.x, shape.scale.y, shape.scale.z),
        )
    elif isinstance(shape, Cylinder):
        ET.SubElement(
            geometry,
            "cylinder",
            radius="%.6f" % (shape.width / 2),
            length="%.6f" % shape.height,
        )
    elif isinstance(shape, Sphere):
        ET.SubElement(geometry, "sphere", radius="%.6f" % shape.radius)
    else:
        logger.debug(
            "unsupported shape type, skipping visual: %s", type(shape).__name__
        )
        return None
    return geometry


def _add_visual(link: ET.Element, shape: Shape, meshes_directory: Path) -> None:
    """
    Append ``<visual>`` for ``shape`` to ``link``, if its geometry can be rendered.

    :param link: The ``<link>`` element to add the ``<visual>`` to.
    :param shape: The shape to render.
    :param meshes_directory: Directory mesh files are copied into.
    """
    geometry = _build_geometry_element(shape, meshes_directory)
    if geometry is None:
        return
    visual = ET.SubElement(link, "visual")
    origin_xyz, origin_rpy = _xyz_rpy(shape.origin)
    ET.SubElement(visual, "origin", xyz=origin_xyz, rpy=origin_rpy)
    visual.append(geometry)
    material = ET.SubElement(visual, "material", name="")
    ET.SubElement(
        material,
        "color",
        rgba="%.6f %.6f %.6f %.6f"
        % (shape.color.R, shape.color.G, shape.color.B, shape.color.A),
    )


def _add_link(root: ET.Element, body: Body, meshes_directory: Path) -> None:
    """
    Append ``<link name="...">`` for ``body``, with one ``<visual>`` per visual shape.
    """
    link = ET.SubElement(root, "link", name=body.name.name)
    for shape in body.visual.shapes:
        _add_visual(link, shape, meshes_directory)


def _joint_type_and_axis(connection: Connection) -> tuple[str, Optional[str]]:
    """
    :return: The URDF joint type for ``connection``, and its axis string if it has one.
    """
    if isinstance(connection, (RevoluteConnection, PrismaticConnection)):
        joint_type = (
            "revolute" if isinstance(connection, RevoluteConnection) else "prismatic"
        )
        axis = connection.axis.to_np().flatten()[:3]
        return joint_type, "%.6f %.6f %.6f" % (
            float(axis[0]),
            float(axis[1]),
            float(axis[2]),
        )
    # Fixed connections, and anything this exporter does not have a moving URDF joint
    # type for (Connection6DoF, ScrewConnection, ...) -- unexpected inside a fixed-base
    # robot's own body tree, so rendered at their current pose instead of failing.
    return "fixed", None


def _add_joint(root: ET.Element, body: Body) -> None:
    """
    Append ``<joint>`` connecting ``body`` to its parent, if it has one.
    """
    connection = body.parent_connection
    if connection is None:
        return
    joint_type, axis = _joint_type_and_axis(connection)
    joint = ET.SubElement(
        root,
        "joint",
        name=connection.name.name if connection.name else body.name.name + "_joint",
        type=joint_type,
    )
    ET.SubElement(joint, "parent", link=connection.parent.name.name)
    ET.SubElement(joint, "child", link=body.name.name)
    origin_xyz, origin_rpy = _xyz_rpy(connection.parent_T_connection_expression)
    ET.SubElement(joint, "origin", xyz=origin_xyz, rpy=origin_rpy)
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=axis)
        limits = connection.dof.limits
        lower = limits.lower.position
        upper = limits.upper.position
        ET.SubElement(
            joint,
            "limit",
            lower="%.6f" % lower if lower is not None else "-3.141593",
            upper="%.6f" % upper if upper is not None else "3.141593",
            effort=str(DEFAULT_JOINT_EFFORT),
            velocity=str(DEFAULT_JOINT_VELOCITY),
        )


def export_robot_urdf(
    world: World, robot: AbstractRobot, output_directory: Path
) -> Optional[Path]:
    """
    Write a URDF (and its meshes, under ``<output_directory>/meshes``) for ``robot``'s
    kinematic tree, from the world's own resolved geometry rather than from a source
    file.

    :param world: The world ``robot`` was annotated on.
    :param robot: The robot to export.
    :param output_directory: Directory the URDF and its meshes are written into.
    :return: The written URDF's path, or ``None`` if ``robot``'s body tree is empty.
    """
    bodies = _robot_subtree(world, robot)
    if not bodies:
        return None
    root_element = ET.Element("robot", name=type(robot).__name__.lower())
    meshes_directory = output_directory / "meshes"
    for body in bodies:
        _add_link(root_element, body, meshes_directory)
    # bodies[0] is always robot.root (see _robot_subtree): a URDF root link has no
    # parent joint of its own, and root's own parent_connection (if any) reaches
    # outside this export -- wherever the robot is mounted in the wider world is the
    # frontend's own concern via the scene's "prefix", not this file's.
    for body in bodies[1:]:
        _add_joint(root_element, body)
    output_directory.mkdir(parents=True, exist_ok=True)
    urdf_path = output_directory / (type(robot).__name__.lower() + ".urdf")
    ET.indent(root_element)
    ET.ElementTree(root_element).write(
        urdf_path, encoding="unicode", xml_declaration=False
    )
    logger.info(
        "exported %s: %d links from the world model (no source URDF was captured)",
        urdf_path.name,
        len(bodies),
    )
    return urdf_path
