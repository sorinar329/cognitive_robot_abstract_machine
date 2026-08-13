"""
Unit tests for exporting a robot's kinematic tree straight from the World model.

Exercised against a small, real ``World``/``Body``/``Connection`` tree (not mimics): the
exporter's own logic is ``isinstance`` checks against the real connection and shape
classes, which a duck-typed stand-in cannot satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera.onboard.world_urdf_export import export_robot_urdf


@dataclass
class FakeRobot:
    """
    Stands in for an :class:`AbstractRobot`: the exporter only ever reads ``root`` and
    the mimic's own class name.
    """

    root: Body


def two_link_world() -> tuple[World, Body, Body]:
    """
    A world with a root body and one child, joined by a revolute connection.
    """
    world = World()
    root = Body(name=PrefixedName("base_link"))
    child = Body(name=PrefixedName("arm_link"))
    with world.modify_world():
        connection = RevoluteConnection.create_with_dofs(
            world,
            parent=root,
            child=child,
            axis=Vector3(x=0.0, y=0.0, z=1.0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-1.5),
                upper=DerivativeMap(position=1.5),
            ),
        )
        world.add_connection(connection)
    return world, root, child


class TestLinksAndJoints:
    def test_every_body_in_the_tree_gets_a_link(self, tmp_path):
        world, root, child = two_link_world()

        urdf_path = export_robot_urdf(world, FakeRobot(root=root), tmp_path)

        text = urdf_path.read_text()
        assert '<link name="base_link">' in text or '<link name="base_link" />' in text
        assert '<link name="arm_link">' in text or '<link name="arm_link" />' in text

    def test_a_revolute_connection_becomes_a_revolute_joint_with_its_limits(
        self, tmp_path
    ):
        world, root, child = two_link_world()

        urdf_path = export_robot_urdf(world, FakeRobot(root=root), tmp_path)

        text = urdf_path.read_text()
        assert 'type="revolute"' in text
        assert 'lower="-1.500000"' in text
        assert 'upper="1.500000"' in text

    def test_a_fixed_connection_becomes_a_fixed_joint_without_an_axis(self, tmp_path):
        world = World()
        root = Body(name=PrefixedName("base_link"))
        child = Body(name=PrefixedName("sensor_link"))
        with world.modify_world():
            world.add_connection(FixedConnection(parent=root, child=child))

        urdf_path = export_robot_urdf(world, FakeRobot(root=root), tmp_path)

        text = urdf_path.read_text()
        assert 'type="fixed"' in text
        assert "<axis" not in text

    def test_a_body_outside_the_robots_tree_is_not_exported(self, tmp_path):
        """
        A sibling of the robot's own root -- part of the same world, but not reachable
        from ``robot.root`` -- must not appear in the export.
        """
        world = World()
        world_origin = Body(name=PrefixedName("world"))
        robot_root = Body(name=PrefixedName("base_link"))
        shelf = Body(name=PrefixedName("shelf"))
        with world.modify_world():
            world.add_connection(FixedConnection(parent=world_origin, child=robot_root))
            world.add_connection(FixedConnection(parent=world_origin, child=shelf))

        urdf_path = export_robot_urdf(world, FakeRobot(root=robot_root), tmp_path)

        text = urdf_path.read_text()
        assert "base_link" in text
        assert "shelf" not in text


class TestGeometry:
    def test_a_box_shape_is_written_as_a_urdf_box(self, tmp_path):
        world, root, child = two_link_world()
        child.visual = ShapeCollection(shapes=[Box(scale=Scale(x=0.1, y=0.2, z=0.3))])

        urdf_path = export_robot_urdf(world, FakeRobot(root=root), tmp_path)

        assert 'size="0.100000 0.200000 0.300000"' in urdf_path.read_text()

    def test_a_mesh_file_is_copied_into_the_output_directory(self, tmp_path):
        world, root, child = two_link_world()
        mesh_source = tmp_path / "source" / "gripper.obj"
        mesh_source.parent.mkdir()
        mesh_source.write_text("mock mesh content")
        child.visual = ShapeCollection(shapes=[Mesh(filename=str(mesh_source))])

        output_directory = tmp_path / "bundle"
        urdf_path = export_robot_urdf(world, FakeRobot(root=root), output_directory)

        assert (output_directory / "meshes" / "gripper.obj").read_text() == (
            "mock mesh content"
        )
        assert 'filename="meshes/gripper.obj"' in urdf_path.read_text()

    def test_a_missing_mesh_file_is_skipped_without_failing(self, tmp_path):
        world, root, child = two_link_world()
        child.visual = ShapeCollection(
            shapes=[Mesh(filename=str(tmp_path / "does_not_exist.obj"))]
        )

        urdf_path = export_robot_urdf(world, FakeRobot(root=root), tmp_path)

        assert "does_not_exist" not in urdf_path.read_text()

    def test_a_body_with_no_visual_shapes_still_gets_a_link_with_no_visuals(
        self, tmp_path
    ):
        world, root, child = two_link_world()

        urdf_path = export_robot_urdf(world, FakeRobot(root=root), tmp_path)

        text = urdf_path.read_text()
        assert "<visual>" not in text


class TestEmptyTree:
    def test_a_robot_whose_root_has_no_children_still_exports_its_own_link(
        self, tmp_path
    ):
        world = World()
        root = Body(name=PrefixedName("lone_link"))
        with world.modify_world():
            world.add_kinematic_structure_entity(root)

        urdf_path = export_robot_urdf(world, FakeRobot(root=root), tmp_path)

        assert urdf_path is not None
        assert "lone_link" in urdf_path.read_text()
