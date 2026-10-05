import os

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="session", autouse=True)
def unique_mujoco_scene_file(tmp_path_factory):
    """
    Give each xdist worker its own MuJoCo scene file to prevent /tmp/scene.xml race
    conditions.
    """
    try:
        from semantic_digital_twin.adapters.multi_sim import MujocoSim

        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        tmp_path = tmp_path_factory.mktemp(f"mujoco_{worker_id}")
        MujocoSim.default_file_path = str(tmp_path / "scene.xml")
    except ImportError:
        pass


@pytest.fixture
def falling_box_world() -> World:
    """
    A world holding one box on a free connection, a metre above the root.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        box = Body(name=PrefixedName("box"))
        geometry = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=box
                    ),
                    scale=Scale(0.1, 0.1, 0.1),
                )
            ],
            reference_frame=box,
        )
        box.collision, box.visual = geometry, geometry
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0, reference_frame=root
                ),
            )
        )
    return world
