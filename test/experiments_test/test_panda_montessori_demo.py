import os

import numpy as np
import pytest

from experiments.montessori.panda_montessori_demo import (
    PANDA_MOUNT_POSITION,
    SHAPE_POSITION,
    TABLE_TOP_SURFACE_Z,
    build_scene,
)
from semantic_digital_twin.world_description.connections import Connection6DoF

panda_scene_missing = not os.path.exists(
    "/home/sorin/dev/manipulation_experiments/resources/generated/stacking_scene.xml"
)

pytestmark = pytest.mark.skipif(
    panda_scene_missing,
    reason="the Panda's MJCF is not installed at PANDA_SCENE_PATH",
)


def test_scene_holds_only_a_floor_a_tabletop_an_arm_and_one_shape():
    """
    The scene carries nothing beyond what a single grasp needs.

    The point of this demo is that a failed grasp is a grasp problem; every extra body is
    something else a reach can collide with or a solver can be slowed by.
    """
    scene = build_scene()

    own_bodies = {
        body.name.name
        for body in scene.world.bodies
        if body.name.prefix == "panda_montessori_demo"
    }

    # panda_mount is the arm's own root, renamed on merge; the arm's links keep their
    # description's names and so are not counted here.
    assert own_bodies == {"root", "floor", "table_top", "shape", "panda_mount"}


def test_shape_rests_on_the_tabletop_as_a_free_body():
    """
    The shape starts on the tabletop and is free to be moved by contact and gravity.

    Its pose has to live in the free joint's own dof values, not in a fixed offset, or
    the simulator starts it at the world origin instead of on the table.
    """
    scene = build_scene()
    scene.world.update_forward_kinematics()

    position = scene.shape.global_transform.to_position()

    assert isinstance(scene.shape.parent_connection, Connection6DoF)
    assert float(position.x) == pytest.approx(float(SHAPE_POSITION.x))
    assert float(position.z) == pytest.approx(float(SHAPE_POSITION.z))
    assert float(position.z) > TABLE_TOP_SURFACE_Z


def test_arm_is_bolted_facing_the_shape_and_within_reach():
    """
    The arm is mounted on the tabletop, turned towards the shape, with the shape neither
    inside its own shoulder nor beyond its reach.
    """
    scene = build_scene()
    scene.world.update_forward_kinematics()

    base = scene.robot.root.global_transform.to_position().to_np()[:3].ravel()
    shape = scene.shape.global_transform.to_position().to_np()[:3].ravel()
    distance = float(np.linalg.norm(shape[:2] - base[:2]))

    assert base[2] == pytest.approx(float(PANDA_MOUNT_POSITION.z))
    assert 0.35 < distance < 0.75
