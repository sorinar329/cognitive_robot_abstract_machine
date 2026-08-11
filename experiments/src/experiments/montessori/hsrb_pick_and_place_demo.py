"""
The simplest possible proof that a real HSRB can move an object from one table to
another: two tables and a loose cube on the first one, the robot drives up to it (see
:mod:`experiments.montessori.hsrb_navigate_demo`), picks it up, drives to the second
table, and puts it down there.

"Picks up" here is a kinematic re-parent, not a real reach-and-grasp: the cube is
re-attached directly to the gripper's tool frame at a fixed offset, the same technique
:mod:`experiments.montessori.montessori_demo`'s own ``PlaceAction`` already uses to
attach a shape once a real grasp finishes. A real grasp needs Giskard/CRAM motion
planning to reach for the cube and a physically stable, articulating gripper to close
on it -- both substantial, independent pieces of work (the gripper specifically needed
the numerical-stability fixes in :mod:`experiments.montessori.hsrb_equipment`'s own
``weld_gripper`` just to hold a fixed pose; making it grasp for real is further work
again) -- out of scope for proving the one thing this demo actually needs to prove:
that a real HSRB, having already been shown to park (see
:mod:`experiments.montessori.hsrb_park_demo`) and drive (see
:mod:`experiments.montessori.hsrb_navigate_demo`), can carry something between two
points while doing both.

Third of three standalone HSRB smoke tests; needs both of the others to already work.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.montessori.hsrb_pick_and_place_demo
    python -m experiments.montessori.hsrb_pick_and_place_demo --viewer
"""

from __future__ import annotations

import argparse
import logging
import time

import mujoco
import numpy as np

from experiments.montessori.hsrb_description_setup import (
    ensure_hsrb_description_available,
)
from experiments.montessori.hsrb_equipment import (
    HSRBMujocoSim,
    disable_robot_self_collision,
    hold_controlled_joints_in_mujoco,
    spawn_mobile_robot,
    weld_gripper,
)
from experiments.montessori.hsrb_navigate_demo import drive_to
from experiments.montessori.hsrb_park_demo import apply_park_configuration
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

NODE_NAME = "hsrb_pick_and_place_demo"
"""
Prefix given to every body this script creates.
"""

TABLE_SCALE = Scale(0.6, 0.6, 0.5)
"""
Size of each table, legs included (one solid box, matching
:mod:`experiments.montessori.hsrb_park_demo`'s own "geometry doesn't need legs" choice).
"""

TABLE_A_POSITION = Point3(0.5, 0.5, TABLE_SCALE.z / 2)
"""
Centre of the table the cube starts on.
"""

TABLE_B_POSITION = Point3(2.5, -0.5, TABLE_SCALE.z / 2)
"""
Centre of the table the cube ends up on.
"""

CUBE_EDGE_LENGTH = 0.05
"""
Edge length of the loose cube.
"""

CUBE_START_POSITION = Point3(
    float(TABLE_A_POSITION.x),
    float(TABLE_A_POSITION.y),
    TABLE_SCALE.z + CUBE_EDGE_LENGTH / 2,
)
"""
Where the cube starts, resting on table A.
"""

FLOOR_SCALE = Scale(6.0, 6.0, 0.02)
"""
Size of the floor slab; its top surface is the world's ``z = 0``.
"""

APPROACH_STANDOFF = 0.55
"""
How far in front of a table's centre (along ``-x``, facing it) the robot stops to reach
it.
"""

GRASP_OFFSET_IN_GRIPPER_FRAME = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.05)
"""
Where the carried cube sits relative to the gripper's tool frame while attached, along
its own forward axis: a plausible "held out in front of the palm" offset, not a solved
grasp pose (see this module's own docstring).
"""

PLACE_HEIGHT_ABOVE_TABLE = CUBE_EDGE_LENGTH / 2
"""
Height above a table's own top surface the cube is placed at, i.e. resting on it.
"""

MUJOCO_STEP_SIZE = 2e-4
"""
MuJoCo simulation step size, matching :mod:`experiments.montessori.hsrb_park_demo`'s own.
"""

MUJOCO_INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
"""
Numerical integrator MuJoCo advances the physics with, matching
:mod:`experiments.montessori.hsrb_park_demo`'s own.
"""

SETTLE_DURATION = 1.0
"""
Real-time seconds held at rest after each pick/place step, before checking or moving on.
"""

PLACE_POSITION_TOLERANCE = 0.05
"""
Maximum allowed horizontal distance, in meters, between the cube's final position and
:data:`TABLE_B_POSITION` for :func:`main` to report the place as successful.
"""


def _box_body(name: str, scale: Scale, color: Color) -> Body:
    """
    A body whose visual and collision geometry are one box.

    :param name: Name of the body.
    :param scale: Size of the box.
    :param color: Colour of the box.
    """
    return Body.from_shape_collection(
        PrefixedName(name, NODE_NAME), ShapeCollection([Box(scale=scale, color=color)])
    )


def build_scene() -> tuple[World, HSRB, OmniDrive, Body]:
    """
    Build a bare floor, two tables, spawn a real HSRB, and place a loose cube on the
    first table.

    :return: The world, the spawned HSRB, the :class:`OmniDrive` connection driving its
        base, and the cube's body.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root", NODE_NAME))
        world.add_kinematic_structure_entity(root)

        floor = _box_body("floor", FLOOR_SCALE, Color.GREY())
        world.add_connection(
            FixedConnection(
                parent=root,
                child=floor,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-FLOOR_SCALE.z / 2
                ),
            )
        )

        for name, position in (
            ("table_a", TABLE_A_POSITION),
            ("table_b", TABLE_B_POSITION),
        ):
            table = _box_body(name, TABLE_SCALE, Color.BEIGE())
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=table,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=position.x, y=position.y, z=position.z
                    ),
                )
            )

        cube = _box_body(
            "cube",
            Scale(CUBE_EDGE_LENGTH, CUBE_EDGE_LENGTH, CUBE_EDGE_LENGTH),
            Color.RED(),
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=cube,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=CUBE_START_POSITION.x,
                    y=CUBE_START_POSITION.y,
                    z=CUBE_START_POSITION.z,
                ),
            )
        )

    robot, drive = spawn_mobile_robot(world, HSRB, Point3(0.0, 0.0, 0.0))
    return world, robot, drive, cube


def attach_to_gripper(world: World, robot: HSRB, item: Body) -> None:
    """
    Kinematically re-parent ``item`` to the robot's gripper tool frame at
    :data:`GRASP_OFFSET_IN_GRIPPER_FRAME`, so it now moves rigidly with the gripper (see
    this module's own docstring for why this stands in for a real grasp).

    :param world: The world ``item`` belongs to, modified in place.
    :param robot: The robot whose gripper ``item`` is attached to.
    :param item: The body to attach.
    """
    tool_frame = robot.end_effector.tool_frame
    with world.modify_world():
        world.remove_connection(item.parent_connection)
        world.add_connection(
            FixedConnection(
                parent=tool_frame,
                child=item,
                parent_T_connection_expression=GRASP_OFFSET_IN_GRIPPER_FRAME,
            )
        )


def place_on_table(world: World, item: Body, table_position: Point3) -> None:
    """
    Kinematically re-parent ``item`` to the world root, resting on top of the table at
    ``table_position``.

    :param world: The world ``item`` belongs to, modified in place.
    :param item: The body to place.
    :param table_position: Centre of the table to place it on.
    """
    with world.modify_world():
        world.remove_connection(item.parent_connection)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=item,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=table_position.x,
                    y=table_position.y,
                    z=float(table_position.z) + TABLE_SCALE.z / 2 + PLACE_HEIGHT_ABOVE_TABLE,
                ),
            )
        )


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments selecting whether a MuJoCo viewer window is opened.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a MuJoCo viewer window; off by default so this runs headless.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Build the scene, park the robot, drive it to table A, attach the cube to its
    gripper, drive to table B, place the cube there, and report whether it ended up
    within :data:`PLACE_POSITION_TOLERANCE` of table B.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    arguments = _parse_arguments()

    ensure_hsrb_description_available()

    world, robot, drive, cube = build_scene()
    apply_park_configuration(robot)
    weld_gripper(robot)
    disable_robot_self_collision(robot)
    hold_controlled_joints_in_mujoco(robot)

    multi_sim = HSRBMujocoSim(
        world=world,
        headless=not arguments.viewer,
        step_size=MUJOCO_STEP_SIZE,
        balanceinertia=True,
        integrator=MUJOCO_INTEGRATOR,
    )
    multi_sim.start_simulation()
    try:
        logger.info("Driving to table A...")
        drive_to(world, drive, float(TABLE_A_POSITION.x) - APPROACH_STANDOFF)
        time.sleep(SETTLE_DURATION)

        logger.info("Picking up the cube...")
        attach_to_gripper(world, robot, cube)
        world.update_forward_kinematics()
        time.sleep(SETTLE_DURATION)

        logger.info("Driving to table B...")
        drive_to(world, drive, float(TABLE_B_POSITION.x) - APPROACH_STANDOFF)
        time.sleep(SETTLE_DURATION)

        logger.info("Placing the cube on table B...")
        place_on_table(world, cube, TABLE_B_POSITION)
        world.update_forward_kinematics()
        time.sleep(SETTLE_DURATION)
    finally:
        multi_sim.stop_simulation()

    world.update_forward_kinematics()
    final_position = world.compute_forward_kinematics(world.root, cube).to_position()
    horizontal_error = float(
        np.linalg.norm(
            [
                float(final_position.x) - float(TABLE_B_POSITION.x),
                float(final_position.y) - float(TABLE_B_POSITION.y),
            ]
        )
    )
    placed = horizontal_error <= PLACE_POSITION_TOLERANCE
    finite = np.isfinite(float(final_position.x)) and np.isfinite(float(final_position.y))

    logger.info(
        "Cube final position: (%.3f, %.3f, %.3f); horizontal error from table B: "
        "%.4f (tolerance %.4f).",
        float(final_position.x),
        float(final_position.y),
        float(final_position.z),
        horizontal_error,
        PLACE_POSITION_TOLERANCE,
    )
    logger.info(
        "PICK AND PLACE: %s",
        "PASS" if placed and finite else "FAIL",
    )


if __name__ == "__main__":
    main()
