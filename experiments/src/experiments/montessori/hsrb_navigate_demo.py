"""
The simplest possible proof that a real HSRB can change its base pose while physically
held together in MuJoCo: drive its :class:`~semantic_digital_twin.world_description.connections.OmniDrive`
base from one point to another in a straight line, arm parked throughout, then confirm
it actually arrived and every other joint stayed put.

Drives the base kinematically (moving :attr:`OmniDrive.origin` in small steps, the same
way :mod:`experiments.montessori.montessori_demo`'s own Giskard/CRAM-driven navigation
already updates it) rather than through real wheel-floor contact dynamics:
:class:`~semantic_digital_twin.adapters.multi_sim.MultiSimBuilder` unconditionally
ignores :class:`OmniDrive` connections when building a MuJoCo scene (an ignored
connection is, per MuJoCo's own semantics, implicitly a rigid weld to the parent body),
so the base has no MuJoCo degree of freedom of its own to actuate physically at all --
only the real wheel/caster joints further down the tree do, and spinning those against
a chassis that is welded to the world is what produces the "strange movements" this demo
exists to sidestep. MuJoCo is still used throughout the drive, physically holding every
other joint (see :mod:`experiments.montessori.hsrb_equipment`) exactly as
:mod:`experiments.montessori.hsrb_park_demo` does.

Second of three standalone HSRB smoke tests; needs
:mod:`experiments.montessori.hsrb_park_demo` to already work, since it reuses that
script's park configuration and physically holds the robot the same way while driving.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.montessori.hsrb_navigate_demo
    python -m experiments.montessori.hsrb_navigate_demo --viewer
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
from experiments.montessori.hsrb_park_demo import apply_park_configuration
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, OmniDrive
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

NODE_NAME = "hsrb_navigate_demo"
"""
Prefix given to every body this script creates.
"""

DRIVE_DISTANCE = 3.0
"""
How far, in meters along the world's +x axis, the robot drives from its start pose.
"""

FLOOR_SCALE = Scale(DRIVE_DISTANCE + 3.0, 4.0, 0.02)
"""
Size of the floor slab, long enough for :data:`DRIVE_DISTANCE` plus clearance on both
ends; its top surface is the world's ``z = 0``.
"""

DRIVE_SPEED = 0.2
"""
Base translation speed, in meters/second, :func:`drive_to` paces the waypoints at.
"""

WAYPOINT_INTERVAL = 0.05
"""
Real-time seconds between successive :attr:`OmniDrive.origin` updates while driving,
fine-grained enough to look like continuous motion in a MuJoCo viewer.
"""

POST_DRIVE_HOLD_DURATION = 2.0
"""
Real-time seconds to keep holding the final pose in MuJoCo after the drive finishes,
before checking whether everything actually stayed put.
"""

POSITION_TOLERANCE = 0.01
"""
Maximum allowed distance, in meters, between the base's final ``x`` and
:data:`DRIVE_DISTANCE` for :func:`main` to report the drive as successful.
"""

MAX_JOINT_DRIFT = 0.05
"""
Maximum allowed change, in radians (or meters for the torso's prismatic joint), of any
held arm/head/torso degree of freedom's position between the start and end of the whole
drive, for :func:`main` to report the hold as successful.
"""

MUJOCO_STEP_SIZE = 2e-4
"""
MuJoCo simulation step size, matching :mod:`experiments.montessori.hsrb_park_demo`'s own.
"""

MUJOCO_INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
"""
Numerical integrator MuJoCo advances the physics with, matching
:mod:`experiments.montessori.hsrb_park_demo`'s own (see that module's docstring for it).
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


def build_scene() -> tuple[World, HSRB, OmniDrive]:
    """
    Build a bare, elongated floor and spawn a real HSRB standing at one end of it,
    facing down its length.

    :return: The world, the spawned HSRB, and the :class:`OmniDrive` connection driving
        its base.
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
                    x=DRIVE_DISTANCE / 2, z=-FLOOR_SCALE.z / 2
                ),
            )
        )

    robot, drive = spawn_mobile_robot(world, HSRB, Point3(0.0, 0.0, 0.0))
    return world, robot, drive


def drive_to(world: World, drive: OmniDrive, target_x: float) -> None:
    """
    Kinematically drive the base from wherever :attr:`OmniDrive.origin` currently has it
    to ``target_x`` (holding ``y``/``yaw`` at ``0``) in a straight line, at
    :data:`DRIVE_SPEED`, updating it every :data:`WAYPOINT_INTERVAL`.

    :param world: The world ``drive`` belongs to.
    :param drive: The base connection to move.
    :param target_x: Where to stop, in ``world``'s root frame.
    """
    start_x = float(world.state[drive.x.id].position)
    distance = target_x - start_x
    duration = abs(distance) / DRIVE_SPEED
    step_count = max(1, int(duration / WAYPOINT_INTERVAL))
    for step in range(1, step_count + 1):
        fraction = step / step_count
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=start_x + distance * fraction, y=0.0, yaw=0.0, reference_frame=world.root
        )
        time.sleep(WAYPOINT_INTERVAL)


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
    Build the scene, park the robot, physically hold it in MuJoCo while driving its base
    :data:`DRIVE_DISTANCE` forward, and report whether it arrived within
    :data:`POSITION_TOLERANCE` with every other joint still within :data:`MAX_JOINT_DRIFT`
    of its parked target.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    arguments = _parse_arguments()

    ensure_hsrb_description_available()

    world, robot, drive = build_scene()
    apply_park_configuration(robot)
    weld_gripper(robot)
    disable_robot_self_collision(robot)
    hold_controlled_joints_in_mujoco(robot)

    held_connections = list(robot.degrees_of_freedom_with_hardware_interface)
    start_joint_positions = {
        dof: float(world.state[dof.id].position) for dof in held_connections
    }

    multi_sim = HSRBMujocoSim(
        world=world,
        headless=not arguments.viewer,
        step_size=MUJOCO_STEP_SIZE,
        balanceinertia=True,
        integrator=MUJOCO_INTEGRATOR,
    )
    multi_sim.start_simulation()
    try:
        logger.info(
            "Driving %.2fm at %.2fm/s...", DRIVE_DISTANCE, DRIVE_SPEED
        )
        drive_to(world, drive, DRIVE_DISTANCE)
        logger.info(
            "Arrived; holding for %.1fs...", POST_DRIVE_HOLD_DURATION
        )
        time.sleep(POST_DRIVE_HOLD_DURATION)
    finally:
        multi_sim.stop_simulation()

    world.update_forward_kinematics()
    final_x = float(world.state[drive.x.id].position)
    position_error = abs(final_x - DRIVE_DISTANCE)
    arrived = position_error <= POSITION_TOLERANCE

    drifts = {
        dof: abs(float(world.state[dof.id].position) - start_joint_positions[dof])
        for dof in held_connections
    }
    max_drift_dof = max(drifts, key=drifts.get)
    max_drift = drifts[max_drift_dof]
    stayed_parked = max_drift <= MAX_JOINT_DRIFT
    finite = all(
        np.isfinite(float(world.state[dof.id].position)) for dof in held_connections
    ) and np.isfinite(final_x)

    logger.info(
        "Final base x: %.4f (target %.4f, error %.4f, tolerance %.4f).",
        final_x,
        DRIVE_DISTANCE,
        position_error,
        POSITION_TOLERANCE,
    )
    logger.info(
        "Largest joint drift while driving: %s moved %.4f (limit %.4f).",
        max_drift_dof.name,
        max_drift,
        MAX_JOINT_DRIFT,
    )
    logger.info(
        "NAVIGATE: %s (arrived: %s, stayed parked: %s, finite: %s)",
        "PASS" if arrived and stayed_parked and finite else "FAIL",
        arrived,
        stayed_parked,
        finite,
    )


if __name__ == "__main__":
    main()
