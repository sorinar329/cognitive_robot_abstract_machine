"""
The simplest possible proof that a real HSRB can hold a static pose under MuJoCo's own
physics: a bare floor and the spawned robot, commanded straight into its arm-park/
torso-low/head-neutral configuration (no motion planning -- the joints are set directly
in the world model), then physically simulated in MuJoCo for a few seconds to confirm
the position-hold actuators (see :mod:`experiments.montessori.hsrb_equipment`) actually
keep it there instead of sagging, spinning, or diverging.

First of three standalone HSRB smoke tests, in order of dependency: this one proves the
robot can hold a pose at all; :mod:`experiments.montessori.hsrb_navigate_demo` (needs
this to work first) then proves it can also change its base pose while staying held
together; a minimal pick-and-place is only worth attempting once both do.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.montessori.hsrb_park_demo
    python -m experiments.montessori.hsrb_park_demo --viewer
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
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

NODE_NAME = "hsrb_park_demo"
"""
Prefix given to every body this script creates.
"""

FLOOR_SCALE = Scale(4.0, 4.0, 0.02)
"""
Size of the floor slab, whose top surface is the world's ``z = 0``.
"""

HEAD_NEUTRAL_JOINT_NAMES = ("head_pan_joint", "head_tilt_joint")
"""
The head's own joints, zeroed directly: unlike the arm and torso,
:class:`~semantic_digital_twin.robots.hsrb.HSRBNeck` defines no
:class:`~semantic_digital_twin.datastructures.definitions.StaticJointState` of its own
to command it into a neutral pose.
"""

MUJOCO_STEP_SIZE = 2e-4
"""
MuJoCo simulation step size, matching :mod:`experiments.montessori.montessori_demo`'s own:
the default step is too large for the wheel joints' low inertia combined with their
position hold and floor contact, and repeatedly drives ``QACC`` to ``NaN``/``Inf``.
"""

MUJOCO_INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
"""
Numerical integrator MuJoCo advances the physics with, in place of its own default
(``RK4``, an explicit method): matching ``coraplex_panda_demo``'s and
``franka_montessori_demo``'s own choice, RK4 was observed to drive ``QACC`` to ``NaN``
within the first physics step for this scene's stiff position-hold actuators (a high
proportional gain relative to a joint's own inertia and the physics step size), which
``implicitfast`` -- designed for exactly this class of stiff system -- does not.
"""

HOLD_DURATION = 5.0
"""
Real-time seconds to physically hold the parked pose in MuJoCo before checking whether
it actually stayed put.
"""

MAX_JOINT_DRIFT = 0.05
"""
Maximum allowed change, in radians (or meters for the torso's prismatic joint), of any
held degree of freedom's position between the start and end of :data:`HOLD_DURATION` for
:func:`main` to report the hold as successful.
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


def build_scene() -> tuple[World, HSRB]:
    """
    Build a bare floor and spawn a real HSRB standing on it at the origin.

    :return: The world, and the spawned HSRB.
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

    robot, _drive = spawn_mobile_robot(world, HSRB, Point3(0.0, 0.0, 0.0))
    return world, robot


def apply_park_configuration(robot: HSRB) -> None:
    """
    Kinematically command the robot straight into its arm-park/torso-low/head-neutral/
    gripper-open configuration, without any motion planning: every controlled joint's
    position is set directly in the world model, for :func:`main` to then physically
    hold in MuJoCo (see :func:`~experiments.montessori.hsrb_equipment.weld_gripper`,
    despite its name applied to whatever pose the gripper is in when it runs).

    The gripper is opened, not left wherever the URDF's own zero default puts it, and
    specifically not closed: with nothing between the fingers to grasp, a fully closed
    gripper's own finger geometry was observed to interpenetrate enough for MuJoCo's
    contact solver to apply a large corrective force in the very first physics step,
    destabilizing the whole arm.

    :param robot: The spawned robot, modified in place.
    """
    arm = robot.mobile_base.torso.arm
    torso = robot.mobile_base.torso
    for connection, target in arm.get_joint_state_by_type(StaticJointState.PARK).items():
        connection.position = target
    for connection, target in torso.get_joint_state_by_type(TorsoState.LOW).items():
        connection.position = target
    for connection, target in robot.end_effector.get_joint_state_by_type(
        GripperState.OPEN
    ).items():
        connection.position = target
    for joint_name in HEAD_NEUTRAL_JOINT_NAMES:
        robot._world.get_connection_by_name(joint_name).position = 0.0
    robot._world.update_forward_kinematics()


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
    Build the scene, park the robot, physically hold it in MuJoCo for
    :data:`HOLD_DURATION`, and report whether every held joint stayed within
    :data:`MAX_JOINT_DRIFT` of where it was commanded.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    arguments = _parse_arguments()

    ensure_hsrb_description_available()

    world, robot = build_scene()
    apply_park_configuration(robot)
    weld_gripper(robot)
    disable_robot_self_collision(robot)
    hold_controlled_joints_in_mujoco(robot)

    held_connections = list(robot.degrees_of_freedom_with_hardware_interface)
    start_positions = {
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
        logger.info("Holding parked pose for %.1fs...", HOLD_DURATION)
        time.sleep(HOLD_DURATION)
    finally:
        multi_sim.stop_simulation()

    world.update_forward_kinematics()
    drifts = {
        dof: abs(float(world.state[dof.id].position) - start_positions[dof])
        for dof in held_connections
    }
    max_drift_dof = max(drifts, key=drifts.get)
    max_drift = drifts[max_drift_dof]
    stayed_still = max_drift <= MAX_JOINT_DRIFT
    finite = all(
        np.isfinite(float(world.state[dof.id].position)) for dof in held_connections
    )

    logger.info(
        "Largest drift: %s moved %.4f (limit %.4f).",
        max_drift_dof.name,
        max_drift,
        MAX_JOINT_DRIFT,
    )
    logger.info(
        "PARK HOLD: %s (finite positions: %s)",
        "PASS" if stayed_still and finite else "FAIL",
        finite,
    )


if __name__ == "__main__":
    main()
