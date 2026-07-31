"""
The smallest scene that can show a Panda picking up a Montessori shape: a floor, one
tabletop, the arm bolted to that tabletop, and a single loose shape on it.

Deliberately not built on :class:`~experiments.montessori.world.MontessoriWorld`. That
scene carries a sorting board with its cut mesh, drawers and seven shapes, and every
failure so far has had to be untangled from one of them. Here a grasp that misses is a
grasp that missed.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.montessori.panda_montessori_demo
    python -m experiments.montessori.panda_montessori_demo --viewer

.. warning::
    The Panda ships no ROS package here -- its
    :meth:`~semantic_digital_twin.robots.panda.Panda.get_ros_file_path` raises -- so its
    description is read from the MJCF at :data:`PANDA_SCENE_PATH`, an absolute path
    outside this repository. Point it at your own copy before running.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from experiments.montessori.montessori_demo import (
    JointServoTuning,
    RobotActuatorTuning,
    _equip_robot_for_physical_simulation,
)
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import (
    MujocoGeom,
    MujocoSim,
    ReparentingMode,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.utils import rclpy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Cylinder, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

NODE_NAME = "panda_montessori_demo"
"""
Name of the ROS 2 node the visualization publishes from.
"""

PANDA_SCENE_PATH = (
    "/home/sorin/dev/manipulation_experiments/resources/generated/stacking_scene.xml"
)
"""
MJCF the Panda's description is read from.

This scene rather than the standalone ``panda.xml`` beside it, because
:class:`~semantic_digital_twin.robots.panda.Panda` resolves its gripper by the prefixed
names (``/hand``, ``/tool_frame``) that only this generated file carries; ``panda.xml``
has no ``tool_frame`` at all.
"""

PANDA_BODIES_TO_DISCARD = frozenset(
    {"target", "cube0", "cube1", "cube2", "cube3", "floor", "stack_pad"}
)
"""
Bodies of :data:`PANDA_SCENE_PATH` belonging to its own stacking task, dropped so only
the arm is merged.
"""

FLOOR_SCALE = Scale(4.0, 4.0, 0.02)
"""
Size of the floor slab, whose top surface is the world's ``z = 0``.
"""

TABLE_TOP_SCALE = Scale(1.0, 0.6, 0.025)
"""
Size of the tabletop. No legs: nothing here stands on the floor, and legs are only more
geometry for a reach to collide with.
"""

TABLE_TOP_POSITION = Point3(-0.1, 0.0, 0.5)
"""
Centre of the tabletop, wide enough in ``x`` to carry both the arm and the shape.
"""

TABLE_TOP_SURFACE_Z = float(TABLE_TOP_POSITION.z) + TABLE_TOP_SCALE.z / 2
"""
Height of the tabletop's upper surface, which everything else rests on.
"""

PANDA_MOUNT_POSITION = Point3(0.25, 0.0, TABLE_TOP_SURFACE_Z)
"""
Where the arm is bolted, on the tabletop.
"""

PANDA_MOUNT_YAW = np.pi
"""
Which way the mounted arm faces: towards the shape, which lies at lower ``x``.
"""

SHAPE_RADIUS = 0.02
"""
Radius of the loose cylinder, narrow enough for the gripper to close around.
"""

SHAPE_HEIGHT = 0.05
"""
Height of the loose cylinder.
"""

SHAPE_POSITION = Point3(-0.25, 0.0, TABLE_TOP_SURFACE_Z + SHAPE_HEIGHT / 2)
"""
Where the cylinder rests, 0.5 m in front of the arm: clear of the arm's own shoulder and
well inside its reach.
"""

GRASP_FRICTION = [1.0, 0.5, 0.5]
"""
Contact friction (sliding, torsional, rolling) of the cylinder and the fingertips.

MuJoCo's defaults leave torsional and rolling friction near zero, so a cylinder pinched
between two flat fingertips spins and rolls out of them however hard they squeeze.
"""

PANDA_ACTUATOR_TUNING = RobotActuatorTuning(
    default=JointServoTuning(
        position_gain=2000.0, velocity_gain=200.0, force_range=(-12.0, 12.0)
    ),
    by_joint_name={
        "joint1": JointServoTuning(4500.0, 450.0, (-87.0, 87.0)),
        "joint2": JointServoTuning(4500.0, 450.0, (-87.0, 87.0)),
        "joint3": JointServoTuning(3500.0, 350.0, (-87.0, 87.0)),
        "joint4": JointServoTuning(3500.0, 350.0, (-87.0, 87.0)),
        "/finger_joint1": JointServoTuning(100.0, 10.0, (-100.0, 100.0)),
    },
)
"""
Gains and force clamps read off the actuators the Panda's own MJCF declares, rather than
reused from the HSR.
"""

SIMULATION_STEP_SIZE = 5e-4
"""
Physics step size.
"""

SYNC_RATE_HZ = 100
"""
Rate at which physically simulated joints are read back into the world model.
"""


@dataclass
class PandaMontessoriScene:
    """
    The built scene and the handles a plan needs to act on it.
    """

    world: World
    """
    The world holding the floor, tabletop, arm and shape.
    """

    robot: Panda
    """
    The arm, bolted to the tabletop.
    """

    shape: Body
    """
    The loose cylinder, free to be moved by contact and gravity.
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


def _parse_panda() -> World:
    """
    Read the arm out of :data:`PANDA_SCENE_PATH`, without the stacking task sharing it.

    The scene's root is renamed, since a body called ``world`` collides with the root the
    simulator's own scene builder creates.

    :return: A world holding only the arm.
    """
    panda_world = MJCFParser(PANDA_SCENE_PATH).parse()
    with panda_world.modify_world():
        for body in [
            body
            for body in panda_world.bodies
            if body.name.name in PANDA_BODIES_TO_DISCARD
        ]:
            panda_world.remove_kinematic_structure_entity(body)
        panda_world.root.name = PrefixedName("panda_mount", NODE_NAME)
        # An actuator parsed into one world cannot be merged into another, and this demo
        # installs its own servos below anyway.
        for actuator in list(panda_world.actuators):
            panda_world.remove_actuator(actuator)
    return panda_world


def build_scene() -> PandaMontessoriScene:
    """
    Build the whole scene: floor, tabletop, arm and one loose cylinder.

    :return: The scene, before any simulation is started.
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

        table_top = _box_body("table_top", TABLE_TOP_SCALE, Color.BEIGE())
        world.add_connection(
            FixedConnection(
                parent=root,
                child=table_top,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=TABLE_TOP_POSITION.x,
                    y=TABLE_TOP_POSITION.y,
                    z=TABLE_TOP_POSITION.z,
                ),
            )
        )

        shape_geometry = Cylinder(
            width=SHAPE_RADIUS * 2, height=SHAPE_HEIGHT, color=Color.BLUE()
        )
        shape_geometry.simulator_additional_properties.append(
            MujocoGeom(friction=list(GRASP_FRICTION))
        )
        shape = Body.from_shape_collection(
            PrefixedName("shape", NODE_NAME), ShapeCollection([shape_geometry])
        )
        shape_connection = Connection6DoF.create_with_dofs(
            world=world, parent=root, child=shape
        )
        world.add_connection(shape_connection)

    # Set after the connection is in the world so the pose lands in the free joint's own
    # dof values, which is what MuJoCo reads as its starting pose; passed to
    # create_with_dofs instead it would be a fixed offset and the body would start at the
    # origin.
    shape_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=SHAPE_POSITION.x, y=SHAPE_POSITION.y, z=SHAPE_POSITION.z
    )

    panda_world = _parse_panda()
    with world.modify_world():
        world.merge_world(
            panda_world,
            FixedConnection(
                parent=root,
                child=panda_world.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=PANDA_MOUNT_POSITION.x,
                    y=PANDA_MOUNT_POSITION.y,
                    z=PANDA_MOUNT_POSITION.z,
                    yaw=PANDA_MOUNT_YAW,
                ),
            ),
        )
    return PandaMontessoriScene(world=world, robot=Panda.from_world(world), shape=shape)


def _pick_up_shape(scene: PandaMontessoriScene) -> None:
    """
    Park the arm, then grasp the cylinder top-down and lift it.

    :param scene: The built scene, already equipped and simulating.
    :raises PlanFailure: If the pick-up does not complete.
    """
    from coraplex.datastructures.dataclasses import Context
    from coraplex.datastructures.enums import (
        ApproachDirection,
        Arms,
        ExecutionType,
        VerticalAlignment,
    )
    from coraplex.datastructures.grasp import GraspDescription
    from coraplex.execution_environment import ExecutionEnvironment
    from coraplex.plans.factories import sequential
    from coraplex.robot_plans.actions.core.pick_up import PickUpAction
    from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

    context = Context(scene.world, scene.robot)
    arm = scene.robot.get_arms()[0]
    plan = sequential(
        [
            ParkArmsAction(Arms.RIGHT),
            PickUpAction(
                scene.shape,
                Arms.RIGHT,
                GraspDescription(
                    ApproachDirection.BACK,
                    VerticalAlignment.NoAlignment,
                    arm.end_effector,
                ),
            ),
        ],
        context=context,
    )
    with ExecutionEnvironment(
        execution_type=ExecutionType.SIMULATED,
        collision_avoidance=False,
        real_time_pacing=True,
    ):
        plan.perform()


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments selecting whether a MuJoCo viewer window is opened.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a MuJoCo viewer window; off by default so the demo runs headless.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Build the scene, simulate it, and try to pick the cylinder up.
    """
    # force: the CRAM/Giskard stack configures the root logger on import, which would
    # otherwise swallow this script's own reporting.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    arguments = _parse_arguments()

    if not rclpy_installed():
        logger.error("rclpy is not installed; this needs the CRAM/Giskard stack.")
        return

    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from coraplex.plans.failures import PlanFailure
    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    spinner.start()

    scene = build_scene()
    physically_simulated_dofs = _equip_robot_for_physical_simulation(
        scene.robot, actuator_tuning=PANDA_ACTUATOR_TUNING
    )
    tf_publisher = TFPublisher(node=node, _world=scene.world)
    viz_marker_publisher = VizMarkerPublisher(_world=scene.world, node=node)

    simulation = MujocoSim(
        world=scene.world,
        headless=False,
        step_size=SIMULATION_STEP_SIZE,
        physically_simulated_dofs=physically_simulated_dofs,
        sync_rate_hz=SYNC_RATE_HZ,
        reparenting_mode=ReparentingMode.CONTACT_ONLY,
    )
    simulation.start_simulation()
    try:
        height_before = float(scene.shape.global_transform.to_position().z)
        try:
            _pick_up_shape(scene)
        except PlanFailure as failure:
            logger.error("pick-up did not finish: %s", failure)
        scene.world.update_forward_kinematics()
        height_after = float(scene.shape.global_transform.to_position().z)
        logger.info(
            "shape z %.4f -> %.4f (lifted: %s)",
            height_before,
            height_after,
            height_after > height_before + SHAPE_HEIGHT / 2,
        )
        logger.info("Done. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        simulation.stop_simulation()
        viz_marker_publisher.stop()
        tf_publisher.stop()
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
