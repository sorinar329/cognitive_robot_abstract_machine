"""
The Montessori pick-and-place of :mod:`experiments.montessori.montessori_demo_mujoco`,
run by a table-mounted Franka Emika Panda instead of a mobile HSR.

Why a second robot at all: the HSR's arm carries no lateral degree of freedom, so its
:attr:`~semantic_digital_twin.robots.robot_part_mixins.HasMobileBase.full_body_controlled`
base is part of every reach -- the motion solver drives the base as it reaches, and a
grasp that misses is never cleanly a grasp problem or a driving problem. The Panda is
bolted down and reaches with its arm alone
(:meth:`~experiments.montessori.world.MontessoriWorld.mount_stationary_robot`), so a
failure here is about the arm and the grasp, and nothing else.

Mounted at :data:`PANDA_MOUNT_POSITION`, every loose shape sits 0.20-0.49 m from its
base and the board 0.45 m, comfortably inside the arm's reach, so one mount covers the
whole task.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.montessori.montessori_demo_panda
    python -m experiments.montessori.montessori_demo_panda --viewer

.. warning::
    The Panda ships no ROS package here -- its
    :meth:`~semantic_digital_twin.robots.panda.Panda.get_ros_file_path` raises -- so its
    description is read from the MJCF scene at :data:`PANDA_SCENE_PATH`, an absolute
    path outside this repository. Point it at your own copy before running.

.. note::
    The Panda has no self-collision matrix
    (:meth:`~semantic_digital_twin.robots.panda.Panda._setup_collision_rules` is empty),
    so unlike the HSR none of its neighbouring links are exported to MuJoCo as contact
    exclusions. Watch for the arm shoving itself apart on the first settle.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import numpy as np

from experiments.montessori.montessori_demo import (
    JointServoTuning,
    RobotActuatorTuning,
    _equip_robot_for_physical_simulation,
    _start_physical_simulation,
)
from experiments.montessori.semantics import MontessoriShape, NoMatchingHoleError
from experiments.montessori.world import (
    TABLE_POSITION,
    TABLE_SCALE,
    MontessoriWorld,
)
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.utils import rclpy_installed
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

NODE_NAME = "montessori_demo_panda"
"""
Name of the ROS 2 node the visualization publishes from.
"""

PANDA_SCENE_PATH = (
    "/home/sorin/dev/manipulation_experiments/resources/generated/stacking_scene.xml"
)
"""
MJCF scene the Panda's description is read from.

Absolute and outside this repository because no ROS package for the Panda ships with it;
see this module's warning.
"""

PANDA_SCENE_BODIES_TO_DISCARD = frozenset(
    {"target", "cube0", "cube1", "cube2", "cube3", "floor", "stack_pad"}
)
"""
Bodies of :data:`PANDA_SCENE_PATH` that belong to its own stacking task rather than to
the robot, dropped so only the arm is merged into the Montessori scene.
"""

PANDA_MOUNT_POSITION = Point3(0.25, 0.0, float(TABLE_POSITION.z) + TABLE_SCALE.z / 2)
"""
Where the Panda's base is bolted: stood off the table's near edge, at table height.

Set back far enough that nothing is too *close* to reach, not just close enough that
nothing is too far. Mounted at the table edge the nearest shape sits 0.20 m out, inside
the arm's own shoulder, and ``PickUpAction`` refuses it on ``IsObjectReachableBy``. From
here every shape falls between 0.40 m and 0.60 m and the board at 0.65 m, clear of both
ends of the workspace.
"""

PANDA_MOUNT_YAW = np.pi
"""
Which way the mounted Panda faces: towards the shape row and the board, both of which lie
at lower ``x`` than :data:`PANDA_MOUNT_POSITION`.

A bolted arm cannot turn, so facing the other way leaves it reaching back over itself for
every grasp -- inside its reach, but in the least dexterous part of its workspace.
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
Gains and force clamps for the Panda's joints, read off the actuators its own MJCF
declares rather than reused from the HSR.

The default covers joints 5-7, which the model drives at 2000/200 within +-12 Nm; the
shoulder and elbow carry far more and are named individually. Driven at the HSR's uniform
1000/100 within +-100 Nm instead, the arm oscillates around a pose it is merely holding:
measured 2.8 cm peak-to-peak at the tool, with per-joint tracking errors up to 0.05 rad.

.. note::
    The finger's entry is the stiffness of the model's own gripper actuator, whose
    ``ctrl`` is in 0-255 units; the servo built here commands the joint in metres, so only
    its gains carry over, not its control range.
"""

INSERTION_HOVER_HEIGHT = 0.03
"""
Height above the target hole at which a shape is released, so the gripper clears the
board's surface before letting go and the shape drops the last centimetres on its own.
"""


def _parse_panda() -> World:
    """
    Read the Panda out of :data:`PANDA_SCENE_PATH`, without the stacking task it shares
    that scene with.

    The scene's root is renamed, since a body called ``world`` would collide with the
    root the simulator's own scene builder creates.

    :return: A world holding only the arm.
    :raises FileNotFoundError: If the scene is not where :data:`PANDA_SCENE_PATH` says.
    """
    panda_world = MJCFParser(PANDA_SCENE_PATH).parse()
    with panda_world.modify_world():
        for body in [
            body
            for body in panda_world.bodies
            if body.name.name in PANDA_SCENE_BODIES_TO_DISCARD
        ]:
            panda_world.remove_kinematic_structure_entity(body)
        panda_world.root.name = PrefixedName("panda_mount", "montessori")
    return panda_world


def _pick_and_place_shape(montessori: MontessoriWorld, shape: MontessoriShape) -> bool:
    """
    Grasp one shape top-down, carry it over its matching hole and release it.

    Mirrors :func:`~experiments.montessori.montessori_demo_mujoco._pick_and_place_shape`
    so the two robots are compared on the same plan, minus everything that plan says
    about a base: a bolted arm has no stance to resolve and no base to keep out of the
    table.

    :param montessori: The Montessori scene, already equipped and simulating.
    :param shape: The shape to insert; must have a matching hole.
    :return: Whether the shape ended up below the board's top surface.
    :raises PlanFailure: If the pick-up or placing does not complete.
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
    from coraplex.robot_plans.actions.core.placing import PlaceAction
    from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

    hole = montessori.board.hole_for(shape)
    insertion_pose = shape.insertion_pose_relative_to_hole(
        hole, Point3(0.0, 0.0, 0.0), INSERTION_HOVER_HEIGHT
    )
    drop_location = montessori.world.transform(insertion_pose, montessori.world.root)
    height_before = float(shape.root.global_transform.to_position().z)

    context = Context(montessori.world, montessori.robot)
    arm = context.robot.get_arms()[0]
    plan = sequential(
        [
            ParkArmsAction(Arms.RIGHT),
            PickUpAction(
                shape.root,
                Arms.RIGHT,
                GraspDescription(
                    ApproachDirection.BACK,
                    VerticalAlignment.TOP,
                    arm.end_effector,
                ),
            ),
            PlaceAction(shape.root, drop_location, Arms.RIGHT),
            ParkArmsAction(Arms.RIGHT),
        ],
        context=context,
    )
    with ExecutionEnvironment(
        execution_type=ExecutionType.SIMULATED,
        collision_avoidance=True,
        real_time_pacing=True,
    ):
        plan.perform()

    montessori.world.update_forward_kinematics()
    height_after = float(shape.root.global_transform.to_position().z)
    board_top_z = (
        montessori.board.root.collision.as_bounding_box_collection_in_frame(
            montessori.world.root
        )
        .bounding_box()
        .max_z
    )
    fell_through = height_after < board_top_z
    logger.info(
        "%s: z %.4f -> %.4f, board top at %.4f, fell through the hole: %s",
        shape.name.name,
        height_before,
        height_after,
        board_top_z,
        fell_through,
    )
    return fell_through


def _run_for_shape(node, shape_name: str, headless: bool) -> bool:
    """
    One self-contained run: build a fresh scene, bolt the Panda down, start its own
    physics simulation, and pick-and-place the named shape into its hole.

    :param node: The ROS 2 node the visualization publishes from.
    :param shape_name: Name of the shape this run inserts.
    :param headless: Whether to run without a MuJoCo viewer window.
    :return: Whether the shape fell through its hole, or ``False`` if the plan failed.
    """
    from coraplex.plans.failures import PlanFailure
    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    # A collidable floor: this run plans no navigation, so it does not pay the costmap
    # cost the visual-only default avoids, and a shape knocked off the table has to land
    # somewhere rather than fall for as long as the run lasts.
    montessori = MontessoriWorld(floor_is_collidable=True)
    [shape] = [
        candidate
        for candidate in montessori.world.get_semantic_annotations_by_type(
            MontessoriShape
        )
        if candidate.name.name == shape_name
    ]
    # The stand carries no part of the plan: a bolted arm is held where it is put either
    # way. It is here so the scene shows the arm standing on something rather than
    # floating at table height.
    montessori.add_robot_stand(PANDA_MOUNT_POSITION)
    montessori.mount_stationary_robot(
        Panda, _parse_panda(), PANDA_MOUNT_POSITION, mount_yaw=PANDA_MOUNT_YAW
    )
    physically_simulated_dofs = _equip_robot_for_physical_simulation(
        montessori.robot, actuator_tuning=PANDA_ACTUATOR_TUNING
    )

    tf_publisher = TFPublisher(node=node, _world=montessori.world)
    viz_marker_publisher = VizMarkerPublisher(_world=montessori.world, node=node)

    multi_sim = _start_physical_simulation(
        montessori, physically_simulated_dofs, headless=False
    )
    try:
        return _pick_and_place_shape(montessori, shape)
    except PlanFailure as failure:
        logger.error("%s: pick-and-place did not finish: %s", shape_name, failure)
        return False
    finally:
        multi_sim.stop_simulation()
        viz_marker_publisher.stop()
        tf_publisher.stop()


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
    Insert every shape that has a matching hole, one run each, and report the outcome.
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

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    spinner.start()

    probe = MontessoriWorld()
    insertable_shape_names = []
    for shape in probe.world.get_semantic_annotations_by_type(MontessoriShape):
        try:
            probe.board.hole_for(shape)
        except NoMatchingHoleError:
            continue
        insertable_shape_names.append(shape.name.name)

    try:
        results = {}
        for shape_name in insertable_shape_names:
            logger.info("=== Run: inserting %s ===", shape_name)
            results[shape_name] = _run_for_shape(
                node, shape_name, headless=not arguments.viewer
            )
        logger.info("=== Summary ===")
        for shape_name, fell_through in results.items():
            logger.info("  %-26s inserted: %s", shape_name, fell_through)
        logger.info("Done. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
