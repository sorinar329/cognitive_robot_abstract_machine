"""
Physical Montessori scene, built up one step at a time: the same world and robot as
:mod:`experiments.montessori.montessori_demo` inside one long-running MuJoCo simulation,
with the robot first parking its arms and then running a basic coraplex pick-and-place
plan for the red cube -- grasp it, carry it over its square hole, and let it drop
through -- in the style of :mod:`coraplex.demos.coraplex_panda_demo.demo`.

Parking comes first deliberately. It exercises the whole physical control loop on its
own -- Giskard commanding joint positions, the world→sim sync driving MuJoCo's
actuators, and the sim→world sync reporting back where the joints physically ended up --
so a joint that cannot reach its commanded position shows up as itself rather than
buried in a grasp failure. Only then does a shape get involved.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.montessori.montessori_demo_mujoco
    python -m experiments.montessori.montessori_demo_mujoco --headless

.. note::
    Needs ROS 2 (``rclpy``) and the robot's own description package, since parking the
    arms goes through the CRAM/Giskard motion stack; see
    :mod:`experiments.montessori.montessori_demo` for the same requirements.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

from typing_extensions import Dict

from experiments.montessori.montessori_demo import (
    DEFAULT_ROBOT_CLASS,
    _enable_robot_table_collision_avoidance,
    _equip_robot_for_physical_simulation,
    _start_physical_simulation,
)
from experiments.montessori.semantics import MontessoriShape
from experiments.montessori.world import MontessoriWorld, robot_installed
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.utils import rclpy_installed
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

logger = logging.getLogger(__name__)

NODE_NAME = "montessori_demo_mujoco"
"""
Name of the ROS 2 node the visualization publishes from.
"""

SETTLE_TIMEOUT = 15.0
"""
Longest the scene is given to come to rest before its joint positions are reported, so a
robot that never settles is reported rather than waited on forever.
"""

SETTLE_SAMPLE_INTERVAL = 0.25
"""
Real-time seconds between the samples :func:`_wait_until_settled` compares.
"""

SETTLE_TOLERANCE = 1e-3
"""
Movement between two samples, in radians or metres, below which a joint counts as at
rest.
"""


def _joint_positions(
    multi_sim: MujocoSim, degrees_of_freedom: set[DegreeOfFreedom]
) -> Dict[str, float]:
    """
    The physically simulated position of every given degree of freedom.

    :param multi_sim: The running simulation.
    :param degrees_of_freedom: The degrees of freedom to read.
    :return: Position by joint name.
    """
    return {
        dof.name.name: float(multi_sim.simulator.get_joint_value(dof.name.name).result)
        for dof in degrees_of_freedom
    }


def _commanded_positions(
    multi_sim: MujocoSim, degrees_of_freedom: set[DegreeOfFreedom]
) -> Dict[str, float]:
    """
    The position each degree of freedom is actually being driven towards, read from its
    actuator's setpoint.

    Read from the actuator rather than from ``world.state``: the sim→world sync
    overwrites the world model's belief with the measured position, so a belief read
    back here describes where the joint *is*, not where it was told to go, and a joint
    failing to track its command would look like it were tracking perfectly.

    :param multi_sim: The running simulation.
    :param degrees_of_freedom: The degrees of freedom to read.
    :return: Commanded position by joint name, omitting dofs driven by no actuator.
    """
    synchronizer = multi_sim.synchronizer
    commanded = {}
    for dof in degrees_of_freedom:
        connection = multi_sim.synchronizer._world.get_connection_by_name(dof.name.name)
        actuator = synchronizer._resolve_actuator(connection)
        control_address = synchronizer._resolve_ctrl_adr(connection)
        if actuator is None or control_address is None:
            continue
        commanded[dof.name.name] = float(
            multi_sim.simulator._mj_data.ctrl[control_address]
        )
    return commanded


def _wait_until_settled(
    multi_sim: MujocoSim, degrees_of_freedom: set[DegreeOfFreedom]
) -> None:
    """
    Block until the physically simulated joints stop moving, or until
    :data:`SETTLE_TIMEOUT` elapses.

    A fixed wait cannot tell a joint that has finished tracking its command from one
    still on its way, which is exactly the distinction this script exists to report.

    :param multi_sim: The running simulation.
    :param degrees_of_freedom: The degrees of freedom to watch.
    """
    deadline = time.time() + SETTLE_TIMEOUT
    previous = _joint_positions(multi_sim, degrees_of_freedom)
    while time.time() < deadline:
        time.sleep(SETTLE_SAMPLE_INTERVAL)
        current = _joint_positions(multi_sim, degrees_of_freedom)
        if all(
            abs(current[name] - previous[name]) < SETTLE_TOLERANCE for name in current
        ):
            return
        previous = current
    logger.warning("Joints were still moving after %.1fs.", SETTLE_TIMEOUT)


def _log_joint_positions(
    label: str, commanded: Dict[str, float], measured: Dict[str, float]
) -> None:
    """
    Report what each joint was commanded to versus where it physically is, so a joint
    that cannot follow its command stands out.

    A joint driven by no actuator is reported without a command: it is moved by physics
    alone (on the HSR the gripper's compliant spring joints), so there is no target for
    it to be failing to reach and an error column would be meaningless.

    :param label: Heading for this report.
    :param commanded: Commanded position by joint name, absent for unactuated joints.
    :param measured: Physically simulated position by joint name.
    """
    logger.info("--- %s ---", label)
    logger.info("%-32s %12s %12s %10s", "joint", "commanded", "measured", "error")
    for name in sorted(measured):
        if name not in commanded:
            logger.info(
                "%-32s %12s %12.5f %10s", name, "unactuated", measured[name], "-"
            )
            continue
        logger.info(
            "%-32s %12.5f %12.5f %10.5f",
            name,
            commanded[name],
            measured[name],
            measured[name] - commanded[name],
        )


def _park_arms(robot: AbstractRobot) -> None:
    """
    Park the robot's arms, executed against the running simulation.

    :param robot: The spawned robot.
    :raises MotionDidNotFinish: If a commanded motion never converged, e.g. because a
        joint cannot physically reach the position it was commanded to.
    """
    # Imported lazily: this chain pulls in rclpy at module level (see
    # experiments.montessori.montessori_demo._insert_all_shapes for the same reasoning).
    from coraplex.datastructures.dataclasses import Context
    from coraplex.datastructures.enums import Arms, ExecutionType
    from coraplex.execution_environment import ExecutionEnvironment
    from coraplex.plans.factories import sequential
    from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

    context = Context(robot._world, robot)
    # real_time_pacing keeps the control loop advancing at the rate of the physics it is
    # driving; without it the planner runs far ahead of the simulation and commands
    # positions the arm has had no wall-clock time to physically reach.
    with ExecutionEnvironment(
        execution_type=ExecutionType.SIMULATED,
        collision_avoidance=True,
        real_time_pacing=True,
    ):
        sequential([ParkArmsAction(Arms.BOTH)], context=context).perform()


INSERTION_HOVER_HEIGHT = 0.03
"""
Height above the target hole at which a shape is released, so the gripper clears the
board's surface before letting go and the shape then drops the last few centimetres
through the hole under its own weight.
"""


def _pick_and_place_shape(montessori: MontessoriWorld, shape: MontessoriShape) -> bool:
    """
    Run a basic coraplex pick-and-place plan for one shape, in the style of
    :mod:`coraplex.demos.coraplex_panda_demo.demo`: park, grasp the shape from the front
    with a top-down alignment, carry it over its matching hole, and release it so it
    falls through.

    No explicit navigation, but the base still moves: the robot's mobile base is
    :attr:`~semantic_digital_twin.robots.robot_part_mixins.HasMobileBase.full_body_controlled`,
    so each reach is solved from the world root through the drive's x/y/yaw and the base
    is driven toward the target as part of the reach itself. The robot is spawned facing
    this shape (see :func:`main`) so that motion stays small. The release only brings the
    shape just above the hole; whether it drops through is left to physics.

    Collision avoidance is on, narrowed to the robot against the table (see
    :func:`~experiments.montessori.montessori_demo._enable_robot_table_collision_avoidance`).
    Without it the whole-body solution drives the base straight through the table's near
    edge: the base's planar joints are unactuated and its pose is written into the
    simulator every sync, so that overlap comes back as contact impulses violent enough
    to throw the loose shapes off the table before the reach arrives.

    Narrowed rather than global because the gripper has to touch the shape it is grasping,
    and because checking the board's ~40-50-piece decomposition overloads the QP solver.

    The grasp approaches from :attr:`~coraplex.datastructures.enums.ApproachDirection.BACK`,
    not ``FRONT``. Under :attr:`~coraplex.datastructures.enums.VerticalAlignment.TOP` the
    approach is top-down whichever direction is named -- the direction only picks the
    gripper's yaw, expressed in the shape's own (world-aligned) frame. ``FRONT`` asks for
    yaw 0 while the base faces yaw pi, so the hand has to point back over the robot and
    the solver turns the whole base to help; ``BACK`` asks for the yaw the base already
    faces, and the base stays put.

    .. note::
        Fine for a shape with a symmetric footprint. An orientation-sensitive one (the
        triangle) is held a half-turn from where ``FRONT`` would hold it, which carries
        through to the yaw it is released at over its hole.

    :param montessori: The Montessori scene, already equipped and simulating.
    :param shape: The shape to insert; must have a matching hole.
    :return: Whether the shape ended up below the board's top surface, i.e. fell through.
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
            ParkArmsAction(Arms.RIGHT),
            PlaceAction(shape.root, drop_location, Arms.RIGHT),
            ParkArmsAction(Arms.RIGHT),
        ],
        context=context,
    )
    with ExecutionEnvironment(
        execution_type=ExecutionType.SIMULATED,
        collision_avoidance=False,
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


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments selecting whether a MuJoCo viewer window is opened.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help=(
            "Open a MuJoCo viewer window. Off by default so the demo runs on a headless "
            "or remote (e.g. VNC) machine without a usable GL context; the scene is "
            "visible in RViz regardless."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Build the Montessori scene with a physically simulated robot, park its arms, and
    keep the simulation running until interrupted.
    """
    # force: the CRAM/Giskard stack configures the root logger on import, which would
    # otherwise swallow this script's own reporting.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    arguments = _parse_arguments()

    if not robot_installed(DEFAULT_ROBOT_CLASS):
        logger.error(
            "%s's description is not installed; nothing to park.",
            DEFAULT_ROBOT_CLASS.__name__,
        )
        return
    if not rclpy_installed():
        logger.error("rclpy is not installed; parking needs the CRAM/Giskard stack.")
        return

    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from experiments.montessori.semantics import NoMatchingHoleError

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    spinner.start()

    # The names of every shape that has a matching hole (all but the sphere), read once
    # from a throwaway scene. Each is inserted in its own run below.
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


def _run_for_shape(node, shape_name: str, headless: bool) -> bool:
    """
    One self-contained run: build a fresh scene, spawn the robot facing ``shape_name``,
    start its own physics simulation, park, and pick-and-place that one shape into its
    hole.

    Each run gets its own world and simulation rather than accumulating shapes across
    runs, so the robot only ever has to reach the single shape it is spawned in front of
    -- no driving between them, which a fixed base cannot do.

    :param node: The ROS 2 node the visualization publishes from.
    :param shape_name: Name of the shape this run inserts.
    :param headless: Whether to run this run's MuJoCo simulation without a viewer window.
    :return: Whether the shape fell through its hole, or ``False`` if the plan failed.
    """
    from coraplex.plans.failures import PlanFailure
    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    # A collidable floor: this run plans no navigation, so it does not pay the costmap
    # cost the visual-only default avoids, and a shape knocked off the table has to land
    # somewhere. Left falling, it accelerates away for as long as the run lasts and drags
    # the reach goal tracking its pose along with it.
    montessori = MontessoriWorld(floor_is_collidable=True)
    [shape] = [
        candidate
        for candidate in montessori.world.get_semantic_annotations_by_type(
            MontessoriShape
        )
        if candidate.name.name == shape_name
    ]
    # Line the *arm* up with the shape, not the base's origin: the arm sits to one side of
    # that origin, so a base centred on the shape leaves the tool frame beside it and the
    # grasp needs base motion to recover ground the stance could have given it for free.
    montessori.world.update_forward_kinematics()
    shape_y = float(shape.root.global_transform.to_position().y)
    montessori.spawn_robot(DEFAULT_ROBOT_CLASS, arm_aligned_with_y=shape_y)
    # Narrows what the plans below check, so their collision_avoidance=True keeps the
    # base out of the table without asking the QP solver to check the whole scene.
    _enable_robot_table_collision_avoidance(montessori)
    physically_simulated_dofs = _equip_robot_for_physical_simulation(montessori.robot)

    # Attached after the robot is spawned so the markers cover it too, and before the
    # simulation starts so every pose the physics produces is published as it happens.
    tf_publisher = TFPublisher(node=node, _world=montessori.world)
    viz_marker_publisher = VizMarkerPublisher(_world=montessori.world, node=node)

    multi_sim = _start_physical_simulation(
        montessori, physically_simulated_dofs, headless=False
    )
    try:
        _wait_until_settled(multi_sim, physically_simulated_dofs)
        try:
            _park_arms(montessori.robot)
        except PlanFailure as failure:
            logger.error("%s: parking did not finish: %s", shape_name, failure)

        try:
            return _pick_and_place_shape(montessori, shape)
        except PlanFailure as failure:
            logger.error("%s: pick-and-place did not finish: %s", shape_name, failure)
            return False
    finally:
        multi_sim.stop_simulation()
        viz_marker_publisher.stop()
        tf_publisher.stop()


if __name__ == "__main__":
    main()
