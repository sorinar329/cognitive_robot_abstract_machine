"""
Build the Montessori shape-sorting world and have an iCub3, standing fixed on the floor
beside the table, sort every loose shape into its matching hole -- the humanoid
counterpart to :mod:`experiments.montessori.franka_montessori_demo`'s table-mounted
Panda, reaching with one arm alone (see
:meth:`~experiments.montessori.world.MontessoriWorld.mount_stationary_robot`) rather
than navigating to it. Mounted as
:class:`~semantic_digital_twin.robots.icub3.ICub3FixedBase`, not
:class:`~semantic_digital_twin.robots.icub3.ICub3`: the latter's mobile base makes
action planning navigate to a resolved standing offset before every reach, which never
converges for a base that is bolted down rather than actually drivable (see
:class:`~semantic_digital_twin.robots.icub3.ICub3FixedBase`'s own docstring), the same
way the Panda's own fixed base has nothing to navigate with.

Run with (the ``experiments`` package must be importable, and the ``iai_icub_description``
ROS package built and sourced -- see :func:`~experiments.montessori.world.robot_installed`)::

    python -m experiments.montessori.icub_montessori_demo
    python -m experiments.montessori.icub_montessori_demo --viewer
    python -m experiments.montessori.icub_montessori_demo --iterations 100

Every run's per-shape results are collected, one :class:`~experiments.montessori.sorting_results.SortingIterationResult` (with
its :class:`~experiments.montessori.sorting_results.ShapeInsertionResult` rows) per
iteration, and logged as a per-shape success-rate summary once
:attr:`~argparse.Namespace.iterations` finish (see :func:`_log_iteration_summary`); not
persisted anywhere.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import logging
import math
import threading
import time
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import mujoco
from typing_extensions import Optional

from experiments.montessori.event_monitoring import (
    MontessoriEventMonitor,
    build_shape_monitor,
)
from experiments.montessori.franka_panda_equipment import (
    BOARD_FRICTION,
    apply_contact_friction,
    apply_montessori_grasp_contact_parameters,
)
from experiments.montessori.icub_equipment import (
    equip_icub_for_physical_simulation,
    parse_icub,
)
from experiments.montessori.semantics import (
    MontessoriShape,
    MontessoriShapeCategory,
    NoMatchingHoleError,
)
from experiments.montessori.sorting_results import (
    InsertionOutcome,
    ShapeInsertionResult,
    SortingIterationResult,
)
from experiments.montessori.world import FLOOR_Z, MontessoriWorld
from segmind.datastructures.events import InsertionEvent, PickUpEvent
from semantic_digital_twin.robots.icub3 import ICub3FixedBase
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.utils import rclpy_installed

if TYPE_CHECKING:
    # coraplex.datastructures.dataclasses and the ROS adapters below all pull in
    # rclpy at module level (see main), so these are only ever imported for type
    # hints, never at runtime.
    from semantic_digital_twin.adapters.multi_sim import MujocoSim
    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
    from experiments.montessori.insert_shape_action import InsertMontessoriShapeAction

logger = logging.getLogger(__name__)

NODE_NAME = "icub_montessori_demo"
"""
Name of the ROS 2 node this demo's visualization runs against.
"""

MOUNT_STANDOFF_DISTANCE = 0.15
"""
How far past the montessori table's near edge (the short edge nearest the loose-shape
row) the iCub3 stands.

Not :data:`~experiments.montessori.franka_montessori_demo.MOUNT_STANDOFF_DISTANCE`'s own
0.35m: that value is tuned for the Panda, which is bolted directly at table height (see
:func:`~experiments.montessori.franka_montessori_demo._mount_position`) and so only ever
has to reach horizontally. The iCub3 instead stands on the floor (see
:func:`_mount_position`) with its shoulders roughly half a meter above this low table's
own top, so every reach also has to cover that vertical drop on top of the horizontal
standoff -- at 0.35m, every insertion's Cartesian goal came back QP-infeasible (out of
reach) in practice; halving the standoff (with the horizontal distance to the board and
shape row shrinking by the same amount as the robot itself moves closer) brought it back
within reach. Still only a first-pass estimate, not verified against every shape's own
reach.
"""

MOUNT_YAW = math.pi
"""
Which way the iCub3 is turned to face once bolted (see :func:`_mount_position`),
matching the Panda's own ``mount_yaw=np.pi`` at the same table edge (see
:func:`~experiments.montessori.franka_montessori_demo._build_world_and_sort`): standing
past the table's near ``max_x`` edge, the iCub3 has to turn a half-turn from its URDF
neutral (assumed, like the Panda's, to face ``+x``) to face back toward the table along
``-x``.
"""

MUJOCO_STEP_SIZE = 1e-4
"""
Physics step size, matching :data:`~experiments.montessori.franka_montessori_demo.MUJOCO_STEP_SIZE`.

No iCub3-specific tuning exists yet; kept at the Panda's own value since
:data:`~experiments.montessori.icub_equipment.ARM_JOINT_SERVO_TUNING`'s gains are of the
same order of magnitude and a coarser step was, for the Panda, observed to make a
high-gain servo shake rather than hold still (see that constant's own docstring).
"""

MUJOCO_INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
"""
Numerical integrator MuJoCo advances the physics with, matching
:data:`~experiments.montessori.franka_montessori_demo.MUJOCO_INTEGRATOR`.
"""

SYNC_RATE_HZ = 100
"""
Rate at which the physically simulated joints' real, physics-driven positions are read
back into the world model, matching
:data:`~experiments.montessori.franka_montessori_demo.SYNC_RATE_HZ`'s own reasoning.
"""

SKIPPED_SHAPE_CATEGORIES = frozenset({MontessoriShapeCategory.DISK})
"""
Shape categories the demo leaves where they are, matching
:data:`~experiments.montessori.franka_montessori_demo.SKIPPED_SHAPE_CATEGORIES`.
"""

MAX_INSERTION_ATTEMPTS = 3
"""
Number of times a single shape's insertion is repeated while the attempt never gets as
far as releasing the shape, before giving up on it and logging a warning.
"""

SHAPE_SETTLE_DURATION = 2.0
"""
Real-time seconds a just-released shape is given to physically fall and come to rest
before it is checked whether it made it through its hole.
"""

MINIMUM_PICKUP_DISPLACEMENT = 0.03
"""
Minimum distance (in meters) a shape must have moved between just before its
:class:`~experiments.montessori.insert_shape_action.InsertMontessoriShapeAction` starts
and right after it finishes, for the pickup to be considered real; see
:data:`~experiments.montessori.franka_montessori_demo.MINIMUM_PICKUP_DISPLACEMENT`'s own
docstring for why this check exists.
"""

TCP_POSITION_THRESHOLD = 0.007
"""
Position tolerance in meters used for every
:class:`~coraplex.robot_plans.motions.gripper.MoveToolCenterPointMotion` in this demo,
matching :data:`~experiments.montessori.franka_montessori_demo.TCP_POSITION_THRESHOLD`
as a first-pass value pending iCub3-specific tuning.
"""

TCP_ORIENTATION_THRESHOLD = 0.03
"""
Orientation tolerance in rad used for every
:class:`~coraplex.robot_plans.motions.gripper.MoveToolCenterPointMotion` in this demo,
matching :data:`~experiments.montessori.franka_montessori_demo.TCP_ORIENTATION_THRESHOLD`.
"""


def _mount_position(montessori: MontessoriWorld) -> Point3:
    """
    Where to bolt the iCub3: past the table's near edge (the short edge nearest the
    loose-shape row), centered on the table's long axis, standing on the floor -- the
    same ``max_x``-edge site :func:`~experiments.montessori.franka_montessori_demo._mount_position`
    bolts the Panda to (see :data:`MOUNT_STANDOFF_DISTANCE`), except at floor height
    (:data:`~experiments.montessori.world.FLOOR_Z`) rather than table height, since the
    iCub3 has legs to stand on the floor with instead of being bolted directly at the
    table's own height.

    :param montessori: The Montessori scene the iCub3 is being mounted next to.
    """
    table_bounding_box = (
        montessori.world.get_body_by_name("table")
        .collision.as_bounding_box_collection_in_frame(montessori.world.root)
        .bounding_box()
    )
    return Point3(
        table_bounding_box.max_x + MOUNT_STANDOFF_DISTANCE,
        0.0,
        FLOOR_Z,
    )


def _build_insert_action(
    shape: MontessoriShape,
    montessori: MontessoriWorld,
    target_horizontal_offset: Optional[Point3] = None,
) -> InsertMontessoriShapeAction:
    """
    Build (without executing) the plan that inserts ``shape`` into its matching hole.

    Unlike :func:`~experiments.montessori.franka_montessori_demo._build_insert_action`,
    no explicit :class:`~coraplex.datastructures.grasp.GraspDescription` is built: that
    function's ``rotate_gripper=True`` works around a Panda-specific wrist-resolution
    quirk (see its own docstring), which is not known to apply to the iCub3's hand, so
    the action's own default grasp resolution is used instead, matching
    :mod:`experiments.montessori.montessori_demo`'s own HSRB call site.

    :param shape: The shape to insert; must have a matching hole.
    :param montessori: The Montessori scene, with the iCub3 already mounted and
        equipped (see :func:`~experiments.montessori.icub_equipment.equip_icub_for_physical_simulation`),
        inside a running simulation.
    :param target_horizontal_offset: Horizontal offset to release the shape at; the
        hole's exact center is used if not given.
    """
    from coraplex.datastructures.enums import Arms
    from experiments.montessori.insert_shape_action import InsertMontessoriShapeAction

    offset = target_horizontal_offset or Point3(0.0, 0.0, 0.0)
    return InsertMontessoriShapeAction(
        montessori_shape=shape,
        board=montessori.board,
        arm=Arms.LEFT,
        target_horizontal_offset=offset,
    )


def _insert_shape(
    action: InsertMontessoriShapeAction,
    montessori: MontessoriWorld,
    context,
) -> bool:
    """
    Run ``action``, then let the shape physically settle under gravity and contacts
    before checking whether it made it through; identical in structure to
    :func:`~experiments.montessori.franka_montessori_demo._insert_shape`, see that
    function's own docstring for the reasoning behind each step.

    :param action: The insertion plan to run, built by :func:`_build_insert_action`.
    :param montessori: The Montessori scene, with the iCub3 already mounted and
        equipped, inside a running simulation.
    :param context: The CRAM execution context to run the insertion action in.
    :raises BodyUnfetchable: If the shape moved less than :data:`MINIMUM_PICKUP_DISPLACEMENT`
        over the whole insertion, i.e. the grasp silently failed to pick it up at all.
    :return: Whether the shape actually fell through its hole after settling.
    """
    from coraplex.datastructures.enums import ExecutionType
    from coraplex.execution_environment import ExecutionEnvironment
    from coraplex.plans.factories import execute_single
    from coraplex.plans.failures import BodyUnfetchable

    shape = action.montessori_shape
    spawn_position = shape.root.global_transform.to_position()
    with ExecutionEnvironment(
        execution_type=ExecutionType.SIMULATED,
        collision_avoidance=False,
        real_time_pacing=False,
        max_ticks_per_motion_mapping=300,
    ):
        node = execute_single(action, context=context)
        insertion_start_time = context.simulation_clock()
        node.perform()
        insertion_duration = context.simulation_clock() - insertion_start_time
        logger.info(
            "%s insertion action took %.3fs of simulated time.",
            shape.name,
            insertion_duration,
        )

    montessori.world.update_forward_kinematics()
    release_position = shape.root.global_transform.to_position()
    displacement = math.dist(
        (float(spawn_position.x), float(spawn_position.y), float(spawn_position.z)),
        (
            float(release_position.x),
            float(release_position.y),
            float(release_position.z),
        ),
    )
    if displacement < MINIMUM_PICKUP_DISPLACEMENT:
        raise BodyUnfetchable(body=shape.root, arm=action.arm)

    hole = montessori.board.hole_for(shape)
    hole_position = hole.root.global_transform.to_position()
    release_position = shape.root.global_transform.to_position()
    logger.info(
        "%s released at (%.4f, %.4f, %.4f); hole center at (%.4f, %.4f, %.4f).",
        shape.name,
        float(release_position.x),
        float(release_position.y),
        float(release_position.z),
        float(hole_position.x),
        float(hole_position.y),
        float(hole_position.z),
    )

    logger.info("Letting %s settle.", shape.name)
    sample_count = 10
    sample_interval = SHAPE_SETTLE_DURATION / sample_count
    for sample_index in range(sample_count):
        time.sleep(sample_interval)
        montessori.world.update_forward_kinematics()
        sample_position = shape.root.global_transform.to_position()
        logger.info(
            "%s settle sample %d/%d: (%.4f, %.4f, %.4f)",
            shape.name,
            sample_index + 1,
            sample_count,
            float(sample_position.x),
            float(sample_position.y),
            float(sample_position.z),
        )

    return action.has_fallen_through_hole()


def _insert_shape_or_none(
    shape: MontessoriShape,
    montessori: MontessoriWorld,
    context,
    attempt: int,
) -> tuple[Optional[bool], InsertMontessoriShapeAction]:
    """
    Attempt one insertion via :func:`_insert_shape`, returning ``None`` instead of
    letting a retryable failure propagate; identical in structure to
    :func:`~experiments.montessori.franka_montessori_demo._insert_shape_or_none`.

    :param shape: The shape to insert; must have a matching hole.
    :param montessori: The Montessori scene, with the iCub3 already mounted and
        equipped, inside a running simulation.
    :param context: The CRAM execution context to run the insertion action in.
    :param attempt: This attempt's 1-based index, used only for the log message.
    :return: Whether the shape fell through its hole (``None`` if this attempt failed in
        a retryable way), and the plan this attempt ran, for the caller to record
        regardless of outcome.
    """
    from coraplex.plans.failures import PlanFailure
    from giskardpy.motion_statechart.exceptions import CollisionViolatedError
    from giskardpy.qp.exceptions import QPSolverException
    from semantic_digital_twin.exceptions import PointOccupiedError

    action = _build_insert_action(shape, montessori)
    try:
        return _insert_shape(action, montessori, context), action
    except (
        PointOccupiedError,
        PlanFailure,
        CollisionViolatedError,
        QPSolverException,
    ) as error:
        logger.warning(
            "%s's insertion attempt %d/%d failed (%s); retrying.",
            shape.name,
            attempt,
            MAX_INSERTION_ATTEMPTS,
            error,
        )
        return None, action


def _log_segmind_verdict(
    shape: MontessoriShape,
    ground_truth_fell_through: Optional[bool],
    monitor: MontessoriEventMonitor,
) -> None:
    """
    Log segmind's own pick-up/insertion verdict for ``shape`` next to the ground truth
    already computed for it; identical to
    :func:`~experiments.montessori.franka_montessori_demo._log_segmind_verdict`.

    :param shape: The shape ``monitor`` was tracking.
    :param ground_truth_fell_through: What :func:`_insert_shape` determined by direct
        geometry, or ``None`` if the attempt never got far enough to check.
    :param monitor: The stopped event monitor that tracked ``shape``.
    """
    events = monitor.events
    pick_up_detected = any(
        isinstance(event, PickUpEvent) and event.tracked_object is shape.root
        for event in events
    )
    insertion_detected = any(
        isinstance(event, InsertionEvent) and event.tracked_object is shape.root
        for event in events
    )
    logger.info(
        "segmind for %s: pick-up detected=%s, insertion detected=%s "
        "(ground truth fell_through=%s).",
        shape.name,
        pick_up_detected,
        insertion_detected,
        ground_truth_fell_through,
    )


def _insert_all_shapes(
    montessori: MontessoriWorld,
    context,
    max_shapes: Optional[int] = None,
    only_shape: Optional[str] = None,
) -> list[ShapeInsertionResult]:
    """
    Have the iCub3 pick up and insert every loose shape that has a matching hole into
    the shape-sorting board; identical in structure to
    :func:`~experiments.montessori.franka_montessori_demo._insert_all_shapes`.

    :param montessori: The Montessori scene, with the iCub3 already mounted and
        equipped, inside a running simulation.
    :param context: The CRAM execution context to run every insertion action in.
    :param max_shapes: Stop after this many shapes have actually been attempted.
    :param only_shape: Attempt only the shape whose name (with the trailing ``_shape``
        removed) equals this, skipping every other shape.
    :return: One :class:`~experiments.montessori.sorting_results.ShapeInsertionResult` per actually attempted shape, in
        attempt order; a skipped shape has no entry.
    """
    results: list[ShapeInsertionResult] = []
    attempted = 0
    for shape in montessori.world.get_semantic_annotations_by_type(MontessoriShape):
        if shape.shape_category in SKIPPED_SHAPE_CATEGORIES:
            logger.info(
                "Skipping %s: %s is not sorted.", shape.name, shape.shape_category
            )
            continue

        try:
            montessori.board.hole_for(shape)
        except NoMatchingHoleError:
            logger.info("Skipping %s: no matching hole.", shape.name)
            continue

        shape_key = shape.name.name.removesuffix("_shape")
        if only_shape is not None and shape_key != only_shape:
            logger.info("Skipping %s: not %s.", shape.name, only_shape)
            continue

        if max_shapes is not None and attempted >= max_shapes:
            logger.info("Reached max_shapes=%d; stopping.", max_shapes)
            break
        attempted += 1

        event_monitor = build_shape_monitor(montessori, shape)
        event_monitor.start()

        fell_through = None
        for attempt in range(1, MAX_INSERTION_ATTEMPTS + 1):
            logger.info(
                "Inserting %s into its matching hole (attempt %d/%d).",
                shape.name,
                attempt,
                MAX_INSERTION_ATTEMPTS,
            )
            fell_through, action = _insert_shape_or_none(
                shape, montessori, context, attempt
            )
            if fell_through is not None:
                break

        event_monitor.stop()
        _log_segmind_verdict(shape, fell_through, event_monitor)

        if fell_through is None:
            logger.warning(
                "%s could not be inserted in %d attempts; moving on to the next shape.",
                shape.name,
                MAX_INSERTION_ATTEMPTS,
            )
            outcome = InsertionOutcome.ATTEMPTS_EXHAUSTED
        elif not fell_through:
            logger.warning(
                "%s did not fall through its hole; it may be resting on the board or "
                "wedged in the opening. Moving on to the next shape.",
                shape.name,
            )
            outcome = InsertionOutcome.DID_NOT_FALL_THROUGH
        else:
            outcome = InsertionOutcome.FELL_THROUGH
        results.append(
            ShapeInsertionResult(shape_key=shape_key, outcome=outcome, plan=action.plan)
        )

    return results


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments selecting whether a MuJoCo viewer window is opened and
    how many shapes to attempt.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a MuJoCo viewer window; off by default so the demo runs headless.",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help=(
            "Stop after this many shapes have been attempted, for fast iteration "
            "while tuning parameters on a single shape. Attempts every shape by "
            "default."
        ),
    )
    parser.add_argument(
        "--only-shape",
        type=str,
        default=None,
        help=(
            "Attempt only the shape with this name (trailing '_shape' removed, e.g. "
            "'square_hole'), skipping every other shape while still spawning them, for "
            "isolating one shape's own tuning. Attempts every shape by default."
        ),
    )
    parser.add_argument(
        "--no-rviz",
        action="store_true",
        help="Don't publish TF/visualization markers to RViz; publishes by default.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help=(
            "Repeat the whole build-world-and-sort cycle this many times, rebuilding "
            "the world and its simulation fresh each time, then log a per-shape "
            "success-rate summary and exit instead of idling. Runs once and keeps the "
            "simulation running afterwards (the original behavior) by default."
        ),
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help=(
            "1-based index recorded on the first iteration's SortingIterationResult, "
            "counting up from there; only the recorded index is affected, not how "
            "many iterations actually run."
        ),
    )
    parser.add_argument(
        "--exit-after-sorting",
        action="store_true",
        help=(
            "Exit as soon as sorting finishes instead of idling afterwards, even with "
            "--iterations 1."
        ),
    )
    return parser.parse_args()


def _build_world_and_sort(node, arguments: argparse.Namespace) -> tuple[
    list[ShapeInsertionResult],
    MujocoSim,
    Optional[TFPublisher],
    Optional[VizMarkerPublisher],
]:
    """
    Build a fresh Montessori world, bolt and equip the iCub3 next to it, start its
    physics simulation, and have it sort every loose shape into the board once.

    :param node: The ROS 2 node TF/marker publishing runs against.
    :param arguments: Parsed command-line arguments selecting the viewer, RViz
        publishing, and shape-attempt limits.
    :return: This run's per-shape results (see :func:`_insert_all_shapes`), and the live
        simulation and publishers, left running for the caller to stop once it is done
        with them.
    """
    from coraplex.datastructures.dataclasses import Context, MotionToleranceConfig
    from semantic_digital_twin.adapters.multi_sim import MujocoSim
    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    montessori = MontessoriWorld(shapes_are_movable=True)
    mount_position = _mount_position(montessori)
    robot = montessori.mount_stationary_robot(
        ICub3FixedBase, parse_icub(), mount_position, mount_yaw=MOUNT_YAW
    )
    physically_simulated_dofs = equip_icub_for_physical_simulation(robot)
    apply_montessori_grasp_contact_parameters(
        montessori.world.get_semantic_annotations_by_type(MontessoriShape)
    )
    apply_contact_friction([montessori.board.root], BOARD_FRICTION)
    logger.info("Built Montessori world with %d bodies.", len(montessori.world.bodies))

    tf_publisher = None
    viz_marker_publisher = None
    if not arguments.no_rviz:
        tf_publisher = TFPublisher(node=node, _world=montessori.world)
        viz_marker_publisher = VizMarkerPublisher(_world=montessori.world, node=node)
        logger.info(
            "Visualizing the Montessori world on topic '%s'.",
            viz_marker_publisher.topic_name,
        )

    multi_sim = MujocoSim(
        world=montessori.world,
        headless=not arguments.viewer,
        step_size=MUJOCO_STEP_SIZE,
        real_time_factor=None if not arguments.viewer else 1.0,
        physically_simulated_dofs=physically_simulated_dofs,
        sync_rate_hz=SYNC_RATE_HZ,
        integrator=MUJOCO_INTEGRATOR,
    )
    context = Context(
        montessori.world,
        robot,
        ros_node=node,
        update_world_model_attachment=False,
        evaluate_conditions=False,
        motion_tolerances=MotionToleranceConfig(
            default_tcp_position_threshold=TCP_POSITION_THRESHOLD,
            tool_orientation_threshold=TCP_ORIENTATION_THRESHOLD,
        ),
    )
    context.simulation_clock = lambda: multi_sim.simulator.current_simulation_time

    multi_sim.start_simulation()
    results = _insert_all_shapes(
        montessori,
        context,
        max_shapes=arguments.max_shapes,
        only_shape=arguments.only_shape,
    )
    return results, multi_sim, tf_publisher, viz_marker_publisher


def _reclaim_native_heap_fragmentation() -> None:
    """
    Collect Python cycles, then ask glibc to release freed-but-unreturned heap back to
    the OS; identical to
    :func:`~experiments.montessori.franka_montessori_demo._reclaim_native_heap_fragmentation`.
    """
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)


def _log_iteration_summary(iteration_results: list[SortingIterationResult]) -> None:
    """
    Log a per-shape success-rate summary across every
    :class:`~experiments.montessori.sorting_results.SortingIterationResult` :func:`main`
    collected, once its :attr:`~argparse.Namespace.iterations` finish.

    :param iteration_results: One entry per iteration :func:`main` ran.
    """
    tallies: dict[str, Counter[InsertionOutcome]] = defaultdict(Counter)
    for iteration_result in iteration_results:
        for shape_result in iteration_result.shape_results:
            tallies[shape_result.shape_key][shape_result.outcome] += 1

    logger.info("=== Summary across %d iteration(s) ===", len(iteration_results))
    total_fell_through = 0
    total_attempted = 0
    for shape_key in sorted(tallies):
        tally = tallies[shape_key]
        attempted = sum(tally.values())
        fell_through = tally[InsertionOutcome.FELL_THROUGH]
        total_fell_through += fell_through
        total_attempted += attempted
        logger.info(
            "%s: %d/%d fell through (%d did not, %d exhausted attempts).",
            shape_key,
            fell_through,
            attempted,
            tally[InsertionOutcome.DID_NOT_FALL_THROUGH],
            tally[InsertionOutcome.ATTEMPTS_EXHAUSTED],
        )

    if total_attempted:
        logger.info(
            "Overall: %d/%d (%.1f%%) fell through across %d iteration(s).",
            total_fell_through,
            total_attempted,
            100.0 * total_fell_through / total_attempted,
            len(iteration_results),
        )


def main() -> None:
    """
    Build the Montessori world, bolt the iCub3 next to it, visualize it in RViz, and
    have it sort the loose shapes into the board; identical in structure to
    :func:`~experiments.montessori.franka_montessori_demo.main`.
    """
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

    keep_simulation_running = (
        arguments.iterations == 1 and not arguments.exit_after_sorting
    )
    iteration_results: list[SortingIterationResult] = []
    multi_sim = None
    tf_publisher = None
    viz_marker_publisher = None
    try:
        for iteration in range(
            arguments.start_iteration,
            arguments.start_iteration + arguments.iterations,
        ):
            if arguments.iterations > 1:
                logger.info(
                    "=== Starting iteration %d/%d ===",
                    iteration,
                    arguments.start_iteration + arguments.iterations - 1,
                )
            shape_results, multi_sim, tf_publisher, viz_marker_publisher = (
                _build_world_and_sort(node, arguments)
            )
            iteration_result = SortingIterationResult(
                iteration=iteration, shape_results=shape_results
            )
            iteration_results.append(iteration_result)

            if keep_simulation_running:
                break

            multi_sim.stop_simulation()
            if viz_marker_publisher is not None:
                viz_marker_publisher.stop()
            if tf_publisher is not None:
                tf_publisher.stop()
            multi_sim = tf_publisher = viz_marker_publisher = None
            _reclaim_native_heap_fragmentation()

        if keep_simulation_running:
            logger.info("Sorting done; the simulation keeps running.")
            logger.info("Done. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
        else:
            _log_iteration_summary(iteration_results)
    except KeyboardInterrupt:
        pass
    finally:
        if multi_sim is not None:
            multi_sim.stop_simulation()
        if viz_marker_publisher is not None:
            viz_marker_publisher.stop()
        if tf_publisher is not None:
            tf_publisher.stop()
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
