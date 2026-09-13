"""
The pickup demo's sorting, run against the digital twin alone: no MuJoCo, no physical
robot, just Giskard's own kinematic simulated execution and RViz to watch it in.

:mod:`~experiments.tracy_experiments.pickup.pickup_demo_real` sorts what the camera
finds; this sorts a fixed layout instead
(:class:`~experiments.tracy_experiments.montessori.scene_builder.TracyOnItsOwnTable`,
the same deterministic row of pieces
:func:`~experiments.montessori.scenarios.SortingScene` scripts a run over), so there is
nothing here to look for and nothing for a look to disagree with the twin about. Each
piece is picked up and let go over the hole it fits through the same way
:class:`~experiments.montessori.scenarios.PickThePieceUp`/
:class:`~experiments.montessori.scenarios.PutThePieceInItsHole` script it: plain
:class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`/
:class:`~coraplex.robot_plans.actions.core.placing.PlaceAction`, ticked by Giskard under
:attr:`~coraplex.datastructures.enums.ExecutionType.SIMULATED` against the twin's own
kinematics, so nothing settles a released piece under gravity: it is left exactly where
the plan let go of it, floating over its hole or resting on the board's lid. Right after
each release, :class:`~experiments.tracy_experiments.pickup.release_simulation.
ReleaseCheck` drops the same piece under real physics in a disposable copy of the scene
and reports what it found -- whether the piece actually fell through -- without changing
what the twin itself believes. Every event the check's own SegMind monitor detected
(support, hole contact, containment, insertion, ...) is streamed to the same live
dashboard :mod:`~experiments.tracy_experiments.montessori.event_dashboard` serves the
real demo from, so an insertion can be watched landing in the browser as each piece is
checked.

Run with (``iai_tracy_description`` must be built and sourced, and RViz2 pointed at a
``MarkerArray`` display on ``/semworld/viz_marker``)::

    python -m experiments.tracy_experiments.pickup.pickup_demo_simulated

While it runs, open ``http://127.0.0.1:5000`` for the event dashboard.
"""

from __future__ import annotations

import logging
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

import rclpy
from rclpy.executors import SingleThreadedExecutor

from coraplex.execution_environment import simulated_robot
from experiments.montessori.pieces import KnownPieceSet, SMALLER_PIECES
from experiments.montessori.scenarios import RELEASE_HEIGHT_ABOVE_THE_HOLE, SortingScene
from experiments.montessori.semantics import MontessoriShapeCategory
from experiments.tracy_experiments.equipment import joint_state_of_type
from experiments.tracy_experiments.montessori.event_dashboard import (
    EventFeed,
    run_dashboard,
)
from experiments.tracy_experiments.montessori.scene_builder import TracyOnItsOwnTable
from experiments.tracy_experiments.pickup.release_simulation import ReleaseCheck
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

NODE_NAME = "tracy_pickup_demo_simulated"
"""
The name this demo's node registers under.
"""

PIECE_SET: KnownPieceSet = SMALLER_PIECES
"""
The set of loose pieces stood on Tracy's table, matching what the physical demo sorts.
"""

EXECUTOR_THREAD_NAME = "rclpy-executor"
"""
The name of the thread the demo's node is spun on.
"""

# %% building the scene


def build_world() -> World:
    """
    Build Tracy, its table, the shape-sorting board and :data:`PIECE_SET` standing in
    the fixed row
    :class:`~experiments.tracy_experiments.montessori.scene_builder.TracyOnItsOwnTable`
    lays them out in, with both arms parked.
    """
    world = TracyOnItsOwnTable(piece_set=PIECE_SET).build(Tracy)
    park_both_arms(world)
    return world


def park_both_arms(world: World) -> None:
    """
    Put both of Tracy's arms in their parked pose and open the picking gripper, the pose
    the real robot starts every sort from.

    Without this, Tracy is left wherever its URDF's own default joint values put it,
    which is not a pose either arm was ever meant to plan a reach from.

    :param world: The world holding Tracy, modified in place.
    """
    [robot] = world.get_semantic_annotations_by_type(Tracy)
    joint_state_of_type(robot.left_arm.end_effector, GripperState.OPEN).apply_to(world)
    joint_state_of_type(robot.left_arm, StaticJointState.PARK).apply_to(world)
    joint_state_of_type(robot.right_arm, StaticJointState.PARK).apply_to(world)
    world.notify_state_change()


# %% sorting the pieces


def sort_every_piece(
    scene: SortingScene, release_check: ReleaseCheck, feed: EventFeed
) -> None:
    """
    Pick up and let go of every loose piece the scene holds, over the hole its own shape
    fits through, in a fixed order.

    :param scene: The scene to sort, built on the twin the plans run against.
    :param release_check: Checks each piece's release under physics; see
        :func:`sort_one_piece`.
    :param feed: Where each release check's detected events are streamed to, for the
        live dashboard.
    """
    for category in sorted(scene.categories, key=lambda category: category.value):
        sort_one_piece(scene, category, release_check, feed)


def sort_one_piece(
    scene: SortingScene,
    category: MontessoriShapeCategory,
    release_check: ReleaseCheck,
    feed: EventFeed,
) -> None:
    """
    Pick up the loose piece of one shape, let it go above the hole it fits through, and
    report what a disposable physics check of that release found.

    :param scene: The scene the piece stands in.
    :param category: The shape of the piece to sort.
    :param release_check: Drops a piece of the same kind under physics once it is
        released, in a scene of its own that never changes what ``scene``'s own twin
        believes.
    :param feed: Where each event the check detected is streamed to, for the live
        dashboard.
    """
    logger.info("Picking up %s.", category.value)
    scene.pick_the_piece_up(category)
    hole_position = scene.hole_for(category).root.global_transform.to_position()
    release_position = Point3(
        float(hole_position.x),
        float(hole_position.y),
        float(hole_position.z) + RELEASE_HEIGHT_ABOVE_THE_HOLE,
    )
    logger.info("Releasing %s over its hole at %s.", category.value, release_position)
    scene.put_the_piece_down_at(category, release_position)

    outcome = release_check.simulate(scene.shape_of(category), scene.board)
    logger.info(
        "%s: physics check -> fell_through=%s, settled at %s.",
        category.value,
        outcome.fell_through,
        outcome.settled_pose.to_position(),
    )
    for event in outcome.events:
        feed.publish(category.value, event)


# %% wiring rviz


def publish_to_rviz(world: World) -> None:
    """
    Start publishing ``world``'s transforms and markers, so it can be watched in RViz2.

    :param world: The world to publish; kept in step for as long as the process runs.
    """
    rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(
        target=executor.spin, daemon=True, name=EXECUTOR_THREAD_NAME
    )
    thread.start()
    # Gives the executor a moment to start spinning before the publishers' first
    # messages are queued, so nothing is published before anything can be subscribed.
    time.sleep(0.1)

    TFPublisher(node=node, _world=world)
    viz_marker_publisher = VizMarkerPublisher(_world=world, node=node)
    logger.info(
        "Visualizing the Tracy pickup demo on topic '%s'.",
        viz_marker_publisher.topic_name,
    )


def main() -> None:
    """
    Build the scene, sort every piece against the digital twin, and keep publishing to
    RViz until interrupted.
    """
    world = build_world()
    publish_to_rviz(world)
    logger.info("Built world with %d bodies.", len(world.bodies))

    feed = EventFeed()
    run_dashboard(feed)

    with simulated_robot:
        sort_every_piece(SortingScene(world), ReleaseCheck(piece_set=PIECE_SET), feed)

    logger.info("Sorting done. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
