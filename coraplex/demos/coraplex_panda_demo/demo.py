import os
import threading
import time
import numpy
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sqlalchemy import text
from sqlalchemy.orm import Session

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    ExecutionType,
)
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import ExecutionEnvironment
from coraplex.orm.ormatic_interface import Base
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoBody
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import Pose

from cramera.live.runner import start as start_live_viz
from cramera.server import start_in_background as start_viz_frontend

start_viz_frontend()
start_live_viz()

time.sleep(8)  # Wait for the launch file to start

execition_mode = ExecutionType.SIMULATED

print("Init ROS")
rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

STACKING_SCENE_PATH: str = os.environ.get(
    "STACKING_SCENE_PATH",
    "/home/nvasant/workspace/ros/src/manipulation_experiments/resources/generated/stacking_scene.xml",
)
"""
Path to the MJCF scene this demo loads.

Override via the ``STACKING_SCENE_PATH`` environment variable for checkouts where the
scene lives elsewhere.
"""

world = MJCFParser(STACKING_SCENE_PATH).parse()
Panda.from_world(world)
publisher = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()


# It is important to have the ros_node in the context for a real robot
context = Context(
    world=world,
    robot=world.get_semantic_annotations_by_type(Panda)[0],
    ros_node=node,
    evaluate_conditions=False,
)

box = world.get_body_by_name("cube0")
box1 = world.get_body_by_name("cube1")
box2 = world.get_body_by_name("cube2")
box3 = world.get_body_by_name("cube3")

print("Perform Plan")

arm = context.robot.get_arms()[0]
gripper = arm.end_effector
physically_simulated_dofs = {c.raw_dof for c in gripper.active_connections} | {
    c.raw_dof for c in arm.active_connections
}

# The arm's actuator gains (parsed from the scene's official mujoco_menagerie
# Franka Panda values) are calibrated assuming gravity is separately cancelled
# out via MuJoCo's own gravcomp mechanism, not held against by the PD gains
# alone. Without it, each joint settles with a steady-state error from gravity
# sag large enough (~0.02 rad) to exceed JointPositionList's default 0.01 rad
# convergence threshold -- so a motion holding the arm under gravity (e.g.
# ParkArmsAction) never registers as converged and Giskard keeps sending
# corrective commands indefinitely, which also stalls the rest of the plan.
for connection in arm.active_connections:
    connection.child.simulator_additional_properties.append(
        MujocoBody(gravitation_compensation_factor=1.0)
    )

multi_sim = MujocoSim(
    world=world,
    headless=False,
    step_size=0.0001,
    real_time_factor=1,
    physically_simulated_dofs=physically_simulated_dofs,
    sync_rate_hz=100,
)
time_start = time.time()

tool_frame = gripper.tool_frame


def print_positions():
    """
    Prints the tool_frame's and cube's position as seen by the world model (Giskard's
    kinematic belief) side by side with MuJoCo's own live simulated position, so a
    divergence between "where Giskard thinks it is" and "where it actually, physically
    is" is visible directly.
    """
    tool_frame_kinematic = numpy.array(
        world.compute_forward_kinematics(world.root, tool_frame)
        .to_position()
        .evaluate()[:3],
        dtype=float,
    )
    box_kinematic = numpy.array(
        world.compute_forward_kinematics(world.root, box).to_position().evaluate()[:3],
        dtype=float,
    )
    tool_frame_mujoco = numpy.array(
        multi_sim.simulator.get_body_position(tool_frame.name.name).result[:3],
        dtype=float,
    )
    box_mujoco = numpy.array(
        multi_sim.simulator.get_body_position(box.name.name).result[:3], dtype=float
    )
    print(
        f"tool_frame: kinematic={tool_frame_kinematic} mujoco={tool_frame_mujoco} | "
        f"cube: kinematic={box_kinematic} mujoco={box_mujoco}"
    )


def print_positions_periodically(stop_event: threading.Event):
    while not stop_event.is_set():
        print_positions()
        time.sleep(0.5)


stop_printing = threading.Event()
printing_thread = threading.Thread(
    target=print_positions_periodically, args=(stop_printing,), daemon=True
)
printing_thread.start()

NUMBER_OF_ITERATIONS = 10
"""
Number of times the full pickup/stack sequence is repeated, so the demo can be left
running unattended instead of re-started by hand for every trial.
"""

ITERATION_TIME_LIMIT = 80.0
"""
Wall-clock budget (in seconds) for one iteration's cube-stacking attempts, checked
between attempts.

A stuck grasp (fingers touching the cube without closing around it) can otherwise stall
an iteration indefinitely. Giskard's own tick budget (see
``max_ticks_per_motion_mapping`` below) already bounds any single stuck attempt, but
this is an additional, coarser safety net: once the elapsed time for an iteration
exceeds this limit, remaining cube attempts are skipped and the loop moves on to the
next iteration. Because an already-in-flight attempt is allowed to finish rather than
being killed mid-motion (forcibly interrupting a running Giskard execution is unsafe),
the actual wall-clock time for an iteration that trips this limit can run somewhat past
it.
"""

STACK_HEIGHT_OFFSET = 0.06
"""
Vertical offset (in meters) above a target cube's center at which a placed cube should
end up -- one cube height plus a small clearance margin.
"""

CUBE_SPAWN_POSITIONS = {
    "cube0": numpy.array([0.40, 0.10, 0.06]),
    "cube1": numpy.array([0.40, -0.04, 0.06]),
    "cube2": numpy.array([0.40, -0.14, 0.06]),
    "cube3": numpy.array([0.40, -0.24, 0.06]),
}
"""
Spawn position of every cube, matching the scene's MJCF definition.
"""

CUBE_SPAWN_ORIENTATION = numpy.array([1.0, 0.0, 0.0, 0.0])
"""
Spawn orientation (identity quaternion) of every cube.
"""

DATABASE_URI: str = os.environ.get(
    "CORAPLEX_PANDA_DEMO_DATABASE_URI",
    "postgresql+psycopg://semantic_digital_twin:naren@localhost:5432/coraplex_panda_demo",
)
"""
Connection string for the database that stores every stacking attempt's
plan -- including all of its action parameters (target poses, arm, grasp
description, per-node status) -- for later analysis.

Reuses the ``semantic_digital_twin`` role already provisioned on this host
for the other demos/experiments in this workspace; only the database itself
is dedicated to this demo. Uses the ``psycopg`` (v3) driver explicitly since
only that, not ``psycopg2``, is installed in this environment.
"""


def _create_database_session(database_uri: str) -> Session:
    """
    Connect to the database, create any missing tables, and return an ORM session that
    plans can be persisted through.
    """
    print(f"[database] Connecting to {database_uri} ...")
    engine = create_engine(database_uri)
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("[database] Schema verified.")
    return Session(engine)


def persist_plan(plan: PlanNode, iteration_index: int, step_name: str) -> None:
    """
    Persists one stacking attempt's plan -- every action's parameters plus its recorded
    per-node status -- to the database.

    Called unconditionally for every step of every iteration, regardless of whether the
    cube actually ended up stacked -- this demo collects every attempt's parameters for
    later analysis; success-gated persistence (only keeping fully-stacked iterations) is
    handled by ``demo2.py`` instead. Persistence failures are logged and swallowed so a
    database hiccup never aborts the run.
    """
    try:
        database_session.add(to_dao(plan))
        database_session.commit()
        print(f"[database] iteration {iteration_index} '{step_name}' persisted")
    except Exception as exc:
        print(
            f"[database] failed to persist iteration {iteration_index} "
            f"'{step_name}': {exc}"
        )
        database_session.rollback()


def persist_world_snapshot() -> None:
    """
    Persists the world's physics configuration -- joint dynamics, actuator gains, geom
    friction/solver parameters, MuJoCo body properties such as the gravity-compensation.

    override applied to the arm, and the arm's rated velocity/acceleration/jerk limits
    -- to the database, via the same ``WorldMappingDAO`` mapping already used elsewhere
    in the workspace.

    The acceleration/jerk limits are set on the robot's DOFs (and stay live for the
    whole run, not just this snapshot) by ``Panda.from_world`` itself; see
    ``coraplex.plans.executables`` for the matching ``prediction_horizon`` that makes
    them feasible to plan under.

    Called once, before the simulation loop starts, since this configuration is static
    for the whole demo run.
    """
    try:
        database_session.add(to_dao(world))
        database_session.commit()
        print("[database] world snapshot persisted")
    except Exception as exc:
        print(f"[database] failed to persist world snapshot: {exc}")
        database_session.rollback()


def reset_cubes() -> None:
    """
    Teleports every cube back to its spawn pose in MuJoCo, undoing the displacement from
    the previous iteration's stacking attempts.

    Only position and orientation are reset -- MuJoCo exposes no safe, synchronized API
    to reset a body's velocity, so residual velocity from the previous iteration can
    carry over as a minor, known limitation.
    """
    for name, position in CUBE_SPAWN_POSITIONS.items():
        multi_sim.simulator.set_body_position(body_name=name, position=position)
        multi_sim.simulator.set_body_quaternion(
            body_name=name, quaternion=CUBE_SPAWN_ORIENTATION
        )


def _build_stack_plan(object_body, target_body, picking_arm) -> PlanNode:
    """
    Builds (without performing) a park/pick/place/park plan that stacks ``object_body``
    centered above ``target_body``, one cube height higher.
    """
    target_pose = target_body.global_pose
    place_location = Pose.from_xyz_rpy(
        x=target_pose.x,
        y=target_pose.y,
        z=target_pose.z + STACK_HEIGHT_OFFSET,
        reference_frame=world.root,
    )
    return sequential(
        [
            ParkArmsAction(Arms.BOTH),
            PickUpAction(
                object_body,
                picking_arm,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.TOP,
                    context.robot.get_arms()[0].end_effector,
                    # The cubes sit in a row; without this, the fingers' opening
                    # axis is exactly parallel to that row (confirmed empirically:
                    # dot product -1.0), so any approach overshoot or the finger
                    # sweep itself can clip a neighboring cube. Rotating 90 degrees
                    # makes the opening axis perpendicular to the row instead
                    # (dot product 0.0).
                    rotate_gripper=True,
                ),
                # A grasp is done once the fingers stop moving against the object,
                # not once they reach their nominal fully-closed target -- an object
                # of any real size blocks full closure, so without this the attempt
                # runs out its tick budget and raises MotionDidNotFinish instead of
                # completing the grasp.
                tolerate_grasp_stall=True,
            ),
            PlaceAction(object_body, place_location, picking_arm),
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    )


def _cube_is_stacked(object_body, target_body) -> bool:
    """
    Whether ``object_body`` actually ended up resting on top of ``target_body`` in the
    physical simulation, judged by its height above ``target_body`` being at least half
    of :data:`STACK_HEIGHT_OFFSET`.

    Checked against the real MuJoCo state rather than the plan's reported
    success/failure: a step can raise (e.g. a re-park afterwards timing out) after
    already having placed the object correctly, and can also report success without the
    object actually ending up in place (see the module docstring's notes on
    ``evaluate_conditions=False``) -- so the reported outcome alone is not a reliable
    signal of whether this step's stacking actually happened.

    This result is informational only (printed in the main loop below) and no longer
    gates persistence -- every step's plan is persisted regardless of this outcome; see
    ``demo2.py`` for the success-gated version.
    """
    object_height = multi_sim.simulator.get_body_position(object_body.name.name).result[
        2
    ]
    target_height = multi_sim.simulator.get_body_position(target_body.name.name).result[
        2
    ]
    return object_height - target_height > STACK_HEIGHT_OFFSET / 2


def attempt_stack(
    object_body, target_body, picking_arm, step_name: str
) -> tuple[PlanNode, bool]:
    """
    Builds and performs one stacking attempt, logging and swallowing any failure instead
    of letting it propagate.

    A single failed grasp/place should not crash the whole run -- it skips to the next
    cube (or iteration) instead, re-parking the arms first so the robot starts the next
    attempt from a known configuration.

    Does not persist the plan itself -- see the main loop below, which persists every
    step of every iteration unconditionally, regardless of whether it actually stacked.

    :return: The performed plan, and whether ``object_body`` actually ended up stacked
        on ``target_body`` afterwards (see :func:`_cube_is_stacked`), for informational
        logging only.
    """
    plan = _build_stack_plan(object_body, target_body, picking_arm)
    try:
        plan.perform()
    except Exception as exc:
        print(f"[warning] {step_name} failed ({type(exc).__name__}: {exc}), moving on")
        try:
            sequential([ParkArmsAction(Arms.BOTH)], context=context).perform()
        except Exception as park_exc:
            print(f"[warning] re-park after {step_name} also failed: {park_exc}")

    return plan, _cube_is_stacked(object_body, target_body)


def print_iteration_summary(iteration_index: int) -> None:
    """
    Prints the final z-height of every cube, a quick visual check of how high the stack
    reached in this iteration.
    """
    heights = {
        name: multi_sim.simulator.get_body_position(name).result[2]
        for name in CUBE_SPAWN_POSITIONS
    }
    print(f"--- iteration {iteration_index} final heights: {heights} ---")


database_session = _create_database_session(DATABASE_URI)
persist_world_snapshot()

# constraints = SimulatorConstraints(max_number_of_steps=10000)
multi_sim.start_simulation()

# MujocoSim rebuilds a fresh MuJoCo model from the World object rather than
# reusing the scene file directly, which silently drops the scene's
# <visual><global azimuth="120" elevation="-20"/> hint -- the viewer would
# otherwise fall back to MuJoCo's own default camera (azimuth=90,
# elevation=-45), making the cubes' row appear rotated instead of matching
# the scene's intended viewing angle.
viewer = multi_sim.simulator.renderer
if hasattr(viewer, "cam"):
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 1.2
    viewer.cam.lookat[:] = [0.3, 0.0, 0.35]
iteration_durations = []
with ExecutionEnvironment(
    execution_type=execition_mode,
    collision_avoidance=False,
    real_time_pacing=True,
    # A stuck grasp can otherwise retry for several minutes (2000 ticks per
    # merged motion at 50 Hz = 40 s per motion) before finally giving up --
    # far too slow across many iterations. 250 still gives each individual
    # motion a 5 s budget, comfortably above how long a successful one
    # actually takes at the tuned approach/lift/transport speeds, while
    # capping a single stuck attempt to well under ITERATION_TIME_LIMIT.
    max_ticks_per_motion_mapping=250,
):
    for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
        iteration_start = time.time()
        print(f"=== starting iteration {iteration}/{NUMBER_OF_ITERATIONS} ===")
        reset_cubes()
        time.sleep(1.5)

        iteration_plans: list[tuple[str, PlanNode]] = []

        for cube_to_pick, cube_to_stack_on, step_label in [
            (box1, box, "cube1 onto cube0"),
            (box2, box1, "cube2 onto cube1"),
            (box3, box2, "cube3 onto cube2"),
        ]:
            elapsed = time.time() - iteration_start
            if elapsed > ITERATION_TIME_LIMIT:
                print(
                    f"[warning] iteration {iteration} already took {elapsed:.1f}s "
                    f"(limit {ITERATION_TIME_LIMIT:.0f}s), skipping remaining "
                    "attempts and moving to the next iteration"
                )
                break
            plan, stacked = attempt_stack(
                cube_to_pick, cube_to_stack_on, Arms.LEFT, step_label
            )
            iteration_plans.append((step_label, plan))
            print(
                f"[info] {step_label} {'stacked' if stacked else 'did NOT stack'} "
                "(informational only -- every step is persisted regardless)"
            )
            time.sleep(1)

        for step_label, plan in iteration_plans:
            persist_plan(plan, iteration, step_label)
        print(
            f"[database] iteration {iteration} persisted {len(iteration_plans)} "
            "step(s) unconditionally"
        )

        print_iteration_summary(iteration)

        iteration_durations.append(time.time() - iteration_start)
        average_duration = sum(iteration_durations) / len(iteration_durations)
        print(
            f"=== iteration {iteration}/{NUMBER_OF_ITERATIONS} took "
            f"{iteration_durations[-1]:.1f}s (average so far: {average_duration:.1f}s) ==="
        )

stop_printing.set()
print("--- final positions ---")
print_positions()

try:
    persisted_plan_count = database_session.execute(
        text('SELECT COUNT(*) FROM "SequentialNodeDAO"')
    ).scalar()
    print(
        f"[database] Total persisted plans (SequentialNodeDAO): {persisted_plan_count}"
    )
except Exception as exc:
    print(f"[database] Could not read row count: {exc}")
database_session.close()

print("Plan finished, keeping the viewer open until it is closed")
while multi_sim.simulator.renderer.is_running():
    time.sleep(0.1)
