"""
Experimental copy of ``demo.py`` that randomizes PickUpAction/PlaceAction velocity/
timing parameters per attempt (see ``pickup_place_parameterization.py``) instead of
always using their fixed defaults, so every attempt persisted to the database shows
real variation, suitable as training data for a probabilistic model.

Every step of every iteration is persisted unconditionally, regardless of whether the
cube actually ended up stacked, matching ``demo.py``'s own persistence behaviour --
success/failure is only used informationally (segmind's support report, log output),
never to gate what gets saved.

Kept as a separate file rather than modifying ``demo.py`` in place: that file reflects
an entire session's worth of validated tuning (grasp reliability, speed, placing-release
verification), and this randomization is new/unvalidated.

No JPT training or model-based sampling happens here yet -- every attempt draws from a
fixed Gaussian prior (see ``pickup_place_parameterization.py``). Training a model on the
resulting data and closing the loop is a follow-up step.
"""

import os
import threading
from pathlib import Path
import time
import numpy
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sqlalchemy import text
from sqlalchemy.orm import Session

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, ExecutionType
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import ExecutionEnvironment
from coraplex.orm.ormatic_interface import Base
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

from pickup_place_parameterization import sample_pickup_instance, sample_place_instance

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.detectors.base import SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoBody
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import Pose


time.sleep(8)  # Wait for the launch file to start

execition_mode = ExecutionType.SIMULATED

print("Init ROS")
rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

world = MJCFParser(
    "/home/nvasant/workspace/ros/src/manipulation_experiments/resources/generated/stacking_scene.xml"
).parse()
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
        world.compute_forward_kinematics(world.root, tool_frame).to_position().evaluate()[:3],
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

STACK_XY_TOLERANCE = 0.03
"""
Maximum horizontal (x/y) distance (in meters) allowed between a stacked cube and its
target's center for the stack to count as properly centered, not just coincidentally at
the right height.

Matches the tolerance already used by :class:`PlaceAction`'s own ``post_condition`` pose
check (``placing.py``), which is otherwise unevaluated here since the demo's ``Context``
is built with ``evaluate_conditions=False``.
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
    cube actually ended up stacked -- this demo collects every attempt's randomized
    parameters for later analysis, success or not. Persistence failures are logged and
    swallowed so a database hiccup never aborts the run.
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

    Unlike ``demo.py``, the ``PickUpAction``/``PlaceAction`` velocity/timing parameters
    are randomly sampled per call (see ``pickup_place_parameterization.py``) instead of
    using their fixed defaults, so that successful attempts show real, varied parameter
    values once persisted.
    """
    target_pose = target_body.global_pose
    place_location = Pose.from_xyz_rpy(
        x=target_pose.x,
        y=target_pose.y,
        z=target_pose.z + STACK_HEIGHT_OFFSET,
        reference_frame=world.root,
    )

    grasp_description = GraspDescription(
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
    )

    pickup_action = sample_pickup_instance(object_body, picking_arm, grasp_description)
    place_action = sample_place_instance(object_body, place_location, picking_arm)

    # object_friction is recorded on PickUpAction purely for persistence -- the
    # action itself never applies it (it's a MuJoCo geom property, not a planner
    # goal), so it must be pushed onto the live simulator here before the pick runs.
    # multi_sim.simulator.set_geom_friction(
    #     f"{object_body.name.name}_geom",
    #     numpy.array([pickup_action.object_friction, 0.05, 0.0005]),
    # )

    print(
        f"[params] pickup {object_body.name.name}: "
        f"pre_approach={pickup_action.pre_approach_linear_velocity:.4f}, "
        f"grasp={pickup_action.grasp_linear_velocity:.4f}, "
        f"closing={pickup_action.grasp_closing_velocity:.4f}, "
        f"lift={pickup_action.lift_linear_velocity:.4f}, "
        f"stall_min_time={pickup_action.grasp_stall_min_time:.4f}, "
        f"object_friction={pickup_action.object_friction:.4f}"
    )
    print(
        f"[params] place {object_body.name.name}: "
        f"transport={place_action.transport_linear_velocity:.4f}, "
        f"placing={place_action.placing_linear_velocity:.4f}, "
        f"release={place_action.release_opening_velocity:.4f}, "
        f"retract={place_action.retract_linear_velocity:.4f}"
    )

    return sequential(
        [
            ParkArmsAction(Arms.BOTH),
            pickup_action,
            place_action,
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    )


def _cube_is_stacked(object_body, target_body) -> bool:
    """
    Whether ``object_body`` actually ended up resting centered on top of ``target_body``
    in the physical simulation.

    Checked against the real MuJoCo state rather than the plan's reported
    success/failure: a step can raise (e.g. a re-park afterwards timing out) after
    already having placed the object correctly, and can also report success without the
    object actually ending up in place (see the module docstring's notes on
    ``evaluate_conditions=False``) -- so the reported outcome alone is not a reliable
    signal of whether this step's stacking actually happened.

    Combines two independent checks, since height alone cannot rule out a cube that
    ended up elevated by coincidence without also being centered above the target (e.g.
    perched on a neighboring cube's edge, or still caught on the retracting gripper):

    - Height: strictly between half and 1.5x :data:`STACK_HEIGHT_OFFSET` above the
      target -- a lower bound (rules out "fell off"/"knocked down") and an upper bound
      (rules out "ended up implausibly high", e.g. still stuck to the gripper).
    - Horizontal (x/y) distance to the target's center stays within
      :data:`STACK_XY_TOLERANCE`, i.e. the cube is actually centered above the target
      rather than merely at the right height somewhere nearby.
    """
    object_position = multi_sim.simulator.get_body_position(object_body.name.name).result
    target_position = multi_sim.simulator.get_body_position(target_body.name.name).result

    height_difference = object_position[2] - target_position[2]
    if not (STACK_HEIGHT_OFFSET / 2 < height_difference < STACK_HEIGHT_OFFSET * 1.5):
        return False

    horizontal_distance = numpy.linalg.norm(object_position[:2] - target_position[:2])
    return horizontal_distance < STACK_XY_TOLERANCE



# %% support verification via segmind

SUPPORT_REPORT_PATH = Path(__file__).parent / "support_report.md"
"""
Markdown file the per-iteration support findings are appended to.
"""

segmind_context = SegmindContext()
"""
Shared context the support detector accumulates its findings in.
"""

support_detector = SupportDetector()
"""
Detects which bodies currently support which other bodies.
"""

motion_statechart_context = MotionStatechartContext(world=world)
"""
Gives the detector access to the world's bodies and their collision geometry.
"""


def expected_supports() -> list[tuple[Body, Body]]:
    """
    The support relations a fully built stack should have, bottom up.

    The bottom cube is left out: what the scene rests it on is not part of what
    the stacking is judged on, and it is reported among the detected supports
    anyway.
    """
    return [
        (box1, box),
        (box2, box1),
        (box3, box2),
    ]


def detected_supports() -> dict[Body, set[Body]]:
    """
    Every support relation segmind currently sees among the cubes.

    The detector reports only relations it has not seen before, so its context
    is cleared first to make each iteration's result independent of earlier
    ones.
    """
    segmind_context.latest_support.clear()
    support_detector.update_context_and_events(
        motion_statechart_context, segmind_context, [box, box1, box2, box3]
    )
    return segmind_context.latest_support


def append_support_report(iteration_index: int) -> None:
    """
    Append this iteration's support findings to :data:`SUPPORT_REPORT_PATH`.

    Records segmind's verdict beside the demo's own geometric verifier for the
    same pair, so the two can be checked against each other.
    """
    supports = detected_supports()

    lines = [f"\n## Iteration {iteration_index}\n"]
    lines.append("| expected support | segmind | demo verifier |")
    lines.append("|---|---|---|")
    for supported, supporter in expected_supports():
        segmind_sees = supporter in supports.get(supported, set())
        verifier = "yes" if _cube_is_stacked(supported, supporter) else "no"
        lines.append(
            f"| {supported.name.name} on {supporter.name.name} "
            f"| {'yes' if segmind_sees else 'no'} | {verifier} |"
        )

    lines.append("\nAll supports segmind detected:\n")
    if supports:
        for supported, supporters in sorted(
            supports.items(), key=lambda item: item[0].name.name
        ):
            names = ", ".join(sorted(s.name.name for s in supporters))
            lines.append(f"- `{supported.name.name}` supported by {names}")
    else:
        lines.append("- none")

    with SUPPORT_REPORT_PATH.open("a") as report:
        report.write("\n".join(lines) + "\n")
    print(f"[segmind] iteration {iteration_index} supports written to {SUPPORT_REPORT_PATH}")


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
            plan, stacked = attempt_stack(cube_to_pick, cube_to_stack_on, Arms.LEFT, step_label)
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
        append_support_report(iteration)

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
    print(f"[database] Total persisted plans (SequentialNodeDAO): {persisted_plan_count}")
except Exception as exc:
    print(f"[database] Could not read row count: {exc}")
database_session.close()

print("Plan finished, keeping the viewer open until it is closed")
while multi_sim.simulator.renderer.is_running():
    time.sleep(0.1)
