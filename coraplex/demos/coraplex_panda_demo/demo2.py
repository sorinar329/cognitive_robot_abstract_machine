"""
Experimental copy of ``demo.py`` that randomizes PickUpAction/PlaceAction velocity/
timing parameters per attempt (see ``pickup_place_parameterization.py``) instead of
always using their fixed defaults, so every attempt persisted to the database shows real
variation, suitable as training data for a probabilistic model.

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

import datetime
import os
import random
import threading
from pathlib import Path
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
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

from pickup_place_parameterization import sample_pickup_instance, sample_place_instance
from stacking_attempt_record import StackingAttemptRecord

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine

from giskardpy.motion_statechart.context import MotionStatechartContext
from physics_simulators.base_simulator import SimulatorCallbackResult
from segmind.detectors.base import SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoBody
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import Pose

from cramera.live.runner import start as start_live_viz
from cramera.server import start_in_background as start_viz_frontend


def verify_workspace_matches_demo() -> None:
    """
    Check that the workspace packages were imported from the same checkout this demo
    file lives in.

    Several checkouts of this repository can be installed in different virtualenvs at
    once. Running the demo with the wrong interpreter loads this file from one checkout
    and its imports from another, which surfaces much later as a missing attribute on a
    class that plainly has it.
    """
    # A concrete module rather than the package: the packages here are
    # namespace packages, whose ``__file__`` is None.
    from physics_simulators import mujoco_simulator

    demo_checkout = Path(__file__).resolve().parents[3]
    package_checkout = Path(mujoco_simulator.__file__).resolve().parents[3]
    if demo_checkout == package_checkout:
        return
    raise RuntimeError(
        f"This demo lives in {demo_checkout} but physics_simulators was imported "
        f"from {package_checkout}. Run it with the interpreter whose packages "
        f"point at {demo_checkout}."
    )


verify_workspace_matches_demo()

start_viz_frontend()
start_live_viz()


RANDOM_SEED = int(os.environ.get("DEMO_RANDOM_SEED", "42"))
"""
Seed for the per-attempt parameter sampling.

The attempt parameters are drawn from a Gaussian prior (see
``pickup_place_parameterization``), which without a fixed seed makes two runs of this
file differ regardless of anything else -- including two runs on the same machine. Set
``DEMO_RANDOM_SEED`` in the environment to vary it deliberately.

..note:: This makes only the sampled *parameters* reproducible. The run as a whole is
    not: the physics runs in its own thread and is paced against the wall clock, so how
    much simulation happens per command still depends on how fast the machine is.
"""

random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)

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
floor = world.get_body_by_name("floor")

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


# The position printing that used to run on its own thread here was removed: it
# evaluated forward kinematics twice a second, and the spatial types backing that
# are CasADi symbolics whose nodes are reference counted without thread safety.
# Doing it alongside the main thread's motion planning corrupted a node's refcount
# and freed it while still referenced, which surfaced hours later as a SIGSEGV in
# casadi::SXElem::is_constant() on an unmapped address. :func:`print_positions` is
# kept for the one call on the main thread once the run is over.

NUMBER_OF_ITERATIONS = 10000
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

WORKSPACE_BOUND = 5.0
"""
Half-extent (in meters) of the region a cube can legitimately be in, in every axis.

The scene is a table-top within arm's reach, so a metre is already far outside it, while
a diverged cube reaches thousands of metres within a single iteration -- anything
between the two works as a divergence threshold.
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

RUN_STARTED_AT = datetime.datetime.now(datetime.timezone.utc)
"""
When this run started, stamped onto every attempt record so runs sharing the database
stay separable even though their iteration numbering restarts at 1.
"""


def _create_database_session(database_uri: str) -> Session:
    """
    Connect to the database, create any missing tables, and return an ORM session that
    plans can be persisted through.
    """
    print(f"[database] Connecting to {database_uri} ...")
    engine = create_engine(database_uri)
    Base.metadata.create_all(bind=engine, checkfirst=True)
    StackingAttemptRecord.create_table(engine)
    print("[database] Schema verified.")
    return Session(engine)


def persist_plan(
    plan: PlanNode, iteration_index: int, step_name: str, simulation_diverged: bool
) -> None:
    """
    Persists one stacking attempt's plan -- every action's parameters plus its recorded
    per-node status -- to the database, alongside a :class:`StackingAttemptRecord`
    naming the iteration and step it came from.

    Called unconditionally for every step of every iteration, regardless of whether the
    cube actually ended up stacked -- this demo collects every attempt's randomized
    parameters for later analysis, success or not. The attempt record is what makes that
    sliceable afterwards, in particular so attempts made while the simulation had
    diverged can be excluded. Persistence failures are logged and swallowed so a
    database hiccup never aborts the run.
    """
    try:
        plan_dao = to_dao(plan)
        database_session.add(plan_dao)
        database_session.flush()
        database_session.add(
            StackingAttemptRecord(
                run_started_at=RUN_STARTED_AT,
                iteration_index=iteration_index,
                step_name=step_name,
                plan_database_id=plan_dao.database_id,
                simulation_diverged=simulation_diverged,
            )
        )
        database_session.commit()
        print(
            f"[database] iteration {iteration_index} '{step_name}' persisted"
            f"{' (DIVERGED)' if simulation_diverged else ''}"
        )
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
    Returns every cube to its spawn pose and brings it to a standstill, undoing both the
    displacement and the motion left over from the previous iteration's attempts.

    Clearing the velocity is what makes this a recovery rather than a cosmetic reset: a
    cube that has picked up a runaway velocity keeps it across a pose reset, so without
    this a single diverged iteration ruins every iteration after it.
    """
    for name, position in CUBE_SPAWN_POSITIONS.items():
        multi_sim.simulator.reset_body_velocity(body_name=name)
        multi_sim.simulator.set_body_position(body_name=name, position=position)
        multi_sim.simulator.set_body_quaternion(
            body_name=name, quaternion=CUBE_SPAWN_ORIENTATION
        )


def diverged_cubes() -> list[str]:
    """
    The cubes that have left the region the scene could plausibly place them in.

    An unstable contact can accelerate a cube to thousands of metres per second, which
    shows up as a position far outside the workspace long before anything else notices.
    """
    escaped = []
    for name in CUBE_SPAWN_POSITIONS:
        position = multi_sim.simulator.get_body_position(name).result[:3]
        outside_horizontally = (
            abs(position[0]) > WORKSPACE_BOUND or abs(position[1]) > WORKSPACE_BOUND
        )
        if outside_horizontally or not -WORKSPACE_BOUND < position[2] < WORKSPACE_BOUND:
            escaped.append(name)
    return escaped


class SimulationDidNotRecover(RuntimeError):
    """
    Raised when cubes are still outside the workspace after being reset to their spawn
    poses and brought to rest.

    Continuing past this point is what turned one diverged iteration into hundreds of
    worthless ones: every later attempt is planned against object poses thousands of
    metres away, and no amount of further iterations brings them back.
    """

    def __init__(self, cube_names: list[str], iteration_index: int) -> None:
        super().__init__(
            f"{', '.join(cube_names)} still outside the workspace after the reset at "
            f"the start of iteration {iteration_index}; stopping rather than collecting "
            "attempts against nonsense object poses"
        )


class FrictionNotApplied(RuntimeError):
    """
    Raised when the sampled object friction could not be pushed onto the simulator.

    Friction is the strongest influence this demo has over whether a grasp holds, so an
    unnoticed failure to apply it would quietly turn every attempt into a run at the
    scene's default friction and make the collected data misleading.
    """

    def __init__(self, body_name: str, reason: str) -> None:
        super().__init__(f"Could not set friction of {body_name}: {reason}")


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
    #
    # Addressed through the cube's body rather than by geom name: MujocoSim rebuilds
    # the model from the World object, and shapes carry no name of their own, so the
    # scene file's geom names do not exist in the running model.
    friction_result = multi_sim.simulator.set_body_friction(
        object_body.name.name,
        numpy.array([pickup_action.object_friction, 0.05, 0.0005]),
    )
    if (
        friction_result.type
        is not SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
    ):
        raise FrictionNotApplied(object_body.name.name, friction_result.info)

    print(
        f"[params] pickup {object_body.name.name}: "
        f"pre_approach={pickup_action.pre_approach_linear_velocity:.4f}, "
        f"grasp={pickup_action.final_approach_linear_velocity:.4f}, "
        f"closing={pickup_action.grasp_closing_velocity:.4f}, "
        f"lift={pickup_action.lift_linear_velocity:.4f}, "
        f"stall_min_time={pickup_action.grasp_stall_minimum_time:.4f}, "
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

    The detector reports only relations it has not seen before, so its context is
    cleared first to make each iteration's result independent of earlier ones.
    """
    segmind_context.latest_support.clear()
    support_detector.update_context_and_events(
        motion_statechart_context, segmind_context, [box, box1, box2, box3]
    )
    return segmind_context.latest_support


def segmind_sees_on_floor(body: Body) -> bool:
    """
    Whether segmind currently sees ``body`` resting directly on the floor.

    Used as the between-step check rather than asking whether the cube is
    supported by the cube below it. Read right after placing, the support onto
    the cube below reads as absent: the cube is still held while the gripper
    lifts away and only settles onto its target a moment later, so the check
    lands in the gap and reports a false negative. Having reached the floor is
    unambiguous by comparison -- a cube down there is not on the stack, whenever
    it is looked at.
    """
    return floor in detected_supports().get(body, set())


def segmind_approved() -> bool:
    """
    Whether segmind sees the whole stack standing, every expected support at once.

    Read in a single pass, so it answers whether the stack is intact now rather than
    whether each step succeeded at the time it ran.
    """
    supports = detected_supports()
    return all(
        supporter in supports.get(supported, set())
        for supported, supporter in expected_supports()
    )


def append_support_report(iteration_index: int, simulation_diverged: bool) -> None:
    """
    Append this iteration's support findings to :data:`SUPPORT_REPORT_PATH`.

    Reports each expected support and whether the stack as a whole stands, read from the
    world as it is once the iteration has finished. A diverged iteration is marked as
    such, since its findings describe cubes that had left the scene rather than a stack
    that failed to hold.
    """
    supports = detected_supports()
    approved = segmind_approved()

    lines = [f"\n## Iteration {iteration_index}\n"]
    lines.append(f"`segmind_approved()`: **{approved}**\n")
    if simulation_diverged:
        lines.append("**simulation diverged -- excluded from results**\n")
    lines.append("| expected support | segmind |")
    lines.append("|---|---|")
    for supported, supporter in expected_supports():
        holds = supporter in supports.get(supported, set())
        lines.append(
            f"| {supported.name.name} on {supporter.name.name} "
            f"| {'yes' if holds else 'no'} |"
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
    print(
        f"[segmind] iteration {iteration_index} supports written to {SUPPORT_REPORT_PATH}"
    )


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

    :return: The performed plan, and whether ``object_body`` avoided ending up on the
        floor (see :func:`segmind_sees_on_floor`), which is what decides whether the
        remaining steps of the iteration are still worth attempting.
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

    return plan, not segmind_sees_on_floor(object_body)


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
        still_escaped = diverged_cubes()
        if still_escaped:
            raise SimulationDidNotRecover(still_escaped, iteration)

        iteration_plans: list[tuple[str, PlanNode]] = []
        simulation_diverged = False

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
            plan, stayed_off_floor = attempt_stack(
                cube_to_pick, cube_to_stack_on, Arms.LEFT, step_label
            )
            iteration_plans.append((step_label, plan))
            print(
                f"[info] {step_label} "
                f"{'stayed off the floor' if stayed_off_floor else 'ended up on the FLOOR'} "
                "(persisted regardless)"
            )
            escaped = diverged_cubes()
            if escaped:
                # Every later step would be planned against a target pose that is
                # now thousands of metres away, and the cubes cannot come back on
                # their own -- the next iteration's reset is what recovers them.
                simulation_diverged = True
                print(
                    f"[warning] simulation DIVERGED: {', '.join(escaped)} left the "
                    f"workspace -- abandoning the rest of iteration {iteration}"
                )
                break
            if not stayed_off_floor:
                # Every later step stacks onto this cube, so continuing would
                # only pile onto a cube that is lying on the floor.
                print(
                    f"[warning] segmind sees {cube_to_pick.name.name} on the floor -- "
                    f"abandoning the rest of iteration {iteration}"
                )
                break
            time.sleep(1)

        for step_label, plan in iteration_plans:
            persist_plan(plan, iteration, step_label, simulation_diverged)
        print(
            f"[database] iteration {iteration} persisted {len(iteration_plans)} "
            "step(s) unconditionally"
        )

        print_iteration_summary(iteration)
        append_support_report(iteration, simulation_diverged)

        iteration_durations.append(time.time() - iteration_start)
        average_duration = sum(iteration_durations) / len(iteration_durations)
        print(
            f"=== iteration {iteration}/{NUMBER_OF_ITERATIONS} took "
            f"{iteration_durations[-1]:.1f}s (average so far: {average_duration:.1f}s) ==="
        )

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
