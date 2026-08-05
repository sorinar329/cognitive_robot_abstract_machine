"""
Closed-loop stacking with JPT-guided sampling and causal failure diagnosis.

Structured like ``demo2.py`` -- same world, simulation and segmind setup -- but with two
differences:

- Every attempt is sampled from *wider* priors than ``demo2.py``'s (see
  :data:`WIDE_PICKUP_PARAMETER_PRIORS`/:data:`WIDE_PLACE_PARAMETER_PRIORS`), deliberately
  reaching further into the ranges noted in ``pickup_place_parameterization.py`` to break
  a grasp or knock the stack over, so failures happen often enough for diagnosis to have
  something to work with.
- When segmind reports a cube ended up on the floor instead of stacked, the trained JPTs'
  causal circuit (see ``causal_diagnosis.py``) diagnoses which parameter of that attempt
  is least consistent with successful attempts, explains why, and proposes a corrected
  value for it. The same cube is retried with that one parameter corrected -- everything
  else about the attempt unchanged -- and segmind validates the retry the same way it
  validated the original attempt. This repeats, capped at
  :data:`MAX_CORRECTION_ATTEMPTS_PER_CUBE`, for every cube and for the stack as a whole.

Run it with the interpreter whose packages point at this checkout, for example::

    /home/sorin/.virtualenvs/cram2-env/bin/python inference.py
"""

import dataclasses
import os
import random
import threading
from pathlib import Path
import time
import numpy
import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    ExecutionType,
)
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import ExecutionEnvironment

# Not used directly (this demo never persists anything to a database) -- imported
# purely so ormatic's DAO registry gets populated as a side effect. Without it,
# sample_pickup_instance/sample_place_instance's underlying UnderspecifiedParameters
# machinery fails with NoDAOFoundError the first time it tries to extract features
# from a literal Body/GraspDescription argument, since that internally calls to_dao().
from coraplex.orm.ormatic_interface import Base  # noqa: F401

from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

from causal_diagnosis import (
    ActionCausalDiagnoser,
    NoRecommendationAvailable,
    PICKUP_CAUSAL_CONFIG,
    PLACE_CAUSAL_CONFIG,
    PICKUP_MODEL_PATH,
    PLACE_MODEL_PATH,
    RootCauseDiagnosis,
)
from parked_arm_detection_gate import (
    ParkedArmDetectionGate,
    RobotArmParkDeviations,
)
from pickup_place_parameterization import (
    ParameterPrior,
    sample_pickup_instance,
    sample_place_instance,
)

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


# %% causal diagnosers, built once from the trained JPTs -- before the ROS/MuJoCo setup
# below, so a missing or untrained model fails immediately rather than after paying for
# that setup.

pickup_diagnoser = ActionCausalDiagnoser(PICKUP_MODEL_PATH, PICKUP_CAUSAL_CONFIG)
"""
Diagnoses failed pickups against the trained :class:`PickUpAction` parameter tree.
"""

place_diagnoser = ActionCausalDiagnoser(PLACE_MODEL_PATH, PLACE_CAUSAL_CONFIG)
"""
Diagnoses failed placements against the trained :class:`PlaceAction` parameter tree.
"""

RANDOM_SEED = int(os.environ.get("INFERENCE_RANDOM_SEED", "42"))
"""
Seed for the per-attempt parameter sampling, see ``demo2.py``'s own
:data:`RANDOM_SEED` for why this only makes the sampled parameters -- not the run as a
whole -- reproducible.
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

world = MJCFParser(
    "/home/sorin/dev/manipulation_experiments/resources/generated/stacking_scene.xml"
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
floor = world.get_body_by_name("floor")

print("Perform Plan")

arm = context.robot.get_arms()[0]
gripper = arm.end_effector
physically_simulated_dofs = {c.raw_dof for c in gripper.active_connections} | {
    c.raw_dof for c in arm.active_connections
}

# See demo2.py's own comment on this block: without gravity compensation, the arm
# never registers as converged and Giskard keeps sending corrective commands
# indefinitely, stalling the rest of the plan.
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


NUMBER_OF_ITERATIONS = int(os.environ.get("INFERENCE_NUMBER_OF_ITERATIONS", "20"))
"""
Number of times the full pickup/stack sequence is repeated.

Much smaller than ``demo2.py``'s: this demo illustrates the diagnose-and-correct loop
rather than collecting a large training dataset, and wide priors make every iteration
slower (a failed cube can trigger several correction attempts before moving on).
"""

MAX_CORRECTION_ATTEMPTS_PER_CUBE = 3
"""
How many times a single cube is retried with a causally corrected sample before its step
is abandoned as a hard failure.
"""

ITERATION_TIME_LIMIT = 120.0
"""
Wall-clock budget (in seconds) for one iteration, checked between cubes.

Twice ``demo2.py``'s: a cube here can go through several correction attempts, each
costing roughly as long as a normal attempt.
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

WORKSPACE_BOUND = 5.0
"""
Half-extent (in meters) of the region a cube can legitimately be in, in every axis, see
``demo2.py``'s own :data:`WORKSPACE_BOUND`.
"""

# %% wide sampling priors, to make failures common enough to diagnose

WIDE_PICKUP_PARAMETER_PRIORS: dict[str, ParameterPrior] = {
    # Default high=0.24; 0.2 was already noted to spike InfeasibleException failures,
    # so this reaches well past that.
    "pre_approach_linear_velocity": ParameterPrior(
        mean=0.2, std=0.08, low=0.04, high=0.35
    ),
    "grasp_linear_velocity": ParameterPrior(mean=0.16, std=0.09, low=0.01, high=0.35),
    # Default high=0.22; the finger joints' 0.2 m/s physical limit is noted to punt the
    # cube out of the grasp past that point, so this reaches well past it.
    "grasp_closing_velocity": ParameterPrior(mean=0.2, std=0.1, low=0.02, high=0.4),
    # Default high=0.28; 0.18 was already noted to knock the stack down, so this
    # reaches well past that.
    "lift_linear_velocity": ParameterPrior(mean=0.2, std=0.09, low=0.04, high=0.4),
    # Default low=0.15; below the validated 0.3 floor the grasp is called complete
    # before the fingers settle, so this reaches well below both.
    "grasp_stall_min_time": ParameterPrior(mean=0.4, std=0.15, low=0.05, high=0.9),
    # Default low=0.15; below roughly 0.3 the cube is noted to slide straight out of
    # the fingers, so this reaches well below that.
    "object_friction": ParameterPrior(mean=1.2, std=0.7, low=0.05, high=2.8),
}
"""
Same fields as :data:`~pickup_place_parameterization.PICKUP_PARAMETER_PRIORS`, widened
to reach past the failure points ``pickup_place_parameterization.py`` notes for each,
deliberately making some fraction of attempts fail so there is something for causal
diagnosis to work with.
"""

WIDE_PLACE_PARAMETER_PRIORS: dict[str, ParameterPrior] = {
    # Default high=0.18; 0.12 was already noted to knock the stack down, so this
    # reaches well past that.
    "transport_linear_velocity": ParameterPrior(
        mean=0.15, std=0.07, low=0.02, high=0.3
    ),
    # Default high=0.14; 0.08 was already noted to scatter the stack, so this
    # reaches well past that.
    "placing_linear_velocity": ParameterPrior(mean=0.12, std=0.06, low=0.01, high=0.25),
    "release_opening_velocity": ParameterPrior(
        mean=0.15, std=0.07, low=0.015, high=0.28
    ),
    # Default high=0.2; 0.14 was already noted to knock the just-placed cube back
    # down, so this reaches well past that.
    "retract_linear_velocity": ParameterPrior(mean=0.17, std=0.07, low=0.02, high=0.3),
}
"""
Same fields as :data:`~pickup_place_parameterization.PLACE_PARAMETER_PRIORS`, widened
the same way as :data:`WIDE_PICKUP_PARAMETER_PRIORS`.
"""


def _observed_parameters(action, field_names: tuple[str, ...]) -> dict[str, float]:
    """
    :return: ``action``'s current value for each of ``field_names``, by name.
    """
    return {name: getattr(action, name) for name in field_names}


def diagnose_cube_failure(
    pickup_action: PickUpAction, place_action: PlaceAction
) -> tuple[str, RootCauseDiagnosis] | None:
    """
    Diagnose a failed cube attempt against both of its actions' trees, and report
    whichever is more likely to be the actual root cause.

    A cube ending up on the floor could stem from either action -- a bad grasp or a bad
    placement -- and nothing short of a finer-grained mid-plan check says which. Both
    are diagnosed and the one whose primary cause has the *lower* support probability
    under successful attempts is reported: the more anomalous of the two is the more
    likely explanation.

    :param pickup_action: The failed attempt's pickup, as actually performed.
    :param place_action: The failed attempt's place, as actually performed.
    :return: Which action was implicated and its diagnosis, or ``None`` if neither
        action's tree could recommend a correction.
    """
    candidates: list[tuple[str, RootCauseDiagnosis]] = []
    try:
        candidates.append(
            (
                "pickup",
                pickup_diagnoser.diagnose(
                    _observed_parameters(
                        pickup_action, PICKUP_CAUSAL_CONFIG.cause_names
                    )
                ),
            )
        )
    except NoRecommendationAvailable:
        pass
    try:
        candidates.append(
            (
                "place",
                place_diagnoser.diagnose(
                    _observed_parameters(place_action, PLACE_CAUSAL_CONFIG.cause_names)
                ),
            )
        )
    except NoRecommendationAvailable:
        pass

    if not candidates:
        return None
    return min(
        candidates, key=lambda candidate: candidate[1].observed_support_probability
    )


def apply_correction(
    pickup_action: PickUpAction,
    place_action: PlaceAction,
    action_name: str,
    diagnosis: RootCauseDiagnosis,
) -> tuple[PickUpAction, PlaceAction]:
    """
    Build the corrected retry's actions: ``diagnosis``'s parameter replaced on whichever
    action it was diagnosed against, everything else unchanged from the failed attempt.

    :param pickup_action: The failed attempt's pickup.
    :param place_action: The failed attempt's place.
    :param action_name: Which action ``diagnosis`` was diagnosed against, ``"pickup"``
        or ``"place"``.
    :param diagnosis: The diagnosis to apply.
    """
    if action_name == "pickup":
        return (
            dataclasses.replace(
                pickup_action, **{diagnosis.variable_name: diagnosis.corrected_value}
            ),
            place_action,
        )
    return (
        pickup_action,
        dataclasses.replace(
            place_action, **{diagnosis.variable_name: diagnosis.corrected_value}
        ),
    )


# %% support verification via segmind

INFERENCE_REPORT_PATH = Path(__file__).parent / "inference_report.md"
"""
Markdown file the per-iteration diagnosis-and-correction findings are appended to.
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

detection_gate = ParkedArmDetectionGate(
    arm=RobotArmParkDeviations(world=world, robot=context.robot)
)
"""
Holds every support detection back until the arm has parked out of the way.
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
    Every support relation segmind currently sees among the cubes, sampled once
    :class:`ParkedArmDetectionGate` reports the arm parked and the scene settled.

    The detector reports only relations it has not seen before, so its context is
    cleared first to make each iteration's result independent of earlier ones.
    """
    if not detection_gate.wait_for_parked_arm():
        print(
            f"[warning] the arm did not reach its park position within "
            f"{detection_gate.arrival_timeout:.1f}s -- detecting supports anyway"
        )
    segmind_context.latest_support.clear()
    support_detector.update_context_and_events(
        motion_statechart_context, segmind_context, [box, box1, box2, box3]
    )
    return segmind_context.latest_support


def segmind_sees_on_floor(body: Body) -> bool:
    """
    Whether segmind currently sees ``body`` resting directly on the floor, see
    ``demo2.py``'s own :func:`segmind_sees_on_floor` for why this, rather than checking
    support from the cube below, is the between-step check.
    """
    return floor in detected_supports().get(body, set())


def segmind_approved() -> bool:
    """
    Whether segmind sees the whole stack standing, every expected support at once.
    """
    supports = detected_supports()
    return all(
        supporter in supports.get(supported, set())
        for supported, supporter in expected_supports()
    )


# %% building and performing one attempt


class FrictionNotApplied(RuntimeError):
    """
    Raised when the sampled object friction could not be pushed onto the simulator, see
    ``demo2.py``'s own :class:`FrictionNotApplied`.
    """

    def __init__(self, body_name: str, reason: str) -> None:
        super().__init__(f"Could not set friction of {body_name}: {reason}")


def _apply_object_friction(pickup_action: PickUpAction) -> None:
    """
    Push ``pickup_action``'s sampled friction onto the simulator's copy of its target
    object, since :class:`PickUpAction` records the value but never applies it itself
    (see ``demo2.py``'s own identical comment on ``_build_stack_plan``).
    """
    friction_result = multi_sim.simulator.set_body_friction(
        pickup_action.object_designator.name.name,
        numpy.array([pickup_action.object_friction, 0.05, 0.0005]),
    )
    if (
        friction_result.type
        is not SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
    ):
        raise FrictionNotApplied(
            pickup_action.object_designator.name.name, friction_result.info
        )


def sample_actions(
    object_body: Body, target_body: Body, picking_arm: Arms
) -> tuple[PickUpAction, PlaceAction]:
    """
    Sample a fresh pickup/place pair from the wide priors, for stacking ``object_body``
    centered above ``target_body``, one cube height higher.
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
        # See demo2.py's own identical comment: rotating 90 degrees keeps the
        # fingers' opening axis from sweeping into a neighboring cube.
        rotate_gripper=True,
    )
    pickup_action = sample_pickup_instance(
        object_body, picking_arm, grasp_description, priors=WIDE_PICKUP_PARAMETER_PRIORS
    )
    place_action = sample_place_instance(
        object_body, place_location, picking_arm, priors=WIDE_PLACE_PARAMETER_PRIORS
    )
    return pickup_action, place_action


def build_stack_plan(
    pickup_action: PickUpAction, place_action: PlaceAction
) -> PlanNode:
    """
    Build (without performing) a park/pick/place/park plan from already-sampled actions,
    applying the pickup's sampled friction first.
    """
    _apply_object_friction(pickup_action)
    print(
        f"[params] pickup {pickup_action.object_designator.name.name}: "
        f"pre_approach={pickup_action.pre_approach_linear_velocity:.4f}, "
        f"grasp={pickup_action.grasp_linear_velocity:.4f}, "
        f"closing={pickup_action.grasp_closing_velocity:.4f}, "
        f"lift={pickup_action.lift_linear_velocity:.4f}, "
        f"stall_min_time={pickup_action.grasp_stall_min_time:.4f}, "
        f"object_friction={pickup_action.object_friction:.4f}"
    )
    print(
        f"[params] place {pickup_action.object_designator.name.name}: "
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


def perform_attempt(
    pickup_action: PickUpAction, place_action: PlaceAction, step_name: str
) -> bool:
    """
    Build and perform one park/pick/place/park attempt, logging and swallowing any
    failure instead of letting it propagate, see ``demo2.py``'s own
    :func:`attempt_stack` for why.

    :return: Whether :attr:`pickup_action.object_designator` avoided ending up on the
        floor.
    """
    plan = build_stack_plan(pickup_action, place_action)
    try:
        plan.perform()
    except Exception as exc:
        print(f"[warning] {step_name} failed ({type(exc).__name__}: {exc}), moving on")
        try:
            sequential([ParkArmsAction(Arms.BOTH)], context=context).perform()
        except Exception as park_exc:
            print(f"[warning] re-park after {step_name} also failed: {park_exc}")

    return not segmind_sees_on_floor(pickup_action.object_designator)


# %% the diagnose-and-correct loop


@dataclasses.dataclass
class CubeAttemptOutcome:
    """
    How one cube's stacking step went, including any diagnosis-and-correction cycle.
    """

    step_label: str
    """
    Which stacking step this outcome is for, e.g. ``"cube1 onto cube0"``.
    """

    initial_succeeded: bool
    """
    Whether the first, uncorrected attempt stayed off the floor.
    """

    correction_attempts: int
    """
    How many corrected retries were performed, 0 if the first attempt already succeeded
    or no diagnosis was available.
    """

    diagnoses: list[tuple[str, RootCauseDiagnosis]]
    """
    Every diagnosis made for this step, one per correction attempt, in order.
    """

    final_succeeded: bool
    """
    Whether the cube ended up off the floor after every attempt made for this step.
    """


@dataclasses.dataclass
class DiagnosisOutcome:
    """
    One causal diagnosis performed during the run, and what happened after applying its
    correction.
    """

    step_label: str
    """
    Which stacking step this diagnosis was for.
    """

    action_name: str
    """
    Which action the diagnosis was made against, ``"pickup"`` or ``"place"``.
    """

    diagnosis: RootCauseDiagnosis
    """
    The diagnosis itself.
    """

    diagnosis_duration_seconds: float
    """
    How long computing this diagnosis took.
    """

    correction_succeeded: bool
    """
    Whether the retry performed with this diagnosis's correction applied stayed off the
    floor.
    """


diagnosis_outcomes: list[DiagnosisOutcome] = []
"""
Every causal diagnosis performed across the whole run, in the order it happened -- see
:func:`diagnosis_summary_lines`.
"""


def attempt_cube_with_correction(
    cube_to_pick: Body, cube_to_stack_on: Body, picking_arm: Arms, step_label: str
) -> CubeAttemptOutcome:
    """
    Attempt one cube's stacking step, diagnosing and correcting up to
    :data:`MAX_CORRECTION_ATTEMPTS_PER_CUBE` times if it fails.
    """
    pickup_action, place_action = sample_actions(
        cube_to_pick, cube_to_stack_on, picking_arm
    )
    succeeded = perform_attempt(pickup_action, place_action, step_label)
    initial_succeeded = succeeded
    print(
        f"[info] {step_label} attempt 1 "
        f"{'stayed off the floor' if succeeded else 'ended up on the FLOOR'}"
    )

    diagnoses: list[tuple[str, RootCauseDiagnosis]] = []
    correction_attempts = 0

    while not succeeded and correction_attempts < MAX_CORRECTION_ATTEMPTS_PER_CUBE:
        diagnosis_start_time = time.time()
        diagnosis_entry = diagnose_cube_failure(pickup_action, place_action)
        diagnosis_duration = time.time() - diagnosis_start_time
        if diagnosis_entry is None:
            print(f"[causal] {step_label}: no correction available, abandoning retries")
            break

        action_name, diagnosis = diagnosis_entry
        diagnoses.append(diagnosis_entry)
        print(
            f"[causal] {step_label} ({action_name}, {diagnosis_duration:.3f}s): "
            f"{diagnosis.explanation()}"
        )

        pickup_action, place_action = apply_correction(
            pickup_action, place_action, action_name, diagnosis
        )
        correction_attempts += 1
        succeeded = perform_attempt(
            pickup_action,
            place_action,
            f"{step_label} (correction {correction_attempts})",
        )
        diagnosis_outcomes.append(
            DiagnosisOutcome(
                step_label=step_label,
                action_name=action_name,
                diagnosis=diagnosis,
                diagnosis_duration_seconds=diagnosis_duration,
                correction_succeeded=succeeded,
            )
        )
        print(
            f"[info] {step_label} correction {correction_attempts} "
            f"{'stayed off the floor' if succeeded else 'ended up on the FLOOR'}"
        )

    return CubeAttemptOutcome(
        step_label=step_label,
        initial_succeeded=initial_succeeded,
        correction_attempts=correction_attempts,
        diagnoses=diagnoses,
        final_succeeded=succeeded,
    )


def diverged_cubes() -> list[str]:
    """
    The cubes that have left the region the scene could plausibly place them in, see
    ``demo2.py``'s own :func:`diverged_cubes`.
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
    Raised when cubes are still outside the workspace after being reset, see
    ``demo2.py``'s own :class:`SimulationDidNotRecover`.
    """

    def __init__(self, cube_names: list[str], iteration_index: int) -> None:
        super().__init__(
            f"{', '.join(cube_names)} still outside the workspace after the reset at "
            f"the start of iteration {iteration_index}; stopping rather than collecting "
            "attempts against nonsense object poses"
        )


def reset_cubes() -> None:
    """
    Returns every cube to its spawn pose and brings it to a standstill, see
    ``demo2.py``'s own :func:`reset_cubes`.
    """
    for name, position in CUBE_SPAWN_POSITIONS.items():
        multi_sim.simulator.reset_body_velocity(body_name=name)
        multi_sim.simulator.set_body_position(body_name=name, position=position)
        multi_sim.simulator.set_body_quaternion(
            body_name=name, quaternion=CUBE_SPAWN_ORIENTATION
        )


def append_inference_report(
    iteration_index: int,
    cube_outcomes: list[CubeAttemptOutcome],
    simulation_diverged: bool,
) -> None:
    """
    Append this iteration's diagnosis-and-correction findings to
    :data:`INFERENCE_REPORT_PATH`.
    """
    stack_approved = segmind_approved()

    lines = [f"\n## Iteration {iteration_index}\n"]
    lines.append(f"`segmind_approved()` (full stack): **{stack_approved}**\n")
    if simulation_diverged:
        lines.append("**simulation diverged -- excluded from results**\n")

    for outcome in cube_outcomes:
        lines.append(f"\n### {outcome.step_label}\n")
        lines.append(
            f"- first attempt: {'success' if outcome.initial_succeeded else 'failure'}"
        )
        for attempt_number, (action_name, diagnosis) in enumerate(
            outcome.diagnoses, start=1
        ):
            lines.append(
                f"- correction {attempt_number} ({action_name}): {diagnosis.explanation()}"
            )
        lines.append(
            f"- final result: {'SUCCESS' if outcome.final_succeeded else 'HARD FAILURE'} "
            f"after {outcome.correction_attempts} correction attempt(s)"
        )

    with INFERENCE_REPORT_PATH.open("a") as report:
        report.write("\n".join(lines) + "\n")
    print(f"[report] iteration {iteration_index} written to {INFERENCE_REPORT_PATH}")


def diagnosis_summary_lines(outcomes: list[DiagnosisOutcome]) -> list[str]:
    """
    :param outcomes: Every diagnosis performed during the run, see
        :data:`diagnosis_outcomes`.
    :return: Markdown lines summarizing how often causal diagnosis triggered, how often
        the resulting correction succeeded, and the average time one diagnosis took.
    """
    lines = ["\n## Causal diagnosis summary\n"]
    if not outcomes:
        lines.append("Causal diagnosis was never triggered.\n")
        return lines

    total = len(outcomes)
    succeeded = sum(1 for outcome in outcomes if outcome.correction_succeeded)
    failed = total - succeeded
    average_duration = (
        sum(outcome.diagnosis_duration_seconds for outcome in outcomes) / total
    )

    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| total diagnoses | {total} |")
    lines.append(f"| succeeded after correction | {succeeded} |")
    lines.append(f"| still failed after correction | {failed} |")
    lines.append(f"| average diagnosis time | {average_duration:.3f}s |")
    return lines


def print_and_append_diagnosis_summary(outcomes: list[DiagnosisOutcome]) -> None:
    """
    Print the causal diagnosis summary and append it to :data:`INFERENCE_REPORT_PATH`.

    :param outcomes: Every diagnosis performed during the run, see
        :data:`diagnosis_outcomes`.
    """
    lines = diagnosis_summary_lines(outcomes)
    print("\n".join(lines))
    with INFERENCE_REPORT_PATH.open("a") as report:
        report.write("\n".join(lines) + "\n")


def print_iteration_summary(iteration_index: int) -> None:
    """
    Prints the final z-height of every cube, see ``demo2.py``'s own
    :func:`print_iteration_summary`.
    """
    heights = {
        name: multi_sim.simulator.get_body_position(name).result[2]
        for name in CUBE_SPAWN_POSITIONS
    }
    print(f"--- iteration {iteration_index} final heights: {heights} ---")


multi_sim.start_simulation()

# See demo2.py's own identical comment: without this, the viewer falls back to
# MuJoCo's default camera instead of the scene's intended viewing angle.
viewer = multi_sim.simulator.renderer
if hasattr(viewer, "cam"):
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 1.2
    viewer.cam.lookat[:] = [0.3, 0.0, 0.35]

iteration_durations = []
successful_iterations = 0
corrected_success_count = 0
hard_failure_count = 0

with ExecutionEnvironment(
    execution_type=execition_mode,
    collision_avoidance=False,
    real_time_pacing=True,
    # See demo2.py's own identical comment on this budget.
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

        cube_outcomes: list[CubeAttemptOutcome] = []
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

            outcome = attempt_cube_with_correction(
                cube_to_pick, cube_to_stack_on, Arms.LEFT, step_label
            )
            cube_outcomes.append(outcome)
            if outcome.final_succeeded:
                if outcome.correction_attempts:
                    corrected_success_count += 1
            else:
                hard_failure_count += 1

            escaped = diverged_cubes()
            if escaped:
                # See demo2.py's own identical comment: every later step would be
                # planned against a nonsense target pose.
                simulation_diverged = True
                print(
                    f"[warning] simulation DIVERGED: {', '.join(escaped)} left the "
                    f"workspace -- abandoning the rest of iteration {iteration}"
                )
                break
            if not outcome.final_succeeded:
                # Every later step stacks onto this cube, so continuing would only
                # pile onto a cube that is lying on the floor.
                print(
                    f"[warning] {cube_to_pick.name.name} ended up on the floor after "
                    f"correction attempts were exhausted -- abandoning the rest of "
                    f"iteration {iteration}"
                )
                break
            time.sleep(1)

        print_iteration_summary(iteration)
        append_inference_report(iteration, cube_outcomes, simulation_diverged)
        if not simulation_diverged and segmind_approved():
            successful_iterations += 1

        iteration_durations.append(time.time() - iteration_start)
        average_duration = sum(iteration_durations) / len(iteration_durations)
        print(
            f"=== iteration {iteration}/{NUMBER_OF_ITERATIONS} took "
            f"{iteration_durations[-1]:.1f}s (average so far: {average_duration:.1f}s) ==="
        )

print("--- final positions ---")
print_positions()
print(
    f"[summary] {successful_iterations}/{NUMBER_OF_ITERATIONS} iterations fully stacked, "
    f"{corrected_success_count} cube(s) recovered via causal correction, "
    f"{hard_failure_count} cube(s) hard-failed after exhausting corrections"
)
print_and_append_diagnosis_summary(diagnosis_outcomes)

print("Plan finished, keeping the viewer open until it is closed")
while multi_sim.simulator.renderer.is_running():
    time.sleep(0.1)
