"""
Build the Montessori shape-sorting world and, if ROS 2 is available, a robot (see
:data:`DEFAULT_ROBOT_CLASS`) in a semantic digital twin world, visualize it live in
RViz, have the robot sort every loose shape into its matching hole (physically settling
each one under gravity in MuJoCo right after it is placed, rather than leaving it
exactly where it was kinematically teleported to), and finally physically simulate the
finished scene live in MuJoCo.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.montessori.montessori_demo
    python -m experiments.montessori.montessori_demo --headless

.. note::
    ROS 2 (``rclpy``) is optional: without it, the scene is still built and viewable
    through :class:`~experiments.montessori.world.MontessoriWorld` directly, but RViz
    visualization is skipped and, since sorting the shapes requires the ROS-dependent
    CRAM/Giskard motion stack, no robot is spawned, no shapes are inserted, and MuJoCo
    is not started, regardless of whether :data:`DEFAULT_ROBOT_CLASS`'s description is
    installed. With ``rclpy`` installed, add a ``MarkerArray`` display in RViz2 for the
    topic printed at startup, with ``DurabilityPolicy.TRANSIENT_LOCAL``, to see the
    scene. The ``mujoco`` dependency (declared by ``semantic_digital_twin``; run
    ``uv sync`` once from the repository root if it is not yet installed) is required
    either way. :data:`DEFAULT_ROBOT_CLASS` additionally requires its own ROS package
    (e.g. ``hsr_description`` for :class:`~semantic_digital_twin.robots.hsrb.HSRB`) to
    be built and sourced.
"""

from __future__ import annotations

import argparse
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco

from typing_extensions import Mapping, Optional, Tuple, Type

from experiments.montessori.semantics import MontessoriShape, NoMatchingHoleError
from experiments.montessori.world import MontessoriWorld, robot_installed
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.utils import clear_memoization_cache
from semantic_digital_twin.adapters.multi_sim import (
    MujocoActuator,
    MujocoBody,
    MujocoGeom,
    MujocoJoint,
    MujocoSim,
    ReparentingMode,
)
from coraplex.datastructures.enums import Arms
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.utils import rclpy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Actuator

if TYPE_CHECKING:
    # coraplex.datastructures.dataclasses pulls in rclpy at module level (see
    # _insert_all_shapes), so this is only ever imported for type hints, never at
    # runtime.
    from coraplex.datastructures.dataclasses import Context

logger = logging.getLogger(__name__)

DEFAULT_ROBOT_CLASS: Type[AbstractRobot] = HSRB
"""
The robot spawned by :func:`main` into the Montessori scene, via
:meth:`~experiments.montessori.world.MontessoriWorld.spawn_robot`.
"""

ACTUATOR_TIME_CONSTANT = 0.1
"""
MuJoCo actuator ``dynamics_parameters[0]`` for the position-hold actuators added to the
robot's joints.
"""

ARM_ACTUATOR_POSITION_GAIN = 1000.0
"""
Proportional gain of the MuJoCo position-servo actuators driving the robot's controlled
joints (arm, torso lift, wrist, head).

A position servo settles with a steady-state error of roughly the residual torque
divided by this gain, and that error has to stay under ``JointPositionList``'s 0.01 rad
convergence threshold or a motion never registers as finished and the plan behind it
never starts. Measured worst-joint error against this gain: 0.044 rad at 100, 0.007 at
500, 0.004 here. Peak torque is unchanged, being clamped by
:data:`JOINT_FORCE_RANGE` rather than by the gain.
"""

ARM_ACTUATOR_VELOCITY_GAIN = 100.0
"""
Derivative (damping) gain of the MuJoCo position-servo actuators driving the robot's
controlled joints, kept at a tenth of :data:`ARM_ACTUATOR_POSITION_GAIN`.
"""

GRIPPER_ACTUATOR_POSITION_GAIN = ARM_ACTUATOR_POSITION_GAIN
"""
Proportional gain of the MuJoCo position-servo actuator driving the gripper.

The sustained squeeze on a grasped shape does not come from this gain alone: once the
fingers stall against the shape,
:meth:`~semantic_digital_twin.adapters.multi_sim.MujocoSynchronizer._integrate_desired_position`
keeps advancing the servo's setpoint past the contact, so the closing force builds up
instead of settling at the contact surface.
"""

GRIPPER_ACTUATOR_VELOCITY_GAIN = ARM_ACTUATOR_VELOCITY_GAIN
"""
Derivative (damping) gain of the MuJoCo position-servo actuator driving the gripper.
"""

ARM_JOINT_ARMATURE = 0.1
"""
Rotor inertia (:attr:`JointDynamics.armature`) added to every physically simulated joint
of the arm, torso lift, head and gripper.

The actuators' velocity feedback is integrated explicitly, so a joint only stays stable
while ``ARM_ACTUATOR_VELOCITY_GAIN * MUJOCO_STEP_SIZE`` remains below its inertia. The
distal joints -- the wrist and arm rolls above all -- carry far less than that, and
without this they visibly shake in place instead of holding still. Chosen as five times
that product; measured peak-to-peak movement while holding a pose drops from 0.0018 rad
to 0.00001 rad, with no loss of tracking accuracy.
"""

JOINT_FORCE_RANGE = [-100.0, 100.0]
"""
Force clamp of every position-servo actuator added to the robot, taken from the effort
limit the HSR's own URDF declares for all of these joints.

Without a force limit, a stiff servo acting on a light link can produce an enormous
instantaneous torque and diverge numerically.
"""

GRIPPER_SPRING_STIFFNESS = 10.0
"""
Spring stiffness given to the gripper's passive joints, which no actuator drives.

These are the compliance that lets the fingers conform to a grasped object, but the
robot's description declares only their damping, so simulating them yields free hinges
that flop about under gravity rather than springs that return to rest. Stiff enough to
carry a finger's own weight near its rest position, soft enough to still deflect under a
grasp.
"""

GRIPPER_FRICTION = [1.0, 0.5, 0.5]
"""
Contact friction (sliding, torsional, rolling) given to the gripper's finger geometry,
matching what the shapes themselves carry (see
:data:`~experiments.montessori.world.GRASP_FRICTION`).

Both sides of a contact need it: MuJoCo combines the two geoms' friction, so leaving the
fingers at the near-zero default torsional/rolling values lets a grasped shape spin and
roll out of them however firmly they squeeze.
"""

BASE_JOINT_DAMPING = 50.0
"""
Viscous friction (:attr:`JointDynamics.damping`) added to the mobile base's wheel
joints, resisting spinning regardless of the (weak) position-hold actuator.
"""

BASE_JOINT_DRY_FRICTION = 5.0
"""
Dry friction (:attr:`JointDynamics.dry_friction`) added to the mobile base's wheel
joints, resisting spinning under gravity and floor contacts.
"""

BASE_JOINT_ARMATURE = 0.02
"""
Rotor inertia (:attr:`JointDynamics.armature`) added to the mobile base's wheel joints.

MuJoCo integrates joint damping explicitly, so a joint stays stable only while
``damping * step_size / inertia`` remains below 1. A caster wheel's own inertia is
several orders of magnitude too small for :data:`BASE_JOINT_DAMPING` at
:data:`MUJOCO_STEP_SIZE`, and its acceleration diverges on the very first step without
this. Chosen as twice ``BASE_JOINT_DAMPING * MUJOCO_STEP_SIZE``, putting that ratio at
one half.
"""

MUJOCO_STEP_SIZE = 5e-4
"""
MuJoCo simulation step size used for the finished scene.

Has to be large enough for this scene to simulate at real time, because the plan is
executed with real-time pacing: the control loop then advances its trajectory against
the wall clock, so a simulation running slower than that leaves the arm progressively
further behind its commanded position until the motion is abandoned as unconverged.
Measured on this scene: 0.59x real time at ``2e-4``, 1.0x from ``5e-4`` up. The lower
step was originally needed because the wheel joints' low inertia drove ``QACC`` to
``NaN``, which the armature they now carry (:data:`BASE_JOINT_ARMATURE`) addresses
directly instead.
"""

MAX_INSERTION_ATTEMPTS = 3
"""
Number of times :func:`_insert_all_shapes` tries inserting a single shape (see
:data:`RETRY_HORIZONTAL_JITTER`) before giving up and logging a warning.
"""

RETRY_HORIZONTAL_JITTER = 0.003
"""
Maximum magnitude, along either axis, of the random horizontal offset (:attr:`~experimen
ts.montessori.insert_shape_action.InsertMontessoriShapeAction.target_horizontal_offset`)
applied to a retried insertion's drop point.

A retry that teleports the shape to the exact same pose and re-settles it in MuJoCo
gives the physics engine no new information, so it is prone to failing the same way
again; a few millimeters of jitter, small next to every hole's own clearance margin
(see :data:`~experiments.montessori.world.SHAPE_FOOTPRINT_CLEARANCE_SCALE`), is enough
to change how the shape first contacts the hole's edge without missing the opening
outright.
"""


def _random_horizontal_jitter() -> Point3:
    """
    A random ``(x, y, 0)`` offset within :data:`RETRY_HORIZONTAL_JITTER` of the origin,
    for :func:`_insert_all_shapes` to retry a failed insertion with an actually
    different drop point.
    """
    return Point3(
        random.uniform(-RETRY_HORIZONTAL_JITTER, RETRY_HORIZONTAL_JITTER),
        random.uniform(-RETRY_HORIZONTAL_JITTER, RETRY_HORIZONTAL_JITTER),
        0.0,
    )


@dataclass(frozen=True)
class InsertionAttemptResult:
    """
    Outcome of a single :func:`_insert_shape` call.
    """

    target_horizontal_offset: Point3
    """
    The horizontal offset the attempt was actually released at (see :attr:`~experiments.
    montessori.insert_shape_action.InsertMontessoriShapeAction.target_horizontal_offset`
    ), whether given by the caller or generated internally.
    """

    fell_through_hole: bool
    """
    Whether the shape actually fell through its hole after settling; see :meth:`~experim
    ents.montessori.insert_shape_action.InsertMontessoriShapeAction.has_fallen_through_h
    ole`.
    """


def _insert_shape(
    shape: MontessoriShape,
    montessori: MontessoriWorld,
    context: Context,
    target_horizontal_offset: Optional[Point3] = None,
) -> InsertionAttemptResult:
    """
    Have the robot pick up and insert a single loose shape into its matching hole once,
    then wait for it to physically come to rest.

    Both the pick-and-place and the drop happen inside the already-running simulation:
    the shape is held by the fingers' friction while carried, and falls through the hole
    on its own once released, rather than being teleported into place.

    Runs with Giskard's collision avoidance off
    (:data:`~coraplex.execution_environment.simulated_robot`'s
    ``collision_avoidance`` default): with it on, even the pre-grasp hover pose (well
    clear of the table) consistently failed to converge for this scene, regardless of
    how far the standing pose or collision buffer were tuned, pointing at the QP
    solver's own performance under collision avoidance rather than at genuinely
    unreachable geometry (see :func:`_enable_robot_table_collision_avoidance`, no
    longer called here). Any resulting table/board interpenetration during the
    kinematic reach is corrected once each shape settles under gravity in MuJoCo.

    :param shape: The shape to insert; must have a matching hole (see
        :meth:`~experiments.montessori.semantics.ShapeSortingBoard.hole_for`).
    :param montessori: The Montessori scene, with :attr:`MontessoriWorld.robot` already
        spawned and equipped (see :func:`_equip_robot_for_physical_simulation`), inside
        a running simulation (see :func:`_start_physical_simulation`).
    :param context: The CRAM execution context to run the insertion action in.
    :param target_horizontal_offset: Horizontal offset to release the shape at; a
        random :func:`_random_horizontal_jitter` is used if not given.
    :return: The attempt's outcome.
    """
    from coraplex.datastructures.enums import Arms, ExecutionType
    from coraplex.execution_environment import ExecutionEnvironment
    from coraplex.plans.factories import execute_single
    from experiments.montessori.insert_shape_action import InsertMontessoriShapeAction

    offset = target_horizontal_offset or _random_horizontal_jitter()

    # World.get_kinematic_structure_entities_of_branch is @memoize'd per
    # (world, root-object) and never invalidated across the attach/detach cycle
    # of the *previous* insertion's pick-and-place, so the next action's
    # gripper-contents query silently returns a stale branch. Clearing it before
    # each insertion forces a fresh read of the actual current world state.
    clear_memoization_cache(montessori.world)
    action = InsertMontessoriShapeAction(
        montessori_shape=shape,
        board=montessori.board,
        arm=Arms.RIGHT,
        target_horizontal_offset=offset,
    )
    # real_time_pacing keeps the control loop advancing at the same rate as the physics
    # it is driving; without it the planner runs far ahead of the simulation and
    # commands joint positions the arm has had no wall-clock time to physically reach.
    with ExecutionEnvironment(
        execution_type=ExecutionType.SIMULATED,
        collision_avoidance=False,
        real_time_pacing=True,
    ):
        node = execute_single(action, context=context)
        node.perform()

    logger.info("Letting %s settle.", shape.name)
    time.sleep(SHAPE_SETTLE_DURATION)
    montessori.world.update_forward_kinematics()

    return InsertionAttemptResult(offset, action.has_fallen_through_hole())


def _insert_shape_or_none(
    shape: MontessoriShape,
    montessori: MontessoriWorld,
    context: Context,
    attempt: int,
) -> Optional[InsertionAttemptResult]:
    """
    Attempt one insertion via :func:`_insert_shape`, returning ``None`` instead of
    letting a retryable failure propagate: either
    :class:`~semantic_digital_twin.exceptions.PointOccupiedError`, if this attempt's
    jittered drop point put the reach target outside the navigation map's free space; a
    :class:`~coraplex.plans.failures.PlanFailure` (e.g. ``EmptyUnderspecified`` if no
    standing offset/grasp satisfies every constraint, or ``MotionDidNotFinish`` if a
    motion failed while executing), while actually reaching for and grasping the shape;
    or a :class:`~giskardpy.motion_statechart.exceptions.CollisionViolatedError`, raised
    when the gripper's approach to a shape resting on the table marginally breaches
    :data:`GRIPPER_TABLE_BUFFER_ZONE_DISTANCE`, since the standing offset
    :meth:`~experiments.montessori.insert_shape_action.InsertMontessoriShapeAction._move_to_reach`
    resolves is not itself constrained to avoid that. None of these indicate a
    fundamentally unreachable shape, only that this specific attempt's resolved
    standing offset or approach did not work out, so retrying (with a jittered drop
    point, changing what gets resolved next) is as reasonable as it is for
    ``PointOccupiedError``.

    :param shape: The shape to insert; must have a matching hole.
    :param montessori: The Montessori scene, with :attr:`MontessoriWorld.robot` already
        spawned and equipped, inside a running simulation.
    :param context: The CRAM execution context to run the insertion action in.
    :param attempt: This attempt's 1-based index, used only for the log message.
    :return: The attempt's outcome, or ``None`` if this attempt failed in a retryable
        way.
    """
    from coraplex.plans.failures import PlanFailure
    from giskardpy.motion_statechart.exceptions import CollisionViolatedError

    try:
        return _insert_shape(shape, montessori, context)
    except (PointOccupiedError, PlanFailure, CollisionViolatedError) as error:
        logger.warning(
            "%s's insertion attempt %d/%d failed (%s); retrying.",
            shape.name,
            attempt,
            MAX_INSERTION_ATTEMPTS,
            error,
        )
        return None


ROBOT_TABLE_BUFFER_ZONE_DISTANCE = 0.02
"""
Buffer-zone distance (see
:attr:`~semantic_digital_twin.collision_checking.collision_rules.AvoidCollisionRule.buffer_zone_distance`)
used for the robot against the table, instead of the robot's default 5cm
(:meth:`~semantic_digital_twin.robots.hsrb.HSRB._setup_collision_rules`).

A grasp pose sits at the target shape's own center, a couple of centimeters above the
table at most for every Montessori shape (see
:meth:`~experiments.montessori.world.MontessoriWorld._resting_position_on_table`); the
default 5cm buffer keeps the whole robot no closer than that to any external body,
which not just the gripper but the wrist and forearm too cannot honor while actually
reaching down to grasp a shape resting on the table, so the reach never leaves its
pre-grasp hover. Narrowing the buffer for the whole robot (not just the gripper; the
lower arm needs to get close to the table too) against only the table (not the shapes
or board, already excluded entirely below) keeps a real, if smaller, standoff instead
of removing avoidance outright.
"""


def _enable_robot_table_collision_avoidance(montessori: MontessoriWorld) -> None:
    """
    Restrict Giskard's collision avoidance, enabled by ``collision_avoidance=True`` on
    :data:`~coraplex.execution_environment.simulated_robot` in :func:`_insert_shape`, to
    the robot and the table, and let the robot approach the table more closely than its
    default standoff.

    .. note::
        :func:`_insert_shape` does not use this and still runs with collision avoidance
        off, since even with this narrowing the reach failed to converge to the pre-grasp
        hover pose from the stance that demo spawns at.
        :func:`~experiments.montessori.montessori_demo_mujoco._run_for_shape` does use
        it, from a stance that lines the arm rather than the base up with its target.

    The robot's own default
    :class:`~semantic_digital_twin.collision_checking.collision_rules.AvoidExternalCollisions`
    rules (added when it is spawned, e.g. :meth:`~semantic_digital_twin.robots.hsrb.HSRB.__post_init__`)
    already check it against every other collision body in the world once that flag is
    on, including the shape-sorting board's ~40-50-piece CoACD decomposition, which
    overloads Giskard's QP solver (a convergence timeout, not a detected collision) for
    the tight-clearance pickup motion. Force-excluding everything but the table, rather
    than adding another avoid-rule on top of those, is what actually narrows the
    checked set: :attr:`~semantic_digital_twin.collision_checking.collision_manager.CollisionManager.ignore_collision_rules`
    are applied last and cannot be overwritten by the broader default rules already in
    place.

    A second, narrower :class:`~semantic_digital_twin.collision_checking.collision_rules.AvoidCollisionBetweenGroups`
    rule is added as a default rule for the robot against the table specifically, with
    :data:`ROBOT_TABLE_BUFFER_ZONE_DISTANCE` instead of the default 5cm:
    :meth:`~semantic_digital_twin.collision_checking.collision_manager.CollisionManager.get_buffer_zone_distance`
    scans rules most-recently-added first, so this rule (added after the robot's own
    default rule) wins for every robot-table pair.

    :param montessori: The Montessori scene, with :attr:`MontessoriWorld.robot` already
        spawned.
    """
    [table] = montessori.world.get_semantic_annotations_by_type(Table)
    robot_bodies = set(montessori.robot.bodies_with_collision)
    gripper_bodies = {
        body
        for end_effector in montessori.robot.get_end_effectors()
        for body in end_effector.bodies_with_collision
    }
    table_bodies = set(table.bodies_with_collision)
    other_bodies = (
        set(montessori.world.bodies_with_collision) - robot_bodies - table_bodies
    )
    with montessori.world.modify_world():
        montessori.world.collision_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(
                body_group_a=list(robot_bodies), body_group_b=list(other_bodies)
            )
        )
        # The fingers are exempt from the table check the rest of the robot keeps: a
        # shape rests on the table, so closing around it puts them at table level, and
        # holding them to the arm's standoff aborts the grasp on its last centimetre
        # with the fingers already at the shape.
        montessori.world.collision_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(
                body_group_a=list(gripper_bodies), body_group_b=list(table_bodies)
            )
        )
        montessori.world.collision_manager.add_default_rule(
            AvoidCollisionBetweenGroups(
                body_group_a=list(robot_bodies - gripper_bodies),
                body_group_b=list(table_bodies),
                buffer_zone_distance=ROBOT_TABLE_BUFFER_ZONE_DISTANCE,
            )
        )


def _insert_all_shapes(montessori: MontessoriWorld) -> None:
    """
    Have the robot pick up and insert every loose shape that has a matching hole into
    the shape-sorting board, skipping any that don't (e.g. the sphere), retrying a shape
    that does not actually fall through its hole (see :func:`_insert_shape`) up to
    :data:`MAX_INSERTION_ATTEMPTS` times with a jittered drop point before giving up on
    it.

    A retry picks the shape up from wherever it physically ended up, which is not
    necessarily where it started: a shape that bounced off the board or slipped out of
    the gripper has genuinely moved.

    :param montessori: The Montessori scene, with :attr:`MontessoriWorld.robot` already
        spawned and equipped (see :func:`_equip_robot_for_physical_simulation`), inside
        a running simulation (see :func:`_start_physical_simulation`).
    """
    # Imported lazily: coraplex.datastructures.dataclasses pulls in
    # coraplex.plans.executables for GiskardExecutable, which imports rclpy at module
    # level, so this whole chain would make the demo unimportable without ROS 2 even
    # though nothing here runs without it anyway (see rclpy_installed() in main()).
    from coraplex.datastructures.dataclasses import Context

    context = Context(
        montessori.world, montessori.robot, query_backend=ProbabilisticBackend()
    )

    for shape in montessori.world.get_semantic_annotations_by_type(MontessoriShape):
        try:
            montessori.board.hole_for(shape)
        except NoMatchingHoleError:
            logger.info("Skipping %s: no matching hole.", shape.name)
            continue

        for attempt in range(1, MAX_INSERTION_ATTEMPTS + 1):
            logger.info(
                "Inserting %s into its matching hole (attempt %d/%d).",
                shape.name,
                attempt,
                MAX_INSERTION_ATTEMPTS,
            )
            result = _insert_shape_or_none(shape, montessori, context, attempt)
            if result is not None and result.fell_through_hole:
                break
        else:
            logger.warning(
                "%s did not fall through its hole after %d attempts; it may be "
                "resting on the board or wedged in the opening.",
                shape.name,
                MAX_INSERTION_ATTEMPTS,
            )


@dataclass(frozen=True)
class JointServoTuning:
    """
    The gains and force clamp one joint's position servo is built with.
    """

    position_gain: float
    """
    Proportional gain of the servo.
    """

    velocity_gain: float
    """
    Derivative (damping) gain of the servo.
    """

    force_range: Tuple[float, float]
    """
    Force clamp of the servo.
    """


@dataclass(frozen=True)
class RobotActuatorTuning:
    """
    How a robot's joints are driven in simulation, per joint wherever they differ.

    Gains that hold one robot's links against gravity make another's oscillate, and a
    single robot's own model often asks for different gains along the arm -- the shoulder
    carrying far more than the wrist. Reading them from the robot rather than assuming one
    setting is what keeps a commanded pose actually held.
    """

    default: JointServoTuning
    """
    Tuning for every joint not named in :attr:`by_joint_name`.
    """

    by_joint_name: Mapping[str, JointServoTuning] = field(default_factory=dict)
    """
    Tuning for individual joints, by joint name.
    """

    def for_joint(self, joint_name: str) -> JointServoTuning:
        """
        The tuning a joint's servo is built with.

        :param joint_name: Name of the joint.
        :return: Its own tuning, or :attr:`default` if it has none.
        """
        return self.by_joint_name.get(joint_name, self.default)


def _servo_tuning_for(
    dof: DegreeOfFreedom,
    gripper_dofs: set[DegreeOfFreedom],
    actuator_tuning: Optional[RobotActuatorTuning],
) -> JointServoTuning:
    """
    The tuning ``dof``'s servo is built with.

    Falls back to this module's own constants, which are the HSR's, when the caller names
    no tuning of its own.

    :param dof: The degree of freedom being driven.
    :param gripper_dofs: The degrees of freedom the gripper drives.
    :param actuator_tuning: The caller's tuning, or ``None`` for the HSR's.
    :return: The gains and force clamp to build the servo with.
    """
    if actuator_tuning is not None:
        return actuator_tuning.for_joint(dof.name.name)
    if dof in gripper_dofs:
        return JointServoTuning(
            GRIPPER_ACTUATOR_POSITION_GAIN,
            GRIPPER_ACTUATOR_VELOCITY_GAIN,
            tuple(JOINT_FORCE_RANGE),
        )
    return JointServoTuning(
        ARM_ACTUATOR_POSITION_GAIN,
        ARM_ACTUATOR_VELOCITY_GAIN,
        tuple(JOINT_FORCE_RANGE),
    )


def _position_servo_actuator(
    position_gain: float,
    velocity_gain: float,
    force_range: Optional[list[float]] = None,
) -> MujocoActuator:
    """
    Build a MuJoCo actuator that servos its degree of freedom to a commanded position
    with a PD law, resisting gravity and contacts.

    The commanded position is whatever the world-model sync last wrote into the
    actuator's ``ctrl`` setpoint, so a joint driven by one of these follows the motion
    planner physically rather than being teleported to it.

    :param position_gain: Proportional gain of the servo.
    :param velocity_gain: Derivative (damping) gain of the servo.
    :param force_range: Force clamp, unlimited when omitted.
    """
    actuator_properties = MujocoActuator(
        dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
        dynamics_parameters=[ACTUATOR_TIME_CONSTANT] + [0.0] * 9,
        gain_type=mujoco.mjtGain.mjGAIN_FIXED,
        gain_parameters=[position_gain] + [0.0] * 9,
        bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
        bias_parameters=[0, -position_gain, -velocity_gain] + [0.0] * 7,
    )
    if force_range is not None:
        actuator_properties.force_limited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator_properties.force_range = list(force_range)
    return actuator_properties


def _add_position_servo_actuator(
    world: World,
    dof: DegreeOfFreedom,
    position_gain: float,
    velocity_gain: float,
    force_range: Optional[list[float]] = None,
) -> None:
    """
    Add a :func:`_position_servo_actuator` for ``dof`` to ``world``.

    :param world: The world to add the actuator to, modified in place.
    :param dof: The degree of freedom to drive.
    :param position_gain: Proportional gain of the servo.
    :param velocity_gain: Derivative (damping) gain of the servo.
    :param force_range: Force clamp, unlimited when omitted.
    """
    actuator = Actuator()
    actuator.add_dof(dof=dof)
    actuator.simulator_additional_properties.append(
        _position_servo_actuator(position_gain, velocity_gain, force_range)
    )
    world.add_actuator(actuator=actuator)


def _base_degrees_of_freedom_without_hardware_interface(
    robot: AbstractRobot,
) -> list[DegreeOfFreedom]:
    """
    The degrees of freedom of :func:`_base_connections_without_hardware_interface`.

    :param robot: The spawned robot.
    """
    return [
        connection.raw_dof
        for connection in _base_connections_without_hardware_interface(robot)
    ]


def _base_connections_without_hardware_interface(
    robot: AbstractRobot,
) -> list[ActiveConnection1DOF]:
    """
    The robot's mobile-base joints (drive wheels, passive caster wheels, base roll, ...)
    that have a degree of freedom but, unlike the arm/wrist/head, are not part of
    :attr:`AbstractRobot.degrees_of_freedom_with_hardware_interface`: they are driven
    indirectly through the :class:`OmniDrive` connection rather than controlled
    directly. Left unactuated, MuJoCo's contact and gravity forces spin them up without
    bound.

    Excludes every end effector's own uncontrolled (mimic/spring) joints, which already
    move together through their mimic relationship and don't need an independent hold.

    :param robot: The spawned robot.
    """
    controlled_dofs = set(robot.degrees_of_freedom_with_hardware_interface)
    gripper_dofs = {
        dof
        for end_effector in robot.get_end_effectors()
        for connection in end_effector.connections
        if isinstance(connection, ActiveConnection)
        for dof in connection.active_dofs
    }

    base_connections = []
    seen_dofs = set(controlled_dofs)
    for connection in robot.connections:
        if not isinstance(connection, ActiveConnection1DOF):
            continue
        if connection.raw_dof in seen_dofs or connection.raw_dof in gripper_dofs:
            continue
        seen_dofs.add(connection.raw_dof)
        base_connections.append(connection)
    return base_connections


def _gripper_drive_degrees_of_freedom(robot: AbstractRobot) -> list[DegreeOfFreedom]:
    """
    The degrees of freedom of every end effector that a MuJoCo actuator has to drive for
    the gripper to physically open and close.

    Read off the end effector's own declared joint states (the open/close configurations
    it commands), so only the joints the gripper actually drives get an actuator and its
    remaining joints are left to physics. On the HSR that distinction matters twice
    over: its four finger joints all mimic one motor joint, so they share a single
    :class:`DegreeOfFreedom` and one actuator on it drives all four (the mimic
    relationships are exported to MuJoCo as joint equality constraints), while its two
    compliant spring joints appear in no joint state at all and must stay free to
    deflect against a grasped object.

    A gripper with no declared joint states drives nothing, and cannot grasp.

    :param robot: The spawned robot.
    """
    driven_dofs = []
    for end_effector in robot.get_end_effectors():
        for joint_state in end_effector.joint_states:
            for connection in joint_state.connections:
                if connection.raw_dof not in driven_dofs:
                    driven_dofs.append(connection.raw_dof)
    return driven_dofs


def _gripper_degrees_of_freedom(robot: AbstractRobot) -> set[DegreeOfFreedom]:
    """
    Every degree of freedom of every end effector, driven or not.

    A superset of :func:`_gripper_drive_degrees_of_freedom`: the remainder are passive
    (e.g. the HSR's compliant spring joints), which physics rather than an actuator has
    to move, or their compliance is teleported away instead of deflecting against a
    grasped object.

    :param robot: The spawned robot.
    """
    return {
        connection.raw_dof
        for end_effector in robot.get_end_effectors()
        for connection in end_effector.connections
        if isinstance(connection, ActiveConnection1DOF)
    }


def _physically_simulated_degrees_of_freedom(
    robot: AbstractRobot,
) -> set[DegreeOfFreedom]:
    """
    Every degree of freedom that MuJoCo's actuator and contact model drives, rather than
    the world-model sync teleporting it (see
    :attr:`~semantic_digital_twin.adapters.multi_sim.MujocoSynchronizer.physically_simulated_dofs`).

    That is the arm, torso lift, head and the whole gripper. The mobile base is
    deliberately left out: it is driven through an :class:`OmniDrive` rather than by its
    wheels, and physically driving those wheels over floor contacts is a separate
    problem from the physical grasp this demo is about.

    :param robot: The spawned robot.
    """
    return set(
        robot.degrees_of_freedom_with_hardware_interface
    ) | _gripper_degrees_of_freedom(robot)


def _connections_driving(
    robot: AbstractRobot, degrees_of_freedom: set[DegreeOfFreedom]
) -> list[ActiveConnection1DOF]:
    """
    The robot's connections moved by any of ``degrees_of_freedom``.

    :param robot: The spawned robot.
    :param degrees_of_freedom: The degrees of freedom to find the connections of.
    """
    return [
        connection
        for connection in robot.connections
        if isinstance(connection, ActiveConnection1DOF)
        and connection.raw_dof in degrees_of_freedom
    ]


def _equip_robot_for_physical_simulation(
    robot: AbstractRobot,
    actuator_tuning: Optional[RobotActuatorTuning] = None,
) -> set[DegreeOfFreedom]:
    """
    Give the robot everything it needs to be driven by MuJoCo's physics rather than
    kinematically teleported, and report which of its degrees of freedom that covers.

    The arm, torso lift, head and the gripper's driven joint get position-servo
    actuators (:data:`ARM_ACTUATOR_POSITION_GAIN`,
    :data:`GRIPPER_ACTUATOR_POSITION_GAIN`) that track whatever the motion planner
    commands. The gripper's passive joints get none, but are still physically simulated,
    so their compliance actually deflects. Every physically simulated joint also gets
    :data:`ARM_JOINT_ARMATURE`, without which the distal ones shake in place rather than
    holding still. Every physically simulated link gets MuJoCo's
    own gravity compensation: without it each joint settles with a steady-state error
    from gravity sag alone, large enough to exceed ``JointPositionList``'s convergence
    threshold, so a motion merely holding the arm never registers as converged and the
    plan behind it never starts. The fingers additionally get the contact friction a
    friction-only grasp needs (:data:`GRIPPER_FRICTION`).

    The mobile base stays kinematic, so its wheels are given no actuator at all -- only
    joint damping and dry friction (:data:`BASE_JOINT_DAMPING`,
    :data:`BASE_JOINT_DRY_FRICTION`) to keep them from spinning up under gravity and
    floor contacts. Servoing them instead diverges immediately: their inertia is tiny
    next to an arm link's, and nothing commands them, so a position servo just fights
    the contact forces at whatever position they happen to be teleported to.

    :param robot: The spawned robot, modified in place.
    :param actuator_tuning: Gains and force clamps to drive the joints with. Defaults to
        this module's own constants, which are the HSR's; another robot's links need its
        own, or its arm oscillates around a pose it is merely asked to hold.
    :return: The degrees of freedom MuJoCo now drives, for
        :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`'s
        ``physically_simulated_dofs``.
    """
    physically_simulated_dofs = _physically_simulated_degrees_of_freedom(robot)
    gripper_dofs = set(_gripper_drive_degrees_of_freedom(robot))
    actuated_dofs = set(robot.degrees_of_freedom_with_hardware_interface) | gripper_dofs

    with robot._world.modify_world():
        for dof in sorted(actuated_dofs, key=lambda d: d.name.name):
            servo = _servo_tuning_for(dof, gripper_dofs, actuator_tuning)
            _add_position_servo_actuator(
                robot._world,
                dof,
                servo.position_gain,
                servo.velocity_gain,
                force_range=list(servo.force_range),
            )
        for connection in _connections_driving(robot, physically_simulated_dofs):
            connection.dynamics.armature = ARM_JOINT_ARMATURE
            connection.child.simulator_additional_properties.append(
                MujocoBody(gravitation_compensation_factor=1.0)
            )
        for connection in _connections_driving(robot, gripper_dofs):
            for shape in connection.child.collision:
                shape.simulator_additional_properties.append(
                    MujocoGeom(friction=list(GRIPPER_FRICTION))
                )
        passive_gripper_dofs = _gripper_degrees_of_freedom(robot) - gripper_dofs
        for connection in _connections_driving(robot, passive_gripper_dofs):
            connection.simulator_additional_properties.append(
                MujocoJoint(stiffness=[GRIPPER_SPRING_STIFFNESS, 0.0, 0.0])
            )
        for connection in _base_connections_without_hardware_interface(robot):
            connection.dynamics.damping = BASE_JOINT_DAMPING
            connection.dynamics.dry_friction = BASE_JOINT_DRY_FRICTION
            connection.dynamics.armature = BASE_JOINT_ARMATURE

    return physically_simulated_dofs


SHAPE_SETTLE_DURATION = 2.0
"""
Real-time seconds a just-released shape is given to physically fall and come to rest
before :meth:`~experiments.montessori.insert_shape_action.InsertMontessoriShapeAction.has_fallen_through_hole`
is asked whether it made it through its hole.

The simulation keeps running throughout, so this is a settling wait, not a separate
physics pass.
"""

SYNC_RATE_HZ = 100
"""
Rate at which the physically simulated joints' real, physics-driven positions are read
back into the world model.

Comfortably above Giskard's own control-loop rate, so it plans against current feedback
of where those joints have actually settled rather than stale readings.
"""


def _start_physical_simulation(
    montessori: MontessoriWorld,
    physically_simulated_dofs: set[DegreeOfFreedom],
    headless: bool,
) -> MujocoSim:
    """
    Start the single, long-running MuJoCo simulation the whole demo executes inside.

    The robot's arm and gripper are driven by their actuators against real contacts, and
    every loose shape is a free body, so a shape is carried only for as long as the
    fingers' friction actually holds it (see
    :attr:`~semantic_digital_twin.adapters.multi_sim.ReparentingMode.CONTACT_ONLY`) and
    falls through a hole because it physically fits, not because it was teleported there.

    The base is exported as movable joints
    (:attr:`~semantic_digital_twin.adapters.multi_sim.MultiSimBuilder.export_omni_drive_as_joints`).
    The robot reaches whole-body -- its mobile base is
    :attr:`~semantic_digital_twin.robots.robot_part_mixins.HasMobileBase.full_body_controlled`,
    so a reach is solved from the world root through the drive's x/y/yaw as if they were
    joints -- and a base welded at its build-time pose leaves the arm executing that
    solution alone, closing the gripper wherever it happens to land rather than on the
    shape.

    :param montessori: The Montessori scene, with :attr:`MontessoriWorld.robot` already
        equipped (see :func:`_equip_robot_for_physical_simulation`).
    :param physically_simulated_dofs: The degrees of freedom MuJoCo drives.
    :param headless: Whether to run without opening a MuJoCo viewer window.
    :return: The running simulation.
    """
    mujoco_sim = MujocoSim(
        world=montessori.world,
        headless=headless,
        step_size=MUJOCO_STEP_SIZE,
        physically_simulated_dofs=physically_simulated_dofs,
        sync_rate_hz=SYNC_RATE_HZ,
        reparenting_mode=ReparentingMode.CONTACT_ONLY,
        export_omni_drive_as_joints=True,
    )
    mujoco_sim.start_simulation()
    return mujoco_sim


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments selecting whether a MuJoCo viewer window is opened.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the MuJoCo simulation without opening a viewer window.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Build the Montessori world, visualize it in RViz, have the robot sort the loose
    shapes into the board, physically simulate the finished scene in MuJoCo, and keep
    the live viewer open until interrupted.
    """
    logging.basicConfig(level=logging.INFO)
    arguments = _parse_arguments()

    montessori = MontessoriWorld()

    physically_simulated_dofs: set[DegreeOfFreedom] = set()
    if robot_installed(DEFAULT_ROBOT_CLASS):
        montessori.spawn_robot(DEFAULT_ROBOT_CLASS)
        # Must happen before the simulation is built: the actuators, gravity
        # compensation and finger friction it adds are all baked into the MuJoCo model
        # at build time, not applied to a running one.
        physically_simulated_dofs = _equip_robot_for_physical_simulation(
            montessori.robot
        )
    else:
        logger.warning(
            "%s's description is not installed; spawning the Montessori scene "
            "without a robot.",
            DEFAULT_ROBOT_CLASS.__name__,
        )
    logger.info("Built Montessori world with %d bodies.", len(montessori.world.bodies))

    # Sorting the shapes goes through CRAM's execute_single, which pulls in
    # coraplex.plans.executables for GiskardExecutable, which imports rclpy at module
    # level regardless of whether a real robot ever executes a real ROS 2 motion; RViz
    # visualization needs rclpy directly. Both are skipped without it, rather than the
    # whole demo failing to even import.
    ros_active = rclpy_installed()
    node = executor = thread = tf_publisher = viz_marker_publisher = None
    if ros_active:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
        from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
            VizMarkerPublisher,
        )

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("montessori_demo")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(
            target=executor.spin, daemon=True, name="rclpy-executor"
        )
        thread.start()
        time.sleep(0.1)

        tf_publisher = TFPublisher(node=node, _world=montessori.world)
        viz_marker_publisher = VizMarkerPublisher(_world=montessori.world, node=node)
        logger.info(
            "Visualizing the Montessori world on topic '%s'.",
            viz_marker_publisher.topic_name,
        )
    else:
        logger.warning("rclpy is not installed; running without RViz visualization.")

    mujoco_sim = None
    if montessori.robot is not None and ros_active:
        import experiments.orm.ormatic_interface  # type: ignore

        mujoco_sim = _start_physical_simulation(
            montessori, physically_simulated_dofs, headless=arguments.headless
        )
        _insert_all_shapes(montessori)
        logger.info("Sorting done; the simulation keeps running.")
    elif montessori.robot is not None:
        logger.warning("rclpy is not installed; skipping sorting and MuJoCo.")

    logger.info("Done. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if mujoco_sim is not None:
            mujoco_sim.stop_simulation()
        if viz_marker_publisher is not None:
            viz_marker_publisher.stop()
        if tf_publisher is not None:
            tf_publisher.stop()
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if ros_active and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
