"""
Spawn a real HSRB and give it what it needs to hold still under MuJoCo's own physics,
for the standalone HSRB demos in this package.

The position-hold tuning here (gains, wheel damping/dry-friction) is the same recipe
:mod:`experiments.montessori.montessori_demo` already proved on the full Montessori
scene: every controlled joint (arm/wrist/head) gets a MuJoCo position-hold actuator, and
the mobile base's own drive/caster wheels -- which have a degree of freedom but no
:class:`~semantic_digital_twin.world_description.connections.OmniDrive`-level hardware
interface -- get joint damping and dry friction on top of a much weaker hold, since
their low inertia under the arm's actuator gain was observed to drive MuJoCo's ``QACC``
to ``NaN``/``Inf`` there. Kept here, rather than imported from that module, since these
functions are pure (only ever take an already-spawned
:class:`~semantic_digital_twin.robots.robot_parts.AbstractRobot` or
:class:`~semantic_digital_twin.world.World`) and this package's standalone HSRB demos
should not depend on ``montessori_demo``'s own module-level state (an ``rclpy`` node,
CRAM/Giskard imports) just to reuse them.
"""

from __future__ import annotations

import mujoco
from typing_extensions import ClassVar, Type

from semantic_digital_twin.adapters.multi_sim import (
    MujocoActuator,
    MujocoBody,
    MujocoBuilder,
    MujocoSim,
    MultiSimBuilder,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Actuator

ACTUATOR_TIME_CONSTANT = 0.1
"""
MuJoCo actuator ``dynamics_parameters[0]`` for every position-hold actuator added here.
"""

ARM_ACTUATOR_POSITION_GAIN = 400.0
"""
Proportional gain of the position-hold actuators added to the robot's controlled joints
(arm, wrist, head).

100 (matching :mod:`experiments.montessori.montessori_demo`'s own, RK4-tuned value) left
a real, if small, residual gravity-sag error even with gravity compensation and
``implicitfast`` (see :data:`~experiments.montessori.hsrb_park_demo.MUJOCO_INTEGRATOR`):
``wrist_flex_joint`` settled at its own ``-1.91`` rad joint limit instead of its
``-1.85`` rad park target. ``implicitfast`` tolerates a much stiffer hold than RK4 does,
so raising the gain closes that gap directly instead of needing yet another
gravity-compensation-adjacent workaround.
"""

ARM_ACTUATOR_VELOCITY_GAIN = 40.0
"""
Derivative (damping) gain of the position-hold actuators added to the robot's controlled
joints (arm, wrist, head).
"""

ARM_JOINT_ARMATURE = 0.1
"""
Rotor inertia (:attr:`~semantic_digital_twin.world_description.degree_of_freedom.JointDynamics.armature`)
added to the robot's controlled joints, matching :mod:`experiments.montessori.franka_panda_equipment`'s
own ``ARM_JOINT_ARMATURE`` for the same reason: the real HSRB wrist link's own physical
inertia is small enough that :data:`ARM_ACTUATOR_POSITION_GAIN` alone drove ``QACC`` to
``NaN`` within the first physics step: an actuator's effective proportional gain and a
joint's own inertia trade off against the step size for numerical stability, the same
relationship :data:`BASE_ACTUATOR_POSITION_GAIN` is kept low to satisfy on the low-
inertia wheels; armature raises the joint's own apparent inertia directly instead of
lowering the gain, keeping the arm's hold stiff.
"""

BASE_ACTUATOR_POSITION_GAIN = 1.0
"""
Proportional gain of the position-hold actuators added to the mobile base's wheel
joints.

Much lower than :data:`ARM_ACTUATOR_POSITION_GAIN`: the wheels have far less inertia
than an arm link, and holding them with the arm's gain makes the simulation numerically
unstable (``QACC`` diverges within the first few milliseconds and never settles).
"""

BASE_ACTUATOR_VELOCITY_GAIN = 0.1
"""
Derivative (damping) gain of the position-hold actuators added to the mobile base's
wheel joints.
"""

BASE_JOINT_DAMPING = 50.0
"""
Viscous friction (:attr:`~semantic_digital_twin.world_description.degree_of_freedom.JointDynamics.damping`)
added to the mobile base's wheel joints, resisting spinning regardless of the (weak)
position-hold actuator.
"""

BASE_JOINT_DRY_FRICTION = 5.0
"""
Dry friction (:attr:`~semantic_digital_twin.world_description.degree_of_freedom.JointDynamics.dry_friction`)
added to the mobile base's wheel joints, resisting spinning regardless of the (weak)
position-hold actuator.
"""


def disable_robot_self_collision(robot: AbstractRobot) -> None:
    """
    Strip every one of the robot's own bodies of collision geometry (their visual
    geometry is untouched), so MuJoCo generates no contacts for it at all -- against
    itself or against anything else in the scene.

    The real HSRB URDF's collision meshes were never authored with every non-adjacent
    body pair in mind: several genuinely overlap in ordinary poses this package's
    standalone demos actually reach (e.g. ``base_link``/``arm_lift_link`` at
    :class:`~semantic_digital_twin.datastructures.definitions.TorsoState`'s own
    ``LOW``), which the real robot's own SRDF self-collision matrix
    (:meth:`~semantic_digital_twin.robots.hsrb.HSRB._setup_collision_rules`) excludes
    for Giskard's kinematic collision *avoidance* during motion planning, but which
    nothing carries over into MuJoCo's own, separate physical collision groups. Observed
    to silently push a joint far from its commanded position with no instability
    warning at all: the contact resolves quietly, just not to the pose anything else in
    this scene expects.

    None of this package's standalone demos need real contact from anywhere on the
    robot: the base itself has no physical MuJoCo degree of freedom to begin with (see
    :func:`spawn_mobile_robot`'s own docstring), so it can neither fall through the
    floor nor be knocked off it regardless of whether its wheels still generate contact.

    :param robot: The spawned robot, modified in place.
    """
    with robot._world.modify_world():
        for body in robot.bodies_with_collision:
            body.collision = ShapeCollection([])


def weld_gripper(robot: AbstractRobot) -> None:
    """
    Rigidly weld every one of the gripper's own joints in place, replacing each with a
    :class:`~semantic_digital_twin.world_description.connections.FixedConnection` at its
    current relative pose (apply a :class:`~semantic_digital_twin.datastructures.definitions.GripperState`
    beforehand, e.g. ``GripperState.CLOSE``, to weld a deliberate pose rather than the
    URDF's own zero default).

    The real HSRB gripper is a single-motor, spring/mimic-linked underactuated
    mechanism: MuJoCo represents it as up to five separate joints per finger, some
    sharing one semantic degree of freedom via a multiplier/offset (see
    :func:`hold_controlled_joints_in_mujoco`'s own docstring) and the rest coupled only
    through a MuJoCo joint-equality constraint mirroring that mimic relationship. That
    combination -- multiple equality constraints referencing one physically reasonable,
    non-degenerate link -- was observed to drive ``QACC`` to ``NaN`` within the first
    physics step regardless of how that joint's own actuator or damping was tuned, with
    no clear culprit short of a deep dive into MuJoCo's own equality-constraint solver.
    None of the standalone HSRB demos in this package need the gripper to actually
    articulate, so welding it sidesteps the whole question rather than resolving it.

    :param robot: The spawned robot, modified in place.
    """
    world = robot._world
    gripper_connections = [
        connection
        for connection in robot.end_effector.connections
        if isinstance(connection, ActiveConnection1DOF)
    ]
    with world.modify_world():
        for connection in gripper_connections:
            parent_T_child = world.compute_forward_kinematics(
                connection.parent, connection.child
            )
            world.remove_connection(connection)
            world.add_connection(
                FixedConnection(
                    parent=connection.parent,
                    child=connection.child,
                    parent_T_connection_expression=parent_T_child,
                )
            )


def spawn_mobile_robot(
    world: World,
    robot_class: Type[AbstractRobot],
    position: Point3,
    yaw: float = 0.0,
) -> tuple[AbstractRobot, OmniDrive]:
    """
    Parse ``robot_class``'s real ROS description and merge it into ``world``, attached
    at its root with a real :class:`OmniDrive` connection.

    Mirrors :meth:`~experiments.montessori.world.MontessoriWorld.spawn_robot`'s own
    technique exactly (a real drive connection, not a plain 6DoF joint, since a mobile
    robot's navigation stack needs one to move it at all), generalized to any position
    rather than always standing in front of the Montessori table.

    :param world: The world to spawn the robot into, modified in place.
    :param robot_class: The robot to spawn, e.g. :class:`~semantic_digital_twin.robots.hsrb.HSRB`.
    :param position: Where the robot's root starts, in ``world``'s root frame.
    :param yaw: Which way the robot starts facing.
    :return: The spawned robot, and the :class:`OmniDrive` connection driving its base.
    """
    from semantic_digital_twin.adapters.urdf import URDFParser

    robot_world = URDFParser.from_file(robot_class.get_ros_file_path()).parse()
    with world.modify_world():
        drive = OmniDrive.create_with_dofs(
            parent=world.root, child=robot_world.root, world=world
        )
        world.merge_world(robot_world, drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=position.x, y=position.y, z=position.z, yaw=yaw, reference_frame=world.root
        )
    return robot_class.from_world(world), drive


def _position_hold_actuator(
    position_gain: float, velocity_gain: float
) -> MujocoActuator:
    """
    Build a MuJoCo actuator that holds its degree of freedom at whatever position it had
    when the simulation started, resisting gravity and contacts with a PD law.

    :param position_gain: Proportional gain of the hold.
    :param velocity_gain: Derivative (damping) gain of the hold.
    """
    return MujocoActuator(
        dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
        dynamics_parameters=[ACTUATOR_TIME_CONSTANT] + [0.0] * 9,
        gain_type=mujoco.mjtGain.mjGAIN_FIXED,
        gain_parameters=[position_gain] + [0.0] * 9,
        bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
        bias_parameters=[0, -position_gain, -velocity_gain] + [0.0] * 7,
    )


def _add_position_hold_actuator(
    world: World, dof: DegreeOfFreedom, position_gain: float, velocity_gain: float
) -> None:
    """
    Add a :func:`_position_hold_actuator` for ``dof`` to ``world``.

    :param world: The world to add the actuator to, modified in place.
    :param dof: The degree of freedom to hold.
    :param position_gain: Proportional gain of the hold.
    :param velocity_gain: Derivative (damping) gain of the hold.
    """
    actuator = Actuator()
    actuator.add_dof(dof=dof)
    actuator.simulator_additional_properties.append(
        _position_hold_actuator(position_gain, velocity_gain)
    )
    world.add_actuator(actuator=actuator)


def base_connections_without_hardware_interface(
    robot: AbstractRobot,
) -> list[ActiveConnection1DOF]:
    """
    The robot's mobile-base joints (drive wheels, passive caster wheels, base roll, ...)
    and end-effector joints (gripper fingers) that have a degree of freedom but, unlike
    the arm/wrist/head, are not part of
    :attr:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.degrees_of_freedom_with_hardware_interface`:
    they are driven indirectly, through the :class:`OmniDrive` connection or a MuJoCo
    joint-equality constraint mirroring the gripper's own mimic linkage, rather than
    controlled directly. Left unactuated, MuJoCo's contact and gravity forces spin or
    swing them up without bound.

    Includes every end effector joint, even ones a mimic/equality constraint already
    couples to another joint: only some of a gripper's own joints turn out to actually
    be constrained that way (on the real HSRB, its motor and two spring-loaded proximal
    joints have no equality driving them at all, only their own distal/mimic joints do),
    and giving every one of them the same weak hold this function already gives real
    wheels is simpler and just as safe as separating the truly free ones out.

    Excludes every zero-range dummy joint (e.g. the real HSRB URDF's bumper mounts and
    wrist force-torque sensor frame, each declared with an explicit ``lower == upper``
    limit and no meaningful inertia of their own): applying a position-hold actuator
    plus damping/dry-friction to a joint that cannot move anyway, on a body MuJoCo
    treats as having negligible mass, was observed to drive ``QACC`` to ``NaN`` within
    the first physics step.

    :param robot: The spawned robot.
    """
    controlled_dofs = set(robot.degrees_of_freedom_with_hardware_interface)

    base_connections = []
    seen_dofs = set(controlled_dofs)
    for connection in robot.connections:
        if not isinstance(connection, ActiveConnection1DOF):
            continue
        if connection.raw_dof in seen_dofs:
            continue
        lower = connection.raw_dof.limits.lower.position
        upper = connection.raw_dof.limits.upper.position
        if lower is not None and upper is not None and lower == upper:
            continue
        seen_dofs.add(connection.raw_dof)
        base_connections.append(connection)
    return base_connections


def hold_controlled_joints_in_mujoco(robot: AbstractRobot) -> None:
    """
    Keep every joint of the robot that would otherwise be left to MuJoCo's own physics
    (arm, wrist, head, the mobile base's wheels, and the gripper) from sagging, spinning,
    or swinging under gravity and contacts once MuJoCo starts stepping the world.

    The arm/wrist/head are held with a MuJoCo position-hold actuator
    (:data:`ARM_ACTUATOR_POSITION_GAIN`) and get :data:`ARM_JOINT_ARMATURE` added, for
    the same numerical-stability reason :data:`BASE_ACTUATOR_POSITION_GAIN` is kept low
    on the wheels and gripper below: a joint whose own physical inertia is small
    relative to its actuator's gain and the physics step size makes ``QACC`` diverge.
    The base's wheels and the gripper's own joints (see
    :func:`base_connections_without_hardware_interface`) additionally get joint damping
    and dry friction (:data:`BASE_JOINT_DAMPING`, :data:`BASE_JOINT_DRY_FRICTION`): their
    low inertia makes the arm's actuator gains numerically unstable, and a weak actuator
    (:data:`BASE_ACTUATOR_POSITION_GAIN`) alone is not enough to stop the wheels
    spinning once the robot has actually driven around and is resting at a real,
    contact-heavy pose rather than its spawn pose.

    :param robot: The spawned robot, modified in place.
    """
    with robot._world.modify_world():
        # Several joints -- HSRB's torso/arm both mark has_hardware_interface on
        # arm_lift_joint, and every one of the gripper's 9 URDF joints turns out to
        # share one of just 3 underlying DegreeOfFreedom objects with its mimic
        # siblings -- wrap the same raw_dof through more than one
        # ActiveConnection1DOF. MuJoCo still exports one real joint per *connection*,
        # not per raw_dof, so a stability property (damping/dry-friction/armature) set
        # on only the one connection this function happens to reach first would leave
        # every sibling connection's own exported joint -- including whichever one
        # :func:`_add_position_hold_actuator`'s single actuator actually ends up
        # driving -- with none, which was observed to still let ``QACC`` diverge on a
        # joint that looked, from this function's own perspective, already handled.
        # Grouping by raw_dof and setting every sibling avoids depending on which one
        # happens to be "first".
        connections_by_dof: dict = {}
        for connection in robot.connections:
            if isinstance(connection, ActiveConnection1DOF):
                connections_by_dof.setdefault(connection.raw_dof, []).append(connection)

        def _hold_dof(
            dof: DegreeOfFreedom, position_gain: float, velocity_gain: float
        ) -> None:
            for sibling in connections_by_dof.get(dof, []):
                sibling.dynamics.armature = ARM_JOINT_ARMATURE
            _add_position_hold_actuator(robot._world, dof, position_gain, velocity_gain)

        for dof in robot.degrees_of_freedom_with_hardware_interface:
            _hold_dof(dof, ARM_ACTUATOR_POSITION_GAIN, ARM_ACTUATOR_VELOCITY_GAIN)
            for sibling in connections_by_dof.get(dof, []):
                # Without this, each joint settles with a steady-state gravity-sag
                # error the position-hold actuator's own proportional gain alone
                # cannot close (matching franka_panda_equipment's own
                # ARM_JOINT_ARMATURE-adjacent gravity compensation, for the same
                # reason): observed to let wrist_flex_joint settle all the way at its
                # own ``-1.91`` rad joint limit instead of its ``-1.85`` rad park
                # target.
                sibling.child.simulator_additional_properties.append(
                    MujocoBody(gravitation_compensation_factor=1.0)
                )
        for connection in base_connections_without_hardware_interface(robot):
            for sibling in connections_by_dof.get(connection.raw_dof, []):
                sibling.dynamics.damping = BASE_JOINT_DAMPING
                sibling.dynamics.dry_friction = BASE_JOINT_DRY_FRICTION
            _add_position_hold_actuator(
                robot._world,
                connection.raw_dof,
                BASE_ACTUATOR_POSITION_GAIN,
                BASE_ACTUATOR_VELOCITY_GAIN,
            )


class _BalancedInertiaMujocoBuilder(MujocoBuilder):
    """
    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoBuilder`, with MuJoCo's
    ``balanceinertia`` compiler option turned on.

    A link's declared inertia can be off by enough numerical noise (typos, mesh-derived
    data rounded per component rather than jointly) to fail MuJoCo's physical-
    realizability check on its diagonalized principal moments, even where every raw
    diagonal entry alone looks plausible -- observed on several links of the real HSRB
    URDF (a rotated inertial frame with off-diagonal terms means the raw frame's
    diagonal entries satisfying the triangle inequality does not guarantee the
    diagonalized principal moments do too). ``balanceinertia`` corrects only such
    otherwise-rejected tensors to the nearest physically valid one; a scene with no such
    tensor compiles identically with or without it. Scoped to this module's own
    :class:`HSRBMujocoSim` rather than changed on the shared
    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoBuilder` itself, since that
    class is used far beyond this package.
    """

    def _start_build(self, file_path: str) -> None:
        super()._start_build(file_path)
        self.spec.compiler.balanceinertia = True


class HSRBMujocoSim(MujocoSim):
    """
    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`, built with
    :class:`_BalancedInertiaMujocoBuilder` instead of the default
    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoBuilder`, for scenes
    containing the real HSRB description (see :class:`_BalancedInertiaMujocoBuilder`'s
    own docstring for why it needs that).
    """

    builder_class: ClassVar[Type[MultiSimBuilder]] = _BalancedInertiaMujocoBuilder
