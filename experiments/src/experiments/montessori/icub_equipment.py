"""
Read the iCub3 out of its own ROS package description and equip it to be driven by
MuJoCo's own physics rather than kinematically teleported, the same way
:mod:`experiments.montessori.franka_panda_equipment` equips the Panda.

Unlike the Panda, the iCub3 has a real ROS package
(:meth:`~semantic_digital_twin.robots.icub3.ICub3.get_ros_file_path`,
``iai_icub_description``), so :func:`parse_icub` reads it through the usual
``package://`` resolution instead of a bundled MJCF. It is also a full legged humanoid
rather than a single arm: only its arms, hands, torso and neck have a hardware interface
(see :attr:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.degrees_of_freedom_with_hardware_interface`)
and are driven by a position servo; its legs are not driven at all -- the montessori
demo bolts the iCub3 to a fixed stance rather than navigating it there (see
:mod:`experiments.montessori.icub_montessori_demo`) -- so they instead get a weaker
position-hold actuator, matching how
:mod:`experiments.montessori.montessori_demo` holds the HSRB's undriven wheel joints,
so they do not sag or spin under gravity once MuJoCo starts stepping the world.

No MJCF with tuned ``<actuator>`` gains for the iCub3 exists in this repository or in
``iai_icub_description``, unlike the Panda's own ``mujoco_menagerie``-sourced
:data:`~experiments.montessori.franka_panda_equipment.PANDA_JOINT_SERVO_TUNING`. An
initial, entirely invented set of gains (one flat tuning for every joint, including the
fingers) turned out to be wrong in the most basic way: its force limit was two orders of
magnitude past what a real actuator this size could produce, and its position gain was
proportioned for a rigid industrial arm rather than a compliant humanoid, so it
oscillated instead of settling. :data:`ARM_JOINT_SERVO_TUNING` and
:data:`TORSO_JOINT_SERVO_TUNING`'s own force ranges are instead read off two real,
comparably-scaled MuJoCo humanoid actuator models found locally under
``~/dev/Multiverse-Resources/robots`` -- Unitree's G1 (``unitree/mjcf/g1/g1_29dof.xml``,
a similarly lightweight research/service humanoid, unlike the much heavier full-size
Unitree H1) for the arm and torso, and IIT's own iCub reference scene
(``iit/iCub/iCub_only_primitives_with_ref_position.xml``) for the fingers, the only one
of the two with actual position-servo ``kp``/``kv`` gains rather than bare torque
actuators. Position/velocity gains for everything but the fingers are still this
module's own heuristic (see :func:`_heuristic_tuning`), since neither reference source
gives a built-in position servo for a torque-controlled joint; both are still first-pass
and unverified beyond the diagnostics :func:`equip_icub_for_physical_simulation`'s own
docstring describes.
"""

from __future__ import annotations

import re

import mujoco
from typing_extensions import Optional

from experiments.montessori.franka_panda_equipment import JointServoTuning
from semantic_digital_twin.adapters.multi_sim import MujocoActuator, MujocoBody
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.icub3 import ICub3, ICub3FixedBase
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Actuator


def _heuristic_tuning(force_range: float) -> JointServoTuning:
    """
    Build a :class:`~experiments.montessori.franka_panda_equipment.JointServoTuning`
    from just a force range, for a joint whose force range is read off a real reference
    actuator (see the module docstring) but whose position/velocity gains are not: a
    position gain of 10x the force range saturates the servo at roughly a tenth of a
    radian of tracking error, comfortably inside every one of the iCub3's own joint
    ranges, with the velocity gain kept at a tenth of the position gain throughout, the
    same ratio the real iCub finger reference data itself uses (see
    :data:`FINGER_JOINT_SERVO_TUNING_BY_INDEX`).

    :param force_range: Symmetric force/torque limit (N or N*m) for the joint.
    """
    return JointServoTuning(
        position_gain=force_range * 10.0,
        velocity_gain=force_range,
        force_range=(-force_range, force_range),
    )


TORSO_JOINT_SERVO_TUNING = _heuristic_tuning(88.0)
"""
Tuning for the torso's own three joints (``torso_roll``/``torso_pitch``/``torso_yaw``).

Force range matches Unitree G1's ``waist_yaw`` actuator (88 N*m; see the module
docstring), the closest real-hardware analogue to a humanoid's own torso joint among the
locally available reference models.
"""

ARM_JOINT_SERVO_TUNING = _heuristic_tuning(25.0)
"""
Tuning for the arms' own "big" joints: every shoulder joint, the elbow, and the wrist's
forearm-rotation joint (``*_wrist_prosup``).

Force range matches Unitree G1's shoulder and elbow actuators (25 N*m each; see the
module docstring) -- two orders of magnitude below this module's own first, invented
150-2000 N*m guesses.
"""

WRIST_JOINT_SERVO_TUNING = _heuristic_tuning(5.0)
"""
Tuning for the wrist's other two joints (``*_wrist_pitch``, ``*_wrist_yaw``) and the neck
(``neck_roll``/``neck_pitch``/``neck_yaw``).

Force range matches Unitree G1's own ``wrist_pitch``/``wrist_yaw`` actuators (5 N*m
each; see the module docstring), the smallest, most precision-oriented joints it
declares a torque limit for. No comparable reference exists for a neck, but it carries
similarly little load (just the head), so the same tuning is reused for it.
"""

EYE_JOINT_SERVO_TUNING = _heuristic_tuning(0.1)
"""
Tuning for the two eye-pan joints and ``eyes_tilt``.

No reference actuator exists for a camera gimbal this light; matches
:data:`FINGER_JOINT_SERVO_TUNING_BY_INDEX`'s own lightest (distal fingertip) tier, the
closest real-hardware analogue to an eye's own negligible mass and inertia among the
locally available reference models.
"""

FINGER_JOINT_SERVO_TUNING_BY_INDEX: tuple[JointServoTuning, ...] = (
    JointServoTuning(position_gain=100.0, velocity_gain=10.0, force_range=(-0.1, 0.1)),
    JointServoTuning(position_gain=80.0, velocity_gain=8.0, force_range=(-0.1, 0.1)),
    JointServoTuning(position_gain=60.0, velocity_gain=6.0, force_range=(-0.1, 0.1)),
    JointServoTuning(position_gain=40.0, velocity_gain=4.0, force_range=(-0.1, 0.1)),
)
"""
Tuning for every finger joint, indexed by its position along the finger (0 = knuckle
closest to the palm, 3 = fingertip; see :func:`_finger_joint_index`).

Read verbatim off IIT's own real iCub reference scene (see the module docstring's own
``iCub_only_primitives_with_ref_position.xml``), the only joints among the locally
available reference models with an actual manufacturer-tuned MuJoCo position servo
rather than a bare torque actuator.
"""

LEG_ACTUATOR_POSITION_GAIN = 300.0
"""
Proportional gain of the position-hold actuator given to every leg/base joint that has
no hardware interface (see :func:`_leg_connections_without_hardware_interface`).

Higher than :mod:`~experiments.montessori.montessori_demo`'s own
``BASE_ACTUATOR_POSITION_GAIN`` (1.0, tuned for a wheel's low inertia): the iCub3's leg
links carry the whole standing robot's weight, and a wheel-scale gain lets the hips sag
under it.
"""

LEG_ACTUATOR_VELOCITY_GAIN = 30.0
"""
Derivative (damping) gain paired with :data:`LEG_ACTUATOR_POSITION_GAIN`.
"""

LEG_ACTUATOR_FORCE_RANGE = 140.0
"""
Force limit of the leg/base position-hold actuator, bounding what would otherwise be
MuJoCo's own default unlimited torque authority. Matches Unitree G1's own knee actuator
(139 N*m, rounded up; see the module docstring), the heaviest-loaded joint of the two
reference models' own leg actuators, since the iCub3's legs (unlike the arms) are only
ever held near their mounted stance rather than tracking a trajectory, and so are not
otherwise scaled per-joint the way :data:`ARM_JOINT_SERVO_TUNING` and
:data:`TORSO_JOINT_SERVO_TUNING` are.
"""

JOINT_ARMATURE = 0.1
"""
Rotor inertia (:attr:`~semantic_digital_twin.world_description.connection_properties.JointDynamics.armature`)
added to every one of the iCub3's joints, servoed or held alike, matching
:data:`~experiments.montessori.franka_panda_equipment.ARM_JOINT_ARMATURE`'s own value:
no iCub3-specific figure is known, and the Panda's own real-hardware armature is a
closer starting point than MuJoCo's unset (zero) default.
"""

_FINGER_JOINT_INDEX_PATTERN = re.compile(r"_(\d)_joint$")


def parse_icub() -> World:
    """
    Read the iCub3 out of its ``iai_icub_description`` ROS package, without any
    actuator: an actuator parsed into one world cannot be merged into another (see
    :func:`~experiments.montessori.world.mount_stationary_robot`), and
    :func:`equip_icub_for_physical_simulation` installs its own once the iCub3 is
    mounted.

    :return: A world holding only the iCub3's body tree.
    """
    icub_world = URDFParser.from_file(ICub3.get_ros_file_path()).parse()
    with icub_world.modify_world():
        for actuator in list(icub_world.actuators):
            icub_world.remove_actuator(actuator)
    return icub_world


def _position_servo_actuator(tuning: JointServoTuning) -> MujocoActuator:
    """
    Build a MuJoCo actuator that servos its degree of freedom to a commanded position
    with a PD law, resisting gravity and contacts; identical in shape to
    :func:`~experiments.montessori.franka_panda_equipment._position_servo_actuator`,
    duplicated rather than imported since it is a module-private implementation detail
    of that module.

    :param tuning: Gains and force clamp to build the servo with.
    """
    return MujocoActuator(
        dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
        dynamics_parameters=[1.0] + [0.0] * 9,
        gain_type=mujoco.mjtGain.mjGAIN_FIXED,
        gain_parameters=[tuning.position_gain] + [0.0] * 9,
        bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
        bias_parameters=[0.0, -tuning.position_gain, -tuning.velocity_gain] + [0.0] * 7,
        force_limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        force_range=list(tuning.force_range),
    )


def _position_hold_actuator(
    position_gain: float, velocity_gain: float, force_range: float
) -> MujocoActuator:
    """
    Build a MuJoCo actuator that holds its degree of freedom at whatever position it had
    when the simulation started, resisting gravity and contacts with a PD law; identical
    in shape to :func:`~experiments.montessori.montessori_demo._position_hold_actuator`,
    except for the added force limit (see :data:`LEG_ACTUATOR_FORCE_RANGE`), duplicated
    for the same reason as :func:`_position_servo_actuator`.

    :param position_gain: Proportional gain of the hold.
    :param velocity_gain: Derivative (damping) gain of the hold.
    :param force_range: Symmetric force/torque limit (N or N*m).
    """
    return MujocoActuator(
        dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
        dynamics_parameters=[0.1] + [0.0] * 9,
        gain_type=mujoco.mjtGain.mjGAIN_FIXED,
        gain_parameters=[position_gain] + [0.0] * 9,
        bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
        bias_parameters=[0, -position_gain, -velocity_gain] + [0.0] * 7,
        force_limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        force_range=[-force_range, force_range],
    )


def _add_actuator(world: World, dof: DegreeOfFreedom, actuator: MujocoActuator) -> None:
    """
    Add ``actuator`` for ``dof`` to ``world``.

    :param world: The world to add the actuator to, modified in place.
    :param dof: The degree of freedom the actuator drives.
    :param actuator: The MuJoCo actuator to attach.
    """
    dof_actuator = Actuator()
    dof_actuator.add_dof(dof=dof)
    dof_actuator.simulator_additional_properties.append(actuator)
    world.add_actuator(actuator=dof_actuator)


def _leg_connections_without_hardware_interface(
    robot: ICub3FixedBase,
) -> list[ActiveConnection1DOF]:
    """
    The iCub3's leg/base joints (hips, knees, ankles): every
    :class:`~semantic_digital_twin.world_description.connections.ActiveConnection1DOF`
    in the robot that is not already covered by
    :attr:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.degrees_of_freedom_with_hardware_interface`
    (the arms, hands, torso and neck).

    Mirrors :mod:`~experiments.montessori.montessori_demo`'s own
    ``_base_connections_without_hardware_interface`` for the HSRB's wheels.

    :param robot: The mounted iCub3.
    """
    hardware_interface_dofs = set(robot.degrees_of_freedom_with_hardware_interface)
    return [
        connection
        for connection in robot.connections
        if isinstance(connection, ActiveConnection1DOF)
        and connection.raw_dof not in hardware_interface_dofs
    ]


def _finger_joint_index(name: str) -> Optional[int]:
    """
    The position of a finger joint along its own finger (0 = knuckle closest to the
    palm, 3 = fingertip), read off the trailing digit in its name (e.g. ``2`` for
    ``"r_hand_thumb_2_joint"``).

    :param name: The degree of freedom's own (unprefixed) name.
    :return: The finger joint's index, or ``None`` if ``name`` does not end in a digit
        followed by ``"_joint"`` (true of every non-finger joint).
    """
    match = _FINGER_JOINT_INDEX_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _servo_tuning_for(dof: DegreeOfFreedom) -> JointServoTuning:
    """
    The :class:`~experiments.montessori.franka_panda_equipment.JointServoTuning`
    ``dof``'s position servo is built with, by name (see the module docstring for where
    each tier's own numbers come from):

    - a finger joint (name contains ``"_hand_"``) gets its own
      :data:`FINGER_JOINT_SERVO_TUNING_BY_INDEX` entry;
    - an eye joint (name contains ``"eye"``) gets :data:`EYE_JOINT_SERVO_TUNING`;
    - a torso joint (name starts with ``"torso_"``) gets :data:`TORSO_JOINT_SERVO_TUNING`;
    - a neck joint (name starts with ``"neck_"``) or one of the wrist's two smaller
      joints (name contains ``"_wrist_pitch"`` or ``"_wrist_yaw"``) gets
      :data:`WRIST_JOINT_SERVO_TUNING`;
    - everything else (shoulders, elbows, ``*_wrist_prosup``) gets
      :data:`ARM_JOINT_SERVO_TUNING`.

    :param dof: The degree of freedom to classify.
    """
    name = dof.name.name
    finger_index = _finger_joint_index(name) if "_hand_" in name else None
    if finger_index is not None:
        return FINGER_JOINT_SERVO_TUNING_BY_INDEX[finger_index]
    if "eye" in name:
        return EYE_JOINT_SERVO_TUNING
    if name.startswith("torso_"):
        return TORSO_JOINT_SERVO_TUNING
    if name.startswith("neck_") or "_wrist_pitch" in name or "_wrist_yaw" in name:
        return WRIST_JOINT_SERVO_TUNING
    return ARM_JOINT_SERVO_TUNING


def equip_icub_for_physical_simulation(robot: ICub3FixedBase) -> set[DegreeOfFreedom]:
    """
    Give the iCub3 everything it needs to be driven by MuJoCo's own physics rather than
    kinematically teleported, and report which of its degrees of freedom that covers.

    Every arm/torso/neck joint (see
    :attr:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.degrees_of_freedom_with_hardware_interface`)
    gets a position-servo actuator that tracks whatever the motion planner commands, its
    tuning selected per-joint by :func:`_servo_tuning_for`. Every leg/base joint instead
    gets a weaker position-hold actuator (:data:`LEG_ACTUATOR_POSITION_GAIN`) that
    simply keeps the stance the iCub3 was mounted in, since
    :mod:`experiments.montessori.icub_montessori_demo` bolts it in place rather than
    navigating it. Every body gets MuJoCo's own gravity compensation and every joint the
    same rotor armature (:data:`JOINT_ARMATURE`), servoed and held alike.

    :param robot: The mounted iCub3, modified in place.
    :return: Every degree of freedom MuJoCo now drives (hardware-interface and
        leg/base), for :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`'s
        ``physically_simulated_dofs``.
    """
    hardware_interface_dofs = set(robot.degrees_of_freedom_with_hardware_interface)
    leg_connections = _leg_connections_without_hardware_interface(robot)

    with robot._world.modify_world():
        for dof in sorted(hardware_interface_dofs, key=lambda d: d.name.name):
            _add_actuator(
                robot._world, dof, _position_servo_actuator(_servo_tuning_for(dof))
            )
        for connection in leg_connections:
            _add_actuator(
                robot._world,
                connection.raw_dof,
                _position_hold_actuator(
                    LEG_ACTUATOR_POSITION_GAIN,
                    LEG_ACTUATOR_VELOCITY_GAIN,
                    LEG_ACTUATOR_FORCE_RANGE,
                ),
            )

        for body in robot.bodies:
            body.simulator_additional_properties.append(
                MujocoBody(gravitation_compensation_factor=1.0)
            )
        for connection in robot.connections:
            if isinstance(connection, ActiveConnection1DOF):
                connection.dynamics.armature = JOINT_ARMATURE

    return hardware_interface_dofs | {
        connection.raw_dof for connection in leg_connections
    }
