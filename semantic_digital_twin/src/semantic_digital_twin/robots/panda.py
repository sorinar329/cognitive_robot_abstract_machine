from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Self

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    JointName,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm, HasTwoFingers
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    EndEffector,
    Finger,
    ServoedRobotPart,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connection_properties import (
    JointDrive,
    ServoGains,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% names in the description


class PandaJoint(JointName):
    """
    Names of the Panda's commandable connections, as spelled in its MJCF.
    """

    JOINT_1 = "joint1"
    JOINT_2 = "joint2"
    JOINT_3 = "joint3"
    JOINT_4 = "joint4"
    JOINT_5 = "joint5"
    JOINT_6 = "joint6"
    JOINT_7 = "joint7"
    LEFT_FINGER = "/finger_joint1"
    RIGHT_FINGER = "/finger_joint2"


class PandaBody(StrEnum):
    """
    Names of the bodies of the Panda's MJCF that its parts are rooted in or end at.
    """

    BASE = "link0"
    WRIST = "link7"
    HAND = "/hand"
    TOOL_FRAME = "/tool_frame"
    LEFT_FINGER = "/left_finger"
    RIGHT_FINGER = "/right_finger"


# %% the hand


@dataclass(eq=False)
class PandaLeftFinger(Finger):
    """
    A finger of the Panda's hand, which is a single body sliding along the hand.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        finger = robot_root._world.get_body_in_branch_by_name(
            robot_root, PandaBody.LEFT_FINGER
        )
        return cls(root=finger, tip=finger)


@dataclass(eq=False)
class PandaRightFinger(Finger):
    """
    A finger of the Panda's hand, which is a single body sliding along the hand.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        finger = robot_root._world.get_body_in_branch_by_name(
            robot_root, PandaBody.RIGHT_FINGER
        )
        return cls(root=finger, tip=finger)


@dataclass(eq=False)
class PandaGripper(
    EndEffector, HasTwoFingers[PandaLeftFinger, PandaRightFinger], ServoedRobotPart
):
    """
    The Franka Hand: two fingers sliding towards each other.
    """

    @classproperty
    def holds_its_own_weight(cls) -> bool:
        return True

    @property
    def drive_of_each_joint(self) -> dict[str, JointDrive]:
        """
        The description drives both fingers together through one tendon; each finger
        is driven here by its share of that tendon's stiffness, damping and force.
        """
        finger_drive = JointDrive(
            gains=ServoGains(stiffness=50.0, damping=5.0, torque_limit=50.0)
        )
        return {
            PandaJoint.LEFT_FINGER: finger_drive,
            PandaJoint.RIGHT_FINGER: finger_drive,
        }

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> list[JointState]:
        """
        The hand opens and closes to the ends of its fingers' travel.
        """
        finger_joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.upper.position
                for connection in finger_joints
            },
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.lower.position
                for connection in finger_joints
            },
            state_type=GripperState.CLOSE,
        )
        return [gripper_close, gripper_open]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.Z(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, PandaBody.HAND
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, PandaBody.TOOL_FRAME
            ),
        )


# %% the arm


@dataclass(eq=False)
class PandaArm(Arm[PandaGripper], ServoedRobotPart):
    """
    The Panda's seven-jointed arm.
    """

    @classproperty
    def holds_its_own_weight(cls) -> bool:
        """
        The arm's own controller cancels its weight.
        """
        return True

    @property
    def drive_of_each_joint(self) -> dict[str, JointDrive]:
        """
        The drives MuJoCo Menagerie gives the Panda: the four joints nearest the base
        carry the rest of the arm and are driven harder than the three wrist joints.
        Menagerie gives every joint an armature of 0.1 and a damping of 1, which the
        description states only for the hand's joints.
        """
        upper_arm = JointDrive(
            gains=ServoGains(stiffness=2000.0, damping=200.0, torque_limit=87.0),
            armature=0.1,
            joint_damping=1.0,
        )
        wrist = JointDrive(
            gains=ServoGains(stiffness=500.0, damping=50.0, torque_limit=12.0),
            armature=0.1,
            joint_damping=1.0,
        )
        return {
            PandaJoint.JOINT_1: upper_arm,
            PandaJoint.JOINT_2: upper_arm,
            PandaJoint.JOINT_3: upper_arm,
            PandaJoint.JOINT_4: upper_arm,
            PandaJoint.JOINT_5: wrist,
            PandaJoint.JOINT_6: wrist,
            PandaJoint.JOINT_7: wrist,
        }

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> list[JointState]:
        """
        The arm parks in Franka's ready pose.
        """
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, PandaBody.BASE
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, PandaBody.WRIST
            ),
        )


# %% the robot


@dataclass(eq=False)
class Panda(AbstractRobot, HasOneArm[PandaArm]):
    """
    The `Franka Emika Panda <https://franka.de/>`_ arm with its hand, described by
    MuJoCo Menagerie's model of it.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_franka_panda_description/mjcf/panda.xml"

    @classmethod
    def from_description(cls) -> Self:
        """
        :return: This robot, read from its MJCF without the world body MJCF puts above
            every robot's root, and without the description's own actuators, since
            each part declares the servo driving its joints.
        """
        world = MJCFParser(cls.get_ros_file_path()).parse()
        with world.modify_world():
            world.remove_kinematic_structure_entity(world.root)
            for actuator in list(world.actuators):
                world.remove_actuator(actuator)
        return cls.from_world(world)

    @classmethod
    def _get_root_body_name(cls) -> str:
        return PandaBody.BASE

    def _setup_collision_rules(self):
        """
        Avoid collisions with everything but the robot itself, since no description
        states which of its own bodies may touch.
        """
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )

    def _setup_velocity_limits(self):
        """
        Franka's published joint velocity limits, and a finger speed of 0.2 m/s.
        """
        velocity_limits = defaultdict(lambda: 2.175)
        for joint in (PandaJoint.JOINT_5, PandaJoint.JOINT_6, PandaJoint.JOINT_7):
            velocity_limits[self._world.get_connection_by_name(joint)] = 2.61
        for joint in (PandaJoint.LEFT_FINGER, PandaJoint.RIGHT_FINGER):
            velocity_limits[self._world.get_connection_by_name(joint)] = 0.2
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=velocity_limits)
