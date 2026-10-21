from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import List, Self

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import EndEffector, Finger
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% joints


class OnRobot2FG7Joint(StrEnum):
    """
    The finger joints of the OnRobot 2FG7 parallel gripper as the MuJoCo scene names
    them.
    """

    LEFT_FINGER = "2fg7_left_finger_joint"
    RIGHT_FINGER = "2fg7_right_finger_joint"


# %% fingers


@dataclass(eq=False)
class OnRobot2FG7LeftFinger(Finger):
    """
    The left sliding finger of the 2FG7.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "mount_2fg7"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "2fg7_left_finger"
            ),
        )


@dataclass(eq=False)
class OnRobot2FG7RightFinger(Finger):
    """
    The right sliding finger of the 2FG7.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "mount_2fg7"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "2fg7_right_finger"
            ),
        )


# %% gripper


@dataclass(eq=False)
class OnRobot2FG7Gripper(
    EndEffector, HasTwoFingers[OnRobot2FG7LeftFinger, OnRobot2FG7RightFinger]
):
    """
    The OnRobot 2FG7 two-finger parallel gripper: both fingers share the mount origin
    and slide apart symmetrically, so positive joint values open the pads and negative
    values close them, with the pads meeting before the travel ends; the stall monitor
    ends a close early on contact.

    The tool frame sits at the TCP between the pads, so grasp goals address the
    pinch point rather than the mount: sending the mount there instead would bury
    the fingers in the table.
    """

    OPEN_FINGER_POSITION = 0.0175
    """
    Finger position with the pads 60 mm apart, a margin short of the travel limit.
    """

    CLOSED_FINGER_POSITION = -0.0126
    """
    Finger position at which the pad faces meet, leaving no gap between them.

    The travel continues to -0.015, where the pads would press into each other.
    """

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        gripper_joints = [
            self._world.get_connection_by_name(OnRobot2FG7Joint.LEFT_FINGER),
            self._world.get_connection_by_name(OnRobot2FG7Joint.RIGHT_FINGER),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(
                zip(gripper_joints, [self.OPEN_FINGER_POSITION] * 2, strict=True)
            ),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(
                zip(gripper_joints, [self.CLOSED_FINGER_POSITION] * 2, strict=True)
            ),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "mount_2fg7"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "tcp_2fg7_frame"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
