from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import List, Self

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasFingers
from semantic_digital_twin.robots.robot_parts import EndEffector, Finger
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% joints


class OnRobot3FG15Joint(StrEnum):
    """
    The finger joints of the OnRobot 3FG15 three-finger gripper as the MuJoCo scene
    names them.
    """

    LEFT_FINGER = "3fg15_left_finger_joint"
    RIGHT_FINGER = "3fg15_right_finger_joint"
    CENTER_FINGER = "3fg15_center_finger_joint"


# %% fingers


@dataclass(eq=False)
class OnRobot3FG15LeftFinger(Finger):
    """
    The left revolving finger of the 3FG15.
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
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "mount_3fg15"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "3fg15_left_finger"
            ),
        )


@dataclass(eq=False)
class OnRobot3FG15RightFinger(Finger):
    """
    The right revolving finger of the 3FG15.
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
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "mount_3fg15"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "3fg15_right_finger"
            ),
        )


@dataclass(eq=False)
class OnRobot3FG15CenterFinger(Finger):
    """
    The center revolving finger of the 3FG15.
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
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "mount_3fg15"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "3fg15_center_finger"
            ),
        )


# %% gripper


@dataclass(eq=False)
class OnRobot3FG15Gripper(
    EndEffector,
    HasFingers[
        OnRobot3FG15CenterFinger,
        OnRobot3FG15LeftFinger,
        OnRobot3FG15RightFinger,
    ],
):
    """
    The OnRobot 3FG15 three-finger gripper: all fingers revolve symmetrically, so open
    and close each name one position for all three joints.

    The center finger takes the thumb slot the mixin requires; the gripper itself is
    rotationally symmetric and has no thumb.

    The tool frame sits at the TCP between the fingers, so grasp goals address the
    pinch point rather than the mount: sending the mount there instead would bury
    the fingers in the table.
    """

    MEDIUM_FINGER_POSITION = 0.6
    """
    Finger position that brings the pads just inside a 35 mm radius, so a medium close meets
    the wall of a 35 mm object without driving the fingers through it.
    """

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        gripper_joints = [
            self._world.get_connection_by_name(OnRobot3FG15Joint.LEFT_FINGER),
            self._world.get_connection_by_name(OnRobot3FG15Joint.RIGHT_FINGER),
            self._world.get_connection_by_name(OnRobot3FG15Joint.CENTER_FINGER),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [-1.0, -1.0, -1.0], strict=True)),
            state_type=GripperState.OPEN,
        )

        gripper_medium = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_medium", prefix=self.name.name),
            mapping=dict(
                zip(gripper_joints, [self.MEDIUM_FINGER_POSITION] * 3, strict=True)
            ),
            state_type=GripperState.MEDIUM,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [1.0, 1.0, 1.0], strict=True)),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_medium, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "mount_3fg15"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "tcp_3fg15_frame"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
