"""
A two-fingered gripper annotation built over bodies that already stand in a world, for
tests that need the world to say which bodies make up a gripper without loading a robot.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import EndEffector, Finger
from semantic_digital_twin.spatial_types.spatial_types import Quaternion
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from typing_extensions import List, Self


@dataclass(eq=False)
class FingerWithoutHardware(Finger):
    """
    A finger that is only its bodies: no joint states, no hardware interface.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        raise NotImplementedError("Built over existing bodies, never set up.")

    def setup_hardware_interfaces(self):
        return

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class ThumbWithoutHardware(FingerWithoutHardware):
    """
    The finger a two-fingered gripper names its thumb.
    """


@dataclass(eq=False)
class OpposingFingerWithoutHardware(FingerWithoutHardware):
    """
    The finger opposite the thumb.
    """


@dataclass(eq=False)
class TwoFingerGripperWithoutHardware(
    EndEffector,
    HasTwoFingers[ThumbWithoutHardware, OpposingFingerWithoutHardware],
):
    """
    A two-fingered gripper that is only its bodies.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        raise NotImplementedError("Built over existing bodies, never set up.")

    def setup_hardware_interfaces(self):
        return

    def setup_joint_states(self) -> List[JointState]:
        return []


def annotate_two_finger_gripper(
    world: World,
    name: str,
    thumb_tip: Body,
    finger_tip: Body,
    tool_frame: Body,
) -> TwoFingerGripperWithoutHardware:
    """
    Declare in ``world`` that the given bodies make up one two-fingered gripper.

    :param world: The world the bodies stand in.
    :param name: The gripper's name; its fingers are named after it.
    :param thumb_tip: The tip of the finger the gripper names its thumb.
    :param finger_tip: The tip of the finger opposite the thumb.
    :param tool_frame: The gripper's tool center point.
    :return: The gripper annotation, already added to ``world``.
    """
    thumb = ThumbWithoutHardware(
        name=PrefixedName(f"{name}_thumb"), root=thumb_tip, tip=thumb_tip
    )
    finger = OpposingFingerWithoutHardware(
        name=PrefixedName(f"{name}_finger"), root=finger_tip, tip=finger_tip
    )
    gripper = TwoFingerGripperWithoutHardware(
        name=PrefixedName(name),
        root=tool_frame,
        tool_frame=tool_frame,
        front_facing_orientation=Quaternion(),
        fingers=[thumb, finger],
    )
    with world.modify_world():
        world.add_semantic_annotation(thumb)
        world.add_semantic_annotation(finger)
        world.add_semantic_annotation(gripper)
    return gripper
