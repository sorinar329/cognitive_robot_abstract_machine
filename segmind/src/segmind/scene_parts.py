"""
The parts of a scene that detectors read from the world, besides the objects they watch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Tuple


@dataclass(frozen=True)
class Gripper:
    """
    A two-fingered gripper, as grasp detection reads it.
    """

    finger_tips: Tuple[Body, Body]
    """
    The tip of the gripper's thumb and the tip of the finger opposite it, in that order.
    """

    tool_frame: Body
    """
    The gripper's tool center point.
    """


@dataclass
class SceneParts:
    """
    The parts of a scene detectors read, as the world declares them.
    """

    grippers: List[Gripper] = field(default_factory=list)
    """
    Every two-fingered gripper in the scene.
    """

    @classmethod
    def of_world(cls, world: World) -> SceneParts:
        """
        The parts the given world declares through its semantic annotations.

        :param world: The world detection runs against.
        """
        return cls(
            grippers=[
                Gripper(
                    finger_tips=(end_effector.thumb.tip, end_effector.finger.tip),
                    tool_frame=end_effector.tool_frame,
                )
                for end_effector in world.get_semantic_annotations_by_type(EndEffector)
                if isinstance(end_effector, HasTwoFingers)
            ]
        )
