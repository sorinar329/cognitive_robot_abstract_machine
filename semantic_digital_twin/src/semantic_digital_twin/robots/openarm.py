import os
from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
    Arm,
    Finger,
    ParallelGripper,
)
from semantic_digital_twin.robots.robot_mixins import HasArms
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world import World


@dataclass(eq=False)
class OpenArm(AbstractRobot, HasArms):
    """
    Class describing the OpenArm bimanual robot based on openarm_bimanual.urdf.
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        # root is 'world' in the provided URDF
        return cls(
            name=PrefixedName("openarm", prefix=world.name),
            root=world.get_body_by_name("world"),
            _world=world,
        )

    def _setup_semantic_annotations(self):
        # We create a helper to define each arm (left and right)
        for side in ["left", "right"]:
            self._setup_arm(side)

    def _setup_arm(self, side: str):
        # Prefix used in the URDF (e.g., openarm_left_)
        urdf_prefix = f"openarm_{side}_"

        # Define the fingers for the parallel gripper
        thumb = Finger(
            name=PrefixedName(f"{side}_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{urdf_prefix}left_finger"),
            tip=self._world.get_body_by_name(f"{urdf_prefix}left_finger"),  # Adjust if a specific tip frame exists
            _world=self._world,
        )

        finger = Finger(
            name=PrefixedName(f"{side}_finger", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{urdf_prefix}right_finger"),
            tip=self._world.get_body_by_name(f"{urdf_prefix}right_finger"),
            _world=self._world,
        )

        # Define the parallel gripper (Hand)
        gripper = ParallelGripper(
            name=PrefixedName(f"{side}_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{urdf_prefix}hand_link"),
            tool_frame=self._world.get_body_by_name(f"{urdf_prefix}hand_link"),  # Often link7 or a dedicated tool link
            thumb=thumb,
            finger=finger,
            front_facing_axis=Vector3(0, 0, 1),  # Based on typical OpenArm orientation
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            _world=self._world,
        )

        # Define the Arm itself (from link0 to the hand)
        arm = Arm(
            name=PrefixedName(f"{side}_arm", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{urdf_prefix}link0"),
            tip=self._world.get_body_by_name(f"{urdf_prefix}link7"),
            manipulator=gripper,
            _world=self._world,
        )

        self.add_arm(arm)

    @property
    def left_arm(self) -> Arm:
        return self.arms[0]

    @property
    def right_arm(self) -> Arm:
        return self.arms[1]