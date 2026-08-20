from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Self, List

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot, MobileBase
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class UnitreeGo2Joint(StrEnum):
    """
    Names of the Go2's commandable connections, as spelled in its MJCF.

    Members are usable wherever a connection name is expected, so a configuration keyed
    by them stays a plain mapping of names to positions.
    """

    FRONT_LEFT_HIP = "FL_hip_joint"
    FRONT_LEFT_THIGH = "FL_thigh_joint"
    FRONT_LEFT_CALF = "FL_calf_joint"

    FRONT_RIGHT_HIP = "FR_hip_joint"
    FRONT_RIGHT_THIGH = "FR_thigh_joint"
    FRONT_RIGHT_CALF = "FR_calf_joint"

    REAR_LEFT_HIP = "RL_hip_joint"
    REAR_LEFT_THIGH = "RL_thigh_joint"
    REAR_LEFT_CALF = "RL_calf_joint"

    REAR_RIGHT_HIP = "RR_hip_joint"
    REAR_RIGHT_THIGH = "RR_thigh_joint"
    REAR_RIGHT_CALF = "RR_calf_joint"


@dataclass(eq=False)
class UnitreeGo2MobileBase(MobileBase[OmniDrive]):
    """
    The Go2's drive: an :class:`OmniDrive` connection put in place of the freejoint
    ``base`` carries in the upstream MJCF, so the robot can be navigated like any other
    :class:`~semantic_digital_twin.robots.robot_part_mixins.HasMobileBase` robot instead
    of falling freely.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(root=robot_root)

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class UnitreeGo2(AbstractRobot, HasMobileBase[UnitreeGo2MobileBase]):
    """
    The Unitree Go2 quadruped, driven as a rigid body on its :class:`OmniDrive` base.

    Its legs are not gaited: they hold whatever configuration they were placed in (see
    ``coraplex_go2_demo``), and only the base itself is commanded when the robot
    navigates.

    https://www.unitree.com/go2
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        raise NotImplementedError(
            "The Go2 has no ROS package; it is parsed from its MJCF scene instead, see "
            "coraplex_go2_demo."
        )

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base"

    def _setup_collision_rules(self):
        # No SRDF-based self-collision matrix exists for the Go2 yet, so only
        # external collisions are guarded against.
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )
