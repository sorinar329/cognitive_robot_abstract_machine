from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Dict

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot


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


HIP_STANCE, THIGH_STANCE, CALF_STANCE = 0.0, 0.9, -1.8
"""Per-leg joint angles of the Go2's standing stance, matching ``go2.xml``'s ``home``
keyframe."""

STANDING_HEIGHT = 0.27
"""Height of the base above the floor with the legs in :data:`STANDING_CONFIGURATION`."""

STANDING_CONFIGURATION: Dict[UnitreeGo2Joint, float] = {
    UnitreeGo2Joint.FRONT_LEFT_HIP: HIP_STANCE,
    UnitreeGo2Joint.FRONT_LEFT_THIGH: THIGH_STANCE,
    UnitreeGo2Joint.FRONT_LEFT_CALF: CALF_STANCE,
    UnitreeGo2Joint.FRONT_RIGHT_HIP: HIP_STANCE,
    UnitreeGo2Joint.FRONT_RIGHT_THIGH: THIGH_STANCE,
    UnitreeGo2Joint.FRONT_RIGHT_CALF: CALF_STANCE,
    UnitreeGo2Joint.REAR_LEFT_HIP: HIP_STANCE,
    UnitreeGo2Joint.REAR_LEFT_THIGH: THIGH_STANCE,
    UnitreeGo2Joint.REAR_LEFT_CALF: CALF_STANCE,
    UnitreeGo2Joint.REAR_RIGHT_HIP: HIP_STANCE,
    UnitreeGo2Joint.REAR_RIGHT_THIGH: THIGH_STANCE,
    UnitreeGo2Joint.REAR_RIGHT_CALF: CALF_STANCE,
}
"""
The twelve leg joint angles that stand the robot up, carrying its base at
:data:`STANDING_HEIGHT`.

A gait works relative to this posture rather than to wherever the legs happen to be:
read live, a leg caught mid-step would become the posture the next gait oscillates
around, and successive walks would wander further from a stance the robot can stand on.
"""


@dataclass(eq=False)
class UnitreeGo2(AbstractRobot):
    """
    The Unitree Go2 quadruped, walking on a physically simulated floating base.

    Its base is attached to the world through the plain 6-DoF connection its MJCF's
    freejoint parses to, so MuJoCo integrates it like any other free body: the base's
    position and orientation are a consequence of gravity and leg-ground contact, not a
    commanded input. It has no wheeled drive and therefore no
    :class:`~semantic_digital_twin.robots.robot_part_mixins.HasMobileBase` mobile base -
    navigation instead comes from gaiting its 12 leg joints, see ``coraplex_go2_demo``.

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
