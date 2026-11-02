"""
The vertical plane the Toyota HSR's arm swings its hand in, and the joint positions that
put the hand at a pose in it: what tells where the HSR has to stand for its hand to
reach a pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% where the hand and the wrist stand


@dataclass
class ArmTilt:
    """
    How far the HSR's hand is tilted in its arm's plane, and where that puts the tool
    frame relative to the wrist.
    """

    tilt: float
    """
    The sum of arm and wrist flex.
    """

    wrist_to_tool: np.ndarray
    """
    The tool frame's position relative to the wrist flex joint, relative to the HSR's
    root.
    """


@dataclass
class WristPlacement:
    """
    A position of the HSR's arm flex and lift that puts the wrist at a height asked
    for.
    """

    flex: float
    """
    The arm flex joint's position.
    """

    lift: float
    """
    The arm lift joint's position.
    """

    wrist_ahead: float
    """
    How far ahead of the HSR's root the wrist flex joint stands.
    """


# %% the arm's plane


@dataclass
class HSRBArmPlane:
    """
    The vertical plane the HSR's arm swings its hand in while its arm and wrist roll
    joints stand at zero, and the joint positions that put the hand at a pose in it.
    In that plane, the hand's tilt depends on the sum of arm and wrist flex alone, and
    the wrist's place on arm flex and lift alone.
    """

    search_steps: int = 600
    """
    How many positions each of the two searches tries within a joint's limits.
    """

    reach_resolution: float = 0.005
    """
    How near, in metres, a goal has to come to a distance the arm reaches for the arm
    to count as reaching it: about how far apart the searches' positions put the tool
    frame.
    """

    world: World = field(init=False)
    """
    A world holding nothing but an HSR, whose joints the searches set, so that no
    simulated robot moves.
    """

    robot: HSRB = field(init=False)
    """
    The HSR of :attr:`world`.
    """

    def __post_init__(self):
        self.world = World()
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
        self.robot = RobotSpecification(HSRB).spawn(self.world)

    def joint_positions_for(self, base_T_tool: np.ndarray) -> dict[HSRBJoint, float]:
        """
        :param base_T_tool: Where the tool frame is to be, relative to the HSR's root;
            it has to lie in the arm's plane.
        :return: The positions of the arm's joints that put the tool frame there, as
            near as the searches get, the wrist rolled to the goal's turn about the
            hand's approach.
        """
        arm_tilt = self._tilt_for(base_T_tool[:3, :3])
        wanted_wrist = base_T_tool[:3, 3] - arm_tilt.wrist_to_tool
        placement = min(
            self._wrist_placements(arm_tilt.tilt, wanted_wrist[2]),
            key=lambda candidate: abs(wanted_wrist[0] - candidate.wrist_ahead),
        )
        positions = {
            HSRBJoint.ARM_LIFT: float(placement.lift),
            HSRBJoint.ARM_FLEX: float(placement.flex),
            HSRBJoint.ARM_ROLL: 0.0,
            HSRBJoint.WRIST_FLEX: float(arm_tilt.tilt - placement.flex),
            HSRBJoint.WRIST_ROLL: 0.0,
        }
        positions[HSRBJoint.WRIST_ROLL] = self._wrist_roll_for(positions, base_T_tool)
        return positions

    def reachable_distances(
        self, tool_height: float, base_R_tool: np.ndarray
    ) -> list[float]:
        """
        :param tool_height: How high above the HSR's root the tool frame is to be.
        :param base_R_tool: How the tool frame is to be turned, relative to the HSR's
            root.
        :return: How far ahead of the HSR's root the arm puts the tool frame at that
            height, tilted as near to that turn as the arm allows, one distance per
            arm flex position the search tries that puts it there, in ascending order.
        """
        arm_tilt = self._tilt_for(base_R_tool)
        return sorted(
            float(placement.wrist_ahead + arm_tilt.wrist_to_tool[0])
            for placement in self._wrist_placements(
                arm_tilt.tilt, tool_height - arm_tilt.wrist_to_tool[2]
            )
        )

    def reaches_from(
        self,
        world_T_base: np.ndarray,
        tool_frame_goals: list[np.ndarray],
        reaches: list[list[float]],
    ) -> bool:
        """
        :param world_T_base: Where the HSR's root stands, in the kitchen's frame.
        :param tool_frame_goals: Where the tool frame is to be, in the kitchen's frame.
        :param reaches: For each goal, the :meth:`reachable_distances` at its height
            and turn.
        :return: Whether each goal lies at a distance ahead that the arm reaches it at.
        """
        base_T_world = np.linalg.inv(world_T_base)
        return all(
            min(abs(distance - (base_T_world @ goal)[0, 3]) for distance in reach)
            <= self.reach_resolution
            for goal, reach in zip(tool_frame_goals, reaches)
        )

    def _tilt_for(self, base_R_tool: np.ndarray) -> ArmTilt:
        """
        Tilt the hand, its arm and wrist roll joints at zero, as near to ``base_R_tool``
        as the arm allows.

        :param base_R_tool: How the tool frame is to be turned, relative to the HSR's
            root.
        :return: The tilt, and where it puts the tool frame relative to the wrist.
        """
        end_effector = self.robot.all_arms[0].end_effector
        wanted_approach = base_R_tool @ end_effector.approach_axis.to_np()[:3]
        self._set({HSRBJoint.ARM_ROLL: 0.0, HSRBJoint.WRIST_ROLL: 0.0})
        tilt = max(
            self._tilts(),
            key=lambda candidate: self._approach_at(candidate) @ wanted_approach,
        )
        self._set_tilt(tilt)
        return ArmTilt(
            tilt=float(tilt),
            wrist_to_tool=self._base_T(end_effector.tool_frame)[:3, 3] - self._wrist(),
        )

    def _wrist_placements(
        self, tilt: float, wrist_height: float
    ) -> list[WristPlacement]:
        """
        :param tilt: The sum of arm and wrist flex the hand is tilted by.
        :param wrist_height: How high above the HSR's root the wrist is to be.
        :return: The arm flex positions the search tries that leave the wrist flex
            within its limits, each with the lift that puts the wrist at
            ``wrist_height``, where that lies within the lift's limits.
        """
        lift_limits = self._limits(HSRBJoint.ARM_LIFT)
        wrist_limits = self._limits(HSRBJoint.WRIST_FLEX)
        placements = []
        for flex in np.linspace(*self._limits(HSRBJoint.ARM_FLEX), self.search_steps):
            if not wrist_limits[0] <= tilt - flex <= wrist_limits[1]:
                continue
            self._set({HSRBJoint.ARM_LIFT: 0.0, HSRBJoint.ARM_FLEX: flex})
            wrist = self._wrist()
            lift = wrist_height - wrist[2]
            if lift_limits[0] <= lift <= lift_limits[1]:
                placements.append(
                    WristPlacement(
                        flex=float(flex), lift=float(lift), wrist_ahead=float(wrist[0])
                    )
                )
        return placements

    def _wrist_roll_for(
        self, positions: dict[HSRBJoint, float], base_T_tool: np.ndarray
    ) -> float:
        """
        :param positions: The arm's joint positions, the wrist not rolled.
        :param base_T_tool: Where the tool frame is to be, relative to the HSR's root.
        :return: The wrist roll that turns the tool frame about the hand's approach as
            far as ``base_T_tool`` is turned.
        """
        tool_frame = self.robot.all_arms[0].end_effector.tool_frame
        self._set(positions)
        unrolled = self._base_T(tool_frame)[:3, :3]
        turn = unrolled.T @ base_T_tool[:3, :3]
        angle = math.atan2(turn[1, 0], turn[0, 0])

        def orientation_error(roll: float) -> float:
            self._set({HSRBJoint.WRIST_ROLL: roll})
            return np.linalg.norm(
                self._base_T(tool_frame)[:3, :3] - base_T_tool[:3, :3]
            )

        return min([angle, -angle], key=orientation_error)

    def _tilts(self) -> np.ndarray:
        """
        :return: The sums of arm and wrist flex the joints' limits allow.
        """
        flex = self._limits(HSRBJoint.ARM_FLEX)
        wrist = self._limits(HSRBJoint.WRIST_FLEX)
        return np.linspace(flex[0] + wrist[0], flex[1] + wrist[1], self.search_steps)

    def _set_tilt(self, tilt: float) -> None:
        """
        Tilt the hand by ``tilt``, the sum of arm and wrist flex, with the arm flexed as
        little as the wrist's limits allow.
        """
        flex = float(np.clip(tilt, *self._limits(HSRBJoint.ARM_FLEX)))
        self._set({HSRBJoint.ARM_FLEX: flex, HSRBJoint.WRIST_FLEX: tilt - flex})

    def _approach_at(self, tilt: float) -> np.ndarray:
        """
        :return: The direction the hand approaches along when tilted by ``tilt``,
            relative to the HSR's root.
        """
        self._set_tilt(tilt)
        end_effector = self.robot.all_arms[0].end_effector
        return (
            self._base_T(end_effector.tool_frame)[:3, :3]
            @ end_effector.approach_axis.to_np()[:3]
        )

    def _wrist(self) -> np.ndarray:
        """
        :return: Where the wrist flex joint is, relative to the HSR's root.
        """
        wrist = self.world.get_connection_by_name(HSRBJoint.WRIST_FLEX).child
        return self._base_T(wrist)[:3, 3]

    def _base_T(self, entity: KinematicStructureEntity) -> np.ndarray:
        """
        :return: The pose of ``entity`` relative to the HSR's root.
        """
        return self.world.compute_forward_kinematics_np(self.robot.root, entity)

    def _limits(self, joint: HSRBJoint) -> tuple[float, float]:
        """
        :return: The lower and upper position limit of ``joint``.
        """
        limits = self.world.get_connection_by_name(joint).dof.limits
        return limits.lower.position, limits.upper.position

    def _set(self, positions: dict[HSRBJoint, float]) -> None:
        """
        Set the joints of :attr:`world`'s HSR to ``positions``.
        """
        for joint, position in positions.items():
            self.world.get_connection_by_name(joint).position = position
