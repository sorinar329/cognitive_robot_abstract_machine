"""
Moving the HSR's arm to joint positions found for it, rather than to a pose of its hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.robot_plans.actions.base import SequentialAction
from cramph.node import StatechartNode
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.hsrb import HSRBJoint

# %% the arm in joint space


@dataclass(eq=False, repr=False, kw_only=True)
class MoveArmJointsAction(SequentialAction):
    """
    Bring the HSR's arm joints to their positions: the arm lift first, so the hand
    leaves its park pose above the counter rather than in front of the cabinets below
    it, then every joint of the arm.
    """

    positions: dict[HSRBJoint, float]
    """
    Where each joint of the arm is to be.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            JointPositionList(goal_state=self._joint_state([HSRBJoint.ARM_LIFT])),
            JointPositionList(goal_state=self._joint_state(list(self.positions))),
        ]

    def _joint_state(self, joints: list[HSRBJoint]) -> JointState:
        """
        :return: The joint state putting ``joints`` at their :attr:`positions`.
        """
        world = self.context.world
        return JointState(
            connections=[world.get_connection_by_name(joint) for joint in joints],
            target_values=[self.positions[joint] for joint in joints],
        )
