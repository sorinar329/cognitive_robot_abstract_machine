from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Dict

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable_from, ConditionType
from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.legged_locomotion import WalkMotion
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_pose_free_for_robot
from semantic_digital_twin.spatial_types.spatial_types import Pose

ARRIVAL_TOLERANCE = 0.35
"""
How close to its target a walk counts as arrived, in metres.

Looser than the gait's own stopping distance
(:attr:`~coraplex.robot_plans.motions.legged_locomotion.TrotGait.position_tolerance`),
because the robot carries on for part of a step after the gait stops commanding one.
"""


@dataclass
class WalkAction(ActionDescription):
    """
    Walks a legged robot to a position by gaiting its legs.

    Unlike :class:`~coraplex.robot_plans.actions.core.navigation.NavigateAction`, the
    target is reached by an open-loop gait rather than a commanded base pose, so only
    the final position is guaranteed to be close to the target -- final orientation is
    not tracked.
    """

    target_location: Pose
    """Location to which the robot should be walked."""

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(WalkMotion(self.target_location))

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The target location needs to be free from obstacles.
        """
        return is_pose_free_for_robot(context.robot, variables["target_location"])

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The robot needs to have arrived within :data:`ARRIVAL_TOLERANCE` of the target.
        """
        return allclose(
            variable_from(context.robot.root).global_pose.to_position(),
            kwargs["target_location"].to_position(),
            atol=ARRIVAL_TOLERANCE,
        )
