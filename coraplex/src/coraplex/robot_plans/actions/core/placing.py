from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict

from coraplex.plans.attachment_nodes import DetachNode
from coraplex.plans.plan_node import PlanNode
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    or_,
    not_,
    and_,
    variable_from,
    ConditionType,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential, execute_single
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import (
    GRASP_DETECTION_THRESHOLD,
    PickUpAction,
)
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PlaceAction(ActionDescription):
    """
    Places an Object at a position using an arm.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be place
    """
    target_location: Pose
    """
    Pose in the world at which the object should be placed.
    """

    arm: Arms
    """
    Arm that is currently holding the object
    """

    placing_linear_velocity: float = 0.03
    """
    Linear reference velocity (in m/s) for the final descent onto the target location.

    Slowed down from the default so the object settles onto the surface instead of being
    dropped/shoved into place.
    """

    transport_linear_velocity: float = 0.05
    """
    Linear reference velocity (in m/s) for carrying the held object to a pose above the
    target location, before the final descent.
    """

    release_opening_velocity: float = 0.05
    """
    Finger joint velocity (in m/s) used while opening the gripper to release the object.
    """

    retract_linear_velocity: float = 0.05
    """
    Linear reference velocity (in m/s) for retracting the end effector away from the
    placed object.

    Kept at the original, careful speed -- tried raising this to 0.15 m/s, but that was
    enough for the retreat itself to knock the just-placed cube back down in testing
    (confirmed via final stack heights collapsing to table level despite every
    pick/place reporting success), since the retreat path runs immediately beside a cube
    that is only resting on the stack by gravity/friction, not attached.
    """

    max_release_attempts: int = 3
    """
    How many times to (re-)issue the OPEN motion, checking afterwards whether the object
    is actually still detected between the fingers, before retracting regardless.

    Guards against retracting while still gripping -- seen carrying the object up a few
    centimeters before finally releasing it, dropping it from height instead of setting
    it down -- by only retracting once the object is confirmed clear of the gripper (or
    attempts run out).
    """

    def _previous_grasp(self) -> GraspDescription:
        """
        The grasp description used by the preceding :class:`PickUpAction` on this
        object, or a default front/no-alignment grasp if none is found.
        """
        end_effector = ViewManager.get_arm_view(self.arm, self.robot).end_effector
        previous_pick = self.plan_node.get_previous_node_by_designator_type(
            PickUpAction
        )
        if previous_pick:
            return previous_pick.designator.grasp_description
        return GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
        )

    def _pose_sequence(self) -> tuple[Pose, Pose, Pose]:
        """
        :return: The (transport, placing, retract) poses for this placement.
        """
        return self._previous_grasp().pose_sequence(
            self.target_location, self.object_designator, reverse=True
        )

    def _transport_and_descend_plan(self, transport_pose: Pose, placing_pose: Pose) -> PlanNode:
        return sequential(
            [
                MoveToolCenterPointMotion(
                    transport_pose,
                    self.arm,
                    allow_gripper_collision=False,
                    reference_linear_velocity=self.transport_linear_velocity,
                ),
                MoveToolCenterPointMotion(
                    placing_pose,
                    self.arm,
                    allow_gripper_collision=False,
                    reference_linear_velocity=self.placing_linear_velocity,
                ),
            ],
        )

    def _retract_plan(self, retract_pose: Pose) -> PlanNode:
        # A DetachNode(body=self.object_designator, new_parent=self.world.root)
        # would normally go here, but is unnecessary: the object is held only
        # by real contact/friction (AttachNode is likewise disabled in
        # PickUpAction), so nothing needs kinematically re-parenting once
        # released.
        return MoveToolCenterPointMotion(
            retract_pose,
            self.arm,
            reference_linear_velocity=self.retract_linear_velocity,
        )

    def _object_released(self) -> bool:
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot)
        return not (
            is_body_in_gripper(self.object_designator, end_effector)
            > GRASP_DETECTION_THRESHOLD
        )

    def execute(self) -> Any:
        transport_pose, placing_pose, retract_pose = self._pose_sequence()

        self.add_subplan(
            self._transport_and_descend_plan(transport_pose, placing_pose)
        ).perform()

        for _ in range(self.max_release_attempts):
            self.add_subplan(
                execute_single(
                    MoveGripperMotion(
                        GripperState.OPEN,
                        self.arm,
                        finger_velocity=self.release_opening_velocity,
                    )
                )
            ).perform()
            if self._object_released():
                break

        self.add_subplan(self._retract_plan(retract_pose)).perform()

    @property
    def _action_plan(self) -> PlanNode:
        transport_pose, placing_pose, retract_pose = self._pose_sequence()

        return sequential(
            [
                self._transport_and_descend_plan(transport_pose, placing_pose),
                MoveGripperMotion(
                    GripperState.OPEN,
                    self.arm,
                    finger_velocity=self.release_opening_velocity,
                ),
                self._retract_plan(retract_pose),
            ],
            self.context,
        )

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The object needs to be in the gripper frame.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return or_(
            not_(GripperIsFree(end_effector)),
            is_body_in_gripper(variable_from(kwargs["object_designator"]), end_effector)
            > GRASP_DETECTION_THRESHOLD,
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper must be free again and the object needs to be at the target
        location.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return and_(
            GripperIsFree(end_effector),
            is_body_in_gripper(variable_from(kwargs["object_designator"]), end_effector)
            < 0.1,
            allclose(
                variable_from(kwargs["object_designator"]).global_pose,
                kwargs["target_location"],
                atol=0.03,
            ),
        )
