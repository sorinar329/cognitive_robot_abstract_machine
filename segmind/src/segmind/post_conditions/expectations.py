"""
What SegMind expects to detect while an action of a plan runs, for the action's
post-condition to hold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from segmind.datastructures.events import (
    DetectionEvent,
    EventWithTrackedObjects,
    GraspEvent,
    LossOfGraspEvent,
    LossOfSupportEvent,
    SupportEvent,
)
from semantic_digital_twin.world_description.world_entity import Body

ExpectedAction = TypeVar("ExpectedAction", bound=ActionDescription)


# %% expectations


@dataclass
class ActionExpectation(Generic[ExpectedAction], SubClassSafeGeneric, ABC):
    """
    The events that show an action of the bound type achieved what it is for.

    An expectation reads only the events detected while the action ran and until SegMind
    saw the world at rest after it.
    """

    @property
    def expected_action_type(self) -> type[ExpectedAction]:
        """
        :return: The type of action this expectation is of.
        """
        return type(self).get_type_of_generic_parameter(ExpectedAction)

    @abstractmethod
    def watched_bodies(self, action: ExpectedAction) -> list[Body]:
        """
        :param action: The action judged.
        :return: The bodies SegMind has to watch to detect what ``action`` achieved.
        """

    @abstractmethod
    def evidence(
        self, action: ExpectedAction, events: list[EventWithTrackedObjects]
    ) -> list[DetectionEvent]:
        """
        :param action: The action judged.
        :param events: The events of its watched bodies detected while it ran and until
            the world was seen at rest after it, in order.
        :return: The events that decide whether the post-condition of ``action`` holds.
        """

    @abstractmethod
    def holds(self, action: ExpectedAction, evidence: list[DetectionEvent]) -> bool:
        """
        :param action: The action judged.
        :param evidence: What :meth:`evidence` chose.
        :return: Whether the post-condition of ``action`` holds.
        """

    @staticmethod
    def last_grasp_change(
        events: list[EventWithTrackedObjects],
        body: Body,
        tool_frame: Body | None = None,
    ) -> list[EventWithTrackedObjects]:
        """
        :param events: The events, in order.
        :param body: The body taken hold of or let go of.
        :param tool_frame: The tool frame that took hold or let go; any, if not given.
        :return: The last :class:`GraspEvent` or :class:`LossOfGraspEvent` of ``body``
            by ``tool_frame``, alone; none when there was none.
        """
        changes = [
            event
            for event in events
            if isinstance(event, (GraspEvent, LossOfGraspEvent))
            and event.tracked_object is body
            and (tool_frame is None or event.with_object is tool_frame)
        ]
        return changes[-1:]


# %% taking hold of and lifting


@dataclass
class GraspingExpectation(ActionExpectation[GraspingAction]):
    """
    A grasp holds when the hand of the action's arm has hold of the object in the end.
    """

    def watched_bodies(self, action: GraspingAction) -> list[Body]:
        return [action.grasp.graspable.root]

    def evidence(
        self, action: GraspingAction, events: list[EventWithTrackedObjects]
    ) -> list[DetectionEvent]:
        return self.last_grasp_change(
            events,
            action.grasp.graspable.root,
            action.arm.end_effector.tool_frame,
        )

    def holds(self, action: GraspingAction, evidence: list[DetectionEvent]) -> bool:
        return any(isinstance(event, GraspEvent) for event in evidence)


@dataclass
class PickUpExpectation(ActionExpectation[PickUpAction]):
    """
    A pick-up holds when the hand of the action's arm has hold of the object in the end
    and the object lost what supported it.
    """

    def watched_bodies(self, action: PickUpAction) -> list[Body]:
        return [action.grasp.graspable.root]

    def evidence(
        self, action: PickUpAction, events: list[EventWithTrackedObjects]
    ) -> list[DetectionEvent]:
        picked = action.grasp.graspable.root
        return self.last_grasp_change(
            events, picked, action.arm.end_effector.tool_frame
        ) + [
            event
            for event in events
            if isinstance(event, LossOfSupportEvent) and event.tracked_object is picked
        ]

    def holds(self, action: PickUpAction, evidence: list[DetectionEvent]) -> bool:
        return any(isinstance(event, GraspEvent) for event in evidence) and any(
            isinstance(event, LossOfSupportEvent) for event in evidence
        )


# %% setting down


@dataclass
class PlaceExpectation(ActionExpectation[PlaceAction]):
    """
    A placing holds when no hand has hold of the object in the end and something other
    than the robot came to support it.
    """

    def watched_bodies(self, action: PlaceAction) -> list[Body]:
        return [action.object_designator.root]

    def evidence(
        self, action: PlaceAction, events: list[EventWithTrackedObjects]
    ) -> list[DetectionEvent]:
        placed = action.object_designator.root
        robot_bodies = set(action.robot.bodies)
        return self.last_grasp_change(events, placed) + [
            event
            for event in events
            if isinstance(event, SupportEvent)
            and event.tracked_object is placed
            and event.with_object not in robot_bodies
        ]

    def holds(self, action: PlaceAction, evidence: list[DetectionEvent]) -> bool:
        return any(isinstance(event, LossOfGraspEvent) for event in evidence) and any(
            isinstance(event, SupportEvent) for event in evidence
        )
