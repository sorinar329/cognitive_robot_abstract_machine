from __future__ import annotations

from dataclasses import dataclass

from giskardpy.motion_statechart.context import MotionStatechartContext
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Generic, List

from segmind.datastructures.events import (
    DetectionEvent,
    GraspEvent,
    GraspingEvent,
    LossOfSupportEvent,
    PickUpEvent,
    PlacingEvent,
    StopTranslationEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.detectors.base import (
    RuleDetector,
    SegmindContext,
    TDetectedEvent,
    TFirstEvidence,
    TSecondEvidence,
)
from segmind.detectors.rules import Evidence, interaction_rule


@dataclass(eq=False, repr=False)
class InteractionDetector(
    RuleDetector[TDetectedEvent, TFirstEvidence, TSecondEvidence],
    Generic[TDetectedEvent, TFirstEvidence, TSecondEvidence],
):
    """
    A detector concluding an interaction from two events about the same object close in
    time, by :func:`~segmind.detectors.rules.interaction_rule`: the object is taken from
    the first event, what it interacted with from the event :meth:`with_object_evidence`
    names.
    """

    @classmethod
    def with_object_evidence(cls) -> Evidence:
        """
        :return: Which of the two events the interaction takes what the object interacted
            with from; the second by default.
        """
        return Evidence.SECOND

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Concludes an interaction for every object whose two events happened within
        :attr:`~segmind.detectors.base.RuleDetector.shift_threshold`, and that was not
        already concluded for the same two entities.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param tracked_objects: List of bodies to analyze for potential interactions.
        :return: The interactions concluded.
        """
        return interaction_rule(
            event_type=self.detected_event_type(),
            primary_event_type=self.first_evidence_type(),
            secondary_event_type=self.second_evidence_type(),
            logged_events=segmind_context.logger.get_events(),
            shift_threshold=self.shift_threshold,
            with_object_from=self.with_object_evidence(),
        ).tolist()


@dataclass(eq=False, repr=False)
class PlacingDetector(
    InteractionDetector[PlacingEvent, StopTranslationEvent, SupportEvent]
):
    """
    Detects that an object was placed on another one: it came to a stop and was
    supported by something soon after.
    """


@dataclass(eq=False, repr=False)
class PickUpDetector(
    InteractionDetector[PickUpEvent, TranslationEvent, LossOfSupportEvent]
):
    """
    Detects that an object was picked up off whatever was supporting it: it started
    moving and lost its support soon after.
    """


@dataclass(eq=False, repr=False)
class GraspingDetector(InteractionDetector[GraspingEvent, GraspEvent, PickUpEvent]):
    """
    Detects that an object was grasped and picked up: a gripper grasped it and it was
    picked up soon after, the gripper being what it happened with.
    """

    @classmethod
    def with_object_evidence(cls) -> Evidence:
        return Evidence.FIRST
