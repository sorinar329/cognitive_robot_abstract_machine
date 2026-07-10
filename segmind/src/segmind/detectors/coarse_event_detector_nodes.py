from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Callable, Any, Set
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    SupportEvent,
    DetectionEvent,
    PlacingEvent, TranslationEvent, LossOfSupportEvent, PickUpEvent, StopTranslationEvent, ContactEvent,
    ContainmentEvent, InsertionEvent, StackingEvent,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootKinematicStructureEntity
from semantic_digital_twin.world_description.world_entity import Body
from segmind.detectors.base import AbstractDetector, SegmindContext


@dataclass
class AbstractInteractionDetector(AbstractDetector):
    """
    Abstract base class for interaction-based detectors.

    Provides shared functionality for monitoring interactions of
    bodies and generating events when detected.
    """

    shift_threshold: timedelta = timedelta(seconds=15)
    """
    The threshold for the time difference between two events to be considered an interaction.
    """

    def _find_interaction_events(
            self,
            segmind_context: SegmindContext,
            primary_event_type: type,
            secondary_event_type: type,
            make_event: Callable[[Any, Any], DetectionEvent],
    ) -> List[DetectionEvent]:
        """
        Scans logged events for correlated pairs of primary and secondary event types
        and emits a detection event for each new, unseen pairing.

        For each secondary event, this method searches for a primary event on the same
        tracked object whose timestamp is within :attr:`shift_threshold`. If such a pair
        is found and has not been recorded in ``segmind_context.placing_pairs`` before,
        the pair is registered and a detection event is produced via ``make_event``.

        :param segmind_context: The shared context holding the event logger and
            previously seen interaction pairs.
        :param primary_event_type: The event type to use as the primary signal
            (e.g. ``StopTranslationEvent`` for placing, ``TranslationEvent`` for pickup).
        :param secondary_event_type: The event type to correlate against the primary
            (e.g. ``SupportEvent`` for placing, ``LossOfSupportEvent`` for pickup).
        :param make_event: Factory called with ``(primary, secondary)`` to produce the
            outgoing :class:`DetectionEvent`. Only called once per unique pair.
        :return: List of newly detected interaction events.
        """
        primary_events = [
            e for e in segmind_context.logger.get_events()
            if isinstance(e, primary_event_type)
        ]
        secondary_events = [
            e for e in segmind_context.logger.get_events()
            if isinstance(e, secondary_event_type)
        ]

        events = []
        by_object = defaultdict(list)
        for e in primary_events:
            by_object[e.tracked_object].append(e)

        for secondary in secondary_events:
            for primary in by_object.get(secondary.tracked_object, []):
                if abs(secondary.timestamp - primary.timestamp) >= self.shift_threshold:
                    continue

                key = (secondary_event_type, secondary.tracked_object.id, secondary.with_object.id)
                if key in segmind_context.placing_pairs:
                    continue

                segmind_context.placing_pairs.add(key)
                events.append(make_event(primary, secondary))
                break

        return events


@dataclass
class PlacingDetector(AbstractInteractionDetector):
    """
    Represents a class detection mechanism for identifying and managing new
    placing events from observed system interactions.

    This class is typically used to analyze specific event types, such as stop
    motion and support events, and identify correlations that form the basis
    of new placing events. By ensuring that placing events are uniquely paired,
    the class helps maintain consistency and prevent duplication of events.
    """

    def update_context_and_events(self, context:MotionStatechartContext, segmind_context:SegmindContext, obj: List[Body]) -> List[DetectionEvent]:
        """
        Updates the system context with new placing event instances based on past
        actions logged in the system. It analyzes and filters specific event types
        to detect new events that can be generated. This function ensures distinct
        events are created by maintaining exclusivity through a pairing mechanism.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param obj: List of bodies to analyze for potential placing events.
        :return: List of generated placing events based on observed interactions.
        """
        return self._find_interaction_events(
            segmind_context,
            primary_event_type=StopTranslationEvent,
            secondary_event_type=SupportEvent,
            make_event=lambda primary, secondary: PlacingEvent(
                tracked_object=primary.tracked_object,
                with_object=secondary.with_object,
            ),
        )


@dataclass
class PickUpDetector(AbstractInteractionDetector):
    """
    Detects and processes interactions suggesting an object has been picked up.

    The PickUpDetector class determines if a "pickup" event has occurred by analyzing
    contextual events such as TranslationEvent and LossOfSupportEvent. It ensures
    that such events are detected and processed by checking their timestamps and
    associating them with corresponding objects. The resulting detected events are
    then returned. This class interfaces with a logger to gather the needed event
    data and uses a context to manage event pairs and thresholds.
    """

    def update_context_and_events(self, context:MotionStatechartContext, segmind_context: SegmindContext,obj: List[Body]) -> List[DetectionEvent]:
        """
        Updates the context and generates a list of events based on translation events and loss
        of support events. The method identifies pairs of related events that are close in
        timestamp, determines if they are exclusive, and creates a new event when conditions
        are met.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param obj: List of bodies to analyze for potential pickup events.
        :return: List of generated pickup events based on observed interactions.
        """
        return self._find_interaction_events(
            segmind_context,
            primary_event_type=TranslationEvent,
            secondary_event_type=LossOfSupportEvent,
            make_event=lambda primary, secondary: PickUpEvent(
                tracked_object=primary.tracked_object,
            ),
        )


@dataclass(eq=False, repr=False)
class StackingDetector(AbstractDetector):
    """
    Detects when a tracked object becomes newly stacked on another, similar object.

    An ``upper`` object counts as stacked on a ``lower`` one once all of the
    following currently hold, per :attr:`SegmindContext.latest_support`:

    * ``upper`` is currently supported by ``lower``.
    * ``upper`` and ``lower`` are similar, i.e. they share at least one
      semantic annotation type (e.g. both are annotated as ``Cube``).
    * ``lower`` is itself currently supported by something.

    The last condition is why a single object resting on the floor never
    produces a StackingEvent by itself: the floor is not itself supported by
    anything. Once a second, similar object is placed on top of it, that
    object's ``lower`` (the first one) is supported by the floor, so the
    condition is met and the pair fires. Reading current support state
    (rather than only past SupportEvents) is what keeps this correct if a
    lower support relation has since been lost -- a broken chain no longer
    satisfies the "lower is itself supported" check.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Finds newly-formed stacking pairs among the currently supported tracked objects.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param tracked_objects: List of bodies to check as the upper object of a stack.
        :return: List of newly detected StackingEvent instances.
        """
        events = []
        latest_support = segmind_context.latest_support

        for upper in tracked_objects:
            for lower in latest_support.get(upper, set()):
                if not latest_support.get(lower):
                    continue
                if not self._are_similar(context, upper, lower):
                    continue

                key = (upper.id, lower.id)
                if key in segmind_context.stacking_pairs:
                    continue

                segmind_context.stacking_pairs.add(key)
                events.append(StackingEvent(tracked_object=upper, with_object=lower))

        return events

    def _are_similar(self, context: MotionStatechartContext, a: Body, b: Body) -> bool:
        """
        Two bodies are similar if they share at least one semantic annotation type.

        :param context: The current motion statechart context, used to look up annotations.
        :param a: The first body.
        :param b: The second body.
        :return: True if ``a`` and ``b`` have a common semantic annotation type.
        """
        return bool(
            self._semantic_annotation_types(context, a)
            & self._semantic_annotation_types(context, b)
        )

    @staticmethod
    def _semantic_annotation_types(context: MotionStatechartContext, body: Body) -> Set[type]:
        """
        The semantic annotation types rooted at ``body``.

        Filters the world's full annotation list by root rather than reading
        it off the body directly: nothing currently populates a body's own
        semantic-annotation backref, so ``body`` itself cannot answer this.

        :param context: The current motion statechart context, used to look up annotations.
        :param body: The body to find semantic annotation types for.
        :return: Set of semantic annotation types rooted at ``body``.
        """
        return {
            type(annotation)
            for annotation in context.world.semantic_annotations
            if isinstance(annotation, HasRootKinematicStructureEntity) and annotation.root is body
        }
