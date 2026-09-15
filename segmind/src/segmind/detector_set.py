"""
Building SegMind out of the detectors a caller asks for, each one bringing along the
detectors it cannot detect without.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from krrood.utils import recursive_subclasses
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, List, Optional, Self, Tuple, Type

from segmind.datastructures.events import DetectionEvent
from segmind.detectors import (
    atomic_event_detectors_nodes,
    coarse_event_detector_nodes,
    grasp_detector_nodes,
    joint_detector_nodes,
    spatial_relation_detector_nodes,
)
from segmind.detectors.base import AbstractDetector, DetectorStateChart
from segmind.exceptions import NoDetectorDetectsEvent
from segmind.scene_parts import SceneParts

DETECTOR_MODULES = (
    atomic_event_detectors_nodes,
    coarse_event_detector_nodes,
    grasp_detector_nodes,
    joint_detector_nodes,
    spatial_relation_detector_nodes,
)
"""
The modules segmind's own kinds of detector are defined in; a module defining a new kind
joins these.
"""


@dataclass(frozen=True)
class DetectorIdentity:
    """
    What tells two detectors apart: their kind and what they watch.
    """

    detector_type: Type[AbstractDetector]
    """
    The detector's kind.
    """

    watched_entities: Tuple[Optional[Body], ...]
    """
    What the detector watches (see :meth:`AbstractDetector.watched_entities`).
    """

    @classmethod
    def of(cls, detector: AbstractDetector) -> DetectorIdentity:
        """
        :param detector: The detector to identify.
        """
        return cls(
            detector_type=type(detector), watched_entities=detector.watched_entities()
        )


@dataclass
class DetectorSet:
    """
    The detectors one run of SegMind ticks, closed under what each of them needs, in the
    order they have to tick.

    A detector the caller adds replaces one of the same identity that was brought along
    for another detector; one brought along never replaces one already there.
    """

    scene: SceneParts
    """
    The parts of the scene the detectors read.
    """

    _detectors: Dict[DetectorIdentity, AbstractDetector] = field(
        init=False, default_factory=dict, repr=False
    )
    """
    Every detector of the set, by identity.
    """

    @classmethod
    def with_all_detectors(cls, scene: SceneParts) -> Self:
        """
        One detector of every kind segmind defines, watching every trackable body,
        leaving out a kind whose needs the scene cannot meet.

        :param scene: The parts of the scene the detectors read.
        """
        detectors = cls(scene=scene)
        for detector_type in cls._detector_types():
            for detector in detector_type.instances_for(None, scene):
                if detectors._needs_can_be_met(detector):
                    detectors.add(detector)
        return detectors

    def add(self, detector: AbstractDetector) -> Self:
        """
        Add a detector, and the detectors it needs.

        :param detector: The detector to add.
        :raises NoDetectorDetectsEvent: If nothing in the scene detects an event it needs.
        """
        self._bring_along_what_is_needed_by(detector)
        self._detectors[DetectorIdentity.of(detector)] = detector
        return self

    def add_detecting(
        self, event_type: Type[DetectionEvent], tracked_object: Optional[Body] = None
    ) -> Self:
        """
        Add the detectors of the kind that detects ``event_type``, and what they need.

        :param event_type: The kind of event to detect.
        :param tracked_object: The body to watch, or None for every trackable body.
        :raises NoDetectorDetectsEvent: If nothing in the scene detects that event.
        """
        for detector in self._detectors_detecting(event_type, tracked_object):
            self.add(detector)
        return self

    @property
    def detectors(self) -> List[AbstractDetector]:
        """
        Every detector of the set, each after the detectors it needs.
        """
        ordered: List[AbstractDetector] = []
        for detector in self._detectors.values():
            self._place_after_what_it_needs(detector, ordered)
        return ordered

    def build_statechart(self) -> DetectorStateChart:
        """
        :return: A statechart ticking every detector of the set, in :attr:`detectors`
            order.
        """
        statechart = DetectorStateChart()
        statechart.add_nodes(self.detectors)
        return statechart

    @classmethod
    def _detector_types(cls) -> List[Type[AbstractDetector]]:
        """
        :return: Every kind of detector segmind defines that binds the events it handles.
        """
        module_names = {module.__name__ for module in DETECTOR_MODULES}
        return [
            detector_type
            for detector_type in recursive_subclasses(AbstractDetector)
            if detector_type.__module__ in module_names
            and not inspect.isabstract(detector_type)
            and all(
                isinstance(parameter, type)
                for parameter in detector_type.get_generic_type_parameters()
            )
        ]

    def _candidates_detecting(
        self, event_type: Type[DetectionEvent], tracked_object: Optional[Body]
    ) -> List[AbstractDetector]:
        """
        :param event_type: The kind of event to detect.
        :param tracked_object: The body to watch, or None for every trackable body.
        :return: New detectors of the kind detecting ``event_type``, one per part of the
            scene it watches; none if no kind detects it or the scene lacks its parts.
        """
        return [
            detector
            for detector_type in self._detector_types()
            if detector_type.detected_event_type() is event_type
            for detector in detector_type.instances_for(tracked_object, self.scene)
        ]

    def _detectors_detecting(
        self, event_type: Type[DetectionEvent], tracked_object: Optional[Body]
    ) -> List[AbstractDetector]:
        """
        Like :meth:`_candidates_detecting`, refusing when there are none.

        :raises NoDetectorDetectsEvent: If nothing in the scene detects ``event_type``.
        """
        candidates = self._candidates_detecting(event_type, tracked_object)
        if not candidates:
            raise NoDetectorDetectsEvent(
                event_type=event_type, tracked_object=tracked_object
            )
        return candidates

    def _needs_can_be_met(self, detector: AbstractDetector) -> bool:
        """
        :return: Whether the scene has detectors for everything ``detector`` needs, and
            for everything those need in turn.
        """
        for needed in detector.required_event_types():
            candidates = self._candidates_detecting(needed, detector.tracked_object)
            if not candidates or not all(
                self._needs_can_be_met(candidate) for candidate in candidates
            ):
                return False
        return True

    def _bring_along_what_is_needed_by(self, detector: AbstractDetector) -> None:
        """
        Add a detector for every event ``detector`` needs that the set does not detect
        yet, and what those need in turn.

        :raises NoDetectorDetectsEvent: If nothing in the scene detects a needed event.
        """
        for needed in detector.required_event_types():
            for producer in self._detectors_detecting(needed, detector.tracked_object):
                identity = DetectorIdentity.of(producer)
                if identity in self._detectors:
                    continue
                self._bring_along_what_is_needed_by(producer)
                self._detectors[identity] = producer

    def _place_after_what_it_needs(
        self, detector: AbstractDetector, ordered: List[AbstractDetector]
    ) -> None:
        """
        Append ``detector`` to ``ordered`` after the set's detectors it needs, unless it
        is already there.
        """
        if any(placed is detector for placed in ordered):
            return
        for needed in detector.required_event_types():
            for producer in self._detectors.values():
                if (
                    producer.detected_event_type() is needed
                    and producer.tracked_object is detector.tracked_object
                ):
                    self._place_after_what_it_needs(producer, ordered)
        ordered.append(detector)
