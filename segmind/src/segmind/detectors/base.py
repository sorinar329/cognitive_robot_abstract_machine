from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Callable

from datetime import timedelta

from typing_extensions import TYPE_CHECKING, Generic, Self, Tuple, Type, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from giskardpy.motion_statechart.context import (
    MotionStatechartContext,
    ContextExtension,
)
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from segmind.datastructures.events import (
    MotionEvent,
    DetectionEvent,
    LiftEvent,
    RotationEvent,
)
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.event_logger import EventLogger
from semantic_digital_twin.semantic_annotations.mixins import HasMechanicalJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.spatial_types.numeric import NumericPose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body, Region

if TYPE_CHECKING:
    from segmind.scene_parts import SceneParts


@dataclass
class DetectorStateChart(MotionStatechart):
    """
    Statechart responsible for running the different motion detectors.

    Currently acts as a container for the detectors and inherits the
    functionality from MotionStatechart.
    """


IndexedBodyPairs = Dict[Body, Set[Body]]
"""
Type hint for dictionaries mapping bodies to sets of bodies
"""


@dataclass
class SegmindContext(ContextExtension):
    """
    Context object shared across the motion statechart detectors.

    Stores the latest detected contact and support relationships
    between bodies in the simulation as well as the event logger.
    """

    latest_contact_bodies: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies it is currently in contact with.
    """

    latest_support: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently support it.
    """

    latest_containments: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently contain it.
    """

    latest_motion_events: Dict[Body, MotionEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active motion event, if any.
    """

    rest_poses: Dict[Body, NumericPose] = field(default_factory=dict)
    """
    Where each tracked body was last at rest as far as its motion has been reported:
    where it was first seen, then wherever a translation of it was reported to end.

    What a change of place no translation event claims is measured from.
    """

    latest_rotation_events: Dict[Body, RotationEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active rotation event, if any.
    """

    latest_lift_events: Dict[Body, LiftEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active lift event, if any.
    """

    latest_grasp: IndexedBodyPairs = field(default_factory=dict)
    """
    Each body currently considered grasped, mapped to the tool frames of the grippers
    holding it (in contact with both of a gripper's fingers and close to its tool center
    point; see :class:`~segmind.detectors.grasp_detector_nodes.GraspDetector`).

    Read by :class:`~segmind.detectors.atomic_event_detectors_nodes.LiftDetector` to
    gate lifting on the object actually being held, not just moving upward on its own.
    """

    logger: EventLogger = field(default_factory=EventLogger)
    """
    The event logger used to record detected events.
    """

    holes: List[Aperture] = field(default_factory=list)
    """
    List of bodies that can be considered holes
    """

    hole_regions: Dict[Aperture, Region] = field(default_factory=dict)
    """
    Every :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Aperture`
    in the world, mapped to its own :class:`Region` root.

    Unlike :attr:`holes` (populated by matching plain :class:`Body` names), an aperture's
    root is a virtual :class:`Region`, never registered with the world's collision
    manager, so it needs its own registry checked by :class:`~segmind.detectors.spatial_relation_detector_nodes.HoleContactDetector`
    and friends instead of ``context.world.bodies_with_collision``.
    """

    latest_hole_contacts: Dict[Body, Set[Region]] = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of hole regions (see :attr:`hole_regions`) it
    is currently overlapping.

    Kept separate from :attr:`latest_contact_bodies` rather than reused for it: a hole
    region is detected by a different (mesh-overlap) check than real body-body
    :attr:`latest_contact_bodies`, and mixing the two would make
    :class:`~segmind.detectors.atomic_event_detectors_nodes.LossOfContactDetector`
    (which only ever recomputes real contact) immediately -- and incorrectly -- drop
    every hole entry the very next tick.
    """

    articulated_parts: List[HasMechanicalJoint] = field(default_factory=list)
    """
    Every part of the scene that moves on a joint and has a handle (see
    :attr:`~segmind.scene_parts.SceneParts.articulated_parts`).
    """

    tracker_registry: ObjectTrackerFactory = field(default_factory=ObjectTrackerFactory)
    """
    The object tracker registry.    
    """


TDetectedEvent = TypeVar("TDetectedEvent", bound=DetectionEvent)
"""
The kind of event a detector detects.
"""


@dataclass(repr=False, eq=False)
class AbstractDetector(
    MotionStatechartNode, Generic[TDetectedEvent], SubClassSafeGeneric, ABC
):
    """
    Abstract base class for all detectors.

    A detector binds the kind of event it detects, and states the kinds of event it
    cannot detect without (see :class:`~segmind.detector_set.DetectorSet`).
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    """
    :param tracked_object: Optional body that should be monitored.
    If None, all trackable objects in the world are checked.
    """

    @classmethod
    def detected_event_type(cls) -> Type[DetectionEvent]:
        """
        :return: The kind of event this detector detects.
        """
        return cls.get_generic_type_parameters()[0]

    @classmethod
    def required_event_types(cls) -> Tuple[Type[DetectionEvent], ...]:
        """
        :return: The kinds of event this detector cannot detect without, because it
            continues or concludes from them; none for a detector that reads only the
            world.
        """
        return ()

    @classmethod
    def instances_for(
        cls, tracked_object: Optional[Body], scene: SceneParts
    ) -> List[Self]:
        """
        The detectors of this kind that watch one object in a scene.

        :param tracked_object: The body to watch, or None for every trackable body.
        :param scene: The parts of the scene detectors read.
        :return: One detector for a kind that watches only the object; one per part for
            a kind that also watches a part of the scene, none when the scene has none.
        """
        return [cls(tracked_object=tracked_object)]

    def watched_entities(self) -> Tuple[Optional[Body], ...]:
        """
        :return: What this detector watches, which tells it apart from another detector
            of the same kind.
        """
        return (self.tracked_object,)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Executes one update cycle of the detector.

        Determines the objects that should be checked for contacts,
        computes new contact relationships, and triggers events if
        contact changes are detected.

        :param context: The current motion statechart context.
        :return: ObservationStateValues.TRUE if events were triggered,
        otherwise ObservationStateValues.FALSE.
        """
        segmind_context_extension = context.require_extension(SegmindContext)

        objects_to_check = self.bodies_to_check(context)
        events = self.update_context_and_events(
            context, segmind_context_extension, objects_to_check
        )
        for e in events:
            segmind_context_extension.logger.log_event(
                e, segmind_context_extension.tracker_registry
            )
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE

    def bodies_to_check(self, context: MotionStatechartContext) -> List[Body]:
        """
        :param context: The current motion statechart context.
        :return: The bodies this detector checks in a tick: its tracked object, or every
            body free to move in the world when it tracks none.
        """
        if self.tracked_object is not None:
            return [self.tracked_object]
        return [
            body
            for body in context.world.bodies
            if type(body.parent_connection) is Connection6DoF
        ]

    def get_relation(
        self,
        context: MotionStatechartContext,
        tracked_objects: List[Body],
        predicate: Callable[[Body, Body], bool],
    ) -> Dict[Body, Set[Body]]:
        """
        Get the relation between tracked objects.

        :param context: The context containing world information.
        :param tracked_objects: List of bodies to check for contact changes.
        :param predicate: Function that returns true if the objects are related.
        :return: Dictionary mapping bodies to sets of related bodies.
        """

        related_bodies: Dict[Body, Set[Body]] = {}
        bodies_with_collision = context.world.bodies_with_collision
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if body is obj:
                    continue
                if predicate(obj, body):
                    related_bodies.setdefault(obj, set()).add(body)
        return related_bodies

    def get_relation_to_regions(
        self,
        tracked_objects: List[Body],
        regions: List[Region],
        predicate: Callable[[Body, Region], bool],
    ) -> Dict[Body, Set[Region]]:
        """
        Like :meth:`get_relation`, but checked against a caller-supplied list of regions
        instead of ``context.world.bodies_with_collision``.

        A :class:`Region` (e.g. a hole's own root) is never registered with the world's
        collision manager, so it is never among ``bodies_with_collision`` and needs its
        candidates supplied directly.

        :param tracked_objects: List of bodies to check for a relation to a region.
        :param regions: The regions to check each tracked object against.
        :param predicate: Function that returns true if a body and a region are related.
        :return: Dictionary mapping bodies to sets of related regions.
        """
        related_regions: Dict[Body, Set[Region]] = {}
        for obj in tracked_objects:
            for region in regions:
                if predicate(obj, region):
                    related_regions.setdefault(obj, set()).add(region)
        return related_regions

    @abstractmethod
    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Core detection logic that updates the internal state and identifies new events.

        This method is called during every tick of the detector. Implementations should
        examine the current state of the world (via the context) for the given
        `tracked_objects`, update the relevant fields in `context` (e.g.,
        `latest_contact_bodies`, `latest_support`), and return a list of any
        `DetectionEvent`s that occurred since the last update.

        Specific implementations may detect:
        * State changes: e.g., a new contact (ContactEvent) or loss of contact.
        * Continuous processes: e.g., ongoing motion or containment.
        * Complex interactions: e.g., insertion or picking up objects.

        :param context: The shared SegmindContext containing the world state,
                        history of relationships, and the event logger.
        :param segmind_context: The SegmindContext extension containing additional states.
        :param tracked_objects: A list of bodies that this detector should focus on
                                during this update cycle.
        :return: A list of DetectionEvent objects representing the events detected
                 in this cycle. Returns an empty list if no events were found.
        """
        pass


DEFAULT_SHIFT_THRESHOLD = timedelta(seconds=15)
"""
The default for :attr:`RuleDetector.shift_threshold`.
"""

TFirstEvidence = TypeVar("TFirstEvidence", bound=DetectionEvent)
"""
The first kind of event a rule detector concludes from.
"""

TSecondEvidence = TypeVar("TSecondEvidence", bound=DetectionEvent)
"""
The second kind of event a rule detector concludes from.
"""


@dataclass(repr=False, eq=False)
class RuleDetector(
    AbstractDetector[TDetectedEvent],
    Generic[TDetectedEvent, TFirstEvidence, TSecondEvidence],
    ABC,
):
    """
    A detector that concludes its event from two earlier kinds of event by a rule (see
    :mod:`segmind.detectors.rules`), instead of reading the world.

    The two kinds it binds are the ones it needs, so a set of detectors brings along the
    detectors producing them.
    """

    shift_threshold: timedelta = DEFAULT_SHIFT_THRESHOLD
    """
    How far apart in time the two events may be and still be one occurrence.
    """

    @classmethod
    def first_evidence_type(cls) -> Type[DetectionEvent]:
        """
        :return: The first kind of event the rule concludes from.
        """
        return cls.get_generic_type_parameters()[1]

    @classmethod
    def second_evidence_type(cls) -> Type[DetectionEvent]:
        """
        :return: The second kind of event the rule concludes from.
        """
        return cls.get_generic_type_parameters()[2]

    @classmethod
    def required_event_types(cls) -> Tuple[Type[DetectionEvent], ...]:
        return (cls.first_evidence_type(), cls.second_evidence_type())
