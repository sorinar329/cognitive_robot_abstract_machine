from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

from typing_extensions import Optional, List

from krrood.entity_query_language.explanation.explanation import explain_inference
from krrood.symbol_graph.symbol_graph import Symbol
from segmind.datastructures.object_tracker import (
    ObjectEventTracker,
    ObjectTrackerFactory,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.spatial_types.numeric import NumericPose
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass
class DetectionEvent(Symbol, ABC):
    """
    Something the segmentation saw happen.

    A symbol, so that what the robot has seen is part of what it can be asked about
    without anyone having to hand the events to the question.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    """
    The time at which the event occurred, defaults to current time.
    """

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    def participating_events(self) -> List[DetectionEvent]:
        """
        The events a rule consumed to conclude this one.

        Empty for an event no rule produced, such as one an atomic detector built
        directly.
        """
        explanation = explain_inference(self)
        if explanation is None:
            return []
        consumed = explanation.get_values_of_variable_nodes_of_given_type(
            DetectionEvent
        )
        return [event for event in consumed.tolist() if event is not self]


@dataclass(kw_only=True)
class EventWithTrackedObjects(DetectionEvent, ABC):
    """
    An abstract event involving one or more tracked objects.

    Provides the primary :attr:`tracked_object` and an optional :attr:`with_object`,
    along with ORM frozen copies and per-object tracker access.
    """

    tracked_object: Body
    """
    The primary object involved in this event.
    """

    with_object: Optional[KinematicStructureEntity] = None
    """
    The secondary object involved in this event, if any.

    Usually a :class:`~semantic_digital_twin.world_description.world_entity.Body`; a
    hole-related event (e.g. contact with an
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Aperture`)
    sets this to that aperture's own
    :class:`~semantic_digital_twin.world_description.world_entity.Region` root instead,
    since an aperture is a virtual opening rather than a collidable body.
    """

    @property
    def tracked_objects(self) -> List[KinematicStructureEntity]:
        """
        :return: the primary object, plus the secondary object when present.
        """
        return (
            [self.tracked_object]
            if self.with_object is None
            else [self.tracked_object, self.with_object]
        )

    @cached_property
    def object_tracker(self) -> ObjectEventTracker:
        """
        :return: the event tracker for :attr:`tracked_object`.
        """
        return ObjectTrackerFactory.get_tracker(self.tracked_object)

    @cached_property
    def with_object_tracker(self) -> Optional[ObjectEventTracker]:
        """
        :return: the event tracker for :attr:`with_object`, or ``None`` if absent.
        """
        return (
            ObjectTrackerFactory.get_tracker(self.with_object)
            if self.with_object is not None
            else None
        )

    def update_object_trackers_with_event(self, factory: ObjectTrackerFactory) -> None:
        """
        Register this event with the tracker of every involved object.

        :param factory: factory used to look up per-object trackers.
        """
        for obj in self.tracked_objects:
            factory.get_tracker(obj).add_event(self)

    def __str__(self) -> str:
        names = " - ".join(str(obj.name) for obj in self.tracked_objects)
        return f"{self.__class__.__name__}: {names} - {self.timestamp}"

    def __eq__(self, other) -> bool:
        return (
            other.__class__ == self.__class__
            and self.tracked_objects == other.tracked_objects
            and self.timestamp == other.timestamp
        )

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(self.tracked_objects), self.timestamp))


# %% relations that begin and end


@dataclass(unsafe_hash=True)
class SupportEvent(EventWithTrackedObjects):
    """
    The SupportEvent class is used to represent an event that involves an object that is
    supported by another object.
    """


@dataclass(unsafe_hash=True)
class LossOfSupportEvent(EventWithTrackedObjects):
    """
    The LossOfSupportEvent class is used to represent an event that involves an object
    that was supported by another object and then lost support.
    """


@dataclass(unsafe_hash=True)
class MotionEvent(EventWithTrackedObjects, ABC):
    """
    Used to represent an event that involves an object that was stationary and then
    moved or vice versa.
    """

    start_pose: NumericPose = field(kw_only=True)
    """
    The pose of the object at the start of the event, in the world frame.
    """

    current_pose: NumericPose = field(kw_only=True)
    """
    The pose of the object at the end of the event, in the world frame.
    """


@dataclass(init=False, unsafe_hash=True)
class TranslationEvent(MotionEvent):
    """
    Represents an event where an object moves from one location to another.
    """

    ...


@dataclass(init=False, unsafe_hash=True)
class RotationEvent(MotionEvent):
    """
    Represents an event where an object rotates around a center point.
    """

    ...


@dataclass(init=False, unsafe_hash=True)
class StopTranslationEvent(MotionEvent):
    """
    Represents an event where an object stops moving.
    """

    ...


@dataclass(init=False, unsafe_hash=True)
class StopRotationEvent(MotionEvent):
    """
    Represents an event where an object stops rotating.
    """

    ...


@dataclass(init=False, unsafe_hash=True)
class LiftEvent(MotionEvent):
    """
    Represents an event where a grasped object starts moving upward along the Z axis.
    """

    ...


@dataclass(init=False, unsafe_hash=True)
class StopLiftEvent(MotionEvent):
    """
    Represents an event where a lifted object stops moving upward.
    """

    ...


@dataclass(unsafe_hash=True)
class AbstractContactEvent(EventWithTrackedObjects, ABC):
    """
    Represents an event where two objects are in contact with each other.
    """

    contact_bodies: list[Body] = field(init=False, default_factory=list)
    """
    The bodies that are in contact with each other.
    """

    latest_contact_bodies: list[Body] = field(init=False, default_factory=list)
    """
    The bodies that were in contact with each other in the previous time step.
    """

    bounding_box: VolumetricBoundingBox = field(init=False)
    """
    Bounding box of the object.
    """

    pose: NumericPose = field(init=False)
    """
    Pose of the object, in the world frame.
    """

    with_object_bounding_box: Optional[VolumetricBoundingBox] = field(
        init=False, default=None
    )
    """
    Bounding box of the second object in contact.
    """

    with_object_pose: Optional[NumericPose] = field(init=False, default=None)
    """
    Pose of the second object in contact, in the world frame.
    """

    def __post_init__(self):
        # an event read back from a record is about a body that stands in no world any
        # more, so there is nothing to read its pose off; the numbers it read when it
        # happened are the record's
        if self.tracked_object._world is None:
            return
        # combined_mesh (not tracked_object.collision.combined_mesh directly) so this
        # also works when with_object is a hole's Region root, which exposes its
        # geometry via .area rather than .collision.
        self.bounding_box = VolumetricBoundingBox.from_mesh(
            self.tracked_object.combined_mesh,
            origin=self.tracked_object.numeric_global_transform,
        )
        self.pose = self.tracked_object.numeric_global_pose

        if self.with_object is not None:
            self.with_object_bounding_box = VolumetricBoundingBox.from_mesh(
                self.with_object.combined_mesh,
                origin=self.with_object.numeric_global_transform,
            )
            self.with_object_pose = self.with_object.numeric_global_pose


@dataclass(init=False, unsafe_hash=True)
class ContactEvent(AbstractContactEvent):
    """
    Represents an event where two objects are in contact with each other.
    """

    ...


@dataclass(init=False, unsafe_hash=True)
class LossOfContactEvent(AbstractContactEvent):
    """
    Represents an event where two objects are no longer in contact with each other.
    """

    ...


@dataclass
class AgentInteractionEvent(EventWithTrackedObjects, ABC):
    """
    An event in which an agent acted on the tracked object rather than one where the
    object was only observed.

    The object acted on is the one the event already tracks, so asking which objects an
    agent acted on is asking these events for their :attr:`tracked_object`.
    """


@dataclass(unsafe_hash=True)
class GraspEvent(EventWithTrackedObjects):
    """
    Represents an event where an object starts being held by a gripper: in contact with
    both of the gripper's fingers, and close to its tool center point.

    :attr:`~EventWithTrackedObjects.with_object` is the gripper's own tool center point
    body (its ``tool_frame``), not either finger individually.
    """

    ...


@dataclass(unsafe_hash=True)
class LossOfGraspEvent(EventWithTrackedObjects):
    """
    Represents an event where an object previously held by a gripper (see
    :class:`GraspEvent`) is no longer in contact with both of its fingers and close to
    its tool center point.
    """

    ...


@dataclass(unsafe_hash=True)
class PickUpEvent(AgentInteractionEvent):
    """
    Represents an event where an object is picked up by another object.
    """


@dataclass(unsafe_hash=True)
class PlacingEvent(AgentInteractionEvent):
    """
    Represents an event where an object is placed on another object.
    """


@dataclass(unsafe_hash=True)
class InsertionEvent(AgentInteractionEvent):
    """
    Represents an event where an object is inserted into another object.
    """

    inserted_into_objects: List[KinematicStructureEntity] = field(default_factory=list)
    """
    List of objects into which the object was inserted.

    A hole-related insertion sets this to the hole's own ``Region`` root (see
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Aperture`),
    not a ``Body``, which is why this is stated over their common base.
    """

    through_hole: Optional[Aperture] = None
    """
    The aperture :attr:`~EventWithTrackedObjects.with_object` (its own ``Region`` root)
    was detected passing through.

    Set directly by the detector that builds this event, which already has the aperture
    in hand (via ``SegmindContext.hole_regions``) rather than derived from
    ``with_object`` here: a hole's root is a virtual ``Region``, not a ``Body``, and has
    no reliable way to look its owning annotation back up on its own.
    """

    def __str__(self) -> str:
        with_object_name = " - " + " - ".join(
            [str(obj.name) for obj in self.inserted_into_objects]
        )
        return f"{self.__class__.__name__}: {self.tracked_object.name}{with_object_name} - {self.timestamp}"


@dataclass(unsafe_hash=True)
class ContainmentEvent(EventWithTrackedObjects):
    """
    Represents an event where an object is contained in another object.
    """

    ...


@dataclass(unsafe_hash=True)
class LossOfContainmentEvent(EventWithTrackedObjects):
    """
    Represents an event where an object is no longer contained in another object.
    """

    ...
