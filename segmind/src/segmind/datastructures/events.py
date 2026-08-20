from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

from typing_extensions import List

from segmind.datastructures.object_tracker import (
    ObjectEventTracker,
    ObjectTrackerFactory,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.spatial_types.numeric import NumericPose
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import Body

# %% identity of a detected event


@dataclass(eq=False)
class DetectionEvent(ABC):
    """
    An occurrence detected in an episode.

    An event is identified by its type, the bodies taking part in it and the time it
    occurred at, so subclasses declare their :attr:`participants` and inherit the rest.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    """
    The time at which the event occurred, defaults to current time.
    """

    @property
    @abstractmethod
    def participants(self) -> List[Body]:
        """
        :return: the bodies taking part in this event.
        """

    def update_object_trackers_with_event(self, factory: ObjectTrackerFactory) -> None:
        """
        Register this event with the tracker of every participant.

        :param factory: factory used to look up per-object trackers.
        """
        for participant in self.participants:
            factory.get_tracker(participant).add_event(self)

    def __str__(self) -> str:
        names = " - ".join(str(participant.name) for participant in self.participants)
        return f"{type(self).__name__}: {names} - {self.timestamp}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is type(self)
            and other.participants == self.participants
            and other.timestamp == self.timestamp
        )

    def __hash__(self) -> int:
        return hash((type(self), tuple(self.participants), self.timestamp))


# %% participant roles


@dataclass(eq=False, kw_only=True)
class EventAboutObject(DetectionEvent, ABC):
    """
    An event predicated of a single body.
    """

    tracked_object: Body
    """The body this event is about."""

    @property
    def participants(self) -> List[Body]:
        return [self.tracked_object]

    @cached_property
    def object_tracker(self) -> ObjectEventTracker:
        """
        :return: the event tracker for :attr:`tracked_object`.
        """
        return ObjectTrackerFactory.get_tracker(self.tracked_object)


@dataclass(eq=False, kw_only=True)
class RelationEvent(EventAboutObject, ABC):
    """
    An event predicated of a body and the body it stands in relation to.

    A subclass naming further participants extends :attr:`participants`, which keeps
    them in the event's identity, its string form and its tracker registration.
    """

    with_object: Body
    """The body the relation holds with."""

    @property
    def participants(self) -> List[Body]:
        return super().participants + [self.with_object]

    @cached_property
    def with_object_tracker(self) -> ObjectEventTracker:
        """
        :return: the event tracker for :attr:`with_object`.
        """
        return ObjectTrackerFactory.get_tracker(self.with_object)


# %% support


@dataclass(eq=False)
class SupportEvent(RelationEvent):
    """
    The SupportEvent class is used to represent an event that involves an object that is supported by another object.
    """


@dataclass(eq=False)
class LossOfSupportEvent(RelationEvent):
    """
    The LossOfSupportEvent class is used to represent an event that involves an object that was supported by another
    object and then lost support.
    """


# %% motion


@dataclass(eq=False)
class MotionEvent(EventAboutObject, ABC):
    """
    Used to represent an event that involves an object that was stationary and then moved or
    vice versa.
    """

    start_pose: Pose = field(default_factory=Pose)
    """
    The pose of the object at the start of the event.
    """
    current_pose: Pose = field(default_factory=Pose)
    """
    The pose of the object at the end of the event.
    """


@dataclass(eq=False, init=False)
class TranslationEvent(MotionEvent):
    """
    Represents an event where an object moves from one location to another.
    """

    ...


@dataclass(eq=False, init=False)
class RotationEvent(MotionEvent):
    """
    Represents an event where an object rotates around a center point.
    """

    ...


@dataclass(eq=False, init=False)
class StopTranslationEvent(MotionEvent):
    """
    Represents an event where an object stops moving.
    """

    ...


@dataclass(eq=False, init=False)
class StopRotationEvent(MotionEvent):
    """
    Represents an event where an object stops rotating.
    """

    ...


# %% contact


@dataclass(eq=False)
class AbstractContactEvent(RelationEvent, ABC):
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

    bounding_box: BoundingBox = field(init=False)
    """
    Bounding box of the object.
    """

    pose: NumericPose = field(init=False)
    """
    Pose of the object, read out into numbers so a detector thread can record it.
    """

    with_object_bounding_box: BoundingBox = field(init=False)
    """
    Bounding box of the second object in contact.
    """

    with_object_pose: NumericPose = field(init=False)
    """
    Pose of the second object in contact, read out into numbers.
    """

    def __post_init__(self):
        self.bounding_box = BoundingBox.from_mesh(
            self.tracked_object.combined_mesh,
            origin=self.tracked_object.numeric_global_transform,
        )
        self.pose = self.tracked_object.numeric_global_pose
        self.with_object_bounding_box = BoundingBox.from_mesh(
            self.with_object.combined_mesh,
            origin=self.with_object.numeric_global_transform,
        )
        self.with_object_pose = self.with_object.numeric_global_pose


@dataclass(eq=False, init=False)
class ContactEvent(AbstractContactEvent):
    """
    Represents an event where two objects are in contact with each other.
    """

    ...


@dataclass(eq=False, init=False)
class LossOfContactEvent(AbstractContactEvent):
    """
    Represents an event where two objects are no longer in contact with each other.
    """

    ...


# %% interactions


@dataclass(eq=False)
class PickUpEvent(RelationEvent):
    """
    Represents an event where an object is picked up off the object supporting it.
    """

    ...


@dataclass(eq=False)
class PlacingEvent(RelationEvent):
    """
    Represents an event where an object is placed on another object.
    """

    ...


@dataclass(eq=False)
class InsertionEvent(RelationEvent):
    """
    Represents an event where an object is inserted through an aperture into other
    objects.
    """

    inserted_into_objects: List[Body] = field(default_factory=list)
    """
    List of objects into which the object was inserted.
    """

    @property
    def participants(self) -> List[Body]:
        return super().participants + self.inserted_into_objects

    @property
    def through_hole(self) -> Aperture:
        """
        :return: the aperture the tracked object was inserted through.
        """
        return self.with_object.get_semantic_annotations_by_type(type_=Aperture)[0]


# %% containment


@dataclass(eq=False)
class ContainmentEvent(RelationEvent):
    """
    Represents an event where an object is contained in another object.
    """

    ...


@dataclass(eq=False)
class LossOfContainmentEvent(RelationEvent):
    """
    Represents an event where an object is no longer contained in another object.
    """

    ...
