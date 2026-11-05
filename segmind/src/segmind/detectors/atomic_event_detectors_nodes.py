"""
Detectors of the simplest events: contacts made and lost, and bodies starting and
stopping to move or turn.
"""

from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing_extensions import Any
import numpy as np

from segmind.datastructures.events import (
    DetectionEvent,
    ContactEvent,
    LossOfContactEvent,
    TranslationEvent,
    RotationEvent,
    StopTranslationEvent,
    StopRotationEvent,
)
from segmind.detectors.base import SegmindContext, AbstractDetector
from semantic_digital_twin.reasoning.predicates import InContactWith
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False, kw_only=True)
class ContactDetector(AbstractDetector):
    """
    Detects contacts between bodies being established and being lost.

    The detector reports a :class:`ContactEvent` when a body touches something new and a
    :class:`LossOfContactEvent` when it stops touching something it touched.

    With the robot left out, what its end effectors touch is still found and kept with
    every other contact, since that is what a grasp is read from, but it is not
    reported: the robot's touches are read as grasps and nothing else.
    """

    def bodies_left_out(self, world: World) -> set[Body]:
        """
        Every body of every robot apart from its end effectors, unless
        :attr:`exclude_robot` says otherwise.
        """
        return super().bodies_left_out(world) - self.end_effector_bodies(world)

    def update_context_and_events(
        self,
        segmind_context: SegmindContext,
        tracked_objects: list[Body],
    ) -> list[DetectionEvent]:
        """
        Detects newly formed and newly lost contacts and updates the stored contact
        state.

        :param segmind_context: The shared SegmindContext containing the information
            required to track events.
        :param tracked_objects: List of bodies to check for contacts.
        :return: The contacts formed, then the contacts lost, apart from those of the
            robot's end effectors while the robot is left out.
        """
        contacts_now = self.get_relation(tracked_objects, InContactWith)
        latest_contacts = segmind_context.latest_contact_bodies
        new_contacts = self.remember_new_relations(latest_contacts, contacts_now)
        lost_contacts = self.forget_lost_relations(
            latest_contacts, contacts_now, tracked_objects
        )
        unreported = (
            self.end_effector_bodies(self.context.world)
            if self.exclude_robot
            else set()
        )
        return [
            ContactEvent(tracked_object=body, with_object=other)
            for body, others in new_contacts.items()
            for other in others
            if other not in unreported
        ] + [
            LossOfContactEvent(tracked_object=body, with_object=other)
            for body, others in lost_contacts.items()
            for other in others
            if other not in unreported
        ]


@dataclass
class ObservedPose:
    """
    A pose of a body, together with when it was observed.
    """

    observed_at: float
    """
    When the pose was observed, in seconds on a clock that never goes backwards.
    """

    pose: Pose
    """
    The pose observed, in the world frame.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class MotionDetector(AbstractDetector):
    """
    Base class for motion-based detectors.

    Provides shared functionality for monitoring poses of bodies and generating events
    when movement is detected.
    """

    window_duration: float = 2.0
    """
    The time, in seconds, a body's motion is read over: where it is now is compared with
    where it was that long ago.
    """

    distance_threshold: float = 0.005
    """
    Threshold for the distance between two poses to be considered movement.
    """

    rotation_threshold: float = 0.1
    """
    Threshold for the rotation error between two poses to be considered rotation.
    """

    _pose_history: dict[Body, list[ObservedPose]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The poses of each tracked body observed within :attr:`window_duration`, oldest
    first, led by the last one observed at its start or before.

    Owned by this detector rather than shared through the context, so that no other
    motion detector in the statechart adds to it.
    """

    def update_context_and_events(
        self,
        segmind_context: SegmindContext,
        tracked_objects: list[Body],
    ) -> list[DetectionEvent]:
        """
        Updates the pose history for each tracked object and checks for motion events,
        once a body has been observed for :attr:`window_duration`.

        :param segmind_context: The shared SegmindContext containing the information
            required to track events.
        :param tracked_objects: List of bodies to update and check.
        :return: A list of events triggered during this update.
        """
        events = []
        window_start = segmind_context.observed_at - self.window_duration
        for tracked_object in tracked_objects:
            history = self._pose_history.setdefault(tracked_object, [])
            history.append(
                ObservedPose(segmind_context.observed_at, tracked_object.global_pose)
            )
            while len(history) > 1 and history[1].observed_at <= window_start:
                history.pop(0)
            if history[0].observed_at > window_start:
                continue

            event = self._check_and_trigger_event(
                segmind_context,
                tracked_object,
                [observed.pose for observed in history],
            )
            if event:
                events.append(event)
        return events

    @abstractmethod
    def _check_and_trigger_event(
        self, segmind_context: SegmindContext, tracked_object: Body, poses: list[Pose]
    ) -> DetectionEvent | None:
        """
        Subclass-specific logic to trigger a Motion or StopMotion event.

        Called once per tick per tracked object, with the full pose window, so each
        detector evaluates only the condition it is responsible for.

        :param segmind_context: The shared SegmindContext containing the information
            required to track events.
        :param tracked_object: The body to check.
        :param poses: The pose window of ``tracked_object``, oldest first.
        :return: A MotionEvent, StopMotionEvent or None.
        """
        pass

    def _is_moving(self, poses: list[Pose]) -> bool:
        """
        Determines whether an object is moving by evaluating the distance between the
        first and the last recorded position of the window.

        :param poses: The pose window of the body, oldest first.
        :return: True if the object is moving, False otherwise.
        """
        return (
            poses[0].position.euclidean_distance(poses[-1].position)
            > self.distance_threshold
        )

    def _is_rotating(self, poses: list[Pose]) -> bool:
        """
        Determines whether an object is rotating by evaluating the rotation error
        between the first and the last recorded pose of the window.

        :param poses: The pose window of the body, oldest first.
        :return: True if the object is rotating, False otherwise.
        """
        return (
            float(
                poses[0].rotation_matrix.rotational_distance(poses[-1].rotation_matrix)
            )
            > self.rotation_threshold
        )


@dataclass(eq=False, repr=False, kw_only=True)
class TranslationDetector(MotionDetector):
    """
    Detector for translations.

    Reports a :class:`TranslationEvent` when an object starts moving and a
    :class:`StopTranslationEvent` when an object that was moving stops.
    """

    def _check_and_trigger_event(
        self, segmind_context: SegmindContext, tracked_object: Body, poses: list[Pose]
    ) -> DetectionEvent | None:
        """
        Reports the object starting to move or, when it was moving, coming to a stop.

        No event is reported while the object stays as it was: stationary, or moving
        while a motion event for it is already active.

        :param segmind_context: The shared SegmindContext containing the information required to
            track events.
        :param tracked_object: The object being monitored for movement.
        :param poses: The pose window of ``tracked_object``, oldest first.
        :return: A TranslationEvent, a StopTranslationEvent or None.
        """
        latest_motion_event = segmind_context.latest_motion_events.get(tracked_object)
        if not self._is_moving(poses):
            if latest_motion_event is None:
                return None
            segmind_context.latest_motion_events.pop(tracked_object)
            return StopTranslationEvent(
                tracked_object=tracked_object,
                world_T_start_pose=latest_motion_event.world_T_start_pose,
                world_T_current_pose=poses[-1],
            )

        if latest_motion_event is not None:
            return None

        new_event = TranslationEvent(
            tracked_object=tracked_object,
            world_T_start_pose=poses[0],
            world_T_current_pose=poses[-1],
        )
        segmind_context.latest_motion_events[tracked_object] = new_event
        return new_event


@dataclass(eq=False, repr=False, kw_only=True)
class RotationDetector(MotionDetector):
    """
    Detector for rotations.

    Reports a :class:`RotationEvent` when an object starts rotating and a
    :class:`StopRotationEvent` when an object that was rotating stops.
    """

    def _check_and_trigger_event(
        self, segmind_context: SegmindContext, tracked_object: Body, poses: list[Pose]
    ) -> DetectionEvent | None:
        """
        Reports the object starting to rotate or, when it was rotating, coming to a
        stop.

        No event is reported while the object stays as it was: not rotating, or
        rotating while a rotation event for it is already active.

        :param segmind_context: The shared SegmindContext containing the information required to
            track events.
        :param tracked_object: The object to check.
        :param poses: The pose window of ``tracked_object``, oldest first.
        :return: A RotationEvent, a StopRotationEvent or None.
        """
        latest_rotation_event = segmind_context.latest_rotation_events.get(
            tracked_object
        )
        if not self._is_rotating(poses):
            if latest_rotation_event is None:
                return None
            segmind_context.latest_rotation_events.pop(tracked_object)
            return StopRotationEvent(
                tracked_object=tracked_object,
                world_T_start_pose=latest_rotation_event.world_T_start_pose,
                world_T_current_pose=poses[-1],
            )

        if latest_rotation_event is not None:
            return None

        new_event = RotationEvent(
            tracked_object=tracked_object,
            world_T_start_pose=poses[0],
            world_T_current_pose=poses[-1],
        )
        segmind_context.latest_rotation_events[tracked_object] = new_event
        return new_event
