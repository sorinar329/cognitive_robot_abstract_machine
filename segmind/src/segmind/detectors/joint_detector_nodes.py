"""
Detectors of what happens to the parts of a scene that move on joints: a joint starting
to move, and a part being opened or closed by its handle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from semantic_digital_twin.semantic_annotations.mixins import HasMechanicalJoint
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import TYPE_CHECKING, Dict, Generic, List, Optional, Self, Set

from segmind.datastructures.events import (
    ClosingEvent,
    DetectionEvent,
    GraspEvent,
    JointDirection,
    JointMotionEvent,
    OpeningEvent,
)
from segmind.detectors.base import (
    AbstractDetector,
    RuleDetector,
    SegmindContext,
    TDetectedEvent,
    TFirstEvidence,
    TSecondEvidence,
)
from segmind.detectors.rules import articulation_rule

if TYPE_CHECKING:
    from segmind.scene_parts import SceneParts

JOINT_MOTION_WINDOW_SIZE = 4
"""
How many readings of a joint's position a joint motion detector compares.
"""

JOINT_POSITION_THRESHOLD = 0.005
"""
The least change of a joint's position across the window that counts as motion, in the
joint's own unit.
"""


class WatchesArticulatedParts:
    """
    A kind of detector that reads the scene's articulated parts (see
    :attr:`~segmind.scene_parts.SceneParts.articulated_parts`): it watches every one of
    them, whatever object it is given to track, and a scene without any gets no detector
    of that kind.
    """

    @classmethod
    def instances_for(
        cls, tracked_object: Optional[Body], scene: SceneParts
    ) -> List[Self]:
        if not scene.articulated_parts:
            return []
        return super().instances_for(tracked_object, scene)


@dataclass(eq=False, repr=False)
class JointMotionDetector(WatchesArticulatedParts, AbstractDetector[JointMotionEvent]):
    """
    Detects the joint of an articulated part starting to move, and which way.
    """

    window_size: int = JOINT_MOTION_WINDOW_SIZE
    """
    How many readings of a joint's position are compared.
    """

    position_threshold: float = JOINT_POSITION_THRESHOLD
    """
    The least change of position across the window that counts as motion.
    """

    _positions: Dict[Body, List[float]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The latest readings of each part's joint position, oldest first, by the part's root.
    """

    _moving: Set[Body] = field(default_factory=set, init=False, repr=False)
    """
    The roots of the parts whose joint motion has been reported and not yet ended.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Read every articulated part's joint position and report each joint that started
        moving.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext holding the articulated parts.
        :param tracked_objects: Not read; every articulated part is watched.
        :return: A JointMotionEvent per joint that started moving this tick.
        """
        events = []
        for part in segmind_context.articulated_parts:
            event = self._motion_starting(part)
            if event is not None:
                events.append(event)
        return events

    def _motion_starting(self, part: HasMechanicalJoint) -> Optional[JointMotionEvent]:
        """
        Record the part's joint position, and report its joint starting to move.

        :param part: The articulated part to read.
        :return: The motion, if the joint moved across the window and was not already
            reported moving; None otherwise.
        """
        positions = self._positions.setdefault(part.root, [])
        positions.append(float(part.mechanical_joint.position))
        if len(positions) > self.window_size:
            positions.pop(0)
        if len(positions) < self.window_size:
            return None

        change = positions[-1] - positions[0]
        if abs(change) <= self.position_threshold:
            self._moving.discard(part.root)
            return None
        if part.root in self._moving:
            return None

        self._moving.add(part.root)
        return JointMotionEvent(
            tracked_object=part.root,
            start_position=positions[0],
            current_position=positions[-1],
            direction=(
                JointDirection.TOWARDS_UPPER_LIMIT
                if change > 0
                else JointDirection.TOWARDS_LOWER_LIMIT
            ),
        )


@dataclass(eq=False, repr=False)
class ArticulationDetector(
    WatchesArticulatedParts,
    RuleDetector[TDetectedEvent, TFirstEvidence, TSecondEvidence],
    Generic[TDetectedEvent, TFirstEvidence, TSecondEvidence],
    ABC,
):
    """
    A detector concluding that a part was moved by its handle, by
    :func:`~segmind.detectors.rules.articulation_rule`: its handle was grasped and its
    joint moved in :meth:`direction` soon after.
    """

    @classmethod
    @abstractmethod
    def direction(cls) -> JointDirection:
        """
        :return: The way the joint moves in the interaction this detector concludes.
        """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Concludes the interaction for every articulated part whose handle was grasped
        and whose joint moved in :meth:`direction` within
        :attr:`~segmind.detectors.base.RuleDetector.shift_threshold`, and that was not
        already concluded for the same part and gripper.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext holding the logged events and the articulated parts.
        :param tracked_objects: Not read; every articulated part is watched.
        :return: The interactions concluded.
        """
        return articulation_rule(
            event_type=self.detected_event_type(),
            direction=self.direction(),
            grasp_event_type=self.first_evidence_type(),
            joint_motion_event_type=self.second_evidence_type(),
            logged_events=segmind_context.logger.get_events(),
            articulated_parts=segmind_context.articulated_parts,
            shift_threshold=self.shift_threshold,
        ).tolist()


@dataclass(eq=False, repr=False)
class OpeningDetector(ArticulationDetector[OpeningEvent, GraspEvent, JointMotionEvent]):
    """
    Detects that a part was opened: its handle was grasped and its joint moved towards
    its upper limit.
    """

    @classmethod
    def direction(cls) -> JointDirection:
        return JointDirection.TOWARDS_UPPER_LIMIT


@dataclass(eq=False, repr=False)
class ClosingDetector(ArticulationDetector[ClosingEvent, GraspEvent, JointMotionEvent]):
    """
    Detects that a part was closed: its handle was grasped and its joint moved towards
    its lower limit.
    """

    @classmethod
    def direction(cls) -> JointDirection:
        return JointDirection.TOWARDS_LOWER_LIMIT
