from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import DetectionEvent, HoldingEvent, LossOfHoldingEvent, LiftingEvent
from segmind.detectors.base import SegmindContext, AbstractDetector
from semantic_digital_twin.reasoning.predicates import contact
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(repr=False, eq=False)
class HoldingDetector(AbstractDetector):
    """
    Detector that identifies when a tracked object starts touching one of a robot's gripper
    bodies, e.g. its finger links.
    """

    gripper_body_names: List[str] = field(kw_only=True, default_factory=list)
    """
    Names of the bodies that count as gripper contact surfaces. An object is considered held
    while it is in contact with at least one of them.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects newly started holds and updates the stored holding state.

        Recomputes gripper contact directly from the world rather than reading
        ``latest_contact_bodies``, so this detector produces correct results regardless of
        where it sits relative to ContactDetector in the tick order.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param tracked_objects: List of bodies to check for newly started holds.
        :return: List of HoldingEvent instances generated during this update.
        """
        gripper_bodies = [
            body for body in context.world.bodies_with_collision
            if body.name.name in self.gripper_body_names
        ]

        events = []
        for obj in tracked_objects:
            if obj in segmind_context.latest_holding:
                continue

            grippers = [gripper for gripper in gripper_bodies if contact(obj, gripper)]
            if not grippers:
                continue

            segmind_context.latest_holding[obj] = set(grippers)
            events.append(HoldingEvent(tracked_object=obj, with_object=grippers[0], grippers=grippers))

        return events


@dataclass(repr=False, eq=False)
class LossOfHoldingDetector(AbstractDetector):
    """
    Detector that identifies when a held object is no longer in contact with any gripper body.
    """

    gripper_body_names: List[str] = field(kw_only=True, default_factory=list)
    """
    Names of the bodies that count as gripper contact surfaces.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects when a held object loses contact with every gripper body and clears its
        holding and lifting state.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param tracked_objects: Unused; every currently held body is re-checked directly.
        :return: List of LossOfHoldingEvent instances generated during this update.
        """
        gripper_bodies = [
            body for body in context.world.bodies_with_collision
            if body.name.name in self.gripper_body_names
        ]

        events = []
        for obj, held_by in list(segmind_context.latest_holding.items()):
            if any(contact(obj, gripper) for gripper in gripper_bodies):
                continue

            segmind_context.latest_holding.pop(obj)
            segmind_context.lifting_baselines.pop(obj, None)
            segmind_context.latest_lifting.discard(obj)

            grippers = list(held_by)
            events.append(
                LossOfHoldingEvent(
                    tracked_object=obj,
                    with_object=grippers[0] if grippers else None,
                    grippers=grippers,
                )
            )

        return events


@dataclass(repr=False, eq=False)
class LiftingDetector(AbstractDetector):
    """
    Detects when a held object rises above a lift threshold relative to the pose it had when
    holding started.
    """

    lift_threshold: float = field(kw_only=True, default=0.05)
    """
    Minimum upward z-displacement, in meters, from the pose at hold-start to count as a lift.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Tracks the pose each held object had when holding started and fires a LiftingEvent
        once it has risen past lift_threshold.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param tracked_objects: List of bodies to check for lifting.
        :return: List of LiftingEvent instances generated during this update.
        """
        events = []
        for obj in tracked_objects:
            if obj not in segmind_context.latest_holding:
                segmind_context.lifting_baselines.pop(obj, None)
                continue

            if obj not in segmind_context.lifting_baselines:
                segmind_context.lifting_baselines[obj] = obj.global_pose
                continue

            if obj in segmind_context.latest_lifting:
                continue

            baseline_pose = segmind_context.lifting_baselines[obj]
            if obj.global_pose.z - baseline_pose.z < self.lift_threshold:
                continue

            grippers = list(segmind_context.latest_holding[obj])
            segmind_context.latest_lifting.add(obj)
            events.append(
                LiftingEvent(
                    tracked_object=obj,
                    with_object=grippers[0] if grippers else None,
                    grippers=grippers,
                    start_pose=baseline_pose,
                    current_pose=obj.global_pose,
                )
            )

        return events
