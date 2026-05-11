from dataclasses import field, dataclass
from typing import List

import numpy as np

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import DetectionEvent, HoldingEvent, LossOfHoldingEvent, LiftingEvent, OpeningEvent
from segmind.detectors.base import SegmindContext, AbstractDetector
from semantic_digital_twin.reasoning.predicates import contact
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(repr=False, eq=False)
class HoldingDetector(AbstractDetector):
    """
    Detector that identifies when an object is being held by at least one
    gripper from a provided list of gripper groups.

    Each entry in gripper_groups is a list of gripper Body names.
    A HoldingEvent is fired when the tracked object is in contact with
    at least one gripper from any group.
    """

    gripper_groups: List[List[str]] = field(kw_only=True, default_factory=list)
    """
    List of gripper groups. Each group is a list of body names.
    A hold is detected if at least one gripper from any group is in contact.
    """

    def update_context_and_events(
            self,
            context: MotionStatechartContext,
            segmind_context: SegmindContext,
            tracked_objects: List[Body],
    ) -> List[DetectionEvent]:

        all_gripper_names = {name for group in self.gripper_groups for name in group}

        events = []
        for obj in tracked_objects:
            current_contacts = segmind_context.latest_contact_bodies.get(obj, set())
            active_grippers = [b for b in current_contacts if b.name.name in all_gripper_names]

            if not active_grippers:
                continue

            if obj in segmind_context.latest_holding:
                continue

            segmind_context.latest_holding[obj] = set(active_grippers)
            events.append(
                HoldingEvent(
                    tracked_object=obj,
                    with_object=active_grippers[0],
                    grippers=active_grippers,
                )
            )

        return events


@dataclass(repr=False, eq=False)
class LossOfHoldingDetector(AbstractDetector):
    """
    Detector that identifies when an object that was being held is no longer
    in contact with any gripper from the provided gripper groups.
    """
    gripper_groups: List[List[str]] = field(kw_only=True, default_factory=list)


    def update_context_and_events(
            self,
            context: MotionStatechartContext,
            segmind_context: SegmindContext,
            tracked_objects: List[Body],
    ) -> List[DetectionEvent]:

        all_gripper_names = {name for group in self.gripper_groups for name in group}

        events = []
        for obj, previously_holding in list(segmind_context.latest_holding.items()):
            current_contacts = segmind_context.latest_contact_bodies.get(obj, set())
            active_grippers = [b for b in current_contacts if b.name.name in all_gripper_names]

            if active_grippers:
                continue

            last_grippers = list(previously_holding)
            segmind_context.latest_holding.pop(obj)
            segmind_context.lifting_baselines.pop(obj, None)
            segmind_context.latest_lifting.discard(obj)
            events.append(
                LossOfHoldingEvent(
                    tracked_object=obj,
                    with_object=last_grippers[0] if last_grippers else None,
                    grippers=last_grippers,
                )
            )

        return events


@dataclass(repr=False, eq=False)
class LiftingDetector(AbstractDetector):
    """
    Detects when a held object is lifted upward in the z-direction.

    Requires HoldingDetector to run first each tick so that
    latest_holding is up to date. Records the z-position at the moment
    holding starts, then fires a LiftingEvent once the object has risen
    above z_threshold.
    """

    z_threshold: float = field(kw_only=True, default=0.05)
    """
    Minimum upward z-displacement (in meters) from the holding pose
    to be considered a lift.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:

        events = []

        for obj in tracked_objects:
            # Only care about objects currently being held
            if obj not in segmind_context.latest_holding:
                # If we were tracking a z-baseline but holding stopped, clean up
                segmind_context.lifting_baselines.pop(obj, None)
                continue

            current_z = obj.global_pose.z


            # Record z-baseline at the moment holding started
            if obj not in segmind_context.lifting_baselines:
                segmind_context.lifting_baselines[obj] = (current_z, obj.global_pose)
                continue

            baseline_z, baseline_pose = segmind_context.lifting_baselines[obj]

            # Already fired a lifting event for this hold — skip
            if obj in segmind_context.latest_lifting:
                continue

            if current_z - baseline_z >= self.z_threshold:
                grippers = list(segmind_context.latest_holding[obj])
                event = LiftingEvent(
                    tracked_object=obj,
                    with_object=grippers[0] if grippers else None,
                    grippers=grippers,
                    start_pose=baseline_pose,
                    current_pose=obj.global_pose,
                )
                segmind_context.latest_lifting.add(obj)
                events.append(event)

        return events


@dataclass(repr=False, eq=False)
class OpeningDetector(AbstractDetector):
    """
    Detects when a handle is being opened by a gripper.

    Requires HoldingDetector to run first. Once holding between
    a gripper and the handle is detected, records translations of
    both over observation_window frames. Fires an OpeningEvent if
    their displacement vectors are similar in both direction and magnitude.
    """

    handle_name: str = field(kw_only=True)
    """
    Name of the handle body to monitor.
    """

    gripper_groups: List[List[str]] = field(kw_only=True, default_factory=list)
    """
    List of gripper groups. A hold is required between any gripper
    in these groups and the handle before tracking begins.
    """

    observation_window: int = field(kw_only=True, default=20)
    """
    Number of frames to observe after holding starts before evaluating.
    """

    distance_threshold: float = field(kw_only=True, default=0.02)
    """
    Minimum displacement (meters) both objects must travel to count as movement.
    """

    angle_threshold: float = field(kw_only=True, default=0.3)
    """
    Maximum angle (radians) between displacement vectors to count as correlated.
    ~0.3 rad ≈ 17 degrees.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:

        all_gripper_names = {name for group in self.gripper_groups for name in group}

        # Find the handle body
        handle = context.world.get_body_by_name(self.handle_name)
        if handle is None:
            return []

        # Already fired for this handle — nothing to do
        if handle in segmind_context.latest_opening:
            return []

        # Check if any gripper is currently holding the handle
        holding_grippers = [
            b for b in segmind_context.latest_contact_bodies.get(handle, set())
            if b.name.name in all_gripper_names
        ]
        if not holding_grippers:
            # No hold — reset tracker if we were tracking
            segmind_context.opening_tracker.pop(handle, None)
            return []

        gripper = holding_grippers[0]

        # Initialise tracker on first frame of holding
        if handle not in segmind_context.opening_tracker:
            segmind_context.opening_tracker[handle] = {
                "gripper": gripper,
                "handle_poses": [handle.global_pose],
                "gripper_poses": [gripper.global_pose],
                "grippers": holding_grippers,
            }
            return []

        tracker = segmind_context.opening_tracker[handle]
        tracker["handle_poses"].append(handle.global_pose)
        tracker["gripper_poses"].append(gripper.global_pose)

        # Not enough frames yet
        if len(tracker["handle_poses"]) < self.observation_window:
            return []

        # Evaluate over the observation window
        events = []
        if self._is_correlated_translation(tracker):
            event = OpeningEvent(
                tracked_object=handle,
                with_object=gripper,
                grippers=tracker["grippers"],
            )
            segmind_context.latest_opening.add(handle)
            segmind_context.opening_tracker.pop(handle)
            events.append(event)
        else:
            # Slide the window forward — drop oldest frame
            tracker["handle_poses"].pop(0)
            tracker["gripper_poses"].pop(0)

        return events

    def _is_correlated_translation(self, tracker: dict) -> bool:
        handle_poses = tracker["handle_poses"]
        gripper_poses = tracker["gripper_poses"]

        handle_disp = self._displacement(handle_poses)
        gripper_disp = self._displacement(gripper_poses)

        # Both must have moved enough
        if np.linalg.norm(handle_disp) < self.distance_threshold:
            return False
        if np.linalg.norm(gripper_disp) < self.distance_threshold:
            return False

        # Displacement vectors must point in similar directions
        angle = self._angle_between(handle_disp, gripper_disp)
        return angle < self.angle_threshold

    def _displacement(self, poses) -> np.ndarray:
        """Displacement vector from first to last pose."""
        start = self._pose_to_xyz(poses[0])
        end = self._pose_to_xyz(poses[-1])
        return end - start

    def _pose_to_xyz(self, pose) -> np.ndarray:
        """Extract xyz from your Pose type — adjust if needed."""
        p = pose.to_position()
        return np.array([float(p.x), float(p.y), float(p.z)])

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return float("inf")
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.arccos(cos_angle))