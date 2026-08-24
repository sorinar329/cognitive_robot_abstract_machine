"""
Optional recording of a real-robot demo run: a ``ros2 bag record`` subprocess covering
Tracy's own joint-state/command topics, plus an action-marker topic that brackets each
pick/place action so a recording can later be sliced back into per-action segments and
replayed against MuJoCo to tune :mod:`~experiments.tracy_experiments.equipment`'s
``ServoGains``. See ``ROSBAG_RECORDING.md`` for the full rationale.

Off by default: a demo's own ``main()`` picks :class:`NullActionRecorder` unless asked
to record, so nothing here runs unless explicitly enabled.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from std_msgs.msg import String
from typing_extensions import Any, Dict, List, Optional, Tuple

from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import code
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.ros.ros2.publisher import create_publisher
from semantic_digital_twin.spatial_types.spatial_types import Pose

logger = logging.getLogger(__name__)

ACTION_MARKER_TOPIC = "/tracy_experiments/action_marker"
"""
Topic a run's start/end action markers are published on; recorded in the same bag as
:data:`RECORDED_ROBOT_TOPICS` so both share one recording clock, see
``ROSBAG_RECORDING.md``.
"""

RECORDED_ROBOT_TOPICS = (
    "/left_arm/joint_states",
    "/right_arm/joint_states",
    "/left_gripper/joint_states",
    "/right_gripper/joint_states",
    "/left_arm/forward_velocity_controller/commands",
    "/right_arm/forward_velocity_controller/commands",
)
"""
Tracy's own feedback (``joint_states``) and command (``forward_velocity_controller/
commands``) topics -- matches what ``TracyVelocityInterface`` itself subscribes to and
publishes on.
"""


class MarkerPhase(StrEnum):
    """Which end of an action a marker message brackets."""

    START = "start"
    END = "end"


class RecordedActionType(StrEnum):
    """Which kind of action a marker message belongs to."""

    PICK = "pick"
    PLACE = "place"


@dataclass
class ActionRecorder(ABC):
    """
    Brackets each pick/place action of a real-robot run with a marker message, and
    optionally records the run to a ``ros2 bag``; see ``ROSBAG_RECORDING.md``.
    """

    def __enter__(self) -> ActionRecorder:
        self.start()
        return self

    def __exit__(self, *exception_info: Any) -> None:
        self.stop()

    @abstractmethod
    def start(self) -> None:
        """Begin recording, if this recorder does any."""

    @abstractmethod
    def stop(self) -> None:
        """End recording, if this recorder does any."""

    @abstractmethod
    def mark_start(
        self,
        index: int,
        action: RecordedActionType,
        object_name: str,
        arm: Arms,
        target_pose: Optional[Pose] = None,
    ) -> None:
        """Publish that action ``index`` has begun."""

    @abstractmethod
    def mark_end(
        self, index: int, action: RecordedActionType, object_name: str, arm: Arms
    ) -> None:
        """Publish that action ``index`` has finished, recording its own outcome."""


@dataclass
class NullActionRecorder(ActionRecorder):
    """
    An :class:`ActionRecorder` that records nothing -- the default, opt-in-only
    behaviour a demo falls back to when recording was not asked for.
    """

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def mark_start(
        self,
        index: int,
        action: RecordedActionType,
        object_name: str,
        arm: Arms,
        target_pose: Optional[Pose] = None,
    ) -> None:
        pass

    def mark_end(
        self, index: int, action: RecordedActionType, object_name: str, arm: Arms
    ) -> None:
        pass


@dataclass
class RosbagActionRecorder(ActionRecorder):
    """
    Records a ``ros2 bag`` of :data:`RECORDED_ROBOT_TOPICS` plus
    :data:`ACTION_MARKER_TOPIC`, and asks the operator to confirm success at each
    action's own end marker -- there is no perception in these demos yet, so a real
    success/failure judgement can only come from a human watching the run (see
    ``ROSBAG_RECORDING.md``).
    """

    node: Any
    """The ROS2 node used to create the marker publisher."""

    output_directory: Path
    """Where ``ros2 bag record`` writes the recording; must not already exist."""

    startup_delay_seconds: float = 1.0
    """
    How long to wait after starting ``ros2 bag record`` before anything is published,
    so the recorder has subscribed to every topic first.
    """

    _bag_process: Optional[subprocess.Popen] = field(default=None, init=False)
    """The running ``ros2 bag record`` subprocess, set once :meth:`start` runs."""

    _publisher: Any = field(init=False)
    """Publisher for :data:`ACTION_MARKER_TOPIC`, created eagerly so it exists even if
    :meth:`start` is never called."""

    def __post_init__(self) -> None:
        self._publisher = create_publisher(
            ACTION_MARKER_TOPIC, String, self.node, queue_size=10
        )

    def start(self) -> None:
        self._bag_process = subprocess.Popen(
            [
                "ros2",
                "bag",
                "record",
                "-o",
                str(self.output_directory),
                *RECORDED_ROBOT_TOPICS,
                ACTION_MARKER_TOPIC,
            ],
            start_new_session=True,
        )
        time.sleep(self.startup_delay_seconds)

    def stop(self) -> None:
        if self._bag_process is None:
            return
        os.killpg(os.getpgid(self._bag_process.pid), signal.SIGTERM)
        self._bag_process.wait()

    def mark_start(
        self,
        index: int,
        action: RecordedActionType,
        object_name: str,
        arm: Arms,
        target_pose: Optional[Pose] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "phase": MarkerPhase.START,
            "index": index,
            "action": action,
            "object": object_name,
            "arm": arm.name.lower(),
        }
        if target_pose is not None:
            position = target_pose.to_position()
            payload["target_pose"] = {
                "x": float(position.x),
                "y": float(position.y),
                "z": float(position.z),
            }
        self._publish(payload)

    def mark_end(
        self, index: int, action: RecordedActionType, object_name: str, arm: Arms
    ) -> None:
        success = self._ask_operator_for_success(index, action, object_name)
        self._publish(
            {
                "phase": MarkerPhase.END,
                "index": index,
                "action": action,
                "object": object_name,
                "arm": arm.name.lower(),
                "success": success,
            }
        )

    @staticmethod
    def _ask_operator_for_success(
        index: int, action: RecordedActionType, object_name: str
    ) -> bool:
        """
        Ask whoever is watching the run whether ``action`` on ``object_name`` actually
        succeeded -- there is no automatic ground truth here, see
        ``ROSBAG_RECORDING.md``.

        :param index: The marker index of the action being confirmed.
        :param action: Which kind of action is being confirmed.
        :param object_name: Name of the object the action was performed on.
        :return: Whether the operator confirmed success.
        """
        answer = input(f"[{index}] did '{action}' on '{object_name}' succeed? [Y/n] ")
        return answer.strip().lower() != "n"

    def _publish(self, payload: Dict[str, Any]) -> None:
        message = String()
        message.data = json.dumps(payload)
        self._publisher.publish(message)


@dataclass
class ActionMarkerNode(ActionDescription):
    """
    Publishes one of ``recorder``'s own marker messages when this plan node is
    reached, without touching the robot -- bracket a pick/place action with one of
    these before and after it in an action sequence so a recording (if any) can later
    be sliced back into per-action segments.
    """

    recorder: ActionRecorder
    """The recorder to publish through; a :class:`NullActionRecorder` makes this a
    no-op."""

    index: int
    """Marker index shared by this node and its start/end counterpart."""

    action: RecordedActionType
    """Which kind of action this node brackets."""

    object_name: str
    """Name of the object the bracketed action acts on."""

    arm: Arms
    """Which arm performs the bracketed action."""

    phase: MarkerPhase
    """Whether this node marks the start or the end of the bracketed action."""

    target_pose: Optional[Pose] = None
    """The bracketed action's own target pose, included on a start marker only."""

    @property
    def _action_plan(self) -> PlanNode:
        return code(self._run)

    def _run(self) -> None:
        if self.phase == MarkerPhase.START:
            self.recorder.mark_start(
                self.index, self.action, self.object_name, self.arm, self.target_pose
            )
        else:
            self.recorder.mark_end(self.index, self.action, self.object_name, self.arm)


def _recorded_action_type(
    action: ActionDescription,
) -> Tuple[Optional[RecordedActionType], Optional[Pose]]:
    """
    Which :class:`RecordedActionType` ``action`` is, and its own target pose if it has
    one, or ``(None, None)`` if it is not a kind of action markers bracket.
    """
    if isinstance(action, PickUpAction):
        return RecordedActionType.PICK, None
    if isinstance(action, PlaceAction):
        return RecordedActionType.PLACE, action.target_location
    return None, None


def bracket_actions_with_markers(
    recorder: ActionRecorder, actions: List[ActionDescription]
) -> List[ActionDescription]:
    """
    Bracket every ``PickUpAction``/``PlaceAction`` in ``actions`` with a start/end
    :class:`ActionMarkerNode` pair, so a recording (if any) can later be sliced back
    into per-action segments; every other action (e.g. a park action) is passed through
    unchanged.

    :param recorder: Recorder the marker nodes publish through; a
        :class:`NullActionRecorder` makes them no-ops.
    :param actions: The action sequence to bracket, as built for a real-robot demo.
    :return: The same sequence with a start/end marker around every pick/place action.
    """
    marked_actions: List[ActionDescription] = []
    index = 0
    for action in actions:
        recorded_type, target_pose = _recorded_action_type(action)
        if recorded_type is None:
            marked_actions.append(action)
            continue

        index += 1
        object_name = action.object_designator.name.name
        marked_actions.append(
            ActionMarkerNode(
                recorder=recorder,
                index=index,
                action=recorded_type,
                object_name=object_name,
                arm=action.arm,
                phase=MarkerPhase.START,
                target_pose=target_pose,
            )
        )
        marked_actions.append(action)
        marked_actions.append(
            ActionMarkerNode(
                recorder=recorder,
                index=index,
                action=recorded_type,
                object_name=object_name,
                arm=action.arm,
                phase=MarkerPhase.END,
            )
        )
    return marked_actions
