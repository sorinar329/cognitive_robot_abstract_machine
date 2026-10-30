"""
SegMind judging, while a plan runs, whether each action that acts on an object caused the
events it is expected to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from coraplex.robot_plans.actions.base import Action
from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.statechart import Statechart
from segmind.action_effects import EffectsOfActions, ExpectedEvent
from segmind.datastructures.events import EventWithTrackedObjects
from segmind.event_segmentation import Segmind

logger = logging.getLogger(__name__)


# %% verdicts


@dataclass
class ActionVerdict:
    """
    Whether one action caused the events it is expected to.
    """

    action: Action
    """
    The action judged.
    """

    expected: list[ExpectedEvent]
    """
    The events the action is expected to cause.
    """

    evidence: list[EventWithTrackedObjects]
    """
    The detected events that show an expected event happened.
    """

    holds: bool
    """
    Whether every expected event happened.
    """

    action_failed: bool
    """
    Whether the action itself ended without succeeding, which does not keep it from
    being judged.
    """

    started: datetime
    """
    When the action started.
    """

    ended: datetime
    """
    When the action ended, or the plan did, for an action that never ended.
    """

    def __str__(self) -> str:
        return (
            f"{type(self.action).__name__} "
            f"{'holds' if self.holds else 'does not hold'}"
            f"{' (the action failed)' if self.action_failed else ''}"
        )


# %% the judge


@dataclass(eq=False)
class ActionJudge(TickingExecutorExtension):
    """
    Judges every action of a plan that acts on an object, as the plan runs, from the
    events :attr:`segmind` detects; :attr:`verdicts` holds one verdict per run of such
    an action. An action held by another such action is part of that one's verdict.

    An action is judged as soon as it ended, once SegMind saw the world at rest after
    it. The tick that ended the action waits until then, so what the next action does
    is never taken for what this one did. An action that never ended is judged when the
    plan stops. A verdict never stops the plan.

    .. note:: :attr:`segmind` has to be watching while the plan runs.
    """

    segmind: Segmind = field(kw_only=True)
    """
    The SegMind watching the objects the plan acts on.
    """

    effects: EffectsOfActions = field(kw_only=True, default_factory=EffectsOfActions)
    """
    The events every action is expected to cause.
    """

    verdicts: list[ActionVerdict] = field(init=False, default_factory=list)
    """
    The verdicts on the actions of the plan, in the order the actions started.
    """

    running_since: dict[Action, datetime] = field(init=False, default_factory=dict)
    """
    The actions to judge that are running, each with when it started.
    """

    def after_tick(self, executor: StatechartExecutor) -> None:
        """
        Note when each action to judge starts, and judge each that ended this tick.
        """
        now = datetime.now()
        for action in self._actions_to_judge(executor.statechart):
            state = action.life_cycle_state
            if state == LifeCycleValues.NOT_STARTED:
                continue
            if action not in self.running_since:
                if state.is_terminal:
                    continue
                self.running_since[action] = now
            if state.is_terminal:
                self._judge(action, ended=now)

    def after_run(self, executor: StatechartExecutor) -> None:
        """
        Judge every action that is still running, since the plan stopped, and order
        :attr:`verdicts` by when the actions started.
        """
        now = datetime.now()
        for action in list(self.running_since):
            self._judge(action, ended=now)
        self.verdicts.sort(key=lambda verdict: verdict.started)

    def _actions_to_judge(self, statechart: Statechart) -> list[Action]:
        """
        :return: The actions of ``statechart`` that act on an object, and that no other
            such action holds.
        """
        acting_on_objects = tuple(
            effects.action_type for effects in self.effects.effects
        )
        return [
            action
            for action in statechart.get_nodes_by_type(Action)
            if isinstance(action, acting_on_objects)
            and not any(
                isinstance(holder, acting_on_objects)
                for holder in self._holders_of(action)
            )
        ]

    @staticmethod
    def _holders_of(action: Action) -> list[Action]:
        """
        :return: The nodes ``action`` lies below, from its parent up.
        """
        holders = []
        holder = action.parent_node
        while holder is not None:
            holders.append(holder)
            holder = holder.parent_node
        return holders

    def _judge(self, action: Action, ended: datetime) -> None:
        """
        Judge ``action``, which ended at ``ended``, from what SegMind detected from its
        start until it saw the world at rest after it; leave it unjudged if SegMind does
        not watch what it acts on.
        """
        started = self.running_since.pop(action)
        expected = self.effects.expected_events(action)
        unwatched = {
            expected_event.tracked_object for expected_event in expected
        } - set(self.segmind.watched_bodies)
        if unwatched:
            logger.warning(
                "%s is not judged: SegMind does not watch %s.",
                type(action).__name__,
                ", ".join(str(body.name) for body in unwatched),
            )
            return
        seen_at_rest = self.segmind.seen_at_rest_after(ended)
        events = sorted(
            (
                event
                for event in self.segmind.event_logger.get_events()
                if isinstance(event, EventWithTrackedObjects)
                and started <= event.timestamp <= seen_at_rest
            ),
            key=lambda event: event.timestamp,
        )
        evidence_of_each = [
            expected_event.evidence_among(events) for expected_event in expected
        ]
        verdict = ActionVerdict(
            action=action,
            expected=expected,
            evidence=[event for evidence in evidence_of_each for event in evidence],
            holds=all(evidence_of_each),
            action_failed=action.life_cycle_state != LifeCycleValues.SUCCEEDED,
            started=started,
            ended=ended,
        )
        self.verdicts.append(verdict)
        logger.info("Verdict: %s", verdict)
