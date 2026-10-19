"""
SegMind judging, action by action, whether a plan achieved what its actions are for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from typing_extensions import Self

from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import ActionNode, PlanNode
from giskardpy.motion_statechart.data_types import LifeCycleValues
from krrood.entity_query_language.factories import evaluate_condition
from segmind.datastructures.events import DetectionEvent, EventWithTrackedObjects
from segmind.event_segmentation import Segmind
from segmind.post_conditions.expectations import (
    ActionExpectation,
    GraspingExpectation,
    PickUpExpectation,
    PlaceExpectation,
)

logger = logging.getLogger(__name__)


# %% verdicts


class JudgedBy(StrEnum):
    """
    What a verdict on an action's post-condition was read from.
    """

    SEGMIND = "segmind"
    """
    The events SegMind detected while the action ran.
    """

    WORLD = "world"
    """
    The action's own post-condition, evaluated on the world once the action ended,
    since SegMind expects nothing of an action of its type.
    """

    NOT_WATCHED = "not watched"
    """
    The action's own post-condition, evaluated on the world once the action ended,
    since SegMind does not watch a body it would have to watch to judge the action.
    """


@dataclass
class PostConditionVerdict:
    """
    Whether one action of a plan achieved what it is for.
    """

    action_node: ActionNode
    """
    The node of the action judged.
    """

    holds: bool
    """
    Whether the action's post-condition holds.
    """

    judged_by: JudgedBy
    """
    What the verdict was read from.
    """

    evidence: list[DetectionEvent]
    """
    The events that decided the verdict; none when it was not judged by SegMind.
    """

    action_failed: bool
    """
    Whether the action itself failed, which does not keep it from being judged.
    """

    started: datetime
    """
    When the first step of the action started.
    """

    def __str__(self) -> str:
        return (
            f"{type(self.action_node.action).__name__} "
            f"{'holds' if self.holds else 'does not hold'} "
            f"(judged by {self.judged_by}"
            f"{', the action failed' if self.action_failed else ''})"
        )


# %% the judge


@dataclass(eq=False)
class PostConditionJudge(PlanCallback):
    """
    SegMind judging every action of a plan as the plan runs: entering it has the plan
    report to it, and :attr:`verdicts` holds one verdict per action that started, once
    the plan ended.

    An action is judged as soon as it ended: by its expectation, from what SegMind
    detected while it ran and until it saw the world at rest after it, or by its own
    post-condition, evaluated on the world, when there is no expectation of it. An
    action that never ended is judged when the plan ended. A verdict is only recorded;
    it never stops the plan.

    .. note:: The plan waits after each action until SegMind saw the world at rest, so
        what the next action does is never taken for what this one did.
    """

    segmind: Segmind = field(kw_only=True)
    """
    The SegMind watching the plan, which has to be watching while the plan runs.
    """

    expectations: list[ActionExpectation] = field(
        kw_only=True,
        default_factory=lambda: [
            PickUpExpectation(),
            GraspingExpectation(),
            PlaceExpectation(),
        ],
    )
    """
    What SegMind expects of the actions it judges, one per type of action.
    """

    verdicts: list[PostConditionVerdict] = field(init=False, default_factory=list)
    """
    The verdicts on the actions of the plan, in the order the actions started.
    """

    def __enter__(self) -> Self:
        self.plan.node_callbacks.append(self)
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.plan.node_callbacks.remove(self)

    def on_start(self, node: PlanNode) -> None:
        """
        Wait, as the plan starts, until SegMind saw the world it starts in.
        """
        if node is self.plan.root:
            self.segmind.seen_at_rest_after(datetime.now())

    def on_end(self, node: PlanNode) -> None:
        """
        Judge every action that ended with ``node``; once the plan ended, every action
        that started and is not judged yet, and order :attr:`verdicts` by when the
        actions started.
        """
        if node is self.plan.root:
            self._judge(
                [
                    action_node
                    for action_node in self.plan.root.descendants
                    if isinstance(action_node, ActionNode)
                    and not self._is_judged(action_node)
                    and self._steps_that_ran(action_node)
                ]
            )
            self.verdicts.sort(key=lambda verdict: verdict.started)
            return
        self._judge(
            [
                action_node
                for action_node in self._action_nodes_around(node)
                if not self._is_judged(action_node) and self._has_ended(action_node)
            ]
        )

    def _judge(self, action_nodes: list[ActionNode]) -> None:
        """
        Judge the actions of ``action_nodes``, which ended now: those judged on the
        world first, before SegMind is waited for.
        """
        judged_by_segmind = []
        for action_node in action_nodes:
            expectation = self._expectation_of(action_node)
            if expectation is None:
                self._judge_on_the_world(action_node, JudgedBy.WORLD)
            elif not set(expectation.watched_bodies(action_node.action)) <= set(
                self.segmind.watched_bodies
            ):
                logger.warning(
                    "SegMind does not watch what %s handles, so it is judged on the "
                    "world.",
                    type(action_node.action).__name__,
                )
                self._judge_on_the_world(action_node, JudgedBy.NOT_WATCHED)
            else:
                judged_by_segmind.append((action_node, expectation))
        if not judged_by_segmind:
            return
        seen_at_rest = self.segmind.seen_at_rest_after(datetime.now())
        for action_node, expectation in judged_by_segmind:
            self._judge_by_segmind(action_node, expectation, seen_at_rest)

    def _judge_by_segmind(
        self,
        action_node: ActionNode,
        expectation: ActionExpectation,
        seen_at_rest: datetime,
    ) -> None:
        """
        Judge ``action_node`` from the events of the bodies ``expectation`` watches,
        detected from its first step on until ``seen_at_rest``.
        """
        started = min(step.start_time for step in self._steps_that_ran(action_node))
        watched = set(expectation.watched_bodies(action_node.action))
        events: list[EventWithTrackedObjects] = [
            event
            for event in self.segmind.event_logger.get_events()
            if isinstance(event, EventWithTrackedObjects)
            and event.tracked_object in watched
            and started <= event.timestamp <= seen_at_rest
        ]
        evidence = expectation.evidence(action_node.action, events)
        self._record(
            PostConditionVerdict(
                action_node=action_node,
                holds=expectation.holds(action_node.action, evidence),
                judged_by=JudgedBy.SEGMIND,
                evidence=evidence,
                action_failed=self._has_failed(action_node),
                started=started,
            )
        )

    def _judge_on_the_world(self, action_node: ActionNode, judged_by: JudgedBy) -> None:
        """
        Judge ``action_node`` by evaluating its own post-condition on the world as it
        is now.
        """
        [post_condition] = [
            child
            for child in action_node.children
            if isinstance(child, ConditionNode) and not child.pre_condition
        ]
        self._record(
            PostConditionVerdict(
                action_node=action_node,
                holds=evaluate_condition(post_condition.condition),
                judged_by=judged_by,
                evidence=[],
                action_failed=self._has_failed(action_node),
                started=min(
                    step.start_time for step in self._steps_that_ran(action_node)
                ),
            )
        )

    def _record(self, verdict: PostConditionVerdict) -> None:
        """
        Keep ``verdict`` and report it.
        """
        self.verdicts.append(verdict)
        logger.info("Post-condition: %s", verdict)

    def _expectation_of(self, action_node: ActionNode) -> ActionExpectation | None:
        """
        :return: The expectation of the most specific type ``action_node``'s action is
            of; none when there is none.
        """
        by_type = {
            expectation.expected_action_type: expectation
            for expectation in self.expectations
        }
        for action_type in type(action_node.action).__mro__:
            if action_type in by_type:
                return by_type[action_type]
        return None

    def _is_judged(self, action_node: ActionNode) -> bool:
        """
        :return: Whether there is a verdict on ``action_node`` already.
        """
        return any(verdict.action_node is action_node for verdict in self.verdicts)

    @staticmethod
    def _action_nodes_around(node: PlanNode) -> list[ActionNode]:
        """
        :return: ``node`` and the nodes it lies below that are actions.
        """
        return [
            candidate
            for candidate in [node, *node.path]
            if isinstance(candidate, ActionNode)
        ]

    @staticmethod
    def _steps(action_node: ActionNode) -> list[PlanNode]:
        """
        :return: The nodes below ``action_node`` that do something themselves: those
            without children, other than conditions.
        """
        return [
            node
            for node in action_node.descendants
            if not node.children and not isinstance(node, ConditionNode)
        ]

    def _steps_that_ran(self, action_node: ActionNode) -> list[PlanNode]:
        """
        :return: The steps of ``action_node`` that started.
        """
        return [
            step
            for step in self._steps(action_node)
            if step.status != LifeCycleValues.NOT_STARTED and step.end_time is not None
        ]

    def _has_ended(self, action_node: ActionNode) -> bool:
        """
        :return: Whether every step of ``action_node`` ran and none is running.
        """
        steps = self._steps(action_node)
        return bool(steps) and len(self._steps_that_ran(action_node)) == len(steps)

    def _has_failed(self, action_node: ActionNode) -> bool:
        """
        :return: Whether a step of ``action_node`` failed or was interrupted.
        """
        return any(
            step.status in (LifeCycleValues.FAILED, LifeCycleValues.INTERRUPTED)
            for step in self._steps(action_node)
        )
