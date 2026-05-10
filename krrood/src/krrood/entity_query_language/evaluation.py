"""
Evaluation context and observer system for the Entity Query Language.

This module provides an aspect-oriented mechanism for hooking into the
evaluation pipeline without polluting the core evaluation methods.
"""

from __future__ import annotations

from abc import ABC
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing_extensions import Any, Dict, List, Optional

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import (
        Bindings,
        OperationResult,
        SymbolicExpression,
    )

_evaluation_context_var: ContextVar[Optional[EvaluationContext]] = ContextVar(
    "_evaluation_context", default=None
)


def get_evaluation_context() -> Optional[EvaluationContext]:
    """Return the current evaluation context, or None if outside evaluation."""
    return _evaluation_context_var.get()


def set_evaluation_context(ctx: Optional[EvaluationContext]) -> None:
    """Set or clear the current evaluation context."""
    _evaluation_context_var.set(ctx)


@dataclass
class EvaluationContext:
    """Carries observer state through the evaluation pipeline."""

    observers: List[EvaluationObserver] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def on_evaluate_enter(self, *, expression, sources):
        for obs in self.observers:
            obs.on_evaluate_enter(expression, sources)

    def on_evaluate_exit(self, *, expression):
        for obs in self.observers:
            obs.on_evaluate_exit(expression)

    def on_result_yielded(self, *, expression, result):
        for obs in self.observers:
            obs.on_result_yielded(expression, result)

    def on_conclusions_processed(self, *, expression, result):
        for obs in self.observers:
            obs.on_conclusions_processed(expression, result)


class EvaluationObserver(ABC):
    """Observer for evaluation events in the EQL evaluation pipeline."""

    def on_evaluate_enter(
            self, expression: SymbolicExpression, sources: Bindings
    ) -> None:
        """Called when entering an expression's _evaluate_ method."""

    def on_evaluate_exit(self, expression: SymbolicExpression) -> None:
        """Called when exiting an expression's _evaluate_ method."""

    def on_result_yielded(
            self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """Called for each OperationResult yielded from _evaluate__."""

    def on_conclusions_processed(
            self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """Called after _evaluate_conclusions_and_update_bindings_ completes."""


def _is_condition_participant(expr) -> bool:
    """Return True if the expression participates in condition evaluation."""
    from krrood.entity_query_language.operators.comparator import Comparator
    from krrood.entity_query_language.predicate import Predicate
    from krrood.entity_query_language.operators.core_logical_operators import (
        LogicalOperator,
    )
    from krrood.entity_query_language.core.base_expressions import (
        TruthValueOperator,
    )

    _condition_types = (Comparator, Predicate, LogicalOperator)
    if isinstance(expr, _condition_types):
        return True
    if isinstance(expr._parent_, TruthValueOperator):
        return True
    return False


def _collect_satisfied_condition_ids(condition_root, bindings: Bindings) -> frozenset:
    """Collect the UUIDs of condition expressions in the condition tree that were satisfied."""
    from krrood.entity_query_language.operators.core_logical_operators import (
        LogicalOperator,
    )

    satisfied = set()
    for expr in condition_root._descendants_:
        if not _is_condition_participant(expr):
            continue
        if expr._id_ in bindings:
            if bindings[expr._id_]:
                satisfied.add(expr._id_)
        elif isinstance(expr, LogicalOperator):
            if any(d._id_ in bindings for d in expr._descendants_):
                if not expr._is_false_:
                    satisfied.add(expr._id_)

    if _is_condition_participant(condition_root):
        if condition_root._id_ in bindings:
            if bindings[condition_root._id_]:
                satisfied.add(condition_root._id_)
        elif (
                isinstance(condition_root, LogicalOperator)
                and not condition_root._is_false_
        ):
            satisfied.add(condition_root._id_)

    return frozenset(satisfied)


SATISFIED_IDS_KEY = "_satisfied_condition_ids"


class SatisfiedConditionTracker(EvaluationObserver):
    """Observer that tracks which condition expressions were satisfied during evaluation.

    Replaces the ad-hoc ``_carried_satisfied_ids_``, ``@captures_satisfied_conditions``,
    and inline propagation that was previously scattered across the evaluation pipeline.
    """

    def on_evaluate_enter(self, expression, sources):
        from krrood.entity_query_language.core.base_expressions import (
            OperationResult,
        )

        ctx = get_evaluation_context()
        if ctx is None:
            return
        satisfied = None
        if isinstance(sources, OperationResult):
            satisfied = sources.satisfied_condition_ids
        elif hasattr(sources, "satisfied_condition_ids"):
            satisfied = sources.satisfied_condition_ids
        if satisfied is not None:
            ctx.data[SATISFIED_IDS_KEY] = satisfied

    def on_result_yielded(self, expression, result):
        ctx = get_evaluation_context()
        if ctx is None:
            return
        satisfied = ctx.data.get(SATISFIED_IDS_KEY)
        if satisfied is not None and result.satisfied_condition_ids is None:
            result.satisfied_condition_ids = satisfied

    def on_conclusions_processed(self, expression, result):

        if expression._conditions_root_ is not expression:
            return
        if result.is_false:
            return
        if expression._conditions_root_ is expression._root_:
            return

        satisfied_ids = _collect_satisfied_condition_ids(expression, result.bindings)
        result.satisfied_condition_ids = satisfied_ids
        ctx = get_evaluation_context()
        if ctx is not None:
            ctx.data[SATISFIED_IDS_KEY] = satisfied_ids


class InferenceRecorder(EvaluationObserver):
    """Observer that records inferred instances for later explanation.

    Replaces the ``@record_inferences`` decorator that was previously applied to
    ``InstantiatedVariable._instantiate_using_child_vars_and_yield_results_``.
    """

    def on_result_yielded(self, expression, result):
        if not getattr(type(expression), "_is_monitored_", False):
            return
        if expression._id_ not in result.bindings:
            return
        # Only record for InstantiatedVariable subclasses whose _evaluate__
        # delegates to _instantiate_using_child_vars_and_yield_results_ (i.e.
        # those that actually create new instances).  Query and its subclasses
        # (Entity, SetOf) override _evaluate__ and merely remap bindings
        # without creating new inferred instances.
        from krrood.entity_query_language.core.variable import (
            InstantiatedVariable,
        )
        from krrood.entity_query_language.query.query import Query

        if not isinstance(expression, InstantiatedVariable):
            return
        if isinstance(expression, Query):
            return
        from krrood.entity_query_language.explanation import register_inference

        register_inference(result.bindings[expression._id_], expression, result)
