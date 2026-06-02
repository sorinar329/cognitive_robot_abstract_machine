"""
Conditional EQL operator constructs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)


@dataclass(eq=False)
class CaseWhen(Selectable):
    """
    Represents a conditional expression: CASE WHEN condition THEN value ELSE else_value END.

    Supports both local Python evaluation and SQL translation via the EQL translator.

    .. code-block:: python

        action = variable(MoveAction, domain=[])
        query = an(set_of(
            min(case_when(action.polymorphic_type == 'PickUpActionDAO', action.database_id))
        ))
    """

    condition: SymbolicExpression
    """The condition to evaluate."""

    then_value: SymbolicExpression
    """The value returned if the condition is true."""

    else_value: Optional[SymbolicExpression] = None
    """The value returned if the condition is false. Defaults to None."""

    @property
    def _children_(self) -> List[SymbolicExpression]:
        """Return child expressions for tree traversal."""
        children = [self.condition, self.then_value]
        if self.else_value is not None:
            children.append(self.else_value)
        return children

    @_children_.setter
    def _children_(self, value: Any) -> None:
        """
        The base dataclass automatically attempts to assign a default value
        to inherited fields during __init__. This setter catches that
        assignment to prevent an AttributeError.
        """
        pass

    def _replace_child_field_(self, old: Any, new: Any) -> None:
        """Replace a child expression node during EQL tree manipulation."""
        if self.condition is old:
            self.condition = new
        if self.then_value is old:
            self.then_value = new
        if self.else_value is old:
            self.else_value = new

    def _name_(self) -> str:
        """Return the symbolic name of this expression node."""
        return "case_when"

    def _evaluate__(self, sources: Any) -> Any:
        """
        Evaluate the condition locally in Python.

        :param sources: The variable bindings for evaluation
        :return: then_value if condition is true, else_value otherwise
        """
        cond_result = self.condition._evaluate__(sources)
        is_true = (
            bool(cond_result)
            if not isinstance(cond_result, list)
            else len(cond_result) > 0
        )

        if is_true:
            if hasattr(self.then_value, "_evaluate__"):
                return self.then_value._evaluate__(sources)
            return self.then_value
        else:
            if self.else_value is not None:
                if hasattr(self.else_value, "_evaluate__"):
                    return self.else_value._evaluate__(sources)
                return self.else_value
            return None


def case_when(
    condition: SymbolicExpression,
    then_value: SymbolicExpression,
    else_value: Optional[SymbolicExpression] = None,
) -> CaseWhen:
    """
    Create a CASE WHEN ... THEN ... ELSE ... END expression.

    .. code-block:: python

        action = variable(MoveAction, domain=[])
        query = an(set_of(
            min(case_when(action.polymorphic_type == 'PickUpActionDAO', action.database_id))
        ))

    :param condition: The condition to evaluate
    :param then_value: The value if condition is true
    :param else_value: The value if condition is false
    :return: A CaseWhen expression
    """
    return CaseWhen(condition, then_value, else_value)

