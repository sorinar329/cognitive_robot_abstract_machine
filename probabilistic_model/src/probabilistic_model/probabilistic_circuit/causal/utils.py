from __future__ import annotations

from typing import Any, Set

from random_events.variable import Variable


def variables_of_simple_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys constrained by a SimpleEvent.

    A SimpleEvent is a VariableMap — directly iterable as a dict via .keys().

    :param support_event: A SimpleEvent whose variable keys to extract.
    :returns: Set of Variable objects keyed in the event.
    """
    return set(support_event.keys())


def variables_of_composite_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys constrained by a composite Event.

    A composite Event exposes .simple_sets, each of which is a SimpleEvent.
    Delegates per-SimpleEvent extraction to variables_of_simple_event.

    :param support_event: A composite Event whose variable keys to extract.
    :returns: Union of Variable keys across all simple sets.
    """
    variable_set: Set[Variable] = set()
    for simple_set in support_event.simple_sets:
        variable_set.update(variables_of_simple_event(simple_set))
    return variable_set


def variables_of_support_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys constrained by a support event.

    Routes to variables_of_simple_event or variables_of_composite_event
    based on whether the event exposes .simple_sets.

    :param support_event: A SimpleEvent or composite Event.
    :returns: Set of Variable objects constrained by the event.
    """
    if hasattr(support_event, "simple_sets"):
        return variables_of_composite_event(support_event)
    return variables_of_simple_event(support_event)