"""
Individuals of an OWL ontology whose classes are still being inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdflib import URIRef
from typing_extensions import Any, Iterable

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.predicate import RenderedFields, SymbolicFunction
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    FunctionVerbalizationTemplates,
)
from krrood.patterns.role_predicates import IsSubClassOrRole


@dataclass(kw_only=True)
class AnonymousClass:
    """
    An individual collecting the classes it is inferred to belong to, before it is
    represented by objects of those classes.
    """

    uri: URIRef
    """
    The IRI of the individual.
    """

    types: set[type] = field(default_factory=set)
    """
    The classes the individual is inferred to belong to.
    """

    final_sorted_types: list[type] = field(default_factory=list)
    """
    The most specific of :attr:`types`, from the least to the most specific.
    """

    def add_type(self, *, cls: type):
        """
        Record that the individual belongs to the class.
        """
        self.types.add(cls)

    @staticmethod
    def variable_for(*, candidate: AnonymousClass | Variable) -> Variable:
        """
        :param candidate: An individual, or a variable standing for one.
        :return: The variable an axiom of a generated class is stated over.
        """
        if isinstance(candidate, Variable):
            return candidate
        return variable(AnonymousClass, [candidate])

    def __hash__(self):
        """
        Individuals are hashed by their IRI.
        """
        return hash(self.uri)

    def __eq__(self, other):
        """
        Individuals are equal when they have the same IRI.
        """
        return hash(self) == hash(other)


@dataclass(eq=False, kw_only=True)
class NumberOfValuesOfClass(SymbolicFunction):
    """
    The number of values of a property of an individual that are inferred to belong to a
    class.
    """

    values: Iterable[AnonymousClass]
    """
    The values of the property.
    """

    value_class: type | tuple[type, ...]
    """
    The class, or any of the classes, a counted value belongs to.
    """

    def __call__(self) -> int:
        return self.count(values=self.values, value_class=self.value_class)

    @staticmethod
    def count(
        *, values: Iterable[AnonymousClass], value_class: type | tuple[type, ...]
    ) -> int:
        """
        :return: The number of *values* inferred to belong to *value_class*.
        """
        return sum(
            1
            for value in values
            if any(
                IsSubClassOrRole.holds(sub_class=cls, super_class=value_class)
                for cls in value.types
            )
        )

    @classmethod
    def _verbalization_fragment_(
        cls, *, fields: RenderedFields
    ) -> VerbalizationFragment:
        return FunctionVerbalizationTemplates.possessive(
            *fields.values(), function_class=cls
        )

    @classmethod
    def _example_operand_values_(cls) -> dict[str, Any]:
        """
        A class stands for :attr:`value_class`, see :meth:`~krrood.entity_query_language
        .predicate.SymbolicCallable._example_operand_values_`.
        """
        return {"value_class": object}
