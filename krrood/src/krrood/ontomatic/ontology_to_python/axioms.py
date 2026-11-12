"""
The conditions OWL class expressions put on the individuals of a class, rendered as the classification axioms of the
generated classes: an entity-query-language form over ``candidate_var`` (``axiom``), after the setup statements
declaring its variables, and a Python form over ``candidate`` (``axiom_python``).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import rdflib
from rdflib import URIRef
from typing_extensions import Any, ClassVar, TYPE_CHECKING

from krrood.ontomatic.naming import NamingRegistry
from krrood.ontomatic.ontology_to_python.property_type import PropertyType

if TYPE_CHECKING:
    from krrood.ontomatic.ontology_to_python.ontology_info import OntologyInfo


# %% conditions


@dataclass(kw_only=True)
class AxiomInfo(ABC):
    """
    A condition on the individuals of a class.
    """

    def setup_statements(self) -> list[str]:
        """
        :return: The statements declaring the variables the entity-query-language form of the condition quantifies
            over.
        """
        return []

    @abstractmethod
    def conditions_eql(self) -> list[str]:
        """
        :return: The condition as entity-query-language expressions over ``candidate_var``, all of which must hold.
        """

    @abstractmethod
    def conditions_python(self) -> list[str]:
        """
        :return: The condition as Python expressions over ``candidate``, all of which must hold.
        """

    @staticmethod
    def variable_name_part(*, class_expression: str) -> str:
        """
        :param class_expression: The Python source of a class or of a tuple of classes.
        :return: A part of a variable name that tells the class expression apart.
        """
        return NamingRegistry.to_snake_case(
            name=re.sub(r"\W+", "_", class_expression).strip("_")
        )


@dataclass(kw_only=True)
class SubClassAxiomInfo(AxiomInfo):
    """
    The individual belongs to a class.
    """

    sub_class: str
    """
    The class the individual belongs to.
    """

    @property
    def types_variable(self) -> str:
        """
        The name of the variable over the classes of the individual.
        """
        return f"candidate_types_for_{self.variable_name_part(class_expression=self.sub_class)}"

    def setup_statements(self) -> list[str]:
        """
        Declares the variable over the classes of the individual, see :meth:`AxiomInfo.setup_statements`.
        """
        return [f"{self.types_variable} = variable_from(candidate_var.types)"]

    def conditions_eql(self) -> list[str]:
        return [
            f"exists({self.types_variable}, IsSubClassOrRole(sub_class={self.types_variable}, super_class={self.sub_class}))"
        ]

    def conditions_python(self) -> list[str]:
        return [
            f"any(IsSubClassOrRole.holds(sub_class=t, super_class={self.sub_class}) for t in candidate.types)"
        ]


@dataclass(kw_only=True)
class PropertyAxiomInfo(AxiomInfo):
    """
    The individual has values of a property.
    """

    property_name: str
    """
    The name of the property in the ontology.
    """
    for_class: str
    """
    The class the condition is a condition of.
    """

    ontology: OntologyInfo
    """
    The ontology the property is declared in.
    """

    @property
    def snake_property_name(self) -> str:
        """
        The name of the attribute holding the values of the property.
        """
        return NamingRegistry.to_snake_case(name=self.property_name)

    @property
    def descriptor_name(self) -> str:
        """
        The name of the descriptor class of the property.
        """
        return NamingRegistry.to_pascal_case(name=self.snake_property_name)

    def conditions_eql(self) -> list[str]:
        return [
            f"HasProperty(instance=candidate_var, property_descriptor_type={self.descriptor_name})"
        ]

    def conditions_python(self) -> list[str]:
        return [
            f"HasProperty(instance=candidate, property_descriptor_type={self.descriptor_name})"
        ]


@dataclass(kw_only=True)
class QualifiedAxiomInfoMixin:
    """
    The values of the property are restricted to a class.
    """

    on_class: str
    """
    The Python source of the class, or of the tuple of classes, the values are restricted to.
    """


@dataclass(kw_only=True)
class ValuesOfClassAxiomInfo(PropertyAxiomInfo, QualifiedAxiomInfoMixin, ABC):
    """
    A restriction quantifying over the values of a property and their classes.
    """

    @property
    def values_variable(self) -> str:
        """
        The name of the variable over the values of the property.
        """
        return f"{self.snake_property_name}_for_{self.variable_name_part(class_expression=self.on_class)}"

    @property
    def value_types_variable(self) -> str:
        """
        The name of the variable over the classes of a value of the property.
        """
        return f"{self.values_variable}_types"

    def setup_statements(self) -> list[str]:
        """
        Declares the variables over the values of the property and over their classes, see
        :meth:`AxiomInfo.setup_statements`.
        """
        return [
            f"{self.values_variable} = variable_from(candidate_var.{self.snake_property_name})",
            f"{self.value_types_variable} = variable_from({self.values_variable}.types)",
        ]

    @property
    def value_is_of_class_eql(self) -> str:
        """
        The entity-query-language condition that a value of the property belongs to the
        class.
        """
        return (
            f"exists({self.value_types_variable}, "
            f"IsSubClassOrRole(sub_class={self.value_types_variable}, super_class={self.on_class}))"
        )

    @property
    def value_is_of_class_python(self) -> str:
        """
        The Python condition that a value ``value`` of the property belongs to the
        class.
        """
        return f"any(IsSubClassOrRole.holds(sub_class=t, super_class={self.on_class}) for t in value.types)"


@dataclass(kw_only=True)
class QuantifiedAxiomInfo(PropertyAxiomInfo, ABC):
    """
    The individual has a number of values of a property.
    """

    quantity: int
    """
    The number the count of values is compared to.
    """
    comparison_operator: ClassVar[str]
    """
    The Python operator comparing the count of values to the quantity.
    """

    @property
    def counted_values_eql(self) -> str:
        """
        The entity-query-language expression counting the values.
        """
        return f"length(iterable=candidate_var.{self.snake_property_name})"

    @property
    def counted_values_python(self) -> str:
        """
        The Python expression counting the values.
        """
        return f"len(candidate.{self.snake_property_name})"

    def conditions_eql(self) -> list[str]:
        """
        Adds the comparison of the count of values to :meth:`PropertyAxiomInfo.conditions_eql`.
        """
        return super().conditions_eql() + [
            f"{self.counted_values_eql} {self.comparison_operator} {self.quantity}"
        ]

    def conditions_python(self) -> list[str]:
        """
        Adds the comparison of the count of values to :meth:`PropertyAxiomInfo.conditions_python`.
        """
        return super().conditions_python() + [
            f"({self.counted_values_python} {self.comparison_operator} {self.quantity})"
        ]


@dataclass(kw_only=True)
class CardinalityAxiomInfo(QuantifiedAxiomInfo):
    """
    The individual has exactly a number of values of a property.
    """

    comparison_operator: ClassVar[str] = "=="
    """
    Exactly the quantity.
    """


@dataclass(kw_only=True)
class MaxCardinalityAxiomInfo(QuantifiedAxiomInfo):
    """
    The individual has at most a number of values of a property.
    """

    comparison_operator: ClassVar[str] = "<="
    """
    At most the quantity.
    """


@dataclass(kw_only=True)
class MinCardinalityAxiomInfo(QuantifiedAxiomInfo):
    """
    The individual has at least a number of values of a property.
    """

    comparison_operator: ClassVar[str] = ">="
    """
    At least the quantity.
    """


@dataclass(kw_only=True)
class QuantifiedQualifiedAxiomInfo(QuantifiedAxiomInfo, QualifiedAxiomInfoMixin, ABC):
    """
    The individual has a number of values of a property that belong to a class.
    """

    @property
    def counted_values_eql(self) -> str:
        """
        Counts only the values belonging to the class, see :attr:`QuantifiedAxiomInfo.counted_values_eql`.
        """
        return f"NumberOfValuesOfClass(values=candidate_var.{self.snake_property_name}, value_class={self.on_class})"

    @property
    def counted_values_python(self) -> str:
        """
        Counts only the values belonging to the class, see :attr:`QuantifiedAxiomInfo.counted_values_python`.
        """
        return f"NumberOfValuesOfClass.count(values=candidate.{self.snake_property_name}, value_class={self.on_class})"


@dataclass(kw_only=True)
class QualifiedCardinalityAxiomInfo(QuantifiedQualifiedAxiomInfo):
    """
    The individual has exactly a number of values of a property that belong to a class.
    """

    comparison_operator: ClassVar[str] = "=="
    """
    Exactly the quantity.
    """


@dataclass(kw_only=True)
class MaxQualifiedCardinalityAxiomInfo(QuantifiedQualifiedAxiomInfo):
    """
    The individual has at most a number of values of a property that belong to a class.
    """

    comparison_operator: ClassVar[str] = "<="
    """
    At most the quantity.
    """


@dataclass(kw_only=True)
class MinQualifiedCardinalityAxiomInfo(QuantifiedQualifiedAxiomInfo):
    """
    The individual has at least a number of values of a property that belong to a class.
    """

    comparison_operator: ClassVar[str] = ">="
    """
    At least the quantity.
    """


@dataclass(kw_only=True)
class HasValueAxiomInfo(PropertyAxiomInfo):
    """
    A value of a property of the individual is a given individual or literal
    (``owl:hasValue``).
    """

    value: Any
    """
    The individual or literal.
    """

    @property
    def value_source(self) -> str:
        """
        The Python source of the value: the IRI of an individual, or the literal.
        """
        if isinstance(self.value, (URIRef, str)):
            return repr(str(self.value))
        return repr(self.value)

    @property
    def has_several_values(self) -> bool:
        """
        Whether the attribute of the property holds a set of values.
        """
        property_information = self.ontology.properties[self.property_name]
        return (
            property_information.type == PropertyType.OBJECT_PROPERTY
            and not property_information.is_functional
        )

    @property
    def values_variable(self) -> str:
        """
        The name of the variable over the values of the property.
        """
        return f"{self.snake_property_name}_values"

    def setup_statements(self) -> list[str]:
        """
        Declares the variable over the values of the property when an individual is among several values, see
        :meth:`AxiomInfo.setup_statements`.
        """
        if not (isinstance(self.value, URIRef) and self.has_several_values):
            return []
        return [
            f"{self.values_variable} = variable_from(candidate_var.{self.snake_property_name})"
        ]

    def conditions_eql(self) -> list[str]:
        """
        Adds the comparison of the values of the property to the value to :meth:`PropertyAxiomInfo.conditions_eql`.
        """
        attribute = f"candidate_var.{self.snake_property_name}"
        if isinstance(self.value, URIRef):
            if self.has_several_values:
                condition = f"exists({self.values_variable}, {self.values_variable}.uri == {self.value_source})"
            else:
                condition = f"{attribute}.uri == {self.value_source}"
        elif self.has_several_values:
            condition = f"contains({attribute}, {self.value_source})"
        else:
            condition = f"{attribute} == {self.value_source}"
        return super().conditions_eql() + [condition]

    def conditions_python(self) -> list[str]:
        """
        Adds the comparison of the values of the property to the value to
        :meth:`PropertyAxiomInfo.conditions_python`.
        """
        attribute = f"candidate.{self.snake_property_name}"
        if isinstance(self.value, URIRef):
            if self.has_several_values:
                condition = f"({self.value_source} in (str(value.uri) for value in {attribute}))"
            else:
                condition = f"(str({attribute}.uri) == {self.value_source})"
        elif self.has_several_values:
            condition = f"({self.value_source} in {attribute})"
        else:
            condition = f"({attribute} == {self.value_source})"
        return super().conditions_python() + [condition]


@dataclass(kw_only=True)
class SomeValuesFromAxiomInfo(ValuesOfClassAxiomInfo):
    """
    A value of a property of the individual belongs to a class (``owl:someValuesFrom``).
    """

    def conditions_eql(self) -> list[str]:
        """
        Adds that a value belongs to the class to :meth:`PropertyAxiomInfo.conditions_eql`.
        """
        return super().conditions_eql() + [
            f"exists({self.values_variable}, {self.value_is_of_class_eql})"
        ]

    def conditions_python(self) -> list[str]:
        """
        Adds that a value belongs to the class to :meth:`PropertyAxiomInfo.conditions_python`.
        """
        return super().conditions_python() + [
            f"any({self.value_is_of_class_python} for value in candidate.{self.snake_property_name})"
        ]


@dataclass(kw_only=True)
class AllValuesFromAxiomInfo(ValuesOfClassAxiomInfo):
    """
    Every value of a property of the individual belongs to a class
    (``owl:allValuesFrom``).
    """

    def conditions_eql(self) -> list[str]:
        """
        Adds that every value belongs to the class to :meth:`PropertyAxiomInfo.conditions_eql`.
        """
        return super().conditions_eql() + [
            f"for_all({self.values_variable}, {self.value_is_of_class_eql})"
        ]

    def conditions_python(self) -> list[str]:
        """
        Adds that every value belongs to the class to :meth:`PropertyAxiomInfo.conditions_python`.
        """
        return super().conditions_python() + [
            f"all({self.value_is_of_class_python} for value in candidate.{self.snake_property_name})"
        ]


@dataclass(kw_only=True)
class ComplementAxiomInfo(AxiomInfo):
    """
    A condition does not hold for the individual (``owl:complementOf``).
    """

    complemented: AxiomInfo
    """
    The condition that must not hold.
    """

    def setup_statements(self) -> list[str]:
        """
        The statements of the complemented condition, see :meth:`AxiomInfo.setup_statements`.
        """
        return self.complemented.setup_statements()

    def conditions_eql(self) -> list[str]:
        conditions = self.complemented.conditions_eql()
        if len(conditions) == 1:
            return [f"not_({conditions[0]})"]
        return [f"not_(and_({', '.join(conditions)}))"]

    def conditions_python(self) -> list[str]:
        return [f"not ({' and '.join(self.complemented.conditions_python())})"]


@dataclass(kw_only=True)
class OneOfAxiomInfo(AxiomInfo):
    """
    The individual is one of a list of individuals (``owl:oneOf``).
    """

    individual_iris: list[str]
    """
    The IRIs of the listed individuals.
    """

    def conditions_eql(self) -> list[str]:
        return [f"in_(candidate_var.uri, {tuple(self.individual_iris)!r})"]

    def conditions_python(self) -> list[str]:
        return [f"(str(candidate.uri) in {tuple(self.individual_iris)!r})"]
