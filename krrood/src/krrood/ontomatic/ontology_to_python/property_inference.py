"""
Domains and ranges of object properties the ontology leaves undeclared, inferred from
related properties.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from krrood.ontomatic.ontology_to_python.ontology_info import (
        OntologyInfo,
        PropertyInfo,
    )


@dataclass(kw_only=True)
class PropertyDomainAndRangeInference:
    """
    Completes the domains and ranges of an object property from its equivalent
    properties, its inverses and its superproperties, and falls back to the base class
    of the ontology when nothing relates it to a class.
    """

    property_information: PropertyInfo
    """
    The property whose domains and ranges are completed.
    """

    ontology: OntologyInfo
    """
    The ontology the property belongs to.
    """

    def complete(self):
        """
        Add the domains and ranges every rule concludes, until no rule adds anything
        new.
        """
        while self._add_new(
            values=self.property_information.domains,
            conclusions=self.inferred_domains(),
        ) | self._add_new(
            values=self.property_information.ranges, conclusions=self.inferred_ranges()
        ):
            pass

    @staticmethod
    def _add_new(*, values: list[str], conclusions: Iterator[list[str]]) -> bool:
        """
        Append the concluded values that are not among the values yet.

        :return: Whether a value was added.
        """
        added = False
        for conclusion in conclusions:
            for value in conclusion:
                if value not in values:
                    values.append(value)
                    added = True
        return added

    @property
    def declared(self) -> PropertyInfo:
        """
        The property as the ontology declares it.
        """
        return self.ontology.original_properties[self.property_information.name]

    @property
    def has_related_properties(self) -> bool:
        """
        Whether the property has superproperties, inverses or equivalent properties.
        """
        return bool(
            self.property_information.superproperties
            or self.property_information.inverses
            or self.property_information.equivalent_properties
        )

    def inferred_domains(self) -> Iterator[list[str]]:
        """
        :return: The domains each applicable rule concludes.
        """
        property_information = self.property_information
        if len(property_information.equivalent_properties) == 1:
            yield self.ontology.properties[
                property_information.equivalent_properties[0]
            ].domains
        if len(property_information.inverses) == 1:
            yield self.ontology.properties[property_information.inverses[0]].ranges
        if self.declared.domains:
            return
        inverses_declare_no_ranges = all(
            not self.ontology.original_properties[inverse].ranges
            for inverse in property_information.inverses
        )
        if inverses_declare_no_ranges and property_information.superproperties:
            yield list(
                dict.fromkeys(
                    domain
                    for superproperty in property_information.superproperties
                    for domain in self.ontology.properties[superproperty].domains
                )
            )
        if not self.has_related_properties:
            yield [self.ontology.base_class_name]
        if (
            inverses_declare_no_ranges
            and not property_information.equivalent_properties
        ):
            yield self.declared_values_of_superproperties(
                values_of=lambda superproperty: superproperty.domains
            )

    def inferred_ranges(self) -> Iterator[list[str]]:
        """
        :return: The ranges each applicable rule concludes.
        """
        property_information = self.property_information
        if property_information.equivalent_properties:
            yield self.ontology.properties[
                property_information.equivalent_properties[0]
            ].ranges
        if property_information.inverses:
            yield self.ontology.properties[property_information.inverses[0]].domains
        if self.declared.ranges:
            return
        inverses_declare_no_domains = all(
            not self.ontology.original_properties[inverse].domains
            for inverse in property_information.inverses
        )
        if (
            inverses_declare_no_domains
            and not property_information.equivalent_properties
        ):
            yield self.declared_values_of_superproperties(
                values_of=lambda superproperty: superproperty.ranges
            )
        if not self.has_related_properties:
            yield [self.ontology.base_class_name]

    def declared_values_of_superproperties(
        self, *, values_of: Callable[[PropertyInfo], list[str]]
    ) -> list[str]:
        """
        :param values_of: Selects the domains or the ranges of a property.
        :return: The declared domains or ranges of the superproperties, from the most general superproperty on.
        """
        return [
            value
            for superproperty in reversed(
                self.property_information.sorted_superproperties
            )
            for value in values_of(self.ontology.original_properties[superproperty])
        ]
