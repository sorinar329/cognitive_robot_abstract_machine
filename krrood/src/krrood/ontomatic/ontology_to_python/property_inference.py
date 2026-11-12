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

    property_info: PropertyInfo
    """
    The property whose domains and ranges are completed.
    """

    onto: OntologyInfo
    """
    The ontology the property belongs to.
    """

    def complete(self):
        """
        Add the domains and ranges every rule concludes, until no rule adds anything
        new.
        """
        while self._add_new(
            values=self.property_info.domains, conclusions=self.inferred_domains()
        ) | self._add_new(
            values=self.property_info.ranges, conclusions=self.inferred_ranges()
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
        return self.onto.original_properties[self.property_info.name]

    @property
    def has_related_properties(self) -> bool:
        """
        Whether the property has superproperties, inverses or equivalent properties.
        """
        return bool(
            self.property_info.superproperties
            or self.property_info.inverses
            or self.property_info.equivalent_properties
        )

    def inferred_domains(self) -> Iterator[list[str]]:
        """
        :return: The domains each applicable rule concludes.
        """
        info = self.property_info
        if len(info.equivalent_properties) == 1:
            yield self.onto.properties[info.equivalent_properties[0]].domains
        if len(info.inverses) == 1:
            yield self.onto.properties[info.inverses[0]].ranges
        if self.declared.domains:
            return
        inverses_declare_no_ranges = all(
            not self.onto.original_properties[inverse].ranges
            for inverse in info.inverses
        )
        if inverses_declare_no_ranges and info.superproperties:
            yield list(
                dict.fromkeys(
                    domain
                    for superproperty in info.superproperties
                    for domain in self.onto.properties[superproperty].domains
                )
            )
        if not self.has_related_properties:
            yield [self.onto.base_cls_name]
        if inverses_declare_no_ranges and not info.equivalent_properties:
            yield self.declared_values_of_superproperties(
                values_of=lambda superproperty: superproperty.domains
            )

    def inferred_ranges(self) -> Iterator[list[str]]:
        """
        :return: The ranges each applicable rule concludes.
        """
        info = self.property_info
        if info.equivalent_properties:
            yield self.onto.properties[info.equivalent_properties[0]].ranges
        if info.inverses:
            yield self.onto.properties[info.inverses[0]].domains
        if self.declared.ranges:
            return
        inverses_declare_no_domains = all(
            not self.onto.original_properties[inverse].domains
            for inverse in info.inverses
        )
        if inverses_declare_no_domains and not info.equivalent_properties:
            yield self.declared_values_of_superproperties(
                values_of=lambda superproperty: superproperty.ranges
            )
        if not self.has_related_properties:
            yield [self.onto.base_cls_name]

    def declared_values_of_superproperties(
        self, *, values_of: Callable[[PropertyInfo], list[str]]
    ) -> list[str]:
        """
        :param values_of: Selects the domains or the ranges of a property.
        :return: The declared domains or ranges of the superproperties, from the most general superproperty on.
        """
        return [
            value
            for superproperty in reversed(self.property_info.sorted_superproperties)
            for value in values_of(self.onto.original_properties[superproperty])
        ]
