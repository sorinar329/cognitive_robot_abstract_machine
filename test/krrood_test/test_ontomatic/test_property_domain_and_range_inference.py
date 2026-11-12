"""
Domains and ranges a property inherits from the properties it specialises.
"""

from __future__ import annotations

import rdflib

from krrood.ontomatic.ontology_to_python.ontology_info import (
    OntologyInfo,
    PropertyInfo,
)
from krrood.ontomatic.ontology_to_python.property_inference import (
    PropertyDomainAndRangeInference,
)
from krrood.ontomatic.ontology_to_python.property_type import PropertyType

# %% inheriting from a superproperty


def ontology_with(*properties: PropertyInfo) -> OntologyInfo:
    """
    :return: An ontology with the properties.
    """
    return OntologyInfo(
        graph=rdflib.Graph(),
        original_properties={
            property_information.name: property_information
            for property_information in properties
        },
    )


def test_a_subproperty_without_a_domain_takes_the_domain_of_its_superproperty():
    """
    A subproperty declaring no domain or range gets those of its superproperty.
    """
    ontology = ontology_with(
        PropertyInfo(
            name="hasTopping",
            uri="http://example.org/pizza#hasTopping",
            type=PropertyType.OBJECT_PROPERTY,
            domains=["Pizza"],
            ranges=["Topping"],
        ),
        PropertyInfo(
            name="hasCheese",
            uri="http://example.org/pizza#hasCheese",
            type=PropertyType.OBJECT_PROPERTY,
            superproperties=["hasTopping"],
        ),
    )
    has_cheese = ontology.properties["hasCheese"]
    PropertyDomainAndRangeInference(
        property_information=has_cheese, ontology=ontology
    ).complete()
    assert (has_cheese.domains, has_cheese.ranges) == (["Pizza"], ["Topping"])
