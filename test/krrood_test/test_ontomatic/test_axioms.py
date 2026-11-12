"""
The Python conditions an axiom generates hold exactly for the individuals the OWL
restriction describes.

Each condition is evaluated on an individual the way a generated class evaluates it.
"""

from __future__ import annotations

import uuid

import pytest
import rdflib
from typing_extensions import Any

from krrood.ontomatic.anonymous_class import AnonymousClass
from krrood.ontomatic.ontology_to_python.axioms import (
    AxiomInfo,
    ComplementAxiomInfo,
    HasValueAxiomInfo,
    MaxCardinalityAxiomInfo,
    SomeValuesFromAxiomInfo,
    SubClassAxiomInfo,
)
from krrood.ontomatic.ontology_to_python.ontology_info import (
    OntologyInfo,
    PropertyInfo,
)
from krrood.ontomatic.ontology_to_python.property_type import PropertyType
from krrood.ontomatic.property_descriptor.property_descriptor import (
    HasProperty,
    PropertyDescriptor,
)
from krrood.patterns.role_predicates import IsSubClassOrRole

# %% the classes and properties the conditions name


class Pizza:
    """
    The class the axioms classify.
    """


class Topping:
    """
    The range of the topping property.
    """


class CheeseTopping(Topping):
    """
    A topping a restriction can require.
    """


class MeatTopping(Topping):
    """
    A topping the restriction does not require.
    """


class HasTopping(PropertyDescriptor):
    """
    A property with several values.
    """


class HasBase(PropertyDescriptor):
    """
    A functional property: it has one value.
    """


@pytest.fixture(scope="module")
def ontology() -> OntologyInfo:
    """
    An ontology with a property of several values and a functional one.
    """
    return OntologyInfo(
        graph=rdflib.Graph(),
        original_properties={
            "hasTopping": PropertyInfo(
                name="hasTopping",
                uri="http://example.org/pizza#hasTopping",
                type=PropertyType.OBJECT_PROPERTY,
            ),
            "hasBase": PropertyInfo(
                name="hasBase",
                uri="http://example.org/pizza#hasBase",
                type=PropertyType.OBJECT_PROPERTY,
                is_functional=True,
            ),
        },
    )


def individual(*types: type, **values: Any) -> AnonymousClass:
    """
    :return: An individual of the classes, with the property values as attributes.
    """
    created = AnonymousClass(uri=f"http://example.org/pizza#{uuid.uuid4().hex}")
    for value_type in types:
        created.add_type(cls=value_type)
    for name, value in values.items():
        setattr(created, name, value)
    return created


def holds(axiom: AxiomInfo, candidate: AnonymousClass) -> bool:
    """
    :return: Whether every Python condition of the axiom holds for the candidate.
    """
    names = {
        "candidate": candidate,
        "HasProperty": HasProperty,
        "IsSubClassOrRole": IsSubClassOrRole,
        **{
            cls.__name__: cls
            for cls in (Pizza, CheeseTopping, MeatTopping, HasTopping, HasBase)
        },
    }
    return all(eval(condition, names) for condition in axiom.conditions_python())


# %% restrictions on the values of a property


def test_some_values_from_holds_when_a_value_is_of_the_class(ontology):
    """
    ``someValuesFrom`` holds when one value of the property is of the class.
    """
    axiom = SomeValuesFromAxiomInfo(
        property_name="hasTopping",
        for_class="Pizza",
        ontology=ontology,
        on_class="CheeseTopping",
    )
    candidate = individual(
        has_topping=[individual(MeatTopping), individual(CheeseTopping)]
    )
    assert holds(axiom, candidate) is True


def test_some_values_from_fails_when_no_value_is_of_the_class(ontology):
    """
    ``someValuesFrom`` fails when no value of the property is of the class.
    """
    axiom = SomeValuesFromAxiomInfo(
        property_name="hasTopping",
        for_class="Pizza",
        ontology=ontology,
        on_class="CheeseTopping",
    )
    assert holds(axiom, individual(has_topping=[individual(MeatTopping)])) is False


def test_a_restriction_fails_for_an_individual_without_the_property(ontology):
    """
    An individual whose classes do not have the property satisfies no restriction on it.
    """
    axiom = SomeValuesFromAxiomInfo(
        property_name="hasTopping",
        for_class="Pizza",
        ontology=ontology,
        on_class="CheeseTopping",
    )
    assert holds(axiom, individual()) is False


def test_has_value_looks_for_the_individual_among_several_values(ontology):
    """
    ``hasValue`` on a property with several values holds when the individual is one of
    them.
    """
    mozzarella = individual(CheeseTopping)
    axiom = HasValueAxiomInfo(
        property_name="hasTopping",
        for_class="Pizza",
        ontology=ontology,
        value=rdflib.URIRef(mozzarella.uri),
    )
    assert holds(axiom, individual(has_topping=[mozzarella])) is True


def test_has_value_compares_the_single_value_of_a_functional_property(ontology):
    """
    ``hasValue`` on a functional property compares its one value with the individual.
    """
    thin_base = individual()
    axiom = HasValueAxiomInfo(
        property_name="hasBase",
        for_class="Pizza",
        ontology=ontology,
        value=rdflib.URIRef(thin_base.uri),
    )
    assert holds(axiom, individual(has_base=thin_base)) is True


# %% cardinalities


def test_a_maximum_cardinality_holds_up_to_the_maximum(ontology):
    """
    ``maxCardinality 1`` holds for one value.
    """
    axiom = MaxCardinalityAxiomInfo(
        property_name="hasTopping", for_class="Pizza", ontology=ontology, quantity=1
    )
    assert holds(axiom, individual(has_topping=[individual(CheeseTopping)])) is True


def test_a_maximum_cardinality_fails_above_the_maximum(ontology):
    """
    ``maxCardinality 1`` fails for two values.
    """
    axiom = MaxCardinalityAxiomInfo(
        property_name="hasTopping", for_class="Pizza", ontology=ontology, quantity=1
    )
    candidate = individual(
        has_topping=[individual(CheeseTopping), individual(MeatTopping)]
    )
    assert holds(axiom, candidate) is False


# %% class expressions


def test_a_complement_holds_where_its_restriction_fails(ontology):
    """
    The complement of a restriction holds for an individual the restriction does not
    describe.
    """
    axiom = ComplementAxiomInfo(
        complemented=SomeValuesFromAxiomInfo(
            property_name="hasTopping",
            for_class="Pizza",
            ontology=ontology,
            on_class="MeatTopping",
        )
    )
    assert holds(axiom, individual(has_topping=[individual(CheeseTopping)])) is True


def test_a_subclass_axiom_holds_for_an_individual_of_the_class():
    """
    A subclass axiom holds for an individual of the class.
    """
    assert holds(SubClassAxiomInfo(sub_class="Pizza"), individual(Pizza)) is True
