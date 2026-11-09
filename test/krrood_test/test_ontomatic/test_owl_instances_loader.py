"""
Individuals loaded into the generated classes get the classes and the facts the ontology entails
(``dataset/ontologies/*_individuals.ttl``).
"""

from __future__ import annotations

import pytest

from krrood.ontomatic.ontology_to_python.owl_instances_loader import (
    OwlInstancesRegistry,
)
from krrood.ontomatic.property_descriptor.property_descriptor_relation import (
    PropertyDescriptorRelation,
)
from krrood.patterns.role import Role
from .generated_ontology import GeneratedOntology

PIZZA = "http://www.co-ode.org/ontologies/pizza/pizza.owl#"
"""
The namespace of the pizza individuals.
"""
BENCHMARK = "http://benchmark/OWL2Bench#"
"""
The namespace of the hobby individuals.
"""
FAMILY = "http://example.org/family#"
"""
The namespace of the family individuals.
"""
ROOMS = "http://example.org/rooms#"
"""
The namespace of the room individuals.
"""


@pytest.fixture(scope="module")
def pizzeria(tmp_path_factory) -> OwlInstancesRegistry:
    """
    A pizza with mozzarella and tomato.
    """
    return GeneratedOntology.of("class_expressions.ttl").loaded_individuals(
        tmp_path_factory.mktemp("pizzeria"), "class_expressions_individuals.ttl"
    )


@pytest.fixture(scope="module")
def hobbies(tmp_path_factory) -> OwlInstancesRegistry:
    """
    A person who loves football, and one without hobbies.
    """
    return GeneratedOntology.of("no_implicit_subsumption.ttl").loaded_individuals(
        tmp_path_factory.mktemp("hobbies"), "no_implicit_subsumption_individuals.ttl"
    )


@pytest.fixture(scope="module")
def family(tmp_path_factory) -> OwlInstancesRegistry:
    """
    Anna, her parent Ben and his parent Clara.
    """
    return GeneratedOntology.of("property_chains.ttl").loaded_individuals(
        tmp_path_factory.mktemp("family"), "property_chains_individuals.ttl"
    )


@pytest.fixture
def rooms(tmp_path) -> OwlInstancesRegistry:
    """
    :return: Three rooms, each connected to the next by a symmetric and transitive property, loaded for each test
        so that the symbol graph still holds them.
    """
    return GeneratedOntology.of("connected_rooms.ttl").loaded_individuals(
        tmp_path, "connected_rooms_individuals.ttl"
    )


def class_names(registry: OwlInstancesRegistry, iri: str) -> set[str]:
    """
    :return: The names of the classes of the objects representing the individual.
    """
    return {
        type(individual_object).__name__ for individual_object in registry.resolve(iri)
    }


# %% classification


def test_an_individual_is_classified_by_a_definition(pizzeria):
    """
    A pizza with a cheese topping is a cheesy pizza.
    """
    assert "CheesyPizza" in class_names(pizzeria, f"{PIZZA}margherita")


def test_an_individual_keeps_its_asserted_class(pizzeria):
    """
    A topping keeps the class it is asserted to have.
    """
    assert class_names(pizzeria, f"{PIZZA}mozzarella") == {"MozzarellaTopping"}


# %% facts


def test_the_inverse_of_an_asserted_fact_is_inferred(pizzeria):
    """
    The topping of a pizza is a topping of the pizza.
    """
    margherita = pizzeria.resolve(f"{PIZZA}margherita")[0]
    mozzarella = pizzeria.resolve(f"{PIZZA}mozzarella")[0]
    assert margherita in mozzarella.is_topping_of


def test_a_fact_of_a_property_chain_is_inferred(family):
    """
    The parent of a parent is a grandparent.
    """
    anna = family.resolve(f"{FAMILY}anna")[0]
    clara = family.resolve(f"{FAMILY}clara")[0]
    assert anna.has_grandparent == {clara}


def test_a_symmetric_and_transitive_property_relates_every_two_connected_individuals(
    rooms,
):
    """
    Rooms connected through a third are connected, in both directions.
    """
    kitchen = rooms.resolve(f"{ROOMS}kitchen")[0]
    bedroom = rooms.resolve(f"{ROOMS}bedroom")[0]
    assert bedroom in kitchen.is_connected_to
    assert kitchen in bedroom.is_connected_to


def test_a_fact_of_a_symmetric_and_transitive_property_is_explained_by_its_component(
    rooms,
):
    """
    The fact that two rooms are connected is explained by the component of connected
    rooms they belong to.
    """
    kitchen = rooms.resolve(f"{ROOMS}kitchen")[0]
    bedroom = rooms.resolve(f"{ROOMS}bedroom")[0]
    is_connected_to = type(vars(type(kitchen))["is_connected_to"])
    explanation = PropertyDescriptorRelation.find(
        kitchen, is_connected_to, bedroom
    ).explain()
    assert explanation.splitlines() == [
        explanation.splitlines()[0],
        f"  component 0 of {is_connected_to.__name__} with 3 members",
    ]
    assert explanation.splitlines()[0].endswith("(symmetric_transitive_component)")


# %% roles


def test_an_individual_entailed_to_have_a_role_is_represented_by_the_role(hobbies):
    """
    A person who loves a sport is a sports lover and a person with a hobby, as roles of the person.
    """
    assert class_names(hobbies, f"{BENCHMARK}bob") == {"SportsLover", "PeopleWithHobby"}
    assert all(isinstance(role, Role) for role in hobbies.resolve(f"{BENCHMARK}bob"))


def test_the_role_taker_of_a_role_is_the_individual(hobbies):
    """
    The role taker of a role of an individual carries the individual's IRI.
    """
    role = hobbies.resolve(f"{BENCHMARK}bob")[0]
    assert role.role_taker.uri == f"{BENCHMARK}bob"
    assert type(role.role_taker).__name__ == "Person"


def test_a_fact_of_a_subproperty_is_a_fact_of_the_superproperty(hobbies):
    """
    Loving football is liking it.
    """
    role = hobbies.resolve(f"{BENCHMARK}bob")[0]
    football = hobbies.resolve(f"{BENCHMARK}football")[0]
    assert football in role.role_taker.likes


def test_an_individual_without_entailed_roles_keeps_its_class(hobbies):
    """
    A person without hobbies is a person.
    """
    assert class_names(hobbies, f"{BENCHMARK}alice") == {"Person"}
