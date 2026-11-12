"""
Every OWL class expression of the pizza ontology (owlcs/pizza-ontology) is translated into the generated code:
definitions, unions as fillers, complements, enumerations, unqualified cardinalities, anonymous intersections as
fillers and values typed by their class (``dataset/ontologies/class_expressions.ttl``).
"""

from __future__ import annotations

import re

import pytest

from krrood.ontomatic.property_descriptor.property_descriptor import HasProperty
from .generated_ontology import GeneratedOntology

PIZZA_NAMESPACE = "http://www.co-ode.org/ontologies/pizza/pizza.owl#"
"""
The namespace of the ontology's classes and individuals.
"""


@pytest.fixture(scope="module")
def generated() -> GeneratedOntology:
    """
    The files ontomatic generates from the ontology.
    """
    return GeneratedOntology.generate_from_file(
        ontology_file_name="class_expressions.ttl"
    )


@pytest.fixture(scope="module")
def generated_module(generated, tmp_path_factory):
    """
    The generated module holding the classes, imported.
    """
    return generated.imported(root=tmp_path_factory.mktemp("generated"))


# %% definitions


def test_a_definition_makes_its_named_conjunct_a_superclass(generated):
    """
    The named conjunct of a definition is a superclass of the defined class.
    """
    assert re.search(r"^class CheesyPizza\(.*\bPizza\b", generated.classes_source, re.M)


def test_a_definition_classifies_by_its_named_conjunct_and_its_restriction(
    generated,
):
    """
    A definition classifies by its named conjunct and by its restriction.
    """
    axiom_python = generated.method_source(
        class_name="CheesyPizza", method_name="axiom_python"
    )
    assert (
        "IsSubClassOrRole.holds(sub_class=t, super_class=Pizza) for t in candidate.types"
        in axiom_python
    )
    assert (
        "IsSubClassOrRole.holds(sub_class=t, super_class=CheeseTopping)" in axiom_python
    )


# %% unions as fillers


def test_a_closure_over_a_union_checks_against_its_members(generated):
    """
    A closure axiom over a union checks the values against the tuple of its members.
    """
    necessary = generated.method_source(
        class_name="Margherita", method_name="necessary_conditions_python"
    )
    assert (
        "IsSubClassOrRole.holds(sub_class=t, super_class=(MozzarellaTopping, TomatoTopping))"
        in necessary
    )


def test_no_class_is_named_after_a_union(generated):
    """
    No class is named after a union.
    """
    assert re.findall(r"\b\w*ORTomatoTopping\w*\b", generated.classes_source) == []


# %% complements


def test_a_complement_of_a_restriction_negates_it(generated):
    """
    The complement of a restriction negates it in both forms of the axiom.
    """
    axiom_python = generated.method_source(
        class_name="VegetarianPizza", method_name="axiom_python"
    )
    assert (
        "not (HasProperty(instance=candidate, property_descriptor_type=HasTopping) and "
        "any(any(IsSubClassOrRole.holds(sub_class=t, super_class=MeatTopping) for t in value.types) "
        "for value in candidate.has_topping))"
    ) in axiom_python
    assert (
        "not_(and_(HasProperty(instance=candidate_var, property_descriptor_type=HasTopping)"
        in generated.method_source(class_name="VegetarianPizza", method_name="axiom")
    )


def test_a_complement_of_a_named_class_negates_its_type_check(generated):
    """
    The complement of a named class negates its type check.
    """
    axiom_python = generated.method_source(
        class_name="NonVegetarianPizza", method_name="axiom_python"
    )
    assert (
        "not (any(IsSubClassOrRole.holds(sub_class=t, super_class=VegetarianPizza) for t in candidate.types))"
        in axiom_python
    )


def test_a_class_defined_by_a_complement_is_classified_by_the_negated_property(
    generated_module,
):
    """
    The property of a negated restriction is a property the class is classified by.
    """
    assert generated_module.HasTopping in HasProperty.descriptor_types_in_axiom_of(
        cls=generated_module.VegetarianPizza
    )


# %% enumerations


def test_an_enumeration_classifies_by_the_individuals_it_lists(generated):
    """
    An enumeration classifies by the IRIs of the individuals it lists.
    """
    iris = (f"{PIZZA_NAMESPACE}America", f"{PIZZA_NAMESPACE}Italy")
    assert f"(str(candidate.uri) in {iris!r})" in generated.method_source(
        class_name="Country", method_name="axiom_python"
    )
    assert f"in_(candidate_var.uri, {iris!r})" in generated.method_source(
        class_name="Country", method_name="axiom"
    )


# %% unqualified cardinalities


def test_an_unqualified_minimum_cardinality_counts_every_value(generated):
    """
    An unqualified minimum cardinality counts every value of the property.
    """
    assert "(len(candidate.has_topping) >= 3)" in generated.method_source(
        class_name="InterestingPizza", method_name="axiom_python"
    )


# %% anonymous intersections as fillers


def test_an_anonymous_intersection_as_filler_is_a_class_classified_by_its_conjuncts(
    generated,
):
    """
    An anonymous intersection a restriction restricts to is a class of its own, classified by its conjuncts.
    """
    filler_name = "CheeseToppingANDhasSpicinessSOMEHot"
    assert (
        f"IsSubClassOrRole.holds(sub_class=t, super_class={filler_name})"
        in generated.method_source(
            class_name="HotCheesePizza", method_name="axiom_python"
        )
    )
    assert re.match(
        rf"class {filler_name}\(CheeseTopping\)",
        generated.class_source(class_name=filler_name),
    )
    assert (
        "IsSubClassOrRole.holds(sub_class=t, super_class=Hot)"
        in generated.method_source(class_name=filler_name, method_name="axiom_python")
    )


def test_an_anonymous_intersection_that_defines_a_named_class_as_filler_is_that_class(
    generated,
):
    """
    An anonymous intersection with the members of a named class's definition stands for that class.
    """
    assert (
        "IsSubClassOrRole.holds(sub_class=t, super_class=SpicyTopping)"
        in generated.method_source(class_name="SpicyPizza", method_name="axiom_python")
    )
    assert re.findall(
        r"^class \w*ANDhasSpicinessSOMEHot\b", generated.classes_source, re.M
    ) == ["class CheeseToppingANDhasSpicinessSOMEHot"]
    assert (
        generated.method_source(
            class_name="SpicyTopping", method_name="axiom_python"
        ).count("IsSubClassOrRole.holds(sub_class=t, super_class=Hot)")
        == 1
    )


# %% values


def test_the_value_of_a_restriction_is_typed_by_its_class_not_by_owl_thing(
    generated,
):
    """
    The individual of a value restriction is typed by its class, not by ``owl:Thing``.
    """
    assert "Set[Thing]" not in generated.classes_source
    assert re.search(
        r"has_country_of_origin: Set\[Country\]",
        generated.class_source(class_name="American"),
    )


# %% properties restricted to several classes


def test_a_property_restricted_to_several_classes_is_declared_once(generated):
    """
    A property a class restricts to several classes is declared on the class at most once.
    """
    assert (
        len(
            re.findall(
                r"^    has_topping: ",
                generated.class_source(class_name="American"),
                re.M,
            )
        )
        <= 1
    )
    assert (
        len(re.findall(r"^American\.has_topping = ", generated.classes_source, re.M))
        <= 1
    )


# %% the generated code


def test_every_generated_file_compiles(generated):
    """
    Every generated file compiles.
    """
    for file_name, source in generated.files.items():
        compile(source, file_name, "exec")


def test_the_generated_module_imports(generated_module):
    """
    The generated module imports, with the class hierarchy of the ontology.
    """
    assert issubclass(generated_module.CheesyPizza, generated_module.Pizza)
