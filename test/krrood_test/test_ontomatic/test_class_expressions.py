"""
Every OWL class expression of the pizza ontology (owlcs/pizza-ontology) is translated into the generated code, not
only those attached to a named class through ``rdfs:subClassOf``:

* ``CheesyPizza EquivalentTo Pizza and (hasTopping some CheeseTopping)`` is a definition whose named conjunct is a
  superclass.
* ``Margherita SubClassOf hasTopping only (MozzarellaTopping or TomatoTopping)`` is a closure axiom over a union.
* ``VegetarianPizza EquivalentTo Pizza and not (hasTopping some MeatTopping)`` and
  ``NonVegetarianPizza EquivalentTo Pizza and not VegetarianPizza`` define classes by complements.
* ``Country EquivalentTo {America, Italy} and DomainConcept`` is an enumeration.
* ``InterestingPizza EquivalentTo Pizza and (hasTopping min 3)`` is an unqualified cardinality.
* ``HotCheesePizza EquivalentTo Pizza and (hasTopping some (CheeseTopping and (hasSpiciness some Hot)))`` restricts to
  an anonymous intersection, and ``SpicyPizza EquivalentTo Pizza and (hasTopping some (PizzaTopping and
  (hasSpiciness some Hot)))`` to one that is the definition of ``SpicyTopping``.
* ``American SubClassOf hasCountryOfOrigin value America`` names an individual that is also typed ``owl:Thing``, and
  ``American SubClassOf hasTopping some MozzarellaTopping, hasTopping some TomatoTopping,
  hasTopping only (MozzarellaTopping or TomatoTopping)`` restricts one property to several classes.
"""

from __future__ import annotations

import importlib
import re

import pytest

from krrood.ontomatic.ontology_to_python.owl_to_python import OwlToPythonConverter
from krrood.ontomatic.property_descriptor.property_descriptor import (
    class_axiomatized_properties,
)

ONTOLOGY = """
@prefix : <http://www.co-ode.org/ontologies/pizza/pizza.owl#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://www.co-ode.org/ontologies/pizza> a owl:Ontology .

:hasTopping a owl:ObjectProperty ; rdfs:domain :Pizza ; rdfs:range :PizzaTopping .
:isToppingOf a owl:ObjectProperty , owl:FunctionalProperty ; owl:inverseOf :hasTopping ;
    rdfs:domain :PizzaTopping ; rdfs:range :Pizza .
:hasSpiciness a owl:ObjectProperty ; rdfs:range :Spiciness .
:hasCountryOfOrigin a owl:ObjectProperty .

:DomainConcept a owl:Class .
:Food a owl:Class ; rdfs:subClassOf :DomainConcept .
:Pizza a owl:Class ; rdfs:subClassOf :Food .
:NamedPizza a owl:Class ; rdfs:subClassOf :Pizza .
:PizzaTopping a owl:Class ; rdfs:subClassOf :Food .
:CheeseTopping a owl:Class ; rdfs:subClassOf :PizzaTopping .
:MozzarellaTopping a owl:Class ; rdfs:subClassOf :CheeseTopping .
:TomatoTopping a owl:Class ; rdfs:subClassOf :PizzaTopping .
:MeatTopping a owl:Class ; rdfs:subClassOf :PizzaTopping .
:Spiciness a owl:Class ; rdfs:subClassOf :DomainConcept .
:Hot a owl:Class ; rdfs:subClassOf :Spiciness .

:CheesyPizza a owl:Class ;
    owl:equivalentClass [ a owl:Class ;
        owl:intersectionOf ( :Pizza
            [ a owl:Restriction ; owl:onProperty :hasTopping ; owl:someValuesFrom :CheeseTopping ] ) ] .

:Margherita a owl:Class ;
    rdfs:subClassOf :NamedPizza ;
    rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :hasTopping ;
        owl:allValuesFrom [ a owl:Class ; owl:unionOf ( :MozzarellaTopping :TomatoTopping ) ] ] .

:VegetarianPizza a owl:Class ;
    owl:equivalentClass [ a owl:Class ;
        owl:intersectionOf ( :Pizza
            [ a owl:Class ; owl:complementOf
                [ a owl:Restriction ; owl:onProperty :hasTopping ; owl:someValuesFrom :MeatTopping ] ] ) ] .

:NonVegetarianPizza a owl:Class ;
    owl:equivalentClass [ a owl:Class ;
        owl:intersectionOf ( :Pizza [ a owl:Class ; owl:complementOf :VegetarianPizza ] ) ] .

:Country a owl:Class ;
    owl:equivalentClass [ a owl:Class ;
        owl:intersectionOf ( [ a owl:Class ; owl:oneOf ( :America :Italy ) ] :DomainConcept ) ] .
:America a owl:NamedIndividual , owl:Thing , :Country .
:Italy a owl:NamedIndividual , :Country .

:InterestingPizza a owl:Class ;
    owl:equivalentClass [ a owl:Class ;
        owl:intersectionOf ( [ a owl:Restriction ; owl:onProperty :hasTopping ;
                               owl:minCardinality "3"^^xsd:nonNegativeInteger ] :Pizza ) ] .

:HotCheesePizza a owl:Class ;
    owl:equivalentClass [ a owl:Class ;
        owl:intersectionOf ( :Pizza
            [ a owl:Restriction ; owl:onProperty :hasTopping ;
              owl:someValuesFrom [ a owl:Class ;
                  owl:intersectionOf ( :CheeseTopping
                      [ a owl:Restriction ; owl:onProperty :hasSpiciness ; owl:someValuesFrom :Hot ] ) ] ] ) ] .

:SpicyTopping a owl:Class ;
    owl:equivalentClass [ a owl:Class ;
        owl:intersectionOf ( :PizzaTopping
            [ a owl:Restriction ; owl:onProperty :hasSpiciness ; owl:someValuesFrom :Hot ] ) ] .

:SpicyPizza a owl:Class ;
    owl:equivalentClass [ a owl:Class ;
        owl:intersectionOf ( :Pizza
            [ a owl:Restriction ; owl:onProperty :hasTopping ;
              owl:someValuesFrom [ a owl:Class ;
                  owl:intersectionOf ( :PizzaTopping
                      [ a owl:Restriction ; owl:onProperty :hasSpiciness ; owl:someValuesFrom :Hot ] ) ] ] ) ] .

:American a owl:Class ;
    rdfs:subClassOf :NamedPizza ;
    rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :hasCountryOfOrigin ; owl:hasValue :America ] ;
    rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :hasTopping ; owl:someValuesFrom :MozzarellaTopping ] ;
    rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :hasTopping ; owl:someValuesFrom :TomatoTopping ] ;
    rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :hasTopping ;
        owl:allValuesFrom [ a owl:Class ; owl:unionOf ( :MozzarellaTopping :TomatoTopping ) ] ] .
"""

PIZZA_NAMESPACE = "http://www.co-ode.org/ontologies/pizza/pizza.owl#"
"""
The namespace of the ontology's classes and individuals.
"""


@pytest.fixture(scope="module")
def generated_files(tmp_path_factory) -> dict[str, str]:
    """
    The Python files ontomatic generates from :data:`ONTOLOGY`, by their name.
    """
    path = tmp_path_factory.mktemp("ontology") / "pizza.ttl"
    path.write_text(ONTOLOGY)
    converter = OwlToPythonConverter()
    converter.load_ontology(str(path))
    return converter.generate_python_code_external("model")


@pytest.fixture(scope="module")
def generated_classes(generated_files) -> str:
    """
    The generated module holding the classes.
    """
    return generated_files["model.py"]


def class_body(source: str, class_name: str) -> str:
    """
    :return: The generated source of the class named ``class_name``, from its ``class`` line to the next class.
    """
    match = re.search(
        rf"^class {class_name}\(.*?(?=^@dataclass|^class |\Z)", source, re.S | re.M
    )
    assert match, f"class {class_name} not generated"
    return match.group(0)


def method_body(class_source: str, method_name: str) -> str:
    """
    :return: The source of the method ``method_name`` of a class, up to the next method.
    """
    start = class_source.index(f"def {method_name}(")
    following = class_source.find("@classmethod", start)
    return class_source[start : following if following != -1 else None]


# %% definitions


def test_a_definition_makes_its_named_conjunct_a_superclass(generated_classes):
    assert re.search(r"^class CheesyPizza\(.*\bPizza\b", generated_classes, re.M)


def test_a_definition_classifies_by_its_named_conjunct_and_its_restriction(
    generated_classes,
):
    axiom_python = method_body(
        class_body(generated_classes, "CheesyPizza"), "axiom_python"
    )
    assert "issubclass_or_role(t, Pizza) for t in candidate.types" in axiom_python
    assert "CheeseTopping" in axiom_python


# %% unions as fillers


def test_a_closure_over_a_union_checks_against_its_members(generated_classes):
    necessary = method_body(
        class_body(generated_classes, "Margherita"), "necessary_conditions_python"
    )
    assert "(MozzarellaTopping, TomatoTopping)" in necessary


def test_no_class_is_named_after_a_union(generated_classes):
    assert re.findall(r"\b\w*ORTomatoTopping\w*\b", generated_classes) == []


# %% complements


def test_a_complement_of_a_restriction_negates_it(generated_classes):
    body = class_body(generated_classes, "VegetarianPizza")
    axiom_python = method_body(body, "axiom_python")
    assert "not (" in axiom_python
    assert "MeatTopping" in axiom_python
    assert "not_(" in method_body(body, "axiom")


def test_a_complement_of_a_named_class_negates_its_type_check(generated_classes):
    axiom_python = method_body(
        class_body(generated_classes, "NonVegetarianPizza"), "axiom_python"
    )
    assert (
        "not (any(issubclass_or_role(t, VegetarianPizza) for t in candidate.types))"
        in axiom_python
    )


# %% enumerations


def test_an_enumeration_classifies_by_the_individuals_it_lists(generated_classes):
    body = class_body(generated_classes, "Country")
    axiom_python = method_body(body, "axiom_python")
    for individual in ["America", "Italy"]:
        assert f"'{PIZZA_NAMESPACE}{individual}'" in axiom_python
    assert "in_(" in method_body(body, "axiom")


# %% unqualified cardinalities


def test_an_unqualified_minimum_cardinality_counts_every_value(generated_classes):
    axiom_python = method_body(
        class_body(generated_classes, "InterestingPizza"), "axiom_python"
    )
    assert "(len(candidate.has_topping) >= 3)" in axiom_python


# %% anonymous intersections as fillers


def test_an_anonymous_intersection_as_filler_is_a_class_classified_by_its_conjuncts(
    generated_classes,
):
    filler_name = "CheeseToppingANDhasSpicinessSOMEHot"
    assert filler_name in method_body(
        class_body(generated_classes, "HotCheesePizza"), "axiom_python"
    )
    filler = class_body(generated_classes, filler_name)
    assert re.match(rf"class {filler_name}\(CheeseTopping\)", filler)
    axiom_python = method_body(filler, "axiom_python")
    assert "issubclass_or_role(t, Hot)" in axiom_python


def test_an_anonymous_intersection_that_defines_a_named_class_as_filler_is_that_class(
    generated_classes,
):
    assert "issubclass_or_role(t, SpicyTopping)" in method_body(
        class_body(generated_classes, "SpicyPizza"), "axiom_python"
    )
    assert re.findall(
        r"^class \w*ANDhasSpicinessSOMEHot\b", generated_classes, re.M
    ) == ["class CheeseToppingANDhasSpicinessSOMEHot"]
    spicy_topping = method_body(
        class_body(generated_classes, "SpicyTopping"), "axiom_python"
    )
    assert spicy_topping.count("issubclass_or_role(t, Hot)") == 1


# %% values


def test_the_value_of_a_restriction_is_typed_by_its_class_not_by_owl_thing(
    generated_classes,
):
    assert "Set[Thing]" not in generated_classes
    assert re.search(
        r"has_country_of_origin: Set\[Country\]",
        class_body(generated_classes, "American"),
    )


# %% properties restricted to several classes


def test_a_property_restricted_to_several_classes_is_declared_once(generated_classes):
    assert (
        len(
            re.findall(
                r"^    has_topping: ", class_body(generated_classes, "American"), re.M
            )
        )
        <= 1
    )
    assert len(re.findall(r"^American\.has_topping = ", generated_classes, re.M)) <= 1


# %% the generated code


def test_every_generated_file_compiles(generated_files):
    for file_name, source in generated_files.items():
        compile(source, file_name, "exec")


@pytest.fixture(scope="module")
def generated_model(generated_files, tmp_path_factory):
    """
    The generated module holding the classes, imported from a package of the generated files.
    """
    root = tmp_path_factory.mktemp("generated")
    package = root / "generated_pizza"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for file_name, source in generated_files.items():
        (package / file_name).write_text(source)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(root))
        yield importlib.import_module("generated_pizza.model")


def test_the_generated_package_imports(generated_model):
    assert issubclass(generated_model.CheesyPizza, generated_model.Pizza)


def test_a_class_defined_by_a_complement_is_classified_by_the_negated_property(
    generated_model,
):
    assert generated_model.HasTopping in class_axiomatized_properties(
        generated_model.VegetarianPizza
    )
