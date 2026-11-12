"""
What the inference engine concludes about the classes and properties of an ontology
(``dataset/ontologies/inference.ttl``), before any code is generated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rdflib

from krrood.ontomatic.ontology_to_python.axioms import SomeValuesFromAxiomInfo
from krrood.ontomatic.ontology_to_python.extraction import OntologyExtractor
from krrood.ontomatic.ontology_to_python.inference import InferenceEngine
from krrood.ontomatic.ontology_to_python.ontology_info import OntologyInfo


@pytest.fixture(scope="module")
def ontology() -> OntologyInfo:
    """
    The ontology after the inference steps code generation runs.
    """
    graph = rdflib.Graph()
    graph.parse(
        Path(__file__).parent.parent / "dataset" / "ontologies" / "inference.ttl"
    )
    ontology = OntologyExtractor(graph=graph).extract()
    engine = InferenceEngine(ontology=ontology)
    engine.compute_ancestors()
    engine.infer_properties()
    engine.apply_predefined_overrides()
    engine.compute_type_hints()
    return ontology


# %% classification conditions


def test_a_definition_becomes_the_classification_condition_of_its_class(ontology):
    """
    A class defined as the individuals with a value of a class for a property is
    classified by the conditions of that ``someValuesFrom`` restriction.
    """
    restriction = SomeValuesFromAxiomInfo(
        property_name="takesCourse",
        for_class="Learner",
        ontology=ontology,
        on_class="Course",
    )
    assert ontology.classes["Learner"].axioms_python == restriction.conditions_python()


# %% properties


def test_a_subproperty_inherits_the_domain_of_its_superproperty(ontology):
    """
    A subproperty declaring no domain has the domain of its superproperty.
    """
    assert ontology.properties["takesGraduateCourse"].domains == ["Student"]


def test_a_subproperty_descriptor_derives_from_its_superproperty_descriptor(
    ontology,
):
    """
    The descriptor class of a subproperty derives from the descriptor class of its
    superproperty.
    """
    assert ontology.properties["takesGraduateCourse"].base_descriptors == [
        ontology.properties["takesCourse"].descriptor_name
    ]


def test_an_object_property_is_typed_by_its_declared_range(ontology):
    """
    The field of an object property is typed by the class of its declared range.
    """
    assert ontology.properties["takesGraduateCourse"].object_range_hint == (
        "GraduateCourse"
    )
