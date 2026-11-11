"""
Only sufficient conditions of a class, such as definitions and general class axioms,
become classification axioms (``axiom_python``); necessary conditions and the inverses
of general class axioms do not (``dataset/ontologies/classification_axioms.ttl``, a
reduced OWL2Bench OWL 2 RL ontology).
"""

from __future__ import annotations

import pytest

from .generated_ontology import GeneratedOntology


@pytest.fixture(scope="module")
def generated() -> GeneratedOntology:
    """
    The files ontomatic generates from the ontology.
    """
    return GeneratedOntology.generate_from_file(
        ontology_file_name="classification_axioms.ttl"
    )


# %% necessary and sufficient conditions


def test_necessary_conditions_are_not_classification_axioms(generated):
    """
    Necessary conditions become no classification axioms.
    """
    for class_name in ["LeisureStudent", "WomanCollege"]:
        source = generated.class_source(class_name=class_name)
        assert "def axiom_python" not in source
        assert "def axiom(" not in source
        assert "def necessary_conditions_python" in source


def test_equivalent_class_definition_is_a_classification_axiom(generated):
    """
    A definition is a classification axiom.
    """
    source = generated.class_source(class_name="T20CricketFan")
    assert "def axiom_python" in source
    assert "http://benchmark/OWL2Bench#T20Cricket" in source


def test_general_class_axiom_keeps_named_conjunct(generated):
    """
    A general class axiom classifies by its named conjunct too.
    """
    axiom_python = generated.method_source(
        class_name="ScienceStudent", method_name="axiom_python"
    )
    assert (
        "IsSubClassOrRole.holds(sub_class=t, super_class=Student) for t in candidate.types"
        in axiom_python
    )
    assert "IsSubClassOrRole.holds(sub_class=t, super_class=Science)" in axiom_python


def test_inverse_of_a_general_class_axiom_is_not_a_classification_axiom(generated):
    """
    The inverse of a general class axiom is not entailed, so it is no classification
    axiom.
    """
    assert "def axiom_python" in generated.class_source(class_name="Employee")
    assert "def axiom_python" not in generated.class_source(class_name="Organization")
