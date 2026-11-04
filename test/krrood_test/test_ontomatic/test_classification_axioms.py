"""
Only sufficient conditions of a class may become classification axioms
(``axiom_python``) in the generated code.

The ontology (``dataset/ontologies/classification_axioms.ttl``) is a reduced version of the OWL2Bench OWL 2 RL
ontology:

* ``LeisureStudent SubClassOf (Student and takesCourse max 1 Course)`` and
  ``WomanCollege SubClassOf (College and hasStudent only (not Man))`` are necessary conditions.
* ``T20CricketFan EquivalentTo isCrazyAbout value T20Cricket`` is a definition (sufficient).
* ``Student and (hasMajor some Science) SubClassOf ScienceStudent`` is a general class axiom (sufficient), whose named
  conjunct ``Student`` must be kept.
* ``Person and (worksFor some Organization) SubClassOf Employee`` with ``hasEmployee`` the inverse of ``worksFor`` does
  not entail ``hasEmployee some Employee SubClassOf Organization``.
"""

from __future__ import annotations

import pytest

from .generated_ontology import GeneratedOntology


@pytest.fixture(scope="module")
def generated() -> GeneratedOntology:
    """
    The files ontomatic generates from the ontology.
    """
    return GeneratedOntology.of("classification_axioms.ttl")


# %% necessary and sufficient conditions


def test_necessary_conditions_are_not_classification_axioms(generated):
    """
    Necessary conditions become no classification axioms.
    """
    for class_name in ["LeisureStudent", "WomanCollege"]:
        source = generated.class_source(class_name)
        assert "def axiom_python" not in source
        assert "def axiom(" not in source
        assert "def necessary_conditions_python" in source


def test_equivalent_class_definition_is_a_classification_axiom(generated):
    """
    A definition is a classification axiom.
    """
    source = generated.class_source("T20CricketFan")
    assert "def axiom_python" in source
    assert "http://benchmark/OWL2Bench#T20Cricket" in source


def test_general_class_axiom_keeps_named_conjunct(generated):
    """
    A general class axiom classifies by its named conjunct too.
    """
    axiom_python = generated.method_source("ScienceStudent", "axiom_python")
    assert "IsSubClassOrRole.holds(t, Student) for t in candidate.types" in axiom_python
    assert "IsSubClassOrRole.holds(t, Science)" in axiom_python


def test_inverse_of_a_general_class_axiom_is_not_a_classification_axiom(generated):
    """
    The inverse of a general class axiom is not entailed, so it is no classification
    axiom.
    """
    assert "def axiom_python" in generated.class_source("Employee")
    assert "def axiom_python" not in generated.class_source("Organization")
