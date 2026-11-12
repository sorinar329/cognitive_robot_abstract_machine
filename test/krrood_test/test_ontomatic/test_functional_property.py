"""
A functional object property holds a single value, so the classification axiom of a
``owl:hasValue`` restriction on it compares that value; a property that is not
functional holds a set of values (``dataset/ontologies/functional_has_value.ttl``).
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
        ontology_file_name="functional_has_value.ttl"
    )


@pytest.fixture(scope="module")
def smith_iri() -> str:
    """
    The IRI of the individual both restrictions name.
    """
    return "http://example.org/functional#drSmith"


# %% single and several values


def test_has_value_on_a_functional_property_compares_the_single_value(
    generated, smith_iri
):
    """
    The attribute of a functional property holds one individual, whose IRI is
    compared.
    """
    axiom_python = generated.method_source(
        class_name="AdvisedBySmith", method_name="axiom_python"
    )
    assert f"(str(candidate.has_advisor.uri) == '{smith_iri}')" in axiom_python


def test_has_value_on_a_property_that_is_not_functional_looks_in_the_set_of_values(
    generated, smith_iri
):
    """
    The attribute of a property that is not functional holds a set of individuals.
    """
    axiom_python = generated.method_source(
        class_name="ReadBySmith", method_name="axiom_python"
    )
    assert (
        f"('{smith_iri}' in (str(value.uri) for value in candidate.has_reader))"
        in axiom_python
    )
