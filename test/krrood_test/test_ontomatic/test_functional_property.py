"""
A functional object property holds a single value, so the classification axiom of a ``owl:hasValue``
restriction on it compares that value; a property that is not functional holds a set of values
(``dataset/ontologies/functional_has_value.ttl``).
"""

from __future__ import annotations

import pytest

from .generated_ontology import GeneratedOntology

SMITH_IRI = "http://example.org/functional#drSmith"


@pytest.fixture(scope="module")
def generated() -> GeneratedOntology:
    """
    The files ontomatic generates from the ontology.
    """
    return GeneratedOntology.of("functional_has_value.ttl")


# %% single and several values


def test_has_value_on_a_functional_property_compares_the_single_value(generated):
    """
    The attribute of a functional property holds one individual, whose IRI is compared.
    """
    axiom_python = generated.method_source("AdvisedBySmith", "axiom_python")
    assert f"(str(candidate.has_advisor.uri) == '{SMITH_IRI}')" in axiom_python


def test_has_value_on_a_property_that_is_not_functional_looks_in_the_set_of_values(
    generated,
):
    """
    The attribute of a property that is not functional holds a set of individuals.
    """
    axiom_python = generated.method_source("ReadBySmith", "axiom_python")
    assert (
        f"('{SMITH_IRI}' in (str(value.uri) for value in candidate.has_reader))"
        in axiom_python
    )
