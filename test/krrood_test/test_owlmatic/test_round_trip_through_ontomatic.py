from __future__ import annotations

import os

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, XSD

from krrood.owlmatic.ontology_comparison import Axiom, AxiomKind, OntologyComparison
from krrood.owlmatic.owl_converter import OWLConverter
from ..dataset.owl_conversion import ontomatic_generated_university
from ..dataset.owl_conversion.ontomatic_generated_university import (
    Person,
    Student,
    UniversityThing,
)

ONTOLOGY_PATH = os.path.join(
    os.path.dirname(ontomatic_generated_university.__file__), "university.ttl"
)


@pytest.fixture(scope="module")
def reference() -> Graph:
    return Graph().parse(ONTOLOGY_PATH)


@pytest.fixture(scope="module")
def comparison(reference: Graph) -> OntologyComparison:
    ontology_iri = next(reference.subjects(predicate=None, object=OWL.Ontology))
    converter = OWLConverter(
        root_classes=[UniversityThing],
        ontology_iri=URIRef(ontology_iri),
        thing_class=UniversityThing,
    )
    return OntologyComparison(reference, converter.convert())


def test_only_the_role_is_lost(comparison: OntologyComparison):
    """
    Ontomatic generates no role taker for the role the ontology restricts on
    ``roleFor``, so the round trip cannot restore the restriction.
    """
    assert comparison.agreement.missing == {
        Axiom(
            AxiomKind.RESTRICTION,
            (
                Student.__name__,
                OWLConverter.role_property_name,
                Axiom.term_of(OWL.someValuesFrom),
                Person.__name__,
            ),
        )
    }


def test_only_what_ontomatic_adds_is_extra(comparison: OntologyComparison):
    """
    Ontomatic adds a ``uri`` field to its base class, and turns every datatype property
    into a field holding one optional value.
    """
    uri = OWLConverter.property_name_of_field("uri")
    name = OWLConverter.property_name_of_field("name")
    assert comparison.agreement.extra == {
        Axiom(AxiomKind.DATATYPE_PROPERTY_DECLARATION, (uri,)),
        Axiom(AxiomKind.RANGE, (uri, Axiom.term_of(XSD.string))),
        Axiom(
            AxiomKind.RESTRICTION,
            (
                Person.__name__,
                name,
                Axiom.term_of(OWL.maxCardinality),
                Axiom.term_of(
                    Literal(
                        OWLConverter.single_valued_field_cardinality,
                        datatype=XSD.nonNegativeInteger,
                    )
                ),
            ),
        ),
    }
