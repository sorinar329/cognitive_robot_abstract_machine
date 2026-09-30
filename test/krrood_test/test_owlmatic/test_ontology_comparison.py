from __future__ import annotations

import pytest
from rdflib import BNode, Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from krrood.owlmatic.ontology_comparison import (
    Axiom,
    AxiomKind,
    OntologyAxioms,
    OntologyComparison,
)

REFERENCE = Namespace("http://example.org/reference#")
CANDIDATE = Namespace("http://example.org/candidate/")


def graph_of(*triples) -> Graph:
    """
    :return: A graph holding *triples*.
    """
    graph = Graph()
    for triple in triples:
        graph.add(triple)
    return graph


# %% axioms


class TestAxioms:

    def test_term_of_an_iri_is_its_local_name(self):
        assert Axiom.term_of(REFERENCE.Drawer) == "Drawer"

    def test_term_of_an_iri_without_fragment_is_its_last_path_segment(self):
        assert Axiom.term_of(CANDIDATE.Drawer) == "Drawer"

    def test_term_of_a_literal_is_its_lexical_form(self):
        assert Axiom.term_of(Literal(1, datatype=XSD.nonNegativeInteger)) == "1"

    def test_declaration_of_a_named_class_is_an_axiom(self):
        graph = graph_of((REFERENCE.Drawer, RDF.type, OWL.Class))
        assert OntologyAxioms(graph).axioms == {
            Axiom(AxiomKind.CLASS_DECLARATION, (Axiom.term_of(REFERENCE.Drawer),))
        }

    def test_anonymous_class_is_not_declared(self):
        graph = graph_of((BNode(), RDF.type, OWL.Class))
        assert OntologyAxioms(graph).axioms == frozenset()

    def test_order_of_a_symmetric_relation_is_ignored(self):
        forwards = graph_of((REFERENCE.member, OWL.inverseOf, REFERENCE.memberOf))
        backwards = graph_of((REFERENCE.memberOf, OWL.inverseOf, REFERENCE.member))
        assert OntologyAxioms(forwards).axioms == OntologyAxioms(backwards).axioms

    def test_order_of_a_directed_relation_is_kept(self):
        forwards = graph_of((REFERENCE.Drawer, RDFS.subClassOf, REFERENCE.Furniture))
        backwards = graph_of((REFERENCE.Furniture, RDFS.subClassOf, REFERENCE.Drawer))
        assert OntologyAxioms(forwards).axioms != OntologyAxioms(backwards).axioms

    def test_property_characteristic_is_an_axiom(self):
        graph = graph_of((REFERENCE.partOf, RDF.type, OWL.TransitiveProperty))
        assert OntologyAxioms(graph).axioms == {
            Axiom(
                AxiomKind.PROPERTY_CHARACTERISTIC,
                (
                    Axiom.term_of(REFERENCE.partOf),
                    Axiom.term_of(OWL.TransitiveProperty),
                ),
            )
        }

    def test_restriction_a_class_is_declared_subclass_of_is_an_axiom(self):
        restriction = BNode()
        graph = graph_of(
            (restriction, RDF.type, OWL.Restriction),
            (restriction, OWL.onProperty, REFERENCE.handle),
            (restriction, OWL.allValuesFrom, REFERENCE.Handle),
            (REFERENCE.Drawer, RDFS.subClassOf, restriction),
        )
        assert OntologyAxioms(graph).axioms == {
            Axiom(
                AxiomKind.RESTRICTION,
                (
                    Axiom.term_of(REFERENCE.Drawer),
                    Axiom.term_of(REFERENCE.handle),
                    Axiom.term_of(OWL.allValuesFrom),
                    Axiom.term_of(REFERENCE.Handle),
                ),
            )
        }

    def test_qualified_cardinality_names_the_counted_class(self):
        restriction = BNode()
        cardinality = Literal(2, datatype=XSD.nonNegativeInteger)
        graph = graph_of(
            (restriction, OWL.onProperty, REFERENCE.handle),
            (restriction, OWL.maxQualifiedCardinality, cardinality),
            (restriction, OWL.onClass, REFERENCE.Handle),
            (REFERENCE.Drawer, RDFS.subClassOf, restriction),
        )
        assert OntologyAxioms(graph).axioms == {
            Axiom(
                AxiomKind.RESTRICTION,
                (
                    Axiom.term_of(REFERENCE.Drawer),
                    Axiom.term_of(REFERENCE.handle),
                    Axiom.term_of(OWL.maxQualifiedCardinality),
                    Axiom.term_of(cardinality),
                    Axiom.term_of(REFERENCE.Handle),
                ),
            )
        }


# %% comparison


class TestComparison:

    @pytest.fixture
    def comparison(self) -> OntologyComparison:
        reference = graph_of(
            (REFERENCE.Drawer, RDF.type, OWL.Class),
            (REFERENCE.Drawer, RDFS.subClassOf, REFERENCE.Furniture),
        )
        candidate = graph_of(
            (CANDIDATE.Drawer, RDF.type, OWL.Class),
            (CANDIDATE.handle, RDF.type, OWL.ObjectProperty),
        )
        return OntologyComparison(reference, candidate)

    def test_terms_named_alike_in_different_namespaces_agree(
        self, comparison: OntologyComparison
    ):
        assert comparison.agreement.shared == {
            Axiom(AxiomKind.CLASS_DECLARATION, (Axiom.term_of(REFERENCE.Drawer),))
        }

    def test_axiom_only_the_reference_states_is_missing(
        self, comparison: OntologyComparison
    ):
        assert comparison.agreement.missing == {
            Axiom(
                AxiomKind.SUBCLASS_OF,
                (
                    Axiom.term_of(REFERENCE.Drawer),
                    Axiom.term_of(REFERENCE.Furniture),
                ),
            )
        }

    def test_axiom_only_the_candidate_states_is_extra(
        self, comparison: OntologyComparison
    ):
        assert comparison.agreement.extra == {
            Axiom(
                AxiomKind.OBJECT_PROPERTY_DECLARATION,
                (Axiom.term_of(CANDIDATE.handle),),
            )
        }

    def test_recall_is_the_share_of_expected_axioms_stated(
        self, comparison: OntologyComparison
    ):
        assert comparison.agreement.recall == 1 / 2

    def test_precision_is_the_share_of_stated_axioms_expected(
        self, comparison: OntologyComparison
    ):
        assert comparison.agreement.precision == 1 / 2

    def test_kind_neither_ontology_states_has_no_agreement(
        self, comparison: OntologyComparison
    ):
        assert set(comparison.agreement_by_kind) == {
            AxiomKind.CLASS_DECLARATION,
            AxiomKind.SUBCLASS_OF,
            AxiomKind.OBJECT_PROPERTY_DECLARATION,
        }

    def test_kind_the_reference_does_not_state_is_fully_recalled(
        self, comparison: OntologyComparison
    ):
        assert (
            comparison.agreement_by_kind[AxiomKind.OBJECT_PROPERTY_DECLARATION].recall
            == 1.0
        )

    def test_kind_the_candidate_does_not_state_is_fully_precise(
        self, comparison: OntologyComparison
    ):
        assert comparison.agreement_by_kind[AxiomKind.SUBCLASS_OF].precision == 1.0

    def test_report_lists_every_missing_axiom(self, comparison: OntologyComparison):
        report = comparison.report()
        assert all(str(axiom) in report for axiom in comparison.agreement.missing)
