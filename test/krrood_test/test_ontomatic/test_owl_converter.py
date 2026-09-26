from __future__ import annotations

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from typing_extensions import Set

from krrood.ontomatic.failures import DuplicateOWLClassName
from krrood.ontomatic.owl_converter import OWLConverter
from ..dataset.owl_conversion.converted_hierarchy import (
    Cabinet,
    ConvertedRoot,
    Drawer,
    DrawerWithSpecializedHandle,
    Handle,
    ReferencedOnlyClass,
    SpecializedHandle,
)
from ..dataset.owl_conversion.duplicate_class_name import SecondRoot

ONTOLOGY_IRI = URIRef("http://example.org/converted_hierarchy")


@pytest.fixture
def converter() -> OWLConverter:
    return OWLConverter(root_classes=[ConvertedRoot], ontology_iri=ONTOLOGY_IRI)


@pytest.fixture
def graph(converter: OWLConverter) -> Graph:
    return converter.convert()


def restriction_values(
    graph: Graph, owl_class: URIRef, property_iri: URIRef, restriction_kind: URIRef
) -> Set:
    """
    :return: The values of the restrictions of *restriction_kind* on *property_iri* that *owl_class* is directly
        declared a subclass of.
    """
    return {
        graph.value(restriction, restriction_kind)
        for restriction in graph.objects(owl_class, RDFS.subClassOf)
        if graph.value(restriction, OWL.onProperty) == property_iri
        and graph.value(restriction, restriction_kind) is not None
    }


# %% ontology


class TestOntology:

    def test_ontology_is_declared(self, graph: Graph):
        assert (ONTOLOGY_IRI, RDF.type, OWL.Ontology) in graph

    def test_names_live_in_the_ontology_namespace(self, converter: OWLConverter):
        assert converter.namespace[Handle.__name__] == URIRef(
            f"{ONTOLOGY_IRI}#{Handle.__name__}"
        )


# %% classes


class TestClasses:

    def test_hierarchy_and_referenced_classes_are_the_only_owl_classes(
        self, graph: Graph, converter: OWLConverter
    ):
        expected_classes = {
            ConvertedRoot,
            Handle,
            SpecializedHandle,
            Drawer,
            DrawerWithSpecializedHandle,
            Cabinet,
            ReferencedOnlyClass,
        }
        assert set(graph.subjects(RDF.type, OWL.Class)) == {
            converter.namespace[python_class.__name__]
            for python_class in expected_classes
        }

    def test_subclass_is_declared_subclass_of_its_superclass_only(
        self, graph: Graph, converter: OWLConverter
    ):
        named_superclasses = {
            superclass
            for superclass in graph.objects(
                converter.namespace[SpecializedHandle.__name__], RDFS.subClassOf
            )
            if isinstance(superclass, URIRef)
        }
        assert named_superclasses == {converter.namespace[Handle.__name__]}

    def test_duplicate_class_names_raise(self):
        converter = OWLConverter(
            root_classes=[ConvertedRoot, SecondRoot], ontology_iri=ONTOLOGY_IRI
        )
        with pytest.raises(DuplicateOWLClassName):
            converter.convert()


# %% relations


class TestRelations:

    def test_relations_are_the_only_object_properties(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.subjects(RDF.type, OWL.ObjectProperty)) == {
            converter.namespace["body"],
            converter.namespace["handle"],
            converter.namespace["front"],
            converter.namespace["drawers"],
        }

    def test_relation_is_restricted_to_the_field_type(
        self, graph: Graph, converter: OWLConverter
    ):
        assert restriction_values(
            graph,
            converter.namespace[Drawer.__name__],
            converter.namespace["handle"],
            OWL.allValuesFrom,
        ) == {converter.namespace[Handle.__name__]}

    def test_relation_to_referenced_class_is_restricted_to_it(
        self, graph: Graph, converter: OWLConverter
    ):
        assert restriction_values(
            graph,
            converter.namespace[Handle.__name__],
            converter.namespace["body"],
            OWL.allValuesFrom,
        ) == {converter.namespace[ReferencedOnlyClass.__name__]}

    def test_inherited_relation_is_not_restricted_again(
        self, graph: Graph, converter: OWLConverter
    ):
        assert (
            restriction_values(
                graph,
                converter.namespace[SpecializedHandle.__name__],
                converter.namespace["body"],
                OWL.allValuesFrom,
            )
            == set()
        )

    def test_narrowed_relation_is_restricted_on_the_subclass(
        self, graph: Graph, converter: OWLConverter
    ):
        assert restriction_values(
            graph,
            converter.namespace[DrawerWithSpecializedHandle.__name__],
            converter.namespace["handle"],
            OWL.allValuesFrom,
        ) == {converter.namespace[SpecializedHandle.__name__]}

    def test_optional_relation_has_at_most_one_value(
        self, graph: Graph, converter: OWLConverter
    ):
        assert restriction_values(
            graph,
            converter.namespace[Drawer.__name__],
            converter.namespace["front"],
            OWL.maxCardinality,
        ) == {
            Literal(
                OWLConverter.single_valued_field_cardinality,
                datatype=XSD.nonNegativeInteger,
            )
        }

    def test_many_valued_relation_has_no_cardinality(
        self, graph: Graph, converter: OWLConverter
    ):
        assert (
            restriction_values(
                graph,
                converter.namespace[Cabinet.__name__],
                converter.namespace["drawers"],
                OWL.maxCardinality,
            )
            == set()
        )

    def test_many_valued_relation_is_restricted_to_the_contained_type(
        self, graph: Graph, converter: OWLConverter
    ):
        assert restriction_values(
            graph,
            converter.namespace[Cabinet.__name__],
            converter.namespace["drawers"],
            OWL.allValuesFrom,
        ) == {converter.namespace[Drawer.__name__]}
