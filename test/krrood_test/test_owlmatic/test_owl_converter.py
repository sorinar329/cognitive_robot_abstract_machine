from __future__ import annotations

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from typing_extensions import Set

from krrood.owlmatic.failures import (
    DuplicateOWLClassName,
    PropertyIsBothObjectAndDatatype,
)
from krrood.owlmatic.owl_converter import OWLConverter
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
from ..dataset.owl_conversion.ontomatic_generated_university import (
    Organization,
    UniversityThing,
)
from ..dataset.owl_conversion.property_of_two_kinds import RootOfFieldsNamedAlike
from ..dataset.owl_conversion.undocumented_class import UndocumentedRoot
from ..dataset.role_and_ontology.university_ontology_like_classes import (
    CEOAsFirstRole,
    HeadOf,
    Member,
    MemberOf,
    Person,
    RecognizedGroup,
    RepresentativeAsSecondRole,
    SubOrganizationOf,
    WorksFor,
)

ONTOLOGY_IRI = URIRef("http://example.org/converted_hierarchy")


@pytest.fixture
def converter() -> OWLConverter:
    return OWLConverter(root_classes=[ConvertedRoot], ontology_iri=ONTOLOGY_IRI)


@pytest.fixture
def graph(converter: OWLConverter) -> Graph:
    return converter.convert()


@pytest.fixture
def described_relations_converter() -> OWLConverter:
    return OWLConverter(
        root_classes=[
            Person,
            RecognizedGroup,
            CEOAsFirstRole,
            RepresentativeAsSecondRole,
        ],
        ontology_iri=ONTOLOGY_IRI,
    )


@pytest.fixture
def described_relations_graph(described_relations_converter: OWLConverter) -> Graph:
    return described_relations_converter.convert()


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


def single_valued_cardinality() -> Literal:
    """
    :return: The cardinality literal a single-valued field is restricted to.
    """
    return Literal(
        OWLConverter.single_valued_field_cardinality, datatype=XSD.nonNegativeInteger
    )


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

    def test_docstring_becomes_the_class_comment(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(
            graph.objects(converter.namespace[ConvertedRoot.__name__], RDFS.comment)
        ) == {Literal(OWLConverter.documentation_of(ConvertedRoot))}

    def test_documentation_is_the_docstring_without_indentation(self):
        assert (
            OWLConverter.documentation_of(ConvertedRoot)
            == "The root class handed to the converter."
        )

    def test_signature_written_by_dataclass_is_not_documentation(self):
        assert OWLConverter.documentation_of(UndocumentedRoot) is None

    def test_class_without_docstring_has_no_comment(self):
        converter = OWLConverter(
            root_classes=[UndocumentedRoot], ontology_iri=ONTOLOGY_IRI
        )
        graph = converter.convert()
        assert (
            set(
                graph.objects(
                    converter.namespace[UndocumentedRoot.__name__], RDFS.comment
                )
            )
            == set()
        )


# %% the class standing for owl:Thing


class TestThingClass:

    @pytest.fixture
    def converter(self) -> OWLConverter:
        return OWLConverter(
            root_classes=[UniversityThing],
            ontology_iri=ONTOLOGY_IRI,
            thing_class=UniversityThing,
        )

    def test_thing_class_is_not_declared(self, graph: Graph, converter: OWLConverter):
        assert (
            converter.namespace[UniversityThing.__name__],
            RDF.type,
            OWL.Class,
        ) not in graph

    def test_subclass_of_thing_class_has_no_named_superclass(
        self, graph: Graph, converter: OWLConverter
    ):
        assert {
            superclass
            for superclass in graph.objects(
                converter.namespace[Organization.__name__], RDFS.subClassOf
            )
            if isinstance(superclass, URIRef)
        } == set()

    def test_property_of_thing_class_has_no_domain(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.objects(converter.namespace["uri"], RDFS.domain)) == set()

    def test_thing_class_is_not_restricted(self, graph: Graph):
        assert set(graph.objects(OWL.Thing, RDFS.subClassOf)) == set()


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

    def test_relation_declared_once_states_its_domain(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.objects(converter.namespace["handle"], RDFS.domain)) == {
            converter.namespace[Drawer.__name__]
        }

    def test_relation_declared_once_states_its_range(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.objects(converter.namespace["handle"], RDFS.range)) == {
            converter.namespace[Handle.__name__]
        }

    def test_relation_whose_range_is_stated_is_not_restricted_again(
        self, graph: Graph, converter: OWLConverter
    ):
        assert (
            restriction_values(
                graph,
                converter.namespace[Drawer.__name__],
                converter.namespace["handle"],
                OWL.allValuesFrom,
            )
            == set()
        )

    def test_relation_to_referenced_class_states_it_as_range(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.objects(converter.namespace["body"], RDFS.range)) == {
            converter.namespace[ReferencedOnlyClass.__name__]
        }

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
        ) == {single_valued_cardinality()}

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

    def test_many_valued_relation_states_the_contained_type_as_range(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.objects(converter.namespace["drawers"], RDFS.range)) == {
            converter.namespace[Drawer.__name__]
        }


# %% fields declared by unrelated classes


class TestFieldDeclaredByUnrelatedClasses:

    def test_property_states_no_domain(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert (
            set(
                described_relations_graph.objects(
                    described_relations_converter.namespace["name"], RDFS.domain
                )
            )
            == set()
        )

    @pytest.mark.parametrize("declaring_class", [Person, RecognizedGroup])
    def test_each_declaring_class_is_restricted_to_the_field_type(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
        declaring_class: type,
    ):
        assert restriction_values(
            described_relations_graph,
            described_relations_converter.namespace[declaring_class.__name__],
            described_relations_converter.namespace["name"],
            OWL.allValuesFrom,
        ) == {OWLConverter.datatypes[str]}


# %% builtin values


class TestDatatypeProperties:

    def test_builtin_fields_are_the_only_datatype_properties(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.subjects(RDF.type, OWL.DatatypeProperty)) == {
            converter.namespace["shelfCount"],
            converter.namespace["labels"],
        }

    def test_property_of_a_field_is_named_in_camel_case(self):
        assert OWLConverter.property_name_of_field("shelf_count") == "shelfCount"

    def test_builtin_type_becomes_the_range(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.objects(converter.namespace["shelfCount"], RDFS.range)) == {
            OWLConverter.datatypes[int]
        }

    def test_single_valued_builtin_field_has_at_most_one_value(
        self, graph: Graph, converter: OWLConverter
    ):
        assert restriction_values(
            graph,
            converter.namespace[Cabinet.__name__],
            converter.namespace["shelfCount"],
            OWL.maxCardinality,
        ) == {single_valued_cardinality()}

    def test_many_valued_builtin_field_states_the_contained_type_as_range(
        self, graph: Graph, converter: OWLConverter
    ):
        assert set(graph.objects(converter.namespace["labels"], RDFS.range)) == {
            OWLConverter.datatypes[str]
        }

    def test_field_of_both_kinds_raises(self):
        converter = OWLConverter(
            root_classes=[RootOfFieldsNamedAlike], ontology_iri=ONTOLOGY_IRI
        )
        with pytest.raises(PropertyIsBothObjectAndDatatype):
            converter.convert()


# %% property descriptors


class TestPropertyDescriptors:

    def test_property_of_a_descriptor_is_named_in_camel_case(self):
        assert OWLConverter.property_name_of_descriptor(WorksFor) == "worksFor"

    def test_described_relation_is_declared_under_its_descriptor_name(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert (
            described_relations_converter.property_iri_of_descriptor(Member),
            RDF.type,
            OWL.ObjectProperty,
        ) in described_relations_graph

    def test_field_named_differently_from_its_descriptor_is_not_a_property(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert (
            described_relations_converter.namespace["members"],
            RDF.type,
            OWL.ObjectProperty,
        ) not in described_relations_graph

    def test_descriptor_subclass_is_a_subproperty(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert set(
            described_relations_graph.objects(
                described_relations_converter.property_iri_of_descriptor(HeadOf),
                RDFS.subPropertyOf,
            )
        ) == {described_relations_converter.property_iri_of_descriptor(WorksFor)}

    def test_descriptor_declaring_an_inverse_states_it(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert set(
            described_relations_graph.objects(
                described_relations_converter.property_iri_of_descriptor(MemberOf),
                OWL.inverseOf,
            )
        ) == {described_relations_converter.property_iri_of_descriptor(Member)}

    def test_descriptor_inheriting_an_inverse_does_not_state_it(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert (
            set(
                described_relations_graph.objects(
                    described_relations_converter.property_iri_of_descriptor(WorksFor),
                    OWL.inverseOf,
                )
            )
            == set()
        )

    def test_transitive_descriptor_is_a_transitive_property(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert (
            described_relations_converter.property_iri_of_descriptor(SubOrganizationOf),
            RDF.type,
            OWL.TransitiveProperty,
        ) in described_relations_graph

    def test_described_relation_states_its_domain(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert set(
            described_relations_graph.objects(
                described_relations_converter.property_iri_of_descriptor(WorksFor),
                RDFS.domain,
            )
        ) == {described_relations_converter.namespace[Person.__name__]}


# %% roles


class TestRoles:

    def test_role_is_restricted_to_some_role_taker(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert restriction_values(
            described_relations_graph,
            described_relations_converter.namespace[CEOAsFirstRole.__name__],
            described_relations_converter.role_property_iri,
            OWL.someValuesFrom,
        ) == {described_relations_converter.namespace[Person.__name__]}

    def test_role_of_a_role_is_restricted_to_some_role(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert restriction_values(
            described_relations_graph,
            described_relations_converter.namespace[
                RepresentativeAsSecondRole.__name__
            ],
            described_relations_converter.role_property_iri,
            OWL.someValuesFrom,
        ) == {described_relations_converter.namespace[CEOAsFirstRole.__name__]}

    def test_role_property_is_declared(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert (
            described_relations_converter.role_property_iri,
            RDF.type,
            OWL.ObjectProperty,
        ) in described_relations_graph

    def test_role_taker_field_is_not_a_property(
        self,
        described_relations_graph: Graph,
        described_relations_converter: OWLConverter,
    ):
        assert (
            described_relations_converter.namespace["roleTaker"],
            RDF.type,
            OWL.ObjectProperty,
        ) not in described_relations_graph

    def test_hierarchy_without_roles_declares_no_role_property(
        self, graph: Graph, converter: OWLConverter
    ):
        assert (converter.role_property_iri, RDF.type, OWL.ObjectProperty) not in graph
