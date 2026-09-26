from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from functools import cached_property

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from typing_extensions import Any, ClassVar, Dict, List, Set, Type, get_origin

from krrood.class_diagrams.class_diagram import (
    Association,
    ClassDiagram,
    Inheritance,
    WrappedClass,
)
from krrood.ontomatic.failures import DuplicateOWLClassName
from krrood.utils import recursive_subclasses


@dataclass
class OWLConverter:
    """
    Converts class hierarchies of dataclasses into an OWL ontology.

    Every root class and all of its subclasses become ``owl:Class`` es, related through
    ``rdfs:subClassOf``. Every field of these classes that refers to instances of another
    dataclass becomes an ``owl:ObjectProperty`` named after the field. The class declaring or
    narrowing the field is restricted to the field's type through ``owl:allValuesFrom``, and,
    unless the field is a container, to at most one value. Dataclasses outside the hierarchies
    that such fields refer to are declared as classes too, without their subclasses and fields.

    .. note:: Subclasses are found through the classes Python has loaded, so the modules
        defining them must be imported before converting.
    """

    root_classes: List[Type]
    """
    The classes that are converted together with all their subclasses.
    """

    ontology_iri: URIRef
    """
    The IRI of the created ontology.

    Its classes and properties are named in the namespace ``<ontology_iri>#``.
    """

    single_valued_field_cardinality: ClassVar[int] = 1
    """
    The maximum number of values of a field that is not a container.
    """

    @cached_property
    def namespace(self) -> Namespace:
        """
        The namespace the classes and properties of the ontology are named in.
        """
        return Namespace(f"{self.ontology_iri}#")

    @cached_property
    def hierarchy_classes(self) -> List[Type]:
        """
        The root classes followed by all their subclasses, without duplicates.
        """
        return list(
            dict.fromkeys(
                python_class
                for root_class in self.root_classes
                for python_class in [root_class, *recursive_subclasses(root_class)]
            )
        )

    @cached_property
    def referenced_classes(self) -> List[Type]:
        """
        The dataclasses outside the hierarchies that fields of hierarchy classes refer
        to.
        """
        hierarchy_diagram = ClassDiagram(self.hierarchy_classes)
        field_types = (
            self._python_class_of(wrapped_field.type_endpoint)
            for wrapped_class in hierarchy_diagram.wrapped_classes
            for wrapped_field in wrapped_class.fields
        )
        return list(
            dict.fromkeys(
                field_type
                for field_type in field_types
                if isinstance(field_type, type)
                and is_dataclass(field_type)
                and field_type not in self._hierarchy_class_set
            )
        )

    @cached_property
    def class_diagram(self) -> ClassDiagram:
        """
        The class diagram of the hierarchy classes and the classes they refer to.
        """
        return ClassDiagram(self.hierarchy_classes + self.referenced_classes)

    @cached_property
    def class_iris(self) -> Dict[Type, URIRef]:
        """
        The OWL class IRI of every class in the class diagram.

        :raises DuplicateOWLClassName: If two of the classes share a name.
        """
        classes_by_name: Dict[str, Type] = {}
        for wrapped_class in self.class_diagram.wrapped_classes:
            python_class = self._python_class_of(wrapped_class.clazz)
            known_class = classes_by_name.setdefault(
                python_class.__name__, python_class
            )
            if known_class is not python_class:
                raise DuplicateOWLClassName(known_class, python_class)
        return {
            python_class: self.namespace[name]
            for name, python_class in classes_by_name.items()
        }

    @cached_property
    def restricted_associations(self) -> List[Association]:
        """
        The relations that hierarchy classes declare or narrow, leaving out those they
        inherit unchanged from a hierarchy superclass.
        """
        return [
            association
            for association in self.class_diagram.associations
            if self._is_relation(association)
            and self._is_declared_in_hierarchy(association)
            and not self._is_inherited_unchanged(association)
        ]

    def convert(self) -> Graph:
        """
        :return: The ontology describing the hierarchy classes, the classes they refer to and
            their relations.
        """
        graph = Graph()
        graph.add((self.ontology_iri, RDF.type, OWL.Ontology))
        for class_iri in self.class_iris.values():
            graph.add((class_iri, RDF.type, OWL.Class))
        for inheritance in self.class_diagram.inheritance_relations:
            graph.add(
                (
                    self._class_iri(inheritance.target),
                    RDFS.subClassOf,
                    self._class_iri(inheritance.source),
                )
            )
        for association in self.restricted_associations:
            self._add_relation(graph, association)
        return graph

    # %% relations

    def _add_relation(self, graph: Graph, association: Association) -> None:
        """
        Declare the object property of *association* and restrict its source class by
        it.
        """
        property_iri = self.namespace[association.wrapped_field.public_name]
        source_iri = self._class_iri(association.source)
        graph.add((property_iri, RDF.type, OWL.ObjectProperty))
        self._add_restriction(
            graph,
            source_iri,
            property_iri,
            OWL.allValuesFrom,
            self._class_iri(association.target),
        )
        if association.wrapped_field.is_container:
            return
        self._add_restriction(
            graph,
            source_iri,
            property_iri,
            OWL.maxCardinality,
            Literal(
                self.single_valued_field_cardinality, datatype=XSD.nonNegativeInteger
            ),
        )

    @staticmethod
    def _add_restriction(
        graph: Graph,
        class_iri: URIRef,
        property_iri: URIRef,
        restriction_kind: URIRef,
        value: URIRef | Literal,
    ) -> None:
        """
        Declare the class of *class_iri* a subclass of the restriction of
        *restriction_kind* with *value* on *property_iri*.
        """
        restriction = BNode()
        graph.add((restriction, RDF.type, OWL.Restriction))
        graph.add((restriction, OWL.onProperty, property_iri))
        graph.add((restriction, restriction_kind, value))
        graph.add((class_iri, RDFS.subClassOf, restriction))

    @staticmethod
    def _is_relation(association: Association) -> bool:
        """
        :return: Whether *association* is declared by a field referring to instances of its
            target class, rather than inferred or referring to the class object itself.
        """
        return not association.inferred and not association.wrapped_field.is_type_type

    def _is_declared_in_hierarchy(self, association: Association) -> bool:
        """
        :return: Whether *association* starts at a hierarchy class itself rather than at a
            specialization of a generic one or a referenced class.
        """
        return association.source.clazz in self._hierarchy_class_set

    def _is_inherited_unchanged(self, association: Association) -> bool:
        """
        :return: Whether a hierarchy superclass of the source has the same relation to the same
            target.
        """
        superclasses = self.class_diagram.get_incoming_neighbors_with_relation_type(
            association.source, Inheritance
        )
        return any(
            inherited.wrapped_field.public_name == association.wrapped_field.public_name
            and self._class_iri(inherited.target) == self._class_iri(association.target)
            for superclass in superclasses
            if superclass.clazz in self._hierarchy_class_set
            for inherited in self.class_diagram.get_outgoing_associations_with_condition(
                superclass, self._is_relation
            )
        )

    # %% class naming

    @cached_property
    def _hierarchy_class_set(self) -> Set[Type]:
        """
        The hierarchy classes, for membership checks.
        """
        return set(self.hierarchy_classes)

    def _class_iri(self, wrapped_class: WrappedClass) -> URIRef:
        """
        :return: The OWL class IRI of *wrapped_class*.
        """
        return self.class_iris[self._python_class_of(wrapped_class.clazz)]

    @staticmethod
    def _python_class_of(type_: Any) -> Any:
        """
        :return: The generic class *type_* specializes, or *type_* itself if it specializes none.
        """
        return get_origin(type_) or type_
