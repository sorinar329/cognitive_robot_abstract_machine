from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from functools import cached_property

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from typing_extensions import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Type,
    get_origin,
)

from krrood.class_diagrams.class_diagram import (
    ClassDiagram,
    Inheritance,
    WrappedClass,
)
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.ontomatic.property_descriptor.mixins import (
    HasInverseProperty,
    TransitiveProperty,
)
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)
from krrood.owlmatic.failures import (
    DuplicateOWLClassName,
    PropertyIsBothObjectAndDatatype,
)
from krrood.patterns.role import Role
from krrood.utils import recursive_subclasses


class PropertyKind(Enum):
    """
    The kinds of OWL property a field can become, each valued by the OWL class of such
    properties.
    """

    OBJECT = OWL.ObjectProperty
    """
    A field referring to instances of another class.
    """

    DATATYPE = OWL.DatatypeProperty
    """
    A field holding builtin values.
    """


@dataclass(frozen=True)
class FieldDeclaration:
    """
    A field that a converted class declares or narrows, as the OWL property it becomes.
    """

    declaring_class: Type
    """
    The class declaring or narrowing the field.
    """

    wrapped_field: WrappedField
    """
    The declared field.
    """

    property_iri: URIRef
    """
    The OWL property the field becomes.
    """

    kind: PropertyKind
    """
    Whether the field refers to instances of a class or holds builtin values.
    """

    value_iri: URIRef
    """
    The OWL class or datatype of the field's values.
    """


@dataclass
class OWLConverter:
    """
    Converts class hierarchies of dataclasses into an OWL ontology.

    The mapping is the reverse of the one Ontomatic uses to generate Python from OWL, so
    Python that Ontomatic generated can be converted back and compared with the ontology it
    came from through :class:`~krrood.owlmatic.ontology_comparison.OntologyComparison`:

    * Every root class and all of its subclasses become ``owl:Class`` es, related through
      ``rdfs:subClassOf``, and their docstrings become ``rdfs:comment`` s.
    * A field referring to instances of another dataclass becomes an ``owl:ObjectProperty``,
      a field holding builtin values an ``owl:DatatypeProperty``. A field managed by a
      :class:`~krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor`
      becomes the property named after the descriptor class, and the descriptor classes'
      inheritance, inverses and transitivity become ``rdfs:subPropertyOf``,
      ``owl:inverseOf`` and ``owl:TransitiveProperty``.
    * A property declared by one class and its subclasses states that class as its
      ``rdfs:domain`` and the field's type as its ``rdfs:range``. A class narrowing the
      type, and every declaring class of a property declared by unrelated classes, is
      restricted through ``owl:allValuesFrom`` instead. Fields that are not containers are
      restricted to ``owl:maxCardinality`` 1.
    * A :class:`~krrood.patterns.role.Role` is restricted to some value of its role taker's
      class on the property :attr:`role_property_name`.

    Dataclasses outside the hierarchies that fields or roles refer to are declared as
    classes too, without their subclasses and fields.

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

    thing_class: Optional[Type] = None
    """
    The class standing for ``owl:Thing``, such as the base class Ontomatic generates for an
    ontology.

    It is not declared, its subclasses are not declared subclasses of it, and the properties
    of its fields state no domain.
    """

    single_valued_field_cardinality: ClassVar[int] = 1
    """
    The maximum number of values of a field that is not a container.
    """

    role_property_name: ClassVar[str] = "roleFor"
    """
    The name of the property relating a role to its role taker, which Ontomatic reads roles
    from.
    """

    datatypes: ClassVar[Dict[Type, URIRef]] = {
        str: XSD.string,
        int: XSD.integer,
        float: XSD.double,
        bool: XSD.boolean,
        datetime: XSD.dateTime,
    }
    """
    The datatype of each builtin type a datatype property can hold.
    """

    property_characteristics: ClassVar[Dict[Type, URIRef]] = {
        TransitiveProperty: OWL.TransitiveProperty,
    }
    """
    The OWL property characteristic each descriptor mixin stands for.
    """

    @cached_property
    def namespace(self) -> Namespace:
        """
        The namespace the classes and properties of the ontology are named in.
        """
        return Namespace(f"{self.ontology_iri}#")

    @cached_property
    def role_property_iri(self) -> URIRef:
        """
        The IRI of the property relating a role to its role taker.
        """
        return self.namespace[self.role_property_name]

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
    def roles(self) -> List[Type[Role]]:
        """
        The hierarchy classes that are roles.
        """
        return [
            python_class
            for python_class in self.hierarchy_classes
            if issubclass(python_class, Role) and python_class is not Role
        ]

    @cached_property
    def referenced_classes(self) -> List[Type]:
        """
        The dataclasses outside the hierarchies that fields of hierarchy classes or roles
        refer to.
        """
        hierarchy_diagram = ClassDiagram(
            self.hierarchy_classes, introspector=DescriptorAwareIntrospector()
        )
        field_types = [
            self._python_class_of(wrapped_field.type_endpoint)
            for wrapped_class in hierarchy_diagram.wrapped_classes
            for wrapped_field in wrapped_class.fields
        ]
        role_takers = [role.get_role_taker_type() for role in self.roles]
        return list(
            dict.fromkeys(
                referenced_type
                for referenced_type in field_types + role_takers
                if isinstance(referenced_type, type)
                and is_dataclass(referenced_type)
                and referenced_type not in self._hierarchy_class_set
            )
        )

    @cached_property
    def class_diagram(self) -> ClassDiagram:
        """
        The class diagram of the hierarchy classes and the classes they refer to.
        """
        return ClassDiagram(
            self.hierarchy_classes + self.referenced_classes,
            introspector=DescriptorAwareIntrospector(),
        )

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
            python_class: (
                OWL.Thing if python_class is self.thing_class else self.namespace[name]
            )
            for name, python_class in classes_by_name.items()
        }

    @cached_property
    def field_declarations(self) -> List[FieldDeclaration]:
        """
        The fields that hierarchy classes declare or narrow, leaving out those they inherit
        unchanged from a hierarchy superclass.

        :raises PropertyIsBothObjectAndDatatype: If a property would be of both kinds.
        """
        declarations = [
            declaration
            for wrapped_class in self._hierarchy_wrapped_classes
            for declaration in self._declarations_of(wrapped_class)
            if not self._is_inherited_unchanged(declaration)
        ]
        self._validate_property_kinds(declarations)
        return declarations

    @cached_property
    def described_properties(self) -> List[Type[PropertyDescriptor]]:
        """
        The descriptor classes managing converted fields, together with the descriptor
        classes they inherit from.
        """
        return list(
            dict.fromkeys(
                descriptor_class
                for declaration in self.field_declarations
                if declaration.wrapped_field.property_descriptor is not None
                for descriptor_class in type(
                    declaration.wrapped_field.property_descriptor
                ).__mro__
                if issubclass(descriptor_class, PropertyDescriptor)
                and descriptor_class is not PropertyDescriptor
            )
        )

    def convert(self) -> Graph:
        """
        :return: The ontology describing the hierarchy classes, the classes they refer to and
            their properties.
        """
        graph = Graph()
        graph.add((self.ontology_iri, RDF.type, OWL.Ontology))
        self._add_classes(graph)
        self._add_roles(graph)
        self._add_field_declarations(graph)
        self._add_described_properties(graph)
        return graph

    # %% naming

    @staticmethod
    def property_name_of_field(field_name: str) -> str:
        """
        :return: The name of the property a field of *field_name* becomes, which is the
            camel case spelling Ontomatic turns back into the field name.
        """
        first_word, *other_words = field_name.split("_")
        return first_word + "".join(word.capitalize() for word in other_words)

    @staticmethod
    def property_name_of_descriptor(descriptor_class: Type[PropertyDescriptor]) -> str:
        """
        :return: The name of the property *descriptor_class* manages, which is the camel
            case spelling Ontomatic turns back into the descriptor class name.
        """
        return descriptor_class.__name__[:1].lower() + descriptor_class.__name__[1:]

    def property_iri_of_descriptor(
        self, descriptor_class: Type[PropertyDescriptor]
    ) -> URIRef:
        """
        :return: The IRI of the property *descriptor_class* manages.
        """
        return self.namespace[self.property_name_of_descriptor(descriptor_class)]

    def property_iri_of_field(self, wrapped_field: WrappedField) -> URIRef:
        """
        :return: The IRI of the property *wrapped_field* becomes.
        """
        if wrapped_field.property_descriptor is not None:
            return self.property_iri_of_descriptor(
                type(wrapped_field.property_descriptor)
            )
        return self.namespace[self.property_name_of_field(wrapped_field.public_name)]

    @staticmethod
    def documentation_of(python_class: Type) -> Optional[str]:
        """
        :return: The docstring written for *python_class*, or ``None`` if it has none. The
            signature ``dataclass`` writes into the docstring of an undocumented class is not
            documentation.
        """
        docstring = python_class.__doc__
        if docstring is None or docstring.startswith(f"{python_class.__name__}("):
            return None
        return inspect.cleandoc(docstring)

    # %% classes

    def _add_classes(self, graph: Graph) -> None:
        """
        Declare the classes with their documentation and their superclasses.
        """
        for python_class, class_iri in self.class_iris.items():
            if class_iri == OWL.Thing:
                continue
            graph.add((class_iri, RDF.type, OWL.Class))
            self._add_documentation(graph, class_iri, python_class)
        for inheritance in self.class_diagram.inheritance_relations:
            superclass_iri = self._class_iri(inheritance.source)
            subclass_iri = self._class_iri(inheritance.target)
            if OWL.Thing in (superclass_iri, subclass_iri):
                continue
            graph.add((subclass_iri, RDFS.subClassOf, superclass_iri))

    def _add_roles(self, graph: Graph) -> None:
        """
        Restrict every role to some value of its role taker's class, and declare the
        property relating them if there are roles.
        """
        if not self.roles:
            return
        graph.add((self.role_property_iri, RDF.type, OWL.ObjectProperty))
        for role in self.roles:
            self._add_restriction(
                graph,
                self.class_iris[role],
                self.role_property_iri,
                OWL.someValuesFrom,
                self.class_iris[role.get_role_taker_type()],
            )

    @classmethod
    def _add_documentation(cls, graph: Graph, iri: URIRef, python_class: Type) -> None:
        """
        Annotate *iri* with the documentation of *python_class*, if it has any.
        """
        documentation = cls.documentation_of(python_class)
        if documentation is not None:
            graph.add((iri, RDFS.comment, Literal(documentation)))

    # %% fields

    def _add_field_declarations(self, graph: Graph) -> None:
        """
        Declare the properties the fields become, state their domains and ranges where one
        class declares them, and restrict the declaring classes by them otherwise.
        """
        declarations_by_property: DefaultDict[URIRef, List[FieldDeclaration]] = (
            defaultdict(list)
        )
        for declaration in self.field_declarations:
            declarations_by_property[declaration.property_iri].append(declaration)
        for property_iri, declarations in declarations_by_property.items():
            graph.add((property_iri, RDF.type, declarations[0].kind.value))
            stated_range = self._add_domain_and_range(graph, declarations)
            for declaration in declarations:
                self._add_restrictions(graph, declaration, stated_range)

    def _add_domain_and_range(
        self, graph: Graph, declarations: List[FieldDeclaration]
    ) -> Optional[URIRef]:
        """
        State the domain and range of the property of *declarations* if one class and its
        subclasses declare it.

        :return: The stated range, or ``None`` if none was stated.
        """
        most_general = [
            declaration
            for declaration in declarations
            if not any(
                other.declaring_class is not declaration.declaring_class
                and issubclass(declaration.declaring_class, other.declaring_class)
                for other in declarations
            )
        ]
        if len(most_general) != 1:
            return None
        declaration = most_general[0]
        domain_iri = self.class_iris[declaration.declaring_class]
        if domain_iri != OWL.Thing:
            graph.add((declaration.property_iri, RDFS.domain, domain_iri))
        graph.add((declaration.property_iri, RDFS.range, declaration.value_iri))
        return declaration.value_iri

    def _add_restrictions(
        self,
        graph: Graph,
        declaration: FieldDeclaration,
        stated_range: Optional[URIRef],
    ) -> None:
        """
        Restrict the declaring class of *declaration* to the field's type unless that is the
        stated range, and to at most one value unless the field is a container.
        """
        class_iri = self.class_iris[declaration.declaring_class]
        if class_iri == OWL.Thing:
            return
        if declaration.value_iri != stated_range:
            self._add_restriction(
                graph,
                class_iri,
                declaration.property_iri,
                OWL.allValuesFrom,
                declaration.value_iri,
            )
        if declaration.wrapped_field.is_container:
            return
        self._add_restriction(
            graph,
            class_iri,
            declaration.property_iri,
            OWL.maxCardinality,
            Literal(
                self.single_valued_field_cardinality, datatype=XSD.nonNegativeInteger
            ),
        )

    def _declarations_of(
        self, wrapped_class: WrappedClass
    ) -> Iterator[FieldDeclaration]:
        """
        :return: The declarations of the fields of *wrapped_class* that become properties.
        """
        declaring_class = self._python_class_of(wrapped_class.clazz)
        for wrapped_field in wrapped_class.fields:
            if wrapped_field.is_type_type or wrapped_field.is_role_taker:
                continue
            value_type = self._python_class_of(wrapped_field.type_endpoint)
            if value_type in self.class_iris:
                kind, value_iri = PropertyKind.OBJECT, self.class_iris[value_type]
            elif value_type in self.datatypes:
                kind, value_iri = PropertyKind.DATATYPE, self.datatypes[value_type]
            else:
                continue
            yield FieldDeclaration(
                declaring_class=declaring_class,
                wrapped_field=wrapped_field,
                property_iri=self.property_iri_of_field(wrapped_field),
                kind=kind,
                value_iri=value_iri,
            )

    def _is_inherited_unchanged(self, declaration: FieldDeclaration) -> bool:
        """
        :return: Whether a hierarchy superclass of the declaring class has a field of the same
            property with the same type.
        """
        superclasses = self.class_diagram.get_incoming_neighbors_with_relation_type(
            self.class_diagram.get_wrapped_class(declaration.declaring_class),
            Inheritance,
        )
        return any(
            inherited.property_iri == declaration.property_iri
            and inherited.value_iri == declaration.value_iri
            for superclass in superclasses
            if superclass.clazz in self._hierarchy_class_set
            for inherited in self._declarations_of(superclass)
        )

    @staticmethod
    def _validate_property_kinds(declarations: List[FieldDeclaration]) -> None:
        """
        :raises PropertyIsBothObjectAndDatatype: If two of *declarations* give one property
            different kinds.
        """
        declaration_by_property: Dict[URIRef, FieldDeclaration] = {}
        for declaration in declarations:
            known = declaration_by_property.setdefault(
                declaration.property_iri, declaration
            )
            if known.kind is declaration.kind:
                continue
            object_declaration, datatype_declaration = (
                (known, declaration)
                if known.kind is PropertyKind.OBJECT
                else (declaration, known)
            )
            raise PropertyIsBothObjectAndDatatype(
                declaration.property_iri,
                object_declaration.declaring_class,
                datatype_declaration.declaring_class,
            )

    # %% property descriptors

    def _add_described_properties(self, graph: Graph) -> None:
        """
        Declare the properties the descriptor classes manage, with their documentation,
        superproperties, inverses and characteristics.
        """
        for descriptor_class in self.described_properties:
            property_iri = self.property_iri_of_descriptor(descriptor_class)
            graph.add((property_iri, RDF.type, OWL.ObjectProperty))
            self._add_documentation(graph, property_iri, descriptor_class)
            for superproperty in descriptor_class.__bases__:
                if (
                    issubclass(superproperty, PropertyDescriptor)
                    and superproperty is not PropertyDescriptor
                ):
                    graph.add(
                        (
                            property_iri,
                            RDFS.subPropertyOf,
                            self.property_iri_of_descriptor(superproperty),
                        )
                    )
            if HasInverseProperty in descriptor_class.__bases__:
                graph.add(
                    (
                        property_iri,
                        OWL.inverseOf,
                        self.property_iri_of_descriptor(descriptor_class.get_inverse()),
                    )
                )
            for mixin, characteristic in self.property_characteristics.items():
                if issubclass(descriptor_class, mixin):
                    graph.add((property_iri, RDF.type, characteristic))

    # %% restrictions

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

    # %% class naming

    @cached_property
    def _hierarchy_class_set(self) -> Set[Type]:
        """
        The hierarchy classes, for membership checks.
        """
        return set(self.hierarchy_classes)

    @cached_property
    def _hierarchy_wrapped_classes(self) -> List[WrappedClass]:
        """
        The wrapped hierarchy classes of the class diagram.
        """
        return [
            wrapped_class
            for wrapped_class in self.class_diagram.wrapped_classes
            if wrapped_class.clazz in self._hierarchy_class_set
        ]

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
