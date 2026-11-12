"""
Inference over the information about an ontology: class hierarchies, restrictions as classification conditions, property domains and ranges, and implicit subsumption.
"""

from __future__ import annotations

from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import rdflib
from ordered_set import OrderedSet
from rdflib.namespace import RDF, RDFS, OWL, XSD
from typing_extensions import ClassVar, Dict, List, Optional, Set, Tuple

from krrood import logger
from krrood.ontomatic.failures import (
    UnionOfAnonymousClassesAsFiller,
    UntranslatedComplement,
)
from krrood.ontomatic.naming import NamingRegistry
from krrood.ontomatic.ontology_to_python.axioms import (
    AxiomInfo,
    SubClassAxiomInfo,
    QualifiedCardinalityAxiomInfo,
    MaxQualifiedCardinalityAxiomInfo,
    MinQualifiedCardinalityAxiomInfo,
    CardinalityAxiomInfo,
    MaxCardinalityAxiomInfo,
    MinCardinalityAxiomInfo,
    HasValueAxiomInfo,
    SomeValuesFromAxiomInfo,
    AllValuesFromAxiomInfo,
    ComplementAxiomInfo,
    OneOfAxiomInfo,
    PropertyAxiomInfo,
)
from krrood.ontomatic.ontology_to_python.property_inference import (
    PropertyDomainAndRangeInference,
)
from krrood.ontomatic.ontology_to_python.property_type import PropertyType
from krrood.ontomatic.property_descriptor.mixins import (
    TransitiveProperty,
    HasInverseProperty,
    HasEquivalentProperties,
    HasDisjointProperties,
    SymmetricProperty,
    ASymmetricProperty,
    ReflexiveProperty,
    IrreflexiveProperty,
)
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)
from krrood.symbol_graph.symbol_graph import Symbol

from krrood.ontomatic.ontology_to_python.ontology_info import (
    ClassInfo,
    OntologyInfo,
    PropertyInfo,
    RoleTakerInfo,
)


class SubsumptionType(Enum):
    """
    How a generated class relates to a class it specialises.
    """

    SUBTYPE = "subtype"
    """
    It is a subtype of the given class (e.g. Math is a subtype of Course). This is the equivalent to OOP 
    inheritance..
    """
    ROLE = "role"
    """
    It is a role that a persistent identifier can take on in a certain context 
    (e.g. Student is a role that a Person can take on in the context of taking a course).
    Thi is the equivalent to OOP composition.
    """


@dataclass(kw_only=True)
class RestrictionCondition:
    """
    A restriction translated into the condition it puts on a class.
    """

    axiom: PropertyAxiomInfo
    """
    The condition.
    """

    value_class: Optional[str]
    """
    The named class the restriction restricts the values of the property to, or None when it names no single class,
    so that the range of the property is taken from its declaration.
    """


@dataclass(kw_only=True)
class PropertyMaps:
    """
    Dataclass for storing property inference maps for domain, range, superclasses, and inverse property pairs.
    """

    domains_by_property: Dict[str, Set[str]] = field(default_factory=dict)
    """
    Map of property names to sets of domain class names.
    """
    declared_domains_by_property: Dict[str, Set[str]] = field(default_factory=dict)
    """
    Map of property names to sets of declared domain class names. These are the classes where the property is declared.
    """
    ranges_by_property: Dict[str, Set[str]] = field(default_factory=dict)
    """
    Map of property names to sets of range class names.
    """
    range_uris_by_property: Dict[str, Set[rdflib.term.URIRef]] = field(
        default_factory=dict
    )
    """
    Map of property names to sets of range URIs.
    """
    superproperties_by_property: Dict[str, List[str]] = field(default_factory=dict)
    """
    Map of property names to lists of superclass names.
    """
    inverse_pairs: List[Tuple[str, str]] = field(default_factory=list)
    """
    List of tuples representing inverse property pairs (property_name, inverse_property_name).
    """
    equivalent_properties_by_property: Dict[str, List[str]] = field(
        default_factory=dict
    )
    """
    Map of property names to lists of equivalent property names.
    """
    properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    """
    Map of property names to PropertyInfo objects.
    """

    @classmethod
    def from_properties(cls, *, properties: Dict[str, PropertyInfo]) -> PropertyMaps:
        """
        Create a PropertyMaps object from a dictionary of PropertyInfo objects.

        :param properties: A dictionary of PropertyInfo objects.
        :return: A PropertyMaps object.
        """
        domains_by_property = {
            name: set(property_information.domains)
            for name, property_information in properties.items()
        }
        declared_domains_by_property = {
            name: set(property_information.domains)
            for name, property_information in properties.items()
        }
        ranges_by_property = {
            name: set(property_information.ranges)
            for name, property_information in properties.items()
        }
        range_uris_by_property = {
            name: set(property_information.range_uris)
            for name, property_information in properties.items()
        }
        superproperties_by_property = {
            name: list(property_information.superproperties)
            for name, property_information in properties.items()
        }
        inverse_pairs = [
            (name, inverse_name)
            for name, property_information in properties.items()
            for inverse_name in property_information.inverses
            if inverse_name in properties
        ]
        equivalent_properties_by_property = {
            name: list(property_information.equivalent_properties)
            for name, property_information in properties.items()
        }
        return cls(
            domains_by_property=domains_by_property,
            declared_domains_by_property=declared_domains_by_property,
            ranges_by_property=ranges_by_property,
            range_uris_by_property=range_uris_by_property,
            superproperties_by_property=superproperties_by_property,
            inverse_pairs=inverse_pairs,
            equivalent_properties_by_property=equivalent_properties_by_property,
            properties=properties,
        )


@dataclass(kw_only=True)
class InferenceEngine:
    """Engine for performing ontological inference and computing class/property relationships."""

    ontology: OntologyInfo
    """
    The ontology whose classes and properties are inferred.
    """
    property_maps: PropertyMaps = field(init=False)
    """
    The domains and ranges of the properties while they are inferred.
    """
    XSD_TO_PYTHON_TYPES: ClassVar[Dict[rdflib.term.URIRef, str]] = {
        XSD.string: "str",
        XSD.normalizedString: "str",
        XSD.token: "str",
        XSD.language: "str",
        XSD.boolean: "bool",
        XSD.decimal: "float",
        XSD.float: "float",
        XSD.double: "float",
        XSD.integer: "int",
        XSD.nonPositiveInteger: "int",
        XSD.negativeInteger: "int",
        XSD.long: "int",
        XSD.int: "int",
        XSD.short: "int",
        XSD.byte: "int",
        XSD.nonNegativeInteger: "int",
        XSD.unsignedLong: "int",
        XSD.unsignedInt: "int",
        XSD.unsignedShort: "int",
        XSD.unsignedByte: "int",
        XSD.positiveInteger: "int",
        XSD.date: "str",
        XSD.dateTime: "str",
        XSD.time: "str",
        XSD.anyURI: "str",
    }
    """
    The Python types of the XML Schema datatypes.
    """

    def __post_init__(self):
        """
        Start from the declared domains and ranges of the properties.
        """
        self.property_maps = PropertyMaps.from_properties(
            properties=self.ontology.properties
        )

    def compute_ancestors(self):
        """
        Compute full ancestor sets for each class (transitive closure).
        """
        # Compute full ancestor sets for each class (transitive closure)
        name_to_bases = {
            name: set(class_information.base_classes)
            for name, class_information in self.ontology.classes.items()
        }
        for class_information in self.ontology.classes.values():
            ancestors = set()
            stack = list(class_information.base_classes)
            while stack:
                base = stack.pop()
                if base in ancestors:
                    continue
                ancestors.add(base)
                stack.extend(name_to_bases.get(base, []))
            class_information.all_base_classes = sorted(ancestors)

    def infer_properties(
        self,
    ):
        """
        Main entry point for property inference.
        Propagates domains, ranges, and handles restrictions and inverses.
        """
        self._infer_properties_data_from_restrictions()

        for property_name, property_information in self.ontology.properties.items():
            if property_name == "roleFor":
                continue
            if property_information.type == PropertyType.DATA_PROPERTY:
                continue
            PropertyDomainAndRangeInference(
                property_information=property_information, ontology=self.ontology
            ).complete()

        self._finalize_properties()
        self._add_property_chain_axioms()
        self._create_specialized_properties()

    def _add_property_chain_axioms(self):
        """
        Add property chain axioms to properties based on owl:propertyChainAxiom.
        """
        for property_name, property_information in self.ontology.properties.items():
            property_uri = rdflib.URIRef(property_information.uri)
            for chain in self.ontology.graph.objects(
                property_uri, OWL.propertyChainAxiom
            ):
                items = []
                node = chain
                while node and node != RDF.nil:
                    first = self.ontology.graph.value(node, RDF.first)
                    if first:
                        name = NamingRegistry.uri_to_python_name(uri=first)
                        if name not in self.ontology.properties:
                            raise ValueError(
                                f"Property chain axiom references unknown property {name}"
                            )
                        items.append(self.ontology.properties[name].descriptor_name)
                    node = self.ontology.graph.value(node, RDF.rest)
                if items:
                    property_information.chain_axioms.append(items)

    def _infer_properties_data_from_restrictions(self):
        """
        Walk through all classes and their restrictions in the graph and update the declared domains of the properties accordingly.
        """
        covered_restrictions = set()
        # Walk class restrictions
        for class_node in self.ontology.graph.subjects(RDF.type, OWL.Class):
            # if isinstance(cls_uri, rdflib.BNode):
            #     continue
            class_name = NamingRegistry.uri_to_python_name(
                uri=class_node, graph=self.ontology.graph
            )
            for_class = (
                self.ontology.description_for(description_name=class_name) or class_name
            )
            if not self._is_extracted(class_node=class_node, class_name=class_name):
                # Its conditions are those of the intersection it duplicates.
                for coll in self.ontology.graph.objects(class_node, OWL.intersectionOf):
                    covered_restrictions.update(
                        NamingRegistry._get_rdf_list(
                            graph=self.ontology.graph, head=coll
                        )
                    )
                continue

            # direct subclass restrictions are necessary conditions of the class
            for restr in self.ontology.graph.objects(class_node, RDFS.subClassOf):
                self._restrictions_handler(
                    for_class=class_name, node=restr, sufficient=False
                )
                # If restriction mentions a property, count this class as declared domain for that property
                covered_restrictions.add(restr)
                restricted_property = self.ontology.graph.value(restr, OWL.onProperty)
                if restricted_property:
                    property_name = NamingRegistry.uri_to_python_name(
                        uri=restricted_property, graph=self.ontology.graph
                    )
                    self.property_maps.declared_domains_by_property[property_name].add(
                        class_name
                    )

            # restrictions inside intersectionOf
            intersection = []
            sufficient = self._is_sufficient_class_expression(class_node=class_node)
            for coll in self.ontology.graph.objects(class_node, OWL.intersectionOf):
                node = coll
                while node and node != RDF.nil:
                    first = self.ontology.graph.value(node, RDF.first)
                    if self._add_complement_or_enumeration_condition(
                        class_name=class_name, member=first, sufficient=sufficient
                    ):
                        covered_restrictions.add(
                            self.ontology.graph.value(first, OWL.complementOf)
                        )
                        node = self.ontology.graph.value(node, RDF.rest)
                        continue
                    restriction_added = self._restrictions_handler(
                        for_class=class_name, node=first, sufficient=sufficient
                    )
                    covered_restrictions.add(first)
                    restricted_property = (
                        self.ontology.graph.value(first, OWL.onProperty)
                        if first
                        else None
                    )
                    if restricted_property and restriction_added:
                        property_name = NamingRegistry.uri_to_python_name(
                            uri=restricted_property, graph=self.ontology.graph
                        )
                        self.property_maps.declared_domains_by_property[
                            property_name
                        ].add(for_class)
                    elif not restricted_property:
                        first_name = NamingRegistry.uri_to_python_name(
                            uri=first, graph=self.ontology.graph
                        )
                        intersection.append(first_name)
                    node = self.ontology.graph.value(node, RDF.rest)
            self._add_named_conjunct_conditions(
                for_class=for_class, named_conjuncts=intersection, sufficient=sufficient
            )
        # Standalone Restrictions
        for restr in self.ontology.graph.subjects(RDF.type, OWL.Restriction):
            if restr in covered_restrictions:
                continue
            covered_restrictions.add(restr)
            restricted_property = self.ontology.graph.value(restr, OWL.onProperty)
            if restricted_property is None:
                continue
            for_class = None
            for subject, predicate in self.ontology.graph.subject_predicates(restr):
                if predicate == OWL.equivalentClass:
                    for_class = NamingRegistry.uri_to_python_name(
                        uri=subject, graph=self.ontology.graph
                    )
                    break
            if not for_class:
                for_class = self.ontology.graph.value(restr, RDFS.subClassOf)
                if for_class:
                    for_class = NamingRegistry.uri_to_python_name(
                        uri=for_class, graph=self.ontology.graph
                    )
            if not for_class:
                raise ValueError(f"Could not determine class for restriction {restr}")
            # Either an owl:equivalentClass definition or a general class axiom (restriction rdfs:subClassOf class),
            # both are sufficient conditions of the class.
            self._restrictions_handler(for_class=for_class, node=restr, sufficient=True)
            property_name = NamingRegistry.uri_to_python_name(
                uri=restricted_property, graph=self.ontology.graph
            )
            for_class = (
                self.ontology.description_for(description_name=for_class) or for_class
            )
            self.property_maps.declared_domains_by_property[property_name].add(
                for_class
            )

    def _is_extracted(self, *, class_node: rdflib.term.Node, class_name: str) -> bool:
        """
        :param class_node: A class.
        :param cls_name: Its name.
        :return: False for an anonymous intersection that was not extracted because another one of the same members
            and name was, such as a filler that stands for the named class it has the definition of.
        """
        if not isinstance(class_node, rdflib.BNode) or not list(
            self.ontology.graph.objects(class_node, OWL.intersectionOf)
        ):
            return True
        description_for = self.ontology.description_for(description_name=class_name)
        class_information = (
            self.ontology.class_descriptions[description_for]
            if description_for is not None
            else self.ontology.classes.get(class_name)
        )
        return class_information is None or class_information.uri == str(class_node)

    def _is_sufficient_class_expression(self, *, class_node: rdflib.term.Node) -> bool:
        """
        Whether an intersection is a sufficient condition of the class it describes: a named class, a general class
        axiom, an ``owl:equivalentClass`` definition, or an anonymous intersection that is a class of its own. One
        appearing only as a superclass of a named class is a necessary condition.

        :param class_node: The class node.
        :return: True if the expression is a sufficient condition.
        """
        if isinstance(class_node, rdflib.URIRef):
            return True
        if any(
            isinstance(superclass, rdflib.URIRef)
            for superclass in self.ontology.graph.objects(class_node, RDFS.subClassOf)
        ):
            return True
        return (
            (None, OWL.equivalentClass, class_node) in self.ontology.graph
            or (class_node, OWL.equivalentClass, None) in self.ontology.graph
            or not (
                any(self.ontology.graph.objects(class_node, RDFS.subClassOf))
                or any(self.ontology.graph.subjects(RDFS.subClassOf, class_node))
            )
        )

    def _add_named_conjunct_conditions(
        self, *, for_class: str, named_conjuncts: List[str], sufficient: bool
    ):
        """
        Add a type check for every named class in an intersection to the conditions of the described class, so that
        e.g. ``Student and (hasMajor some Science) SubClassOf ScienceStudent`` checks that the individual is a Student.

        :param for_class: The class the intersection describes.
        :param named_conjuncts: Names of the named classes of the intersection.
        :param sufficient: Whether the intersection is a sufficient condition.
        """
        class_information = self.ontology.classes.get(for_class)
        if class_information is None:
            return
        for conjunct in reversed(named_conjuncts):
            if conjunct == for_class or conjunct not in self.ontology.classes:
                continue
            subclass_axiom = SubClassAxiomInfo(sub_class=conjunct)
            python_condition = subclass_axiom.conditions_python()[0]
            if not sufficient:
                if python_condition not in class_information.necessary_axioms_python:
                    class_information.necessary_axioms_python.insert(
                        0, python_condition
                    )
                continue
            if not class_information.axioms:
                # The class has no sufficient restriction conditions, a named conjunct alone is a plain subsumption.
                continue
            if python_condition not in class_information.axioms_python:
                class_information.axioms_setup[0:0] = subclass_axiom.setup_statements()
                class_information.axioms.insert(0, subclass_axiom.conditions_eql()[0])
                class_information.axioms_python.insert(0, python_condition)

    def _restrictions_handler(
        self, *, for_class: str, node: rdflib.term.Node, sufficient: bool = True
    ):
        """
        Handle restrictions for a given class and node in the ontology graph.

        :param for_class: The class name.
        :param node: The restriction node.
        :param sufficient: Whether the restriction is (part of) a sufficient condition of the class. Only sufficient
         conditions become classification axioms, necessary conditions are kept as documentation.
        """
        if not node:
            return True
        restricted_property = self.ontology.graph.value(node, OWL.onProperty)
        if not restricted_property:
            return True
        description_for = self.ontology.description_for(description_name=for_class)
        for_class = description_for or for_class
        property_name = NamingRegistry.uri_to_python_name(
            uri=restricted_property, graph=self.ontology.graph
        )
        if property_name in self.ontology.properties:
            self.property_maps.domains_by_property[property_name].add(for_class)
        restriction = self._restriction_condition(
            node=node, for_class=for_class, property_name=property_name
        )
        if restriction:
            axiom = restriction.axiom
            value_type = restriction.value_class
            try:
                range_name = value_type
                if property_name == "roleFor":
                    class_information = self.ontology.classes.get(for_class)
                    if class_information:
                        class_information.role_taker = RoleTakerInfo(
                            class_name=range_name,
                            field_name=NamingRegistry.to_snake_case(name=range_name),
                        )
                    return True
                self._add_condition(
                    for_class=for_class, axiom=axiom, sufficient=sufficient
                )
                if range_name is None:
                    if self.ontology.original_properties[property_name].ranges:
                        range_name = self.ontology.original_properties[
                            property_name
                        ].ranges[0]
                    else:
                        return True
                existing_ranges = self.property_maps.ranges_by_property.get(
                    property_name, set()
                )
                contesting_ranges = set(existing_ranges)
                for (
                    domain_class,
                    ranges_by_restricted_property,
                ) in self.ontology.property_restrictions.items():
                    if property_name in ranges_by_restricted_property:
                        for existing_range in existing_ranges:
                            if (
                                existing_range
                                in ranges_by_restricted_property[property_name]
                            ):
                                contesting_ranges.remove(existing_range)
                if self.ontology.classes[for_class].role_taker:
                    role_taker = self.ontology.classes[for_class].role_taker.class_name
                    if (
                        role_taker
                        in self.ontology.original_properties[property_name].domains
                        and range_name
                        in self.ontology.original_properties[property_name].ranges
                    ):
                        self.property_maps.domains_by_property[property_name].remove(
                            for_class
                        )
                        self.ontology.property_restrictions.setdefault(
                            for_class, {}
                        ).setdefault(property_name, set()).add(range_name)
                        return False
                if contesting_ranges and not any(
                    contesting_range
                    in self.ontology.classes[range_name].all_base_classes + [range_name]
                    for contesting_range in contesting_ranges
                ):
                    self.property_maps.domains_by_property[property_name].remove(
                        for_class
                    )
                    return False

                self.property_maps.ranges_by_property[property_name].add(range_name)
                range_uri = (
                    value_type
                    if value_type is not None
                    else rdflib.URIRef(self.ontology.classes[range_name].uri)
                )
                self.property_maps.range_uris_by_property[property_name].add(range_uri)
                self.ontology.property_restrictions.setdefault(
                    for_class, {}
                ).setdefault(property_name, set()).add(range_name)
                # The inverse of a restriction (e.g. Organization <- hasEmployee some Employee from
                # Person and (worksFor some Organization) SubClassOf Employee) is not entailed, so it only
                # contributes to the attribute types and never becomes a classification axiom of the range class.
                for inverse in self.ontology.properties[property_name].inverses:
                    self.ontology.property_restrictions.setdefault(
                        range_name, {}
                    ).setdefault(inverse, set()).add(for_class)
            except Exception as e:
                logger.warning(f"[owl_to_python] Error processing restriction: {e}")
            return True

    def _restriction_condition(
        self, *, node: rdflib.term.Node, for_class: str, property_name: str
    ) -> Optional[RestrictionCondition]:
        """
        :param node: A restriction.
        :param for_class: The class the restriction is a condition of.
        :param prop_name: The name of the property the restriction is on.
        :return: The condition the restriction puts on the class, or None for a kind of restriction that is not
            translated.
        """
        graph = self.ontology.graph
        if graph.value(node, OWL.hasValue):
            has_value = graph.value(node, OWL.hasValue)
            value_type = [
                value_class
                for value_class in graph.objects(has_value, RDF.type)
                if value_class not in (OWL.NamedIndividual, OWL.Thing)
            ][0]
            return RestrictionCondition(
                axiom=HasValueAxiomInfo(
                    property_name=property_name,
                    value=has_value,
                    for_class=for_class,
                    ontology=self.ontology,
                ),
                value_class=NamingRegistry.uri_to_python_name(
                    uri=value_type, graph=graph
                ),
            )
        for qualified, axiom_type in (
            (OWL.maxQualifiedCardinality, MaxQualifiedCardinalityAxiomInfo),
            (OWL.minQualifiedCardinality, MinQualifiedCardinalityAxiomInfo),
            (OWL.qualifiedCardinality, QualifiedCardinalityAxiomInfo),
        ):
            if graph.value(node, qualified):
                on_class = NamingRegistry.uri_to_python_name(
                    uri=graph.value(node, OWL.onClass), graph=graph
                )
                return RestrictionCondition(
                    axiom=axiom_type(
                        property_name=property_name,
                        quantity=graph.value(node, qualified).toPython(),
                        on_class=on_class,
                        for_class=for_class,
                        ontology=self.ontology,
                    ),
                    value_class=on_class,
                )
        for unqualified, axiom_type in (
            (OWL.maxCardinality, MaxCardinalityAxiomInfo),
            (OWL.minCardinality, MinCardinalityAxiomInfo),
            (OWL.cardinality, CardinalityAxiomInfo),
        ):
            if graph.value(node, unqualified) is not None:
                return RestrictionCondition(
                    axiom=axiom_type(
                        property_name=property_name,
                        quantity=graph.value(node, unqualified).toPython(),
                        for_class=for_class,
                        ontology=self.ontology,
                    ),
                    value_class=None,
                )
        for quantified, axiom_type in (
            (OWL.someValuesFrom, SomeValuesFromAxiomInfo),
            (OWL.allValuesFrom, AllValuesFromAxiomInfo),
        ):
            filler = graph.value(node, quantified)
            if filler:
                union = graph.value(filler, OWL.unionOf)
                return RestrictionCondition(
                    axiom=axiom_type(
                        property_name=property_name,
                        on_class=self._classes_of_filler(filler=filler),
                        for_class=for_class,
                        ontology=self.ontology,
                    ),
                    value_class=None
                    if union is not None
                    else self._class_of_filler(filler=filler),
                )
        return None

    def _classes_of_filler(self, *, filler: rdflib.term.Node) -> str:
        """
        :param filler: The class expression a restriction restricts the values of a property to.
        :return: The class a value is checked against, or the tuple of the members of a union, as Python source.
        :raises UnionOfAnonymousClassesAsFiller: If the filler is a union holding an anonymous class expression.
        """
        graph = self.ontology.graph
        union = graph.value(filler, OWL.unionOf)
        if union is None:
            return self._class_of_filler(filler=filler)
        members = NamingRegistry._get_rdf_list(graph=graph, head=union)
        if not all(isinstance(member, rdflib.URIRef) for member in members):
            raise UnionOfAnonymousClassesAsFiller(
                union_name=NamingRegistry.uri_to_python_name(uri=filler, graph=graph)
            )
        return f"({', '.join(NamingRegistry.uri_to_python_name(uri=member) for member in members)})"

    def _class_of_filler(self, *, filler: rdflib.term.Node) -> str:
        """
        :param filler: A class expression that is not a union.
        :return: The name of its class: the named class an anonymous intersection is the definition of, otherwise the
            name of the expression.
        """
        name = NamingRegistry.uri_to_python_name(uri=filler, graph=self.ontology.graph)
        return self.ontology.description_for(description_name=name) or name

    def _add_condition(self, *, for_class: str, axiom: AxiomInfo, sufficient: bool):
        """
        Add a condition to a class: to its classification axioms when it is sufficient, to its necessary conditions
        otherwise.

        :param for_class: The name of the class.
        :param axiom: The condition.
        :param sufficient: Whether the condition is (part of) a sufficient condition of the class.
        """
        class_information = self.ontology.classes[for_class]
        if not sufficient:
            class_information.necessary_axioms_python.extend(axiom.conditions_python())
            return
        class_information.axioms_setup.extend(axiom.setup_statements())
        class_information.axioms.extend(axiom.conditions_eql())
        class_information.axioms_python.extend(axiom.conditions_python())

    def _add_complement_or_enumeration_condition(
        self, *, class_name: str, member: rdflib.term.Node, sufficient: bool
    ) -> bool:
        """
        Add the condition a member of an intersection stands for when it is a complement (``owl:complementOf``) or
        an enumeration of individuals (``owl:oneOf``).

        :param cls_name: The name of the class, or of the description, the intersection belongs to.
        :param member: The member of the intersection.
        :param sufficient: Whether the intersection is a sufficient condition.
        :return: Whether the member is a complement or an enumeration.
        :raises UntranslatedComplement: If the member is the complement of something other than a named class or
            a restriction.
        """
        graph = self.ontology.graph
        complemented = graph.value(member, OWL.complementOf)
        enumerated = graph.value(member, OWL.oneOf)
        if complemented is None and enumerated is None:
            return False
        for_class = (
            self.ontology.description_for(description_name=class_name) or class_name
        )
        if enumerated is not None:
            individuals = NamingRegistry._get_rdf_list(graph=graph, head=enumerated)
            self._add_condition(
                for_class=for_class,
                axiom=OneOfAxiomInfo(
                    individual_iris=[str(individual) for individual in individuals]
                ),
                sufficient=sufficient,
            )
            return True
        if isinstance(complemented, rdflib.URIRef):
            inner = SubClassAxiomInfo(
                sub_class=NamingRegistry.uri_to_python_name(uri=complemented)
            )
        else:
            restricted_property = graph.value(complemented, OWL.onProperty)
            restriction = (
                self._restriction_condition(
                    node=complemented,
                    for_class=for_class,
                    property_name=NamingRegistry.uri_to_python_name(
                        uri=restricted_property, graph=graph
                    ),
                )
                if restricted_property is not None
                else None
            )
            if restriction is None:
                raise UntranslatedComplement(
                    complemented_name=NamingRegistry.uri_to_python_name(
                        uri=complemented, graph=graph
                    )
                )
            inner = restriction.axiom
        self._add_condition(
            for_class=for_class,
            axiom=ComplementAxiomInfo(complemented=inner),
            sufficient=sufficient,
        )
        return True

    def _propagate_types(self):
        """
        Perform iterative propagation of domains and ranges along property hierarchy and inverses.
        """
        changed = True
        while changed:
            changed = False
            for property_name, inverse_name in self.property_maps.inverse_pairs:
                before_domain_count, before_range_count = (
                    len(self.property_maps.domains_by_property[property_name]),
                    len(self.property_maps.ranges_by_property[property_name]),
                )
                before_inverse_domain_count, before_inverse_range_count = (
                    len(self.property_maps.domains_by_property[inverse_name]),
                    len(self.property_maps.ranges_by_property[inverse_name]),
                )
                if not before_domain_count:
                    self.property_maps.domains_by_property[property_name].update(
                        self.property_maps.ranges_by_property[inverse_name]
                    )
                if not before_range_count:
                    self.property_maps.ranges_by_property[property_name].update(
                        self.property_maps.domains_by_property[inverse_name]
                    )
                if not before_inverse_domain_count:
                    self.property_maps.domains_by_property[inverse_name].update(
                        self.property_maps.ranges_by_property[property_name]
                    )
                if not before_inverse_range_count:
                    self.property_maps.ranges_by_property[inverse_name].update(
                        self.property_maps.domains_by_property[property_name]
                    )
                if (
                    len(self.property_maps.domains_by_property[property_name])
                    != before_domain_count
                    or len(self.property_maps.ranges_by_property[property_name])
                    != before_range_count
                    or len(self.property_maps.domains_by_property[inverse_name])
                    != before_inverse_domain_count
                    or len(self.property_maps.ranges_by_property[inverse_name])
                    != before_inverse_range_count
                ):
                    changed = True
            for (
                name,
                superproperties,
            ) in self.property_maps.superproperties_by_property.items():
                supers_and_equivalents = set(superproperties).union(
                    self.property_maps.equivalent_properties_by_property[name]
                )
                for superproperty in supers_and_equivalents:
                    if superproperty not in self.property_maps.domains_by_property:
                        continue
                    before_range_count, before_range_uri_count, before_domain_count = (
                        len(self.property_maps.ranges_by_property[name]),
                        len(self.property_maps.range_uris_by_property[name]),
                        len(self.property_maps.domains_by_property[name]),
                    )
                    if not before_range_count:
                        self.property_maps.ranges_by_property[name].update(
                            self.property_maps.ranges_by_property[superproperty]
                        )
                    if not before_range_uri_count:
                        self.property_maps.range_uris_by_property[name].update(
                            self.property_maps.range_uris_by_property[superproperty]
                        )
                    if not before_domain_count:
                        self.property_maps.domains_by_property[name].update(
                            self.property_maps.domains_by_property[superproperty]
                        )
                    if (
                        len(self.property_maps.ranges_by_property[name])
                        != before_range_count
                        or len(self.property_maps.range_uris_by_property[name])
                        != before_range_uri_count
                        or len(self.property_maps.domains_by_property[name])
                        != before_domain_count
                    ):
                        changed = True

    def _finalize_properties(self):
        """
        Update PropertyInfo objects with inferred domain and range information.
        """
        for name, property_information in self.ontology.properties.items():
            property_information.domains = sorted(property_information.domains)
            property_information.ranges = sorted(property_information.ranges)
            property_information.range_uris = [
                self.ontology.classes[range_name].uri
                for range_name in property_information.ranges
                if range_name in self.ontology.classes
            ]
            property_information.declared_domains = copy(property_information.domains)

    def _create_specialized_properties(self):
        """
        Create specialized versions of properties based on class-level restrictions.
        Used for narrowing property ranges in specific subclasses.
        """
        specialized_properties: Dict[str, PropertyInfo] = {}
        for (
            class_name,
            ranges_by_restricted_property,
        ) in self.ontology.property_restrictions.items():
            for property_name, range_names in ranges_by_restricted_property.items():
                base = self.ontology.properties.get(property_name)
                if not base or base.type != PropertyType.OBJECT_PROPERTY:
                    continue
                if range_names.issubset(
                    set(self.ontology.original_properties[property_name].ranges)
                ):
                    continue
                if len(range_names) > 1:
                    # Restricted to several classes (e.g. hasTopping some MozzarellaTopping and hasTopping some
                    # TomatoTopping), the property is narrowed to none of them; one field per class would declare it
                    # several times.
                    continue
                if class_name in base.declared_domains:
                    base.declared_domains.remove(class_name)
                for range_name in sorted(range_names):
                    if range_name in base.ranges:
                        base.ranges.remove(range_name)
                    if len(base.ranges) == 0:
                        for declared_domain in base.declared_domains:
                            if (
                                property_name
                                in self.ontology.classes[
                                    declared_domain
                                ].declared_properties
                            ):
                                self.ontology.classes[
                                    declared_domain
                                ].declared_properties.remove(property_name)
                        base.declared_domains = []
                    spec_key = f"{property_name}{{{range_name}}}"
                    if (
                        spec_key in self.ontology.properties
                        or spec_key in specialized_properties
                    ):
                        continue
                    specialized_properties[spec_key] = PropertyInfo(
                        name=property_name,
                        uri=base.uri,
                        type=PropertyType.OBJECT_PROPERTY,
                        domains=[class_name],
                        ranges=[range_name],
                        label=base.label,
                        comment=base.comment,
                        field_name=base.field_name,
                        descriptor_name=NamingRegistry.to_pascal_case(
                            name=base.descriptor_name or property_name
                        ),
                        superproperties=[property_name],
                        inverses=[],
                        inverse_of=None,
                        is_transitive=base.is_transitive,
                        declared_domains=[class_name],
                        is_specialized=True,
                    )
        self.ontology.properties.update(specialized_properties)

    def apply_predefined_overrides(
        self,
    ):
        """
        Apply manual type overrides for specific class properties.
        """
        for class_name, overrides in (
            self.ontology.predefined_data_types or {}
        ).items():
            if class_name == "Thing":
                class_name = self.ontology.base_class_name
            for field_name, python_type in overrides.items():
                target_property_name = next(
                    (
                        name
                        for name, property_information in self.ontology.properties.items()
                        if property_information.field_name == field_name
                    ),
                    None,
                )
                if not target_property_name:
                    logger.info(
                        f"[owl_to_python] Override not applied: property '{field_name}' not found"
                    )
                    continue
                property_information = self.ontology.properties[target_property_name]
                if property_information.type is None:
                    property_information.type = PropertyType.DATA_PROPERTY
                if property_information.type == PropertyType.DATA_PROPERTY:
                    property_information.data_type_hint_inner = python_type
                else:
                    property_information.object_range_hint = python_type
                property_information._predefined_data_type = True
                overridden_for = set(property_information._overrides_for)
                overridden_for.add(class_name)
                property_information._overrides_for = sorted(overridden_for)
                if class_name not in property_information.declared_domains:
                    property_information.declared_domains.append(class_name)
                logger.info(
                    f"[owl_to_python] Applied override: {class_name}.{field_name} -> {python_type}"
                )

    def compute_type_hints(self):
        """
        Compute Python type hints for all properties.
        Handles both object properties (referencing classes) and data properties (XSD types).
        """
        for property_information in self.ontology.properties.values():
            self._set_base_descriptors(property_information=property_information)
            if property_information.type == PropertyType.OBJECT_PROPERTY:
                self._set_object_range_hint(property_information=property_information)
            elif not (
                property_information._predefined_data_type
                and property_information.data_type_hint_inner
            ):
                self._set_data_type_hint(property_information=property_information)

    def _set_base_descriptors(self, *, property_information: PropertyInfo):
        """
        Determine the base descriptor classes for a property.
        :param info: The PropertyInfo to update.
        """
        bases = [
            self.ontology.properties[superproperty].descriptor_name
            for superproperty in property_information.superproperties
            if superproperty in self.ontology.properties
        ] or [PropertyDescriptor.__name__]
        if property_information.is_transitive:
            bases.append(TransitiveProperty.__name__)
        if property_information.inverse_of:
            bases.append(HasInverseProperty.__name__)
        if property_information.equivalent_properties:
            bases.append(HasEquivalentProperties.__name__)
        if property_information.disjoint_properties:
            bases.append(HasDisjointProperties.__name__)
        if property_information.is_symmetric:
            bases.append(SymmetricProperty.__name__)
        if property_information.is_asymmetric:
            bases.append(ASymmetricProperty.__name__)
        if property_information.is_reflexive:
            bases.append(ReflexiveProperty.__name__)
        if property_information.is_irreflexive:
            bases.append(IrreflexiveProperty.__name__)
        property_information.base_descriptors = bases

    def _set_object_range_hint(self, *, property_information: PropertyInfo):
        """
        Compute and set the object_range_hint for an ObjectProperty.
        :param info: The PropertyInfo to update.
        """
        ranges = list(property_information.ranges)
        if ranges:
            range_set = set(ranges)
            simplified = [
                range_name
                for range_name in sorted(range_set)
                if not any(
                    ancestor in range_set
                    for ancestor in self.ontology.class_ancestors.get(range_name, set())
                )
            ]
            ranges = simplified or ranges

        if len(ranges) > 1:
            for index, range_name in enumerate(ranges[:-1]):
                for other_range_name in ranges[index + 1 :]:
                    if (
                        other_range_name
                        in self.ontology.classes[range_name].all_base_classes
                    ):
                        try:
                            ranges.remove(other_range_name)
                        except ValueError:
                            pass
                    elif (
                        range_name
                        in self.ontology.classes[other_range_name].all_base_classes
                    ):
                        try:
                            ranges.remove(range_name)
                        except ValueError:
                            pass
        if len(ranges) > 1:
            property_information.object_range_hint = (
                f"Union[{', '.join(sorted(ranges))}]"
            )
        elif len(ranges) == 1:
            property_information.object_range_hint = ranges[0]
        else:
            logger.warning(
                f"[owl_to_python]: Could not infer object range type for property '{property_information.name}'. Using {self.ontology.base_class_name} instead."
            )
            property_information.object_range_hint = self.ontology.base_class_name

    def _set_data_type_hint(self, *, property_information: PropertyInfo):
        """
        Compute and set the data_type_hint_inner for a DataProperty.
        :param info: The PropertyInfo to update.
        """
        python_types: List[str] = []
        # 1. Try mapping from XSD URIs
        for uri in property_information.range_uris:
            try:
                if isinstance(uri, rdflib.URIRef) and uri in self.XSD_TO_PYTHON_TYPES:
                    python_types.append(self.XSD_TO_PYTHON_TYPES[uri])
            except Exception:
                pass

        # 2. Try mapping from range names if URI mapping failed
        if not python_types:
            python_types = self._map_range_names_to_python_types(
                range_names=property_information.ranges
            )

        if not python_types:
            logger.warning(
                f"[owl_to_python]: Could not infer data type for property '{property_information.name}'. Using str."
            )
            python_types.append("str")

        unique_types = list(OrderedSet(python_types))
        property_information.data_type_hint_inner = (
            f"Union[{', '.join(unique_types)}]"
            if len(unique_types) > 1
            else unique_types[0]
        )

    @staticmethod
    def _map_range_names_to_python_types(*, range_names: List[str]) -> List[str]:
        """Map OWL range names (as strings) to Python types."""
        python_types = []
        textual = [range_name.lower() for range_name in range_names]
        for range_text in textual:
            if range_text in (
                "string",
                "normalizedstring",
                "token",
                "language",
                "anyuri",
                "datetime",
                "date",
                "time",
            ):
                python_types.append("str")
            elif range_text in (
                "integer",
                "int",
                "long",
                "short",
                "byte",
                "nonnegativeinteger",
                "positiveinteger",
                "unsignedlong",
                "unsignedint",
                "unsignedshort",
                "unsignedbyte",
            ):
                python_types.append("int")
            elif range_text in ("float", "double", "decimal"):
                python_types.append("float")
            elif range_text == "boolean":
                python_types.append("bool")
        return python_types

    def find_implicit_subtypes(self, *, property_order: List[str]):
        """
        Identify implicit subtype or role relationships between classes based on property commonality.
        """
        candidate_parents: Dict[
            str,
            Dict[Tuple[str, ...], List[Tuple[str, SubsumptionType, Dict[str, str]]]],
        ] = defaultdict(lambda: defaultdict(list))
        most_matched_properties_by_child: Dict[str, int] = {
            class_name: 0 for class_name in self.ontology.classes
        }
        for parent_name, parent_information in self.ontology.classes.items():
            for child_name, child_information in self.ontology.classes.items():
                if not self._is_subsumption_candidate(
                    parent_name=parent_name,
                    parent_information=parent_information,
                    child_name=child_name,
                    child_information=child_information,
                ):
                    continue

                matched_properties, child_matched_properties = (
                    self._get_matched_properties(
                        parent_information=parent_information,
                        child_information=child_information,
                    )
                )
                if not matched_properties:
                    continue

                subsumption_type = self._determine_subsumption_type(
                    matched_properties=matched_properties,
                    parent_information=parent_information,
                    child_information=child_information,
                )
                if subsumption_type:
                    matched_properties = tuple(sorted(matched_properties))
                    if (
                        len(matched_properties)
                        >= most_matched_properties_by_child[child_name]
                    ):
                        most_matched_properties_by_child[child_name] = len(
                            matched_properties
                        )
                        candidate_parents[child_name][matched_properties].append(
                            (parent_name, subsumption_type, child_matched_properties)
                        )
                    continue
        for child_name, candidates in candidate_parents.items():
            if len(candidates) == 1:
                parent_name, subsumption_type, _ = next(iter(candidates.values()))[0]
                self._apply_implicit_subsumption(
                    parent_name=parent_name,
                    child_name=child_name,
                    subsumption_type=subsumption_type,
                )
                continue
            min_mro = float("inf")
            selected = None
            for matched_properties, subsumption_data in candidates.items():
                for (
                    parent_name,
                    subsumption_type,
                    child_matched_properties,
                ) in subsumption_data:
                    parent_property_name = list(child_matched_properties.keys())[0]
                    child_property_name = child_matched_properties[parent_property_name]
                    length = self.ontology.properties_inheritance_path_length(
                        child_class=child_property_name,
                        parent_class=parent_property_name,
                    )
                    if length < min_mro:
                        min_mro = length
                        selected = (parent_name, subsumption_type)
            if selected:
                parent_name, subsumption_type = selected
                self._apply_implicit_subsumption(
                    parent_name=parent_name,
                    child_name=child_name,
                    subsumption_type=subsumption_type,
                )
                continue

    def _is_subsumption_candidate(
        self,
        *,
        parent_name: str,
        parent_information: ClassInfo,
        child_name: str,
        child_information: ClassInfo,
    ) -> bool:
        """Check if parent and child are candidates for implicit subsumption."""
        if parent_name == child_name:
            return False
        if parent_name in child_information.all_base_classes_including_role_takers:
            return False
        if child_name in parent_information.all_base_classes_including_role_takers:
            return False
        base_classes = [
            base_class
            for base_class in child_information.base_classes
            if base_class != self.ontology.base_class_name
            and base_class != Symbol.__name__
            and base_class != self.ontology.role_class_name
        ]
        if len(base_classes) >= 1:
            return False
        return True

    def _get_matched_properties(
        self,
        *,
        parent_information: ClassInfo,
        child_information: ClassInfo,
    ) -> Tuple[Set[str], Dict[str, str]]:
        """
        Find property base names that are compatible between parent and child.
        """

        parent_properties = self.declared_and_restricted_properties(
            class_name=parent_information.name
        )
        child_properties = self.declared_and_restricted_properties(
            class_name=child_information.name
        )

        parent_property_names = [
            property_name.split("{")[0] for property_name in parent_properties
        ]
        child_property_names = [
            property_name.split("{")[0] for property_name in child_properties
        ]

        matched_property_names = set(parent_property_names).intersection(
            set(child_property_names)
        )

        for matched_property_name in copy(matched_property_names):
            parent_index = parent_property_names.index(matched_property_name)
            child_index = child_property_names.index(matched_property_name)
            parent_property_name = parent_properties[parent_index]
            child_property_name = child_properties[child_index]
            if (
                self.ontology.properties[parent_property_name].is_specialized
                and not self.ontology.properties[child_property_name].is_specialized
            ):
                matched_property_names.remove(matched_property_name)

        # Re-verify based on original logic: check all combinations of parent/child properties
        # and remove/add base name based on range and superproperty compatibility.
        child_matched_property_names = {}
        for parent_property_name in parent_properties:
            parent_base_name = parent_property_name.split("{")[0]
            for child_property_name in child_properties:
                child_property_information, parent_property_information = (
                    self.ontology.properties.get(child_property_name),
                    self.ontology.properties.get(parent_property_name),
                )
                if not child_property_information or not parent_property_information:
                    continue
                if (
                    child_property_information.type == PropertyType.DATA_PROPERTY
                    or parent_property_information.type == PropertyType.DATA_PROPERTY
                ):
                    continue
                if (
                    parent_base_name
                    not in child_property_information.all_superproperties
                ):
                    continue

                child_property_range, parent_property_range = (
                    child_property_information.object_range_hint,
                    parent_property_information.object_range_hint,
                )
                if parent_property_range not in self.ontology.class_ancestors.get(
                    child_property_range, set()
                ).union({child_property_range}):
                    if parent_base_name in matched_property_names:
                        matched_property_names.remove(parent_base_name)
                    continue
                matched_property_names.add(parent_base_name)
                child_matched_property_names[parent_base_name] = [
                    pname for pname in child_properties if child_property_name in pname
                ][0]

        return matched_property_names, child_matched_property_names

    @lru_cache
    def declared_and_restricted_properties(self, *, class_name: str) -> List[str]:
        """Get declared properties of a class, including those from restrictions."""
        class_information = self.ontology.classes[class_name]
        class_properties = copy(class_information.declared_properties)
        class_property_names = [
            property_name.split("{")[0] for property_name in class_properties
        ]
        if class_information.name in self.ontology.property_restrictions:
            for property_name, ranges in self.ontology.property_restrictions[
                class_information.name
            ].items():
                if property_name in class_property_names:
                    continue
                class_properties.append(property_name)
        return class_properties

    def _determine_subsumption_type(
        self,
        *,
        matched_properties: Set[str],
        parent_information: ClassInfo,
        child_information: ClassInfo,
    ) -> Optional[SubsumptionType]:
        """Determine if the relationship is a SUBTYPE or a ROLE."""
        parent_properties = self.declared_and_restricted_properties(
            class_name=parent_information.name
        )
        parent_property_names = {
            property_name.split("{")[0] for property_name in parent_properties
        }

        if matched_properties == parent_property_names:
            if parent_information.role_taker:
                if (
                    not child_information.role_taker
                    or child_information.role_taker.class_name
                    != parent_information.role_taker.class_name
                ):
                    return None
            return SubsumptionType.SUBTYPE
        return SubsumptionType.ROLE

    def _apply_implicit_subsumption(
        self,
        *,
        parent_name: str,
        child_name: str,
        subsumption_type: SubsumptionType,
    ):
        """Apply the determined subsumption to the child class."""
        child_information = self.ontology.classes[child_name]
        parent_information = self.ontology.classes[parent_name]
        if self.ontology.base_class_name in child_information.base_classes:
            child_information.base_classes.remove(self.ontology.base_class_name)

        if subsumption_type == SubsumptionType.ROLE:
            self._apply_role_subsumption(
                child_information=child_information, parent_name=parent_name
            )
        else:
            self._apply_subtype_subsumption(
                child_information=child_information,
                parent_name=parent_name,
                parent_information=parent_information,
            )

    def _apply_role_subsumption(
        self, *, child_information: ClassInfo, parent_name: str
    ):
        """Add a role taker relationship to the class."""
        child_information.role_taker = RoleTakerInfo(
            class_name=parent_name,
            field_name=NamingRegistry.to_snake_case(name=parent_name),
        )
        if self.ontology.role_class_name not in child_information.all_base_classes:
            child_information.base_classes = [
                self.ontology.role_class_name
            ] + child_information.base_classes
            child_information.all_base_classes = [
                self.ontology.role_class_name
            ] + child_information.all_base_classes
            if Symbol.__name__ not in child_information.base_classes:
                child_information.base_classes.append(Symbol.__name__)
                child_information.all_base_classes.append(Symbol.__name__)
                child_information.all_base_classes_including_role_takers.append(
                    Symbol.__name__
                )
        child_information.all_base_classes_including_role_takers.append(parent_name)

    @staticmethod
    def _apply_subtype_subsumption(
        *, child_information: ClassInfo, parent_name: str, parent_information: ClassInfo
    ):
        """Make the class a subtype of another class."""
        child_information.base_classes = []
        if parent_name not in child_information.base_classes:
            child_information.base_classes.append(parent_name)
            child_information.all_base_classes.append(parent_name)
            child_information.all_base_classes_including_role_takers.append(parent_name)

        # Remove redundant properties already declared in the parent
        parent_properties = set(parent_information.declared_properties)
        child_information.declared_properties = [
            property_name
            for property_name in child_information.declared_properties
            if property_name not in parent_properties
        ]

    def __hash__(self):
        """
        Inference engines are hashed by identity.
        """
        return hash(id(self))
