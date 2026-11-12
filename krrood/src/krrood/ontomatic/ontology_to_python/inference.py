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

    dom_map: Dict[str, Set[str]] = field(default_factory=dict)
    """
    Map of property names to sets of domain class names.
    """
    declared_dom_map: Dict[str, Set[str]] = field(default_factory=dict)
    """
    Map of property names to sets of declared domain class names. These are the classes where the property is declared.
    """
    rng_map: Dict[str, Set[str]] = field(default_factory=dict)
    """
    Map of property names to sets of range class names.
    """
    rng_uri_map: Dict[str, Set[rdflib.term.URIRef]] = field(default_factory=dict)
    """
    Map of property names to sets of range URIs.
    """
    super_map: Dict[str, List[str]] = field(default_factory=dict)
    """
    Map of property names to lists of superclass names.
    """
    inverse_pairs: List[Tuple[str, str]] = field(default_factory=list)
    """
    List of tuples representing inverse property pairs (property_name, inverse_property_name).
    """
    equivalent_map: Dict[str, List[str]] = field(default_factory=dict)
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
        dom_map = {n: set(p.domains) for n, p in properties.items()}
        declared_dom_map = {n: set(p.domains) for n, p in properties.items()}
        rng_map = {n: set(p.ranges) for n, p in properties.items()}
        rng_uri_map = {n: set(p.range_uris) for n, p in properties.items()}
        super_map = {n: list(p.superproperties) for n, p in properties.items()}
        inverse_pairs = [
            (n, inv)
            for n, p in properties.items()
            for inv in p.inverses
            if inv in properties
        ]
        equivalent_map = {
            n: list(p.equivalent_properties) for n, p in properties.items()
        }
        return cls(
            dom_map=dom_map,
            declared_dom_map=declared_dom_map,
            rng_map=rng_map,
            rng_uri_map=rng_uri_map,
            super_map=super_map,
            inverse_pairs=inverse_pairs,
            equivalent_map=equivalent_map,
            properties=properties,
        )


@dataclass(kw_only=True)
class InferenceEngine:
    """Engine for performing ontological inference and computing class/property relationships."""

    onto: OntologyInfo
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
            properties=self.onto.properties
        )

    def compute_ancestors(self):
        """
        Compute full ancestor sets for each class (transitive closure).
        """
        # Compute full ancestor sets for each class (transitive closure)
        name_to_bases = {
            name: set(info.base_classes) for name, info in self.onto.classes.items()
        }
        for info in self.onto.classes.values():
            ancestors = set()
            stack = list(info.base_classes)
            while stack:
                base = stack.pop()
                if base in ancestors:
                    continue
                ancestors.add(base)
                stack.extend(name_to_bases.get(base, []))
            info.all_base_classes = sorted(ancestors)

    def infer_properties(
        self,
    ):
        """
        Main entry point for property inference.
        Propagates domains, ranges, and handles restrictions and inverses.
        """
        self._infer_properties_data_from_restrictions()

        for prop_name, prop_info in self.onto.properties.items():
            if prop_name == "roleFor":
                continue
            if prop_info.type == PropertyType.DATA_PROPERTY:
                continue
            PropertyDomainAndRangeInference(
                property_info=prop_info, onto=self.onto
            ).complete()

        self._finalize_properties()
        self._add_property_chain_axioms()
        self._create_specialized_properties()

    def _add_property_chain_axioms(self):
        """
        Add property chain axioms to properties based on owl:propertyChainAxiom.
        """
        for prop_name, prop_info in self.onto.properties.items():
            prop_uri = rdflib.URIRef(prop_info.uri)
            for chain in self.onto.graph.objects(prop_uri, OWL.propertyChainAxiom):
                items = []
                node = chain
                while node and node != RDF.nil:
                    first = self.onto.graph.value(node, RDF.first)
                    if first:
                        name = NamingRegistry.uri_to_python_name(uri=first)
                        if name not in self.onto.properties:
                            raise ValueError(
                                f"Property chain axiom references unknown property {name}"
                            )
                        items.append(self.onto.properties[name].descriptor_name)
                    node = self.onto.graph.value(node, RDF.rest)
                if items:
                    prop_info.chain_axioms.append(items)

    def _infer_properties_data_from_restrictions(self):
        """
        Walk through all classes and their restrictions in the graph and update declared_dom_map accordingly.
        """
        covered_restrictions = set()
        # Walk class restrictions
        for cls_uri in self.onto.graph.subjects(RDF.type, OWL.Class):
            # if isinstance(cls_uri, rdflib.BNode):
            #     continue
            cls_name = NamingRegistry.uri_to_python_name(
                uri=cls_uri, graph=self.onto.graph
            )
            for_class = self.onto.description_for(description_name=cls_name) or cls_name
            if not self._is_extracted(class_node=cls_uri, cls_name=cls_name):
                # Its conditions are those of the intersection it duplicates.
                for coll in self.onto.graph.objects(cls_uri, OWL.intersectionOf):
                    covered_restrictions.update(
                        NamingRegistry._get_rdf_list(graph=self.onto.graph, head=coll)
                    )
                continue

            # direct subclass restrictions are necessary conditions of the class
            for restr in self.onto.graph.objects(cls_uri, RDFS.subClassOf):
                self._restrictions_handler(
                    for_class=cls_name, node=restr, sufficient=False
                )
                # If restriction mentions a property, count this class as declared domain for that property
                covered_restrictions.add(restr)
                on_prop = self.onto.graph.value(restr, OWL.onProperty)
                if on_prop:
                    prop_name = NamingRegistry.uri_to_python_name(
                        uri=on_prop, graph=self.onto.graph
                    )
                    self.property_maps.declared_dom_map[prop_name].add(cls_name)

            # restrictions inside intersectionOf
            intersection = []
            sufficient = self._is_sufficient_class_expression(class_node=cls_uri)
            for coll in self.onto.graph.objects(cls_uri, OWL.intersectionOf):
                node = coll
                while node and node != RDF.nil:
                    first = self.onto.graph.value(node, RDF.first)
                    if self._add_complement_or_enumeration_condition(
                        cls_name=cls_name, member=first, sufficient=sufficient
                    ):
                        covered_restrictions.add(
                            self.onto.graph.value(first, OWL.complementOf)
                        )
                        node = self.onto.graph.value(node, RDF.rest)
                        continue
                    restriction_added = self._restrictions_handler(
                        for_class=cls_name, node=first, sufficient=sufficient
                    )
                    covered_restrictions.add(first)
                    on_prop = (
                        self.onto.graph.value(first, OWL.onProperty) if first else None
                    )
                    if on_prop and restriction_added:
                        prop_name = NamingRegistry.uri_to_python_name(
                            uri=on_prop, graph=self.onto.graph
                        )
                        self.property_maps.declared_dom_map[prop_name].add(for_class)
                    elif not on_prop:
                        first_name = NamingRegistry.uri_to_python_name(
                            uri=first, graph=self.onto.graph
                        )
                        intersection.append(first_name)
                    node = self.onto.graph.value(node, RDF.rest)
            self._add_named_conjunct_conditions(
                for_class=for_class, named_conjuncts=intersection, sufficient=sufficient
            )
        # Standalone Restrictions
        for restr in self.onto.graph.subjects(RDF.type, OWL.Restriction):
            if restr in covered_restrictions:
                continue
            covered_restrictions.add(restr)
            on_prop = self.onto.graph.value(restr, OWL.onProperty)
            if on_prop is None:
                continue
            for_class = None
            for subj, pred in self.onto.graph.subject_predicates(restr):
                if pred == OWL.equivalentClass:
                    for_class = NamingRegistry.uri_to_python_name(
                        uri=subj, graph=self.onto.graph
                    )
                    break
            if not for_class:
                for_class = self.onto.graph.value(restr, RDFS.subClassOf)
                if for_class:
                    for_class = NamingRegistry.uri_to_python_name(
                        uri=for_class, graph=self.onto.graph
                    )
            if not for_class:
                raise ValueError(f"Could not determine class for restriction {restr}")
            # Either an owl:equivalentClass definition or a general class axiom (restriction rdfs:subClassOf class),
            # both are sufficient conditions of the class.
            self._restrictions_handler(for_class=for_class, node=restr, sufficient=True)
            prop_name = NamingRegistry.uri_to_python_name(
                uri=on_prop, graph=self.onto.graph
            )
            for_class = (
                self.onto.description_for(description_name=for_class) or for_class
            )
            self.property_maps.declared_dom_map[prop_name].add(for_class)

    def _is_extracted(self, *, class_node: rdflib.term.Node, cls_name: str) -> bool:
        """
        :param class_node: A class.
        :param cls_name: Its name.
        :return: False for an anonymous intersection that was not extracted because another one of the same members
            and name was, such as a filler that stands for the named class it has the definition of.
        """
        if not isinstance(class_node, rdflib.BNode) or not list(
            self.onto.graph.objects(class_node, OWL.intersectionOf)
        ):
            return True
        description_for = self.onto.description_for(description_name=cls_name)
        info = (
            self.onto.class_descriptions[description_for]
            if description_for is not None
            else self.onto.classes.get(cls_name)
        )
        return info is None or info.uri == str(class_node)

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
            for superclass in self.onto.graph.objects(class_node, RDFS.subClassOf)
        ):
            return True
        return (
            (None, OWL.equivalentClass, class_node) in self.onto.graph
            or (class_node, OWL.equivalentClass, None) in self.onto.graph
            or not (
                any(self.onto.graph.objects(class_node, RDFS.subClassOf))
                or any(self.onto.graph.subjects(RDFS.subClassOf, class_node))
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
        cls_info = self.onto.classes.get(for_class)
        if cls_info is None:
            return
        for conjunct in reversed(named_conjuncts):
            if conjunct == for_class or conjunct not in self.onto.classes:
                continue
            subclass_axiom = SubClassAxiomInfo(sub_class=conjunct)
            python_condition = subclass_axiom.conditions_python()[0]
            if not sufficient:
                if python_condition not in cls_info.necessary_axioms_python:
                    cls_info.necessary_axioms_python.insert(0, python_condition)
                continue
            if not cls_info.axioms:
                # The class has no sufficient restriction conditions, a named conjunct alone is a plain subsumption.
                continue
            if python_condition not in cls_info.axioms_python:
                cls_info.axioms_setup[0:0] = subclass_axiom.setup_statements()
                cls_info.axioms.insert(0, subclass_axiom.conditions_eql()[0])
                cls_info.axioms_python.insert(0, python_condition)

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
        on_prop = self.onto.graph.value(node, OWL.onProperty)
        if not on_prop:
            return True
        description_for = self.onto.description_for(description_name=for_class)
        for_class = description_for or for_class
        prop_name = NamingRegistry.uri_to_python_name(
            uri=on_prop, graph=self.onto.graph
        )
        if prop_name in self.onto.properties:
            self.property_maps.dom_map[prop_name].add(for_class)
        restriction = self._restriction_condition(
            node=node, for_class=for_class, prop_name=prop_name
        )
        if restriction:
            axiom = restriction.axiom
            value_type = restriction.value_class
            try:
                rng_name = value_type
                if prop_name == "roleFor":
                    cls_info = self.onto.classes.get(for_class)
                    if cls_info:
                        cls_info.role_taker = RoleTakerInfo(
                            class_name=rng_name,
                            field_name=NamingRegistry.to_snake_case(name=rng_name),
                        )
                    return True
                self._add_condition(
                    for_class=for_class, axiom=axiom, sufficient=sufficient
                )
                if rng_name is None:
                    if self.onto.original_properties[prop_name].ranges:
                        rng_name = self.onto.original_properties[prop_name].ranges[0]
                    else:
                        return True
                existing_ranges = self.property_maps.rng_map.get(prop_name, set())
                contesting_ranges = set(existing_ranges)
                for dom_cls, pred_obj in self.onto.property_restrictions.items():
                    if prop_name in pred_obj:
                        for er in existing_ranges:
                            if er in pred_obj[prop_name]:
                                contesting_ranges.remove(er)
                if self.onto.classes[for_class].role_taker:
                    role_taker = self.onto.classes[for_class].role_taker.class_name
                    if (
                        role_taker in self.onto.original_properties[prop_name].domains
                        and rng_name in self.onto.original_properties[prop_name].ranges
                    ):
                        self.property_maps.dom_map[prop_name].remove(for_class)
                        self.onto.property_restrictions.setdefault(
                            for_class, {}
                        ).setdefault(prop_name, set()).add(rng_name)
                        return False
                if contesting_ranges and not any(
                    cr in self.onto.classes[rng_name].all_base_classes + [rng_name]
                    for cr in contesting_ranges
                ):
                    self.property_maps.dom_map[prop_name].remove(for_class)
                    return False

                self.property_maps.rng_map[prop_name].add(rng_name)
                rng_uri = (
                    value_type
                    if value_type is not None
                    else rdflib.URIRef(self.onto.classes[rng_name].uri)
                )
                self.property_maps.rng_uri_map[prop_name].add(rng_uri)
                self.onto.property_restrictions.setdefault(for_class, {}).setdefault(
                    prop_name, set()
                ).add(rng_name)
                # The inverse of a restriction (e.g. Organization <- hasEmployee some Employee from
                # Person and (worksFor some Organization) SubClassOf Employee) is not entailed, so it only
                # contributes to the attribute types and never becomes a classification axiom of the range class.
                for inverse in self.onto.properties[prop_name].inverses:
                    self.onto.property_restrictions.setdefault(rng_name, {}).setdefault(
                        inverse, set()
                    ).add(for_class)
            except Exception as e:
                logger.warning(f"[owl_to_python] Error processing restriction: {e}")
            return True

    def _restriction_condition(
        self, *, node: rdflib.term.Node, for_class: str, prop_name: str
    ) -> Optional[RestrictionCondition]:
        """
        :param node: A restriction.
        :param for_class: The class the restriction is a condition of.
        :param prop_name: The name of the property the restriction is on.
        :return: The condition the restriction puts on the class, or None for a kind of restriction that is not
            translated.
        """
        graph = self.onto.graph
        if graph.value(node, OWL.hasValue):
            has_value = graph.value(node, OWL.hasValue)
            value_type = [
                v
                for v in graph.objects(has_value, RDF.type)
                if v not in (OWL.NamedIndividual, OWL.Thing)
            ][0]
            return RestrictionCondition(
                axiom=HasValueAxiomInfo(
                    property_name=prop_name,
                    value=has_value,
                    for_class=for_class,
                    onto=self.onto,
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
                        property_name=prop_name,
                        quantity=graph.value(node, qualified).toPython(),
                        on_class=on_class,
                        for_class=for_class,
                        onto=self.onto,
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
                        property_name=prop_name,
                        quantity=graph.value(node, unqualified).toPython(),
                        for_class=for_class,
                        onto=self.onto,
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
                        property_name=prop_name,
                        on_class=self._classes_of_filler(filler=filler),
                        for_class=for_class,
                        onto=self.onto,
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
        graph = self.onto.graph
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
        name = NamingRegistry.uri_to_python_name(uri=filler, graph=self.onto.graph)
        return self.onto.description_for(description_name=name) or name

    def _add_condition(self, *, for_class: str, axiom: AxiomInfo, sufficient: bool):
        """
        Add a condition to a class: to its classification axioms when it is sufficient, to its necessary conditions
        otherwise.

        :param for_class: The name of the class.
        :param axiom: The condition.
        :param sufficient: Whether the condition is (part of) a sufficient condition of the class.
        """
        cls_info = self.onto.classes[for_class]
        if not sufficient:
            cls_info.necessary_axioms_python.extend(axiom.conditions_python())
            return
        cls_info.axioms_setup.extend(axiom.setup_statements())
        cls_info.axioms.extend(axiom.conditions_eql())
        cls_info.axioms_python.extend(axiom.conditions_python())

    def _add_complement_or_enumeration_condition(
        self, *, cls_name: str, member: rdflib.term.Node, sufficient: bool
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
        graph = self.onto.graph
        complemented = graph.value(member, OWL.complementOf)
        enumerated = graph.value(member, OWL.oneOf)
        if complemented is None and enumerated is None:
            return False
        for_class = self.onto.description_for(description_name=cls_name) or cls_name
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
            on_prop = graph.value(complemented, OWL.onProperty)
            restriction = (
                self._restriction_condition(
                    node=complemented,
                    for_class=for_class,
                    prop_name=NamingRegistry.uri_to_python_name(
                        uri=on_prop, graph=graph
                    ),
                )
                if on_prop is not None
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
            for a, b in self.property_maps.inverse_pairs:
                before_da, before_ra = (
                    len(self.property_maps.dom_map[a]),
                    len(self.property_maps.rng_map[a]),
                )
                before_db, before_rb = (
                    len(self.property_maps.dom_map[b]),
                    len(self.property_maps.rng_map[b]),
                )
                if not before_da:
                    self.property_maps.dom_map[a].update(self.property_maps.rng_map[b])
                if not before_ra:
                    self.property_maps.rng_map[a].update(self.property_maps.dom_map[b])
                if not before_db:
                    self.property_maps.dom_map[b].update(self.property_maps.rng_map[a])
                if not before_rb:
                    self.property_maps.rng_map[b].update(self.property_maps.dom_map[a])
                if (
                    len(self.property_maps.dom_map[a]) != before_da
                    or len(self.property_maps.rng_map[a]) != before_ra
                    or len(self.property_maps.dom_map[b]) != before_db
                    or len(self.property_maps.rng_map[b]) != before_rb
                ):
                    changed = True
            for name, supers in self.property_maps.super_map.items():
                supers_and_equivalents = set(supers).union(
                    self.property_maps.equivalent_map[name]
                )
                for sp in supers_and_equivalents:
                    if sp not in self.property_maps.dom_map:
                        continue
                    before_range_len, before_range_uri_len, before_domain_len = (
                        len(self.property_maps.rng_map[name]),
                        len(self.property_maps.rng_uri_map[name]),
                        len(self.property_maps.dom_map[name]),
                    )
                    if not before_range_len:
                        self.property_maps.rng_map[name].update(
                            self.property_maps.rng_map[sp]
                        )
                    if not before_range_uri_len:
                        self.property_maps.rng_uri_map[name].update(
                            self.property_maps.rng_uri_map[sp]
                        )
                    if not before_domain_len:
                        self.property_maps.dom_map[name].update(
                            self.property_maps.dom_map[sp]
                        )
                    if (
                        len(self.property_maps.rng_map[name]) != before_range_len
                        or len(self.property_maps.rng_uri_map[name])
                        != before_range_uri_len
                        or len(self.property_maps.dom_map[name]) != before_domain_len
                    ):
                        changed = True

    def _finalize_properties(self):
        """
        Update PropertyInfo objects with inferred domain and range information.
        """
        for name, info in self.onto.properties.items():
            info.domains = sorted(info.domains)
            info.ranges = sorted(info.ranges)
            info.range_uris = [
                self.onto.classes[r].uri for r in info.ranges if r in self.onto.classes
            ]
            info.declared_domains = copy(info.domains)

    def _create_specialized_properties(self):
        """
        Create specialized versions of properties based on class-level restrictions.
        Used for narrowing property ranges in specific subclasses.
        """
        specialized_props: Dict[str, PropertyInfo] = {}
        for cls_name, props in self.onto.property_restrictions.items():
            for prop_name, rng_names in props.items():
                base = self.onto.properties.get(prop_name)
                if not base or base.type != PropertyType.OBJECT_PROPERTY:
                    continue
                if rng_names.issubset(
                    set(self.onto.original_properties[prop_name].ranges)
                ):
                    continue
                if len(rng_names) > 1:
                    # Restricted to several classes (e.g. hasTopping some MozzarellaTopping and hasTopping some
                    # TomatoTopping), the property is narrowed to none of them; one field per class would declare it
                    # several times.
                    continue
                if cls_name in base.declared_domains:
                    base.declared_domains.remove(cls_name)
                for rng_name in sorted(rng_names):
                    if rng_name in base.ranges:
                        base.ranges.remove(rng_name)
                    if len(base.ranges) == 0:
                        for dd in base.declared_domains:
                            if prop_name in self.onto.classes[dd].declared_properties:
                                self.onto.classes[dd].declared_properties.remove(
                                    prop_name
                                )
                        base.declared_domains = []
                    spec_key = f"{prop_name}{{{rng_name}}}"
                    if (
                        spec_key in self.onto.properties
                        or spec_key in specialized_props
                    ):
                        continue
                    specialized_props[spec_key] = PropertyInfo(
                        name=prop_name,
                        uri=base.uri,
                        type=PropertyType.OBJECT_PROPERTY,
                        domains=[cls_name],
                        ranges=[rng_name],
                        label=base.label,
                        comment=base.comment,
                        field_name=base.field_name,
                        descriptor_name=NamingRegistry.to_pascal_case(
                            name=base.descriptor_name or prop_name
                        ),
                        superproperties=[prop_name],
                        inverses=[],
                        inverse_of=None,
                        is_transitive=base.is_transitive,
                        declared_domains=[cls_name],
                        is_specialized=True,
                    )
        self.onto.properties.update(specialized_props)

    def apply_predefined_overrides(
        self,
    ):
        """
        Apply manual type overrides for specific class properties.
        """
        for cls_name, overrides in (self.onto.predefined_data_types or {}).items():
            if cls_name == "Thing":
                cls_name = self.onto.base_cls_name
            for field_snake, py_type in overrides.items():
                target_prop_name = next(
                    (
                        n
                        for n, p in self.onto.properties.items()
                        if p.field_name == field_snake
                    ),
                    None,
                )
                if not target_prop_name:
                    logger.info(
                        f"[owl_to_python] Override not applied: property '{field_snake}' not found"
                    )
                    continue
                p = self.onto.properties[target_prop_name]
                if p.type is None:
                    p.type = PropertyType.DATA_PROPERTY
                if p.type == PropertyType.DATA_PROPERTY:
                    p.data_type_hint_inner = py_type
                else:
                    p.object_range_hint = py_type
                p._predefined_data_type = True
                ov = set(p._overrides_for)
                ov.add(cls_name)
                p._overrides_for = sorted(ov)
                if cls_name not in p.declared_domains:
                    p.declared_domains.append(cls_name)
                logger.info(
                    f"[owl_to_python] Applied override: {cls_name}.{field_snake} -> {py_type}"
                )

    def compute_type_hints(self):
        """
        Compute Python type hints for all properties.
        Handles both object properties (referencing classes) and data properties (XSD types).
        """
        for info in self.onto.properties.values():
            self._set_base_descriptors(info=info)
            if info.type == PropertyType.OBJECT_PROPERTY:
                self._set_object_range_hint(info=info)
            elif not (info._predefined_data_type and info.data_type_hint_inner):
                self._set_data_type_hint(info=info)

    def _set_base_descriptors(self, *, info: PropertyInfo):
        """
        Determine the base descriptor classes for a property.
        :param info: The PropertyInfo to update.
        """
        bases = [
            self.onto.properties[sp].descriptor_name
            for sp in info.superproperties
            if sp in self.onto.properties
        ] or [PropertyDescriptor.__name__]
        if info.is_transitive:
            bases.append(TransitiveProperty.__name__)
        if info.inverse_of:
            bases.append(HasInverseProperty.__name__)
        if info.equivalent_properties:
            bases.append(HasEquivalentProperties.__name__)
        if info.disjoint_properties:
            bases.append(HasDisjointProperties.__name__)
        if info.is_symmetric:
            bases.append(SymmetricProperty.__name__)
        if info.is_asymmetric:
            bases.append(ASymmetricProperty.__name__)
        if info.is_reflexive:
            bases.append(ReflexiveProperty.__name__)
        if info.is_irreflexive:
            bases.append(IrreflexiveProperty.__name__)
        info.base_descriptors = bases

    def _set_object_range_hint(self, *, info: PropertyInfo):
        """
        Compute and set the object_range_hint for an ObjectProperty.
        :param info: The PropertyInfo to update.
        """
        ranges = list(info.ranges)
        if ranges:
            rng_set = set(ranges)
            simplified = [
                r
                for r in sorted(rng_set)
                if not any(
                    a in rng_set for a in self.onto.class_ancestors.get(r, set())
                )
            ]
            ranges = simplified or ranges

        if len(ranges) > 1:
            for i, r in enumerate(ranges[:-1]):
                for r2 in ranges[i + 1 :]:
                    if r2 in self.onto.classes[r].all_base_classes:
                        try:
                            ranges.remove(r2)
                        except ValueError:
                            pass
                    elif r in self.onto.classes[r2].all_base_classes:
                        try:
                            ranges.remove(r)
                        except ValueError:
                            pass
        if len(ranges) > 1:
            info.object_range_hint = f"Union[{', '.join(sorted(ranges))}]"
        elif len(ranges) == 1:
            info.object_range_hint = ranges[0]
        else:
            logger.warning(
                f"[owl_to_python]: Could not infer object range type for property '{info.name}'. Using {self.onto.base_cls_name} instead."
            )
            info.object_range_hint = self.onto.base_cls_name

    def _set_data_type_hint(self, *, info: PropertyInfo):
        """
        Compute and set the data_type_hint_inner for a DataProperty.
        :param info: The PropertyInfo to update.
        """
        py_types: List[str] = []
        # 1. Try mapping from XSD URIs
        for uri in info.range_uris:
            try:
                if isinstance(uri, rdflib.URIRef) and uri in self.XSD_TO_PYTHON_TYPES:
                    py_types.append(self.XSD_TO_PYTHON_TYPES[uri])
            except Exception:
                pass

        # 2. Try mapping from range names if URI mapping failed
        if not py_types:
            py_types = self._map_range_names_to_python_types(range_names=info.ranges)

        if not py_types:
            logger.warning(
                f"[owl_to_python]: Could not infer data type for property '{info.name}'. Using str."
            )
            py_types.append("str")

        unique_types = list(OrderedSet(py_types))
        info.data_type_hint_inner = (
            f"Union[{', '.join(unique_types)}]"
            if len(unique_types) > 1
            else unique_types[0]
        )

    @staticmethod
    def _map_range_names_to_python_types(*, range_names: List[str]) -> List[str]:
        """Map OWL range names (as strings) to Python types."""
        py_types = []
        textual = [r.lower() for r in range_names]
        for t in textual:
            if t in (
                "string",
                "normalizedstring",
                "token",
                "language",
                "anyuri",
                "datetime",
                "date",
                "time",
            ):
                py_types.append("str")
            elif t in (
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
                py_types.append("int")
            elif t in ("float", "double", "decimal"):
                py_types.append("float")
            elif t == "boolean":
                py_types.append("bool")
        return py_types

    def find_implicit_subtypes(self, *, props_order: List[str]):
        """
        Identify implicit subtype or role relationships between classes based on property commonality.
        """
        candidate_parents: Dict[
            str,
            Dict[Tuple[str, ...], List[Tuple[str, SubsumptionType, Dict[str, str]]]],
        ] = defaultdict(lambda: defaultdict(list))
        child_max_matched_props: Dict[str, int] = {n: 0 for n in self.onto.classes}
        for parent_name, parent_info in self.onto.classes.items():
            for child_name, child_info in self.onto.classes.items():
                if not self._is_subsumption_candidate(
                    parent_name=parent_name,
                    parent_info=parent_info,
                    child_name=child_name,
                    child_info=child_info,
                ):
                    continue

                matched_props, child_matched_props = self._get_matched_properties(
                    parent_info=parent_info, child_info=child_info
                )
                if not matched_props:
                    continue

                subsumption_type = self._determine_subsumption_type(
                    matched_props=matched_props,
                    parent_info=parent_info,
                    child_info=child_info,
                )
                if subsumption_type:
                    matched_props = tuple(sorted(matched_props))
                    if len(matched_props) >= child_max_matched_props[child_name]:
                        child_max_matched_props[child_name] = len(matched_props)
                        candidate_parents[child_name][matched_props].append(
                            (parent_name, subsumption_type, child_matched_props)
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
            for matched_props, subsumption_data in candidates.items():
                for (
                    parent_name,
                    subsumption_type,
                    child_matched_props,
                ) in subsumption_data:
                    parent_prop_name = list(child_matched_props.keys())[0]
                    child_prop_name = child_matched_props[parent_prop_name]
                    length = self.onto.properties_inheritance_path_length(
                        child_class=child_prop_name, parent_class=parent_prop_name
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
        parent_info: ClassInfo,
        child_name: str,
        child_info: ClassInfo,
    ) -> bool:
        """Check if parent and child are candidates for implicit subsumption."""
        if parent_name == child_name:
            return False
        if parent_name in child_info.all_base_classes_including_role_takers:
            return False
        if child_name in parent_info.all_base_classes_including_role_takers:
            return False
        base_classes = [
            bc
            for bc in child_info.base_classes
            if bc != self.onto.base_cls_name
            and bc != Symbol.__name__
            and bc != self.onto.role_cls_name
        ]
        if len(base_classes) >= 1:
            return False
        return True

    def _get_matched_properties(
        self,
        *,
        parent_info: ClassInfo,
        child_info: ClassInfo,
    ) -> Tuple[Set[str], Dict[str, str]]:
        """
        Find property base names that are compatible between parent and child.
        """

        parent_props = self.get_declared_and_restricted_properties_of_cls(
            cls_name=parent_info.name
        )
        child_props = self.get_declared_and_restricted_properties_of_cls(
            cls_name=child_info.name
        )

        parent_props_filtered = [p.split("{")[0] for p in parent_props]
        child_props_filtered = [p.split("{")[0] for p in child_props]

        matched_prop_names = set(parent_props_filtered).intersection(
            set(child_props_filtered)
        )

        for mpn in copy(matched_prop_names):
            mpn_parent_idx = parent_props_filtered.index(mpn)
            mpn_child_idx = child_props_filtered.index(mpn)
            parent_p_name = parent_props[mpn_parent_idx]
            child_p_name = child_props[mpn_child_idx]
            if (
                self.onto.properties[parent_p_name].is_specialized
                and not self.onto.properties[child_p_name].is_specialized
            ):
                matched_prop_names.remove(mpn)

        # Re-verify based on original logic: check all combinations of parent/child properties
        # and remove/add base name based on range and superproperty compatibility.
        child_matched_prop_names = {}
        for parent_p_name in parent_props:
            parent_base_name = parent_p_name.split("{")[0]
            for child_p_name in child_props:
                child_p_info, parent_p_info = (
                    self.onto.properties.get(child_p_name),
                    self.onto.properties.get(parent_p_name),
                )
                if not child_p_info or not parent_p_info:
                    continue
                if (
                    child_p_info.type == PropertyType.DATA_PROPERTY
                    or parent_p_info.type == PropertyType.DATA_PROPERTY
                ):
                    continue
                if parent_base_name not in child_p_info.all_superproperties:
                    continue

                child_prop_range, parent_prop_range = (
                    child_p_info.object_range_hint,
                    parent_p_info.object_range_hint,
                )
                if parent_prop_range not in self.onto.class_ancestors.get(
                    child_prop_range, set()
                ).union({child_prop_range}):
                    if parent_base_name in matched_prop_names:
                        matched_prop_names.remove(parent_base_name)
                    continue
                matched_prop_names.add(parent_base_name)
                child_matched_prop_names[parent_base_name] = [
                    pname for pname in child_props if child_p_name in pname
                ][0]

        return matched_prop_names, child_matched_prop_names

    @lru_cache
    def get_declared_and_restricted_properties_of_cls(
        self, *, cls_name: str
    ) -> List[str]:
        """Get declared properties of a class, including those from restrictions."""
        cls_info = self.onto.classes[cls_name]
        cls_props = copy(cls_info.declared_properties)
        cls_props_filtered = [p.split("{")[0] for p in cls_props]
        if cls_info.name in self.onto.property_restrictions:
            for prop_name, ranges in self.onto.property_restrictions[
                cls_info.name
            ].items():
                if prop_name in cls_props_filtered:
                    continue
                cls_props.append(prop_name)
        return cls_props

    def _determine_subsumption_type(
        self, *, matched_props: Set[str], parent_info: ClassInfo, child_info: ClassInfo
    ) -> Optional[SubsumptionType]:
        """Determine if the relationship is a SUBTYPE or a ROLE."""
        parent_props = self.get_declared_and_restricted_properties_of_cls(
            cls_name=parent_info.name
        )
        parent_props_filtered = {p.split("{")[0] for p in parent_props}

        if matched_props == parent_props_filtered:
            if parent_info.role_taker:
                if (
                    not child_info.role_taker
                    or child_info.role_taker.class_name
                    != parent_info.role_taker.class_name
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
        child_info = self.onto.classes[child_name]
        parent_info = self.onto.classes[parent_name]
        if self.onto.base_cls_name in child_info.base_classes:
            child_info.base_classes.remove(self.onto.base_cls_name)

        if subsumption_type == SubsumptionType.ROLE:
            self._apply_role_subsumption(child_info=child_info, parent_name=parent_name)
        else:
            self._apply_subtype_subsumption(
                child_info=child_info, parent_name=parent_name, parent_info=parent_info
            )

    def _apply_role_subsumption(self, *, child_info: ClassInfo, parent_name: str):
        """Add a role taker relationship to the class."""
        child_info.role_taker = RoleTakerInfo(
            class_name=parent_name,
            field_name=NamingRegistry.to_snake_case(name=parent_name),
        )
        if self.onto.role_cls_name not in child_info.all_base_classes:
            child_info.base_classes = [
                self.onto.role_cls_name
            ] + child_info.base_classes
            child_info.all_base_classes = [
                self.onto.role_cls_name
            ] + child_info.all_base_classes
            if Symbol.__name__ not in child_info.base_classes:
                child_info.base_classes.append(Symbol.__name__)
                child_info.all_base_classes.append(Symbol.__name__)
                child_info.all_base_classes_including_role_takers.append(
                    Symbol.__name__
                )
        child_info.all_base_classes_including_role_takers.append(parent_name)

    @staticmethod
    def _apply_subtype_subsumption(
        *, child_info: ClassInfo, parent_name: str, parent_info: ClassInfo
    ):
        """Make the class a subtype of another class."""
        child_info.base_classes = []
        if parent_name not in child_info.base_classes:
            child_info.base_classes.append(parent_name)
            child_info.all_base_classes.append(parent_name)
            child_info.all_base_classes_including_role_takers.append(parent_name)

        # Remove redundant properties already declared in the parent
        parent_props = set(parent_info.declared_properties)
        child_info.declared_properties = [
            p for p in child_info.declared_properties if p not in parent_props
        ]

    def __hash__(self):
        """
        Inference engines are hashed by identity.
        """
        return hash(id(self))
