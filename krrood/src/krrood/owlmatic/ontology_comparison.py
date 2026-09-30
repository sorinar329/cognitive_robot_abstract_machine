from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from typing_extensions import ClassVar, Dict, FrozenSet, Iterator, List, Tuple, Union


class AxiomKind(StrEnum):
    """
    The kinds of axiom two ontologies are compared by.
    """

    CLASS_DECLARATION = "class declaration"
    OBJECT_PROPERTY_DECLARATION = "object property declaration"
    DATATYPE_PROPERTY_DECLARATION = "datatype property declaration"
    SUBCLASS_OF = "subclass of"
    EQUIVALENT_CLASS = "equivalent class"
    DISJOINT_WITH = "disjoint with"
    SUB_PROPERTY_OF = "subproperty of"
    INVERSE_OF = "inverse of"
    EQUIVALENT_PROPERTY = "equivalent property"
    PROPERTY_DISJOINT_WITH = "property disjoint with"
    DOMAIN = "domain"
    RANGE = "range"
    PROPERTY_CHARACTERISTIC = "property characteristic"
    RESTRICTION = "restriction"


@dataclass(frozen=True)
class Axiom:
    """
    One axiom of an ontology, with its terms named by their local names so that
    ontologies naming their terms in different namespaces can be compared.
    """

    kind: AxiomKind
    """
    The kind of the axiom.
    """

    terms: Tuple[str, ...]
    """
    The local names of the terms the axiom relates, in the order the axiom kind gives
    them.
    """

    @staticmethod
    def term_of(node: Union[URIRef, Literal]) -> str:
        """
        :return: The local name of *node* if it is an IRI, otherwise its lexical form.
        """
        if isinstance(node, Literal):
            return str(node)
        return str(node).rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    def __str__(self) -> str:
        return f"{self.kind}: {' '.join(self.terms)}"


@dataclass(frozen=True)
class NamedRelation:
    """
    An RDF predicate relating two named terms that makes one axiom per triple.
    """

    predicate: URIRef
    """
    The predicate of the triples.
    """

    kind: AxiomKind
    """
    The kind of axiom each triple makes.
    """

    is_symmetric: bool
    """
    Whether the axiom holds the same with subject and object swapped, so their order
    does not distinguish two axioms.
    """


@dataclass
class OntologyAxioms:
    """
    The axioms of an ontology that are compared: the declarations of named classes and
    properties, the relations between them, the characteristics of properties, and the
    restrictions named classes are declared subclasses of.

    Anonymous class expressions other than restrictions on a property are not compared.
    """

    graph: Graph
    """
    The ontology.
    """

    declarations: ClassVar[Dict[URIRef, AxiomKind]] = {
        OWL.Class: AxiomKind.CLASS_DECLARATION,
        OWL.ObjectProperty: AxiomKind.OBJECT_PROPERTY_DECLARATION,
        OWL.DatatypeProperty: AxiomKind.DATATYPE_PROPERTY_DECLARATION,
    }
    """
    The axiom kind of each type a named term is declared to be of.
    """

    named_relations: ClassVar[List[NamedRelation]] = [
        NamedRelation(RDFS.subClassOf, AxiomKind.SUBCLASS_OF, is_symmetric=False),
        NamedRelation(
            OWL.equivalentClass, AxiomKind.EQUIVALENT_CLASS, is_symmetric=True
        ),
        NamedRelation(OWL.disjointWith, AxiomKind.DISJOINT_WITH, is_symmetric=True),
        NamedRelation(
            RDFS.subPropertyOf, AxiomKind.SUB_PROPERTY_OF, is_symmetric=False
        ),
        NamedRelation(OWL.inverseOf, AxiomKind.INVERSE_OF, is_symmetric=True),
        NamedRelation(
            OWL.equivalentProperty, AxiomKind.EQUIVALENT_PROPERTY, is_symmetric=True
        ),
        NamedRelation(
            OWL.propertyDisjointWith,
            AxiomKind.PROPERTY_DISJOINT_WITH,
            is_symmetric=True,
        ),
        NamedRelation(RDFS.domain, AxiomKind.DOMAIN, is_symmetric=False),
        NamedRelation(RDFS.range, AxiomKind.RANGE, is_symmetric=False),
    ]
    """
    The predicates relating named terms.
    """

    property_characteristics: ClassVar[FrozenSet[URIRef]] = frozenset(
        {
            OWL.FunctionalProperty,
            OWL.InverseFunctionalProperty,
            OWL.TransitiveProperty,
            OWL.SymmetricProperty,
            OWL.AsymmetricProperty,
            OWL.ReflexiveProperty,
            OWL.IrreflexiveProperty,
        }
    )
    """
    The types stating a characteristic of a property.
    """

    restriction_kinds: ClassVar[FrozenSet[URIRef]] = frozenset(
        {
            OWL.allValuesFrom,
            OWL.someValuesFrom,
            OWL.hasValue,
            OWL.hasSelf,
            OWL.cardinality,
            OWL.minCardinality,
            OWL.maxCardinality,
            OWL.qualifiedCardinality,
            OWL.minQualifiedCardinality,
            OWL.maxQualifiedCardinality,
        }
    )
    """
    The predicates stating what a restriction restricts its property to.
    """

    @cached_property
    def axioms(self) -> FrozenSet[Axiom]:
        """
        The compared axioms of the ontology.
        """
        return frozenset(
            [
                *self._declaration_axioms(),
                *self._named_relation_axioms(),
                *self._characteristic_axioms(),
                *self._restriction_axioms(),
            ]
        )

    def _declaration_axioms(self) -> Iterator[Axiom]:
        """
        :return: The declarations of named classes and properties.
        """
        for declared_type, kind in self.declarations.items():
            for subject in self.graph.subjects(RDF.type, declared_type):
                if isinstance(subject, URIRef):
                    yield Axiom(kind, (Axiom.term_of(subject),))

    def _named_relation_axioms(self) -> Iterator[Axiom]:
        """
        :return: The axioms relating two named terms.
        """
        for relation in self.named_relations:
            for subject, named_object in self.graph.subject_objects(relation.predicate):
                if not isinstance(subject, URIRef) or not isinstance(
                    named_object, URIRef
                ):
                    continue
                terms = (Axiom.term_of(subject), Axiom.term_of(named_object))
                if relation.is_symmetric:
                    terms = tuple(sorted(terms))
                yield Axiom(relation.kind, terms)

    def _characteristic_axioms(self) -> Iterator[Axiom]:
        """
        :return: The characteristics stated of named properties.
        """
        for characteristic in self.property_characteristics:
            for subject in self.graph.subjects(RDF.type, characteristic):
                if isinstance(subject, URIRef):
                    yield Axiom(
                        AxiomKind.PROPERTY_CHARACTERISTIC,
                        (Axiom.term_of(subject), Axiom.term_of(characteristic)),
                    )

    def _restriction_axioms(self) -> Iterator[Axiom]:
        """
        :return: The restrictions on a property that named classes are declared subclasses
            of, as the class, the property, the kind of restriction, the value it restricts
            to and, for a qualified cardinality, the class it counts.
        """
        for owl_class, restriction in self.graph.subject_objects(RDFS.subClassOf):
            if not isinstance(owl_class, URIRef) or not isinstance(restriction, BNode):
                continue
            restricted_property = self.graph.value(restriction, OWL.onProperty)
            if restricted_property is None:
                continue
            counted_class = self.graph.value(restriction, OWL.onClass)
            for restriction_kind, value in self.graph.predicate_objects(restriction):
                if restriction_kind not in self.restriction_kinds or isinstance(
                    value, BNode
                ):
                    continue
                terms = (
                    Axiom.term_of(owl_class),
                    Axiom.term_of(restricted_property),
                    Axiom.term_of(restriction_kind),
                    Axiom.term_of(value),
                )
                if counted_class is not None:
                    terms += (Axiom.term_of(counted_class),)
                yield Axiom(AxiomKind.RESTRICTION, terms)


@dataclass
class AxiomAgreement:
    """
    How far the axioms a candidate ontology states agree with those a reference ontology
    states.
    """

    expected: FrozenSet[Axiom]
    """
    The axioms of the reference ontology.
    """

    produced: FrozenSet[Axiom]
    """
    The axioms of the candidate ontology.
    """

    @property
    def shared(self) -> FrozenSet[Axiom]:
        """
        The axioms both ontologies state.
        """
        return self.expected & self.produced

    @property
    def missing(self) -> FrozenSet[Axiom]:
        """
        The axioms only the reference ontology states.
        """
        return self.expected - self.produced

    @property
    def extra(self) -> FrozenSet[Axiom]:
        """
        The axioms only the candidate ontology states.
        """
        return self.produced - self.expected

    @property
    def recall(self) -> float:
        """
        The share of the expected axioms the candidate states; 1 if none are expected.
        """
        if not self.expected:
            return 1.0
        return len(self.shared) / len(self.expected)

    @property
    def precision(self) -> float:
        """
        The share of the candidate's axioms that are expected; 1 if it states none.
        """
        if not self.produced:
            return 1.0
        return len(self.shared) / len(self.produced)


@dataclass
class OntologyComparison:
    """
    Compares a candidate ontology with a reference ontology axiom by axiom.

    Converting the Python that Ontomatic generated from an ontology back with
    :class:`~krrood.owlmatic.owl_converter.OWLConverter` and comparing the result with the
    original ontology evaluates what the round trip through Python keeps.
    """

    reference: Graph
    """
    The ontology the candidate should agree with.
    """

    candidate: Graph
    """
    The ontology that is evaluated.
    """

    @cached_property
    def agreement(self) -> AxiomAgreement:
        """
        The agreement over all compared axioms.
        """
        return AxiomAgreement(
            OntologyAxioms(self.reference).axioms, OntologyAxioms(self.candidate).axioms
        )

    @cached_property
    def agreement_by_kind(self) -> Dict[AxiomKind, AxiomAgreement]:
        """
        The agreement over the axioms of each kind either ontology states.
        """
        stated_kinds = {
            axiom.kind for axiom in self.agreement.expected | self.agreement.produced
        }
        return {
            kind: AxiomAgreement(
                frozenset(
                    axiom for axiom in self.agreement.expected if axiom.kind is kind
                ),
                frozenset(
                    axiom for axiom in self.agreement.produced if axiom.kind is kind
                ),
            )
            for kind in AxiomKind
            if kind in stated_kinds
        }

    def report(self) -> str:
        """
        :return: A table of the recall and precision of every axiom kind and of all axioms,
            followed by the missing and the extra axioms.
        """
        rows = [*self.agreement_by_kind.items(), ("all", self.agreement)]
        lines = [f"{'axiom kind':<32}{'recall':>8}{'precision':>11}"]
        lines.extend(
            f"{kind:<32}{agreement.recall:>8.2f}{agreement.precision:>11.2f}"
            for kind, agreement in rows
        )
        lines.extend(["", "missing:", *sorted(map(str, self.agreement.missing))])
        lines.extend(["", "extra:", *sorted(map(str, self.agreement.extra))])
        return "\n".join(lines)
