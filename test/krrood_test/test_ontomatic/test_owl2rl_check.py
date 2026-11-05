"""
The OWL 2 RL check reads the terminology of an ontology and reports the equalities, inconsistencies and missing
conclusions of a knowledge base (``dataset/ontologies/owl2rl_check.ttl``).
"""

from __future__ import annotations

import rdflib
import pytest
from rdflib import URIRef

from krrood.ontomatic.ontology_to_python.owl2rl_check import Owl2RlCheck, Owl2RlRule
from krrood.ontomatic.ontology_to_python.owl2rl_tbox import (
    AllValuesFrom,
    ComplementOf,
    HasValue,
    IntersectionOf,
    MaxCardinality,
    NamedClass,
    Owl2RlTerminology,
    OutsideOwl2Rl,
)
from ..dataset.owl2rl_knowledge_base import (
    CLASSES,
    College,
    Course,
    FootballFan,
    Interest,
    LeisureStudent,
    Man,
    Person,
    Student,
    Thing,
    Woman,
    WomanCollege,
)
from .generated_ontology import ONTOLOGIES_DIRECTORY

NAMESPACE = "http://example.org/test#"
"""
The namespace of the ontology's classes and individuals.
"""


@pytest.fixture
def terminology() -> Owl2RlTerminology:
    """
    The terminology of the ontology, mapped to the classes of the knowledge base.
    """
    graph = rdflib.Graph()
    graph.parse(ONTOLOGIES_DIRECTORY / "owl2rl_check.ttl", format="turtle")
    return Owl2RlTerminology(graph, CLASSES)


# %% helpers


def individual(cls, name: str, **values) -> Thing:
    """
    :return: An individual of the class, named in the namespace of the ontology.
    """
    return cls(uri=NAMESPACE + name, **values)


def check(terminology: Owl2RlTerminology, *objects: Thing):
    """
    :return: The report of checking the individuals, each represented by one object.
    """
    return Owl2RlCheck(terminology, {URIRef(o.uri): [o] for o in objects}).run()


def rules(findings) -> set:
    """
    :return: The rules of the findings.
    """
    return {finding.rule for finding in findings}


# %% rules


def test_terminology_reads_superclass_expressions_and_constraints(terminology):
    """
    The terminology reads the superclass expressions and the property and class constraints.
    """
    assert terminology.superclass_expressions[WomanCollege] == [
        IntersectionOf(
            (
                NamedClass(College),
                AllValuesFrom("has_student", ComplementOf(NamedClass(Man))),
            )
        )
    ]
    assert terminology.superclass_expressions[FootballFan] == [
        HasValue("likes", URIRef(NAMESPACE + "Football"))
    ]
    assert terminology.superclass_expressions[LeisureStudent] == [
        IntersectionOf(
            (NamedClass(Student), MaxCardinality("takes_course", 1, NamedClass(Course)))
        )
    ]
    assert terminology.functional_properties == ["enroll_in", "has_age"]
    assert terminology.inverse_functional_properties == ["is_head_of"]
    assert terminology.disjoint_property_groups == [("likes", "dislikes")]
    assert terminology.disjoint_class_groups == [(Man, Woman)]
    assert [(key.python_class, key.field_names) for key in terminology.keys] == [
        (Student, ("has_id",))
    ]
    assert [type(e) for _, e in terminology.outside_owl2_rl()] == [OutsideOwl2Rl]


def test_a_knowledge_base_without_violations_passes(terminology):
    """
    A knowledge base without violations passes, reporting what lies outside OWL 2 RL.
    """
    football = individual(Interest, "Football")
    fan = individual(FootballFan, "fan", likes=[football])
    report = check(terminology, football, fan)
    assert report.passed
    assert report.outside_owl2_rl == [
        "Role SubClassOf ObjectSomeValuesFrom on roleFor in superclass position"
    ]


def test_equality_rules(terminology):
    """
    The equality rules report the individuals they make equal.
    """
    first, second = individual(College, "c1"), individual(College, "c2")
    student = individual(Student, "s", enroll_in=[first, second])
    head_one = individual(Person, "h1", is_head_of=[first])
    head_two = individual(Person, "h2", is_head_of=[first])
    course_one, course_two = individual(Course, "k1"), individual(Course, "k2")
    leisure = individual(LeisureStudent, "l", takes_course=[course_one, course_two])
    alice, bob = individual(Student, "alice"), individual(Student, "bob")
    report = check(
        terminology,
        first,
        second,
        student,
        head_one,
        head_two,
        course_one,
        course_two,
        leisure,
        alice,
        bob,
    )
    assert rules(report.equalities) == {
        Owl2RlRule.PRP_FP,
        Owl2RlRule.PRP_IFP,
        Owl2RlRule.CLS_MAXC,
        Owl2RlRule.PRP_KEY,
    }
    assert any(
        f.rule == Owl2RlRule.PRP_KEY
        and f.individuals == (NAMESPACE + "alice", NAMESPACE + "bob")
        for f in report.equalities
    )


def test_inconsistency_rules(terminology):
    """
    The inconsistency rules report the individuals that violate them.
    """
    man = individual(Man, "m")
    both = individual(Man, "both")
    both_as_woman = Woman(uri=both.uri)
    college = individual(WomanCollege, "wc", has_student=[man])
    football = individual(Interest, "Football")
    carol = individual(Person, "carol", likes=[football])
    selfish = individual(Person, "selfish")
    selfish.likes.append(selfish)
    torn = individual(Person, "torn", likes=[football], dislikes=[football])
    left, right = individual(College, "left"), individual(College, "right")
    left.affiliated_with.append(right)
    right.affiliated_with.append(left)
    report = Owl2RlCheck(
        terminology,
        {
            URIRef(o.uri): objects
            for o, objects in [
                (man, [man]),
                (both, [both, both_as_woman]),
                (college, [college]),
                (football, [football]),
                (carol, [carol]),
                (selfish, [selfish]),
                (torn, [torn]),
                (left, [left]),
                (right, [right]),
                (individual(Person, "alice"), [individual(Person, "alice")]),
            ]
        },
    ).run()
    assert rules(report.inconsistencies) == {
        Owl2RlRule.CAX_DW,
        Owl2RlRule.CLS_COM,
        Owl2RlRule.PRP_NPA,
        Owl2RlRule.PRP_IRP,
        Owl2RlRule.PRP_PDW,
        Owl2RlRule.PRP_ASYP,
        Owl2RlRule.PRP_FP_DATA,
    }


def test_missing_conclusions_are_reported(terminology):
    """
    A conclusion the knowledge base lacks is reported.
    """
    fan = individual(FootballFan, "fan")
    report = check(terminology, fan, individual(Interest, "Football"))
    assert rules(report.inconsistencies) == {Owl2RlRule.MISSING}


def test_equal_individuals_asserted_to_be_different_are_inconsistent(terminology):
    """
    Equal individuals asserted to be different are inconsistent.
    """
    college = individual(College, "d0")
    first, second = individual(Course, "d1"), individual(Course, "d2")
    student = individual(Student, "s", enroll_in=[first, second])
    report = check(terminology, college, first, second, student)
    assert Owl2RlRule.EQ_DIFF in rules(report.inconsistencies)
