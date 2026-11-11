"""
Objects of a knowledge base checked against the OWL 2 RL rules of the ontology
``ontologies/owl2rl_check.ttl``: one class per class of the ontology, one list field per object property.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List


@dataclass(eq=False)
class Thing:
    """
    An individual of the knowledge base.
    """

    uri: str
    """
    The IRI of the individual.
    """
    enroll_in: List[Thing] = field(default_factory=list)
    """
    The values of the functional property ``enrollIn``.
    """
    is_head_of: List[Thing] = field(default_factory=list)
    """
    The values of the inverse functional property ``isHeadOf``.
    """
    likes: List[Thing] = field(default_factory=list)
    """
    The values of the irreflexive property ``likes``.
    """
    dislikes: List[Thing] = field(default_factory=list)
    """
    The values of ``dislikes``, disjoint with ``likes``.
    """
    affiliated_with: List[Thing] = field(default_factory=list)
    """
    The values of the asymmetric property ``affiliatedWith``.
    """
    has_student: List[Thing] = field(default_factory=list)
    """
    The values of ``hasStudent``.
    """
    takes_course: List[Thing] = field(default_factory=list)
    """
    The values of ``takesCourse``.
    """


@dataclass(eq=False)
class Person(Thing):
    """
    ``Person``.
    """


@dataclass(eq=False)
class Man(Person):
    """
    ``Man``, disjoint with ``Woman``.
    """


@dataclass(eq=False)
class Woman(Person):
    """
    ``Woman``.
    """


@dataclass(eq=False)
class Student(Person):
    """
    ``Student``, with the key ``hasId``.
    """


@dataclass(eq=False)
class LeisureStudent(Student):
    """
    ``LeisureStudent``, taking at most one course.
    """


@dataclass(eq=False)
class Course(Thing):
    """
    ``Course``.
    """


@dataclass(eq=False)
class College(Thing):
    """
    ``College``.
    """


@dataclass(eq=False)
class WomanCollege(College):
    """
    ``WomanCollege``, whose students are no men.
    """


@dataclass(eq=False)
class FootballFan(Person):
    """
    ``FootballFan``, everyone who likes football.
    """


@dataclass(eq=False)
class Interest(Thing):
    """
    ``Interest``.
    """


@dataclass(eq=False)
class Role(Thing):
    """
    ``Role``, whose superclass expression lies outside OWL 2 RL.
    """


CLASSES = {
    cls.__name__: cls
    for cls in (Person, Man, Woman, Student, LeisureStudent, Course, College)
    + (WomanCollege, FootballFan, Interest, Role)
}
"""
The classes of the knowledge base, by the local name of their class in the ontology.
"""
