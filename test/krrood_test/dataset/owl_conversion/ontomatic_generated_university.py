"""
The Python that Ontomatic's ``OwlToPythonConverter`` generates from ``university.ttl``.

Generated on the ``dl2`` branch (commit ``df5158820e``), which spreads it over a base, a
properties and a classes module. They are joined here in that order, and what ``main``
does not provide is left out: the ``IsBaseClass`` marker of the base class, the query
language imports the class module declares for axioms none of these classes have, the
property mixins no descriptor below uses, and the closing call to
``PropertyDescriptor.update_domains_that_are_axiomatized_on_properties``. Apart from
formatting, everything else is the generated code unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import ClassVar, Optional, Set, Type

from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.mixins import (
    HasInverseProperty,
    TransitiveProperty,
)
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)

# %% base module


@dataclass(eq=False)
class UniversityThing(Symbol):
    """
    Base class for University.
    """

    # URI of the ontology element - The unique resource identifier (URI) of the ontology element.
    uri: Optional[str] = field(kw_only=True, default=None)

    def __hash__(self):
        return hash(id(self))


# %% properties module


@dataclass(eq=False)
class Member(PropertyDescriptor, HasInverseProperty):
    """
    Member.
    """

    @classmethod
    def get_inverse(cls) -> Optional[Type[MemberOf]]:
        if cls is Member:
            return MemberOf
        return None


@dataclass(eq=False)
class MemberOf(PropertyDescriptor, HasInverseProperty):
    """
    MemberOf.
    """

    @classmethod
    def get_inverse(cls) -> Optional[Type[Member]]:
        if cls is MemberOf:
            return Member
        return None


@dataclass(eq=False)
class SubOrganizationOf(PropertyDescriptor, TransitiveProperty):
    """
    SubOrganizationOf.
    """


@dataclass(eq=False)
class TakesCourse(PropertyDescriptor):
    """
    TakesCourse.
    """


@dataclass(eq=False)
class WorksFor(MemberOf):
    """
    WorksFor.
    """


# %% classes module


@dataclass(eq=False)
class Course(UniversityThing):
    cls_uri: ClassVar[str] = "http://example.org/university#Course"


@dataclass(eq=False)
class Organization(UniversityThing):
    """
    A group of people with a shared purpose.
    """

    cls_uri: ClassVar[str] = "http://example.org/university#Organization"
    member: Set[Person] = field(kw_only=True, default_factory=set)
    sub_organization_of: Set[Organization] = field(kw_only=True, default_factory=set)


@dataclass(eq=False)
class Person(UniversityThing):
    """
    A human being.
    """

    cls_uri: ClassVar[str] = "http://example.org/university#Person"
    member_of: Set[Organization] = field(kw_only=True, default_factory=set)
    name: Optional[str] = field(kw_only=True, default=None)
    works_for: Set[Organization] = field(kw_only=True, default_factory=set)


@dataclass(eq=False)
class Department(Organization):
    cls_uri: ClassVar[str] = "http://example.org/university#Department"


@dataclass(eq=False)
class Student(UniversityThing):
    cls_uri: ClassVar[str] = "http://example.org/university#Student"
    takes_course: Set[Course] = field(kw_only=True, default_factory=set)


@dataclass(eq=False)
class University(Organization):
    cls_uri: ClassVar[str] = "http://example.org/university#University"


# %% descriptor assignments

Organization.member = Member(Organization, "member")
Organization.sub_organization_of = SubOrganizationOf(
    Organization, "sub_organization_of"
)
Person.member_of = MemberOf(Person, "member_of")
Person.works_for = WorksFor(Person, "works_for")
Student.takes_course = TakesCourse(Student, "takes_course")
