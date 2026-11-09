"""
A role counts as a subclass of the type of its role taker when classes are ranked by how
specific they are.
"""

from __future__ import annotations

from krrood.patterns.role_aware_class_hierarchy import RoleAwareClassHierarchy
from krrood.symbol_graph.symbol_graph import Symbol
from ..dataset.role_and_ontology.hobbies import PersonWithHobbies, SportsLover
from ..dataset.role_and_ontology.university_ontology_like_classes import (
    CEOAsFirstRole,
    Company,
    Person,
    RecognizedGroup,
    RepresentativeAsSecondRole,
)

# %% inheritance path lengths


def test_a_role_is_one_step_below_the_type_of_its_role_taker():
    """
    A role is one inheritance step below the type of its role taker.
    """
    assert RoleAwareClassHierarchy.inheritance_path_length(CEOAsFirstRole, Person) == 1


def test_a_role_of_a_role_is_two_steps_below_the_root_role_taker_type():
    """
    A role of a role is two inheritance steps below the type of its root role taker.
    """
    assert (
        RoleAwareClassHierarchy.inheritance_path_length(
            RepresentativeAsSecondRole, Person
        )
        == 2
    )


def test_an_unrelated_class_is_infinitely_far():
    """
    A class that is neither a subclass nor a role of another is infinitely far from it.
    """
    assert RoleAwareClassHierarchy.inheritance_path_length(
        CEOAsFirstRole, RecognizedGroup
    ) == float("inf")


# %% most specific classes


def test_the_most_specific_classes_leave_out_the_types_of_role_takers():
    """
    The type of a role taker is less specific than its role.
    """
    assert RoleAwareClassHierarchy.most_specific(
        (Person, CEOAsFirstRole, Company, RecognizedGroup)
    ) == [CEOAsFirstRole, Company]


# %% common ancestors


def test_the_nearest_common_ancestor_of_roles_of_one_type_is_that_type():
    """
    The nearest common ancestor of roles of one type is the type of their role taker.
    """
    assert (
        RoleAwareClassHierarchy.nearest_common_ancestor(
            (SportsLover, PersonWithHobbies)
        )
        is PersonWithHobbies
    )


# %% sorting


def test_classes_are_sorted_from_the_least_to_the_most_specific():
    """
    Classes are sorted by their distance from their nearest common ancestor.
    """
    assert RoleAwareClassHierarchy.sorted_by_inheritance_path_length(
        (RepresentativeAsSecondRole, CEOAsFirstRole, Person)
    ) == [Person, CEOAsFirstRole, RepresentativeAsSecondRole]


def test_among_classes_equally_far_a_class_comes_before_a_role():
    """
    Among classes equally far from the common ancestor, a class that is no role comes
    first.
    """
    assert RoleAwareClassHierarchy.sorted_by_inheritance_path_length(
        (CEOAsFirstRole, Person), common_ancestor=Symbol
    ) == [Person, CEOAsFirstRole]
