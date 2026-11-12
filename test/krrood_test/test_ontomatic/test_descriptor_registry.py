"""
Finding the descriptor that manages a property's field on a class.
"""

from __future__ import annotations

import pytest

from krrood.ontomatic.failures import DescriptorNotDeclaredForDomain
from ..dataset.role_and_ontology.university_ontology_like_classes import (
    CEOAsFirstRole,
    Company,
    Person,
    WorksFor,
)

# %% declared descriptors


def test_the_descriptor_declared_on_a_class_is_found():
    """
    The descriptor declared on a class manages the field for that class.
    """
    assert (
        WorksFor.get_descriptor_instance_for_domain_type(domain_type=Person)
        is vars(Person)["works_for"]
    )


def test_a_role_finds_the_descriptor_of_its_role_taker():
    """
    A role without a declaration of its own uses the one of its role taker's class.
    """
    assert (
        WorksFor.get_descriptor_instance_for_domain_type(domain_type=CEOAsFirstRole)
        is vars(Person)["works_for"]
    )


def test_a_class_that_declares_no_descriptor_is_refused():
    """
    A class that neither declares the descriptor nor inherits it is refused.
    """
    with pytest.raises(DescriptorNotDeclaredForDomain):
        WorksFor.get_descriptor_instance_for_domain_type(domain_type=Company)
