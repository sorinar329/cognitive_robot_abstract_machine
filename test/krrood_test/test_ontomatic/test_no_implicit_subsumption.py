"""
Subclass and role relations of the generated classes come only from the ontology, never
from properties two classes share: ``SportsLover`` and ``PeopleWithHobby`` stay
unrelated roles of ``Person`` although ``loves`` is a subproperty of ``likes``
(``dataset/ontologies/no_implicit_subsumption.ttl``).
"""

from __future__ import annotations

import re

from .generated_ontology import GeneratedOntology

# %% subclasses


def test_shared_properties_do_not_create_subclasses():
    """
    Classes sharing a property are not made subclasses of each other.
    """
    source = GeneratedOntology.generate_from_file(
        ontology_file_name="no_implicit_subsumption.ttl"
    ).classes_source
    bases = re.search(r"^class SportsLover\((.*?)\):", source, re.M).group(1)
    assert "PeopleWithHobby" not in bases
    assert "Role[Person]" in bases
