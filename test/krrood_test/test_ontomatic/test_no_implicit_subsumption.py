"""
Subclass and role relations of the generated classes must only come from the ontology
(rdfs:subClassOf, owl:equivalentClass and roleFor restrictions), never from properties
that two classes happen to share.

In the OWL2Bench RL ontology (``dataset/ontologies/no_implicit_subsumption.ttl``), ``SportsLover`` and
``PeopleWithHobby`` are both roles of ``Person`` and unrelated otherwise; ``loves`` is a sub-property of ``likes``.
Ontomatic used to make ``SportsLover`` a subclass of ``PeopleWithHobby`` because of the shared (super-)property, which
OWL 2 RL does not entail.
"""

from __future__ import annotations

import re

from .generated_ontology import GeneratedOntology

# %% subclasses


def test_shared_properties_do_not_create_subclasses():
    """
    Classes sharing a property are not made subclasses of each other.
    """
    source = GeneratedOntology.of("no_implicit_subsumption.ttl").classes_source
    bases = re.search(r"^class SportsLover\((.*?)\):", source, re.M).group(1)
    assert "PeopleWithHobby" not in bases
    assert "Role[Person]" in bases
