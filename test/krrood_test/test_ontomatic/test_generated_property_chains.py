"""
A property chain axiom of an ontology becomes a :class:`PropertyChain` of the generated
property it implies (``dataset/ontologies/property_chains.ttl``: ``hasParent o
hasParent`` implies ``hasGrandparent``).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from krrood.ontomatic.property_descriptor.property_descriptor import PropertyChain
from .generated_ontology import GeneratedOntology


@pytest.fixture(scope="module")
def properties(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """
    The generated module holding the property descriptors.
    """
    generated = GeneratedOntology.generate_from_file(
        ontology_file_name="property_chains.ttl"
    )
    classes = generated.imported(root=Path(tmp_path_factory.mktemp("property_chains")))
    package = classes.__name__.rsplit(".", 1)[0]
    return importlib.import_module(f"{package}.{generated.module_name}_properties")


# %% chain axioms


def test_a_chain_axiom_is_a_property_chain_of_the_implied_property(properties):
    """
    The implied property returns the chain as a :class:`PropertyChain` of the generated
    properties.
    """
    assert properties.HasGrandparent.get_chain_axioms() == [
        PropertyChain(properties=(properties.HasParent, properties.HasParent))
    ]
