from typing import Container

import pytest
from sqlalchemy import select
from krrood.ormatic.dao import to_dao
from ..dataset.cyclic_dao_dependency import IssueMain, IssueDependency, PlanReproduction
from ..dataset.cyclic_dependent_alternative_mappings import (
    CyclicDependentAlternativeMappingContainer,
    MainMapping,
    Dependency,
    Main,
)
from ..dataset.ormatic_interface import (
    IssueMainMappingDAO,
    PlanReproductionDAO,
    CyclicDependentAlternativeMappingContainerDAO,
    MainMappingDAO,
    ContainerDAO,
)


def test_alternative_mapping_hash_failure(session, database):
    """
    Test that reproducing the issue where an AlternativeMapping's to_domain_object
    calls hash() on its dependencies, which are not yet filled with their relationships.
    """
    # Setup domain objects
    main = IssueMain(name="root")
    dep = IssueDependency(name="dep1", parent=main)
    main.dependencies = [dep]
    plan = PlanReproduction(dependency=dep)

    # Persist
    dao = to_dao(plan)
    session.add(dao)
    session.commit()
    session.expunge_all()

    # Restore
    # This should trigger the bug because PlanReproductionDAO.from_dao()
    # will discover 'dependency' FIRST, and 'main' SECOND (via dependency).
    # discovery_order: ['plan', 'dependency', 'main']
    # Reversed discovery order: ['main', 'dependency', 'plan']
    # So 'main' is filled FIRST.
    # main._fill_from_dao() calls to_domain_object()
    # to_domain_object() calls hash(dep)
    # BUT 'dep' has NOT been filled yet!
    # So 'dep' has NO 'parent' attribute!
    fetched_dao = session.scalars(select(PlanReproductionDAO)).one()

    # This raises AttributeError: 'IssueDependency' object has no attribute 'parent'
    recreated = fetched_dao.from_dao()

    assert isinstance(recreated.dependency.parent, IssueMain)
    assert recreated.dependency.parent.name == "root"
    assert isinstance(recreated.dependency.parent.dependencies[0], IssueDependency)
    assert recreated.dependency.parent.dependencies[0].name == "dep1"
    assert (
        recreated.dependency.parent.dependencies[0].parent
        is recreated.dependency.parent
    )


def test_chained_alternative_mapping_fix(session, database):
    """
    Test that the refcount solution correctly resolves chained AlternativeMappings.
    """
    # Setup domain objects
    dependency = Dependency(name="dep", value=42)
    main = Main(name="main", dependency=dependency)
    container = CyclicDependentAlternativeMappingContainer(
        dependency=dependency, main=main
    )
    container_dao = to_dao(container)
    # Re-mocking for each call if needed, but the current mock should suffice

    # Discovery order: cont_dao -> dep_dao -> main_dao
    # Reversed: main_dao -> dep_dao -> cont_dao
    # With refcount:
    # dep_dao has 0 incoming dependencies from these three.
    # main_dao depends on dep_dao.
    # cont_dao depends on dep_dao and main_dao.

    # Order should be: dep_dao resolved, then main_dao resolved, then cont_dao resolved.

    recreated = container_dao.from_dao()

    assert isinstance(recreated.dependency, Dependency)
    assert recreated.dependency.value == 42
    assert isinstance(recreated.main.dependency, Dependency)
    assert recreated.main.dependency.value == 42
    assert recreated.main.dependency is recreated.dependency
