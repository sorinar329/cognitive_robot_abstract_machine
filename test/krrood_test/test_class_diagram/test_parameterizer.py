
from __future__ import annotations

import pytest
from dataclasses import dataclass
from enum import Enum
from typing_extensions import List, Optional

from krrood.class_diagrams.parameterizer import Parameterizer
from random_events.variable import Continuous, Integer, Symbolic
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

from ..dataset.example_classes import (
    Position,
    Orientation,
    Pose,
    Element,
)


@pytest.fixture
def parameterizer() -> Parameterizer:
    return Parameterizer()


def test_parameterizer_position(parameterizer: Parameterizer):
    """
    Test parameterizing a simple dataclass with float fields.
    """
    variables = parameterizer(Position)
    assert len(variables) == 3
    assert all(isinstance(v, Continuous) for v in variables)
    assert variables[0].name == "Position.x"
    assert variables[1].name == "Position.y"
    assert variables[2].name == "Position.z"


def test_parameterizer_orientation(parameterizer: Parameterizer):
    """
    Test parameterizing a dataclass with an Optional float field.
    """
    variables = parameterizer(Orientation)
    assert len(variables) == 4
    assert all(isinstance(v, Continuous) for v in variables)
    assert variables[3].name == "Orientation.w"


def test_parameterizer_pose(parameterizer: Parameterizer):
    """
    Test parameterizing a nested dataclass.
    """
    variables = parameterizer(Pose)
    # Pose has position (3) and orientation (4)
    assert len(variables) == 7
    assert variables[0].name == "Pose.position.x"
    assert variables[4].name == "Pose.orientation.y"


@dataclass
class IntegerClass:
    count: int


def test_parameterizer_integer(parameterizer: Parameterizer):
    """
    Test parameterizing a dataclass with an integer field.
    """
    variables = parameterizer(IntegerClass)
    assert len(variables) == 1
    assert isinstance(variables[0], Integer)
    assert variables[0].name == "IntegerClass.count"


@dataclass
class BooleanEnumClass:
    active: bool
    element: Element


def test_parameterizer_symbolic(parameterizer: Parameterizer):
    """
    Test parameterizing a dataclass with boolean and Enum fields.
    """
    variables = parameterizer(BooleanEnumClass)
    assert len(variables) == 2
    assert all(isinstance(v, Symbolic) for v in variables)
    assert variables[0].name == "BooleanEnumClass.active"
    assert variables[1].name == "BooleanEnumClass.element"


@dataclass
class ListClass:
    values: List[float]


def test_parameterizer_list(parameterizer: Parameterizer):
    """
    Test parameterizing a dataclass with a List field.
    """
    variables = parameterizer(ListClass)
    assert len(variables) == 1
    assert isinstance(variables[0], Continuous)
    assert variables[0].name == "ListClass.values"


def test_create_fully_factorized_distribution(parameterizer: Parameterizer):
    """
    Test creating a fully factorized distribution.
    """
    variables = parameterizer(Position)
    distribution = parameterizer.create_fully_factorized_distribution(variables)
    assert isinstance(distribution, ProbabilisticCircuit)
    assert set(distribution.variables) == set(variables)


def test_parameterizer_non_dataclass_error(parameterizer: Parameterizer):
    """
    Test that a TypeError is raised when a non-dataclass is passed.
    """
    with pytest.raises(TypeError):
        parameterizer(int)


@dataclass
class UnsupportedTypeClass:
    data: dict


def test_parameterizer_unsupported_type_error(parameterizer: Parameterizer):
    """
    Test that a NotImplementedError is raised for unsupported types.
    """
    with pytest.raises(NotImplementedError):
        parameterizer(UnsupportedTypeClass)


