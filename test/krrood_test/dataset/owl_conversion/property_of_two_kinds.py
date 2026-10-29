from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RootOfFieldsNamedAlike:
    """
    A root class whose subclasses declare fields of one name holding different kinds of
    values.
    """


@dataclass
class Target(RootOfFieldsNamedAlike):
    """
    A class that fields refer to.
    """


@dataclass
class HoldsAnInstance(RootOfFieldsNamedAlike):
    """
    A class whose field refers to instances of another class.
    """

    content: Target
    """
    A relation to another class.
    """


@dataclass
class HoldsABuiltinValue(RootOfFieldsNamedAlike):
    """
    A class whose field of the same name holds a builtin value.
    """

    content: str
    """
    A builtin value.
    """
