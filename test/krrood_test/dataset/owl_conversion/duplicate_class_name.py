from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SecondRoot:
    """
    A root class whose hierarchy contains a class named like one in another hierarchy.
    """


@dataclass
class Handle(SecondRoot):
    """
    A class sharing its name with a class of the converted hierarchy.
    """
