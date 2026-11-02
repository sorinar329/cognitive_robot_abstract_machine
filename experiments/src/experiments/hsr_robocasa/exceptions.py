"""
Errors the HSR's RoboCasa tasks raise when the HSR cannot do what a task asks.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class NoApproachError(DataclassException):
    """
    Raised when no approach of the hand to an object is free and leaves the HSR
    somewhere to stand.
    """

    body: Body
    """
    The object the hand was to take hold of.
    """

    def error_message(self) -> str:
        return (
            f"No approach of the HSR's hand to {self.body.name.name} is free and leaves "
            f"the HSR somewhere to stand."
        )

    def suggest_correction(self) -> str:
        return (
            "Try more approaches, another seed that places the object elsewhere, or a "
            "lowered kitchen."
        )
