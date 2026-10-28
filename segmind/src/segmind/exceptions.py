"""
Errors SegMind raises when it is asked for something it cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Type

from krrood.exceptions import DataclassException
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@dataclass
class NoSemanticAnnotationToWatch(DataclassException):
    """
    Raised when a run asks for every semantic annotation of a type to be watched and the
    world holds none of that type.
    """

    semantic_annotation_type: Type[SemanticAnnotation]
    """
    The type asked for.
    """

    def error_message(self) -> str:
        return f"The world holds no {self.semantic_annotation_type.__name__} to watch."

    def suggest_correction(self) -> str:
        return "Annotate a body of the world with that type, or ask for another type."


@dataclass
class NoEffectsOfAction(DataclassException):
    """
    Raised when the events SegMind is expected to detect while an action runs are asked
    for, and no effects are known for its type.
    """

    action_type: type
    """
    The type of the action asked about.
    """

    def error_message(self) -> str:
        return f"No effects are known for {self.action_type.__name__}."

    def suggest_correction(self) -> str:
        return (
            "Add the effects of this type of action to "
            "segmind.action_effects.EffectsOfActions."
        )
