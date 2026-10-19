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
class SegmindIsNotWatching(DataclassException):
    """
    Raised when SegMind is asked what it saw after a moment while its detectors are not
    being ticked, so it would never see it.
    """

    def error_message(self) -> str:
        return "SegMind is not watching the world, so it cannot see it after a moment."

    def suggest_correction(self) -> str:
        return "Ask while SegMind is entered, between its start and its stop."
