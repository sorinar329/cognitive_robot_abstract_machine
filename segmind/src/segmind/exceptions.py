"""
Errors SegMind raises when it is asked for something it cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Type

from krrood.exceptions import DataclassException
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

if TYPE_CHECKING:
    from coraplex.episodes import Episode


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
class EpisodeWithoutFrames(DataclassException):
    """
    Raised when an episode is to be annotated that holds no frame to replay.
    """

    episode: Episode
    """
    The episode asked for.
    """

    def error_message(self) -> str:
        return "The episode holds no frame of its world to replay."

    def suggest_correction(self) -> str:
        return "Record the plan's run with an EpisodeRecorder before annotating it."
