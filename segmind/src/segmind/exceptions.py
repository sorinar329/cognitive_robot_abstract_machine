"""
Errors SegMind raises when it is asked for something it cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass


from krrood.exceptions import DataclassException
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@dataclass
class NoSemanticAnnotationToWatch(DataclassException):
    """
    Raised when a run asks for every semantic annotation of a type to be watched and the
    world holds none of that type.
    """

    semantic_annotation_type: type[SemanticAnnotation]
    """
    The type asked for.
    """

    def error_message(self) -> str:
        return f"The world holds no {self.semantic_annotation_type.__name__} to watch."

    def suggest_correction(self) -> str:
        return "Annotate a body of the world with that type, or ask for another type."


@dataclass
class SegmindProcessFailed(DataclassException):
    """
    Raised when the process SegMind watched a run from ended with an error.
    """

    exit_code: int | None
    """
    The exit code the process ended with.
    """

    def error_message(self) -> str:
        return f"SegMind's own process ended with exit code {self.exit_code}."

    def suggest_correction(self) -> str:
        return "Read the traceback the process printed above to see what failed."


@dataclass
class SegmindProcessDidNotStart(DataclassException):
    """
    Raised when the process SegMind is to watch a run from is not watching in time.
    """

    seconds_waited: float
    """
    How long the run waited for the process to start watching.
    """

    def error_message(self) -> str:
        return f"SegMind's own process was not watching after {self.seconds_waited} s."

    def suggest_correction(self) -> str:
        return "Allow the process more time to start, or watch a smaller world."
