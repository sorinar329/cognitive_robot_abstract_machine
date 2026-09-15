"""
Exceptions raised by segmind.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from segmind.datastructures.events import DetectionEvent
    from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class NoDetectorDetectsEvent(DataclassException):
    """
    Raised when a detector needs a kind of event that no detector detects in the scene.
    """

    event_type: Type[DetectionEvent]
    """
    The kind of event that was needed.
    """

    tracked_object: Optional[Body]
    """
    The body it was needed about, or None for every trackable body.
    """

    def error_message(self) -> str:
        watched = (
            "every trackable body"
            if self.tracked_object is None
            else self.tracked_object.name
        )
        return f"No detector detects {self.event_type.__name__} about {watched}."

    def suggest_correction(self) -> str:
        return (
            "Declare in the world the part of the scene that kind of detector watches, "
            "such as a gripper, or leave out the detector that needs it."
        )


class OptionalDependency(StrEnum):
    """
    A package segmind uses only for a feature installed as one of its extras.
    """

    FLASK = "flask"
    """
    Serves the live event dashboard (segmind's ``dashboard`` extra).
    """


@dataclass
class DashboardNeedsFlask(DataclassException):
    """
    Raised when the live event dashboard is loaded without flask installed.
    """

    def error_message(self) -> str:
        return (
            f"The live event dashboard needs {OptionalDependency.FLASK}, which is not "
            f"installed."
        )

    def suggest_correction(self) -> str:
        return "Install segmind with its dashboard extra: pip install 'segmind[dashboard]'."
