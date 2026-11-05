"""
Detectors that do something other than detect, to watch the watching itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from segmind.datastructures.events import DetectionEvent
from segmind.detectors.base import AbstractDetector, SegmindContext
from semantic_digital_twin.world_description.world_entity import Body


# %% a detector that fails


@dataclass
class DetectorFailedOnPurpose(DataclassException):
    """
    Raised by a detector that is made to fail.
    """

    def error_message(self) -> str:
        return "The detector was made to fail."

    def suggest_correction(self) -> str:
        return "Use a detector that is not made to fail."


@dataclass(eq=False, repr=False)
class DetectorFailingOnItsFirstTick(AbstractDetector):
    """
    Fails the first time it is ticked.
    """

    def update_context_and_events(
        self,
        segmind_context: SegmindContext,
        tracked_objects: list[Body],
    ) -> list[DetectionEvent]:
        raise DetectorFailedOnPurpose()
