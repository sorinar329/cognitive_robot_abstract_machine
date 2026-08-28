"""The interface every detection approach implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from montessori_vision.board import BoardConfiguration
from montessori_vision.detections import ImageDetections
from montessori_vision.image import Image


@dataclass
class DetectionPipeline(ABC):
    """Finds the shapes and holes of a montessori board in a single image.

    Implementations differ in how they get there; they agree on what a caller asks for and what
    comes back, so the robot and the evaluation harness can swap one for another.
    """

    board: BoardConfiguration
    """
    The shape vocabulary the pipeline reports against.
    """

    @property
    def name(self) -> str:
        """A readable name for this pipeline, used when comparing several of them."""
        return type(self).__name__

    @abstractmethod
    def detect(self, image: Image) -> ImageDetections:
        """Return every board shape and hole found in the image."""
