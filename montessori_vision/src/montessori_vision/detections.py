"""What a pipeline reports after looking at one image."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import TYPE_CHECKING, Any, Optional

from montessori_vision.board import BoardConfiguration, ShapeCategory, TargetKind
from montessori_vision.geometry import BoundingBox

if TYPE_CHECKING:
    import numpy.typing as npt


# %% annotation format


class AnnotationKey(StrEnum):
    """The keys of the annotation format that ground truth and predictions are stored in."""

    IMAGE = "image"
    """File name of the annotated image."""

    WIDTH = "width"
    """Width of the annotated image in pixels."""

    HEIGHT = "height"
    """Height of the annotated image in pixels."""

    DETECTIONS = "detections"
    """The shapes found in the image."""

    CATEGORY = "category"
    """Name of the shape category of one detection."""

    KIND = "kind"
    """Role of one detection, a :class:`montessori_vision.board.TargetKind` value."""

    LEFT = "left"
    """Leftmost pixel column of a detection."""

    TOP = "top"
    """Topmost pixel row of a detection."""

    RIGHT = "right"
    """Pixel column just past the right edge of a detection."""

    BOTTOM = "bottom"
    """Pixel row just past the bottom edge of a detection."""

    CONFIDENCE = "confidence"
    """How certain the pipeline is about a detection."""


# %% labels


@dataclass(frozen=True)
class ShapeLabel:
    """A shape of the board in one of its two roles, the thing a pipeline has to name."""

    category: ShapeCategory
    """Which shape of the board this is."""

    kind: TargetKind
    """Whether it is a loose piece or a hole in the board."""

    @property
    def name(self) -> str:
        """A single readable name such as ``star_hole``."""
        return f"{self.category.name}_{self.kind}"

    def matches(self, other: ShapeLabel) -> bool:
        """Whether both labels name the same shape in the same role."""
        return self.category.name == other.category.name and self.kind is other.kind


# %% detections


@dataclass
class Detection:
    """One shape a pipeline found in an image."""

    label: ShapeLabel
    """The shape and role the pipeline assigned."""

    bounding_box: BoundingBox
    """Where the shape sits in the image."""

    confidence: float = 1.0
    """How certain the pipeline is, between zero and one.

    Ground truth uses one.
    """

    mask: Optional[npt.NDArray[np.bool_]] = None
    """The pixels the shape covers, when the pipeline produces one.

    Box only pipelines leave this empty.
    """

    def to_json(self) -> dict[str, Any]:
        """Render the detection as the annotation format, dropping any mask."""
        return {
            AnnotationKey.CATEGORY: self.label.category.name,
            AnnotationKey.KIND: str(self.label.kind),
            AnnotationKey.LEFT: self.bounding_box.left,
            AnnotationKey.TOP: self.bounding_box.top,
            AnnotationKey.RIGHT: self.bounding_box.right,
            AnnotationKey.BOTTOM: self.bounding_box.bottom,
            AnnotationKey.CONFIDENCE: self.confidence,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any], board: BoardConfiguration) -> Detection:
        """Read a detection from the annotation format."""
        return cls(
            label=ShapeLabel(
                category=board.category(payload[AnnotationKey.CATEGORY]),
                kind=TargetKind(payload[AnnotationKey.KIND]),
            ),
            bounding_box=BoundingBox(
                left=payload[AnnotationKey.LEFT],
                top=payload[AnnotationKey.TOP],
                right=payload[AnnotationKey.RIGHT],
                bottom=payload[AnnotationKey.BOTTOM],
            ),
            confidence=payload.get(AnnotationKey.CONFIDENCE, 1.0),
        )


@dataclass
class ImageDetections:
    """Everything a pipeline found in one image, and the single return type of every pipeline."""

    image_name: str
    """The file name of the image these detections belong to."""

    width: int
    """Width of that image in pixels."""

    height: int
    """Height of that image in pixels."""

    detections: list[Detection] = field(default_factory=list)
    """The shapes that were found, in no particular order."""

    def of_kind(self, kind: TargetKind) -> list[Detection]:
        """Return only the detections that play the given role."""
        return [detection for detection in self.detections if detection.label.kind is kind]

    def to_json(self) -> dict[str, Any]:
        """Render the detections as the annotation format."""
        return {
            AnnotationKey.IMAGE: self.image_name,
            AnnotationKey.WIDTH: self.width,
            AnnotationKey.HEIGHT: self.height,
            AnnotationKey.DETECTIONS: [detection.to_json() for detection in self.detections],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any], board: BoardConfiguration) -> ImageDetections:
        """Read detections from the annotation format."""
        return cls(
            image_name=payload[AnnotationKey.IMAGE],
            width=payload[AnnotationKey.WIDTH],
            height=payload[AnnotationKey.HEIGHT],
            detections=[
                Detection.from_json(entry, board) for entry in payload[AnnotationKey.DETECTIONS]
            ],
        )
