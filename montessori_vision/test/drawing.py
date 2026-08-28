"""Drawing board pictures from the real outlines, so tests need no recorded images."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from typing_extensions import TYPE_CHECKING

from montessori_vision.board import ShapeCategory, TargetKind
from montessori_vision.detections import Detection, ImageDetections, ShapeLabel
from montessori_vision.geometry import BoundingBox, Point2D
from montessori_vision.image import Image

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class DrawnShape:
    """One shape to put on a drawn picture."""

    category: ShapeCategory
    """The shape to draw."""

    kind: TargetKind
    """Whether it is drawn as a loose piece or as a hole."""

    center: Point2D
    """Where the middle of the shape sits in the picture."""

    radius: float
    """Pixels from the middle of the shape to its furthest corner."""

    @property
    def label(self) -> ShapeLabel:
        """What a pipeline would have to report for this shape."""
        return ShapeLabel(self.category, self.kind)

    def corners(self) -> npt.NDArray[np.int32]:
        """The corners of the shape in picture coordinates, as OpenCV wants them."""
        return np.array(
            [
                (round(self.center.x + point.x), round(self.center.y + point.y))
                for point in self.category.outline.vertices(self.radius)
            ],
            dtype=np.int32,
        )

    def mask(self, width: int, height: int) -> npt.NDArray[np.bool_]:
        """The pixels this shape covers in a picture of the given size."""
        drawn = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(drawn, [self.corners()], color=1)
        return drawn.astype(bool)


@dataclass
class DrawnBoard:
    """A picture of a board with its holes and a few loose pieces, drawn rather than photographed.

    The drawing uses the outlines of the board configuration itself, so a test that passes against
    it is testing the shapes the pipelines will actually be given.
    """

    width: int = 320
    """Width of the drawn picture in pixels."""

    height: int = 240
    """Height of the drawn picture in pixels."""

    shapes: list[DrawnShape] = field(default_factory=list)
    """The shapes on the picture, in the order they are drawn."""

    name: str = "drawn_board.png"
    """The file name the picture is reported under."""

    piece_colour: tuple[int, int, int] = (200, 120, 60)
    """The red, green and blue a loose piece is drawn in."""

    hole_colour: tuple[int, int, int] = (30, 30, 30)
    """The red, green and blue a hole is drawn in, dark like an opening."""

    background_colour: tuple[int, int, int] = (160, 150, 140)
    """The red, green and blue the table is drawn in."""

    def image(self) -> Image:
        """Draw the picture."""
        pixels = np.full((self.height, self.width, 3), self.background_colour, dtype=np.uint8)
        for shape in self.shapes:
            colour = self.hole_colour if shape.kind is TargetKind.HOLE else self.piece_colour
            cv2.fillPoly(pixels, [shape.corners()], color=colour)
        return Image(name=self.name, pixels=pixels)

    def ground_truth(self) -> ImageDetections:
        """What a perfect pipeline would report about this picture."""
        return ImageDetections(
            image_name=self.name,
            width=self.width,
            height=self.height,
            detections=[
                Detection(
                    label=shape.label,
                    bounding_box=BoundingBox.from_mask(shape.mask(self.width, self.height)),
                )
                for shape in self.shapes
            ],
        )

    def masks(self) -> list[npt.NDArray[np.bool_]]:
        """The mask of every shape, as a perfect segmenter would propose them."""
        return [shape.mask(self.width, self.height) for shape in self.shapes]
