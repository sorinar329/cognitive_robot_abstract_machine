"""Throwing away the proposals that cannot be a board shape before they are classified."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING

from montessori_vision.geometry import BoundingBox
from montessori_vision.image import Image

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class MaskFilter:
    """Drops the proposals a segmenter makes that cannot be a piece or a hole.

    A segmenter that is not told what to look for also proposes the table, the whole board and
    speckle. Classifying those wastes the most expensive step of the pipeline and invites false
    positives, so they are removed by size and shape first.

    ..note:: The defaults are starting values for a board filling a good part of the frame; tune
        them against your own recordings.
    """

    minimum_area_fraction: float = 0.0005
    """Smallest share of the image a proposal may cover to be worth classifying."""

    maximum_area_fraction: float = 0.25
    """Largest share of the image a proposal may cover; above this it is the board or the table."""

    minimum_side_length: int = 12
    """Shortest edge in pixels a proposal's bounding box may have."""

    maximum_aspect_ratio: float = 6.0
    """Largest ratio between the long and the short edge of a proposal's bounding box."""

    minimum_fill_ratio: float = 0.25
    """Smallest share of its own bounding box a proposal must cover, which rejects thin outlines and
    scattered speckle that happen to span a plausible box."""

    def keep(self, mask: npt.NDArray[np.bool_], image: Image) -> bool:
        """Whether a proposal is worth classifying."""
        covered_pixels = int(mask.sum())
        if covered_pixels == 0:
            return False
        area_fraction = covered_pixels / (image.width * image.height)
        if not self.minimum_area_fraction <= area_fraction <= self.maximum_area_fraction:
            return False
        box = BoundingBox.from_mask(mask)
        if min(box.width, box.height) < self.minimum_side_length:
            return False
        if max(box.width, box.height) / min(box.width, box.height) > self.maximum_aspect_ratio:
            return False
        return covered_pixels / box.area >= self.minimum_fill_ratio

    def apply(
        self, masks: list[npt.NDArray[np.bool_]], image: Image
    ) -> list[npt.NDArray[np.bool_]]:
        """Return only the proposals worth classifying, in the order they came in."""
        return [mask for mask in masks if self.keep(mask, image)]
