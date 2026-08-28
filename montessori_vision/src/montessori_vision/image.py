"""The images the pipelines look at."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class Image:
    """One picture a pipeline is asked about, kept together with the name it is reported under.

    Pixels are red, green, blue in that order, so a pipeline never has to guess a channel order.
    """

    name: str
    """
    The file name the image was read from, used to tie detections back to their annotation.
    """

    pixels: npt.NDArray[np.uint8]
    """The picture as a height by width by three array of red, green and blue values."""

    @property
    def width(self) -> int:
        """Width of the image in pixels."""
        return self.pixels.shape[1]

    @property
    def height(self) -> int:
        """Height of the image in pixels."""
        return self.pixels.shape[0]

    @classmethod
    def read(cls, path: Path) -> Image:
        """Read an image from disk and convert it to red, green, blue order."""
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cls(name=path.name, pixels=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))

    def write(self, path: Path) -> None:
        """Write the image to disk."""
        cv2.imwrite(str(path), cv2.cvtColor(self.pixels, cv2.COLOR_RGB2BGR))
