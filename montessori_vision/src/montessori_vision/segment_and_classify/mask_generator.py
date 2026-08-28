"""Proposing a mask for everything in an image, without knowing what is in it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING

from montessori_vision.image import Image

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class MaskGenerator(ABC):
    """Proposes one mask per object it finds in an image, with no notion of what the objects are.

    Keeping this an interface is what lets the pipeline be tested, and lets a different segmenter be
    tried, without loading a segmentation model.
    """

    @abstractmethod
    def generate(self, image: Image) -> list[npt.NDArray[np.bool_]]:
        """Return a boolean mask of the image size for every object proposal."""
