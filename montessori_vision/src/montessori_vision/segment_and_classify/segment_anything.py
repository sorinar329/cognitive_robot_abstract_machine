"""Segment Anything as the source of object proposals.

.. warning:: This module imports torch and the ``segment-anything`` package, both of which come with
    the ``segment_and_classify`` extra. Import it only where those are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from typing_extensions import TYPE_CHECKING, Any

from montessori_vision.image import Image
from montessori_vision.segment_and_classify.device import TorchDevice
from montessori_vision.segment_and_classify.mask_generator import MaskGenerator

if TYPE_CHECKING:
    import numpy.typing as npt


class SegmentAnythingModelSize(StrEnum):
    """The published Segment Anything checkpoints, from the cheapest to the most accurate."""

    BASE = "vit_b"
    """The smallest checkpoint, fast enough to iterate with."""

    LARGE = "vit_l"
    """The middle checkpoint."""

    HUGE = "vit_h"
    """The largest checkpoint, the one the published results use."""


class MaskKey(StrEnum):
    """The keys of the dictionaries Segment Anything returns per proposal."""

    SEGMENTATION = "segmentation"
    """The boolean mask of one proposal."""

    AREA = "area"
    """The number of pixels one proposal covers."""


@dataclass
class SegmentAnythingMaskGenerator(MaskGenerator):
    """Proposes masks with Segment Anything in its prompt free mode.

    The model is loaded the first time a mask is asked for, so building the pipeline stays cheap.
    """

    checkpoint_path: Path
    """The downloaded Segment Anything checkpoint to load."""

    model_size: SegmentAnythingModelSize = SegmentAnythingModelSize.HUGE
    """Which checkpoint architecture ``checkpoint_path`` holds."""

    device: TorchDevice = field(default_factory=TorchDevice.available)
    """The device the model runs on."""

    points_per_side: int = 32
    """How densely the image is probed for objects; more points find smaller shapes and cost
    more."""

    prediction_quality_threshold: float = 0.9
    """The model's own quality score a proposal must reach to be returned."""

    stability_threshold: float = 0.95
    """How stable a proposal has to be under a shifted cutoff to be returned."""

    extra_arguments: dict[str, Any] = field(default_factory=dict)
    """Further arguments passed straight to ``SamAutomaticMaskGenerator`` for experiments that need
    settings this class does not name."""

    @cached_property
    def model(self) -> SamAutomaticMaskGenerator:
        """The loaded Segment Anything mask generator."""
        model = sam_model_registry[str(self.model_size)](checkpoint=str(self.checkpoint_path))
        model.to(device=self.device)
        return SamAutomaticMaskGenerator(
            model=model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.prediction_quality_threshold,
            stability_score_thresh=self.stability_threshold,
            **self.extra_arguments,
        )

    def generate(self, image: Image) -> list[npt.NDArray[np.bool_]]:
        """Return one mask per object proposal, largest proposal first."""
        proposals = self.model.generate(image.pixels)
        proposals.sort(key=lambda proposal: proposal[MaskKey.AREA], reverse=True)
        return [np.asarray(proposal[MaskKey.SEGMENTATION], dtype=bool) for proposal in proposals]
