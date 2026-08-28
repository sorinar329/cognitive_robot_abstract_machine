"""Where the segmentation and text image models run.

.. warning:: This module imports torch, which comes with the ``segment_and_classify`` extra.
"""

from __future__ import annotations

from enum import StrEnum

import torch


class TorchDevice(StrEnum):
    """The devices a model of this package can run on."""

    CUDA = "cuda"
    """A graphics card, which every model here is fast enough on to iterate with."""

    CPU = "cpu"
    """The processor, correct but slow enough that a whole validation folder takes a while."""

    @classmethod
    def available(cls) -> TorchDevice:
        """Return the graphics card when torch can reach one, and the processor otherwise."""
        return cls.CUDA if torch.cuda.is_available() else cls.CPU
