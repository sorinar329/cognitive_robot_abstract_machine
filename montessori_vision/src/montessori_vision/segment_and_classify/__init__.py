"""Approach one: segment everything, then name each segment.

A segmenter proposes a mask for every object it can find without being told what to look for, and a
text image model decides which board shape, if any, each proposal shows. Nothing here is trained on
the board, so a new shape is added by editing the board configuration.

.. note::
   :mod:`montessori_vision.segment_and_classify.segment_anything` and
   :mod:`montessori_vision.segment_and_classify.clip` import torch and are deliberately not imported
   here, so the package stays usable without the ``segment_and_classify`` extra installed.
"""

from montessori_vision.segment_and_classify.classifier import CropClassification, CropClassifier
from montessori_vision.segment_and_classify.mask_filter import MaskFilter
from montessori_vision.segment_and_classify.mask_generator import MaskGenerator
from montessori_vision.segment_and_classify.pipeline import SegmentAndClassifyPipeline

__all__ = [
    "CropClassification",
    "CropClassifier",
    "MaskFilter",
    "MaskGenerator",
    "SegmentAndClassifyPipeline",
]
