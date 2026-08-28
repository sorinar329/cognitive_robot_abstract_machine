"""The segment and classify pipeline itself."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING

from montessori_vision.detections import Detection, ImageDetections
from montessori_vision.exceptions import MismatchedBatchSize
from montessori_vision.geometry import BoundingBox
from montessori_vision.image import Image
from montessori_vision.pipeline import DetectionPipeline
from montessori_vision.segment_and_classify.classifier import CropClassifier
from montessori_vision.segment_and_classify.mask_filter import MaskFilter
from montessori_vision.segment_and_classify.mask_generator import MaskGenerator

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class SegmentAndClassifyPipeline(DetectionPipeline):
    """Finds board shapes by segmenting everything and then naming what was segmented.

    The segmenter and the classifier are given to the pipeline rather than built by it, so the same
    pipeline runs against a real segmentation model, a cheaper one, or a stand-in in a test.
    """

    mask_generator: MaskGenerator
    """
    Proposes a mask for every object in the image.
    """

    classifier: CropClassifier
    """Decides which board shape each surviving proposal shows."""

    mask_filter: MaskFilter = field(default_factory=MaskFilter)
    """Removes the proposals that cannot be a piece or a hole before they are classified."""

    minimum_confidence: float = 0.25
    """Confidence a classification must reach to be reported as a detection."""

    crop_padding: int = 8
    """Pixels of context kept around a proposal, so a shape is not cut flush at its edge."""

    isolate_crops: bool = True
    """Whether the pixels outside a proposal's mask are replaced by grey before classifying, which
    stops a neighbouring shape in the same box from deciding the answer."""

    background_value: int = 128
    """The grey that replaces the pixels outside a proposal, mid grey so neither a bright nor a dark
    shape is turned into a silhouette of itself."""

    def detect(self, image: Image) -> ImageDetections:
        """Return every board shape and hole the segmenter proposed and the classifier named."""
        masks = self.mask_filter.apply(self.mask_generator.generate(image), image)
        boxes = [
            BoundingBox.from_mask(mask).padded(self.crop_padding, image.width, image.height)
            for mask in masks
        ]
        crops = [self.crop(image, mask, box) for mask, box in zip(masks, boxes)]
        classifications = self.classifier.classify(crops)
        if len(classifications) != len(crops):
            raise MismatchedBatchSize(len(crops), len(classifications))

        detections = [
            Detection(
                label=classification.label,
                bounding_box=box,
                confidence=classification.confidence,
                mask=mask,
            )
            for classification, box, mask in zip(classifications, boxes, masks)
            if classification.label is not None
            and classification.confidence >= self.minimum_confidence
        ]
        return ImageDetections(
            image_name=image.name,
            width=image.width,
            height=image.height,
            detections=detections,
        )

    def crop(self, image: Image, mask: npt.NDArray[np.bool_], box: BoundingBox) -> Image:
        """Cut one proposal out of the image, optionally greying out everything around it."""
        pixels = image.pixels
        if self.isolate_crops:
            pixels = np.where(mask[:, :, np.newaxis], pixels, self.background_value)
        return Image(name=image.name, pixels=box.crop(pixels).astype(np.uint8))
