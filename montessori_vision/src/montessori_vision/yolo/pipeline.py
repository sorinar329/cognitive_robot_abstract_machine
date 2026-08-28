"""Running a trained detector on real camera images.

.. warning:: This module imports ultralytics, which comes with the ``yolo`` extra. Import it only
    where that is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from ultralytics import YOLO

from montessori_vision.detections import Detection, ImageDetections
from montessori_vision.geometry import BoundingBox
from montessori_vision.image import Image
from montessori_vision.pipeline import DetectionPipeline
from montessori_vision.yolo.class_names import YoloClassNames


@dataclass
class YoloPipeline(DetectionPipeline):
    """Finds board shapes with a detector trained on synthetic renders.

    Class indices are read back through the mapping the dataset was written with, so a prediction
    means the same shape it meant during training.
    """

    weights_path: Path
    """
    The trained weights to run, as written by :class:`montessori_vision.yolo.training.YoloTrainer`.
    """

    minimum_confidence: float = 0.25
    """Confidence a prediction must reach to be reported as a detection."""

    class_names: YoloClassNames = field(init=False)
    """The mapping from class indices back to board shapes, derived from the board."""

    def __post_init__(self) -> None:
        self.class_names = YoloClassNames.from_board(self.board)

    @cached_property
    def model(self) -> YOLO:
        """The loaded detector."""
        return YOLO(str(self.weights_path))

    def detect(self, image: Image) -> ImageDetections:
        """Return every board shape and hole the detector predicts in the image."""
        prediction = self.model.predict(
            source=image.pixels, conf=self.minimum_confidence, verbose=False
        )[0]
        detections = [
            Detection(
                label=self.class_names.label_at(int(class_index)),
                bounding_box=BoundingBox(
                    left=int(left), top=int(top), right=int(right), bottom=int(bottom)
                ),
                confidence=float(confidence),
            )
            for (left, top, right, bottom), class_index, confidence in zip(
                prediction.boxes.xyxy.tolist(),
                prediction.boxes.cls.tolist(),
                prediction.boxes.conf.tolist(),
            )
        ]
        return ImageDetections(
            image_name=image.name,
            width=image.width,
            height=image.height,
            detections=detections,
        )
