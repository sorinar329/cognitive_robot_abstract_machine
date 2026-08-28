"""Training a detector on the synthetic renders.

.. warning:: This module imports ultralytics, which comes with the ``yolo`` extra. Import it only
    where that is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ultralytics import YOLO


class RunArtifact(StrEnum):
    """The parts of a training run's output folder that are read back."""

    WEIGHTS_FOLDER = "weights"
    """Holds the checkpoints a run saved."""

    BEST_WEIGHTS = "best.pt"
    """The checkpoint that scored best on the validation split."""


@dataclass
class YoloTrainer:
    """Trains a detector on a folder written by
    :class:`montessori_vision.yolo.dataset.YoloDatasetWriter`.

    ..note:: The settings are starting values for a few thousand synthetic renders; tune them once
        the real validation frames show where the model falls short.
    """

    dataset_description: Path
    """The ``data.yaml`` the dataset writer produced."""

    base_weights: str = "yolo11n.pt"
    """The pretrained weights training starts from; the small model is enough for six shapes."""

    epochs: int = 100
    """How many passes over the renders training makes."""

    image_size: int = 640
    """The edge length images are scaled to for training."""

    batch_size: int = 16
    """How many images are trained on at once."""

    run_name: str = "montessori"
    """The name the run and its output folder are given."""

    def train(self) -> Path:
        """Train the detector and return the path of the best weights the run produced."""
        model = YOLO(self.base_weights)
        results = model.train(
            data=str(self.dataset_description),
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            name=self.run_name,
        )
        return Path(results.save_dir) / RunArtifact.WEIGHTS_FOLDER / RunArtifact.BEST_WEIGHTS
