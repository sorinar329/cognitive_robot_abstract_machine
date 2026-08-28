"""Approach two: train a detector on synthetic renders of the board.

A Blender scene renders the board and its pieces under randomised viewpoints, lighting and
materials, labelling every shape for free because the renderer knows where it put them. A detector
is then trained on those renders and run on the real camera images.

.. note::
   :mod:`montessori_vision.yolo.pipeline` and :mod:`montessori_vision.yolo.training` import
   ultralytics and are deliberately not imported here, so the package stays usable without the
   ``yolo`` extra installed.
"""

from montessori_vision.yolo.class_names import YoloClassNames
from montessori_vision.yolo.dataset import DatasetSplit, YoloBox, YoloDatasetWriter

__all__ = ["DatasetSplit", "YoloBox", "YoloClassNames", "YoloDatasetWriter"]
