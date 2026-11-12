"""
Naming the pieces of the Montessori shape-sorting set on Tracy's table.

The set has four pieces in two colours: a cyan cube and cylinder and a yellow
rectangular and triangular prism. Every object hypothesis with a footprint is named by
its colour and the shape of its footprint, and given a pose at its centre.
"""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.annotation import (
    Classification,
    Footprint,
    FootprintShape,
    PoseAnnotation,
)
from robokudo.types.scene import ObjectHypothesis

# %% the pieces


class MontessoriPiece(StrEnum):
    """
    The pieces of the set, named after the hole each goes through.
    """

    SQUARE = "square"
    """
    The cyan cube.
    """

    CIRCLE = "circle"
    """
    The cyan cylinder.
    """

    RECTANGLE = "rectangle"
    """
    The yellow rectangular prism, twice as long as it is wide.
    """

    TRIANGLE = "triangle"
    """
    The yellow triangular prism.
    """


# %% naming the pieces


class MontessoriPieceAnnotator(BaseAnnotator):
    """
    Names every object hypothesis that is one of the Montessori pieces, by its colour
    and the shape of its :class:`~robokudo.types.annotation.Footprint`, and poses it at
    the footprint's centre. Hypotheses that are no piece are left as they are.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the piece annotator.
        """

        class Parameters:
            """
            The colours and proportions the pieces are told apart by.
            """

            def __init__(self) -> None:
                self.yellow_hues: tuple[int, int] = (15, 35)
                """The OpenCV hues (0 to 180) the yellow pieces cover."""
                self.cyan_hues: tuple[int, int] = (85, 110)
                """The OpenCV hues (0 to 180) the cyan pieces cover."""
                self.minimum_saturation: int = 100
                """How saturated a hypothesis' median colour has to be to be a piece."""
                self.largest_extent: float = 0.08
                """The longest a footprint may be to be a piece's, in metres."""

        parameters = Parameters()

    def __init__(
        self,
        name: str = "MontessoriPieceAnnotator",
        descriptor: Descriptor | None = None,
    ) -> None:
        super().__init__(name=name, descriptor=descriptor)

    def update(self) -> Status:
        named = 0
        for hypothesis in self.get_cas().filter_annotations_by_type(ObjectHypothesis):
            footprints = [a for a in hypothesis.annotations if isinstance(a, Footprint)]
            if not footprints:
                continue
            piece = self.identify_piece(hypothesis, footprints[0])
            if piece is None:
                continue
            hypothesis.annotations.append(
                Classification(
                    classification_type="CLASS",
                    classname=str(piece),
                    confidence=1.0,
                    source=type(self).__name__,
                )
            )
            hypothesis.annotations.append(self.pose_at(footprints[0].centre))
            named += 1
        self.feedback_message = f"Named {named} Montessori pieces"
        return Status.SUCCESS

    def identify_piece(
        self, hypothesis: ObjectHypothesis, footprint: Footprint
    ) -> MontessoriPiece | None:
        """
        :param hypothesis: An object hypothesis.
        :param footprint: The hypothesis' footprint.
        :return: The piece the hypothesis is, or None if it is none of them.
        """
        parameters = self.descriptor.parameters
        if not self._has_colours(hypothesis.points):
            return None
        if footprint.length > parameters.largest_extent:
            return None
        hue, saturation = self._median_hue_and_saturation(hypothesis.points)
        if saturation < parameters.minimum_saturation:
            return None
        if parameters.cyan_hues[0] <= hue <= parameters.cyan_hues[1]:
            return {
                FootprintShape.SQUARE: MontessoriPiece.SQUARE,
                FootprintShape.CIRCLE: MontessoriPiece.CIRCLE,
            }.get(footprint.shape)
        if parameters.yellow_hues[0] <= hue <= parameters.yellow_hues[1]:
            return {
                FootprintShape.RECTANGLE: MontessoriPiece.RECTANGLE,
                FootprintShape.TRIANGLE: MontessoriPiece.TRIANGLE,
            }.get(footprint.shape)
        return None

    @staticmethod
    def pose_at(centre: list[float]) -> PoseAnnotation:
        """
        :param centre: The middle of a piece, in the camera's frame.
        :return: A pose there, turned as the camera is.
        """
        pose = PoseAnnotation()
        pose.translation = list(centre)
        pose.rotation = [0.0, 0.0, 0.0, 1.0]
        pose.source = MontessoriPieceAnnotator.__name__
        return pose

    @staticmethod
    def _has_colours(points: o3d.geometry.PointCloud | None) -> bool:
        """
        :param points: The points of a hypothesis, if it has any.
        :return: Whether there are points and they carry colours.
        """
        return (
            isinstance(points, o3d.geometry.PointCloud)
            and len(points.points) > 0
            and points.has_colors()
        )

    @staticmethod
    def _median_hue_and_saturation(
        points: o3d.geometry.PointCloud,
    ) -> tuple[float, float]:
        """
        :param points: Coloured points.
        :return: The median OpenCV hue (0 to 180) and saturation (0 to 255) of their
            colours.
        """
        rgb = (np.asarray(points.colors) * 255).astype(np.uint8).reshape(-1, 1, 3)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        return float(np.median(hsv[:, 0])), float(np.median(hsv[:, 1]))
