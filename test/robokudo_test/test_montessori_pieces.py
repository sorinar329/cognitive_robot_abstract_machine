"""
Tests for naming the Montessori pieces by their colour and their footprint.
"""

# %% imports

from __future__ import annotations

from enum import Enum

import numpy as np
import open3d
import pytest
from py_trees.common import Status

from robokudo.annotators.montessori_pieces import (
    MontessoriPiece,
    MontessoriPieceAnnotator,
)
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import (
    Classification,
    Footprint,
    FootprintShape,
    PoseAnnotation,
)
from robokudo.types.scene import ObjectHypothesis

# %% helpers


def annotated_hypothesis(
    colour: tuple[float, float, float], footprint: Footprint
) -> ObjectHypothesis:
    """
    :param colour: The colour of every point of the hypothesis, as RGB from 0 to 1.
    :param footprint: The footprint the hypothesis already has.
    :return: The hypothesis after the annotator ran.
    """
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(
        np.random.default_rng(0).random((50, 3))
    )
    cloud.colors = open3d.utility.Vector3dVector(np.tile(colour, (50, 1)))
    hypothesis = ObjectHypothesis(points=cloud)
    hypothesis.annotations.append(footprint)
    pipeline = Pipeline()
    annotator = MontessoriPieceAnnotator()
    pipeline.add_child(annotator)
    pipeline.setup()
    pipeline.cas.annotations.append(hypothesis)

    assert annotator.update() == Status.SUCCESS
    return hypothesis


def named_piece(colour: tuple[float, float, float], footprint: Footprint) -> str | None:
    """
    :return: The class name the annotator gave the hypothesis, or None if it gave
        none.
    """
    hypothesis = annotated_hypothesis(colour, footprint)
    names = [
        a.classname for a in hypothesis.annotations if isinstance(a, Classification)
    ]
    return names[0] if names else None


class PointColour(Enum):
    """
    Colours the points of a hypothesis are painted in, as RGB from 0 to 1.
    """

    CYAN = (0.0, 0.75, 0.86)
    """
    The colour of the cube and the cylinder.
    """

    YELLOW = (0.9, 0.78, 0.0)
    """
    The colour of the rectangular and the triangular prism.
    """

    GREY = (0.5, 0.5, 0.5)
    """
    The colour of no piece.
    """


# %% naming the pieces


@pytest.mark.parametrize(
    "colour, shape, piece",
    [
        (PointColour.CYAN, FootprintShape.SQUARE, MontessoriPiece.SQUARE),
        (PointColour.CYAN, FootprintShape.CIRCLE, MontessoriPiece.CIRCLE),
        (PointColour.YELLOW, FootprintShape.RECTANGLE, MontessoriPiece.RECTANGLE),
        (PointColour.YELLOW, FootprintShape.TRIANGLE, MontessoriPiece.TRIANGLE),
    ],
)
def test_piece_is_named_by_colour_and_footprint(
    colour: PointColour,
    shape: FootprintShape,
    piece: MontessoriPiece,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Each piece is told apart by its colour and the shape of its footprint.
    """
    footprint = Footprint(shape=shape, length=0.03, width=0.03)

    assert named_piece(colour.value, footprint) == piece


def test_object_larger_than_a_piece_is_not_named(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    An object whose footprint is longer than any piece, such as the board, is no
    piece, whatever its colour.
    """
    largest = MontessoriPieceAnnotator.Descriptor.Parameters().largest_extent
    footprint = Footprint(
        shape=FootprintShape.SQUARE, length=largest + 0.01, width=0.03
    )

    assert named_piece(PointColour.CYAN.value, footprint) is None


def test_grey_object_is_not_named(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    An object whose colour is not saturated enough to be a piece is not named.
    """
    footprint = Footprint(shape=FootprintShape.SQUARE, length=0.03, width=0.03)

    assert named_piece(PointColour.GREY.value, footprint) is None


def test_piece_is_posed_at_the_centre_of_its_footprint(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    A named piece is posed at the centre of its footprint, in the camera's frame.
    """
    footprint = Footprint(
        shape=FootprintShape.CIRCLE, length=0.03, width=0.03, centre=[0.1, -0.2, 0.9]
    )

    hypothesis = annotated_hypothesis(PointColour.CYAN.value, footprint)

    [pose] = [a for a in hypothesis.annotations if isinstance(a, PoseAnnotation)]
    assert pose.translation == footprint.centre


@pytest.mark.parametrize(
    "colour, shape",
    [
        (PointColour.CYAN, FootprintShape.TRIANGLE),
        (PointColour.CYAN, FootprintShape.RECTANGLE),
        (PointColour.YELLOW, FootprintShape.SQUARE),
        (PointColour.YELLOW, FootprintShape.CIRCLE),
    ],
)
def test_object_whose_colour_and_footprint_match_no_piece_is_not_named(
    colour: PointColour,
    shape: FootprintShape,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    An object of a piece's colour but no piece's footprint in that colour, such as a
    cyan part of the robot, is not named.
    """
    footprint = Footprint(shape=shape, length=0.03, width=0.03)

    assert named_piece(colour.value, footprint) is None
