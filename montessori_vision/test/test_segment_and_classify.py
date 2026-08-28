"""The pipeline is exercised with stand-in models, so no weights are needed to check its wiring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from typing_extensions import TYPE_CHECKING

from montessori_vision.board import BoardConfiguration, TargetKind
from montessori_vision.exceptions import MismatchedBatchSize
from montessori_vision.image import Image
from montessori_vision.segment_and_classify.classifier import (
    CropClassification,
    CropClassifier,
    PromptSet,
)
from montessori_vision.segment_and_classify.mask_filter import MaskFilter
from montessori_vision.segment_and_classify.mask_generator import MaskGenerator
from montessori_vision.segment_and_classify.pipeline import SegmentAndClassifyPipeline

from .drawing import DrawnBoard

if TYPE_CHECKING:
    import numpy.typing as npt


# %% stand-ins for the models


@dataclass
class PreparedMaskGenerator(MaskGenerator):
    """Hands out masks decided in advance, standing in for a segmenter."""

    masks: list[npt.NDArray[np.bool_]]
    """
    The proposals every call returns.
    """

    def generate(self, image: Image) -> list[npt.NDArray[np.bool_]]:
        return list(self.masks)


@dataclass
class PreparedClassifier(CropClassifier):
    """Hands out classifications decided in advance, standing in for a text image model."""

    classifications: list[CropClassification] = field(default_factory=list)
    """The answers handed out, one per crop, in order."""

    received_crops: list[Image] = field(default_factory=list)
    """Every crop the pipeline handed over, kept so the cropping itself can be checked."""

    def classify(self, crops: list[Image]) -> list[CropClassification]:
        self.received_crops.extend(crops)
        return self.classifications[: len(crops)]


# %% the pipeline


def build_pipeline(
    board: BoardConfiguration, drawn: DrawnBoard, **overrides
) -> tuple[SegmentAndClassifyPipeline, PreparedClassifier]:
    """A pipeline whose segmenter proposes exactly the drawn shapes and whose answers are known."""
    classifier = PreparedClassifier(
        board=board,
        classifications=[
            CropClassification(label=shape.label, confidence=0.9) for shape in drawn.shapes
        ],
    )
    pipeline = SegmentAndClassifyPipeline(
        board=board,
        mask_generator=PreparedMaskGenerator(masks=drawn.masks()),
        classifier=classifier,
        **overrides,
    )
    return pipeline, classifier


def test_every_proposal_the_classifier_names_becomes_a_detection(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    pipeline, _ = build_pipeline(board, drawn_board)
    found = pipeline.detect(drawn_board.image())
    assert [detection.label.name for detection in found.detections] == [
        shape.label.name for shape in drawn_board.shapes
    ]


def test_a_detection_is_boxed_where_the_shape_was_drawn(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    pipeline, _ = build_pipeline(board, drawn_board, crop_padding=0)
    found = pipeline.detect(drawn_board.image())
    expected = drawn_board.ground_truth()
    for detection, annotated in zip(found.detections, expected.detections):
        assert detection.bounding_box == annotated.bounding_box


def test_the_image_a_detection_came_from_is_reported_with_it(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    pipeline, _ = build_pipeline(board, drawn_board)
    found = pipeline.detect(drawn_board.image())
    assert (found.image_name, found.width, found.height) == (
        drawn_board.name,
        drawn_board.width,
        drawn_board.height,
    )


def test_a_crop_a_background_description_won_is_not_reported(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    pipeline, classifier = build_pipeline(board, drawn_board)
    classifier.classifications[1] = CropClassification(label=None, confidence=0.99)
    found = pipeline.detect(drawn_board.image())
    assert [detection.label.name for detection in found.detections] == [
        drawn_board.shapes[0].label.name,
        drawn_board.shapes[2].label.name,
    ]


def test_a_crop_the_classifier_is_unsure_about_is_not_reported(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    pipeline, classifier = build_pipeline(board, drawn_board, minimum_confidence=0.5)
    classifier.classifications[0] = CropClassification(
        label=drawn_board.shapes[0].label, confidence=0.4
    )
    found = pipeline.detect(drawn_board.image())
    assert drawn_board.shapes[0].label.name not in [
        detection.label.name for detection in found.detections
    ]


def test_the_pixels_around_a_proposal_are_greyed_out_before_it_is_classified(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    pipeline, classifier = build_pipeline(board, drawn_board, crop_padding=4)
    pipeline.detect(drawn_board.image())
    corner = classifier.received_crops[0].pixels[0, 0]
    assert list(corner) == [pipeline.background_value] * 3


def test_a_classifier_that_answers_the_wrong_number_of_times_is_refused(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    pipeline, classifier = build_pipeline(board, drawn_board)
    classifier.classifications.pop()
    with pytest.raises(MismatchedBatchSize):
        pipeline.detect(drawn_board.image())


# %% the filter in front of the classifier


def test_a_proposal_covering_most_of_the_picture_never_reaches_the_classifier(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    whole_picture = np.ones((drawn_board.height, drawn_board.width), dtype=bool)
    classifier = PreparedClassifier(
        board=board, classifications=[CropClassification(label=None, confidence=1.0)]
    )
    pipeline = SegmentAndClassifyPipeline(
        board=board,
        mask_generator=PreparedMaskGenerator(masks=[whole_picture]),
        classifier=classifier,
    )
    pipeline.detect(drawn_board.image())
    assert classifier.received_crops == []


def test_a_speck_of_a_proposal_never_reaches_the_classifier(drawn_board: DrawnBoard) -> None:
    speck = np.zeros((drawn_board.height, drawn_board.width), dtype=bool)
    speck[10:13, 10:13] = True
    assert MaskFilter().keep(speck, drawn_board.image()) is False


def test_a_drawn_shape_passes_the_filter(drawn_board: DrawnBoard) -> None:
    for mask in drawn_board.masks():
        assert MaskFilter().keep(mask, drawn_board.image()) is True


# %% prompts


def test_every_shape_and_role_of_the_board_is_described_to_the_classifier(
    board: BoardConfiguration,
) -> None:
    prompts = PromptSet.from_board(board)
    described = {prompt.label for prompt in prompts.prompts if prompt.label is not None}
    assert len(described) == len(board.categories) * len(TargetKind)


def test_the_background_is_described_too_so_a_crop_can_be_rejected(
    board: BoardConfiguration,
) -> None:
    prompts = PromptSet.from_board(board)
    rejecting = [prompt.text for prompt in prompts.prompts if prompt.label is None]
    assert rejecting == list(board.background_prompts)
