"""The comparison is the answer the repository exists for, so its arithmetic is pinned down."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from montessori_vision.board import BoardConfiguration, TargetKind
from montessori_vision.dataset import AnnotationFile, ImageFolderDataset
from montessori_vision.detections import Detection, ImageDetections, ShapeLabel
from montessori_vision.evaluation.comparison import PipelineComparison
from montessori_vision.evaluation.matching import DetectionMatcher
from montessori_vision.evaluation.score import PipelineScore
from montessori_vision.exceptions import MissingAnnotation
from montessori_vision.geometry import BoundingBox
from montessori_vision.image import Image
from montessori_vision.pipeline import DetectionPipeline

from .drawing import DrawnBoard

# %% a stand-in for a pipeline


@dataclass
class PreparedPipeline(DetectionPipeline):
    """Reports detections decided in advance, standing in for a real approach."""

    prepared: ImageDetections
    """
    What every call reports.
    """

    def detect(self, image: Image) -> ImageDetections:
        return self.prepared


# %% matching


def test_a_prediction_on_top_of_an_annotation_is_a_hit(drawn_board: DrawnBoard) -> None:
    truth = drawn_board.ground_truth()
    result = DetectionMatcher().match(truth, truth)
    assert len(result.matches) == len(truth.detections)
    assert result.false_positives == []
    assert result.false_negatives == []


def test_a_prediction_naming_the_wrong_shape_is_a_miss_and_a_false_alarm(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    truth = drawn_board.ground_truth()
    mislabelled = ImageDetections(
        image_name=truth.image_name,
        width=truth.width,
        height=truth.height,
        detections=[
            Detection(
                label=ShapeLabel(board.category("circle"), truth.detections[0].label.kind),
                bounding_box=truth.detections[0].bounding_box,
            )
        ],
    )
    result = DetectionMatcher().match(truth, mislabelled)
    assert result.matches == []
    assert len(result.false_positives) == 1
    assert len(result.false_negatives) == len(truth.detections)


def test_a_prediction_naming_the_wrong_role_is_a_miss_and_a_false_alarm(
    drawn_board: DrawnBoard,
) -> None:
    truth = drawn_board.ground_truth()
    annotated = truth.detections[0]
    swapped_role = TargetKind.PIECE if annotated.label.kind is TargetKind.HOLE else TargetKind.HOLE
    predicted = ImageDetections(
        image_name=truth.image_name,
        width=truth.width,
        height=truth.height,
        detections=[
            Detection(
                label=ShapeLabel(annotated.label.category, swapped_role),
                bounding_box=annotated.bounding_box,
            )
        ],
    )
    result = DetectionMatcher().match(truth, predicted)
    assert len(result.false_positives) == 1


def test_a_prediction_that_barely_overlaps_is_not_a_hit(drawn_board: DrawnBoard) -> None:
    truth = drawn_board.ground_truth()
    annotated = truth.detections[0]
    box = annotated.bounding_box
    shifted = ImageDetections(
        image_name=truth.image_name,
        width=truth.width,
        height=truth.height,
        detections=[
            Detection(
                label=annotated.label,
                bounding_box=BoundingBox(
                    left=box.left + box.width,
                    top=box.top,
                    right=box.right + box.width,
                    bottom=box.bottom,
                ),
            )
        ],
    )
    assert DetectionMatcher().match(truth, shifted).matches == []


def test_only_the_more_confident_of_two_predictions_claims_one_annotation(
    drawn_board: DrawnBoard,
) -> None:
    truth = drawn_board.ground_truth()
    annotated = truth.detections[0]
    doubled = ImageDetections(
        image_name=truth.image_name,
        width=truth.width,
        height=truth.height,
        detections=[
            Detection(label=annotated.label, bounding_box=annotated.bounding_box, confidence=0.4),
            Detection(label=annotated.label, bounding_box=annotated.bounding_box, confidence=0.9),
        ],
    )
    result = DetectionMatcher().match(truth, doubled)
    assert len(result.matches) == 1
    assert result.matches[0].prediction.confidence == 0.9
    assert len(result.false_positives) == 1


# %% scoring


def test_a_perfect_run_scores_one_everywhere(drawn_board: DrawnBoard) -> None:
    truth = drawn_board.ground_truth()
    score = PipelineScore.of("perfect", DetectionMatcher().match(truth, truth))
    assert (score.counts.precision, score.counts.recall, score.counts.harmonic_mean) == (
        1.0,
        1.0,
        1.0,
    )
    assert score.counts.mean_overlap == 1.0


def test_a_run_that_finds_nothing_scores_no_recall(drawn_board: DrawnBoard) -> None:
    truth = drawn_board.ground_truth()
    empty = ImageDetections(image_name=truth.image_name, width=truth.width, height=truth.height)
    score = PipelineScore.of("silent", DetectionMatcher().match(truth, empty))
    assert score.counts.recall == 0.0
    assert score.counts.missed == len(truth.detections)


def test_finding_one_of_two_annotations_halves_the_recall(
    board: BoardConfiguration, drawn_board: DrawnBoard
) -> None:
    truth = drawn_board.ground_truth()
    half = ImageDetections(
        image_name=truth.image_name,
        width=truth.width,
        height=truth.height,
        detections=truth.detections[:1],
    )
    score = PipelineScore.of("half", DetectionMatcher().match(truth, half))
    assert score.counts.precision == 1.0
    assert score.counts.recall == pytest.approx(1 / len(truth.detections))


def test_the_score_is_broken_down_by_shape_and_role(drawn_board: DrawnBoard) -> None:
    truth = drawn_board.ground_truth()
    score = PipelineScore.of("perfect", DetectionMatcher().match(truth, truth))
    assert {entry.label.name for entry in score.per_label} == {
        shape.label.name for shape in drawn_board.shapes
    }


def test_pieces_and_holes_are_scored_apart(drawn_board: DrawnBoard) -> None:
    truth = drawn_board.ground_truth()
    score = PipelineScore.of("perfect", DetectionMatcher().match(truth, truth))
    holes = [shape for shape in drawn_board.shapes if shape.kind is TargetKind.HOLE]
    assert score.of_kind(TargetKind.HOLE).found == len(holes)
    assert score.of_kind(TargetKind.PIECE).found == len(drawn_board.shapes) - len(holes)


# %% comparing over a folder


def write_folder(folder: Path, drawn: DrawnBoard) -> None:
    """Put one drawn picture and its ground truth into a folder."""
    drawn.image().write(folder / drawn.name)
    AnnotationFile(annotations={drawn.name: drawn.ground_truth()}).write(
        folder / AnnotationFile.default_file_name
    )


def test_an_approach_is_scored_over_every_picture_of_a_folder(
    board: BoardConfiguration, drawn_board: DrawnBoard, tmp_path: Path
) -> None:
    write_folder(tmp_path, drawn_board)
    comparison = PipelineComparison(dataset=ImageFolderDataset(folder=tmp_path, board=board))
    score = comparison.score(PreparedPipeline(board=board, prepared=drawn_board.ground_truth()))
    assert score.counts.found == len(drawn_board.shapes)
    assert score.pipeline_name == PreparedPipeline.__name__


def test_several_approaches_are_scored_on_the_same_pictures(
    board: BoardConfiguration, drawn_board: DrawnBoard, tmp_path: Path
) -> None:
    write_folder(tmp_path, drawn_board)
    truth = drawn_board.ground_truth()
    empty = ImageDetections(image_name=truth.image_name, width=truth.width, height=truth.height)
    comparison = PipelineComparison(dataset=ImageFolderDataset(folder=tmp_path, board=board))
    scores = comparison.compare(
        [
            PreparedPipeline(board=board, prepared=truth),
            PreparedPipeline(board=board, prepared=empty),
        ]
    )
    assert [score.counts.recall for score in scores] == [1.0, 0.0]
    assert comparison.format_table(scores).count(PreparedPipeline.__name__) == len(scores)


def test_a_picture_nobody_annotated_is_refused_rather_than_scored(
    board: BoardConfiguration, drawn_board: DrawnBoard, tmp_path: Path
) -> None:
    drawn_board.image().write(tmp_path / drawn_board.name)
    comparison = PipelineComparison(dataset=ImageFolderDataset(folder=tmp_path, board=board))
    with pytest.raises(MissingAnnotation):
        comparison.score(PreparedPipeline(board=board, prepared=drawn_board.ground_truth()))
