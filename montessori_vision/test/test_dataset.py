"""Ground truth is read once and used by every approach, so its format round-trips exactly."""

from __future__ import annotations

from pathlib import Path

import pytest

from montessori_vision.board import BoardConfiguration, TargetKind
from montessori_vision.dataset import AnnotationFile, ImageFolderDataset, ImageSuffix
from montessori_vision.detections import AnnotationKey, Detection, ImageDetections, ShapeLabel
from montessori_vision.exceptions import MissingAnnotation, UnknownShapeCategory
from montessori_vision.geometry import BoundingBox
from montessori_vision.image import Image

from .drawing import DrawnBoard

# %% the annotation format


def test_annotations_survive_being_written_and_read_back(
    board: BoardConfiguration, drawn_board: DrawnBoard, tmp_path: Path
) -> None:
    truth = drawn_board.ground_truth()
    path = tmp_path / AnnotationFile.default_file_name
    AnnotationFile(annotations={truth.image_name: truth}).write(path)
    read_back = AnnotationFile.read(path, board).of(truth.image_name)

    assert read_back.image_name == truth.image_name
    assert (read_back.width, read_back.height) == (truth.width, truth.height)
    assert [detection.bounding_box for detection in read_back.detections] == [
        detection.bounding_box for detection in truth.detections
    ]
    assert [detection.label.name for detection in read_back.detections] == [
        detection.label.name for detection in truth.detections
    ]


def test_a_detection_is_written_with_the_keys_of_the_annotation_format(
    board: BoardConfiguration,
) -> None:
    detection = Detection(
        label=ShapeLabel(board.category("star"), TargetKind.HOLE),
        bounding_box=BoundingBox(left=1, top=2, right=3, bottom=4),
        confidence=0.5,
    )
    written = detection.to_json()
    assert written[AnnotationKey.CATEGORY] == "star"
    assert written[AnnotationKey.KIND] == TargetKind.HOLE
    assert written[AnnotationKey.CONFIDENCE] == 0.5


def test_an_annotation_naming_a_shape_the_board_lacks_is_refused(
    board: BoardConfiguration,
) -> None:
    payload = {
        AnnotationKey.IMAGE: "a.png",
        AnnotationKey.WIDTH: 10,
        AnnotationKey.HEIGHT: 10,
        AnnotationKey.DETECTIONS: [
            {
                AnnotationKey.CATEGORY: "trapezoid",
                AnnotationKey.KIND: TargetKind.PIECE,
                AnnotationKey.LEFT: 1,
                AnnotationKey.TOP: 1,
                AnnotationKey.RIGHT: 5,
                AnnotationKey.BOTTOM: 5,
            }
        ],
    }
    with pytest.raises(UnknownShapeCategory):
        ImageDetections.from_json(payload, board)


def test_asking_for_an_annotation_that_was_never_made_is_refused() -> None:
    with pytest.raises(MissingAnnotation):
        AnnotationFile().of("never_annotated.png")


# %% a folder of pictures


def test_a_folder_lists_its_pictures_and_ignores_everything_else(
    board: BoardConfiguration, drawn_board: DrawnBoard, tmp_path: Path
) -> None:
    drawn_board.image().write(tmp_path / drawn_board.name)
    (tmp_path / "notes.txt").write_text("not a picture")
    dataset = ImageFolderDataset(folder=tmp_path, board=board)
    assert [path.name for path in dataset.image_paths] == [drawn_board.name]


def test_a_picture_survives_being_written_and_read_back(
    drawn_board: DrawnBoard, tmp_path: Path
) -> None:
    written = drawn_board.image()
    path = (tmp_path / "written").with_suffix(ImageSuffix.PNG)
    written.write(path)
    read_back = Image.read(path)
    assert read_back.pixels.tolist() == written.pixels.tolist()
    assert (read_back.width, read_back.height) == (written.width, written.height)


def test_a_folder_without_annotations_yields_an_empty_ground_truth(
    board: BoardConfiguration, tmp_path: Path
) -> None:
    assert ImageFolderDataset(folder=tmp_path, board=board).ground_truth().annotations == {}
