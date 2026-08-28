"""The class list and the label format are what the two halves of approach two agree on."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from montessori_vision.board import BoardConfiguration, TargetKind
from montessori_vision.detections import Detection, ImageDetections, ShapeLabel
from montessori_vision.exceptions import UnknownClassIndex, UnknownShapeLabel
from montessori_vision.geometry import BoundingBox
from montessori_vision.yolo.class_names import YoloClassNames
from montessori_vision.yolo.dataset import (
    DatasetFolder,
    DatasetSplit,
    DescriptionKey,
    LabelSuffix,
    YoloBox,
    YoloDatasetWriter,
)

# %% class names


def test_a_board_yields_one_class_per_shape_and_role(board: BoardConfiguration) -> None:
    class_names = YoloClassNames.from_board(board)
    assert len(class_names.labels) == len(board.categories) * len(TargetKind)


def test_a_class_index_leads_back_to_the_shape_it_stands_for(board: BoardConfiguration) -> None:
    class_names = YoloClassNames.from_board(board)
    for index, label in enumerate(class_names.labels):
        assert class_names.index_of(label) == index
        assert class_names.label_at(index) is label


def test_a_class_name_joins_the_shape_and_its_role(board: BoardConfiguration) -> None:
    label = ShapeLabel(board.category("star"), TargetKind.HOLE)
    assert label.name == f"star_{TargetKind.HOLE}"


def test_an_index_outside_the_class_list_is_refused(board: BoardConfiguration) -> None:
    class_names = YoloClassNames.from_board(board)
    with pytest.raises(UnknownClassIndex) as raised:
        class_names.label_at(len(class_names.labels))
    assert raised.value.label_count == len(class_names.labels)


def test_a_shape_the_detector_never_saw_is_refused(board: BoardConfiguration) -> None:
    class_names = YoloClassNames(labels=(ShapeLabel(board.category("star"), TargetKind.HOLE),))
    with pytest.raises(UnknownShapeLabel):
        class_names.index_of(ShapeLabel(board.category("circle"), TargetKind.PIECE))


# %% label lines


def test_a_box_survives_being_written_as_fractions_and_read_back() -> None:
    box = BoundingBox(left=10, top=20, right=110, bottom=220)
    written = YoloBox.from_bounding_box(box, class_index=3, image_width=640, image_height=480)
    assert written.to_bounding_box(640, 480) == box


def test_a_label_line_survives_being_written_and_parsed() -> None:
    written = YoloBox.from_bounding_box(
        BoundingBox(left=10, top=20, right=110, bottom=220),
        class_index=3,
        image_width=640,
        image_height=480,
    )
    assert YoloBox.parse(written.format()).format() == written.format()


def test_a_centred_half_sized_box_is_written_as_the_expected_fractions() -> None:
    box = BoundingBox(left=160, top=120, right=480, bottom=360)
    written = YoloBox.from_bounding_box(box, class_index=0, image_width=640, image_height=480)
    assert (written.center_x, written.center_y, written.width, written.height) == (
        0.5,
        0.5,
        0.5,
        0.5,
    )


# %% writing a dataset


def test_a_written_dataset_holds_a_label_file_per_image(
    board: BoardConfiguration, tmp_path: Path
) -> None:
    class_names = YoloClassNames.from_board(board)
    writer = YoloDatasetWriter(root=tmp_path, class_names=class_names)
    writer.prepare()
    label = ShapeLabel(board.category("hexagon"), TargetKind.PIECE)
    detections = ImageDetections(
        image_name="000000.png",
        width=640,
        height=480,
        detections=[Detection(label=label, bounding_box=BoundingBox(10, 20, 110, 220))],
    )
    writer.write_labels(detections, DatasetSplit.TRAIN)

    written = (tmp_path / DatasetFolder.LABELS / DatasetSplit.TRAIN / "000000").with_suffix(
        LabelSuffix.TEXT
    )
    assert written.read_text().split()[0] == str(class_names.index_of(label))
    assert writer.written_image_counts[DatasetSplit.TRAIN] == 1


def test_the_dataset_description_lists_the_classes_in_index_order(
    board: BoardConfiguration, tmp_path: Path
) -> None:
    class_names = YoloClassNames.from_board(board)
    writer = YoloDatasetWriter(root=tmp_path, class_names=class_names)
    writer.prepare()
    description = yaml.safe_load(writer.write_description().read_text())
    assert description[DescriptionKey.NAMES] == class_names.names
    assert description[DescriptionKey.TRAIN].endswith(str(DatasetSplit.TRAIN))
