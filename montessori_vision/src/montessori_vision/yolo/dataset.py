"""The on disk format a detector is trained from."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml
from typing_extensions import ClassVar

from montessori_vision.detections import Detection, ImageDetections
from montessori_vision.geometry import BoundingBox
from montessori_vision.image import Image
from montessori_vision.yolo.class_names import YoloClassNames

# %% file layout


class DatasetSplit(StrEnum):
    """The parts a training dataset is divided into."""

    TRAIN = "train"
    """The renders the detector learns from."""

    VALIDATION = "val"
    """The renders the detector is measured on while training."""


class DatasetFolder(StrEnum):
    """The folder names the training format expects."""

    IMAGES = "images"
    """Holds the rendered pictures."""

    LABELS = "labels"
    """Holds one label file per picture, under the same name."""


class LabelSuffix(StrEnum):
    """The file type one picture's labels are written in."""

    TEXT = ".txt"
    """One line of plain text per labelled shape, next to the picture of the same name."""


class DescriptionKey(StrEnum):
    """The keys of the dataset description a trainer is pointed at."""

    PATH = "path"
    """The root folder the other paths are relative to."""

    TRAIN = "train"
    """Where the training images live."""

    VALIDATION = "val"
    """Where the validation images live."""

    NAMES = "names"
    """The class names, in class index order."""


# %% labels


@dataclass(frozen=True)
class YoloBox:
    """
    One line of a label file: a class and a box given relative to the image size.

    Storing the box as fractions is what makes a label survive being resized, and it is the format
    the trainer reads, so the conversion lives here rather than at every call site.
    """

    class_index: int
    """The class the box belongs to, an index into a :class:`YoloClassNames`."""

    center_x: float
    """Horizontal centre of the box as a fraction of the image width."""

    center_y: float
    """Vertical centre of the box as a fraction of the image height."""

    width: float
    """Width of the box as a fraction of the image width."""

    height: float
    """Height of the box as a fraction of the image height."""

    decimal_places: ClassVar[int] = 6
    """How precisely a fraction is written, enough to place an edge on a pixel of an 8K image."""

    def format(self) -> str:
        """Render the box as the single line a label file holds for it."""
        numbers = (self.center_x, self.center_y, self.width, self.height)
        return " ".join(
            [str(self.class_index)] + [f"{number:.{self.decimal_places}f}" for number in numbers]
        )

    def to_bounding_box(self, image_width: int, image_height: int) -> BoundingBox:
        """Return the box in pixels for an image of the given size."""
        half_width = self.width * image_width / 2
        half_height = self.height * image_height / 2
        center_x = self.center_x * image_width
        center_y = self.center_y * image_height
        return BoundingBox(
            left=max(0, round(center_x - half_width)),
            top=max(0, round(center_y - half_height)),
            right=min(image_width, round(center_x + half_width)),
            bottom=min(image_height, round(center_y + half_height)),
        )

    @classmethod
    def parse(cls, line: str) -> YoloBox:
        """Read a box back from one line of a label file."""
        class_index, center_x, center_y, width, height = line.split()
        return cls(
            class_index=int(class_index),
            center_x=float(center_x),
            center_y=float(center_y),
            width=float(width),
            height=float(height),
        )

    @classmethod
    def from_bounding_box(
        cls, box: BoundingBox, class_index: int, image_width: int, image_height: int
    ) -> YoloBox:
        """Express a pixel box as fractions of the image it was found in."""
        return cls(
            class_index=class_index,
            center_x=box.center.x / image_width,
            center_y=box.center.y / image_height,
            width=box.width / image_width,
            height=box.height / image_height,
        )


# %% writing


@dataclass
class YoloDatasetWriter:
    """Lays out rendered images and their labels the way a trainer expects to find them.

    The class list comes from the board configuration, so the dataset description a trainer reads
    and the mapping used to interpret its predictions cannot drift apart.
    """

    root: Path
    """The folder the dataset is written into."""

    class_names: YoloClassNames
    """The mapping from board shapes to class indices this dataset is labelled with."""

    description_file_name: ClassVar[str] = "data.yaml"
    """The dataset description a trainer is pointed at."""

    written_image_counts: dict[DatasetSplit, int] = field(default_factory=dict)
    """How many images have been labelled per split so far."""

    def image_folder(self, split: DatasetSplit) -> Path:
        """Where the pictures of one split are written."""
        return self.root / DatasetFolder.IMAGES / split

    def label_folder(self, split: DatasetSplit) -> Path:
        """Where the label files of one split are written."""
        return self.root / DatasetFolder.LABELS / split

    def prepare(self) -> None:
        """Create the folders of every split."""
        for split in DatasetSplit:
            self.image_folder(split).mkdir(parents=True, exist_ok=True)
            self.label_folder(split).mkdir(parents=True, exist_ok=True)

    def write(self, image: Image, detections: ImageDetections, split: DatasetSplit) -> None:
        """Write one picture and the labels of the shapes it shows."""
        image.write(self.image_folder(split) / image.name)
        self.write_labels(detections, split)

    def write_labels(self, detections: ImageDetections, split: DatasetSplit) -> None:
        """Write the labels of one picture that is already on disk.

        A renderer writes its own image file, so labelling it is kept separate from writing it.
        """
        label_path = (self.label_folder(split) / detections.image_name).with_suffix(
            LabelSuffix.TEXT
        )
        label_path.write_text(self.format_labels(detections))
        self.written_image_counts[split] = self.written_image_counts.get(split, 0) + 1

    def format_labels(self, detections: ImageDetections) -> str:
        """Render every detection of an image as the lines of its label file."""
        lines = [self.format_label(detection, detections) for detection in detections.detections]
        return "\n".join(lines) + "\n" if lines else ""

    def format_label(self, detection: Detection, detections: ImageDetections) -> str:
        """Render one detection as the line of a label file that describes it."""
        return YoloBox.from_bounding_box(
            box=detection.bounding_box,
            class_index=self.class_names.index_of(detection.label),
            image_width=detections.width,
            image_height=detections.height,
        ).format()

    def write_description(self) -> Path:
        """Write the dataset description a trainer is pointed at and return its path."""
        description = {
            DescriptionKey.PATH: str(self.root.resolve()),
            DescriptionKey.TRAIN: f"{DatasetFolder.IMAGES}/{DatasetSplit.TRAIN}",
            DescriptionKey.VALIDATION: f"{DatasetFolder.IMAGES}/{DatasetSplit.VALIDATION}",
            DescriptionKey.NAMES: self.class_names.names,
        }
        path = self.root / self.description_file_name
        path.write_text(yaml.safe_dump({str(key): value for key, value in description.items()}))
        return path
