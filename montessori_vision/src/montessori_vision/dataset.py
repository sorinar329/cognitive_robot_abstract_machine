"""Reading the validation images and their ground truth off disk."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import ClassVar, Iterator

from montessori_vision.board import BoardConfiguration
from montessori_vision.detections import ImageDetections
from montessori_vision.exceptions import MissingAnnotation
from montessori_vision.image import Image

# %% file layout


class ImageSuffix(StrEnum):
    """The image file types a dataset folder may hold."""

    PNG = ".png"
    """Portable network graphics, the lossless default."""

    JPEG = ".jpg"
    """Joint photographic experts group, as most camera recordings arrive."""

    JPEG_LONG = ".jpeg"
    """The longer spelling of the same file type."""


# %% ground truth


@dataclass
class AnnotationFile:
    """Ground truth for a folder of images, one entry per annotated image.

    The file is a json list of :func:`montessori_vision.detections.ImageDetections.to_json`
    documents, so the same type describes what was annotated and what a pipeline predicted.
    """

    default_file_name: ClassVar[str] = "annotations.json"
    """The name looked for next to an image folder."""

    annotations: dict[str, ImageDetections] = field(default_factory=dict)
    """The ground truth of each annotated image, by image file name."""

    path: Path = Path(default_file_name)
    """Where the annotations were read from, reported when one is missing."""

    def of(self, image_name: str) -> ImageDetections:
        """Return the ground truth of one image.

        :raises MissingAnnotation: if the image was never annotated.
        """
        if image_name not in self.annotations:
            raise MissingAnnotation(image_name, str(self.path))
        return self.annotations[image_name]

    def write(self, path: Path) -> None:
        """Write every annotation to a json file."""
        payload = [annotation.to_json() for annotation in self.annotations.values()]
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def read(cls, path: Path, board: BoardConfiguration) -> AnnotationFile:
        """Read annotations from a json file, resolving category names against the board."""
        entries = json.loads(path.read_text())
        annotations = [ImageDetections.from_json(entry, board) for entry in entries]
        return cls(
            annotations={annotation.image_name: annotation for annotation in annotations},
            path=path,
        )


# %% images


@dataclass
class ImageFolderDataset:
    """A folder of images, optionally with the ground truth that belongs to them.

    This is the shape the validation frames take once they have been extracted from a recording:
    drop the pictures in a folder and put an annotation file next to them.
    """

    folder: Path
    """The folder the images are read from."""

    board: BoardConfiguration
    """The shape vocabulary the annotations are read against."""

    @property
    def image_paths(self) -> list[Path]:
        """Every image file of the folder, sorted by name."""
        suffixes = {str(suffix) for suffix in ImageSuffix}
        return sorted(path for path in self.folder.iterdir() if path.suffix.lower() in suffixes)

    @property
    def annotation_path(self) -> Path:
        """Where the ground truth of this folder is expected."""
        return self.folder / AnnotationFile.default_file_name

    def images(self) -> Iterator[Image]:
        """Read the images one after another, so a large folder is never held in memory at once."""
        for path in self.image_paths:
            yield Image.read(path)

    def ground_truth(self) -> AnnotationFile:
        """Read the ground truth of this folder, empty when the folder has none."""
        if not self.annotation_path.exists():
            return AnnotationFile(path=self.annotation_path)
        return AnnotationFile.read(self.annotation_path, self.board)
