"""Rendering a whole training dataset.

.. warning:: This module imports bpy through :mod:`montessori_vision.synthetic.scene`, which comes
    with the ``blender`` extra. Import it only where that is installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Optional

from montessori_vision.board import BoardConfiguration
from montessori_vision.dataset import ImageSuffix
from montessori_vision.synthetic.layout import BoardLayout
from montessori_vision.synthetic.randomization import RandomizationRanges, SceneSampler
from montessori_vision.synthetic.scene import BlenderScene
from montessori_vision.yolo.class_names import YoloClassNames
from montessori_vision.yolo.dataset import DatasetSplit, YoloDatasetWriter


@dataclass
class SyntheticDatasetGenerator:
    """Renders a dataset of montessori boards and writes it in the format a detector trains from.

    Every render is labelled from the positions the scene was built with, so the dataset needs no
    annotation pass at all.
    """

    board: BoardConfiguration
    """
    The shapes the rendered boards hold.
    """

    root: Path
    """The folder the dataset is written into."""

    image_count: int
    """
    How many images are rendered in total, split between training and validation.
    """

    validation_fraction: float = 0.1
    """The share of the images kept aside to measure the detector on while it trains."""

    seed: Optional[int] = None
    """The seed the scenes are drawn with; the same seed renders the same dataset again."""

    ranges: RandomizationRanges = field(default_factory=RandomizationRanges)
    """How far each part of a scene may vary between renders."""

    layout: BoardLayout = field(default_factory=BoardLayout)
    """The measurements of the rendered board."""

    image_width: int = 640
    """Width of every rendered image in pixels."""

    image_height: int = 480
    """Height of every rendered image in pixels."""

    def generate(self) -> Path:
        """Render every image, write it with its labels, and return the dataset description."""
        writer = YoloDatasetWriter(
            root=self.root, class_names=YoloClassNames.from_board(self.board)
        )
        writer.prepare()
        sampler = SceneSampler(
            board=self.board, ranges=self.ranges, layout=self.layout, seed=self.seed
        )
        scene = BlenderScene(
            board=self.board,
            layout=self.layout,
            image_width=self.image_width,
            image_height=self.image_height,
        )

        for index in range(self.image_count):
            split = self.split_of(index)
            image_name = f"{index:06d}{ImageSuffix.PNG}"
            image_path = writer.image_folder(split) / image_name
            writer.write_labels(scene.render(sampler.sample(), image_path), split)
        return writer.write_description()

    def split_of(self, index: int) -> DatasetSplit:
        """Return which split the image of the given number belongs to."""
        validation_every = max(1, round(1 / self.validation_fraction))
        if self.validation_fraction > 0 and index % validation_every == 0:
            return DatasetSplit.VALIDATION
        return DatasetSplit.TRAIN


def main() -> None:
    """Render a dataset from the command line."""
    parser = argparse.ArgumentParser(description=SyntheticDatasetGenerator.__doc__)
    parser.add_argument("--output", type=Path, required=True, help="folder to write the dataset to")
    parser.add_argument("--images", type=int, default=2000, help="how many images to render")
    parser.add_argument("--seed", type=int, default=None, help="seed the scenes are drawn with")
    parser.add_argument(
        "--board",
        type=Path,
        default=None,
        help="board configuration to render, defaults to the one shipped with the package",
    )
    parser.add_argument("--width", type=int, default=640, help="width of a rendered image")
    parser.add_argument("--height", type=int, default=480, help="height of a rendered image")
    arguments = parser.parse_args()

    board = (
        BoardConfiguration.default()
        if arguments.board is None
        else BoardConfiguration.from_yaml(arguments.board)
    )
    description = SyntheticDatasetGenerator(
        board=board,
        root=arguments.output,
        image_count=arguments.images,
        seed=arguments.seed,
        image_width=arguments.width,
        image_height=arguments.height,
    ).generate()
    print(f"wrote {description}")


if __name__ == "__main__":
    main()
