"""Score both approaches on the same annotated folder and print the comparison.

This lives outside the package because it is the one place that loads the segmentation model, the
text image model and the trained detector at once, and so needs every optional extra installed.

Run it as::

    python scripts/compare_pipelines.py --images data/frames \
        --sam-checkpoint weights/sam_vit_h.pth --yolo-weights runs/montessori/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from montessori_vision.board import BoardConfiguration
from montessori_vision.dataset import ImageFolderDataset
from montessori_vision.evaluation.comparison import PipelineComparison
from montessori_vision.evaluation.matching import DetectionMatcher
from montessori_vision.pipeline import DetectionPipeline
from montessori_vision.segment_and_classify.clip import ClipClassifier
from montessori_vision.segment_and_classify.pipeline import SegmentAndClassifyPipeline
from montessori_vision.segment_and_classify.segment_anything import (
    SegmentAnythingMaskGenerator,
    SegmentAnythingModelSize,
)
from montessori_vision.yolo.pipeline import YoloPipeline


def parse_arguments() -> argparse.Namespace:
    """Read what to compare and where from."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="folder of annotated pictures")
    parser.add_argument(
        "--board",
        type=Path,
        default=None,
        help="board configuration, defaults to the one shipped with the package",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="Segment Anything checkpoint; the segment and classify approach is skipped without it",
    )
    parser.add_argument(
        "--sam-size",
        type=SegmentAnythingModelSize,
        default=SegmentAnythingModelSize.HUGE,
        choices=list(SegmentAnythingModelSize),
        help="which architecture the checkpoint holds",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=None,
        help="trained detector weights; the synthetic approach is skipped without them",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DetectionMatcher.minimum_overlap,
        help="how much a prediction must overlap an annotation to count as having found it",
    )
    return parser.parse_args()


def build_pipelines(
    arguments: argparse.Namespace, board: BoardConfiguration
) -> list[DetectionPipeline]:
    """Build the approaches whose models were pointed at."""
    pipelines: list[DetectionPipeline] = []
    if arguments.sam_checkpoint is not None:
        pipelines.append(
            SegmentAndClassifyPipeline(
                board=board,
                mask_generator=SegmentAnythingMaskGenerator(
                    checkpoint_path=arguments.sam_checkpoint, model_size=arguments.sam_size
                ),
                classifier=ClipClassifier(board=board),
            )
        )
    if arguments.yolo_weights is not None:
        pipelines.append(YoloPipeline(board=board, weights_path=arguments.yolo_weights))
    return pipelines


def main() -> None:
    """Score every approach that was pointed at a model and print the table."""
    arguments = parse_arguments()
    board = (
        BoardConfiguration.default()
        if arguments.board is None
        else BoardConfiguration.from_yaml(arguments.board)
    )
    comparison = PipelineComparison(
        dataset=ImageFolderDataset(folder=arguments.images, board=board),
        matcher=DetectionMatcher(minimum_overlap=arguments.overlap),
    )
    pipelines = build_pipelines(arguments, board)
    print(comparison.format_table(comparison.compare(pipelines)))


if __name__ == "__main__":
    main()
