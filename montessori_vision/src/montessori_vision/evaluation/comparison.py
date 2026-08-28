"""Running several approaches over the same pictures and putting the numbers side by side."""

from __future__ import annotations

from dataclasses import dataclass, field

from montessori_vision.board import TargetKind
from montessori_vision.dataset import ImageFolderDataset
from montessori_vision.evaluation.matching import DetectionMatcher, MatchResult
from montessori_vision.evaluation.score import DetectionCounts, PipelineScore
from montessori_vision.pipeline import DetectionPipeline


@dataclass
class PipelineComparison:
    """Scores several approaches on the same annotated folder.

    Every approach sees the same pictures and is matched the same way, so the difference in the
    numbers is a difference between the approaches rather than between two evaluations.
    """

    dataset: ImageFolderDataset
    """
    The annotated pictures every approach is run over.
    """

    matcher: DetectionMatcher = field(default_factory=DetectionMatcher)
    """How a prediction is decided to have found an annotated shape."""

    def score(self, pipeline: DetectionPipeline) -> PipelineScore:
        """Run one approach over every annotated picture and score what it found."""
        ground_truth = self.dataset.ground_truth()
        total = MatchResult()
        for image in self.dataset.images():
            annotated = ground_truth.of(image.name)
            total.extend(self.matcher.match(annotated, pipeline.detect(image)))
        return PipelineScore.of(pipeline.name, total)

    def compare(self, pipelines: list[DetectionPipeline]) -> list[PipelineScore]:
        """Score every approach, in the order they were given."""
        return [self.score(pipeline) for pipeline in pipelines]

    def format_table(self, scores: list[PipelineScore]) -> str:
        """Render the scores as a table, overall and split into pieces and holes."""
        header = f"{'pipeline':<28}{'precision':>10}{'recall':>9}{'f1':>7}{'overlap':>9}"
        lines = [header, "-" * len(header)]
        for score in scores:
            lines.append(self.format_row(score.pipeline_name, score.counts))
            for kind in TargetKind:
                lines.append(self.format_row(f"  {kind}", score.of_kind(kind)))
        return "\n".join(lines)

    @staticmethod
    def format_row(name: str, counts: DetectionCounts) -> str:
        """Render one line of the table."""
        return (
            f"{name:<28}{counts.precision:>10.3f}{counts.recall:>9.3f}"
            f"{counts.harmonic_mean:>7.3f}{counts.mean_overlap:>9.3f}"
        )
