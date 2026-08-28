"""Turning matched detections into the numbers an approach is judged by."""

from __future__ import annotations

from dataclasses import dataclass

from montessori_vision.board import TargetKind
from montessori_vision.detections import Detection, ShapeLabel
from montessori_vision.evaluation.matching import MatchResult

# %% counts


@dataclass(frozen=True)
class DetectionCounts:
    """How often a detector was right, wrong, and how often it missed."""

    found: int
    """Annotated shapes a prediction correctly found."""

    false_alarms: int
    """Predictions that found nothing that was annotated."""

    missed: int
    """Annotated shapes no prediction found."""

    total_overlap: float = 0.0
    """The overlaps of the correct predictions added up, from which the mean overlap is taken."""

    @property
    def precision(self) -> float:
        """The share of predictions that were right; one when nothing was predicted."""
        predicted = self.found + self.false_alarms
        return 1.0 if predicted == 0 else self.found / predicted

    @property
    def recall(self) -> float:
        """The share of annotated shapes that were found; one when nothing was annotated."""
        annotated = self.found + self.missed
        return 1.0 if annotated == 0 else self.found / annotated

    @property
    def harmonic_mean(self) -> float:
        """Precision and recall combined, low whenever either of them is low."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def mean_overlap(self) -> float:
        """How tightly the correct predictions sat on the shapes they found."""
        return 0.0 if self.found == 0 else self.total_overlap / self.found

    @classmethod
    def of(cls, result: MatchResult) -> DetectionCounts:
        """Count a matching result."""
        return cls(
            found=len(result.matches),
            false_alarms=len(result.false_positives),
            missed=len(result.false_negatives),
            total_overlap=sum(match.overlap for match in result.matches),
        )


# %% scores


@dataclass(frozen=True)
class CategoryScore:
    """How an approach did on one shape in one role."""

    label: ShapeLabel
    """The shape and role these numbers are about."""

    counts: DetectionCounts
    """How often the approach was right, wrong and missing on this shape."""


@dataclass(frozen=True)
class PipelineScore:
    """How an approach did over a whole folder of images.

    The breakdown per shape is what tells the two approaches apart in practice: an overall number
    hides that one of them never finds the holes.
    """

    pipeline_name: str
    """The approach these numbers belong to."""

    counts: DetectionCounts
    """How the approach did over every shape at once."""

    per_label: tuple[CategoryScore, ...]
    """How the approach did on each shape and role it was asked about."""

    def of_kind(self, kind: TargetKind) -> DetectionCounts:
        """Add up the scores of every shape in one role, so pieces and holes can be compared."""
        selected = [score.counts for score in self.per_label if score.label.kind is kind]
        return DetectionCounts(
            found=sum(counts.found for counts in selected),
            false_alarms=sum(counts.false_alarms for counts in selected),
            missed=sum(counts.missed for counts in selected),
            total_overlap=sum(counts.total_overlap for counts in selected),
        )

    @classmethod
    def of(cls, pipeline_name: str, result: MatchResult) -> PipelineScore:
        """Score a matching result, overall and per shape."""
        return cls(
            pipeline_name=pipeline_name,
            counts=DetectionCounts.of(result),
            per_label=tuple(
                CategoryScore(label=label, counts=DetectionCounts.of(subset))
                for label, subset in cls.split_by_label(result).items()
            ),
        )

    @staticmethod
    def split_by_label(result: MatchResult) -> dict[ShapeLabel, MatchResult]:
        """Split a matching result into one result per shape and role that appears in it."""
        per_label: dict[ShapeLabel, MatchResult] = {}

        def result_for(detection: Detection) -> MatchResult:
            return per_label.setdefault(detection.label, MatchResult())

        for match in result.matches:
            result_for(match.ground_truth).matches.append(match)
        for false_positive in result.false_positives:
            result_for(false_positive).false_positives.append(false_positive)
        for false_negative in result.false_negatives:
            result_for(false_negative).false_negatives.append(false_negative)
        return per_label
