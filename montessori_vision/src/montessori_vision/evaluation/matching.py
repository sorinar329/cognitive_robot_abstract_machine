"""Deciding which prediction belongs to which annotated shape."""

from __future__ import annotations

from dataclasses import dataclass, field

from montessori_vision.detections import Detection, ImageDetections


@dataclass(frozen=True)
class DetectionMatch:
    """One prediction paired with the annotated shape it found."""

    ground_truth: Detection
    """The shape that was annotated."""

    prediction: Detection
    """The prediction that was accepted as having found it."""

    overlap: float
    """How much the two boxes overlap, as an intersection over union."""


@dataclass
class MatchResult:
    """How predictions lined up with what was annotated, for one image or for a whole folder."""

    matches: list[DetectionMatch] = field(default_factory=list)
    """The predictions that found an annotated shape."""

    false_positives: list[Detection] = field(default_factory=list)
    """The predictions that found nothing that was annotated."""

    false_negatives: list[Detection] = field(default_factory=list)
    """The annotated shapes no prediction found."""

    def extend(self, other: MatchResult) -> None:
        """Fold another image's result into this one, so a whole folder adds up."""
        self.matches.extend(other.matches)
        self.false_positives.extend(other.false_positives)
        self.false_negatives.extend(other.false_negatives)


@dataclass(frozen=True)
class DetectionMatcher:
    """Pairs the predictions of an image up with what was annotated in it.

    The most confident prediction picks first and takes the annotated shape it overlaps most, which
    is how detection benchmarks resolve two predictions competing for one shape.
    """

    minimum_overlap: float = 0.5
    """How much a prediction's box must overlap an annotated box to count as having found it."""

    def match(self, ground_truth: ImageDetections, predictions: ImageDetections) -> MatchResult:
        """Pair up one image's predictions with its annotations.

        A pair only counts when both name the same shape in the same role: finding a star in the
        right place but calling it a hexagon is a miss and a false alarm, not a hit.
        """
        result = MatchResult(false_negatives=list(ground_truth.detections))
        ordered = sorted(
            predictions.detections, key=lambda prediction: prediction.confidence, reverse=True
        )
        for prediction in ordered:
            found = self.best_remaining(prediction, result.false_negatives)
            if found is None:
                result.false_positives.append(prediction)
                continue
            result.false_negatives.remove(found.ground_truth)
            result.matches.append(found)
        return result

    def best_remaining(
        self, prediction: Detection, candidates: list[Detection]
    ) -> DetectionMatch | None:
        """Return the still unclaimed annotation this prediction found, if it found one."""
        overlapping = [
            DetectionMatch(
                ground_truth=candidate,
                prediction=prediction,
                overlap=candidate.bounding_box.intersection_over_union(prediction.bounding_box),
            )
            for candidate in candidates
            if candidate.label.matches(prediction.label)
        ]
        best = max(overlapping, key=lambda match: match.overlap, default=None)
        if best is None or best.overlap < self.minimum_overlap:
            return None
        return best
