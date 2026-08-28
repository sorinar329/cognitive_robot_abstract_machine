"""Measuring how well an approach works, so the two can be compared on the same pictures.

This is the part that answers the question the repository exists for: point it at a folder of camera
frames with their ground truth, hand it both pipelines, and read off which one finds the shapes and
the holes.
"""

from montessori_vision.evaluation.comparison import PipelineComparison
from montessori_vision.evaluation.matching import DetectionMatch, DetectionMatcher, MatchResult
from montessori_vision.evaluation.score import CategoryScore, PipelineScore

__all__ = [
    "CategoryScore",
    "DetectionMatch",
    "DetectionMatcher",
    "MatchResult",
    "PipelineComparison",
    "PipelineScore",
]
