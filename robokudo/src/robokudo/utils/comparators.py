from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from typing_extensions import TYPE_CHECKING, List, Any, Tuple

if TYPE_CHECKING:
    from robokudo.types.annotation import (
        BoundingBox3DAnnotation,
        Classification,
        ColorHistogram,
        SemanticColor,
    )
    from robokudo.types.cv import Rect


class FeatureComparator:
    """Base class for feature comparators."""

    def __init__(self, weight: float):
        self.weight = weight
        """Weight of this comparator in the final similarity score."""

    def compute_similarity(self, query_value: Any, obj_value: Any) -> float:
        """Computes similarity between query and object values.

        :param query_value: The translation to use as a baseline for comparison.
        :param obj_value: The value to compare against query_value.
        :returns: A similarity score between 0.0 (no similarity at all) and 1.0 (completely identical).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class TranslationComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on translation distance between query and object values."""

    def __init__(self, weight: float, max_distance: float):
        super().__init__(weight)

        self.max_distance = max_distance
        """Maximum distance between query and object values for which the similarity is 0.0."""

    def compute_similarity(
        self,
        query_value: Tuple[float, float, float],
        obj_value: Tuple[float, float, float],
    ) -> float:
        """Computes similarity based on translation distance between query and object values.

        :param query_value: The translation to use as a baseline for comparison.
        :param obj_value: The translation to compare against `query_value`.
        :returns: A similarity score between 0.0 (distance equal to or larger than `self.max_distance`) and 1.0 (completely identical).
        """
        distance = euclidean(query_value, obj_value)
        return max(min(1.0 - (distance / self.max_distance), 1.0), 0.0)


class BboxComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on the bounding box size difference between query and object values."""

    def compute_similarity(
        self, query_value: BoundingBox3DAnnotation, obj_value: BoundingBox3DAnnotation
    ) -> float:
        """Computes similarity between bounding boxes based on their sizes.

        :param query_value: The bounding box to use as a baseline for comparison.
        :param obj_value: The bounding box to compare against `query_value`.
        :returns: A similarity score between 0.0 (`obj_value` is 0% the size of `query_value`) and 1.0 (identical bounding box sizes).
        """
        sorted_obj_size = np.sort(
            [obj_value.x_length, obj_value.y_length, obj_value.z_length]
        )
        sorted_query_size = np.sort(
            [query_value.x_length, query_value.y_length, query_value.z_length]
        )
        size_diff = np.abs(sorted_obj_size - sorted_query_size).sum()
        return 1.0 / (1.0 + size_diff)


class SizeComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on the sum difference between query and object values."""

    def compute_similarity(self, query_value: List, obj_value: List) -> float:
        """Computes similarity of two lists based on their sum differences.

        :param query_value: The list to use as a baseline for comparison.
        :param obj_value: The list to compare against `query_value`.
        :returns: A similarity score between 0.0 (sum of `obj_value` is 100% smaller or larger than sum of `query_value`) and 1.0 (identical sums).
        """
        sorted_obj_size = np.sort(query_value)
        sorted_query_size = np.sort(obj_value)
        size_diff = np.abs(sorted_obj_size - sorted_query_size).sum()
        return 1.0 / (1.0 + size_diff)


class ClassnameComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on the string equality between query and object values."""

    def compute_similarity(
        self, query_value: Classification, obj_value: Classification
    ) -> float:
        """Computes similarity between classnames based on string equality.

        :param query_value: The classname to use as a baseline for comparison.
        :param obj_value: The classname to compare against `query_value`.
        :returns: A similarity score between 0.0 (not identical) and 1.0 (identical).
        """
        return 1.0 if query_value.classname == obj_value.classname else 0.0


class HistogramComparator(FeatureComparator):
    """Extended FeatureComparator that computes similarity based on the cv2.HISTCMP_CORREL comparison between query and object values."""

    def compute_similarity(
        self, query_value: ColorHistogram, obj_value: ColorHistogram
    ) -> float:
        """Computes similarity between ColorHistogram objects based on the cv2.HISTCMP_CORREL metric.

        :param query_value: The histogram to use as a baseline for comparison.
        :param obj_value: The histogram to compare against `query_value`.
        :returns: A similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        return abs(
            cv2.compareHist(query_value.hist, obj_value.hist, cv2.HISTCMP_CORREL)
        )


class SemanticColorComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes similarity based on string similarity and color ratio difference between query and object values."""

    def compute_similarity(
        self, query_value: SemanticColor, obj_value: SemanticColor
    ) -> float:
        """Computes similarity between SemanticColor objects based on the cv2.HISTCMP_CORREL metric.

        :param query_value: The semantic color annotation to use as a baseline for comparison.
        :param obj_value: The semantic color annotation to compare against `query_value`.
        :returns: A similarity score between 0.0 (completely different colors or color ratio) and 1.0 (identical color and color ratio).
        """
        same_color = query_value.color == obj_value.color
        if not same_color:
            return 0.0
        return 1.0 - abs(query_value.ratio - obj_value.ratio)


class AdditionalDataComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes similarity based on numerical difference or simple value comparison between query and object values."""

    def compute_similarity(self, query_value: Any, obj_value: Any) -> float:
        """Computes similarity between any value or object based on numerical difference or value equality.

        If `query_value` and `obj_value` are numerical this FeatureComparator will normalize their numerical difference.
        Otherwise a simple equality check is performed, 0.0 is returned if they are not equal, 1.0 if they are equal.

        :param query_value: The object or value to use as a baseline for comparison.
        :param obj_value: The object or value to compare against `query_value`.
        :returns: A similarity score between 0.0 (no similarity) and 1.0 (identical values).
        """
        # Assuming simple equality check for additional data. Modify if needed.
        if isinstance(query_value, (int, float)) and isinstance(
            obj_value, (int, float)
        ):
            return 1.0 / (
                1.0 + abs(query_value - obj_value)
            )  # Normalize numerical difference
        return 1.0 if query_value == obj_value else 0.0


class RoiComparator(FeatureComparator):
    """Extended `FeatureComparator` that computes similarity based on overlap percentage between query and object values."""

    def compute_similarity(self, query_value: Rect, obj_value: Rect) -> float:
        """Computes the similarity of two Region of Interests by calculating their area overlap and returning it as a percentage.

        :param query_value: The rectangle to use as a baseline for comparison.
        :param obj_value: The rectangle to compare against `query_value`.
        :returns: A similarity score between 0.0 (no overlap) and 1.0 (100% overlap).
        """
        rect1 = query_value
        rect1_xyxy = (
            rect1.pos.x,
            rect1.pos.y,
            rect1.pos.x + rect1.width,
            rect1.pos.y + rect1.height,
        )

        rect2 = obj_value
        rect2_xyxy = (
            rect2.pos.x,
            rect2.pos.y,
            rect2.pos.x + rect2.width,
            rect2.pos.y + rect2.height,
        )

        overlap_width = min(rect1_xyxy[2], rect2_xyxy[2]) - max(
            rect1_xyxy[0], rect2_xyxy[0]
        )
        overlap_height = min(rect1_xyxy[3], rect2_xyxy[3]) - max(
            rect1_xyxy[1], rect2_xyxy[1]
        )

        if overlap_width <= 0.0 or overlap_height <= 0.0:
            return 0.0

        overlap_area = overlap_width * overlap_height

        rect1_area = query_value.width * query_value.height
        rect2_area = obj_value.width * obj_value.height

        total_area = (rect1_area + rect2_area) - overlap_area
        return float(overlap_area / total_area)
