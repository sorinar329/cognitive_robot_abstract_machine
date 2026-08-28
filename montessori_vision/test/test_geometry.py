"""Boxes decide whether a detection counts as a hit, so their arithmetic is pinned down."""

from __future__ import annotations

import numpy as np
import pytest

from montessori_vision.exceptions import EmptyBoundingBox, EmptyMask
from montessori_vision.geometry import BoundingBox, Point2D

# %% measurements


def test_a_box_reports_the_pixels_it_spans() -> None:
    box = BoundingBox(left=10, top=20, right=40, bottom=60)
    assert (box.width, box.height, box.area) == (30, 40, 1200)


def test_the_centre_of_a_box_sits_between_its_edges() -> None:
    assert BoundingBox(left=10, top=20, right=40, bottom=60).center == Point2D(25.0, 40.0)


def test_a_box_with_no_extent_is_refused() -> None:
    with pytest.raises(EmptyBoundingBox):
        BoundingBox(left=10, top=20, right=10, bottom=60)


# %% overlap


def test_two_boxes_overlapping_in_a_quarter_report_that_share() -> None:
    first = BoundingBox(left=0, top=0, right=10, bottom=10)
    second = BoundingBox(left=5, top=5, right=15, bottom=15)
    # Twenty five overlapping pixels out of the one hundred and seventy five they jointly cover.
    assert first.intersection_over_union(second) == pytest.approx(25 / 175)


def test_a_box_overlaps_itself_completely() -> None:
    box = BoundingBox(left=3, top=4, right=9, bottom=11)
    assert box.intersection_over_union(box) == 1.0


def test_boxes_that_only_touch_do_not_overlap() -> None:
    first = BoundingBox(left=0, top=0, right=10, bottom=10)
    second = BoundingBox(left=10, top=0, right=20, bottom=10)
    assert first.intersection_over_union(second) == 0.0


# %% derived boxes


def test_a_box_around_a_mask_hugs_its_set_pixels() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:8] = True
    assert BoundingBox.from_mask(mask) == BoundingBox(left=3, top=2, right=8, bottom=5)


def test_a_mask_without_a_set_pixel_has_no_box() -> None:
    with pytest.raises(EmptyMask):
        BoundingBox.from_mask(np.zeros((4, 4), dtype=bool))


def test_padding_a_box_stops_at_the_edge_of_the_image() -> None:
    box = BoundingBox(left=2, top=2, right=8, bottom=8)
    assert box.padded(padding=5, width=10, height=10) == BoundingBox(
        left=0, top=0, right=10, bottom=10
    )


def test_a_box_crops_exactly_the_pixels_it_covers() -> None:
    pixels = np.arange(100).reshape(10, 10)
    cropped = BoundingBox(left=1, top=2, right=4, bottom=5).crop(pixels)
    assert cropped.shape == (3, 3)
    assert cropped[0, 0] == pixels[2, 1]
