"""A wrong projection silently mislabels every render, so it is checked against known geometry."""

from __future__ import annotations

import math

import pytest

from montessori_vision.board import Circle
from montessori_vision.geometry import Point3D
from montessori_vision.synthetic.layout import BoardLayout
from montessori_vision.synthetic.prism import Prism
from montessori_vision.synthetic.projection import CameraProjection

# %% fixtures


IMAGE_WIDTH = 640
"""Width of the images the projections in this module are checked against."""

IMAGE_HEIGHT = 480
"""Height of the images the projections in this module are checked against."""


def camera(elevation: float = math.radians(90), azimuth: float = 0.0) -> CameraProjection:
    """A camera one metre from the origin, looking at it."""
    return CameraProjection.looking_at_origin(
        distance=1.0,
        elevation=elevation,
        azimuth=azimuth,
        focal_length=50.0,
        sensor_width=36.0,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )


# %% projecting


def test_the_point_the_camera_looks_at_lands_in_the_middle_of_the_image() -> None:
    projected = camera().project(Point3D(0.0, 0.0, 0.0))
    assert (projected.x, projected.y) == pytest.approx((IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2))


def test_a_point_behind_the_camera_does_not_land_on_the_image() -> None:
    assert camera(elevation=math.radians(90)).project(Point3D(0.0, 0.0, 5.0)) is None


def test_a_point_twice_as_far_from_the_axis_lands_twice_as_far_from_the_middle() -> None:
    looking_down = camera()
    near = looking_down.project(Point3D(0.05, 0.0, 0.0))
    far = looking_down.project(Point3D(0.10, 0.0, 0.0))
    assert far.x - IMAGE_WIDTH / 2 == pytest.approx(2 * (near.x - IMAGE_WIDTH / 2))


def test_the_projected_size_follows_the_focal_length() -> None:
    point = Point3D(0.05, 0.0, 0.0)
    wide = camera().project(point)
    narrow = CameraProjection.looking_at_origin(
        distance=1.0,
        elevation=math.radians(90),
        azimuth=0.0,
        focal_length=100.0,
        sensor_width=36.0,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    ).project(point)
    assert narrow.x - IMAGE_WIDTH / 2 == pytest.approx(2 * (wide.x - IMAGE_WIDTH / 2))


# %% boxes around solids


def test_a_solid_under_the_camera_is_boxed_around_the_middle_of_the_image() -> None:
    looking_down = camera()
    prism = Prism.extruded(Circle(segments=32), radius=0.05, bottom=0.0, top=0.02)
    box = looking_down.bounding_box(prism.corners)
    assert box.center.x == pytest.approx(IMAGE_WIDTH / 2, abs=1.0)
    assert box.center.y == pytest.approx(IMAGE_HEIGHT / 2, abs=1.0)


def test_a_solid_off_to_the_side_of_the_frame_gets_no_box() -> None:
    prism = Prism.extruded(Circle(segments=16), radius=0.01, bottom=0.0, top=0.01).moved(
        x=5.0, y=5.0, rotation=0.0
    )
    assert camera().bounding_box(prism.corners) is None


def test_a_wider_solid_gets_a_wider_box() -> None:
    looking_down = camera()
    small = Prism.extruded(Circle(segments=32), radius=0.03, bottom=0.0, top=0.01)
    large = Prism.extruded(Circle(segments=32), radius=0.06, bottom=0.0, top=0.01)
    assert (
        looking_down.bounding_box(large.corners).width
        > looking_down.bounding_box(small.corners).width
    )


# %% solids and the board they sit in


def test_extruding_a_silhouette_gives_two_faces_and_one_side_per_edge() -> None:
    prism = Prism.extruded(Circle(segments=8), radius=1.0, bottom=0.0, top=0.5)
    assert prism.corner_count_per_face == 8
    assert len(prism.faces) == 8 + 2


def test_moving_a_solid_carries_every_corner_with_it() -> None:
    prism = Prism.extruded(Circle(segments=4), radius=1.0, bottom=0.0, top=1.0)
    moved = prism.moved(x=10.0, y=0.0, rotation=0.0)
    assert [corner.x - original.x for corner, original in zip(moved.corners, prism.corners)] == [
        pytest.approx(10.0)
    ] * len(prism.corners)


def test_holes_are_cut_wider_than_the_pieces_that_drop_through_them() -> None:
    layout = BoardLayout()
    assert layout.hole_radius > layout.shape_radius


def test_the_board_grows_to_hold_every_hole_of_its_configuration() -> None:
    layout = BoardLayout(columns=3)
    assert layout.rows(6) == 2
    assert layout.board_size(6).depth < layout.board_size(9).depth


def test_the_holes_are_laid_out_around_the_middle_of_the_board() -> None:
    positions = BoardLayout(columns=3).hole_positions(6)
    assert sum(position.x for position in positions) == pytest.approx(0.0)
    assert sum(position.y for position in positions) == pytest.approx(0.0)
