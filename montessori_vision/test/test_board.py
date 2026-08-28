"""The board configuration is what every part of the package agrees on, so it is read exactly."""

from __future__ import annotations

import math
from importlib import resources

import pytest

from montessori_vision.board import (
    BoardConfiguration,
    Circle,
    ConfigurationKey,
    Polygon,
    RegularPolygon,
    ShapeOutline,
    Star,
    TargetKind,
)
from montessori_vision.exceptions import (
    DegenerateOutline,
    UnknownOutlineType,
    UnknownShapeCategory,
)

# %% reading a configuration


def test_shipped_configuration_names_its_shapes(board: BoardConfiguration) -> None:
    assert board.category_names == (
        "circle",
        "square",
        "triangle",
        "rectangle",
        "star",
        "hexagon",
    )


def test_configuration_read_from_a_file_matches_the_shipped_one(
    board: BoardConfiguration,
) -> None:
    shipped = resources.files("montessori_vision.resources") / BoardConfiguration.default_file_name
    with resources.as_file(shipped) as path:
        assert BoardConfiguration.from_yaml(path) == board


def test_every_shape_describes_both_of_its_roles(board: BoardConfiguration) -> None:
    for category in board.categories:
        for kind in TargetKind:
            assert category.prompts_for(kind), f"{category.name} has no {kind} description"


def test_an_unknown_shape_is_refused_with_the_names_that_exist(board: BoardConfiguration) -> None:
    with pytest.raises(UnknownShapeCategory) as raised:
        board.category("trapezoid")
    assert raised.value.known_names == board.category_names


# %% outlines


def test_a_regular_polygon_has_one_corner_per_side() -> None:
    assert len(RegularPolygon(sides=6).vertices(1.0)) == 6


def test_a_star_alternates_between_its_outer_and_inner_circle() -> None:
    star = Star(points=5, inner_radius_ratio=0.5)
    radii = [math.hypot(point.x, point.y) for point in star.vertices(2.0)]
    assert radii == pytest.approx([2.0, 1.0] * star.points)


def test_a_circle_is_approximated_by_the_configured_number_of_segments() -> None:
    assert len(Circle(segments=12).vertices(1.0)) == 12


def test_a_polygon_scales_its_corners_to_the_requested_size() -> None:
    outline = Polygon(corners=((1.0, 0.5), (-1.0, 0.5), (0.0, -1.0)))
    scaled = outline.vertices(10.0)
    assert (scaled[0].x, scaled[0].y) == (10.0, 5.0)


def test_an_outline_that_encloses_no_area_is_refused() -> None:
    with pytest.raises(DegenerateOutline) as raised:
        RegularPolygon(sides=2)
    assert raised.value.minimum_corner_count == ShapeOutline.minimum_corner_count


def test_an_outline_type_that_has_no_implementation_is_refused() -> None:
    with pytest.raises(UnknownOutlineType) as raised:
        ShapeOutline.from_configuration({ConfigurationKey.TYPE: "spiral"})
    assert Star.configuration_type in raised.value.known_types
