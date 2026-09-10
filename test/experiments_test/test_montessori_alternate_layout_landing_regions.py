"""
The two analytically-placed Montessori layouts -- the board on Tracy's own table and
:class:`~experiments.montessori.world2.MontessoriWorld2`'s free-standing board -- give
every hole a landing region that spans its shaft, from just below the surface the board
rests on up to just below the board's top.

:mod:`~experiments.montessori.world`'s own layout measures the open space under each hole
instead and is covered by ``test_montessori_scenarios``; these two build the region
analytically from the two surfaces' heights via
:func:`~experiments.montessori.world._landing_region_height` and
:func:`~experiments.montessori.world._landing_region_position`.
"""

from __future__ import annotations

import pytest
from typing_extensions import List

from experiments.montessori.world import (
    BOARD_SCALE,
    LANDING_REGION_BOTTOM_MARGIN,
    LANDING_REGION_TOP_CLEARANCE,
)
from experiments.montessori.world2 import (
    BOARD_POSITION_2,
    BOARD_TABLE_POSITION,
    BOARD_TABLE_SCALE,
    MontessoriWorld2,
    _HOLES_2,
)
from experiments.tracy_experiments.montessori.world import TracyMontessoriWorld

TRACY_TABLE_TOP_Z = 0.79
"""
Stand-in for the height
:func:`~experiments.tracy_experiments.equipment.tracy_table_mount_position` measures
once Tracy is mounted; the landing-region geometry under test is analytic in this value,
so its exact number does not matter here.
"""


def _axis_extents(area) -> List[float]:
    """
    How far a region's shapes reach along each world axis, in metres.

    :param area: The :class:`~semantic_digital_twin.world_description.shape_collection.ShapeCollection`
        to measure.
    """
    lower, upper = area.combined_mesh.bounds
    return [float(u - l) for l, u in zip(lower, upper)]


def _assert_landing_regions_span_every_shaft(
    landing_regions, hole_specs, table_top_z: float, board_top_z: float
) -> None:
    """
    Assert each hole spec has a landing region whose footprint matches the hole's and
    whose vertical span reaches from :data:`LANDING_REGION_BOTTOM_MARGIN` below
    ``table_top_z`` up to :data:`LANDING_REGION_TOP_CLEARANCE` below ``board_top_z``.
    """
    assert set(landing_regions) == {spec.key for spec in hole_specs}

    for spec in hole_specs:
        region = landing_regions[spec.key]
        width, depth, height = _axis_extents(region.area)
        assert (width, depth) == pytest.approx((spec.shape.size.x, spec.shape.size.y))

        center_z = float(region.global_transform.to_position().z)
        assert center_z - height / 2 == pytest.approx(
            table_top_z - LANDING_REGION_BOTTOM_MARGIN
        )
        assert center_z + height / 2 == pytest.approx(
            board_top_z - LANDING_REGION_TOP_CLEARANCE
        )


def test_tracy_layout_gives_every_hole_a_landing_region_spanning_its_shaft():
    world = TracyMontessoriWorld(
        shapes_are_movable=False, table_top_z=TRACY_TABLE_TOP_Z
    )

    _assert_landing_regions_span_every_shaft(
        world.landing_regions,
        world._hole_specs,
        table_top_z=TRACY_TABLE_TOP_Z,
        board_top_z=TRACY_TABLE_TOP_Z + float(BOARD_SCALE.z),
    )


def test_world2_layout_gives_every_hole_a_landing_region_spanning_its_shaft():
    world = MontessoriWorld2()

    _assert_landing_regions_span_every_shaft(
        world.landing_regions,
        _HOLES_2,
        table_top_z=float(BOARD_TABLE_POSITION.z) + float(BOARD_TABLE_SCALE.z) / 2,
        board_top_z=float(BOARD_POSITION_2.z) + float(BOARD_SCALE.z) / 2,
    )
