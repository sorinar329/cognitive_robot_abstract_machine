"""
Tests for :mod:`experiments.tracy_experiments.pickup.release_simulation`: a released
piece dropped into a disposable, robot-free copy of the shape-sorting scene falls
through its own hole, rests on the lid over a hole it does not fit through, and never
touches the belief it was read from.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.montessori.pieces import KnownPieceSet
from experiments.montessori.scenarios import SortingScene
from experiments.montessori.semantics import MontessoriShapeCategory
from experiments.tracy_experiments.pickup.pickup_demo_simulated import build_world
from experiments.tracy_experiments.pickup.release_simulation import (
    NoKnownPieceOfCategoryError,
    ReleaseCheck,
)
from segmind.datastructures.events import (
    ContainmentEvent,
    PlacingEvent,
    StopTranslationEvent,
    SupportEvent,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)

RELEASE_CLEARANCE = 0.15
"""
Height above a hole a piece is released at, in metres: clear of the board's lid, so the
release check has a real drop to settle instead of starting in contact.
"""


def _release_above(
    scene: SortingScene, category: MontessoriShapeCategory, over: Point3
):
    """
    Stand the loose piece of ``category`` :data:`RELEASE_CLEARANCE` above ``over``.

    :param scene: The scene the piece stands in, modified in place.
    :param category: The shape of the piece to release.
    :param over: The point, in the world root frame, to release it above.
    :return: The piece, as the scene now holds it.
    """
    scene.stand_the_piece_at(
        category,
        Point3(
            float(over.x),
            float(over.y),
            float(over.z) + RELEASE_CLEARANCE,
            reference_frame=scene.world.root,
        ),
    )
    return scene.shape_of(category)


def test_a_piece_released_over_its_own_hole_falls_through():
    scene = SortingScene(build_world())
    category = MontessoriShapeCategory.CUBE
    hole_position = scene.hole_for(category).root.global_transform.to_position()
    piece = _release_above(scene, category, hole_position)

    outcome = ReleaseCheck().simulate(piece, scene.board)

    assert outcome.fell_through is True
    assert any(isinstance(event, ContainmentEvent) for event in outcome.events)


def test_a_settled_piece_is_detected_as_placed():
    """
    A piece stopping and coming to rest is reported as a placing: the monitor keeps
    ticking a few windows past the moment the piece is found still, so segmind's own
    (coarser) sliding window has entirely stationary samples to conclude "stopped" from.
    """
    scene = SortingScene(build_world())
    category = MontessoriShapeCategory.CUBE
    hole_position = scene.hole_for(category).root.global_transform.to_position()
    piece = _release_above(scene, category, hole_position)

    outcome = ReleaseCheck().simulate(piece, scene.board)

    assert any(isinstance(event, StopTranslationEvent) for event in outcome.events)
    assert any(isinstance(event, PlacingEvent) for event in outcome.events)


def test_a_piece_released_over_a_hole_it_does_not_fit_through_rests_on_the_lid():
    scene = SortingScene(build_world())
    category = MontessoriShapeCategory.CUBE
    wrong_hole_position = scene.hole_for(
        MontessoriShapeCategory.RECTANGULAR_PRISM
    ).root.global_transform.to_position()
    piece = _release_above(scene, category, wrong_hole_position)

    outcome = ReleaseCheck().simulate(piece, scene.board)

    assert outcome.fell_through is False
    assert any(isinstance(event, SupportEvent) for event in outcome.events)
    assert not any(isinstance(event, ContainmentEvent) for event in outcome.events)


def test_simulate_leaves_the_belief_untouched():
    scene = SortingScene(build_world())
    category = MontessoriShapeCategory.CUBE
    hole_position = scene.hole_for(category).root.global_transform.to_position()
    piece = _release_above(scene, category, hole_position)
    pose_before = piece.root.global_transform.to_np().copy()
    connection_before = piece.root.parent_connection

    ReleaseCheck().simulate(piece, scene.board)

    assert np.array_equal(piece.root.global_transform.to_np(), pose_before)
    assert piece.root.parent_connection is connection_before


def test_simulate_raises_for_a_category_the_piece_set_does_not_have():
    scene = SortingScene(build_world())
    category = MontessoriShapeCategory.CUBE
    hole_position = scene.hole_for(category).root.global_transform.to_position()
    piece = _release_above(scene, category, hole_position)

    with pytest.raises(NoKnownPieceOfCategoryError) as excinfo:
        ReleaseCheck(piece_set=KnownPieceSet(pieces=())).simulate(piece, scene.board)

    assert excinfo.value.category is category


BOARD_OFFSET = Point3(0.3, 0.1, 0.0)
"""
How far the belief's board is moved for :func:`test_the_check_follows_the_board_to_its_
own_measured_pose`, in metres.
"""


def _move_board(scene: SortingScene, offset: Point3) -> None:
    """
    Move the scene's board by ``offset``, carrying its holes' landing regions along.

    A hole itself is a child of the board's own body, so moving the board alone already
    carries it along; a landing region hangs off the world root independently of both
    and is moved by hand.

    :param scene: The scene whose board is moved, modified in place.
    :param offset: How far to move it, along the world root frame's own axes.
    """
    world = scene.world

    def _shifted(entity) -> HomogeneousTransformationMatrix:
        moved = entity.global_transform.to_np().copy()
        moved[0, 3] += float(offset.x)
        moved[1, 3] += float(offset.y)
        moved[2, 3] += float(offset.z)
        return HomogeneousTransformationMatrix(moved, reference_frame=world.root)

    landing_regions = [
        hole.landing_region
        for hole in scene.board.apertures
        if hole.landing_region is not None
    ]
    shifted_landing_region_poses = [_shifted(region) for region in landing_regions]

    world.move_branch_to(scene.board.root, _shifted(scene.board.root))
    for region, shifted_pose in zip(landing_regions, shifted_landing_region_poses):
        world.move_branch_to(region, shifted_pose)


def test_the_check_follows_the_board_to_its_own_measured_pose():
    """
    A board standing somewhere other than the check's own default build position -- as a
    perceived one, found wherever the camera saw it, always does -- is still followed:

    the check relocates its own modelled board to match rather than assuming the
    default.
    """
    scene = SortingScene(build_world())
    _move_board(scene, BOARD_OFFSET)
    category = MontessoriShapeCategory.CUBE
    hole_position = scene.hole_for(category).root.global_transform.to_position()
    piece = _release_above(scene, category, hole_position)

    outcome = ReleaseCheck().simulate(piece, scene.board)

    assert outcome.fell_through is True
    assert any(isinstance(event, ContainmentEvent) for event in outcome.events)
