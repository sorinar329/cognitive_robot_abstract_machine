"""Randomising the scenes is what makes the renders worth training on, so the ranges are checked."""

from __future__ import annotations

import math
import random

import pytest

from montessori_vision.board import BoardConfiguration
from montessori_vision.synthetic.layout import BoardLayout
from montessori_vision.synthetic.randomization import (
    IntegerRange,
    RandomizationRanges,
    Range,
    SceneSampler,
)

# %% reproducibility


def test_the_same_seed_draws_the_same_scene(board: BoardConfiguration) -> None:
    assert (
        SceneSampler(board=board, seed=11).sample() == SceneSampler(board=board, seed=11).sample()
    )


def test_different_seeds_draw_different_scenes(board: BoardConfiguration) -> None:
    assert (
        SceneSampler(board=board, seed=11).sample() != SceneSampler(board=board, seed=12).sample()
    )


def test_successive_scenes_of_one_sampler_differ(board: BoardConfiguration) -> None:
    sampler = SceneSampler(board=board, seed=11)
    assert sampler.sample() != sampler.sample()


# %% staying inside the ranges


def test_every_drawn_value_stays_inside_its_range(board: BoardConfiguration) -> None:
    ranges = RandomizationRanges()
    sampler = SceneSampler(board=board, ranges=ranges, seed=5)
    for _ in range(20):
        scene = sampler.sample()
        assert (
            ranges.camera_distance.lowest <= scene.camera_distance <= ranges.camera_distance.highest
        )
        assert (
            ranges.camera_elevation.lowest
            <= scene.camera_elevation
            <= ranges.camera_elevation.highest
        )
        assert ranges.light_energy.lowest <= scene.light_energy <= ranges.light_energy.highest
        assert (
            ranges.loose_piece_count.lowest
            <= len(scene.loose_pieces)
            <= ranges.loose_piece_count.highest
        )


def test_a_colour_is_drawn_for_every_loose_piece(board: BoardConfiguration) -> None:
    scene = SceneSampler(board=board, seed=5).sample()
    assert len(scene.piece_colours) == len(scene.loose_pieces)


def test_the_board_and_the_table_are_coloured_apart(board: BoardConfiguration) -> None:
    drawn = [SceneSampler(board=board, seed=seed).sample() for seed in range(10)]
    assert any(scene.board_colour != scene.table_colour for scene in drawn)


# %% pieces beside the board rather than inside it


def test_a_loose_piece_is_never_dropped_inside_the_board(board: BoardConfiguration) -> None:
    layout = BoardLayout()
    sampler = SceneSampler(
        board=board,
        # A range that would drop pieces on top of the board if it were followed blindly.
        ranges=RandomizationRanges(piece_scatter_radius=Range(0.0, 0.01)),
        layout=layout,
        seed=5,
    )
    for _ in range(20):
        for piece in sampler.sample().loose_pieces:
            distance = math.hypot(piece.position.x, piece.position.y)
            # The whole range lies inside the board, so every piece is pushed out to the clearance.
            assert distance == pytest.approx(sampler.minimum_scatter_radius)


def test_the_clearance_a_piece_keeps_grows_with_the_board(board: BoardConfiguration) -> None:
    narrow = SceneSampler(board=board, layout=BoardLayout(hole_spacing=0.06))
    wide = SceneSampler(board=board, layout=BoardLayout(hole_spacing=0.12))
    assert wide.minimum_scatter_radius > narrow.minimum_scatter_radius


# %% the ranges themselves


def test_a_range_only_ever_yields_values_between_its_ends() -> None:
    generator = random.Random(0)
    drawn = [Range(2.0, 3.0).sample(generator) for _ in range(50)]
    assert all(2.0 <= value <= 3.0 for value in drawn)


def test_a_whole_number_range_includes_both_of_its_ends() -> None:
    generator = random.Random(0)
    drawn = {IntegerRange(1, 3).sample(generator) for _ in range(50)}
    assert drawn == {1, 2, 3}
