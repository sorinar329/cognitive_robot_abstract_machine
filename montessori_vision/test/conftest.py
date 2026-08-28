"""Fixtures shared by the tests."""

from __future__ import annotations

import pytest

from montessori_vision.board import BoardConfiguration, TargetKind
from montessori_vision.geometry import Point2D

from .drawing import DrawnBoard, DrawnShape


@pytest.fixture
def board() -> BoardConfiguration:
    """The board configuration shipped with the package."""
    return BoardConfiguration.default()


@pytest.fixture
def drawn_board(board: BoardConfiguration) -> DrawnBoard:
    """A picture holding one hole and two loose pieces, well apart from each other."""
    return DrawnBoard(
        shapes=[
            DrawnShape(board.category("star"), TargetKind.HOLE, Point2D(70, 60), 28),
            DrawnShape(board.category("hexagon"), TargetKind.PIECE, Point2D(230, 70), 30),
            DrawnShape(board.category("square"), TargetKind.PIECE, Point2D(150, 180), 26),
        ]
    )
