"""Where the holes sit on the rendered board and how large everything is."""

from __future__ import annotations

from dataclasses import dataclass

from montessori_vision.geometry import Point2D


@dataclass(frozen=True)
class BoardSize:
    """How far the board reaches across the table."""

    width: float
    """Metres along the board's first axis."""

    depth: float
    """Metres along the board's second axis."""


@dataclass(frozen=True)
class BoardLayout:
    """The measurements of the rendered board and the grid its holes are cut in.

    ..note:: These are the measurements of a typical wooden shape sorter. Measure the board in front
        of the robot and put its numbers here, so the renders match what the camera will see.
    """

    shape_radius: float = 0.035
    """Metres from the centre of a shape to its furthest corner."""

    piece_thickness: float = 0.02
    """Metres a loose piece stands above the table."""

    board_thickness: float = 0.012
    """Metres the board itself is thick."""

    hole_clearance: float = 1.08
    """How much wider a hole is cut than its piece, so a piece drops through rather than jamming."""

    hole_spacing: float = 0.09
    """Metres between the centres of two neighbouring holes."""

    columns: int = 3
    """How many holes are cut per row before the next row starts."""

    margin: float = 0.03
    """Metres of board left around the outermost holes."""

    @property
    def hole_radius(self) -> float:
        """Metres from the centre of a hole to its furthest corner."""
        return self.shape_radius * self.hole_clearance

    def rows(self, hole_count: int) -> int:
        """How many rows the given number of holes needs."""
        return -(-hole_count // self.columns)

    def hole_positions(self, hole_count: int) -> list[Point2D]:
        """Return the centre of every hole, laid out in a grid centred on the board.

        Holes are placed left to right and front to back, so the order matches the board
        configuration and a render can be read against it.
        """
        rows = self.rows(hole_count)
        positions = []
        for index in range(hole_count):
            column = index % self.columns
            row = index // self.columns
            positions.append(
                Point2D(
                    x=(column - (self.columns - 1) / 2) * self.hole_spacing,
                    y=((rows - 1) / 2 - row) * self.hole_spacing,
                )
            )
        return positions

    def board_size(self, hole_count: int) -> BoardSize:
        """Return how large the board has to be to hold the given number of holes."""
        return BoardSize(
            width=(self.columns - 1) * self.hole_spacing + 2 * (self.hole_radius + self.margin),
            depth=(self.rows(hole_count) - 1) * self.hole_spacing
            + 2 * (self.hole_radius + self.margin),
        )
