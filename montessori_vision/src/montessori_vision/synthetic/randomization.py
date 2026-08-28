"""Deciding what one rendered scene looks like."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from typing_extensions import Optional

from montessori_vision.board import BoardConfiguration, ShapeCategory
from montessori_vision.geometry import Point2D
from montessori_vision.synthetic.layout import BoardLayout

# %% ranges


@dataclass(frozen=True)
class Range:
    """An interval a value is drawn from, inclusive at both ends."""

    lowest: float
    """Smallest value that can be drawn."""

    highest: float
    """Largest value that can be drawn."""

    def sample(self, generator: random.Random) -> float:
        """Draw one value from the interval."""
        return generator.uniform(self.lowest, self.highest)


@dataclass(frozen=True)
class IntegerRange:
    """A whole number interval a count is drawn from, inclusive at both ends."""

    lowest: int
    """Smallest count that can be drawn."""

    highest: int
    """Largest count that can be drawn."""

    def sample(self, generator: random.Random) -> int:
        """Draw one count from the interval."""
        return generator.randint(self.lowest, self.highest)


@dataclass(frozen=True)
class RandomizationRanges:
    """How far each part of the scene is allowed to vary between renders.

    Widening these is what stops a detector from learning the one viewpoint and the one wood colour
    it was shown.

    ..note:: The values are starting points for a board roughly a third of a metre across, filling a
        good part of the frame from a table mounted camera. Measure your own setup and narrow them
        to the range the robot's camera actually sees.
    """

    camera_distance: Range = Range(0.6, 1.2)
    """Metres between the camera and the centre of the board."""

    camera_elevation: Range = Range(math.radians(35), math.radians(85))
    """Radians the camera sits above the plane of the board; straight down is the upper end."""

    camera_azimuth: Range = Range(0.0, 2 * math.pi)
    """Radians the camera is rotated around the board, covering every side of it."""

    focal_length: Range = Range(28.0, 55.0)
    """Millimetres of focal length, covering wide and narrow lenses."""

    board_rotation: Range = Range(0.0, 2 * math.pi)
    """Radians the board itself is turned by on the table."""

    light_energy: Range = Range(30.0, 150.0)
    """Watts of the key light, spanning a dim room and bright daylight for a light at
    :attr:`montessori_vision.synthetic.scene.BlenderScene.light_distance` from the board."""

    light_elevation: Range = Range(math.radians(25), math.radians(90))
    """Radians the key light sits above the plane of the board."""

    light_azimuth: Range = Range(0.0, 2 * math.pi)
    """Radians the key light is rotated around the board."""

    loose_piece_count: IntegerRange = IntegerRange(1, 6)
    """How many pieces lie next to the board rather than in a hole."""

    piece_scatter_radius: Range = Range(0.25, 0.6)
    """Metres from the centre of the board a loose piece may be dropped at.

    A piece is never dropped close enough to stand inside the board, however low this reaches.
    """

    piece_rotation: Range = Range(0.0, 2 * math.pi)
    """Radians a loose piece is turned by where it lies."""

    colour_brightness: Range = Range(0.1, 0.7)
    """How light the randomly tinted materials are, kept off pure black and off the brightness where
    a lit surface washes out."""

    world_brightness: Range = Range(0.02, 0.3)
    """How much light the surroundings add, from a dark room to an overcast window."""

    roughness: Range = Range(0.2, 0.9)
    """How matte the materials are, spanning varnished and bare wood."""


# %% one scene


@dataclass(frozen=True)
class PlacedPiece:
    """One loose piece, with where it was dropped and how it was turned."""

    category: ShapeCategory
    """Which shape the piece is."""

    position: Point2D
    """Where the piece lies on the table, in metres from the centre of the board."""

    rotation: float
    """Radians the piece is turned by around its own vertical axis."""


@dataclass(frozen=True)
class SampledScene:
    """Everything one render needs, drawn from the ranges and holding no renderer state.

    A scene is plain data, so a render can be inspected, stored and reproduced without Blender.
    """

    camera_distance: float
    """Metres between the camera and the centre of the board."""

    camera_elevation: float
    """Radians the camera sits above the plane of the board."""

    camera_azimuth: float
    """Radians the camera is rotated around the board."""

    focal_length: float
    """Millimetres of focal length of the camera."""

    board_rotation: float
    """Radians the board is turned by on the table."""

    light_energy: float
    """Watts of the key light."""

    light_elevation: float
    """Radians the key light sits above the plane of the board."""

    light_azimuth: float
    """Radians the key light is rotated around the board."""

    board_colour: tuple[float, float, float]
    """The red, green and blue of the board material."""

    table_colour: tuple[float, float, float]
    """The red, green and blue of the surface the board rests on, drawn apart from the board so the
    two never blend into one another."""

    world_brightness: float
    """How much light the surroundings add."""

    roughness: float
    """How matte the board and piece materials are."""

    loose_pieces: tuple[PlacedPiece, ...]
    """The pieces lying next to the board rather than sitting in a hole."""

    piece_colours: tuple[tuple[float, float, float], ...]
    """The red, green and blue of each loose piece, in the order the pieces are listed."""


@dataclass
class SceneSampler:
    """Draws scenes from the ranges, reproducibly for a given seed."""

    board: BoardConfiguration
    """The shapes that can appear in a scene."""

    ranges: RandomizationRanges = field(default_factory=RandomizationRanges)
    """How far each part of the scene may vary."""

    layout: BoardLayout = field(default_factory=BoardLayout)
    """The measurements of the board, which decide where a loose piece still lies beside it."""

    seed: Optional[int] = None
    """The seed the scenes are drawn with; the same seed gives the same dataset again."""

    def __post_init__(self) -> None:
        self.generator = random.Random(self.seed)

    def sample(self) -> SampledScene:
        """Draw one scene."""
        pieces = tuple(
            PlacedPiece(
                category=self.generator.choice(self.board.categories),
                position=self.scatter_position(),
                rotation=self.ranges.piece_rotation.sample(self.generator),
            )
            for _ in range(self.ranges.loose_piece_count.sample(self.generator))
        )
        return SampledScene(
            camera_distance=self.ranges.camera_distance.sample(self.generator),
            camera_elevation=self.ranges.camera_elevation.sample(self.generator),
            camera_azimuth=self.ranges.camera_azimuth.sample(self.generator),
            focal_length=self.ranges.focal_length.sample(self.generator),
            board_rotation=self.ranges.board_rotation.sample(self.generator),
            light_energy=self.ranges.light_energy.sample(self.generator),
            light_elevation=self.ranges.light_elevation.sample(self.generator),
            light_azimuth=self.ranges.light_azimuth.sample(self.generator),
            board_colour=self.colour(),
            table_colour=self.colour(),
            world_brightness=self.ranges.world_brightness.sample(self.generator),
            roughness=self.ranges.roughness.sample(self.generator),
            loose_pieces=pieces,
            piece_colours=tuple(self.colour() for _ in pieces),
        )

    @property
    def minimum_scatter_radius(self) -> float:
        """Metres from the centre a loose piece has to be to lie beside the board rather than inside
        it.

        A piece standing in the board would be a solid growing through another, which no camera will
        ever see and which would teach a detector a shape that does not occur.
        """
        size = self.layout.board_size(len(self.board.categories))
        return math.hypot(size.width, size.depth) / 2 + self.layout.shape_radius

    def scatter_position(self) -> Point2D:
        """Draw a spot on the table for a loose piece, spread evenly around the board."""
        radius = max(
            self.ranges.piece_scatter_radius.sample(self.generator), self.minimum_scatter_radius
        )
        angle = self.generator.uniform(0.0, 2 * math.pi)
        return Point2D(radius * math.cos(angle), radius * math.sin(angle))

    def colour(self) -> tuple[float, float, float]:
        """Draw a material colour, kept away from pure black and pure white."""
        return tuple(self.ranges.colour_brightness.sample(self.generator) for _ in range(3))
