"""The board a robot works on: which shapes exist, how they look and how they are described."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path

import yaml
from typing_extensions import Any, ClassVar

from montessori_vision.exceptions import (
    DegenerateOutline,
    UnknownOutlineType,
    UnknownShapeCategory,
)
from montessori_vision.geometry import Point2D

# %% what a detection can be


class TargetKind(StrEnum):
    """Whether a detected shape is something to pick up or somewhere to insert it."""

    PIECE = "piece"
    """A loose block lying next to or on the board."""

    HOLE = "hole"
    """A cutout in the board that a piece of the same category fits into."""


class ConfigurationKey(StrEnum):
    """The keys a board configuration file uses."""

    CATEGORIES = "categories"
    """The list of shape categories the board holds."""

    NAME = "name"
    """The name of a shape category."""

    OUTLINE = "outline"
    """The description of a category's silhouette."""

    TYPE = "type"
    """The outline type selecting which :class:`ShapeOutline` is built."""

    PIECE_PROMPTS = "piece_prompts"
    """The text descriptions of a category's loose block."""

    HOLE_PROMPTS = "hole_prompts"
    """The text descriptions of a category's cutout."""

    BACKGROUND_PROMPTS = "background_prompts"
    """The text descriptions of everything that is not a board shape."""


# %% silhouettes


@dataclass(frozen=True)
class ShapeOutline(ABC):
    """The silhouette of a shape category, expressed as a closed polygon on the unit circle.

    One outline description drives the rendered mesh, the drawn reference images and the shape
    vocabulary, so a board is described in a single place.
    """

    registered_types: ClassVar[dict[str, type[ShapeOutline]]] = {}
    """Outline implementations by the type name a configuration file uses."""

    configuration_type: ClassVar[str] = ""
    """The type name that selects this implementation in a configuration file."""

    minimum_corner_count: ClassVar[int] = 3
    """The number of corners below which a silhouette encloses no area."""

    def __init_subclass__(cls, **keyword_arguments: Any) -> None:
        super().__init_subclass__(**keyword_arguments)
        if cls.configuration_type:
            ShapeOutline.registered_types[cls.configuration_type] = cls

    @abstractmethod
    def vertices(self, radius: float) -> list[Point2D]:
        """Return the corners of the silhouette, counterclockwise, for a shape of the given size."""

    @classmethod
    def from_configuration(cls, configuration: dict[str, Any]) -> ShapeOutline:
        """Build the outline a configuration entry describes.

        :raises UnknownOutlineType: if the entry names a type that has no implementation.
        """
        arguments = dict(configuration)
        outline_type = arguments.pop(ConfigurationKey.TYPE)
        if outline_type not in cls.registered_types:
            raise UnknownOutlineType(outline_type, tuple(sorted(cls.registered_types)))
        return cls.registered_types[outline_type](**arguments)


@dataclass(frozen=True)
class RegularPolygon(ShapeOutline):
    """A polygon with equally long sides, such as a triangle, square or hexagon."""

    configuration_type: ClassVar[str] = "regular_polygon"

    sides: int
    """The number of sides of the polygon."""

    rotation: float = 0.0
    """Angle in radians the polygon is turned by, so a square can sit axis aligned."""

    def __post_init__(self) -> None:
        if self.sides < self.minimum_corner_count:
            raise DegenerateOutline("A regular polygon", self.sides, self.minimum_corner_count)

    def vertices(self, radius: float) -> list[Point2D]:
        step = 2 * math.pi / self.sides
        return [
            Point2D(
                radius * math.cos(self.rotation + index * step),
                radius * math.sin(self.rotation + index * step),
            )
            for index in range(self.sides)
        ]


@dataclass(frozen=True)
class Circle(ShapeOutline):
    """A circle, approximated by a polygon fine enough that its edges are not visible."""

    configuration_type: ClassVar[str] = "circle"

    segments: int = 64
    """The number of straight segments the circle is approximated by."""

    def __post_init__(self) -> None:
        if self.segments < self.minimum_corner_count:
            raise DegenerateOutline("A circle", self.segments, self.minimum_corner_count)

    def vertices(self, radius: float) -> list[Point2D]:
        step = 2 * math.pi / self.segments
        return [
            Point2D(radius * math.cos(index * step), radius * math.sin(index * step))
            for index in range(self.segments)
        ]


@dataclass(frozen=True)
class Star(ShapeOutline):
    """A star, alternating between points on the outer and on an inner circle."""

    configuration_type: ClassVar[str] = "star"

    points: int
    """The number of outer points of the star."""

    inner_radius_ratio: float
    """The inner radius as a fraction of the outer radius; smaller values give sharper points."""

    rotation: float = math.pi / 2
    """Angle in radians the star is turned by; the default puts one point straight up."""

    def __post_init__(self) -> None:
        if self.points < self.minimum_corner_count:
            raise DegenerateOutline("A star", self.points, self.minimum_corner_count)

    def vertices(self, radius: float) -> list[Point2D]:
        step = math.pi / self.points
        corners = []
        for index in range(2 * self.points):
            corner_radius = radius if index % 2 == 0 else radius * self.inner_radius_ratio
            angle = self.rotation + index * step
            corners.append(
                Point2D(corner_radius * math.cos(angle), corner_radius * math.sin(angle))
            )
        return corners


@dataclass(frozen=True)
class Ellipse(ShapeOutline):
    """An oval, a circle squeezed along one axis."""

    configuration_type: ClassVar[str] = "ellipse"

    width_ratio: float = 1.0
    """The horizontal extent as a fraction of the radius."""

    height_ratio: float = 0.6
    """The vertical extent as a fraction of the radius."""

    segments: int = 64
    """The number of straight segments the ellipse is approximated by."""

    def __post_init__(self) -> None:
        if self.segments < self.minimum_corner_count:
            raise DegenerateOutline("An ellipse", self.segments, self.minimum_corner_count)

    def vertices(self, radius: float) -> list[Point2D]:
        step = 2 * math.pi / self.segments
        return [
            Point2D(
                radius * self.width_ratio * math.cos(index * step),
                radius * self.height_ratio * math.sin(index * step),
            )
            for index in range(self.segments)
        ]


@dataclass(frozen=True)
class Polygon(ShapeOutline):
    """A silhouette spelled out corner by corner, for shapes the other outlines cannot express."""

    configuration_type: ClassVar[str] = "polygon"

    corners: tuple[tuple[float, float], ...]
    """The corners on the unit circle, counterclockwise, as horizontal and vertical fractions."""

    def __post_init__(self) -> None:
        # A configuration file yields nested lists; store tuples so the outline stays hashable.
        object.__setattr__(self, "corners", tuple(tuple(corner) for corner in self.corners))
        if len(self.corners) < self.minimum_corner_count:
            raise DegenerateOutline("A polygon", len(self.corners), self.minimum_corner_count)

    def vertices(self, radius: float) -> list[Point2D]:
        return [Point2D(x * radius, y * radius) for x, y in self.corners]


# %% shape vocabulary


@dataclass(frozen=True)
class ShapeCategory:
    """One shape of the board, together with everything needed to recognise it."""

    name: str
    """The name the board uses for this shape, such as ``star``."""

    outline: ShapeOutline
    """The silhouette used to render the shape and to draw reference images of it."""

    piece_prompts: tuple[str, ...]
    """Text descriptions of the loose block, matched against image crops by a text image model."""

    hole_prompts: tuple[str, ...]
    """Text descriptions of the cutout, matched against image crops by a text image model."""

    def prompts_for(self, kind: TargetKind) -> tuple[str, ...]:
        """Return the text descriptions of this shape in the given role."""
        return self.piece_prompts if kind is TargetKind.PIECE else self.hole_prompts

    @classmethod
    def from_configuration(cls, configuration: dict[str, Any]) -> ShapeCategory:
        """Build a category from one entry of a board configuration file."""
        return cls(
            name=configuration[ConfigurationKey.NAME],
            outline=ShapeOutline.from_configuration(configuration[ConfigurationKey.OUTLINE]),
            piece_prompts=tuple(configuration[ConfigurationKey.PIECE_PROMPTS]),
            hole_prompts=tuple(configuration[ConfigurationKey.HOLE_PROMPTS]),
        )


@dataclass(frozen=True)
class BoardConfiguration:
    """The shape vocabulary of one montessori board.

    The same configuration drives the text prompts of the segment and classify pipeline, the class
    list of the trained detector and the meshes of the synthetic renderer, so a board is described
    once rather than three times.
    """

    default_file_name: ClassVar[str] = "board.yaml"
    """The configuration shipped with the package, describing a common wooden shape sorter."""

    categories: tuple[ShapeCategory, ...]
    """The shapes of the board, in the order the configuration lists them."""

    background_prompts: tuple[str, ...]
    """Text descriptions of everything that is not a board shape, so a classifier can reject a crop
    instead of forcing it into the closest category."""

    @property
    def category_names(self) -> tuple[str, ...]:
        """The names of the shapes of the board, in configuration order."""
        return tuple(category.name for category in self.categories)

    def category(self, name: str) -> ShapeCategory:
        """Return the shape of the given name.

        :raises UnknownShapeCategory: if the board defines no shape of that name.
        """
        for category in self.categories:
            if category.name == name:
                return category
        raise UnknownShapeCategory(name, self.category_names)

    @classmethod
    def from_yaml(cls, path: Path) -> BoardConfiguration:
        """Read a board configuration from a yaml file."""
        return cls.from_configuration(yaml.safe_load(path.read_text()))

    @classmethod
    def from_configuration(cls, configuration: dict[str, Any]) -> BoardConfiguration:
        """Build a board configuration from an already parsed configuration document."""
        return cls(
            categories=tuple(
                ShapeCategory.from_configuration(entry)
                for entry in configuration[ConfigurationKey.CATEGORIES]
            ),
            background_prompts=tuple(configuration[ConfigurationKey.BACKGROUND_PROMPTS]),
        )

    @classmethod
    def default(cls) -> BoardConfiguration:
        """Read the board configuration shipped with the package."""
        package_resources = resources.files("montessori_vision.resources")
        with resources.as_file(package_resources / cls.default_file_name) as path:
            return cls.from_yaml(path)
