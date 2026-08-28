"""Turning a flat silhouette into the solid the renderer places in the world."""

from __future__ import annotations

import math
from dataclasses import dataclass

from montessori_vision.board import ShapeOutline
from montessori_vision.geometry import Point3D


@dataclass(frozen=True)
class Prism:
    """A silhouette given a thickness: the shape of both a montessori piece and the hole it fits.

    The corners are what the label of a rendered shape is derived from, so this is kept apart from
    the renderer and checked on its own.
    """

    corners: tuple[Point3D, ...]
    """The corners of the solid, the bottom face first and the top face after it, in the order the
    silhouette lists them."""

    faces: tuple[tuple[int, ...], ...]
    """The faces as indices into the corners: the bottom, the top, and one quadrilateral per
    side."""

    @property
    def corner_count_per_face(self) -> int:
        """How many corners the silhouette has."""
        return len(self.corners) // 2

    @classmethod
    def extruded(cls, outline: ShapeOutline, radius: float, bottom: float, top: float) -> Prism:
        """Raise a silhouette of the given size from one height to another."""
        silhouette = outline.vertices(radius)
        count = len(silhouette)
        corners = [Point3D(point.x, point.y, bottom) for point in silhouette]
        corners += [Point3D(point.x, point.y, top) for point in silhouette]

        bottom_face = tuple(reversed(range(count)))
        top_face = tuple(range(count, 2 * count))
        sides = tuple(
            (index, (index + 1) % count, (index + 1) % count + count, index + count)
            for index in range(count)
        )
        return cls(corners=tuple(corners), faces=(bottom_face, top_face) + sides)

    def moved(self, x: float, y: float, rotation: float) -> Prism:
        """Return the solid turned around its own vertical axis and set down somewhere else."""
        cosine, sine = math.cos(rotation), math.sin(rotation)
        return Prism(
            corners=tuple(
                Point3D(
                    x=x + corner.x * cosine - corner.y * sine,
                    y=y + corner.x * sine + corner.y * cosine,
                    z=corner.z,
                )
                for corner in self.corners
            ),
            faces=self.faces,
        )
