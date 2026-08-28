"""Pixel and plane geometry shared by the detection pipelines and the synthetic renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING, Iterable

from montessori_vision.exceptions import EmptyBoundingBox, EmptyMask

if TYPE_CHECKING:
    import numpy.typing as npt


# %% points


@dataclass(frozen=True)
class Point2D:
    """A point on a plane, in whatever unit the owning object uses."""

    x: float
    """Coordinate along the horizontal axis."""

    y: float
    """Coordinate along the vertical axis."""

    def scaled(self, factor: float) -> Point2D:
        """Return this point moved away from the origin by the given factor."""
        return Point2D(self.x * factor, self.y * factor)

    def rotated(self, angle: float) -> Point2D:
        """Return this point rotated around the origin by an angle in radians."""
        cosine, sine = math.cos(angle), math.sin(angle)
        return Point2D(self.x * cosine - self.y * sine, self.x * sine + self.y * cosine)


@dataclass(frozen=True)
class Point3D:
    """A point in world space, in metres."""

    x: float
    """Coordinate along the world's forward axis."""

    y: float
    """Coordinate along the world's left axis."""

    z: float
    """Coordinate along the world's up axis."""

    def as_array(self) -> npt.NDArray[np.float64]:
        """Return the coordinates as a length three array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)


# %% bounding boxes


@dataclass(frozen=True)
class BoundingBox:
    """An axis aligned box in pixel coordinates.

    The right and bottom edges are exclusive, matching how numpy slices an image.
    """

    left: int
    """Leftmost pixel column inside the box."""

    top: int
    """Topmost pixel row inside the box."""

    right: int
    """Pixel column just past the right edge of the box."""

    bottom: int
    """Pixel row just past the bottom edge of the box."""

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise EmptyBoundingBox(self.left, self.top, self.right, self.bottom)

    @property
    def width(self) -> int:
        """The horizontal extent of the box in pixels."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """The vertical extent of the box in pixels."""
        return self.bottom - self.top

    @property
    def area(self) -> int:
        """The number of pixels the box covers."""
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        """The centre of the box in pixel coordinates."""
        return Point2D((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def intersection_over_union(self, other: BoundingBox) -> float:
        """Return the overlap of two boxes as a fraction of the area they jointly cover."""
        overlap_left = max(self.left, other.left)
        overlap_top = max(self.top, other.top)
        overlap_right = min(self.right, other.right)
        overlap_bottom = min(self.bottom, other.bottom)
        if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
            return 0.0
        intersection = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        return intersection / (self.area + other.area - intersection)

    def padded(self, padding: int, width: int, height: int) -> BoundingBox:
        """Return the box grown on every side by the given padding, clipped to an image size."""
        return BoundingBox(
            left=max(0, self.left - padding),
            top=max(0, self.top - padding),
            right=min(width, self.right + padding),
            bottom=min(height, self.bottom + padding),
        )

    def crop(self, pixels: npt.NDArray) -> npt.NDArray:
        """Return the part of an image array that the box covers."""
        return pixels[self.top : self.bottom, self.left : self.right]

    @classmethod
    def around(cls, points: Iterable[Point2D], width: int, height: int) -> BoundingBox:
        """Return the smallest box that contains the given points, clipped to an image size.

        :raises EmptyBoundingBox: if the points lie entirely outside the image.
        """
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        return cls(
            left=max(0, math.floor(min(xs))),
            top=max(0, math.floor(min(ys))),
            right=min(width, math.ceil(max(xs))),
            bottom=min(height, math.ceil(max(ys))),
        )

    @classmethod
    def from_mask(cls, mask: npt.NDArray[np.bool_]) -> BoundingBox:
        """Return the smallest box that contains every set pixel of a binary mask.

        :raises EmptyMask: if no pixel of the mask is set.
        """
        set_rows = np.flatnonzero(mask.any(axis=1))
        set_columns = np.flatnonzero(mask.any(axis=0))
        if set_rows.size == 0:
            raise EmptyMask()
        return cls(
            left=int(set_columns[0]),
            top=int(set_rows[0]),
            right=int(set_columns[-1]) + 1,
            bottom=int(set_rows[-1]) + 1,
        )
