"""Turning the world positions the renderer knows about into boxes on the rendered image."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING, Iterable, Optional

from montessori_vision.exceptions import EmptyBoundingBox
from montessori_vision.geometry import BoundingBox, Point2D, Point3D

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class CameraProjection:
    """A pinhole camera, enough of one to place a world point on the rendered image.

    Keeping this apart from the renderer is what lets the labels be checked without Blender, and it
    is the piece most likely to be silently wrong.
    """

    world_to_camera: npt.NDArray[np.float64]
    """The four by four matrix taking a world point into camera space, where the camera sits at the
    origin looking down its negative third axis."""

    focal_length: float
    """The focal length in millimetres."""

    sensor_width: float
    """The width of the sensor in millimetres."""

    image_width: int
    """The width of the rendered image in pixels."""

    image_height: int
    """The height of the rendered image in pixels."""

    @property
    def pixels_per_millimetre(self) -> float:
        """How many pixels one millimetre of sensor covers."""
        return self.image_width / self.sensor_width

    def project(self, point: Point3D) -> Optional[Point2D]:
        """Return where a world point lands on the image, or nothing when it is behind the camera.

        The image origin is the top left corner, so a projected point can be compared with a
        detection directly.
        """
        homogeneous = np.append(point.as_array(), 1.0)
        in_camera = self.world_to_camera @ homogeneous
        depth = -in_camera[2]
        if depth <= 0:
            return None
        scale = self.focal_length * self.pixels_per_millimetre / depth
        return Point2D(
            x=self.image_width / 2 + in_camera[0] * scale,
            y=self.image_height / 2 - in_camera[1] * scale,
        )

    def bounding_box(self, points: Iterable[Point3D]) -> Optional[BoundingBox]:
        """Return the box covering every world point that lands on the image.

        Points behind the camera are dropped; a shape entirely behind or beside the camera has no
        box and is left unlabelled rather than labelled wrongly.
        """
        projected = [self.project(point) for point in points]
        visible = [point for point in projected if point is not None]
        if not visible:
            return None
        if not any(
            0 <= point.x <= self.image_width and 0 <= point.y <= self.image_height
            for point in visible
        ):
            return None
        try:
            return BoundingBox.around(visible, self.image_width, self.image_height)
        except EmptyBoundingBox:
            return None

    @classmethod
    def looking_at_origin(
        cls,
        distance: float,
        elevation: float,
        azimuth: float,
        focal_length: float,
        sensor_width: float,
        image_width: int,
        image_height: int,
    ) -> CameraProjection:
        """Build a camera orbiting the world origin, which is where the board is placed.

        The elevation is measured up from the ground plane and the azimuth around the vertical axis.
        """
        position = Point3D(
            x=distance * math.cos(elevation) * math.cos(azimuth),
            y=distance * math.cos(elevation) * math.sin(azimuth),
            z=distance * math.sin(elevation),
        )
        forward = -position.as_array() / np.linalg.norm(position.as_array())
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        rotation = np.stack([right, up, -forward])
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ position.as_array()
        return cls(
            world_to_camera=world_to_camera,
            focal_length=focal_length,
            sensor_width=sensor_width,
            image_width=image_width,
            image_height=image_height,
        )
