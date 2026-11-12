"""
Naming the shape of the footprint each object leaves on the table, from its top face in
the colour image: a light alternative to fitting 3D shapes to depth.
"""

from __future__ import annotations

import cv2
import numpy as np
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS
from robokudo.types.annotation import Footprint, FootprintShape
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image

# %% casting top faces onto the table


class FootprintShapeAnnotator(BaseAnnotator):
    """
    Adds a :class:`~robokudo.types.annotation.Footprint` to every object hypothesis that
    has a mask, cast onto the table from the brightest part of the mask.

    The objects must be upright prisms of one height with convex footprints, standing
    on a horizontal table and lit from above.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the footprint annotator.
        """

        class Parameters:
            """
            Where the objects stand and how their footprints are told apart.
            """

            def __init__(self) -> None:
                self.table_height: float = 0.0
                """The height of the table's surface along the world's z-axis, in metres."""
                self.object_height: float = 0.0
                """How tall the objects are, in metres."""
                self.triangle_fill_limit: float = 0.75
                """A footprint filling less of its smallest rectangle than this is a triangle; a triangle fills 1/2 of it, a circle pi/4."""
                self.corner_tolerance: float = 0.03
                """How far, as a share of its perimeter, a footprint's outline may stray from the polygon its corners are counted on."""
                self.circle_corner_count: int = 5
                """A footprint that is no triangle and has at least this many corners is a circle."""
                self.rectangle_elongation: float = 1.5
                """A four-cornered footprint whose smallest rectangle is this many times as long as it is wide is a rectangle, otherwise a square."""

        parameters = Parameters()

    def __init__(
        self,
        name: str = "FootprintShapeAnnotator",
        descriptor: Descriptor | None = None,
    ) -> None:
        super().__init__(name=name, descriptor=descriptor)

    def update(self) -> Status:
        cas = self.get_cas()
        world_T_camera = (
            cas.require_camera_observation().world_T_camera_or_raise().to_np()
        )
        pixels_to_rays = np.linalg.inv(self.mask_intrinsic_matrix(cas))
        brightness = cv2.cvtColor(
            get_scaled_color_image_for_depth_image(cas, cas.color_image),
            cv2.COLOR_BGR2HSV_FULL,
        )[..., 2]
        measured = 0
        for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis):
            if hypothesis.roi.mask is None:
                continue
            footprint = self.outline_on_top_faces(
                self.top_face_outline(hypothesis, brightness),
                pixels_to_rays,
                world_T_camera,
            )
            if cv2.contourArea(footprint) == 0.0:
                continue
            hypothesis.annotations.append(
                self.measure_footprint(footprint, world_T_camera)
            )
            measured += 1
        self.feedback_message = f"Measured {measured} footprints"
        return Status.SUCCESS

    @staticmethod
    def mask_intrinsic_matrix(cas: CAS) -> np.ndarray:
        """
        :param cas: The CAS of the current frame.
        :return: The intrinsic matrix of the image the masks are in, which has the
            depth image's resolution.
        """
        ratio = cas.color_to_depth_ratio
        return (
            cas.require_camera_observation()
            .effective_camera_model.scaled(
                scale_x=ratio.horizontal, scale_y=ratio.vertical
            )
            .intrinsic_matrix
        )

    @staticmethod
    def top_face_outline(
        hypothesis: ObjectHypothesis, brightness: np.ndarray
    ) -> np.ndarray:
        """
        :param hypothesis: A hypothesis with a mask.
        :param brightness: The brightness of every pixel of the image the mask is in.
        :return: The pixels on the outline of the largest bright patch of the mask, in
            the whole image, shape ``(n, 2)``; the whole mask's outline where the mask
            is evenly bright.
        """
        region = hypothesis.roi.roi
        mask = hypothesis.roi.mask > 0
        values = brightness[
            region.pos.y : region.pos.y + region.height,
            region.pos.x : region.pos.x + region.width,
        ]
        threshold, _ = cv2.threshold(
            values[mask].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        bright = ((values > threshold) & mask).astype(np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if not bright.any():
            bright = mask.astype(np.uint8)
        _, labels, statistics, _ = cv2.connectedComponentsWithStats(bright)
        largest = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
        contours, _ = cv2.findContours(
            (labels == largest).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return contours[0].reshape(-1, 2) + [region.pos.x, region.pos.y]

    def outline_on_top_faces(
        self,
        pixels: np.ndarray,
        pixels_to_rays: np.ndarray,
        world_T_camera: np.ndarray,
    ) -> np.ndarray:
        """
        :param pixels: Outline pixels of a top face.
        :param pixels_to_rays: The inverse of the intrinsic matrix of their image.
        :param world_T_camera: The camera's pose in the world.
        :return: The convex hull of where the rays through the pixels meet the plane
            of the objects' top faces, as world ``(x, y)`` corners.
        """
        top = self.descriptor.parameters.table_height + (
            self.descriptor.parameters.object_height
        )
        homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
        directions = homogeneous @ pixels_to_rays.T @ world_T_camera[:3, :3].T
        camera = world_T_camera[:3, 3]
        distances = (top - camera[2]) / directions[:, 2]
        on_plane = camera[:2] + distances[:, None] * directions[:, :2]
        return cv2.convexHull(on_plane.astype(np.float32)).reshape(-1, 2)

    def measure_footprint(
        self, footprint: np.ndarray, world_T_camera: np.ndarray
    ) -> Footprint:
        """
        :param footprint: A convex footprint with an area, as world ``(x, y)`` corners.
        :param world_T_camera: The camera's pose in the world.
        :return: The footprint's shape, size and centre.
        """
        parameters = self.descriptor.parameters
        _, sides, _ = cv2.minAreaRect(footprint)
        length, width = max(sides), min(sides)
        perimeter = cv2.arcLength(footprint, True)
        corners = cv2.approxPolyDP(
            footprint, parameters.corner_tolerance * perimeter, True
        )
        if cv2.contourArea(footprint) / (length * width) < (
            parameters.triangle_fill_limit
        ):
            shape = FootprintShape.TRIANGLE
        elif len(corners) >= parameters.circle_corner_count:
            shape = FootprintShape.CIRCLE
        elif length >= parameters.rectangle_elongation * width:
            shape = FootprintShape.RECTANGLE
        else:
            shape = FootprintShape.SQUARE
        moments = cv2.moments(footprint)
        middle = np.array(
            [
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
                parameters.table_height + parameters.object_height / 2,
            ]
        )
        centre = (middle - world_T_camera[:3, 3]) @ world_T_camera[:3, :3]
        return Footprint(
            shape=shape,
            length=float(length),
            width=float(width),
            centre=centre.tolist(),
            source=type(self).__name__,
        )
