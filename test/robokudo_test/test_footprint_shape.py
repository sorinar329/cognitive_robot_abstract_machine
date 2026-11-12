"""
Tests for naming the shape of the footprint an object leaves on the table, from its
outline in the colour image.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point2
from py_trees.common import Status
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.footprint_shape import FootprintShapeAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CAS
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Footprint, FootprintShape
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.scene import ObjectHypothesis

# %% a camera looking down at a table at a slant


def tracy_like_camera_model() -> PinholeCameraModel:
    """
    :return: A full-HD camera with the focal length of Tracy's camera.
    """
    return PinholeCameraModel(
        image_resolution=CameraResolution(width=1920, height=1080),
        focal_length_x=1117.0,
        focal_length_y=1117.0,
        principal_point=Point2(x=960.0, y=540.0),
    )


@dataclass
class StandingPrism:
    """
    An upright prism standing on the table.
    """

    outline: np.ndarray
    """
    The corners of its footprint, centred on the origin, in metres.
    """

    centre: np.ndarray
    """
    Where on the table its footprint is centred, as world ``(x, y)``.
    """

    height: float = 0.03
    """
    How tall it is, in metres.
    """

    def corners(self, table_height: float, side_share: float) -> np.ndarray:
        """
        :param table_height: The height of the table it stands on.
        :param side_share: How much of its sides, from the top down, to include.
        :return: The corners of the lowest included part of its sides, then those of
            its top face, in the world.
        """
        on_table = self.outline + self.centre
        top = table_height + self.height
        return np.vstack(
            [
                np.column_stack([on_table, np.full(len(on_table), height)])
                for height in (top - side_share * self.height, top)
            ]
        )


@dataclass
class SlantedCamera:
    """
    A camera above a table, looking down at it at a slant, as Tracy's camera does.
    """

    table_height: float = 0.88
    """
    The height of the table's surface in the world, in metres.
    """

    height_above_table: float = 0.9
    """
    How far the camera is above the table, in metres.
    """

    tilt: float = 0.45
    """
    How far the camera's view is turned away from straight down, in radians.
    """

    model: PinholeCameraModel = field(default_factory=tracy_like_camera_model)
    """
    The camera's projection.
    """

    optical_frame: Body = field(
        default_factory=lambda: Body(name=PrefixedName(name="camera_optical_frame"))
    )
    """
    The frame the camera's image is taken in.
    """

    def world_T_camera(self) -> HomogeneousTransformationMatrix:
        """
        :return: The camera's optical frame in the world.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.0,
            y=0.0,
            z=self.table_height + self.height_above_table,
            roll=math.pi + self.tilt,
            reference_frame=Body(name=PrefixedName(name="map")),
            child_frame=self.optical_frame,
        )

    def middle_of_view(self) -> np.ndarray:
        """
        :return: Where the middle of the image meets the table, as world ``(x, y)``.
        """
        return np.array([0.0, self.height_above_table * math.tan(self.tilt)])

    def pixels(self, points: np.ndarray) -> np.ndarray:
        """
        :param points: World points, shape ``(n, 3)``.
        :return: The pixels they are seen at, shape ``(n, 2)``.
        """
        world_T_camera = self.world_T_camera().to_np()
        in_camera = (points - world_T_camera[:3, 3]) @ world_T_camera[:3, :3]
        projected = in_camera @ self.model.intrinsic_matrix.T
        return projected[:, :2] / projected[:, 2:]

    def cas_seeing(
        self, prism: StandingPrism, side_share: float = 1.0
    ) -> tuple[CAS, ObjectHypothesis]:
        """
        :param prism: A prism standing on the table.
        :param side_share: How much of the prism's sides, from the top down, its mask
            covers, as a colour threshold may leave out shaded sides.
        :return: A CAS from this camera, whose colour image shows the prism lit from
            above on a grey table, and the hypothesis whose mask is the prism's
            silhouette, top face and sides together.
        """
        corners = prism.corners(self.table_height, side_share)
        silhouette = cv2.convexHull(self.pixels(corners).astype(np.float32))
        top_face = self.pixels(corners[len(corners) // 2 :])
        hue_saturation_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        hue_saturation_value[...] = (0, 10, 130)
        cv2.fillConvexPoly(
            hue_saturation_value,
            np.round(silhouette).astype(np.int32),
            (138, 140, 120),
        )
        cv2.fillConvexPoly(
            hue_saturation_value,
            np.round(top_face).astype(np.int32),
            (138, 140, 210),
        )
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(silhouette).astype(np.int32), 255)
        left, top, width, height = cv2.boundingRect(mask)
        hypothesis = ObjectHypothesis()
        hypothesis.roi.roi.pos.x = left
        hypothesis.roi.roi.pos.y = top
        hypothesis.roi.roi.width = width
        hypothesis.roi.roi.height = height
        hypothesis.roi.mask = mask[top : top + height, left : left + width]

        cas = CAS(timestamp=1)
        cas.color_image = cv2.cvtColor(hue_saturation_value, cv2.COLOR_HSV2BGR_FULL)
        cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=1.0, vertical=1.0)
        cas.camera_observation = CameraObservation(
            camera=Camera(
                name=PrefixedName(name="camera"),
                root=self.optical_frame,
                camera_model=self.model,
            ),
            effective_camera_model=self.model,
            world_T_camera=self.world_T_camera(),
            timestamp_nanoseconds=1,
        )
        cas.annotations.append(hypothesis)
        return cas, hypothesis


def rectangle(length: float, width: float) -> np.ndarray:
    """
    :return: The corners of a rectangle centred on the origin.
    """
    return np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ]
    )


def regular_polygon(corner_count: int, circumradius: float) -> np.ndarray:
    """
    :return: The corners of a regular polygon centred on the origin.
    """
    angles = np.linspace(0.0, 2 * math.pi, corner_count, endpoint=False)
    return circumradius * np.column_stack([np.cos(angles), np.sin(angles)])


def turned(outline: np.ndarray, angle: float) -> np.ndarray:
    """
    :return: The outline turned counterclockwise by ``angle`` radians.
    """
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    return outline @ rotation.T


def annotated(cas: CAS, camera: SlantedCamera, object_height: float) -> None:
    """
    Runs the annotator on the CAS, configured for the camera's table.
    """
    pipeline = Pipeline()
    annotator = FootprintShapeAnnotator()
    annotator.descriptor.parameters.table_height = camera.table_height
    annotator.descriptor.parameters.object_height = object_height
    pipeline.add_child(annotator)
    pipeline.setup()
    annotator.get_cas = lambda: cas

    assert annotator.update() == Status.SUCCESS


def footprint_seen(
    camera: SlantedCamera, prism: StandingPrism, side_share: float = 1.0
) -> Footprint:
    """
    :return: The one footprint the annotator gives the prism seen by the camera.
    """
    cas, hypothesis = camera.cas_seeing(prism, side_share)
    annotated(cas, camera, prism.height)
    [footprint] = [a for a in hypothesis.annotations if isinstance(a, Footprint)]
    return footprint


# %% naming the shape


@pytest.mark.parametrize("angle", [0.0, 0.5])
@pytest.mark.parametrize(
    "outline, shape",
    [
        (rectangle(0.03, 0.03), FootprintShape.SQUARE),
        (rectangle(0.06, 0.03), FootprintShape.RECTANGLE),
        (regular_polygon(3, 0.025), FootprintShape.TRIANGLE),
        (regular_polygon(64, 0.015), FootprintShape.CIRCLE),
    ],
)
def test_footprint_shape_is_named_from_its_silhouette(
    outline: np.ndarray,
    shape: FootprintShape,
    angle: float,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    The shape is named from the outline the object leaves on the table, although a
    slanted camera sees its sides along with its top face.
    """
    camera = SlantedCamera()
    prism = StandingPrism(
        outline=turned(outline, angle), centre=camera.middle_of_view() + [0.1, 0.05]
    )

    assert footprint_seen(camera, prism).shape == shape


@pytest.mark.parametrize("side_share", [1.0, 0.4, 0.0])
def test_footprint_length_and_width_are_the_sides_of_its_smallest_rectangle(
    side_share: float, annotator_output_buffer: AnnotatorOutputPerPipelineMap
) -> None:
    """
    The length and width are those of the smallest rectangle around the footprint,
    longer side first, in metres, however much of its sides the mask covers.
    """
    camera = SlantedCamera()
    prism = StandingPrism(
        outline=turned(rectangle(0.03, 0.06), 0.5), centre=camera.middle_of_view()
    )

    footprint = footprint_seen(camera, prism, side_share)

    assert (footprint.length, footprint.width) == pytest.approx((0.06, 0.03), abs=0.002)


def test_footprint_centre_is_the_middle_of_the_object_in_the_camera_frame(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    The centre lies in the middle of the object, half its height above the table, given
    in the camera's frame.
    """
    camera = SlantedCamera()
    prism = StandingPrism(
        outline=rectangle(0.03, 0.03), centre=camera.middle_of_view() + [0.1, 0.05]
    )

    footprint = footprint_seen(camera, prism)

    world_T_camera = camera.world_T_camera().to_np()
    middle = np.array([*prism.centre, camera.table_height + prism.height / 2])
    expected = (middle - world_T_camera[:3, 3]) @ world_T_camera[:3, :3]
    np.testing.assert_allclose(footprint.centre, expected, atol=0.002)


def test_hypothesis_without_mask_gets_no_footprint(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    A hypothesis whose outline was never found in the image is left without a footprint.
    """
    camera = SlantedCamera()
    cas, hypothesis = camera.cas_seeing(
        StandingPrism(outline=rectangle(0.03, 0.03), centre=camera.middle_of_view())
    )
    hypothesis.roi.mask = None

    annotated(cas, camera, object_height=0.03)

    assert not [a for a in hypothesis.annotations if isinstance(a, Footprint)]
