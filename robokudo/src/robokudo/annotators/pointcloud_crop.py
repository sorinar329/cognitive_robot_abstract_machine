"""Point cloud cropping and visualization.

This module provides an annotator for:

* Cropping point clouds using axis-aligned bounding boxes
* Supporting both sensor and world coordinate frames
* Generating and visualizing point cloud masks
* Combining masks with color images

.. note::
   Cropping can be done in either sensor coordinates (default) or world coordinates.
"""
from timeit import default_timer

import cv2
import open3d as o3d
import py_trees

import robokudo.annotators
import robokudo.annotators.core
import robokudo.annotators.outputs
import robokudo.utils.annotator_helper
from robokudo.cas import CASViews
from robokudo.utils.o3d_helper import get_mask_from_pointcloud


class PointcloudCropAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Point cloud cropping using axis-aligned bounding boxes.

    Crop a subset of points from a pointcloud data based on min/max X,Y,Z values.
    The crop is either done in sensor coordinates (default) or relative to the world frame.

    .. warning::
       When using world coordinates, requires valid camera-to-world transform in CAS.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for point cloud cropping."""

        class Parameters:
            """Parameters for configuring point cloud cropping.

            Bounding box parameters:

            :ivar min_x: Minimum X coordinate, defaults to -2.0
            :type min_x: float
            :ivar min_y: Minimum Y coordinate, defaults to -2.0
            :type min_y: float
            :ivar min_z: Minimum Z coordinate, defaults to -9.0
            :type min_z: float
            :ivar max_x: Maximum X coordinate, defaults to 2.0
            :type max_x: float
            :ivar max_y: Maximum Y coordinate, defaults to 2.0
            :type max_y: float
            :ivar max_z: Maximum Z coordinate, defaults to 3.0
            :type max_z: float

            Coordinate frame:

            :ivar relative_to_world: Whether to crop in world coordinates, defaults to False
            :type relative_to_world: bool
            """

            def __init__(self):
                self.min_x = -2.0
                self.min_y = -2.0
                self.min_z = -9.0
                self.max_x = 2.0
                self.max_y = 2.0
                self.max_z = 3.0
                self.relative_to_world = False  # Decide if the Crop should be done in the sensor/camera coordinates
                # or if the PC should be transformed with CASViews.VIEWPOINT_CAM_TO_WORLD first

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="PointcloudCropAnnotator", descriptor=Descriptor()):
        """Initialize the point cloud cropper.

        :param name: Name of this annotator instance, defaults to "PointcloudCropAnnotator"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: PointcloudCropAnnotator.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.color = None

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        """Process and crop point cloud data.

        The method:

        * Loads point cloud and camera data from CAS
        * Optionally transforms to world coordinates
        * Crops using axis-aligned bounding box
        * Generates visualization mask
        * Updates CAS with cropped cloud
        * Creates combined visualization

        :return: SUCCESS after processing
        :rtype: py_trees.Status
        :raises Exception: If world transform not found when needed
        """
        start_timer = default_timer()
        self.rk_logger.warning(f"{self.__class__.__name__} called for update()")
        # q = self.get_cas().get(CASViews.QUERY)
        # if q.type == 'failc':
        #     raise Exception("Some Foobar happened")

        cloud = self.get_cas().get(CASViews.CLOUD)
        # cloud_organized = self.get_cas().get(CASViews.CLOUD_ORGANIZED)
        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        pc_cam_intrinsics = self.get_cas().get(CASViews.PC_CAM_INTRINSIC)
        color2depth_ratio = self.get_cas().get(CASViews.COLOR2DEPTH_RATIO)

        #
        # Lookup camera information if the crop should be done relative to the world frame
        #

        if self.descriptor.parameters.relative_to_world:
            try:
                cloud = robokudo.utils.annotator_helper.transform_cloud_from_cam_to_world(self.get_cas(), cloud)
            except Exception as e:
                self.rk_logger.warning(f"Couldn't find camera viewpoint in the CAS and relative_to_world is true. "
                                       f"Fail. Error: {e}")
            return py_trees.common.Status.FAILURE

        #
        # Crop the point cloud
        #
        assert (isinstance(cloud, o3d.geometry.PointCloud))

        abb = o3d.geometry.AxisAlignedBoundingBox(
            [self.descriptor.parameters.min_x,
             self.descriptor.parameters.min_y,
             self.descriptor.parameters.min_z, ],
            [self.descriptor.parameters.max_x,
             self.descriptor.parameters.max_y,
             self.descriptor.parameters.max_z, ]
        )

        cropped_cloud = cloud.crop(abb)

        # Transform cloud back to camera coordinates if it has been transformed to world before
        if self.descriptor.parameters.relative_to_world:
            cropped_cloud_transformed = robokudo.utils.annotator_helper.transform_cloud_from_world_to_cam(
                self.get_cas(), cropped_cloud)
            cropped_cloud = cropped_cloud_transformed

        assert (isinstance(cropped_cloud, o3d.geometry.PointCloud))

        self.get_cas().set_ref(CASViews.CLOUD, cropped_cloud)
        self.get_annotator_output_struct().set_geometries(cropped_cloud)

        mask_scale = 1.0 / color2depth_ratio[0]
        mask = get_mask_from_pointcloud(cropped_cloud, self.color, pc_cam_intrinsics, mask_scale_factor=mask_scale,
                                        crop_to_ref=True)
        combined_mask_color = cv2.addWeighted(self.color, 0.5, mask, 0.5, 0)
        self.get_annotator_output_struct().set_image(combined_mask_color)

        # plt.cla()
        # plt.imshow(mask)
        # plt.pause(0.1)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS
