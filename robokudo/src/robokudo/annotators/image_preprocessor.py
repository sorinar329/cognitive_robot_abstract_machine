"""Image preprocessing and point cloud generation.

This module provides an annotator for:

* Converting RGB-D camera data to point clouds
* Handling color and depth image synchronization
* Managing camera intrinsic parameters
* Providing visualization modes for debugging

.. note::
   Depth values are expected to be in millimeters.
"""
import copy
from timeit import default_timer

import cv2
import numpy
import open3d as o3d
import py_trees

import robokudo.annotators
import robokudo.annotators.core
import robokudo.annotators.outputs
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
import robokudo.utils.o3d_helper
from robokudo.cas import CASViews


class ImagePreprocessorAnnotator(robokudo.annotators.core.BaseAnnotator):
    """RGB-D image preprocessor and point cloud generator.
    
    This annotator:
    
    * Converts RGB-D camera data to point clouds
    * Handles color/depth image synchronization
    * Manages camera intrinsic parameters
    * Provides visualization modes for debugging
    
    .. warning::
       Requires properly configured camera intrinsics and color-to-depth ratio.
    """

    class ViewMode:
        """Visualization mode enumeration.
        
        :cvar color: Display color image (1)
        :type color: int
        :cvar depth: Display depth image (2)
        :type depth: int
        """
        color = 1
        depth = 2

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for image preprocessing."""

        class Parameters:
            """Parameters for configuring preprocessing.

            :ivar depth_trunc: Maximum depth distance in meters, points beyond are discarded
            :type depth_trunc: float
            """
            def __init__(self):
                self.depth_trunc = 9.0  # distance parameter used in the cloud creation. Larger dists will be discarded.

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="ImagePreprocessor", descriptor=Descriptor()):
        """Initialize the image preprocessor.

        :param name: Name of this annotator instance, defaults to "ImagePreprocessor"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: ImagePreprocessorAnnotator.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.color = None
        self.depth = None

        self.display_mode = self.ViewMode.color

    def update(self):
        """Process RGB-D images and generate point cloud.

        The method:
        
        * Loads color and depth images from CAS
        * Synchronizes and scales images
        * Converts to Open3D format
        * Generates point cloud
        * Updates visualization based on view mode

        :return: SUCCESS after processing
        :rtype: py_trees.Status
        :raises RuntimeError: If color-to-depth ratio is not set
        """
        start_timer = default_timer()

        self.depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.cam_intrinsics = copy.deepcopy(self.get_cas().get(CASViews.CAM_INTRINSIC))

        if self.display_mode == self.ViewMode.depth:
            self.get_annotator_output_struct().set_image(self.depth)
        else:
            self.get_annotator_output_struct().set_image(self.color)

        robokudo.utils.annotator_helper.scale_cam_intrinsics(self)

        resized_color = None
        try:
            resized_color = robokudo.utils.cv_helper.get_scaled_color_image_for_depth_image(self.get_cas(), self.color)
        except RuntimeError as e:
            self.rk_logger.error("No color to depth ratio set by your camera driver! Can't preprocess.")

        # o3d expects color images in RGB order
        color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = None
        try:
            o3d_depth = o3d.geometry.Image(self.depth)  # Please note that depth values should be in mm
        except RuntimeError:
            # Even though you might have a uint16 already, it might be required to explicitly cast to uint16
            # (Note: the byteorder field might be different after the cast to make o3d happy)
            o3d_depth = o3d.geometry.Image(self.depth.astype(numpy.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth,
                                                                        convert_rgb_to_intensity=False,
                                                                        depth_trunc=self.descriptor.parameters.depth_trunc)

        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.cam_intrinsics)

        self.get_cas().set(CASViews.PC_CAM_INTRINSIC, self.cam_intrinsics)

        self.get_cas().set_ref(CASViews.CLOUD, cloud)
        self.get_annotator_output_struct().set_geometries(cloud)  # the visualizer can only show non-NaN pointclouds

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS

    def key_callback(self, key):
        """Handle keyboard input for view mode switching.

        :param key: ASCII value of pressed key
        :type key: int
        
        Available keys:
        
        * '1': Switch to color view mode
        * '2': Switch to depth view mode
        """
        if key == ord('1'):
            self.display_mode = self.ViewMode.color
        if key == ord('2'):
            self.display_mode = self.ViewMode.depth
