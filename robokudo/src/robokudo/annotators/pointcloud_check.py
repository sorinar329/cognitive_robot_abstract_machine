"""Point cloud validation and threshold checking.

This module provides an annotator for:

* Validating point cloud size
* Configurable threshold checking
* Customizable status responses
* Optional error raising

The annotator supports:

* Configurable point count threshold
* Different status returns for above/below threshold
* Optional exception raising on failure
* Detailed logging of check results

.. note::
   Point cloud must be available in CAS under CASViews.CLOUD.
"""
from timeit import default_timer

import open3d as o3d
import py_trees

import robokudo.utils.annotator_helper
import robokudo.utils.error_handling
from robokudo.cas import CASViews


class PointcloudCheckAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Check if the CASViews.Cloud contains more than X points.

    .. warning::
       Will raise exception if point cloud not found in CAS.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for point cloud checking."""

        class Parameters:
            """Parameters for configuring point cloud validation.

            Threshold parameters:

            :ivar point_threshold: Minimum required points, defaults to 100
            :type point_threshold: int

            Status parameters:

            :ivar status_below_threshold: Status when below threshold, defaults to FAILURE
            :type status_below_threshold: py_trees.Status
            :ivar status_above_threshold: Status when above threshold, defaults to SUCCESS
            :type status_above_threshold: py_trees.Status

            Error handling:

            :ivar raise_on_failure: Whether to raise exception on failure, defaults to True
            :type raise_on_failure: bool
            """

            def __init__(self):
                # Decision boundary for: if CASViews.CLOUD has less than this amount of points
                self.point_threshold = 100
                self.status_below_threshold = py_trees.Status.FAILURE
                self.status_above_threshold = py_trees.Status.SUCCESS
                self.raise_on_failure = True

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="PointcloudcloudCheckAnnotator", descriptor=Descriptor()):
        """Initialize the point cloud checker.

        :param name: Name of this annotator instance, defaults to "PointcloudcloudCheckAnnotator"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: PointcloudCheckAnnotator.Descriptor, optional
        """
        super().__init__(name, descriptor)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        """Check point cloud size against threshold.

        The method:

        * Loads point cloud from CAS
        * Counts number of points
        * Compares against threshold:
          * Below: Returns status_below_threshold
          * Above: Returns status_above_threshold
        * Optionally raises exceptions on failure
        * Logs detailed check results

        :return: Configured status based on point count
        :rtype: py_trees.Status
        :raises Exception: If point count fails threshold and raise_on_failure is True
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        assert (isinstance(cloud, o3d.geometry.PointCloud))
        point_count = len(cloud.points)
        if point_count < self.descriptor.parameters.point_threshold:
            if self.descriptor.parameters.status_below_threshold == py_trees.Status.FAILURE and \
                    self.descriptor.parameters.raise_on_failure:
                raise Exception(f"Scene Pointcloud size({point_count}) is below "
                                f"threshold of {self.descriptor.parameters.point_threshold}")

            self.rk_logger.info(f"Scene Pointcloud size({point_count}) is below "
                                f"threshold of {self.descriptor.parameters.point_threshold}")
            end_timer = default_timer()
            self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
            return self.descriptor.parameters.status_below_threshold
        else:
            if self.descriptor.parameters.status_above_threshold == py_trees.Status.FAILURE and \
                    self.descriptor.parameters.raise_on_failure:
                raise Exception(f"Scene Pointcloud size({point_count}) is above "
                                f"threshold of {self.descriptor.parameters.point_threshold}")

            self.rk_logger.info(f"Scene Pointcloud size({point_count}) is above "
                                f"threshold of {self.descriptor.parameters.point_threshold}")
            end_timer = default_timer()
            self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
            return self.descriptor.parameters.status_above_threshold

        #
        #
        # end_timer = default_timer()
        # self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        # return py_trees.common.Status.SUCCESS
