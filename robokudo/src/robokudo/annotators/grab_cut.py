"""Image segmentation using GrabCut algorithm.

This module provides an annotator for:

* Refining object ROIs using GrabCut
* Processing color image regions
* Creating segmentation masks
* Visualizing segmentation results

The module uses:

* OpenCV GrabCut implementation
* ROI-based initialization
* Iterative segmentation
* Mask-based visualization

.. warning::
   Current implementation is incomplete and for reference only.
"""
from timeit import default_timer

import cv2
import numpy as np
import py_trees

import robokudo.annotators
import robokudo.annotators.core
import robokudo.annotators.outputs
import robokudo.types.scene
import robokudo.utils.error_handling
from robokudo.cas import CASViews


class GrabCutAnnotator(robokudo.annotators.core.ThreadedAnnotator):
    """Refine the ROI of an Object Hypothesis with the GrabCut Algorithm

    This annotator:

    * Refines object ROIs using GrabCut
    * Processes color image regions
    * Creates foreground/background models
    * Generates segmentation masks
    * Visualizes segmented regions

    .. warning::
       Current implementation only processes one object and lacks annotation creation.
    """

    def __init__(self, name="GrabCutAnnotator"):
        """Initialize the GrabCut segmenter.

        :param name: Name of this annotator instance, defaults to "GrabCutAnnotator"
        :type name: str, optional
        """
        super().__init__(name)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self):
        """Process object hypotheses using GrabCut.

        The method:

        * Loads color image and object hypotheses
        * For each hypothesis:
          * Extracts ROI rectangle
          * Creates initial mask
          * Initializes foreground/background models
          * Runs GrabCut segmentation
          * Creates visualization

        .. note::
           Currently only processes one object and lacks annotation creation.

        :return: SUCCESS after processing
        :rtype: py_trees.Status
        """
        start_timer = default_timer()

        # TODO This implementation is incomplete and just serves as a quick reference what we can expect
        # from GrabCut. As of today, it will only show one of the segmented objects
        # And will not create an Annotation

        cloud = self.get_cas().get(CASViews.CLOUD)
        color_image = self.get_cas().get(CASViews.COLOR_IMAGE)
        visualization_image = self.get_cas().get(CASViews.COLOR_IMAGE)

        object_hypotheses = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)

        # Iterate over everything that is a Object hypothesis and calculate the centroid
        for object_hypothesis in object_hypotheses:
            assert (isinstance(object_hypothesis, robokudo.types.scene.ObjectHypothesis))
            roi = object_hypothesis.roi.roi
            mask = np.zeros(color_image.shape[:2], dtype="uint8")
            rect = (roi.pos.x, roi.pos.y, roi.width, roi.height)

            fg_model = np.zeros((1, 65), dtype="float")
            bg_model = np.zeros((1, 65), dtype="float")
            cv2.grabCut(color_image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            visualization_image = color_image * mask2[:, :, np.newaxis]

            # cluster_cloud = object_hypothesis.points
            # centroid, covariance = cluster_cloud.compute_mean_and_covariance()

            # position = robokudo.types.annotation.PositionAnnotation()
            # position.source = type(self).__name__
            # position.translation = centroid
            # object_hypothesis.annotations.append(position)

            # centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)  # in meters
            # centroid_sphere.paint_uniform_color([255, 0, 0])
            # centroid_sphere.translate(centroid)
            # centroids_to_visualize.append(centroid_sphere)

        # Visualization
        # vis_geometries = [
        #    {"name": "Cloud", "geometry": cloud},
        # ]
        # vis_geometries.extend(centroids_to_visualize)

        self.get_annotator_output_struct().set_image(visualization_image)
        # self.get_annotator_output_struct().set_geometries(vis_geometries)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS
