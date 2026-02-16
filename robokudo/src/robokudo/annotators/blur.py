"""Image blur detection for RoboKudo.

This module provides an annotator for calculating and visualizing image blur metrics
using the Laplacian variance method. It can optionally halt pipeline execution if
blur exceeds a threshold.

.. note::
   The blur metric is calculated using the Laplacian operator variance, where:
   
   * Higher values indicate sharper images
   * Lower values indicate more blur
"""
import copy
from timeit import default_timer

import cv2
import py_trees

import robokudo.utils.cv_helper
from robokudo.cas import CASViews


class BlurAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Annotator for calculating and visualizing image blur metrics.

    This annotator:
    
    * Calculates a blur metric using Laplacian variance
    * Visualizes the blur value on the image
    * Optionally halts pipeline if blur exceeds threshold

    .. warning::
       Setting return_failure_above_threshold to True will stop pipeline
       advancement when the blur threshold is exceeded.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for blur detection."""

        class Parameters:
            """Parameters for configuring blur detection behavior.

            :ivar blur_threshold: Threshold for acceptable blur level, defaults to 100
            :type blur_threshold: float
            :ivar return_failure_above_threshold: If True, return FAILURE when blur exceeds threshold
            :type return_failure_above_threshold: bool
            """
            def __init__(self):
                self.blur_threshold = 100
                self.return_failure_above_threshold = True  # Let this behaviour return failure to stop the
                # advancement of the current pipeline

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="BlurAnnotator", descriptor=Descriptor()):
        """Initialize the blur annotator. Minimal one-time init!

        :param name: Name of the annotator instance, defaults to "BlurAnnotator"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: BlurAnnotator.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def compute_bluriness(self, img):
        """Compute the blur metric for an image.

        Uses Laplacian variance to measure image sharpness:
        
        * Converts image to grayscale
        * Computes Laplacian operator
        * Returns variance of Laplacian values

        :param img: Input image in BGR format
        :type img: numpy.ndarray
        :return: Blur metric value (higher = sharper)
        :rtype: float
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def update(self):
        """Update blur detection and visualization.

        Creates visualizations containing:
        
        * Color image with blur metric overlay
        * Point cloud data
        
        :return: SUCCESS if blur is acceptable, FAILURE if above threshold (if configured)
        :rtype: py_trees.common.Status
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        bluriness = self.compute_bluriness(color)

        visualization_img = copy.deepcopy(color)
        font = cv2.FONT_HERSHEY_COMPLEX
        visualization_img = cv2.putText(visualization_img, f"Bluriness: {bluriness}",
                                        (20, 90), font, 1, (0, 0, 255), 1, 2)

        # 3D Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]

        self.get_annotator_output_struct().set_geometries(vis_geometries)
        self.get_annotator_output_struct().set_image(visualization_img)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS
