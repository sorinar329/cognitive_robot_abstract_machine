"""ROI adjustment for RoboKudo.

This module provides an annotator for adjusting the size of Region of Interest (ROI)
boundaries for object hypotheses. It can grow or shrink ROIs and their associated
masks by a specified pixel offset.
"""
import copy
from timeit import default_timer

import py_trees

import robokudo
import robokudo.annotators.core
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
from robokudo.cas import CASViews


class ROIAdjusterAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Annotator for adjusting ROI sizes of object hypotheses.

    This annotator can grow or shrink ROIs (Regions of Interest) on object
    hypotheses by a specified pixel offset. It handles both the ROI boundaries
    and their associated masks, ensuring proper adjustment of both.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for ROI adjustment."""

        class Parameters:
            """Parameters for configuring ROI adjustment behavior.

            :ivar offset_pixel: Pixels to add/subtract from ROI sides (positive grows, negative shrinks)
            :type offset_pixel: int
            :ivar analysis_scope: Type of annotations to process, defaults to ObjectHypothesis
            :type analysis_scope: type
            :ivar fill_value_mask: Value to fill new mask areas when growing ROIs, defaults to 0
            :type fill_value_mask: int
            """

            def __init__(self):
                self.offset_pixel = 20  # in pixels to be added to either side of the ROI. Can be negative or positive.
                self.analysis_scope = robokudo.types.scene.ObjectHypothesis
                # This is the value that will be added to an associated mask if the ROI is grown
                self.fill_value_mask = 0

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="ROIAdjusterAnnotator", descriptor=Descriptor()):
        """Initialize the ROI adjuster.

        :param name: Name of the annotator instance, defaults to "ROIAdjusterAnnotator"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: ROIAdjusterAnnotator.Descriptor, optional
        """
        super().__init__(name, descriptor)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        """Update ROIs by applying the configured pixel offset.

        For each object hypothesis in the analysis scope:
        - Adjusts the ROI boundaries by the specified pixel offset
        - If a mask exists, adjusts it accordingly with the specified fill value

        :return: SUCCESS after adjusting all ROIs
        :rtype: py_trees.common.Status
        """
        start_timer = default_timer()
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        object_hypotheses = self.get_cas().filter_annotations_by_type(self.descriptor.parameters.analysis_scope)
        for object_hypothesis in object_hypotheses:
            robokudo.utils.cv_helper.adjust_image_roi(color, object_hypothesis.roi,
                                                      self.descriptor.parameters.offset_pixel)

            if object_hypothesis.roi.mask is not None:
                object_hypothesis.roi.mask = \
                    robokudo.utils.cv_helper.adjust_mask(object_hypothesis.roi.mask,
                                                         self.descriptor.parameters.offset_pixel,
                                                         fill_value=self.descriptor.parameters.fill_value_mask)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
