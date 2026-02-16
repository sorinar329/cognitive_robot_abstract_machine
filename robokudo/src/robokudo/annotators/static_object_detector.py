"""
Robokudo Static Object Detector Module

This module provides functionality for detecting objects at predefined locations
using either manually configured bounding boxes or object knowledge from a database.
The detector can create object hypotheses with poses, masks, and class labels.

.. note::
   All poses are defined relative to the camera frame by default unless pose_in_world_coordinates is True.
"""
import copy
from enum import Enum
from timeit import default_timer

import cv2
import numpy as np
import py_trees

import robokudo
import robokudo.annotators.core
import robokudo.types.annotation
import robokudo.types.cv
import robokudo.types.scene
import robokudo.types.tf
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
import robokudo.utils.knowledge
import robokudo.utils.o3d_helper
import robokudo.utils.type_conversion
from robokudo.cas import CASViews
from robokudo.object_knowledge_base import ObjectKnowledge
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.transform import get_rotation_matrix_from_euler_angles


class StaticObjectMode(Enum):
    BOUNDING_BOX = "bounding_box"
    OBJECT_KNOWLEDGE_INSTANCE = "object_knowledge_instance"
    OBJECT_KNOWLEDGE_BASE = "object_knowledge_base"


class StaticObjectDetectorAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Find a cluster based on a preconfigured Bounding Box, Pose and Class name.

    This annotator can:

    * Create object hypotheses at fixed locations using manual bounding box coordinates
    * Load object knowledge from a database to automatically infer bounding boxes
    * Generate pose annotations in either camera or world coordinates
    * Create masks for the detected regions

    .. note::
       The detector supports both Euler angles and quaternions for rotation specification.
       Parameters can be dynamically reconfigured through ROS dynamic reconfigure.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for the StaticObjectDetectorAnnotator.

        Defines all configurable parameters including:

        * Bounding box dimensions and position
        * Object knowledge database settings
        * Pose generation options
        * Coordinate frame settings
        """

        class Parameters:
            """Parameters controlling the static object detection behavior."""

            def __init__(self):
                self.bounding_box_x = 1
                self.bounding_box_y = 1
                self.bounding_box_width = 10
                self.bounding_box_height = 10

                # Defines the mode which mainly decide which sources of information are used to generate the Object
                # Hypothesis
                self.mode = StaticObjectMode.BOUNDING_BOX

                # # If setting this to True, we detect a certain object stored in the ObjectKnowledgeBase
                # # This allows us to automatically infer the BoundingBox coordinates
                # self.detect_object_from_object_knowledge = False

                # Define the class_name which is used for the object of interest
                # Only used for StaticObjectMode.BOUNDING_BOX and StaticObjectMode.OBJECT_KNOWLEDGE_INSTANCE
                self.class_name = "unknown"
                # Used for StaticObjectMode.OBJECT_KNOWLEDGE_BASE
                self.class_names = []

                # Shall we create a Pose and BoundingBox Annotation for the object?
                # Please note that poses are defined relative to the camera frame by default
                # Only effective in Mode=StaticObjectDetectorAnnotator.Mode.OBJECT_KNOWLEDGE_*
                self.create_pose_annotation = False
                self.create_bounding_box_annotation = False

                # If this is a true, a mask based on the ROI will be generated that marks every pixel as ON
                self.create_mask = True

                # If you use ObjectKnowledge to generate the detection, provide the knowledge base here
                self.object_knowledge_base_ros_package = "robokudo"
                self.object_knowledge_base_name = "object_knowledge_iai_kitchen"

                # If StaticObjectDetectorAnnotator.Mode.OBJECT_KNOWLEDGE_INSTANCE is used, set the desired instance here
                self.object_knowledge_instance: ObjectKnowledge | None = None

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="StaticObjectDetector", descriptor=Descriptor()):
        """Default construction. Minimal one-time init!

        :param name: Name of the annotator instance, defaults to "StaticObjectDetector"
        :type name: str
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: StaticObjectDetectorAnnotator.Descriptor
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.color = None
        self.depth = None
        self.cloud = None
        self.cam_intrinsics = None
        self.object_kb = None
        self.object_knowledge = None

    def detect_from_bb_descriptor(self, color_rgb: np.ndarray) -> ObjectHypothesis:
        """
        Detect only based on the BB.


        :param color_rgb: Image in RGB order
        :type color_rgb: np.ndarray
        :return: robokudo.types.scene.ObjectHypothesis
        """
        object_hypothesis = robokudo.types.scene.ObjectHypothesis()
        object_hypothesis.id = str(0)
        object_hypothesis.source = self.name

        object_hypothesis.points = self.get_cloud_from_2d_bb_roi(color_rgb)
        # Use the non-scaled BB coordinates relative to the color image
        object_hypothesis.roi.roi.pos.x = self.descriptor.parameters.bounding_box_x
        object_hypothesis.roi.roi.pos.y = self.descriptor.parameters.bounding_box_y
        object_hypothesis.roi.roi.width = self.descriptor.parameters.bounding_box_width
        object_hypothesis.roi.roi.height = self.descriptor.parameters.bounding_box_height
        if self.descriptor.parameters.create_mask:
            object_hypothesis.roi.mask = np.ones((object_hypothesis.roi.roi.height,
                                                  object_hypothesis.roi.roi.width), dtype=np.uint8) * 255
        StaticObjectDetectorAnnotator.add_classification_annotation(object_hypothesis=object_hypothesis,
                                                                    class_name=self.descriptor.parameters.class_name)
        return object_hypothesis

    def detect_from_object_knowledge(self, object_knowledge: ObjectKnowledge,
                                     object_id: int = 0) -> ObjectHypothesis | None:
        """
        Detect from singular, passed ObjectKnowledge instance

        :return: robokudo.types.scene.ObjectHypothesis
        """
        object_hypothesis = robokudo.types.scene.ObjectHypothesis()
        object_hypothesis.id = str(object_id)
        object_hypothesis.source = self.name
        object_hypothesis.object_knowledge = object_knowledge

        # Check pose 2: Defined in cam or other frame?
        if object_knowledge.is_frame_in_camera_coordinates():
            object_knowledge_transform_in_cam = robokudo.utils.knowledge.get_transform_matrix_from_object_knowledge(
                object_knowledge)
        else:
            # Currently, we only support cam or world frames. No tf lookup for other frames! Therefore, consider
            # pose in object_knowledge as world frame
            world_to_cam_transform_matrix = robokudo.utils.annotator_helper.get_world_to_cam_transform_matrix(
                self.get_cas())
            object_knowledge_transform_in_world = robokudo.utils.knowledge.get_transform_matrix_from_object_knowledge(
                object_knowledge)

            object_knowledge_transform_in_cam = world_to_cam_transform_matrix @ object_knowledge_transform_in_world

        # Calculate Bounding Box and resulting 2D Image Corner points based on pose in cam coordinates

        obb = robokudo.utils.knowledge.get_obb_for_object_and_transform(object_knowledge,
                                                                        object_knowledge_transform_in_cam)
        corner_points = robokudo.utils.o3d_helper.get_2d_bounding_rect_from_3d_bb(self.get_cas(), obb)

        image_height = self.get_cas().get(CASViews.COLOR_IMAGE).shape[0]
        image_width = self.get_cas().get(CASViews.COLOR_IMAGE).shape[1]

        if robokudo.utils.cv_helper.rect_outside_image(corner_points, image_width, image_height):
            self.rk_logger.info("ROI of object would be completely out of camera frame. Skipping ...")
            return None

        corner_points = robokudo.utils.cv_helper.clamp_bounding_rect(corner_points, image_width=image_width,
                                                                     image_height=image_height)

        roi = robokudo.types.cv.ImageROI()
        roi.roi.pos.x = corner_points[0]
        roi.roi.pos.y = corner_points[1]
        roi.roi.width = corner_points[2]
        roi.roi.height = corner_points[3]
        object_hypothesis.roi = roi
        object_hypothesis.points = self.cloud.crop(obb)

        object_knowledge_translation_in_cam = list(
            robokudo.utils.transform.get_translation_from_transform_matrix(object_knowledge_transform_in_cam))
        object_knowledge_rotation_in_cam = list(
            robokudo.utils.transform.get_quaternion_from_transform_matrix(object_knowledge_transform_in_cam))

        if self.descriptor.parameters.create_pose_annotation:
            pose_annotation = robokudo.types.annotation.PoseAnnotation()
            pose_annotation.source = 'StaticObjectDetectorAnnotator'
            pose_annotation.translation = object_knowledge_translation_in_cam
            pose_annotation.rotation = object_knowledge_rotation_in_cam

            object_hypothesis.annotations.append(pose_annotation)

        if self.descriptor.parameters.create_bounding_box_annotation:
            bb_annotation = robokudo.types.annotation.BoundingBox3DAnnotation()
            bb_annotation.source = 'StaticObjectDetectorAnnotator'
            bb_annotation.pose = robokudo.types.tf.Pose()
            bb_annotation.pose.translation = object_knowledge_translation_in_cam
            bb_annotation.pose.rotation = object_knowledge_rotation_in_cam

            bb_annotation.x_length = object_knowledge.x_size
            bb_annotation.y_length = object_knowledge.y_size
            bb_annotation.z_length = object_knowledge.z_size

            object_hypothesis.annotations.append(bb_annotation)

        StaticObjectDetectorAnnotator.add_classification_annotation(object_hypothesis=object_hypothesis,
                                                                    class_name=object_knowledge.name)

        return object_hypothesis

    def detect_from_object_knowledge_base(self) -> list[ObjectHypothesis]:
        """
        Detect from a completed object knowledge base

        :return: robokudo.types.scene.ObjectHypothesis
        """
        object_hypotheses = []
        for class_name in self.descriptor.parameters.class_names:
            object_knowledge = self.object_kb.entries[class_name]
            object_hypothesis = self.detect_from_object_knowledge(object_knowledge)
            if object_hypothesis is not None:
                object_hypotheses.append(object_hypothesis)

        return object_hypotheses

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        """Process current scene to detect configured static objects.

        Steps:

        * Gets current color image, depth image and point cloud
        * If using object knowledge, validates object class exists
        * Scales color image to match depth image
        * Creates object hypothesis with:

          * Bounding box from configuration or object knowledge
          * Pose annotation if enabled
          * Point cloud from ROI
          * Mask if enabled

        :return: SUCCESS if detection completed, FAILURE if required transforms not found
        :rtype: py_trees.common.Status
        :raises Exception: If camera parameters are invalid or missing
        """
        start_timer = default_timer()

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        self.cloud = self.get_cas().get(CASViews.CLOUD)
        self.cam_intrinsics = copy.deepcopy(self.get_cas().get(CASViews.CAM_INTRINSIC))

        world_frame_required = False
        if self.descriptor.parameters.mode == StaticObjectMode.OBJECT_KNOWLEDGE_BASE:
            self.object_kb = robokudo.utils.knowledge.load_object_knowledge_base(self)
            self.rk_logger.info(f"Loaded KB: {self.object_kb.entries}")

            # Do some sanity checks and quit early if necessary
            for class_name in self.descriptor.parameters.class_names:
                if class_name not in self.object_kb.entries:
                    self.feedback_message = f"Couldn't find {class_name} in Object Knowledgebase"
                    self.rk_logger.warning(self.feedback_message)
                    return py_trees.common.Status.SUCCESS

                if self.object_kb.entries[class_name].is_frame_in_camera_coordinates():
                    world_frame_required = True

        if self.descriptor.parameters.mode == StaticObjectMode.OBJECT_KNOWLEDGE_INSTANCE:
            self.object_knowledge = self.descriptor.parameters.object_knowledge_instance

            # Do some sanity checks and quit early if necessary
            if self.descriptor.parameters.class_name is not self.object_knowledge.name:
                self.feedback_message = f"Couldn't find {self.descriptor.parameters.class_name} in Object Knowledge Instance"
                self.rk_logger.warning(self.feedback_message)
                return py_trees.common.Status.SUCCESS

            world_frame_required = self.object_knowledge.is_frame_in_camera_coordinates()

        if world_frame_required:
            try:
                cam_to_world_transform = self.get_cas().get(CASViews.VIEWPOINT_CAM_TO_WORLD)
            except:
                self.rk_logger.warning("Couldn't find viewpoint in the CAS")
                return py_trees.common.Status.FAILURE

            assert (isinstance(cam_to_world_transform, robokudo.types.tf.StampedTransform))

        # Scale the image down so that it matches the depth image size
        resized_color = None
        try:
            resized_color = robokudo.utils.cv_helper.get_scaled_color_image_for_depth_image(self.get_cas(), self.color)
            robokudo.utils.annotator_helper.scale_cam_intrinsics(self)
        except RuntimeError as e:
            self.rk_logger.error(
                "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation.")
            raise Exception(
                "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation.")

        color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)

        object_hypotheses = []

        if self.descriptor.parameters.mode == StaticObjectMode.BOUNDING_BOX:
            object_hypothesis = self.detect_from_bb_descriptor(color_rgb)
            if object_hypothesis is None:
                return py_trees.common.Status.SUCCESS
            object_hypotheses.append(object_hypothesis)
        elif self.descriptor.parameters.mode == StaticObjectMode.OBJECT_KNOWLEDGE_BASE:
            object_hypotheses = self.detect_from_object_knowledge_base()
            if len(object_hypotheses) == 0:
                # Simply return early but don't die
                return py_trees.common.Status.SUCCESS
        elif self.descriptor.parameters.mode == StaticObjectMode.OBJECT_KNOWLEDGE_INSTANCE:
            object_hypothesis = self.detect_from_object_knowledge(self.descriptor.parameters.object_knowledge_instance)
            if object_hypothesis is None:
                return py_trees.common.Status.SUCCESS
            object_hypotheses.append(object_hypothesis)
        else:
            raise Exception("Unknown static object mode")

        self.get_cas().annotations.extend(object_hypotheses)

        #
        # Create visualization Output
        #
        visualization_img = copy.deepcopy(self.color)

        for oh in object_hypotheses:
            assert (isinstance(oh, robokudo.types.scene.ObjectHypothesis))
            oh_roi = oh.roi.roi
            upper_left = (oh_roi.pos.x, oh_roi.pos.y)
            upper_left_text = (oh_roi.pos.x, oh_roi.pos.y - 5)

            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_img = cv2.putText(visualization_img, f"ROI-{oh.id}({len(oh.points.points)})",
                                            upper_left_text, font, 0.5, (0, 0, 255), 1, 2)
            visualization_img = cv2.rectangle(visualization_img,
                                              upper_left,
                                              (oh_roi.pos.x + oh_roi.width, oh_roi.pos.y + oh_roi.height),
                                              (0, 0, 255), 2)

            self.get_annotator_output_struct().set_geometries(oh.points)

        self.get_annotator_output_struct().set_image(visualization_img)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS

    @staticmethod
    def add_classification_annotation(object_hypothesis: ObjectHypothesis, class_name: str):
        classification_annotation = robokudo.types.annotation.Classification()
        classification_annotation.classname = class_name
        classification_annotation.source = 'StaticObjectDetectorAnnotator'
        classification_annotation.confidence = 1.0
        object_hypothesis.annotations.append(classification_annotation)

    def get_rotation_list_based_on_parameters(self, rotation_x: float, rotation_y: float, rotation_z: float,
                                              rotation_w: float) -> list:
        """
        Return a quaternion based on the parametrization of the Annotator.

        :param rotation_x:
        :param rotation_y:
        :param rotation_z:
        :param rotation_w:
        :return: 4-dim list with Quaternion
        """
        rotation_list = None
        if self.descriptor.parameters.pose_use_euler_angles:
            rot_matrix = get_rotation_matrix_from_euler_angles(rotation_x,
                                                               rotation_y,
                                                               rotation_z)
            rotation_list = list(
                robokudo.utils.transform.get_quaternion_from_rotation_matrix(rot_matrix))
        else:
            # Interpret values directly as a quaternion
            rotation_list = [rotation_x,
                             rotation_y,
                             rotation_z,
                             rotation_w]
        return rotation_list

    def get_cloud_from_2d_bb_roi(self, color_rgb):
        mask = np.zeros_like(self.depth, dtype=np.uint8)
        x1 = self.descriptor.parameters.bounding_box_x
        x2 = self.descriptor.parameters.bounding_box_x + self.descriptor.parameters.bounding_box_width
        y1 = self.descriptor.parameters.bounding_box_y
        y2 = self.descriptor.parameters.bounding_box_y + self.descriptor.parameters.bounding_box_height
        # Respect possible color2depth scaling also for BoundingBox coordinates
        color2depth_ratio = self.get_cas().get(robokudo.cas.CASViews.COLOR2DEPTH_RATIO)
        (sx1, sy1) = robokudo.utils.cv_helper.get_scale_coordinates(color2depth_ratio, (x1, y1))
        (sx2, sy2) = robokudo.utils.cv_helper.get_scale_coordinates(color2depth_ratio, (x2, y2))
        mask[int(sy1): int(sy2), int(sx1): int(sx2)] = 255
        cloud = robokudo.utils.o3d_helper.get_cloud_from_rgb_depth_and_mask(color_rgb, self.depth,
                                                                            mask, self.cam_intrinsics,
                                                                            mask_true_val=255)
        return cloud
