"""Annotation writer classes for RoboKudo.

This module provides annotators for writing and publishing annotations.
"""
import copy
import os
import shutil
from timeit import default_timer

import py_trees
import rospy
import std_msgs

import robokudo.io.storage
import robokudo.utils.serialization as serializer


class AnnotationStorageWriter(robokudo.annotators.core.BaseAnnotator):
    """Annotator for writing annotations to storage in JSON format.

    This annotator writes the current CAS annotations to files in a specified
    directory, using JSON serialization.

    :ivar counter: Counter for generating sequential filenames
    :type counter: int
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for annotation storage writer."""

        class Parameters:
            """Parameters for configuring annotation storage.

            :ivar basic_path: Base directory for storing annotations, defaults to "annotations"
            :type basic_path: str
            :ivar suffix: File extension for annotation files, defaults to "json"
            :type suffix: str
            """

            def __init__(self):
                self.basic_path = "annotations"
                self.suffix = "json"

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="AnnotationStorageWriter", descriptor=Descriptor()):
        """Initialize the annotation storage writer. Minimal one-time init!

        :param name: Name of the annotator instance, defaults to "AnnotationStorageWriter"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: AnnotationStorageWriter.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.counter = -1

    def update(self):
        """Write current CAS annotations to storage.

        Serializes the current annotations to JSON and writes them to a file
        in the configured directory. Files are numbered sequentially.

        :return: SUCCESS after writing annotations
        :rtype: py_trees.common.Status
        """
        start_timer = default_timer()

        # encode data as json string
        json_data_string = serializer.encode(self.get_cas().annotations)

        dir_path = os.path.join(self.descriptor.parameters.basic_path, self.name)

        # increase count
        self.counter += 1

        if self.counter == 0 and os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        os.makedirs(dir_path, exist_ok=True)

        json_path = os.path.join(dir_path, "{}.{}".format(self.counter, self.descriptor.parameters.suffix))
        with open(json_path, "w") as f:
            f.write(json_data_string)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS


class AnnotationPublisherWriter(robokudo.annotators.core.BaseAnnotator):
    """Annotator for publishing annotations via ROS topics.

    This annotator publishes the current CAS annotations as JSON-encoded
    strings over a ROS topic.

    :ivar pub: ROS publisher for annotations
    :type pub: rospy.Publisher
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Configuration descriptor for annotation publisher."""

        class Parameters:
            """Parameters for configuring annotation publishing.

            :ivar topic_name: Name of the ROS topic to publish on, defaults to "/annotations"
            :type topic_name: str
            """

            def __init__(self):
                self.topic_name = "/annotations"

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="AnnotationPublisherWriter", descriptor=Descriptor()):
        """Initialize the annotation publisher. Minimal one-time init!

        :param name: Name of the annotator instance, defaults to "AnnotationPublisherWriter"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: AnnotationPublisherWriter.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.pub = None

    def setup(self, timeout):
        """Set up the ROS publisher. Useful for delayed initialization. For example ROS pub/sub, drivers.

        :param timeout: Maximum time to wait for setup completion
        :type timeout: float
        :return: True if setup was successful
        :rtype: bool
        """
        self.rk_logger.debug("%s.setup()" % self.__class__.__name__)

        self.pub = rospy.Publisher(self.descriptor.parameters.topic_name,
                                   std_msgs.msg.String, queue_size=10)

    def update(self):
        """Publish current CAS annotations.

        Serializes the current annotations to JSON and publishes them
        on the configured ROS topic.

        :return: SUCCESS after publishing annotations
        :rtype: py_trees.common.Status
        """
        start_timer = default_timer()

        # encode data as json string
        json_data_string = serializer.encode(self.get_cas().annotations)

        # publish encoded data
        self.pub.publish(json_data_string)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
