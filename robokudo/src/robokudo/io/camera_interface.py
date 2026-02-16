"""
Camera interface module for RoboKudo.

This module provides base classes and implementations for interfacing with
various camera types in RoboKudo. It supports:

* ROS camera interfaces (raw and compressed)
* Kinect-style RGB-D cameras
* Camera calibration handling
* Transform lookups
* Synchronized data acquisition
* Thread-safe operation

The module handles:

* RGB and depth image acquisition
* Camera calibration information
* Camera-to-world transforms
* Data synchronization
* Format conversions
"""
import logging
import struct
import threading

import cv2
import numpy as np
import open3d as o3d
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.impl import rcutils_logger
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, CameraInfo, Image

import robokudo.cas
import robokudo.io.tf_listener_proxy
import robokudo.types.tf
from robokudo.cas import CASViews


class CameraInterface(object):
    """
    Base class for all camera interfaces in RoboKudo.

    This class defines the basic interface that all camera implementations
    must provide. It handles configuration and data availability tracking.

    :ivar _has_new_data: Whether new data is available
    :type _has_new_data: bool
    :ivar camera_config: Camera configuration object
    :type camera_config: Any
    :ivar rk_logger: RoboKudo logger instance
    :type rk_logger: logging.Logger
    """

    def __init__(self, camera_config):
        """
        Initialize the camera interface.

        :param camera_config: Configuration for the camera
        :type camera_config: Any
        """
        self._has_new_data = False
        self.camera_config = camera_config
        self.rk_logger = rcutils_logger.RcutilsLogger(name=robokudo.defs.PACKAGE_NAME)

    def has_new_data(self):
        """
        Check if new data is available.

        :return: True if new data is available, False otherwise
        :rtype: bool
        """
        return self._has_new_data

    def set_data(self, cas: robokudo.cas.CAS):
        """
        This method is supposed to read in, convert (if needed) and put the data into the CAS.
        If you are running a CameraInterface which is getting data via callback methods, please make sure
        to keep callbacks light and do the main conversion work here!
        Callbacks should be short.

        :param cas: The CAS where the data should be placed in
        :type cas: robokudo.cas.CAS
        """
        raise NotImplementedError


class ROSCameraInterface(CameraInterface):
    """
    Base class for ROS-based camera interfaces.

    This class extends the base camera interface with ROS-specific
    functionality like transform lookups and camera intrinsics handling.

    :ivar lookup_viewpoint: Whether to look up camera transforms
    :type lookup_viewpoint: bool
    :ivar tf_from: Transform source frame
    :type tf_from: str
    :ivar tf_to: Transform target frame
    :type tf_to: str
    :ivar transform_listener: ROS transform listener
    :type transform_listener: tf.TransformListener
    """

    def __init__(self, camera_config, node=None):
        """
        Initialize the ROS camera interface.

        :param camera_config: Configuration for the ROS camera
        :type camera_config: Any
        :param node: A ROS node for transform lookups
        :type node: rclpy.node.Node
        """
        super().__init__(camera_config)
        self.node = node if node is not None else Node("ros_camera_node")
        if hasattr(self.camera_config, "lookup_viewpoint"):
            self.lookup_viewpoint = self.camera_config.lookup_viewpoint
            self.tf_from = camera_config.tf_from
            self.tf_to = camera_config.tf_to
        else:
            self.lookup_viewpoint = False

        if self.lookup_viewpoint:
            self.transform_listener = robokudo.io.tf_listener_proxy.instance(self.node)

    def lookup_transform(self):
        """
        Look up the camera transform from TF.

        :return: True if transform lookup succeeded, False otherwise
        :rtype: bool
        """
        if self.lookup_viewpoint:
            time = Time()
            try:
                tf = self.transform_listener.lookup_transform(self.tf_to,
                                                              self.tf_from,
                                                              time,
                                                              timeout=Duration(
                                                                  seconds=0.1))
                translation = tf.transform.translation
                rotation = tf.transform.rotation
                self.cam_translation = [translation.x, translation.y, translation.z]
                self.cam_quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
            except Exception as err:
                self.rk_logger.warn(
                    f"cannot transform from {self.tf_from} to {self.tf_to} at ts {time}: {err}")
                return False
        return True

    def set_o3d_cam_intrinsics_from_ros_cam_info(self):
        """
        Convert ROS camera info to Open3D camera intrinsics.

        Creates an Open3D camera intrinsics object from the ROS camera
        calibration parameters.
        """
        # Construct o3d camera intrinsics from cam info in CAS
        self.cam_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        width = self.cam_info.width
        height = self.cam_info.height
        fx = self.cam_info.K[0]
        cx = self.cam_info.K[2]
        fy = self.cam_info.K[4]
        cy = self.cam_info.K[5]
        self.cam_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)


def depth_convert_workaround(msg):
    """
    Convert compressed depth image to proper depth format.

    This is a workaround for handling compressed depth images in ROS.
    Source: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/

    :param msg: Compressed depth image message
    :type msg: sensor_msgs.msg.CompressedImage
    :return: Depth image as numpy array
    :rtype: numpy.ndarray
    :raises Exception: If compression type is wrong or decoding fails
    """
    # 'msg' as type CompressedImage
    depth_fmt, compr_type = msg.format.split(';')
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if "compressedDepth" not in compr_type:
        raise Exception("Compression type is not 'compressedDepth'."
                        "You probably subscribed to the wrong topic.")

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
    # replaced np.fromstring with np.frombuffer because np.fromstring is deprecated in newer versions of numpy
    if depth_img_raw is None:
        # probably wrong header size
        raise Exception("Could not decode compressed depth image."
                        "You may need to change 'depth_header_size'!")

    if depth_fmt == "16UC1":
        # write raw image data
        return depth_img_raw
    elif depth_fmt == "32FC1":
        raw_header = msg.data[:depth_header_size]
        # header: int, float, float
        [compfmt, depthQuantA, depthQuantB] = struct.unpack('iff', raw_header)
        depth_img_scaled = depthQuantA / (depth_img_raw.astype(np.float32) - depthQuantB)
        # filter max values
        depth_img_scaled[depth_img_raw == 0] = 0

        # depth_img_scaled provides distance in meters as f32
        # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
        depth_img_mm = (depth_img_scaled * 1000).astype(np.uint16)
        return depth_img_mm
    else:
        raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")


class KinectCameraInterface(ROSCameraInterface):
    """
    Interface for Kinect-style RGB-D cameras using ROS.

    This class implements a camera interface for RGB-D cameras that publish
    color and depth images through ROS topics. It supports both raw and
    compressed image formats.

    :ivar color_subscriber: Color image subscriber
    :type color_subscriber: message_filters.Subscriber
    :ivar depth_subscriber: Depth image subscriber
    :type depth_subscriber: message_filters.Subscriber
    :ivar cam_info_subscriber: Camera info subscriber
    :type cam_info_subscriber: message_filters.Subscriber
    :ivar color: Latest color image
    :type color: numpy.ndarray
    :ivar depth: Latest depth image
    :type depth: numpy.ndarray
    :ivar cam_info: Latest camera calibration
    :type cam_info: sensor_msgs.msg.CameraInfo
    :ivar cam_intrinsic: Open3D camera intrinsics
    :type cam_intrinsic: o3d.camera.PinholeCameraIntrinsic
    :ivar color2depth_ratio: Ratio between color and depth image sizes
    :type color2depth_ratio: tuple
    :ivar cam_translation: Camera translation from TF
    :type cam_translation: list
    :ivar cam_quaternion: Camera rotation from TF
    :type cam_quaternion: list
    :ivar timestamp: Latest message timestamp
    :type timestamp: rospy.Time
    :ivar lock: Thread synchronization lock
    :type lock: threading.Lock
    """

    def __init__(self, camera_config):
        """
        Initialize the Kinect camera interface.

        Sets up ROS subscribers and synchronization for color, depth,
        and camera info topics.

        :param camera_config: Configuration for the Kinect camera
        :type camera_config: Any
        """
        super().__init__(camera_config, node=Node("kinect_camera_node"))

        if self.compressed_color_configured():
            self.color_subscriber = Subscriber(self.node, CompressedImage, camera_config.topic_color)
            # self.color_topic_sub = self.node.create_subscription(CompressedImage, camera_config.topic_color,
            #                                                      self.blackhole_callback, 10)
        else:
            self.color_subscriber = Subscriber(self.node, Image, camera_config.topic_color)

        if self.compressed_depth_configured():
            self.depth_subscriber = Subscriber(self.node, CompressedImage, camera_config.topic_depth)
        else:
            self.depth_subscriber = Subscriber(self.node, Image, camera_config.topic_depth)
            # self.depth_topic_sub = self.node.create_subscription(Image, camera_config.topic_depth,
            #                                                      self.blackhole_callback, 10)

        self.cam_info_subscriber = Subscriber(self.node, CameraInfo, camera_config.topic_cam_info)
        # self.cam_info_sub = self.node.create_subscription(CameraInfo, camera_config.topic_cam_info,
        #                                                   self.blackhole_callback, 10)

        ts = ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber, self.cam_info_subscriber],
            queue_size=10, slop=0.4)
        ts.registerCallback(self.callback)

        self.rk_logger.info("Subscribed to: ")
        self.rk_logger.info(f"  {camera_config.topic_color}")
        self.rk_logger.info(f"  {camera_config.topic_depth}")
        self.rk_logger.info(f"  {camera_config.topic_cam_info}")

        self.color = None
        self.depth = None
        self.cam_info = None
        self.cam_intrinsic = None
        self.color2depth_ratio = None
        self.cam_translation = None
        self.cam_quaternion = None
        self.timestamp = None
        self.lock = threading.Lock()
        # rclpy.spin_once(self.node)

        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True, name="Cam Interface Thread").start()

        # threading.Thread(target=rclpy.spin_once(self.node), args=(self.node,), daemon=True).start()

    # doubt

    # self.kexecutor = MultiThreadedExecutor()
    # self.kexecutor.add_node(self.node)
    # threading.Thread(target=self.kexecutor.spin, daemon=True).start()
    # threading.Thread(target=rclpy.spin_once, args=(self.node,), daemon=True).start()

    def compressed_depth_configured(self):
        """
        Check if compressed depth images are configured.

        :return: True if compressed depth is configured, False otherwise
        :rtype: bool
        """
        return hasattr(self.camera_config, "depth_hints") and self.camera_config.depth_hints == "compressedDepth"

    def compressed_color_configured(self):
        """
        Check if compressed color images are configured.

        :return: True if compressed color is configured, False otherwise
        :rtype: bool
        """
        return hasattr(self.camera_config, "color_hints") and self.camera_config.color_hints == "compressed"

    def get_node(self):
        return self.node

    def blackhole_callback(self, data):
        """
        This callback is just a dummy to receive data coming from a workaround subscription to handle problems with
        the ApproximateTimeSynchronizer
        :param data:
        :return:
        """
        pass

    def debug_callback(self, data):
        pass
        # self.rk_logger.info(f"DEBUG color ping: {data.header.stamp}")

    def debug_depth_callback(self, data):
        pass
        # self.rk_logger.info(f"DEBUG depth ping: {data.header.stamp}")

    def debug_cam_info_callback(self, data):
        pass
        # self.rk_logger.info(f"DEBUG info ping: {data.header.stamp}")

    def callback(self, color_data, depth_data=None, cam_info=None):
        """
        Process synchronized camera data.

        This callback handles incoming color, depth, and camera info messages.
        It converts the data to OpenCV format and stores it for later use.

        TODO make this generic. handle the encoding and order properly.
        For standard and compressed images.
        this might also depend on the fix of image_transport_plugins being published as a package.
        Startpoint can be found at the bottom of this method.

        :param color_data: Color image message
        :type color_data: sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage
        :param depth_data: Depth image message
        :type depth_data: sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage
        :param cam_info: Camera calibration message
        :type cam_info: sensor_msgs.msg.CameraInfo
        """
        self.lock.acquire()
        if self.rk_logger.is_enabled_for(logging.DEBUG):
            self.rk_logger.debug("Received data:")

            color_time = Time(seconds=color_data.header.stamp.sec, nanoseconds=color_data.header.stamp.nanosec)

            if depth_data is not None:
                depth_time = Time(seconds=depth_data.header.stamp.sec, nanoseconds=depth_data.header.stamp.nanosec)
                self.rk_logger.debug(f"  Color time - Depth time: {(color_time - depth_time).nanoseconds / 1e9:.6f}")

            if cam_info is not None:
                cam_info_time = Time(seconds=cam_info.header.stamp.sec, nanoseconds=cam_info.header.stamp.nanosec)
                self.rk_logger.debug(
                    f"  Color time - Cam Info time: {(color_time - cam_info_time).nanoseconds / 1e9:.6f}")

        if self.compressed_color_configured():
            color_arr = np.frombuffer(color_data.data, np.uint8)
            self.color = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
        else:
            bridge = CvBridge()
            self.color = bridge.imgmsg_to_cv2(color_data, "bgr8")

        self.timestamp = color_data.header.stamp

        if self.compressed_depth_configured():
            self.depth = depth_convert_workaround(depth_data)
        else:
            bridge = CvBridge()
            self.depth = bridge.imgmsg_to_cv2(depth_data, "32FC1")

        self.cam_info = cam_info

        # self.rk_logger.info("Callback processing done - Final steps")
        if not self.lookup_transform():
            self._has_new_data = False
            self.lock.release()
            return

        self._has_new_data = True
        self.lock.release()

    def set_data(self, cas: robokudo.cas.CAS):
        if not self.has_new_data():
            return

        self.lock.acquire()
        if self.camera_config.hi_res_mode:
            self.color = self.color[0:960, 0:1280]

        self.cam_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        width = self.cam_info.width
        height = self.cam_info.height
        if self.camera_config.hi_res_mode:
            height = 960

        fx = self.cam_info.k[0]
        cx = self.cam_info.k[2]
        fy = self.cam_info.k[4]
        cy = self.cam_info.k[5]
        self.cam_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

        self.color2depth_ratio = self.camera_config.color2depth_ratio

        cas.set(CASViews.COLOR_IMAGE, self.color)
        cas.set(CASViews.DEPTH_IMAGE, self.depth)
        cas.set(CASViews.CAM_INFO, self.cam_info)
        cas.set(CASViews.CAM_INTRINSIC, self.cam_intrinsic)
        cas.set(CASViews.COLOR2DEPTH_RATIO, self.color2depth_ratio)

        if self.lookup_viewpoint:
            st = robokudo.types.tf.StampedTransform()
            st.rotation = self.cam_quaternion
            st.translation = self.cam_translation
            st.frame = self.tf_from
            st.child_frame = self.tf_to
            st.timestamp = self.timestamp
            cas.set(CASViews.VIEWPOINT_CAM_TO_WORLD, st)

        self._has_new_data = False

        self.lock.release()


# Archived ROS lookup
# '16UC1; compressedDepth'
# bridge = CvBridge()
# self.lock.acquire()
# try:
#     cv_rgb_image = bridge.imgmsg_to_cv2(image_data)
#     cv_depth_image = bridge.imgmsg_to_cv2(depth_data)
# except CvBridgeError as e:
#     print(e)
#     self.lock.release()
#     return
#
# (rows, cols, channels) = cv_rgb_image.shape
# print(f'RGB image parameters (rows,cols,channels,pixels,type): {rows}, {cols}, {channels}, {cv_rgb_image.size}, {cv_rgb_image.dtype}')
# (rows, cols) = cv_depth_image.shape
# print(f'Depth image parameters (rows,cols,pixels,type): {rows}, {cols}, {cv_depth_image.size}, {cv_depth_image.dtype}')
#
"""ROS1 TO ROS2 
Adapts to changes in CvBridge and uses np.frombuffer instead of np.fromstring to avoid deprecation issues.

 ROS2 explicitly creates a node (self.node = Node('kinect_camera_node')), which is then used to initialize the subscribers. 
 
 The import statements and the way Subscriber and ApproximateTimeSynchronizer are used differ slightly due to the differences between ROS1 and ROS2 libraries.
 
 bridge.imgmsg_to_cv2(color_data, "bgr8") converts the ROS image message directly to an OpenCV image in BGR format, bypassing the need for a separate RGB to BGR conversion step.
 
 In ROS2 need to explicitly specify the format (like "32FC1") to ensure the depth image is correctly interpreted and converted

"""
