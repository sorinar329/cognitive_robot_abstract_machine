from dataclasses import dataclass

# Camera config for Realsense Camera devices.
# Tested with a D435
# Please start the ROS Realsense driver like this:
#   roslaunch realsense2_camera rs_aligned_depth.launch
@dataclass
class CameraConfig:
    """Configuration class for Intel RealSense cameras.

    This class defines the configuration parameters for RealSense cameras,
    particularly tested with the D435 model. It assumes the use of aligned depth
    data and requires the RealSense ROS driver to be running.

    :ivar interface_type: Type of camera interface, set to "Kinect" for compatibility
    :type interface_type: str
    :ivar depthOffset: Offset value for depth measurements
    :type depthOffset: int
    :ivar filterBlurredImages: Flag to enable/disable filtering of blurred images
    :type filterBlurredImages: bool
    :ivar color2depth_ratio: Ratio between color and depth image resolution (x, y)
    :type color2depth_ratio: tuple[float, float]
    :ivar hi_res_mode: Enable high resolution mode (disabled for RealSense)
    :type hi_res_mode: bool
    :ivar topic_depth: ROS topic for aligned depth image data
    :type topic_depth: str
    :ivar topic_color: ROS topic for color image data
    :type topic_color: str
    :ivar topic_cam_info: ROS topic for camera information
    :type topic_cam_info: str
    :ivar depth_hints: Transport hints for depth image subscription
    :type depth_hints: str
    :ivar color_hints: Transport hints for color image subscription
    :type color_hints: str
    :ivar tf_from: Frame ID of the camera's color optical frame
    :type tf_from: str
    :ivar tf_to: Target frame ID for transformations
    :type tf_to: str
    :ivar lookup_viewpoint: Flag to enable viewpoint lookup (disabled by default)
    :type lookup_viewpoint: bool
    :ivar only_stable_viewpoints: Flag to use only stable viewpoints
    :type only_stable_viewpoints: bool
    :ivar max_viewpoint_distance: Maximum allowed distance for viewpoint changes
    :type max_viewpoint_distance: float
    :ivar max_viewpoint_rotation: Maximum allowed rotation for viewpoint changes
    :type max_viewpoint_rotation: float
    :ivar semantic_map: Filename of the semantic map configuration
    :type semantic_map: str

    .. note::
        Requires the RealSense ROS driver to be running with aligned depth:
        roslaunch realsense2_camera rs_aligned_depth.launch
    """
    # camera
    interface_type = "Kinect"
    depthOffset = 0
    filterBlurredImages = True
    # We currently assume that the Color and Depth topics run at the same resolution
    color2depth_ratio = (1.0, 1.0)
    hi_res_mode = False

    # camera topics
    topic_depth = "/camera/aligned_depth_to_color/image_raw/compressedDepth"
    topic_color = "/camera/color/image_raw/compressed"
    topic_cam_info = "/camera/color/camera_info"
    depth_hints = "compressedDepth"
    color_hints = "compressed"

    # tf
    tf_from = "/camera_color_optical_frame"
    tf_to = "/map"
    lookup_viewpoint = False
    only_stable_viewpoints = True
    max_viewpoint_distance = 0.01
    max_viewpoint_rotation = 1.0
    semantic_map = "semantic_map_iai_kitchen.yaml"
