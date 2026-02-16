from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration class for a Kinect camera setup in a robotic environment.

    This class defines the configuration parameters for a Kinect camera, including
    interface settings, topic names, and transformation settings. It is designed to
    work with ROS topics and TF transformations.

    :ivar interface_type: Type of camera interface, set to "Kinect"
    :type interface_type: str
    :ivar depthOffset: Offset value for depth measurements
    :type depthOffset: int
    :ivar filterBlurredImages: Flag to enable/disable filtering of blurred images
    :type filterBlurredImages: bool
    :ivar color2depth_ratio: Ratio between color and depth image resolution (x, y)
    :type color2depth_ratio: tuple[float, float]
    :ivar hi_res_mode: Enable high resolution mode for better depth-to-RGB matching
    :type hi_res_mode: bool
    :ivar topic_depth: ROS topic for depth image data
    :type topic_depth: str
    :ivar topic_color: ROS topic for color image data
    :type topic_color: str
    :ivar topic_cam_info: ROS topic for camera information
    :type topic_cam_info: str
    :ivar depth_hints: Transport hints for depth image subscription
    :type depth_hints: str
    :ivar color_hints: Transport hints for color image subscription
    :type color_hints: str
    :ivar tf_from: Frame ID of the camera's optical frame
    :type tf_from: str
    :ivar tf_to: Target frame ID for transformations
    :type tf_to: str
    :ivar lookup_viewpoint: Flag to enable viewpoint lookup
    :type lookup_viewpoint: bool
    :ivar only_stable_viewpoints: Flag to use only stable viewpoints
    :type only_stable_viewpoints: bool
    :ivar max_viewpoint_distance: Maximum allowed distance for viewpoint changes
    :type max_viewpoint_distance: float
    :ivar max_viewpoint_rotation: Maximum allowed rotation for viewpoint changes
    :type max_viewpoint_rotation: float
    :ivar semantic_map: Filename of the semantic map configuration
    :type semantic_map: str
    """
    # camera
    interface_type = "Kinect"
    depthOffset = 0
    filterBlurredImages = True
    # If the resolution of the depth image differs from the color image, we need to define the factor for (x, y).
    # Example: (0.5,0.5) for a 640x480 depth image compared to a 1280x960 rgb image Otherwise, just put (1,1) here
    color2depth_ratio = (0.5, 0.5)
    hi_res_mode = True  # Setting this to true will apply some workarounds to match the depth data to RGB on the Kinect

    # camera topics
    # topic_depth = "/kinect_head/depth_registered/image_raw"
    topic_depth = "/kinect_head/depth_registered/image_raw/compressedDepth"
    topic_color = "/kinect_head/rgb/image_color/compressed"
    topic_cam_info = "/kinect_head/rgb/camera_info"
    # depth_hints = "raw"
    depth_hints = "compressedDepth"
    color_hints = "compressed"

    # tf
    tf_from = "head_mount_kinect_rgb_optical_frame"
    tf_to = "map"
    lookup_viewpoint = True
    only_stable_viewpoints = True
    max_viewpoint_distance = 0.01
    max_viewpoint_rotation = 1.0
    semantic_map = "semantic_map_iai_kitchen.yaml"
