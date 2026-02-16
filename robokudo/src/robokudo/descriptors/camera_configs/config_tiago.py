from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration class for the TIAGo robot's Xtion camera.

    This class defines the configuration parameters for the Xtion RGB-D camera
    mounted on the TIAGo robot. It includes settings for camera interface,
    topic names, and transformation settings.

    :ivar interface_type: Type of camera interface, set to "Kinect" for compatibility
    :type interface_type: str
    :ivar depthOffset: Offset value for depth measurements
    :type depthOffset: int
    :ivar filterBlurredImages: Flag to enable/disable filtering of blurred images
    :type filterBlurredImages: bool
    :ivar color2depth_ratio: Ratio between color and depth image resolution (x, y)
    :type color2depth_ratio: tuple[float, float]
    :ivar hi_res_mode: Enable high resolution mode (disabled for Xtion)
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
    :ivar tf_from: Frame ID of the Xtion's RGB optical frame
    :type tf_from: str
    :ivar tf_to: Target frame ID for transformations (odom frame)
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

    .. note::
        The configuration uses the Kinect interface type for compatibility with
        the Xtion camera, as both cameras use similar RGB-D data formats.
    """
    # camera
    interface_type = "Kinect"
    depthOffset = 0
    filterBlurredImages = True
    # If the resolution of the depth image differs from the color image, we need to define the factor for (x, y).
    # Example: (0.5,0.5) for a 640x480 depth image compared to a 1280x960 rgb image Otherwise, just put (1,1) here
    color2depth_ratio = (1, 1)
    hi_res_mode = False

    # camera topics
    topic_depth = "/xtion/depth_registered/image_raw/compressedDepth"
    topic_color = "/xtion/rgb/image_raw/compressed"
    topic_cam_info = "/xtion/rgb/camera_info"
    depth_hints = "compressedDepth"
    color_hints = "compressed"

    # tf
    tf_from = "/xtion_rgb_optical_frame"
    tf_to = "/odom"
    lookup_viewpoint = True
    only_stable_viewpoints = True
    max_viewpoint_distance = 0.01
    max_viewpoint_rotation = 1.0
    semantic_map = "semantic_map_iai_kitchen.yaml"
