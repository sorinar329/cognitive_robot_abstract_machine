from dataclasses import dataclass

"""
Camera config for the camera-in-hand on the Toyota HSR robot.
"""


@dataclass
class CameraConfig:
    """Configuration class for the hand-mounted camera on the Toyota HSR robot.

    This class defines the configuration parameters for the HSR's hand camera,
    including interface settings, topic names, and transformation settings. The camera
    provides color images without depth information.

    :ivar interface_type: Type of camera interface, set to "ROSCameraWithoutDepthInterface"
    :type interface_type: str
    :ivar depthOffset: Offset value for depth measurements (unused in this config)
    :type depthOffset: int
    :ivar rotate_image: Image rotation setting ('90_ccw', '90_cw', '180', or None)
    :type rotate_image: str
    :ivar topic_color: ROS topic for color image data
    :type topic_color: str
    :ivar topic_cam_info: ROS topic for camera information
    :type topic_cam_info: str
    :ivar tf_from: Frame ID of the hand camera
    :type tf_from: str
    :ivar tf_to: Target frame ID for transformations
    :type tf_to: str
    :ivar lookup_viewpoint: Flag to enable viewpoint lookup (currently disabled)
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
    interface_type = "ROSCameraWithoutDepthInterface"
    depthOffset = 0
    rotate_image = '90_ccw'  # Possible values: None, '90_ccw', '90_cw', '180' with ccw = counter-clockwise, cw = clockwise
    # filterBlurredImages = True

    # camera topics
    # topic_color = "hsrb/hand_camera/image_rect_color"
    topic_color = "/hsrb/hand_camera/image_raw"
    topic_cam_info = "/hsrb/hand_camera/camera_info"
    # color_hints = "compressed"

    # tf
    tf_from = "/hand_camera_frame"
    tf_to = "/map"
    lookup_viewpoint = False  # TODO: CHange this for TODO for viewpoint of cam in world
    only_stable_viewpoints = True
    max_viewpoint_distance = 0.01
    max_viewpoint_rotation = 1.0
    semantic_map = "semantic_map_iai_kitchen.yaml"
