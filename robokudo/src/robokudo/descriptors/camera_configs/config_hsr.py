"""
This is the camera config for the HSR robot.
"""
from dataclasses import dataclass


@dataclass
class CameraConfig:
    """
    A config
    """
    # camera
    interface_type = "Kinect"
    depthOffset = 0
    filterBlurredImages = True
    # If the resolution of the depth image differs from the color image,
    # we need to define the factor for (x, y).
    # Example: (0.5,0.5) for a 640x480 depth image compared to a
    # 1280x960 rgb image Otherwise, just put (1,1) here
    color2depth_ratio = (1, 1)
    # Setting this to true will apply some workarounds
    # to match the depth data to RGB on the Kinect
    hi_res_mode = False

    # camera topics
    # topic_depth = "hsrb/head_rgbd_sensor/depth_registered/image_raw"
    topic_depth = "hsrb/head_rgbd_sensor/depth_registered/image/compressedDepth"
    topic_color = "hsrb/head_rgbd_sensor/rgb/image_raw/compressed"
    topic_cam_info = "hsrb/head_rgbd_sensor/rgb/camera_info"
    # depth_hints = "raw"
    depth_hints = "compressedDepth"
    color_hints = "compressed"

    # tf
    tf_from = "head_rgbd_sensor_rgb_frame"
    tf_to = "map"
    lookup_viewpoint = True  # TODO: CHange this for TODO for viewpoint of cam in world
    only_stable_viewpoints = True
    max_viewpoint_distance = 0.01
    max_viewpoint_rotation = 1.0
    semantic_map = "semantic_map_iai_kitchen.yaml"
