from dataclasses import dataclass


@dataclass
class CameraConfig:
    # camera
    interface_type = "Kinect"
    depthOffset = 0
    filterBlurredImages = False
    # If the resolution of the depth image differs from the color image, we need to define the factor for (x, y).
    # Example: (0.5,0.5) for a 640x480 depth image compared to a 1280x960 rgb image Otherwise, just put (1,1) here
    color2depth_ratio = (1.0, 1.0)
    hi_res_mode = False  # Setting this to true will apply some workarounds to match the depth data to RGB on the Kinect

    # camera topics
    # topic_depth = "/camera/depth/image_raw/compressedDepth"
    topic_depth = "/camera/depth/image_raw"
    topic_color = "/camera/color/image_raw/compressed"
    topic_cam_info = "/camera/color/camera_info"
    depth_hints = "raw"
    color_hints = "compressed"

    # tf
    lookup_viewpoint = False
    tf_from = "camera_color_optical_frame"
    tf_to = "map"
    only_stable_viewpoints = True
    max_viewpoint_distance = 0.01
    max_viewpoint_rotation = 1.0
    semantic_map = "semantic_map_iai_kitchen.yaml"
