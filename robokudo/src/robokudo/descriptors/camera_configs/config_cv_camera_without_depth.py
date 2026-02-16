"""
This is the camera config for a openCV camera without depth information.
"""
from dataclasses import dataclass

import cv2


@dataclass
class CameraConfig:
    """Configuration class for OpenCV-based cameras without depth information.

    This class defines the configuration parameters for cameras that can be accessed
    through OpenCV's video capture interface. It supports various input sources
    including physical cameras, video files, image sequences, and network streams.

    :ivar interface_type: Type of camera interface, set to "OpenCV"
    :type interface_type: str
    :ivar device: Input source identifier (camera index, file path, or URL)
    :type device: Union[int, str]
    :ivar api_preference: Preferred OpenCV capture API backend
    :type api_preference: int
    :ivar device_driver_flag: Flag for frame retrieval configuration
    :type device_driver_flag: int
    :ivar normalize_rgb: Flag to enable RGB image normalization
    :type normalize_rgb: bool
    :ivar loop_mode: Video playback loop configuration (-1: infinite, 0: no loop, >0: n loops)
    :type loop_mode: int
    :ivar depth: Static depth image to use (if any)
    :type depth: Optional[numpy.ndarray]
    :ivar update_global_with_depth_parameter: Flag to update global depth parameters
    :type update_global_with_depth_parameter: bool
    :ivar cam_info: Camera configuration dictionary
    :type cam_info: Optional[dict]
    :ivar cam_intrinsic: Camera intrinsic parameters
    :type cam_intrinsic: Optional[numpy.ndarray]
    :ivar color2depth_ratio: Ratio between color and depth image resolution (x, y)
    :type color2depth_ratio: tuple[float, float]
    :ivar viewpoint_cam_to_world: Camera to world transformation
    :type viewpoint_cam_to_world: Optional[numpy.ndarray]

    .. note::
        When using image sequences as input, the first image must have a number
        between 0 and 4, and there cannot be any gaps in the sequence numbering.
    """
    # camera
    interface_type: str = "OpenCV"

    # device (integer for I/O device or path to image/video file) and flag
    # integer:  id of the video capturing device to open
    #           Use 0 to open default camera using default backend
    # string:   path to a video file
    #           path to image sequence (e.g. 'my/path/img_%02d.jpg').
    #               Note: First image needs number between 0 and 4 and the following numbers cannot contain any gaps.
    #           URL of video/camera stream or image
    device = 0
    api_preference = cv2.CAP_ANY  # preferred Capture API backends to use.

    # flag argument to use when retrieving/reading the frames
    device_driver_flag: int = 0

    # normalize/stabilise rgb image contrast/brightness
    normalize_rgb: bool = True

    # loop after iterating over all frames of the video file
    # > 0: number of repetitions before to stop
    # = 0: never loop
    # < 0: infinite loop
    # Note: Only works with a file as source
    loop_mode: int = -1

    # static depth image, which is used as depth image for all rgb images
    depth = None
    update_global_with_depth_parameter: bool = True

    # camera config as dict
    cam_info = None

    # camera intrinsic
    cam_intrinsic = None

    # if the resolution of the depth image differs from the color image, we need to define the factor for (x, y).
    color2depth_ratio = (1, 1)

    viewpoint_cam_to_world = None
