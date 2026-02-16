from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration class for file-based camera data playback.

    This class defines the configuration parameters for reading camera data from files,
    typically used for offline processing or testing. It supports reading from both
    ROS package directories and regular filesystem paths.

    :ivar interface_type: Type of camera interface, set to "FileReader"
    :type interface_type: str
    :ivar loop: Flag to enable looping over directory contents
    :type loop: bool
    :ivar target_ros_package: ROS package name containing the data files (optional)
    :type target_ros_package: Optional[str]
    :ivar target_dir: Directory path containing the data files
    :type target_dir: str
    :ivar color2depth_ratio: Ratio between color and depth image resolution (x, y)
    :type color2depth_ratio: tuple[float, float]
    :ivar filename_prefix: Prefix for filtering relevant files
    :type filename_prefix: str
    :ivar kinect_height_fix_mode: Flag to enable Kinect-specific height corrections
    :type kinect_height_fix_mode: bool

    .. note::
        If target_ros_package is None, target_dir is used as an absolute or relative
        path. Otherwise, target_dir is appended to the ROS package path.
    """
    # camera
    interface_type = "FileReader"

    # shall we loop after iterating over a directory?
    loop = True

    # Files for the FileReaderInterface should be in a ROS package
    # If you don't want that, leave target_ros_package to None and define target_dir
    # either absolute or relative to your executable.
    target_ros_package = None
    # If target_ros_package is None, try to load target_dir directly.
    # Otherwise, append target_dir to target_ros_package and load the files from there.
    target_dir = "/tmp"

    # If you have depth data to read from, please set the ratio in x,y here
    # If it's not set, (1,1) will be the default. Set to None if you don't have that.
    color2depth_ratio = (0.5, 0.5)

    # Define the prefix of all the files that shall be loaded into the FileReaderInterface
    filename_prefix = "rk_"

    # Apply kinect hack
    kinect_height_fix_mode = True