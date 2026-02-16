from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration class for MongoDB-based camera data playback.

    This class defines the configuration parameters for reading camera data from a
    MongoDB database, typically used for offline processing or testing with stored
    scene data.

    :ivar interface_type: Type of camera interface, set to "StorageReader"
    :type interface_type: str
    :ivar loop: Flag to enable looping over database entries
    :type loop: bool
    :ivar db_name: Name of the MongoDB database to read from
    :type db_name: str
    :ivar semantic_map: Filename of the semantic map configuration
    :type semantic_map: str

    .. note::
        This configuration is used to read previously stored camera data from a
        MongoDB database, allowing for replay and analysis of recorded scenes.
    """
    # camera
    interface_type = "StorageReader"

    # mongo
    loop = True
    db_name = 'rk_scenes'  # database name
    # tf
    semantic_map = "semantic_map_iai_kitchen.yaml"