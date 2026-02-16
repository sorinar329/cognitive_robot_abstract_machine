"""
MongoDB storage interface for RoboKudo.

This module provides the core interface between RoboKudo and MongoDB for storing
and retrieving sensor data and annotations. It supports:

* Serialization of complex data types
* Conversion between ROS and MongoDB formats
* Efficient binary storage of numpy arrays
* Configurable view storage and restoration
* Thread-safe database operations

The module handles:

* RGB-D camera data
* Camera calibration information
* ROS messages and transforms
* Open3D data structures
* Custom RoboKudo types
"""
import array
import io
import os
import sys
import open3d as o3d
from typing import Optional, Dict, Union

import builtin_interfaces.msg
import numpy as np
from pymongo import MongoClient
from sensor_msgs.msg import CameraInfo

import robokudo.analysis_engine
import robokudo.analysis_engine
import robokudo.cas
import robokudo.types.tf
import robokudo.utils.serialization as serializer
from robokudo.cas import CAS, CASViews
from robokudo.types.tf import StampedTransform


def recursive_convert(value):
    """
    Recursively converts non-BSON types into Python-native types.
    In particular, converts array.array and numpy.ndarray objects into lists.
    """
    # Convert Python array.array to a list
    if isinstance(value, array.array):
        return list(value)
    # Convert numpy arrays to lists
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [recursive_convert(item) for item in value]
    elif isinstance(value, dict):
        return {k: recursive_convert(v) for k, v in value.items()}
    else:
        return value


def ros_message_to_dict(msg):
    """
    Recursively converts a ROS2 message into a dictionary using introspection.
    Ensures that any array.array or numpy.ndarray objects are converted to lists.
    """
    result = {}
    for field, field_type in msg.get_fields_and_field_types().items():
        value = getattr(msg, field)
        if hasattr(value, "get_fields_and_field_types"):
            result[field] = ros_message_to_dict(value)
        elif isinstance(value, list):
            result[field] = [recursive_convert(item) for item in value]
        else:
            result[field] = recursive_convert(value)
    return result


def dict_to_ros_message(message_type, data_dict):
    """
    Recursively converts a dictionary into a ROS2 message of the specified type.
    This example assumes that the dictionary values are basic types.
    """
    msg = message_type()
    for field, field_type in msg.get_fields_and_field_types().items():
        if field in data_dict:
            value = data_dict[field]
            current_field = getattr(msg, field)
            if hasattr(current_field, "get_fields_and_field_types"):
                setattr(msg, field, dict_to_ros_message(type(current_field), value))
            elif isinstance(current_field, list):
                if current_field and hasattr(current_field[0], "get_fields_and_field_types"):
                    new_list = [dict_to_ros_message(type(current_field[0]), item) for item in value]
                else:
                    new_list = value
                setattr(msg, field, new_list)
            else:
                setattr(msg, field, value)
    return msg


class Storage:
    """Main interface between RoboKudo and MongoDB.

    This class holds the main interface code between RoboKudo and the MongoDB database.
    It stores sensor data and CAS views by converting specialized types (NumPy arrays,
    ROS messages, etc.) into a BSON‑encodable dictionary format.

    :ivar db_name: Name of the MongoDB database
    :type db_name: str
    :ivar client: MongoDB client connection
    :type client: pymongo.MongoClient
    :ivar db: MongoDB database instance
    :type db: pymongo.database.Database
    """

    BLACKLISTED_TYPES = (type(lambda x: x),)  # Example: functions

    def is_blacklisted(obj):
        """
        Check if an object is of a blacklisted type.

        :param obj: Object to check
        :type obj: Any
        :return: True if object type is blacklisted, False otherwise
        :rtype: bool
        """
        return isinstance(obj, Storage.BLACKLISTED_TYPES)

    @staticmethod
    def instantiate_mongo_client():
        """
        Create a MongoDB client instance.

        Uses environment variables RK_MONGO_HOST and RK_MONGO_PORT if set,
        otherwise defaults to localhost:27017.

        :return: MongoDB client instance
        :rtype: pymongo.MongoClient
        """
        # Fetch environment variables which might have been used to configure
        # another MongoDB Host and Port
        # This was originally introduced to support unit tests.
        mongo_host = os.getenv("RK_MONGO_HOST", "localhost")
        mongo_port = int(os.getenv("RK_MONGO_PORT", 27017))

        return MongoClient(host=mongo_host, port=mongo_port)

    @DeprecationWarning  # Use ListReader instead for better pymongo compatibility across versions
    class Reader:
        """
        Deprecated cursor-based MongoDB reader.

        .. deprecated:: Use ListReader instead for better pymongo compatibility
        """

        def __init__(self, db_name):
            """
            Initialize the reader.

            :param db_name: Name of the MongoDB database
            :type db_name: str
            """
            client = Storage.instantiate_mongo_client()

            self.db_reader = client[db_name]
            self.reset_cursor()

        def reset_cursor(self):
            """Reset the cursor to the start of the collection."""
            self.cursor = self.db_reader.cas.find()

        def collection_has_frames(self):
            """
            Check if collection has any frames.

            :return: True if frames exist, False otherwise
            :rtype: bool
            """
            return self.db_reader.cas.find().alive

        def cursor_has_frames(self):
            """
            Check if cursor has more frames.

            :return: True if more frames exist, False otherwise
            :rtype: bool
            """
            return self.cursor.alive

        def get_next_frame(self) -> Optional[dict]:
            """
            Get the next frame from the cursor.

            :return: Next frame data or None if no more frames
            :rtype: dict or None
            """
            if not self.cursor_has_frames():
                return None
            return self.cursor.next()

    class ListReader:
        """
        List-based MongoDB reader.

        This class reads all matching records into a list for iteration,
        providing better compatibility across pymongo versions and simpler
        cursor management.

        :ivar db_reader: MongoDB database instance
        :type db_reader: pymongo.database.Database
        :ivar index: Current position in data list
        :type index: int or None
        :ivar data: List of loaded documents
        :type data: list
        """

        def __init__(self, db_name):
            """
            Initialize the list reader.

            :param db_name: Name of the MongoDB database
            :type db_name: str
            """
            client = Storage.instantiate_mongo_client()
            self.db_reader = client[db_name]
            self.index = None
            self.data = []
            self.reset_cursor()

        def reset_cursor(self):
            """
            Reset the reader state.

            Clears and reloads all documents from the database.
            """
            self.data.clear()
            for data in self.db_reader.cas.find():
                self.data.append(data)
            self.index = 0 if self.data else None

        def cursor_has_frames(self):
            """
            Check if more frames are available.

            :return: True if more frames exist, False otherwise
            :rtype: bool
            """
            return self.index is not None and self.index < len(self.data)

        def get_next_frame(self) -> Optional[dict]:
            """
            Get the next frame from the data list.

            :return: Next frame data or None if no more frames
            :rtype: dict or None
            """
            if not self.cursor_has_frames():
                return None
            data = self.data[self.index]
            self.index += 1
            return data

    def __init__(self, db_name):
        """
        Initialize the storage interface.

        :param db_name: Name of the MongoDB database
        :type db_name: str
        """
        self.db_name = db_name
        self.client = Storage.instantiate_mongo_client()
        self.db = self.client[db_name]

    def drop_database(self):
        """Drop the entire database."""
        self.client.drop_database(self.db_name)

    @staticmethod
    def nd_array_to_numpy_binary(arr: np.ndarray) -> bytes:
        """
        Convert numpy array to binary format for storage.

        :param arr: Numpy array to convert
        :type arr: numpy.ndarray
        :return: Binary representation of array
        :rtype: bytes
        """
        memfile = io.BytesIO()
        np.save(memfile, arr)
        return memfile.getvalue()

    @staticmethod
    def numpy_binary_to_nd_array(binary: bytes) -> np.ndarray:
        """
        Convert binary data back to numpy array.

        :param bin: Binary data to convert
        :type bin: bytes
        :return: Reconstructed numpy array
        :rtype: numpy.ndarray
        """
        memfile = io.BytesIO()
        memfile.write(binary)
        memfile.seek(0)
        return np.load(memfile)

    # Conversion functions for ROS messages and transforms

    @staticmethod
    def ros_cam_info_to_mongo(cam_info: CameraInfo) -> dict:
        """Convert ROS camera info to MongoDB format.

        Convert a ROS2 CameraInfo message into a dictionary using our custom introspection,
        ensuring that all non-BSON types (numpy.ndarray, array.array) are converted.

        :param cam_info: ROS camera info message
        :type cam_info: sensor_msgs.msg.CameraInfo
        :return: Dictionary representation of camera info
        :rtype: dict
        """
        return ros_message_to_dict(cam_info)

    @staticmethod
    def ros_cam_info_from_mongo(mongo_cam_info: dict) -> CameraInfo:
        """Convert a dictionary from MongoDB back into a ROS2 CameraInfo message.

        :param mongo_cam_info: Dictionary representation of camera info
        :type mongo_cam_info: dict
        :return: ROS camera info message
        :rtype: sensor_msgs.msg.CameraInfo
        """
        return dict_to_ros_message(CameraInfo, mongo_cam_info)

    @staticmethod
    def camera_intrinsic_to_mongo(camera_intrinsic: o3d.camera.PinholeCameraIntrinsic) -> Dict[str, Union[int, float]]:
        intrinsic_matrix = camera_intrinsic.intrinsic_matrix       # 3 x 3
        result = {
            "width": camera_intrinsic.width,
            "height": camera_intrinsic.height,
            "fx": intrinsic_matrix[0, 0],
            "fy": intrinsic_matrix[1, 1],
            "cx": intrinsic_matrix[0, 2],
            "cy": intrinsic_matrix[1, 2]
        }
        return result

    @staticmethod
    def camera_intrinsic_from_mongo(camera_intrinsic_dict: Dict[str, Union[int, float]]) \
            -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(**camera_intrinsic_dict)


    @staticmethod
    def rk_stamped_transform_to_mongo(stamped_transform: StampedTransform) -> dict:
        """
        Convert RoboKudo stamped transform to MongoDB format.

        :param stamped_transform: RoboKudo stamped transform
        :type stamped_transform: robokudo.types.tf.StampedTransform
        :return: Dictionary representation of transform
        :rtype: dict
        """
        result = stamped_transform.__dict__
        time = result['timestamp']  # this is a ros type that has no __dict__
        result['timestamp'] = {
            'secs': time.sec,
            'nsecs': time.nanosec,
        }
        return result

    @staticmethod
    def rk_stamped_transform_from_mongo(stamped_transform_dict: dict) -> StampedTransform:
        """
        Convert MongoDB transform back to RoboKudo format.

        :param stamped_transform_dict: Dictionary representation of transform
        :type stamped_transform_dict: dict
        :return: RoboKudo stamped transform
        :rtype: robokudo.types.tf.StampedTransform
        """
        st = robokudo.types.tf.StampedTransform()
        st.rotation = stamped_transform_dict['rotation']
        st.translation = stamped_transform_dict['translation']
        st.frame = stamped_transform_dict['frame']
        st.child_frame = stamped_transform_dict['child_frame']
        st.timestamp = builtin_interfaces.msg.Time(
            sec=stamped_transform_dict['timestamp']['secs'],
            nanosec=stamped_transform_dict['timestamp']['nsecs']
        )
        return st

    # TODO Have a seperate module for conversion handling

    # TODO This should be overwritable in the Descriptor/Config. Atleast the Views you want to save/restore.
    view_configuration = {
        CASViews.COLOR_IMAGE: {
            'to_mongo': lambda x: Storage.nd_array_to_numpy_binary(x),
            'from_mongo': lambda x: Storage.numpy_binary_to_nd_array(x)
        },
        CASViews.DEPTH_IMAGE: {
            'to_mongo': lambda x: Storage.nd_array_to_numpy_binary(x),
            'from_mongo': lambda x: Storage.numpy_binary_to_nd_array(x)
        },
        CASViews.COLOR2DEPTH_RATIO: {
            'from_mongo': lambda x: tuple(x),  # mongodb will serialize and load to lists
        },
        CASViews.CAM_INFO: {
            'to_mongo': lambda x: Storage.ros_cam_info_to_mongo(x),
            'from_mongo': lambda x: Storage.ros_cam_info_from_mongo(x)
        },
        CASViews.CAM_INTRINSIC: {
            'to_mongo': lambda x: Storage.camera_intrinsic_to_mongo(x),
            'from_mongo': lambda x: Storage.camera_intrinsic_from_mongo(x)
        },
        CASViews.VIEWPOINT_CAM_TO_WORLD: {
            'to_mongo': lambda x: Storage.rk_stamped_transform_to_mongo(x),
            'from_mongo': lambda x: Storage.rk_stamped_transform_from_mongo(x)
        }
    }

    def store_views_in_mongo(self, cas_dict: Dict) -> None:
        """Store CAS views in MongoDB.

        Store the views present in cas.views.
        Please note that this method will change cas. So make a deepcopy if you want to leave your
        true CAS untouched!

        :param cas_dict: CAS as a dictionary to persist
        :type cas_dict: dict
        """
        for view in Storage.view_configuration:
            if view not in cas_dict['views']:
                print(f'You want to store view "{view}" but it is not in the CAS. Ignoring...')
                continue

            mongo_view_data = cas_dict['views'][view]
            if mongo_view_data is not None:
                # get a transform function or get identity function if no special function is set
                to_mongo_fun = Storage.view_configuration[view].get('to_mongo', lambda x: x)
                mongo_view_data = to_mongo_fun(mongo_view_data)

            # Construct the document (record) we want to save in the DB
            mongo_view_document = {'data': mongo_view_data}
            result = self.db[view].insert_one(mongo_view_document)
            if not result.acknowledged:
                print(f'Mongo DB Error when trying to store view "{view}"')
                sys.exit(1)

            cas_dict['view_ids'][view] = result.inserted_id

    def load_views_from_mongo_in_cas(self, cas_document: Dict) -> None:
        """Load views from MongoDB into a CAS document.

        Retrieves and converts each view referenced in the CAS document from its
        MongoDB representation back to its original format.

        :param cas_document: CAS document to update with loaded views
        :type cas_document: dict
        """
        for view_name, view_id in cas_document['view_ids'].items():
            view_document = self.db[view_name].find_one({'_id': view_id})
            if not view_document:
                print(f"Couldn't find View '{view_name}' with id={view_id}")
                sys.exit(1)
            from_mongo_fun = Storage.view_configuration[view_name].get('from_mongo', lambda x: x)
            view_data = from_mongo_fun(view_document['data'])

            view_data = view_document['data']
            if view_data is not None:
                # Put the view document into the CAS we are restoring - Respect transform method if present
                from_mongo_fun = Storage.view_configuration[view_name].get('from_mongo', lambda x: x)
                view_data = from_mongo_fun(view_data)

            cas_document['views'][view_name] = view_data

    def load_annotations_from_mongo_in_cas(self, cas_document: Dict, cas: CAS) -> None:
        """Load annotations from MongoDB into a CAS.

        Restore the annotations from the database and insert them into the CAS.
        If no (pickled) annotations are available, cas will be untouched.

        :param cas_document: A dict representing a frame of a CAS in the database
        :type cas_document: dict
        :param cas: The 'robokudo.cas.CAS' instance where the annotations shall be inserted to
        :type cas: robokudo.cas.CAS
        """
        if cas_document['annotations']:
            cas.annotations = serializer.unflatten(cas_document['annotations'])

    def generate_dict_from_real_cas(self, cas: CAS) -> Dict:
        """Convert a CAS instance to a MongoDB-compatible dictionary.

        Generate dict that we can put in mongo from CAS

        :param cas: Input CAS that should be used to create a dict-representation of it.
        :type cas: robokudo.cas.CAS
        :return: A dict with references to parts of the input CAS.
        :rtype: dict
        """
        serialized_annotations = serializer.flatten(cas.annotations)

        result = {
            'timestamp': cas.timestamp,
            'timestamp_readable': cas.timestamp_readable,
            'annotations': serialized_annotations,
            'views': cas.views
        }

        return result

    def store_cas_dict(self, cas_dict: Dict):
        """
        Store a CAS dictionary in MongoDB.

        Stores the views and creates a CAS document in MongoDB that references
        them.

        :param cas_dict: Dictionary representation of a CAS
        :type cas_dict: dict
        :return: MongoDB ObjectId of the stored CAS document
        :rtype: bson.objectid.ObjectId
        """
        return self.db.cas.insert_one(cas_dict)
