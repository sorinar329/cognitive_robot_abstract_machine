"""
Storage writer annotator for RoboKudo.

This module provides an annotator for storing sensor data in MongoDB.
It supports:

* MongoDB integration
* CAS data persistence
* View data storage
* Database management
* Configurable database settings

The module is used for:

* Data recording
* Offline processing
* Dataset creation
* Experiment logging
"""
import copy
from timeit import default_timer
import py_trees
import robokudo.annotators.outputs
import robokudo.io.storage


class StorageWriter(robokudo.annotators.core.BaseAnnotator):
    """
    Annotator for storing sensor data in MongoDB.

    This annotator provides methods to store sensor data in a MongoDB database,
    allowing for data recording and offline processing without using ROS bag files.

    :ivar storage: MongoDB storage interface
    :type storage: robokudo.io.storage.Storage
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """
        Configuration descriptor for storage writer.

        :ivar parameters: Storage parameters
        :type parameters: StorageWriter.Descriptor.Parameters
        """

        class Parameters:
            """
            Parameter container for storage configuration.

            :ivar db_name: Name of MongoDB database
            :type db_name: str
            :ivar drop_database_on_storage: Whether to clear database before recording
            :type drop_database_on_storage: bool
            """
            def __init__(self):
                self.db_name = "rk_scenes"  # database name
                self.drop_database_on_storage = True  # True or False to wipe the database completely before
                # recording data
        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="StorageWriter", descriptor=Descriptor()):
        """
        Initialize the storage writer. Minimal one-time init!

        :param name: Annotator name, defaults to "StorageWriter"
        :type name: str, optional
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        :type descriptor: StorageWriter.Descriptor, optional
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.storage = robokudo.io.storage.Storage(self.descriptor.parameters.db_name)

        # Wipe the database completely before recording data
        if self.descriptor.parameters.drop_database_on_storage:
            self.storage.drop_database()

    def update(self):
        """
        Store current CAS data in MongoDB.

        Creates a deep copy of the CAS, flattens it into a dictionary,
        and stores both views and CAS data in the database.

        :return: SUCCESS if storage successful, FAILURE otherwise
        :rtype: py_trees.Status
        """
        start_timer = default_timer()

        persist_cas = copy.deepcopy(self.get_cas())

        flat_cas = self.storage.generate_dict_from_real_cas(persist_cas)
        flat_cas['view_ids'] = {}

        # step 1: persist each view
        self.storage.store_views_in_mongo(flat_cas)

        # step 2: persist cas and pointers to each view
        del flat_cas['views']
        result = self.storage.store_cas_dict(flat_cas)
        if not result.acknowledged:
            print(f'mongo db error when trying to store cas')
            return py_trees.common.Status.FAILURE

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS
